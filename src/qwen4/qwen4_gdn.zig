const trf = @import("../transformer.zig");
const std = @import("std");
const gdn_decode = @import("gdn_decode.zig");
const mlx = @import("../mlx.zig");
const log = @import("../log.zig");
const testing = std.testing;
const test_support = @import("../test_support.zig");
const attn256MaxDiff = test_support.attn256MaxDiff;
const attn256RandBf16 = test_support.attn256RandBf16;
const attn256RandBf16Scaled = test_support.attn256RandBf16Scaled;
const bf16Trunc = test_support.bf16Trunc;
const evalToF32 = test_support.evalToF32;
const qkvExpectBitsEqual = test_support.qkvExpectBitsEqual;
const GDN_KERNEL_HEADER = trf.GDN_KERNEL_HEADER;
const KVCache = trf.KVCache;
const LinearAttnWeights = trf.LinearAttnWeights;
const SSMCacheEntry = trf.SSMCacheEntry;
const Transformer = trf.Transformer;
const GdnProj = Transformer.GdnProj;
const GdnVerifyInputs = Transformer.GdnVerifyInputs;
const PREFILL_EVAL_MIN_SEQ = Transformer.PREFILL_EVAL_MIN_SEQ;
const bf16Scalar = trf.bf16Scalar;
const compactConvStateEnabled = trf.compactConvStateEnabled;
const conv1dExpectBitEqual = trf.conv1dExpectBitEqual;
const diagEnvOnCached = trf.diagEnvOnCached;
const getGdnKernel = trf.getGdnKernel;
const getGdnKernelFor = trf.getGdnKernelFor;
const materializedOwnedCopy = trf.materializedOwnedCopy;
const qwen4Standin = trf.qwen4Standin;
const ssmFreeSpecCapture = trf.ssmFreeSpecCapture;
const ssmRollbackFromCapture = trf.ssmRollbackFromCapture;
const standinRef = trf.standinRef;
const verifyQmmNaxAvailable = trf.verifyQmmNaxAvailable;



pub fn getGdnKernelSeq(vector_gate: bool) !mlx.mlx_fast_metal_kernel {
    return if (vector_gate) getGdnKernelFor(true, true) else getGdnKernelFor(false, true);
}

// ── GatedDeltaNet blocked-seq PREFILL kernel ──
// Port of oMLX's `gated_delta_blocked_seq` (custom_kernels/qwen35_prefill/
// gdn.py, Apache-2.0, oMLX by jundot — keep this provenance chain). Same
// EXACT recurrence as GDN_KERNEL_SOURCE — no chunked/WY reformulation —
// restructured for Apple-GPU efficiency: k/q/v staged into threadgroup
// memory in TB-token blocks with coalesced cooperative loads (the stock
// kernel re-reads k/q from device once per (Dv/4)-slice threadgroup => 32x
// redundant traffic, ~13 GB per 16K-token layer), state register-resident
// as (dv row, 16-wide d segment) fragments, and the k·state / q·state
// contractions reduced across the 8 segment-threads of a dv row via
// simd_shuffle_down — no threadgroup barriers in the hot token loop. Their
// measurement: 14.9 ms vs 29.7 ms per layer @16K = ~2x over the stock
// kernel shape; 48 of the 27B's 64 layers are GDN.
// Geometry contract (checked by gdnBlockedEligible): Dk == 128 exactly
// (8 threads x 16-wide fragments), Dv % 32 == 0 (DB=32 dv rows per
// threadgroup). Anything else — and decode (T==1), PLD/MTP verify, and the
// per-position-state capture path — stays on the stock kernels.
// Block size 32 for bf16 — Metal's 32 KiB threadgroup limit governs,
// and it is a function of the INPUT WIDTH, not a constant: fp32 activations
// stage twice the bytes and clamp to 16 via gdnBlockTFor. GDN inputs are NOT
// always bf16 — an f16 checkpoint promotes its activations to f32.

/// Test seam: false takes the stock recurrence for every prefill width.
pub var gdn_blocked_override: ?bool = null;

pub fn gdnBlockedEnabled() bool {
    return gdn_blocked_override orelse true;
}

/// Prefill-width floor for the blocked kernel (mirrors oMLX's OMLX_GDN_MIN_T
/// default). Below this the per-token stock kernel wins on launch overhead.
pub const GDN_BLOCKED_MIN_T: c_int = 64;

/// Pure routing predicate: geometry + width gate for the blocked-seq kernel.
pub fn gdnBlockedEligible(seq_len: c_int, dk: c_int, dv: c_int, num_k_heads: c_int, num_v_heads: c_int) bool {
    if (seq_len < GDN_BLOCKED_MIN_T) return false;
    if (dk != 128) return false;
    if (dv < 32 or @rem(dv, 32) != 0) return false;
    if (num_k_heads <= 0 or @rem(num_v_heads, num_k_heads) != 0) return false;
    return true;
}

/// Metal's threadgroup memory budget per threadgroup on Apple GPUs. The
/// blocked kernel stages k/q/v in threadgroup memory, so this is a HARD cap on
/// TB x input width — exceed it and the kernel does not compile.
const METAL_TG_BYTES: usize = 32 * 1024;

/// Threadgroup bytes the blocked kernel stages per TB-token block:
/// `k_s`+`q_s` [TB][Dk+8] and `v_s` [TB][DB+8] at the input width, plus the
/// `g_s`/`b_s` [TB] f32 rows. Mirrors the declarations in
/// GDN_KERNEL_BLOCKED_BODY — change one, change the other.
pub fn gdnBlockedTgBytes(tb: u32, dk: c_int, in_elem_size: usize) usize {
    const DB: usize = 32; // dv rows per threadgroup
    const t: usize = tb;
    const k: usize = @intCast(dk);
    return t * (2 * (k + 8) * in_elem_size + (DB + 8) * in_elem_size + 2 * @sizeOf(f32));
}

/// Largest supported block size <= `requested` whose staging fits the budget
/// at this input width, or null when even the smallest does not (decline the
/// blocked route and stay on the stock kernel).
///
/// The width is NOT a constant: bf16 and f16 checkpoints stage 2 bytes and
/// keep the requested size, but an f16 checkpoint promotes its activations to
/// f32 (f16 (x) f32-scalar -> f32), and at 4 bytes TB=32 needs 40 KiB. Block
/// size and input dtype are therefore ONE decision — picking them apart is how
/// a valid-looking config produces a kernel that cannot compile.
pub fn gdnBlockTFor(requested: u32, dk: c_int, in_elem_size: usize) ?u32 {
    var best: ?u32 = null;
    for (GDN_BLOCKED_TBS) |cand| {
        if (cand > requested) continue;
        if (gdnBlockedTgBytes(cand, dk, in_elem_size) > METAL_TG_BYTES) continue;
        if (best == null or cand > best.?) best = cand;
    }
    return best;
}

const GDN_BLOCKED_TBS = [_]u32{ 16, 32, 48 };
var gdn_blocked_cached: [GDN_BLOCKED_TBS.len]?mlx.mlx_fast_metal_kernel = @splat(null);
const GDN_BLOCK_T: u32 = 32;

// Body of the blocked-seq kernel; the `constexpr int TB = N;` line is
// prepended per block size by gdnBlockedSource (one cached kernel per TB,
// DISTINCT names — two specializations sharing a name bind the wrong binary).
// Faithful transcription of oMLX's _KERNEL_S_SRC with one generalization:
// state loads/stores go through vec<StT,4> (theirs hard-codes float4 — their
// state is fp32, ours rides the bf16 ssm_state buffer exactly like the stock
// kernel, so cross-chunk behavior is unchanged).
const GDN_KERNEL_BLOCKED_BODY =
    \\constexpr int DB = 32;                             // dv rows per threadgroup
    \\const int tid = thread_position_in_threadgroup.x;  // 0..255
    \\const int blk = threadgroup_position_in_grid.x;    // Dv/DB block
    \\const int hv  = threadgroup_position_in_grid.y;
    \\const int b   = threadgroup_position_in_grid.z;
    \\const int hk  = hv / (Hv / Hk);
    \\const int dv0 = blk * DB;
    \\
    \\// thread -> (dv row, 16-wide d segment); 8 threads per dv row, all in
    \\// the same simdgroup (lane = (dvr%4)*8 + seg).
    \\const int dvr = tid / 8;            // 0..31
    \\const int seg = tid % 8;            // 0..7
    \\const int d0  = seg * 16;
    \\
    \\threadgroup InT k_s[TB][Dk + 8];
    \\threadgroup InT q_s[TB][Dk + 8];
    \\threadgroup InT v_s[TB][DB + 8];
    \\threadgroup float g_s[TB];
    \\threadgroup float b_s[TB];
    \\
    \\auto k_base = k + ((size_t)b * T * Hk + hk) * Dk;
    \\auto q_base = q + ((size_t)b * T * Hk + hk) * Dk;
    \\auto v_base = v + ((size_t)b * T * Hv + hv) * Dv + dv0;
    \\const size_t krow = (size_t)Hk * Dk;
    \\
    \\// state fragment in registers: [dv0+dvr][d0..d0+16]
    \\float4 st[4];
    \\{
    \\    const device vec<StT,4>* S_in = (const device vec<StT,4>*)(
    \\        state_in + (((size_t)b * Hv + hv) * Dv + dv0 + dvr) * Dk + d0);
    \\    for (int i = 0; i < 4; ++i) st[i] = float4(S_in[i]);
    \\}
    \\
    \\device OutT* y_base = y + ((size_t)b * T * Hv + hv) * Dv + dv0;
    \\
    \\for (int t0 = 0; t0 < T; t0 += TB) {
    \\    const int tt = min(TB, T - t0);
    \\    // cooperative staging (coalesced): k/q rows, v slice, g/beta
    \\    for (int p = tid; p < tt * Dk; p += 256) {
    \\        const int r = p / Dk, d = p % Dk;
    \\        k_s[r][d] = static_cast<InT>(k_base[(size_t)(t0 + r) * krow + d]);
    \\        q_s[r][d] = static_cast<InT>(q_base[(size_t)(t0 + r) * krow + d]);
    \\    }
    \\    for (int p = tid; p < tt * DB; p += 256) {
    \\        const int r = p / DB, d = p % DB;
    \\        v_s[r][d] = static_cast<InT>(v_base[(size_t)(t0 + r) * Hv * Dv + d]);
    \\    }
    \\    for (int p = tid; p < tt; p += 256) {
    \\        g_s[p] = (float)g[((size_t)b * T + t0 + p) * Hv + hv];
    \\        b_s[p] = (float)beta[((size_t)b * T + t0 + p) * Hv + hv];
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\    for (int t = 0; t < tt; ++t) {
    \\        const float gt = g_s[t];
    \\        const float bt = b_s[t];
    \\        const threadgroup vec<InT,4>* k4 =
    \\            (const threadgroup vec<InT,4>*)&k_s[t][d0];
    \\        const threadgroup vec<InT,4>* q4 =
    \\            (const threadgroup vec<InT,4>*)&q_s[t][d0];
    \\        float4 kf[4];
    \\        for (int i = 0; i < 4; ++i) kf[i] = float4(k4[i]);
    \\        // kv_mem = (g*state) . k ; decay applied to state first
    \\        float4 p4 = 0.0f;
    \\        for (int i = 0; i < 4; ++i) {
    \\            st[i] *= gt;
    \\            p4 += st[i] * kf[i];
    \\        }
    \\        float part = p4.x + p4.y + p4.z + p4.w;
    \\        // reduce across the 8 segment-threads of this dv row
    \\        part += simd_shuffle_down(part, 4);
    \\        part += simd_shuffle_down(part, 2);
    \\        part += simd_shuffle_down(part, 1);
    \\        const float kv_mem = simd_shuffle(part, (tid % 32) / 8 * 8);
    \\        const float delta = ((float)v_s[t][dvr] - kv_mem) * bt;
    \\
    \\        float4 o4 = 0.0f;
    \\        for (int i = 0; i < 4; ++i) {
    \\            st[i] += kf[i] * delta;
    \\            o4 += st[i] * float4(q4[i]);
    \\        }
    \\        float out = o4.x + o4.y + o4.z + o4.w;
    \\        out += simd_shuffle_down(out, 4);
    \\        out += simd_shuffle_down(out, 2);
    \\        out += simd_shuffle_down(out, 1);
    \\        if (seg == 0) {
    \\            y_base[(size_t)(t0 + t) * Hv * Dv + dvr] = static_cast<OutT>(out);
    \\        }
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\}
    \\
    \\{
    \\    device vec<StT,4>* S_out = (device vec<StT,4>*)(
    \\        state_out + (((size_t)b * Hv + hv) * Dv + dv0 + dvr) * Dk + d0);
    \\    for (int i = 0; i < 4; ++i) S_out[i] = vec<StT,4>(st[i]);
    \\}
;

fn gdnBlockedSource(comptime tb: u32) [:0]const u8 {
    return std.fmt.comptimePrint("constexpr int TB = {d};\n", .{tb}) ++ GDN_KERNEL_BLOCKED_BODY;
}

fn getGdnKernelBlocked(tb: u32) !mlx.mlx_fast_metal_kernel {
    inline for (GDN_BLOCKED_TBS, 0..) |cand, i| {
        if (cand == tb) {
            if (gdn_blocked_cached[i]) |k| return k;
            const input_names = [_][*:0]const u8{ "q", "k", "v", "g", "beta", "state_in", "T" };
            const output_names = [_][*:0]const u8{ "y", "state_out" };
            const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
            defer _ = mlx.mlx_vector_string_free(in_vec);
            const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
            defer _ = mlx.mlx_vector_string_free(out_vec);
            const kernel = mlx.mlx_fast_metal_kernel_new(
                std.fmt.comptimePrint("gated_delta_blocked_tb{d}", .{cand}),
                in_vec,
                out_vec,
                gdnBlockedSource(cand),
                "",
                true,
                false,
            );
            if (kernel.ctx == null) return error.MetalKernelCompileFailed;
            gdn_blocked_cached[i] = kernel;
            return kernel;
        }
    }
    return error.UnsupportedGdnBlockT;
}

pub const GdnRoute = enum { stock, blocked, pipelined };

var gdn_pipelined_cached: ?mlx.mlx_fast_metal_kernel = null;

fn gdnPipelinedEnabled(model_type: []const u8) bool {
    return verifyQmmNaxAvailable() and std.mem.eql(u8, model_type, "qwen4_exp");
}

fn getGdnKernelPipelined() !mlx.mlx_fast_metal_kernel {
    if (gdn_pipelined_cached) |k| return k;
    const input_names = [_][*:0]const u8{ "q", "k", "v", "g", "beta", "state_in", "T" };
    const output_names = [_][*:0]const u8{ "y", "state_out" };
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new("gated_delta_pipelined_tb12", in_vec, out_vec, @embedFile("../kernels/gdn_pipelined.metal"), "", true, false);
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    gdn_pipelined_cached = kernel;
    return kernel;
}

pub fn gdnRouteKernel(route: GdnRoute, tb: u32, vector_gate: bool) !mlx.mlx_fast_metal_kernel {
    return switch (route) {
        .stock => getGdnKernel(vector_gate),
        .blocked => getGdnKernelBlocked(tb),
        .pipelined => getGdnKernelPipelined(),
    };
}

pub fn gdnSetGrid(config: mlx.mlx_fast_metal_kernel_config, route: GdnRoute, dv: c_int, hv: c_int, batch: c_int) !void {
    switch (route) {
        .stock => {
            // Grid: (32, Dv, B*Hv) threads; threadgroup: (32, 4, 1). Matches mlx-lm.
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32, dv, batch * hv));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, 4, 1));
        },
        .blocked => {
            // Grid: (256*(Dv/32), Hv, B) threads; threadgroup (256,1,1).
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 256 * @divExact(dv, 32), hv, batch));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1));
        },
        .pipelined => {
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 128 * @divExact(dv, 16), hv, batch));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 128, 1, 1));
        },
    }
}

const GdnPrefillRoute = struct { route: GdnRoute, tb: u32 = 0 };

pub fn gdnPrefillRoute(model_type: []const u8, seq_len: c_int, dk: c_int, dv: c_int, num_k_heads: c_int, num_v_heads: c_int, vector_gate: bool, in_itemsize: usize) GdnPrefillRoute {
    if (vector_gate or !gdnBlockedEnabled() or !gdnBlockedEligible(seq_len, dk, dv, num_k_heads, num_v_heads)) return .{ .route = .stock };
    const tb = gdnBlockTFor(GDN_BLOCK_T, dk, in_itemsize) orelse return .{ .route = .stock };
    return .{ .route = if (gdnPipelinedEnabled(model_type)) .pipelined else .blocked, .tb = tb };
}

pub var gdn_batched_logged: bool = false;
pub var gdn_batched_verify_logged: bool = false;

/// MoE routing chain (negate→argpartition→slice→softmax→take→sum→expand→divide).
/// Free-function variant of `Transformer.moeRoutingUncompiled` so unit tests can
/// exercise the pure subgraph without constructing a full Transformer. Returns
/// owned `inds` (int32, [..., k]) and `norm_scores` (bf16, [..., k]) — caller
/// must free both.
/// GatedDeltaNet gating chain: g = exp(-exp(A_log) * softplus(a + dt_bias)),
/// computed in float32 for stability and returned as `gate_dtype` (an f32 gate also adds
/// a + dt_bias in f32, as the HF reference does). Mirrors mlx-lm's `compute_g` (which is
/// `@mx.compile`d). Pure — serves as both the compiled-closure body and the uncompiled
/// fallback. Returns owned array.
pub fn gdnGateChain(A_log: mlx.mlx_array, a: mlx.mlx_array, dt_bias: mlx.mlx_array, gate_dtype: mlx.mlx_dtype, s: mlx.mlx_stream) !mlx.mlx_array {
    var A_log_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(A_log_f32);
    try mlx.check(mlx.mlx_astype(&A_log_f32, A_log, .float32, s));
    var exp_A = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(exp_A);
    try mlx.check(mlx.mlx_exp(&exp_A, A_log_f32, s));

    // softplus(a + dt_bias) = log1p(exp(a + dt_bias))
    var a_plus_dt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a_plus_dt);
    var a_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a_f32);
    if (gate_dtype == .float32) {
        var a_wide = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(a_wide);
        var dt_wide = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dt_wide);
        try mlx.check(mlx.mlx_astype(&a_wide, a, .float32, s));
        try mlx.check(mlx.mlx_astype(&dt_wide, dt_bias, .float32, s));
        try mlx.check(mlx.mlx_add(&a_f32, a_wide, dt_wide, s));
    } else {
        try mlx.check(mlx.mlx_add(&a_plus_dt, a, dt_bias, s));
        try mlx.check(mlx.mlx_astype(&a_f32, a_plus_dt, .float32, s));
    }
    var exp_a = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(exp_a);
    try mlx.check(mlx.mlx_exp(&exp_a, a_f32, s));
    var sp_inner = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sp_inner);
    try mlx.check(mlx.mlx_log1p(&sp_inner, exp_a, s));

    var neg_decay = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(neg_decay);
    try mlx.check(mlx.mlx_multiply(&neg_decay, exp_A, sp_inner, s));
    var neg_neg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(neg_neg);
    try mlx.check(mlx.mlx_negative(&neg_neg, neg_decay, s));
    var g_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g_f32);
    try mlx.check(mlx.mlx_exp(&g_f32, neg_neg, s));
    var g = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(g);
    try mlx.check(mlx.mlx_astype(&g, g_f32, gate_dtype, s));
    return g;
}

const GDN_PREWORK_SOURCE =
    \\uint lane = thread_position_in_threadgroup.x;
    \\// Grid row = b*S + r over the batch: q/k/v/g/beta/b/a are [B,S,..] flat
    \\// so `row` indexes them directly; the conv taps + next state are per batch.
    \\uint row = threadgroup_position_in_grid.y;
    \\uint b = row / uint(seq);
    \\uint r = row - b * uint(seq);
    \\uint logical_head = threadgroup_position_in_grid.z;
    \\constexpr uint q_heads = uint(HK);
    \\constexpr uint k_head_base = uint(HK);
    \\constexpr uint v_head_base = 2 * uint(HK);
    \\bool is_q = logical_head < q_heads;
    \\bool is_k = logical_head >= k_head_base && logical_head < v_head_base;
    \\uint head = is_q ? logical_head
    \\           : (is_k ? logical_head - k_head_base : logical_head - v_head_base);
    \\uint channel_base = is_q ? head * uint(DK)
    \\                   : (is_k ? uint(HK) * uint(DK) + head * uint(DK)
    \\                           : 2 * uint(HK) * uint(DK) + head * uint(DV));
    \\T activated[4];
    \\float sumsq = 0.0f;
    \\for (uint i = 0; i < 4; ++i) {
    \\    uint channel = channel_base + lane * 4 + i;
    \\    float acc = 0.0f;
    \\    for (uint tap = 0; tap < 4; ++tap) {
    \\        uint input_row = r + tap;
    \\        const T xv = input_row < uint(NKEEP)
    \\            ? conv_state[(b * uint(NKEEP) + input_row) * uint(C) + channel]
    \\            : qkv[(row + tap - uint(NKEEP)) * uint(QSTRIDE) + uint(QOFF) + channel];
    \\        acc += float(xv) * float(conv_w[channel * 4 + tap]);
    \\    }
    \\    const T conv = T(acc);
    \\    // MLX's unary Sigmoid formula (unary_ops.h), verbatim, in the tensor dtype.
    \\    T sig;
    \\    if constexpr (TAB) sig = sigtab[as_type<ushort>(conv)];
    \\    else { T sy = T(1) / (T(1) + metal::exp(metal::abs(conv))); sig = conv < T(0) ? sy : T(1) - sy; }
    \\    const T act = conv * sig;
    \\    activated[i] = act;
    \\    float value = float(act);
    \\    sumsq += value * value;
    \\}
    \\if (is_q || is_k) {
    \\    sumsq = simd_sum(sumsq);
    \\    float inv = metal::precise::rsqrt(sumsq / float(DK) + 1e-6f);
    \\    const T scale = is_q ? q_scale : k_scale;
    \\    uint out_base = (row * uint(HK) + head) * uint(DK) + lane * 4;
    \\    for (uint i = 0; i < 4; ++i) {
    \\        // ones-weight rms_norm rounding (T(x*inv)), then the separate
    \\        // scalar multiply's rounding — the composed chain's two casts.
    \\        const T rms = T(1) * T(float(activated[i]) * inv);
    \\        const T value = scale * rms;
    \\        if (is_q) {
    \\            q_out[out_base + i] = value;
    \\        } else {
    \\            k_out[out_base + i] = value;
    \\        }
    \\    }
    \\} else {
    \\    uint out_base = (row * uint(HV) + head) * uint(DV) + lane * 4;
    \\    for (uint i = 0; i < 4; ++i) {
    \\        v_out[out_base + i] = activated[i];
    \\    }
    \\    if (lane == 0) {
    \\        // beta = sigmoid(b) (MLX unary formula); g = exp(-exp(A_log) *
    \\        // softplus(a + dt_bias)) with the compiled chain's own casts:
    \\        // GT add, f32 precise exp / log1p / exp, GT store.
    \\        const T bv = b_in[row * uint(BSTRIDE) + uint(BOFF) + head];
    \\        T bsig;
    \\        if constexpr (TAB) bsig = sigtab[as_type<ushort>(bv)];
    \\        else { T by = T(1) / (T(1) + metal::exp(metal::abs(bv))); bsig = bv < T(0) ? by : T(1) - by; }
    \\        beta_out[row * uint(HV) + head] = bsig;
    \\        const GT apd = GT(float(a_in[row * uint(ASTRIDE) + uint(AOFF) + head]) + float(dt_bias[head]));
    \\        float sp = sushi_log1p(metal::precise::exp(float(apd)));
    \\        float ea = metal::precise::exp(float(A_log[head]));
    \\        g_out[row * uint(HV) + head] = GT(metal::precise::exp(-(ea * sp)));
    \\    }
    \\}
    \\// Next conv state = rows [S, S+NKEEP) of concat(conv_state, qkv).
    \\if (r + uint(NKEEP) >= uint(seq)) {
    \\    uint state_row = r + uint(NKEEP) - uint(seq);
    \\    uint raw_base = row * uint(QSTRIDE) + uint(QOFF) + channel_base + lane * 4;
    \\    uint state_base = (b * uint(NKEEP) + state_row) * uint(C) + channel_base + lane * 4;
    \\    for (uint i = 0; i < 4; ++i) {
    \\        conv_out[state_base + i] = qkv[raw_base + i];
    \\    }
    \\}
    \\if (r == 0) {
    \\    for (uint rr = 0; rr + uint(seq) < uint(NKEEP); ++rr) {
    \\        uint src_base = (b * uint(NKEEP) + rr + uint(seq)) * uint(C) + channel_base + lane * 4;
    \\        uint dst_base = (b * uint(NKEEP) + rr) * uint(C) + channel_base + lane * 4;
    \\        for (uint i = 0; i < 4; ++i) {
    \\            conv_out[dst_base + i] = conv_state[src_base + i];
    \\        }
    \\    }
    \\}
;

var gdn_prework_kernel: ?mlx.mlx_fast_metal_kernel = null;
var gdn_prework_engaged: bool = false;
pub var gdn_prework_override: ?bool = null; // test seam

fn gdnPreworkEnabled() bool {
    return gdn_prework_override orelse true;
}

pub var gdn_decode_fused_override: ?bool = null; // test seam

/// The decode-width GDN fusions (prework at S < 3, in-kernel gate/beta, the
/// norm-gate epilogue). Off = the pre-fusion decode path, op for op.
pub fn gdnDecodeFusedEnabled() bool {
    return gdn_decode_fused_override orelse true;
}

pub var gdn_decode_recur_override: ?bool = null; // test seam
pub var gdn_decode_recur_engaged: bool = false;
pub var gdn_verify_recur_engaged: bool = false;
pub var gdn_verify_fold_override: ?bool = null; // parity test seam
pub var gdn_verify_fold_engaged = false;

pub fn gdnVerifyFoldFor(seq_len: c_int) bool {
    // Wider captures increase shared-memory pressure; keep their existing path.
    return seq_len >= 2 and seq_len <= gdn_decode.MAX_SEQ and (gdn_verify_fold_override orelse (seq_len == 2));
}

pub var gdn_check_env: ?bool = null;

/// `gdn_decode.step` + the norm-gate in place of prework -> recurrence -> norm-gate, so it
/// needs both of that chain's fusions.
pub fn gdnDecodeRecurEnabled() bool {
    return (gdn_decode_recur_override orelse true) and gdnDecodeFusedEnabled() and gdnPreworkEnabled();
}

fn getGdnPreworkKernel() !mlx.mlx_fast_metal_kernel {
    if (gdn_prework_kernel) |k| return k;
    const input_names = [_][*:0]const u8{ "qkv", "conv_state", "conv_w", "q_scale", "k_scale", "b_in", "a_in", "A_log", "dt_bias", "sigtab", "seq" };
    const output_names = [_][*:0]const u8{ "q_out", "k_out", "v_out", "conv_out", "g_out", "beta_out" };
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "sushi_gdn_prework2",
        in_vec,
        out_vec,
        GDN_PREWORK_SOURCE,
        GDN_KERNEL_HEADER,
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    gdn_prework_kernel = kernel;
    return kernel;
}

const GdnPreworkCfgKey = struct { hk: c_int, hv: c_int, seq: c_int, batch: c_int, c: c_int, qs: c_int, qo: c_int, bs: c_int, bo: c_int, as: c_int, ao: c_int, gate: mlx.mlx_dtype };
var gdn_prework_cfg: ?mlx.mlx_fast_metal_kernel_config = null;
var gdn_prework_cfg_key: GdnPreworkCfgKey = std.mem.zeroes(GdnPreworkCfgKey);

pub const GdnPrework = struct {
    q: mlx.mlx_array, // [B, S, Hk, 128] normed + scaled
    k: mlx.mlx_array, // [B, S, Hk, 128] normed + scaled
    v: mlx.mlx_array, // [B, S, Hv, 128] silu-activated
    conv_state: mlx.mlx_array, // [B, NKEEP, C] next conv state
    g: mlx.mlx_array, // [B, S, Hv] forget gate in `GdnPreworkArgs.gate_dtype`
    beta: mlx.mlx_array, // [B, S, Hv] sigmoid(b)
};
/// Rows (batch*seq) the decode-width GDN fusions serve.
pub const GDN_FUSED_MAX_ROWS: c_int = 16;

var gdn_prefill_fused_override: ?bool = null;
pub var gdn_prefill_engaged = false;
pub var gdn_state_f32_logged = false;

pub fn gdnPrefillFusedFor(seq: c_int, batch: c_int) bool {
    if (seq < 17 or batch < 1 or batch > 2 or seq > @divTrunc(@import("../hc_prefill.zig").max_seq, batch)) return false;
    return gdn_prefill_fused_override orelse true;
}

/// Inputs of the packed prework. `qkv`/`b`/`a` are read at `(off, stride)`
/// per row, so they may all alias one folded in_proj output.
pub const GdnPreworkArgs = struct {
    qkv: mlx.mlx_array, // rows of [.. QOFF + C ..] bf16
    qkv_off: c_int,
    qkv_stride: c_int,
    b: mlx.mlx_array, // rows of [.. BOFF + Hv ..] bf16
    b_off: c_int,
    b_stride: c_int,
    a: mlx.mlx_array,
    a_off: c_int,
    a_stride: c_int,
    A_log: mlx.mlx_array, // [Hv]
    dt_bias: mlx.mlx_array, // [Hv] bf16
    conv_state: mlx.mlx_array, // [B, NKEEP, C] bf16
    conv_w: mlx.mlx_array, // [C, 4, 1] bf16
    q_scale: mlx.mlx_array, // 0-dim bf16 (1/dk)
    k_scale: mlx.mlx_array, // 0-dim bf16 (1/sqrt(dk))
    hk: c_int,
    hv: c_int,
    dk: c_int,
    dv: c_int,
    seq: c_int,
    batch: c_int = 1,
    gate_dtype: mlx.mlx_dtype = .bfloat16, // `ModelConfig.gdnGateDtype()`
};

/// One fused dispatch for the GDN prework at S 1..9 x B batched slots
/// (batch*seq <= GDN_FUSED_MAX_ROWS; per-head gate archs only). Null →
/// caller keeps the composed chain. The caller owns all six outputs (and
/// installs `conv_state` into the SSM cache entry).
pub fn gdnPreworkFused(s: mlx.mlx_stream, in: GdnPreworkArgs) !?GdnPrework {
    if (!gdnPreworkEnabled()) return null;
    if (!gdnPrefillFusedFor(in.seq, in.batch) and (in.seq < 1 or in.seq > 9 or in.batch < 1 or in.batch * in.seq > GDN_FUSED_MAX_ROWS)) return null;
    if (in.seq < 3 and !gdnDecodeFusedEnabled()) return null;
    if (in.dk != 128 or in.dv != 128 or (in.gate_dtype != .bfloat16 and in.gate_dtype != .float32)) return null;
    inline for (.{ in.qkv, in.b, in.a, in.conv_state, in.conv_w, in.dt_bias }) |arr| {
        if (mlx.mlx_array_dtype(arr) != .bfloat16) return null;
    }
    const wsh = mlx.getShape(in.conv_w);
    if (wsh.len < 2 or wsh[1] != 4) return null;
    const c_dim: c_int = in.hk * in.dk * 2 + in.hv * in.dv;
    const qsh = mlx.getShape(in.qkv);
    if (qsh.len != 3 or qsh[0] != in.batch or qsh[1] != in.seq or qsh[2] != in.qkv_stride or in.qkv_off + c_dim > in.qkv_stride) return null;
    const bsh = mlx.getShape(in.b);
    if (bsh.len != 3 or bsh[0] != in.batch or bsh[1] != in.seq or bsh[2] != in.b_stride or in.b_off + in.hv > in.b_stride) return null;
    const ash = mlx.getShape(in.a);
    if (ash.len != 3 or ash[0] != in.batch or ash[1] != in.seq or ash[2] != in.a_stride or in.a_off + in.hv > in.a_stride) return null;
    const csh = mlx.getShape(in.conv_state);
    if (csh.len != 3 or csh[0] != in.batch or csh[1] != 3 or csh[2] != c_dim) return null;

    const nkeep: c_int = 3;
    const key = GdnPreworkCfgKey{ .hk = in.hk, .hv = in.hv, .seq = in.seq, .batch = in.batch, .c = c_dim, .qs = in.qkv_stride, .qo = in.qkv_off, .bs = in.b_stride, .bo = in.b_off, .as = in.a_stride, .ao = in.a_off, .gate = in.gate_dtype };
    if (gdn_prework_cfg == null or !std.meta.eql(gdn_prework_cfg_key, key)) {
        if (gdn_prework_cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const config = mlx.mlx_fast_metal_kernel_config_new();
        const q_shape = [_]c_int{ in.batch, in.seq, in.hk, in.dk };
        const v_shape = [_]c_int{ in.batch, in.seq, in.hv, in.dv };
        const st_shape = [_]c_int{ in.batch, nkeep, c_dim };
        const g_shape = [_]c_int{ in.batch, in.seq, in.hv };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &q_shape, 4, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &q_shape, 4, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &v_shape, 4, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &st_shape, 3, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &g_shape, 3, in.gate_dtype));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &g_shape, 3, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32, in.batch * in.seq, 2 * in.hk + in.hv));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "GT", in.gate_dtype));
        const ints = .{ .{ "HK", in.hk }, .{ "HV", in.hv }, .{ "DK", in.dk }, .{ "DV", in.dv }, .{ "NKEEP", nkeep }, .{ "C", c_dim }, .{ "TAB", @as(c_int, @intFromBool(in.seq >= 17)) }, .{ "QSTRIDE", in.qkv_stride }, .{ "QOFF", in.qkv_off }, .{ "BSTRIDE", in.b_stride }, .{ "BOFF", in.b_off }, .{ "ASTRIDE", in.a_stride }, .{ "AOFF", in.a_off } };
        inline for (ints) |kv| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, kv[0], kv[1]));
        gdn_prework_cfg = config;
        gdn_prework_cfg_key = key;
    }

    const kernel = try getGdnPreworkKernel();
    const seq_arr = mlx.mlx_array_new_int(in.seq);
    defer _ = mlx.mlx_array_free(seq_arr);
    const inputs_arr = [_]mlx.mlx_array{ in.qkv, in.conv_state, in.conv_w, in.q_scale, in.k_scale, in.b, in.a, in.A_log, in.dt_bias, if (in.seq >= 17) try @import("../hc_prefill.zig").sigmoidTable(s) else in.qkv, seq_arr };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, gdn_prework_cfg.?, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 6) return error.MetalKernelBadOutputCount;
    var out = GdnPrework{
        .q = mlx.mlx_array_new(),
        .k = mlx.mlx_array_new(),
        .v = mlx.mlx_array_new(),
        .conv_state = mlx.mlx_array_new(),
        .g = mlx.mlx_array_new(),
        .beta = mlx.mlx_array_new(),
    };
    errdefer {
        inline for (.{ &out.q, &out.k, &out.v, &out.conv_state, &out.g, &out.beta }) |p| {
            if (p.*.ctx != null) _ = mlx.mlx_array_free(p.*);
        }
    }
    try mlx.check(mlx.mlx_vector_array_get(&out.q, outputs_vec, 0));
    try mlx.check(mlx.mlx_vector_array_get(&out.k, outputs_vec, 1));
    try mlx.check(mlx.mlx_vector_array_get(&out.v, outputs_vec, 2));
    try mlx.check(mlx.mlx_vector_array_get(&out.conv_state, outputs_vec, 3));
    try mlx.check(mlx.mlx_vector_array_get(&out.g, outputs_vec, 4));
    try mlx.check(mlx.mlx_vector_array_get(&out.beta, outputs_vec, 5));
    if (!gdn_prework_engaged) {
        gdn_prework_engaged = true;
        log.info("[gdn] packed prework engaged: Hk={d} Hv={d} S={d} B={d}\n", .{ in.hk, in.hv, in.seq, in.batch });
    }
    return out;
}

// ── Fused GDN norm-gate epilogue ──
// rms_norm(y, norm_w) * silu(z) → flat [1, S, Hv*Dv] in ONE launch (the
// composed chain is rms_norm + sigmoid + multiply + multiply). Mirrors
// `rms_single_row`'s reduction (per-lane 4-element partials, simd_sum,
// precise rsqrt, `w * T(x * inv)`) and MLX's sigmoid, so it is bit-identical.
// z is read at (ZOFF, ZSTRIDE) so a wider projection row can hand it over.
const GDN_NORMGATE_SOURCE =
    \\uint lane = thread_position_in_threadgroup.x;
    \\uint row = threadgroup_position_in_grid.y;
    \\uint head = threadgroup_position_in_grid.z;
    \\uint base = (row * uint(HV) + head) * uint(DV) + lane * 4;
    \\float xs[4];
    \\float sumsq = 0.0f;
    \\for (uint i = 0; i < 4; ++i) {
    \\    xs[i] = float(y[base + i]);
    \\    sumsq += xs[i] * xs[i];
    \\}
    \\sumsq = simd_sum(sumsq);
    \\float inv = metal::precise::rsqrt(sumsq / float(DV) + eps);
    \\uint zbase = row * uint(ZSTRIDE) + uint(ZOFF) + head * uint(DV) + lane * 4;
    \\for (uint i = 0; i < 4; ++i) {
    \\    const T normed = norm_w[lane * 4 + i] * T(xs[i] * inv);
    \\    const T zv = z[zbase + i];
    \\    T sig;
    \\    if constexpr (TAB) sig = sigtab[as_type<ushort>(zv)];
    \\    else { T sy = T(1) / (T(1) + metal::exp(metal::abs(zv))); sig = zv < T(0) ? sy : T(1) - sy; }
    \\    // swish gate: silu(z) * normed (qwen3.5); sigmoid gate: normed * sigmoid(z) (qwen4_exp, KDA)
    \\    out[base + i] = SWISH ? (zv * sig) * normed : normed * sig;
    \\}
;

var gdn_normgate_kernel: ?mlx.mlx_fast_metal_kernel = null;
var gdn_normgate_engaged: bool = false;
var gdn_normgate_declined: bool = false;
const GdnNormGateCfgKey = struct { hv: c_int, seq: c_int, batch: c_int, zs: c_int, zo: c_int, swish: bool };
var gdn_normgate_cfg: ?mlx.mlx_fast_metal_kernel_config = null;
var gdn_normgate_cfg_key: GdnNormGateCfgKey = std.mem.zeroes(GdnNormGateCfgKey);

fn getGdnNormGateKernel() !mlx.mlx_fast_metal_kernel {
    if (gdn_normgate_kernel) |k| return k;
    const input_names = [_][*:0]const u8{ "y", "z", "norm_w", "eps", "sigtab" };
    const output_names = [_][*:0]const u8{"out"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new("sushi_gdn_normgate", in_vec, out_vec, GDN_NORMGATE_SOURCE, "", true, false);
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    gdn_normgate_kernel = kernel;
    return kernel;
}

/// `rms_norm(y) * silu(z)` flattened to [B, S, Hv*Dv]. Null → composed chain.
pub fn gdnNormGateFused(
    s: mlx.mlx_stream,
    y: mlx.mlx_array, // [B, S, Hv, Dv] bf16
    z: mlx.mlx_array, // rows of [.. ZOFF + Hv*Dv ..] bf16
    z_off: c_int,
    z_stride: c_int,
    norm_w: mlx.mlx_array, // [Dv] bf16
    eps: mlx.mlx_array, // 0-dim f32
    swish: bool, // true: silu(z) * y (qwen3.5); false: y * sigmoid(z) (qwen4_exp, KDA)
    hv: c_int,
    dv: c_int,
    batch: c_int,
    seq: c_int,
) !?mlx.mlx_array {
    if (dv != 128) return null;
    const prefill = gdnPrefillFusedFor(seq, batch);
    if (!prefill and !gdnDecodeFusedEnabled()) return null;
    if (!prefill and (seq < 1 or seq > 9 or batch < 1 or batch * seq > GDN_FUSED_MAX_ROWS)) return null;
    inline for (.{ y, z, norm_w }, 0..) |arr, i| {
        if (mlx.mlx_array_dtype(arr) != .bfloat16) {
            if (!gdn_normgate_declined) {
                gdn_normgate_declined = true;
                log.info("[gdn] fused norm-gate declined: input {d} dtype {s}\n", .{ i, @tagName(mlx.mlx_array_dtype(arr)) });
            }
            return null;
        }
    }
    const ysh = mlx.getShape(y);
    const zsh = mlx.getShape(z);
    if (ysh.len != 4 or ysh[0] != batch or ysh[1] != seq or ysh[2] != hv or ysh[3] != dv or zsh.len != 3 or zsh[0] != batch or zsh[1] != seq or zsh[2] != z_stride or z_off + hv * dv > z_stride) {
        if (!gdn_normgate_declined) {
            gdn_normgate_declined = true;
            log.info("[gdn] fused norm-gate declined: y {any} z {any} z_off={d} z_stride={d} hv={d} dv={d} seq={d}\n", .{ ysh, zsh, z_off, z_stride, hv, dv, seq });
        }
        return null;
    }
    const key = GdnNormGateCfgKey{ .hv = hv, .seq = seq, .batch = batch, .zs = z_stride, .zo = z_off, .swish = swish };
    if (gdn_normgate_cfg == null or !std.meta.eql(gdn_normgate_cfg_key, key)) {
        if (gdn_normgate_cfg) |c| _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const config = mlx.mlx_fast_metal_kernel_config_new();
        const out_shape = [_]c_int{ batch, seq, hv * dv };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &out_shape, 3, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32, batch * seq, hv));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", .bfloat16));
        const ints = .{ .{ "HV", hv }, .{ "DV", dv }, .{ "ZSTRIDE", z_stride }, .{ "ZOFF", z_off }, .{ "TAB", @as(c_int, @intFromBool(seq >= 17)) } };
        inline for (ints) |kv| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, kv[0], kv[1]));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "SWISH", @intFromBool(swish)));
        gdn_normgate_cfg = config;
        gdn_normgate_cfg_key = key;
    }
    const kernel = try getGdnNormGateKernel();
    const inputs_arr = [_]mlx.mlx_array{ y, z, norm_w, eps, if (seq >= 17) try @import("../hc_prefill.zig").sigmoidTable(s) else y };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, gdn_normgate_cfg.?, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 0));
    if (!gdn_normgate_engaged) {
        gdn_normgate_engaged = true;
        log.info("[gdn] fused norm-gate engaged: Hv={d} S={d} gate={s}\n", .{ hv, seq, if (swish) "swish" else "sigmoid" });
    }
    return out;
}

/// `arr[:, :, start..stop]` as a lazy view.
fn sliceLastDim(arr: mlx.mlx_array, start: c_int, stop: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const sh = mlx.getShape(arr);
    if (sh.len != 3) return error.BadShape;
    const st = [_]c_int{ 0, 0, start };
    const en = [_]c_int{ sh[0], sh[1], stop };
    const strides = [_]c_int{ 1, 1, 1 };
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&out, arr, &st, 3, &en, 3, &strides, 3, s));
    return out;
}

fn gdnTestRun(comptime seq: bool, comptime vec: bool, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, g: mlx.mlx_array, beta: mlx.mlx_array, state_in: mlx.mlx_array, B: c_int, T: c_int, Hk: c_int, Hv: c_int, Dk: c_int, Dv: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const T_scalar = mlx.mlx_array_new_int(T);
    defer _ = mlx.mlx_array_free(T_scalar);
    const y_shape = [_]c_int{ B, T, Hv, Dv };
    const config = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &y_shape, 4, .bfloat16));
    if (seq) {
        const ss_shape = [_]c_int{ T, B, Hv, Dv, Dk };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &ss_shape, 5, .bfloat16));
        // Final state is its own output (written from registers); the seq
        // buffer's last row is deliberately UNWRITTEN (capture-tail trim).
        const so_shape = [_]c_int{ B, Hv, Dv, Dk };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &so_shape, 4, .bfloat16));
    } else {
        const so_shape = [_]c_int{ B, Hv, Dv, Dk };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &so_shape, 4, .bfloat16));
    }
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32, Dv, B * Hv));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, 4, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "InT", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "StT", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "OutT", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dk", Dk));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dv", Dv));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hk", Hk));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hv", Hv));

    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    if (seq) {
        const seq_stride = mlx.mlx_array_new_int(B * Hv * Dv * Dk);
        defer _ = mlx.mlx_array_free(seq_stride);
        const inputs_arr = [_]mlx.mlx_array{ q, k, v, g, beta, state_in, T_scalar, seq_stride };
        const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
        defer _ = mlx.mlx_vector_array_free(inputs_vec);
        const kern = try getGdnKernelSeq(vec);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kern, inputs_vec, config, s));
        // Final state comes from the dedicated state_out output — after the
        // capture-tail trim, state_seq[T-1] is never written.
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 2));
        return out;
    } else {
        const inputs_arr = [_]mlx.mlx_array{ q, k, v, g, beta, state_in, T_scalar };
        const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
        defer _ = mlx.mlx_vector_array_free(inputs_vec);
        const kern = try getGdnKernel(vec);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kern, inputs_vec, config, s));
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 1));
        return out;
    }
}

fn gdnTestRunSerial(comptime vec: bool, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, g: mlx.mlx_array, beta: mlx.mlx_array, initial: mlx.mlx_array, B: c_int, T: c_int, Hk: c_int, Hv: c_int, Dk: c_int, Dv: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    var state = try standinRef(initial);
    errdefer _ = mlx.mlx_array_free(state);
    const sources = [_]mlx.mlx_array{ q, k, v, g, beta };
    for (0..@intCast(T)) |t| {
        var rows: [5]mlx.mlx_array = @splat(.{});
        defer for (rows) |row| {
            if (row.ctx != null) _ = mlx.mlx_array_free(row);
        };
        for (sources, 0..) |src, i| {
            const shape = mlx.getShape(src);
            var starts: [4]c_int = @splat(0);
            var stops: [4]c_int = @splat(1);
            const steps: [4]c_int = @splat(1);
            @memcpy(stops[0..shape.len], shape);
            starts[1] = @intCast(t);
            stops[1] = @intCast(t + 1);
            rows[i] = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(&rows[i], src, &starts, shape.len, &stops, shape.len, &steps, shape.len, s));
        }
        const next = try gdnTestRun(false, vec, rows[0], rows[1], rows[2], rows[3], rows[4], state, B, 1, Hk, Hv, Dk, Dv, s);
        _ = mlx.mlx_array_free(state);
        state = next;
    }
    return state;
}

fn gdnTestRunSeqAt(q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, g: mlx.mlx_array, beta: mlx.mlx_array, state_in: mlx.mlx_array, B: c_int, T: c_int, Hk: c_int, Hv: c_int, Dk: c_int, Dv: c_int, pos: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const T_scalar = mlx.mlx_array_new_int(T);
    defer _ = mlx.mlx_array_free(T_scalar);
    const y_shape = [_]c_int{ B, T, Hv, Dv };
    const config = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &y_shape, 4, .bfloat16));
    const ss_shape = [_]c_int{ T, B, Hv, Dv, Dk };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &ss_shape, 5, .bfloat16));
    const so_shape = [_]c_int{ B, Hv, Dv, Dk };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &so_shape, 4, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32, Dv, B * Hv));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, 4, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "InT", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "StT", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "OutT", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dk", Dk));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dv", Dv));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hk", Hk));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hv", Hv));
    const seq_stride = mlx.mlx_array_new_int(B * Hv * Dv * Dk);
    defer _ = mlx.mlx_array_free(seq_stride);
    const inputs_arr = [_]mlx.mlx_array{ q, k, v, g, beta, state_in, T_scalar, seq_stride };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    const kern = try getGdnKernelSeq(false);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kern, inputs_vec, config, s));
    var ss = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ss);
    try mlx.check(mlx.mlx_vector_array_get(&ss, outputs_vec, 1));
    const start = [_]c_int{ pos, 0, 0, 0, 0 };
    const stop = [_]c_int{ pos + 1, B, Hv, Dv, Dk };
    const strides = [_]c_int{ 1, 1, 1, 1, 1 };
    var sliced = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sliced);
    try mlx.check(mlx.mlx_slice(&sliced, ss, &start, 5, &stop, 5, &strides, 5, s));
    const fshape = [_]c_int{ B, Hv, Dv, Dk };
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&out, sliced, &fshape, 4, s));
    return out;
}

const GdnHostRef = struct { y: []f32, state: []f32 };

/// f64 host ground truth of the exact GDN recurrence (the stock kernel's
/// math, per house rule: GPU parity is judged vs fp32+ ground truth, never
/// kernel-vs-kernel bf16 agreement).
fn gdnHostRef(
    al: std.mem.Allocator,
    qd: []const f32,
    kd: []const f32,
    vd: []const f32,
    gd: []const f32,
    bd: []const f32,
    sd: []const f32,
    B: usize,
    T: usize,
    Hk: usize,
    Hv: usize,
    Dk: usize,
    Dv: usize,
) !GdnHostRef {
    const y = try al.alloc(f32, B * T * Hv * Dv);
    errdefer al.free(y);
    const st_out = try al.alloc(f32, B * Hv * Dv * Dk);
    errdefer al.free(st_out);
    const S = try al.alloc(f64, Dv * Dk);
    defer al.free(S);
    const group = Hv / Hk;
    for (0..B) |b| {
        for (0..Hv) |hv| {
            const hk = hv / group;
            for (0..Dv) |dvi| {
                for (0..Dk) |dki| S[dvi * Dk + dki] = sd[((b * Hv + hv) * Dv + dvi) * Dk + dki];
            }
            for (0..T) |t| {
                const gt: f64 = gd[(b * T + t) * Hv + hv];
                const bt: f64 = bd[(b * T + t) * Hv + hv];
                const kbase = ((b * T + t) * Hk + hk) * Dk;
                for (S) |*x| x.* *= gt;
                for (0..Dv) |dvi| {
                    var kv: f64 = 0;
                    for (0..Dk) |dki| kv += S[dvi * Dk + dki] * @as(f64, kd[kbase + dki]);
                    const delta = (@as(f64, vd[((b * T + t) * Hv + hv) * Dv + dvi]) - kv) * bt;
                    var out: f64 = 0;
                    for (0..Dk) |dki| {
                        S[dvi * Dk + dki] += @as(f64, kd[kbase + dki]) * delta;
                        out += S[dvi * Dk + dki] * @as(f64, qd[kbase + dki]);
                    }
                    y[((b * T + t) * Hv + hv) * Dv + dvi] = @floatCast(out);
                }
            }
            for (0..Dv) |dvi| {
                for (0..Dk) |dki| st_out[((b * Hv + hv) * Dv + dvi) * Dk + dki] = @floatCast(S[dvi * Dk + dki]);
            }
        }
    }
    return .{ .y = y, .state = st_out };
}

const GdnRunOut = struct { y: mlx.mlx_array, state: mlx.mlx_array };

/// Run the GDN recurrence via the stock single-state kernel (blocked=false)
/// or the blocked-seq prefill kernel (blocked=true) and return BOTH outputs.
fn gdnRunYState(route: GdnRoute, tb_requested: u32, q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, g: mlx.mlx_array, beta: mlx.mlx_array, state_in: mlx.mlx_array, B: c_int, T: c_int, Hk: c_int, Hv: c_int, Dk: c_int, Dv: c_int, s: mlx.mlx_stream) !GdnRunOut {
    // Mirror the production route exactly: dtypes read off the arrays, block
    // size clamped to what the staging budget allows at that width.
    const in_dtype = mlx.mlx_array_dtype(q);
    const st_dtype = mlx.mlx_array_dtype(state_in);
    const tb = gdnBlockTFor(tb_requested, Dk, mlx.mlx_array_itemsize(q)) orelse return error.GdnBlockedDeclined;
    const T_scalar = mlx.mlx_array_new_int(T);
    defer _ = mlx.mlx_array_free(T_scalar);
    const y_shape = [_]c_int{ B, T, Hv, Dv };
    const so_shape = [_]c_int{ B, Hv, Dv, Dk };
    const config = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &y_shape, 4, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &so_shape, 4, st_dtype));
    try gdnSetGrid(config, route, Dv, Hv, B);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "InT", in_dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "StT", st_dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "OutT", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dk", Dk));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dv", Dv));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hk", Hk));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hv", Hv));

    const inputs_arr = [_]mlx.mlx_array{ q, k, v, g, beta, state_in, T_scalar };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    const kern = try gdnRouteKernel(route, tb, false);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kern, inputs_vec, config, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 2) return error.MetalKernelBadOutputCount;
    var yo = mlx.mlx_array_new();
    var so = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&yo, outputs_vec, 0));
    try mlx.check(mlx.mlx_vector_array_get(&so, outputs_vec, 1));
    return .{ .y = yo, .state = so };
}

fn maxAbsDiff(a: []const f32, b: []const f32) f32 {
    var m: f32 = 0;
    for (a, b) |x, y| m = @max(m, @abs(x - y));
    return m;
}

/// `in_dtype` is the ACTIVATION width q/k/v arrive in. bf16 is the common
/// case; an f16 checkpoint promotes its activations to f32 (f16 ⊕ f32 scalar),
/// which is a different kernel specialization AND a different threadgroup
/// budget — both covered by the sweep below.
const GdnCase = struct { B: c_int, T: c_int, Hk: c_int, Hv: c_int, Dv: c_int, tb: u32, in_dtype: mlx.mlx_dtype = .bfloat16, state_dtype: mlx.mlx_dtype = .bfloat16 };

/// Runs one geometry through host ref + stock + blocked and asserts the
/// blocked kernel is no less accurate than the stock kernel it replaces.
fn gdnBlockedParityCase(case: GdnCase, s: mlx.mlx_stream) !void {
    const al = testing.allocator;
    const B = case.B;
    const T = case.T;
    const Hk = case.Hk;
    const Hv = case.Hv;
    const Dk: c_int = 128;
    const Dv = case.Dv;
    const qn: usize = @intCast(B * T * Hk * Dk);
    const vn: usize = @intCast(B * T * Hv * Dv);
    const gn: usize = @intCast(B * T * Hv);
    const sn: usize = @intCast(B * Hv * Dv * Dk);

    var prng = std.Random.DefaultPrng.init(0xB10C5ED);
    const rnd = prng.random();
    const qd = try al.alloc(f32, qn);
    defer al.free(qd);
    const kd = try al.alloc(f32, qn);
    defer al.free(kd);
    const vd = try al.alloc(f32, vn);
    defer al.free(vd);
    const gd = try al.alloc(f32, gn);
    defer al.free(gd);
    const bd = try al.alloc(f32, gn);
    defer al.free(bd);
    const sd = try al.alloc(f32, sn);
    defer al.free(sd);
    for (qd) |*x| x.* = bf16Trunc(rnd.float(f32) - 0.5);
    for (kd) |*x| x.* = bf16Trunc(rnd.float(f32) - 0.5);
    for (vd) |*x| x.* = bf16Trunc(rnd.float(f32) - 0.5);
    for (gd) |*x| x.* = bf16Trunc(0.5 + 0.4 * rnd.float(f32));
    for (bd) |*x| x.* = bf16Trunc(rnd.float(f32));
    for (sd) |*x| x.* = bf16Trunc(rnd.float(f32) - 0.5);

    const ref = try gdnHostRef(al, qd, kd, vd, gd, bd, sd, @intCast(B), @intCast(T), @intCast(Hk), @intCast(Hv), @intCast(Dk), @intCast(Dv));
    defer al.free(ref.y);
    defer al.free(ref.state);

    const qsh = [_]c_int{ B, T, Hk, Dk };
    const vsh = [_]c_int{ B, T, Hv, Dv };
    const gsh = [_]c_int{ B, T, Hv };
    const ssh = [_]c_int{ B, Hv, Dv, Dk };
    const q32 = mlx.mlx_array_new_data(qd.ptr, &qsh, 4, .float32);
    defer _ = mlx.mlx_array_free(q32);
    const k32 = mlx.mlx_array_new_data(kd.ptr, &qsh, 4, .float32);
    defer _ = mlx.mlx_array_free(k32);
    const v32 = mlx.mlx_array_new_data(vd.ptr, &vsh, 4, .float32);
    defer _ = mlx.mlx_array_free(v32);
    const g32 = mlx.mlx_array_new_data(gd.ptr, &gsh, 3, .float32);
    defer _ = mlx.mlx_array_free(g32);
    const b32 = mlx.mlx_array_new_data(bd.ptr, &gsh, 3, .float32);
    defer _ = mlx.mlx_array_free(b32);
    const st32 = mlx.mlx_array_new_data(sd.ptr, &ssh, 4, .float32);
    defer _ = mlx.mlx_array_free(st32);

    var q = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q);
    var kk = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kk);
    var v = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v);
    var g = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g);
    var beta = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(beta);
    var st = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(st);
    // Mirrors the live dtype signature: q/k/v/beta follow the ACTIVATION width
    // (bf16 checkpoints stay bf16; f16 checkpoints promote to f32), `g` is
    // always bf16 because it comes out of our fused gate kernel, and the SSM
    // state buffer is always bf16 because we allocate it that way. The mixed
    // set is the point — the kernel must read each input at its own width.
    try mlx.check(mlx.mlx_astype(&q, q32, case.in_dtype, s));
    try mlx.check(mlx.mlx_astype(&kk, k32, case.in_dtype, s));
    try mlx.check(mlx.mlx_astype(&v, v32, case.in_dtype, s));
    try mlx.check(mlx.mlx_astype(&g, g32, .bfloat16, s));
    try mlx.check(mlx.mlx_astype(&beta, b32, case.in_dtype, s));
    try mlx.check(mlx.mlx_astype(&st, st32, case.state_dtype, s));

    const stock = try gdnRunYState(.stock, 16, q, kk, v, g, beta, st, B, T, Hk, Hv, Dk, Dv, s);
    defer _ = mlx.mlx_array_free(stock.y);
    defer _ = mlx.mlx_array_free(stock.state);
    const stock_y = try evalToF32(al, stock.y, vn, s);
    defer al.free(stock_y);
    const stock_st = try evalToF32(al, stock.state, sn, s);
    defer al.free(stock_st);
    // No-worse-than-reference: the blocked kernel's error vs the f64 ground
    // truth must not exceed the stock kernel's (both bf16-out; 1.5x headroom
    // covers fp32 summation-order differences, gross bugs blow way past it).
    for (stock_y) |x| try testing.expect(std.math.isFinite(x));
    for (stock_st) |x| try testing.expect(std.math.isFinite(x));
    for (ref.y) |x| try testing.expect(std.math.isFinite(x));
    for (ref.state) |x| try testing.expect(std.math.isFinite(x));
    const stock_y_err = maxAbsDiff(stock_y, ref.y);
    const stock_st_err = maxAbsDiff(stock_st, ref.state);
    for ([_]GdnRoute{ .blocked, .pipelined }) |route| {
        const got = try gdnRunYState(route, case.tb, q, kk, v, g, beta, st, B, T, Hk, Hv, Dk, Dv, s);
        defer _ = mlx.mlx_array_free(got.y);
        defer _ = mlx.mlx_array_free(got.state);
        try testing.expectEqual(case.state_dtype, mlx.mlx_array_dtype(got.state));
        const got_y = try evalToF32(al, got.y, vn, s);
        defer al.free(got_y);
        const got_st = try evalToF32(al, got.state, sn, s);
        defer al.free(got_st);
        for (got_y) |x| try testing.expect(std.math.isFinite(x));
        for (got_st) |x| try testing.expect(std.math.isFinite(x));
        const y_err = maxAbsDiff(got_y, ref.y);
        const st_err = maxAbsDiff(got_st, ref.state);
        if (y_err > 1.5 * stock_y_err + 0.02 or st_err > 1.5 * stock_st_err + 0.02) {
            std.debug.print(
                "GDN {s} parity FAIL (B={d} T={d} Hk={d} Hv={d} Dv={d} tb={d}): y {d:.5} vs stock {d:.5}, state {d:.5} vs stock {d:.5}\n",
                .{ @tagName(route), case.B, case.T, case.Hk, case.Hv, case.Dv, case.tb, y_err, stock_y_err, st_err, stock_st_err },
            );
            return error.GdnBlockedParityFailed;
        }
    }
}

fn gdnChunkContinuity(route: GdnRoute, split: c_int, state_dtype: mlx.mlx_dtype) !void {
    const s = mlx.gpuStream();
    const al = testing.allocator;
    const B: c_int = 1;
    const T: c_int = 100;
    const T1: c_int = split; // non-multiple of TB=32: exercises the ragged tail mid-sequence
    const Hk: c_int = 2;
    const Hv: c_int = 4;
    const Dk: c_int = 128;
    const Dv: c_int = 64;
    const qn: usize = @intCast(B * T * Hk * Dk);
    const vn: usize = @intCast(B * T * Hv * Dv);
    const gn: usize = @intCast(B * T * Hv);
    const sn: usize = @intCast(B * Hv * Dv * Dk);

    var prng = std.Random.DefaultPrng.init(0x5EC0DD);
    const rnd = prng.random();
    const qd = try al.alloc(f32, qn);
    defer al.free(qd);
    const kd = try al.alloc(f32, qn);
    defer al.free(kd);
    const vd = try al.alloc(f32, vn);
    defer al.free(vd);
    const gd = try al.alloc(f32, gn);
    defer al.free(gd);
    const bd = try al.alloc(f32, gn);
    defer al.free(bd);
    const sd = try al.alloc(f32, sn);
    defer al.free(sd);
    for (qd) |*x| x.* = bf16Trunc(rnd.float(f32) - 0.5);
    for (kd) |*x| x.* = bf16Trunc(rnd.float(f32) - 0.5);
    for (vd) |*x| x.* = bf16Trunc(rnd.float(f32) - 0.5);
    for (gd) |*x| x.* = bf16Trunc(0.5 + 0.4 * rnd.float(f32));
    for (bd) |*x| x.* = bf16Trunc(rnd.float(f32));
    for (sd) |*x| x.* = bf16Trunc(rnd.float(f32) - 0.5);

    const qsh = [_]c_int{ B, T, Hk, Dk };
    const vsh = [_]c_int{ B, T, Hv, Dv };
    const gsh = [_]c_int{ B, T, Hv };
    const ssh = [_]c_int{ B, Hv, Dv, Dk };
    const q32 = mlx.mlx_array_new_data(qd.ptr, &qsh, 4, .float32);
    defer _ = mlx.mlx_array_free(q32);
    const k32 = mlx.mlx_array_new_data(kd.ptr, &qsh, 4, .float32);
    defer _ = mlx.mlx_array_free(k32);
    const v32 = mlx.mlx_array_new_data(vd.ptr, &vsh, 4, .float32);
    defer _ = mlx.mlx_array_free(v32);
    const g32 = mlx.mlx_array_new_data(gd.ptr, &gsh, 3, .float32);
    defer _ = mlx.mlx_array_free(g32);
    const b32 = mlx.mlx_array_new_data(bd.ptr, &gsh, 3, .float32);
    defer _ = mlx.mlx_array_free(b32);
    const st32 = mlx.mlx_array_new_data(sd.ptr, &ssh, 4, .float32);
    defer _ = mlx.mlx_array_free(st32);

    var q = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q);
    var kk = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kk);
    var v = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v);
    var g = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g);
    var beta = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(beta);
    var st = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(st);
    try mlx.check(mlx.mlx_astype(&q, q32, .bfloat16, s));
    try mlx.check(mlx.mlx_astype(&kk, k32, .bfloat16, s));
    try mlx.check(mlx.mlx_astype(&v, v32, .bfloat16, s));
    try mlx.check(mlx.mlx_astype(&g, g32, .bfloat16, s));
    try mlx.check(mlx.mlx_astype(&beta, b32, .bfloat16, s));
    try mlx.check(mlx.mlx_astype(&st, st32, state_dtype, s));

    // Full run.
    const full = try gdnRunYState(route, 32, q, kk, v, g, beta, st, B, T, Hk, Hv, Dk, Dv, s);
    defer _ = mlx.mlx_array_free(full.y);
    defer _ = mlx.mlx_array_free(full.state);

    // Split run: [0, T1) then [T1, T), carrying the bf16 state — exactly what
    // chunked prefill does (ssm_state hand-off between forwardWith chunks).
    const strides4 = [_]c_int{ 1, 1, 1, 1 };
    const strides3 = [_]c_int{ 1, 1, 1 };
    var q1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q1);
    var k1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k1);
    var v1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v1);
    var g1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g1);
    var b1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b1);
    try mlx.check(mlx.mlx_slice(&q1, q, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ B, T1, Hk, Dk }, 4, &strides4, 4, s));
    try mlx.check(mlx.mlx_slice(&k1, kk, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ B, T1, Hk, Dk }, 4, &strides4, 4, s));
    try mlx.check(mlx.mlx_slice(&v1, v, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ B, T1, Hv, Dv }, 4, &strides4, 4, s));
    try mlx.check(mlx.mlx_slice(&g1, g, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ B, T1, Hv }, 3, &strides3, 3, s));
    try mlx.check(mlx.mlx_slice(&b1, beta, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ B, T1, Hv }, 3, &strides3, 3, s));
    var q2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q2);
    var k2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k2);
    var v2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v2);
    var g2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g2);
    var b2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b2);
    try mlx.check(mlx.mlx_slice(&q2, q, &[_]c_int{ 0, T1, 0, 0 }, 4, &[_]c_int{ B, T, Hk, Dk }, 4, &strides4, 4, s));
    try mlx.check(mlx.mlx_slice(&k2, kk, &[_]c_int{ 0, T1, 0, 0 }, 4, &[_]c_int{ B, T, Hk, Dk }, 4, &strides4, 4, s));
    try mlx.check(mlx.mlx_slice(&v2, v, &[_]c_int{ 0, T1, 0, 0 }, 4, &[_]c_int{ B, T, Hv, Dv }, 4, &strides4, 4, s));
    try mlx.check(mlx.mlx_slice(&g2, g, &[_]c_int{ 0, T1, 0 }, 3, &[_]c_int{ B, T, Hv }, 3, &strides3, 3, s));
    try mlx.check(mlx.mlx_slice(&b2, beta, &[_]c_int{ 0, T1, 0 }, 3, &[_]c_int{ B, T, Hv }, 3, &strides3, 3, s));

    const part1 = try gdnRunYState(route, 32, q1, k1, v1, g1, b1, st, B, T1, Hk, Hv, Dk, Dv, s);
    defer _ = mlx.mlx_array_free(part1.y);
    defer _ = mlx.mlx_array_free(part1.state);
    const part2 = try gdnRunYState(route, 32, q2, k2, v2, g2, b2, part1.state, B, T - T1, Hk, Hv, Dk, Dv, s);
    defer _ = mlx.mlx_array_free(part2.y);
    defer _ = mlx.mlx_array_free(part2.state);

    const full_st = try evalToF32(al, full.state, sn, s);
    defer al.free(full_st);
    const split_st = try evalToF32(al, part2.state, sn, s);
    defer al.free(split_st);

    // The only divergence allowed is the one bf16 state rounding injected at
    // the boundary (live chunked prefill injects the identical rounding).
    for (full_st) |x| try testing.expect(std.math.isFinite(x));
    for (split_st) |x| try testing.expect(std.math.isFinite(x));
    var max_mag: f32 = 0;
    for (full_st) |x| max_mag = @max(max_mag, @abs(x));
    const tol = 0.02 * max_mag + 0.02;
    const st_diff = maxAbsDiff(full_st, split_st);
    try testing.expect(st_diff < tol);

    // y of the second chunk must also line up with the full run's tail.
    var y_tail = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y_tail);
    try mlx.check(mlx.mlx_slice(&y_tail, full.y, &[_]c_int{ 0, T1, 0, 0 }, 4, &[_]c_int{ B, T, Hv, Dv }, 4, &strides4, 4, s));
    const n2: usize = @intCast(B * (T - T1) * Hv * Dv);
    const tail_f = try evalToF32(al, y_tail, n2, s);
    defer al.free(tail_f);
    const part2_y = try evalToF32(al, part2.y, n2, s);
    defer al.free(part2_y);
    for (tail_f) |x| try testing.expect(std.math.isFinite(x));
    for (part2_y) |x| try testing.expect(std.math.isFinite(x));
    try testing.expect(maxAbsDiff(tail_f, part2_y) < tol);
}

/// A gate probe: exact at bf16, a few ulps of g at f32 (precise::exp against MLX's exp).
fn gdnGateExpectClose(got: mlx.mlx_array, want: mlx.mlx_array, gate_dt: mlx.mlx_dtype, s: mlx.mlx_stream) !void {
    try testing.expectEqual(gate_dt, mlx.mlx_array_dtype(got));
    try testing.expectEqual(gate_dt, mlx.mlx_array_dtype(want));
    const d = try attn256MaxDiff(got, want, s);
    if (gate_dt == .bfloat16) try testing.expectEqual(@as(f32, 0.0), d) else try testing.expect(d <= 1e-6);
}

fn gdnPreworkCase(gate_dt: mlx.mlx_dtype) !void {
    const s = mlx.gpuStream();
    gdn_prefill_fused_override = true;
    defer gdn_prefill_fused_override = null;
    gdn_prework_override = true;
    defer gdn_prework_override = null;
    gdn_decode_fused_override = true;
    defer gdn_decode_fused_override = null;
    var cache = try KVCache.init(testing.allocator, 0);
    defer cache.deinit();
    var xfm = std.mem.zeroInit(Transformer, .{ .s = s, .allocator = testing.allocator, .cache = cache });
    xfm.compileGdnGate();
    defer if (xfm.compiled_gdn_gate) |closure| {
        _ = mlx.mlx_closure_free(closure);
    };
    try testing.expect(xfm.compiled_gdn_gate != null);
    var prng = std.Random.DefaultPrng.init(0x6D1);
    const rnd = prng.random();

    const hk: c_int = 2;
    const hv: c_int = 4;
    const dk: c_int = 128;
    const dv: c_int = 128;
    const c_dim: c_int = hk * dk * 2 + hv * dv;
    const key_dim: c_int = hk * dk;
    const value_dim: c_int = hv * dv;

    const q_scale = bf16Scalar(1.0 / 128.0, s);
    defer _ = mlx.mlx_array_free(q_scale);
    const k_scale = bf16Scalar(@sqrt(1.0 / 128.0), s);
    defer _ = mlx.mlx_array_free(k_scale);
    var ones_w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ones_w);
    const ones_shape = [_]c_int{dk};
    try mlx.check(mlx.mlx_ones(&ones_w, &ones_shape, 1, .bfloat16, s));
    const hv_shape = [_]c_int{hv};
    const A_log = try attn256RandBf16(rnd, &hv_shape, s);
    defer _ = mlx.mlx_array_free(A_log);
    const dt_bias = try attn256RandBf16(rnd, &hv_shape, s);
    defer _ = mlx.mlx_array_free(dt_bias);

    for ([_]c_int{ 1, 2, 3, 5, 9, 17, 65, 513, 8703 }) |seq| {
        const qkv_shape = [_]c_int{ 1, seq, c_dim };
        const qkv = try attn256RandBf16(rnd, &qkv_shape, s);
        defer _ = mlx.mlx_array_free(qkv);
        const st_shape = [_]c_int{ 1, 3, c_dim };
        const conv_state = if (seq == 17) blk: {
            var zero = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_zeros(&zero, &st_shape, 3, .bfloat16, s));
            break :blk zero;
        } else try attn256RandBf16(rnd, &st_shape, s);
        defer _ = mlx.mlx_array_free(conv_state);
        const w_shape = [_]c_int{ c_dim, 4, 1 };
        const conv_w = try attn256RandBf16(rnd, &w_shape, s);
        defer _ = mlx.mlx_array_free(conv_w);
        const ba_shape = [_]c_int{ 1, seq, hv };
        // Wide-range b/a so the gate's exp/log1p branches are all exercised.
        const b_in = try attn256RandBf16Scaled(rnd, &ba_shape, 16.0, s);
        defer _ = mlx.mlx_array_free(b_in);
        const a_in = try attn256RandBf16Scaled(rnd, &ba_shape, 16.0, s);
        defer _ = mlx.mlx_array_free(a_in);

        const pre = (try gdnPreworkFused(s, .{
            .qkv = qkv,
            .qkv_off = 0,
            .qkv_stride = c_dim,
            .b = b_in,
            .b_off = 0,
            .b_stride = hv,
            .a = a_in,
            .a_off = 0,
            .a_stride = hv,
            .A_log = A_log,
            .dt_bias = dt_bias,
            .conv_state = conv_state,
            .conv_w = conv_w,
            .q_scale = q_scale,
            .k_scale = k_scale,
            .hk = hk,
            .hv = hv,
            .dk = dk,
            .dv = dv,
            .seq = seq,
            .gate_dtype = gate_dt,
        })) orelse return error.FusedDeclined;
        defer _ = mlx.mlx_array_free(pre.q);
        defer _ = mlx.mlx_array_free(pre.k);
        defer _ = mlx.mlx_array_free(pre.v);
        defer _ = mlx.mlx_array_free(pre.conv_state);
        defer _ = mlx.mlx_array_free(pre.g);
        defer _ = mlx.mlx_array_free(pre.beta);

        // Composed reference — the production chain, op for op.
        var conv_input = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(conv_input);
        {
            const arr = [_]mlx.mlx_array{ conv_state, qkv };
            const vec = mlx.mlx_vector_array_new_data(&arr, 2);
            defer _ = mlx.mlx_vector_array_free(vec);
            try mlx.check(mlx.mlx_concatenate_axis(&conv_input, vec, 1, s));
        }
        var ref_state = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ref_state);
        {
            const start = [_]c_int{ 0, seq, 0 };
            const stop = [_]c_int{ 1, seq + 3, c_dim };
            const strides = [_]c_int{ 1, 1, 1 };
            try mlx.check(mlx.mlx_slice(&ref_state, conv_input, &start, 3, &stop, 3, &strides, 3, s));
        }
        var conv_raw = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(conv_raw);
        try mlx.check(mlx.mlx_conv1d(&conv_raw, conv_input, conv_w, 1, 0, 1, c_dim, s));
        var sig = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sig);
        try mlx.check(mlx.mlx_sigmoid(&sig, conv_raw, s));
        var conv_out = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(conv_out);
        try mlx.check(mlx.mlx_multiply(&conv_out, conv_raw, sig, s));

        const strides3 = [_]c_int{ 1, 1, 1 };
        {
            var v_flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(v_flat);
            const start = [_]c_int{ 0, 0, key_dim * 2 };
            const stop = [_]c_int{ 1, seq, key_dim * 2 + value_dim };
            try mlx.check(mlx.mlx_slice(&v_flat, conv_out, &start, 3, &stop, 3, &strides3, 3, s));
            var v_heads = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(v_heads);
            const v_shape = [_]c_int{ 1, seq, hv, dv };
            try mlx.check(mlx.mlx_reshape(&v_heads, v_flat, &v_shape, 4, s));
            try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(pre.v, v_heads, s));
        }
        inline for (.{
            .{ @as(c_int, 0), key_dim, hk, dk, q_scale, pre.q },
            .{ key_dim, key_dim * 2, hk, dk, k_scale, pre.k },
        }) |case| {
            var flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(flat);
            const start = [_]c_int{ 0, 0, case[0] };
            const stop = [_]c_int{ 1, seq, case[1] };
            try mlx.check(mlx.mlx_slice(&flat, conv_out, &start, 3, &stop, 3, &strides3, 3, s));
            var heads = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(heads);
            const h_shape = [_]c_int{ 1, seq, case[2], case[3] };
            try mlx.check(mlx.mlx_reshape(&heads, flat, &h_shape, 4, s));
            var nrm = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(nrm);
            try mlx.check(mlx.mlx_fast_rms_norm(&nrm, heads, ones_w, 1e-6, s));
            var scaled = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(scaled);
            try mlx.check(mlx.mlx_multiply(&scaled, nrm, case[4], s));
            try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(case[5], scaled, s));
        }
        try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(pre.conv_state, ref_state, s));

        // Gate + beta against the graph chain.
        const g_ref = try gdnGateChain(A_log, a_in, dt_bias, gate_dt, s);
        defer _ = mlx.mlx_array_free(g_ref);
        try gdnGateExpectClose(pre.g, g_ref, gate_dt, s);
        if (gate_dt == xfm.config.gdnGateDtype()) {
            const g_compiled = try xfm.computeGdnGate(A_log, a_in, dt_bias);
            defer _ = mlx.mlx_array_free(g_compiled);
            try gdnGateExpectClose(pre.g, g_compiled, gate_dt, s);
        }
        var beta_ref = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(beta_ref);
        try mlx.check(mlx.mlx_sigmoid(&beta_ref, b_in, s));
        try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(pre.beta, beta_ref, s));

        // Folded layout: [qkv | z | b | a] rows read through offsets must
        // reproduce the unfolded call exactly.
        var raw = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(raw);
        {
            const z_shape = [_]c_int{ 1, seq, value_dim };
            const z = try attn256RandBf16(rnd, &z_shape, s);
            defer _ = mlx.mlx_array_free(z);
            const arr = [_]mlx.mlx_array{ qkv, z, b_in, a_in };
            const vec = mlx.mlx_vector_array_new_data(&arr, 4);
            defer _ = mlx.mlx_vector_array_free(vec);
            try mlx.check(mlx.mlx_concatenate_axis(&raw, vec, 2, s));
        }
        const width = c_dim + value_dim + 2 * hv;
        const pre2 = (try gdnPreworkFused(s, .{
            .qkv = raw,
            .qkv_off = 0,
            .qkv_stride = width,
            .b = raw,
            .b_off = c_dim + value_dim,
            .b_stride = width,
            .a = raw,
            .a_off = c_dim + value_dim + hv,
            .a_stride = width,
            .A_log = A_log,
            .dt_bias = dt_bias,
            .conv_state = conv_state,
            .conv_w = conv_w,
            .q_scale = q_scale,
            .k_scale = k_scale,
            .hk = hk,
            .hv = hv,
            .dk = dk,
            .dv = dv,
            .seq = seq,
            .gate_dtype = gate_dt,
        })) orelse return error.FusedDeclined;
        defer _ = mlx.mlx_array_free(pre2.q);
        defer _ = mlx.mlx_array_free(pre2.k);
        defer _ = mlx.mlx_array_free(pre2.v);
        defer _ = mlx.mlx_array_free(pre2.conv_state);
        defer _ = mlx.mlx_array_free(pre2.g);
        defer _ = mlx.mlx_array_free(pre2.beta);
        inline for (.{ .{ pre.q, pre2.q }, .{ pre.k, pre2.k }, .{ pre.v, pre2.v }, .{ pre.conv_state, pre2.conv_state }, .{ pre.g, pre2.g }, .{ pre.beta, pre2.beta } }) |pair| {
            try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(pair[0], pair[1], s));
        }

        // Batched arm (B=2 merged slots): row 0 of the batched outputs is
        // bit-identical to the single-batch outputs above; row 1 to a
        // second slot's own single-batch run.
        if (seq <= 8 or seq == 65) {
            const qkv_b = try attn256RandBf16(rnd, &qkv_shape, s);
            defer _ = mlx.mlx_array_free(qkv_b);
            const st_b = try attn256RandBf16(rnd, &st_shape, s);
            defer _ = mlx.mlx_array_free(st_b);
            const b_b = try attn256RandBf16Scaled(rnd, &ba_shape, 16.0, s);
            defer _ = mlx.mlx_array_free(b_b);
            const a_b = try attn256RandBf16Scaled(rnd, &ba_shape, 16.0, s);
            defer _ = mlx.mlx_array_free(a_b);
            const cat0 = struct {
                fn f(st: mlx.mlx_stream, x0: mlx.mlx_array, x1: mlx.mlx_array) !mlx.mlx_array {
                    const arr = [_]mlx.mlx_array{ x0, x1 };
                    const vec = mlx.mlx_vector_array_new_data(&arr, 2);
                    defer _ = mlx.mlx_vector_array_free(vec);
                    var out = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 0, st));
                    return out;
                }
            }.f;
            const row0 = struct {
                fn f(st: mlx.mlx_stream, x: mlx.mlx_array, i: c_int) !mlx.mlx_array {
                    const sh = mlx.getShape(x);
                    var start: [4]c_int = .{ i, 0, 0, 0 };
                    var stop: [4]c_int = .{ i + 1, 0, 0, 0 };
                    const strides: [4]c_int = .{ 1, 1, 1, 1 };
                    for (1..sh.len) |d| stop[d] = sh[d];
                    _ = &start;
                    var out = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_slice(&out, x, &start, @intCast(sh.len), &stop, @intCast(sh.len), &strides, @intCast(sh.len), st));
                    return out;
                }
            }.f;
            const qkv2 = try cat0(s, qkv, qkv_b);
            defer _ = mlx.mlx_array_free(qkv2);
            const st2 = try cat0(s, conv_state, st_b);
            defer _ = mlx.mlx_array_free(st2);
            const b2 = try cat0(s, b_in, b_b);
            defer _ = mlx.mlx_array_free(b2);
            const a2 = try cat0(s, a_in, a_b);
            defer _ = mlx.mlx_array_free(a2);
            var args_b = GdnPreworkArgs{ .qkv = qkv2, .qkv_off = 0, .qkv_stride = c_dim, .b = b2, .b_off = 0, .b_stride = hv, .a = a2, .a_off = 0, .a_stride = hv, .A_log = A_log, .dt_bias = dt_bias, .conv_state = st2, .conv_w = conv_w, .q_scale = q_scale, .k_scale = k_scale, .hk = hk, .hv = hv, .dk = dk, .dv = dv, .seq = seq, .batch = 2, .gate_dtype = gate_dt };
            const preb = (try gdnPreworkFused(s, args_b)) orelse return error.FusedDeclined;
            defer inline for (.{ preb.q, preb.k, preb.v, preb.conv_state, preb.g, preb.beta }) |arr| {
                _ = mlx.mlx_array_free(arr);
            };
            args_b.qkv = qkv_b;
            args_b.conv_state = st_b;
            args_b.b = b_b;
            args_b.a = a_b;
            args_b.batch = 1;
            const pre_b1 = (try gdnPreworkFused(s, args_b)) orelse return error.FusedDeclined;
            defer inline for (.{ pre_b1.q, pre_b1.k, pre_b1.v, pre_b1.conv_state, pre_b1.g, pre_b1.beta }) |arr| {
                _ = mlx.mlx_array_free(arr);
            };
            inline for (.{ .{ preb.q, pre.q, pre_b1.q }, .{ preb.k, pre.k, pre_b1.k }, .{ preb.v, pre.v, pre_b1.v }, .{ preb.conv_state, pre.conv_state, pre_b1.conv_state }, .{ preb.g, pre.g, pre_b1.g }, .{ preb.beta, pre.beta, pre_b1.beta } }) |tri| {
                const r0 = try row0(s, tri[0], 0);
                defer _ = mlx.mlx_array_free(r0);
                const r1 = try row0(s, tri[0], 1);
                defer _ = mlx.mlx_array_free(r1);
                try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(r0, tri[1], s));
                try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(r1, tri[2], s));
            }
        }
    }

    // Unsupported widths/head dimensions and disabled decode/prefill arms decline.
    {
        const qkv_shape = [_]c_int{ 1, 2, c_dim };
        const qkv = try attn256RandBf16(rnd, &qkv_shape, s);
        defer _ = mlx.mlx_array_free(qkv);
        const st_shape = [_]c_int{ 1, 3, c_dim };
        const conv_state = try attn256RandBf16(rnd, &st_shape, s);
        defer _ = mlx.mlx_array_free(conv_state);
        const w_shape = [_]c_int{ c_dim, 4, 1 };
        const conv_w = try attn256RandBf16(rnd, &w_shape, s);
        defer _ = mlx.mlx_array_free(conv_w);
        const ba_shape = [_]c_int{ 1, 2, hv };
        const ba = try attn256RandBf16(rnd, &ba_shape, s);
        defer _ = mlx.mlx_array_free(ba);
        var args = GdnPreworkArgs{
            .qkv = qkv,
            .qkv_off = 0,
            .qkv_stride = c_dim,
            .b = ba,
            .b_off = 0,
            .b_stride = hv,
            .a = ba,
            .a_off = 0,
            .a_stride = hv,
            .A_log = A_log,
            .dt_bias = dt_bias,
            .conv_state = conv_state,
            .conv_w = conv_w,
            .q_scale = q_scale,
            .k_scale = k_scale,
            .hk = hk,
            .hv = hv,
            .dk = dk,
            .dv = dv,
            .seq = 2,
            .gate_dtype = gate_dt,
        };
        args.seq = 17;
        gdn_prefill_fused_override = false;
        try std.testing.expect((try gdnPreworkFused(s, args)) == null);
        gdn_prefill_fused_override = true;
        args.seq = @import("../hc_prefill.zig").max_seq + 1;
        try std.testing.expect((try gdnPreworkFused(s, args)) == null);
        args.seq = 17;
        args.batch = 3;
        try std.testing.expect((try gdnPreworkFused(s, args)) == null);
        args.batch = 1;
        args.seq = 10;
        try std.testing.expect((try gdnPreworkFused(s, args)) == null);
        args.seq = 2;
        args.dk = 64;
        try std.testing.expect((try gdnPreworkFused(s, args)) == null);
        args.dk = 128;
        gdn_decode_fused_override = false;
        try std.testing.expect((try gdnPreworkFused(s, args)) == null);
        gdn_decode_fused_override = true;
    }
}

fn gdnPreworkBatchedCase(gate_dt: mlx.mlx_dtype) !void {
    const s = mlx.gpuStream();
    gdn_prework_override = true;
    defer gdn_prework_override = null;
    gdn_decode_fused_override = true;
    defer gdn_decode_fused_override = null;
    var prng = std.Random.DefaultPrng.init(0xB16);
    const rnd = prng.random();
    const hk: c_int = 16;
    const hv: c_int = 48;
    const dk: c_int = 128;
    const dv: c_int = 128;
    const c_dim: c_int = hk * dk * 2 + hv * dv;
    const seq: c_int = 1;
    const q_scale = bf16Scalar(1.0 / 128.0, s);
    defer _ = mlx.mlx_array_free(q_scale);
    const k_scale = bf16Scalar(@sqrt(1.0 / 128.0), s);
    defer _ = mlx.mlx_array_free(k_scale);
    var ones_w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ones_w);
    const ones_shape = [_]c_int{dk};
    try mlx.check(mlx.mlx_ones(&ones_w, &ones_shape, 1, .bfloat16, s));
    const key_dim: c_int = hk * dk;
    const value_dim: c_int = hv * dv;
    const hv_shape = [_]c_int{hv};
    const A_log = try attn256RandBf16(rnd, &hv_shape, s);
    defer _ = mlx.mlx_array_free(A_log);
    const dt_bias = try attn256RandBf16(rnd, &hv_shape, s);
    defer _ = mlx.mlx_array_free(dt_bias);
    const w_shape = [_]c_int{ c_dim, 4, 1 };
    const conv_w = try attn256RandBf16(rnd, &w_shape, s);
    defer _ = mlx.mlx_array_free(conv_w);

    const row0 = struct {
        fn f(st: mlx.mlx_stream, x: mlx.mlx_array, i: c_int) !mlx.mlx_array {
            const sh = mlx.getShape(x);
            var start: [4]c_int = .{ i, 0, 0, 0 };
            var stop: [4]c_int = .{ i + 1, 0, 0, 0 };
            const strides: [4]c_int = .{ 1, 1, 1, 1 };
            for (1..sh.len) |d| stop[d] = sh[d];
            var out = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(&out, x, &start, @intCast(sh.len), &stop, @intCast(sh.len), &strides, @intCast(sh.len), st));
            return out;
        }
    }.f;

    for ([_]c_int{ 6, 8, 16 }) |batch| {
        const qkv_shape = [_]c_int{ batch, seq, c_dim };
        const qkv = try attn256RandBf16(rnd, &qkv_shape, s);
        defer _ = mlx.mlx_array_free(qkv);
        const st_shape = [_]c_int{ batch, 3, c_dim };
        const conv_state = try attn256RandBf16(rnd, &st_shape, s);
        defer _ = mlx.mlx_array_free(conv_state);
        const ba_shape = [_]c_int{ batch, seq, hv };
        const b_in = try attn256RandBf16Scaled(rnd, &ba_shape, 16.0, s);
        defer _ = mlx.mlx_array_free(b_in);
        const a_in = try attn256RandBf16Scaled(rnd, &ba_shape, 16.0, s);
        defer _ = mlx.mlx_array_free(a_in);
        const preb = (try gdnPreworkFused(s, .{
            .qkv = qkv,
            .qkv_off = 0,
            .qkv_stride = c_dim,
            .b = b_in,
            .b_off = 0,
            .b_stride = hv,
            .a = a_in,
            .a_off = 0,
            .a_stride = hv,
            .A_log = A_log,
            .dt_bias = dt_bias,
            .conv_state = conv_state,
            .conv_w = conv_w,
            .q_scale = q_scale,
            .k_scale = k_scale,
            .hk = hk,
            .hv = hv,
            .dk = dk,
            .dv = dv,
            .seq = seq,
            .batch = batch,
            .gate_dtype = gate_dt,
        })) orelse return error.FusedDeclined;
        defer inline for (.{ preb.q, preb.k, preb.v, preb.conv_state, preb.g, preb.beta }) |arr| {
            _ = mlx.mlx_array_free(arr);
        };
        var conv_input = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(conv_input);
        {
            const arr = [_]mlx.mlx_array{ conv_state, qkv };
            const vec = mlx.mlx_vector_array_new_data(&arr, 2);
            defer _ = mlx.mlx_vector_array_free(vec);
            try mlx.check(mlx.mlx_concatenate_axis(&conv_input, vec, 1, s));
        }
        var ref_state = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ref_state);
        {
            const start = [_]c_int{ 0, seq, 0 };
            const stop = [_]c_int{ batch, seq + 3, c_dim };
            const strides = [_]c_int{ 1, 1, 1 };
            try mlx.check(mlx.mlx_slice(&ref_state, conv_input, &start, 3, &stop, 3, &strides, 3, s));
        }
        var conv_raw = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(conv_raw);
        try mlx.check(mlx.mlx_conv1d(&conv_raw, conv_input, conv_w, 1, 0, 1, c_dim, s));
        var sig = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sig);
        try mlx.check(mlx.mlx_sigmoid(&sig, conv_raw, s));
        var conv_out = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(conv_out);
        try mlx.check(mlx.mlx_multiply(&conv_out, conv_raw, sig, s));
        const strides3 = [_]c_int{ 1, 1, 1 };
        {
            var v_flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(v_flat);
            const start = [_]c_int{ 0, 0, key_dim * 2 };
            const stop = [_]c_int{ batch, seq, key_dim * 2 + value_dim };
            try mlx.check(mlx.mlx_slice(&v_flat, conv_out, &start, 3, &stop, 3, &strides3, 3, s));
            var v_heads = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(v_heads);
            const v_shape = [_]c_int{ batch, seq, hv, dv };
            try mlx.check(mlx.mlx_reshape(&v_heads, v_flat, &v_shape, 4, s));
            const d_v = try attn256MaxDiff(preb.v, v_heads, s);
            if (d_v != 0) std.debug.print("prework B={d} v maxdiff={e}\n", .{ batch, d_v });
            try std.testing.expectEqual(@as(f32, 0.0), d_v);
        }
        inline for (.{
            .{ @as(c_int, 0), key_dim, hk, dk, q_scale, preb.q },
            .{ key_dim, key_dim * 2, hk, dk, k_scale, preb.k },
        }) |case| {
            var flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(flat);
            const start = [_]c_int{ 0, 0, case[0] };
            const stop = [_]c_int{ batch, seq, case[1] };
            try mlx.check(mlx.mlx_slice(&flat, conv_out, &start, 3, &stop, 3, &strides3, 3, s));
            var heads = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(heads);
            const h_shape = [_]c_int{ batch, seq, case[2], case[3] };
            try mlx.check(mlx.mlx_reshape(&heads, flat, &h_shape, 4, s));
            var nrm = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(nrm);
            try mlx.check(mlx.mlx_fast_rms_norm(&nrm, heads, ones_w, 1e-6, s));
            var scaled = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(scaled);
            try mlx.check(mlx.mlx_multiply(&scaled, nrm, case[4], s));
            const dh = try attn256MaxDiff(case[5], scaled, s);
            if (dh != 0) std.debug.print("prework B={d} q/k maxdiff={e}\n", .{ batch, dh });
            try std.testing.expectEqual(@as(f32, 0.0), dh);
        }
        const ds = try attn256MaxDiff(preb.conv_state, ref_state, s);
        if (ds != 0) std.debug.print("prework B={d} conv_state maxdiff={e}\n", .{ batch, ds });
        try std.testing.expectEqual(@as(f32, 0.0), ds);
        const g_ref = try gdnGateChain(A_log, a_in, dt_bias, gate_dt, s);
        defer _ = mlx.mlx_array_free(g_ref);
        try gdnGateExpectClose(preb.g, g_ref, gate_dt, s);
        var beta_ref = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(beta_ref);
        try mlx.check(mlx.mlx_sigmoid(&beta_ref, b_in, s));
        const db = try attn256MaxDiff(preb.beta, beta_ref, s);
        if (db > 7.7e-6) std.debug.print("prework B={d} beta maxdiff={e}\n", .{ batch, db });
        try std.testing.expect(db <= 7.7e-6);
        var args1 = GdnPreworkArgs{
            .qkv = qkv,
            .qkv_off = 0,
            .qkv_stride = c_dim,
            .b = b_in,
            .b_off = 0,
            .b_stride = hv,
            .a = a_in,
            .a_off = 0,
            .a_stride = hv,
            .A_log = A_log,
            .dt_bias = dt_bias,
            .conv_state = conv_state,
            .conv_w = conv_w,
            .q_scale = q_scale,
            .k_scale = k_scale,
            .hk = hk,
            .hv = hv,
            .dk = dk,
            .dv = dv,
            .seq = seq,
            .batch = 1,
            .gate_dtype = gate_dt,
        };
        const last: c_int = batch - 1;
        const q1 = try row0(s, qkv, last);
        defer _ = mlx.mlx_array_free(q1);
        const st1 = try row0(s, conv_state, last);
        defer _ = mlx.mlx_array_free(st1);
        const b1 = try row0(s, b_in, last);
        defer _ = mlx.mlx_array_free(b1);
        const a1 = try row0(s, a_in, last);
        defer _ = mlx.mlx_array_free(a1);
        args1.qkv = q1;
        args1.conv_state = st1;
        args1.b = b1;
        args1.a = a1;
        const pre1 = (try gdnPreworkFused(s, args1)) orelse return error.FusedDeclined;
        defer inline for (.{ pre1.q, pre1.k, pre1.v, pre1.conv_state, pre1.g, pre1.beta }) |arr| {
            _ = mlx.mlx_array_free(arr);
        };
        inline for (.{ .{ preb.q, pre1.q }, .{ preb.k, pre1.k }, .{ preb.v, pre1.v }, .{ preb.conv_state, pre1.conv_state }, .{ preb.g, pre1.g }, .{ preb.beta, pre1.beta } }) |pair| {
            const r = try row0(s, pair[0], last);
            defer _ = mlx.mlx_array_free(r);
            try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(r, pair[1], s));
        }
        const y_shape = [_]c_int{ batch, seq, hv, dv };
        const y = try attn256RandBf16Scaled(rnd, &y_shape, 8.0, s);
        defer _ = mlx.mlx_array_free(y);
        const z_shape = [_]c_int{ batch, seq, hv * dv };
        const z = try attn256RandBf16Scaled(rnd, &z_shape, 12.0, s);
        defer _ = mlx.mlx_array_free(z);
        const nw_shape = [_]c_int{dv};
        const norm_w = try attn256RandBf16(rnd, &nw_shape, s);
        defer _ = mlx.mlx_array_free(norm_w);
        const eps_arr = mlx.mlx_array_new_float(1e-6);
        defer _ = mlx.mlx_array_free(eps_arr);
        const fused = (try gdnNormGateFused(s, y, z, 0, hv * dv, norm_w, eps_arr, false, hv, dv, batch, seq)) orelse return error.FusedDeclined;
        defer _ = mlx.mlx_array_free(fused);
        const y1 = try row0(s, y, last);
        defer _ = mlx.mlx_array_free(y1);
        const z1 = try row0(s, z, last);
        defer _ = mlx.mlx_array_free(z1);
        const fused1 = (try gdnNormGateFused(s, y1, z1, 0, hv * dv, norm_w, eps_arr, false, hv, dv, 1, seq)) orelse return error.FusedDeclined;
        defer _ = mlx.mlx_array_free(fused1);
        const fr = try row0(s, fused, last);
        defer _ = mlx.mlx_array_free(fr);
        try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(fr, fused1, s));
    }
}

/// Uniform [-scale/2, scale/2) drawn in f32, then cast: an f32 array keeps every mantissa bit.
fn gdnRecurRand(rnd: std.Random, shape: []const c_int, scale: f32, dt: mlx.mlx_dtype, s: mlx.mlx_stream) !mlx.mlx_array {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const data = try testing.allocator.alloc(f32, n);
    defer testing.allocator.free(data);
    for (data) |*x| x.* = (rnd.float(f32) - 0.5) * scale;
    const f32arr = mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), .float32);
    if (dt == .float32) return f32arr;
    defer _ = mlx.mlx_array_free(f32arr);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, f32arr, dt, s));
    return out;
}

/// `x[:, from:to]` on axis 1, any rank.
fn gdnRecurRows(x: mlx.mlx_array, from: c_int, to: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    var start: [5]c_int = @splat(0);
    var stop: [5]c_int = @splat(0);
    const strides: [5]c_int = @splat(1);
    for (sh, 0..) |d, i| stop[i] = d;
    start[1] = from;
    stop[1] = to;
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&out, x, &start, sh.len, &stop, sh.len, &strides, sh.len, s));
    return out;
}

fn gdnRecurRef(a: mlx.mlx_array) !mlx.mlx_array {
    var r = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_array_set(&r, a));
    return r;
}

/// The chain's recurrence dispatch as `gatedDeltaNetProjected` issues it at B=1:
/// y, the final state and (capturing) the per-step states.
fn gdnChainRecurrence(s: mlx.mlx_stream, pre: *const GdnPrework, state: mlx.mlx_array, hk: c_int, hv: c_int, t_len: c_int, capture: bool) ![3]mlx.mlx_array {
    const dk: c_int = 128;
    const dv: c_int = 128;
    const config = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &[_]c_int{ 1, t_len, hv, dv }, 4, .bfloat16));
    if (capture) try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &[_]c_int{ t_len, 1, hv, dv, dk }, 5, mlx.mlx_array_dtype(state)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &[_]c_int{ 1, hv, dv, dk }, 4, mlx.mlx_array_dtype(state)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32, dv, hv));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, 4, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "InT", mlx.mlx_array_dtype(pre.q)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "StT", mlx.mlx_array_dtype(state)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "OutT", .bfloat16));
    inline for (.{ .{ "Dk", dk }, .{ "Dv", dv }, .{ "Hk", hk }, .{ "Hv", hv } }) |kv|
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, kv[0], kv[1]));
    const T_scalar = mlx.mlx_array_new_int(t_len);
    defer _ = mlx.mlx_array_free(T_scalar);
    const seq_stride = mlx.mlx_array_new_int(hv * dv * dk);
    defer _ = mlx.mlx_array_free(seq_stride);
    const ins = [_]mlx.mlx_array{ pre.q, pre.k, pre.v, pre.g, pre.beta, state, T_scalar, seq_stride };
    const vec = mlx.mlx_vector_array_new_data(&ins, if (capture) 8 else 7);
    defer _ = mlx.mlx_vector_array_free(vec);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, if (capture) try getGdnKernelSeq(false) else try getGdnKernel(false), vec, config, s));
    var r: [3]mlx.mlx_array = .{ mlx.mlx_array_new(), mlx.mlx_array_new(), .{} };
    try mlx.check(mlx.mlx_vector_array_get(&r[0], outs, 0));
    try mlx.check(mlx.mlx_vector_array_get(&r[1], outs, if (capture) 2 else 1));
    if (capture) {
        r[2] = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&r[2], outs, 1));
    }
    return r;
}

/// The gate is f32 at either width; `wide` selects the f32 state, else the bf16 state of `--qwen-gdn 16`.
fn gdnRecurParityCase(s: mlx.mlx_stream, hk: c_int, hv: c_int, t_len: c_int, folded: bool, wide: bool) !void {
    const gate_dt: mlx.mlx_dtype = .float32;
    var prng = std.Random.DefaultPrng.init(0x6D5E + @as(u64, @intCast(t_len * 131 + hv)));
    const rnd = prng.random();
    const dk: c_int = 128;
    const dv: c_int = 128;
    const c_dim = 2 * hk * dk + hv * dv;
    const value_dim = hv * dv;
    const q_scale = bf16Scalar(1.0 / 128.0, s);
    defer _ = mlx.mlx_array_free(q_scale);
    const k_scale = bf16Scalar(@sqrt(1.0 / 128.0), s);
    defer _ = mlx.mlx_array_free(k_scale);
    const eps_arr = mlx.mlx_array_new_float(1e-6);
    defer _ = mlx.mlx_array_free(eps_arr);
    // Wide a/b reach every branch of the gate's exp/log1p.
    const qkv = try gdnRecurRand(rnd, &.{ 1, t_len, c_dim }, 4.0, .bfloat16, s);
    defer _ = mlx.mlx_array_free(qkv);
    const z = try gdnRecurRand(rnd, &.{ 1, t_len, value_dim }, 4.0, .bfloat16, s);
    defer _ = mlx.mlx_array_free(z);
    const a_in = try gdnRecurRand(rnd, &.{ 1, t_len, hv }, 16.0, .bfloat16, s);
    defer _ = mlx.mlx_array_free(a_in);
    const b_in = try gdnRecurRand(rnd, &.{ 1, t_len, hv }, 16.0, .bfloat16, s);
    defer _ = mlx.mlx_array_free(b_in);
    const conv_state = try gdnRecurRand(rnd, &.{ 1, 3, c_dim }, 4.0, .bfloat16, s);
    defer _ = mlx.mlx_array_free(conv_state);
    const conv_w = try gdnRecurRand(rnd, &.{ c_dim, 4, 1 }, 1.0, .bfloat16, s);
    defer _ = mlx.mlx_array_free(conv_w);
    const A_log = try gdnRecurRand(rnd, &.{hv}, 2.0, .bfloat16, s);
    defer _ = mlx.mlx_array_free(A_log);
    const dt_bias = try gdnRecurRand(rnd, &.{hv}, 2.0, .bfloat16, s);
    defer _ = mlx.mlx_array_free(dt_bias);
    const norm_w = try gdnRecurRand(rnd, &.{dv}, 2.0, .bfloat16, s);
    defer _ = mlx.mlx_array_free(norm_w);
    const ssm_state = try gdnRecurRand(rnd, &.{ 1, hv, dv, dk }, 1.0, if (wide) .float32 else .bfloat16, s);
    defer _ = mlx.mlx_array_free(ssm_state);

    const pre = (try gdnPreworkFused(s, .{ .gate_dtype = gate_dt, .qkv = qkv, .qkv_off = 0, .qkv_stride = c_dim, .b = b_in, .b_off = 0, .b_stride = hv, .a = a_in, .a_off = 0, .a_stride = hv, .A_log = A_log, .dt_bias = dt_bias, .conv_state = conv_state, .conv_w = conv_w, .q_scale = q_scale, .k_scale = k_scale, .hk = hk, .hv = hv, .dk = dk, .dv = dv, .seq = t_len })) orelse return error.FusedDeclined;
    defer inline for (.{ pre.q, pre.k, pre.v, pre.conv_state, pre.g, pre.beta }) |arr| {
        _ = mlx.mlx_array_free(arr);
    };
    const capture = t_len > 1;
    const ref = try gdnChainRecurrence(s, &pre, ssm_state, hk, hv, t_len, capture);
    defer for (ref) |arr| {
        if (arr.ctx != null) _ = mlx.mlx_array_free(arr);
    };
    const flat_ref = (try gdnNormGateFused(s, ref[0], z, 0, value_dim, norm_w, eps_arr, false, hv, dv, 1, t_len)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(flat_ref);

    const geo = gdn_decode.Geometry{ .hk = hk, .hv = hv, .dk = dk, .dv = dv };
    const in = gdn_decode.Inputs{ .qkv = qkv, .a = a_in, .b = b_in, .conv_state = conv_state, .ssm_state = ssm_state, .conv_w = conv_w, .A_log = A_log, .dt_bias = dt_bias, .q_scale = q_scale, .k_scale = k_scale, .gate_dtype = gate_dt };
    const got = (try gdn_decode.step(geo, t_len, in, true, s)) orelse return error.FusedDeclined;
    defer got.deinit();
    const flat = (try gdnNormGateFused(s, got.y, z, 0, value_dim, norm_w, eps_arr, false, hv, dv, 1, t_len)) orelse return error.FusedDeclined;
    defer _ = mlx.mlx_array_free(flat);

    if (folded) {
        for ([_]bool{ false, true }) |swish| {
            var available = false;
            if (!swish) {
                try mlx.check(mlx._mlx_array_is_available(&available, qkv));
                try testing.expect(!available);
            }
            const f = (try gdn_decode.stepFold(geo, t_len, in, z, norm_w, eps_arr, swish, s)) orelse {
                // A GPU whose threadgroup limit is below the fold's width declines it and serves the unfolded
                // path; only the two-row default must fold.
                if (gdn_decode.foldDeclined(t_len) and (gdn_decode.fold_nt_override != null or t_len > 2)) return;
                return error.FoldDeclined;
            };
            defer f.deinit();
            if (!swish) {
                try mlx.check(mlx._mlx_array_is_available(&available, qkv));
                try testing.expect(!available);
            }
            const want = (try gdnNormGateFused(s, got.y, z, 0, value_dim, norm_w, eps_arr, swish, hv, dv, 1, t_len)) orelse return error.FusedDeclined;
            defer _ = mlx.mlx_array_free(want);
            try qkvExpectBitsEqual(s, want, f.gated);
            try qkvExpectBitsEqual(s, got.conv_state, f.conv_state);
            try qkvExpectBitsEqual(s, got.ssm_state, f.ssm_state);
            var t: c_int = 0;
            while (t + 1 < t_len) : (t += 1) {
                const a_row = try gdnRecurRowsAxis0(got.state_seq, t, s);
                defer _ = mlx.mlx_array_free(a_row);
                const b_row = try gdnRecurRowsAxis0(f.state_seq, t, s);
                defer _ = mlx.mlx_array_free(b_row);
                try qkvExpectBitsEqual(s, a_row, b_row);
            }
            const vec = mlx.mlx_vector_array_new_data(&.{ conv_state, qkv }, 2);
            defer _ = mlx.mlx_vector_array_free(vec);
            var conv_input = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(conv_input);
            try mlx.check(mlx.mlx_concatenate_axis(&conv_input, vec, 1, s));
            try qkvExpectBitsEqual(s, conv_input, f.conv_input);
        }
    }
    try qkvExpectBitsEqual(s, ref[0], got.y);
    try qkvExpectBitsEqual(s, flat_ref, flat);
    try qkvExpectBitsEqual(s, pre.conv_state, got.conv_state);
    try qkvExpectBitsEqual(s, ref[1], got.ssm_state);
    if (!capture) return;

    // Serial decode token by token: its state after token t is capture row t (the rollback
    // source), its conv state the rollback's conv slice, and after the last token the final state.
    var conv_input = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(conv_input);
    {
        const arr = [_]mlx.mlx_array{ conv_state, qkv };
        const vec = mlx.mlx_vector_array_new_data(&arr, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&conv_input, vec, 1, s));
    }
    var conv = try gdnRecurRef(conv_state);
    defer _ = mlx.mlx_array_free(conv);
    var state = try gdnRecurRef(ssm_state);
    defer _ = mlx.mlx_array_free(state);
    var t: c_int = 0;
    while (t < t_len) : (t += 1) {
        var row: [3]mlx.mlx_array = undefined;
        inline for (.{ qkv, a_in, b_in }, 0..) |x, i| row[i] = try gdnRecurRows(x, t, t + 1, s);
        defer for (row) |arr| {
            _ = mlx.mlx_array_free(arr);
        };
        var one_in = in;
        one_in.qkv = row[0];
        one_in.a = row[1];
        one_in.b = row[2];
        one_in.conv_state = conv;
        one_in.ssm_state = state;
        const one = (try gdn_decode.step(geo, 1, one_in, true, s)) orelse return error.FusedDeclined;
        defer one.deinit();
        _ = mlx.mlx_array_free(conv);
        conv = try gdnRecurRef(one.conv_state);
        _ = mlx.mlx_array_free(state);
        state = try gdnRecurRef(one.ssm_state);
        const y_t = try gdnRecurRows(got.y, t, t + 1, s);
        defer _ = mlx.mlx_array_free(y_t);
        try qkvExpectBitsEqual(s, y_t, one.y);
        if (t + 1 == t_len) {
            try qkvExpectBitsEqual(s, got.ssm_state, state);
            try qkvExpectBitsEqual(s, got.conv_state, conv);
            break;
        }
        var row_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(row_t);
        const seq_view = try gdnRecurRowsAxis0(got.state_seq, t, s);
        defer _ = mlx.mlx_array_free(seq_view);
        try mlx.check(mlx.mlx_reshape(&row_t, seq_view, &[_]c_int{ 1, hv, dv, dk }, 4, s));
        try qkvExpectBitsEqual(s, row_t, state);
        const conv_t = try gdnRecurRows(conv_input, 1 + t, 4 + t, s);
        defer _ = mlx.mlx_array_free(conv_t);
        try qkvExpectBitsEqual(s, conv_t, conv);
    }
}

/// `x[i]` on axis 0, kept as a leading 1.
fn gdnRecurRowsAxis0(x: mlx.mlx_array, i: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    var start: [5]c_int = @splat(0);
    var stop: [5]c_int = @splat(0);
    const strides: [5]c_int = @splat(1);
    for (sh, 0..) |d, k| stop[k] = d;
    start[0] = i;
    stop[0] = i + 1;
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&out, x, &start, sh.len, &stop, sh.len, &strides, sh.len, s));
    return out;
}

/// Both arms of one GDN layer call on twin entries; asserts the fused arm engaged the
/// same bytes as the chain and returns its output.
fn gdnRecurArms(xfm: *Transformer, la: *const LinearAttnWeights, chain: *SSMCacheEntry, fused: *SSMCacheEntry, x: mlx.mlx_array, seq: c_int, capture: bool) !mlx.mlx_array {
    const s = xfm.s;
    xfm.spec_capture_ssm = capture;
    defer xfm.spec_capture_ssm = false;
    defer gdn_decode_recur_override = null;
    gdn_decode_recur_override = false;
    const out_chain = try xfm.gatedDeltaNetProjected(x, la, chain, 0, 1, seq, seq > 1, null, false);
    defer _ = mlx.mlx_array_free(out_chain);
    const engaged = if (seq == 1) &gdn_decode_recur_engaged else &gdn_verify_recur_engaged;
    engaged.* = false;
    gdn_verify_fold_engaged = false;
    gdn_decode_recur_override = true;
    const out = try xfm.gatedDeltaNetProjected(x, la, fused, 0, 1, seq, seq > 1, null, false);
    errdefer _ = mlx.mlx_array_free(out);
    try testing.expect(engaged.*);
    if (capture and gdnVerifyFoldFor(seq) and !gdn_decode.foldDeclined(seq)) try testing.expect(gdn_verify_fold_engaged);
    try qkvExpectBitsEqual(s, out_chain, out);
    try qkvExpectBitsEqual(s, chain.conv_state, fused.conv_state);
    try qkvExpectBitsEqual(s, chain.ssm_state, fused.ssm_state);
    if (capture) {
        try qkvExpectBitsEqual(s, chain.spec_conv_input, fused.spec_conv_input);
        // From three rows on the fused verify stores its final state only and rollback replays.
        const replays = seq >= 3;
        try testing.expectEqual(replays, fused.spec_replay.held());
        try testing.expectEqual(replays, fused.spec_state_seq.ctx == null);
        var i: c_int = 0;
        while (!replays and i + 1 < seq) : (i += 1) {
            const ra = try gdnRecurRowsAxis0(chain.spec_state_seq, i, s);
            defer _ = mlx.mlx_array_free(ra);
            const rb = try gdnRecurRowsAxis0(fused.spec_state_seq, i, s);
            defer _ = mlx.mlx_array_free(rb);
            try qkvExpectBitsEqual(s, ra, rb);
        }
    }
    return out;
}

/// The gate is f32 at either width; `wide` selects the f32 state, else the bf16 state of `--qwen-gdn 16`.
fn gdnRecurLayerCase(wide: bool) !void {
    const s = mlx.gpuStream();
    gdn_prework_override = true;
    defer gdn_prework_override = null;
    gdn_decode_fused_override = true;
    defer gdn_decode_fused_override = null;
    var cache = try KVCache.init(testing.allocator, 0);
    defer cache.deinit();
    var xfm = std.mem.zeroInit(Transformer, .{ .s = s, .allocator = testing.allocator, .cache = cache });
    defer inline for (.{ &xfm.gdn_q_scale, &xfm.gdn_k_scale, &xfm.gdn_eps }) |p| {
        if (p.*) |arr| _ = mlx.mlx_array_free(arr);
    };
    // The served geometry (Hk 16, Hv 48, 128-wide heads, sigmoid out gate) over a narrow hidden.
    const hk: c_int = 16;
    const hv: c_int = 48;
    const hidden: c_int = 128;
    const c_dim: c_int = 2 * hk * 128 + hv * 128;
    xfm.config.linear_num_key_heads = @intCast(hk);
    xfm.config.linear_num_value_heads = @intCast(hv);
    xfm.config.linear_key_head_dim = 128;
    xfm.config.linear_value_head_dim = 128;
    xfm.config.linear_conv_kernel_dim = 4;
    xfm.config.rms_norm_eps = 1e-6;
    xfm.config.kda_sigmoid_out_gate = true;
    xfm.config.gdn_state_f32 = wide;
    var prng = std.Random.DefaultPrng.init(0x6D5E1);
    const rnd = prng.random();
    const la = LinearAttnWeights{
        .qkv_w = try gdnRecurRand(rnd, &.{ hidden, c_dim }, 0.5, .bfloat16, s),
        .qkv_s = .{},
        .qkv_b = .{},
        .z_w = try gdnRecurRand(rnd, &.{ hidden, hv * 128 }, 0.5, .bfloat16, s),
        .z_s = .{},
        .z_b = .{},
        .a_w = try gdnRecurRand(rnd, &.{ hidden, hv }, 2.0, .bfloat16, s),
        .a_s = .{},
        .a_b = .{},
        .b_w = try gdnRecurRand(rnd, &.{ hidden, hv }, 2.0, .bfloat16, s),
        .b_s = .{},
        .b_b = .{},
        .conv1d_w = try gdnRecurRand(rnd, &.{ c_dim, 4, 1 }, 1.0, .bfloat16, s),
        .A_log = try gdnRecurRand(rnd, &.{hv}, 2.0, .bfloat16, s),
        .dt_bias = try gdnRecurRand(rnd, &.{hv}, 2.0, .bfloat16, s),
        .norm_w = try gdnRecurRand(rnd, &.{128}, 2.0, .bfloat16, s),
        .out_w = try gdnRecurRand(rnd, &.{ hv * 128, hidden }, 0.1, .bfloat16, s),
        .out_s = .{},
        .out_b = .{},
    };
    defer inline for (.{ la.qkv_w, la.z_w, la.a_w, la.b_w, la.conv1d_w, la.A_log, la.dt_bias, la.norm_w, la.out_w }) |arr| {
        _ = mlx.mlx_array_free(arr);
    };
    const conv0 = try gdnRecurRand(rnd, &.{ 1, 3, c_dim }, 4.0, .bfloat16, s);
    defer _ = mlx.mlx_array_free(conv0);
    const state0 = try gdnRecurRand(rnd, &.{ 1, hv, 128, 128 }, 1.0, if (wide) .float32 else .bfloat16, s);
    defer _ = mlx.mlx_array_free(state0);
    var chain = SSMCacheEntry{ .conv_state = try gdnRecurRef(conv0), .ssm_state = try gdnRecurRef(state0), .initialized = true };
    var fused = SSMCacheEntry{ .conv_state = try gdnRecurRef(conv0), .ssm_state = try gdnRecurRef(state0), .initialized = true };
    defer inline for (.{ &chain, &fused }) |e| {
        ssmFreeSpecCapture(e);
        _ = mlx.mlx_array_free(e.conv_state);
        _ = mlx.mlx_array_free(e.ssm_state);
    };

    var seq: c_int = 1;
    while (seq <= gdn_decode.MAX_SEQ) : (seq += 1) {
        // A decode step between verifies, as the engine runs them.
        {
            const x = try gdnRecurRand(rnd, &.{ 1, 1, hidden }, 2.0, .bfloat16, s);
            defer _ = mlx.mlx_array_free(x);
            _ = mlx.mlx_array_free(try gdnRecurArms(&xfm, &la, &chain, &fused, x, 1, false));
        }
        if (seq == 1) continue;
        var serial = SSMCacheEntry{ .conv_state = try gdnRecurRef(fused.conv_state), .ssm_state = try gdnRecurRef(fused.ssm_state), .initialized = true };
        defer {
            _ = mlx.mlx_array_free(serial.conv_state);
            _ = mlx.mlx_array_free(serial.ssm_state);
        }
        const x = try gdnRecurRand(rnd, &.{ 1, seq, hidden }, 2.0, .bfloat16, s);
        defer _ = mlx.mlx_array_free(x);
        const out = try gdnRecurArms(&xfm, &la, &chain, &fused, x, seq, true);
        defer _ = mlx.mlx_array_free(out);

        // Accepting rows 0..j rolls back to what serial decode of those rows holds.
        var j: c_int = 0;
        while (j < seq) : (j += 1) {
            const x_j = try gdnRecurRows(x, j, j + 1, s);
            defer _ = mlx.mlx_array_free(x_j);
            gdn_decode_recur_override = true;
            defer gdn_decode_recur_override = null;
            const out_j = try xfm.gatedDeltaNetProjected(x_j, &la, &serial, 0, 1, 1, false, null, false);
            defer _ = mlx.mlx_array_free(out_j);
            const row_j = try gdnRecurRows(out, j, j + 1, s);
            defer _ = mlx.mlx_array_free(row_j);
            try qkvExpectBitsEqual(s, row_j, out_j);
            if (j + 1 == seq) {
                try qkvExpectBitsEqual(s, fused.ssm_state, serial.ssm_state);
                try qkvExpectBitsEqual(s, fused.conv_state, serial.conv_state);
                break;
            }
            for ([_]*SSMCacheEntry{ &chain, &fused }) |e| {
                var back = SSMCacheEntry{
                    .conv_state = try gdnRecurRef(e.conv_state),
                    .ssm_state = try gdnRecurRef(e.ssm_state),
                    .initialized = true,
                    .spec_state_seq = if (e.spec_state_seq.ctx != null) try gdnRecurRef(e.spec_state_seq) else .{ .ctx = null },
                    .spec_conv_input = try gdnRecurRef(e.spec_conv_input),
                };
                if (e.spec_replay.held()) {
                    var rep = e.spec_replay;
                    rep.pre_state = try gdnRecurRef(rep.pre_state);
                    try back.spec_replay.hold(rep.pre_state, rep.geo, .{
                        .qkv = .{},
                        .a = rep.a,
                        .b = rep.b,
                        .conv_state = .{},
                        .ssm_state = rep.pre_state,
                        .conv_w = rep.conv_w,
                        .A_log = rep.A_log,
                        .dt_bias = rep.dt_bias,
                        .q_scale = rep.q_scale,
                        .k_scale = rep.k_scale,
                        .gate_dtype = rep.gate_dtype,
                    });
                }
                defer {
                    ssmFreeSpecCapture(&back);
                    _ = mlx.mlx_array_free(back.conv_state);
                    _ = mlx.mlx_array_free(back.ssm_state);
                }
                try ssmRollbackFromCapture(&back, @intCast(j), @intCast(seq), s);
                try qkvExpectBitsEqual(s, back.ssm_state, serial.ssm_state);
                try qkvExpectBitsEqual(s, back.conv_state, serial.conv_state);
            }
        }
        ssmFreeSpecCapture(&chain);
        ssmFreeSpecCapture(&fused);
    }
}

const GdnDecayRoute = enum { chain, prework_chain, decode, verify, fold, prefill };

/// Per head: the gate input `a` (dt_bias 0) and A_log. Every exact gate lies in (0.5, 1); the slow ones
/// sit inside bf16's rounding to 1.0 or to its 2^-8 grid.
const gdn_decay_a = [8]f32{ -8, -7, -6, -9, -5, -2, 0, -10 };
const gdn_decay_A_log = [8]f32{ 0, 0, 0, -1, 0, 0, 0, 0 };
/// Divisible by every width 1..MAX_SEQ.
const gdn_decay_steps: c_int = 840;

fn gdnDecayGate(head: usize) f64 {
    const sp = @log(1.0 + @exp(@as(f64, gdn_decay_a[head])));
    return @exp(-@exp(@as(f64, gdn_decay_A_log[head])) * sp);
}

fn gdnDecayHost(arr: mlx.mlx_array, s: mlx.mlx_stream) ![]f32 {
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, arr, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f));
    const p = mlx.mlx_array_data_float32(f) orelse return error.InvalidDtype;
    return testing.allocator.dupe(f32, p[0..mlx.mlx_array_size(f)]);
}

/// `state0` decayed by `steps` exact gates: with q, k and v all zero the recurrence is `state *= g`.
fn gdnDecayExpectExact(state0: []const f32, got: []const f32, hv: usize, steps: usize, tol: f64) !void {
    const per_head = state0.len / hv;
    for (0..hv) |h| {
        const decay = std.math.pow(f64, gdnDecayGate(h), @floatFromInt(steps));
        for (0..per_head) |i| {
            const want = @as(f64, state0[h * per_head + i]) * decay;
            const have = @as(f64, got[h * per_head + i]);
            if (@abs(have - want) > tol * @abs(want) + 1e-9) {
                std.debug.print("head {d} elem {d} steps {d}: got {e} want {e}\n", .{ h, i, steps, have, want });
                return error.TestExpectedApproxEqRel;
            }
        }
    }
}

fn gdnDecayEqualBits(a: []const f32, b: []const f32) !void {
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(a), std.mem.sliceAsBytes(b));
}

fn gdnDecayClose(a: []const f32, b: []const f32, tol: f64) !void {
    for (a, b) |x, y| try testing.expect(@abs(@as(f64, x) - @as(f64, y)) <= tol * @abs(@as(f64, x)) + 1e-9);
}

/// Final state (and, when capturing, the last call's captured rows) after `gdn_decay_steps` tokens in
/// calls of `per_call`, on the layer path `route` selects. Row j of the last call holds the state after
/// token j of that call.
fn gdnDecayRun(xfm: *Transformer, la: *const LinearAttnWeights, route: GdnDecayRoute, per_call: c_int, state0: mlx.mlx_array, conv_dim: c_int, rows: *[]f32) ![]f32 {
    const s = xfm.s;
    gdn_prework_override = route != .chain;
    defer gdn_prework_override = null;
    gdn_decode_recur_override = route == .decode or route == .verify or route == .fold;
    defer gdn_decode_recur_override = null;
    gdn_verify_fold_override = route == .fold;
    defer gdn_verify_fold_override = null;
    gdn_decode_recur_engaged = false;
    gdn_verify_recur_engaged = false;
    gdn_verify_fold_engaged = false;
    xfm.spec_capture_ssm = route == .verify or route == .fold;
    defer xfm.spec_capture_ssm = false;

    var entry = SSMCacheEntry{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = true };
    defer {
        ssmFreeSpecCapture(&entry);
        _ = mlx.mlx_array_free(entry.conv_state);
        _ = mlx.mlx_array_free(entry.ssm_state);
    }
    try mlx.check(mlx.mlx_zeros(&entry.conv_state, &[_]c_int{ 1, 3, conv_dim }, 3, .bfloat16, s));
    try mlx.check(mlx.mlx_astype(&entry.ssm_state, state0, xfm.config.gdnStateDtype(), s));

    var ones = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ones);
    try mlx.check(mlx.mlx_ones(&ones, &[_]c_int{ 1, per_call, 128 }, 3, .bfloat16, s));
    var call: c_int = 0;
    while (call < @divExact(gdn_decay_steps, per_call)) : (call += 1) {
        _ = mlx.mlx_array_free(try xfm.gatedDeltaNetProjected(ones, la, &entry, 0, 1, per_call, route == .prefill, null, false));
    }
    switch (route) {
        .decode => try testing.expect(gdn_decode_recur_engaged),
        .verify => try testing.expect(gdn_verify_recur_engaged),
        .fold => try testing.expect(gdn_verify_fold_engaged or gdn_decode.foldDeclined(per_call)),
        else => {},
    }
    rows.* = if (entry.spec_state_seq.ctx != null) try gdnDecayHost(entry.spec_state_seq, s) else &.{};
    return gdnDecayHost(entry.ssm_state, s);
}

/// The gate is f32 either way. An f32 state forgets at the exact rate; a bf16 state rounds a slow head's
/// decay away (its step is under half a bf16 ulp), so only the routes' agreement is asserted there.
fn gdnDecayCase(state_f32: bool) !void {
    const s = mlx.gpuStream();
    var cache = try KVCache.init(testing.allocator, 0);
    defer cache.deinit();
    var xfm = std.mem.zeroInit(Transformer, .{ .s = s, .allocator = testing.allocator, .cache = cache });
    defer inline for (.{ &xfm.gdn_q_scale, &xfm.gdn_k_scale, &xfm.gdn_eps, &xfm.gdn_ones_w }) |p| {
        if (p.*) |arr| _ = mlx.mlx_array_free(arr);
    };
    const hk: c_int = 2;
    const hv: c_int = 8;
    const hidden: c_int = 128;
    const c_dim: c_int = 2 * hk * 128 + hv * 128;
    xfm.config.linear_num_key_heads = @intCast(hk);
    xfm.config.linear_num_value_heads = @intCast(hv);
    xfm.config.linear_key_head_dim = 128;
    xfm.config.linear_value_head_dim = 128;
    xfm.config.linear_conv_kernel_dim = 4;
    xfm.config.rms_norm_eps = 1e-6;
    xfm.config.gdn_state_f32 = state_f32;

    // Zero q/k/v projections and a ones input make `a` the column sum of a_w: each head's gate input is
    // exact and the recurrence reduces to state *= g.
    var a_w_host: [128 * 8]f32 = undefined;
    for (0..128) |r| for (0..8) |h| {
        a_w_host[r * 8 + h] = gdn_decay_a[h] / 128.0;
    };
    const a_w_f32 = mlx.mlx_array_new_data(&a_w_host, &[_]c_int{ hidden, hv }, 2, .float32);
    defer _ = mlx.mlx_array_free(a_w_f32);
    var a_w = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&a_w, a_w_f32, .bfloat16, s));
    const A_log_f32 = mlx.mlx_array_new_data(&gdn_decay_A_log, &[_]c_int{hv}, 1, .float32);
    defer _ = mlx.mlx_array_free(A_log_f32);
    var A_log = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&A_log, A_log_f32, .bfloat16, s));
    var zeros: [5]mlx.mlx_array = undefined;
    const zero_shapes = [5][2]c_int{ .{ hidden, c_dim }, .{ hidden, hv * 128 }, .{ hidden, hv }, .{ hv * 128, hidden }, .{ hv, 0 } };
    for (&zeros, zero_shapes, 0..) |*z, sh, i| {
        z.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_zeros(z, &sh, if (i == 4) 1 else 2, .bfloat16, s));
    }
    var ones_norm = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_ones(&ones_norm, &[_]c_int{128}, 1, .bfloat16, s));
    var prng = std.Random.DefaultPrng.init(0xDECA7);
    const la = LinearAttnWeights{
        .qkv_w = zeros[0],
        .qkv_s = .{},
        .qkv_b = .{},
        .z_w = zeros[1],
        .z_s = .{},
        .z_b = .{},
        .a_w = a_w,
        .a_s = .{},
        .a_b = .{},
        .b_w = zeros[2],
        .b_s = .{},
        .b_b = .{},
        .conv1d_w = try gdnRecurRand(prng.random(), &.{ c_dim, 4, 1 }, 1.0, .bfloat16, s),
        .A_log = A_log,
        .dt_bias = zeros[4],
        .norm_w = ones_norm,
        .out_w = zeros[3],
        .out_s = .{},
        .out_b = .{},
    };
    defer inline for (.{ la.qkv_w, la.z_w, la.a_w, la.b_w, la.conv1d_w, la.A_log, la.dt_bias, la.norm_w, la.out_w }) |arr| {
        _ = mlx.mlx_array_free(arr);
    };
    const state0 = try gdnRecurRand(prng.random(), &.{ 1, hv, 128, 128 }, 1.0, .bfloat16, s);
    defer _ = mlx.mlx_array_free(state0);
    const start = try gdnDecayHost(state0, s);
    defer testing.allocator.free(start);

    const Case = struct { route: GdnDecayRoute, per_call: c_int };
    const cases = [_]Case{
        .{ .route = .chain, .per_call = 1 },
        .{ .route = .prework_chain, .per_call = 1 },
        .{ .route = .decode, .per_call = 1 },
        .{ .route = .verify, .per_call = 2 },
        .{ .route = .verify, .per_call = 3 },
        .{ .route = .verify, .per_call = 4 },
        .{ .route = .verify, .per_call = 5 },
        .{ .route = .verify, .per_call = 6 },
        .{ .route = .verify, .per_call = 7 },
        .{ .route = .verify, .per_call = 8 },
        .{ .route = .fold, .per_call = 2 },
        .{ .route = .prefill, .per_call = gdn_decay_steps },
    };
    var finals: [cases.len][]f32 = undefined;
    var made: usize = 0;
    defer for (finals[0..made]) |f| testing.allocator.free(f);
    for (cases, 0..) |c, i| {
        var rows: []f32 = &.{};
        defer if (rows.len > 0) testing.allocator.free(rows);
        finals[i] = try gdnDecayRun(&xfm, &la, c.route, c.per_call, state0, c_dim, &rows);
        made += 1;
        if (state_f32) try gdnDecayExpectExact(start, finals[i], hv, @intCast(gdn_decay_steps), 1e-3);
        // Row j of the last call is the state after that many tokens: the rollback source.
        var j: usize = 0;
        while (state_f32 and rows.len > 0 and j + 1 < @as(usize, @intCast(c.per_call))) : (j += 1) {
            try gdnDecayExpectExact(start, rows[j * start.len ..][0..start.len], hv, @as(usize, @intCast(gdn_decay_steps - c.per_call)) + j + 1, 1e-3);
        }
    }
    // The fused arms share one gate formula and the chain arms MLX's: bit-equal inside a family.
    for (finals[3..11]) |f| try gdnDecayEqualBits(finals[2], f);
    try gdnDecayEqualBits(finals[1], finals[2]);
    // A prefill chunk holds its state in f32 across tokens; a bf16 state rounds it every token.
    if (state_f32) try gdnDecayEqualBits(finals[0], finals[11]);
    // Across families, precise::exp against MLX's exp differ by ulps of g, not a bf16 step.
    if (state_f32) try gdnDecayClose(finals[0], finals[2], 1e-4);
}

fn conv1dTailRetentionDelta(batch: c_int, seq: c_int, cdim: c_int, kernel: c_int) !usize {
    const t = std.testing;
    const s = mlx.gpuStream();
    const saved = trf.compact_conv_state_override;
    trf.compact_conv_state_override = true;
    defer trf.compact_conv_state_override = saved;
    var xfm: Transformer = undefined;
    xfm.s = s;
    xfm.spec_capture_ssm = false;

    var prng = std.Random.DefaultPrng.init(0xC0A1_7A11);
    const x = try attn256RandBf16(prng.random(), &[_]c_int{ batch, seq, cdim }, s);
    var x_live = true;
    defer if (x_live) {
        _ = mlx.mlx_array_free(x);
    };
    var w = mlx.mlx_array_new();
    var w_live = true;
    defer if (w_live) {
        _ = mlx.mlx_array_free(w);
    };
    const wshape = [_]c_int{ cdim, kernel, 1 };
    try mlx.check(mlx.mlx_ones(&w, &wshape, 3, .bfloat16, s));

    var entry: SSMCacheEntry = .{
        .conv_state = mlx.mlx_array_new(),
        .ssm_state = mlx.mlx_array_new(),
        .initialized = false,
    };
    defer {
        _ = mlx.mlx_array_free(entry.conv_state);
        _ = mlx.mlx_array_free(entry.ssm_state);
    }

    _ = mlx.mlx_synchronize(s);
    _ = mlx.mlx_clear_cache();
    var baseline: usize = 0;
    _ = mlx.mlx_get_active_memory(&baseline);

    const out = try xfm.conv1dWithCache(x, w, null, &entry, batch, cdim, kernel, false);
    var out_live = true;
    defer if (out_live) {
        _ = mlx.mlx_array_free(out);
    };
    try mlx.check(mlx.mlx_array_eval(out));
    try mlx.check(mlx.mlx_array_eval(entry.conv_state));
    out_live = false;
    x_live = false;
    w_live = false;
    _ = mlx.mlx_array_free(out);
    _ = mlx.mlx_array_free(x);
    _ = mlx.mlx_array_free(w);
    _ = mlx.mlx_synchronize(s);
    _ = mlx.mlx_clear_cache();
    var after: usize = 0;
    _ = mlx.mlx_get_active_memory(&after);
    const delta = if (after > baseline) after - baseline else 0;
    try t.expect(entry.initialized);
    const keep = kernel - 1;
    const sh = mlx.getShape(entry.conv_state);
    try t.expectEqual(@as(c_int, batch), sh[0]);
    try t.expectEqual(keep, sh[1]);
    try t.expectEqual(cdim, sh[2]);
    return delta;
}

fn conv1dBitCopy(batch: c_int, seq: c_int, cdim: c_int, kernel: c_int, compact: bool, s: mlx.mlx_stream) !struct { out: mlx.mlx_array, conv_state: mlx.mlx_array } {
    const saved = trf.compact_conv_state_override;
    trf.compact_conv_state_override = compact;
    defer trf.compact_conv_state_override = saved;

    var xfm: Transformer = undefined;
    xfm.s = s;
    xfm.spec_capture_ssm = false;
    var prng = std.Random.DefaultPrng.init(0xB17E_C0DE);
    const x = try attn256RandBf16(prng.random(), &[_]c_int{ batch, seq, cdim }, s);
    defer _ = mlx.mlx_array_free(x);
    var w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w);
    const wshape = [_]c_int{ cdim, kernel, 1 };
    try mlx.check(mlx.mlx_ones(&w, &wshape, 3, .bfloat16, s));
    var entry: SSMCacheEntry = .{
        .conv_state = mlx.mlx_array_new(),
        .ssm_state = mlx.mlx_array_new(),
        .initialized = false,
    };
    errdefer _ = mlx.mlx_array_free(entry.conv_state);
    defer _ = mlx.mlx_array_free(entry.ssm_state);
    const out = try xfm.conv1dWithCache(x, w, null, &entry, batch, cdim, kernel, true);
    return .{ .out = out, .conv_state = entry.conv_state };
}

fn conv1dTwoChunk(batch: c_int, t1: c_int, t2: c_int, cdim: c_int, kernel: c_int, compact: bool, s: mlx.mlx_stream) !struct { out: mlx.mlx_array, conv_state: mlx.mlx_array } {
    const saved = trf.compact_conv_state_override;
    trf.compact_conv_state_override = compact;
    defer trf.compact_conv_state_override = saved;

    var xfm: Transformer = undefined;
    xfm.s = s;
    xfm.spec_capture_ssm = false;
    var prng = std.Random.DefaultPrng.init(0xA11C_EDED);
    const x1 = try attn256RandBf16(prng.random(), &[_]c_int{ batch, t1, cdim }, s);
    defer _ = mlx.mlx_array_free(x1);
    const x2 = try attn256RandBf16(prng.random(), &[_]c_int{ batch, t2, cdim }, s);
    defer _ = mlx.mlx_array_free(x2);
    var w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w);
    const wshape = [_]c_int{ cdim, kernel, 1 };
    try mlx.check(mlx.mlx_ones(&w, &wshape, 3, .bfloat16, s));
    var entry: SSMCacheEntry = .{
        .conv_state = mlx.mlx_array_new(),
        .ssm_state = mlx.mlx_array_new(),
        .initialized = false,
    };
    errdefer _ = mlx.mlx_array_free(entry.conv_state);
    defer _ = mlx.mlx_array_free(entry.ssm_state);
    const out1 = try xfm.conv1dWithCache(x1, w, null, &entry, batch, cdim, kernel, true);
    _ = mlx.mlx_array_free(out1);
    const out2 = try xfm.conv1dWithCache(x2, w, null, &entry, batch, cdim, kernel, true);
    return .{ .out = out2, .conv_state = entry.conv_state };
}

/// Takes ownership of supplied projection slices; recurrent state remains per request.
pub fn gatedDeltaNetProjected(
    self: *Transformer,
    x: mlx.mlx_array,
    la: *const LinearAttnWeights,
    ssm: *SSMCacheEntry,
    layer_idx: usize,
    batch: c_int,
    seq_len: c_int,
    is_prefill: bool,
    projected: ?GdnVerifyInputs,
    skip_output: bool,
) !mlx.mlx_array {
    const cfg = &self.config;
    const num_k_heads: c_int = @intCast(cfg.linear_num_key_heads);
    const num_v_heads: c_int = @intCast(cfg.linear_num_value_heads);
    const dk: c_int = @intCast(cfg.linear_key_head_dim);
    const dv: c_int = @intCast(cfg.linear_value_head_dim);
    const key_dim: c_int = dk * num_k_heads;
    const value_dim: c_int = dv * num_v_heads;
    const conv_dim: c_int = key_dim * 2 + value_dim;
    const kernel: c_int = @intCast(cfg.linear_conv_kernel_dim);

    // Projections: the caller's, or separate (qkv+z+a+b)
    var qkv: mlx.mlx_array = undefined;
    var z_proj: mlx.mlx_array = undefined;
    var a_proj: mlx.mlx_array = undefined;
    var b_proj: mlx.mlx_array = undefined;
    defer _ = mlx.mlx_array_free(qkv);
    defer _ = mlx.mlx_array_free(z_proj);
    defer _ = mlx.mlx_array_free(a_proj);
    defer _ = mlx.mlx_array_free(b_proj);
    if (projected) |values| {
        qkv = values.qkv;
        z_proj = values.z;
        a_proj = values.a;
        b_proj = values.b;
    } else {
        const proj = try self.gdnProjGpu(x, la);
        qkv = proj.qkv;
        z_proj = proj.z;
        a_proj = try self.qmatmul(x, la.a_w, la.a_s, la.a_b);
        b_proj = try self.qmatmul(x, la.b_w, la.b_s, la.b_b);
    }
    // Scale scalars + the parameter-free rms_norm ones-weight (mlx-c
    // requires a non-empty weight) are cached on the Transformer — they
    // used to be rebuilt every layer every decode step.
    if (self.gdn_q_scale == null) {
        const inv_scale = 1.0 / @as(f32, @floatFromInt(cfg.linear_key_head_dim));
        self.gdn_q_scale = bf16Scalar(inv_scale, self.s);
        self.gdn_k_scale = bf16Scalar(@sqrt(inv_scale), self.s);
    }
    const inv_scale_sq = self.gdn_q_scale.?;
    const inv_sqrt_sc = self.gdn_k_scale.?;

    // A restored state continues in the width this load stores.
    const state_dtype = cfg.gdnStateDtype();
    if (ssm.ssm_state.ctx != null and mlx.mlx_array_dtype(ssm.ssm_state) != state_dtype) {
        var cast = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(cast);
        try mlx.check(mlx.mlx_astype(&cast, ssm.ssm_state, state_dtype, self.s));
        _ = mlx.mlx_array_free(ssm.ssm_state);
        ssm.ssm_state = cast;
    }

    // Port of mlx-serve #517: prework + recurrence in one dispatch, then the norm-gate,
    // at decode (S=1) and capturing verify widths (S 2..8); the chain below otherwise.
    if (batch == 1 and (projected == null or !self.spec_capture_ssm) and kernel == 4 and ssm.initialized and ssm.conv_state.ctx != null and
        ssm.ssm_state.ctx != null and (if (self.spec_capture_ssm) seq_len >= 2 else seq_len == 1) and
        !cfg.kda_vector_gate and !qwen4Standin().gdn_recur and
        !diagEnvOnCached(&gdn_check_env, "QWEN4_GDN_CHECK") and gdnDecodeRecurEnabled())
    recur: {
        const geo = gdn_decode.Geometry{ .hk = num_k_heads, .hv = num_v_heads, .dk = dk, .dv = dv };
        const inputs = gdn_decode.Inputs{
            .qkv = qkv,
            .a = a_proj,
            .b = b_proj,
            .conv_state = ssm.conv_state,
            .ssm_state = ssm.ssm_state,
            .conv_w = la.conv1d_w,
            .A_log = la.A_log,
            .dt_bias = la.dt_bias,
            .q_scale = inv_scale_sq,
            .k_scale = inv_sqrt_sc,
            .gate_dtype = cfg.gdnGateDtype(),
        };
        if (self.gdn_eps == null) self.gdn_eps = mlx.mlx_array_new_float(cfg.rms_norm_eps);
        if (self.spec_capture_ssm and gdnVerifyFoldFor(seq_len)) {
            if (try gdn_decode.stepFold(geo, seq_len, inputs, z_proj, la.norm_w, self.gdn_eps.?, !cfg.kda_sigmoid_out_gate, self.s)) |result| {
                var f = result;
                defer f.deinit();
                if (ssm.spec_conv_input.ctx != null) _ = mlx.mlx_array_free(ssm.spec_conv_input);
                ssm.spec_conv_input = f.conv_input;
                f.conv_input = .{ .ctx = null };
                if (ssm.spec_state_seq.ctx != null) _ = mlx.mlx_array_free(ssm.spec_state_seq);
                ssm.spec_state_seq = f.state_seq;
                f.state_seq = .{ .ctx = null };
                _ = mlx.mlx_array_free(ssm.conv_state);
                ssm.conv_state = f.conv_state;
                f.conv_state = .{ .ctx = null };
                _ = mlx.mlx_array_free(ssm.ssm_state);
                ssm.ssm_state = f.ssm_state;
                f.ssm_state = .{ .ctx = null };
                gdn_verify_recur_engaged = true;
                if (!gdn_verify_fold_engaged) {
                    gdn_verify_fold_engaged = true;
                    log.info("[gdn] verify fold engaged: S={d} Hk={d} Hv={d}\n", .{ seq_len, num_k_heads, num_v_heads });
                }
                return if (skip_output) standinRef(f.gated) else self.qmatmul(f.gated, la.out_w, la.out_s, la.out_b);
            }
        }
        var r = (try gdn_decode.step(geo, seq_len, inputs, false, self.s)) orelse break :recur;
        defer r.deinit();
        if (self.gdn_eps == null) self.gdn_eps = mlx.mlx_array_new_float(cfg.rms_norm_eps);
        const flat = (try gdnNormGateFused(self.s, r.y, z_proj, 0, value_dim, la.norm_w, self.gdn_eps.?, !cfg.kda_sigmoid_out_gate, num_v_heads, dv, 1, seq_len)) orelse break :recur;
        defer _ = mlx.mlx_array_free(flat);
        if (self.spec_capture_ssm) {
            // Rollback slices the conv INPUT, as the chain stashes it.
            var conv_input = mlx.mlx_array_new();
            errdefer _ = mlx.mlx_array_free(conv_input);
            const arr = [_]mlx.mlx_array{ ssm.conv_state, qkv };
            const vec = mlx.mlx_vector_array_new_data(&arr, 2);
            defer _ = mlx.mlx_vector_array_free(vec);
            try mlx.check(mlx.mlx_concatenate_axis(&conv_input, vec, 1, self.s));
            if (ssm.spec_conv_input.ctx != null) _ = mlx.mlx_array_free(ssm.spec_conv_input);
            ssm.spec_conv_input = conv_input;
            if (ssm.spec_state_seq.ctx != null) _ = mlx.mlx_array_free(ssm.spec_state_seq);
            ssm.spec_state_seq = .{ .ctx = null };
            const pre_state = ssm.ssm_state;
            ssm.ssm_state = .{ .ctx = null };
            try ssm.spec_replay.hold(pre_state, geo, inputs);
        }
        _ = mlx.mlx_array_free(ssm.conv_state);
        ssm.conv_state = r.conv_state;
        r.conv_state = .{ .ctx = null };
        if (ssm.ssm_state.ctx != null) _ = mlx.mlx_array_free(ssm.ssm_state);
        ssm.ssm_state = r.ssm_state;
        r.ssm_state = .{ .ctx = null };
        const once = if (seq_len == 1) &gdn_decode_recur_engaged else &gdn_verify_recur_engaged;
        if (!once.*) {
            once.* = true;
            log.info("[gdn] {s} recur engaged: S={d} Hk={d} Hv={d}\n", .{ if (seq_len == 1) "decode" else "verify", seq_len, num_k_heads, num_v_heads });
        }
        return if (skip_output) standinRef(flat) else self.qmatmul(flat, la.out_w, la.out_s, la.out_b);
    }

    var q_scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_scaled);
    var k_scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k_scaled);
    var v_heads = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v_heads);
    var fused_prework = false;
    var g_fused: ?mlx.mlx_array = null;
    var beta_fused: ?mlx.mlx_array = null;

    // Wide prework is token-local; a cold chunk supplies the same zero history as conv1dWithCache.
    const prefill_fused = cfg.longCtxGated() and is_prefill and gdnPrefillFusedFor(seq_len, batch);
    if (!cfg.kda_vector_gate and
        (batch * seq_len <= GDN_FUSED_MAX_ROWS or prefill_fused) and
        kernel == 4 and (ssm.initialized or prefill_fused))
    blk: {
        var zero_conv = mlx.mlx_array{ .ctx = null };
        defer if (zero_conv.ctx != null) {
            _ = mlx.mlx_array_free(zero_conv);
        };
        const cold = !ssm.initialized;
        if (cold) {
            zero_conv = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_zeros(&zero_conv, &[_]c_int{ batch, 3, conv_dim }, 3, .bfloat16, self.s));
        }
        const incoming_conv = if (cold) zero_conv else ssm.conv_state;
        const pre = (gdnPreworkFused(self.s, .{
            .qkv = qkv,
            .qkv_off = 0,
            .qkv_stride = conv_dim,
            .b = b_proj,
            .b_off = 0,
            .b_stride = num_v_heads,
            .a = a_proj,
            .a_off = 0,
            .a_stride = num_v_heads,
            .A_log = la.A_log,
            .dt_bias = la.dt_bias,
            .conv_state = incoming_conv,
            .conv_w = la.conv1d_w,
            .q_scale = inv_scale_sq,
            .k_scale = inv_sqrt_sc,
            .hk = num_k_heads,
            .hv = num_v_heads,
            .dk = dk,
            .dv = dv,
            .seq = seq_len,
            .batch = batch,
            .gate_dtype = cfg.gdnGateDtype(),
        }) catch null) orelse break :blk;
        // Spec-capture rollback slices the conv INPUT; build the concat
        // the composed path would have stashed (lazy, one launch).
        if (self.spec_capture_ssm) {
            const arr = [_]mlx.mlx_array{ incoming_conv, qkv };
            const vec = mlx.mlx_vector_array_new_data(&arr, 2);
            defer _ = mlx.mlx_vector_array_free(vec);
            if (ssm.spec_conv_input.ctx != null) _ = mlx.mlx_array_free(ssm.spec_conv_input);
            ssm.spec_conv_input = mlx.mlx_array_new();
            mlx.check(mlx.mlx_concatenate_axis(&ssm.spec_conv_input, vec, 1, self.s)) catch {
                // Free the fused outputs and fall back to the composed
                // chain rather than adopt state with no rollback capture.
                _ = mlx.mlx_array_free(pre.q);
                _ = mlx.mlx_array_free(pre.k);
                _ = mlx.mlx_array_free(pre.v);
                _ = mlx.mlx_array_free(pre.conv_state);
                _ = mlx.mlx_array_free(pre.g);
                _ = mlx.mlx_array_free(pre.beta);
                break :blk;
            };
        }
        if (diagEnvOnCached(&gdn_check_env, "QWEN4_GDN_CHECK")) self.gdnCheckFused(la, qkv, incoming_conv, b_proj, a_proj, &pre, layer_idx, num_k_heads, num_v_heads, dk, dv, batch, seq_len) catch |e| log.warn("[gdn-check] failed: {s}\n", .{@errorName(e)});
        _ = mlx.mlx_array_free(ssm.conv_state);
        ssm.conv_state = pre.conv_state;
        ssm.initialized = true;
        if (prefill_fused and seq_len >= 512 and !gdn_prefill_engaged) {
            gdn_prefill_engaged = true;
            log.info("[gdn-prefill] engaged: S={d} B={d} cold={}\n", .{ seq_len, batch, cold });
        }
        _ = mlx.mlx_array_free(q_scaled);
        _ = mlx.mlx_array_free(k_scaled);
        _ = mlx.mlx_array_free(v_heads);
        q_scaled = pre.q;
        k_scaled = pre.k;
        v_heads = pre.v;
        fused_prework = true;
        if (gdnDecodeFusedEnabled()) {
            g_fused = pre.g;
            beta_fused = pre.beta;
        } else {
            _ = mlx.mlx_array_free(pre.g);
            _ = mlx.mlx_array_free(pre.beta);
        }
    }

    if (!fused_prework) {
        // Conv1d with cache: prepend conv_state, apply depthwise conv + silu
        const conv_out = try self.conv1dWithCache(qkv, la.conv1d_w, null, ssm, batch, conv_dim, kernel, true);
        defer _ = mlx.mlx_array_free(conv_out);

        // Split conv output into Q, K, V
        // Q: [B, S, key_dim] → [B, S, num_k_heads, dk]
        // K: [B, S, key_dim] → [B, S, num_k_heads, dk]
        // V: [B, S, value_dim] → [B, S, num_v_heads, dv]
        const strides3 = [_]c_int{ 1, 1, 1 };
        var q_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_flat);
        {
            const start = [_]c_int{ 0, 0, 0 };
            const stop = [_]c_int{ batch, seq_len, key_dim };
            try mlx.check(mlx.mlx_slice(&q_flat, conv_out, &start, 3, &stop, 3, &strides3, 3, self.s));
        }
        var k_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_flat);
        {
            const start = [_]c_int{ 0, 0, key_dim };
            const stop = [_]c_int{ batch, seq_len, key_dim * 2 };
            try mlx.check(mlx.mlx_slice(&k_flat, conv_out, &start, 3, &stop, 3, &strides3, 3, self.s));
        }
        var v_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v_flat);
        {
            const start = [_]c_int{ 0, 0, key_dim * 2 };
            const stop = [_]c_int{ batch, seq_len, key_dim * 2 + value_dim };
            try mlx.check(mlx.mlx_slice(&v_flat, conv_out, &start, 3, &stop, 3, &strides3, 3, self.s));
        }

        // Reshape to head dims
        const q_shape = [_]c_int{ batch, seq_len, num_k_heads, dk };
        const k_shape = [_]c_int{ batch, seq_len, num_k_heads, dk };
        const v_shape = [_]c_int{ batch, seq_len, num_v_heads, dv };
        var q_heads = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_heads);
        var k_heads = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_heads);
        try mlx.check(mlx.mlx_reshape(&q_heads, q_flat, &q_shape, 4, self.s));
        try mlx.check(mlx.mlx_reshape(&k_heads, k_flat, &k_shape, 4, self.s));
        try mlx.check(mlx.mlx_reshape(&v_heads, v_flat, &v_shape, 4, self.s));

        // Q/K normalization: q = (1/dk) * rms_norm(q, null), k = (1/sqrt(dk)) * rms_norm(k, null)
        if (self.gdn_ones_w == null) {
            const ones_shape = [_]c_int{dk};
            var w = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_ones(&w, &ones_shape, 1, .bfloat16, self.s));
            self.gdn_ones_w = w;
        }
        const ones_w = self.gdn_ones_w.?;

        var q_norm = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_norm);
        try mlx.check(mlx.mlx_fast_rms_norm(&q_norm, q_heads, ones_w, 1e-6, self.s));
        try mlx.check(mlx.mlx_multiply(&q_scaled, q_norm, inv_scale_sq, self.s));

        var k_norm = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k_norm);
        try mlx.check(mlx.mlx_fast_rms_norm(&k_norm, k_heads, ones_w, 1e-6, self.s));
        try mlx.check(mlx.mlx_multiply(&k_scaled, k_norm, inv_sqrt_sc, self.s));
    }

    // Gating. GatedDeltaNet: g = exp(-exp(A_log) * softplus(a + dt_bias)),
    // one scalar per (token, value head) — one fused kernel via compiled
    // closure (mirrors mlx-lm's compute_g), raw chain as fallback.
    // [B, S, Hv]. KDA: one value per KEY CHANNEL — [B, S, Hv*Dk], which the
    // `_vec` kernel variant indexes.
    const vector_gate = cfg.kda_vector_gate;
    const g = g_fused orelse try self.computeGdnGate(la.A_log, a_proj, la.dt_bias);
    defer _ = mlx.mlx_array_free(g);

    // beta = sigmoid(b)
    const beta = beta_fused orelse blk: {
        var bt = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_sigmoid(&bt, b_proj, self.s)); // [B, S, Hv]
        break :blk bt;
    };
    defer _ = mlx.mlx_array_free(beta);

    // Initialize SSM state if needed.
    // Can't use ssm.initialized — conv1dWithCache already set it to true.
    // Check if ssm_state is empty (ctx == null) as the actual init indicator.
    if (ssm.ssm_state.ctx == null) {
        const state_shape = [_]c_int{ batch, num_v_heads, dv, dk };
        ssm.ssm_state = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_zeros(&ssm.ssm_state, &state_shape, 4, state_dtype, self.s));
        if (state_dtype == .float32 and !gdn_state_f32_logged) {
            gdn_state_f32_logged = true;
            log.info("[gdn] recurrent state f32\n", .{});
        }
    }

    // Fused Metal kernel: runs the full T-step delta recurrence in one dispatch.
    // Inputs (shapes): q,k [B,T,Hk,Dk] (GQA handled in kernel), v [B,T,Hv,Dv],
    //                  g,beta [B,T,Hv], state_in [B,Hv,Dv,Dk], T scalar.
    // Outputs: y [B,T,Hv,Dv], state_out [B,Hv,Dv,Dk].
    const T_scalar = mlx.mlx_array_new_int(seq_len);
    defer _ = mlx.mlx_array_free(T_scalar);

    const y_shape = [_]c_int{ batch, seq_len, num_v_heads, dv };

    // The kernels specialize on the ACTUAL widths, never an assumed one.
    // bf16 checkpoints land here as bf16, but an f16 checkpoint promotes
    // its activations to f32 (f16 ⊕ f32-scalar → f32) and a hardcoded
    // bf16 pointer type then fails to COMPILE, which is an uncatchable
    // MLX error that kills the server mid-request. `g`/`beta` are read
    // through their raw buffer names with an explicit float cast, so they
    // are width-agnostic and need no template arg. Outputs are ours: y is
    // bf16 and the state buffers keep the stored state's width (StT).
    const gdn_in_dtype = mlx.mlx_array_dtype(q_scaled);
    const gdn_in_itemsize = mlx.mlx_array_itemsize(q_scaled);
    const gdn_state_dtype = mlx.mlx_array_dtype(ssm.ssm_state);

    const config = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &y_shape, 4, .bfloat16));

    var y_bthd = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y_bthd);

    if (self.spec_capture_ssm) {
        // Spec verify pass: emit per-position states so partial-accept
        // rollback needs no re-forward. state_seq: [T, B, Hv, Dv, Dk]
        // (last row unwritten — capture-tail trim), final state as its
        // own [B, Hv, Dv, Dk] output.
        const state_seq_shape = [_]c_int{ seq_len, batch, num_v_heads, dv, dk };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &state_seq_shape, 5, gdn_state_dtype));
        const state_out_shape = [_]c_int{ batch, num_v_heads, dv, dk };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &state_out_shape, 4, gdn_state_dtype));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32, dv, batch * num_v_heads));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32, 4, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "InT", gdn_in_dtype));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "StT", gdn_state_dtype));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "OutT", .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dk", dk));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dv", dv));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hk", num_k_heads));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hv", num_v_heads));

        // seq_stride = per-timestep element stride into state_seq.
        const seq_stride = mlx.mlx_array_new_int(batch * num_v_heads * dv * dk);
        defer _ = mlx.mlx_array_free(seq_stride);
        const inputs_arr = [_]mlx.mlx_array{ q_scaled, k_scaled, v_heads, g, beta, ssm.ssm_state, T_scalar, seq_stride };
        const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
        defer _ = mlx.mlx_vector_array_free(inputs_vec);

        const gdn_kernel = try getGdnKernelSeq(vector_gate);
        var outputs_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(outputs_vec);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, gdn_kernel, inputs_vec, config, self.s));
        if (mlx.mlx_vector_array_size(outputs_vec) != 3) return error.MetalKernelBadOutputCount;
        try mlx.check(mlx.mlx_vector_array_get(&y_bthd, outputs_vec, 0));

        // Stash the per-position sequence for rollback (takes ownership).
        var state_seq = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&state_seq, outputs_vec, 1));
        if (ssm.spec_state_seq.ctx != null) _ = mlx.mlx_array_free(ssm.spec_state_seq);
        ssm.spec_state_seq = state_seq;

        // Continue normal flow with the kernel's own final state — no
        // slice view into the capture buffer (a view would pin the whole
        // [T,...] buffer across rounds and add 2 graph nodes per layer).
        var final_state = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&final_state, outputs_vec, 2));
        _ = mlx.mlx_array_free(ssm.ssm_state);
        ssm.ssm_state = final_state;
    } else {
        const state_shape_out = [_]c_int{ batch, num_v_heads, dv, dk };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &state_shape_out, 4, gdn_state_dtype));
        // Blocked-seq kernel at prefill widths on the 128-Dk/32-aligned-Dv
        // geometry (oMLX port, ~2x per GDN layer at 16K). Decode (T==1),
        // spec verify widths, and off-geometry shapes keep the stock
        // per-token kernel; the PLD capture path above is untouched.
        // Block size and input width are ONE decision: the staging arrays
        // are declared in the input dtype, so fp32 activations need a
        // smaller TB to fit the threadgroup budget. A null means even the
        // smallest block overflows — decline and stay on the stock kernel
        // rather than emit a kernel that cannot compile.
        // The blocked kernel reads a PER-HEAD gate; a per-channel (KDA)
        // gate is a different indexing contract, so it declines outright
        // rather than silently reading the wrong element.
        const prefill = gdnPrefillRoute(cfg.model_type, seq_len, dk, dv, num_k_heads, num_v_heads, vector_gate, gdn_in_itemsize);
        try gdnSetGrid(config, prefill.route, dv, num_v_heads, batch);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "InT", gdn_in_dtype));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "StT", gdn_state_dtype));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "OutT", .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dk", dk));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Dv", dv));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hk", num_k_heads));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "Hv", num_v_heads));

        const inputs_arr = [_]mlx.mlx_array{ q_scaled, k_scaled, v_heads, g, beta, ssm.ssm_state, T_scalar };
        const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
        defer _ = mlx.mlx_vector_array_free(inputs_vec);

        const gdn_kernel = try gdnRouteKernel(prefill.route, prefill.tb, vector_gate);
        var outputs_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(outputs_vec);
        if (qwen4Standin().gdn_recur) {
            _ = mlx.mlx_array_free(y_bthd);
            y_bthd = try standinRef(v_heads);
        } else {
            try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, gdn_kernel, inputs_vec, config, self.s));
            if (mlx.mlx_vector_array_size(outputs_vec) != 2) return error.MetalKernelBadOutputCount;
            try mlx.check(mlx.mlx_vector_array_get(&y_bthd, outputs_vec, 0));

            var new_state = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(&new_state, outputs_vec, 1));
            _ = mlx.mlx_array_free(ssm.ssm_state);
            ssm.ssm_state = new_state;
        }
    }

    // Fused epilogue: rms_norm(y) * silu(z) straight to the flat
    // out_proj input (decode/verify widths, per-head swish gate archs).
    if (fused_prework) {
        if (self.gdn_eps == null) self.gdn_eps = mlx.mlx_array_new_float(cfg.rms_norm_eps);
        if (try gdnNormGateFused(self.s, y_bthd, z_proj, 0, value_dim, la.norm_w, self.gdn_eps.?, !cfg.kda_sigmoid_out_gate, num_v_heads, dv, batch, seq_len)) |flat| {
            defer _ = mlx.mlx_array_free(flat);
            return if (skip_output) standinRef(flat) else self.qmatmul(flat, la.out_w, la.out_s, la.out_b);
        }
    }

    // Reshape z to [B, S, Hv, Dv]
    const z_shape = [_]c_int{ batch, seq_len, num_v_heads, dv };
    var z_heads = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(z_heads);
    try mlx.check(mlx.mlx_reshape(&z_heads, z_proj, &z_shape, 4, self.s));

    // RMSNormGated. FLA's FusedRMSNormGated applies the gate's ACTIVATION,
    // which the checkpoint chooses: qwen3.5 uses swish (silu(z) * y), KDA
    // uses a plain sigmoid (y * sigmoid(z)).
    const y_normed = try self.rmsNorm(y_bthd, la.norm_w);
    defer _ = mlx.mlx_array_free(y_normed);
    const out_gated = if (cfg.kda_sigmoid_out_gate) blk: {
        var gate = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gate);
        try mlx.check(mlx.mlx_sigmoid(&gate, z_heads, self.s));
        var gated = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&gated, y_normed, gate, self.s));
        break :blk gated;
    } else try self.swiglu(z_heads, y_normed);
    defer _ = mlx.mlx_array_free(out_gated);

    // Flatten [B, S, Hv, Dv] → [B, S, value_dim]
    const out_flat_shape = [_]c_int{ batch, seq_len, value_dim };
    var out_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(out_flat);
    try mlx.check(mlx.mlx_reshape(&out_flat, out_gated, &out_flat_shape, 3, self.s));

    return if (skip_output) standinRef(out_flat) else self.qmatmul(out_flat, la.out_w, la.out_s, la.out_b);
}

/// Prepends cached conv state, applies depthwise conv1d, updates cache.
/// If apply_silu is true, applies SiLU activation after conv.
/// conv_b is an optional bias added after conv1d.
pub fn conv1dWithCache(
    self: *Transformer,
    x: mlx.mlx_array,
    conv_w: mlx.mlx_array,
    conv_b: ?mlx.mlx_array,
    ssm: *SSMCacheEntry,
    batch: c_int,
    cdim: c_int,
    kernel: c_int,
    apply_silu: bool,
) !mlx.mlx_array {
    // Prepend conv_state or zeros
    var conv_input: mlx.mlx_array = undefined;
    defer _ = mlx.mlx_array_free(conv_input);
    if (ssm.initialized) {
        const arr = [_]mlx.mlx_array{ ssm.conv_state, x };
        const vec = mlx.mlx_vector_array_new_data(&arr, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        conv_input = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_concatenate_axis(&conv_input, vec, 1, self.s));
    } else {
        const zero_shape = [_]c_int{ batch, kernel - 1, cdim };
        var zero_state = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(zero_state);
        try mlx.check(mlx.mlx_zeros(&zero_state, &zero_shape, 3, .bfloat16, self.s));
        const arr = [_]mlx.mlx_array{ zero_state, x };
        const vec = mlx.mlx_vector_array_new_data(&arr, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        conv_input = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_concatenate_axis(&conv_input, vec, 1, self.s));
    }

    // Update conv_state: keep last (kernel-1) positions
    {
        const ci_shape = mlx.getShape(conv_input);
        const ci_len = ci_shape[1];
        const keep_start = ci_len - (kernel - 1);
        const start = [_]c_int{ 0, keep_start, 0 };
        const stop = [_]c_int{ batch, ci_len, cdim };
        const strides = [_]c_int{ 1, 1, 1 };
        var new_conv_state = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_slice(&new_conv_state, conv_input, &start, 3, &stop, 3, &strides, 3, self.s));
        errdefer _ = mlx.mlx_array_free(new_conv_state);
        // The tail must be an OWNED copy: a slice of the chunk-wide `conv_input` keeps
        // the whole chunk alive in this layer until its next forward. Only a chunk-wide
        // parent is worth the copy; at decode width the parent is already tiny.
        const t_rows = ci_len - (kernel - 1);
        if (compactConvStateEnabled() and t_rows >= PREFILL_EVAL_MIN_SEQ) {
            const owned = try materializedOwnedCopy(self.s, new_conv_state);
            _ = mlx.mlx_array_free(new_conv_state);
            new_conv_state = owned;
            trf.compact_conv_state_copy_count += 1;
        }
        _ = mlx.mlx_array_free(ssm.conv_state);
        ssm.conv_state = new_conv_state;
        ssm.initialized = true;
    }

    // PLD capture: stash the full conv input ([B, (kernel-1)+T, conv_dim])
    // so partial-accept rollback can slice the accepted-position conv_state
    // without a re-forward. Refcount-shared; freed at the end of the round.
    if (self.spec_capture_ssm) {
        if (ssm.spec_conv_input.ctx != null) _ = mlx.mlx_array_free(ssm.spec_conv_input);
        ssm.spec_conv_input = mlx.mlx_array_new();
        _ = mlx.mlx_array_set(&ssm.spec_conv_input, conv_input);
    }

    // Depthwise conv1d (groups = cdim)
    var conv_out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_conv1d(&conv_out, conv_input, conv_w, 1, 0, 1, cdim, self.s));

    // Optional bias
    if (conv_b) |cb| {
        var biased = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_add(&biased, conv_out, cb, self.s));
        _ = mlx.mlx_array_free(conv_out);
        conv_out = biased;
    }

    // Optional SiLU activation
    if (apply_silu) {
        const activated = try self.silu(conv_out);
        _ = mlx.mlx_array_free(conv_out);
        return activated;
    }
    return conv_out;
}

/// QWEN4_GDN_CHECK=1: run the composed prework beside the fused kernel
/// and log per-component max |diff| (diagnostic).
pub fn gdnCheckFused(self: *Transformer, la: *const LinearAttnWeights, qkv: mlx.mlx_array, old_state: mlx.mlx_array, b_proj: mlx.mlx_array, a_proj: mlx.mlx_array, pre: *const GdnPrework, layer_idx: usize, hk: c_int, hv: c_int, dk: c_int, dv: c_int, batch: c_int, seq: c_int) !void {
    const s = self.s;
    const key_dim = hk * dk;
    const value_dim = hv * dv;
    const c_dim = key_dim * 2 + value_dim;
    if (self.gdn_ones_w == null) {
        var w = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_ones(&w, &[_]c_int{dk}, 1, .bfloat16, s));
        self.gdn_ones_w = w;
    }
    var conv_input = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(conv_input);
    {
        const arr = [_]mlx.mlx_array{ old_state, qkv };
        const vec = mlx.mlx_vector_array_new_data(&arr, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&conv_input, vec, 1, s));
    }
    var ref_state = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref_state);
    try mlx.check(mlx.mlx_slice(&ref_state, conv_input, &[_]c_int{ 0, seq, 0 }, 3, &[_]c_int{ batch, seq + 3, c_dim }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    var conv_raw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(conv_raw);
    try mlx.check(mlx.mlx_conv1d(&conv_raw, conv_input, la.conv1d_w, 1, 0, 1, c_dim, s));
    const conv_out = try self.silu(conv_raw);
    defer _ = mlx.mlx_array_free(conv_out);
    const strides3 = [_]c_int{ 1, 1, 1 };
    var v_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v_flat);
    try mlx.check(mlx.mlx_slice(&v_flat, conv_out, &[_]c_int{ 0, 0, key_dim * 2 }, 3, &[_]c_int{ batch, seq, c_dim }, 3, &strides3, 3, s));
    var v_heads = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v_heads);
    try mlx.check(mlx.mlx_reshape(&v_heads, v_flat, &[_]c_int{ batch, seq, hv, dv }, 4, s));
    const dv_ = try attn256MaxDiff(pre.v, v_heads, s);
    var dq: f32 = 0;
    var dk_: f32 = 0;
    inline for (.{ .{ @as(c_int, 0), key_dim, self.gdn_q_scale.?, pre.q, &dq }, .{ key_dim, key_dim * 2, self.gdn_k_scale.?, pre.k, &dk_ } }) |case| {
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_slice(&flat, conv_out, &[_]c_int{ 0, 0, case[0] }, 3, &[_]c_int{ batch, seq, case[1] }, 3, &strides3, 3, s));
        var heads = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(heads);
        try mlx.check(mlx.mlx_reshape(&heads, flat, &[_]c_int{ batch, seq, hk, dk }, 4, s));
        var nrm = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(nrm);
        try mlx.check(mlx.mlx_fast_rms_norm(&nrm, heads, self.gdn_ones_w.?, 1e-6, s));
        var scaled = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(scaled);
        try mlx.check(mlx.mlx_multiply(&scaled, nrm, case[2], s));
        case[4].* = try attn256MaxDiff(case[3], scaled, s);
    }
    const dst = try attn256MaxDiff(pre.conv_state, ref_state, s);
    const g_ref = try self.computeGdnGate(la.A_log, a_proj, la.dt_bias);
    defer _ = mlx.mlx_array_free(g_ref);
    const dg = try attn256MaxDiff(pre.g, g_ref, s);
    var beta_ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(beta_ref);
    try mlx.check(mlx.mlx_sigmoid(&beta_ref, b_proj, s));
    const db = try attn256MaxDiff(pre.beta, beta_ref, s);
    log.info("[gdn-check] layer {d} S={d}: q {e} k {e} v {e} state {e} g {e} beta {e}\n", .{ layer_idx, seq, dq, dk_, dv_, dst, dg, db });
}

/// Apply the compiled GDN gate closure if available, else the raw chain.
/// Returns owned g (`config.gdnGateDtype()`, shape of `a`).
pub fn computeGdnGate(self: *const Transformer, A_log: mlx.mlx_array, a: mlx.mlx_array, dt_bias: mlx.mlx_array) !mlx.mlx_array {
    if (self.compiled_gdn_gate) |compiled| {
        const in_arr = [_]mlx.mlx_array{ A_log, a, dt_bias };
        const in_vec = mlx.mlx_vector_array_new_data(&in_arr, 3);
        defer _ = mlx.mlx_vector_array_free(in_vec);
        var out_vec = mlx.mlx_vector_array{ .ctx = null };
        try mlx.check(mlx.mlx_closure_apply(&out_vec, compiled, in_vec));
        defer _ = mlx.mlx_vector_array_free(out_vec);
        if (mlx.mlx_vector_array_size(out_vec) == 1) {
            var g = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_vector_array_get(&g, out_vec, 0));
            return g;
        }
    }
    return gdnGateChain(A_log, a, dt_bias, self.config.gdnGateDtype(), self.s);
}

pub fn gdnProjGpu(self: *Transformer, x: mlx.mlx_array, la: *const LinearAttnWeights) !GdnProj {
    const qkv = try self.qmatmul(x, la.qkv_w, la.qkv_s, la.qkv_b);
    errdefer _ = mlx.mlx_array_free(qkv);
    const z = try self.qmatmul(x, la.z_w, la.z_s, la.z_b);
    return .{ .qkv = qkv, .z = z };
}

test "gdnGateChain matches g = exp(-exp(A_log) * softplus(a + dt_bias))" {
    const s = mlx.gpuStream();

    // Hv = 2 heads, B=1, S=1. Hand-computable fixtures:
    //   head 0: A_log=0, a=0, dt_bias=0 → g = exp(-1 * softplus(0)) = exp(-ln2) = 0.5
    //   head 1: A_log=ln2, a=0, dt_bias=0 → g = exp(-2 * ln2) = 0.25
    const a_log_data = [_]f32{ 0.0, @log(2.0) };
    const hv_shape = [_]c_int{2};
    const A_log = mlx.mlx_array_new_data(&a_log_data, &hv_shape, 1, .float32);
    defer _ = mlx.mlx_array_free(A_log);

    const a_data = [_]f32{ 0.0, 0.0 };
    const a_shape = [_]c_int{ 1, 1, 2 };
    const a = mlx.mlx_array_new_data(&a_data, &a_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(a);

    const dt_data = [_]f32{ 0.0, 0.0 };
    const dt_bias = mlx.mlx_array_new_data(&dt_data, &hv_shape, 1, .float32);
    defer _ = mlx.mlx_array_free(dt_bias);

    const g = try gdnGateChain(A_log, a, dt_bias, .bfloat16, s);
    defer _ = mlx.mlx_array_free(g);

    var g_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g_f32);
    try mlx.check(mlx.mlx_astype(&g_f32, g, .float32, s));
    try mlx.check(mlx.mlx_array_eval(g_f32));

    const gd = mlx.mlx_array_data_float32(g_f32) orelse return error.InvalidDtype;
    // bf16 round-trip tolerance
    const tol: f32 = 5e-3;
    try testing.expect(@abs(gd[0] - 0.5) < tol);
    try testing.expect(@abs(gd[1] - 0.25) < tol);
}

test "GDN verify recurrence matches serial state rounding at every position" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const B: c_int = 1;
    const T: c_int = 4;
    const Hk: c_int = 1;
    const Hv: c_int = 2;
    const Dk: c_int = 128; // n_per_t = Dk/32 = 4 (matches real GatedDeltaNet head dim)
    const Dv: c_int = 4;

    // Deterministic pseudo-random inputs in [-0.5, 0.5].
    var prng = std.Random.DefaultPrng.init(0xC0FFEE);
    const rnd = prng.random();
    const qn: usize = @intCast(B * T * Hk * Dk);
    const vn: usize = @intCast(B * T * Hv * Dv);
    const gn: usize = @intCast(B * T * Hv);
    const sn: usize = @intCast(B * Hv * Dv * Dk);
    const qd = try testing.allocator.alloc(f32, qn);
    defer testing.allocator.free(qd);
    const kd = try testing.allocator.alloc(f32, qn);
    defer testing.allocator.free(kd);
    const vd = try testing.allocator.alloc(f32, vn);
    defer testing.allocator.free(vd);
    const gd = try testing.allocator.alloc(f32, gn);
    defer testing.allocator.free(gd);
    const bd = try testing.allocator.alloc(f32, gn);
    defer testing.allocator.free(bd);
    const sd = try testing.allocator.alloc(f32, sn);
    defer testing.allocator.free(sd);
    for (qd) |*x| x.* = rnd.float(f32) - 0.5;
    for (kd) |*x| x.* = rnd.float(f32) - 0.5;
    for (vd) |*x| x.* = rnd.float(f32) - 0.5;
    for (gd) |*x| x.* = 0.5 + 0.4 * rnd.float(f32); // decay in (0.5,0.9)
    for (bd) |*x| x.* = rnd.float(f32);
    for (sd) |*x| x.* = rnd.float(f32) - 0.5;

    const qsh = [_]c_int{ B, T, Hk, Dk };
    const vsh = [_]c_int{ B, T, Hv, Dv };
    const gsh = [_]c_int{ B, T, Hv };
    const ssh = [_]c_int{ B, Hv, Dv, Dk };
    const q32 = mlx.mlx_array_new_data(qd.ptr, &qsh, 4, .float32);
    defer _ = mlx.mlx_array_free(q32);
    const k32 = mlx.mlx_array_new_data(kd.ptr, &qsh, 4, .float32);
    defer _ = mlx.mlx_array_free(k32);
    const v32 = mlx.mlx_array_new_data(vd.ptr, &vsh, 4, .float32);
    defer _ = mlx.mlx_array_free(v32);
    const g32 = mlx.mlx_array_new_data(gd.ptr, &gsh, 3, .float32);
    defer _ = mlx.mlx_array_free(g32);
    const b32 = mlx.mlx_array_new_data(bd.ptr, &gsh, 3, .float32);
    defer _ = mlx.mlx_array_free(b32);
    const st32 = mlx.mlx_array_new_data(sd.ptr, &ssh, 4, .float32);
    defer _ = mlx.mlx_array_free(st32);

    // Cast inputs to bf16 (kernel expects InT/StT = bf16).
    var q = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q);
    var kk = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(kk);
    var v = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v);
    var g = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g);
    var beta = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(beta);
    var st = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(st);
    try mlx.check(mlx.mlx_astype(&q, q32, .bfloat16, s));
    try mlx.check(mlx.mlx_astype(&kk, k32, .bfloat16, s));
    try mlx.check(mlx.mlx_astype(&v, v32, .bfloat16, s));
    try mlx.check(mlx.mlx_astype(&g, g32, .bfloat16, s));
    try mlx.check(mlx.mlx_astype(&beta, b32, .bfloat16, s));
    try mlx.check(mlx.mlx_astype(&st, st32, .bfloat16, s));

    const state_a = try gdnTestRunSerial(false, q, kk, v, g, beta, st, B, T, Hk, Hv, Dk, Dv, s);
    defer _ = mlx.mlx_array_free(state_a);
    const state_b = try gdnTestRun(true, false, q, kk, v, g, beta, st, B, T, Hk, Hv, Dk, Dv, s);
    defer _ = mlx.mlx_array_free(state_b);

    try qkvExpectBitsEqual(s, state_a, state_b);

    const acc_run: c_int = 2; // accepted+1 = 2 tokens
    const q2_sh = [_]c_int{ B, acc_run, Hk, Dk };
    const v2_sh = [_]c_int{ B, acc_run, Hv, Dv };
    const g2_sh = [_]c_int{ B, acc_run, Hv };
    const z3 = [_]c_int{ 1, 1, 1, 1 };
    var q2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q2);
    var k2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k2);
    var v2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v2);
    var g2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g2);
    var b2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b2);
    try mlx.check(mlx.mlx_slice(&q2, q, &[_]c_int{ 0, 0, 0, 0 }, 4, &q2_sh, 4, &z3, 4, s));
    try mlx.check(mlx.mlx_slice(&k2, kk, &[_]c_int{ 0, 0, 0, 0 }, 4, &q2_sh, 4, &z3, 4, s));
    try mlx.check(mlx.mlx_slice(&v2, v, &[_]c_int{ 0, 0, 0, 0 }, 4, &v2_sh, 4, &z3, 4, s));
    try mlx.check(mlx.mlx_slice(&g2, g, &[_]c_int{ 0, 0, 0 }, 3, &g2_sh, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
    try mlx.check(mlx.mlx_slice(&b2, beta, &[_]c_int{ 0, 0, 0 }, 3, &g2_sh, 3, &[_]c_int{ 1, 1, 1 }, 3, s));

    const ref2 = try gdnTestRunSerial(false, q2, k2, v2, g2, b2, st, B, acc_run, Hk, Hv, Dk, Dv, s);
    defer _ = mlx.mlx_array_free(ref2);

    // Pull position 1 out of the length-T seq run.
    const state_seq_full = try gdnTestRunSeqAt(q, kk, v, g, beta, st, B, T, Hk, Hv, Dk, Dv, 1, s);
    defer _ = mlx.mlx_array_free(state_seq_full);

    try qkvExpectBitsEqual(s, ref2, state_seq_full);
}

test "GDN vectorized-gate kernel: a per-channel gate that is UNIFORM per head equals the scalar kernel" {
    try mlx.requireGpu();
    // KDA gates each state element by its own key channel. The invariant that
    // pins the indexing: broadcast a per-head decay across that head's Dk
    // channels and the `_vec` kernel must reproduce the scalar kernel BIT FOR
    // BIT — same recurrence, same reduction order, only the gate fetch differs.
    // A wrong stride/base in the `_vec` arm (reading head 0's channel for every
    // head, or walking Hv instead of Hv*Dk per step) survives every "output is
    // finite and roughly decays" check but fails this one.
    const s = mlx.gpuStream();
    const B: c_int = 1;
    const T: c_int = 5;
    const Hk: c_int = 2;
    const Hv: c_int = 2;
    const Dk: c_int = 128;
    const Dv: c_int = 128;

    var prng = std.Random.DefaultPrng.init(0x11CE);
    const rnd = prng.random();
    const qn: usize = @intCast(B * T * Hk * Dk);
    const vn: usize = @intCast(B * T * Hv * Dv);
    const gn: usize = @intCast(B * T * Hv);
    const gvn: usize = @intCast(B * T * Hv * Dk);
    const sn: usize = @intCast(B * Hv * Dv * Dk);
    const qd = try testing.allocator.alloc(f32, qn);
    defer testing.allocator.free(qd);
    const kd = try testing.allocator.alloc(f32, qn);
    defer testing.allocator.free(kd);
    const vd = try testing.allocator.alloc(f32, vn);
    defer testing.allocator.free(vd);
    const gd = try testing.allocator.alloc(f32, gn);
    defer testing.allocator.free(gd);
    const gvd = try testing.allocator.alloc(f32, gvn);
    defer testing.allocator.free(gvd);
    const bd = try testing.allocator.alloc(f32, gn);
    defer testing.allocator.free(bd);
    const sd = try testing.allocator.alloc(f32, sn);
    defer testing.allocator.free(sd);
    for (qd) |*x| x.* = rnd.float(f32) - 0.5;
    for (kd) |*x| x.* = rnd.float(f32) - 0.5;
    for (vd) |*x| x.* = rnd.float(f32) - 0.5;
    for (gd) |*x| x.* = 0.5 + 0.4 * rnd.float(f32);
    for (bd) |*x| x.* = rnd.float(f32);
    for (sd) |*x| x.* = rnd.float(f32) - 0.5;
    // Broadcast each (token, head) decay across that head's Dk channels.
    for (0..gn) |i| {
        for (0..@as(usize, @intCast(Dk))) |c| gvd[i * @as(usize, @intCast(Dk)) + c] = gd[i];
    }

    const qsh = [_]c_int{ B, T, Hk, Dk };
    const vsh = [_]c_int{ B, T, Hv, Dv };
    const gsh = [_]c_int{ B, T, Hv };
    const gvsh = [_]c_int{ B, T, Hv, Dk };
    const ssh = [_]c_int{ B, Hv, Dv, Dk };
    const raw = [_]mlx.mlx_array{
        mlx.mlx_array_new_data(qd.ptr, &qsh, 4, .float32),
        mlx.mlx_array_new_data(kd.ptr, &qsh, 4, .float32),
        mlx.mlx_array_new_data(vd.ptr, &vsh, 4, .float32),
        mlx.mlx_array_new_data(gd.ptr, &gsh, 3, .float32),
        mlx.mlx_array_new_data(gvd.ptr, &gvsh, 4, .float32),
        mlx.mlx_array_new_data(bd.ptr, &gsh, 3, .float32),
        mlx.mlx_array_new_data(sd.ptr, &ssh, 4, .float32),
    };
    defer for (raw) |r| {
        _ = mlx.mlx_array_free(r);
    };
    var bf: [raw.len]mlx.mlx_array = undefined;
    for (raw, 0..) |r, i| {
        bf[i] = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&bf[i], r, .bfloat16, s));
    }
    defer for (bf) |a| {
        _ = mlx.mlx_array_free(a);
    };
    const q = bf[0];
    const kk = bf[1];
    const v = bf[2];
    const g_head = bf[3];
    const g_chan = bf[4];
    const beta = bf[5];
    const st = bf[6];

    const state_scalar = try gdnTestRun(false, false, q, kk, v, g_head, beta, st, B, T, Hk, Hv, Dk, Dv, s);
    defer _ = mlx.mlx_array_free(state_scalar);
    const state_vec = try gdnTestRun(false, true, q, kk, v, g_chan, beta, st, B, T, Hk, Hv, Dk, Dv, s);
    defer _ = mlx.mlx_array_free(state_vec);

    var a32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a32);
    var b32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b32);
    try mlx.check(mlx.mlx_astype(&a32, state_scalar, .float32, s));
    try mlx.check(mlx.mlx_astype(&b32, state_vec, .float32, s));
    try mlx.check(mlx.mlx_array_eval(a32));
    try mlx.check(mlx.mlx_array_eval(b32));
    const ad = mlx.mlx_array_data_float32(a32) orelse return error.InvalidDtype;
    const bd2 = mlx.mlx_array_data_float32(b32) orelse return error.InvalidDtype;
    var max_diff: f32 = 0;
    var any_nonzero = false;
    for (0..sn) |i| {
        max_diff = @max(max_diff, @abs(ad[i] - bd2[i]));
        if (@abs(ad[i]) > 1e-3) any_nonzero = true;
    }
    // Guard the guard: an all-zero state would pass any diff check.
    try testing.expect(any_nonzero);
    try testing.expectEqual(@as(f32, 0), max_diff);

    // And the seq-capture variant of the same kernel agrees with its own
    // scalar twin, so spec rollback stays correct on a vector-gated arch.
    const seq_scalar = try gdnTestRun(true, false, q, kk, v, g_head, beta, st, B, T, Hk, Hv, Dk, Dv, s);
    defer _ = mlx.mlx_array_free(seq_scalar);
    const seq_vec = try gdnTestRun(true, true, q, kk, v, g_chan, beta, st, B, T, Hk, Hv, Dk, Dv, s);
    defer _ = mlx.mlx_array_free(seq_vec);
    var c32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c32);
    var d32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(d32);
    try mlx.check(mlx.mlx_astype(&c32, seq_scalar, .float32, s));
    try mlx.check(mlx.mlx_astype(&d32, seq_vec, .float32, s));
    try mlx.check(mlx.mlx_array_eval(c32));
    try mlx.check(mlx.mlx_array_eval(d32));
    const cd = mlx.mlx_array_data_float32(c32) orelse return error.InvalidDtype;
    const dd = mlx.mlx_array_data_float32(d32) orelse return error.InvalidDtype;
    var seq_diff: f32 = 0;
    for (0..sn) |i| seq_diff = @max(seq_diff, @abs(cd[i] - dd[i]));
    try testing.expectEqual(@as(f32, 0), seq_diff);
}

test "GDN pipelined route: NAX Qwen default, eligibility" {
    const saved_blocked = gdn_blocked_override;
    const saved_nax = trf.vqmm_nax_probe_override;
    defer {
        gdn_blocked_override = saved_blocked;
        trf.vqmm_nax_probe_override = saved_nax;
    }
    gdn_blocked_override = true;
    trf.vqmm_nax_probe_override = true;
    try testing.expectEqual(GdnRoute.pipelined, gdnPrefillRoute("qwen4_exp", 64, 128, 128, 2, 4, false, 2).route);
    for ([_][]const u8{ "mimo_v2", "glm5_next", "not_a_served_arch" }) |model_type| {
        try testing.expectEqual(GdnRoute.blocked, gdnPrefillRoute(model_type, 64, 128, 128, 2, 4, false, 2).route);
    }
    try testing.expectEqual(GdnRoute.pipelined, gdnPrefillRoute("qwen4_exp", 64, 128, 128, 2, 4, false, 2).route);
    trf.vqmm_nax_probe_override = false;
    try testing.expectEqual(GdnRoute.blocked, gdnPrefillRoute("qwen4_exp", 64, 128, 128, 2, 4, false, 2).route);
    trf.vqmm_nax_probe_override = true;
    try testing.expectEqual(GdnRoute.pipelined, gdnPrefillRoute("qwen4_exp", 64, 128, 128, 2, 4, false, 2).route);
    for ([_]c_int{ 1, 2, 8, 63 }) |seq_len| {
        try testing.expectEqual(GdnRoute.stock, gdnPrefillRoute("qwen4_exp", seq_len, 128, 128, 2, 4, false, 2).route);
    }
    const shapes = [_][4]c_int{
        .{ 64, 128, 2, 4 },
        .{ 128, 16, 2, 4 },
        .{ 128, 48, 2, 4 },
        .{ 128, 128, 0, 4 },
        .{ 128, 128, 3, 4 },
    };
    for (shapes) |shape| {
        try testing.expectEqual(GdnRoute.stock, gdnPrefillRoute("qwen4_exp", 64, shape[0], shape[1], shape[2], shape[3], false, 2).route);
    }
    try testing.expectEqual(GdnRoute.stock, gdnPrefillRoute("qwen4_exp", 64, 128, 128, 2, 4, true, 2).route);
    try testing.expectEqual(GdnRoute.stock, gdnPrefillRoute("qwen4_exp", 64, 128, 128, 2, 4, false, 16).route);
    const fp32 = gdnPrefillRoute("qwen4_exp", 64, 128, 128, 2, 4, false, 4);
    try testing.expectEqual(GdnRoute.pipelined, fp32.route);
    try testing.expectEqual(@as(u32, 16), fp32.tb);
    gdn_blocked_override = false;
    try testing.expectEqual(GdnRoute.stock, gdnPrefillRoute("qwen4_exp", 64, 128, 128, 2, 4, false, 2).route);
}

test "GDN blocked-seq and pipelined kernels: no worse than stock vs f64 ground truth (T/GQA/Dv/TB sweep)" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    // T deliberately hits non-multiples of every supported TB (16/32/48);
    // Hk<Hv exercises GQA head mapping; Dv 32..128 exercises 1..4 DB blocks.
    const cases = [_]GdnCase{
        .{ .B = 1, .T = 100, .Hk = 2, .Hv = 4, .Dv = 128, .tb = 32 }, // live-like 27B geometry (fewer heads)
        .{ .B = 2, .T = 64, .Hk = 1, .Hv = 2, .Dv = 32, .tb = 32 }, // batch>1, minimal Dv
        .{ .B = 1, .T = 65, .Hk = 1, .Hv = 2, .Dv = 64, .tb = 16 }, // TB=16 variant, ragged tail 1
        .{ .B = 1, .T = 49, .Hk = 2, .Hv = 2, .Dv = 32, .tb = 48 }, // TB=48 variant, ragged tail 1
        // fp32 activations (f16 checkpoints promote to them — the geometry
        // that killed the server on prism-ml/Ternary-Bonsai-27B-mlx-2bit).
        // The kernel must specialize on the real width AND clamp the block
        // size to fit the doubled staging; requesting 32/48 exercises both.
        .{ .B = 1, .T = 100, .Hk = 2, .Hv = 4, .Dv = 128, .tb = 32, .in_dtype = .float32 },
        .{ .B = 1, .T = 49, .Hk = 2, .Hv = 2, .Dv = 32, .tb = 48, .in_dtype = .float32 },
        .{ .B = 2, .T = 64, .Hk = 1, .Hv = 2, .Dv = 64, .tb = 16, .in_dtype = .float32 },
        // f16 activations: same 2-byte staging as bf16, different kernel.
        .{ .B = 1, .T = 100, .Hk = 2, .Hv = 4, .Dv = 64, .tb = 32, .in_dtype = .float16 },
        .{ .B = 1, .T = 100, .Hk = 2, .Hv = 4, .Dv = 128, .tb = 32, .state_dtype = .float32 },
        .{ .B = 1, .T = 65, .Hk = 1, .Hv = 2, .Dv = 64, .tb = 16, .in_dtype = .float32, .state_dtype = .float32 },
        .{ .B = 1, .T = 100, .Hk = 2, .Hv = 4, .Dv = 64, .tb = 32, .in_dtype = .float16, .state_dtype = .float32 },
        .{ .B = 1, .T = 72, .Hk = 2, .Hv = 4, .Dv = 128, .tb = 48 },
    };
    for (cases) |case| try gdnBlockedParityCase(case, s);
}

test "GDN blocked-seq kernel: chunk-boundary state continuity (split run == full run)" {
    try mlx.requireGpu();
    try gdnChunkContinuity(.blocked, 48, .bfloat16);
}

test "GDN pipelined kernel: chunk-boundary state continuity (split run == full run)" {
    try mlx.requireGpu();
    for ([_]c_int{ 48, 49 }) |split| {
        for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |state_dtype| {
            try gdnChunkContinuity(.pipelined, split, state_dtype);
        }
    }
}

test "gdnBlockTFor: the block size follows the INPUT WIDTH, not a bf16 assumption" {
    // bf16/f16 stage 2 bytes — every supported block size fits, so the
    // requested one stands.
    try testing.expectEqual(@as(?u32, 16), gdnBlockTFor(16, 128, 2));
    try testing.expectEqual(@as(?u32, 32), gdnBlockTFor(32, 128, 2));
    try testing.expectEqual(@as(?u32, 48), gdnBlockTFor(48, 128, 2));

    // fp32 activations (what an f16 checkpoint promotes to) stage 4 bytes:
    // TB=32 needs ~40 KiB and TB=48 ~60 KiB, both past Metal's 32 KiB
    // threadgroup budget, so each clamps DOWN rather than emitting a kernel
    // that fails to compile and takes the server with it.
    try testing.expectEqual(@as(?u32, 16), gdnBlockTFor(16, 128, 4));
    try testing.expectEqual(@as(?u32, 16), gdnBlockTFor(32, 128, 4));
    try testing.expectEqual(@as(?u32, 16), gdnBlockTFor(48, 128, 4));

    // The staging budget the clamp is derived from — pinned so a change to
    // the kernel's threadgroup declarations has to come here too.
    try testing.expect(gdnBlockedTgBytes(48, 128, 2) <= METAL_TG_BYTES);
    try testing.expect(gdnBlockedTgBytes(16, 128, 4) <= METAL_TG_BYTES);
    try testing.expect(gdnBlockedTgBytes(32, 128, 4) > METAL_TG_BYTES);

    // Nothing fits at all → decline the blocked route (null), never fall
    // through to an over-budget dispatch.
    try testing.expectEqual(@as(?u32, null), gdnBlockTFor(48, 128, 16));
    // A request below the smallest supported block declines too.
    try testing.expectEqual(@as(?u32, null), gdnBlockTFor(8, 128, 2));
}

test "gdnBlockedEligible: width floor + exact-128 Dk + Dv/head-group alignment" {
    // Live 27B GDN geometry at prefill width: eligible.
    try testing.expect(gdnBlockedEligible(2048, 128, 128, 16, 32));
    try testing.expect(gdnBlockedEligible(GDN_BLOCKED_MIN_T, 128, 128, 16, 32));
    // Below the width floor (decode, spec verify widths): stock kernel.
    try testing.expect(!gdnBlockedEligible(GDN_BLOCKED_MIN_T - 1, 128, 128, 16, 32));
    try testing.expect(!gdnBlockedEligible(1, 128, 128, 16, 32));
    // Off-geometry: Dk != 128, Dv not a multiple of 32, ragged GQA grouping.
    try testing.expect(!gdnBlockedEligible(2048, 96, 128, 16, 32));
    try testing.expect(!gdnBlockedEligible(2048, 256, 128, 16, 32));
    try testing.expect(!gdnBlockedEligible(2048, 128, 48, 16, 32));
    try testing.expect(!gdnBlockedEligible(2048, 128, 16, 16, 32));
    try testing.expect(!gdnBlockedEligible(2048, 128, 128, 3, 32));
}

test "gdn packed prework: bit-identical to the composed chain at decode and prefill widths incl. gate/beta + folded layout" {
    try mlx.requireGpu();
    try gdnPreworkCase(.bfloat16);
    try gdnPreworkCase(.float32);
}

test "gdn packed prework: fused at production Hk/Hv serves B=6,8,16 S=1 per row" {
    try mlx.requireGpu();
    try gdnPreworkBatchedCase(.bfloat16);
    try gdnPreworkBatchedCase(.float32);
}

test "gdn norm-gate fused: bit-identical to rms_norm + silu(z) * y at decode and prefill widths, z at a folded offset" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    gdn_prefill_fused_override = true;
    defer gdn_prefill_fused_override = null;
    gdn_decode_fused_override = true;
    defer gdn_decode_fused_override = null;
    var prng = std.Random.DefaultPrng.init(0x9A7E);
    const rnd = prng.random();
    const hv: c_int = 4;
    const dv: c_int = 128;
    const value_dim = hv * dv;
    const w_shape = [_]c_int{dv};
    const norm_w = try attn256RandBf16(rnd, &w_shape, s);
    defer _ = mlx.mlx_array_free(norm_w);
    const eps: f32 = 1e-6;
    const eps_arr = mlx.mlx_array_new_float(eps);
    defer _ = mlx.mlx_array_free(eps_arr);

    for ([_]c_int{ 1, 4, 9, 17, 65, 513, 8703 }) |seq| {
        const y_shape = [_]c_int{ 1, seq, hv, dv };
        const y = try attn256RandBf16Scaled(rnd, &y_shape, 8.0, s);
        defer _ = mlx.mlx_array_free(y);
        // z lives at column offset 96 of a wider row.
        const z_off: c_int = 96;
        const width: c_int = z_off + value_dim + 40;
        const raw_shape = [_]c_int{ 1, seq, width };
        const raw = try attn256RandBf16Scaled(rnd, &raw_shape, 12.0, s);
        defer _ = mlx.mlx_array_free(raw);

        if (seq >= 17) {
            gdn_prefill_fused_override = false;
            try testing.expect((try gdnNormGateFused(s, y, raw, z_off, width, norm_w, eps_arr, false, hv, dv, 1, seq)) == null);
            gdn_prefill_fused_override = true;
        }

        // Composed: rms_norm(y, w), then either silu(z) * normed (swish) or
        // normed * sigmoid(z) (sigmoid gate), flattened.
        var normed = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(normed);
        try mlx.check(mlx.mlx_fast_rms_norm(&normed, y, norm_w, eps, s));
        const z_flat = try sliceLastDim(raw, z_off, z_off + value_dim, s);
        defer _ = mlx.mlx_array_free(z_flat);
        var z_heads = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(z_heads);
        try mlx.check(mlx.mlx_reshape(&z_heads, z_flat, &y_shape, 4, s));
        var sig = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sig);
        try mlx.check(mlx.mlx_sigmoid(&sig, z_heads, s));
        for ([_]bool{ true, false }) |swish| {
            const fused = (try gdnNormGateFused(s, y, raw, z_off, width, norm_w, eps_arr, swish, hv, dv, 1, seq)) orelse return error.FusedDeclined;
            defer _ = mlx.mlx_array_free(fused);
            var gated = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(gated);
            if (swish) {
                var act = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(act);
                try mlx.check(mlx.mlx_multiply(&act, z_heads, sig, s));
                try mlx.check(mlx.mlx_multiply(&gated, act, normed, s));
            } else {
                try mlx.check(mlx.mlx_multiply(&gated, normed, sig, s));
            }
            const flat_shape = [_]c_int{ 1, seq, value_dim };
            var ref = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ref);
            try mlx.check(mlx.mlx_reshape(&ref, gated, &flat_shape, 3, s));
            try std.testing.expectEqual(@as(f32, 0.0), try attn256MaxDiff(fused, ref, s));
        }
    }
    // Decline: fusions off.
    gdn_decode_fused_override = false;
    const y_shape = [_]c_int{ 1, 1, hv, dv };
    const y = try attn256RandBf16(rnd, &y_shape, s);
    defer _ = mlx.mlx_array_free(y);
    const z_shape = [_]c_int{ 1, 1, value_dim };
    const z = try attn256RandBf16(rnd, &z_shape, s);
    defer _ = mlx.mlx_array_free(z);
    try std.testing.expect((try gdnNormGateFused(s, y, z, 0, value_dim, norm_w, eps_arr, true, hv, dv, 1, 1)) == null);
}

test "gdn_decode.step: bit-identical to prework -> recurrence -> norm-gate at S 1..8; every captured row is serial decode" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    gdn_prework_override = true;
    defer gdn_prework_override = null;
    gdn_decode_fused_override = true;
    defer gdn_decode_fused_override = null;
    for ([_][2]c_int{ .{ 16, 48 }, .{ 2, 8 } }) |heads| {
        var t: c_int = 1;
        while (t <= gdn_decode.MAX_SEQ) : (t += 1) {
            try gdnRecurParityCase(s, heads[0], heads[1], t, false, false);
            try gdnRecurParityCase(s, heads[0], heads[1], t, false, true);
        }
    }

    // The chain's prework and norm-gate serve bf16 activations only: f16 activations or a width
    // past MAX_SEQ keep the chain. An f32 state is served, and keeps its width.
    var prng = std.Random.DefaultPrng.init(0xF16);
    const rnd = prng.random();
    const hk: c_int = 2;
    const hv: c_int = 8;
    const c_dim: c_int = 2 * hk * 128 + hv * 128;
    const scale = bf16Scalar(0.1, s);
    defer _ = mlx.mlx_array_free(scale);
    for ([_][2]mlx.mlx_dtype{ .{ .float16, .bfloat16 }, .{ .bfloat16, .bfloat16 } }, 0..) |dts, case| {
        const dt = dts[0];
        const t_len: c_int = if (case == 1) gdn_decode.MAX_SEQ + 1 else 1;
        const qkv = try gdnRecurRand(rnd, &.{ 1, t_len, c_dim }, 1.0, dt, s);
        defer _ = mlx.mlx_array_free(qkv);
        const ab = try gdnRecurRand(rnd, &.{ 1, t_len, hv }, 1.0, dt, s);
        defer _ = mlx.mlx_array_free(ab);
        const conv_state = try gdnRecurRand(rnd, &.{ 1, 3, c_dim }, 1.0, dt, s);
        defer _ = mlx.mlx_array_free(conv_state);
        const ssm_state = try gdnRecurRand(rnd, &.{ 1, hv, 128, 128 }, 1.0, dts[1], s);
        defer _ = mlx.mlx_array_free(ssm_state);
        const conv_w = try gdnRecurRand(rnd, &.{ c_dim, 4, 1 }, 1.0, dt, s);
        defer _ = mlx.mlx_array_free(conv_w);
        const per_head = try gdnRecurRand(rnd, &.{hv}, 1.0, dt, s);
        defer _ = mlx.mlx_array_free(per_head);
        const in = gdn_decode.Inputs{ .qkv = qkv, .a = ab, .b = ab, .conv_state = conv_state, .ssm_state = ssm_state, .conv_w = conv_w, .A_log = per_head, .dt_bias = per_head, .q_scale = scale, .k_scale = scale };
        try testing.expect((try gdn_decode.step(.{ .hk = hk, .hv = hv, .dk = 128, .dv = 128 }, t_len, in, true, s)) == null);
    }
}

test "gdn_decode.step in gatedDeltaNetProjected: equals the chain at decode and verify widths; rollback to every row is serial decode" {
    try mlx.requireGpu();
    try gdnRecurLayerCase(false);
    try gdnRecurLayerCase(true);
}

test "gdn state f32: prefill, decode, a restored bf16 state and verify capture all store f32; rollback is serial decode" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var cache = try KVCache.init(testing.allocator, 0);
    defer cache.deinit();
    var xfm = std.mem.zeroInit(Transformer, .{ .s = s, .allocator = testing.allocator, .cache = cache });
    defer inline for (.{ &xfm.gdn_q_scale, &xfm.gdn_k_scale, &xfm.gdn_eps, &xfm.gdn_ones_w }) |p| {
        if (p.*) |arr| _ = mlx.mlx_array_free(arr);
    };
    const hk: c_int = 2;
    const hv: c_int = 8;
    const hidden: c_int = 128;
    const c_dim: c_int = 2 * hk * 128 + hv * 128;
    xfm.config.linear_num_key_heads = @intCast(hk);
    xfm.config.linear_num_value_heads = @intCast(hv);
    xfm.config.linear_key_head_dim = 128;
    xfm.config.linear_value_head_dim = 128;
    xfm.config.linear_conv_kernel_dim = 4;
    xfm.config.rms_norm_eps = 1e-6;
    var prng = std.Random.DefaultPrng.init(0xF32);
    const rnd = prng.random();
    const la = LinearAttnWeights{
        .qkv_w = try gdnRecurRand(rnd, &.{ hidden, c_dim }, 0.5, .bfloat16, s),
        .qkv_s = .{},
        .qkv_b = .{},
        .z_w = try gdnRecurRand(rnd, &.{ hidden, hv * 128 }, 0.5, .bfloat16, s),
        .z_s = .{},
        .z_b = .{},
        .a_w = try gdnRecurRand(rnd, &.{ hidden, hv }, 2.0, .bfloat16, s),
        .a_s = .{},
        .a_b = .{},
        .b_w = try gdnRecurRand(rnd, &.{ hidden, hv }, 2.0, .bfloat16, s),
        .b_s = .{},
        .b_b = .{},
        .conv1d_w = try gdnRecurRand(rnd, &.{ c_dim, 4, 1 }, 1.0, .bfloat16, s),
        .A_log = try gdnRecurRand(rnd, &.{hv}, 2.0, .bfloat16, s),
        .dt_bias = try gdnRecurRand(rnd, &.{hv}, 2.0, .bfloat16, s),
        .norm_w = try gdnRecurRand(rnd, &.{128}, 2.0, .bfloat16, s),
        .out_w = try gdnRecurRand(rnd, &.{ hv * 128, hidden }, 0.1, .bfloat16, s),
        .out_s = .{},
        .out_b = .{},
    };
    defer inline for (.{ la.qkv_w, la.z_w, la.a_w, la.b_w, la.conv1d_w, la.A_log, la.dt_bias, la.norm_w, la.out_w }) |arr| {
        _ = mlx.mlx_array_free(arr);
    };
    const f32_dt: mlx.mlx_dtype = .float32;
    gdn_decode_recur_override = true;
    defer gdn_decode_recur_override = null;

    var fresh = SSMCacheEntry{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = false };
    defer inline for (.{ fresh.conv_state, fresh.ssm_state }) |arr| {
        if (arr.ctx != null) _ = mlx.mlx_array_free(arr);
    };
    {
        const x = try gdnRecurRand(rnd, &.{ 1, 5, hidden }, 2.0, .bfloat16, s);
        defer _ = mlx.mlx_array_free(x);
        _ = mlx.mlx_array_free(try xfm.gatedDeltaNetProjected(x, &la, &fresh, 0, 1, 5, true, null, false));
        try testing.expectEqual(f32_dt, mlx.mlx_array_dtype(fresh.ssm_state));
    }

    const conv0 = try gdnRecurRand(rnd, &.{ 1, 3, c_dim }, 4.0, .bfloat16, s);
    defer _ = mlx.mlx_array_free(conv0);
    const restored = try gdnRecurRand(rnd, &.{ 1, hv, 128, 128 }, 1.0, .bfloat16, s);
    defer _ = mlx.mlx_array_free(restored);
    var live = SSMCacheEntry{ .conv_state = try gdnRecurRef(conv0), .ssm_state = try gdnRecurRef(restored), .initialized = true };
    defer {
        ssmFreeSpecCapture(&live);
        _ = mlx.mlx_array_free(live.conv_state);
        _ = mlx.mlx_array_free(live.ssm_state);
    }
    {
        const x = try gdnRecurRand(rnd, &.{ 1, 1, hidden }, 2.0, .bfloat16, s);
        defer _ = mlx.mlx_array_free(x);
        _ = mlx.mlx_array_free(try xfm.gatedDeltaNetProjected(x, &la, &live, 0, 1, 1, false, null, false));
        try testing.expectEqual(f32_dt, mlx.mlx_array_dtype(live.ssm_state));
    }

    var serial = SSMCacheEntry{ .conv_state = try gdnRecurRef(live.conv_state), .ssm_state = try gdnRecurRef(live.ssm_state), .initialized = true };
    defer {
        _ = mlx.mlx_array_free(serial.conv_state);
        _ = mlx.mlx_array_free(serial.ssm_state);
    }
    const seq: c_int = 3;
    const x = try gdnRecurRand(rnd, &.{ 1, seq, hidden }, 2.0, .bfloat16, s);
    defer _ = mlx.mlx_array_free(x);
    xfm.spec_capture_ssm = true;
    const out = try xfm.gatedDeltaNetProjected(x, &la, &live, 0, 1, seq, false, null, false);
    xfm.spec_capture_ssm = false;
    defer _ = mlx.mlx_array_free(out);
    try testing.expectEqual(f32_dt, mlx.mlx_array_dtype(live.ssm_state));
    try testing.expectEqual(f32_dt, mlx.mlx_array_dtype(if (live.spec_replay.held()) live.spec_replay.pre_state else live.spec_state_seq));

    const x0 = try gdnRecurRows(x, 0, 1, s);
    defer _ = mlx.mlx_array_free(x0);
    _ = mlx.mlx_array_free(try xfm.gatedDeltaNetProjected(x0, &la, &serial, 0, 1, 1, false, null, false));
    try ssmRollbackFromCapture(&live, 0, @intCast(seq), s);
    try qkvExpectBitsEqual(s, live.ssm_state, serial.ssm_state);
    try qkvExpectBitsEqual(s, live.conv_state, serial.conv_state);
}

test "gdn decay: at an f32 state and gate every route forgets at the exact rate; at --qwen-gdn 16 the routes agree" {
    try mlx.requireGpu();
    try gdnDecayCase(true);
    try gdnDecayCase(false);
}

test "gdn gate: compiled closure vs graph chain vs prework kernel (bit-identity probe)" {
    const s = mlx.gpuStream();
    const Cb = struct {
        fn cb(res: *mlx.mlx_vector_array, input: mlx.mlx_vector_array, payload: ?*anyopaque) callconv(.c) c_int {
            const st: *mlx.mlx_stream = @ptrCast(@alignCast(payload.?));
            var A_log = mlx.mlx_array_new();
            if (mlx.mlx_vector_array_get(&A_log, input, 0) != 0) return -1;
            defer _ = mlx.mlx_array_free(A_log);
            var a = mlx.mlx_array_new();
            if (mlx.mlx_vector_array_get(&a, input, 1) != 0) return -1;
            defer _ = mlx.mlx_array_free(a);
            var dt = mlx.mlx_array_new();
            if (mlx.mlx_vector_array_get(&dt, input, 2) != 0) return -1;
            defer _ = mlx.mlx_array_free(dt);
            const g = gdnGateChain(A_log, a, dt, .bfloat16, st.*) catch return -1;
            const arr = [_]mlx.mlx_array{g};
            res.* = mlx.mlx_vector_array_new_data(&arr, 1);
            _ = mlx.mlx_array_free(g);
            return 0;
        }
    };
    var stream_box = s;
    const raw = mlx.mlx_closure_new_func_payload(&Cb.cb, @ptrCast(&stream_box), null);
    var compiled = mlx.mlx_closure{ .ctx = null };
    try mlx.check(mlx.mlx_compile(&compiled, raw, true));
    _ = mlx.mlx_closure_free(raw);
    defer _ = mlx.mlx_closure_free(compiled);

    var prng = std.Random.DefaultPrng.init(0x6A7E);
    const rnd = prng.random();
    const hv: c_int = 48;
    const hv_shape = [_]c_int{hv};
    const A_log = try attn256RandBf16(rnd, &hv_shape, s);
    defer _ = mlx.mlx_array_free(A_log);
    const dt_bias = try attn256RandBf16(rnd, &hv_shape, s);
    defer _ = mlx.mlx_array_free(dt_bias);
    const a_shape = [_]c_int{ 1, 9, hv };
    const a = try attn256RandBf16Scaled(rnd, &a_shape, 16.0, s);
    defer _ = mlx.mlx_array_free(a);

    const in_arr = [_]mlx.mlx_array{ A_log, a, dt_bias };
    const in_vec = mlx.mlx_vector_array_new_data(&in_arr, 3);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var out_vec = mlx.mlx_vector_array{ .ctx = null };
    try mlx.check(mlx.mlx_closure_apply(&out_vec, compiled, in_vec));
    defer _ = mlx.mlx_vector_array_free(out_vec);
    var g_compiled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g_compiled);
    try mlx.check(mlx.mlx_vector_array_get(&g_compiled, out_vec, 0));
    const g_chain = try gdnGateChain(A_log, a, dt_bias, .bfloat16, s);
    defer _ = mlx.mlx_array_free(g_chain);
    const d = try attn256MaxDiff(g_compiled, g_chain, s);
    errdefer std.debug.print("[gate probe] compiled vs chain max |diff| = {e}\n", .{d});
    try std.testing.expectEqual(@as(f32, 0.0), d);
}

test "gdn gate at f32: compiled closure, graph chain and prework kernel agree far inside a bf16 step and carry the f32 width" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var cache = try KVCache.init(testing.allocator, 0);
    defer cache.deinit();
    var xfm = std.mem.zeroInit(Transformer, .{ .s = s, .allocator = testing.allocator, .cache = cache });
    xfm.compileGdnGate();
    defer if (xfm.compiled_gdn_gate) |closure| {
        _ = mlx.mlx_closure_free(closure);
    };
    try testing.expect(xfm.compiled_gdn_gate != null);
    var prng = std.Random.DefaultPrng.init(0x6A7F);
    const rnd = prng.random();
    const hk: c_int = 2;
    const hv: c_int = 48;
    const c_dim: c_int = 2 * hk * 128 + hv * 128;
    const q_scale = bf16Scalar(1.0 / 128.0, s);
    defer _ = mlx.mlx_array_free(q_scale);
    const k_scale = bf16Scalar(@sqrt(1.0 / 128.0), s);
    defer _ = mlx.mlx_array_free(k_scale);
    const A_log = try attn256RandBf16(rnd, &[_]c_int{hv}, s);
    defer _ = mlx.mlx_array_free(A_log);
    const dt_bias = try attn256RandBf16(rnd, &[_]c_int{hv}, s);
    defer _ = mlx.mlx_array_free(dt_bias);
    var worst: f64 = 0;
    for ([_]c_int{ 1, 9, 65, 513 }) |seq| {
        const qkv = try attn256RandBf16(rnd, &[_]c_int{ 1, seq, c_dim }, s);
        defer _ = mlx.mlx_array_free(qkv);
        const conv_state = try attn256RandBf16(rnd, &[_]c_int{ 1, 3, c_dim }, s);
        defer _ = mlx.mlx_array_free(conv_state);
        const conv_w = try attn256RandBf16(rnd, &[_]c_int{ c_dim, 4, 1 }, s);
        defer _ = mlx.mlx_array_free(conv_w);
        const b_in = try attn256RandBf16Scaled(rnd, &[_]c_int{ 1, seq, hv }, 16.0, s);
        defer _ = mlx.mlx_array_free(b_in);
        const a_in = try attn256RandBf16Scaled(rnd, &[_]c_int{ 1, seq, hv }, 16.0, s);
        defer _ = mlx.mlx_array_free(a_in);
        const pre = (try gdnPreworkFused(s, .{ .qkv = qkv, .qkv_off = 0, .qkv_stride = c_dim, .b = b_in, .b_off = 0, .b_stride = hv, .a = a_in, .a_off = 0, .a_stride = hv, .A_log = A_log, .dt_bias = dt_bias, .conv_state = conv_state, .conv_w = conv_w, .q_scale = q_scale, .k_scale = k_scale, .hk = hk, .hv = hv, .dk = 128, .dv = 128, .seq = seq, .gate_dtype = .float32 })) orelse return error.FusedDeclined;
        defer inline for (.{ pre.q, pre.k, pre.v, pre.conv_state, pre.g, pre.beta }) |arr| {
            _ = mlx.mlx_array_free(arr);
        };
        const g_closure = try xfm.computeGdnGate(A_log, a_in, dt_bias);
        defer _ = mlx.mlx_array_free(g_closure);
        const g_chain = try gdnGateChain(A_log, a_in, dt_bias, .float32, s);
        defer _ = mlx.mlx_array_free(g_chain);
        for ([_]mlx.mlx_array{ pre.g, g_closure, g_chain }) |g| try testing.expectEqual(mlx.mlx_dtype.float32, mlx.mlx_array_dtype(g));
        const want = try gdnDecayHost(g_chain, s);
        defer testing.allocator.free(want);
        for ([_]mlx.mlx_array{ pre.g, g_closure }) |g| {
            const got = try gdnDecayHost(g, s);
            defer testing.allocator.free(got);
            for (want, got) |w, x| worst = @max(worst, @abs(@as(f64, w) - @as(f64, x)) / @max(@abs(@as(f64, w)), 1e-30));
        }
    }
    // precise::exp and MLX's exp differ by more than a few f32 ulps on some Apple GPUs at a large
    // argument; a bf16 gate is ~2e-3 off near one, so 1e-4 still tells the widths apart.
    if (worst > 1e-4) std.debug.print("gdn gate f32 worst relative error {e}\n", .{worst});
    try testing.expect(worst <= 1e-4);
}

test "gdn verify fold: output and every rollback state remain bit-identical" {
    try mlx.requireGpu();
    mlx.installErrorHandler();
    const s = mlx.gpuStream();
    gdn_prework_override = true;
    defer gdn_prework_override = null;
    gdn_decode_fused_override = true;
    defer gdn_decode_fused_override = null;
    for ([_][2]c_int{ .{ 2, 8 }, .{ 16, 48 } }) |heads| {
        var t: c_int = 2;
        while (t <= gdn_decode.MAX_SEQ) : (t += 1) {
            try gdnRecurParityCase(s, heads[0], heads[1], t, true, false);
            try gdnRecurParityCase(s, heads[0], heads[1], t, true, true);
        }
    }
}


test "gdn verify fold: unsupported threadgroup declines without poisoning later work" {
    try mlx.requireGpu();
    mlx.installErrorHandler();
    const s = mlx.gpuStream();
    gdn_prework_override = true;
    defer gdn_prework_override = null;
    gdn_decode_fused_override = true;
    defer gdn_decode_fused_override = null;
    gdn_decode.fold_nt_override = 2048;
    defer gdn_decode.fold_nt_override = null;
    try gdnRecurParityCase(s, 2, 8, 2, true, false);
    try testing.expect(gdn_decode.foldDeclined(2));
    try testing.expect(!mlx.errorPending());
    gdn_decode.fold_nt_override = null;
    try gdnRecurParityCase(s, 2, 8, 2, true, false);
    try testing.expect(!mlx.errorPending());
}


test "gdn verify fold: default serves two rows and timing overrides stay within the kernel envelope" {
    const saved = gdn_verify_fold_override;
    defer gdn_verify_fold_override = saved;
    gdn_verify_fold_override = null;
    for (0..10) |rows| try testing.expectEqual(rows == 2, gdnVerifyFoldFor(@intCast(rows)));
    gdn_verify_fold_override = true;
    for (0..10) |rows| try testing.expectEqual(rows >= 2 and rows <= 8, gdnVerifyFoldFor(@intCast(rows)));
    gdn_verify_fold_override = false;
    try testing.expect(!gdnVerifyFoldFor(2));
}

test "conv1dWithCache tail drops the prefill chunk parent" {
    try mlx.requireGpu();
    const t = std.testing;
    const batch: c_int = 1;
    const seq: c_int = 4096;
    const cdim: c_int = 256;
    const kernel: c_int = 4;
    const delta = try conv1dTailRetentionDelta(batch, seq, cdim, kernel);
    const chunk: usize = @as(usize, @intCast(batch)) * @as(usize, @intCast(seq)) * @as(usize, @intCast(cdim)) * 2;
    if (delta >= 64 * 1024) std.debug.print("retention delta={d} chunk={d}\n", .{ delta, chunk });
    try t.expect(delta < 64 * 1024);
}

// Bar: same as above with a strided batch-2 parent.
test "conv1dWithCache tail drops a batch-2 strided prefill parent" {
    try mlx.requireGpu();
    const t = std.testing;
    const batch: c_int = 2;
    const seq: c_int = 4096;
    const cdim: c_int = 256;
    const kernel: c_int = 4;
    const delta = try conv1dTailRetentionDelta(batch, seq, cdim, kernel);
    const chunk: usize = @as(usize, @intCast(batch)) * @as(usize, @intCast(seq)) * @as(usize, @intCast(cdim)) * 2;
    if (delta >= 64 * 1024) std.debug.print("retention delta={d} chunk={d}\n", .{ delta, chunk });
    try t.expect(delta < 64 * 1024);
}

test "conv1dWithCache compact conv_state is bit-identical on and off" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const kernel: c_int = 4;
    const cdim: c_int = 32;
    const cases = [_][2]c_int{
        .{ 1, 1 },
        .{ 1, 5 },
        .{ 1, 4096 },
        .{ 2, 1 },
        .{ 2, 5 },
        .{ 2, 4096 },
    };
    for (cases) |c| {
        const batch = c[0];
        const seq = c[1];
        const off = try conv1dBitCopy(batch, seq, cdim, kernel, false, s);
        defer {
            _ = mlx.mlx_array_free(off.out);
            _ = mlx.mlx_array_free(off.conv_state);
        }
        const on = try conv1dBitCopy(batch, seq, cdim, kernel, true, s);
        defer {
            _ = mlx.mlx_array_free(on.out);
            _ = mlx.mlx_array_free(on.conv_state);
        }
        try conv1dExpectBitEqual(off.out, on.out, s);
        try conv1dExpectBitEqual(off.conv_state, on.conv_state, s);
    }
}

// Bar: below the chunk bar the tail stays a view (no copy kernel per decode step).
test "conv1dWithCache decode width does not copy the tail" {
    try mlx.requireGpu();
    const t = std.testing;
    const s = mlx.gpuStream();
    const saved = trf.compact_conv_state_override;
    defer trf.compact_conv_state_override = saved;
    trf.compact_conv_state_override = true;
    trf.compact_conv_state_copy_count = 0;
    const on = try conv1dBitCopy(1, 1, 32, 4, true, s);
    defer {
        _ = mlx.mlx_array_free(on.out);
        _ = mlx.mlx_array_free(on.conv_state);
    }
    try t.expectEqual(@as(u64, 0), trf.compact_conv_state_copy_count);
    const off = try conv1dBitCopy(1, 1, 32, 4, false, s);
    defer {
        _ = mlx.mlx_array_free(off.out);
        _ = mlx.mlx_array_free(off.conv_state);
    }
    try conv1dExpectBitEqual(on.out, off.out, s);
    try conv1dExpectBitEqual(on.conv_state, off.conv_state, s);
}

// Bar: the compacted state prepended to the next chunk gives byte-equal results.
test "conv1dWithCache compact conv_state is bit-identical across two chunks" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const off = try conv1dTwoChunk(1, 64, 64, 32, 4, false, s);
    defer {
        _ = mlx.mlx_array_free(off.out);
        _ = mlx.mlx_array_free(off.conv_state);
    }
    const on = try conv1dTwoChunk(1, 64, 64, 32, 4, true, s);
    defer {
        _ = mlx.mlx_array_free(on.out);
        _ = mlx.mlx_array_free(on.conv_state);
    }
    try conv1dExpectBitEqual(off.out, on.out, s);
    try conv1dExpectBitEqual(off.conv_state, on.conv_state, s);
}
