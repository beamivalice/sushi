const trf = @import("transformer.zig");
const std = @import("std");
const mlx = @import("mlx.zig");
const test_support = @import("test_support.zig");
const testReadF32 = test_support.testReadF32;
const log = @import("log.zig");
const testing = std.testing;
const verify_moe = @import("verify_moe.zig");
const qmatmulBits = verify_moe.qmatmulBits;
const Transformer = trf.Transformer;
const transposeBf16Weight = trf.transposeBf16Weight;

const routerRefRow = test_support.routerRefRow;

pub fn moeRoutingChain(router_logits: mlx.mlx_array, k: c_int, s: mlx.mlx_stream) !Transformer.MoeRouting {
    var neg_logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(neg_logits);
    try mlx.check(mlx.mlx_negative(&neg_logits, router_logits, s));

    var partitioned = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(partitioned);
    try mlx.check(mlx.mlx_argpartition_axis(&partitioned, neg_logits, k - 1, -1, s));

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

    var probs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(probs);
    try mlx.check(mlx.mlx_softmax_axis(&probs, router_logits, -1, true, s));

    var top_weights = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(top_weights);
    try mlx.check(mlx.mlx_take_along_axis(&top_weights, probs, inds, -1, s));

    var weight_sum_raw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(weight_sum_raw);
    try mlx.check(mlx.mlx_sum_axis(&weight_sum_raw, top_weights, -1, false, s));
    var weight_sum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(weight_sum);
    try mlx.check(mlx.mlx_expand_dims(&weight_sum, weight_sum_raw, -1, s));
    var norm_scores = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(norm_scores);
    try mlx.check(mlx.mlx_divide(&norm_scores, top_weights, weight_sum, s));

    return .{ .inds = inds, .norm_scores = norm_scores };
}


// ── Fused MoE router (top-K selection + weight normalization in ONE kernel) ──
//
// The op chains above are ~11 dispatches each — sigmoid/softmax, bias add,
// negate, argpartition (a full 256-wide sort), slice, take_along_axis, sum,
// eps add, divide, scale multiply, cast — over a [rows, E] tensor that is 512
// bytes wide at decode. Every one of them is launch-latency bound, and a MoE
// model pays the whole chain once per LAYER: 40 layers x 11 = 440 dispatches
// per token, none of which move meaningful data.
//
// One threadgroup per row does all of it: phase 1 computes the selection key
// for every expert into threadgroup memory, phase 2 runs K rounds of
// simdgroup-wide argmax with a LOWEST-INDEX tie-break, phase 3 normalizes the
// selected weights. Selection is exact integer/compare logic, so the picked
// expert set is the same set the chain picks whenever the keys are distinct.
//
// The ARITHMETIC deliberately replicates the chain's rather than improving on
// it — same `fast::exp`, same reciprocal-multiply softmax normalization, same
// MLX `Sigmoid` two-sided form, same intermediate rounding to the output dtype
// at the same points. A "more accurate" fused router is still a DIFFERENT
// model: the first live A/B of this kernel computed the softmax weights in one
// f32 pass and diverged from the chain at token ~80, which
// is the known INT4 near-tie argmax class amplified by an avoidable numerical
// change. The tests below therefore assert BIT equality with the chain, not
// just no-worse-than-reference; only the reduction ORDER over the expert axis
// differs (a simd tree vs MLX's N_READS blocking), which cannot move a bf16
// result except at an exact rounding boundary.
//
// Engagement is LOGGED, never inferred — a declined kernel is output-identical to the chain.
/// `sigmoid_bias_grouped` is `sigmoid_bias` with a GROUP LIMIT spliced
/// between the bias and the top-k: the experts are split into NG equal groups,
/// each group is scored by the SUM OF ITS TOP TWO, and every expert outside the
/// best TOPKG groups is pushed to the composed chain's own finite -1e4 penalty
/// before selection. The composed chain is ~28 dispatches per MoE layer x 41 layers,
/// and a dispatch prices at 1.76 us on this arch (SUSHI_DISPATCH_PROBE
/// slope over 1000 injected ops), so that chain alone is ~2 ms of a 19 ms token.
/// `softmax_gate` is `softmax` plus the shared-expert gate's logit, a dot of one x row with a flat
/// weight vector that the simdgroups sitting out the selection compute meanwhile.
const RouterMode = enum(u8) { softmax = 0, sigmoid_bias = 1, sigmoid_bias_grouped = 2, softmax_gate = 3 };

/// Largest expert count the kernel will stage in threadgroup memory. The
/// sigmoid arm keeps two f32 planes (key + raw weight), so 2048 experts is
/// 16 KiB of the 32 KiB budget.
const ROUTER_MAX_EXPERTS: c_int = 2048;

fn moeRouterSource(comptime mode: RouterMode) [:0]const u8 {
    const sigmoid = mode == .sigmoid_bias or mode == .sigmoid_bias_grouped;
    const grouped = mode == .sigmoid_bias_grouped;
    const gate = mode == .softmax_gate;
    return std.fmt.comptimePrint(
        \\uint row = thread_position_in_grid.y;
        \\uint tid = thread_position_in_threadgroup.x;
        \\uint lane = thread_index_in_simdgroup;
        \\
        \\threadgroup float rk[NE];
        \\{s}
        \\threadgroup uint sel[NK];
        \\{s}
        \\size_t rbase = (size_t)row * (size_t)NE;
        \\for (uint e = tid; e < NE; e += TG) {{
        \\{s}
        \\}}
        \\threadgroup_barrier(mem_flags::mem_threadgroup);
        \\{s}
        \\// Selection runs on simdgroup 0 only: K rounds of "max, then mask".
        \\// simd_min over the candidate indices makes the tie-break the LOWEST
        \\// expert id, so the pick is a pure function of the keys.
        \\if (tid < 32) {{
        \\{s}
        \\{s}
        \\  for (uint r = 0; r < NK; ++r) {{
        \\    float best = -INFINITY;
        \\    uint bidx = 0xFFFFFFFFu;
        \\    for (uint e = lane; e < NE; e += 32) {{
        \\      float v = rk[e];
        \\      if (v > best) {{ best = v; bidx = e; }}
        \\    }}
        \\    float gmax = simd_max(best);
        \\    uint cand = (best == gmax) ? bidx : 0xFFFFFFFFu;
        \\    uint gidx = metal::min(simd_min(cand), uint(NE - 1));
        \\    if (lane == 0) {{ sel[r] = gidx; rk[gidx] = -INFINITY; }}
        \\    simdgroup_barrier(mem_flags::mem_threadgroup);
        \\  }}
        \\
        \\  float w = 0.0f;
        \\  uint myidx = 0;
        \\  if (lane < NK) {{
        \\    myidx = sel[lane];
        \\{s}
        \\  }}
        \\  // MLX's `row_reduce_small` owns an 8-wide row in ONE thread and
        \\  // folds it in ASCENDING order in the reduce's accumulator type —
        \\  // which for a bf16 input is bf16, rounding after every add. A simd
        \\  // butterfly over the same values is a different number.
        \\  float tot = 0.0f;
        \\{s}
        \\{s}
        \\  if (lane < NK) {{
        \\    inds[(size_t)row * (size_t)NK + lane] = myidx;
        \\    scores[(size_t)row * (size_t)NK + lane] = TOUT(w);
        \\  }}
        \\}}
        \\{s}
    , .{
        if (grouped)
            "threadgroup float rw[NE];\nthreadgroup float gs[NG];"
        else if (sigmoid)
            "threadgroup float rw[NE];"
        else
            "",
        if (gate) "threadgroup float gp[16];" else "",
        if (sigmoid)
            // MLX's `Sigmoid` unary op, verbatim (unary_ops.h): the two-sided
            // form, not 1/(1+exp(-x)) — they differ in the last ulp. `exp` is
            // spelled PRECISE because copying upstream's TEXT is not copying
            // its ARITHMETIC: MLX's metallib is built `-fno-fast-math`
            // (kernels/CMakeLists.txt), so its bare `metal::exp` is the precise
            // one, while this source is JIT-compiled at MathMode::Safe where
            // the same spelling lowers to the fast variant. Measured 13.8% of
            // 1M random f32 sigmoids 1 ulp apart, and 6.1% of the routing
            // weights that come out of them; precise is 0/1M. The softmax arm
            // below spells `fast::exp` for the mirror-image reason (softmax.h
            // asks for fast explicitly). Costs ~1% of the kernel at an 8192-row
            // prefill width and nothing measurable at decode.
            \\  float l = float(logits[rbase + e]);
            \\  float sy = 1.0f / (1.0f + metal::precise::exp(metal::abs(l)));
            \\  float sg = (l < 0.0f) ? sy : (1.0f - sy);
            \\  rk[e] = sg + float(bias[e]);
            \\  rw[e] = sg;
        else
            \\  rk[e] = float(logits[rbase + e]);
        ,
        if (gate)
            // MLX's `dot_product` for a bf16/f16 [1,K]x[K,1] matmul, reproduced lane for lane: one
            // virtual simdgroup per 1024 elements, a float4 accumulator over four strided 8-element
            // chunks, then c0+c1+c2+c3 and a simd_sum. Any other order changes the rounded logit.
            \\const uint GSG = (GK + 1023) / 1024;
            \\if (tid >= 32 && tid < 32 * (1 + GSG)) {
            \\  const uint vs = tid / 32 - 1;
            \\  float4 c = 0.0f;
            \\  for (uint i = 0; i < 32; i += 8) {
            \\    const uint idx = vs * 1024 + lane * 8 + i * 32;
            \\    if (idx + 8 <= uint(GK)) {
            \\      for (uint j = 0; j < 8; j += 4) {
            \\        c += float4(*reinterpret_cast<const device metal::vec<TX, 4>*>(gx + (size_t)row * (size_t)GK + idx + j)) *
            \\             float4(*reinterpret_cast<const device metal::vec<TX, 4>*>(gw + idx + j));
            \\      }
            \\    }
            \\  }
            \\  float ga = c[0] + c[1] + c[2] + c[3];
            \\  ga = simd_sum(ga);
            \\  if (lane == 0) gp[vs] = ga;
            \\}
        else
            "",
        if (grouped)
            // The group limit (`group_limited_topk` shape). One lane
            // per group scores it by the SUM OF ITS TOP TWO biased experts, then
            // TOPKG rounds of the same max-then-mask/lowest-index selection the
            // expert top-k uses, then every expert in a losing group is pushed to
            // -1e4 — the composed chain's own FINITE penalty, not -inf: biased
            // scores are sigmoid+bias so they live well inside +/-2, and a
            // non-finite sentinel in a tensor that feeds selection is how the
            // composed causal-mask arm once shipped an all-NaN path that scored a
            // perfect parity max_err of 0.
            \\  for (uint g = lane; g < NG; g += 32) {
            \\    float m1 = -INFINITY, m2 = -INFINITY;
            \\    for (uint j = 0; j < GSZ; ++j) {
            \\      float v = rk[g * GSZ + j];
            \\      if (v > m1) { m2 = m1; m1 = v; } else if (v > m2) { m2 = v; }
            \\    }
            \\    gs[g] = m1 + m2;
            \\  }
            \\  simdgroup_barrier(mem_flags::mem_threadgroup);
            \\  uint keep_mask = 0u;
            \\  for (uint r = 0; r < TOPKG; ++r) {
            \\    float best = -INFINITY;
            \\    uint bidx = 0xFFFFFFFFu;
            \\    for (uint g = lane; g < NG; g += 32) {
            \\      float v = gs[g];
            \\      if (v > best) { best = v; bidx = g; }
            \\    }
            \\    float gmax = simd_max(best);
            \\    uint cand = (best == gmax) ? bidx : 0xFFFFFFFFu;
            \\    uint gidx = metal::min(simd_min(cand), uint(NG - 1));
            \\    keep_mask |= (1u << gidx);
            \\    if (lane == 0) gs[gidx] = -INFINITY;
            \\    simdgroup_barrier(mem_flags::mem_threadgroup);
            \\  }
            \\  for (uint e = lane; e < NE; e += 32) {
            \\    if ((keep_mask & (1u << (e / GSZ))) == 0u) rk[e] = -1.0e4f;
            \\  }
            \\  simdgroup_barrier(mem_flags::mem_threadgroup);
        else
            "",
        if (sigmoid)
            ""
        else
            // MLX's `softmax_single_row`, reduction tree included. It launches
            // ceil(NE/4) threads, each summing FOUR consecutive elements in
            // order, then one `simd_sum` per simdgroup, then a final `simd_sum`
            // over the per-simdgroup partials sitting in lanes 0..SIMDS-1 with
            // zeros above. Emulating those virtual threads 32 at a time
            // reproduces the identical butterfly on identical values, which is
            // what makes the rounded bf16 probability bit-equal; a plain
            // strided sum over the same data was 1 ulp off on real inputs.
            \\  constexpr int VT = (NE + 3) / 4;
            \\  constexpr int SIMDS = (VT + 31) / 32;
            \\  float lane_max = -INFINITY;
            \\  for (int g = 0; g < SIMDS; ++g) {
            \\    int vt = g * 32 + int(lane);
            \\    float m = -INFINITY;
            \\    for (int i = 0; i < 4; ++i) {
            \\      int e = vt * 4 + i;
            \\      if (e < NE) m = metal::max(m, rk[e]);
            \\    }
            \\    float mg = simd_max(m);
            \\    if (int(lane) == g) lane_max = mg;
            \\  }
            \\  float smax = simd_max(lane_max);
            \\  float lane_sum = 0.0f;
            \\  for (int g = 0; g < SIMDS; ++g) {
            \\    int vt = g * 32 + int(lane);
            \\    float acc = 0.0f;
            \\    for (int i = 0; i < 4; ++i) {
            \\      int e = vt * 4 + i;
            \\      acc += (e < NE) ? fast::exp(rk[e] - smax) : 0.0f;
            \\    }
            \\    float pg = simd_sum(acc);
            \\    if (int(lane) == g) lane_sum = pg;
            \\  }
            \\  float snorm = 1.0f / simd_sum(lane_sum);
        ,
        if (sigmoid)
            \\    w = rw[myidx];
        else
            // softmax(...) writes probabilities in the OUTPUT dtype, so the
            // per-expert rounding happens here, before the top-K sum.
            \\    w = float(TOUT(fast::exp(float(logits[rbase + myidx]) - smax) * snorm));
        ,
        if (sigmoid)
            // The chain sums FLOAT32 sigmoid scores, so the accumulator is
            // float — but still ascending, still sequential.
            \\  for (int j = 0; j < NK; ++j) tot += simd_shuffle(w, ushort(j));
        else
            \\  for (int j = 0; j < NK; ++j) tot = float(TOUT(simd_shuffle(w, ushort(j)) + tot));
        ,
        if (sigmoid)
            \\  if (NORM != 0 && NK > 1) {
            \\    w = w / (tot + 1e-20f);
            \\  }
            \\  // `scale` is a 0-dim array, which MLX binds as `const constant
            \\  // float&` — a reference, not a subscriptable pointer.
            \\  w = w * scale;
        else
            // `sum` over the K probabilities lands back in the output dtype
            // before the divide, exactly as mlx_sum_axis + mlx_divide do.
            \\  w = w / float(TOUT(tot));
        ,
        if (gate)
            // MLX folds the per-simdgroup sums with a second simd_sum over 16 lanes, zeros past the last.
            \\threadgroup_barrier(mem_flags::mem_threadgroup);
            \\if (tid < 16) {
            \\  float gt = (tid < GSG) ? gp[tid] : 0.0f;
            \\  gt = simd_sum(gt);
            \\  if (tid == 0) glogit[row] = TOUT(gt);
            \\}
        else
            "",
    });
}

const ROUTER_SOURCES = [4][:0]const u8{ moeRouterSource(.softmax), moeRouterSource(.sigmoid_bias), moeRouterSource(.sigmoid_bias_grouped), moeRouterSource(.softmax_gate) };
const ROUTER_NAMES = [4][*:0]const u8{ "sushi_moe_router_softmax", "sushi_moe_router_sigmoid", "sushi_moe_router_sigmoid_grouped", "sushi_moe_router_softmax_gate" };
var router_kernels: [4]?mlx.mlx_fast_metal_kernel = @splat(null);

fn getMoeRouterKernel(mode: RouterMode) !mlx.mlx_fast_metal_kernel {
    const mi: usize = @backingInt(mode);
    if (router_kernels[mi]) |k| return k;
    const softmax_inputs = [_][*:0]const u8{"logits"};
    const gate_inputs = [_][*:0]const u8{ "logits", "gx", "gw" };
    const sigmoid_inputs = [_][*:0]const u8{ "logits", "bias", "scale" };
    const input_names: []const [*:0]const u8 = switch (mode) {
        .softmax => &softmax_inputs,
        .softmax_gate => &gate_inputs,
        .sigmoid_bias, .sigmoid_bias_grouped => &sigmoid_inputs,
    };
    const plain_outputs = [_][*:0]const u8{ "inds", "scores" };
    const gate_outputs = [_][*:0]const u8{ "inds", "scores", "glogit" };
    const output_names: []const [*:0]const u8 = if (mode == .softmax_gate) &gate_outputs else &plain_outputs;
    const in_vec = mlx.mlx_vector_string_new_data(input_names.ptr, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(output_names.ptr, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        ROUTER_NAMES[mi],
        in_vec,
        out_vec,
        ROUTER_SOURCES[mi],
        "",
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    router_kernels[mi] = kernel;
    return kernel;
}

/// Threads per row. Selection needs a full simdgroup; phase 1 wants enough
/// threads to cover the expert row without a long strided loop.
fn moeRouterThreadGroup(num_experts: c_int) c_int {
    const rounded = @divTrunc(num_experts + 31, 32) * 32;
    return @max(@as(c_int, 32), @min(@as(c_int, 256), rounded));
}

/// Everything the cached kernel config bakes in. A mismatch re-arms it.
/// Full-shape component for cached metal-kernel config keys. A cached config
/// bakes the OUTPUT SHAPES (and grid) of the call that built it, so the key
/// must carry the caller's actual dims: keying on a PRODUCT (rows, n) collides
/// across layouts with equal element counts and hands the colliding caller an
/// output labeled with the FIRST caller's shape. Live 2026-07-29:
/// a MoE-layer SwiGLU at a 16-token hot-cache-restore prefill is (1,16,512) =
/// 8192 elements — the same count/rank/dtype as a dense mlp_only layer at
/// decode (1,1,8192) — and the mislabeled activation killed the server in the
/// dense down-proj matmul. Every cached config key whose config bakes an
/// input-derived shape uses this instead of derived products.
pub const ShapeKey = struct {
    dims: [8]c_int = @splat(0),
    len: u8 = 0,
    pub fn from(sh: []const c_int) ShapeKey {
        var key = ShapeKey{};
        for (sh, 0..) |d, i| key.dims[i] = d;
        key.len = @intCast(sh.len);
        return key;
    }
};

const RouterCfgKey = struct {
    shape: ShapeKey,
    k: c_int,
    out_dtype: mlx.mlx_dtype,
    norm: bool,
    scale: f32,
    n_group: c_int = 0,
    topk_group: c_int = 0,
    gate_k: c_int = 0,
    gate_dtype: mlx.mlx_dtype = .bfloat16,
};
const RouterCfgSlot = struct {
    cfg: mlx.mlx_fast_metal_kernel_config = .{ .ctx = null },
    key: RouterCfgKey = std.mem.zeroes(RouterCfgKey),
    scale_arr: mlx.mlx_array = .{ .ctx = null },
};
var router_cfg_cache: [4]RouterCfgSlot = .{ .{}, .{}, .{}, .{} };

pub var moe_router_fused_override: ?bool = null;
var moe_router_engaged: bool = false;

pub fn moeRouterFusedEnabled() bool {
    return moe_router_fused_override orelse true;
}

/// The shared-expert gate's logit folded into a softmax router: one `x` row and the flat weight vector `w`.
pub const RouterGate = struct { x: mlx.mlx_array, w: mlx.mlx_array };

/// One-kernel replacement for `moeRoutingChain` / `sigmoidBiasRoutingChain`.
/// Returns null when the geometry or dtype is outside the supported set, so
/// every caller keeps its chain fallback.
pub fn moeRouterTopK(
    s: mlx.mlx_stream,
    logits: mlx.mlx_array,
    bias: mlx.mlx_array,
    k: c_int,
    mode: RouterMode,
    route_norm: bool,
    route_scale: f32,
    out_dtype: mlx.mlx_dtype,
    n_group: c_int,
    topk_group: c_int,
) !?Transformer.MoeRouting {
    return moeRouterTopKGate(s, logits, bias, k, mode, route_norm, route_scale, out_dtype, n_group, topk_group, null);
}

pub fn moeRouterTopKGate(
    s: mlx.mlx_stream,
    logits: mlx.mlx_array,
    bias: mlx.mlx_array,
    k: c_int,
    mode_in: RouterMode,
    route_norm: bool,
    route_scale: f32,
    out_dtype: mlx.mlx_dtype,
    n_group: c_int,
    topk_group: c_int,
    gate: ?RouterGate,
) !?Transformer.MoeRouting {
    if (!moeRouterFusedEnabled()) return null;
    const mode: RouterMode = if (gate != null and mode_in == .softmax) .softmax_gate else mode_in;
    const sigmoid_mode = mode == .sigmoid_bias or mode == .sigmoid_bias_grouped;
    const lsh = mlx.getShape(logits);
    if (lsh.len < 1 or lsh.len > 4) return null;
    const num_experts = lsh[lsh.len - 1];
    if (num_experts < k or num_experts > ROUTER_MAX_EXPERTS) return null;
    if (k < 1 or k > 32) return null;
    const ldt = mlx.mlx_array_dtype(logits);
    if (ldt != .bfloat16 and ldt != .float16 and ldt != .float32) return null;
    if (sigmoid_mode and bias.ctx == null) return null;
    if (mode == .sigmoid_bias_grouped) {
        // One lane per group, one bit per group in the keep mask, and a group
        // score that is a top-2 sum. Anything outside that declines to the
        // composed chain rather than computing a different model.
        if (n_group <= 1 or n_group > 32) return null;
        if (topk_group <= 0 or topk_group > n_group) return null;
        if (@rem(num_experts, n_group) != 0) return null;
        if (@divExact(num_experts, n_group) < 2) return null;
    }

    var rows: c_int = 1;
    for (lsh[0 .. lsh.len - 1]) |d| rows *= d;
    if (rows < 1) return null;

    const tg = moeRouterThreadGroup(num_experts);
    // Bit-equality holds only for the one-row dot MLX runs as a single 512-thread block, and each
    // 1024 elements of it needs a simdgroup besides the selecting one.
    var gate_k: c_int = 0;
    var gate_dtype: mlx.mlx_dtype = .bfloat16;
    if (gate) |gt| {
        const wn = mlx.mlx_array_size(gt.w);
        gate_dtype = mlx.mlx_array_dtype(gt.x);
        if (rows != 1 or wn == 0 or wn > 16384 or wn % 8 != 0 or mlx.mlx_array_size(gt.x) != wn) return null;
        if ((gate_dtype != .bfloat16 and gate_dtype != .float16) or mlx.mlx_array_dtype(gt.w) != gate_dtype or out_dtype != gate_dtype) return null;
        if (@divTrunc(tg, 32) - 1 < @as(c_int, @intCast((wn + 1023) / 1024))) return null;
        gate_k = @intCast(wn);
    }

    // The whole point of the kernel is to spend ONE dispatch where the chain
    // spent a dozen; rebuilding the config every call would just move that cost
    // to the CPU side (measured +0.16 ms of graph-build per token on a 40-layer MoE).
    // A model's routing geometry is fixed, so one cached config per mode covers
    // decode, and prefill re-arms it once per width.
    const key = RouterCfgKey{
        .shape = ShapeKey.from(lsh),
        .k = k,
        .out_dtype = out_dtype,
        .norm = route_norm,
        .scale = route_scale,
        .n_group = n_group,
        .topk_group = topk_group,
        .gate_k = gate_k,
        .gate_dtype = gate_dtype,
    };
    const slot = &router_cfg_cache[@backingInt(mode)];
    if (slot.cfg.ctx == null or !std.meta.eql(slot.key, key)) {
        if (slot.cfg.ctx != null) _ = mlx.mlx_fast_metal_kernel_config_free(slot.cfg);
        if (slot.scale_arr.ctx != null) _ = mlx.mlx_array_free(slot.scale_arr);
        const config = mlx.mlx_fast_metal_kernel_config_new();
        // Outputs keep the caller's leading dims with the expert axis replaced by K.
        var out_shape: [4]c_int = undefined;
        for (lsh[0 .. lsh.len - 1], 0..) |d, i| out_shape[i] = d;
        out_shape[lsh.len - 1] = k;
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &out_shape, lsh.len, .uint32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &out_shape, lsh.len, out_dtype));
        if (mode == .softmax_gate) {
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &[_]c_int{1}, 1, out_dtype));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GK", gate_k));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "TX", gate_dtype));
        }
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, tg, rows, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, tg, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "TOUT", out_dtype));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "NE", num_experts));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "NK", k));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "TG", tg));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "NORM", @intFromBool(route_norm)));
        if (mode == .sigmoid_bias_grouped) {
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "NG", n_group));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GSZ", @divExact(num_experts, n_group)));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "TOPKG", topk_group));
        }
        slot.* = .{ .cfg = config, .key = key, .scale_arr = mlx.mlx_array_new_float(route_scale) };
    }
    const config = slot.cfg;

    const softmax_inputs = [_]mlx.mlx_array{logits};
    const gate_inputs = [_]mlx.mlx_array{ logits, if (gate) |gt| gt.x else logits, if (gate) |gt| gt.w else logits };
    const sigmoid_inputs = [_]mlx.mlx_array{ logits, bias, slot.scale_arr };
    const inputs_arr: []const mlx.mlx_array = switch (mode) {
        .softmax => &softmax_inputs,
        .softmax_gate => &gate_inputs,
        .sigmoid_bias, .sigmoid_bias_grouped => &sigmoid_inputs,
    };
    const inputs_vec = mlx.mlx_vector_array_new_data(inputs_arr.ptr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);

    const kernel = try getMoeRouterKernel(mode);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, config, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != (if (mode == .softmax_gate) @as(usize, 3) else 2)) return error.MetalKernelBadOutputCount;

    var inds = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(inds);
    try mlx.check(mlx.mlx_vector_array_get(&inds, outputs_vec, 0));
    var scores = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(scores);
    try mlx.check(mlx.mlx_vector_array_get(&scores, outputs_vec, 1));
    var gate_logit = mlx.mlx_array{ .ctx = null };
    if (mode == .softmax_gate) {
        gate_logit = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&gate_logit, outputs_vec, 2));
    }

    if (!moe_router_engaged) {
        moe_router_engaged = true;
        log.info("[moe] fused router kernel engaged: mode={s} E={d} K={d} tg={d}\n", .{ @tagName(mode), num_experts, k, tg });
    }
    return .{ .inds = inds, .norm_scores = scores, .gate_logit = gate_logit };
}

test "fused MoE router reproduces the softmax routing chain's top-K and beats its weights" {
    try mlx.requireGpu();
    // Selection is discrete: a different expert set is a different model, so
    // the ids are asserted EXACTLY against an fp64 host reference. The weights
    // follow the kernel rule — no worse than the fp64 reference, never
    // kernel-vs-kernel equality (the chain rounds a full bf16 softmax; the
    // kernel rounds once).
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xC0FFEE11);
    const rnd = prng.random();
    moe_router_fused_override = true;
    defer moe_router_fused_override = null;

    const ROWS: c_int = 5;
    const E: c_int = 256;
    const K: c_int = 8;
    const n: usize = @intCast(ROWS * E);

    const buf = try allocator.alloc(f32, n);
    defer allocator.free(buf);
    // Spread wide enough that a bf16 round-trip leaves the top-K boundary
    // unambiguous; ties there would make "the chain's set" undefined, not wrong.
    for (buf, 0..) |*v, i| v.* = (rnd.float(f32) - 0.5) * 8.0 + @as(f32, @floatFromInt(i % 7)) * 0.011;
    const lsh = [_]c_int{ ROWS, E };
    const l32 = mlx.mlx_array_new_data(buf.ptr, &lsh, 2, .float32);
    defer _ = mlx.mlx_array_free(l32);
    var lbf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(lbf);
    try mlx.check(mlx.mlx_astype(&lbf, l32, .bfloat16, s));

    // The kernel and the reference must see the SAME bf16 values.
    const host = try allocator.alloc(f32, n);
    defer allocator.free(host);
    try testReadF32(lbf, host, s);

    const fused = (try moeRouterTopK(s, lbf, .{ .ctx = null }, K, .softmax, true, 1.0, .bfloat16, 0, 0)) orelse
        return error.FusedRouterDeclined;
    defer _ = mlx.mlx_array_free(fused.inds);
    defer _ = mlx.mlx_array_free(fused.norm_scores);
    const chain = try moeRoutingChain(lbf, K, s);
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
    var ref_ids: [8]usize = undefined;
    var fused_err: f64 = 0;
    var chain_err: f64 = 0;
    for (0..@intCast(ROWS)) |r| {
        const row = host[r * @as(usize, @intCast(E)) ..][0..@intCast(E)];
        for (row, 0..) |v, i| keys[i] = @floatCast(v);
        routerRefRow(keys, @intCast(K), &ref_ids);

        // Exact ids, in the kernel's descending-key order.
        for (0..@intCast(K)) |j| {
            try testing.expectEqual(ref_ids[j], @as(usize, @intFromFloat(f_ids[r * @as(usize, @intCast(K)) + j])));
        }
        // The chain's order is argpartition's, so compare it as a SET.
        var chain_set: [8]usize = undefined;
        for (0..@intCast(K)) |j| chain_set[j] = @intFromFloat(c_ids[r * @as(usize, @intCast(K)) + j]);
        std.mem.sort(usize, &chain_set, {}, std.sort.asc(usize));
        var ref_sorted = ref_ids;
        std.mem.sort(usize, &ref_sorted, {}, std.sort.asc(usize));
        try testing.expectEqualSlices(usize, &ref_sorted, &chain_set);

        // fp64 reference weights: softmax over the selected logits.
        var denom: f64 = 0;
        var num: [8]f64 = undefined;
        for (0..@intCast(K)) |j| {
            num[j] = @exp(keys[ref_ids[j]] - keys[ref_ids[0]]);
            denom += num[j];
        }
        for (0..@intCast(K)) |j| {
            const want = num[j] / denom;
            const got = f_w[r * @as(usize, @intCast(K)) + j];
            fused_err = @max(fused_err, @abs(want - @as(f64, got)));
            // Chain rows are in argpartition order — find this id's slot.
            var slot: usize = 0;
            while (slot < @as(usize, @intCast(K))) : (slot += 1) {
                if (@as(usize, @intFromFloat(c_ids[r * @as(usize, @intCast(K)) + slot])) == ref_ids[j]) break;
            }
            const chain_got = c_w[r * @as(usize, @intCast(K)) + slot];
            chain_err = @max(chain_err, @abs(want - @as(f64, chain_got)));
            // BIT equality with the chain: the kernel replicates the chain's
            // arithmetic, and anything else is a different model (a live A/B on
            // a live A/B diverged at token ~80 when it did not).
            try testing.expectEqual(chain_got, got);
        }
    }
    try testing.expect(fused_err <= chain_err);
}

test "fused MoE router breaks ties on the lowest expert id at every width and mode" {
    try mlx.requireGpu();
    // Duplicated keys put ties inside the top-K, across lanes and within one
    // lane's strided slice; the pick must stay lowest-index-first.
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x71E5_0001);
    const rnd = prng.random();
    moe_router_fused_override = true;
    defer moe_router_fused_override = null;

    const widths = [_]c_int{ 100, 256, 512, 2048 };
    const modes = [_]RouterMode{ .softmax, .sigmoid_bias };
    const ROWS: c_int = 48;
    const K: c_int = 8;
    for (modes) |mode| for (widths) |E| {
        const ne: usize = @intCast(E);
        const n: usize = @as(usize, @intCast(ROWS)) * ne;
        const buf = try allocator.alloc(f32, n);
        defer allocator.free(buf);
        for (buf) |*v| v.* = @floatFromInt(rnd.uintLessThan(u32, 5));
        const lsh = [_]c_int{ ROWS, E };
        const l32 = mlx.mlx_array_new_data(buf.ptr, &lsh, 2, .float32);
        defer _ = mlx.mlx_array_free(l32);
        const zeros = try allocator.alloc(f32, ne);
        defer allocator.free(zeros);
        @memset(zeros, 0);
        const bsh = [_]c_int{E};
        const bias = mlx.mlx_array_new_data(zeros.ptr, &bsh, 1, .float32);
        defer _ = mlx.mlx_array_free(bias);

        const fused = (try moeRouterTopK(s, l32, bias, K, mode, true, 1.0, .float32, 0, 0)) orelse
            return error.FusedRouterDeclined;
        defer _ = mlx.mlx_array_free(fused.inds);
        defer _ = mlx.mlx_array_free(fused.norm_scores);
        const ids = try allocator.alloc(f32, @as(usize, @intCast(ROWS)) * 8);
        defer allocator.free(ids);
        try testReadF32(fused.inds, ids, s);

        const keys = try allocator.alloc(f64, ne);
        defer allocator.free(keys);
        var ref_ids: [8]usize = undefined;
        for (0..@intCast(ROWS)) |r| {
            for (buf[r * ne ..][0..ne], 0..) |v, i| keys[i] = v;
            routerRefRow(keys, 8, &ref_ids);
            for (0..8) |j| try testing.expectEqual(ref_ids[j], @as(usize, @intFromFloat(ids[r * 8 + j])));
        }
    };
}

test "fused softmax router folds the shared-expert gate dot bit-identically to the matmul it replaces" {
    try mlx.requireGpu();
    // The dot rides the router's idle simdgroups in MLX's own `dot_product` order; the cancelling
    // +-2^16 terms make any other summation order change the bf16 result.
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x6A7E_0002);
    const rnd = prng.random();
    moe_router_fused_override = true;
    defer moe_router_fused_override = null;

    const E: c_int = 512;
    const K: c_int = 10;
    const GK: usize = 2560;
    const bf = struct {
        fn f(buf: []const f32, shape: []const c_int, st: mlx.mlx_stream) !mlx.mlx_array {
            const a32 = mlx.mlx_array_new_data(buf.ptr, shape.ptr, @intCast(shape.len), .float32);
            defer _ = mlx.mlx_array_free(a32);
            var out = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_astype(&out, a32, .bfloat16, st));
            return out;
        }
    }.f;
    const lbuf = try allocator.alloc(f32, @intCast(E));
    defer allocator.free(lbuf);
    const xbuf = try allocator.alloc(f32, GK);
    defer allocator.free(xbuf);
    const wbuf = try allocator.alloc(f32, GK);
    defer allocator.free(wbuf);

    for (0..300) |_| {
        for (lbuf) |*v| v.* = (rnd.float(f32) - 0.5) * 8.0;
        for (xbuf) |*v| v.* = (rnd.float(f32) - 0.5) * 4.0;
        for (wbuf) |*v| v.* = (rnd.float(f32) - 0.5) * 0.4;
        for (0..3) |_| {
            const i = rnd.uintLessThan(usize, GK);
            const j = rnd.uintLessThan(usize, GK);
            if (i == j) continue;
            xbuf[i] = 65536.0;
            wbuf[i] = 1.0;
            xbuf[j] = -65536.0;
            wbuf[j] = 1.0;
        }
        const logits = try bf(lbuf, &.{ 1, E }, s);
        defer _ = mlx.mlx_array_free(logits);
        const x = try bf(xbuf, &.{ 1, 1, @intCast(GK) }, s);
        defer _ = mlx.mlx_array_free(x);
        const w_row = try bf(wbuf, &.{ 1, @intCast(GK) }, s);
        defer _ = mlx.mlx_array_free(w_row);
        const w_t = try transposeBf16Weight(w_row, s);
        defer _ = mlx.mlx_array_free(w_t);
        var w_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(w_flat);
        try mlx.check(mlx.mlx_reshape(&w_flat, w_t, &[_]c_int{@intCast(GK)}, 1, s));
        var w_vec = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(w_vec);
        try mlx.check(mlx.mlx_contiguous(&w_vec, w_flat, false, s));

        const want = try qmatmulBits(x, w_t, .{ .ctx = null }, .{ .ctx = null }, 0, 0, .affine, s);
        defer _ = mlx.mlx_array_free(want);
        const plain = (try moeRouterTopK(s, logits, .{ .ctx = null }, K, .softmax, true, 1.0, .bfloat16, 0, 0)) orelse
            return error.FusedRouterDeclined;
        defer _ = mlx.mlx_array_free(plain.inds);
        defer _ = mlx.mlx_array_free(plain.norm_scores);
        const gated = (try moeRouterTopKGate(s, logits, .{ .ctx = null }, K, .softmax, true, 1.0, .bfloat16, 0, 0, .{ .x = x, .w = w_vec })) orelse
            return error.FusedRouterDeclined;
        defer _ = mlx.mlx_array_free(gated.inds);
        defer _ = mlx.mlx_array_free(gated.norm_scores);
        defer _ = mlx.mlx_array_free(gated.gate_logit);
        try testing.expect(gated.gate_logit.ctx != null);

        var a: [1]f32 = undefined;
        var b: [1]f32 = undefined;
        try testReadF32(want, &a, s);
        try testReadF32(gated.gate_logit, &b, s);
        try testing.expectEqual(@as(u32, @bitCast(a[0])), @as(u32, @bitCast(b[0])));
        var pi: [10]f32 = undefined;
        var gi: [10]f32 = undefined;
        var ps: [10]f32 = undefined;
        var gs: [10]f32 = undefined;
        try testReadF32(plain.inds, &pi, s);
        try testReadF32(gated.inds, &gi, s);
        try testReadF32(plain.norm_scores, &ps, s);
        try testReadF32(gated.norm_scores, &gs, s);
        try testing.expectEqualSlices(f32, &pi, &gi);
        try testing.expectEqualSlices(f32, &ps, &gs);
    }
}

test "the router declines the gate fold when it has no idle simdgroups or the dot leaves MLX's single-block shape" {
    const s = mlx.gpuStream();
    moe_router_fused_override = true;
    defer moe_router_fused_override = null;
    const ones = struct {
        fn f(n: c_int, st: mlx.mlx_stream) !mlx.mlx_array {
            var o = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_ones(&o, &[_]c_int{n}, 1, .bfloat16, st));
            return o;
        }
    }.f;
    const small = try ones(32, s);
    defer _ = mlx.mlx_array_free(small);
    const wide = try ones(512, s);
    defer _ = mlx.mlx_array_free(wide);
    const x2560 = try ones(2560, s);
    defer _ = mlx.mlx_array_free(x2560);
    const x2564 = try ones(2564, s);
    defer _ = mlx.mlx_array_free(x2564);
    const x20000 = try ones(20000, s);
    defer _ = mlx.mlx_array_free(x20000);
    // 32 experts: one simdgroup, none left for the dot.
    try testing.expect((try moeRouterTopKGate(s, small, .{ .ctx = null }, 4, .softmax, true, 1.0, .bfloat16, 0, 0, .{ .x = x2560, .w = x2560 })) == null);
    // K not a multiple of 8, and K past one 512-thread block of MLX's kernel.
    try testing.expect((try moeRouterTopKGate(s, wide, .{ .ctx = null }, 4, .softmax, true, 1.0, .bfloat16, 0, 0, .{ .x = x2564, .w = x2564 })) == null);
    try testing.expect((try moeRouterTopKGate(s, wide, .{ .ctx = null }, 4, .softmax, true, 1.0, .bfloat16, 0, 0, .{ .x = x20000, .w = x20000 })) == null);
}

test "moeRouterThreadGroup keeps a full simdgroup and caps at 256" {
    try testing.expectEqual(@as(c_int, 32), moeRouterThreadGroup(8));
    try testing.expectEqual(@as(c_int, 32), moeRouterThreadGroup(32));
    try testing.expectEqual(@as(c_int, 64), moeRouterThreadGroup(33));
    try testing.expectEqual(@as(c_int, 128), moeRouterThreadGroup(128));
    try testing.expectEqual(@as(c_int, 256), moeRouterThreadGroup(256));
    try testing.expectEqual(@as(c_int, 256), moeRouterThreadGroup(2048));
}

test "moeRoutingChain produces top-K indices and renormalized softmax weights" {
    const s = mlx.gpuStream();

    // Two rows of router logits over 6 experts. Top-2 of each row is unambiguous:
    //   row 0: experts {0, 3} (logits 10 and 5)
    //   row 1: experts {1, 4} (logits 10 and 5)
    const n_rows: c_int = 2;
    const n_exp: c_int = 6;
    const k: c_int = 2;
    const data = [_]f32{
        10.0, 0.0,  0.0, 5.0, 0.0, 0.0,
        0.0,  10.0, 0.0, 0.0, 5.0, 0.0,
    };
    const shape = [_]c_int{ n_rows, n_exp };
    const logits = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);

    const routed = try moeRoutingChain(logits, k, s);
    defer _ = mlx.mlx_array_free(routed.inds);
    defer _ = mlx.mlx_array_free(routed.norm_scores);

    // Shape check (cheap, no data read needed).
    {
        const inds_shape = mlx.getShape(routed.inds);
        try testing.expectEqual(@as(usize, 2), inds_shape.len);
        try testing.expectEqual(n_rows, inds_shape[0]);
        try testing.expectEqual(k, inds_shape[1]);
        const sc_shape = mlx.getShape(routed.norm_scores);
        try testing.expectEqual(@as(usize, 2), sc_shape.len);
        try testing.expectEqual(n_rows, sc_shape[0]);
        try testing.expectEqual(k, sc_shape[1]);
    }

    // To verify top-K correctness without reading non-contiguous slice memory
    // directly, gather the original logits at the selected indices: gathered[i,j]
    // == logits[i, inds[i,j]]. Then sum across K — for our fixture, the top-2
    // logits in each row are {10, 5}, so the per-row sum must be 15.
    var gathered = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gathered);
    try mlx.check(mlx.mlx_take_along_axis(&gathered, logits, routed.inds, -1, s));
    var gathered_sum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gathered_sum);
    try mlx.check(mlx.mlx_sum_axis(&gathered_sum, gathered, -1, false, s));

    // norm_scores must sum to 1 along K (verifies the renormalize step).
    var scores_sum = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scores_sum);
    try mlx.check(mlx.mlx_sum_axis(&scores_sum, routed.norm_scores, -1, false, s));

    {
        const ev = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(ev);
        _ = mlx.mlx_vector_array_append_value(ev, gathered_sum);
        _ = mlx.mlx_vector_array_append_value(ev, scores_sum);
        try mlx.check(mlx.mlx_eval(ev));
    }

    // gathered_sum and scores_sum are 1D outputs of sum_axis (contiguous).
    const gs = mlx.mlx_array_data_float32(gathered_sum) orelse return error.InvalidDtype;
    const ss = mlx.mlx_array_data_float32(scores_sum) orelse return error.InvalidDtype;
    const tol: f32 = 1e-3;
    try testing.expect(@abs(gs[0] - 15.0) < tol);
    try testing.expect(@abs(gs[1] - 15.0) < tol);
    try testing.expect(@abs(ss[0] - 1.0) < tol);
    try testing.expect(@abs(ss[1] - 1.0) < tol);
}
