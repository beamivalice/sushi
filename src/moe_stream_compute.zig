const trf = @import("transformer.zig");
const std = @import("std");
const expert_stream_mod = @import("expert_stream.zig");
const imatrix_capture = @import("imatrix.zig");
const mlx = @import("mlx.zig");
const test_support = @import("test_support.zig");
const testReadF32 = test_support.testReadF32;
const model_mod = @import("model.zig");
const log = @import("log.zig");
const QuantMode = model_mod.QuantMode;
const testing = std.testing;
const MoeMlpWeights = trf.MoeMlpWeights;
const SeErr = trf.SeErr;
const fusedSwiGLU = trf.fusedSwiGLU;
const seBf16 = trf.seBf16;
const seFillBf16 = trf.seFillBf16;
const seHash = trf.seHash;
const transposeBf16Weight = trf.transposeBf16Weight;

const expert_bf16 = @import("expert_bf16_kernels.zig");

pub var expert_swap_logged: bool = false;
pub var expert_swap_total_swapped: u64 = 0;
pub var expert_swap_total_ids: u64 = 0;
pub var expert_swap_last_report: u64 = 0;

/// The running swap count is logged at the first swap, so a short run shows the pick swapping, then every 100k ids.
pub fn swapReportDue(first_swap: bool, total_ids: u64, last_report: u64) bool {
    return first_swap or total_ids -% last_report >= 100_000;
}

const STREAM_DECODE_MAX_ROWS: c_int = 16;

pub fn streamRows(shape: []const c_int) c_int {
    var rows: c_int = 1;
    for (shape[0 .. shape.len - 1]) |d| rows *= d;
    return rows;
}

fn streamHandoffApplies(rows: c_int) bool {
    return rows <= STREAM_DECODE_MAX_ROWS;
}

pub fn streamSharedEarlyApplies(eligible: bool, inds_shape: []const c_int) bool {
    return eligible and streamRows(inds_shape) <= STREAM_DECODE_MAX_ROWS;
}

/// What the lossy pick compares: softmax logits differ by the log of a probability ratio as they are; a
/// sigmoid router's do once mapped to log sigmoid, unbiased (the correction bias only selects).
const SwapLogits = enum { softmax, log_sigmoid };

pub fn expertSwapKind(tolerance: f32, biased_router: bool, sigmoid_router: bool, imatrix_armed: bool, inds_shape: []const c_int) ?SwapLogits {
    if (!(tolerance > 0) or imatrix_armed or streamRows(inds_shape) > STREAM_DECODE_MAX_ROWS) return null;
    if (sigmoid_router) return .log_sigmoid;
    return if (biased_router) null else .softmax;
}

/// The pick's f32 logits; log sigmoid as -logaddexp(0, -x), finite where sigmoid underflows.
pub fn routerSwapLogits(out: *mlx.mlx_array, router_logits: mlx.mlx_array, kind: SwapLogits, s: mlx.mlx_stream) !void {
    var wide = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wide);
    try mlx.check(mlx.mlx_astype(&wide, router_logits, .float32, s));
    if (kind == .softmax) return mlx.check(mlx.mlx_contiguous(out, wide, false, s));
    var neg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(neg);
    try mlx.check(mlx.mlx_negative(&neg, wide, s));
    const zero = mlx.mlx_array_new_float(0);
    defer _ = mlx.mlx_array_free(zero);
    var softplus = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(softplus);
    try mlx.check(mlx.mlx_logaddexp(&softplus, neg, zero, s));
    var log_sigmoid = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(log_sigmoid);
    try mlx.check(mlx.mlx_negative(&log_sigmoid, softplus, s));
    try mlx.check(mlx.mlx_contiguous(out, log_sigmoid, false, s));
}

/// One-row twin of `expert_stream.substituteMisses` on the GPU, so a lossy pick
/// needs no host round trip before the layer's experts are queued: misses in
/// descending logit order, each takes the highest-logit ready expert not yet
/// selected within `gap` (lowest index on ties) unless it is starved. Writes the
/// picked ids and their slots. The host still runs its own pick and keeps the
/// GPU result only when both agree.
const EXPERT_PICK_SOURCE =
    \\uint t = thread_position_in_threadgroup.x;
    \\uint T = threads_per_threadgroup.x;
    \\uint lane = thread_index_in_simdgroup;
    \\uint sg = simdgroup_index_in_threadgroup;
    \\threadgroup bool taken[E];
    \\threadgroup int sel[K];
    \\threadgroup int order[K];
    \\threadgroup int nmiss;
    \\threadgroup float red_v[32];
    \\threadgroup int red_i[32];
    \\for (uint e = t; e < uint(E); e += T) taken[e] = false;
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if (t == 0) {
    \\  int n = 0;
    \\  for (int p = 0; p < K; ++p) {
    \\    int e = ids[p];
    \\    sel[p] = e;
    \\    taken[e] = true;
    \\    if (map[e] >= 0) continue;
    \\    int at = n;
    \\    while (at > 0 && logits[ids[order[at - 1]]] < logits[e]) { order[at] = order[at - 1]; at -= 1; }
    \\    order[at] = p;
    \\    n += 1;
    \\  }
    \\  nmiss = n;
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\float g = gap[0];
    \\for (int i = 0; i < nmiss; ++i) {
    \\  int pos = order[i];
    \\  int missed = sel[pos];
    \\  if (int(starved[missed]) >= LIMIT) continue;
    \\  float lm = logits[missed];
    \\  float best_v = -INFINITY;
    \\  int best_i = -1;
    \\  for (uint e = t; e < uint(E); e += T) {
    \\    if (taken[e] || map[e] < 0) continue;
    \\    float v = logits[e];
    \\    if (!(v - lm >= g) || !(v > best_v)) continue;
    \\    best_v = v;
    \\    best_i = int(e);
    \\  }
    \\  for (uint off = 16; off > 0; off >>= 1) {
    \\    float ov = simd_shuffle_down(best_v, off);
    \\    int oi = simd_shuffle_down(best_i, off);
    \\    if (lane + off < 32 && oi >= 0 && (best_i < 0 || ov > best_v || (ov == best_v && oi < best_i))) { best_v = ov; best_i = oi; }
    \\  }
    \\  if (lane == 0) { red_v[sg] = best_v; red_i[sg] = best_i; }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  if (t == 0) {
    \\    float bv = -INFINITY;
    \\    int bi = -1;
    \\    for (uint s2 = 0; s2 < (T + 31) / 32; ++s2) {
    \\      int oi = red_i[s2];
    \\      float ov = red_v[s2];
    \\      if (oi >= 0 && (bi < 0 || ov > bv || (ov == bv && oi < bi))) { bv = ov; bi = oi; }
    \\    }
    \\    if (bi >= 0) { sel[pos] = bi; taken[bi] = true; }
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\}
    \\if (t < uint(K)) {
    \\  int e = sel[t];
    \\  picked[t] = e;
    \\  slots[t] = max(map[e], 0);
    \\}
;

var expert_pick_kernel: ?mlx.mlx_fast_metal_kernel = null;

const ExpertPick = struct { slots: mlx.mlx_array, picked: mlx.mlx_array };

pub fn expertPickGpu(ids: mlx.mlx_array, logits: mlx.mlx_array, map: mlx.mlx_array, starved: mlx.mlx_array, tolerance: f32, s: mlx.mlx_stream) !ExpertPick {
    const k: c_int = @intCast(mlx.mlx_array_size(ids));
    const experts: c_int = @intCast(mlx.mlx_array_size(map));
    if (k < 1 or k > 32 or experts < 1 or experts > 1024 or mlx.mlx_array_size(logits) != @as(usize, @intCast(experts))) return error.UnsupportedShape;
    const kernel = expert_pick_kernel orelse blk: {
        const input_names = [_][*:0]const u8{ "ids", "logits", "map", "starved", "gap" };
        const output_names = [_][*:0]const u8{ "slots", "picked" };
        const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
        defer _ = mlx.mlx_vector_string_free(in_vec);
        const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
        defer _ = mlx.mlx_vector_string_free(out_vec);
        const made = mlx.mlx_fast_metal_kernel_new("sushi_expert_pick", in_vec, out_vec, EXPERT_PICK_SOURCE, "", true, false);
        if (made.ctx == null) return error.MetalKernelCompileFailed;
        expert_pick_kernel = made;
        break :blk made;
    };
    const gap_value: f32 = @log(1.0 - @min(tolerance, expert_stream_mod.MAX_PICK_TOLERANCE));
    const gap = mlx.mlx_array_new_data(&gap_value, &[_]c_int{1}, 1, .float32);
    defer _ = mlx.mlx_array_free(gap);
    const config = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    const out_shape = [_]c_int{k};
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &out_shape, 1, .int32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &out_shape, 1, .int32));
    const threads: c_int = @min(1024, @divTrunc(experts + 31, 32) * 32);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, threads, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, threads, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "K", k));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "E", experts));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "LIMIT", expert_stream_mod.PICK_STARVE_LIMIT));
    const inputs = [_]mlx.mlx_array{ ids, logits, map, starved, gap };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs, kernel, inputs_vec, config, s));
    var slots = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(slots);
    try mlx.check(mlx.mlx_vector_array_get(&slots, outputs, 0));
    var picked = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&picked, outputs, 1));
    return .{ .slots = slots, .picked = picked };
}

pub fn slabMoeWeights(mw: *const MoeMlpWeights, raw: *const [expert_stream_mod.quant.component_count]mlx.mlx_array) MoeMlpWeights {
    const C = expert_stream_mod.quant.Component;
    var out = mw.*;
    out.switch_gate_w = raw[@backingInt(C.gate_w)];
    out.switch_gate_s = raw[@backingInt(C.gate_s)];
    out.switch_gate_b = raw[@backingInt(C.gate_b)];
    out.switch_up_w = raw[@backingInt(C.up_w)];
    out.switch_up_s = raw[@backingInt(C.up_s)];
    out.switch_up_b = raw[@backingInt(C.up_b)];
    out.switch_down_w = raw[@backingInt(C.down_w)];
    out.switch_down_s = raw[@backingInt(C.down_s)];
    out.switch_down_b = raw[@backingInt(C.down_b)];
    return out;
}

/// A decode-width streamed layer queues its expert compute from the GPU copy
/// of the cache map before the host reads the router ids; the host keeps it
/// when every id resolved to the gathered slot, else rebuilds it.
pub fn expertSpecApplies(imatrix_armed: bool, inds_shape: []const c_int) bool {
    return !imatrix_armed and streamRows(inds_shape) <= STREAM_DECODE_MAX_ROWS;
}

var expert_spec_kept: u64 = 0;
var expert_spec_total: u64 = 0;
pub fn noteExpertSpec(kept: bool) void {
    if (kept) expert_spec_kept += 1;
    expert_spec_total += 1;
    if (expert_spec_total == 1 or expert_spec_total % 20_000 == 0)
        log.info("[expert-spec] kept {d} of {d} speculated layers\n", .{ expert_spec_kept, expert_spec_total });
}

pub fn handOffStreamedResult(result: mlx.mlx_array, rows: c_int) !void {
    if (!streamHandoffApplies(rows)) return;
    const vec = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{result}, 1);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_async_eval(vec));
}

/// Gathered matmul for MoE expert dispatch — handles both quantized and dense bf16.
/// Quantized (sc.ctx != null): mlx_gather_qmm with transpose_w=true, w stored [E,out,in].
/// Dense bf16 (sc.ctx == null): mlx_gather_mm; w was pre-transposed to [E,in,out] at load
/// (maybeTransposeForBf16 + generalized transposeBf16Weight), so x @ w is correct with no
/// transpose flag — mirrors mlx-lm's `gather_mm(x, weight.swapaxes(-1,-2))`.
const EXPERT_SLAB_ALIGN_BYTES: usize = 16;

const StreamedExpertOperands = struct {
    x_rows: mlx.mlx_array,
    gate: mlx.mlx_array,
    up: mlx.mlx_array,
    down: mlx.mlx_array,
    slab_gate_up: mlx.mlx_array,
    slab_down: mlx.mlx_array,
    slots: mlx.mlx_array,
    weights: mlx.mlx_array,
    imatrix: ?ImatrixTap = null,
};

/// imatrix capture through a streamed MoE layer. Every compute arm indexes the
/// slab by REMAPPED slot, so the tap carries the routing's own [rows, top_k]
/// GLOBAL expert ids: the statistics are keyed by those.
pub const ImatrixTap = struct {
    collector: *imatrix_capture.Collector,
    layer: u16,
    global_ids: mlx.mlx_array,

    /// The tap for one streamed layer, owning its [rows, top_k] view of the
    /// routing's ids; null when no capture is armed. The arms only read it, so
    /// the layer's own `deinit` frees it once.
    pub fn forLayer(col: ?*imatrix_capture.Collector, layer: u16, ids: mlx.mlx_array, rows: c_int, topk: c_int, s: mlx.mlx_stream) !?ImatrixTap {
        const collector = col orelse return null;
        var ids_2d = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(ids_2d);
        try mlx.check(mlx.mlx_reshape(&ids_2d, ids, &[_]c_int{ rows, topk }, 2, s));
        return .{ .collector = collector, .layer = layer, .global_ids = ids_2d };
    }

    pub fn deinit(self: ImatrixTap) void {
        _ = mlx.mlx_array_free(self.global_ids);
    }

    /// The layer's MLP input rows, [.., hidden] over `rows` tokens.
    pub fn observeInput(self: ImatrixTap, s: mlx.mlx_stream, x: mlx.mlx_array, rows: c_int, hidden: c_int) !void {
        var x_rows = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_rows);
        try mlx.check(mlx.mlx_reshape(&x_rows, x, &[_]c_int{ rows, hidden }, 2, s));
        try self.collector.observeGateUp(self.layer, x_rows, self.global_ids);
    }

    /// One MoE layer's SwiGLU activation rows. `act` is [.., inter] in whatever
    /// row order the arm produced; `ids_flat` names each of those `n` rows'
    /// global expert in the SAME order.
    pub fn observeDown(self: ImatrixTap, s: mlx.mlx_stream, act: mlx.mlx_array, ids_flat: mlx.mlx_array, n: c_int) !void {
        const ash = mlx.getShape(act);
        var act_2d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(act_2d);
        try mlx.check(mlx.mlx_reshape(&act_2d, act, &[_]c_int{ n, ash[ash.len - 1] }, 2, s));
        var ids_2d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ids_2d);
        try mlx.check(mlx.mlx_reshape(&ids_2d, ids_flat, &[_]c_int{ n, 1 }, 2, s));
        try self.collector.observeDown(self.layer, act_2d, ids_2d);
    }

    /// `act` in the sorted arm's row order; `inv_order` is that arm's own
    /// permutation back to the routing's order, where the global ids live.
    pub fn observeDownSorted(self: ImatrixTap, s: mlx.mlx_stream, act: mlx.mlx_array, inv_order: mlx.mlx_array, n: c_int) !void {
        const ash = mlx.getShape(act);
        var act_2d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(act_2d);
        try mlx.check(mlx.mlx_reshape(&act_2d, act, &[_]c_int{ n, ash[ash.len - 1] }, 2, s));
        var routed = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(routed);
        try mlx.check(mlx.mlx_take_axis(&routed, act_2d, inv_order, 0, s));
        try self.observeDown(s, routed, self.global_ids, n);
    }
};

var streamed_expert_kernels_engaged: bool = false;

fn expertSlabAligned(slab: mlx.mlx_array) !void {
    try mlx.check(mlx.mlx_array_eval(slab));
    const p = mlx.mlx_array_data_uint8(slab) orelse return error.ExpertSlabUnaligned;
    if (@intFromPtr(p) % EXPERT_SLAB_ALIGN_BYTES != 0) return error.ExpertSlabUnaligned;
}

fn streamedExpertSwiglu(s: mlx.mlx_stream, gate: mlx.mlx_array, up: mlx.mlx_array) !mlx.mlx_array {
    if (try fusedSwiGLU(s, gate, up)) |fused| return fused;
    var sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sig);
    try mlx.check(mlx.mlx_sigmoid(&sig, gate, s));
    var activated = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(activated);
    try mlx.check(mlx.mlx_multiply(&activated, gate, sig, s));
    var result = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&result, activated, up, s));
    return result;
}

fn streamedExpertUnsortedDown(s: mlx.mlx_stream, op: StreamedExpertOperands, activation: mlx.mlx_array, rows: c_int, topk: c_int) !mlx.mlx_array {
    const no_idx = mlx.mlx_array{ .ctx = null };
    const no_side = mlx.mlx_array{ .ctx = null };
    var inds3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(inds3);
    try mlx.check(mlx.mlx_reshape(&inds3, op.slots, &[_]c_int{ 1, rows, topk }, 3, s));
    var act_exp = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(act_exp);
    try mlx.check(mlx.mlx_expand_dims(&act_exp, activation, -2, s));
    var down_5d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(down_5d);
    try gatherExpertMm(&down_5d, act_exp, op.down, no_side, no_side, no_idx, inds3, 0, 64, .affine, false, s);
    var squeezed = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(squeezed);
    try mlx.check(mlx.mlx_squeeze(&squeezed, down_5d, s));
    const sh = mlx.getShape(squeezed);
    const hidden = sh[sh.len - 1];
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&out, squeezed, &[_]c_int{ rows, topk, hidden }, 3, s));
    return out;
}

fn streamedExpertCompositeDown(s: mlx.mlx_stream, op: StreamedExpertOperands, rows: c_int, topk: c_int) !mlx.mlx_array {
    const D = mlx.getShape(op.x_rows)[1];
    const total_inds: c_int = rows * topk;
    const no_idx = mlx.mlx_array{ .ctx = null };
    const no_side = mlx.mlx_array{ .ctx = null };
    if (rows > 1 or total_inds >= 64) {
        var flat_inds = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat_inds);
        try mlx.check(mlx.mlx_reshape(&flat_inds, op.slots, &[_]c_int{total_inds}, 1, s));
        var order = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(order);
        try mlx.check(mlx.mlx_argsort_axis(&order, flat_inds, 0, s));
        var inv_order = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(inv_order);
        try mlx.check(mlx.mlx_argsort_axis(&inv_order, order, 0, s));
        var sorted_inds = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sorted_inds);
        try mlx.check(mlx.mlx_take_axis(&sorted_inds, flat_inds, order, 0, s));
        const k_arr = mlx.mlx_array_new_int(topk);
        defer _ = mlx.mlx_array_free(k_arr);
        var lhs_idx = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(lhs_idx);
        try mlx.check(mlx.mlx_floor_divide(&lhs_idx, order, k_arr, s));
        var x_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_flat);
        try mlx.check(mlx.mlx_reshape(&x_flat, op.x_rows, &[_]c_int{ rows, D }, 2, s));
        var x_gathered = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_gathered);
        try mlx.check(mlx.mlx_take_axis(&x_gathered, x_flat, lhs_idx, 0, s));
        var x_rep = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_rep);
        try mlx.check(mlx.mlx_reshape(&x_rep, x_gathered, &[_]c_int{ total_inds, 1, D }, 3, s));
        var gate_3d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gate_3d);
        try gatherExpertMm(&gate_3d, x_rep, op.gate, no_side, no_side, no_idx, sorted_inds, 0, 64, .affine, true, s);
        var gate = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gate);
        try mlx.check(mlx.mlx_squeeze(&gate, gate_3d, s));
        var up_3d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(up_3d);
        try gatherExpertMm(&up_3d, x_rep, op.up, no_side, no_side, no_idx, sorted_inds, 0, 64, .affine, true, s);
        var up = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(up);
        try mlx.check(mlx.mlx_squeeze(&up, up_3d, s));
        const activation = try streamedExpertSwiglu(s, gate, up);
        defer _ = mlx.mlx_array_free(activation);
        if (op.imatrix) |tap| {
            var flat_global = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(flat_global);
            try mlx.check(mlx.mlx_reshape(&flat_global, tap.global_ids, &[_]c_int{total_inds}, 1, s));
            var sorted_global = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sorted_global);
            try mlx.check(mlx.mlx_take_axis(&sorted_global, flat_global, order, 0, s));
            try tap.observeDown(s, activation, sorted_global, total_inds);
        }
        var act_exp = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(act_exp);
        try mlx.check(mlx.mlx_expand_dims(&act_exp, activation, -2, s));
        var down_3d = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(down_3d);
        try gatherExpertMm(&down_3d, act_exp, op.down, no_side, no_side, no_idx, sorted_inds, 0, 64, .affine, true, s);
        var down_squeezed = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(down_squeezed);
        try mlx.check(mlx.mlx_squeeze(&down_squeezed, down_3d, s));
        var down_unsorted = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(down_unsorted);
        try mlx.check(mlx.mlx_take_axis(&down_unsorted, down_squeezed, inv_order, 0, s));
        const hidden = mlx.getShape(down_unsorted)[1];
        var down_out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&down_out, down_unsorted, &[_]c_int{ rows, topk, hidden }, 3, s));
        return down_out;
    }
    var x_exp = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x_exp);
    try mlx.check(mlx.mlx_reshape(&x_exp, op.x_rows, &[_]c_int{ 1, rows, 1, 1, D }, 5, s));
    var inds3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(inds3);
    try mlx.check(mlx.mlx_reshape(&inds3, op.slots, &[_]c_int{ 1, rows, topk }, 3, s));
    var gate_5d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gate_5d);
    try gatherExpertMm(&gate_5d, x_exp, op.gate, no_side, no_side, no_idx, inds3, 0, 64, .affine, false, s);
    var gate = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gate);
    try mlx.check(mlx.mlx_squeeze(&gate, gate_5d, s));
    var up_5d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(up_5d);
    try gatherExpertMm(&up_5d, x_exp, op.up, no_side, no_side, no_idx, inds3, 0, 64, .affine, false, s);
    var up = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(up);
    try mlx.check(mlx.mlx_squeeze(&up, up_5d, s));
    const activation = try streamedExpertSwiglu(s, gate, up);
    defer _ = mlx.mlx_array_free(activation);
    if (op.imatrix) |tap| try tap.observeDown(s, activation, tap.global_ids, rows * topk);
    return streamedExpertUnsortedDown(s, op, activation, rows, topk);
}

fn streamedExpertWeightedReduce(s: mlx.mlx_stream, down_out: mlx.mlx_array, weights: mlx.mlx_array) !mlx.mlx_array {
    var w_exp = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_exp);
    try mlx.check(mlx.mlx_expand_dims(&w_exp, weights, -1, s));
    var weighted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(weighted);
    try mlx.check(mlx.mlx_multiply(&weighted, down_out, w_exp, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_sum_axis(&out, weighted, -2, false, s));
    return out;
}

pub fn streamedExpertCompute(s: mlx.mlx_stream, op: StreamedExpertOperands, rows: c_int, use_kernels: bool) !mlx.mlx_array {
    const x_shape = mlx.getShape(op.x_rows);
    if (x_shape.len != 2 or x_shape[0] != rows) return error.StreamedExpertShapeMismatch;
    const slot_shape = mlx.getShape(op.slots);
    if (slot_shape.len != 2 or slot_shape[0] != rows) return error.StreamedExpertShapeMismatch;
    const topk = slot_shape[1];
    if (op.imatrix) |tap| try tap.observeInput(s, op.x_rows, rows, x_shape[1]);
    if (use_kernels and rows >= 1 and rows <= expert_bf16.MAX_ROWS and topk >= 1 and topk <= expert_bf16.MAX_TOPK) {
        try expertSlabAligned(op.slab_gate_up);
        try expertSlabAligned(op.slab_down);
        const h = try expert_bf16.gateUpSwiglu(s, op.x_rows, op.slab_gate_up, op.slots);
        defer _ = mlx.mlx_array_free(h);
        if (op.imatrix) |tap| try tap.observeDown(s, h, tap.global_ids, rows * topk);
        const down_kernel = expert_bf16.downKernelPreferred(@intCast(rows));
        if (!streamed_expert_kernels_engaged) {
            streamed_expert_kernels_engaged = true;
            log.info("[expert-bf16] streamed MoE kernel arm engaged: rows={d} topk={d} down={s}\n", .{ rows, topk, if (down_kernel) "kernel" else "gather_mm" });
        }
        if (down_kernel) {
            var w32 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(w32);
            try mlx.check(mlx.mlx_astype(&w32, op.weights, .float32, s));
            return expert_bf16.downReduce(s, h, op.slab_down, op.slots, w32);
        }
        var act = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(act);
        if (rows == 1) {
            try mlx.check(mlx.mlx_reshape(&act, h, &[_]c_int{ topk, mlx.getShape(h)[2] }, 2, s));
        } else {
            try mlx.check(mlx.mlx_array_set(&act, h));
        }
        const down_out = try streamedExpertUnsortedDown(s, op, act, rows, topk);
        defer _ = mlx.mlx_array_free(down_out);
        return streamedExpertWeightedReduce(s, down_out, op.weights);
    }
    const down_out = try streamedExpertCompositeDown(s, op, rows, topk);
    defer _ = mlx.mlx_array_free(down_out);
    return streamedExpertWeightedReduce(s, down_out, op.weights);
}

pub fn gatherExpertMm(res: *mlx.mlx_array, x: mlx.mlx_array, w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array, lhs_idx: mlx.mlx_array, rhs_idx: mlx.mlx_array, bits: u32, group_size: u32, mode: QuantMode, sorted: bool, s: mlx.mlx_stream) !void {
    if (sc.ctx == null) {
        // mlx 0.31.2's `mlx_gather_mm` returns WRONG results with sorted_indices=true
        // for the dense (non-quantized) path — the quantized `mlx_gather_qmm` honors
        // the flag correctly, but the dense kernel does not. The sorted-indices flag is
        // only a performance hint (rhs_indices ARE sorted here), so forcing false is
        // always numerically correct; it just forgoes the sorted-stream optimization.
        // Without this, dense bf16 MoE prefill (the sorted path, S>1) produced fluent
        // but semantically-wrong output (e.g. a bf16 MoE calling clean prompts
        // "jumbled"). Verified byte-identical to mlx-lm once forced false. `sorted` is
        // still honored by the quantized branch below, where gather_qmm handles it
        // correctly.
        try mlx.check(mlx.mlx_gather_mm(res, x, w, lhs_idx, rhs_idx, false, s));
    } else {
        try mlx.check(mlx.mlx_gather_qmm(res, x, w, sc, bi, lhs_idx, rhs_idx, true, mlx.mlx_optional_int.some(@intCast(group_size)), mlx.mlx_optional_int.some(@intCast(bits)), mode.cstr(), sorted, s));
    }
}

fn seSplitViews(s: mlx.mlx_stream, gate_up: mlx.mlx_array, down_raw: mlx.mlx_array, u: c_int, inter: c_int, hidden: c_int) !struct { gate: mlx.mlx_array, up: mlx.mlx_array, down: mlx.mlx_array } {
    const strides = [_]c_int{ 1, 1, 1 };
    const axes = [_]c_int{ 0, 2, 1 };
    var gate_raw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gate_raw);
    try mlx.check(mlx.mlx_slice(&gate_raw, gate_up, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ u, inter, hidden }, 3, &strides, 3, s));
    var up_raw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(up_raw);
    try mlx.check(mlx.mlx_slice(&up_raw, gate_up, &[_]c_int{ 0, inter, 0 }, 3, &[_]c_int{ u, 2 * inter, hidden }, 3, &strides, 3, s));
    var gate = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(gate);
    try mlx.check(mlx.mlx_transpose_axes(&gate, gate_raw, &axes, 3, s));
    var up = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(up);
    try mlx.check(mlx.mlx_transpose_axes(&up, up_raw, &axes, 3, s));
    var down = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(down);
    try mlx.check(mlx.mlx_transpose_axes(&down, down_raw, &axes, 3, s));
    return .{ .gate = gate, .up = up, .down = down };
}

fn seTruth(
    alloc: std.mem.Allocator,
    x: [*]const u16,
    gu: [*]const u16,
    dn: [*]const u16,
    slots: []const i32,
    w: [*]const u16,
    rows: usize,
    topk: usize,
    hidden: usize,
    inter: usize,
) ![]f32 {
    const out = try alloc.alloc(f32, rows * hidden);
    @memset(out, 0);
    const hvals = try alloc.alloc(f32, inter);
    defer alloc.free(hvals);
    for (0..rows) |r| {
        for (0..topk) |k| {
            const slot: usize = @intCast(slots[r * topk + k]);
            for (0..inter) |j| {
                var g: f32 = 0;
                var u: f32 = 0;
                const grow = (slot * 2 * inter + j) * hidden;
                const urow = (slot * 2 * inter + inter + j) * hidden;
                for (0..hidden) |c| {
                    const xv = seBf16(x[r * hidden + c]);
                    g += xv * seBf16(gu[grow + c]);
                    u += xv * seBf16(gu[urow + c]);
                }
                hvals[j] = (g / (1.0 + @exp(-g))) * u;
            }
            const wv = seBf16(w[r * topk + k]);
            for (0..hidden) |o| {
                var acc: f32 = 0;
                const drow = (slot * hidden + o) * inter;
                for (0..inter) |j| acc += hvals[j] * seBf16(dn[drow + j]);
                out[r * hidden + o] += wv * acc;
            }
        }
    }
    return out;
}

fn seUlpBf16(x: f32) f32 {
    const a = @abs(x);
    if (a == 0 or !std.math.isFinite(a)) return 0;
    return @exp2(@floor(@log2(a)) - 7.0);
}

fn seErrStats(alloc: std.mem.Allocator, got: mlx.mlx_array, truth: []const f32) !SeErr {
    try mlx.check(mlx.mlx_array_eval(got));
    const p = mlx.mlx_array_data_bfloat16(got) orelse return error.Bf16Unreadable;
    _ = alloc;
    var max: f32 = 0;
    var sum: f64 = 0;
    for (truth, 0..) |t, i| {
        const v = seBf16(p[i]);
        if (!std.math.isFinite(v)) return error.NonFiniteOutput;
        const d = @abs(v - t);
        if (d > max) max = d;
        sum += @as(f64, d) * @as(f64, d);
    }
    return .{ .max = max, .rms = @floatCast(@sqrt(sum / @as(f64, @floatFromInt(truth.len)))) };
}

fn seSlots(alloc: std.mem.Allocator, rows: usize, topk: usize, u: usize) ![]i32 {
    const out = try alloc.alloc(i32, rows * topk);
    for (0..rows) |r| {
        for (0..topk) |k| out[r * topk + k] = @intCast((r * 7 + k * 5 + 1) % u);
        out[r * topk + 2] = out[r * topk + 0];
        out[r * topk + 5] = out[r * topk + 1];
        if (r > 0) out[r * topk + 7] = out[0 * topk + 3];
    }
    return out;
}

fn gpuPickMatchesHost(comptime E: usize, comptime K: usize, seed: u64, log_sigmoid: bool) !void {
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var prng = std.Random.DefaultPrng.init(seed);
    const rand = prng.random();
    for (0..200) |case| {
        var logits: [E]f32 = undefined;
        for (&logits) |*v| v.* = rand.floatNorm(f32) * 2;
        if (case % 7 == 0) logits[rand.uintLessThan(usize, E)] = logits[3];
        if (log_sigmoid) {
            // Every value <= 0, some where sigmoid underflows: the pick reads what the served path feeds it.
            for (&logits) |*v| if (rand.float(f32) < 0.05) {
                v.* = -120;
            };
            const raw = mlx.mlx_array_new_data(&logits, &[_]c_int{ 1, 1, E }, 3, .float32);
            defer _ = mlx.mlx_array_free(raw);
            var swap = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(swap);
            try routerSwapLogits(&swap, raw, .log_sigmoid, s);
            try mlx.check(mlx.mlx_array_eval(swap));
            @memcpy(&logits, mlx.mlx_array_data_float32(swap).?[0..E]);
        }
        var map: [E]i32 = undefined;
        var cached: [E]bool = undefined;
        for (&map, &cached, 0..) |*m, *c, e| {
            c.* = rand.float(f32) < 0.5;
            m.* = if (c.*) @intCast(e % 251) else -1;
        }
        var starved: [E]u8 = undefined;
        for (&starved) |*v| v.* = if (rand.float(f32) < 0.1) expert_stream_mod.PICK_STARVE_LIMIT else 0;
        var ids: [K]u16 = undefined;
        var picked_n: usize = 0;
        while (picked_n < K) {
            const e: u16 = @intCast(rand.uintLessThan(usize, E));
            if (std.mem.indexOfScalar(u16, ids[0..picked_n], e) != null) continue;
            ids[picked_n] = e;
            picked_n += 1;
        }
        const tolerance: f32 = if (case % 2 == 0) 0.2 else 0.6;
        var host_ids = ids;
        var taken: [E]bool = undefined;
        var loading: [E]bool = undefined;
        var host_starved = starved;
        _ = expert_stream_mod.substituteMisses(&host_ids, &logits, &cached, tolerance, K, &taken, &loading, &host_starved);

        var ids_i32: [K]i32 = undefined;
        for (&ids_i32, ids) |*d, v| d.* = v;
        const a_ids = mlx.mlx_array_new_data(&ids_i32, &[_]c_int{ 1, 1, K }, 3, .int32);
        defer _ = mlx.mlx_array_free(a_ids);
        const a_logits = mlx.mlx_array_new_data(&logits, &[_]c_int{ 1, 1, E }, 3, .float32);
        defer _ = mlx.mlx_array_free(a_logits);
        const a_map = mlx.mlx_array_new_data(&map, &[_]c_int{E}, 1, .int32);
        defer _ = mlx.mlx_array_free(a_map);
        const a_starved = mlx.mlx_array_new_data(&starved, &[_]c_int{E}, 1, .uint8);
        defer _ = mlx.mlx_array_free(a_starved);
        const pick = try expertPickGpu(a_ids, a_logits, a_map, a_starved, tolerance, s);
        defer _ = mlx.mlx_array_free(pick.slots);
        defer _ = mlx.mlx_array_free(pick.picked);
        try mlx.check(mlx.mlx_array_eval(pick.picked));
        try mlx.check(mlx.mlx_array_eval(pick.slots));
        const got = mlx.mlx_array_data_int32(pick.picked).?;
        const slots = mlx.mlx_array_data_int32(pick.slots).?;
        for (host_ids, 0..) |want, i| {
            try testing.expectEqual(@as(i32, want), got[i]);
            try testing.expectEqual(@max(map[want], 0), slots[i]);
        }
    }
}

test "the swap count is reported at the first swap and then every 100k routed ids" {
    try testing.expect(swapReportDue(true, 376, 0));
    try testing.expect(!swapReportDue(false, 376, 0));
    try testing.expect(swapReportDue(false, 100_000, 0));
    try testing.expect(!swapReportDue(false, 199_999, 100_000));
}

test "streamed expert work is handed to the GPU early only at decode widths" {
    try testing.expect(streamHandoffApplies(1));
    try testing.expect(streamHandoffApplies(16));
    try testing.expect(!streamHandoffApplies(17));
    try testing.expect(streamSharedEarlyApplies(true, &.{ 1, 1, 10 }));
    try testing.expect(!streamSharedEarlyApplies(true, &.{ 1, 512, 10 }));
    try testing.expect(!streamSharedEarlyApplies(false, &.{ 1, 1, 10 }));
}

test "expert swap reads softmax logits raw and a biased sigmoid router through log sigmoid, at decode widths without a capture" {
    try testing.expectEqual(@as(?SwapLogits, .softmax), expertSwapKind(0.3, false, false, false, &.{ 1, 4, 10 }));
    try testing.expectEqual(@as(?SwapLogits, .log_sigmoid), expertSwapKind(0.3, true, true, false, &.{ 1, 1, 8 }));
    try testing.expectEqual(@as(?SwapLogits, null), expertSwapKind(0.3, true, false, false, &.{ 1, 1, 10 }));
    try testing.expectEqual(@as(?SwapLogits, null), expertSwapKind(0, true, true, false, &.{ 1, 1, 8 }));
    try testing.expectEqual(@as(?SwapLogits, null), expertSwapKind(0.3, false, false, true, &.{ 1, 1, 10 }));
    try testing.expectEqual(@as(?SwapLogits, null), expertSwapKind(0.3, true, true, false, &.{ 1, 512, 8 }));
}

test "a sigmoid router's swap logits make the pick test its sigmoid probability ratio" {
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    // (missed logit, cached candidate logit, tolerance): near saturation a raw logit gap would refuse a
    // substitute the sigmoid ratio accepts, and -120 must stay finite.
    const cases = [_][3]f32{ .{ 8, 3, 0.2 }, .{ 8, 3, 0.04 }, .{ -2, -2.3, 0.2 }, .{ -2, -2.1, 0.2 }, .{ 0, -0.2, 0.2 }, .{ 20, -5, 0.6 }, .{ -120, -120.1, 0.2 } };
    for (cases) |c| {
        const raw = [_]f32{ c[0], c[1], -50, -50 };
        const arr = mlx.mlx_array_new_data(&raw, &[_]c_int{ 1, 1, 4 }, 3, .float32);
        defer _ = mlx.mlx_array_free(arr);
        var swap = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(swap);
        try routerSwapLogits(&swap, arr, .log_sigmoid, s);
        try mlx.check(mlx.mlx_array_eval(swap));
        const got = mlx.mlx_array_data_float32(swap).?;
        var ids = [_]u16{0};
        const cached = [_]bool{ false, true, false, false };
        var taken: [4]bool = undefined;
        var loading: [4]bool = undefined;
        var starved = [_]u8{ 0, 0, 0, 0 };
        _ = expert_stream_mod.substituteMisses(&ids, got[0..4], &cached, c[2], 1, &taken, &loading, &starved);
        const sig = struct {
            fn f(x: f64) f64 {
                return 1.0 / (1.0 + @exp(-x));
            }
        }.f;
        const ratio = if (c[0] < -100) @exp(@as(f64, c[1] - c[0])) else sig(c[1]) / sig(c[0]);
        try testing.expect(std.math.isFinite(got[0]) and std.math.isFinite(got[1]));
        try testing.expectEqual(@as(u16, if (ratio >= 1.0 - c[2]) 1 else 0), ids[0]);
    }
}

test "speculative expert compute runs at decode widths without an imatrix capture" {
    try testing.expect(expertSpecApplies(false, &.{ 1, 1, 10 }));
    try testing.expect(expertSpecApplies(false, &.{ 1, 16, 10 }));
    try testing.expect(!expertSpecApplies(false, &.{ 1, 17, 10 }));
    try testing.expect(!expertSpecApplies(true, &.{ 1, 1, 10 }));
}

test "gatherExpertMm dense bf16 matches per-expert ground truth (decode + prefill shapes)" {
    try mlx.requireGpu();
    // Centerpiece TDD gate for fully-dense bf16 MoE. Proves the dense gather_mm
    // path (gatherExpertMm with null scales + generalized transposeBf16Weight)
    // computes the same per-expert matmul as (a) an independent fp32 ground truth
    // and (b) the established quantized gather_qmm path — for BOTH the decode
    // (S=1, unsorted, 5D x) and prefill ([N,1,in], sorted) call shapes used by
    // moeMLP2. If the historical Qwen3.6-A3B-bf16 generation bug lived in the
    // expert gather, this test fails for the right reason.
    const s = mlx.gpuStream();
    const alloc = testing.allocator;

    const E = 4;
    const IN = 32; // must be a multiple of gs
    const OUT = 8;
    const gs = 32; // mlx_quantize supports group sizes 32, 64, 128
    const bits = 8; // near-lossless quant for a tight cross-check

    // Build w_orig [E, OUT, IN] bf16 from a deterministic small-valued buffer.
    var w_host: [E * OUT * IN]f32 = undefined;
    for (&w_host, 0..) |*v, i| v.* = (@as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6))) * 0.05;
    var w_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_f32);
    {
        const sh = [_]c_int{ E, OUT, IN };
        w_f32 = mlx.mlx_array_new_data(&w_host, &sh, 3, .float32);
    }
    var w_bf16 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_bf16);
    try mlx.check(mlx.mlx_astype(&w_bf16, w_f32, .bfloat16, s));

    // Quantize → [q, sc, bi]; dequantize back so the dense path consumes the
    // exact numbers gather_qmm sees internally (cancels quant error in the
    // dense-vs-quant comparison).
    var qvec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(qvec);
    try mlx.check(mlx.mlx_quantize(&qvec, w_bf16, mlx.mlx_optional_int.some(gs), mlx.mlx_optional_int.some(bits), "affine", .{}, s));
    var q_w = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_w);
    var q_sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_sc);
    var q_bi = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q_bi);
    try mlx.check(mlx.mlx_vector_array_get(&q_w, qvec, 0));
    try mlx.check(mlx.mlx_vector_array_get(&q_sc, qvec, 1));
    try mlx.check(mlx.mlx_vector_array_get(&q_bi, qvec, 2));

    var w_deq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_deq);
    try mlx.check(mlx.mlx_dequantize(&w_deq, q_w, q_sc, q_bi, mlx.mlx_optional_int.some(gs), mlx.mlx_optional_int.some(bits), "affine", .{}, .{ .value = .bfloat16, .has_value = true }, s));
    // w_deq_t [E, IN, OUT] — the dense weight layout the loader produces.
    const w_deq_t = try transposeBf16Weight(w_deq, s);
    defer _ = mlx.mlx_array_free(w_deq_t);

    // Read dequantized weights to host for ground truth: w_deq_host[e*OUT*IN + o*IN + k].
    var w_deq_host: [E * OUT * IN]f32 = undefined;
    try testReadF32(w_deq, &w_deq_host, s);

    const null_sc = mlx.mlx_array{ .ctx = null };
    const null_bi = mlx.mlx_array{ .ctx = null };
    const no_idx = mlx.mlx_array{ .ctx = null };

    // ── Decode shape: x_exp [1,1,1,1,IN], inds [1,1,K], sorted=false ──
    {
        const K = 2;
        var x_host: [IN]f32 = undefined;
        for (&x_host, 0..) |*v, i| v.* = (@as(f32, @floatFromInt(@as(i32, @intCast(i % 5)) - 2))) * 0.1;
        var x_f32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_f32);
        {
            const sh = [_]c_int{IN};
            x_f32 = mlx.mlx_array_new_data(&x_host, &sh, 1, .float32);
        }
        var x_bf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_bf);
        try mlx.check(mlx.mlx_astype(&x_bf, x_f32, .bfloat16, s));
        var x_exp = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_exp);
        {
            const sh = [_]c_int{ 1, 1, 1, 1, IN };
            try mlx.check(mlx.mlx_reshape(&x_exp, x_bf, &sh, 5, s));
        }
        var x_bf_host: [IN]f32 = undefined;
        try testReadF32(x_bf, &x_bf_host, s);

        const inds_host = [_]u32{ 1, 3 };
        var inds = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(inds);
        {
            const sh = [_]c_int{ 1, 1, K };
            inds = mlx.mlx_array_new_data(&inds_host, &sh, 3, .uint32);
        }

        // Ground truth: gt[k*OUT + o] = sum_in x[in] * w_deq[inds[k]][o][in].
        var gt: [K * OUT]f32 = undefined;
        for (0..K) |k| {
            const e = inds_host[k];
            for (0..OUT) |o| {
                var acc: f32 = 0;
                for (0..IN) |in| acc += x_bf_host[in] * w_deq_host[e * OUT * IN + o * IN + in];
                gt[k * OUT + o] = acc;
            }
        }

        // Dense path.
        var dense5 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dense5);
        try gatherExpertMm(&dense5, x_exp, w_deq_t, null_sc, null_bi, no_idx, inds, bits, gs, .affine, false, s);
        var dense = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dense);
        try mlx.check(mlx.mlx_squeeze(&dense, dense5, s)); // [K, OUT]
        const dense_host = try alloc.alloc(f32, K * OUT);
        defer alloc.free(dense_host);
        try testReadF32(dense, dense_host, s);

        // Quantized path (cross-check).
        var quant5 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(quant5);
        try gatherExpertMm(&quant5, x_exp, q_w, q_sc, q_bi, no_idx, inds, bits, gs, .affine, false, s);
        var quant = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(quant);
        try mlx.check(mlx.mlx_squeeze(&quant, quant5, s));
        const quant_host = try alloc.alloc(f32, K * OUT);
        defer alloc.free(quant_host);
        try testReadF32(quant, quant_host, s);

        for (0..K * OUT) |i| {
            try testing.expectApproxEqAbs(gt[i], dense_host[i], 2e-2); // dense == ground truth
            try testing.expectApproxEqAbs(gt[i], quant_host[i], 2e-2); // quant agrees too
        }
    }

    // ── Prefill/sorted shape: x_rep [N,1,IN], sorted_inds [N], sorted=true ──
    {
        const N = 5;
        var x_host: [N * IN]f32 = undefined;
        for (&x_host, 0..) |*v, i| v.* = (@as(f32, @floatFromInt(@as(i32, @intCast(i % 7)) - 3))) * 0.07;
        var x_f32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_f32);
        {
            const sh = [_]c_int{ N, 1, IN };
            x_f32 = mlx.mlx_array_new_data(&x_host, &sh, 3, .float32);
        }
        var x_rep = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_rep);
        try mlx.check(mlx.mlx_astype(&x_rep, x_f32, .bfloat16, s));
        var x_bf_host: [N * IN]f32 = undefined;
        try testReadF32(x_rep, &x_bf_host, s);

        const sorted_host = [_]u32{ 0, 0, 1, 2, 3 }; // sorted experts
        var sorted_inds = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sorted_inds);
        {
            const sh = [_]c_int{N};
            sorted_inds = mlx.mlx_array_new_data(&sorted_host, &sh, 1, .uint32);
        }

        // Ground truth: gt[i*OUT + o] = sum_in x[i][in] * w_deq[sorted[i]][o][in].
        var gt: [N * OUT]f32 = undefined;
        for (0..N) |i| {
            const e = sorted_host[i];
            for (0..OUT) |o| {
                var acc: f32 = 0;
                for (0..IN) |in| acc += x_bf_host[i * IN + in] * w_deq_host[e * OUT * IN + o * IN + in];
                gt[i * OUT + o] = acc;
            }
        }

        var dense3 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dense3);
        try gatherExpertMm(&dense3, x_rep, w_deq_t, null_sc, null_bi, no_idx, sorted_inds, bits, gs, .affine, true, s);
        var dense = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dense);
        try mlx.check(mlx.mlx_squeeze(&dense, dense3, s)); // [N, OUT]
        const dense_host = try alloc.alloc(f32, N * OUT);
        defer alloc.free(dense_host);
        try testReadF32(dense, dense_host, s);

        var quant3 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(quant3);
        try gatherExpertMm(&quant3, x_rep, q_w, q_sc, q_bi, no_idx, sorted_inds, bits, gs, .affine, true, s);
        var quant = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(quant);
        try mlx.check(mlx.mlx_squeeze(&quant, quant3, s));
        const quant_host = try alloc.alloc(f32, N * OUT);
        defer alloc.free(quant_host);
        try testReadF32(quant, quant_host, s);

        for (0..N * OUT) |i| {
            try testing.expectApproxEqAbs(gt[i], dense_host[i], 2e-2);
            try testing.expectApproxEqAbs(gt[i], quant_host[i], 2e-2);
        }
    }
}

test "streamed expert kernel arm is no worse than the composite arm against fp32 truth" {
    const alloc = std.testing.allocator;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const hidden: c_int = 2560;
    const inter: c_int = 640;
    const topk: c_int = 10;
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, 0x5eed51ab));
    var worse: usize = 0;
    const SE_DRAWS: usize = 4;

    for ([_]c_int{ 16, 128 }) |u| {
        var gu = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gu);
        try mlx.check(mlx.mlx_random_normal(&gu, &[_]c_int{ u, 2 * inter, hidden }, 3, .bfloat16, 0.0, 0.02, key, s));
        try mlx.check(mlx.mlx_array_eval(gu));
        var dn = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(dn);
        try mlx.check(mlx.mlx_random_normal(&dn, &[_]c_int{ u, hidden, inter }, 3, .bfloat16, 0.0, 0.02, key, s));
        try mlx.check(mlx.mlx_array_eval(dn));
        const views = try seSplitViews(s, gu, dn, u, inter, hidden);
        defer _ = mlx.mlx_array_free(views.gate);
        defer _ = mlx.mlx_array_free(views.up);
        defer _ = mlx.mlx_array_free(views.down);

        for ([_]c_int{ 1, 2, 4 }) |rows| {
            for (0..SE_DRAWS) |draw| {
                var dkey = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(dkey);
                try mlx.check(mlx.mlx_random_key(&dkey, 0xa5a50000 + @as(u64, draw) * 7919 + @as(u64, @intCast(u))));
                var x = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(x);
                try mlx.check(mlx.mlx_random_normal(&x, &[_]c_int{ rows, hidden }, 2, .bfloat16, 0.0, 1.0, dkey, s));
                try mlx.check(mlx.mlx_array_eval(x));
                var w = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(w);
                try mlx.check(mlx.mlx_random_normal(&w, &[_]c_int{ rows, topk }, 2, .bfloat16, 0.0, 1.0, dkey, s));
                try mlx.check(mlx.mlx_array_eval(w));
                const slots_host = try seSlots(alloc, @intCast(rows), @intCast(topk), @intCast(u));
                defer alloc.free(slots_host);
                const slots = mlx.mlx_array_new_data(slots_host.ptr, &[_]c_int{ rows, topk }, 2, .int32);
                defer _ = mlx.mlx_array_free(slots);

                const op = StreamedExpertOperands{
                    .x_rows = x,
                    .gate = views.gate,
                    .up = views.up,
                    .down = views.down,
                    .slab_gate_up = gu,
                    .slab_down = dn,
                    .slots = slots,
                    .weights = w,
                };
                const truth = try seTruth(
                    alloc,
                    mlx.mlx_array_data_bfloat16(x).?,
                    mlx.mlx_array_data_bfloat16(gu).?,
                    mlx.mlx_array_data_bfloat16(dn).?,
                    slots_host,
                    mlx.mlx_array_data_bfloat16(w).?,
                    @intCast(rows),
                    @intCast(topk),
                    @intCast(hidden),
                    @intCast(inter),
                );
                defer alloc.free(truth);

                const composite = try streamedExpertCompute(s, op, rows, false);
                defer _ = mlx.mlx_array_free(composite);
                const kernel = try streamedExpertCompute(s, op, rows, true);
                defer _ = mlx.mlx_array_free(kernel);
                try std.testing.expectEqualSlices(c_int, &[_]c_int{ rows, hidden }, mlx.getShape(composite));
                try std.testing.expectEqualSlices(c_int, &[_]c_int{ rows, hidden }, mlx.getShape(kernel));
                const ce = try seErrStats(alloc, composite, truth);
                const ke = try seErrStats(alloc, kernel, truth);
                var tmax: f32 = 0;
                for (truth) |v| {
                    if (@abs(v) > tmax) tmax = @abs(v);
                }
                const allow: f32 = if (expert_bf16.downKernelPreferred(@intCast(rows))) 0 else seUlpBf16(tmax);
                const bad = ke.max > ce.max + allow or ke.rms > ce.rms;
                if (bad) {
                    std.debug.print("streamed-expert U={d:>4} R={d} draw={d}: kernel max={e:.3} rms={e:.3} | composite max={e:.3} rms={e:.3} | tmax={e:.3} allow={e:.3} [WORSE]\n", .{ u, rows, draw, ke.max, ke.rms, ce.max, ce.rms, tmax, allow });
                    worse += 1;
                }
            }
        }
        _ = mlx.mlx_clear_cache();
    }
    try std.testing.expectEqual(@as(usize, 0), worse);
}

test "streamed expert composite arm keeps the pre-change bytes" {
    try mlx.requireGpu();
    const alloc = std.testing.allocator;
    const s = mlx.gpuStream();
    const hidden: c_int = 64;
    const inter: c_int = 32;
    const u: c_int = 8;
    const topk: c_int = 10;
    var prng = std.Random.DefaultPrng.init(0xbf16c0de);
    const rnd = prng.random();

    const gu_host = try alloc.alloc(u16, @intCast(u * 2 * inter * hidden));
    defer alloc.free(gu_host);
    seFillBf16(rnd, gu_host);
    const dn_host = try alloc.alloc(u16, @intCast(u * hidden * inter));
    defer alloc.free(dn_host);
    seFillBf16(rnd, dn_host);
    const gu = mlx.mlx_array_new_data(gu_host.ptr, &[_]c_int{ u, 2 * inter, hidden }, 3, .bfloat16);
    defer _ = mlx.mlx_array_free(gu);
    const dn = mlx.mlx_array_new_data(dn_host.ptr, &[_]c_int{ u, hidden, inter }, 3, .bfloat16);
    defer _ = mlx.mlx_array_free(dn);
    const views = try seSplitViews(s, gu, dn, u, inter, hidden);
    defer _ = mlx.mlx_array_free(views.gate);
    defer _ = mlx.mlx_array_free(views.up);
    defer _ = mlx.mlx_array_free(views.down);

    const pinned = [_]u64{ 0xc39bfc6d04f4e917, 0x1e43226d66b1fd96 };
    for ([_]c_int{ 1, 3 }, pinned) |rows, want| {
        const x_host = try alloc.alloc(u16, @intCast(rows * hidden));
        defer alloc.free(x_host);
        seFillBf16(rnd, x_host);
        const w_host = try alloc.alloc(u16, @intCast(rows * topk));
        defer alloc.free(w_host);
        seFillBf16(rnd, w_host);
        const slots_host = try alloc.alloc(i32, @intCast(rows * topk));
        defer alloc.free(slots_host);
        for (slots_host, 0..) |*v, i| v.* = @intCast((i * 3 + 1) % @as(usize, @intCast(u)));

        const x = mlx.mlx_array_new_data(x_host.ptr, &[_]c_int{ rows, hidden }, 2, .bfloat16);
        defer _ = mlx.mlx_array_free(x);
        const slots = mlx.mlx_array_new_data(slots_host.ptr, &[_]c_int{ rows, topk }, 2, .int32);
        defer _ = mlx.mlx_array_free(slots);
        const w = mlx.mlx_array_new_data(w_host.ptr, &[_]c_int{ rows, topk }, 2, .bfloat16);
        defer _ = mlx.mlx_array_free(w);

        const out = try streamedExpertCompute(s, .{
            .x_rows = x,
            .gate = views.gate,
            .up = views.up,
            .down = views.down,
            .slab_gate_up = gu,
            .slab_down = dn,
            .slots = slots,
            .weights = w,
        }, rows, false);
        defer _ = mlx.mlx_array_free(out);
        try std.testing.expectEqualSlices(c_int, &[_]c_int{ rows, hidden }, mlx.getShape(out));
        try std.testing.expectEqual(want, try seHash(out));
    }
}

test "imatrix capture keys the streamed MoE statistics on GLOBAL expert ids, on every arm" {
    const alloc = std.testing.allocator;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const hidden: c_int = 64;
    const inter: c_int = 32;
    const u: c_int = 8;
    const experts: c_int = 16; // the slab holds 8 of them; ids 8..15 are the resident half
    const topk: c_int = 10;
    var prng = std.Random.DefaultPrng.init(0x1A7A1);
    const rnd = prng.random();

    const gu_host = try alloc.alloc(u16, @intCast(u * 2 * inter * hidden));
    defer alloc.free(gu_host);
    seFillBf16(rnd, gu_host);
    const dn_host = try alloc.alloc(u16, @intCast(u * hidden * inter));
    defer alloc.free(dn_host);
    seFillBf16(rnd, dn_host);
    const gu = mlx.mlx_array_new_data(gu_host.ptr, &[_]c_int{ u, 2 * inter, hidden }, 3, .bfloat16);
    defer _ = mlx.mlx_array_free(gu);
    const dn = mlx.mlx_array_new_data(dn_host.ptr, &[_]c_int{ u, hidden, inter }, 3, .bfloat16);
    defer _ = mlx.mlx_array_free(dn);
    const views = try seSplitViews(s, gu, dn, u, inter, hidden);
    defer _ = mlx.mlx_array_free(views.gate);
    defer _ = mlx.mlx_array_free(views.up);
    defer _ = mlx.mlx_array_free(views.down);

    // rows 1 (unsorted composite) and 2 (sorted composite), each on the kernel arm too.
    for ([_]c_int{ 1, 2 }) |rows| {
        for ([_]bool{ false, true }) |kernels| {
            const col = try imatrix_capture.Collector.init(alloc, s, "/dev/null", 3, experts, .qwen4_exp);
            defer col.deinit();
            const x_host = try alloc.alloc(u16, @intCast(rows * hidden));
            defer alloc.free(x_host);
            seFillBf16(rnd, x_host);
            const w_host = try alloc.alloc(u16, @intCast(rows * topk));
            defer alloc.free(w_host);
            seFillBf16(rnd, w_host);
            const slots_host = try seSlots(alloc, @intCast(rows), @intCast(topk), @intCast(u));
            defer alloc.free(slots_host);
            const global_host = try alloc.alloc(i32, slots_host.len);
            defer alloc.free(global_host);
            for (slots_host, 0..) |slot, i| global_host[i] = slot + u;

            const x = mlx.mlx_array_new_data(x_host.ptr, &[_]c_int{ rows, hidden }, 2, .bfloat16);
            defer _ = mlx.mlx_array_free(x);
            const w = mlx.mlx_array_new_data(w_host.ptr, &[_]c_int{ rows, topk }, 2, .bfloat16);
            defer _ = mlx.mlx_array_free(w);
            const slots = mlx.mlx_array_new_data(slots_host.ptr, &[_]c_int{ rows, topk }, 2, .int32);
            defer _ = mlx.mlx_array_free(slots);
            const globals = mlx.mlx_array_new_data(global_host.ptr, &[_]c_int{ rows, topk }, 2, .int32);
            defer _ = mlx.mlx_array_free(globals);

            const out = try streamedExpertCompute(s, .{
                .x_rows = x,
                .gate = views.gate,
                .up = views.up,
                .down = views.down,
                .slab_gate_up = gu,
                .slab_down = dn,
                .slots = slots,
                .weights = w,
                .imatrix = .{ .collector = col, .layer = 2, .global_ids = globals },
            }, rows, kernels);
            defer _ = mlx.mlx_array_free(out);

            const slot = &col.layers[2];
            try std.testing.expectEqual(@as(u64, @intCast(rows)), slot.tokens);
            try std.testing.expectEqualSlices(c_int, &[_]c_int{ experts, hidden }, mlx.getShape(slot.gu));
            try std.testing.expectEqualSlices(c_int, &[_]c_int{ experts, inter }, mlx.getShape(slot.down));
            try mlx.check(mlx.mlx_array_eval(slot.rows));
            const routed = mlx.mlx_array_data_float32(slot.rows).?;
            var low: f32 = 0;
            var high: f32 = 0;
            for (0..@intCast(u)) |e| low += routed[e];
            for (@intCast(u)..@intCast(experts)) |e| high += routed[e];
            try std.testing.expectEqual(@as(f32, 0), low);
            try std.testing.expectEqual(@as(f32, @floatFromInt(rows * topk)), high);
            // Every expert the slab served owns some mass on both statistics.
            try mlx.check(mlx.mlx_array_eval(slot.gu));
            try mlx.check(mlx.mlx_array_eval(slot.down));
            const gu_acc = mlx.mlx_array_data_float32(slot.gu).?;
            const dn_acc = mlx.mlx_array_data_float32(slot.down).?;
            for (0..@intCast(u * hidden)) |i| try std.testing.expectEqual(@as(f32, 0), gu_acc[i]);
            for (0..@intCast(u * inter)) |i| try std.testing.expectEqual(@as(f32, 0), dn_acc[i]);
            var seen: f32 = 0;
            for (@intCast(u * inter)..@intCast(experts * inter)) |i| seen += dn_acc[i];
            try std.testing.expect(seen > 0);
        }
    }
}

test "a streamed layer's imatrix tap views the routing ids and is absent with no capture" {
    const alloc = std.testing.allocator;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const ids_host = [_]i32{ 7, 3, 1, 5 };
    const ids = mlx.mlx_array_new_data(&ids_host, &[_]c_int{ 1, 2, 2 }, 3, .int32);
    defer _ = mlx.mlx_array_free(ids);
    try std.testing.expect(try ImatrixTap.forLayer(null, 0, ids, 2, 2, s) == null);

    const col = try imatrix_capture.Collector.init(alloc, s, "/dev/null", 2, 8, .mimo_v2);
    defer col.deinit();
    const tap = (try ImatrixTap.forLayer(col, 1, ids, 2, 2, s)).?;
    defer tap.deinit();
    try std.testing.expectEqual(@as(u16, 1), tap.layer);
    try std.testing.expectEqualSlices(c_int, &[_]c_int{ 2, 2 }, mlx.getShape(tap.global_ids));
    try mlx.check(mlx.mlx_array_eval(tap.global_ids));
    const got = mlx.mlx_array_data_int32(tap.global_ids).?;
    for (ids_host, 0..) |want, i| try std.testing.expectEqual(want, got[i]);
}

test "streamed expert compute refuses a misaligned slab by name" {
    const alloc = std.testing.allocator;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const hidden: c_int = 64;
    const inter: c_int = 32;
    const u: c_int = 8;
    const topk: c_int = 10;
    const rows: c_int = 1;

    const n: c_int = u * 2 * inter * hidden + 1;
    const gu_host = try alloc.alloc(u16, @intCast(n));
    defer alloc.free(gu_host);
    var prng = std.Random.DefaultPrng.init(7);
    seFillBf16(prng.random(), gu_host);
    const gu_base = mlx.mlx_array_new_data(gu_host.ptr, &[_]c_int{n}, 1, .bfloat16);
    defer _ = mlx.mlx_array_free(gu_base);
    var gu_sliced = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gu_sliced);
    try mlx.check(mlx.mlx_slice(&gu_sliced, gu_base, &[_]c_int{1}, 1, &[_]c_int{n}, 1, &[_]c_int{1}, 1, s));
    var gu_bad = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gu_bad);
    try mlx.check(mlx.mlx_reshape(&gu_bad, gu_sliced, &[_]c_int{ u, 2 * inter, hidden }, 3, s));
    const gu_good = mlx.mlx_array_new_data(gu_host.ptr + 1, &[_]c_int{ u, 2 * inter, hidden }, 3, .bfloat16);
    defer _ = mlx.mlx_array_free(gu_good);
    try mlx.check(mlx.mlx_array_eval(gu_good));

    const dn_host = try alloc.alloc(u16, @intCast(u * hidden * inter));
    defer alloc.free(dn_host);
    seFillBf16(prng.random(), dn_host);
    const dn = mlx.mlx_array_new_data(dn_host.ptr, &[_]c_int{ u, hidden, inter }, 3, .bfloat16);
    defer _ = mlx.mlx_array_free(dn);
    const views = try seSplitViews(s, gu_good, dn, u, inter, hidden);
    defer _ = mlx.mlx_array_free(views.gate);
    defer _ = mlx.mlx_array_free(views.up);
    defer _ = mlx.mlx_array_free(views.down);

    const x_host = try alloc.alloc(u16, @intCast(rows * hidden));
    defer alloc.free(x_host);
    seFillBf16(prng.random(), x_host);
    const x = mlx.mlx_array_new_data(x_host.ptr, &[_]c_int{ rows, hidden }, 2, .bfloat16);
    defer _ = mlx.mlx_array_free(x);
    const slots_host = try seSlots(alloc, @intCast(rows), @intCast(topk), @intCast(u));
    defer alloc.free(slots_host);
    const slots = mlx.mlx_array_new_data(slots_host.ptr, &[_]c_int{ rows, topk }, 2, .int32);
    defer _ = mlx.mlx_array_free(slots);
    const w_host = try alloc.alloc(u16, @intCast(rows * topk));
    defer alloc.free(w_host);
    seFillBf16(prng.random(), w_host);
    const w = mlx.mlx_array_new_data(w_host.ptr, &[_]c_int{ rows, topk }, 2, .bfloat16);
    defer _ = mlx.mlx_array_free(w);

    var op = StreamedExpertOperands{
        .x_rows = x,
        .gate = views.gate,
        .up = views.up,
        .down = views.down,
        .slab_gate_up = gu_bad,
        .slab_down = dn,
        .slots = slots,
        .weights = w,
    };
    try mlx.check(mlx.mlx_array_eval(gu_bad));
    try std.testing.expect(@intFromPtr(mlx.mlx_array_data_uint8(gu_bad).?) % EXPERT_SLAB_ALIGN_BYTES != 0);
    try std.testing.expectError(error.ExpertSlabUnaligned, streamedExpertCompute(s, op, rows, true));

    op.slab_gate_up = gu_good;
    const ok = try streamedExpertCompute(s, op, rows, true);
    defer _ = mlx.mlx_array_free(ok);
    try std.testing.expectEqualSlices(c_int, &[_]c_int{ rows, hidden }, mlx.getShape(ok));
}

test "streamed expert compute falls back to the composite arm past the kernel row cap" {
    const alloc = std.testing.allocator;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const hidden: c_int = 64;
    const inter: c_int = 32;
    const u: c_int = 8;
    const topk: c_int = 10;
    const rows: c_int = expert_bf16.MAX_ROWS + 1;
    var prng = std.Random.DefaultPrng.init(0xcafe17);
    const rnd = prng.random();

    const gu_host = try alloc.alloc(u16, @intCast(u * 2 * inter * hidden));
    defer alloc.free(gu_host);
    seFillBf16(rnd, gu_host);
    const dn_host = try alloc.alloc(u16, @intCast(u * hidden * inter));
    defer alloc.free(dn_host);
    seFillBf16(rnd, dn_host);
    const x_host = try alloc.alloc(u16, @intCast(rows * hidden));
    defer alloc.free(x_host);
    seFillBf16(rnd, x_host);
    const w_host = try alloc.alloc(u16, @intCast(rows * topk));
    defer alloc.free(w_host);
    seFillBf16(rnd, w_host);
    const slots_host = try seSlots(alloc, @intCast(rows), @intCast(topk), @intCast(u));
    defer alloc.free(slots_host);

    const gu = mlx.mlx_array_new_data(gu_host.ptr, &[_]c_int{ u, 2 * inter, hidden }, 3, .bfloat16);
    defer _ = mlx.mlx_array_free(gu);
    const dn = mlx.mlx_array_new_data(dn_host.ptr, &[_]c_int{ u, hidden, inter }, 3, .bfloat16);
    defer _ = mlx.mlx_array_free(dn);
    const x = mlx.mlx_array_new_data(x_host.ptr, &[_]c_int{ rows, hidden }, 2, .bfloat16);
    defer _ = mlx.mlx_array_free(x);
    const w = mlx.mlx_array_new_data(w_host.ptr, &[_]c_int{ rows, topk }, 2, .bfloat16);
    defer _ = mlx.mlx_array_free(w);
    const slots = mlx.mlx_array_new_data(slots_host.ptr, &[_]c_int{ rows, topk }, 2, .int32);
    defer _ = mlx.mlx_array_free(slots);
    const views = try seSplitViews(s, gu, dn, u, inter, hidden);
    defer _ = mlx.mlx_array_free(views.gate);
    defer _ = mlx.mlx_array_free(views.up);
    defer _ = mlx.mlx_array_free(views.down);
    const op = StreamedExpertOperands{
        .x_rows = x,
        .gate = views.gate,
        .up = views.up,
        .down = views.down,
        .slab_gate_up = gu,
        .slab_down = dn,
        .slots = slots,
        .weights = w,
    };
    const asked = try streamedExpertCompute(s, op, rows, true);
    defer _ = mlx.mlx_array_free(asked);
    const composite = try streamedExpertCompute(s, op, rows, false);
    defer _ = mlx.mlx_array_free(composite);
    try std.testing.expectEqualSlices(c_int, &[_]c_int{ rows, hidden }, mlx.getShape(asked));
    try std.testing.expectEqual(try seHash(composite), try seHash(asked));
}

test "the GPU expert pick matches the host substitution on one row" {
    try gpuPickMatchesHost(512, 10, 0x5eed, false);
}

test "the GPU expert pick matches the host on a sigmoid router's log-sigmoid logits, MiMo's E and K" {
    try gpuPickMatchesHost(256, 8, 0x51a7, true);
}

test "the GPU expert pick declines a shape its threadgroup cannot hold" {
    const s = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var ids = [_]i32{ 0, 1 };
    var logits: [2048]f32 = @splat(0);
    var map: [2048]i32 = @splat(-1);
    var starved: [2048]u8 = @splat(0);
    const a_ids = mlx.mlx_array_new_data(&ids, &[_]c_int{2}, 1, .int32);
    defer _ = mlx.mlx_array_free(a_ids);
    const a_logits = mlx.mlx_array_new_data(&logits, &[_]c_int{2048}, 1, .float32);
    defer _ = mlx.mlx_array_free(a_logits);
    const a_map = mlx.mlx_array_new_data(&map, &[_]c_int{2048}, 1, .int32);
    defer _ = mlx.mlx_array_free(a_map);
    const a_starved = mlx.mlx_array_new_data(&starved, &[_]c_int{2048}, 1, .uint8);
    defer _ = mlx.mlx_array_free(a_starved);
    try testing.expectError(error.UnsupportedShape, expertPickGpu(a_ids, a_logits, a_map, a_starved, 0.2, s));
}
