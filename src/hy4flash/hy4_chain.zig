//! Hy4's small f32 elementwise chains, each one launch that repeats the ops' arithmetic in their order with no fma
//! contraction, so every element is the chain's bytes: the engine's `fusedChain` takes BF16 chains only.
const std = @import("std");
const mlx = @import("../mlx.zig");
const Arr = mlx.mlx_array;
const kcfg = @import("hy4_kcfg.zig");

/// MLX's `Sigmoid` functor (`unary_ops.h`, the engine's `sushi_sigmoid`) on a float.
const header: [:0]const u8 =
    \\inline float hy4_sigmoid(float x) {
    \\  auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
    \\  return (x < 0) ? y : 1 - y;
    \\}
;

/// `(g · σ(g)) · u`: the SwiGLU's `silu` (`Ops.silu`, the Sigmoid then the Multiply) times the up rows.
const swiglu_source: [:0]const u8 =
    \\#pragma clang fp contract(off)
    \\uint i = thread_position_in_grid.x;
    \\float x = g[i];
    \\float a = x * hy4_sigmoid(x);
    \\out[i] = a * u[i];
;

var swiglu_kernel: ?mlx.mlx_fast_metal_kernel = null;

fn getKernel(slot: *?mlx.mlx_fast_metal_kernel, name: [*:0]const u8, ins: []const [*:0]const u8, src: [:0]const u8) !mlx.mlx_fast_metal_kernel {
    if (slot.*) |k| return k;
    const outs = [_][*:0]const u8{"out"};
    const iv = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new(name, iv, ov, src, header, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    slot.* = k;
    return k;
}

/// Which launch a config is for, and its shape.
const Kind = enum { swiglu, groups, rows, gated, sink_sm_causal, sink_sm_mask, ring_keys };
const Key = struct { kind: Kind, a: c_int, b: c_int, c: c_int = 0, d: c_int = 0 };
var cfgs: kcfg.Cache(Key) = .{};

fn sinkSoftmaxGroup(n: c_int) c_int {
    const axis = n + 1;
    return 32 * @divTrunc(@divTrunc(axis + 3, 4) + 31, 32);
}

fn build(k: Key) !kcfg.Config {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    // (outputs, grid x and y, threads a group)
    switch (k.kind) {
        .swiglu => {
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ k.a, k.b }, 2, .float32));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, k.a * k.b, 1, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, @min(k.a * k.b, 256), 1, 1));
        },
        .groups => {
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ k.a, k.b + 1 }, 2, .float32));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, k.b + 1, k.a, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, @min(k.b + 1, 256), 1, 1));
        },
        .rows => {
            const width = k.b * k.c;
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ k.a, width }, 2, .uint32));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ k.a, width }, 2, .bool_));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, width, k.a, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, @min(width, 256), 1, 1));
        },
        .gated => {
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ k.a, k.b }, 2, .float32));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, k.b, k.a, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, @min(k.b, 256), 1, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "V", k.c));
        },
        .sink_sm_causal, .sink_sm_mask => {
            const heads_first = (k.d & 1) != 0;
            const out_shape = if (heads_first) [_]c_int{ k.a, k.b, k.c } else [_]c_int{ k.b, k.a, k.c };
            const group = sinkSoftmaxGroup(k.c);
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &out_shape, 3, .float32));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, k.a * k.b * group, 1, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, group, 1, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "HEADS_FIRST", heads_first));
            if (k.kind == .sink_sm_mask) {
                try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "MASK_BOOL", (k.d & 2) != 0));
            }
        },
        .ring_keys => {
            const width = k.c + k.d;
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ k.a, k.b, k.c }, 3, .float32));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ k.a, k.b, k.d }, 3, .float32));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, width, k.b, k.a));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, @min(width, 256), 1, 1));
        },
    }
    return cfg;
}

/// Whether `swiglu` takes `g` and `u`: f32 of one shape on a GPU stream.
pub fn swigluApplies(s: mlx.mlx_stream, g: Arr, u: Arr) bool {
    return mlx.streamIsGpu(s) and mlx.mlx_array_dtype(g) == .float32 and mlx.mlx_array_dtype(u) == .float32 and
        std.mem.eql(c_int, mlx.getShape(g), mlx.getShape(u)) and mlx.mlx_array_size(g) > 0;
}

/// `silu(g) · u` in f32, `g` and `u` of one shape. The caller frees the result.
pub fn swiglu(s: mlx.mlx_stream, g: Arr, u: Arr) !Arr {
    const k = try getKernel(&swiglu_kernel, "sushi_hy4_swiglu", &.{ "g", "u" }, swiglu_source);
    const sh = mlx.getShape(g);
    const rows: c_int = if (sh.len == 2) sh[0] else 1;
    const cfg = try cfgs.get(.{ .kind = .swiglu, .a = rows, .b = @divExact(@as(c_int, @intCast(mlx.mlx_array_size(g))), rows) }, build);
    var out: [1]Arr = undefined;
    try applyOuts(s, k, &.{ g, u }, cfg, &out);
    if (sh.len == 2) return out[0];
    defer _ = mlx.mlx_array_free(out[0]);
    var shaped = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(shaped);
    try mlx.check(mlx.mlx_reshape(&shaped, out[0], sh.ptr, sh.len, s));
    return shaped;
}

/// `recallAt`'s groups [T, take + 1] f32: the picks `order` [T, take] (int32) widened, then each row's own group
/// `floor((p0 + t) / ratio)` (the arange, the Divide and the Floor).
const groups_source: [:0]const u8 =
    \\#pragma clang fp contract(off)
    \\uint c = thread_position_in_grid.x;
    \\uint t = thread_position_in_grid.y;
    \\uint k = threads_per_grid.x;
    \\out[t * k + c] = c + 1 < k ? static_cast<float>(order[t * (k - 1) + c]) : metal::floor(static_cast<float>(meta[0] + int(t)) / static_cast<float>(meta[1]));
;

/// `recalledRows` over the groups [T, k] f32 (own group last): each row's `k · ratio` key rows `pos = g · ratio + r`
/// clamped to `n - 1` (uint32), and whether it reads each (`pos` at or before its position, and from an earlier group
/// or the own group's column), in the ops' f32 order.
const rows_source: [:0]const u8 =
    \\#pragma clang fp contract(off)
    \\uint s = thread_position_in_grid.x;
    \\uint t = thread_position_in_grid.y;
    \\uint width = threads_per_grid.x;
    \\int ratio = meta[1];
    \\uint k = width / uint(ratio);
    \\uint j = s / uint(ratio);
    \\uint r = s % uint(ratio);
    \\float g = groups[t * k + j];
    \\float own_group = groups[t * k + k - 1];
    \\bool ok = (g < own_group) || (static_cast<float>(k - 1) <= static_cast<float>(j));
    \\float base = g * static_cast<float>(ratio);
    \\float pos = base + static_cast<float>(r);
    \\bool reads = (pos <= static_cast<float>(meta[0] + int(t))) && ok;
    \\float last = static_cast<float>(meta[2] - 1);
    \\float clamped = metal::isnan(pos) ? pos : (pos < last ? pos : last);
    \\rows[t * width + s] = static_cast<uint>(clamped);
    \\keep[t * width + s] = reads;
;

var groups_kernel: ?mlx.mlx_fast_metal_kernel = null;
var rows_kernel: ?mlx.mlx_fast_metal_kernel = null;

fn intsArr(v: []const i32) !Arr {
    return mlx.newData(v.ptr, &.{@intCast(v.len)}, .int32);
}

/// `recallAt`'s groups on a GPU stream from the top-k picks `order` [T, take] int32 (take ≥ 1). The caller frees.
pub fn recallGroups(s: mlx.mlx_stream, order: Arr, p0: c_int, ratio: c_int) !Arr {
    const k = try getKernel(&groups_kernel, "sushi_hy4_recall_groups", &.{ "order", "meta" }, groups_source);
    const sh = mlx.getShape(order);
    const mv = [_]i32{ p0, ratio };
    const meta = try intsArr(&mv);
    defer _ = mlx.mlx_array_free(meta);
    const cfg = try cfgs.get(.{ .kind = .groups, .a = sh[0], .b = sh[1] }, build);
    var out: [1]Arr = undefined;
    try applyOuts(s, k, &.{ order, meta }, cfg, &out);
    return out[0];
}

/// `fwd.recalledRows` on a GPU stream over `groups` [T, k] f32: the key rows [T, k·ratio] uint32 and the keep mask
/// (bool). The caller frees both.
pub fn recalledRows(s: mlx.mlx_stream, groups: Arr, p0: c_int, n: c_int, ratio: c_int) !struct { rows: Arr, keep: Arr } {
    if (rows_kernel == null) {
        const ins = [_][*:0]const u8{ "groups", "meta" };
        const outs = [_][*:0]const u8{ "rows", "keep" };
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const made = mlx.mlx_fast_metal_kernel_new("sushi_hy4_recalled_rows", iv, ov, rows_source, header, true, false);
        if (made.ctx == null) return error.MetalKernelCompileFailed;
        rows_kernel = made;
    }
    const sh = mlx.getShape(groups);
    const mv = [_]i32{ p0, ratio, n };
    const meta = try intsArr(&mv);
    defer _ = mlx.mlx_array_free(meta);
    const cfg = try cfgs.get(.{ .kind = .rows, .a = sh[0], .b = sh[1], .c = ratio }, build);
    var out: [2]Arr = undefined;
    try applyOuts(s, rows_kernel.?, &.{ groups, meta }, cfg, &out);
    return .{ .rows = out[0], .keep = out[1] };
}

fn applyOuts(s: mlx.mlx_stream, k: mlx.mlx_fast_metal_kernel, inputs: []const Arr, cfg: mlx.mlx_fast_metal_kernel_config, outs: []Arr) !void {
    const iv = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, k, iv, cfg, s));
    for (outs, 0..) |*o, i| {
        o.* = mlx.mlx_array_new();
        mlx.check(mlx.mlx_vector_array_get(o, ov, i)) catch |e| {
            for (outs[0 .. i + 1]) |prev| _ = mlx.mlx_array_free(prev);
            return e;
        };
    }
}

/// The attention output gate: `u ⊙ σ(gate)` with `u` [H, T, v] (any strides) and the gate's rows [T, H·v] as stored,
/// widened, written [T, H·v] as `W_o` reads them (the Cast, the Sigmoid, the Multiply and the transpose's copy).
const gate_source: [:0]const u8 =
    \\#pragma clang fp contract(off)
    \\uint i = thread_position_in_grid.x;
    \\uint t = thread_position_in_grid.y;
    \\uint hv = threads_per_grid.x;
    \\uint h = i / uint(V);
    \\uint c = i % uint(V);
    \\float g = static_cast<float>(gate[(size_t)t * hv + i]);
    \\float uv = u[(size_t)h * (size_t)u_strides[0] + (size_t)t * (size_t)u_strides[1] + (size_t)c * (size_t)u_strides[2]];
    \\out[(size_t)t * hv + i] = uv * hy4_sigmoid(g);
;

var gate_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// Whether `gated` takes `u` [H, T, v] f32 and `gate` [T, H·v] (BF16 or f32) on a GPU stream.
pub fn gatedApplies(s: mlx.mlx_stream, u: Arr, gate: Arr) bool {
    const us = mlx.getShape(u);
    const gs = mlx.getShape(gate);
    const gd = mlx.mlx_array_dtype(gate);
    return mlx.streamIsGpu(s) and us.len == 3 and gs.len == 2 and mlx.mlx_array_dtype(u) == .float32 and
        (gd == .bfloat16 or gd == .float32) and gs[0] == us[1] and gs[1] == us[0] * us[2];
}

/// `u ⊙ σ(gate)` as [T, H·v] f32. The caller frees the result.
pub fn gated(s: mlx.mlx_stream, u: Arr, gate: Arr) !Arr {
    if (gate_kernel == null) {
        const ins = [_][*:0]const u8{ "u", "gate" };
        const outs = [_][*:0]const u8{"out"};
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        // `u` is read through its strides: the value product's [H, T, v] as it stands.
        const made = mlx.mlx_fast_metal_kernel_new("sushi_hy4_gated", iv, ov, gate_source, header, false, false);
        if (made.ctx == null) return error.MetalKernelCompileFailed;
        gate_kernel = made;
    }
    const gs = mlx.getShape(gate);
    const cfg = try cfgs.get(.{ .kind = .gated, .a = gs[0], .b = gs[1], .c = mlx.getShape(u)[2] }, build);
    var out: [1]Arr = undefined;
    try applyOuts(s, gate_kernel.?, &.{ u, gate }, cfg, &out);
    return out[0];
}

const sink_softmax_tail: [:0]const u8 =
    \\threadgroup float local_max[32];
    \\threadgroup float local_normalizer[32];
    \\if (simd_group_id == 0) {
    \\  local_max[simd_lane_id] = -metal::numeric_limits<float>::infinity();
    \\  local_normalizer[simd_lane_id] = 0.0f;
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\float maxval = -metal::numeric_limits<float>::max();
    \\for (int i = 0; i < 4; i++) {
    \\  maxval = (maxval < ld[i]) ? ld[i] : maxval;
    \\}
    \\maxval = simd_max(maxval);
    \\if (simd_lane_id == 0) {
    \\  local_max[simd_group_id] = maxval;
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if (simd_group_id == 0) {
    \\  maxval = simd_max(local_max[simd_lane_id]);
    \\  if (simd_lane_id == 0) {
    \\    local_max[0] = maxval;
    \\  }
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\maxval = local_max[0];
    \\float normalizer = 0.0f;
    \\for (int i = 0; i < 4; i++) {
    \\  float exp_x = fast::exp(ld[i] - maxval);
    \\  ld[i] = exp_x;
    \\  normalizer += exp_x;
    \\}
    \\normalizer = simd_sum(normalizer);
    \\if (simd_lane_id == 0) {
    \\  local_normalizer[simd_group_id] = normalizer;
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if (simd_group_id == 0) {
    \\  normalizer = simd_sum(local_normalizer[simd_lane_id]);
    \\  if (simd_lane_id == 0) {
    \\    local_normalizer[0] = normalizer;
    \\  }
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\normalizer = 1.0f / local_normalizer[0];
    \\device float* dst = out + (size_t)gid * (size_t)N + (size_t)lid * 4;
    \\if (lid * 4 + 4 <= N) {
    \\  for (int i = 0; i < 4; i++) {
    \\    dst[i] = ld[i] * normalizer;
    \\  }
    \\} else {
    \\  for (int i = 0; i < 4; i++) {
    \\    if (lid * 4 + i < N) {
    \\      dst[i] = ld[i] * normalizer;
    \\    }
    \\  }
    \\}
;

const sink_softmax_causal_source: [:0]const u8 =
    \\#pragma clang fp contract(off)
    \\uint gid = threadgroup_position_in_grid.x;
    \\int lid = int(thread_index_in_threadgroup);
    \\uint simd_lane_id = thread_index_in_simdgroup;
    \\uint simd_group_id = simdgroup_index_in_threadgroup;
    \\int H = meta[0];
    \\int T = meta[1];
    \\int N = meta[2];
    \\int p0 = meta[3];
    \\int window = meta[4];
    \\float scale = as_type<float>(meta[5]);
    \\int h = HEADS_FIRST ? (int(gid) / T) : (int(gid) % H);
    \\int t = HEADS_FIRST ? (int(gid) % T) : (int(gid) / H);
    \\float rf = static_cast<float>(p0 + t);
    \\float wf = static_cast<float>(window);
    \\float ld[4];
    \\for (int i = 0; i < 4; i++) {
    \\  int col = lid * 4 + i;
    \\  if (col < N) {
    \\    size_t idx = (size_t)gid * (size_t)N + (size_t)col;
    \\    float s = a[idx] + b[idx];
    \\    float sc = s * scale;
    \\    float cf = static_cast<float>(col);
    \\    bool keep = (cf <= rf) && (window <= 0 || (cf > (rf - wf)));
    \\    float m = keep ? 0.0f : -metal::numeric_limits<float>::infinity();
    \\    ld[i] = sc + m;
    \\  } else if (col == N) {
    \\    ld[i] = static_cast<float>(sinks[h]);
    \\  } else {
    \\    ld[i] = -metal::numeric_limits<float>::infinity();
    \\  }
    \\}
++ sink_softmax_tail;

const sink_softmax_mask_source: [:0]const u8 =
    \\#pragma clang fp contract(off)
    \\uint gid = threadgroup_position_in_grid.x;
    \\int lid = int(thread_index_in_threadgroup);
    \\uint simd_lane_id = thread_index_in_simdgroup;
    \\uint simd_group_id = simdgroup_index_in_threadgroup;
    \\int H = meta[0];
    \\int T = meta[1];
    \\int N = meta[2];
    \\float scale = as_type<float>(meta[3]);
    \\int h = HEADS_FIRST ? (int(gid) / T) : (int(gid) % H);
    \\int t = HEADS_FIRST ? (int(gid) % T) : (int(gid) / H);
    \\float ld[4];
    \\for (int i = 0; i < 4; i++) {
    \\  int col = lid * 4 + i;
    \\  if (col < N) {
    \\    size_t idx = (size_t)gid * (size_t)N + (size_t)col;
    \\    float s = a[idx] + b[idx];
    \\    float sc = s * scale;
    \\    size_t midx = (size_t)t * (size_t)N + (size_t)col;
    \\    float m = MASK_BOOL ? (mask[midx] ? 0.0f : -metal::numeric_limits<float>::infinity()) : static_cast<float>(mask[midx]);
    \\    ld[i] = sc + m;
    \\  } else if (col == N) {
    \\    ld[i] = static_cast<float>(sinks[h]);
    \\  } else {
    \\    ld[i] = -metal::numeric_limits<float>::infinity();
    \\  }
    \\}
++ sink_softmax_tail;

var sink_sm_causal_kernel: ?mlx.mlx_fast_metal_kernel = null;
var sink_sm_mask_kernel: ?mlx.mlx_fast_metal_kernel = null;

pub fn sinkSoftmaxCausalApplies(s: mlx.mlx_stream, a: Arr, b: Arr, sinks: Arr, heads_first: bool) bool {
    const sa = mlx.getShape(a);
    const sb = mlx.getShape(b);
    const ss = mlx.getShape(sinks);
    const sdt = mlx.mlx_array_dtype(sinks);
    if (!mlx.streamIsGpu(s) or sa.len != 3 or !std.mem.eql(c_int, sa, sb) or ss.len != 1) return false;
    if (mlx.mlx_array_dtype(a) != .float32 or mlx.mlx_array_dtype(b) != .float32) return false;
    if (sdt != .float32 and sdt != .bfloat16) return false;
    const h = if (heads_first) sa[0] else sa[1];
    const t = if (heads_first) sa[1] else sa[0];
    const n = sa[2];
    if (h <= 0 or t <= 0 or n <= 0 or n >= 4096 or ss[0] != h) return false;
    return @as(i64, h) * @as(i64, t) * @as(i64, sinkSoftmaxGroup(n)) <= std.math.maxInt(c_int);
}

pub fn sinkSoftmaxMaskApplies(s: mlx.mlx_stream, a: Arr, b: Arr, mask: Arr, sinks: Arr, heads_first: bool) bool {
    if (!sinkSoftmaxCausalApplies(s, a, b, sinks, heads_first)) return false;
    const sa = mlx.getShape(a);
    const sm = mlx.getShape(mask);
    const mdt = mlx.mlx_array_dtype(mask);
    const t = if (heads_first) sa[1] else sa[0];
    const n = sa[2];
    return sm.len == 2 and sm[0] == t and sm[1] == n and (mdt == .bool_ or mdt == .float32);
}

pub fn sinkSoftmaxCausal(s: mlx.mlx_stream, a: Arr, b: Arr, sinks: Arr, scale: f32, p0: c_int, window: c_int, heads_first: bool) !Arr {
    const k = try getKernel(&sink_sm_causal_kernel, "sushi_hy4_sink_softmax_causal", &.{ "a", "b", "sinks", "meta" }, sink_softmax_causal_source);
    const sa = mlx.getShape(a);
    const h = if (heads_first) sa[0] else sa[1];
    const t = if (heads_first) sa[1] else sa[0];
    const n = sa[2];
    const mv = [_]i32{ h, t, n, p0, window, kcfg.bits(scale) };
    const meta = try intsArr(&mv);
    defer _ = mlx.mlx_array_free(meta);
    const cfg = try cfgs.get(.{ .kind = .sink_sm_causal, .a = h, .b = t, .c = n, .d = @intFromBool(heads_first) }, build);
    var out: [1]Arr = undefined;
    try applyOuts(s, k, &.{ a, b, sinks, meta }, cfg, &out);
    return out[0];
}

pub fn sinkSoftmaxMask(s: mlx.mlx_stream, a: Arr, b: Arr, mask: Arr, sinks: Arr, scale: f32, heads_first: bool) !Arr {
    const k = try getKernel(&sink_sm_mask_kernel, "sushi_hy4_sink_softmax_mask", &.{ "a", "b", "mask", "sinks", "meta" }, sink_softmax_mask_source);
    const sa = mlx.getShape(a);
    const h = if (heads_first) sa[0] else sa[1];
    const t = if (heads_first) sa[1] else sa[0];
    const n = sa[2];
    const mask_bool = mlx.mlx_array_dtype(mask) == .bool_;
    const mv = [_]i32{ h, t, n, kcfg.bits(scale) };
    const meta = try intsArr(&mv);
    defer _ = mlx.mlx_array_free(meta);
    const flags: c_int = @as(c_int, @intFromBool(heads_first)) | (@as(c_int, @intFromBool(mask_bool)) << 1);
    const cfg = try cfgs.get(.{ .kind = .sink_sm_mask, .a = h, .b = t, .c = n, .d = flags }, build);
    var out: [1]Arr = undefined;
    try applyOuts(s, k, &.{ a, b, mask, sinks, meta }, cfg, &out);
    return out[0];
}

const ring_keys_source: [:0]const u8 =
    \\uint col = thread_position_in_grid.x;
    \\uint i = thread_position_in_grid.y;
    \\uint r = thread_position_in_grid.z;
    \\int w = meta[1];
    \\int p0 = meta[2];
    \\int rp0 = meta[3];
    \\int lat = meta[4];
    \\int rope = meta[5];
    \\int lo = p0 + int(r) + 1 - w;
    \\int slot0 = (rp0 + int(r) + 1 - w) % w;
    \\if (slot0 < 0) slot0 += w;
    \\int src = lo + (int(i) < slot0 ? (w - slot0 + int(i)) : (int(i) - slot0));
    \\if (int(col) < lat) {
    \\  c_out[((size_t)r * (size_t)w + (size_t)i) * (size_t)lat + col] = static_cast<float>(c[(size_t)src * (size_t)lat + col]);
    \\} else {
    \\  uint pc = col - uint(lat);
    \\  pe_out[((size_t)r * (size_t)w + (size_t)i) * (size_t)rope + pc] = static_cast<float>(k_pe[(size_t)src * (size_t)rope + pc]);
    \\}
;

var ring_keys_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// Whether `ringKeys` builds a sliding block's windows: a GPU stream, f32 or bf16 latent and rope, the ring full
/// from the block's first row on (the caller decides the block is a verify block).
pub fn ringKeysApplies(s: mlx.mlx_stream, c: Arr, k_pe: Arr, t: c_int, p0: c_int, window: c_int, ring_p0: c_int) bool {
    const cs = mlx.getShape(c);
    const ps = mlx.getShape(k_pe);
    const cd = mlx.mlx_array_dtype(c);
    const pd = mlx.mlx_array_dtype(k_pe);
    return mlx.streamIsGpu(s) and cs.len == 2 and ps.len == 2 and cs[0] == ps[0] and cs[1] > 0 and ps[1] > 0 and
        (cd == .float32 or cd == .bfloat16) and (pd == .float32 or pd == .bfloat16) and
        t > 0 and window > 0 and ring_p0 + 1 > window and p0 + 1 >= window and p0 + t <= cs[0];
}

/// Each of a verify block's `t` rows' sliding window, widened to f32 in the order its one-row step reads the ring
/// (rotated at its own slot): [t, W, latent] and [t, W, rope], one launch for the block. The caller frees both.
pub fn ringKeys(s: mlx.mlx_stream, c: Arr, k_pe: Arr, t: c_int, p0: c_int, window: c_int, ring_p0: c_int) !struct { c: Arr, pe: Arr } {
    if (ring_keys_kernel == null) {
        const ins = [_][*:0]const u8{ "c", "k_pe", "meta" };
        const outs = [_][*:0]const u8{ "c_out", "pe_out" };
        const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_string_free(iv);
        const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
        defer _ = mlx.mlx_vector_string_free(ov);
        const made = mlx.mlx_fast_metal_kernel_new("sushi_hy4_ring_keys", iv, ov, ring_keys_source, header, true, false);
        if (made.ctx == null) return error.MetalKernelCompileFailed;
        ring_keys_kernel = made;
    }
    const lat = mlx.getShape(c)[1];
    const rope = mlx.getShape(k_pe)[1];
    const mv = [_]i32{ t, window, p0, ring_p0, lat, rope };
    const meta = try intsArr(&mv);
    defer _ = mlx.mlx_array_free(meta);
    const cfg = try cfgs.get(.{ .kind = .ring_keys, .a = t, .b = window, .c = lat, .d = rope }, build);
    var out: [2]Arr = undefined;
    try applyOuts(s, ring_keys_kernel.?, &.{ c, k_pe, meta }, cfg, &out);
    return .{ .c = out[0], .pe = out[1] };
}

// ── tests ────────────────────────────────────────────────────────────────────────────────────────

const testing = std.testing;
const Ops = @import("../mlx_ops.zig").Ops;

fn hostF32(o: *Ops, a: Arr) ![]const u8 {
    const c = try o.contiguous(a);
    try mlx.check(mlx.mlx_array_eval(c));
    return std.mem.sliceAsBytes((mlx.mlx_array_data_float32(c) orelse return error.HostRead)[0..mlx.mlx_array_size(c)]);
}

fn randomF32(o: *Ops, shape: []const c_int, seed: u64, scale: f64) !Arr {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const v = try testing.allocator.alloc(f32, n);
    defer testing.allocator.free(v);
    var r = std.Random.DefaultPrng.init(seed);
    for (v) |*d| d.* = @floatCast(r.random().floatNorm(f64) * scale);
    // Exact zeros, both signs, and values past the sigmoid's saturation.
    if (n > 4) {
        v[0] = 0;
        v[1] = -0.0;
        v[2] = 120;
        v[3] = -120;
    }
    return o.own(mlx.mlx_array_new_data(v.ptr, shape.ptr, @intCast(shape.len), .float32));
}

test "hy4 chain: the SwiGLU in one launch gives the bytes of Ops.silu then the multiply, f32, at the shared expert's and the dense FFN's widths" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    for ([_][2]c_int{ .{ 1, 1536 }, .{ 1, 13824 }, .{ 7, 1536 }, .{ 1, 256 }, .{ 3, 128 } }, 0..) |sh, i| {
        var o: Ops = .{ .s = es };
        defer o.deinit();
        const g = try randomF32(&o, &sh, 2 * i + 1, 4);
        const u = try randomF32(&o, &sh, 2 * i + 2, 3);
        const want = try hostF32(&o, try o.binary(.mul, try o.silu(g), u));
        try testing.expect(swigluApplies(es, g, u));
        const before = mlx.engine.stats().dispatches;
        const got = try o.own(try swiglu(es, g, u));
        try mlx.check(mlx.mlx_array_eval(got));
        try testing.expectEqual(@as(u64, 1), mlx.engine.stats().dispatches - before);
        try testing.expectEqualSlices(u8, want, try hostF32(&o, got));
    }
}

fn hostAny(o: *Ops, a: Arr) ![]const u8 {
    const c = try o.contiguous(a);
    try mlx.check(mlx.mlx_array_eval(c));
    const n = mlx.mlx_array_size(c);
    return switch (mlx.mlx_array_dtype(c)) {
        .float32 => std.mem.sliceAsBytes((mlx.mlx_array_data_float32(c) orelse return error.HostRead)[0..n]),
        .uint32 => std.mem.sliceAsBytes((mlx.mlx_array_data_uint32(c) orelse return error.HostRead)[0..n]),
        .bool_ => std.mem.sliceAsBytes((mlx.mlx_array_data_bool(c) orelse return error.HostRead)[0..n]),
        else => error.UnexpectedDtype,
    };
}

fn dispatchesOf(comptime f: anytype, args: anytype) !struct { u64, @typeInfo(@typeInfo(@TypeOf(f)).@"fn".return_type.?).error_union.payload } {
    const before = mlx.engine.stats().dispatches;
    const r = try @call(.auto, f, args);
    return .{ mlx.engine.stats().dispatches - before, r };
}

test "hy4 chain: the indexer's groups and recalled rows in one launch each give the bytes of the ops, at the release's 511 picks of groups of 4 and the tiny's, decode and verify rows, near the context's start and far out" {
    try mlx.requireGpu();
    const fwd = @import("hy4_forward.zig");
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    defer fwd.chain_kernels_on = true;
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const Case = struct { t: c_int, take: c_int, ratio: c_int, p0: c_int };
    for ([_]Case{
        .{ .t = 1, .take = 511, .ratio = 4, .p0 = 4096 },
        .{ .t = 6, .take = 511, .ratio = 4, .p0 = 1_000_003 },
        .{ .t = 1, .take = 3, .ratio = 4, .p0 = 600 },
        .{ .t = 3, .take = 3, .ratio = 4, .p0 = 17 },
    }) |c| {
        var o: Ops = .{ .s = es };
        defer o.deinit();
        const n = c.p0 + c.t;
        const groups_seen = @divTrunc(n, c.ratio);
        const v = try testing.allocator.alloc(i32, @intCast(c.t * c.take));
        defer testing.allocator.free(v);
        var r = std.Random.DefaultPrng.init(@intCast(c.p0));
        // Picks anywhere in the pooled groups, the row's own and later ones among them (a row with few earlier groups).
        for (v) |*d| d.* = r.random().intRangeLessThan(i32, 0, @max(1, groups_seen + 1));
        const order = try o.own(mlx.mlx_array_new_data(v.ptr, &[_]c_int{ c.t, c.take }, 2, .int32));
        const own_v = try testing.allocator.alloc(f32, @intCast(c.t));
        defer testing.allocator.free(own_v);
        for (own_v, 0..) |*d, i| d.* = @floatFromInt(c.p0 + @as(c_int, @intCast(i)));
        const pos = try o.own(mlx.mlx_array_new_data(own_v.ptr, &[_]c_int{c.t}, 1, .float32));
        const fd = try o.slot();
        try mlx.check(mlx.mlx_floor_divide(fd, pos, try o.scalar(@floatFromInt(c.ratio), .float32), es));
        const want_groups = try o.concat(&.{ try o.cast(order, .float32), try o.reshape(fd.*, &.{ c.t, 1 }) }, 1);
        const want_g = try hostAny(&o, want_groups);
        try mlx.check(mlx.mlx_array_eval(order));
        const ng, const groups = try dispatchesOf(recallGroups, .{ es, order, c.p0, c.ratio });
        _ = try o.own(groups);
        try mlx.check(mlx.mlx_array_eval(groups));
        try testing.expectEqual(@as(u64, 1), ng);
        try testing.expectEqualSlices(u8, want_g, try hostAny(&o, groups));

        fwd.chain_kernels_on = false;
        const want = try fwd.recalledRows(&o, c.t, n, c.p0, groups, c.ratio);
        const want_rows = try hostAny(&o, want.rows);
        const want_keep = try hostAny(&o, want.keep);
        fwd.chain_kernels_on = true;
        const nr, const got = try dispatchesOf(recalledRows, .{ es, groups, c.p0, n, c.ratio });
        _ = try o.own(got.rows);
        _ = try o.own(got.keep);
        try mlx.check(mlx.mlx_array_eval(got.keep));
        try testing.expectEqual(@as(u64, 1), nr);
        try testing.expectEqualSlices(u8, want_rows, try hostAny(&o, got.rows));
        try testing.expectEqualSlices(u8, want_keep, try hostAny(&o, got.keep));
    }
}

test "hy4 chain: the attention output gate in one launch gives the bytes of the cast, the Sigmoid, the Multiply and the transpose, BF16 and f32 gate rows, the release's 64 and 32 heads of 256, a decode row and a block" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    for ([_][3]c_int{ .{ 64, 1, 256 }, .{ 32, 1, 256 }, .{ 64, 6, 256 }, .{ 8, 3, 32 } }, 0..) |sh, i| for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |gdt| {
        var o: Ops = .{ .s = es };
        defer o.deinit();
        const h = sh[0];
        const t = sh[1];
        const v = sh[2];
        // `u` as the value product leaves it: [T, H, v] seen as [H, T, v].
        const u = try o.transpose(try randomF32(&o, &.{ t, h, v }, 3 * i + 1, 2), &.{ 1, 0, 2 });
        const gate = try o.cast(try randomF32(&o, &.{ t, h * v }, 3 * i + 2, 3), gdt);
        const g32 = try o.unary(.sigmoid, try o.transpose(try o.reshape(try o.cast(gate, .float32), &.{ t, h, v }), &.{ 1, 0, 2 }));
        const want = try hostF32(&o, try o.reshape(try o.transpose(try o.binary(.mul, u, g32), &.{ 1, 0, 2 }), &.{ t, h * v }));
        try testing.expect(gatedApplies(es, u, gate));
        try mlx.check(mlx.mlx_array_eval(u));
        try mlx.check(mlx.mlx_array_eval(gate));
        const before = mlx.engine.stats().dispatches;
        const got = try o.own(try gated(es, u, gate));
        try mlx.check(mlx.mlx_array_eval(got));
        try testing.expectEqual(@as(u64, 1), mlx.engine.stats().dispatches - before);
        try testing.expectEqualSlices(u8, want, try hostF32(&o, got));
    };
}

test "hy4 chain: attention score add, scale, mask, sink concat and softmax in one launch gives the op chain's bytes at the release's sliding and gathered shapes, decode and verify blocks, causal, bool keep and additive masks" {
    try mlx.requireGpu();
    const fwd = @import("hy4_forward.zig");
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();

    const Case = struct { h: c_int, t: c_int, s: c_int, heads_first: bool, scale: f32 };
    const cases = [_]Case{
        .{ .h = 64, .t = 1, .s = 128, .heads_first = true, .scale = 1.0 / @sqrt(@as(f32, 192.0)) },
        .{ .h = 64, .t = 1, .s = 512, .heads_first = true, .scale = 0.0625 },
        .{ .h = 64, .t = 6, .s = 512, .heads_first = true, .scale = 0.0625 },
        .{ .h = 32, .t = 1, .s = 128, .heads_first = false, .scale = 1.0 / @sqrt(@as(f32, 192.0)) },
        .{ .h = 32, .t = 1, .s = 512, .heads_first = false, .scale = 0.0625 },
        .{ .h = 32, .t = 1, .s = 2048, .heads_first = false, .scale = 0.0625 },
        .{ .h = 32, .t = 6, .s = 2048, .heads_first = false, .scale = 0.0625 },
    };

    for (cases, 0..) |c, ci| {
        var o: Ops = .{ .s = es };
        defer o.deinit();
        const sh = if (c.heads_first) [3]c_int{ c.h, c.t, c.s } else [3]c_int{ c.t, c.h, c.s };
        const a = try randomF32(&o, &sh, 100 * ci + 1, 5.0);
        const b = try randomF32(&o, &sh, 100 * ci + 2, 5.0);
        const sinks = try randomF32(&o, &.{c.h}, 100 * ci + 3, 2.0);
        const sink_sh: [3]c_int = if (c.heads_first) .{ c.h, 1, 1 } else .{ 1, c.h, 1 };
        const sink_full: [3]c_int = if (c.heads_first) .{ c.h, c.t, 1 } else .{ c.t, c.h, 1 };
        const sink_col = try o.broadcast(try o.reshape(sinks, &sink_sh), &sink_full);

        const kn: usize = @intCast(c.t * c.s);
        const kv = try testing.allocator.alloc(bool, kn);
        defer testing.allocator.free(kv);
        var prng = std.Random.DefaultPrng.init(100 * ci + 4);
        for (0..@intCast(c.t)) |row| {
            for (0..@intCast(c.s)) |col| {
                kv[row * @as(usize, @intCast(c.s)) + col] = if (c.t > 1 and row == 0)
                    false
                else if (c.t > 1 and row == 1)
                    true
                else
                    prng.random().boolean();
            }
        }
        const keep = try o.own(mlx.mlx_array_new_data(kv.ptr, &[_]c_int{ c.t, c.s }, 2, .bool_));
        const add_m = try o.slot();
        try mlx.check(mlx.mlx_where(add_m, keep, try o.scalar(0, .float32), try o.scalar(-std.math.inf(f32), .float32), es));
        const m_bcast = if (c.heads_first) add_m.* else try o.reshape(add_m.*, &.{ c.t, 1, c.s });
        const want_m_arr = try o.slice(try o.softmax(try o.concat(&.{ try o.binary(.add, try o.binary(.mul, try o.binary(.add, a, b), try o.scalar(c.scale, .float32)), m_bcast), sink_col }, 2), 2), 2, 0, c.s);
        const want_m = try hostF32(&o, want_m_arr);

        for ([_]Arr{ a, b, sinks, keep, add_m.* }) |arr| try mlx.check(mlx.mlx_array_eval(arr));

        try testing.expect(sinkSoftmaxMaskApplies(es, a, b, keep, sinks, c.heads_first));
        const d_bool, const got_bool = try dispatchesOf(sinkSoftmaxMask, .{ es, a, b, keep, sinks, c.scale, c.heads_first });
        _ = try o.own(got_bool);
        try mlx.check(mlx.mlx_array_eval(got_bool));
        try testing.expectEqual(@as(u64, 1), d_bool);
        try testing.expectEqualSlices(u8, want_m, try hostF32(&o, got_bool));

        try testing.expect(sinkSoftmaxMaskApplies(es, a, b, add_m.*, sinks, c.heads_first));
        const d_f32, const got_f32 = try dispatchesOf(sinkSoftmaxMask, .{ es, a, b, add_m.*, sinks, c.scale, c.heads_first });
        _ = try o.own(got_f32);
        try mlx.check(mlx.mlx_array_eval(got_f32));
        try testing.expectEqual(@as(u64, 1), d_f32);
        try testing.expectEqualSlices(u8, want_m, try hostF32(&o, got_f32));

        if (c.heads_first) {
            const p0: c_int = if (c.t == 1) c.s - 1 else c.s - c.t;
            const win: c_int = if (c.t == 1) 0 else @divTrunc(c.s, 2);
            const cm = try fwd.causalMask(&o, c.t, c.s, p0, win);
            const want_c_arr = try o.slice(try o.softmax(try o.concat(&.{ try o.binary(.add, try o.binary(.mul, try o.binary(.add, a, b), try o.scalar(c.scale, .float32)), cm), sink_col }, 2), 2), 2, 0, c.s);
            const want_c = try hostF32(&o, want_c_arr);
            try testing.expect(sinkSoftmaxCausalApplies(es, a, b, sinks, true));
            const d_c, const got_c = try dispatchesOf(sinkSoftmaxCausal, .{ es, a, b, sinks, c.scale, p0, win, true });
            _ = try o.own(got_c);
            try mlx.check(mlx.mlx_array_eval(got_c));
            try testing.expectEqual(@as(u64, 1), d_c);
            try testing.expectEqualSlices(u8, want_c, try hostF32(&o, got_c));
        }
    }
}
