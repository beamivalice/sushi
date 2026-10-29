const trf = @import("transformer.zig");
const std = @import("std");
const imatrix_capture = @import("imatrix.zig");
const sushi_exl3 = @import("sushi_exl3");
const expert_exl3 = sushi_exl3.format;
const mlx = @import("mlx.zig");
const test_support = @import("test_support.zig");
const attn256RandBf16 = test_support.attn256RandBf16;
const model_mod = @import("model.zig");
const log = @import("log.zig");
const ModelConfig = model_mod.ModelConfig;
const QuantMode = model_mod.QuantMode;
const Weights = model_mod.Weights;
const testing = std.testing;
const KVCache = trf.KVCache;
const Transformer = trf.Transformer;
const seFillBf16 = trf.seFillBf16;

pub const expert_exl3_kernels = sushi_exl3.kernels;

const expert_quant_mod = @import("expert_quant.zig");

/// `std.meta.fields` was replaced by parallel `fieldNames`/`fieldTypes`
/// arrays — zip them into the `.name`/`.type` shape every call site below
/// already expects.
const WeightField = struct { name: [:0]const u8, type: type };

fn structFields(comptime T: type) []const WeightField {
    return comptime blk: {
        const names = std.meta.fieldNames(T);
        const types = std.meta.fieldTypes(T);
        var result: [names.len]WeightField = undefined;
        for (names, types, 0..) |name, ty, i| result[i] = .{ .name = name, .type = ty };
        const final = result;
        break :blk &final;
    };
}

pub const FullAttnWeights = struct {
    q_w: mlx.mlx_array,
    q_s: mlx.mlx_array,
    q_b: mlx.mlx_array,
    k_w: mlx.mlx_array,
    k_s: mlx.mlx_array,
    k_b: mlx.mlx_array,
    v_w: mlx.mlx_array,
    v_s: mlx.mlx_array,
    v_b: mlx.mlx_array,
    o_w: mlx.mlx_array,
    o_s: mlx.mlx_array,
    o_b: mlx.mlx_array,
    q_norm: mlx.mlx_array,
    k_norm: mlx.mlx_array,
    // qwen4_exp QSA indexer: one projection for the index queries + the
    // single shared key head, and their per-head RMS norms. Null-ctx on
    // every other arch.
    idx_qk_w: mlx.mlx_array = .{ .ctx = null },
    idx_qk_s: mlx.mlx_array = .{ .ctx = null },
    idx_qk_b: mlx.mlx_array = .{ .ctx = null },
    idx_q_norm: mlx.mlx_array = .{ .ctx = null },
    idx_k_norm: mlx.mlx_array = .{ .ctx = null },
    // MiMo: learned per-head attention sinks (`self_attn.attention_sink_bias`,
    // [n_heads]). One extra column in the softmax DENOMINATOR, letting a head
    // attend to "nothing". Null-ctx on every other arch.
    sinks: mlx.mlx_array = .{ .ctx = null },
};

/// qwen4_exp gated residual ("hyper connection"). `norm_w` is the grouped
/// hc_norm weight viewed as [hc, hidden]; `inject_*` is absent on the final
/// mixer (`use_combine=False`).
pub const HcWeights = struct {
    norm_w: mlx.mlx_array,
    down_w: mlx.mlx_array,
    down_s: mlx.mlx_array,
    down_b: mlx.mlx_array,
    up_w: mlx.mlx_array,
    up_s: mlx.mlx_array,
    up_b: mlx.mlx_array,
    inject_w: mlx.mlx_array = .{ .ctx = null },
    inject_s: mlx.mlx_array = .{ .ctx = null },
    inject_b: mlx.mlx_array = .{ .ctx = null },
    /// Dense inject weight materialized row-major `[hc*hidden, hc]` for the
    /// fused decode kernel (the qmatmul path reads the lazy transpose view).
    inject_flat: mlx.mlx_array = .{ .ctx = null },
};

/// qwen4_exp PLE: n-gram embedding → per-stream keys + one value, gated by
/// the normalized streams, plus a dilated depthwise short conv.
pub const PleWeights = struct {
    key_w: mlx.mlx_array,
    key_s: mlx.mlx_array,
    key_b: mlx.mlx_array,
    value_w: mlx.mlx_array,
    value_s: mlx.mlx_array,
    value_b: mlx.mlx_array,
    norm_key: mlx.mlx_array, // [hc, hidden]
    norm_query: mlx.mlx_array,
    norm_conv: mlx.mlx_array,
    conv_w: mlx.mlx_array, // [hc*hidden, K, 1]
};


pub const LinearAttnWeights = struct {
    // Separate projections: qkv=QKV, z=Z, a=A, b=B
    qkv_w: mlx.mlx_array,
    qkv_s: mlx.mlx_array,
    qkv_b: mlx.mlx_array,
    z_w: mlx.mlx_array,
    z_s: mlx.mlx_array,
    z_b: mlx.mlx_array,
    a_w: mlx.mlx_array,
    a_s: mlx.mlx_array,
    a_b: mlx.mlx_array,
    b_w: mlx.mlx_array,
    b_s: mlx.mlx_array,
    b_b: mlx.mlx_array,
    conv1d_w: mlx.mlx_array,
    A_log: mlx.mlx_array,
    dt_bias: mlx.mlx_array,
    norm_w: mlx.mlx_array,
    out_w: mlx.mlx_array,
    out_s: mlx.mlx_array,
    out_b: mlx.mlx_array,
};

pub const DenseMlpWeights = struct {
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

pub const MoeMlpWeights = struct {
    /// A rate-grouped EXL3 layer's banks; its handles are the transformer's to free.
    exl3_ragged: ?sushi_exl3.Ragged = null,
    router_w: mlx.mlx_array,
    router_s: mlx.mlx_array,
    router_b: mlx.mlx_array,
    switch_gate_w: mlx.mlx_array,
    switch_gate_s: mlx.mlx_array,
    switch_gate_b: mlx.mlx_array,
    switch_up_w: mlx.mlx_array,
    switch_up_s: mlx.mlx_array,
    switch_up_b: mlx.mlx_array,
    switch_down_w: mlx.mlx_array,
    switch_down_s: mlx.mlx_array,
    switch_down_b: mlx.mlx_array,
    shared_gate_w: mlx.mlx_array,
    shared_gate_s: mlx.mlx_array,
    shared_gate_b: mlx.mlx_array,
    shared_up_w: mlx.mlx_array,
    shared_up_s: mlx.mlx_array,
    shared_up_b: mlx.mlx_array,
    shared_down_w: mlx.mlx_array,
    shared_down_s: mlx.mlx_array,
    shared_down_b: mlx.mlx_array,
    // The dense shared-expert gate as one contiguous [hidden] vector (null ctx otherwise), for the router kernel's dot.
    shared_gate_vec: mlx.mlx_array = .{ .ctx = null },
    // Shared expert gating (qwen4_exp; null for MiMo)
    shared_expert_gate_w: ?mlx.mlx_array = null,
    shared_expert_gate_s: ?mlx.mlx_array = null,
    shared_expert_gate_b: ?mlx.mlx_array = null,
    // MiMo sigmoid routing: f32 [num_experts] bias added to the sigmoid scores for
    // top-k SELECTION only (weights come from the unbiased scores). Non-null
    // expert_bias routes moeMLP2 through the sigmoid chain.
    expert_bias: ?mlx.mlx_array = null,
    route_norm: bool = true,
    route_scale: f32 = 1.0,
};

const HybridMlpWeights = union(enum) {
    dense: DenseMlpWeights,
    moe: MoeMlpWeights,
};

pub const MoeLayerWeights = struct {
    input_norm: mlx.mlx_array,
    post_attn_norm: mlx.mlx_array,
    is_linear: bool,
    attn: union(enum) { full: FullAttnWeights, linear: LinearAttnWeights },
    mlp: HybridMlpWeights,
    // qwen4_exp: the two gated residual reads/writes around the block and
    // the n-gram PLE on its one layer. Null for every other arch.
    hc_attn: ?HcWeights = null,
    hc_mlp: ?HcWeights = null,
    ple: ?PleWeights = null,
};


// ── Quantization params cache ──
// A tiny lock-free pointer → (bits, group_size) cache. Quantized weights are loaded
// once at init and reused for every forward pass; we detect (or pre-bind) params on
// first touch and serve hits thereafter. Uses open addressing with linear probing on
// a fixed-size array — this fits in L1 and keeps the cost of a lookup to ~5ns,
// matching the perf commit's intent of "eliminate per-call detect overhead" while
// supporting mixed-precision quantization (per-layer bits).
pub const BITS_CACHE_CAP: usize = 1024; // plenty for 60 layers × ~10 quant weights × factor
pub const QuantParams = struct { bits: u32, group_size: u32, mode: QuantMode = .affine };

const QuantParamsCache = struct {
    keys: [BITS_CACHE_CAP]?*anyopaque = @splat(null),
    vals_bits: [BITS_CACHE_CAP]u8 = @splat(0),
    // group_size is always a small power of two in MLX (16, 32, 64, 128).
    // Store as group_size/8 to fit u8 with headroom up to 2040.
    vals_gs_div8: [BITS_CACHE_CAP]u8 = @splat(0),
    vals_mode: [BITS_CACHE_CAP]u8 = @splat(0),

    pub inline fn slot(key: *anyopaque) usize {
        const h: usize = @intFromPtr(key);
        // Golden-ratio multiplier for quick hash on pointer values (high bits).
        return (h *% 0x9E3779B97F4A7C15) >> @as(u6, @intCast(@bitSizeOf(usize) - 10));
    }

    /// Insert params for `key`. Returns false if the probe window saturated
    /// (caller should treat as "fall through to detection" — but realistically
    /// never fires given BITS_CACHE_CAP is 1024 vs ~2k weights max).
    pub fn put(self: *QuantParamsCache, key: *anyopaque, qp: QuantParams) bool {
        const start = slot(key);
        var i: usize = 0;
        while (i < 4) : (i += 1) {
            const idx = (start + i) & (BITS_CACHE_CAP - 1);
            if (self.keys[idx] == null or self.keys[idx] == key) {
                self.keys[idx] = key;
                self.vals_bits[idx] = @intCast(qp.bits);
                self.vals_gs_div8[idx] = @intCast(qp.group_size / 8);
                self.vals_mode[idx] = @backingInt(qp.mode);
                return true;
            }
        }
        return false;
    }
};

// Backwards-compatibility alias — keeps existing field names readable.
pub const BitsCache = QuantParamsCache;

/// Pre-transpose a plain-BF16 weight stored as `[out, in]` to `[in, out]` so
/// `mlx_matmul(x, w_t)` lands the contraction over the input axis. Used by
/// Unsloth Dynamic checkpoints that leave linear-attention projections
/// unquantized while quantizing the rest. Caller owns the returned array.
pub fn transposeBf16Weight(w: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    // Swap the last two axes for any rank: 2D weights [out, in] → [in, out];
    // stacked MoE expert tensors [experts, out, in] → [experts, in, out].
    const ndim = mlx.getShape(w).len;
    var perm: [8]c_int = undefined; // mlx arrays never exceed 8 dims here
    for (0..ndim) |i| perm[i] = @intCast(i);
    perm[ndim - 2] = @intCast(ndim - 1);
    perm[ndim - 1] = @intCast(ndim - 2);
    var w_t = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_transpose_axes(&w_t, w, &perm, @intCast(ndim), s));
    return w_t;
}

/// In-place: if `*sc` is null-ctx, we treat the matching `*w` as plain bf16.
/// Replace `*w` with its pre-transposed `[in, out]` form and record the new
/// array in `owned` so we can free it on Transformer.deinit.
pub fn maybeTransposeForBf16(
    w: *mlx.mlx_array,
    sc: mlx.mlx_array,
    owned: *std.ArrayList(mlx.mlx_array),
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,
) !void {
    if (sc.ctx != null) return; // quantized weight — leave as-is
    if (w.ctx == null) return; // absent weight (e.g. KV-shared layer) — nothing to transpose
    const transposed = try transposeBf16Weight(w.*, s);
    w.* = transposed;
    try owned.append(allocator, transposed);
}

/// One quantized-or-dense projection triple by full tensor base name.
pub fn weightTriple(weights: *const Weights, base: []const u8, owned_bf16: *std.ArrayList(mlx.mlx_array), allocator: std.mem.Allocator, s: mlx.mlx_stream) !struct { w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array } {
    var nb: [160]u8 = undefined;
    const w = weights.get(std.fmt.bufPrint(&nb, "{s}.weight", .{base}) catch unreachable) orelse {
        log.err("MISSING WEIGHT: {s}.weight\n", .{base});
        return error.MissingWeight;
    };
    const sc = weights.get(std.fmt.bufPrint(&nb, "{s}.scales", .{base}) catch unreachable) orelse mlx.mlx_array_new();
    const bi = weights.get(std.fmt.bufPrint(&nb, "{s}.biases", .{base}) catch unreachable) orelse mlx.mlx_array_new();
    var wm = w;
    try maybeTransposeForBf16(&wm, sc, owned_bf16, allocator, s);
    return .{ .w = wm, .sc = sc, .bi = bi };
}

/// A grouped-norm weight `[hc*hidden]` viewed as `[hc, hidden]` (owned).
fn groupedNormView(weights: *const Weights, name: []const u8, hc: u32, hidden: u32, owned_bf16: *std.ArrayList(mlx.mlx_array), allocator: std.mem.Allocator, s: mlx.mlx_stream) !mlx.mlx_array {
    const raw = weights.get(name) orelse {
        log.err("MISSING WEIGHT: {s}\n", .{name});
        return error.MissingWeight;
    };
    const shape = [_]c_int{ @intCast(hc), @intCast(hidden) };
    var view = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&view, raw, &shape, 2, s));
    try owned_bf16.append(allocator, view);
    return view;
}

pub fn loadHcWeights(weights: *const Weights, base: []const u8, with_inject: bool, hc: u32, hidden: u32, owned_bf16: *std.ArrayList(mlx.mlx_array), allocator: std.mem.Allocator, s: mlx.mlx_stream) !HcWeights {
    var nb: [160]u8 = undefined;
    const down = try weightTriple(weights, std.fmt.bufPrint(&nb, "{s}.input_mix_weight_down", .{base}) catch unreachable, owned_bf16, allocator, s);
    const up = try weightTriple(weights, std.fmt.bufPrint(&nb, "{s}.input_mix_weight_up", .{base}) catch unreachable, owned_bf16, allocator, s);
    var out: HcWeights = .{
        .norm_w = try groupedNormView(weights, std.fmt.bufPrint(&nb, "{s}.hc_norm.weight", .{base}) catch unreachable, hc, hidden, owned_bf16, allocator, s),
        .down_w = down.w,
        .down_s = down.sc,
        .down_b = down.bi,
        .up_w = up.w,
        .up_s = up.sc,
        .up_b = up.bi,
    };
    // The reference divides the down and inject projections by hc_count;
    // fold that into the (de)quantization affine or the dense weight.
    const inv_hc = mlx.mlx_array_new_float(1.0 / @as(f32, @floatFromInt(hc)));
    defer _ = mlx.mlx_array_free(inv_hc);
    try scaleTriple(&out.down_w, &out.down_s, &out.down_b, inv_hc, owned_bf16, allocator, s);
    if (with_inject) {
        const inj = try weightTriple(weights, std.fmt.bufPrint(&nb, "{s}.block_inject_weight", .{base}) catch unreachable, owned_bf16, allocator, s);
        out.inject_w = inj.w;
        out.inject_s = inj.sc;
        out.inject_b = inj.bi;
        try scaleTriple(&out.inject_w, &out.inject_s, &out.inject_b, inv_hc, owned_bf16, allocator, s);
        if (out.inject_s.ctx == null) {
            var flat = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_contiguous(&flat, out.inject_w, false, s));
            try mlx.check(mlx.mlx_array_eval(flat));
            try owned_bf16.append(allocator, flat);
            out.inject_flat = flat;
        }
    }
    return out;
}

/// Multiply a projection by a scalar: scales+biases of an affine weight, or
/// the dense weight itself. Replacement arrays are owned (and evaluated).
fn scaleTriple(w: *mlx.mlx_array, sc: *mlx.mlx_array, bi: *mlx.mlx_array, k: mlx.mlx_array, owned_bf16: *std.ArrayList(mlx.mlx_array), allocator: std.mem.Allocator, s: mlx.mlx_stream) !void {
    // The scalar must carry the TARGET's dtype: an f32 scalar promotes the
    // bf16 weight/scales, and a widened inject gate then widened the whole
    // residual stream to f32 (1/hc is exact in bf16 for hc <= 8).
    const Scaled = struct {
        fn f(a: mlx.mlx_array, k_raw: mlx.mlx_array, st: mlx.mlx_stream) !mlx.mlx_array {
            var kt = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(kt);
            try mlx.check(mlx.mlx_astype(&kt, k_raw, mlx.mlx_array_dtype(a), st));
            var out = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_multiply(&out, a, kt, st));
            try mlx.check(mlx.mlx_array_eval(out));
            return out;
        }
    }.f;
    if (sc.ctx != null) {
        const s2 = try Scaled(sc.*, k, s);
        try owned_bf16.append(allocator, s2);
        sc.* = s2;
        if (bi.ctx != null) {
            const b2 = try Scaled(bi.*, k, s);
            try owned_bf16.append(allocator, b2);
            bi.* = b2;
        }
    } else {
        const w2 = try Scaled(w.*, k, s);
        try owned_bf16.append(allocator, w2);
        w.* = w2;
    }
}

pub fn loadPleWeights(weights: *const Weights, name_buf: *[256]u8, prefix: []const u8, li: u32, hc: u32, hidden: u32, owned_bf16: *std.ArrayList(mlx.mlx_array), allocator: std.mem.Allocator, s: mlx.mlx_stream) !PleWeights {
    var nb: [160]u8 = undefined;
    const key = try weightTriple(weights, std.fmt.bufPrint(&nb, "{s}.layers.{d}.ple.key_proj", .{ prefix, li }) catch unreachable, owned_bf16, allocator, s);
    const value = try weightTriple(weights, std.fmt.bufPrint(&nb, "{s}.layers.{d}.ple.value_proj", .{ prefix, li }) catch unreachable, owned_bf16, allocator, s);
    return .{
        .key_w = key.w,
        .key_s = key.sc,
        .key_b = key.bi,
        .value_w = value.w,
        .value_s = value.sc,
        .value_b = value.bi,
        .norm_key = try groupedNormView(weights, std.fmt.bufPrint(&nb, "{s}.layers.{d}.ple.norm_key.weight", .{ prefix, li }) catch unreachable, hc, hidden, owned_bf16, allocator, s),
        .norm_query = try groupedNormView(weights, std.fmt.bufPrint(&nb, "{s}.layers.{d}.ple.norm_query.weight", .{ prefix, li }) catch unreachable, hc, hidden, owned_bf16, allocator, s),
        .norm_conv = try groupedNormView(weights, std.fmt.bufPrint(&nb, "{s}.layers.{d}.ple.norm_conv.weight", .{ prefix, li }) catch unreachable, hc, hidden, owned_bf16, allocator, s),
        .conv_w = try getLayerWeight(weights, name_buf, prefix, li, "ple.conv1d.weight"),
    };
}

pub fn shouldStreamExpertBank(config: *const ModelConfig, prefix: []const u8) bool {
    return config.expert_streaming and expert_quant_mod.isExpertStreamingArch(config.model_type) and
        std.mem.eql(u8, prefix, config.weight_prefix);
}

/// Append every non-null mlx_array field of a plain weight struct.
pub fn appendStructArrays(vec: mlx.mlx_vector_array, w: anytype) void {
    inline for (comptime structFields(@TypeOf(w.*))) |field| {
        if (comptime field.type != mlx.mlx_array) continue;
        const arr = @field(w, field.name);
        if (arr.ctx != null) _ = mlx.mlx_vector_array_append_value(vec, arr);
    }
}

pub fn appendFullAttnWeights(vec: mlx.mlx_vector_array, fa: *const FullAttnWeights) void {
    inline for (comptime structFields(FullAttnWeights)) |field| {
        // Dense bf16 full-attn layers carry null-ctx scales/biases — skip those
        // so null arrays don't poison the eval batch. Mirrors the linear/mlp paths.
        const arr = @field(fa, field.name);
        if (arr.ctx != null) _ = mlx.mlx_vector_array_append_value(vec, arr);
    }
}

pub fn appendLinearAttnWeights(vec: mlx.mlx_vector_array, la: *const LinearAttnWeights) void {
    inline for (comptime structFields(LinearAttnWeights)) |field| {
        if (comptime field.type != mlx.mlx_array) continue;
        {
            const arr = @field(la, field.name);
            // Plain-bf16 layers (Unsloth Dynamic) carry null scales/biases — skip those.
            if (arr.ctx != null) {
                _ = mlx.mlx_vector_array_append_value(vec, arr);
            }
        }
    }
}

pub fn appendHybridMlpWeights(vec: mlx.mlx_vector_array, hw: *const HybridMlpWeights) void {
    // Plain-bf16 layers (Unsloth Dynamic) carry null-ctx scales/biases — skip those
    // so they don't pollute the eval batch. Mirrors `appendLinearAttnWeights`.
    switch (hw.*) {
        .moe => |*mw| {
            if (mw.exl3_ragged) |*rb| {
                for (rb.storage()) |arr| _ = mlx.mlx_vector_array_append_value(vec, arr);
                for (rb.views[0..rb.count]) |bank| {
                    for ([_]sushi_exl3.Proj{ bank.gate, bank.up, bank.down }) |proj| {
                        for ([_]mlx.mlx_array{ proj.trellis, proj.suh, proj.svh }) |arr| _ = mlx.mlx_vector_array_append_value(vec, arr);
                    }
                }
            }
            inline for (comptime structFields(MoeMlpWeights)) |field| {
                if (field.type == ?mlx.mlx_array) {
                    if (@field(mw, field.name)) |arr| {
                        if (arr.ctx != null) _ = mlx.mlx_vector_array_append_value(vec, arr);
                    }
                } else if (field.type == mlx.mlx_array) {
                    const arr = @field(mw, field.name);
                    if (arr.ctx != null) _ = mlx.mlx_vector_array_append_value(vec, arr);
                }
            }
        },
        .dense => |*dw| {
            inline for (comptime structFields(DenseMlpWeights)) |field| {
                const arr = @field(dw, field.name);
                if (arr.ctx != null) _ = mlx.mlx_vector_array_append_value(vec, arr);
            }
        },
    }
}


/// Detect quantization bits from weight and scales shapes: bits = w_cols * 32 / (s_cols * group_size)
fn detectQuantBits(w: mlx.mlx_array, sc: mlx.mlx_array, group_size: u32) u32 {
    const w_shape = mlx.getShape(w);
    const s_shape = mlx.getShape(sc);
    if (w_shape.len < 2 or s_shape.len < 2) return 4;
    const w_cols: u32 = @intCast(w_shape[w_shape.len - 1]);
    const s_cols: u32 = @intCast(s_shape[s_shape.len - 1]);
    if (s_cols == 0) return 4;
    return (w_cols * 32) / (s_cols * group_size);
}

/// Last-axis size of an array, as the input-dimension hint for
/// `quantParamsHinted`. Null for null-ctx handles or degenerate shapes.
pub inline fn lastDim(x: mlx.mlx_array) ?u32 {
    if (x.ctx == null) return null;
    const shape = mlx.getShape(x);
    if (shape.len == 0) return null;
    const d = shape[shape.len - 1];
    return if (d > 0) @as(u32, @intCast(d)) else null;
}

/// Detect a quantized weight's (bits, group_size, mode) from its scales tensor
/// plus the model config:
/// - uint8 scales → fp8-family. In an mxfp4/mxfp8 model the config's
///   mode/bits/group_size apply (the fp8 scheme is uniform within a model).
///   Under an affine config this is the legacy "mxfp8 tensor inside an affine
///   checkpoint" case: bits is always 8, group size from the shape ratio.
/// - float scales → affine. In an affine model, bits via detectQuantBits
///   against the config group_size (mixed-precision models share one gs). For
///   affine OVERRIDES inside a non-affine model (a QAT checkpoint keeping its
///   shared MLP at affine 8-bit/gs64) the config group_size doesn't
///   apply: w_cols*32/s_cols only pins bits×gs, so solve exactly with the
///   caller's `in_dim` (activation inner dim); without a hint, assume mlx-lm's
///   override default group size 64.
/// Solve an affine weight's exact (bits, group_size) from packed-column
/// geometry when the input dimension is known:
///   bits = w_cols * 32 / in_dim,  group_size = in_dim / s_cols
/// Returns null unless both divide EXACTLY and land on values MLX's affine
/// packing supports — callers then fall back to config-based detection.
/// This is what lets a sidecar (e.g. an MTP head quantized 5-bit/gs-128 over
/// a 4-bit/gs-64 trunk) resolve per-weight instead of inheriting the trunk's
/// group size.
/// True when an embedding/lm_head TABLE is stored dense (float dtype) even
/// though the config declares global quantization. Mixed checkpoints quantize
/// per-tensor — a 2-bit pack can ship a bf16 embed_tokens beside 2-bit experts —
/// so scales-presence must be decided by the WEIGHT's dtype, never the
/// config's bits (a packed uint32 table missing scales still crashes
/// honestly at the mandatory fetch).
pub fn floatDtypeTable(dtype: mlx.mlx_dtype) bool {
    return dtype == .bfloat16 or dtype == .float16 or dtype == .float32;
}

pub fn affineParamsFromGeometry(w: mlx.mlx_array, sc: mlx.mlx_array, in_dim: u32) ?QuantParams {
    if (sc.ctx == null) return null;
    const w_shape = mlx.getShape(w);
    const s_shape = mlx.getShape(sc);
    if (w_shape.len < 2 or s_shape.len < 2) return null;
    const geom = expert_quant_mod.affineGeomFromShapes(
        @intCast(w_shape[w_shape.len - 1]),
        @intCast(s_shape[s_shape.len - 1]),
        in_dim,
    ) orelse return null;
    return .{ .bits = geom.bits, .group_size = geom.group_size, .mode = .affine };
}

/// The fp mode a solved (bits, group_size) pair belongs to. Each format
/// DEFINES its block size — MXFP8/MXFP4 are 32 — so the pair
/// names exactly one of them. Anything else is a layout we don't serve;
/// returning null keeps a checkpoint's own declaration rather than inventing a
/// mode for it.
fn fpQuantModeFor(bits: u32, group_size: u32) ?QuantMode {
    if (bits == 8 and group_size == 32) return .mxfp8;
    if (bits == 4 and group_size == 32) return .mxfp4;
    return null;
}

/// Per-weight (bits, group_size, mode) for an fp-quantized tensor, solved from
/// the packed geometry against the activation's inner dim: a row holds
/// `w_cols * 32` packed bits over `in_dim` values, and `s_cols` scales cover
/// them in groups.
///
/// Returns null when the geometry AGREES with the model-wide declaration —
/// the caller then returns the config verbatim, so a consistent checkpoint
/// resolves exactly as it did before this existed.
fn fpParamsFromGeometry(config: *const ModelConfig, w_cols: u32, s_cols: u32, in_dim: u32) ?QuantParams {
    if (w_cols == 0 or s_cols == 0 or in_dim == 0) return null;
    const packed_bits = w_cols * 32;
    if (packed_bits % in_dim != 0 or in_dim % s_cols != 0) return null;
    const bits = packed_bits / in_dim;
    const group_size = in_dim / s_cols;
    if (bits == config.quant_bits and group_size == config.quant_group_size) return null;
    const mode = fpQuantModeFor(bits, group_size) orelse return null;
    return .{ .bits = bits, .group_size = group_size, .mode = mode };
}

pub fn computeQuantParams(config: *const ModelConfig, w: mlx.mlx_array, sc: mlx.mlx_array, in_dim: ?u32) QuantParams {
    const w_shape = mlx.getShape(w);
    const s_shape = mlx.getShape(sc);
    const w_cols: u32 = if (w_shape.len >= 2) @intCast(w_shape[w_shape.len - 1]) else 0;
    const s_cols: u32 = if (s_shape.len >= 2) @intCast(s_shape[s_shape.len - 1]) else 0;

    if (mlx.mlx_array_dtype(sc) == .uint8) {
        if (config.quant_mode != .affine) {
            // A per-tensor override can live INSIDE the fp family: an
            // mxfp4 build can store `model.embed_tokens` as mxfp8. Every fp mode
            // carries uint8 (E8M0/E4M3) scales, so the scales-dtype rule above
            // cannot separate them and the embedding resolved at the model-wide
            // 4 bits — an `mlx_dequantize` shape rejection, i.e. an MLX error,
            // i.e. a dead server on the first request. The packed geometry says
            // it exactly; solve, exactly as the affine arm below does.
            if (in_dim) |in| {
                if (fpParamsFromGeometry(config, w_cols, s_cols, in)) |qp| return qp;
            }
            return .{ .bits = config.quant_bits, .group_size = config.quant_group_size, .mode = config.quant_mode };
        }
        const gs: u32 = if (w_cols > 0 and s_cols > 0) (w_cols * 32) / (s_cols * 8) else 32;
        return .{ .bits = 8, .group_size = gs, .mode = .mxfp8 };
    }

    if (config.quant_mode == .affine) {
        // Exact per-weight solve when the caller knows the input dim — a
        // hinted tensor whose geometry contradicts the config's group size
        // (MTP sidecars mix gs 128/64/32 over a gs-64 trunk) resolves to its
        // TRUE params. Consistent tensors solve to the config values anyway.
        if (in_dim) |in| {
            if (affineParamsFromGeometry(w, sc, in)) |qp| return qp;
        }
        return .{
            .bits = detectQuantBits(w, sc, config.quant_group_size),
            .group_size = config.quant_group_size,
            .mode = .affine,
        };
    }

    if (in_dim) |in| {
        if (in > 0 and s_cols > 0 and w_cols > 0) {
            return .{ .bits = (w_cols * 32) / in, .group_size = in / s_cols, .mode = .affine };
        }
    }
    if (w_cols > 0 and s_cols > 0) {
        const ratio = (w_cols * 32) / s_cols; // = bits × group_size
        return .{ .bits = @max(ratio / 64, 1), .group_size = 64, .mode = .affine };
    }
    return .{ .bits = 8, .group_size = 64, .mode = .affine };
}

pub fn getWeightFmt(weights: *const Weights, buf: *[256]u8, comptime fmt: []const u8, prefix: []const u8) error{MissingWeight}!mlx.mlx_array {
    const name = std.fmt.bufPrint(buf, fmt, .{prefix}) catch unreachable;
    return weights.get(name) orelse {
        log.err("MISSING WEIGHT: {s}\n", .{name});
        return error.MissingWeight;
    };
}

/// The 4-expert 128x128 pack the synthetic switch_mlp binds answer to.
fn exl3SwitchBankConfig(rate: expert_exl3.Rate) ModelConfig {
    return .{
        .expert_layout = .exl3_k4,
        .expert_quant_rate = rate,
        .num_experts = 4,
        .hidden_size = 128,
        .moe_intermediate_size = 128,
    };
}

/// Bind an EXL3 switch_mlp bank whose three trellises carry `trellis`, against
/// a config that names `rate`.
fn bindExl3SwitchBank(st: mlx.mlx_stream, trellis: []const c_int, rate: expert_exl3.Rate) !SwitchMlpBank {
    var w = Weights.init(std.testing.allocator);
    defer w.deinit();
    const scales = [_]c_int{ 4, 128 };
    for ([_][]const u8{ "gate", "up", "down" }) |proj| {
        for ([_]struct { leaf: []const u8, dtype: mlx.mlx_dtype, shape: []const c_int }{
            .{ .leaf = "trellis", .dtype = .uint16, .shape = trellis },
            .{ .leaf = "suh", .dtype = .float16, .shape = &scales },
            .{ .leaf = "svh", .dtype = .float16, .shape = &scales },
        }) |part| {
            var tbuf: [96]u8 = undefined;
            const key = try std.fmt.bufPrint(&tbuf, "language_model.mtp.layers.0.mlp.switch_mlp.{s}_proj.{s}", .{ proj, part.leaf });
            var arr = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_zeros(&arr, part.shape.ptr, @intCast(part.shape.len), part.dtype, st));
            try w.map.put(try w.allocator.dupe(u8, key), arr);
        }
    }
    var buf: [256]u8 = undefined;
    const config = exl3SwitchBankConfig(rate);
    return loadSwitchMlpBank(&w, &buf, "language_model.mtp", 0, true, &config);
}

/// `{prefix}.{base}.{suffix}` with a RUNTIME base — the embedding table's name
/// is the checkpoint's (embed_tokens / word_embeddings / embed / embeddings)
/// and its three lookups must agree, which a comptime format can't express
/// without repeating the arch chain per suffix.
pub fn getNamedWeight(weights: *const Weights, buf: *[256]u8, prefix: []const u8, base: []const u8, suffix: []const u8) ?mlx.mlx_array {
    const name = std.fmt.bufPrint(buf, "{s}.{s}.{s}", .{ prefix, base, suffix }) catch unreachable;
    return weights.get(name);
}

pub fn getWeightFmtOpt(weights: *const Weights, buf: *[256]u8, comptime fmt: []const u8, prefix: []const u8) ?mlx.mlx_array {
    const name = std.fmt.bufPrint(buf, fmt, .{prefix}) catch unreachable;
    return weights.get(name);
}

pub fn getLayerWeightOpt(weights: *const Weights, buf: *[256]u8, prefix: []const u8, layer: u32, suffix: []const u8) ?mlx.mlx_array {
    const name = std.fmt.bufPrint(buf, "{s}.layers.{d}.{s}", .{ prefix, layer, suffix }) catch unreachable;
    return weights.get(name);
}

pub fn getLayerWeight(weights: *const Weights, buf: *[256]u8, prefix: []const u8, layer: u32, suffix: []const u8) error{MissingWeight}!mlx.mlx_array {
    const name = std.fmt.bufPrint(buf, "{s}.layers.{d}.{s}", .{ prefix, layer, suffix }) catch unreachable;
    return weights.get(name) orelse {
        log.err("MISSING WEIGHT: {s}\n", .{name});
        return error.MissingWeight;
    };
}

pub const SwitchMlpBank = struct {
    ragged: ?sushi_exl3.Ragged = null,
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

/// A trellis the EXL3 kernels can decode AND the memory plan can price:
/// `[E, in/16, out/16, n]` at a rate the CONFIG's bill covers. Decode reads n
/// off the tensor and a layer may carry fewer bits than the pack's widest — a
/// wider one under-bills, and an under-bill here is an uncatchable Metal OOM.
pub fn loadSwitchMlpBank(weights: *const Weights, buf: *[256]u8, prefix: []const u8, layer: u32, exl3: bool, config: *const ModelConfig) !SwitchMlpBank {
    if (exl3) {
        var plan = sushi_exl3.GroupLayout.init(config.hidden_size, config.moe_intermediate_size, config.expert_quant_rate);
        plan.rates = config.expert_quant_rates;
        var groups: [sushi_exl3.group_layout.max_groups]sushi_exl3.Bank = undefined;
        var start_buf: [256]u8 = undefined;
        const start = try std.fmt.bufPrint(&start_buf, "{s}.layers.{d}.mlp.switch_mlp.", .{ prefix, layer });
        var it = weights.map.iterator();
        while (it.next()) |entry| {
            if (!std.mem.startsWith(u8, entry.key_ptr.*, start)) continue;
            const arr = entry.value_ptr.*;
            const shape = mlx.getShape(arr);
            if (shape.len > 4) return error.Exl3TrellisGeometry;
            var dims: [4]u64 = undefined;
            for (shape, 0..) |d, i| {
                if (d <= 0) return error.Exl3TrellisGeometry;
                dims[i] = @intCast(d);
            }
            const name = plan.add(entry.key_ptr.*[start.len..], dims[0..shape.len], switch (mlx.mlx_array_dtype(arr)) {
                .uint16 => .u16,
                .float16 => .f16,
                else => .other,
            }) catch |err| {
                if (err == error.Exl3GroupGeometry) return error.Exl3TrellisGeometry;
                return err;
            };
            const bank = &groups[name.group];
            const proj = switch (name.projection) {
                0 => &bank.gate,
                1 => &bank.up,
                else => &bank.down,
            };
            switch (name.part) {
                0 => proj.trellis = arr,
                1 => proj.suh = arr,
                else => proj.svh = arr,
            }
        }
        const router = getLayerWeightOpt(weights, buf, prefix, layer, "mlp.gate.weight");
        var experts = config.num_experts;
        if (router) |r| {
            const sh = mlx.getShape(r);
            if (sh.len != 2 or sh[0] <= 0) return error.Exl3RouterWidthMismatch;
            experts = @intCast(sh[0]);
        }
        try plan.finish(experts, config.num_experts_per_tok, config.expert_streaming, config.num_experts);
        if (config.moe_n_group > 1 and (plan.grouped.? or experts != config.num_experts)) return error.Exl3RouterGroupsUnsupported;
        var ragged: ?sushi_exl3.Ragged = null;
        if (plan.grouped.?) {
            ragged = try sushi_exl3.Ragged.init(mlx.gpuStream(), groups[0..plan.count]);
            errdefer ragged.?.deinit();
            // The pack's group tensors become views of the ragged storage, so the loaded originals
            // are freed once it is evaluated and the layer's bytes stay resident once.
            var views = weights.map.iterator();
            while (views.next()) |entry| {
                if (!std.mem.startsWith(u8, entry.key_ptr.*, start)) continue;
                const name = try sushi_exl3.group_layout.Name.parse(entry.key_ptr.*[start.len..]);
                const bank = ragged.?.views[name.group];
                const proj = switch (name.projection) {
                    0 => bank.gate,
                    1 => bank.up,
                    else => bank.down,
                };
                var view = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_array_set(&view, switch (name.part) {
                    0 => proj.trellis,
                    1 => proj.suh,
                    else => proj.svh,
                }));
                _ = mlx.mlx_array_free(entry.value_ptr.*);
                entry.value_ptr.* = view;
            }
        }
        const g = if (ragged) |rb| rb.views[0] else groups[0];
        return .{
            .gate_w = g.gate.trellis,
            .gate_s = g.gate.suh,
            .gate_b = g.gate.svh,
            .up_w = g.up.trellis,
            .up_s = g.up.suh,
            .up_b = g.up.svh,
            .down_w = g.down.trellis,
            .down_s = g.down.suh,
            .down_b = g.down.svh,
            .ragged = ragged,
        };
    }
    return .{
        .gate_w = try getLayerWeight(weights, buf, prefix, layer, "mlp.switch_mlp.gate_proj.weight"),
        .gate_s = getLayerWeightOpt(weights, buf, prefix, layer, "mlp.switch_mlp.gate_proj.scales") orelse mlx.mlx_array_new(),
        .gate_b = getLayerWeightOpt(weights, buf, prefix, layer, "mlp.switch_mlp.gate_proj.biases") orelse mlx.mlx_array_new(),
        .up_w = try getLayerWeight(weights, buf, prefix, layer, "mlp.switch_mlp.up_proj.weight"),
        .up_s = getLayerWeightOpt(weights, buf, prefix, layer, "mlp.switch_mlp.up_proj.scales") orelse mlx.mlx_array_new(),
        .up_b = getLayerWeightOpt(weights, buf, prefix, layer, "mlp.switch_mlp.up_proj.biases") orelse mlx.mlx_array_new(),
        .down_w = try getLayerWeight(weights, buf, prefix, layer, "mlp.switch_mlp.down_proj.weight"),
        .down_s = getLayerWeightOpt(weights, buf, prefix, layer, "mlp.switch_mlp.down_proj.scales") orelse mlx.mlx_array_new(),
        .down_b = getLayerWeightOpt(weights, buf, prefix, layer, "mlp.switch_mlp.down_proj.biases") orelse mlx.mlx_array_new(),
    };
}


/// Fetch a quantization scale/bias tensor, tolerant of dense bf16 models.
/// quant_bits == 0 ⇒ dense bf16 (no .scales/.biases exist anywhere) ⇒ return a
/// null-ctx array, which downstream code (qmatmulBits, gatherExpertMm, append
/// guards) reads as "this weight is plain bf16". quant_bits > 0 ⇒ a genuinely
/// quantized model, so fetch mandatorily — a missing scale stays a clear
/// MISSING WEIGHT error rather than silently degrading to a dense path.
pub fn getLayerScaleOrEmpty(weights: *const Weights, buf: *[256]u8, prefix: []const u8, layer: u32, suffix: []const u8, quant_bits: u32) error{MissingWeight}!mlx.mlx_array {
    if (quant_bits == 0) return mlx.mlx_array_new();
    return try getLayerWeight(weights, buf, prefix, layer, suffix);
}

/// Split a native fused QKV projection whose value width is independent of its
/// key width. MiMo's `[q | k | v]` rows are `[nq*hd | nkv*hd | nkv*vhd]`;
/// `splitFusedQkvRows` remains the symmetric compatibility wrapper.
pub fn splitFusedQkvAsymRows(
    arr: mlx.mlx_array,
    q_rows: u32,
    k_rows: u32,
    v_rows: u32,
    transpose: bool,
    owned: *std.ArrayList(mlx.mlx_array),
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,
) ![3]mlx.mlx_array {
    if (arr.ctx == null) return .{ mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new() };
    const shape = mlx.getShape(arr);
    const total_rows = std.math.add(u32, q_rows, std.math.add(u32, k_rows, v_rows) catch return error.BadFusedQkvShape) catch return error.BadFusedQkvShape;
    if (shape.len != 2 or shape[0] != @as(c_int, @intCast(total_rows))) return error.BadFusedQkvShape;
    const bounds = [_]c_int{ 0, @intCast(q_rows), @intCast(q_rows + k_rows), shape[0] };
    var out: [3]mlx.mlx_array = undefined;
    for (0..3) |i| {
        const start = [_]c_int{ bounds[i], 0 };
        const stop = [_]c_int{ bounds[i + 1], shape[1] };
        const strides = [_]c_int{ 1, 1 };
        var view = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(view);
        try mlx.check(mlx.mlx_slice(&view, arr, &start, 2, &stop, 2, &strides, 2, s));
        var part = mlx.mlx_array_new();
        if (transpose) {
            part = try transposeBf16Weight(view, s);
        } else {
            try mlx.check(mlx.mlx_contiguous(&part, view, false, s));
        }
        try owned.append(allocator, part);
        out[i] = part;
    }
    return out;
}

pub fn getLayerBias(weights: *const Weights, buf: *[256]u8, prefix: []const u8, layer: u32, suffix: []const u8, config: *const ModelConfig) error{MissingWeight}!mlx.mlx_array {
    if (config.quant_bits == 0) return mlx.mlx_array_new();
    if (config.quant_mode.hasBiases()) return try getLayerWeight(weights, buf, prefix, layer, suffix);
    return getLayerWeightOpt(weights, buf, prefix, layer, suffix) orelse mlx.mlx_array_new();
}

/// A 0-d scalar in `dt` (a bare f32 scalar array promotes bf16 operands).
pub fn scalarOf(val: f32, dt: mlx.mlx_dtype, s: mlx.mlx_stream) !mlx.mlx_array {
    const raw = mlx.mlx_array_new_float(val);
    defer _ = mlx.mlx_array_free(raw);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, raw, dt, s));
    return out;
}

pub fn bf16Scalar(val: f32, s: mlx.mlx_stream) mlx.mlx_array {
    const f32_arr = mlx.mlx_array_new_float(val);
    defer _ = mlx.mlx_array_free(f32_arr);
    var bf16_arr = mlx.mlx_array_new();
    _ = mlx.mlx_astype(&bf16_arr, f32_arr, .bfloat16, s);
    return bf16_arr;
}

pub fn initModuleShell(allocator: std.mem.Allocator, config: ModelConfig, s: mlx.mlx_stream) !Transformer {
    const cache = try KVCache.init(allocator, 0);
    return .{
        .config = config,
        .cache = cache,
        .s = s,
        .allocator = allocator,
        .emb_w = mlx.mlx_array_new(),
        .emb_s = mlx.mlx_array_new(),
        .emb_b = mlx.mlx_array_new(),
        .emb_scale = null,
        .final_norm = mlx.mlx_array_new(),
        .lm_head_w = mlx.mlx_array_new(),
        .lm_head_s = mlx.mlx_array_new(),
        .lm_head_b = mlx.mlx_array_new(),
        .owns_lm_head = false,
        .owns_norms = false,
        .gelu_coeff = bf16Scalar(0.7978845608028654, s),
        .gelu_inner = bf16Scalar(0.044715, s),
        .half = bf16Scalar(0.5, s),
        .one = bf16Scalar(1.0, s),
        .rms_eps_arr = mlx.mlx_array_new_float(config.rms_norm_eps),
        .three = bf16Scalar(3.0, s),
        .neg_one = null,
        .moe_layers = null,
        .ssm_entries = null,
        .moe_seq_offset = 0,
    };
}

pub fn addOne(arr: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const one = bf16Scalar(1.0, s);
    defer _ = mlx.mlx_array_free(one);
    var result = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&result, one, arr, s));
    return result;
}

/// [E, out, in] affine-4/g64 expert bank — the stand-in for a streamed MXFP4
/// slab, which reaches the same quantized arms through the routing override.
fn imxQuantBank(s: mlx.mlx_stream, alloc: std.mem.Allocator, rnd: std.Random, e: c_int, out_dim: c_int, in_dim: c_int, owned: *std.ArrayList(mlx.mlx_array)) ![3]mlx.mlx_array {
    const buf = try alloc.alloc(f32, @intCast(e * out_dim * in_dim));
    defer alloc.free(buf);
    for (buf) |*v| v.* = (rnd.float(f32) - 0.5) * 0.4;
    const a32 = mlx.mlx_array_new_data(buf.ptr, &[_]c_int{ e, out_dim, in_dim }, 3, .float32);
    defer _ = mlx.mlx_array_free(a32);
    var bf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bf);
    try mlx.check(mlx.mlx_astype(&bf, a32, .bfloat16, s));
    var triple = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(triple);
    try mlx.check(mlx.mlx_quantize(&triple, bf, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{}, s));
    var out: [3]mlx.mlx_array = undefined;
    for (&out, 0..) |*a, i| {
        a.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(a, triple, i));
        try owned.append(alloc, a.*);
    }
    return out;
}

test "floatDtypeTable: float tables are dense, packed/quantized ones are not" {
    try std.testing.expect(floatDtypeTable(.bfloat16));
    try std.testing.expect(floatDtypeTable(.float16));
    try std.testing.expect(floatDtypeTable(.float32));
    try std.testing.expect(!floatDtypeTable(.uint32)); // packed affine quant
    try std.testing.expect(!floatDtypeTable(.uint8)); // fp8-encoded scales/modes
    try std.testing.expect(!floatDtypeTable(.int32));
}

test "a missing weight is a load ERROR, not a process exit (issue #217)" {
    var w = model_mod.Weights.init(std.testing.allocator);
    defer w.deinit();
    var buf: [256]u8 = undefined;
    try std.testing.expectError(error.MissingWeight, getLayerWeight(&w, &buf, "model", 0, "mlp.w1.weight"));
    try std.testing.expectError(error.MissingWeight, getWeightFmt(&w, &buf, "{s}.norm.weight", "model"));
}

test "exl3 MTP switch_mlp binds restacked trellis/suh/svh under the head prefix" {
    const t = std.testing;
    const s = mlx.gpuStream();
    const put = struct {
        fn add(w: *Weights, name: []const u8, dtype: mlx.mlx_dtype, shape: []const c_int, st: mlx.mlx_stream) !void {
            var arr = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_zeros(&arr, shape.ptr, @intCast(shape.len), dtype, st));
            try w.map.put(try w.allocator.dupe(u8, name), arr);
        }
    }.add;
    var w = Weights.init(t.allocator);
    defer w.deinit();
    const trellis = [_]c_int{ 4, 8, 8, 64 };
    const suh = [_]c_int{ 4, 128 };
    const svh = [_]c_int{ 4, 128 };
    for ([_][]const u8{ "gate", "up", "down" }) |proj| {
        var tbuf: [96]u8 = undefined;
        const tk = try std.fmt.bufPrint(&tbuf, "language_model.mtp.layers.0.mlp.switch_mlp.{s}_proj.trellis", .{proj});
        try put(&w, tk, .uint16, &trellis, s);
        const sk = try std.fmt.bufPrint(&tbuf, "language_model.mtp.layers.0.mlp.switch_mlp.{s}_proj.suh", .{proj});
        try put(&w, sk, .float16, &suh, s);
        const vk = try std.fmt.bufPrint(&tbuf, "language_model.mtp.layers.0.mlp.switch_mlp.{s}_proj.svh", .{proj});
        try put(&w, vk, .float16, &svh, s);
    }
    var buf: [256]u8 = undefined;
    const config = exl3SwitchBankConfig(.{ .n = 64 });
    const bank = try loadSwitchMlpBank(&w, &buf, "language_model.mtp", 0, true, &config);
    try t.expectEqual(@as(usize, 4), mlx.getShape(bank.gate_w).len);
    try t.expectEqual(@as(c_int, 4), mlx.getShape(bank.gate_w)[0]);
    try t.expectEqual(mlx.mlx_dtype.uint16, mlx.mlx_array_dtype(bank.gate_w));
    try t.expectEqual(mlx.mlx_dtype.float16, mlx.mlx_array_dtype(bank.gate_s));
    try t.expectEqual(mlx.mlx_dtype.float16, mlx.mlx_array_dtype(bank.gate_b));
    try t.expectEqualSlices(c_int, mlx.getShape(bank.gate_w), mlx.getShape(bank.up_w));
    try t.expectEqualSlices(c_int, mlx.getShape(bank.down_s), mlx.getShape(bank.gate_s));
}

test "exl3 MTP switch_mlp refuses a missing restacked tensor by MissingWeight" {
    const t = std.testing;
    const s = mlx.gpuStream();
    const put = struct {
        fn add(w: *Weights, name: []const u8, dtype: mlx.mlx_dtype, shape: []const c_int, st: mlx.mlx_stream) !void {
            var arr = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_zeros(&arr, shape.ptr, @intCast(shape.len), dtype, st));
            try w.map.put(try w.allocator.dupe(u8, name), arr);
        }
    }.add;
    var w = Weights.init(t.allocator);
    defer w.deinit();
    const trellis = [_]c_int{ 4, 8, 8, 64 };
    const svh = [_]c_int{ 4, 128 };
    for ([_][]const u8{ "gate", "up", "down" }) |proj| {
        var tbuf: [96]u8 = undefined;
        const tk = try std.fmt.bufPrint(&tbuf, "language_model.mtp.layers.0.mlp.switch_mlp.{s}_proj.trellis", .{proj});
        try put(&w, tk, .uint16, &trellis, s);
        const vk = try std.fmt.bufPrint(&tbuf, "language_model.mtp.layers.0.mlp.switch_mlp.{s}_proj.svh", .{proj});
        try put(&w, vk, .float16, &svh, s);
    }
    var buf: [256]u8 = undefined;
    const config = exl3SwitchBankConfig(.{ .n = 64 });
    try t.expectError(error.MissingWeight, loadSwitchMlpBank(&w, &buf, "language_model.mtp", 0, true, &config));
}

test "exl3 a trellis the kernels cannot decode refuses at load" {
    const t = std.testing;
    const s = mlx.gpuStream();
    _ = try bindExl3SwitchBank(s, &[_]c_int{ 4, 8, 8, 48 }, .{ .n = 48 });
    _ = try bindExl3SwitchBank(s, &[_]c_int{ 4, 8, 8, 40 }, .{ .n = 40 });
    _ = try bindExl3SwitchBank(s, &[_]c_int{ 4, 8, 8, 44 }, .{ .n = 44 });
    try t.expectError(error.Exl3TrellisGeometry, bindExl3SwitchBank(s, &[_]c_int{ 4, 8, 8, 130 }, .{ .n = 130 }));
    try t.expectError(error.Exl3TrellisGeometry, bindExl3SwitchBank(s, &[_]c_int{ 4, 8, 8, 14 }, .{ .n = 14 }));
    try t.expectError(error.Exl3TrellisGeometry, bindExl3SwitchBank(s, &[_]c_int{ 4, 8, 8, 41 }, .{ .n = 41 }));
    try t.expectError(error.Exl3TrellisGeometry, bindExl3SwitchBank(s, &[_]c_int{ 4, 8, 64 }, .{ .n = 64 }));
}

test "exl3 a trellis the config's rate or expert geometry does not name refuses at load" {
    const t = std.testing;
    const s = mlx.gpuStream();
    // The streaming bill prices n from the config, so a shard WIDER than the
    // config names must never load; a narrower one is over-billed, not wrong.
    try t.expectError(error.Exl3TrellisGeometry, bindExl3SwitchBank(s, &[_]c_int{ 4, 8, 8, 48 }, .{ .n = 40 }));
    _ = try bindExl3SwitchBank(s, &[_]c_int{ 4, 8, 8, 40 }, .{ .n = 48 });
    try t.expectError(error.Exl3TrellisGeometry, bindExl3SwitchBank(s, &[_]c_int{ 8, 8, 8, 64 }, .{ .n = 64 }));
    try t.expectError(error.Exl3TrellisGeometry, bindExl3SwitchBank(s, &[_]c_int{ 4, 4, 8, 64 }, .{ .n = 64 }));
    try t.expectError(error.Exl3TrellisGeometry, bindExl3SwitchBank(s, &[_]c_int{ 4, 8, 16, 64 }, .{ .n = 64 }));
    _ = try bindExl3SwitchBank(s, &[_]c_int{ 4, 8, 8, 64 }, .{ .n = 64 });
}

test "affineParamsFromGeometry: exact per-weight solve for off-config sidecar quants" {
    // Real geometries from the stamsam 35B-A3B MTP sidecar over a 4-bit/gs-64
    // affine trunk: q_proj 5-bit/gs-128, v_proj 6-bit/gs-128, shared expert
    // 8-bit/gs-128, switch experts 4-bit/gs-64. The old config-gs detection
    // (detectQuantBits with gs 64) returns garbage for the gs-128 tensors.
    const s = mlx.gpuStream();
    const H = 2048; // sidecar hidden

    const mk = struct {
        fn arr(shape: []const c_int, dt: mlx.mlx_dtype, st: mlx.mlx_stream) !mlx.mlx_array {
            var a = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_zeros(&a, shape.ptr, shape.len, dt, st));
            return a;
        }
    };

    // q_proj: w [8192, 320] u32, scales [8192, 16] bf16 → 5-bit / gs 128.
    const wq = try mk.arr(&.{ 8192, 320 }, .uint32, s);
    defer _ = mlx.mlx_array_free(wq);
    const sq = try mk.arr(&.{ 8192, 16 }, .bfloat16, s);
    defer _ = mlx.mlx_array_free(sq);
    const qp_q = affineParamsFromGeometry(wq, sq, H) orelse return error.TestUnexpectedResult;
    try testing.expectEqual(@as(u32, 5), qp_q.bits);
    try testing.expectEqual(@as(u32, 128), qp_q.group_size);

    // v_proj: w [512, 384] u32, scales [512, 16] bf16 → 6-bit / gs 128.
    const wv = try mk.arr(&.{ 512, 384 }, .uint32, s);
    defer _ = mlx.mlx_array_free(wv);
    const sv = try mk.arr(&.{ 512, 16 }, .bfloat16, s);
    defer _ = mlx.mlx_array_free(sv);
    const qp_v = affineParamsFromGeometry(wv, sv, H) orelse return error.TestUnexpectedResult;
    try testing.expectEqual(@as(u32, 6), qp_v.bits);
    try testing.expectEqual(@as(u32, 128), qp_v.group_size);

    // Switch experts (3D): w [256, 512, 256] u32, scales [256, 512, 32] bf16
    // → 4-bit / gs 64 (trunk-standard — must still solve).
    const we = try mk.arr(&.{ 256, 512, 256 }, .uint32, s);
    defer _ = mlx.mlx_array_free(we);
    const se = try mk.arr(&.{ 256, 512, 32 }, .bfloat16, s);
    defer _ = mlx.mlx_array_free(se);
    const qp_e = affineParamsFromGeometry(we, se, H) orelse return error.TestUnexpectedResult;
    try testing.expectEqual(@as(u32, 4), qp_e.bits);
    try testing.expectEqual(@as(u32, 64), qp_e.group_size);

    // Degenerate: bits wouldn't divide exactly → null (caller falls back).
    const qp_bad = affineParamsFromGeometry(wq, sq, 3000);
    try testing.expect(qp_bad == null);

    // And through computeQuantParams on an AFFINE gs-64 config: the hinted
    // exact solve must WIN over the config group size (pre-fix this returned
    // bits 10 / gs 64 for the q_proj geometry).
    var cfg = ModelConfig{};
    cfg.quant_bits = 4;
    cfg.quant_group_size = 64;
    cfg.quant_mode = .affine;
    const qp_via = computeQuantParams(&cfg, wq, sq, H);
    try testing.expectEqual(@as(u32, 5), qp_via.bits);
    try testing.expectEqual(@as(u32, 128), qp_via.group_size);
}

test "computeQuantParams resolves an fp override INSIDE the fp family (mxfp8 embed under mxfp4)" {
    // A checkpoint can declare a model-wide mxfp4/gs32 AND a
    // per-tensor override: `"model.embed_tokens": {bits 8, group_size 32,
    // mode "mxfp8"}`. Both modes carry uint8 (E8M0) scales, so the
    // scales-dtype rule cannot tell them apart and the embedding resolved at
    // the model-wide 4 bits — `mlx_dequantize` then rejected the shapes
    // ("matrix of shape (1,512) and scales of shape (1,64)"), and an MLX error
    // is not a Zig error: the first request killed the server.
    //
    // The packed geometry states the answer exactly, the same way the affine
    // path already solves off-config sidecar quants.
    const s = mlx.gpuStream();
    const H = 2048; // hidden

    const mk = struct {
        fn arr(shape: []const c_int, dt: mlx.mlx_dtype, st: mlx.mlx_stream) !mlx.mlx_array {
            var a = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_zeros(&a, shape.ptr, shape.len, dt, st));
            return a;
        }
    };

    var cfg = ModelConfig{};
    cfg.quant_bits = 4;
    cfg.quant_group_size = 32;
    cfg.quant_mode = .mxfp4;

    // embed_tokens: w [V, 512] u32 + scales [V, 64] u8 → 2048 values/row at
    // 8 bits, gs 32. (Real shapes; V shrunk — only the last axis is read.)
    const we = try mk.arr(&.{ 1024, 512 }, .uint32, s);
    defer _ = mlx.mlx_array_free(we);
    const se = try mk.arr(&.{ 1024, 64 }, .uint8, s);
    defer _ = mlx.mlx_array_free(se);
    const qp_e = computeQuantParams(&cfg, we, se, H);
    try testing.expectEqual(QuantMode.mxfp8, qp_e.mode);
    try testing.expectEqual(@as(u32, 8), qp_e.bits);
    try testing.expectEqual(@as(u32, 32), qp_e.group_size);

    // An ordinary mxfp4 linear in the SAME model: w [2048, 256] + [2048, 64].
    // Unchanged — the config already describes it.
    const wl = try mk.arr(&.{ 2048, 256 }, .uint32, s);
    defer _ = mlx.mlx_array_free(wl);
    const sl = try mk.arr(&.{ 2048, 64 }, .uint8, s);
    defer _ = mlx.mlx_array_free(sl);
    const qp_l = computeQuantParams(&cfg, wl, sl, H);
    try testing.expectEqual(QuantMode.mxfp4, qp_l.mode);
    try testing.expectEqual(@as(u32, 4), qp_l.bits);
    try testing.expectEqual(@as(u32, 32), qp_l.group_size);

    // A geometry that maps to no fp format we serve is NOT invented into one:
    // the config stands, and the load fails honestly if it really disagrees.
    const s_odd = try mk.arr(&.{ 2048, 21 }, .uint8, s);
    defer _ = mlx.mlx_array_free(s_odd);
    const qp_odd = computeQuantParams(&cfg, wl, s_odd, H);
    try testing.expectEqual(QuantMode.mxfp4, qp_odd.mode);
    try testing.expectEqual(@as(u32, 4), qp_odd.bits);
    try testing.expectEqual(@as(u32, 32), qp_odd.group_size);

    // No inner-dim hint ⇒ nothing to solve against; config stands (every
    // non-hinted fp call site keeps its pre-fix behavior).
    const qp_nohint = computeQuantParams(&cfg, we, se, null);
    try testing.expectEqual(QuantMode.mxfp4, qp_nohint.mode);
    try testing.expectEqual(@as(u32, 4), qp_nohint.bits);
}

test "computeQuantParams resolves mixed mxfp4 + affine-override weights" {
    // Mixed shape: most tensors mxfp4 (uint8 fp8 scales, gs 32) with an
    // affine override on one linear
    // 8-bit/gs64 (bf16 scales + biases). Detection key: scales dtype, then
    // the activation inner dim to pin (bits, group_size) for the override.
    const s = mlx.gpuStream();
    const IN = 2560;
    const OUT = 8;

    var cfg = ModelConfig{};
    cfg.quant_bits = 4;
    cfg.quant_group_size = 32;
    cfg.quant_mode = .mxfp4;

    // mxfp4 tensor: w [OUT, IN*4/32] u32, scales [OUT, IN/32] u8.
    var w_nv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_nv);
    var sc_nv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc_nv);
    const w_nv_sh = [_]c_int{ OUT, IN * 4 / 32 };
    const sc_nv_sh = [_]c_int{ OUT, IN / 32 };
    try mlx.check(mlx.mlx_zeros(&w_nv, &w_nv_sh, 2, .uint32, s));
    try mlx.check(mlx.mlx_zeros(&sc_nv, &sc_nv_sh, 2, .uint8, s));
    const qp_nv = computeQuantParams(&cfg, w_nv, sc_nv, null);
    try testing.expectEqual(QuantMode.mxfp4, qp_nv.mode);
    try testing.expectEqual(@as(u32, 4), qp_nv.bits);
    try testing.expectEqual(@as(u32, 32), qp_nv.group_size);

    // Affine 8-bit/gs64 override: w [OUT, IN*8/32] u32, scales [OUT, IN/64] bf16.
    var w_af = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_af);
    var sc_af = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc_af);
    const w_af_sh = [_]c_int{ OUT, IN * 8 / 32 };
    const sc_af_sh = [_]c_int{ OUT, IN / 64 };
    try mlx.check(mlx.mlx_zeros(&w_af, &w_af_sh, 2, .uint32, s));
    try mlx.check(mlx.mlx_zeros(&sc_af, &sc_af_sh, 2, .bfloat16, s));

    // With the activation inner-dim hint: exact.
    const qp_af = computeQuantParams(&cfg, w_af, sc_af, IN);
    try testing.expectEqual(QuantMode.affine, qp_af.mode);
    try testing.expectEqual(@as(u32, 8), qp_af.bits);
    try testing.expectEqual(@as(u32, 64), qp_af.group_size);

    // Without a hint: falls back to mlx-lm's override default gs=64.
    const qp_af_nohint = computeQuantParams(&cfg, w_af, sc_af, null);
    try testing.expectEqual(QuantMode.affine, qp_af_nohint.mode);
    try testing.expectEqual(@as(u32, 8), qp_af_nohint.bits);
    try testing.expectEqual(@as(u32, 64), qp_af_nohint.group_size);

    // Plain affine model (config mode affine, bf16 scales): unchanged
    // detect-against-config-gs behavior.
    var cfg_affine = ModelConfig{};
    cfg_affine.quant_bits = 4;
    cfg_affine.quant_group_size = 64;
    cfg_affine.quant_mode = .affine;
    const qp_plain = computeQuantParams(&cfg_affine, w_af, sc_af, null);
    try testing.expectEqual(QuantMode.affine, qp_plain.mode);
    try testing.expectEqual(@as(u32, 8), qp_plain.bits);
    try testing.expectEqual(@as(u32, 64), qp_plain.group_size);
}

test "QuantParamsCache put/lookup round-trip" {
    var cache: BitsCache = .{};
    const s = mlx.gpuStream();

    // Three real arrays with distinct ctx pointers — the cache keys.
    var a = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a);
    var b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b);
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    const shape = [_]c_int{8};
    try mlx.check(mlx.mlx_zeros(&a, &shape, 1, .bfloat16, s));
    try mlx.check(mlx.mlx_zeros(&b, &shape, 1, .bfloat16, s));
    try mlx.check(mlx.mlx_zeros(&c, &shape, 1, .bfloat16, s));

    try testing.expect(cache.put(a.ctx.?, .{ .bits = 4, .group_size = 32, .mode = .affine }));
    try testing.expect(cache.put(b.ctx.?, .{ .bits = 4, .group_size = 64, .mode = .mxfp4 }));
    try testing.expect(cache.put(c.ctx.?, .{ .bits = 8, .group_size = 128, .mode = .affine }));

    // Build a Transformer-like view over the cache via quantParamsFor — we
    // need a real Transformer to call the method, so verify by direct slot
    // lookup instead. (The forward path goes through quantParamsFor, which is
    // exercised by integration tests; this test pins the data structure.)
    {
        const idx = BitsCache.slot(a.ctx.?);
        var found = false;
        for (0..4) |i| {
            const j = (idx + i) & (BITS_CACHE_CAP - 1);
            if (cache.keys[j] == a.ctx) {
                try testing.expectEqual(@as(u8, 4), cache.vals_bits[j]);
                try testing.expectEqual(@as(u8, 32 / 8), cache.vals_gs_div8[j]);
                try testing.expectEqual(@backingInt(QuantMode.affine), cache.vals_mode[j]);
                found = true;
                break;
            }
        }
        try testing.expect(found);
    }
    {
        const idx = BitsCache.slot(c.ctx.?);
        var found = false;
        for (0..4) |i| {
            const j = (idx + i) & (BITS_CACHE_CAP - 1);
            if (cache.keys[j] == c.ctx) {
                try testing.expectEqual(@as(u8, 8), cache.vals_bits[j]);
                try testing.expectEqual(@as(u8, 128 / 8), cache.vals_gs_div8[j]);
                try testing.expectEqual(@backingInt(QuantMode.affine), cache.vals_mode[j]);
                found = true;
                break;
            }
        }
        try testing.expect(found);
    }
}

test "qwen4 expert streaming targets only the trunk bank" {
    var config = ModelConfig{ .model_type = "qwen4_exp", .weight_prefix = "language_model.model", .expert_streaming = true };
    try std.testing.expect(shouldStreamExpertBank(&config, "language_model.model"));
    try std.testing.expect(!shouldStreamExpertBank(&config, "language_model.mtp"));
    config.expert_streaming = false;
    try std.testing.expect(!shouldStreamExpertBank(&config, "language_model.model"));
    var moe = ModelConfig{ .model_type = "not_a_served_arch", .weight_prefix = "language_model.model", .expert_streaming = true };
    try std.testing.expect(!shouldStreamExpertBank(&moe, "language_model.model"));
}

test "appendLinearAttnWeights skips fields with null ctx (plain bf16 layers)" {
    const s = mlx.gpuStream();
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);

    // Real arrays for the 9 non-scale/bias fields so they have non-null ctx.
    const sh = [_]c_int{1};
    var arrs: [9]mlx.mlx_array = undefined;
    for (&arrs) |*a| {
        a.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_zeros(a, &sh, 1, .bfloat16, s));
    }
    defer for (arrs) |a| {
        _ = mlx.mlx_array_free(a);
    };

    // Simulate the UD layout: weights set, scales/biases null.
    const la: LinearAttnWeights = .{
        .qkv_w = arrs[0],
        .qkv_s = mlx.mlx_array{ .ctx = null },
        .qkv_b = mlx.mlx_array{ .ctx = null },
        .z_w = arrs[1],
        .z_s = mlx.mlx_array{ .ctx = null },
        .z_b = mlx.mlx_array{ .ctx = null },
        .a_w = arrs[2],
        .a_s = mlx.mlx_array{ .ctx = null },
        .a_b = mlx.mlx_array{ .ctx = null },
        .b_w = arrs[3],
        .b_s = mlx.mlx_array{ .ctx = null },
        .b_b = mlx.mlx_array{ .ctx = null },
        .conv1d_w = arrs[4],
        .A_log = arrs[5],
        .dt_bias = arrs[6],
        .norm_w = arrs[7],
        .out_w = arrs[8],
        .out_s = mlx.mlx_array{ .ctx = null },
        .out_b = mlx.mlx_array{ .ctx = null },
    };

    appendLinearAttnWeights(vec, &la);

    // Of 19 mlx_array fields: 5 weights (qkv/z/a/b/out) + 4 SSM bits
    // (conv1d/A_log/dt_bias/norm_w) = 9 expected. The 10 null-ctx scales/biases
    // are skipped — confirms the optional-bf16 path doesn't poison the eval batch.
    try testing.expectEqual(@as(usize, 9), mlx.mlx_vector_array_size(vec));
}

test "appendHybridMlpWeights skips MoE fields with null ctx (UD MoE bf16 router/SEG)" {
    const s = mlx.gpuStream();
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);

    // Real arrays for every weight (`*_w`) — non-null ctx. Quantized projections
    // also have real scales/biases. UD bf16 layers (router, shared_expert_gate)
    // get null-ctx scales/biases.
    const sh = [_]c_int{1};
    var arrs: [16]mlx.mlx_array = undefined;
    for (&arrs) |*a| {
        a.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_zeros(a, &sh, 1, .bfloat16, s));
    }
    defer for (arrs) |a| {
        _ = mlx.mlx_array_free(a);
    };

    // Unsloth-dynamic MoE layout: router + shared_expert_gate are bf16 (null s/b);
    // routed experts (switch_*) and shared_expert (shared_*) stay quantized.
    const mw: MoeMlpWeights = .{
        .router_w = arrs[0],
        .router_s = mlx.mlx_array{ .ctx = null }, // UD bf16
        .router_b = mlx.mlx_array{ .ctx = null }, // UD bf16
        .switch_gate_w = arrs[1],
        .switch_gate_s = arrs[2],
        .switch_gate_b = arrs[3],
        .switch_up_w = arrs[4],
        .switch_up_s = arrs[5],
        .switch_up_b = arrs[6],
        .switch_down_w = arrs[7],
        .switch_down_s = arrs[8],
        .switch_down_b = arrs[9],
        .shared_gate_w = arrs[10],
        .shared_gate_s = arrs[11],
        .shared_gate_b = arrs[12],
        .shared_up_w = arrs[13],
        .shared_up_s = arrs[14],
        .shared_up_b = arrs[15],
        .shared_down_w = arrs[0], // reuse — only ctx-null check matters here
        .shared_down_s = arrs[1],
        .shared_down_b = arrs[2],
        .shared_expert_gate_w = arrs[3],
        .shared_expert_gate_s = mlx.mlx_array{ .ctx = null }, // UD bf16
        .shared_expert_gate_b = mlx.mlx_array{ .ctx = null }, // UD bf16
    };
    const hw: HybridMlpWeights = .{ .moe = mw };

    appendHybridMlpWeights(vec, &hw);

    // Counted by hand: the non-optional `mlx.mlx_array` fields minus the 2 null-ctx
    // ones (router_s, router_b) → 19 appended. Plus the optional `?mlx.mlx_array`
    // fields: shared_expert_gate_w is Some(real) → +1; SEG scales/biases are
    // Some(null-ctx) → +0 each. Total: 19 + 1 = 20.
    try testing.expectEqual(@as(usize, 20), mlx.mlx_vector_array_size(vec));
}

test "appendHybridMlpWeights skips dense fields with null ctx (UD dense bf16)" {
    const s = mlx.gpuStream();
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);

    const sh = [_]c_int{1};
    var arrs: [3]mlx.mlx_array = undefined;
    for (&arrs) |*a| {
        a.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_zeros(a, &sh, 1, .bfloat16, s));
    }
    defer for (arrs) |a| {
        _ = mlx.mlx_array_free(a);
    };

    // All-bf16 dense MLP: weights set, scales/biases null.
    const dw: DenseMlpWeights = .{
        .gate_w = arrs[0],
        .gate_s = mlx.mlx_array{ .ctx = null },
        .gate_b = mlx.mlx_array{ .ctx = null },
        .up_w = arrs[1],
        .up_s = mlx.mlx_array{ .ctx = null },
        .up_b = mlx.mlx_array{ .ctx = null },
        .down_w = arrs[2],
        .down_s = mlx.mlx_array{ .ctx = null },
        .down_b = mlx.mlx_array{ .ctx = null },
    };
    const hw: HybridMlpWeights = .{ .dense = dw };

    appendHybridMlpWeights(vec, &hw);

    // 9 fields, 3 weights non-null + 6 null-ctx scales/biases skipped → 3.
    try testing.expectEqual(@as(usize, 3), mlx.mlx_vector_array_size(vec));
}

test "imatrix capture on the streamed QUANTIZED MoE path keys the routing's global ids" {
    const alloc = std.testing.allocator;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const hidden: c_int = 128;
    const inter: c_int = 64;
    const u: c_int = 4; // experts the slab holds; global id = slot + u
    const experts: c_int = 8;
    const topk: c_int = 2;
    var prng = std.Random.DefaultPrng.init(0x515A);
    const rnd = prng.random();

    var owned: std.ArrayList(mlx.mlx_array) = .empty;
    defer {
        for (owned.items) |a| _ = mlx.mlx_array_free(a);
        owned.deinit(alloc);
    }
    const gate = try imxQuantBank(s, alloc, rnd, u, inter, hidden, &owned);
    const up = try imxQuantBank(s, alloc, rnd, u, inter, hidden, &owned);
    const down = try imxQuantBank(s, alloc, rnd, u, hidden, inter, &owned);
    const none = mlx.mlx_array{ .ctx = null };
    const one = mlx.mlx_array_new_float(1.0);
    try owned.append(alloc, one);
    const mw = MoeMlpWeights{
        .router_w = none,
        .router_s = none,
        .router_b = none,
        .switch_gate_w = gate[0],
        .switch_gate_s = gate[1],
        .switch_gate_b = gate[2],
        .switch_up_w = up[0],
        .switch_up_s = up[1],
        .switch_up_b = up[2],
        .switch_down_w = down[0],
        .switch_down_s = down[1],
        .switch_down_b = down[2],
        .shared_gate_w = none,
        .shared_gate_s = none,
        .shared_gate_b = none,
        .shared_up_w = none,
        .shared_up_s = none,
        .shared_up_b = none,
        .shared_down_w = none,
        .shared_down_s = none,
        .shared_down_b = none,
    };

    const col = try imatrix_capture.Collector.init(alloc, s, "/dev/null", 2, experts, .mimo_v2);
    defer col.deinit();
    var xfm: Transformer = undefined;
    xfm.s = s;
    xfm.config = .{
        .model_type = "mimo_v2",
        .num_experts = @intCast(experts),
        .num_experts_per_tok = @intCast(topk),
        .hidden_size = hidden,
        .moe_intermediate_size = inter,
        .quant_bits = 4,
        .quant_group_size = 64,
        .quant_mode = .affine,
        .hidden_act = .silu,
    };
    xfm.one = one;
    xfm.bits_cache = .{};
    xfm.qwen4 = null;
    xfm.compiled_moe_routing = null;
    xfm.compiled_gelu = null;
    xfm.compiled_geglu = null;
    xfm.cost_trace_active = false;
    xfm.imatrix = col;

    // Slots DESCEND, so the sorted arm's row order is not the routing's: a
    // statistic folded in sorted order lands on the wrong expert. Token 0 is
    // all zeros, so the experts it routes to (and only those) owe zero mass.
    for ([_]c_int{ 2, 1 }) |rows| {
        const layer: u16 = if (rows == 2) 1 else 0;
        const x_host = try alloc.alloc(u16, @intCast(rows * hidden));
        defer alloc.free(x_host);
        seFillBf16(rnd, x_host);
        if (rows == 2) @memset(x_host[0..@intCast(hidden)], 0);
        const x = mlx.mlx_array_new_data(x_host.ptr, &[_]c_int{ 1, rows, hidden }, 3, .bfloat16);
        defer _ = mlx.mlx_array_free(x);
        const slots_host = try alloc.alloc(i32, @intCast(rows * topk));
        defer alloc.free(slots_host);
        const globals_host = try alloc.alloc(i32, slots_host.len);
        defer alloc.free(globals_host);
        for (slots_host, 0..) |*slot, i| {
            slot.* = @intCast(u - 1 - @as(i32, @intCast(i)));
            globals_host[i] = slot.* + u;
        }
        const slab_inds = mlx.mlx_array_new_data(slots_host.ptr, &[_]c_int{ 1, rows, topk }, 3, .int32);
        defer _ = mlx.mlx_array_free(slab_inds);
        const globals = mlx.mlx_array_new_data(globals_host.ptr, &[_]c_int{ rows, topk }, 2, .int32);
        defer _ = mlx.mlx_array_free(globals);
        const w_host = try alloc.alloc(u16, @intCast(rows * topk));
        defer alloc.free(w_host);
        seFillBf16(rnd, w_host);
        const scores = mlx.mlx_array_new_data(w_host.ptr, &[_]c_int{ 1, rows, topk }, 3, .bfloat16);
        defer _ = mlx.mlx_array_free(scores);

        const out = try xfm.moeMLP2WithRouter(x, x, &mw, null, false, null, .{
            .inds = slab_inds,
            .norm_scores = scores,
            .imatrix = .{ .collector = col, .layer = layer, .global_ids = globals },
        });
        defer _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_array_eval(out));

        const slot = &col.layers[layer];
        try std.testing.expectEqual(@as(u64, @intCast(rows)), slot.tokens);
        try std.testing.expect(slot.gu.ctx != null and slot.down.ctx != null and slot.rows.ctx != null);
        try std.testing.expectEqualSlices(c_int, &[_]c_int{ experts, hidden }, mlx.getShape(slot.gu));
        try std.testing.expectEqualSlices(c_int, &[_]c_int{ experts, inter }, mlx.getShape(slot.down));
        try mlx.check(mlx.mlx_array_eval(slot.rows));
        try mlx.check(mlx.mlx_array_eval(slot.gu));
        try mlx.check(mlx.mlx_array_eval(slot.down));
        const routed = mlx.mlx_array_data_float32(slot.rows).?;
        const gu_acc = mlx.mlx_array_data_float32(slot.gu).?;
        const dn_acc = mlx.mlx_array_data_float32(slot.down).?;
        // Nothing lands on a slab SLOT id, everything on the global id.
        for (0..@intCast(u)) |e| try std.testing.expectEqual(@as(f32, 0), routed[e]);
        for (0..@intCast(u * hidden)) |i| try std.testing.expectEqual(@as(f32, 0), gu_acc[i]);
        for (0..@intCast(u * inter)) |i| try std.testing.expectEqual(@as(f32, 0), dn_acc[i]);
        var high: f32 = 0;
        for (@intCast(u)..@intCast(experts)) |e| high += routed[e];
        try std.testing.expectEqual(@as(f32, @floatFromInt(rows * topk)), high);
        // Row order: token 0's experts (the two HIGHEST global ids, slots
        // descend) carry the zero row, every other routed expert real mass.
        for (@intCast(u)..@intCast(experts)) |e| {
            if (routed[e] == 0) continue;
            var mass: f32 = 0;
            for (0..@intCast(inter)) |c| mass += dn_acc[e * @as(usize, @intCast(inter)) + c];
            const zero_row = rows == 2 and e >= @as(usize, @intCast(experts - topk));
            if (zero_row) try std.testing.expectEqual(@as(f32, 0), mass) else try std.testing.expect(mass > 0);
        }
    }
}

test "exl3 shared add releases routed output study3" {
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var xfm: Transformer = undefined;
    xfm.config = .{};
    xfm.bits_cache = .{};
    xfm.compiled_geglu = null;
    xfm.allocator = testing.allocator;
    xfm.s = s;
    xfm.config.expert_layout = .exl3_k4;
    xfm.config.hidden_act = .silu;
    xfm.config.quant_bits = 4;
    xfm.config.quant_group_size = 64;
    xfm.config.moe_intermediate_size = 128;
    var rng = std.Random.DefaultPrng.init(1103);
    const x = try attn256RandBf16(rng.random(), &.{ 1, 64, 128 }, s);
    defer _ = mlx.mlx_array_free(x);
    const shared = try attn256RandBf16(rng.random(), &.{ 128, 128 }, s);
    defer _ = mlx.mlx_array_free(shared);
    const gate = try attn256RandBf16(rng.random(), &.{ 128, 1 }, s);
    defer _ = mlx.mlx_array_free(gate);
    const trh: [8 * 8 * 64]u16 = @splat(1234);
    const tr = mlx.mlx_array_new_data(&trh, &.{ 1, 8, 8, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr);
    const sh: [128]u16 = @splat(0x3c00);
    const scale = mlx.mlx_array_new_data(&sh, &.{ 1, 128 }, 2, .float16);
    defer _ = mlx.mlx_array_free(scale);
    const ih: [64]u32 = @splat(0);
    const inds = mlx.mlx_array_new_data(&ih, &.{ 1, 64, 1 }, 3, .uint32);
    defer _ = mlx.mlx_array_free(inds);
    const sc: [64]u16 = @splat(0x3f80);
    const scores = mlx.mlx_array_new_data(&sc, &.{ 1, 64, 1 }, 3, .bfloat16);
    defer _ = mlx.mlx_array_free(scores);
    var mw = std.mem.zeroes(MoeMlpWeights);
    mw.switch_gate_w = tr;
    mw.switch_up_w = tr;
    mw.switch_down_w = tr;
    mw.switch_gate_s = scale;
    mw.switch_up_s = scale;
    mw.switch_down_s = scale;
    mw.switch_gate_b = scale;
    mw.switch_up_b = scale;
    mw.switch_down_b = scale;
    mw.shared_gate_w = shared;
    mw.shared_up_w = shared;
    mw.shared_down_w = shared;
    mw.shared_expert_gate_w = gate;
    mw.shared_expert_gate_s = .{ .ctx = null };
    mw.shared_expert_gate_b = .{ .ctx = null };
    var before: usize = 0;
    var after: usize = 0;
    for (0..5) |rep| {
        const out = try xfm.moeMLP2WithRouter(x, x, &mw, null, false, null, .{ .inds = inds, .norm_scores = scores });
        try mlx.check(mlx.mlx_array_eval(out));
        _ = mlx.mlx_array_free(out);
        try mlx.check(mlx.mlx_synchronize(s));
        _ = mlx.mlx_clear_cache();
        if (rep == 0) try mlx.check(mlx.mlx_get_active_memory(&before));
    }
    try mlx.check(mlx.mlx_get_active_memory(&after));
    if (after > before + 4096) {
        std.debug.print("[study3 ownership] before={d} after={d} retained={d}\n", .{ before, after, after -| before });
        return error.TestUnexpectedResult;
    }
}

test "mimo v2 fused qkv split keeps asymmetric value rows" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    const q_rows: usize = 4 * 3;
    const k_rows: usize = 2 * 3;
    const v_rows: usize = 2 * 2;
    const in_dim: usize = 5;
    var raw_data: [((q_rows + k_rows + v_rows) * in_dim)]f32 = undefined;
    for (&raw_data, 0..) |*v, i| v.* = @floatFromInt(i);
    const raw = mlx.mlx_array_new_data(
        raw_data[0..].ptr,
        &[_]c_int{ @intCast(q_rows + k_rows + v_rows), @intCast(in_dim) },
        2,
        .float32,
    );
    defer _ = mlx.mlx_array_free(raw);
    var owned: std.ArrayList(mlx.mlx_array) = .empty;
    defer {
        for (owned.items) |arr| _ = mlx.mlx_array_free(arr);
        owned.deinit(allocator);
    }

    const parts = try splitFusedQkvAsymRows(
        raw,
        @intCast(q_rows),
        @intCast(k_rows),
        @intCast(v_rows),
        false,
        &owned,
        allocator,
        s,
    );
    try testing.expectEqualSlices(c_int, &[_]c_int{ @intCast(q_rows), @intCast(in_dim) }, mlx.getShape(parts[0]));
    try testing.expectEqualSlices(c_int, &[_]c_int{ @intCast(k_rows), @intCast(in_dim) }, mlx.getShape(parts[1]));
    try testing.expectEqualSlices(c_int, &[_]c_int{ @intCast(v_rows), @intCast(in_dim) }, mlx.getShape(parts[2]));
}

test "sushi coder uniform qwen4 and mimo banks preserve stored handles" {
    const t = std.testing;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    for ([_][]const u8{ "language_model.model", "model", "language_model.mtp" }) |prefix| {
        var w = Weights.init(t.allocator);
        defer w.deinit();
        var buf: [256]u8 = undefined;
        for ([_][]const u8{ "gate", "up", "down" }) |proj| {
            for ([_][]const u8{ "trellis", "suh", "svh" }) |part| {
                const shape: []const c_int = if (std.mem.eql(u8, part, "trellis")) &.{ 4, 8, 8, 48 } else &.{ 4, 128 };
                var arr = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_zeros(&arr, shape.ptr, @intCast(shape.len), if (shape.len == 4) .uint16 else .float16, s));
                const key = try std.fmt.allocPrint(t.allocator, "{s}.layers.0.mlp.switch_mlp.{s}_proj.{s}", .{ prefix, proj, part });
                try w.map.put(key, arr);
            }
        }
        const cfg = exl3SwitchBankConfig(.{ .n = 48 });
        const bank = try loadSwitchMlpBank(&w, &buf, prefix, 0, true, &cfg);
        const key = try std.fmt.bufPrint(&buf, "{s}.layers.0.mlp.switch_mlp.gate_proj.trellis", .{prefix});
        try t.expectEqual(w.get(key).?.ctx, bank.gate_w.ctx);
        try t.expectEqualSlices(c_int, &.{ 4, 8, 8, 48 }, mlx.getShape(bank.gate_w));
    }
}

test "sushi coder GPU grouped binders cover qwen mimo and MTP" {
    const t = std.testing;
    const s = mlx.gpuStream();
    for ([_][]const u8{ "language_model.model", "model", "language_model.mtp" }) |prefix| {
        var w = Weights.init(t.allocator);
        defer w.deinit();
        var buf: [256]u8 = undefined;
        var router = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_zeros(&router, &[_]c_int{ 3, 128 }, 2, .bfloat16, s));
        try w.map.put(try std.fmt.allocPrint(t.allocator, "{s}.layers.0.mlp.gate.weight", .{prefix}), router);
        for (0..2) |group| {
            const e: c_int = @intCast(group + 1);
            for ([_][]const u8{ "gate", "up", "down" }, 0..) |proj, p| {
                const n: c_int = @intCast(32 + 16 * ((group + p) % 3));
                for ([_][]const u8{ "trellis", "suh", "svh" }, 0..) |part, i| {
                    const shape: []const c_int = if (i == 0) &.{ e, 8, 8, n } else &.{ e, 128 };
                    var arr = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_zeros(&arr, shape.ptr, @intCast(shape.len), if (i == 0) .uint16 else .float16, s));
                    try w.map.put(try std.fmt.allocPrint(t.allocator, "{s}.layers.0.mlp.switch_mlp.{s}_proj.g{d}.{s}", .{ prefix, proj, group, part }), arr);
                }
            }
        }
        var cfg = exl3SwitchBankConfig(.{ .n = 64 });
        cfg.num_experts = 512;
        cfg.num_experts_per_tok = 2;
        var bank = try loadSwitchMlpBank(&w, &buf, prefix, 0, true, &cfg);
        defer bank.ragged.?.deinit();
        try t.expectEqual(@as(usize, 2), bank.ragged.?.count);
        try t.expectEqual(@as(c_int, 2), mlx.getShape(bank.ragged.?.views[1].up.trellis)[0]);
        try t.expectEqual(@as(c_int, 64), mlx.getShape(bank.ragged.?.views[1].up.trellis)[3]);
        // The pack's tensors now share the ragged storage.
        const key = try std.fmt.bufPrint(&buf, "{s}.layers.0.mlp.switch_mlp.down_proj.g1.trellis", .{prefix});
        try t.expectEqualSlices(c_int, &.{ 2, 8, 8, 32 }, mlx.getShape(w.get(key).?));
        cfg.num_experts_per_tok = 4;
        try t.expectError(error.Exl3TopKExceedsExperts, loadSwitchMlpBank(&w, &buf, prefix, 0, true, &cfg));
    }
}

test "sushi coder GPU a loaded grouped layer keeps its stored bytes resident once" {
    const t = std.testing;
    const s = mlx.gpuStream();
    const prefix = "language_model.model";
    var w = Weights.init(t.allocator);
    defer w.deinit();
    var buf: [256]u8 = undefined;
    // A cached buffer up to two pages larger can serve an allocation, which would count its slack as resident;
    // with no cache every buffer is fresh and the count is exact.
    var cache_limit: usize = 0;
    try mlx.check(mlx.mlx_set_cache_limit(&cache_limit, 0));
    defer {
        var unused: usize = 0;
        _ = mlx.mlx_set_cache_limit(&unused, cache_limit);
    }
    try mlx.check(mlx.mlx_clear_cache());
    var before: usize = 0;
    try mlx.check(mlx.mlx_get_active_memory(&before));
    var stored: usize = 0;
    for (0..2) |group| {
        for ([_][]const u8{ "gate", "up", "down" }, 0..) |proj, p| {
            const n: c_int = if ((group + p) % 2 == 0) 24 else 48;
            for ([_][]const u8{ "trellis", "suh", "svh" }, 0..) |part, i| {
                const shape: []const c_int = if (i == 0) &.{ 64, 16, 16, n } else &.{ 64, 256 };
                var arr = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_ones(&arr, shape.ptr, shape.len, if (i == 0) .uint16 else .float16, s));
                try mlx.check(mlx.mlx_array_eval(arr));
                stored += mlx.mlx_array_size(arr) * mlx.mlx_array_itemsize(arr);
                try w.map.put(try std.fmt.allocPrint(t.allocator, "{s}.layers.0.mlp.switch_mlp.{s}_proj.g{d}.{s}", .{ prefix, proj, group, part }), arr);
            }
        }
    }
    var cfg = exl3SwitchBankConfig(.{ .n = 64 });
    cfg.hidden_size = 256;
    cfg.moe_intermediate_size = 256;
    cfg.num_experts = 128;
    cfg.num_experts_per_tok = 8;
    var bank = try loadSwitchMlpBank(&w, &buf, prefix, 0, true, &cfg);
    defer bank.ragged.?.deinit();
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    var hw: HybridMlpWeights = .{ .moe = std.mem.zeroInit(MoeMlpWeights, .{ .exl3_ragged = bank.ragged }) };
    appendHybridMlpWeights(vec, &hw);
    try mlx.check(mlx.mlx_eval(vec));
    try mlx.check(mlx.mlx_synchronize(s));
    var after: usize = 0;
    try mlx.check(mlx.mlx_get_active_memory(&after));
    // The stored bytes once, plus the per-expert table; the loaded originals are gone.
    try t.expectEqual(stored + 128 * sushi_exl3.rate_groups.META_WORDS * 4, after - before);
}
