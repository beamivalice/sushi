// KV-cache quantization backend.
//
// This module owns the storage / dispatch contract between the cache
// (`KVCache` in `transformer.zig`) and the attention call sites. SDPA always
// reads dense `[B, H, T, head_dim]` tensors via `KVCache.denseView` — what the
// cache buffers actually hold is decided here.
//
// Two non-trivial schemes:
//   * affine group-wise quantization at 4 or 8 bits, mathematically identical
//     to mlx-c's `mlx_quantize` weight path. The buffers grow to 3 arrays per
//     K and V (q, scales, biases); attention is unchanged because `denseView`
//     calls `dequantizeAffine` before returning.
//   * TBQ rotated-codebook quantization at 4/3 bits (port of the
//     llama.cpp-turboq-mtp TBQ4_0/TBQ3_0 format, `tbq_tables.zig`). Same
//     3-array plumbing: q holds packed uint8 codes in the rotated domain,
//     scales the fp16 block norm `d`, biases a shallow copy of `d` the cache
//     machinery never reads. `dequantizeTBQ` inverse-rotates on read.
//
// Adding a scheme: one `Scheme` variant, a quantize/dequantize pair here,
// one arm each in `KVCache.update` (write) and `KVCache.denseView` (read).
// SDPA call sites never change; the cache contract holds. The fused
// quantized-attention kernels key on the affine triple (`transformer.zig`).

const std = @import("std");
const mlx = @import("mlx.zig");
const model_settings = @import("model_settings.zig");

/// KV-cache storage scheme.
///   * `off`      — dense bf16.
///   * `affine`   — group-wise affine quant via `mlx_quantize`/`mlx_dequantize`.
///   * `tbq`      — 128-block rotated (seed-42 ±1 Hadamard) fixed-codebook
///     quant at 4 or 3 bits; packed uint8 + fp16 `d` per block.
pub const Scheme = enum { off, affine, tbq };

/// Configuration for the cache's storage backend. Stored on `KVCache.config`
/// and switched on at every read/write boundary.
pub const KVQuantConfig = struct {
    scheme: Scheme,
    /// Affine: 4 or 8. TBQ: 4 or 3. Ignored when `scheme == .off`.
    bits: u8,
    /// Affine group size — number of consecutive elements that share one
    /// scale+bias pair along the last axis. mlx-c convention is 64 for
    /// 4-bit and 8-bit weights; we match that. TBQ keeps it at 128 (the
    /// rotation block: one fp16 `d` per 128 elements); the fused affine
    /// kernels never see a TBQ cache.
    group_size: u32,

    pub const dense: KVQuantConfig = .{ .scheme = .off, .bits = 0, .group_size = 0 };
    /// What a load stores at when neither `--kv-quant` nor `model-settings.json` names a scheme.
    pub const engine_default: KVQuantConfig = .{ .scheme = .affine, .bits = 8, .group_size = 64 };

    pub fn affine(bits: u8) KVQuantConfig {
        std.debug.assert(bits == 4 or bits == 8);
        return .{ .scheme = .affine, .bits = bits, .group_size = 64 };
    }

    pub fn tbq(bits: u8) KVQuantConfig {
        std.debug.assert(bits == 3 or bits == 4);
        return .{ .scheme = .tbq, .bits = bits, .group_size = 128 };
    }

    pub fn isQuant(self: KVQuantConfig) bool {
        return self.scheme != .off;
    }

    /// The wire vocabulary shared by the per-request `kv_quant` body field and
    /// `model-settings.json`: "off"/0/16 (BF16), "4", "8", "tbq4", "tbq3".
    /// Null = unrecognized.
    pub fn fromJsonValue(v: std.json.Value) ?KVQuantConfig {
        switch (v) {
            .string => |s| {
                if (std.mem.eql(u8, s, "off") or std.mem.eql(u8, s, "0") or std.mem.eql(u8, s, "16")) return dense;
                if (std.mem.eql(u8, s, "4")) return affine(4);
                if (std.mem.eql(u8, s, "8")) return affine(8);
                if (std.mem.eql(u8, s, "tbq4")) return tbq(4);
                if (std.mem.eql(u8, s, "tbq3")) return tbq(3);
                return null;
            },
            .integer => |i| {
                if (i == 0 or i == 16) return dense;
                if (i == 4) return affine(4);
                if (i == 8) return affine(8);
                return null;
            },
            else => return null,
        }
    }

    /// GLM's MLA latent bits for this scheme (`glm5_latent.zig`); null = no GLM backend (kv4).
    pub fn glmLatentBits(self: KVQuantConfig) ?u8 {
        return switch (self.scheme) {
            .off => 0,
            .affine => if (self.bits == 8 and self.group_size == 64) 8 else null,
        };
    }

    /// The same vocabulary, for reporting (`/v1/models` `meta.kv_quant`).
    pub fn wireName(self: KVQuantConfig) []const u8 {
        return switch (self.scheme) {
            .off => "off",
            .affine => if (self.bits == 4) "4" else "8",
            .tbq => if (self.bits == 4) "tbq4" else "tbq3",
        };
    }
};

/// The KV scheme a load stores at, with its provenance: `--kv-quant` > per-model setting >
/// `KVQuantConfig.engine_default`. `launch` is the flag's value, or the default when unflagged.
/// A request's own `kv_quant` outranks all of them.
pub const KvCacheChoice = struct {
    config: KVQuantConfig,
    source: model_settings.Source,

    pub fn resolve(setting: ?KVQuantConfig, launch: KVQuantConfig, launch_explicit: bool) KvCacheChoice {
        const p = model_settings.pick(KVQuantConfig, model_settings.launchFlag(KVQuantConfig, launch, launch_explicit), setting, launch);
        return .{ .config = p.value, .source = p.source };
    }

    pub fn label(self: KvCacheChoice) []const u8 {
        return switch (self.config.scheme) {
            .off => "off",
            .affine => if (self.config.bits == 4) "kv4" else "kv8",
            .tbq => if (self.config.bits == 4) "kvtbq4" else "kvtbq3",
        };
    }

    pub fn sourceName(self: KvCacheChoice) []const u8 {
        return model_settings.sourceLabel(self.source, "--kv-quant");
    }
};

/// One quantized K or V triple. Layout for input shape `[..., D]`:
///   q      : `[..., D * bits / 32]` uint32   (packed)
///   scales : `[..., D / group_size]` bf16
///   biases : `[..., D / group_size]` bf16
///
/// Owns its three array handles. Caller frees via `deinit`.
pub const QuantizedKV = struct {
    q: mlx.mlx_array,
    scales: mlx.mlx_array,
    biases: mlx.mlx_array,

    pub fn deinit(self: *QuantizedKV) void {
        _ = mlx.mlx_array_free(self.q);
        _ = mlx.mlx_array_free(self.scales);
        _ = mlx.mlx_array_free(self.biases);
        self.q = mlx.mlx_array_new();
        self.scales = mlx.mlx_array_new();
        self.biases = mlx.mlx_array_new();
    }
};

/// Affine quantize `dense_x` group-wise along the last axis. Returns a
/// `QuantizedKV` triple owned by the caller. Caller's input `dense_x` is
/// not consumed (refcount semantics — the caller still owns it).
pub fn quantizeAffine(
    s: mlx.mlx_stream,
    dense_x: mlx.mlx_array,
    group_size: u32,
    bits: u8,
) !QuantizedKV {
    var vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);

    try mlx.check(mlx.mlx_quantize(
        &vec,
        dense_x,
        mlx.mlx_optional_int.some(@intCast(group_size)),
        mlx.mlx_optional_int.some(@intCast(bits)),
        "affine",
        .{}, // global_scale (null)
        s,
    ));

    // mlx-c convention: the vector contains [q, scales, biases] in that order.
    // Mirror the unpack pattern used elsewhere when consuming a
    // `mlx_vector_array` (e.g. concatenate fan-out).
    const n = mlx.mlx_vector_array_size(vec);
    if (n != 3) return error.UnexpectedQuantizeOutput;

    var out: QuantizedKV = .{
        .q = mlx.mlx_array_new(),
        .scales = mlx.mlx_array_new(),
        .biases = mlx.mlx_array_new(),
    };
    errdefer out.deinit();

    try mlx.check(mlx.mlx_vector_array_get(&out.q, vec, 0));
    try mlx.check(mlx.mlx_vector_array_get(&out.scales, vec, 1));
    try mlx.check(mlx.mlx_vector_array_get(&out.biases, vec, 2));
    return out;
}

/// Affine dequantize a `(q, scales, biases)` triple to the scales dtype. Caller
/// owns the returned array.
pub fn dequantizeAffine(
    s: mlx.mlx_stream,
    q: mlx.mlx_array,
    scales: mlx.mlx_array,
    biases: mlx.mlx_array,
    group_size: u32,
    bits: u8,
) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_dequantize(
        &out,
        q,
        scales,
        biases,
        mlx.mlx_optional_int.some(@intCast(group_size)),
        mlx.mlx_optional_int.some(@intCast(bits)),
        "affine",
        .{}, // global_scale (null)
        .{ .value = mlx.mlx_array_dtype(scales), .has_value = true },
        s,
    ));
    return out;
}

// ── TBQ: rotated fixed-codebook quantization ──
//
// Port of the llama.cpp-turboq-mtp TBQ4_0 / TBQ3_0 block format
// (ref/ggml-turboq.c) onto MLX array ops. The last axis is split into
// 128-wide blocks; each block is unit-normalized, rotated by
// `diag(s2) · H128/√128 · diag(s1)` (4-bit) or the same WITHOUT the
// √128 factor (3-bit — its codebook is fit to the unnormalized domain and
// the dequant multiplies it back on the output), snapped to the Lloyd-Max
// centroids by counting midpoint boundaries (ties round UP, matching the
// C `if (v < mp[j]) idx = j` chain), then packed:
//   * 4-bit: one byte per element PAIR — even element in the low nibble.
//   * 3-bit: eight 3-bit indices per little-endian 24-bit group.
// Storage for input `[..., D]`:
//   q      : `[..., D/2]` (4-bit) / `[..., D*3/8]` (3-bit) uint8
//   scales : `[..., D/128]` float16 — the block scale `d`
//   biases : shallow refcount copy of scales. TBQ has no bias term, but
//            every 6-buffer cache path (grow / COW / disk / snapshot)
//            writes and grows one, so `d` is billed twice in bytes.
// The rotation is a dense ±1 matmul, so accumulation order differs from
// the C butterfly — near-tie elements can land in the neighboring bucket.
// RMSE is preserved; bit-exactness with the fork's files is NOT.

const tbq_tables = @import("tbq_tables.zig");

fn tbqHost(data: ?*const anyopaque, shape: []const c_int, dt: mlx.mlx_dtype) mlx.mlx_array {
    return mlx.mlx_array_new_data(data, shape.ptr, @intCast(shape.len), dt);
}

/// 8^j, j = 0..7 — the 3-bit field weights (8·3 = 24 bits, no carry).
const pow8_u32: [8]u32 = blk: {
    var w: [8]u32 = .{ 1, 1, 1, 1, 1, 1, 1, 1 };
    for (0..8) |j| w[j] = std.math.powi(u32, 8, @intCast(j)) catch unreachable;
    break :blk w;
};

/// Module-lifetime host scalars wrapped non-owningly by `tbqHost`. The lazy
/// graph captures the pointed-to data, so these must outlive any single call
/// — never wrap function-local storage in a host array.
const tbq_u8_16: u8 = 16;
const tbq_u32_256: u32 = 256;
const tbq_u32_65536: u32 = 65536;
const tbq_u32_8: u32 = 8;
const tbq_w3_u32: [3]u32 = .{ 1, 256, 65536 };

const TBQTables = struct {
    centroids: []const f32,
    midpoints: []const f32,
    signs1: []const f32,
    signs2: []const f32,
    /// Index-code width (3 or 4) — the pack layout selector; the codebook
    /// length is 2^bits, so `centroids.len` is NOT it.
    bits: u8,
    /// True for 4-bit (codebook fit to the normalized FWHT domain).
    normalized: bool,

    fn forBits(bits: u8) TBQTables {
        return switch (bits) {
            4 => .{
                .centroids = &tbq_tables.centroids4,
                .midpoints = &tbq_tables.midpoints4,
                .signs1 = &tbq_tables.signs1_4,
                .signs2 = &tbq_tables.signs2_4,
                .bits = 4,
                .normalized = true,
            },
            3 => .{
                .centroids = &tbq_tables.centroids3,
                .midpoints = &tbq_tables.midpoints3,
                .signs1 = &tbq_tables.signs1_3,
                .signs2 = &tbq_tables.signs2_3,
                .bits = 3,
                .normalized = false,
            },
            else => unreachable,
        };
    }
};

/// Folded query/output rotations for fused packed reads: row-major 128x128
/// f32, built at module load. Module-static so non-owning host wraps into
/// lazy graphs stay alive.
pub const TbqRot = struct { m1: [128 * 128]f32, m2: [128 * 128]f32 };

fn tbqRotBuild(s1: []const f32, s2: []const f32) TbqRot {
    var r: TbqRot = .{ .m1 = undefined, .m2 = undefined };
    @setEvalBranchQuota(100000); // built once at comptime: 2 x 128 x 128 products
    for (0..128) |i| for (0..128) |j| {
        const h = tbq_tables.h128[i * 128 + j];
        r.m1[i * 128 + j] = s1[i] * h * s2[j] * tbq_tables.inv_sqrt_128;
        r.m2[i * 128 + j] = s2[i] * h * s1[j] * tbq_tables.inv_sqrt_128;
    };
    return r;
}

// M1 folds the whole forward rotation (c -> c.s2 @H .s1 .isv): dot(q@M1, raw)
// equals dot(q, dequant(raw)). M2 = M1^T inverse-rotates the merged output,
// which the kernel accumulates against raw V codes.
const tbq_rot_4 = tbqRotBuild(&tbq_tables.signs1_4, &tbq_tables.signs2_4);
const tbq_rot_3 = tbqRotBuild(&tbq_tables.signs1_3, &tbq_tables.signs2_3);

pub fn tbqRot(bits: u8) *const TbqRot {
    return if (bits == 4) &tbq_rot_4 else if (bits == 3) &tbq_rot_3 else unreachable;
}

pub fn tbqCentroids(bits: u8) []const f32 {
    return if (bits == 4) &tbq_tables.centroids4 else &tbq_tables.centroids3;
}

/// Rotated-codebook quantize `[B, H, T, D]` (D a multiple of 128).
pub fn quantizeTBQ(s: mlx.mlx_stream, dense_x: mlx.mlx_array, bits: u8) !QuantizedKV {
    if (bits != 3 and bits != 4) return error.UnsupportedKvBits;
    return quantizeTBQInner(s, dense_x, TBQTables.forBits(bits));
}

/// Rotated-codebook dequantize a `(q, d)` pair to `out_dtype`, in the
/// ORIGINAL (pre-rotation) domain — callers (SDPA, QSA) cannot tell the
/// difference from a dense cache.
pub fn dequantizeTBQ(
    s: mlx.mlx_stream,
    q: mlx.mlx_array,
    scales: mlx.mlx_array,
    bits: u8,
    out_dtype: mlx.mlx_dtype,
) !mlx.mlx_array {
    if (bits != 3 and bits != 4) return error.UnsupportedKvBits;
    return dequantizeTBQInner(s, q, scales, TBQTables.forBits(bits), out_dtype);
}

fn quantizeTBQInner(
    s: mlx.mlx_stream,
    dense_x: mlx.mlx_array,
    t: TBQTables,
) !QuantizedKV {
    const shape = mlx.getShape(dense_x);
    if (shape.len != 4) return error.UnexpectedKVShape;
    const B = shape[0];
    const Hh = shape[1];
    const T = shape[2];
    const D = shape[3];
    if (D <= 0 or @mod(D, 128) != 0) return error.UnsupportedHeadDim;
    const BLK: c_int = @divExact(D, 128);
    const nc: c_int = @intCast(t.centroids.len);
    const bits = t.bits;

    const sh5 = [_]c_int{ B, Hh, T, BLK, 128 };
    const sh_d = [_]c_int{ B, Hh, T, BLK };
    const sh_tab = [_]c_int{ 1, 1, 1, 1, nc };
    const sh_sign = [_]c_int{ 1, 1, 1, 1, 128 };
    const sh_h = [_]c_int{ 128, 128 };
    const sh_one = [_]c_int{1};
    const packed_dim: c_int = if (bits == 4) BLK * 64 else BLK * 48;
    const sh_q = [_]c_int{ B, Hh, T, packed_dim };

    const hd = tbqHost(&tbq_tables.h128, &sh_h, .float32);
    defer _ = mlx.mlx_array_free(hd);
    const s1 = tbqHost(t.signs1.ptr, &sh_sign, .float32);
    defer _ = mlx.mlx_array_free(s1);
    const s2w = tbqHost(t.signs2.ptr, &sh_sign, .float32);
    defer _ = mlx.mlx_array_free(s2w);
    const ctab = tbqHost(t.centroids.ptr, &sh_tab, .float32);
    defer _ = mlx.mlx_array_free(ctab);

    var x32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x32);
    try mlx.check(mlx.mlx_astype(&x32, dense_x, .float32, s));
    var xr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xr);
    try mlx.check(mlx.mlx_reshape(&xr, x32, &sh5, 5, s));

    // Block L2 norm, kept as [.., BLK, 1] for broadcasting.
    var xsq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xsq);
    try mlx.check(mlx.mlx_multiply(&xsq, xr, xr, s));
    var n2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(n2);
    try mlx.check(mlx.mlx_sum_axis(&n2, xsq, -1, true, s));
    var norm = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(norm);
    try mlx.check(mlx.mlx_sqrt(&norm, n2, s));

    // t = x / norm; an all-zero block goes to exactly zero (C reference:
    // `inv_norm = norm > 1e-10 ? 1/norm : 0` ⇒ v = 0 ⇒ idx = 8, d = 0).
    // TBQ3 instead clamps the norm itself, faithful to its quantize_row_tbq3.
    var tvec = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tvec);
    var zero = mlx.mlx_array_new();
    if (t.normalized) {
        const eps = mlx.mlx_array_new_float(1e-38);
        defer _ = mlx.mlx_array_free(eps);
        var safe = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(safe);
        try mlx.check(mlx.mlx_maximum(&safe, norm, eps, s));
        try mlx.check(mlx.mlx_divide(&tvec, xr, safe, s));
        try mlx.check(mlx.mlx_zeros(&zero, &sh5, 5, .float32, s));
        const thr = mlx.mlx_array_new_float(1e-10);
        defer _ = mlx.mlx_array_free(thr);
        var cond = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cond);
        try mlx.check(mlx.mlx_greater(&cond, norm, thr, s));
        var picked = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_where(&picked, cond, tvec, zero, s));
        _ = mlx.mlx_array_free(tvec);
        tvec = picked;
    } else {
        const thr = mlx.mlx_array_new_float(1e-10);
        defer _ = mlx.mlx_array_free(thr);
        var clamped = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(clamped);
        try mlx.check(mlx.mlx_maximum(&clamped, norm, thr, s));
        try mlx.check(mlx.mlx_divide(&tvec, xr, clamped, s));
    }
    defer _ = mlx.mlx_array_free(zero);

    // Forward rotation v = ((t · s1) @ H) [· 1/√128 when normalized].
    var u = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(u);
    try mlx.check(mlx.mlx_multiply(&u, tvec, s1, s));
    var v = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v);
    try mlx.check(mlx.mlx_matmul(&v, u, hd, s));
    if (t.normalized) {
        const is = mlx.mlx_array_new_float(tbq_tables.inv_sqrt_128);
        defer _ = mlx.mlx_array_free(is);
        var vn = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&vn, v, is, s));
        _ = mlx.mlx_array_free(v);
        v = vn;
    }
    // The second sign flip applies to BOTH schemes (C: s1 → fwht → s2;
    // inverse: s2 → fwht → s1), so the codebook domain is ((t·s1)@H·[is])·s2.
    {
        var vs = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_multiply(&vs, v, s2w, s));
        _ = mlx.mlx_array_free(v);
        v = vs;
    }

    // Codebook index: idx = #{ j : v >= midpoint[j] }, ascending boundaries.
    var idx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(idx);
    try mlx.check(mlx.mlx_zeros(&idx, &sh5, 5, .float32, s));
    for (t.midpoints) |mp| {
        const thr = mlx.mlx_array_new_float(mp);
        defer _ = mlx.mlx_array_free(thr);
        var ge = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ge);
        try mlx.check(mlx.mlx_greater_equal(&ge, v, thr, s));
        var in_ = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_add(&in_, idx, ge, s));
        _ = mlx.mlx_array_free(idx);
        idx = in_;
    }
    var idx_u32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(idx_u32);
    try mlx.check(mlx.mlx_astype(&idx_u32, idx, .uint32, s));

    // Scale d = ‖x‖ / ‖recon(unit)‖ so the codebook vector is exact up to
    // one fp16 rounding. Both divisions clamp; 4-bit falls back to `norm`
    // when the reconstruction is degenerate (matches quantize_row_tbq4).
    var r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r);
    try mlx.check(mlx.mlx_take_along_axis(&r, ctab, idx_u32, -1, s));
    var rsq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rsq);
    var rr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rr);
    try mlx.check(mlx.mlx_multiply(&rr, r, r, s));
    try mlx.check(mlx.mlx_sum_axis(&rsq, rr, -1, true, s));
    var rn = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rn);
    try mlx.check(mlx.mlx_sqrt(&rn, rsq, s));
    var d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(d);
    const dnorm = mlx.mlx_array_new_float(1e-10);
    defer _ = mlx.mlx_array_free(dnorm);
    var rn_clamped = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rn_clamped);
    try mlx.check(mlx.mlx_maximum(&rn_clamped, rn, dnorm, s));
    if (t.normalized) {
        // d = norm / rn, except rn ≈ 0 → d = norm.
        var q_ = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(q_);
        try mlx.check(mlx.mlx_divide(&q_, norm, rn_clamped, s));
        var cond = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cond);
        try mlx.check(mlx.mlx_greater(&cond, rn, dnorm, s));
        var picked = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_where(&picked, cond, q_, norm, s));
        _ = mlx.mlx_array_free(d);
        d = picked;
        _ = mlx.mlx_array_free(q_);
    } else {
        var n_clamped = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(n_clamped);
        try mlx.check(mlx.mlx_maximum(&n_clamped, norm, dnorm, s));
        try mlx.check(mlx.mlx_divide(&d, n_clamped, rn_clamped, s));
    }
    var d16 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(d16);
    try mlx.check(mlx.mlx_astype(&d16, d, .float16, s));
    var dsh = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(dsh);
    try mlx.check(mlx.mlx_reshape(&dsh, d16, &sh_d, 4, s));

    // Pack indices.
    var idx_u8 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(idx_u8);
    try mlx.check(mlx.mlx_astype(&idx_u8, idx_u32, .uint8, s));
    var q = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(q);
    if (bits == 4) {
        // [.., BLK, 64, 2]: q[..., j] = idx[2j] | idx[2j+1] << 4.
        const sh6 = [_]c_int{ B, Hh, T, BLK, 64, 2 };
        const start_l = [_]c_int{ 0, 0, 0, 0, 0, 0 };
        const stop_l = [_]c_int{ B, Hh, T, BLK, 64, 1 };
        const start_h = [_]c_int{ 0, 0, 0, 0, 0, 1 };
        const stop_h = [_]c_int{ B, Hh, T, BLK, 64, 2 };
        const str_l = [_]c_int{ 1, 1, 1, 1, 1, 2 };
        const s16 = tbqHost(&tbq_u8_16, &sh_one, .uint8);
        defer _ = mlx.mlx_array_free(s16);
        var ir = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ir);
        try mlx.check(mlx.mlx_reshape(&ir, idx_u8, &sh6, 6, s));
        var lo = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(lo);
        try mlx.check(mlx.mlx_slice(&lo, ir, &start_l, 6, &stop_l, 6, &str_l, 6, s));
        var hi = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(hi);
        try mlx.check(mlx.mlx_slice(&hi, ir, &start_h, 6, &stop_h, 6, &str_l, 6, s));
        var him = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(him);
        try mlx.check(mlx.mlx_multiply(&him, hi, s16, s));
        var sum = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_add(&sum, lo, him, s));
        try mlx.check(mlx.mlx_reshape(&q, sum, &sh_q, 4, s));
        _ = mlx.mlx_array_free(sum);
    } else {
        // [.., BLK, 16, 8] × 8^j → 24-bit groups → 3 LE bytes.
        const sh6 = [_]c_int{ B, Hh, T, BLK, 16, 8 };
        const sh_w8 = [_]c_int{ 1, 1, 1, 1, 1, 8 };
        const w8 = tbqHost(&pow8_u32, &sh_w8, .uint32);
        defer _ = mlx.mlx_array_free(w8);
        const s256 = tbqHost(&tbq_u32_256, &sh_one, .uint32);
        defer _ = mlx.mlx_array_free(s256);
        const s65536 = tbqHost(&tbq_u32_65536, &sh_one, .uint32);
        defer _ = mlx.mlx_array_free(s65536);
        var ir = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ir);
        try mlx.check(mlx.mlx_reshape(&ir, idx_u8, &sh6, 6, s));
        var iru = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(iru);
        try mlx.check(mlx.mlx_astype(&iru, ir, .uint32, s));
        var gmul = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gmul);
        try mlx.check(mlx.mlx_multiply(&gmul, iru, w8, s));
        var gsum = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gsum);
        try mlx.check(mlx.mlx_sum_axis(&gsum, gmul, -1, false, s));
        var b0 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(b0);
        try mlx.check(mlx.mlx_remainder(&b0, gsum, s256, s));
        var t1 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(t1);
        try mlx.check(mlx.mlx_floor_divide(&t1, gsum, s256, s));
        var b1 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(b1);
        try mlx.check(mlx.mlx_remainder(&b1, t1, s256, s));
        var b2 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(b2);
        try mlx.check(mlx.mlx_floor_divide(&b2, t1, s256, s));
        var b0u8 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(b0u8);
        try mlx.check(mlx.mlx_astype(&b0u8, b0, .uint8, s));
        var b1u8 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(b1u8);
        try mlx.check(mlx.mlx_astype(&b1u8, b1, .uint8, s));
        var b2u8 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(b2u8);
        try mlx.check(mlx.mlx_astype(&b2u8, b2, .uint8, s));
        var e0 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(e0);
        try mlx.check(mlx.mlx_expand_dims(&e0, b0u8, -1, s));
        var e1 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(e1);
        try mlx.check(mlx.mlx_expand_dims(&e1, b1u8, -1, s));
        var e2 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(e2);
        try mlx.check(mlx.mlx_expand_dims(&e2, b2u8, -1, s));
        const vec = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{ e0, e1, e2 }, 3);
        defer _ = mlx.mlx_vector_array_free(vec);
        var cat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cat);
        try mlx.check(mlx.mlx_concatenate_axis(&cat, vec, -1, s));
        // cat has shape [B,H,T,BLK,16,3]; verify against sh6_3 for sanity.
        const cat_shape = mlx.getShape(cat);
        if (cat_shape.len != 6 or cat_shape[5] != 3) return error.UnexpectedQuantizeOutput;
        try mlx.check(mlx.mlx_reshape(&q, cat, &sh_q, 4, s));
    }

    // The 6-buffer contract wants a biases array; d is its own (borrowed).
    var biases = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(biases);
    try mlx.check(mlx.mlx_array_set(&biases, dsh));

    return .{ .q = q, .scales = dsh, .biases = biases };
}

fn dequantizeTBQInner(
    s: mlx.mlx_stream,
    q: mlx.mlx_array,
    d16: mlx.mlx_array,
    t: TBQTables,
    out_dtype: mlx.mlx_dtype,
) !mlx.mlx_array {
    const qs = mlx.getShape(q);
    if (qs.len != 4) return error.UnexpectedKVShape;
    const B = qs[0];
    const Hh = qs[1];
    const T = qs[2];
    const P = qs[3];
    const bits = t.bits;
    const D: c_int = if (bits == 4) P * 2 else @divExact(P * 8, 3);
    if (D <= 0 or @mod(D, 128) != 0) return error.UnsupportedHeadDim;
    const BLK: c_int = @divExact(D, 128);
    const nc: c_int = @intCast(t.centroids.len);

    const sh5 = [_]c_int{ B, Hh, T, BLK, 128 };
    const sh_out = [_]c_int{ B, Hh, T, D };
    const sh_tab = [_]c_int{ 1, 1, 1, 1, nc };
    const sh_sign = [_]c_int{ 1, 1, 1, 1, 128 };
    const sh_h = [_]c_int{ 128, 128 };
    const sh_one = [_]c_int{1};

    const hd = tbqHost(&tbq_tables.h128, &sh_h, .float32);
    defer _ = mlx.mlx_array_free(hd);
    const s1 = tbqHost(t.signs1.ptr, &sh_sign, .float32);
    defer _ = mlx.mlx_array_free(s1);
    const s2 = tbqHost(t.signs2.ptr, &sh_sign, .float32);
    defer _ = mlx.mlx_array_free(s2);
    const ctab = tbqHost(t.centroids.ptr, &sh_tab, .float32);
    defer _ = mlx.mlx_array_free(ctab);

    // Unpack indices → [B, H, T, BLK, 128] uint32 (row-major (j, lane)
    // relabels to element 2j + lane for 4-bit, 8g + lane for 3-bit).
    var idx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(idx);
    if (bits == 4) {
        // q byte j holds idx[2j] in the low nibble, idx[2j+1] in the high.
        // Reshape the byte axis to [64,1] (element-preserving), split the
        // two nibbles by rem/floor-div, then concat back to [64,2] so the
        // [BLK,128] row-major relabel yields idx[2j], idx[2j+1] in order.
        const sh6e = [_]c_int{ B, Hh, T, BLK, 64, 1 };
        const s16 = tbqHost(&tbq_u8_16, &sh_one, .uint8);
        defer _ = mlx.mlx_array_free(s16);
        var qr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(qr);
        try mlx.check(mlx.mlx_reshape(&qr, q, &sh6e, 6, s));
        var nib = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(nib);
        try mlx.check(mlx.mlx_remainder(&nib, qr, s16, s));
        var byt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(byt);
        try mlx.check(mlx.mlx_floor_divide(&byt, qr, s16, s));
        const vec = mlx.mlx_vector_array_new_data(&[_]mlx.mlx_array{ nib, byt }, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        var cat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cat);
        try mlx.check(mlx.mlx_concatenate_axis(&cat, vec, -1, s)); // [.., 64, 2]
        var idx5u8 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(idx5u8);
        try mlx.check(mlx.mlx_reshape(&idx5u8, cat, &sh5, 5, s));
        try mlx.check(mlx.mlx_astype(&idx, idx5u8, .uint32, s));
    } else {
        // Little-endian 24-bit groups → eight 3-bit fields per group.
        const sh6 = [_]c_int{ B, Hh, T, BLK, 16, 3 };
        const sh6_3 = [_]c_int{ 1, 1, 1, 1, 1, 3 };
        const sh6_8 = [_]c_int{ 1, 1, 1, 1, 1, 8 };
        const s8 = tbqHost(&tbq_u32_8, &sh_one, .uint32);
        defer _ = mlx.mlx_array_free(s8);
        const w3b = tbqHost(&tbq_w3_u32, &sh6_3, .uint32);
        defer _ = mlx.mlx_array_free(w3b);
        const w8 = tbqHost(&pow8_u32, &sh6_8, .uint32);
        defer _ = mlx.mlx_array_free(w8);
        var qr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(qr);
        try mlx.check(mlx.mlx_reshape(&qr, q, &sh6, 6, s));
        var qru = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(qru);
        try mlx.check(mlx.mlx_astype(&qru, qr, .uint32, s));
        var gmul = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gmul);
        try mlx.check(mlx.mlx_multiply(&gmul, qru, w3b, s));
        var gsum = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gsum);
        try mlx.check(mlx.mlx_sum_axis(&gsum, gmul, -1, false, s)); // [.., 16]
        var eb = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(eb);
        try mlx.check(mlx.mlx_expand_dims(&eb, gsum, -1, s)); // [.., 16, 1]
        var shifted = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(shifted);
        try mlx.check(mlx.mlx_floor_divide(&shifted, eb, w8, s)); // [.., 16, 8]
        var idx6 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(idx6);
        try mlx.check(mlx.mlx_remainder(&idx6, shifted, s8, s));
        try mlx.check(mlx.mlx_reshape(&idx, idx6, &sh5, 5, s)); // uint32 [.., 128]
    }

    // Inverse rotation, original domain: ((c[idx] · s2) @ H) · s1 · (1/√128) · d.
    var r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(r);
    try mlx.check(mlx.mlx_take_along_axis(&r, ctab, idx, -1, s));
    var rot = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rot);
    try mlx.check(mlx.mlx_multiply(&rot, r, s2, s));
    var inv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(inv);
    try mlx.check(mlx.mlx_matmul(&inv, rot, hd, s));
    var inv1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(inv1);
    try mlx.check(mlx.mlx_multiply(&inv1, inv, s1, s));
    const isv = mlx.mlx_array_new_float(tbq_tables.inv_sqrt_128);
    defer _ = mlx.mlx_array_free(isv);
    var inv2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(inv2);
    try mlx.check(mlx.mlx_multiply(&inv2, inv1, isv, s));
    var dd = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dd);
    try mlx.check(mlx.mlx_expand_dims(&dd, d16, -1, s));
    var scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scaled);
    try mlx.check(mlx.mlx_multiply(&scaled, inv2, dd, s));
    var out4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(out4);
    try mlx.check(mlx.mlx_reshape(&out4, scaled, &sh_out, 4, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_astype(&out, out4, out_dtype, s));
    return out;
}

// ── Fused quant-attention path (opt-in via the request's `kv_attn_mode: "fused"`) ──
//
// `quantAttention` reads K and V directly from their quantized triples,
// avoiding the dense materialization that `KVCache.denseView` performs in
// the default path. The fusion comes from `mlx_quantized_matmul`, Apple's
// kernel that internally dequantizes one operand and multiplies in a
// single Metal pass — same primitive `qmatmulBits` already uses for weight
// quantization throughout `transformer.zig`.
//
// Tradeoff vs `mlx_fast_scaled_dot_product_attention` (flash-attention):
//   * Dense SDPA fuses Q@K^T → scale → mask → softmax → @V into a single
//     tiled pass with no intermediate HBM writes. Wins at short context.
//   * Hand-rolled qmm × 2 + softmax loses tile-level fusion but skips the
//     dense K/V materialization. Wins at long context where K/V bandwidth
//     dominates.
// The crossover is data-driven; v1 ships behind `kv_attn_mode: "fused"` so the
// default is unchanged.
//
// Shape contract:
//   q_dense      : [B, H,    T_q, D] bf16 (Q already scaled or not; we apply scale below)
//   k_q/sc/bi    : K affine-quantized along last axis (D). Shape of the
//                  triple is whatever `mlx_quantize` produced.
//   v_q/sc/bi    : V same as K.
// GQA: when H > H_kv, Q is RESHAPED down to H_kv groups of g·T_q rows
// (never the triples expanded up to H_q — that materializes the packed
// bank ×g per layer per token). See the body comment in `quantAttention`.

/// A read-only borrow of a `QuantizedKV` triple — the cache owns the
/// arrays; the call site borrows them for the duration of one attention.
/// Used to thread quant triples through `DenseKVView` without altering
/// refcount semantics.
pub const BorrowedTriple = struct {
    q: mlx.mlx_array,
    scales: mlx.mlx_array,
    biases: mlx.mlx_array,
};

/// Hand-rolled attention that consumes K and V triples directly.
/// `scale` is the standard 1/sqrt(D) factor SDPA applies before softmax.
/// `mask_mode` mirrors `mlx_fast_scaled_dot_product_attention`:
///   * "causal": apply a Q-aligned causal mask (upper triangular -inf).
///     For T_q == 1 (decode tick) this is a no-op.
///   * "":       no mask.
///   * "array":  add `mask_arr` to the pre-softmax scores. Must be
///               additive (mlx convention: -inf for masked positions).
/// Returns a `[B, H, T_q, D]` bf16 array; caller owns and frees.
pub fn quantAttention(
    q_dense: mlx.mlx_array,
    k_triple: BorrowedTriple,
    v_triple: BorrowedTriple,
    bits: u8,
    group_size: u32,
    scale: f32,
    mask_mode: []const u8,
    mask_arr: mlx.mlx_array,
    s: mlx.mlx_stream,
) !mlx.mlx_array {
    // GQA handling (docs/kv-quant-perf.md Phase 1). mlx_quantized_matmul
    // does not broadcast across the head dim the way
    // mlx_fast_scaled_dot_product_attention does, and expanding the K/V
    // triples to H_q materializes the packed bank ×(H_q/H_kv) per layer per
    // token (mlx_quantized_matmul makes non-contiguous operands contiguous
    // on read). Instead we reshape Q the other way:
    //   [B, H_q, T_q, D] → [B, H_kv, g·T_q, D]   where g = H_q / H_kv.
    // Pure relabel in logical row-major order: element (b, hkv·g+gi, t, d)
    // maps to row r = gi·T_q + t of block hkv. The packed triples are then
    // read IN PLACE with no expansion at all; the output reshapes back.
    // Masking under grouped rows: row r is query position r % T_q, so a
    // [T_q, T_k] mask TILED g times along rows (g-major, matching the row
    // order above) masks every group identically.
    const q_shape = mlx.getShape(q_dense);
    if (q_shape.len < 4) return error.UnexpectedQShape;
    const k_q_shape = mlx.getShape(k_triple.q);
    if (k_q_shape.len < 4) return error.UnexpectedKShape;
    const B: c_int = q_shape[0];
    const H_q: c_int = q_shape[1];
    const T_q: c_int = q_shape[2];
    const D: c_int = q_shape[3];
    const H_kv: c_int = k_q_shape[1];
    if (H_kv <= 0 or @mod(H_q, H_kv) != 0) return error.UnsupportedGqaRatio;
    const g: c_int = @divExact(H_q, H_kv);

    // Group Q down to H_kv blocks. Borrow when g == 1 (no GQA).
    var q_used = q_dense;
    var owns_q = false;
    defer if (owns_q) {
        _ = mlx.mlx_array_free(q_used);
    };
    if (g > 1) {
        const gshape = [_]c_int{ B, H_kv, g * T_q, D };
        var qg = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&qg, q_dense, &gshape, 4, s));
        q_used = qg;
        owns_q = true;
    }

    // 1) scores = Q @ K^T. transpose_w=true contracts on K's quantized
    //    last axis (D), the same dim Q contracts. Output: [B, H_kv, g·T_q, T_k].
    var scores = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scores);
    try mlx.check(mlx.mlx_quantized_matmul(
        &scores,
        q_used,
        k_triple.q,
        k_triple.scales,
        k_triple.biases,
        true,
        mlx.mlx_optional_int.some(@intCast(group_size)),
        mlx.mlx_optional_int.some(@intCast(bits)),
        "affine",
        s,
    ));

    // 2) scale. Fold into a single multiply rather than dividing inside
    //    softmax — saves one op per call.
    const scale_arr = mlx.mlx_array_new_float(scale);
    defer _ = mlx.mlx_array_free(scale_arr);
    var scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scaled);
    try mlx.check(mlx.mlx_multiply(&scaled, scores, scale_arr, s));

    // 3) mask. We build masked_scores into `pre_softmax` and free the
    //    intermediates at scope exit. The branches assign `pre_softmax`
    //    to either an owned array or a borrow of `scaled`; `owns_pre`
    //    tracks which.
    var pre_softmax: mlx.mlx_array = .{};
    var owns_pre = false;
    defer {
        if (owns_pre) _ = mlx.mlx_array_free(pre_softmax);
    }
    if (std.mem.eql(u8, mask_mode, "causal")) {
        // Causal mask over the UNGROUPED query axis: T_q comes from the
        // caller's Q shape, never from the (possibly grouped) scores rows.
        // For decode (T_q == 1) every grouped row is the same position and
        // the mask is identically zero — skip building it.
        const scores_shape = mlx.getShape(scaled);
        if (scores_shape.len < 2) return error.UnexpectedShape;
        const t_k: c_int = scores_shape[scores_shape.len - 1];
        if (T_q == 1) {
            pre_softmax = scaled;
        } else {
            // Build a `[T_q, T_k]` additive mask: -inf above the diagonal
            // anchored to the right edge (Q position 0 corresponds to
            // K position (T_k - T_q)).
            const offset: c_int = t_k - T_q;
            const shape2 = [_]c_int{ T_q, t_k };
            var ones2 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ones2);
            try mlx.check(mlx.mlx_ones(&ones2, &shape2, 2, .bfloat16, s));
            var upper = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(upper);
            // triu(ones, k=offset+1) yields 1s strictly above the causal
            // diagonal (positions Q can't see) and 0s on/below.
            try mlx.check(mlx.mlx_triu(&upper, ones2, offset + 1, s));
            const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
            defer _ = mlx.mlx_array_free(neg_inf);
            var neg_inf_bf16 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(neg_inf_bf16);
            try mlx.check(mlx.mlx_astype(&neg_inf_bf16, neg_inf, .bfloat16, s));
            // mask = where(upper, -inf, 0). NEVER `upper * -inf`: the
            // below-diagonal zeros make 0 x -inf = NaN, which poisons the
            // whole softmax (live symptom: gemma answered "<pad><pad>" —
            // caught only once the parity loops became NaN-aware).
            const half = mlx.mlx_array_new_float(0.5);
            defer _ = mlx.mlx_array_free(half);
            var upper_bool = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(upper_bool);
            try mlx.check(mlx.mlx_greater(&upper_bool, upper, half, s));
            const zero_f = mlx.mlx_array_new_float(0.0);
            defer _ = mlx.mlx_array_free(zero_f);
            var zero_bf16 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(zero_bf16);
            try mlx.check(mlx.mlx_astype(&zero_bf16, zero_f, .bfloat16, s));
            var add_mask = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(add_mask);
            try mlx.check(mlx.mlx_where(&add_mask, upper_bool, neg_inf_bf16, zero_bf16, s));
            // Tile g-major along rows so every Q-group sees the same mask.
            var tiled_mask = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(tiled_mask);
            if (g > 1) {
                const reps = [_]c_int{ g, 1 };
                try mlx.check(mlx.mlx_tile(&tiled_mask, add_mask, &reps, 2, s));
            } else {
                try mlx.check(mlx.mlx_array_set(&tiled_mask, add_mask));
            }
            pre_softmax = mlx.mlx_array_new();
            owns_pre = true;
            try mlx.check(mlx.mlx_add(&pre_softmax, scaled, tiled_mask, s));
        }
    } else if (std.mem.eql(u8, mask_mode, "array")) {
        // External additive mask `[..., T_q, T_k]`. When Q is grouped and
        // T_q > 1 the mask's row axis must be tiled ×g (g-major). At
        // T_q == 1 the mask row broadcasts over the g grouped rows as-is.
        var mask_used = mask_arr;
        var owns_mask = false;
        defer if (owns_mask) {
            _ = mlx.mlx_array_free(mask_used);
        };
        if (g > 1 and T_q > 1) {
            const m_shape = mlx.getShape(mask_arr);
            if (m_shape.len < 2) return error.UnexpectedMaskShape;
            var reps_buf: [8]c_int = .{ 1, 1, 1, 1, 1, 1, 1, 1 };
            if (m_shape.len > reps_buf.len) return error.UnexpectedMaskShape;
            reps_buf[m_shape.len - 2] = g;
            var tm = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_tile(&tm, mask_arr, &reps_buf, m_shape.len, s));
            mask_used = tm;
            owns_mask = true;
        }
        pre_softmax = mlx.mlx_array_new();
        owns_pre = true;
        try mlx.check(mlx.mlx_add(&pre_softmax, scaled, mask_used, s));
    } else {
        // No mask: borrow `scaled` for the softmax input.
        pre_softmax = scaled;
    }

    // 4) softmax along last axis. `precise=true` matches the fast SDPA
    //    kernel's accumulation order (f32 reductions inside softmax) more
    //    closely than the default; trades a small perf hit for tighter
    //    equivalence to dense SDPA.
    var attn = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(attn);
    try mlx.check(mlx.mlx_softmax_axis(&attn, pre_softmax, -1, true, s));

    // 5) out = attn @ V. transpose_w=false contracts attn's last axis
    //    (T_k) with V's second-to-last (T_k) — V's quantized last axis
    //    (D) becomes the output dim. mlx_quantized_matmul dequantizes V
    //    on-the-fly without materializing a dense intermediate.
    var out_grouped = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(out_grouped);
    try mlx.check(mlx.mlx_quantized_matmul(
        &out_grouped,
        attn,
        v_triple.q,
        v_triple.scales,
        v_triple.biases,
        false,
        mlx.mlx_optional_int.some(@intCast(group_size)),
        mlx.mlx_optional_int.some(@intCast(bits)),
        "affine",
        s,
    ));

    // 6) Ungroup: [B, H_kv, g·T_q, D] → [B, H_q, T_q, D] (same relabel,
    //    inverted). No-op reshape when g == 1.
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    if (g > 1) {
        const oshape = [_]c_int{ B, H_q, T_q, D };
        try mlx.check(mlx.mlx_reshape(&out, out_grouped, &oshape, 4, s));
    } else {
        try mlx.check(mlx.mlx_array_set(&out, out_grouped));
    }
    return out;
}

// ── Tests ──

const testing = std.testing;

/// Build a `[1, 1, 8, head_dim]` bf16 tensor whose values vary smoothly per
/// position so quantization error is non-trivial but bounded.
fn buildSmoothBf16(s: mlx.mlx_stream, head_dim: c_int) !mlx.mlx_array {
    const T: usize = 8;
    const D: usize = @intCast(head_dim);
    const buf = try testing.allocator.alloc(f32, T * D);
    defer testing.allocator.free(buf);
    for (0..T) |t| {
        for (0..D) |d| {
            // Range roughly [-1, 1]; smooth so adjacent group elements are
            // close (best case for affine), with some variation across groups.
            const fi: f32 = @floatFromInt(t * D + d);
            const denom: f32 = @floatFromInt(T * D);
            buf[t * D + d] = (fi / denom) * 2.0 - 1.0;
        }
    }
    const shape = [_]c_int{ 1, 1, @intCast(T), head_dim };
    const f32_arr = mlx.mlx_array_new_data(buf.ptr, &shape, 4, .float32);
    defer _ = mlx.mlx_array_free(f32_arr);
    var bf16_arr = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&bf16_arr, f32_arr, .bfloat16, s));
    return bf16_arr;
}

/// Read a flat float32 host buffer for a small array (eval-and-copy via
/// `mlx_astype` to float32 then reshape to 1D).
fn readF32Flat(s: mlx.mlx_stream, arr: mlx.mlx_array, allocator: std.mem.Allocator) ![]f32 {
    var f32_view = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f32_view);
    try mlx.check(mlx.mlx_astype(&f32_view, arr, .float32, s));

    const n: c_int = @intCast(mlx.mlx_array_size(f32_view));
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    {
        const sh = [_]c_int{n};
        try mlx.check(mlx.mlx_reshape(&flat, f32_view, &sh, 1, s));
    }
    {
        const ev = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(ev);
        _ = mlx.mlx_vector_array_append_value(ev, flat);
        try mlx.check(mlx.mlx_eval(ev));
    }
    const ptr = mlx.mlx_array_data_float32(flat) orelse return error.NullData;
    const out = try allocator.alloc(f32, @intCast(n));
    @memcpy(out, ptr[0..@intCast(n)]);
    return out;
}

test "quantizeAffine + dequantizeAffine round-trip at 4 bits" {
    const s = mlx.gpuStream();
    const src = try buildSmoothBf16(s, 256);
    defer _ = mlx.mlx_array_free(src);

    var qkv = try quantizeAffine(s, src, 64, 4);
    defer qkv.deinit();

    // Shape sanity: q is [..., D * bits / 32] = [..., 256 * 4 / 32] = [..., 32]
    const q_shape = mlx.getShape(qkv.q);
    try testing.expectEqual(@as(c_int, 32), q_shape[q_shape.len - 1]);
    // scales/biases last dim = D / group_size = 256 / 64 = 4
    const sc_shape = mlx.getShape(qkv.scales);
    try testing.expectEqual(@as(c_int, 4), sc_shape[sc_shape.len - 1]);

    const deq = try dequantizeAffine(s, qkv.q, qkv.scales, qkv.biases, 64, 4);
    defer _ = mlx.mlx_array_free(deq);

    const orig = try readF32Flat(s, src, testing.allocator);
    defer testing.allocator.free(orig);
    const got = try readF32Flat(s, deq, testing.allocator);
    defer testing.allocator.free(got);

    try testing.expectEqual(orig.len, got.len);
    var max_err: f32 = 0;
    for (orig, got) |o, g| {
        try testing.expect(!std.math.isNan(g));
        const e = @abs(o - g);
        if (e > max_err) max_err = e;
    }
    // 4-bit affine on smooth data with group=64: empirical ceiling well under 0.05.
    try testing.expect(max_err < 0.05);
}

test "quantizeAffine + dequantizeAffine round-trip at 8 bits" {
    const s = mlx.gpuStream();
    const src = try buildSmoothBf16(s, 256);
    defer _ = mlx.mlx_array_free(src);

    var qkv = try quantizeAffine(s, src, 64, 8);
    defer qkv.deinit();

    // Shape: q = [..., 256 * 8 / 32] = [..., 64]
    const q_shape = mlx.getShape(qkv.q);
    try testing.expectEqual(@as(c_int, 64), q_shape[q_shape.len - 1]);

    const deq = try dequantizeAffine(s, qkv.q, qkv.scales, qkv.biases, 64, 8);
    defer _ = mlx.mlx_array_free(deq);

    const orig = try readF32Flat(s, src, testing.allocator);
    defer testing.allocator.free(orig);
    const got = try readF32Flat(s, deq, testing.allocator);
    defer testing.allocator.free(got);

    var max_err: f32 = 0;
    for (orig, got) |o, g| {
        try testing.expect(!std.math.isNan(g));
        const e = @abs(o - g);
        if (e > max_err) max_err = e;
    }
    // 8-bit affine: ~256x finer steps than 4-bit; expect < 0.005 on smooth data.
    try testing.expect(max_err < 0.01);
}

// ── TBQ (rotated fixed-codebook) tests ──

fn buildGaussianBf16(s: mlx.mlx_stream, t: c_int, d: c_int) !mlx.mlx_array {
    // Pseudo-Gaussian: mean of 4 LCG uniforms, scaled to ~unit variance —
    // the heavy-tailed-free regime the codebooks are fitted for.
    const T: usize = @intCast(t);
    const D: usize = @intCast(d);
    const buf = try testing.allocator.alloc(f32, T * D);
    defer testing.allocator.free(buf);
    var seed: u32 = 0x9E3779B9;
    for (0..T * D) |i| {
        var acc: f32 = 0;
        for (0..4) |_| {
            seed = seed *% 1664525 +% 1013904223;
            const u: f32 = @as(f32, @floatFromInt(seed >> 8)) / 8388608.0 - 1.0;
            acc += u;
        }
        buf[i] = acc * 0.577; // 4 uniforms: var 4/12 → normalize to ~1.33σ→≈0.77σ
    }
    const shape = [_]c_int{ 1, 1, t, d };
    const f32_arr = mlx.mlx_array_new_data(buf.ptr, &shape, 4, .float32);
    defer _ = mlx.mlx_array_free(f32_arr);
    var bf16_arr = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&bf16_arr, f32_arr, .bfloat16, s));
    return bf16_arr;
}

fn rmse(a: []const f32, b: []const f32) f32 {
    var sse: f64 = 0;
    for (a, b) |x, y| {
        const diff: f64 = @as(f64, x) - @as(f64, y);
        sse += diff * diff;
    }
    return @floatCast(@sqrt(sse / @as(f64, @floatFromInt(a.len))));
}

fn flatBytes(arr: mlx.mlx_array) ![]const u8 {
    try mlx.check(mlx.mlx_array_eval(arr));
    // The uint8 data accessor reinterprets the host buffer — enough for the
    // byte-identity checks (the f16 accessor's `f16` type is never declared
    // in mlx.zig).
    const p = mlx.mlx_array_data_uint8(arr) orelse return error.NullData;
    const n = mlx.mlx_array_size(arr) * mlx.mlx_array_itemsize(arr);
    return p[0..n];
}

test "tbq4: packed buffers keep the 6-slot contract" {
    const s = mlx.gpuStream();
    const src = try buildSmoothBf16(s, 256);
    defer _ = mlx.mlx_array_free(src);

    var qkv = try quantizeTBQ(s, src, 4);
    defer qkv.deinit();

    const q_shape = mlx.getShape(qkv.q);
    try testing.expectEqual(@as(c_int, 256 / 2), q_shape[q_shape.len - 1]);
    const sc_shape = mlx.getShape(qkv.scales);
    try testing.expectEqual(@as(c_int, 256 / 128), sc_shape[sc_shape.len - 1]);
    try testing.expectEqual(mlx.mlx_dtype.uint8, mlx.mlx_array_dtype(qkv.q));
    try testing.expectEqual(mlx.mlx_dtype.float16, mlx.mlx_array_dtype(qkv.scales));
    // biases is a shallow copy of scales: same buffer, same size.
    try testing.expectEqual(mlx.mlx_array_size(qkv.scales), mlx.mlx_array_size(qkv.biases));

    const deq = try dequantizeTBQ(s, qkv.q, qkv.scales, 4, .bfloat16);
    defer _ = mlx.mlx_array_free(deq);
    const orig = try readF32Flat(s, src, testing.allocator);
    defer testing.allocator.free(orig);
    const got = try readF32Flat(s, deq, testing.allocator);
    defer testing.allocator.free(got);
    try testing.expectEqual(orig.len, got.len);
    var max_err: f32 = 0;
    for (orig, got) |o, g| {
        try testing.expect(!std.math.isNan(g));
        const e = @abs(o - g);
        if (e > max_err) max_err = e;
    }
    // A DC-heavy linear ramp is near-1-sparse in the Hadamard domain; the
    // fixed codebook clips the dominant coordinate at ±0.2416 and d only
    // restores the block norm — measured max abs err ≈ 0.24 (numpy mirror).
    // This test's job is the 6-slot contract (shapes, dtypes, no NaN).
    try testing.expect(max_err < 0.26);
}

test "tbq: RMSE on pseudo-Gaussian KV — tbq4 near-parity with affine4, tbq3 degrades gracefully" {
    const s = mlx.gpuStream();
    const src = try buildGaussianBf16(s, 64, 256);
    defer _ = mlx.mlx_array_free(src);
    const orig = try readF32Flat(s, src, testing.allocator);
    defer testing.allocator.free(orig);

    var aq = try quantizeAffine(s, src, 64, 4);
    defer aq.deinit();
    const adeq = try dequantizeAffine(s, aq.q, aq.scales, aq.biases, 64, 4);
    defer _ = mlx.mlx_array_free(adeq);
    const agot = try readF32Flat(s, adeq, testing.allocator);
    defer testing.allocator.free(agot);

    var t4 = try quantizeTBQ(s, src, 4);
    defer t4.deinit();
    const d4 = try dequantizeTBQ(s, t4.q, t4.scales, 4, .bfloat16);
    defer _ = mlx.mlx_array_free(d4);
    const g4 = try readF32Flat(s, d4, testing.allocator);
    defer testing.allocator.free(g4);

    var t3 = try quantizeTBQ(s, src, 3);
    defer t3.deinit();
    const d3 = try dequantizeTBQ(s, t3.q, t3.scales, 3, .bfloat16);
    defer _ = mlx.mlx_array_free(d3);
    const g3 = try readF32Flat(s, d3, testing.allocator);
    defer testing.allocator.free(g3);

    const r_affine = rmse(orig, agot);
    const r4 = rmse(orig, g4);
    const r3 = rmse(orig, g3);
    try testing.expect(!std.math.isNan(r4) and !std.math.isNan(r3));
    // Ground truth (numpy mirror of these exact formulas): on this bounded
    // LCG pseudo-Gaussian tbq4 = 0.0631, tbq3 = 0.1423, affine4 = 0.0569;
    // on a true Gaussian tbq4 = 0.0314 vs affine4 = 0.0294. Group min/max
    // affine is already near-optimal on well-behaved data — the rotated
    // fixed codebook wins in BYTES (4.25 vs 4.5 B/2-elem at 4-bit, 3.25 at
    // 3-bit), not in error at equal bits. Measured on this data: r_affine =
    // 0.0575, r4 = 0.0631, r3 = 0.1423 (r3/r4 = 2.25). Robust invariants:
    try testing.expect(r4 < r_affine * 1.2); // tbq4 ≈ parity
    try testing.expect(r3 > r4); // fewer levels → worse…
    try testing.expect(r3 < r4 * 2.7); // …but gracefully (measured 2.25-2.5)
    try testing.expect(r3 < r_affine * 3.0); // and beats affine-3 territory
}

test "tbq: all-zero head dimension dequantizes to zero, scale 0, idx 8" {
    const s = mlx.gpuStream();
    const shape = [_]c_int{ 1, 1, 4, 256 };
    const zeros = std.mem.zeroes([1024]f32);
    const f32_arr = mlx.mlx_array_new_data(&zeros, &shape, 4, .float32);
    defer _ = mlx.mlx_array_free(f32_arr);
    var src = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(src);
    try mlx.check(mlx.mlx_astype(&src, f32_arr, .bfloat16, s));

    var qkv = try quantizeTBQ(s, src, 4);
    defer qkv.deinit();
    const bytes = try flatBytes(qkv.q);
    for (bytes) |b| try testing.expectEqual(@as(u8, 0x88), b);
    const sbytes = try flatBytes(qkv.scales);
    for (sbytes) |b| try testing.expectEqual(@as(u8, 0), b);

    const deq = try dequantizeTBQ(s, qkv.q, qkv.scales, 4, .bfloat16);
    defer _ = mlx.mlx_array_free(deq);
    const got = try readF32Flat(s, deq, testing.allocator);
    defer testing.allocator.free(got);
    for (got) |v| try testing.expectApproxEqAbs(@as(f32, 0), v, 1e-6);
}

test "tbq: fixed-point pack/unpack is byte-identical through quant→dequant→quant" {
    const s = mlx.gpuStream();
    const src = try buildGaussianBf16(s, 16, 256);
    defer _ = mlx.mlx_array_free(src);

    var q1 = try quantizeTBQ(s, src, 4);
    defer q1.deinit();
    const dq = try dequantizeTBQ(s, q1.q, q1.scales, 4, .bfloat16);
    defer _ = mlx.mlx_array_free(dq);
    var q2 = try quantizeTBQ(s, dq, 4);
    defer q2.deinit();

    const b1 = try flatBytes(q1.q);
    const b2 = try flatBytes(q2.q);
    try testing.expectEqualSlices(u8, b1, b2);
    // Scales are not byte-identical: dequant stores d as f16 and the
    // bf16-rounded reconstruction shifts each block norm by ~0.4% (measured
    // ≤1 f16 ULP per 8 requant steps… actually ≤0.4% absolute). The pack
    // path itself is fixed-point — q bytes must match exactly.
    const sc1 = try readF32Flat(s, q1.scales, testing.allocator);
    defer testing.allocator.free(sc1);
    const sc2 = try readF32Flat(s, q2.scales, testing.allocator);
    defer testing.allocator.free(sc2);
    for (sc1, sc2) |a, b| {
        const tol = @max(@abs(a), @abs(b)) * 0.01 + 1e-3;
        try testing.expectApproxEqAbs(a, b, tol);
    }
}

test "tbq fused: M1 rotation preserves scores (tbq4, tbq3)" {
    // The fused decode/verify kernel's identity: with q_rot = q @ M1,
    // dot(q_rot, c·d) equals the dense dequant's dot(q, ((c∘s2)@H∘s1∘isv)·d)
    // for one 128-block — the kernel never leaves the rotated domain.
    for ([_]u8{ 4, 3 }) |bits| {
        const rot = tbqRot(bits);
        const tb = TBQTables.forBits(bits);
        var prng = std.Random.DefaultPrng.init(0xF0 + bits);
        const rnd = prng.random();
        var q: [128]f32 = undefined;
        for (&q) |*x| x.* = rnd.float(f32) * 2.0 - 1.0;
        var qrot: [128]f32 = @splat(0);
        for (0..128) |j| {
            var acc: f32 = 0;
            for (0..128) |e| acc += q[e] * rot.m1[e * 128 + j];
            qrot[j] = acc;
        }
        var codes: [128]f32 = undefined;
        for (&codes) |*x| x.* = tb.centroids[rnd.uintLessThan(usize, tb.centroids.len)];
        var s2c: [128]f32 = undefined;
        for (0..128) |j| s2c[j] = codes[j] * tb.signs2[j];
        const d: f32 = 0.25 + rnd.float(f32);
        var lhs: f32 = 0;
        var rhs: f32 = 0;
        for (0..128) |j| lhs += qrot[j] * codes[j];
        for (0..128) |e| {
            var inner: f32 = 0;
            for (0..128) |j| inner += s2c[j] * tbq_tables.h128[j * 128 + e];
            rhs += q[e] * inner * tbq_tables.inv_sqrt_128 * tb.signs1[e];
        }
        lhs *= d;
        rhs *= d;
        try testing.expectApproxEqAbs(lhs, rhs, 1e-3 * @max(1.0, @abs(rhs)));
    }
}

test "tbq fused: M2 inverts the rotation on outputs (tbq4, tbq3)" {
    // o_orig = o_rot @ M2, element-for-element with the dense dequant's
    // inverse-rotation formula; and the fold keeps (o@M2)@M1 == o.
    for ([_]u8{ 4, 3 }) |bits| {
        const rot = tbqRot(bits);
        const tb = TBQTables.forBits(bits);
        var prng = std.Random.DefaultPrng.init(0xBA + bits);
        const rnd = prng.random();
        var o: [128]f32 = undefined;
        for (&o) |*x| x.* = rnd.float(f32) * 2.0 - 1.0;
        var o2: [128]f32 = @splat(0);
        for (0..128) |e| {
            var acc: f32 = 0;
            for (0..128) |j| acc += o[j] * rot.m2[j * 128 + e];
            o2[e] = acc;
        }
        var want: [128]f32 = undefined;
        for (0..128) |e| {
            var inner: f32 = 0;
            for (0..128) |j| inner += o[j] * tb.signs2[j] * tbq_tables.h128[j * 128 + e];
            want[e] = inner * tbq_tables.inv_sqrt_128 * tb.signs1[e];
        }
        for (o2, want) |a, b| {
            try testing.expectApproxEqAbs(a, b, 1e-3 * @max(1.0, @abs(b)));
        }
        var back: [128]f32 = @splat(0);
        for (0..128) |i| {
            var acc: f32 = 0;
            for (0..128) |j| acc += o2[j] * rot.m1[j * 128 + i];
            back[i] = acc;
        }
        for (back, o) |a, b| {
            try testing.expectApproxEqAbs(a, b, 1e-2 * @max(1.0, @abs(b)));
        }
    }
}

test "a load's KV choice: --kv-quant > setting > the kv8 default, each named by its source" {
    const t = std.testing;
    try t.expectEqual(KVQuantConfig.affine(8), KVQuantConfig.engine_default);
    const unflagged = KvCacheChoice.resolve(null, KVQuantConfig.engine_default, false);
    try t.expectEqual(KVQuantConfig.affine(8), unflagged.config);
    try t.expectEqualStrings("kv8", unflagged.label());
    try t.expectEqualStrings("default", unflagged.sourceName());
    const flagged = KvCacheChoice.resolve(null, KVQuantConfig.affine(4), true);
    try t.expectEqualStrings("kv4", flagged.label());
    try t.expectEqualStrings("--kv-quant", flagged.sourceName());
    const flag_wins = KvCacheChoice.resolve(KVQuantConfig.dense, KVQuantConfig.affine(4), true);
    try t.expectEqual(KVQuantConfig.affine(4), flag_wins.config);
    try t.expectEqualStrings("--kv-quant", flag_wins.sourceName());
    const set = KvCacheChoice.resolve(KVQuantConfig.dense, KVQuantConfig.engine_default, false);
    try t.expectEqual(KVQuantConfig.dense, set.config);
    try t.expectEqualStrings("off", set.label());
    try t.expectEqualStrings("model-settings.json", set.sourceName());
}

test "KVQuantConfig.affine builds a sane config" {
    const c4 = KVQuantConfig.affine(4);
    try testing.expectEqual(Scheme.affine, c4.scheme);
    try testing.expectEqual(@as(u8, 4), c4.bits);
    try testing.expectEqual(@as(u32, 64), c4.group_size);

    const c8 = KVQuantConfig.affine(8);
    try testing.expectEqual(@as(u8, 8), c8.bits);

    const cd = KVQuantConfig.dense;
    try testing.expectEqual(Scheme.off, cd.scheme);
}

// ── Fused-attention validation ──
//
// The two tests below are the validation harness Phase 2 v1 relies on
// before wiring `quantAttention` into transformer.zig SDPA call sites.
// They prove (a) `mlx_quantized_matmul` semantics match dequant+matmul
// in both transpose modes, and (b) the assembled `quantAttention`
// produces logits within the same loose tolerance (0.05 max-abs-diff)
// the existing affine round-trip tests use.

/// Build a `[B, H, T, D]` dense bf16 with smooth ramped values per
/// `(b, h, t, d)` so quantization is non-trivial. Caller frees.
fn buildSmoothBHTD(s: mlx.mlx_stream, B: c_int, H: c_int, T: c_int, D: c_int) !mlx.mlx_array {
    const total: usize = @intCast(B * H * T * D);
    const buf = try testing.allocator.alloc(f32, total);
    defer testing.allocator.free(buf);
    var i: usize = 0;
    for (0..@intCast(B)) |b| {
        for (0..@intCast(H)) |h| {
            for (0..@intCast(T)) |t| {
                for (0..@intCast(D)) |d| {
                    const f: f32 = @floatFromInt(b * 17 + h * 5 + t * 3 + d);
                    const denom: f32 = @floatFromInt(total);
                    buf[i] = (f / denom) * 2.0 - 1.0;
                    i += 1;
                }
            }
        }
    }
    const shape = [_]c_int{ B, H, T, D };
    const f32_arr = mlx.mlx_array_new_data(buf.ptr, &shape, 4, .float32);
    defer _ = mlx.mlx_array_free(f32_arr);
    var bf16_arr = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&bf16_arr, f32_arr, .bfloat16, s));
    return bf16_arr;
}

test "mlx_quantized_matmul transpose=true matches dequant+matmul (4-bit, 4D)" {
    const s = mlx.gpuStream();
    // x: [1, 2, 3, 64]  (Q-shaped: B, H, T_q, D)
    // w: [1, 2, 5, 64]  (K-shaped: B, H_kv, T_k, D, will be quantized along D)
    // Expected: x @ w.T → [1, 2, 3, 5]
    const x = try buildSmoothBHTD(s, 1, 2, 3, 64);
    defer _ = mlx.mlx_array_free(x);
    const w = try buildSmoothBHTD(s, 1, 2, 5, 64);
    defer _ = mlx.mlx_array_free(w);

    var qw = try quantizeAffine(s, w, 64, 4);
    defer qw.deinit();

    // Reference: dequantize w, then dense matmul with explicit transpose.
    const w_deq = try dequantizeAffine(s, qw.q, qw.scales, qw.biases, 64, 4);
    defer _ = mlx.mlx_array_free(w_deq);
    // Transpose w_deq along last two dims: [1,2,5,64] → [1,2,64,5]
    var w_deq_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_deq_t);
    const axes_t = [_]c_int{ 0, 1, 3, 2 };
    try mlx.check(mlx.mlx_transpose_axes(&w_deq_t, w_deq, &axes_t, 4, s));
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    try mlx.check(mlx.mlx_matmul(&ref, x, w_deq_t, s));

    // Candidate: fused qmm with transpose=true.
    var cand = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cand);
    try mlx.check(mlx.mlx_quantized_matmul(
        &cand,
        x,
        qw.q,
        qw.scales,
        qw.biases,
        true,
        mlx.mlx_optional_int.some(64),
        mlx.mlx_optional_int.some(4),
        "affine",
        s,
    ));

    // Shape sanity.
    const ref_shape = mlx.getShape(ref);
    const cand_shape = mlx.getShape(cand);
    try testing.expectEqual(ref_shape.len, cand_shape.len);
    for (ref_shape, cand_shape) |r, c| try testing.expectEqual(r, c);

    const ref_flat = try readF32Flat(s, ref, testing.allocator);
    defer testing.allocator.free(ref_flat);
    const cand_flat = try readF32Flat(s, cand, testing.allocator);
    defer testing.allocator.free(cand_flat);
    var max_err: f32 = 0;
    var max_ref: f32 = 0;
    for (ref_flat, cand_flat) |r, c| {
        try testing.expect(!std.math.isNan(c));
        const e = @abs(r - c);
        if (e > max_err) max_err = e;
        if (@abs(r) > max_ref) max_ref = @abs(r);
    }
    // bf16 reductions inside qmm don't bit-match an explicit dequant +
    // mlx_matmul because the order of additions differs. Compare in
    // relative terms (the smooth-ramp test produces outputs in the
    // [-N*D*1.0, N*D*1.0] range so absolute thresholds are misleading).
    const rel_err = if (max_ref > 0) max_err / max_ref else max_err;
    if (rel_err >= 0.02) {
        std.debug.print("max_err={d} max_ref={d} rel_err={d}\n", .{ max_err, max_ref, rel_err });
    }
    try testing.expect(rel_err < 0.02);
}

test "mlx_quantized_matmul transpose=false matches dequant+matmul (4-bit, 4D)" {
    const s = mlx.gpuStream();
    // x: [1, 2, 3, 5]  (attn-shaped: B, H, T_q, T_k)
    // w: [1, 2, 5, 64] (V-shaped:    B, H_kv, T_k, D)
    // Expected: x @ w → [1, 2, 3, 64], contracting over T_k.
    // For V, the quantized last axis is D — so transpose=false should
    // dequantize each D-column on the fly while contracting over T_k.
    const x = try buildSmoothBHTD(s, 1, 2, 3, 5);
    defer _ = mlx.mlx_array_free(x);
    const w = try buildSmoothBHTD(s, 1, 2, 5, 64);
    defer _ = mlx.mlx_array_free(w);

    var qw = try quantizeAffine(s, w, 64, 4);
    defer qw.deinit();

    // Reference: dequant + plain matmul.
    const w_deq = try dequantizeAffine(s, qw.q, qw.scales, qw.biases, 64, 4);
    defer _ = mlx.mlx_array_free(w_deq);
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    try mlx.check(mlx.mlx_matmul(&ref, x, w_deq, s));

    var cand = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cand);
    try mlx.check(mlx.mlx_quantized_matmul(
        &cand,
        x,
        qw.q,
        qw.scales,
        qw.biases,
        false,
        mlx.mlx_optional_int.some(64),
        mlx.mlx_optional_int.some(4),
        "affine",
        s,
    ));

    const ref_shape = mlx.getShape(ref);
    const cand_shape = mlx.getShape(cand);
    try testing.expectEqual(ref_shape.len, cand_shape.len);
    for (ref_shape, cand_shape) |r, c| try testing.expectEqual(r, c);

    const ref_flat = try readF32Flat(s, ref, testing.allocator);
    defer testing.allocator.free(ref_flat);
    const cand_flat = try readF32Flat(s, cand, testing.allocator);
    defer testing.allocator.free(cand_flat);
    var max_err: f32 = 0;
    for (ref_flat, cand_flat) |r, c| {
        // NaN-blind comparisons pass on an all-NaN candidate (NaN > x is
        // false) — the composed causal arm shipped NaN for months behind
        // exactly this hole. Finiteness is asserted BEFORE the diff.
        try testing.expect(!std.math.isNan(c));
        const e = @abs(r - c);
        if (e > max_err) max_err = e;
    }
    try testing.expect(max_err < 0.05);
}

test "quantAttention matches dense SDPA at 4-bit (decode, T_q=1)" {
    const s = mlx.gpuStream();
    const B: c_int = 1;
    const H: c_int = 2;
    const T_k: c_int = 8;
    const D: c_int = 64;
    const q = try buildSmoothBHTD(s, B, H, 1, D);
    defer _ = mlx.mlx_array_free(q);
    const k_dense = try buildSmoothBHTD(s, B, H, T_k, D);
    defer _ = mlx.mlx_array_free(k_dense);
    const v_dense = try buildSmoothBHTD(s, B, H, T_k, D);
    defer _ = mlx.mlx_array_free(v_dense);

    var qk = try quantizeAffine(s, k_dense, 64, 4);
    defer qk.deinit();
    var qv = try quantizeAffine(s, v_dense, 64, 4);
    defer qv.deinit();

    // Dense reference. We pass the dequantized K/V so any mismatch
    // attributable to quantization itself is folded into both paths.
    const k_ref = try dequantizeAffine(s, qk.q, qk.scales, qk.biases, 64, 4);
    defer _ = mlx.mlx_array_free(k_ref);
    const v_ref = try dequantizeAffine(s, qv.q, qv.scales, qv.biases, 64, 4);
    defer _ = mlx.mlx_array_free(v_ref);

    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    const none_mask = mlx.mlx_array{ .ctx = null };
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(
        &ref,
        q,
        k_ref,
        v_ref,
        scale,
        "",
        none_mask,
        .{ .ctx = null },
        false,
        s,
    ));

    const cand = try quantAttention(
        q,
        .{ .q = qk.q, .scales = qk.scales, .biases = qk.biases },
        .{ .q = qv.q, .scales = qv.scales, .biases = qv.biases },
        4,
        64,
        scale,
        "",
        none_mask,
        s,
    );
    defer _ = mlx.mlx_array_free(cand);

    const ref_flat = try readF32Flat(s, ref, testing.allocator);
    defer testing.allocator.free(ref_flat);
    const cand_flat = try readF32Flat(s, cand, testing.allocator);
    defer testing.allocator.free(cand_flat);
    try testing.expectEqual(ref_flat.len, cand_flat.len);
    var max_err: f32 = 0;
    for (ref_flat, cand_flat) |r, c| {
        // NaN-blind comparisons pass on an all-NaN candidate (NaN > x is
        // false) — the composed causal arm shipped NaN for months behind
        // exactly this hole. Finiteness is asserted BEFORE the diff.
        try testing.expect(!std.math.isNan(c));
        const e = @abs(r - c);
        if (e > max_err) max_err = e;
    }
    // Matches the bf16 reduction tolerance used elsewhere in this file.
    try testing.expect(max_err < 0.05);
}

test "quantAttention reshapes Q for GQA instead of repeating K/V triples" {
    // Phase 1 (docs/kv-quant-perf.md): the GQA repeat materializes the packed
    // bank ×(H_q/H_kv) per layer per token — the reshape-Q form reads the
    // triples IN PLACE. Pin that the repeat op never comes back. The needle
    // is concatenated so this test's own source can't match it.
    const src = @embedFile("kv_quant.zig");
    const start = std.mem.indexOf(u8, src, "pub fn quantAttention").?;
    const end = std.mem.indexOfPos(u8, src, start, "// ── Tests ──").?;
    const body = src[start..end];
    try testing.expect(std.mem.indexOf(u8, body, "mlx_repeat" ++ "_axis") == null);
}

/// Shared harness for the grouped-Q GQA parity tests: build smooth Q at
/// `[1, H_q, T_q, D]` and K/V at `[1, H_kv, T_k, D]`, quantize K/V, then
/// compare `quantAttention` against dense SDPA fed the SAME dequantized K/V
/// (no-worse-than-reference — quantization error folds into both paths;
/// what's under test is the grouped-Q contraction + mask tiling).
fn gqaParityCase(
    H_q: c_int,
    H_kv: c_int,
    T_q: c_int,
    T_k: c_int,
    bits: u8,
    mask_mode: [:0]const u8,
    mask_arr: mlx.mlx_array,
) !void {
    return gqaParityCaseD(H_q, H_kv, T_q, T_k, 64, bits, mask_mode, mask_arr);
}

fn gqaParityCaseD(
    H_q: c_int,
    H_kv: c_int,
    T_q: c_int,
    T_k: c_int,
    D: c_int,
    bits: u8,
    mask_mode: [:0]const u8,
    mask_arr: mlx.mlx_array,
) !void {
    const s = mlx.gpuStream();
    const q = try buildSmoothBHTD(s, 1, H_q, T_q, D);
    defer _ = mlx.mlx_array_free(q);
    const k_dense = try buildSmoothBHTD(s, 1, H_kv, T_k, D);
    defer _ = mlx.mlx_array_free(k_dense);
    const v_dense = try buildSmoothBHTD(s, 1, H_kv, T_k, D);
    defer _ = mlx.mlx_array_free(v_dense);

    var qk = try quantizeAffine(s, k_dense, 64, bits);
    defer qk.deinit();
    var qv = try quantizeAffine(s, v_dense, 64, bits);
    defer qv.deinit();

    const k_ref = try dequantizeAffine(s, qk.q, qk.scales, qk.biases, 64, bits);
    defer _ = mlx.mlx_array_free(k_ref);
    const v_ref = try dequantizeAffine(s, qv.q, qv.scales, qv.biases, 64, bits);
    defer _ = mlx.mlx_array_free(v_ref);

    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    // MLX's fast SDPA broadcasts GQA natively (H_q vs H_kv).
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(
        &ref,
        q,
        k_ref,
        v_ref,
        scale,
        mask_mode,
        mask_arr,
        .{ .ctx = null },
        false,
        s,
    ));

    const cand = try quantAttention(
        q,
        .{ .q = qk.q, .scales = qk.scales, .biases = qk.biases },
        .{ .q = qv.q, .scales = qv.scales, .biases = qv.biases },
        bits,
        64,
        scale,
        mask_mode,
        mask_arr,
        s,
    );
    defer _ = mlx.mlx_array_free(cand);

    // Output shape must be the ungrouped [1, H_q, T_q, D].
    const cand_shape = mlx.getShape(cand);
    try testing.expectEqual(@as(usize, 4), cand_shape.len);
    try testing.expectEqual(H_q, cand_shape[1]);
    try testing.expectEqual(T_q, cand_shape[2]);
    try testing.expectEqual(D, cand_shape[3]);

    const ref_flat = try readF32Flat(s, ref, testing.allocator);
    defer testing.allocator.free(ref_flat);
    const cand_flat = try readF32Flat(s, cand, testing.allocator);
    defer testing.allocator.free(cand_flat);
    try testing.expectEqual(ref_flat.len, cand_flat.len);
    var max_err: f32 = 0;
    for (ref_flat, cand_flat) |r, c| {
        // NaN-blind comparisons pass on an all-NaN candidate (NaN > x is
        // false) — the composed causal arm shipped NaN for months behind
        // exactly this hole. Finiteness is asserted BEFORE the diff.
        try testing.expect(!std.math.isNan(c));
        const e = @abs(r - c);
        if (e > max_err) max_err = e;
    }
    if (max_err >= 0.05) {
        std.debug.print("gqaParityCase Hq={d} Hkv={d} Tq={d} Tk={d} D={d} bits={d} mode={s}: max_err={d}\n", .{ H_q, H_kv, T_q, T_k, D, bits, mask_mode, max_err });
    }
    try testing.expect(max_err < 0.05);
}

test "quantAttention GQA 4:1 decode (T_q=1) matches dense SDPA, 4/8-bit" {
    const none = mlx.mlx_array{ .ctx = null };
    try gqaParityCase(8, 2, 1, 16, 4, "", none);
    try gqaParityCase(8, 2, 1, 16, 8, "", none);
}

test "quantAttention GQA 6:1 decode (T_q=1) matches dense SDPA (laguna ratio)" {
    const none = mlx.mlx_array{ .ctx = null };
    try gqaParityCase(48, 8, 1, 24, 8, "", none);
}

test "quantAttention GQA 4:1 causal T_q=4 matches dense SDPA" {
    const none = mlx.mlx_array{ .ctx = null };
    try gqaParityCase(8, 2, 4, 12, 4, "causal", none);
    try gqaParityCase(8, 2, 4, 12, 8, "causal", none);
}

test "quantAttention GQA 6:1 causal T_q=9 matches dense SDPA (MTP verify width)" {
    const none = mlx.mlx_array{ .ctx = null };
    try gqaParityCase(12, 2, 9, 24, 8, "causal", none);
}

test "quantAttention GQA 4:1 causal at gemma geometry (D=256, T=15, 4-bit)" {
    // The exact live shape that answered "<pad><pad>" on gemma-4-e4b under
    // kv_attn_mode "fused" (Hq=8, Hkv=2, hd 256, 4-bit, 15-token prefill).
    const none = mlx.mlx_array{ .ctx = null };
    try gqaParityCaseD(8, 2, 15, 15, 256, 4, "causal", none);
}

test "quantAttention GQA 4:1 array mask T_q=4 matches dense SDPA" {
    const s = mlx.gpuStream();
    // Additive [1, 1, 4, 12] mask: block the first 3 kv positions for every
    // query row (sliding-window-ish shape), 0 elsewhere.
    const t_q: usize = 4;
    const t_k: usize = 12;
    var buf: [t_q * t_k]f32 = undefined;
    for (0..t_q) |r| {
        for (0..t_k) |c| {
            buf[r * t_k + c] = if (c < 3) -std.math.inf(f32) else 0.0;
        }
    }
    const shape = [_]c_int{ 1, 1, t_q, t_k };
    const mask_f32 = mlx.mlx_array_new_data(&buf, &shape, 4, .float32);
    defer _ = mlx.mlx_array_free(mask_f32);
    var mask_bf16 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask_bf16);
    try mlx.check(mlx.mlx_astype(&mask_bf16, mask_f32, .bfloat16, s));

    try gqaParityCase(8, 2, 4, 12, 8, "array", mask_bf16);
}

test "kv-quant decode µbench (env-gated: SUSHI_KVQ_UBENCH=1)" {
    // Phase 0 deliverable (docs/kv-quant-perf.md): time the three decode
    // read paths at T_q=1 — (a) fused grouped-Q quantAttention, (b) dense
    // mode (full-cache dequant + SDPA, what --kv-quant users get today),
    // (c) kv-quant off (SDPA over resident dense K/V). µbench wins can lose
    // live — this SCOPES the work; the live A/B decides defaults.
    const raw = std.c.getenv("SUSHI_KVQ_UBENCH");
    if (raw == null or std.mem.eql(u8, std.mem.sliceTo(raw.?, 0), "0")) return;

    const s = mlx.gpuStream();
    const D: c_int = 128;
    const ratios = [_][2]c_int{ .{ 8, 8 }, .{ 32, 8 }, .{ 48, 8 } };
    const lens = [_]c_int{ 2048, 8192, 32768 };
    const bits_list = [_]u8{ 4, 8 };
    const iters: usize = 20;
    const warmup: usize = 3;

    std.debug.print("\n[kvq-ubench] Hq/Hkv  T_k    bits  fused_ms  dense_ms  off_ms\n", .{});
    for (ratios) |r| {
        const H_q = r[0];
        const H_kv = r[1];
        for (lens) |T_k| {
            for (bits_list) |bits| {
                const q = try buildSmoothBHTD(s, 1, H_q, 1, D);
                defer _ = mlx.mlx_array_free(q);
                const k_dense = try buildSmoothBHTD(s, 1, H_kv, T_k, D);
                defer _ = mlx.mlx_array_free(k_dense);
                const v_dense = try buildSmoothBHTD(s, 1, H_kv, T_k, D);
                defer _ = mlx.mlx_array_free(v_dense);
                var qk = try quantizeAffine(s, k_dense, 64, bits);
                defer qk.deinit();
                var qv = try quantizeAffine(s, v_dense, 64, bits);
                defer qv.deinit();
                const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));
                const none = mlx.mlx_array{ .ctx = null };

                // Pre-evaluate inputs so setup cost never bills a path.
                {
                    const ev = mlx.mlx_vector_array_new();
                    defer _ = mlx.mlx_vector_array_free(ev);
                    _ = mlx.mlx_vector_array_append_value(ev, q);
                    _ = mlx.mlx_vector_array_append_value(ev, k_dense);
                    _ = mlx.mlx_vector_array_append_value(ev, qk.q);
                    _ = mlx.mlx_vector_array_append_value(ev, qv.q);
                    try mlx.check(mlx.mlx_eval(ev));
                }

                // Zig 0.17 has no std.time.Timer; time via std.Io
                // (same pattern as transformer.zig's ProfClock).
                const io = std.Io.Threaded.global_single_threaded.io();
                var mark = std.Io.Timestamp.now(io, .boot);
                const Lap = struct {
                    fn reset(m: *std.Io.Timestamp, io_: std.Io) void {
                        m.* = std.Io.Timestamp.now(io_, .boot);
                    }
                    fn read(m: *const std.Io.Timestamp, io_: std.Io) u64 {
                        return @intCast(m.untilNow(io_, .boot).nanoseconds);
                    }
                };

                // (a) fused grouped-Q
                var fused_ns: u64 = 0;
                for (0..warmup + iters) |i| {
                    if (i == warmup) Lap.reset(&mark, io);
                    const out = try quantAttention(
                        q,
                        .{ .q = qk.q, .scales = qk.scales, .biases = qk.biases },
                        .{ .q = qv.q, .scales = qv.scales, .biases = qv.biases },
                        bits,
                        64,
                        scale,
                        "",
                        none,
                        s,
                    );
                    defer _ = mlx.mlx_array_free(out);
                    const ev = mlx.mlx_vector_array_new();
                    defer _ = mlx.mlx_vector_array_free(ev);
                    _ = mlx.mlx_vector_array_append_value(ev, out);
                    try mlx.check(mlx.mlx_eval(ev));
                }
                fused_ns = Lap.read(&mark, io);

                // (b) dense mode: full dequant + SDPA per step
                var dense_ns: u64 = 0;
                for (0..warmup + iters) |i| {
                    if (i == warmup) Lap.reset(&mark, io);
                    const kd = try dequantizeAffine(s, qk.q, qk.scales, qk.biases, 64, bits);
                    defer _ = mlx.mlx_array_free(kd);
                    const vd = try dequantizeAffine(s, qv.q, qv.scales, qv.biases, 64, bits);
                    defer _ = mlx.mlx_array_free(vd);
                    var out = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(out);
                    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&out, q, kd, vd, scale, "", none, .{ .ctx = null }, false, s));
                    const ev = mlx.mlx_vector_array_new();
                    defer _ = mlx.mlx_vector_array_free(ev);
                    _ = mlx.mlx_vector_array_append_value(ev, out);
                    try mlx.check(mlx.mlx_eval(ev));
                }
                dense_ns = Lap.read(&mark, io);

                // (c) kv-quant off: SDPA over resident dense K/V
                var off_ns: u64 = 0;
                for (0..warmup + iters) |i| {
                    if (i == warmup) Lap.reset(&mark, io);
                    var out = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(out);
                    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&out, q, k_dense, v_dense, scale, "", none, .{ .ctx = null }, false, s));
                    const ev = mlx.mlx_vector_array_new();
                    defer _ = mlx.mlx_vector_array_free(ev);
                    _ = mlx.mlx_vector_array_append_value(ev, out);
                    try mlx.check(mlx.mlx_eval(ev));
                }
                off_ns = Lap.read(&mark, io);

                const to_ms = struct {
                    fn f(ns: u64, n: usize) f64 {
                        return @as(f64, @floatFromInt(ns)) / @as(f64, @floatFromInt(n)) / 1e6;
                    }
                }.f;
                std.debug.print("[kvq-ubench] {d}/{d}  {d}  {d}  {d:.3}  {d:.3}  {d:.3}\n", .{
                    H_q, H_kv, T_k, bits, to_ms(fused_ns, iters), to_ms(dense_ns, iters), to_ms(off_ns, iters),
                });
                _ = mlx.mlx_clear_cache();
            }
        }
    }
}

test "quantAttention causal mask matches dense SDPA (prefill, T_q=T_k=4)" {
    const s = mlx.gpuStream();
    const B: c_int = 1;
    const H: c_int = 2;
    const T: c_int = 4;
    const D: c_int = 64;
    const q = try buildSmoothBHTD(s, B, H, T, D);
    defer _ = mlx.mlx_array_free(q);
    const k_dense = try buildSmoothBHTD(s, B, H, T, D);
    defer _ = mlx.mlx_array_free(k_dense);
    const v_dense = try buildSmoothBHTD(s, B, H, T, D);
    defer _ = mlx.mlx_array_free(v_dense);

    var qk = try quantizeAffine(s, k_dense, 64, 4);
    defer qk.deinit();
    var qv = try quantizeAffine(s, v_dense, 64, 4);
    defer qv.deinit();

    const k_ref = try dequantizeAffine(s, qk.q, qk.scales, qk.biases, 64, 4);
    defer _ = mlx.mlx_array_free(k_ref);
    const v_ref = try dequantizeAffine(s, qv.q, qv.scales, qv.biases, 64, 4);
    defer _ = mlx.mlx_array_free(v_ref);

    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    const none_mask = mlx.mlx_array{ .ctx = null };
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(
        &ref,
        q,
        k_ref,
        v_ref,
        scale,
        "causal",
        none_mask,
        .{ .ctx = null },
        false,
        s,
    ));

    const cand = try quantAttention(
        q,
        .{ .q = qk.q, .scales = qk.scales, .biases = qk.biases },
        .{ .q = qv.q, .scales = qv.scales, .biases = qv.biases },
        4,
        64,
        scale,
        "causal",
        none_mask,
        s,
    );
    defer _ = mlx.mlx_array_free(cand);

    const ref_flat = try readF32Flat(s, ref, testing.allocator);
    defer testing.allocator.free(ref_flat);
    const cand_flat = try readF32Flat(s, cand, testing.allocator);
    defer testing.allocator.free(cand_flat);
    var max_err: f32 = 0;
    for (ref_flat, cand_flat) |r, c| {
        // NaN-blind comparisons pass on an all-NaN candidate (NaN > x is
        // false) — the composed causal arm shipped NaN for months behind
        // exactly this hole. Finiteness is asserted BEFORE the diff.
        try testing.expect(!std.math.isNan(c));
        const e = @abs(r - c);
        if (e > max_err) max_err = e;
    }
    try testing.expect(max_err < 0.05);
}

test "every model, GLM included, defaults to kv8 and 16 selects BF16" {
    const choice = KvCacheChoice.resolve(null, KVQuantConfig.engine_default, false);
    try std.testing.expectEqual(KVQuantConfig.affine(8), choice.config);
    try std.testing.expectEqualStrings("default", choice.sourceName());
    try std.testing.expectEqual(KVQuantConfig.dense, KVQuantConfig.fromJsonValue(.{ .string = "16" }).?);
    try std.testing.expectEqual(KVQuantConfig.dense, KVQuantConfig.fromJsonValue(.{ .integer = 16 }).?);
    try std.testing.expectEqual(KVQuantConfig.dense, KvCacheChoice.resolve(null, KVQuantConfig.fromJsonValue(.{ .string = "16" }).?, true).config);
}

test "GLM latent stores BF16 or kv8 and names kv4 unsupported" {
    try std.testing.expectEqual(@as(?u8, 0), KVQuantConfig.dense.glmLatentBits());
    try std.testing.expectEqual(@as(?u8, 8), KVQuantConfig.affine(8).glmLatentBits());
    try std.testing.expectEqual(@as(?u8, null), KVQuantConfig.affine(4).glmLatentBits());
    try std.testing.expectEqual(@as(?u8, 8), KVQuantConfig.fromJsonValue(.{ .integer = 8 }).?.glmLatentBits());
}
