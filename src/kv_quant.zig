// KV-cache quantization backend.
//
// This module owns the storage / dispatch contract between the cache
// (`KVCache` in `transformer.zig`) and the attention call sites. SDPA always
// reads dense `[B, H, T, head_dim]` tensors via `KVCache.denseView` — what the
// cache buffers actually hold is decided here.
//
// One non-trivial scheme: affine group-wise quantization at 4, 6 or 8 bits,
// mathematically identical to mlx-c's `mlx_quantize` weight path. The buffers
// grow to 3 arrays per K and V (q, scales, biases); attention is unchanged
// because `denseView` calls `dequantizeAffine` before returning.
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
pub const Scheme = enum { off, affine };

/// Configuration for the cache's storage backend. Stored on `KVCache.config`
/// and switched on at every read/write boundary.
pub const KVQuantConfig = struct {
    scheme: Scheme,
    /// Affine: 4, 6 or 8. Ignored when `scheme == .off`.
    bits: u8,
    /// Affine group size — number of consecutive elements that share one
    /// scale+bias pair along the last axis. mlx-c convention is 64 for
    /// 4-, 6- and 8-bit weights; we match that.
    group_size: u32,

    pub const dense: KVQuantConfig = .{ .scheme = .off, .bits = 0, .group_size = 0 };
    /// What a load stores at when neither `--kv-quant` nor `model-settings.json` names a scheme.
    pub const engine_default: KVQuantConfig = .{ .scheme = .affine, .bits = 8, .group_size = 64 };

    /// Affine widths the cache stores and every packed reader serves.
    pub fn supportedBits(bits: u8) bool {
        return bits == 4 or bits == 6 or bits == 8;
    }

    pub fn affine(bits: u8) KVQuantConfig {
        std.debug.assert(supportedBits(bits));
        return .{ .scheme = .affine, .bits = bits, .group_size = 64 };
    }

    pub fn isQuant(self: KVQuantConfig) bool {
        return self.scheme != .off;
    }

    /// The wire vocabulary shared by the per-request `kv_quant` body field and
    /// `model-settings.json`: "off"/0/16 (BF16), "4", "6", "8". Null = unrecognized.
    pub fn fromJsonValue(v: std.json.Value) ?KVQuantConfig {
        switch (v) {
            .string => |s| {
                if (std.mem.eql(u8, s, "off") or std.mem.eql(u8, s, "0") or std.mem.eql(u8, s, "16")) return dense;
                if (std.mem.eql(u8, s, "4")) return affine(4);
                if (std.mem.eql(u8, s, "6")) return affine(6);
                if (std.mem.eql(u8, s, "8")) return affine(8);
                return null;
            },
            .integer => |i| {
                if (i == 0 or i == 16) return dense;
                if (i == 4) return affine(4);
                if (i == 6) return affine(6);
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
            .affine => if ((self.bits == 8 or self.bits == 6) and self.group_size == 64) self.bits else null,
        };
    }

    /// The same vocabulary, for reporting (`/v1/models` `meta.kv_quant`).
    pub fn wireName(self: KVQuantConfig) []const u8 {
        return switch (self.scheme) {
            .off => "off",
            .affine => switch (self.bits) {
                4 => "4",
                6 => "6",
                else => "8",
            },
        };
    }
};

/// uint32 words one row of `dim` values packs into; a layout that rounds is never a cache's.
pub fn packedWords(dim: c_int, bits: u8) c_int {
    return @divTrunc(dim * @as(c_int, bits), 32);
}

/// Values a packed row of `words` uint32 words holds; the inverse of `packedWords`.
pub fn unpackedDim(words: c_int, bits: u8) c_int {
    return @divTrunc(words * 32, @as(c_int, bits));
}

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
            .affine => switch (self.config.bits) {
                4 => "kv4",
                6 => "kv6",
                else => "kv8",
            },
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

test "kv6 is a first-class wire value: parsed, named and served by GLM's latent backend" {
    const t = std.testing;
    const six = KVQuantConfig.affine(6);
    try t.expectEqual(six, KVQuantConfig.fromJsonValue(.{ .string = "6" }).?);
    try t.expectEqual(six, KVQuantConfig.fromJsonValue(.{ .integer = 6 }).?);
    try t.expectEqual(@as(?KVQuantConfig, null), KVQuantConfig.fromJsonValue(.{ .string = "5" }));
    try t.expectEqual(@as(?KVQuantConfig, null), KVQuantConfig.fromJsonValue(.{ .integer = 7 }));
    try t.expectEqualStrings("6", six.wireName());
    try t.expectEqualStrings("kv6", (KvCacheChoice{ .config = six, .source = .default }).label());
    try t.expectEqual(@as(?u8, 6), six.glmLatentBits());
    // Every wire name parses back to its own config.
    for ([_]KVQuantConfig{ KVQuantConfig.dense, KVQuantConfig.affine(4), six, KVQuantConfig.affine(8) }) |c| {
        try t.expectEqual(c, KVQuantConfig.fromJsonValue(.{ .string = c.wireName() }).?);
    }
}

fn roundTripRms(bits: u8) !f64 {
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(0x6b76);
    const rnd = prng.random();
    const n = 64 * 256;
    const host = try testing.allocator.alloc(f32, n);
    defer testing.allocator.free(host);
    for (host) |*v| v.* = rnd.floatNorm(f32) * 3.0;
    const shape = [_]c_int{ 64, 256 };
    const f32_arr = mlx.mlx_array_new_data(host.ptr, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(f32_arr);
    var src = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(src);
    try mlx.check(mlx.mlx_astype(&src, f32_arr, .bfloat16, s));

    var qkv = try quantizeAffine(s, src, 64, bits);
    defer qkv.deinit();
    try testing.expectEqual(@as(c_int, 8) * bits, mlx.getShape(qkv.q)[1]);
    const deq = try dequantizeAffine(s, qkv.q, qkv.scales, qkv.biases, 64, bits);
    defer _ = mlx.mlx_array_free(deq);

    const orig = try readF32Flat(s, src, testing.allocator);
    defer testing.allocator.free(orig);
    const got = try readF32Flat(s, deq, testing.allocator);
    defer testing.allocator.free(got);
    var sum: f64 = 0;
    for (orig, got) |o, g| sum += (o - g) * (o - g);
    return @sqrt(sum / n);
}

test "kv6 round-trip error sits between kv4 and kv8" {
    const e4 = try roundTripRms(4);
    const e6 = try roundTripRms(6);
    const e8 = try roundTripRms(8);
    // Step size shrinks 4x per two bits; bf16 scales put a floor under kv8.
    try testing.expect(e6 < e4 / 3.0);
    try testing.expect(e6 > e8 * 2.0);
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

test "every model, GLM included, defaults to kv8 and 16 selects BF16" {
    const choice = KvCacheChoice.resolve(null, KVQuantConfig.engine_default, false);
    try std.testing.expectEqual(KVQuantConfig.affine(8), choice.config);
    try std.testing.expectEqualStrings("default", choice.sourceName());
    try std.testing.expectEqual(KVQuantConfig.dense, KVQuantConfig.fromJsonValue(.{ .string = "16" }).?);
    try std.testing.expectEqual(KVQuantConfig.dense, KVQuantConfig.fromJsonValue(.{ .integer = 16 }).?);
    try std.testing.expectEqual(KVQuantConfig.dense, KvCacheChoice.resolve(null, KVQuantConfig.fromJsonValue(.{ .string = "16" }).?, true).config);
}

test "GLM latent stores BF16, kv8 or kv6 and names kv4 unsupported" {
    try std.testing.expectEqual(@as(?u8, 0), KVQuantConfig.dense.glmLatentBits());
    try std.testing.expectEqual(@as(?u8, 8), KVQuantConfig.affine(8).glmLatentBits());
    try std.testing.expectEqual(@as(?u8, null), KVQuantConfig.affine(4).glmLatentBits());
    try std.testing.expectEqual(@as(?u8, 6), KVQuantConfig.affine(6).glmLatentBits());
    try std.testing.expectEqual(@as(?u8, 8), KVQuantConfig.fromJsonValue(.{ .integer = 8 }).?.glmLatentBits());
}
