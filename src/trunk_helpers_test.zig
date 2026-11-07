//! Test-only fixtures shared by the trunk, KV cache, attention-kernel and MoE tests: random KV appends, sliding-window
//! and fused-attention truth, verify-GEMM parity, a bank hash. Imported inside test blocks only.
const std = @import("std");
const mlx = @import("mlx.zig");
const testing = std.testing;
const kv_cache = @import("kv_cache.zig");
const attn_kernels = @import("attn_kernels.zig");
const model_mod = @import("model.zig");
const mlx_util = @import("mlx_util.zig");
const KVCache = kv_cache.KVCache;
const KVCacheEntry = kv_cache.KVCacheEntry;
const AttnPdArm = attn_kernels.AttnPdArm;
const ModelConfig = model_mod.ModelConfig;
const bf16Scalar = mlx_util.bf16Scalar;
const expectError = @import("test_expect.zig").expectError;
const test_support = @import("test_support.zig");
const qsaArraysAllEqual = test_support.qsaArraysAllEqual;
const testRandUniformBf16 = test_support.testRandUniformBf16;
const attn256Reference = test_support.attn256Reference;
const DenseKVView = kv_cache.DenseKVView;
const slidingTailSpan = kv_cache.slidingTailSpan;
const SLIDING_TRIM_UNBOUNDED = kv_cache.SLIDING_TRIM_UNBOUNDED;
const kv_quant = @import("kv_quant.zig");
const qsaNaxOsOk = @import("nax_gate.zig").qsaNaxOsOk;
const materializedOwnedCopy = mlx_util.materializedOwnedCopy;
const verifyQmmNaxAvailable = @import("nax_gate.zig").verifyQmmNaxAvailable;

const kv_append_kv_h: c_int = 2;

const kv_append_hd: c_int = 256;

pub fn kvAppendPacked(n: c_int, seq: c_int, seed: u32, s: mlx.mlx_stream) !mlx.mlx_array {
    const nn: usize = @intCast(n * kv_append_kv_h * seq * kv_append_hd);
    const buf = try testing.allocator.alloc(f32, nn);
    defer testing.allocator.free(buf);
    var st: u32 = seed;
    for (buf) |*x| {
        st = st *% 1664525 +% 1013904223;
        x.* = @as(f32, @floatFromInt(st >> 16)) / 65536.0 - 0.5;
    }
    const shape = [_]c_int{ n, kv_append_kv_h, seq, kv_append_hd };
    const f32_arr = mlx.mlx_array_new_data(buf.ptr, &shape, 4, .float32);
    defer _ = mlx.mlx_array_free(f32_arr);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, f32_arr, .bfloat16, s));
    return out;
}

pub fn kvAppendPrefill(cache: *KVCache, tokens: usize, seed: u32, s: mlx.mlx_stream) !void {
    const k = try kvAppendPacked(1, @intCast(tokens), seed, s);
    defer _ = mlx.mlx_array_free(k);
    const v = try kvAppendPacked(1, @intCast(tokens), seed ^ 0x9E3779B9, s);
    defer _ = mlx.mlx_array_free(v);
    var dv = try cache.update(0, k, v, s, 0);
    dv.deinit();
}

pub fn kvAppendOperandsEqual(a: *const KVCacheEntry, b: *const KVCacheEntry, s: mlx.mlx_stream) !void {
    try testing.expectEqual(a.offset, b.offset);
    try testing.expect(try qsaArraysAllEqual(a.keys, b.keys, s));
    try testing.expect(try qsaArraysAllEqual(a.values, b.values, s));
    try testing.expect(try qsaArraysAllEqual(a.keys_scales, b.keys_scales, s));
    try testing.expect(try qsaArraysAllEqual(a.keys_biases, b.keys_biases, s));
    try testing.expect(try qsaArraysAllEqual(a.values_scales, b.values_scales, s));
    try testing.expect(try qsaArraysAllEqual(a.values_biases, b.values_biases, s));
}

pub fn kvAppendDummyAttn(cache: *KVCache, s: mlx.mlx_stream) !mlx.mlx_array {
    var dv = try cache.denseView(0, s);
    defer dv.deinit();
    const ksh = mlx.getShape(dv.k);
    const q_shape = [_]c_int{ ksh[0], ksh[1], 1, ksh[3] };
    var q = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(q);
    try mlx.check(mlx.mlx_ones(&q, &q_shape, 4, .bfloat16, s));
    const none = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(none);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(kv_append_hd)));
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(
        &out,
        q,
        dv.k,
        dv.v,
        scale,
        "",
        none,
        .{ .ctx = null },
        false,
        s,
    ));
    _ = mlx.mlx_array_free(q);
    return out;
}

pub fn conv1dExpectBitEqual(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !void {
    var eq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eq);
    try mlx.check(mlx.mlx_array_equal(&eq, a, b, false, s));
    try mlx.check(mlx.mlx_array_eval(eq));
    var ok: bool = false;
    try mlx.check(mlx.mlx_array_item_bool(&ok, eq));
    try std.testing.expect(ok);
}

pub fn moeRowsAffineBank(s: mlx.mlx_stream, rnd: std.Random, E: c_int, N: c_int, K: c_int, bits: u32, gs: u32) !struct { w: mlx.mlx_array, sc: mlx.mlx_array, bi: mlx.mlx_array } {
    const words: c_int = @intCast(@divExact(@as(u32, @intCast(K)) * bits, 32));
    const groups: c_int = @divExact(K, @as(c_int, @intCast(gs)));
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, rnd.int(u64)));
    const w_shape = [_]c_int{ E, N, words };
    var w = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_random_bits(&w, &w_shape, 3, 4, key, s));
    const s_shape = [_]c_int{ E, N, groups };
    const sc = try testRandUniformBf16(rnd, &s_shape, 0.01, 0.06, s);
    const bi = try testRandUniformBf16(rnd, &s_shape, -0.5, 0.5, s);
    return .{ .w = w, .sc = sc, .bi = bi };
}

pub fn moeRowsSliceRow(s: mlx.mlx_stream, x: mlx.mlx_array, i: c_int) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    var start: [3]c_int = .{ i, 0, 0 };
    var stop: [3]c_int = .{ i + 1, 1, 1 };
    const strides = [_]c_int{ 1, 1, 1 };
    for (1..sh.len) |d| stop[d] = sh[d];
    var sl = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sl);
    try mlx.check(mlx.mlx_slice(&sl, x, &start, @intCast(sh.len), &stop, @intCast(sh.len), &strides, @intCast(sh.len), s));
    var sq = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(sq);
    try mlx.check(mlx.mlx_squeeze(&sq, sl, s));
    return sq;
}

/// `[1, heads, len, dim]` of distinct, position-dependent values — two caches
/// fed the same chunk must return byte-identical views, so only distinctness
/// matters, not the numbers.
pub fn swaRingChunk(s: mlx.mlx_stream, start: c_int, len: c_int, heads: c_int, dim: c_int) !mlx.mlx_array {
    const n: f64 = @floatFromInt(heads * len * dim);
    const base: f64 = @floatFromInt(start * heads * dim);
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_arange(&flat, base, base + n, 1.0, .float32, s));
    var scaled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(scaled);
    const inv = mlx.mlx_array_new_float(1.0 / 4096.0);
    defer _ = mlx.mlx_array_free(inv);
    try mlx.check(mlx.mlx_multiply(&scaled, flat, inv, s));
    var bf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bf);
    try mlx.check(mlx.mlx_astype(&bf, scaled, .bfloat16, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    const shape = [_]c_int{ 1, heads, len, dim };
    try mlx.check(mlx.mlx_reshape(&out, bf, &shape, 4, s));
    return out;
}

/// Layer 0 global, layer 1 sliding: rows `[from, to)` in `chunk`-wide forwards.
pub fn ringCheckpointFeed(a: *KVCache, b: *KVCache, window: u32, from: usize, to: usize, chunk: usize) !void {
    const s = mlx.gpuStream();
    var pos = from;
    while (pos < to) {
        const q: c_int = @intCast(@min(chunk, to - pos));
        const k = try swaRingChunk(s, @intCast(pos), q, 1, 64);
        defer _ = mlx.mlx_array_free(k);
        const v = try swaRingChunk(s, @as(c_int, @intCast(pos)) + 7919, q, 1, 64);
        defer _ = mlx.mlx_array_free(v);
        for ([_]u32{ 0, slidingTailSpan(window, @intCast(q), SLIDING_TRIM_UNBOUNDED) }, 0..) |span, li| {
            var av = try a.update(@intCast(li), k, v, s, span);
            defer av.deinit();
            var bv = try b.update(@intCast(li), k, v, s, span);
            defer bv.deinit();
            try std.testing.expectEqual(@as(f32, 0), try maxAbsDiffF32(av.k, bv.k, s));
            try std.testing.expectEqual(@as(f32, 0), try maxAbsDiffF32(av.v, bv.v, s));
        }
        pos += @intCast(q);
    }
}

/// Max |a-b| between two same-shaped float arrays, with a NaN guard (a NaN
/// candidate must FAIL, never score 0 — the parity-loop finiteness rule).
pub fn maxAbsDiffF32(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    // materializedOwnedCopy: a same-dtype astype is a NO-OP view and a slice's
    // raw data pointer walks the PARENT's physical layout — force a contiguous
    // owned buffer before touching bytes.
    var a_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a_t);
    try mlx.check(mlx.mlx_astype(&a_t, a, .float32, s));
    const a32 = try materializedOwnedCopy(s, a_t);
    defer _ = mlx.mlx_array_free(a32);
    var b_t = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b_t);
    try mlx.check(mlx.mlx_astype(&b_t, b, .float32, s));
    const b32 = try materializedOwnedCopy(s, b_t);
    defer _ = mlx.mlx_array_free(b32);
    try mlx.check(mlx.mlx_array_eval(a32));
    try mlx.check(mlx.mlx_array_eval(b32));
    const n = mlx.mlx_array_size(a32);
    if (n != mlx.mlx_array_size(b32)) return error.ShapeMismatch;
    const pa = mlx.mlx_array_data_float32(a32) orelse return error.NoData;
    const pb = mlx.mlx_array_data_float32(b32) orelse return error.NoData;
    var max_err: f32 = 0;
    for (0..n) |i| {
        const d = @abs(pa[i] - pb[i]);
        if (!std.math.isFinite(d)) return error.NonFiniteValue;
        if (d > max_err) max_err = d;
    }
    return max_err;
}

/// Assert a verify-kernel output is NO LESS ACCURATE than the stock qmm it
/// replaces, both measured against `wdq_t` — the SAME 4-bit weights dequantized
/// to fp32 (so quantization error cancels and only the two kernels' own
/// arithmetic is under test).
///
/// This is the machine-independent invariant. A direct kernel-vs-stock bound is
/// NOT: both paths accumulate fp32 in a different order and round to bf16, so on
/// heavily-cancelling data each already sits ~0.03 from truth on its worst
/// element and they may legitimately differ from each other by more than that.
/// Tuning a pair-agreement threshold on one GPU produces a test that is green on
/// the dev machine and red on the CI runner while the kernel is bit-for-bit as
/// accurate (measured: kernel-vs-truth == stock-vs-truth to four decimals).
/// A genuine defect — a partial-sum race, a register spill, bad indexing —
/// pushes kernel-vs-truth far past stock-vs-truth and is still caught here.
/// Parity bars for a verify lane, stated so that no bar is fit per GPU.
///
/// A per-chip max-error slack was tried first (M1 Pro measured kern_max 0.0313
/// against stock 0.0039 while cosine stayed at 0.999998, so the bar was
/// widened to 0.04 there). That is the wrong shape of guard twice over:
///
///   - Widening the MAX blinds the test to what it exists to catch. Cosine is
///     a GLOBAL measure, so a partial-sum race, a register spill or a bad
///     index — one element far off, everything else intact — passes a widened
///     max with cosine unmoved.
///   - `stock_max` is not a stable reference. The same shapes measure
///     stock_max 0.02-0.035 on M4 Max, so the M1 Pro row was not a lane
///     landing 8x worse than normal, it was STOCK landing unusually well
///     there. Anything keyed to stock's worst element inherits that.
///
/// So neither bar references stock's max any more:
///
///   - `GROSS_CEILING` is an ABSOLUTE ceiling on the worst element. On the
///     clamped-relative metric below, a bf16 output's own rounding noise sits
///     at 0.02-0.035 on every machine measured; a corrupted accumulator lands
///     orders of magnitude past 0.5. This is the single-element bar.
///   - `RMS_FACTOR` bounds the whole error distribution against stock's,
///     which is scale-free: when stock is unusually accurate the ratio is
///     what stays meaningful. This is the systematic bar. (M1 Pro's cosines
///     imply a ratio near 1.4; M4 Max measures 1.00 across every lane.)
///   - The cosine bars are unchanged and catch a broken reference.
pub const VerifyQmmParity = struct {
    /// Absolute ceiling on the worst element's clamped relative error.
    pub const GROSS_CEILING: f64 = 0.5;
    /// How much of stock's RMS error the lane may carry.
    pub const RMS_FACTOR: f64 = 3.0;
    /// Cosine may not fall below stock's by more than this.
    pub const COS_SLACK: f64 = 1e-5;
    /// …and both must actually be tracking the truth.
    pub const COS_FLOOR: f64 = 0.999;
};

/// Why a verify-lane parity check failed, or null when it passed. Pure so the
/// decision itself is unit-testable without a GPU — each arm is a distinct
/// failure shape the bars are meant to separate.
const VerifyQmmParityFail = enum { gross_element, systematic, cosine, not_tracking };

pub fn verifyQmmParityVerdict(
    kern_max: f64,
    rms_stock: f64,
    rms_kern: f64,
    cos_stock: f64,
    cos_kern: f64,
) ?VerifyQmmParityFail {
    if (cos_kern < VerifyQmmParity.COS_FLOOR) return .not_tracking;
    if (cos_kern < cos_stock - VerifyQmmParity.COS_SLACK) return .cosine;
    if (kern_max > VerifyQmmParity.GROSS_CEILING) return .gross_element;
    // A zero stock RMS means the reference is exact on this data; the ratio
    // then demands the lane be exact too, which is the strictest reading and
    // the one we want — an exactly-representable case a lane cannot reproduce
    // is a defect, not reduction-order noise.
    if (rms_kern > VerifyQmmParity.RMS_FACTOR * rms_stock) return .systematic;
    return null;
}

/// fp32 ground truth for an attention shape: the same composed chain MLX runs
/// for an unfused head dim, with every operand widened first. Both the bf16
/// composed path and the kernel are measured against THIS, never each other.
pub fn attnPdTruthF32(
    q: mlx.mlx_array,
    k: mlx.mlx_array,
    v: mlx.mlx_array,
    scale: f32,
    s: mlx.mlx_stream,
) !mlx.mlx_array {
    var q32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q32);
    var k32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k32);
    var v32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v32);
    try mlx.check(mlx.mlx_astype(&q32, q, .float32, s));
    try mlx.check(mlx.mlx_astype(&k32, k, .float32, s));
    try mlx.check(mlx.mlx_astype(&v32, v, .float32, s));
    return attn256Reference(q32, k32, v32, scale, "causal", .{ .ctx = null }, s);
}

/// A `DenseKVView` over a freshly quantized K/V pair, as `KVCache.denseView`
/// hands one out: lazy dense arrays plus the packed triples they stand for.
pub const TestQuantView = struct {
    view: DenseKVView,
    kq: kv_quant.QuantizedKV,
    vq: kv_quant.QuantizedKV,

    pub fn init(s: mlx.mlx_stream, k: mlx.mlx_array, v: mlx.mlx_array, bits: u8, group_size: u32) !TestQuantView {
        var kq = try kv_quant.quantizeAffine(s, k, group_size, bits);
        errdefer kq.deinit();
        var vq = try kv_quant.quantizeAffine(s, v, group_size, bits);
        errdefer vq.deinit();
        const dk = try kv_quant.dequantizeAffine(s, kq.q, kq.scales, kq.biases, group_size, bits);
        errdefer _ = mlx.mlx_array_free(dk);
        const dv = try kv_quant.dequantizeAffine(s, vq.q, vq.scales, vq.biases, group_size, bits);
        return .{
            .kq = kq,
            .vq = vq,
            .view = .{
                .k = dk,
                .v = dv,
                .owned = true,
                .k_triple_q = kq.q,
                .k_triple_scales = kq.scales,
                .k_triple_biases = kq.biases,
                .v_triple_q = vq.q,
                .v_triple_scales = vq.scales,
                .v_triple_biases = vq.biases,
                .has_quant_triple = true,
                .bits = bits,
                .group_size = group_size,
            },
        };
    }

    pub fn deinit(self: *TestQuantView) void {
        self.view.deinit();
        self.kq.deinit();
        self.vq.deinit();
    }
};

pub const SWA_TEST_WINDOW: u32 = 128;

/// mimo_v2's sliding-layer attention geometry, enough for the ring and the arm.
pub fn swaSinkTestConfig() ModelConfig {
    var c = ModelConfig{};
    c.model_type = "mimo_v2";
    c.has_sliding_window = true;
    c.sliding_window = SWA_TEST_WINDOW;
    c.head_dim = 192;
    c.v_head_dim = 128;
    return c;
}

pub fn swaHostF32(alloc: std.mem.Allocator, a: mlx.mlx_array, s: mlx.mlx_stream) ![]f32 {
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_contiguous(&c, a, false, s));
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, c, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f));
    const n = mlx.mlx_array_size(f);
    const src = mlx.mlx_array_data_float32(f) orelse return error.Unreadable;
    const out = try alloc.alloc(f32, n);
    @memcpy(out, src[0..n]);
    return out;
}

pub const SwaGeom = struct { hq: usize, hk: usize, qk: usize, vd: usize, t: usize };

/// fp64 truth for one chunk of a sliding layer with sinks, read off the FULL
/// history at absolute positions: a ring or band offset bug cannot hide in it.
/// `q` is the chunk `[hq, ql, qk]`; `k`/`v` the whole history `[hk, t, *]`.
pub fn swaSinkTruth(
    alloc: std.mem.Allocator,
    g: SwaGeom,
    q: []const f32,
    k: []const f32,
    v: []const f32,
    sinks: []const f32,
    pos: usize,
    ql: usize,
    scale: f64,
) ![]f64 {
    const out = try alloc.alloc(f64, g.hq * ql * g.vd);
    @memset(out, 0);
    var logits: [SWA_TEST_WINDOW]f64 = undefined;
    for (0..g.hq) |h| {
        const kh = h / (g.hq / g.hk);
        for (0..ql) |r| {
            const p = pos + r;
            const lo = (p + 1) -| SWA_TEST_WINDOW;
            const qrow = q[(h * ql + r) * g.qk ..][0..g.qk];
            var m: f64 = sinks[h];
            for (lo..p + 1, 0..) |j, i| {
                const krow = k[(kh * g.t + j) * g.qk ..][0..g.qk];
                var dot: f64 = 0;
                for (qrow, krow) |a, b| dot += @as(f64, a) * b;
                logits[i] = dot * scale;
                m = @max(m, logits[i]);
            }
            var denom: f64 = @exp(@as(f64, sinks[h]) - m);
            for (logits[0 .. p + 1 - lo]) |l| denom += @exp(l - m);
            const orow = out[(h * ql + r) * g.vd ..][0..g.vd];
            for (lo..p + 1, 0..) |j, i| {
                const w = @exp(logits[i] - m) / denom;
                const vrow = v[(kh * g.t + j) * g.vd ..][0..g.vd];
                for (orow, vrow) |*o, x| o.* += w * x;
            }
        }
    }
    return out;
}

/// `[1, 1, ql, kl]` bool band: the mask the composed arm reads, bottom-right aligned.
pub fn swaBandMask(ql: c_int, kl: c_int) !mlx.mlx_array {
    const alloc = std.testing.allocator;
    const n: usize = @intCast(ql * kl);
    const buf = try alloc.alloc(bool, n);
    defer alloc.free(buf);
    for (0..@intCast(ql)) |r| {
        const row: c_int = kl - ql + @as(c_int, @intCast(r));
        for (0..@intCast(kl)) |c| {
            const col: c_int = @intCast(c);
            buf[r * @as(usize, @intCast(kl)) + c] = col <= row and row - col < @as(c_int, SWA_TEST_WINDOW);
        }
    }
    const shape = [_]c_int{ 1, 1, ql, kl };
    return mlx.mlx_array_new_data(buf.ptr, &shape, 4, .bool_);
}

pub fn absMax(xs: []const f32) f64 {
    var m: f64 = 0;
    for (xs) |x| m = @max(m, @abs(@as(f64, x)));
    return m;
}

pub fn swaMaxErr(got: []const f32, truth: []const f64) f64 {
    var e: f64 = 0;
    for (got, truth) |a, b| e = @max(e, @abs(@as(f64, a) - b));
    return e;
}

pub fn sliceSeq(s: mlx.mlx_stream, a: mlx.mlx_array, start: c_int, end: c_int) !mlx.mlx_array {
    const sh = mlx.getShape(a);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&out, a, &.{ 0, 0, start, 0 }, 4, &.{ sh[0], sh[1], end, sh[3] }, 4, &.{ 1, 1, 1, 1 }, 4, s));
    return out;
}

/// Sum-of-squares slack (RMS within 0.5%): bf16 store rounding flips land either way between the arms.
const ATTN_PD_NAX_RMS_SLACK: f64 = 1.01;

/// The kernel arms this machine can run, SIMD first: every parity case covers each of them.
pub fn attnPdTestArms() []const AttnPdArm {
    const all = [_]AttnPdArm{ .simd, .nax };
    return if (verifyQmmNaxAvailable() and qsaNaxOsOk()) &all else all[0..1];
}

/// The NAX arm against fp64 truth, element by element, may exceed the SIMD kernel's error only by
/// the bf16 store's own rounding flip plus its f16 P's rounding (2^-11 of the largest |V|, the
/// weights summing to one), and its RMS error by rounding noise.
pub fn expectAttnPdNoWorse(nax: []const f32, simd: []const f32, truth: []const f64, v_absmax: f64) !void {
    const p16 = v_absmax * 0.00048828125;
    var scale_of: f64 = 0;
    for (truth) |t| scale_of = @max(scale_of, @abs(t));
    var sq_nax: f64 = 0;
    var sq_simd: f64 = 0;
    for (nax, simd, truth) |a, b, t| {
        const ea = @abs(@as(f64, a) - t);
        const eb = @abs(@as(f64, b) - t);
        try std.testing.expect(std.math.isFinite(ea) and std.math.isFinite(eb));
        // One bf16 ulp of the element, floored at one ulp of a value 2^-8 below the tensor's scale
        // (near zero the fp32 accumulation, not the store, sets the error).
        try std.testing.expect(ea <= eb + (@abs(t) + scale_of * 0.00390625) * 0.00390625 + p16);
        sq_nax += ea * ea;
        sq_simd += eb * eb;
    }
    const n: f64 = @floatFromInt(truth.len);
    try std.testing.expect(sq_nax <= sq_simd * ATTN_PD_NAX_RMS_SLACK + n * (p16 * 0.5) * (p16 * 0.5));
}

/// fp64 truth for a global layer's causal attention over `g.t` keys, bottom-right aligned.
/// `q` is `[hq, ql, qk]`, `k`/`v` are `[hk, t, *]`.
pub fn attnPdCausalTruth(alloc: std.mem.Allocator, g: SwaGeom, q: []const f32, k: []const f32, v: []const f32, ql: usize, scale: f64) ![]f64 {
    const out = try alloc.alloc(f64, g.hq * ql * g.vd);
    @memset(out, 0);
    const logits = try alloc.alloc(f64, g.t);
    defer alloc.free(logits);
    for (0..g.hq) |h| {
        const kh = h / (g.hq / g.hk);
        for (0..ql) |r| {
            const p = g.t - ql + r;
            const qrow = q[(h * ql + r) * g.qk ..][0..g.qk];
            var m: f64 = -std.math.inf(f64);
            for (0..p + 1) |j| {
                const krow = k[(kh * g.t + j) * g.qk ..][0..g.qk];
                var dot: f64 = 0;
                for (qrow, krow) |a, b| dot += @as(f64, a) * b;
                logits[j] = dot * scale;
                m = @max(m, logits[j]);
            }
            var denom: f64 = 0;
            for (logits[0 .. p + 1]) |l| denom += @exp(l - m);
            const orow = out[(h * ql + r) * g.vd ..][0..g.vd];
            for (0..p + 1) |j| {
                const w = @exp(logits[j] - m) / denom;
                const vrow = v[(kh * g.t + j) * g.vd ..][0..g.vd];
                for (orow, vrow) |*o, x| o.* += w * x;
            }
        }
    }
    return out;
}

pub fn verifyRollbackEvalStorage(entry: *const KVCacheEntry) !void {
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    inline for (.{ "keys", "values", "keys_scales", "keys_biases", "values_scales", "values_biases" }) |field| {
        const value = @field(entry, field);
        if (value.ctx != null) _ = mlx.mlx_vector_array_append_value(vec, value);
    }
    try mlx.check(mlx.mlx_eval(vec));
}

pub fn verifyRollbackValidPrefixEqual(a: *const KVCacheEntry, b: *const KVCacheEntry, s: mlx.mlx_stream) !void {
    try testing.expectEqual(a.initialized, b.initialized);
    try testing.expectEqual(a.offset, b.offset);
    if (!a.initialized) return;
    inline for (.{ "keys", "values", "keys_scales", "keys_biases", "values_scales", "values_biases" }) |field| {
        const av = @field(a, field);
        const bv = @field(b, field);
        try testing.expectEqual(av.ctx == null, bv.ctx == null);
        if (av.ctx != null) {
            const ash = mlx.getShape(av);
            const bsh = mlx.getShape(bv);
            var aslice = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(aslice);
            var bslice = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(bslice);
            try mlx.check(mlx.mlx_slice(&aslice, av, &.{ 0, 0, 0, 0 }, 4, &.{ ash[0], ash[1], @intCast(a.offset), ash[3] }, 4, &.{ 1, 1, 1, 1 }, 4, s));
            try mlx.check(mlx.mlx_slice(&bslice, bv, &.{ 0, 0, 0, 0 }, 4, &.{ bsh[0], bsh[1], @intCast(b.offset), bsh[3] }, 4, &.{ 1, 1, 1, 1 }, 4, s));
            try testing.expect(try qsaArraysAllEqual(aslice, bslice, s));
        }
    }
}

pub fn seBf16(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
}

pub fn seFillBf16(rnd: std.Random, buf: []u16) void {
    for (buf) |*v| {
        const f = rnd.float(f32) * 2.0 - 1.0;
        v.* = @truncate(@as(u32, @bitCast(f)) >> 16);
    }
}

pub fn seHash(arr: mlx.mlx_array) !u64 {
    try mlx.check(mlx.mlx_array_eval(arr));
    const p = mlx.mlx_array_data_bfloat16(arr) orelse return error.Bf16Unreadable;
    var h: u64 = 0xcbf29ce484222325;
    for (p[0..mlx.mlx_array_size(arr)]) |v| {
        h ^= v;
        h *%= 0x100000001b3;
    }
    return h;
}

pub const SeErr = struct { max: f32, rms: f32 };

test "verifyQmmParityVerdict: reduction-order noise passes, a gross or systematic error does not" {
    // Reduction-order noise: a worse worst-element than stock, cosine intact.
    // This is the M1 Pro measurement (kern_max 0.0313 while stock landed an
    // unusually good 0.0039 at cos 0.999998) and it must PASS with no
    // per-chip bar — the max is no longer judged against stock's at all.
    try testing.expect(verifyQmmParityVerdict(0.0313, 0.0009, 0.0013, 0.999999, 0.999998) == null);
    // M4 Max: every shipped lane measures ratio 1.00.
    try testing.expect(verifyQmmParityVerdict(0.0308, 0.0012, 0.0012, 0.999998, 0.999998) == null);
    // One accumulator corrupted: a single element, rms and cosine still fine
    // — the exact shape a max-slack widened to 0.04 would have waved through.
    try testing.expectEqual(
        VerifyQmmParityFail.gross_element,
        verifyQmmParityVerdict(12.0, 0.0012, 0.0013, 0.999999, 0.999997).?,
    );
    // Systematically noisier without a single gross element: rms is the bar
    // that sees it, and it is scale-free so an accurate stock cannot hide it.
    try testing.expectEqual(
        VerifyQmmParityFail.systematic,
        verifyQmmParityVerdict(0.4, 0.0009, 0.0040, 0.999999, 0.999995).?,
    );
    try testing.expectEqual(
        VerifyQmmParityFail.cosine,
        verifyQmmParityVerdict(0.005, 0.0012, 0.0012, 0.999999, 0.99990).?,
    );
    try testing.expectEqual(
        VerifyQmmParityFail.not_tracking,
        verifyQmmParityVerdict(0.005, 0.0012, 0.0012, 0.5, 0.4).?,
    );
    // A lane BETTER than stock is never a failure.
    try testing.expect(verifyQmmParityVerdict(0.0001, 0.0012, 0.0002, 0.999998, 0.999999) == null);
}
