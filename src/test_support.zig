const trf = @import("transformer.zig");
const io_util_mod = @import("io_util.zig");
const diagEnvOn = io_util_mod.diagEnvOn;
const std = @import("std");
const sushi_exl3 = @import("sushi_exl3");
const expert_exl3 = sushi_exl3.format;
const mlx = @import("mlx.zig");
const testing = std.testing;
const KVCacheSnapshot = trf.KVCacheSnapshot;
const MoeMlpWeights = trf.MoeMlpWeights;
const Transformer = trf.Transformer;


pub const Exl3MoeHarness = struct {
    xfm: Transformer,
    x: mlx.mlx_array,
    mw: MoeMlpWeights,
    owned: [10]mlx.mlx_array,

    pub const E: usize = 4;
    pub const dim: usize = 128;

    pub fn init(alloc: std.mem.Allocator, s: mlx.mlx_stream, rows: usize, seed: u64) !Exl3MoeHarness {
        return initDtype(alloc, s, rows, seed, .float16);
    }

    /// `act_dtype` is the dtype of x and of every dense weight; the EXL3 scale vectors stay f16.
    pub fn initDtype(alloc: std.mem.Allocator, s: mlx.mlx_stream, rows: usize, seed: u64, act_dtype: mlx.mlx_dtype) !Exl3MoeHarness {
        const exl3 = expert_exl3;
        const fixture = exl3.fixtures.k4;
        const header_len = std.mem.readInt(u64, fixture[0..8], .little);
        const header = fixture[8 .. 8 + header_len];
        const data = fixture[8 + header_len ..];
        const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header, .{});
        defer parsed.deinit();
        const trellis_meta = parsed.value.object.get("trellis").?.object;
        const suh_meta = parsed.value.object.get("suh").?.object;
        const svh_meta = parsed.value.object.get("svh").?.object;
        const t0: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[0].integer);
        const t1: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[1].integer);
        const s0: usize = @intCast(suh_meta.get("data_offsets").?.array.items[0].integer);
        const s1: usize = @intCast(suh_meta.get("data_offsets").?.array.items[1].integer);
        const v0: usize = @intCast(svh_meta.get("data_offsets").?.array.items[0].integer);
        const v1: usize = @intCast(svh_meta.get("data_offsets").?.array.items[1].integer);
        const trellis_bits = std.mem.bytesAsSlice(u16, data[t0..t1]);
        const suh_bits = std.mem.bytesAsSlice(u16, data[s0..s1]);
        const svh_bits = std.mem.bytesAsSlice(u16, data[v0..v1]);
        const tile_n = 8 * 8 * 64;
        const stacked_t = try alloc.alloc(u16, E * tile_n);
        const stacked_suh = try alloc.alloc(u16, E * dim);
        const stacked_svh = try alloc.alloc(u16, E * dim);
        for (0..E) |e| {
            @memcpy(stacked_t[e * tile_n ..][0..tile_n], trellis_bits);
            @memcpy(stacked_suh[e * dim ..][0..dim], suh_bits);
            @memcpy(stacked_svh[e * dim ..][0..dim], svh_bits);
        }
        var prng = std.Random.DefaultPrng.init(seed);
        const rnd = prng.random();
        const xh = try alloc.alloc(u16, rows * dim);
        for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
        const rw = try alloc.alloc(u16, dim * E);
        for (rw) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 0.05);
        const ident = try alloc.alloc(u16, dim * dim);
        @memset(ident, 0);
        for (0..dim) |i| ident[i * dim + i] = exl3.f32ToF16Bits(1.0);
        const gbits = try alloc.alloc(u16, dim);
        @memset(gbits, 0);
        const tr = mlx.mlx_array_new_data(stacked_t.ptr, &[_]c_int{ @intCast(E), 8, 8, 64 }, 4, .uint16);
        const suh = mlx.mlx_array_new_data(stacked_suh.ptr, &[_]c_int{ @intCast(E), @intCast(dim) }, 2, .float16);
        const svh = mlx.mlx_array_new_data(stacked_svh.ptr, &[_]c_int{ @intCast(E), @intCast(dim) }, 2, .float16);
        const router = mlx.mlx_array_new_data(rw.ptr, &[_]c_int{ @intCast(dim), @intCast(E) }, 2, .float16);
        const sh_gate = mlx.mlx_array_new_data(ident.ptr, &[_]c_int{ @intCast(dim), @intCast(dim) }, 2, .float16);
        const sh_up = mlx.mlx_array_new_data(ident.ptr, &[_]c_int{ @intCast(dim), @intCast(dim) }, 2, .float16);
        const sh_down = mlx.mlx_array_new_data(ident.ptr, &[_]c_int{ @intCast(dim), @intCast(dim) }, 2, .float16);
        const sh_eg = mlx.mlx_array_new_data(gbits.ptr, &[_]c_int{ @intCast(dim), 1 }, 2, .float16);
        const x = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ 1, @intCast(rows), @intCast(dim) }, 3, .float16);
        const one = mlx.mlx_array_new_float(1.0);
        const none = mlx.mlx_array{ .ctx = null };
        var h: Exl3MoeHarness = .{
            .xfm = undefined,
            .x = x,
            .owned = .{ tr, suh, svh, router, sh_gate, sh_up, sh_down, sh_eg, x, one },
            .mw = .{
                .router_w = router,
                .router_s = none,
                .router_b = none,
                .switch_gate_w = tr,
                .switch_gate_s = suh,
                .switch_gate_b = svh,
                .switch_up_w = tr,
                .switch_up_s = suh,
                .switch_up_b = svh,
                .switch_down_w = tr,
                .switch_down_s = suh,
                .switch_down_b = svh,
                .shared_gate_w = sh_gate,
                .shared_gate_s = none,
                .shared_gate_b = none,
                .shared_up_w = sh_up,
                .shared_up_s = none,
                .shared_up_b = none,
                .shared_down_w = sh_down,
                .shared_down_s = none,
                .shared_down_b = none,
                .shared_expert_gate_w = sh_eg,
                .shared_expert_gate_s = none,
                .shared_expert_gate_b = none,
            },
        };
        h.xfm.s = s;
        h.xfm.config = .{
            .expert_layout = .exl3_k4,
            .num_experts = @intCast(E),
            .num_experts_per_tok = @intCast(E),
            .hidden_size = @intCast(dim),
            .moe_intermediate_size = @intCast(dim),
            .quant_bits = 0,
            .quant_group_size = 64,
            .quant_mode = .affine,
            .hidden_act = .silu,
        };
        h.xfm.one = one;
        h.xfm.bits_cache = .{};
        h.xfm.compiled_moe_routing = null;
        h.xfm.compiled_gelu = null;
        h.xfm.compiled_geglu = null;
        h.xfm.cost_trace_active = false;
        if (act_dtype != .float16) {
            const dense = [_]*mlx.mlx_array{ &h.owned[3], &h.owned[4], &h.owned[5], &h.owned[6], &h.owned[7], &h.owned[8] };
            for (dense) |slot| {
                var cast = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_astype(&cast, slot.*, act_dtype, s));
                _ = mlx.mlx_array_free(slot.*);
                slot.* = cast;
            }
            h.x = h.owned[8];
            h.mw.router_w = h.owned[3];
            h.mw.shared_gate_w = h.owned[4];
            h.mw.shared_up_w = h.owned[5];
            h.mw.shared_down_w = h.owned[6];
            h.mw.shared_expert_gate_w = h.owned[7];
        }
        for (h.owned) |a| try mlx.check(mlx.mlx_array_eval(a));
        return h;
    }

    pub fn deinit(self: *Exl3MoeHarness) void {
        for (self.owned) |a| _ = mlx.mlx_array_free(a);
    }
};

pub fn testKVWide(seq_len: usize, head_dim: c_int, s: mlx.mlx_stream) mlx.mlx_array {
    const sl: c_int = @intCast(seq_len);
    const shape = [_]c_int{ 1, 2, sl, head_dim };
    var arr = mlx.mlx_array_new();
    _ = mlx.mlx_zeros(&arr, &shape, 4, .float32, s);
    return arr;
}

pub fn qkvIdentityArray(s: mlx.mlx_stream, heads: c_int, rows: c_int, dim: c_int, seed: f32, dtype: mlx.mlx_dtype) !mlx.mlx_array {
    const buf = try testing.allocator.alloc(f32, @intCast(heads * rows * dim));
    defer testing.allocator.free(buf);
    for (buf, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.173 + seed) * 1.7;
    const raw = mlx.mlx_array_new_data(buf.ptr, &[_]c_int{ 1, heads, rows, dim }, 4, .float32);
    defer _ = mlx.mlx_array_free(raw);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_astype(&out, raw, dtype, s));
    return out;
}

pub fn qkvExpectBitsEqual(s: mlx.mlx_stream, a: mlx.mlx_array, b: mlx.mlx_array) !void {
    try testing.expectEqual(mlx.mlx_array_dtype(a), mlx.mlx_array_dtype(b));
    try testing.expectEqualSlices(c_int, mlx.getShape(a), mlx.getShape(b));
    var ac = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ac);
    var bc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bc);
    try mlx.check(mlx.mlx_contiguous(&ac, a, false, s));
    try mlx.check(mlx.mlx_contiguous(&bc, b, false, s));
    const ah = try qwen4ReadF32(testing.allocator, ac, s);
    defer testing.allocator.free(ah);
    const bh = try qwen4ReadF32(testing.allocator, bc, s);
    defer testing.allocator.free(bh);
    try testing.expectEqual(ah.len, bh.len);
    for (ah, bh, 0..) |x, y, i| {
        try testing.expect(std.math.isFinite(x) and std.math.isFinite(y));
        if (@as(u32, @bitCast(x)) != @as(u32, @bitCast(y))) {
            std.debug.print("kv8 row differs at {d}: {d} vs {d}\n", .{ i, x, y });
            return error.RowNotBitIdentical;
        }
    }
}

/// Test helper: eval `arr`, flatten, and copy the first `out.len` f32 values to host.
pub fn testReadF32(arr: mlx.mlx_array, out: []f32, s: mlx.mlx_stream) !void {
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, arr, .float32, s));
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    const sh = [_]c_int{@intCast(out.len)};
    try mlx.check(mlx.mlx_reshape(&flat, f, &sh, 1, s));
    const ev = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ev);
    _ = mlx.mlx_vector_array_append_value(ev, flat);
    try mlx.check(mlx.mlx_eval(ev));
    const data = mlx.mlx_array_data_float32(flat) orelse return error.TestUnexpectedNullData;
    @memcpy(out, data[0..out.len]);
}

/// Host-side reference for one router row: exact top-K by key with the
/// LOWEST-INDEX tie-break, weights in f64. Returns the selected ids in
/// descending-key order, matching the kernel's emission order.
pub fn routerRefRow(keys: []const f64, k: usize, out_ids: []usize) void {
    var taken = std.testing.allocator.alloc(bool, keys.len) catch unreachable;
    defer std.testing.allocator.free(taken);
    @memset(taken, false);
    for (0..k) |r| {
        var best: f64 = -std.math.inf(f64);
        var bidx: usize = 0;
        var found = false;
        for (keys, 0..) |v, i| {
            if (taken[i]) continue;
            if (!found or v > best) {
                best = v;
                bidx = i;
                found = true;
            }
        }
        taken[bidx] = true;
        out_ids[r] = bidx;
    }
}

pub fn snapshotsBitEqual(a: *const KVCacheSnapshot, b: *const KVCacheSnapshot, s: mlx.mlx_stream) !bool {
    if (a.step != b.step or a.entries.len != b.entries.len) return false;
    for (a.entries, b.entries) |*x, *y| {
        if (x.initialized != y.initialized or x.ringed != y.ringed) return false;
        if (!x.initialized) continue;
        if (x.base != y.base or x.offset != y.offset) return false;
        const pairs = [_][2]mlx.mlx_array{
            .{ x.keys, y.keys },                   .{ x.values, y.values },
            .{ x.keys_scales, y.keys_scales },     .{ x.keys_biases, y.keys_biases },
            .{ x.values_scales, y.values_scales }, .{ x.values_biases, y.values_biases },
        };
        for (pairs) |p| {
            if ((p[0].ctx == null) != (p[1].ctx == null)) return false;
            if (p[0].ctx == null) continue;
            var eq = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(eq);
            try mlx.check(mlx.mlx_array_equal(&eq, p[0], p[1], false, s));
            try mlx.check(mlx.mlx_array_eval(eq));
            var same = false;
            try mlx.check(mlx.mlx_array_item_bool(&same, eq));
            if (!same) return false;
        }
    }
    return true;
}

/// Truncate an f32 to a bf16-representable value (top 16 bits). Test inputs
/// pass through this so the f64 host reference and the bf16 kernels consume
/// IDENTICAL values — the only error left is kernel accumulation/rounding.
pub fn bf16Trunc(x: f32) f32 {
    const bits: u32 = @bitCast(x);
    return @bitCast(bits & 0xFFFF_0000);
}

/// Eval an mlx array as f32 and copy its data into a fresh slice.
pub fn evalToF32(al: std.mem.Allocator, arr: mlx.mlx_array, n: usize, s: mlx.mlx_stream) ![]f32 {
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, arr, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f));
    const d = mlx.mlx_array_data_float32(f) orelse return error.InvalidDtype;
    const out = try al.alloc(f32, n);
    @memcpy(out, d[0..n]);
    return out;
}

/// Composed reference for the fused kernel tests: MLX's own SDPA (which takes
/// the composed matmul→softmax→matmul path at head_dim 256).
pub fn attn256Reference(
    q: mlx.mlx_array,
    k: mlx.mlx_array,
    v: mlx.mlx_array,
    scale: f32,
    mode: [*:0]const u8,
    mask: mlx.mlx_array,
    s: mlx.mlx_stream,
) !mlx.mlx_array {
    var ref = mlx.mlx_array_new();
    const none = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(&ref, q, k, v, scale, mode, if (mask.ctx != null) mask else none, .{ .ctx = null }, false, s));
    return ref;
}

pub fn attn256RandBf16(rnd: std.Random, shape: []const c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const data = try std.testing.allocator.alloc(f32, n);
    defer std.testing.allocator.free(data);
    for (data) |*x| x.* = rnd.float(f32) - 0.5;
    const f32arr = mlx.mlx_array_new_data(data.ptr, shape.ptr, @intCast(shape.len), .float32);
    defer _ = mlx.mlx_array_free(f32arr);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, f32arr, .bfloat16, s));
    return out;
}

/// Uniform [lo, hi) drawn by MLX: a host fill of a vocab-sized weight in a
/// Debug test build dominated the whole suite's runtime.
pub fn testRandUniformBf16(rnd: std.Random, shape: []const c_int, lo: f32, hi: f32, s: mlx.mlx_stream) !mlx.mlx_array {
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, rnd.int(u64)));
    const lo_a = mlx.mlx_array_new_float(lo);
    defer _ = mlx.mlx_array_free(lo_a);
    const hi_a = mlx.mlx_array_new_float(hi);
    defer _ = mlx.mlx_array_free(hi_a);
    var u = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(u);
    try mlx.check(mlx.mlx_random_uniform(&u, lo_a, hi_a, shape.ptr, shape.len, .float32, key, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, u, .bfloat16, s));
    return out;
}

pub fn testRandWeightBf16(rnd: std.Random, shape: []const c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    return testRandUniformBf16(rnd, shape, -0.5, 0.5, s);
}

pub fn attn256RandBf16Scaled(rnd: std.Random, shape: []const c_int, scale: f32, s: mlx.mlx_stream) !mlx.mlx_array {
    const unit = try attn256RandBf16(rnd, shape, s);
    defer _ = mlx.mlx_array_free(unit);
    const k = mlx.mlx_array_new_float(scale);
    defer _ = mlx.mlx_array_free(k);
    var wide = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wide);
    try mlx.check(mlx.mlx_multiply(&wide, unit, k, s));
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, wide, .bfloat16, s));
    return out;
}

/// Largest |element| of an array: the scale a bf16 rounding bar is relative to.
pub fn attn256MaxAbs(a: mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    var a_c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a_c);
    try mlx.check(mlx.mlx_contiguous(&a_c, a, false, s));
    var a32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a32);
    try mlx.check(mlx.mlx_astype(&a32, a_c, .float32, s));
    try mlx.check(mlx.mlx_array_eval(a32));
    const ad = mlx.mlx_array_data_float32(a32) orelse return error.InvalidDtype;
    const n = mlx.mlx_array_size(a32);
    var m: f32 = 0;
    for (0..n) |i| m = @max(m, @abs(ad[i]));
    return m;
}

pub fn attn256MaxDiff(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    // Raw data-pointer reads below: force row-major first. The fused sdpa
    // kernels hand back a [B,L,H,D]-strided VIEW and astype keeps the strides.
    var a_c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a_c);
    var b_c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b_c);
    try mlx.check(mlx.mlx_contiguous(&a_c, a, false, s));
    try mlx.check(mlx.mlx_contiguous(&b_c, b, false, s));
    var a32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a32);
    var b32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b32);
    try mlx.check(mlx.mlx_astype(&a32, a_c, .float32, s));
    try mlx.check(mlx.mlx_astype(&b32, b_c, .float32, s));
    try mlx.check(mlx.mlx_array_eval(a32));
    try mlx.check(mlx.mlx_array_eval(b32));
    const ad = mlx.mlx_array_data_float32(a32) orelse return error.InvalidDtype;
    const bd = mlx.mlx_array_data_float32(b32) orelse return error.InvalidDtype;
    const n = mlx.mlx_array_size(a32);
    if (n != mlx.mlx_array_size(b32)) return error.ShapeMismatch;
    var max_diff: f32 = 0;
    for (0..n) |i| max_diff = @max(max_diff, @abs(ad[i] - bd[i]));
    return max_diff;
}

pub fn qwen4ReadF32(alloc: std.mem.Allocator, arr: mlx.mlx_array, s: mlx.mlx_stream) ![]f32 {
    var f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f);
    try mlx.check(mlx.mlx_astype(&f, arr, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f));
    const n = mlx.mlx_array_size(f);
    const out = try alloc.alloc(f32, n);
    const src = mlx.mlx_array_data_float32(f) orelse return error.Unreadable;
    @memcpy(out, src[0..n]);
    return out;
}

/// Metal bytes held by LIVE arrays, with the allocator's free list flushed so
/// the number is a function of ownership alone.
pub fn mlxSettledActiveBytes(s: mlx.mlx_stream) usize {
    _ = mlx.mlx_synchronize(s);
    _ = mlx.mlx_clear_cache();
    var active: usize = 0;
    _ = mlx.mlx_get_active_memory(&active);
    return active;
}

pub fn qsaArraysAllEqual(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream) !bool {
    var eq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eq);
    try mlx.check(mlx.mlx_equal(&eq, a, b, s));
    var all = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(all);
    try mlx.check(mlx.mlx_all(&all, eq, false, s));
    try mlx.check(mlx.mlx_array_eval(all));
    var out: bool = false;
    try mlx.check(mlx.mlx_array_item_bool(&out, all));
    return out;
}

fn mimoFixtureMaxAbs(a: []const f32, b: []const f32) !f32 {
    if (a.len != b.len) return error.FixtureShapeMismatch;
    var max_err: f32 = 0;
    for (a, b) |x, y| {
        if (!std.math.isFinite(x) or !std.math.isFinite(y)) return error.NonFiniteFixture;
        max_err = @max(max_err, @abs(x - y));
    }
    return max_err;
}

pub fn mimoExpectClose(label: []const u8, actual: []const f32, expected: []const f32, limit: f32) !void {
    const err = try mimoFixtureMaxAbs(actual, expected);
    if (err > limit) {
        var peak: usize = 0;
        for (actual, expected, 0..) |x, y, i| {
            if (@abs(x - y) > @abs(actual[peak] - expected[peak])) peak = i;
        }
        std.debug.print("{s}: max absolute error {d} exceeds {d}; index {d}, actual {d}, expected {d}; first-row error {d}\n", .{
            label, err, limit, peak, actual[peak], expected[peak],
            try mimoFixtureMaxAbs(actual[0..@min(384, actual.len)], expected[0..@min(384, expected.len)]),
        });
        return error.TestExpectedEqual;
    }
}

/// A MiMo-shaped EXL3 MoE layer: hidden != inter, sigmoid routing with a
/// selection bias, no shared expert. Every expert holds the SAME bank and the
/// router is flat, so the normalized routed sum is one expert's own output
/// whichever top-k the GPU picks — that is the host oracle.
pub const MimoExl3MoeHarness = struct {
    xfm: Transformer,
    x: mlx.mlx_array,
    mw: MoeMlpWeights,
    owned: [10]mlx.mlx_array,
    gate_t: []u16,
    down_t: []u16,
    suh_h: []u16,
    svh_i: []u16,
    suh_i: []u16,
    svh_h: []u16,
    xf: []f32,
    dec: expert_exl3.Decode,
    rows: usize,

    const E: usize = 4;
    const hidden: usize = 128;
    const inter: usize = 256;

    pub fn init(alloc: std.mem.Allocator, s: mlx.mlx_stream, rate: expert_exl3.Rate, rows: usize, seed: u64, dec: expert_exl3.Decode) !MimoExl3MoeHarness {
        const packed_n = rate.halfwords();
        const gu_tile = (hidden / 16) * (inter / 16) * packed_n;
        const d_tile = (inter / 16) * (hidden / 16) * packed_n;
        var prng = std.Random.DefaultPrng.init(seed);
        const rnd = prng.random();
        const gate_t = try alloc.alloc(u16, E * gu_tile);
        const down_t = try alloc.alloc(u16, E * d_tile);
        for (gate_t[0..gu_tile]) |*v| v.* = @truncate(rnd.int(u32));
        for (down_t[0..d_tile]) |*v| v.* = @truncate(rnd.int(u32));
        for (1..E) |e| {
            @memcpy(gate_t[e * gu_tile ..][0..gu_tile], gate_t[0..gu_tile]);
            @memcpy(down_t[e * d_tile ..][0..d_tile], down_t[0..d_tile]);
        }
        const scale = struct {
            fn fill(bank: []u16, n: usize, r: std.Random) void {
                for (bank[0..n]) |*v| v.* = expert_exl3.f32ToF16Bits(if (r.boolean()) @as(f32, 0.9) else -0.9);
                var e: usize = 1;
                while (e * n < bank.len) : (e += 1) @memcpy(bank[e * n ..][0..n], bank[0..n]);
            }
        }.fill;
        const suh_h = try alloc.alloc(u16, E * hidden);
        const svh_i = try alloc.alloc(u16, E * inter);
        const suh_i = try alloc.alloc(u16, E * inter);
        const svh_h = try alloc.alloc(u16, E * hidden);
        scale(suh_h, hidden, rnd);
        scale(svh_i, inter, rnd);
        scale(suh_i, inter, rnd);
        scale(svh_h, hidden, rnd);
        const xh = try alloc.alloc(u16, rows * hidden);
        const xf = try alloc.alloc(f32, rows * hidden);
        for (xh, xf) |*b, *v| {
            b.* = expert_exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
            v.* = expert_exl3.f16BitsToF32(b.*);
        }
        // A flat router: every expert scores the same, so the normalized
        // weights sum to one whichever top-k the selection returns.
        const rw = try alloc.alloc(u16, hidden * E);
        @memset(rw, 0);
        const bias = try alloc.alloc(f32, E);
        @memset(bias, 0);
        const ci = struct {
            fn i(v: usize) c_int {
                return @intCast(v);
            }
        }.i;
        const gt = mlx.mlx_array_new_data(gate_t.ptr, &[_]c_int{ ci(E), ci(hidden / 16), ci(inter / 16), ci(packed_n) }, 4, .uint16);
        const dt = mlx.mlx_array_new_data(down_t.ptr, &[_]c_int{ ci(E), ci(inter / 16), ci(hidden / 16), ci(packed_n) }, 4, .uint16);
        const a_suh_h = mlx.mlx_array_new_data(suh_h.ptr, &[_]c_int{ ci(E), ci(hidden) }, 2, .float16);
        const a_svh_i = mlx.mlx_array_new_data(svh_i.ptr, &[_]c_int{ ci(E), ci(inter) }, 2, .float16);
        const a_suh_i = mlx.mlx_array_new_data(suh_i.ptr, &[_]c_int{ ci(E), ci(inter) }, 2, .float16);
        const a_svh_h = mlx.mlx_array_new_data(svh_h.ptr, &[_]c_int{ ci(E), ci(hidden) }, 2, .float16);
        const router = mlx.mlx_array_new_data(rw.ptr, &[_]c_int{ ci(hidden), ci(E) }, 2, .float16);
        const eb = mlx.mlx_array_new_data(bias.ptr, &[_]c_int{ci(E)}, 1, .float32);
        const x = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ 1, ci(rows), ci(hidden) }, 3, .float16);
        const one = mlx.mlx_array_new_float(1.0);
        const none = mlx.mlx_array{ .ctx = null };
        var h: MimoExl3MoeHarness = .{
            .xfm = undefined,
            .x = x,
            .owned = .{ gt, dt, a_suh_h, a_svh_i, a_suh_i, a_svh_h, router, eb, x, one },
            .gate_t = gate_t,
            .down_t = down_t,
            .suh_h = suh_h,
            .svh_i = svh_i,
            .suh_i = suh_i,
            .svh_h = svh_h,
            .xf = xf,
            .dec = dec,
            .rows = rows,
            .mw = .{
                .router_w = router,
                .router_s = none,
                .router_b = none,
                .switch_gate_w = gt,
                .switch_gate_s = a_suh_h,
                .switch_gate_b = a_svh_i,
                .switch_up_w = gt,
                .switch_up_s = a_suh_h,
                .switch_up_b = a_svh_i,
                .switch_down_w = dt,
                .switch_down_s = a_suh_i,
                .switch_down_b = a_svh_h,
                .shared_gate_w = none,
                .shared_gate_s = none,
                .shared_gate_b = none,
                .shared_up_w = none,
                .shared_up_s = none,
                .shared_up_b = none,
                .shared_down_w = none,
                .shared_down_s = none,
                .shared_down_b = none,
                .expert_bias = eb,
                .route_norm = true,
                .route_scale = 1.0,
            },
        };
        h.xfm.s = s;
        h.xfm.config = .{
            .model_type = "mimo_v2",
            .expert_layout = .exl3_k4,
            .expert_quant_rate = rate,
            .expert_quant_codebook = dec.codebook,
            .expert_quant_window = dec.window,
            .expert_quant_table = dec.table,
            .num_experts = @intCast(E),
            .num_experts_per_tok = 2,
            .hidden_size = @intCast(hidden),
            .moe_intermediate_size = @intCast(inter),
            .quant_bits = 0,
            .quant_group_size = 64,
            .quant_mode = .affine,
            .hidden_act = .silu,
            .moe_sigmoid_router = true,
            .moe_n_group = 1,
            .moe_topk_group = 1,
            .moe_route_norm = true,
        };
        h.xfm.one = one;
        h.xfm.bits_cache = .{};
        h.xfm.compiled_moe_routing = null;
        h.xfm.compiled_gelu = null;
        h.xfm.compiled_geglu = null;
        h.xfm.cost_trace_active = false;
        h.xfm.expert_stream = null;
        for (h.owned) |a| try mlx.check(mlx.mlx_array_eval(a));
        return h;
    }

    /// One expert's SwiGLU on the host, at the rate the trellis names.
    fn hostRow(self: *MimoExl3MoeHarness, alloc: std.mem.Allocator, rate: expert_exl3.Rate, row: usize, out: []f32) !void {
        const x = self.xf[row * hidden ..][0..hidden];
        const t_hidden = try alloc.alloc(f32, hidden);
        defer alloc.free(t_hidden);
        const t_inter = try alloc.alloc(f32, inter);
        defer alloc.free(t_inter);
        const inner = try alloc.alloc(f32, inter);
        defer alloc.free(inner);
        const gate_y = try alloc.alloc(f32, inter);
        defer alloc.free(gate_y);
        const up_y = try alloc.alloc(f32, inter);
        defer alloc.free(up_y);
        const gu_tile = (hidden / 16) * (inter / 16) * rate.halfwords();
        const d_tile = (inter / 16) * (hidden / 16) * rate.halfwords();
        expert_exl3.project(x, self.gate_t[0..gu_tile], self.suh_h[0..hidden], self.svh_i[0..inter], hidden, inter, rate, self.dec, t_hidden, inner, gate_y);
        expert_exl3.project(x, self.gate_t[0..gu_tile], self.suh_h[0..hidden], self.svh_i[0..inter], hidden, inter, rate, self.dec, t_hidden, inner, up_y);
        for (gate_y, up_y) |*g, u| g.* = (g.* / (1.0 + @exp(-g.*))) * u;
        expert_exl3.project(gate_y, self.down_t[0..d_tile], self.suh_i[0..inter], self.svh_h[0..hidden], inter, hidden, rate, self.dec, t_inter, inner[0..hidden], out);
    }

    /// One forward through the engine, scored against this harness's own
    /// codebook on the host.
    fn expectMatchesHost(self: *MimoExl3MoeHarness, alloc: std.mem.Allocator, rate: expert_exl3.Rate) !void {
        const y = try self.xfm.moeMLP(self.x, &self.mw);
        defer _ = mlx.mlx_array_free(y);
        var c = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(c);
        try mlx.check(mlx.mlx_contiguous(&c, y, false, self.xfm.s));
        try mlx.check(mlx.mlx_array_eval(c));
        const got = mlx.mlx_array_data_float16(c) orelse return error.F16Unreadable;
        const want = try alloc.alloc(f32, hidden);
        defer alloc.free(want);
        var ss: f64 = 0;
        var ref: f64 = 0;
        for (0..self.rows) |r| {
            try self.hostRow(alloc, rate, r, want);
            for (want, 0..) |w, i| {
                const a: f64 = @floatCast(got[r * hidden + i]);
                ss += (a - w) * (a - w);
                ref += @as(f64, w) * @as(f64, w);
            }
        }
        const rel = @sqrt(ss / @max(ref, 1e-20));
        if (!(rel < 0.01)) {
            std.debug.print("mimo exl3 forward codebook={s} window={d} rel_rms={d:.6}\n", .{ @tagName(self.dec.codebook), self.dec.window.bits(), rel });
            return error.TestExpectedEqual;
        }
    }

    pub fn deinit(self: *MimoExl3MoeHarness) void {
        for (self.owned) |a| _ = mlx.mlx_array_free(a);
    }
};

pub fn mimoExl3ForwardMatchesHost(rate: expert_exl3.Rate, rows: usize, seed: u64) !void {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var h = try MimoExl3MoeHarness.init(alloc, s, rate, rows, seed, .mcg);
    defer h.deinit();
    try h.expectMatchesHost(alloc, rate);
}

/// Two EXL3 packs on different codebooks, resident together and dispatching
/// alternately: every forward must decode with ITS OWN model's codebook.
pub fn exl3CodebookFollowsModel(rows: usize) !void {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const rate: expert_exl3.Rate = .{ .n = 40 };
    var mul1 = try MimoExl3MoeHarness.init(alloc, s, rate, rows, 101, .mul1);
    defer mul1.deinit();
    var mcg = try MimoExl3MoeHarness.init(alloc, s, rate, rows, 103, .mcg);
    defer mcg.deinit();
    for (0..2) |_| {
        try mul1.expectMatchesHost(alloc, rate);
        try mcg.expectMatchesHost(alloc, rate);
    }
}

pub fn mimoExl3DistinctRouting(rows: usize) !void {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const H = MimoExl3MoeHarness.hidden;
    const I = MimoExl3MoeHarness.inter;
    const E = MimoExl3MoeHarness.E;
    const rate: expert_exl3.Rate = .{ .n = 40 };
    var h = try MimoExl3MoeHarness.init(alloc, s, rate, rows, 191, .{ .codebook = .mcg, .window = .w12 });
    defer h.deinit();
    var prng = std.Random.DefaultPrng.init(193);
    const rnd = prng.random();
    for (h.gate_t) |*v| v.* = rnd.int(u16);
    for (h.down_t) |*v| v.* = rnd.int(u16);
    const router = try alloc.alloc(f32, H * E);
    for (router) |*v| {
        const value = (rnd.float(f32) * 2 - 1) * 0.2;
        const bits: u32 = @bitCast(value);
        v.* = @bitCast((bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000);
    }
    const bias = [_]f32{ 0.03, -0.08, 0.11, -0.02 };
    const replace = struct {
        fn array(dst: *mlx.mlx_array, src: mlx.mlx_array) void {
            _ = mlx.mlx_array_free(dst.*);
            dst.* = src;
        }
    }.array;
    replace(&h.owned[0], mlx.mlx_array_new_data(h.gate_t.ptr, &.{ E, H / 16, I / 16, 40 }, 4, .uint16));
    replace(&h.owned[1], mlx.mlx_array_new_data(h.down_t.ptr, &.{ E, I / 16, H / 16, 40 }, 4, .uint16));
    replace(&h.owned[6], mlx.mlx_array_new_data(router.ptr, &.{ H, E }, 2, .float32));
    replace(&h.owned[7], mlx.mlx_array_new_data(&bias, &.{E}, 1, .float32));
    var xb = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&xb, h.x, .bfloat16, s));
    replace(&h.owned[8], xb);
    h.x = xb;
    h.mw.switch_gate_w = h.owned[0];
    h.mw.switch_up_w = h.owned[0];
    h.mw.switch_down_w = h.owned[1];
    h.mw.router_w = h.owned[6];
    h.mw.expert_bias = h.owned[7];
    var x32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x32);
    try mlx.check(mlx.mlx_astype(&x32, h.x, .float32, s));
    try mlx.check(mlx.mlx_array_eval(x32));
    @memcpy(h.xf, (mlx.mlx_array_data_float32(x32) orelse return error.F32Unreadable)[0..h.xf.len]);
    const logits = try h.xfm.mimoRouterLogits(h.x, &h.mw);
    defer _ = mlx.mlx_array_free(logits);
    const routed = try h.xfm.computeMimoRouting(logits, h.mw.expert_bias.?);
    defer _ = mlx.mlx_array_free(routed.inds);
    defer _ = mlx.mlx_array_free(routed.norm_scores);
    try t.expectEqual(mlx.mlx_dtype.float32, mlx.mlx_array_dtype(routed.norm_scores));
    var ids32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids32);
    try mlx.check(mlx.mlx_astype(&ids32, routed.inds, .uint32, s));
    try mlx.check(mlx.mlx_array_eval(ids32));
    try mlx.check(mlx.mlx_array_eval(routed.norm_scores));
    const ids = mlx.mlx_array_data_uint32(ids32) orelse return error.ExpertIdsUnreadable;
    const scores = mlx.mlx_array_data_float32(routed.norm_scores) orelse return error.F32Unreadable;
    const y = try h.xfm.moeMLP(h.x, &h.mw);
    defer _ = mlx.mlx_array_free(y);
    try t.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(y));
    var y32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y32);
    try mlx.check(mlx.mlx_astype(&y32, y, .float32, s));
    try mlx.check(mlx.mlx_array_eval(y32));
    const got = mlx.mlx_array_data_float32(y32) orelse return error.F32Unreadable;
    var err2: f64 = 0;
    var ref2: f64 = 0;
    var seen: u32 = 0;
    for (0..rows) |r| {
        var probabilities: [E]f32 = undefined;
        for (0..E) |e| {
            var dot: f32 = 0;
            for (0..H) |d| dot += h.xf[r * H + d] * router[d * E + e];
            probabilities[e] = 1 / (1 + @exp(-dot));
        }
        const den = probabilities[ids[r * 2]] + probabilities[ids[r * 2 + 1]];
        var want: [H]f32 = @splat(0);
        for (0..2) |k| {
            const e = ids[r * 2 + k];
            try t.expect(e < E);
            seen |= @as(u32, 1) << @intCast(e);
            const score = probabilities[e] / den;
            try t.expectApproxEqAbs(score, scores[r * 2 + k], 1e-6);
            for (0..E) |other| {
                if (other == ids[r * 2] or other == ids[r * 2 + 1]) continue;
                try t.expect(probabilities[e] + bias[e] >= probabilities[other] + bias[other]);
            }
            var oracle = h;
            oracle.gate_t = h.gate_t[e * (h.gate_t.len / E) ..];
            oracle.down_t = h.down_t[e * (h.down_t.len / E) ..];
            oracle.suh_h = h.suh_h[e * H ..];
            oracle.svh_i = h.svh_i[e * I ..];
            oracle.suh_i = h.suh_i[e * I ..];
            oracle.svh_h = h.svh_h[e * H ..];
            var out: [H]f32 = undefined;
            try oracle.hostRow(alloc, rate, r, &out);
            for (&want, out) |*w, value| w.* += value * score;
        }
        for (want, 0..) |w, d| {
            const actual = got[r * H + d];
            try t.expect(std.math.isFinite(actual) and std.math.isFinite(w));
            const delta: f64 = actual - w;
            err2 += delta * delta;
            ref2 += @as(f64, w) * w;
        }
    }
    if (rows > 1) try t.expect(@popCount(seen) > 2);
    const relative = @sqrt(err2 / @max(ref2, 1e-20));
    if (diagEnvOn("SUSHI_EXL3_PARITY_TRACE"))
        std.debug.print("mimo distinct routed MoE rows={d} relative RMS={d:.8}\n", .{ rows, relative });
    if (!(relative < 0.01)) {
        std.debug.print("mimo distinct routed MoE rows={d} relative RMS={d:.8}\n", .{ rows, relative });
        return error.TestExpectedEqual;
    }
}
