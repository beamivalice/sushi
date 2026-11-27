//! MLX's fused scaled dot-product attention on the engine: `fast::scaled_dot_product_attention` (mlx/fast.cpp) and
//! `ScaledDotProductAttention::eval_gpu` with `sdpa_full_self_attention_nax`, `sdpa_vector` and `sdpa_vector_2pass`
//! (mlx/backend/metal/scaled_dot_product_attention.cpp) over the compiled kernels of `sushi.metallib`, so the output
//! has the bytes and the strides it has under MLX. Ported from MLX's Metal backend (MIT, see NOTICE). `routeOf` is
//! MLX's choice between these kernels and its op chain (the route runs the chain on the engine's own ops). Ported, with
//! no array mask and no sinks and one dtype: the vector kernels at head dimensions 64 to 256, causal or not; the full
//! kernel on NAX at 64 and, head-split, at 256, causal or not; on the SIMD tiles (no matrix units, or f32 without TF32)
//! at 64 with no mask. Any other fused call is refused by name.

const std = @import("std");
const metal = @import("../metal.zig");
const core = @import("core.zig");
const matmul = @import("matmul.zig");
const ops = @import("ops.zig");
const stock = @import("stock.zig");

const Tensor = core.Tensor;
const Error = core.Error;

/// `mlx::steel::AttnParams` (kernels/steel/attn/params.h).
const AttnParams = extern struct {
    b: c_int,
    h: c_int,
    d: c_int,
    q_len: c_int,
    k_len: c_int,
    gqa_factor: c_int,
    scale: f32,
    nq: c_int,
    nk: c_int,
    nq_aligned: c_int,
    nk_aligned: c_int,
    q_len_rem: c_int,
    k_len_rem: c_int,
    q_len_off: c_int,
    q_strides: [3]i64,
    k_strides: [3]i64,
    v_strides: [3]i64,
    o_strides: [3]i64,
};

const head_dim = 64;
/// The vector kernels take at most this many query rows; a longer query takes the full kernel.
const vector_rows = 8;
/// The full kernel's query and key tiles, on the NAX arm and on the SIMD arm.
const Tiles = struct { bq: usize, bk: usize };
const nax_tiles: Tiles = .{ .bq = 64, .bk = 32 };
const simd_tiles: Tiles = .{ .bq = 32, .bk = 32 };

/// The tiles MLX's `sdpa_full_self_attention_metal` takes for a head dimension of 64: NAX with matrix units, and for f32
/// only with TF32.
fn tilesOf(g: matmul.Gpu, dtype: core.Dtype) Tiles {
    return if (g.nax and (dtype != .float32 or g.tf32)) nax_tiles else simd_tiles;
}

/// The pipeline `attention` runs for `q_len` query rows over `k_len` keys, `gqa` query heads to a key head, on `g`:
/// what its name and function constants follow. A load's warm reads it to reach every pipeline its shapes take.
pub const Plan = union(enum) {
    full: Aligned,
    one_pass,
    two_pass: struct { blocks: c_int, wide: bool },
};

/// The full kernel's alignment constants: whether the query and the keys are whole tiles.
const Aligned = struct { align_q: bool, align_k: bool };

fn alignedOf(t: Tiles, q_len: usize, k_len: usize) Aligned {
    return .{ .align_q = q_len % t.bq == 0, .align_k = k_len % t.bk == 0 };
}

pub fn planOf(g: matmul.Gpu, dtype: core.Dtype, q_len: usize, k_len: usize, gqa: usize, dim: usize) Plan {
    if (q_len > vector_rows) return .{ .full = alignedOf(tilesOf(g, dtype), q_len, k_len) };
    // Two passes for a long key axis on a Max or an Ultra, or for keys shared by many heads.
    if (((g.class == .d or g.class == .s) and k_len >= 1024) or (gqa > 1 and k_len >= 4096)) {
        const gqa_dims = (gqa == 8 and dim == 64) or (gqa == 8 and dim == 128) or ((gqa == 12 or gqa == 16) and dim == 128);
        return .{ .two_pass = .{ .blocks = blocksOf(g, k_len, gqa * q_len), .wide = q_len == 1 and gqa_dims and k_len >= 8192 } };
    }
    return .one_pass;
}

pub const MaskKind = enum { none, causal, array };

/// One `fast::scaled_dot_product_attention` call as MLX decides on it: shapes of `[B, H, L, D]` queries and keys.
pub const Ask = struct {
    q_len: usize,
    k_len: usize,
    heads: usize,
    kv_heads: usize,
    q_dim: usize,
    v_dim: usize,
    dtype: core.Dtype,
    mask: MaskKind,
    force_fused: bool = false,
};

/// Where MLX sends a call: its op chain (`fallback`), a fused kernel, or an `invalid_argument` it throws.
pub const Route = union(enum) { fallback, vector, full, invalid: []const u8 };

/// `has_fused_kernel` and `ScaledDotProductAttention::use_fallback` (mlx/backend/metal/scaled_dot_product_attention.cpp)
/// for an inference call on a GPU stream. Head dimension 512 is left out: its arms read an environment variable of MLX's.
pub fn routeOf(g: matmul.Gpu, a: Ask) Route {
    const fused = fusedGap(a) == null;
    if (a.force_fused) return if (fused) (if (a.q_len > vector_rows) .full else .vector) else .{ .invalid = fusedGap(a).? };
    if (!fused) return .fallback;
    if (a.q_len > vector_rows) {
        const tiles_nax = g.nax and (g.tf32 or a.dtype != .float32);
        if (tiles_nax and a.q_len >= 1024 and a.q_dim == 256 and a.mask != .none) return .full;
        if (!g.nax and a.dtype != .float32 and a.mask == .causal and a.q_dim == 256 and a.q_len >= 2048 and a.q_len == a.k_len) return .full;
        return if (a.q_dim == 192 or a.q_dim == 256) .fallback else .full;
    }
    return if (a.q_dim == a.v_dim and a.q_dim == 192) .fallback else .vector;
}

/// A call MLX runs, as the call tally names it: the route, mask, head dimension, query rows by MLX's thresholds, keys
/// against queries, GQA factor and dtype.
pub fn tallyKey(buf: []u8, a: Ask) Error![]const u8 {
    const rows = if (a.q_len == 1) "1" else if (a.q_len <= vector_rows) "2-8" else if (a.q_len < 1024) "9-1023" else if (a.q_len < 2048) "1024-2047" else "2048+";
    const keys = if (a.k_len == a.q_len) "k=q" else if (a.k_len > a.q_len) "k>q" else "k<q";
    return std.fmt.bufPrint(buf, "{s} {s} D{d} q{s} {s} gqa{d} {s}{s}", .{
        @tagName(routeOf(try matmul.gpu(), a)), @tagName(a.mask), a.q_dim, rows, keys, a.heads / a.kv_heads, @tagName(a.dtype), if (a.force_fused) " forced" else "",
    }) catch error.TensorShape;
}

/// Why no fused kernel takes `a` (`has_fused_kernel`), or null.
fn fusedGap(a: Ask) ?[]const u8 {
    if (a.q_dim == 512) return "head dimension 512 is not ported";
    const gqa = a.heads / a.kv_heads;
    if (a.q_len > vector_rows) {
        const same = a.q_dim == a.v_dim;
        const dims = (a.q_dim == 96 and a.v_dim == 64) or (same and switch (a.q_dim) {
            64, 72, 80, 96, 128, 192, 256 => true,
            else => false,
        });
        if (!dims) return "the full attention kernel takes no such head dimensions";
        if (a.mask == .causal and a.q_len > a.k_len) return "the full attention kernel with a causal mask needs no more queries than keys";
        return null;
    }
    const same = a.q_dim == a.v_dim;
    const dims = (same and switch (a.q_dim) {
        64, 96, 128, 192, 256 => true,
        else => false,
    }) or (a.q_dim == 192 and a.v_dim == 128) or (a.q_dim == 96 and a.v_dim == 64);
    if (!dims) return "the vector attention kernel takes no such head dimensions";
    if (a.q_len > a.k_len) return "the vector attention kernel needs no more queries than keys";
    if (a.q_len * gqa > 32) return "the vector attention kernel takes at most 32 query rows per key head";
    return null;
}

/// `causal`: the causal mask, which only the vector kernels take here.
pub fn attention(q_pre: *Tensor, k_pre: *Tensor, v_pre: *Tensor, scale: f32, causal: bool) Error!*Tensor {
    if (q_pre.ndim != 4 or k_pre.ndim != 4 or v_pre.ndim != 4) return error.TensorShape;
    if (q_pre.dtype != k_pre.dtype or q_pre.dtype != v_pre.dtype) return core.refusing("attention over q, k and v of different dtypes", error.TensorDtype);
    switch (q_pre.dtype) {
        .float16, .bfloat16, .float32 => {},
        else => return core.refusing("attention in a dtype other than f32, f16 and bf16", error.TensorDtype),
    }
    const batch = q_pre.shape[0];
    const heads = q_pre.shape[1];
    const q_len: usize = @intCast(q_pre.shape[2]);
    const k_heads = k_pre.shape[1];
    if (k_pre.shape[0] != batch or v_pre.shape[0] != batch or q_pre.shape[3] != k_pre.shape[3] or k_pre.shape[1] != v_pre.shape[1] or
        k_pre.shape[2] != v_pre.shape[2] or k_heads == 0 or @rem(heads, k_heads) != 0) return error.TensorShape;
    if (q_len == 0 or k_pre.shape[2] == 0 or batch == 0 or heads == 0) return core.refusing("an empty query, key, batch or head axis", error.TensorUnsupported);
    const k_len: usize = @intCast(k_pre.shape[2]);
    const g = try matmul.gpu();
    const gqa: usize = @intCast(@divExact(heads, k_heads));

    if (q_len > vector_rows) {
        const dim = q_pre.shape[3];
        if (v_pre.shape[3] != dim) return core.refusing("a full attention kernel with value and query head dimensions that differ", error.TensorUnsupported);
        const tiles = tilesOf(g, q_pre.dtype);
        if (tiles.bq == nax_tiles.bq) {
            if (dim != 64 and dim != 256) return core.refusing("a full attention kernel on NAX at a head dimension other than 64 and 256", error.TensorUnsupported);
        } else {
            if (dim != head_dim) return core.refusing("a full attention kernel on SIMD tiles at a head dimension other than 64", error.TensorUnsupported);
            if (causal) return core.refusing("a full attention kernel on SIMD tiles with a causal mask", error.TensorUnsupported);
        }
        return full(tiles, q_pre, k_pre, v_pre, scale, causal);
    }
    if (q_pre.shape[3] != v_pre.shape[3] or switch (q_pre.shape[3]) {
        64, 96, 128, 256 => false,
        else => true,
    }) return core.refusing("a vector attention kernel at a head dimension other than 64, 96, 128 and 256", error.TensorUnsupported);
    if (q_len > k_len or q_len * gqa > 32) return core.refusing("a vector attention with more queries than keys, or over 32 query rows per key head", error.TensorUnsupported);
    return vector(g, q_pre, k_pre, v_pre, scale, gqa, causal and q_len > 1);
}

fn strideOf(t: *const Tensor, axis: usize) i64 {
    return @intCast(t.strides[axis]);
}

fn copyUnlessLastDense(t: *Tensor) Error!*Tensor {
    return if (t.strides[3] == 1) core.share(t) else ops.contiguousCopy(t);
}

/// `sdpa_full_self_attention_nax` or `_metal`: the output is laid out `[B, qL, H, D]` and read as `[B, H, qL, D]`. On
/// NAX a head dimension of 256 takes the head-split kernel (`steel_attention_dsplit_`), two simdgroup columns of 128.
fn full(tiles: Tiles, q_pre: *Tensor, k_pre: *Tensor, v_pre: *Tensor, scale: f32, causal: bool) Error!*Tensor {
    const q = try copyUnlessLastDense(q_pre);
    defer core.release(q);
    const k = try copyUnlessLastDense(k_pre);
    defer core.release(k);
    const v = try copyUnlessLastDense(v_pre);
    defer core.release(v);

    const dim: usize = @intCast(q.shape[3]);
    const split = dim == 256;
    const bq = tiles.bq;
    const bk = tiles.bk;
    const wm: usize = 4;
    const wn: usize = if (split) dim / 128 else 1;
    const b: usize = @intCast(q.shape[0]);
    const h: usize = @intCast(q.shape[1]);
    const q_len: usize = @intCast(q.shape[2]);
    const true_k_len: usize = @intCast(k.shape[2]);
    // The causal diagonal stays at the true key length; a whole last tile is read when the cache backs its rows.
    var k_len = true_k_len;
    if (split and causal and k_len % bk != 0) {
        const widened = bk * ((k_len + bk - 1) / bk);
        if (backsRows(k, widened) and backsRows(v, widened)) k_len = widened;
    }
    const laid = try core.fresh(q.dtype, &.{ q.shape[0], q.shape[2], q.shape[1], q.shape[3] });
    defer core.release(laid);
    const out = try ops.transpose(laid, &.{ 0, 2, 1, 3 });
    errdefer core.release(out);

    const aligned = alignedOf(tiles, q_len, k_len);
    const align_q = aligned.align_q;
    const align_k = aligned.align_k;
    const type_name = try stock.typeName(q.dtype);
    var name_buf: [160]u8 = undefined;
    const name = try stock.named(&name_buf, "{s}{s}_bq{d}_bk{d}_bd{d}_bv{d}_wm{d}_wn{d}_mask{s}", .{ if (split) "steel_attention_dsplit_" else "steel_attention_", type_name, bq, bk, dim, dim, wm, wn, type_name });
    var hash_buf: [256]u8 = undefined;
    const hash = try stock.named(&hash_buf, "{s}_align_Q_{c}_align_K_{c}_has_mask_n_do_causal_{c}_has_sinks_n", .{ name, tf(align_q), tf(align_k), tf(causal) });
    const kern = try stock.kernelFlags(name, hash, &.{
        .{ .index = 200, .value = align_q },
        .{ .index = 201, .value = align_k },
        .{ .index = 300, .value = false },
        .{ .index = 301, .value = causal },
        .{ .index = 302, .value = false },
    });
    const nq = (q_len + bq - 1) / bq;
    const nk = (k_len + bk - 1) / bk;
    const params: AttnParams = .{
        .b = @intCast(b),
        .h = @intCast(h),
        .d = @intCast(dim),
        .q_len = @intCast(q_len),
        .k_len = @intCast(k_len),
        .gqa_factor = @intCast(h / @as(usize, @intCast(k.shape[1]))),
        .scale = scale,
        .nq = @intCast(nq),
        .nk = @intCast(nk),
        .nq_aligned = @intCast(q_len / bq),
        .nk_aligned = @intCast(k_len / bk),
        .q_len_rem = @intCast(q_len - (q_len / bq) * bq),
        .k_len_rem = @intCast(k_len - (k_len / bk) * bk),
        .q_len_off = @intCast(true_k_len - q_len),
        .q_strides = .{ strideOf(q, 0), strideOf(q, 1), strideOf(q, 2) },
        .k_strides = .{ strideOf(k, 0), strideOf(k, 1), strideOf(k, 2) },
        .v_strides = .{ strideOf(v, 0), strideOf(v, 1), strideOf(v, 2) },
        .o_strides = .{ strideOf(out, 0), strideOf(out, 1), strideOf(out, 2) },
    };
    const p = try core.pass();
    p.pipeline(kern);
    core.bind(p, q, 0);
    core.bind(p, k, 1);
    core.bind(p, v, 2);
    core.bindOut(p, out, 3);
    p.constant(params, 4);
    core.dispatchGroups(p, .{ .x = nq, .y = h, .z = b }, .{ .x = 32, .y = wm, .z = wn });
    return out;
}

/// MLX's `has_backing_rows`: `rows` rows of each head sit inside the head's row pitch and inside the store, as when
/// keys are a slice of a longer cache.
pub fn backsRows(t: *const Tensor, rows: usize) bool {
    const st = t.strides;
    if (st[1] == 0 or st[2] == 0 or st[1] % st[2] != 0) return false;
    const item = core.itemSize(t.dtype);
    const pitch = st[1] / st[2];
    const row0 = ((t.offset / item) % st[1]) / st[2];
    if (row0 + rows > pitch) return false;
    const end = (@as(usize, @intCast(t.shape[0])) - 1) * st[0] + (@as(usize, @intCast(t.shape[1])) - 1) * st[1] + (rows - 1) * st[2] + @as(usize, @intCast(t.shape[3]));
    return t.offset + end * item <= t.store.len;
}

fn tf(yes: bool) u8 {
    return if (yes) 't' else 'n';
}

/// The vector kernels' layout rules: a query is copied unless it is row-contiguous or a transposed singleton batch
/// or head; keys and values unless their rows are dense and their batch and head axes are one run.
fn queryKept(t: *const Tensor) bool {
    if (t.rowContiguous()) return true;
    if (t.shape[0] == 1 or t.shape[1] == 1) {
        const other: usize = if (t.shape[0] == 1) 1 else 0;
        return t.strides[3] == 1 and t.strides[2] == @as(usize, @intCast(t.shape[3])) * @as(usize, @intCast(t.shape[other])) and t.strides[other] == @as(usize, @intCast(t.shape[3]));
    }
    return false;
}

fn keysKept(t: *const Tensor) bool {
    if (t.strides[3] != 1) return false;
    if (t.shape[0] == 1 or t.shape[1] == 1) return true;
    return t.strides[0] == t.strides[1] * @as(usize, @intCast(t.shape[1]));
}

fn vector(g: matmul.Gpu, q_pre: *Tensor, k_pre: *Tensor, v_pre: *Tensor, scale: f32, gqa: usize, causal: bool) Error!*Tensor {
    const q = if (queryKept(q_pre)) try core.share(q_pre) else try ops.contiguousCopy(q_pre);
    defer core.release(q);
    const k = if (keysKept(k_pre)) try core.share(k_pre) else try ops.contiguousCopy(k_pre);
    defer core.release(k);
    const v = if (keysKept(v_pre)) try core.share(v_pre) else try ops.contiguousCopy(v_pre);
    defer core.release(v);
    const out = try core.fresh(q.dtype, &.{ q.shape[0], q.shape[1], q.shape[2], v.shape[3] });
    errdefer core.release(out);
    switch (planOf(g, q.dtype, @intCast(q.shape[2]), @intCast(k.shape[2]), gqa, @intCast(q.shape[3]))) {
        .two_pass => |plan| try twoPass(plan.blocks, plan.wide, q, k, v, out, scale, gqa, causal),
        else => try onePass(q, k, v, out, scale, gqa, causal),
    }
    return out;
}

fn headStride(t: *const Tensor) u64 {
    return @intCast(if (t.shape[1] == 1) t.strides[0] else t.strides[1]);
}

/// The bool function constants 20 to 25 of the vector kernels with no array mask and no sinks, and the name suffix MLX
/// gives them.
fn vectorFlags(q: *const Tensor, causal: bool, flags: *[6]metal.Flag, hash: []u8, base: []const u8) Error![]const u8 {
    const transposed = !q.rowContiguous();
    flags.* = .{
        .{ .index = 20, .value = false },
        .{ .index = 21, .value = transposed },
        .{ .index = 22, .value = causal },
        .{ .index = 23, .value = false },
        .{ .index = 24, .value = false },
        .{ .index = 25, .value = false },
    };
    return std.fmt.bufPrint(hash, "{s}_nomask{s}{s}", .{ base, if (transposed) "_qt" else "_qnt", if (causal) "_c" else "_nc" }) catch error.TensorShape;
}

fn onePass(q: *Tensor, k: *Tensor, v: *Tensor, out: *Tensor, scale: f32, gqa: usize, causal: bool) Error!void {
    var name_buf: [96]u8 = undefined;
    const name = try stock.named(&name_buf, "sdpa_vector_{s}_{d}_{d}", .{ try stock.cType(q.dtype), q.shape[3], v.shape[3] });
    var flags: [6]metal.Flag = undefined;
    var hash_buf: [160]u8 = undefined;
    const prefix = try vectorFlags(q, causal, &flags, &hash_buf, name);
    var full_hash: [192]u8 = undefined;
    const hash = try stock.named(&full_hash, "{s}_nosinks", .{prefix});
    const kern = try stock.kernelFlags(name, hash, &flags);
    const p = try core.pass();
    p.pipeline(kern);
    core.bind(p, q, 0);
    core.bind(p, k, 1);
    core.bind(p, v, 2);
    core.bindOut(p, out, 3);
    p.constant(@as(c_int, @intCast(gqa)), 4);
    p.constant(@as(c_int, @intCast(k.shape[2])), 5);
    p.constant(headStride(k), 6);
    p.constant(@as(u64, @intCast(k.strides[2])), 7);
    p.constant(headStride(v), 8);
    p.constant(@as(u64, @intCast(v.strides[2])), 9);
    p.constant(scale, 10);
    core.dispatchGroups(p, .{ .x = @as(usize, @intCast(q.shape[0])) * @as(usize, @intCast(q.shape[1])), .y = @intCast(q.shape[2]), .z = 1 }, .{ .x = 1024, .y = 1, .z = 1 });
}

/// MLX's count of partial blocks the keys are cut into.
fn blocksOf(g: matmul.Gpu, n: usize, n_simds: usize) c_int {
    var blocks: c_int = undefined;
    if (g.class == .s) {
        blocks = 64;
        if (n > 1024 and n_simds > 4) {
            blocks = if (n <= 8192) 128 else if (n <= 32768) 256 else if (n <= 65536) 512 else 1024;
        }
    } else if (g.class == .d) {
        blocks = 128;
        if (n_simds <= 2 and n > 8192) {
            blocks = 256;
        } else if (n_simds >= 6) {
            if (n >= 16384 and n < 65536) blocks = 512 else if (n >= 65536) blocks = 1024;
        }
    } else {
        blocks = if (n_simds >= 4) 64 else 32;
    }
    return blocks;
}

fn twoPass(blocks: c_int, wide: bool, q: *Tensor, k: *Tensor, v: *Tensor, out: *Tensor, scale: f32, gqa: usize, causal: bool) Error!void {
    const q_len: usize = @intCast(q.shape[2]);
    const n: usize = @intCast(k.shape[2]);
    var name_buf: [96]u8 = undefined;
    const name = if (wide)
        try stock.named(&name_buf, "sdpa_vector_2pass_1_gqa_{d}_{s}_{d}_{d}", .{ gqa, try stock.cType(q.dtype), q.shape[3], v.shape[3] })
    else
        try stock.named(&name_buf, "sdpa_vector_2pass_1_{s}_{d}_{d}", .{ try stock.cType(q.dtype), q.shape[3], v.shape[3] });

    var mid_shape: [5]c_int = undefined;
    @memcpy(mid_shape[0..3], out.shape[0..3]);
    mid_shape[3] = blocks;
    mid_shape[4] = out.shape[3];
    const partial = try core.fresh(q.dtype, &mid_shape);
    defer core.release(partial);
    const sums = try core.fresh(.float32, mid_shape[0..4]);
    defer core.release(sums);
    const maxs = try core.fresh(.float32, mid_shape[0..4]);
    defer core.release(maxs);

    var flags: [6]metal.Flag = undefined;
    var hash_buf: [160]u8 = undefined;
    const prefix = try vectorFlags(q, causal, &flags, &hash_buf, name);
    var full_hash: [192]u8 = undefined;
    const hash = try stock.named(&full_hash, "{s}_nosinks_{d}", .{ prefix, blocks });
    const kern = try stock.kernelConstants(name, hash, &flags, &.{.{ .index = 26, .value = blocks }});
    const p = try core.pass();
    p.pipeline(kern);
    core.bind(p, q, 0);
    core.bind(p, k, 1);
    core.bind(p, v, 2);
    core.bindOut(p, partial, 3);
    core.bindOut(p, sums, 4);
    core.bindOut(p, maxs, 5);
    p.constant(@as(c_int, @intCast(n)), 7);
    p.constant(headStride(k), 8);
    p.constant(@as(u64, @intCast(k.strides[2])), 9);
    p.constant(headStride(v), 10);
    p.constant(@as(u64, @intCast(v.strides[2])), 11);
    p.constant(scale, 12);
    core.dispatchGroups(p, .{ .x = @intCast(k.shape[1]), .y = @intCast(q.shape[0]), .z = @intCast(blocks) }, .{ .x = 32, .y = gqa, .z = q_len });

    var final_buf: [64]u8 = undefined;
    const final = try stock.kernel(try stock.named(&final_buf, "sdpa_vector_2pass_2_{s}_{d}", .{ try stock.cType(q.dtype), v.shape[3] }));
    const f = try core.pass();
    f.pipeline(final);
    core.bind(f, partial, 0);
    core.bind(f, sums, 1);
    core.bind(f, maxs, 2);
    core.bindOut(f, out, 3);
    f.constant(blocks, 4);
    core.dispatchGroups(f, .{ .x = @as(usize, @intCast(q.shape[0])) * @as(usize, @intCast(q.shape[1])), .y = q_len, .z = 1 }, .{ .x = 1024, .y = 1, .z = 1 });
}
