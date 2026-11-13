//! MLX's fused scaled dot-product attention on the engine: `fast::scaled_dot_product_attention` (mlx/fast.cpp) and
//! `ScaledDotProductAttention::eval_gpu` with `sdpa_full_self_attention_nax`, `sdpa_vector` and `sdpa_vector_2pass`
//! (mlx/backend/metal/scaled_dot_product_attention.cpp) over the compiled kernels of `mlx.metallib`, so the output
//! has the bytes and the strides it has under MLX. Ported from MLX's Metal backend (MIT, see NOTICE). Ported: no mask
//! and no sinks, one dtype, head dimension 64 on the NAX tiles. Refused by name, each one because the tower never asks:
//! a mask or sinks (it passes none), another head dimension (the release's is 64, and `engineGap` checks it at the
//! load), mixed dtypes (q, k and v are all bf16), the SIMD tiles (DeepSeek serves on the engine only with matrix units).

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

pub fn attention(q_pre: *Tensor, k_pre: *Tensor, v_pre: *Tensor, scale: f32) Error!*Tensor {
    if (q_pre.ndim != 4 or k_pre.ndim != 4 or v_pre.ndim != 4) return error.TensorShape;
    if (q_pre.dtype != k_pre.dtype or q_pre.dtype != v_pre.dtype) return error.TensorDtype;
    switch (q_pre.dtype) {
        .float16, .bfloat16, .float32 => {},
        else => return error.TensorDtype,
    }
    const batch = q_pre.shape[0];
    const heads = q_pre.shape[1];
    const q_len: usize = @intCast(q_pre.shape[2]);
    const k_heads = k_pre.shape[1];
    if (k_pre.shape[0] != batch or v_pre.shape[0] != batch or q_pre.shape[3] != k_pre.shape[3] or k_pre.shape[1] != v_pre.shape[1] or
        k_pre.shape[2] != v_pre.shape[2] or k_heads == 0 or @rem(heads, k_heads) != 0) return error.TensorShape;
    if (q_pre.shape[3] != head_dim or v_pre.shape[3] != head_dim) return error.TensorUnsupported;
    if (q_len == 0 or k_pre.shape[2] == 0 or batch == 0 or heads == 0) return error.TensorUnsupported;
    const k_len: usize = @intCast(k_pre.shape[2]);
    const g = try matmul.gpu();
    const gqa: usize = @intCast(@divExact(heads, k_heads));

    if (q_len > vector_rows) {
        // `has_fused_kernel` and `use_fallback` take this shape to the full kernel; the SIMD tiles are not ported.
        if (!g.nax or (q_pre.dtype == .float32 and !g.tf32)) return error.TensorUnsupported;
        return full(q_pre, k_pre, v_pre, scale);
    }
    if (q_len > k_len or q_len * gqa > 32) return error.TensorUnsupported;
    return vector(g, q_pre, k_pre, v_pre, scale, gqa);
}

fn strideOf(t: *const Tensor, axis: usize) i64 {
    return @intCast(t.strides[axis]);
}

fn copyUnlessLastDense(t: *Tensor) Error!*Tensor {
    return if (t.strides[3] == 1) core.share(t) else ops.contiguousCopy(t);
}

/// `sdpa_full_self_attention_nax`: the output is laid out `[B, qL, H, D]` and read as `[B, H, qL, D]`.
fn full(q_pre: *Tensor, k_pre: *Tensor, v_pre: *Tensor, scale: f32) Error!*Tensor {
    const q = try copyUnlessLastDense(q_pre);
    defer core.release(q);
    const k = try copyUnlessLastDense(k_pre);
    defer core.release(k);
    const v = try copyUnlessLastDense(v_pre);
    defer core.release(v);

    const bq: usize = 64;
    const bk: usize = 32;
    const wm: usize = 4;
    const b: usize = @intCast(q.shape[0]);
    const h: usize = @intCast(q.shape[1]);
    const q_len: usize = @intCast(q.shape[2]);
    const k_len: usize = @intCast(k.shape[2]);
    const laid = try core.fresh(q.dtype, &.{ q.shape[0], q.shape[2], q.shape[1], q.shape[3] });
    defer core.release(laid);
    const out = try ops.transpose(laid, &.{ 0, 2, 1, 3 });
    errdefer core.release(out);

    const align_q = q_len % bq == 0;
    const align_k = k_len % bk == 0;
    const type_name = try stock.typeName(q.dtype);
    var name_buf: [160]u8 = undefined;
    const name = try stock.named(&name_buf, "steel_attention_{s}_bq{d}_bk{d}_bd{d}_bv{d}_wm{d}_wn1_mask{s}", .{ type_name, bq, bk, head_dim, head_dim, wm, type_name });
    var hash_buf: [256]u8 = undefined;
    const hash = try stock.named(&hash_buf, "{s}_align_Q_{c}_align_K_{c}_has_mask_n_do_causal_n_has_sinks_n", .{ name, tf(align_q), tf(align_k) });
    const kern = try stock.kernelFlags(name, hash, &.{
        .{ .index = 200, .value = align_q },
        .{ .index = 201, .value = align_k },
        .{ .index = 300, .value = false },
        .{ .index = 301, .value = false },
        .{ .index = 302, .value = false },
    });
    const nq = (q_len + bq - 1) / bq;
    const nk = (k_len + bk - 1) / bk;
    const params: AttnParams = .{
        .b = @intCast(b),
        .h = @intCast(h),
        .d = head_dim,
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
        .q_len_off = @intCast(k_len - q_len),
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
    core.dispatchGroups(p, .{ .x = nq, .y = h, .z = b }, .{ .x = 32, .y = wm, .z = 1 });
    return out;
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

fn vector(g: matmul.Gpu, q_pre: *Tensor, k_pre: *Tensor, v_pre: *Tensor, scale: f32, gqa: usize) Error!*Tensor {
    const q = if (queryKept(q_pre)) try core.share(q_pre) else try ops.contiguousCopy(q_pre);
    defer core.release(q);
    const k = if (keysKept(k_pre)) try core.share(k_pre) else try ops.contiguousCopy(k_pre);
    defer core.release(k);
    const v = if (keysKept(v_pre)) try core.share(v_pre) else try ops.contiguousCopy(v_pre);
    defer core.release(v);
    const out = try core.fresh(q.dtype, &.{ q.shape[0], q.shape[1], q.shape[2], v.shape[3] });
    errdefer core.release(out);
    const k_len: usize = @intCast(k.shape[2]);
    // Two passes for a long key axis on a Max or an Ultra, or for keys shared by many heads.
    if (((g.class == .d or g.class == .s) and k_len >= 1024) or (k.shape[1] < q.shape[1] and k_len >= 4096)) {
        try twoPass(g, q, k, v, out, scale, gqa);
    } else {
        try onePass(q, k, v, out, scale, gqa);
    }
    return out;
}

fn headStride(t: *const Tensor) u64 {
    return @intCast(if (t.shape[1] == 1) t.strides[0] else t.strides[1]);
}

/// The bool function constants 20 to 25 of the vector kernels with no mask and no sinks, and the name suffix MLX gives them.
fn vectorFlags(q: *const Tensor, flags: *[6]metal.Flag, hash: []u8, base: []const u8) Error![]const u8 {
    const transposed = !q.rowContiguous();
    flags.* = .{
        .{ .index = 20, .value = false },
        .{ .index = 21, .value = transposed },
        .{ .index = 22, .value = false },
        .{ .index = 23, .value = false },
        .{ .index = 24, .value = false },
        .{ .index = 25, .value = false },
    };
    return std.fmt.bufPrint(hash, "{s}_nomask{s}_nc", .{ base, if (transposed) "_qt" else "_qnt" }) catch error.TensorShape;
}

fn onePass(q: *Tensor, k: *Tensor, v: *Tensor, out: *Tensor, scale: f32, gqa: usize) Error!void {
    var name_buf: [96]u8 = undefined;
    const name = try stock.named(&name_buf, "sdpa_vector_{s}_{d}_{d}", .{ try stock.cType(q.dtype), head_dim, head_dim });
    var flags: [6]metal.Flag = undefined;
    var hash_buf: [160]u8 = undefined;
    const prefix = try vectorFlags(q, &flags, &hash_buf, name);
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
    if (std.c.getenv("MLX_SDPA_BLOCKS")) |text| {
        const forced = std.fmt.parseInt(c_int, std.mem.span(text), 10) catch 0;
        if (forced > 0) blocks = @divTrunc(forced + 31, 32) * 32;
    }
    return blocks;
}

fn twoPass(g: matmul.Gpu, q: *Tensor, k: *Tensor, v: *Tensor, out: *Tensor, scale: f32, gqa: usize) Error!void {
    const q_len: usize = @intCast(q.shape[2]);
    const n: usize = @intCast(k.shape[2]);
    const gqa_dims = (gqa == 8 and head_dim == 64) or (gqa == 8 and head_dim == 128) or ((gqa == 12 or gqa == 16) and head_dim == 128);
    const wide = q_len == 1 and q.shape[1] == @as(c_int, @intCast(gqa)) * k.shape[1] and gqa_dims and n >= 8192;
    var name_buf: [96]u8 = undefined;
    const name = if (wide)
        try stock.named(&name_buf, "sdpa_vector_2pass_1_gqa_{d}_{s}_{d}_{d}", .{ gqa, try stock.cType(q.dtype), head_dim, head_dim })
    else
        try stock.named(&name_buf, "sdpa_vector_2pass_1_{s}_{d}_{d}", .{ try stock.cType(q.dtype), head_dim, head_dim });
    const blocks = blocksOf(g, n, gqa * q_len);

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
    const prefix = try vectorFlags(q, &flags, &hash_buf, name);
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
    const final = try stock.kernel(try stock.named(&final_buf, "sdpa_vector_2pass_2_{s}_{d}", .{ try stock.cType(q.dtype), head_dim }));
    const f = try core.pass();
    f.pipeline(final);
    core.bind(f, partial, 0);
    core.bind(f, sums, 1);
    core.bind(f, maxs, 2);
    core.bindOut(f, out, 3);
    f.constant(blocks, 4);
    core.dispatchGroups(f, .{ .x = @as(usize, @intCast(q.shape[0])) * @as(usize, @intCast(q.shape[1])), .y = q_len, .z = 1 }, .{ .x = 1024, .y = 1, .z = 1 });
}
