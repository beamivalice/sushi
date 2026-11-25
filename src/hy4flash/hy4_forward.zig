//! Hy4-Flash's forward, composed from the engine's existing ops (`mlx_ops.Ops`): the same code runs on Sushi Cooler's
//! stream when served and on the CPU stream against the oracle (tests/dump_hy4_fixtures.py). Each piece takes
//! the release's own tensors by their checkpoint names (docs/hy4-flash-arch.md#the-forward-component-by-component).
const std = @import("std");
const mlx = @import("../mlx.zig");
const Ops = @import("../mlx_ops.zig").Ops;
const hck = @import("../hc_kernels.zig");
const sparse = @import("../sparse_select.zig");
const Arr = mlx.mlx_array;
const hy4_stq = @import("hy4_stq.zig");
const expert_mv = @import("hy4_expert_mv.zig");
const hy4_absorb = @import("hy4_absorb.zig");
const hy4_attn = @import("hy4_attn.zig");
const hy4_rope = @import("hy4_rope.zig");
const hy4_norm = @import("hy4_norm.zig");
const hy4_chain = @import("hy4_chain.zig");
const hy4_engram_gate = @import("hy4_engram_gate.zig");
const hy4_route = @import("hy4_route.zig");
const hy4_gemv = @import("hy4_gemv.zig");
const hy4_row = @import("hy4_row.zig");
const hy4_hc = @import("hy4_hc.zig");
const hy4_index = @import("hy4_index.zig");
const hy4_front = @import("hy4_front.zig");
/// A trunk linear: the stored weight, or an STQ pack's trellis bank (`hy4_stq`).
pub const Lin = hy4_stq.Lin;

/// One hyper-connection mix's tensors: `hc_fn` [2n, n·D], `hc_base` [2n], `hc_scale` [2] (f32 in the release).
pub const Mix = struct { fn_: Arr, base: Arr, scale: Arr };

/// What a mix gives: `post` [T, n], the weight each stream takes of the sublayer's output, and `x` [T, D], the
/// collapsed row the sublayer reads.
pub const Mixed = struct { post: Arr, x: Arr };

/// What a calibration tap reads, by the linears that read it.
pub const TapKind = enum {
    /// The input norm's output [T, D]: q_a, kv_a, the attention gate, the indexer's token_key and head_weight.
    attn_in,
    /// `q_resid` [T, q_lora]: q_b and the indexer's query.
    q_lora,
    /// The normed latent [T, latent]: kv_b's k and v halves (absorbed).
    kv_latent,
    /// The gated attention [T, heads·v], one call per attention span of rows: o.
    o_in,
    /// The FFN norm's output [T, D]: lat_down, the router, the shared expert's gate and up; layer 0's dense gate and up.
    ffn_in,
    /// The routed experts' input `lat` [T, latent] in f32 (BF16 values at `products` .bf16).
    expert_in,
    /// The routed sum in the latent [T, latent]: lat_up.
    lat_up_in,
    /// The shared expert's SwiGLU [T, I] (layer 0's dense one [T, dense I]): its down.
    shared_act,
    /// The concatenated normed head rows `u` [T, heads·hd]: Engram's value_w and key_w.
    engram_in,
    /// The final norm's output on every row [T, D], whatever is read out: lm_head.
    head_in,
};

/// Reads a forward without changing it. Arrays are valid for the call only; a sink that keeps one copies it.
pub const Tap = struct {
    ctx: *anyopaque,
    input: *const fn (ctx: *anyopaque, layer: usize, kind: TapKind, x: Arr) anyerror!void,
    /// The router's picks (host, [T·k] expert ids, row-major) and the weights [T, k] f32 the layer applies.
    routes: *const fn (ctx: *anyopaque, layer: usize, picks: []const u32, weights: Arr) anyerror!void,
    /// The routed experts' down input `act` [P, I] as down reads it (BF16 at `products` .bf16, else f32; unweighted: the
    /// weight applies after down), and each row's expert id (host, [P]). A layer may call it once per group or block.
    expert_act: *const fn (ctx: *anyopaque, layer: usize, act: Arr, experts: []const u32) anyerror!void,
};

/// The tap and the layer a piece of the forward runs for; null on the default path, which then makes no call of the
/// tap's.
pub const At = struct {
    tap: *const Tap,
    layer: usize,

    pub fn input(at: ?At, kind: TapKind, x: Arr) !void {
        if (at) |a| try a.tap.input(a.tap.ctx, a.layer, kind, x);
    }
};

/// The iHC pre/post mix over the `n` streams `h` [T, n, D] (transformers `HYV4HyperConnection`, llama.cpp `hc_mixes`):
/// `m = (flat · fnᵀ) · rsqrt(mean(flat²) + ε_norm)` over `flat` = the streams side by side in f32;
/// `pre = σ(m[:n]·s₀ + b[:n]) + ε`, `post = magnitude·σ(m[n:]·s₁ + b[n:]) + ε`, `x = Σ_c pre_c · h_c`.
pub fn mix(o: *Ops, h: Arr, w: Mix, n: c_int, magnitude: f32, eps: f32, norm_eps: f32) !Mixed {
    const shape = mlx.getShape(h);
    const t = shape[0];
    const d = shape[2];
    const flat = try o.cast(try o.reshape(h, &.{ t, n * d }), .float32);
    const mixes = try o.binary(.mul, try o.binary(.mm, flat, try o.transpose(try o.cast(w.fn_, .float32), &.{ 1, 0 })), try rowRsqrt(o, flat, norm_eps));
    const scale = try o.cast(w.scale, .float32);
    const base = try o.cast(w.base, .float32);
    const pre_logit = try o.binary(.add, try o.binary(.mul, try o.slice(mixes, 1, 0, n), try o.slice(scale, 0, 0, 1)), try o.slice(base, 0, 0, n));
    const post_logit = try o.binary(.add, try o.binary(.mul, try o.slice(mixes, 1, n, 2 * n), try o.slice(scale, 0, 1, 2)), try o.slice(base, 0, n, 2 * n));
    const epsilon = try o.scalar(eps, .float32);
    const pre = try o.binary(.add, try o.unary(.sigmoid, pre_logit), epsilon);
    const post = try o.binary(.add, try o.binary(.mul, try o.unary(.sigmoid, post_logit), try o.scalar(magnitude, .float32)), epsilon);
    return .{ .post = post, .x = try collapse(o, h, pre) };
}

/// Whether a GPU or Sushi Cooler stream runs the iHC through the shared identity-comb kernels (`hc_kernels.zig`);
/// the CPU stream, the oracle tests' exact reference, always takes the op chain. A test seam.
pub var hc_kernels_on = true;

/// Whether a GPU or Sushi Cooler stream scores and selects the indexer's groups through the shared kernels
/// (`sparse_select.zig`); the CPU stream takes the op chain. A test seam.
pub var select_kernels_on = true;

/// Whether a one-row forward's absorbed products and f32 linears read their BF16 weights as stored (`hy4_absorb`)
/// rather than widened. A test seam.
pub var absorb_kernels_on = true;

fn hcKernels(o: *Ops) bool {
    return hc_kernels_on and mlx.streamIsGpu(o.s);
}

/// `mix`, then the collapsed row rms-normed by `norm_w`: on a GPU stream one mix kernel and one split, collapse and
/// norm kernel, else the op chain.
pub fn mixNorm(o: *Ops, h: Arr, w: Mix, n: c_int, magnitude: f32, eps: f32, norm_eps: f32, norm_w: Arr) !Mixed {
    if (!hcKernels(o)) {
        const m = try mix(o, h, w, n, magnitude, eps, norm_eps);
        return .{ .post = m.post, .x = try norm(o, m.x, norm_w, norm_eps) };
    }
    const k = try o.own(try hck.identityEps(norm_eps, eps, magnitude));
    const hf = try o.cast(h, .float32);
    const hw: hck.HcWeights = .{ .fn_w = w.fn_, .scale = w.scale, .base = w.base };
    if (mix_norm_on and hy4_hc.applies(o.s, hf, w.fn_, w.scale)) {
        const r = try hy4_hc.mixNorm(o.s, try o.contiguous(hf), hw, k, norm_w);
        return .{ .post = try o.own(r.post), .x = try o.own(r.h) };
    }
    const z = try o.own(try hck.mixTiled(o.s, hf, hw, k, hck.hc_mix_tile));
    const r = try hck.normIdentity(o.s, z, hf, norm_w, k);
    _ = try o.own(r.pre);
    return .{ .post = try o.own(r.post), .x = try o.own(r.h) };
}

/// Whether a decode block's mix and its collapse and norm run as one launch (`hy4_hc`) rather than the tiled mix and
/// the identity norm. A test seam: both give the same bytes.
pub var mix_norm_on = true;

/// Whether a GPU stream expands a sublayer's output inside the next mix's launch (`hck.mixExpandIdentity`) rather than
/// in a launch of its own. A test seam: both give the same bytes.
pub var mix_expand_on = true;

/// Whether a decode block's W_o leaves its partials for the iHC mix's launch to finish and norm (`expandMixNormParts`)
/// rather than its own finish_rms launch. A test seam: both give the same bytes.
pub var o_parts_on = true;

/// Whether W_o's launch finishes its own partials on a stream with a counter page (`hy4_stq.applyFinished`) and the
/// mix norms its rows (`expandMixNormRows`), rather than the mix's every threadgroup finishing them. A test seam: both
/// give the same bytes.
pub var o_rows_on = true;

/// The most rows W_o finishes its own partials for: a block's last threadgroup to count in finishes every row alone, a
/// tail that past three rows outweighs the partials the mix's threadgroups no longer re-read (hy4-flash-arch.md).
pub const o_rows_max = @import("sushi_exl3").project_rows.COUNTED_ROWS_MAX;

/// Whether `expandMixNormParts` (or `expandMixNormRows`) will take W_o's partials (or rows) for the streams `h`
/// [rows, n, D] and the mix `w`, with W_o's `attn.o` and its norm: a caller asks before it makes them.
pub fn oPartsReady(o: *Ops, h: Arr, w: Mix, attn: Attn) bool {
    if (!o_parts_on or !chain_kernels_on or !mix_expand_on or !mix_norm_on or !hcKernels(o)) return false;
    const b = attn.o.stq orelse return false;
    const d = mlx.getShape(h)[mlx.getShape(h).len - 1];
    const nd = mlx.mlx_array_dtype(attn.post_norm);
    return hy4_hc.applies(o.s, h, w.fn_, w.scale) and @rem(d, 128) == 0 and d <= 4096 and b.outDim() == d and
        mlx.mlx_array_size(attn.post_norm) == @as(usize, @intCast(d)) and (nd == .float32 or nd == .bfloat16);
}

/// `expandMixNorm(h, post, norm(W_o's rows))` from W_o's k-split partials `op` (`osplit` of them a row, `oPartsReady`),
/// the finish and `attn.post_norm` in the mix's launch (`hy4_hc.mixExpandNormParts`): the same bytes, one launch fewer.
pub fn expandMixNormParts(o: *Ops, h: Arr, post: Arr, op: Arr, osplit: c_int, attn: Attn, o_eps: f32, w: Mix, magnitude: f32, eps: f32, norm_eps: f32, norm_w: Arr) !Expanded {
    const k = try o.own(try hck.identityEps(norm_eps, eps, magnitude));
    const hw: hck.HcWeights = .{ .fn_w = w.fn_, .scale = w.scale, .base = w.base };
    const r = try hy4_hc.mixExpandNormParts(o.s, try o.contiguous(h), try o.contiguous(try o.cast(post, .float32)), op, attn.o.stq.?.svh, osplit, attn.post_norm, o_eps, hw, k, norm_w);
    return .{ .m = try o.own(r.m.?), .mixed = .{ .post = try o.own(r.post), .x = try o.own(r.h) } };
}

/// `expandMixNormParts` from W_o's finished rows `y` [T, D] f32 (`OLeft.rows`), their norm in the mix's launch
/// (`hy4_hc.mixExpandNormRows`): the same bytes.
pub fn expandMixNormRows(o: *Ops, h: Arr, post: Arr, y: Arr, attn: Attn, o_eps: f32, w: Mix, magnitude: f32, eps: f32, norm_eps: f32, norm_w: Arr) !Expanded {
    const k = try o.own(try hck.identityEps(norm_eps, eps, magnitude));
    const hw: hck.HcWeights = .{ .fn_w = w.fn_, .scale = w.scale, .base = w.base };
    const r = try hy4_hc.mixExpandNormRows(o.s, try o.contiguous(h), try o.contiguous(try o.cast(post, .float32)), y, attn.post_norm, o_eps, hw, k, norm_w);
    return .{ .m = try o.own(r.m.?), .mixed = .{ .post = try o.own(r.post), .x = try o.own(r.h) } };
}

/// `expandMixNorm` with the attention's `post_proj_norm` of `y` in the same launch on a GPU stream (`hy4_hc.expandNormMixNorm`).
pub fn expandNormMixNorm(o: *Ops, h: Arr, post: Arr, y: Arr, y_norm_w: Arr, y_eps: f32, w: Mix, n: c_int, magnitude: f32, eps: f32, norm_eps: f32, norm_w: Arr) !Expanded {
    if (norm_kernel_on and mix_expand_on and mix_norm_on and hcKernels(o)) {
        const hf = try o.contiguous(try o.cast(h, .float32));
        const pf = try o.contiguous(try o.cast(post, .float32));
        if (hy4_hc.expandNormApplies(o.s, hf, pf, y, y_norm_w, w.fn_, w.scale)) {
            const k = try o.own(try hck.identityEps(norm_eps, eps, magnitude));
            const hw: hck.HcWeights = .{ .fn_w = w.fn_, .scale = w.scale, .base = w.base };
            const r = try hy4_hc.expandNormMixNorm(o.s, hf, pf, try o.contiguous(y), y_norm_w, y_eps, hw, k, norm_w);
            return .{ .m = try o.own(r.m.?), .mixed = .{ .post = try o.own(r.post), .x = try o.own(r.h) } };
        }
    }
    return expandMixNorm(o, h, post, try norm(o, y, y_norm_w, y_eps), w, n, magnitude, eps, norm_eps, norm_w);
}

/// The streams after a sublayer's write-back `m` [T, n, D] and the next mix of them.
pub const Expanded = struct { m: Arr, mixed: Mixed };

/// `expand(h, post, y)`, then `mixNorm` of it: on a GPU stream the expand rides the mix's launch.
pub fn expandMixNorm(o: *Ops, h: Arr, post: Arr, y: Arr, w: Mix, n: c_int, magnitude: f32, eps: f32, norm_eps: f32, norm_w: Arr) !Expanded {
    if (!hcKernels(o) or !mix_expand_on) {
        const m = try expand(o, h, post, y);
        return .{ .m = m, .mixed = try mixNorm(o, m, w, n, magnitude, eps, norm_eps, norm_w) };
    }
    const k = try o.own(try hck.identityEps(norm_eps, eps, magnitude));
    const hf = try o.cast(h, .float32);
    const hw: hck.HcWeights = .{ .fn_w = w.fn_, .scale = w.scale, .base = w.base };
    if (mix_norm_on and hy4_hc.applies(o.s, hf, w.fn_, w.scale) and mlx.mlx_array_dtype(y) == .float32) {
        const r = try hy4_hc.mixExpandNorm(o.s, try o.contiguous(hf), try o.contiguous(y), try o.contiguous(try o.cast(post, .float32)), hw, k, norm_w);
        return .{ .m = try o.own(r.m.?), .mixed = .{ .post = try o.own(r.post), .x = try o.own(r.h) } };
    }
    const r = try hck.mixExpandIdentity(o.s, hf, y, try o.cast(post, .float32), hw, k, hck.hc_mix_tile);
    const m = try o.own(r.m);
    const z = try o.own(r.z);
    const nr = try hck.normIdentity(o.s, z, m, norm_w, k);
    _ = try o.own(nr.pre);
    return .{ .m = m, .mixed = .{ .post = try o.own(nr.post), .x = try o.own(nr.h) } };
}

/// `head`, then the collapsed row rms-normed by `norm_w` (the final norm), as `mixNorm` runs it.
pub fn headNorm(o: *Ops, h: Arr, w: Mix, n: c_int, eps: f32, norm_eps: f32, norm_w: Arr) !Arr {
    if (!hcKernels(o)) return norm(o, try head(o, h, w, n, eps, norm_eps), norm_w, norm_eps);
    const k = try o.own(try hck.identityEps(norm_eps, eps, 1.0));
    const hf = try o.cast(h, .float32);
    const z = try o.own(try hck.mixTiled(o.s, hf, .{ .fn_w = w.fn_, .scale = w.scale, .base = w.base }, k, hck.hc_mix_tile));
    const r = try hck.headIdentity(o.s, z, hf, norm_w, k);
    _ = try o.own(r.pre);
    return o.own(r.h);
}

/// The head's collapse (`HYV4HyperHead`): one weight a stream, `pre = σ(m·s + b) + ε`, `x = Σ_c pre_c · h_c`; `hc_fn`
/// [n, n·D], `hc_base` [n], `hc_scale` [1].
pub fn head(o: *Ops, h: Arr, w: Mix, n: c_int, eps: f32, norm_eps: f32) !Arr {
    const shape = mlx.getShape(h);
    const flat = try o.cast(try o.reshape(h, &.{ shape[0], n * shape[2] }), .float32);
    const mixes = try o.binary(.mul, try o.binary(.mm, flat, try o.transpose(try o.cast(w.fn_, .float32), &.{ 1, 0 })), try rowRsqrt(o, flat, norm_eps));
    const logit = try o.binary(.add, try o.binary(.mul, mixes, try o.cast(w.scale, .float32)), try o.cast(w.base, .float32));
    const pre = try o.binary(.add, try o.unary(.sigmoid, logit), try o.scalar(eps, .float32));
    return collapse(o, h, pre);
}

/// A sublayer's output `y` [T, D] written back into the streams: `h'_c = post_c · y + h_c`, in f32 (llama.cpp keeps
/// the streams f32; transformers computes this sum in f32 too).
pub fn expand(o: *Ops, h: Arr, post: Arr, y: Arr) !Arr {
    if (hcKernels(o)) return o.own(try hck.expandIdentity(o.s, try o.cast(h, .float32), y, try o.cast(post, .float32)));
    const shape = mlx.getShape(post);
    const yd = mlx.getShape(y);
    const weighted = try o.binary(.mul, try o.reshape(post, &.{ shape[0], shape[1], 1 }), try o.reshape(try o.cast(y, .float32), &.{ yd[0], 1, yd[1] }));
    return o.binary(.add, weighted, try o.cast(h, .float32));
}

/// `Σ_c w_c · h_c` over the stream axis, in f32: `w` [T, n], `h` [T, n, D].
fn collapse(o: *Ops, h: Arr, w: Arr) !Arr {
    const shape = mlx.getShape(w);
    const weighted = try o.binary(.mul, try o.reshape(w, &.{ shape[0], shape[1], 1 }), try o.cast(h, .float32));
    return o.reduce(weighted, 1, false, false);
}

/// `rsqrt(mean(x²) + ε)` a row, [T, 1]: the unweighted RMS norm's factor (`HYV4UnweightedRMSNorm`).
fn rowRsqrt(o: *Ops, x: Arr, eps: f32) !Arr {
    const mean = try o.reduce(try o.binary(.mul, x, x), -1, true, true);
    return o.unary(.rsqrt, try o.binary(.add, mean, try o.scalar(eps, .float32)));
}

// ── norms, linears and the feed-forward ─────────────────────────────────────────────────────────────────────────────

/// `RMSNorm_w(x)` (`HYV4RMSNorm`): `w ⊙ x · rsqrt(mean(x²) + ε)`, in f32.
pub fn norm(o: *Ops, x: Arr, w: Arr, eps: f32) !Arr {
    return normAs(o, x, null, w, eps, .float32);
}

/// Whether a GPU stream norms through `hy4_norm.rows`' one launch (the weight read as stored, an add and the cast after
/// folded in) rather than the casts, the add and MLX's norm. A test seam: both give the same bytes.
pub var norm_kernel_on = true;

/// `norm(x + y, w, eps)` (`y` absent: of `x`; the add in f32) cast to `out_dt`.
pub fn normAs(o: *Ops, x: Arr, y: ?Arr, w: Arr, eps: f32, out_dt: mlx.mlx_dtype) !Arr {
    if (norm_kernel_on and hy4_norm.applies(o.s, x, y, w)) return o.own(try hy4_norm.rows(o.s, x, y, w, eps, out_dt));
    const v = if (y) |b| try o.binary(.add, try o.cast(x, .float32), try o.cast(b, .float32)) else x;
    return o.cast(try o.rms(try o.cast(v, .float32), try o.cast(w, .float32), eps), out_dt);
}

/// The most of a weight a linear widens to f32 at once.
pub var widen_block_bytes: u64 = default_widen_block_bytes;
pub const default_widen_block_bytes: u64 = 64 << 20;

/// The precision the trunk's and the experts' products take: BF16 as stored (served), or f32 throughout, which the
/// transformers oracle's f32 forward is held to. A rounding moves the tiny's random router's picks, so its oracle tests
/// run at f32; the BF16 forward is judged on the release (greedy text and the KLD against f32).
pub const Products = enum { bf16, f32 };
pub var products: Products = .bf16;

/// `x · Wᵀ` for a linear's weight `W` [out, in], as f32. At `products` .bf16 a BF16 weight multiplies as stored: the rows rounded to BF16,
/// the sums in f32 (MLX's BF16 product, as transformers' BF16 forward and llama.cpp's BF16 kernels take it), the result
/// widened. Any other weight takes `linearF32`.
pub fn linear(o: *Ops, x: Arr, lin: Lin) !Arr {
    if (lin.stq) |b| return hy4_stq.apply(o, x, b);
    // One row through a BF16 weight with the casts in the launch, the product's bytes.
    const rows = mlx.getShape(x)[0];
    if (chain_kernels_on and products == .bf16 and (rows == 1 or isVerifyBlock(rows)) and hy4_gemv.applies(o.s, x, lin.w)) return o.own(try hy4_gemv.rowsThru(o.s, x, lin.w));
    return o.cast(try linearStored(o, x, lin), .float32);
}

/// `linear` before its widening, in the dtype its product rounds to: BF16 through a trellis bank or a BF16 weight at
/// `.bf16`, else f32. Widening it is exact, so a caller may hold it narrow.
pub fn linearStored(o: *Ops, x: Arr, lin: Lin) !Arr {
    return linearStoredAs(o, x, lin, null);
}

/// `linearStored` with its product's kernel chosen as for `rows_as` rows on the engine (`matmulRowsAs`). A trellis
/// bank's GEMM gives a row its bytes at any count past its rows arm (`trunk.matmul`), so it takes none.
fn linearStoredAs(o: *Ops, x: Arr, lin: Lin, rows_as: ?c_int) !Arr {
    if (lin.stq) |b| return hy4_stq.applyStored(o, x, b);
    const w = lin.w;
    if (products == .f32 or mlx.mlx_array_dtype(w) != .bfloat16) return linearF32As(o, x, w, rows_as);
    return mmByRow(o, try o.cast(x, .bfloat16), try o.transpose(w, &.{ 1, 0 }), rows_as);
}

/// `x` cast to the dtype `linear` of `lin` reads it in, so a caller may hold it narrow.
pub fn linearInput(o: *Ops, x: Arr, lin: Lin) !Arr {
    return o.cast(x, linearInputDtype(lin));
}

/// The dtype `linearInput` gives `lin`'s rows.
pub fn linearInputDtype(lin: Lin) mlx.mlx_dtype {
    const narrow = lin.stq != null or (products == .bf16 and mlx.mlx_array_dtype(lin.w) == .bfloat16);
    return if (narrow) .bfloat16 else .float32;
}

/// `a · b` of matrices, on the engine with its kernel, tile and k split chosen as for `rows_as` rows of `a`.
fn mmAs(o: *Ops, a: Arr, b: Arr, rows_as: ?c_int) !Arr {
    const r = rows_as orelse return o.binary(.mm, a, b);
    if (r < 2 or mlx.getShape(a)[0] <= r or !mlx.engine.isStream(o.s)) return o.binary(.mm, a, b);
    const out = try o.slot();
    try mlx.check(mlx.engine.matmulRowsAs(out, a, b, @intCast(r)));
    return out.*;
}

/// The widest block whose rows take their one-row step's shapes (a speculative verify).
pub const verify_rows_max: c_int = @min(hy4_row.max_rows, hy4_absorb.max_rows);

/// Whether a block (a verify block, or one span of a chunk) of `rows` rows takes the one-row shapes row by row, so
/// each row's bytes are its decode step's. Every 2..8-row site asks this, by the span's own width.
pub fn isVerifyBlock(rows: c_int) bool {
    return rows >= 2 and rows <= verify_rows_max;
}

/// `a · b` with each of `a`'s 2..8 rows its own one-row product, so a verify row has the bytes its decode step has
/// (MLX's kernel and tile change with the row count). The head reads its widened block once a row.
fn mmByRow(o: *Ops, a: Arr, b: Arr, rows_as: ?c_int) !Arr {
    const t = mlx.getShape(a)[0];
    if (rows_as != null or !isVerifyBlock(t)) return mmAs(o, a, b, rows_as);
    const tu: usize = @intCast(t);
    var rows: [verify_rows_max]Arr = undefined;
    for (0..tu) |r| {
        const ri: c_int = @intCast(r);
        rows[r] = try o.binary(.mm, try o.slice(a, 0, ri, ri + 1), b);
    }
    return o.concat(rows[0..tu], 0);
}

/// `a · b` per head with each of `a`'s 2..8 rows its own one-row product: the absorb and value-out fallbacks when
/// `hy4_absorb` does not serve.
fn mmHeadsByRow(o: *Ops, a: Arr, b: Arr) !Arr {
    const t = mlx.getShape(a)[1];
    if (!isVerifyBlock(t)) return o.binary(.mm, a, b);
    const tu: usize = @intCast(t);
    var rows: [verify_rows_max]Arr = undefined;
    for (0..tu) |r| {
        const ri: c_int = @intCast(r);
        rows[r] = try o.binary(.mm, try o.slice(a, 1, ri, ri + 1), b);
    }
    return o.concat(rows[0..tu], 1);
}

/// `mmHeadsByRow` with its rows computed as `rows_as`-row launches would compute them (`matmulRowsAs`), so a row's
/// bytes do not depend on the span's width.
fn mmHeadsAs(o: *Ops, a: Arr, b: Arr, rows_as: ?c_int) !Arr {
    const r = rows_as orelse return mmHeadsByRow(o, a, b);
    const t = mlx.getShape(a)[1];
    if (r < 2 or t <= r or !mlx.engine.isStream(o.s)) return mmHeadsByRow(o, a, b);
    const full = t - @rem(t, r);
    if (full == t) {
        const out = try o.slot();
        try mlx.check(mlx.engine.matmulRowsAs(out, a, b, @intCast(r)));
        return out.*;
    }
    const whole = if (full > r) blk: {
        const out = try o.slot();
        try mlx.check(mlx.engine.matmulRowsAs(out, try o.slice(a, 1, 0, full), b, @intCast(r)));
        break :blk out.*;
    } else try mmHeadsByRow(o, try o.slice(a, 1, 0, full), b);
    return o.concat(&.{ whole, try mmHeadsByRow(o, try o.slice(a, 1, full, t), b) }, 1);
}

/// Whether a chunk's attention spans share one launch of the gate and of `W_o` (`linearSpans`) and one widened `W_kvb`,
/// or each span makes its own (the per-span order, a test seam).
pub var spans_share = true;

/// `linearStored` of `x`'s rows as attention spans of `span` rows give each row its own span's launch: the whole spans
/// in one launch, its product chosen as for one span, and a short last span in a launch of its own (a launch of a few
/// rows takes the vector kernels, other sums).
pub fn linearSpans(o: *Ops, x: Arr, lin: Lin, span: c_int) !Arr {
    const t = mlx.getShape(x)[0];
    if (!spans_share and t > span) {
        var parts: [512]Arr = undefined;
        var count: usize = 0;
        var r0: c_int = 0;
        while (r0 < t) : (r0 += span) {
            if (count == parts.len) return error.Hy4ChunkTooWide;
            parts[count] = try linearStored(o, try o.slice(x, 0, r0, @min(t, r0 + span)), lin);
            count += 1;
        }
        return o.concat(parts[0..count], 0);
    }
    if (t <= span) return linearStored(o, x, lin);
    const full = t - @rem(t, span);
    if (full == t) return linearStoredAs(o, x, lin, span);
    const whole = if (full > span) try linearStoredAs(o, try o.slice(x, 0, 0, full), lin, span) else try linearStored(o, try o.slice(x, 0, 0, full), lin);
    return o.concat(&.{ whole, try linearStored(o, try o.slice(x, 0, full, t), lin) }, 0);
}

/// `norm(linear(x, lin), w, eps)`, a trellis bank's finish and the norm in one dispatch where its launch serves it.
pub fn linearNormed(o: *Ops, x: Arr, lin: Lin, w: Arr, eps: f32) !Arr {
    return linearNormedAs(o, x, lin, w, eps, .float32);
}

/// Whether a linear may take `x` as stored: a BF16 or plain weight, a BF16 input, or rows past a verify block's width;
/// a decode or verify row through an STQ bank keeps its BF16 product in the launch.
fn storedCheap(x: Arr, lin: Lin) bool {
    return lin.stq == null or mlx.mlx_array_dtype(x) == .bfloat16 or mlx.getShape(x)[0] > 16;
}

fn linearNormedAs(o: *Ops, x: Arr, lin: Lin, w: Arr, eps: f32, out_dt: mlx.mlx_dtype) !Arr {
    if (out_dt == .float32) if (lin.stq) |b| if (try hy4_stq.applyNormed(o, x, b, w, eps)) |y| return y;
    if (norm_kernel_on and mlx.streamIsGpu(o.s) and storedCheap(x, lin) and (lin.stq != null or mlx.getShape(x)[0] > 1))
        return normAs(o, try linearStored(o, x, lin), null, w, eps, out_dt);
    return normAs(o, try linear(o, x, lin), null, w, eps, out_dt);
}

/// `x · Wᵀ` in f32 throughout (f32 rows, the stored weight's values, f32 sums): the head (llama.cpp's f32 head), the
/// router and the indexer, whose picks a rounding could move, and Engram. `W` is widened a block of output rows at a
/// time, each in a scope of its own: a forward holds at most one block's f32 copy (`widen_block_bytes`).
pub fn linearF32(o: *Ops, x: Arr, w: Arr) !Arr {
    return linearF32As(o, x, w, null);
}

/// `linearF32` over `x`'s rows in launches of `span` rows each (`linearF32As`), the tail its own: a row's bytes do not
/// depend on how many rows run beside it.
pub fn linearF32Spans(o: *Ops, x: Arr, w: Arr, span: c_int) !Arr {
    const t = mlx.getShape(x)[0];
    if (t <= span) return linearF32(o, x, w);
    const full = t - @rem(t, span);
    if (full == t) return linearF32As(o, x, w, span);
    const whole = if (full > span) try linearF32As(o, try o.slice(x, 0, 0, full), w, span) else try linearF32(o, try o.slice(x, 0, 0, full), w);
    return o.concat(&.{ whole, try linearF32(o, try o.slice(x, 0, full, t), w) }, 0);
}

fn linearF32As(o: *Ops, x: Arr, w: Arr, rows_as: ?c_int) !Arr {
    const shape = mlx.getShape(w);
    const n: usize = @intCast(shape[0]);
    const k: usize = @intCast(shape[1]);
    const x32 = try o.cast(x, .float32);
    // One row reads a BF16 weight as stored, with the widened product's bytes.
    if (absorb_kernels_on and (rows_as == null or mlx.getShape(x32)[0] == 1) and hy4_row.applies(o.s, x32, w)) return o.own(try hy4_row.rowF32(o.s, try o.contiguous(x32), w));
    var per: usize = @max(1, widen_block_bytes / (k * 4));
    if (per >= n) {
        var ob: Ops = .{ .s = o.s };
        defer ob.deinit();
        const wf = try ob.cast(w, .float32);
        return o.own(try ob.result(try mmByRow(&ob, x32, try ob.transpose(wf, &.{ 1, 0 }), rows_as)));
    }
    const max_blocks = 128;
    if ((n + per - 1) / per > max_blocks) per = (n + max_blocks - 1) / max_blocks;
    var parts: [max_blocks]Arr = undefined;
    var count: usize = 0;
    var r0: usize = 0;
    while (r0 < n) : (r0 += per) {
        const r1 = @min(n, r0 + per);
        var ob: Ops = .{ .s = o.s };
        defer ob.deinit();
        const wb = try ob.cast(try ob.slice(w, 0, @intCast(r0), @intCast(r1)), .float32);
        parts[count] = try o.own(try ob.result(try mmByRow(&ob, x32, try ob.transpose(wb, &.{ 1, 0 }), rows_as)));
        count += 1;
    }
    return o.concat(parts[0..count], -1);
}

/// Whether a GPU stream runs Hy4's small f32 chains as `hy4_chain`'s one launch each rather than the ops. A test seam:
/// both give the same bytes.
pub var chain_kernels_on = true;

/// A SwiGLU block's three linears.
pub const Swiglu = struct { gate: Lin, up: Lin, down: Lin };

/// `down(silu(gate·x) ⊙ up·x)`: the dense layer's FFN and the shared expert (`HYV4FlashMLP`, before its norm).
pub fn swiglu(o: *Ops, x: Arr, w: Swiglu, at: ?At) !Arr {
    return linear(o, try swigluActAs(o, x, w, if (at == null) linearInputDtype(w.down) else .float32, at), w.down);
}

fn swigluStored(o: *Ops, x: Arr, w: Swiglu, at: ?At) !Arr {
    return linearStored(o, try swigluActAs(o, x, w, if (at == null) linearInputDtype(w.down) else .float32, at), w.down);
}

/// `norm(swiglu(x, w), norm_w, eps)`: the dense layer's FFN and its `post_fc2_norm` (`linearNormed`).
pub fn swigluNormed(o: *Ops, x: Arr, w: Swiglu, norm_w: Arr, eps: f32, at: ?At) !Arr {
    return linearNormed(o, try swigluActAs(o, x, w, if (at == null) linearInputDtype(w.down) else .float32, at), w.down, norm_w, eps);
}

/// `silu(gate·x) ⊙ up·x` in f32, the rows `down` reads: through trellis banks one launch where it serves
/// (`hy4_stq.applySwiglu`), else the two linears and the SwiGLU's launch.
pub fn swigluAct(o: *Ops, x: Arr, w: Swiglu, at: ?At) !Arr {
    return swigluActAs(o, x, w, .float32, at);
}

fn swigluActAs(o: *Ops, x: Arr, w: Swiglu, out_dt: mlx.mlx_dtype, at: ?At) !Arr {
    if (chain_kernels_on) if (w.gate.stq) |gb| if (w.up.stq) |ub| if (try hy4_stq.applySwiglu(o, x, gb, ub)) |act| {
        try At.input(at, .shared_act, act);
        return act;
    };
    const narrow = at == null and chain_kernels_on and mlx.streamIsGpu(o.s) and mlx.getShape(x)[0] > 1;
    const stored = narrow and storedCheap(x, w.gate) and storedCheap(x, w.up);
    const g = if (stored) try linearStored(o, x, w.gate) else try linear(o, x, w.gate);
    const u = if (stored) try linearStored(o, x, w.up) else try linear(o, x, w.up);
    const act = if (chain_kernels_on and hy4_chain.swigluApplies(o.s, g, u))
        try o.own(try hy4_chain.swigluAs(o.s, g, u, if (narrow) out_dt else .float32))
    else
        try o.binary(.mul, try o.silu(try o.cast(g, .float32)), try o.cast(u, .float32));
    try At.input(at, .shared_act, act);
    return act;
}

/// The router's tensors: `gate.weight` [E, D] and the f32 `e_score_correction_bias` [E].
pub const Router = struct { weight: Arr, bias: Arr };

/// The experts a row routes to and the weight each takes: `picks` [T, k] int32, `weights` [T, k] f32.
pub const Routed = struct { picks: Arr, weights: Arr };

/// `HYV4TopkRouter` (one group): `s = σ(x·Wᵀ)` in f32, the top `k` of `s + bias` picked, their unbiased `s`
/// normalized to sum 1 and scaled by `scale` (3.998 in the release).
pub fn route(o: *Ops, x: Arr, w: Router, k: c_int, scale: f32) !Routed {
    return (try routeMapped(o, x, w, k, scale, null)).routed;
}

/// `route`, and with `map` [E] int32 (a resident stream's slots) each pick's slot [T, k] uint32: on a GPU stream a
/// decode row's picks, weights and slots are one launch past the logits (`hy4_route.pick`).
pub fn routeMapped(o: *Ops, x: Arr, w: Router, k: c_int, scale: f32, map: ?Arr) !struct { routed: Routed, slots: ?Arr } {
    const logits = try linearF32(o, x, w.weight);
    if (chain_kernels_on and hy4_route.applies(o.s, logits, w.bias, k)) {
        const p = try hy4_route.pick(o.s, logits, w.bias, k, scale, map);
        return .{ .routed = .{ .picks = try o.own(p.picks), .weights = try o.own(p.weights) }, .slots = if (p.slots) |sl| try o.own(sl) else null };
    }
    const r = try routeFrom(o, logits, w, k, scale);
    return .{ .routed = r, .slots = if (map) |m| try o.cast(try o.take(m, try o.cast(r.picks, .int32), 0), .uint32) else null };
}

/// `w` with the cache-favoring pick's penalty `pen` (`--expert-cache-bias`, f32 [E]: 0 where the expert cache holds an
/// expert, -n elsewhere) added to its bias, so a held expert within n of a missed one wins; the weights stay the
/// unbiased scores. Null `pen` is `w` as it is.
pub fn cacheBiased(o: *Ops, w: Router, pen: ?Arr) !Router {
    const p = pen orelse return w;
    return .{ .weight = w.weight, .bias = try o.binary(.add, try o.cast(w.bias, .float32), p) };
}

fn routeFrom(o: *Ops, logits: Arr, w: Router, k: c_int, scale: f32) !Routed {
    const scores = try o.unary(.sigmoid, logits);
    const choice = try o.binary(.add, scores, try o.cast(w.bias, .float32));
    const order = try o.slot();
    try mlx.check(mlx.mlx_argpartition_axis(order, try o.unary(.negative, choice), k - 1, -1, o.s));
    const picks = try o.slice(order.*, 1, 0, k);
    const picked = try o.slot();
    try mlx.check(mlx.mlx_take_along_axis(picked, scores, picks, -1, o.s));
    const total = try o.binary(.add, try o.reduce(picked.*, -1, false, true), try o.scalar(1e-20, .float32));
    const weights = try o.binary(.mul, try o.binary(.div, picked.*, total), try o.scalar(scale, .float32));
    return .{ .picks = picks, .weights = weights };
}

/// A latent MoE layer's tensors (`HYV4FlashMoE`): the router, the fused banks `gate_up` [E, 2I, L] (gate rows first)
/// and `down` [E, L, I], the latent `lat_down` [L, D] and `lat_up` [D, L], the shared expert and the combine norm.
pub const Moe = struct {
    router: Router,
    gate_up: Arr,
    down: Arr,
    lat_down: Lin,
    lat_up: Lin,
    shared: Swiglu,
    combine: Arr,
};

pub const MoeGeometry = struct { top_k: c_int, inter: c_int, scale: f32, swiglu_limit: f32, eps: f32 };

/// The latent MoE over `x` [T, D] with every expert resident: route, read the picks, then `moeTail` over the banks by
/// pick.
pub fn moe(gpa: std.mem.Allocator, o: *Ops, x: Arr, w: Moe, g: MoeGeometry, at: ?At) !struct { y: Arr, routed: Routed } {
    const r = try route(o, x, w.router, g.top_k, g.scale);
    const picks = try hostPicks(gpa, o, r.picks, @intCast(mlx.getShape(w.router.weight)[0]));
    defer gpa.free(picks);
    if (at) |a| try a.tap.routes(a.tap.ctx, a.layer, picks, r.weights);
    return .{ .y = try moeTail(gpa, o, x, w, g, r, w.gate_up, w.down, picks, at), .routed = r };
}

/// The picks [T, k] as host ids, position-major. With `result` (the experts they index), a readback that steers no
/// shape, made up in a dry recording: the resident route bounds the groups they make (`hy4_expert_mv.plan`). Without, a
/// plain readback, which a dry recording refuses: a streamed route's groups follow the picks.
pub fn hostPicks(gpa: std.mem.Allocator, o: *Ops, picks: Arr, result: ?u32) ![]u32 {
    const c = try o.contiguous(try o.cast(picks, .int32));
    try mlx.check(mlx.mlx_array_eval(c));
    const raw = if (result) |experts| try mlx.hostResult(i32, c, experts) else (mlx.mlx_array_data_int32(c) orelse return error.HostRead)[0..mlx.mlx_array_size(c)];
    const out = try gpa.alloc(u32, raw.len);
    for (raw, out) |v, *d| d.* = @intCast(v);
    return out;
}

/// The latent MoE after its route: `lat = x·lat_downᵀ`, the routed experts (`routedExperts`) over `rows` (host, [T·k]:
/// each pick's row in `gate_up_bank` [S, 2I, L] and `down_bank` [S, L, I], the layer's banks by expert when resident,
/// the expert stream's slots when streamed), `lat_up` back to D, the shared expert added, then the combine norm.
pub fn moeTail(gpa: std.mem.Allocator, o: *Ops, x: Arr, w: Moe, g: MoeGeometry, r: Routed, gate_up_bank: Arr, down_bank: Arr, rows: []const u32, at: ?At) !Arr {
    const lat = try linear(o, x, w.lat_down);
    try At.input(at, .expert_in, lat);
    const tap: ?ExpertTap = if (at) |a| .{ .at = a, .experts = rows } else null;
    const routed = try o.own(try routedExperts(gpa, o.s, lat, r.weights, null, rows, g, gate_up_bank, down_bank, tap));
    return moeFinish(o, x, w, g, routed, at);
}

/// The MoE after its routed sum `routed` (in the latent): out through W_lat_up, the shared expert added, the combine
/// norm. The routed sum is linear in the picks, so a caller may build it from groups of picks.
pub fn moeFinish(o: *Ops, x: Arr, w: Moe, g: MoeGeometry, routed: Arr, at: ?At) !Arr {
    return (try moeFinishBack(o, x, w, g, routed, at, null)).y;
}

/// The next layer's attention mix and norm (`mixNorm`'s operands), which a layer's write-back may run in its launch.
pub const NextMix = struct { mix: Mix, n: c_int, magnitude: f32, eps: f32, norm_eps: f32, norm_w: Arr };

/// The streams a layer's FFN output is written back into (`expand`): `h` [T, n, D] and the mix's `post` [T, n]; with
/// `next`, the next layer's mix of the streams after it.
pub const Back = struct { h: Arr, post: Arr, next: ?NextMix = null };

/// A sublayer's output, or with `expanded` the streams after its write-back; `next` is `Back.next`'s mix of them when
/// the launch ran it.
pub const Fin = struct { y: Arr, expanded: bool, next: ?Mixed = null };

/// Whether a layer's combine norm and write-back run the next layer's mix and norm in their launch (`hy4_hc.backMixNorm`)
/// rather than the norm's launch and the mix's. A test seam: both give the same bytes.
pub var back_mix_on = true;

/// `moeFinish`, and with `back` on a GPU stream the write-back `expand(back.h, back.post, y)` in the combine norm's
/// launch (`hy4_norm.rowsExpand`, `hc_ident_expand`'s arithmetic), the streams returned instead of `y`; with
/// `back.next`, that launch also runs the next layer's `mixNorm` of them (`hy4_hc.backMixNorm`).
pub fn moeFinishBack(o: *Ops, x: Arr, w: Moe, g: MoeGeometry, routed: Arr, at: ?At, back: ?Back) !Fin {
    try At.input(at, .lat_up_in, routed);
    // The shared expert first: launches that finish themselves run in the order made, so it runs beside the routed
    // experts rather than after W_lat_up.
    const narrow = norm_kernel_on and mlx.streamIsGpu(o.s) and mlx.getShape(x)[0] > 1;
    const shared = if (narrow and storedCheap(x, w.shared.down)) try swigluStored(o, x, w.shared, at) else try swiglu(o, x, w.shared, at);
    const up = if (narrow and storedCheap(routed, w.lat_up)) try linearStored(o, routed, w.lat_up) else try linear(o, routed, w.lat_up);
    if (back) |b| if (norm_kernel_on and mix_expand_on and hcKernels(o)) {
        const h = try o.contiguous(try o.cast(b.h, .float32));
        const post = try o.contiguous(try o.cast(b.post, .float32));
        if (b.next) |nx| if (back_mix_on and mix_norm_on and hy4_hc.backApplies(o.s, h, post, up, shared, w.combine, nx.mix.fn_, nx.mix.scale)) {
            const k = try o.own(try hck.identityEps(nx.norm_eps, nx.eps, nx.magnitude));
            const hw: hck.HcWeights = .{ .fn_w = nx.mix.fn_, .scale = nx.mix.scale, .base = nx.mix.base };
            const r = try hy4_hc.backMixNorm(o.s, h, post, up, shared, w.combine, g.eps, hw, k, nx.norm_w);
            return .{ .y = try o.own(r.m.?), .expanded = true, .next = .{ .post = try o.own(r.post), .x = try o.own(r.h) } };
        };
        if (hy4_norm.expandApplies(o.s, up, shared, w.combine, h, post))
            return .{ .y = try o.own(try hy4_norm.rowsExpand(o.s, up, shared, w.combine, g.eps, h, post)), .expanded = true };
    };
    return .{ .y = try normAs(o, up, shared, w.combine, g.eps, .float32), .expanded = false };
}

/// The latent `x · W_lat_downᵀ` [T, L] the trellis experts read: as stored, BF16, from a trellis bank where no tap reads
/// it (its launch rounds the rows, the bytes of the f32 latent cast), else f32 as `linear` gives it.
pub fn expertLatent(o: *Ops, x: Arr, w: Lin, at: ?At) !Arr {
    return if (at == null and w.stq != null) linearStored(o, x, w) else linear(o, x, w);
}

/// A tap on the routed experts: each pick's expert id, beside `rows` (its bank row, a slot when streamed).
pub const ExpertTap = struct { at: At, experts: []const u32 };

/// The most picks one gathered block runs (`routedExperts`): a 4,096-row chunk's 65,536 picks in one block would hold
/// about 5 GB of planes.
pub const pick_block: usize = 8192;

/// The routed experts over `lat` [T, L], summed by their weights [T, k] into [T, L] f32. `picks`: the flat [T, k]
/// positions this call runs (a group of the stream's), or null for every pick in order; `rows`: each one's bank row.
/// Each runs its bank row's clamped SwiGLU (`gate ≤ limit`, `|up| ≤ limit`) by two gathered products over the banks as
/// stored (BF16, f32 sums; the SwiGLU in f32), is weighted, and lands at its position; each row then sums its k. A
/// position no pick of this call names adds exactly nothing. The caller frees the result.
pub fn routedExperts(gpa: std.mem.Allocator, s: mlx.mlx_stream, lat: Arr, weights: Arr, picks: ?[]const u32, rows: []const u32, g: MoeGeometry, gate_up_bank: Arr, down_bank: Arr, tap: ?ExpertTap) !Arr {
    if (picks == null and rows.len <= pick_block) return (try runExperts(gpa, s, lat, weights, null, rows, g, gate_up_bank, down_bank, tap, null)).?;
    const t = mlx.getShape(lat)[0];
    var placed = try placedZeros(s, t, g.top_k, mlx.getShape(lat)[1]);
    defer _ = mlx.mlx_array_free(placed);
    _ = try runExperts(gpa, s, lat, weights, picks, rows, g, gate_up_bank, down_bank, tap, &placed);
    return sumPlaced(s, placed, t, g.top_k);
}

/// The most sorted picks one `routedBatched` gate-up block holds, bounding the SwiGLU's f32 planes.
pub const batched_block: usize = 32768;

/// `routedExperts` over the rows of several pieces at once, each piece's sum as `routedExperts` gives it, byte for byte:
/// the picks are sorted by bank row so each expert's matrix is read once for many rows (`hy4_expert_mv`), where
/// `routedExperts` reads it once per pick. `lat[i]` [T_i, L] and `weights[i]` [T_i, k] are piece `i`'s, `bank_rows` the
/// bank row of each pick of every piece in order and `experts` its expert id (the tap's). `sums[i]` are the pieces'
/// [T_i, L] f32 sums, lazy over the banks: the caller evaluates them before the banks are given back. The products are
/// `products`' (BF16 or f32) over the BF16 banks, on the GPU.
pub fn routedBatched(gpa: std.mem.Allocator, s: mlx.mlx_stream, lat: []const Arr, weights: []const Arr, bank_rows: []const u32, experts: []const u32, g: MoeGeometry, gate_up_bank: Arr, down_bank: Arr, at: ?At, sums: []Arr) !void {
    if (mlx.mlx_array_dtype(gate_up_bank) != .bfloat16 or !mlx.streamIsGpu(s)) return error.Hy4BatchedNeedsBf16BanksOnGpu;
    const round: mlx.mlx_dtype = if (products == .bf16) .bfloat16 else .float32;
    const k: usize = @intCast(g.top_k);
    var total_rows: usize = 0;
    for (lat) |a| total_rows += @intCast(mlx.getShape(a)[0]);
    const picks = total_rows * k;
    if (bank_rows.len != picks or experts.len != picks) return error.Hy4PicksNotRows;
    var o: Ops = .{ .s = s };
    defer o.deinit();
    const l = mlx.getShape(lat[0])[1];
    const slots_n = mlx.getShape(gate_up_bank)[0];
    const gu_bank = try o.reshape(gate_up_bank, &.{ slots_n, 2 * g.inter, l });
    const down = try o.reshape(down_bank, &.{ slots_n, l, g.inter });
    const x = try o.cast(try o.concat(lat, 0), round);
    const w_all = try o.reshape(try o.cast(try o.concat(weights, 0), .float32), &.{ @intCast(picks), 1 });

    // The picks by bank row, a row's picks in position order.
    const slots: usize = @intCast(slots_n);
    const cursor = try gpa.alloc(u32, slots + 1);
    defer gpa.free(cursor);
    @memset(cursor, 0);
    for (bank_rows) |r| {
        if (r >= slots) return error.ExpertOutOfRange;
        cursor[r + 1] += 1;
    }
    for (1..cursor.len) |i| cursor[i] += cursor[i - 1];
    const order = try gpa.alloc(u32, picks);
    defer gpa.free(order);
    for (bank_rows, 0..) |r, q| {
        order[cursor[r]] = @intCast(q);
        cursor[r] += 1;
    }
    const operand = try gpa.alloc(u32, picks);
    defer gpa.free(operand);
    for (order, operand) |q, *d| d.* = q / @as(u32, @intCast(k));
    const iota = try gpa.alloc(u32, picks);
    defer gpa.free(iota);
    for (iota, 0..) |*d, i| d.* = @intCast(i);
    const tapped = try gpa.alloc(u32, if (at != null) picks else 0);
    defer gpa.free(tapped);
    for (tapped, order) |*d, q| d.* = experts[q];

    var acts: std.ArrayList(Arr) = .empty;
    defer {
        for (acts.items) |a| _ = mlx.mlx_array_free(a);
        acts.deinit(gpa);
    }
    var b0: usize = 0;
    while (b0 < picks) : (b0 += batched_block) {
        const b1 = @min(picks, b0 + batched_block);
        const pb: c_int = @intCast(b1 - b0);
        var ob: Ops = .{ .s = s };
        defer ob.deinit();
        const planned = try expert_mv.plan(gpa, bank_rows, order[b0..b1], operand[b0..b1], iota[0 .. b1 - b0]);
        defer planned.deinit();
        const gu = try ob.reshape(try ob.cast(try ob.own(try expert_mv.product(s, gu_bank, x, planned, pb)), .float32), &.{ pb, 2 * g.inter });
        const limit = try ob.scalar(g.swiglu_limit, .float32);
        const gate = try ob.binary(.min, try ob.slice(gu, 1, 0, g.inter), limit);
        const up = try ob.binary(.max, try ob.binary(.min, try ob.slice(gu, 1, g.inter, 2 * g.inter), limit), try ob.unary(.negative, limit));
        const act = try ob.cast(try ob.binary(.mul, try ob.silu(gate), up), round);
        if (at) |a| try a.tap.expert_act(a.tap.ctx, a.layer, act, tapped[b0..b1]);
        try acts.ensureUnusedCapacity(gpa, 1);
        const kept = try ob.result(act);
        errdefer _ = mlx.mlx_array_free(kept);
        try mlx.check(mlx.mlx_array_eval(kept));
        acts.appendAssumeCapacity(kept);
    }
    const act_all = try o.concat(acts.items, 0);
    const planned = try expert_mv.plan(gpa, bank_rows, order, iota, order);
    defer planned.deinit();
    const out = try o.own(try expert_mv.product(s, down, act_all, planned, @intCast(picks)));
    try mlx.check(mlx.mlx_array_eval(out));
    var q0: c_int = 0;
    for (lat, sums) |piece, *sum| {
        const t = mlx.getShape(piece)[0];
        var ob: Ops = .{ .s = s };
        defer ob.deinit();
        const rows = try ob.slice(out, 0, q0 * g.top_k, (q0 + t) * g.top_k);
        const weighted = try ob.binary(.mul, try ob.cast(rows, .float32), try ob.slice(w_all, 0, q0 * g.top_k, (q0 + t) * g.top_k));
        sum.* = try ob.result(try ob.reduce(try ob.reshape(weighted, &.{ t, g.top_k, mlx.getShape(piece)[1] }), 1, false, false));
        q0 += t;
    }
}

/// The [T·k, L] f32 zeros `placeExperts` lands picks in, written on the GPU on the engine (a prefill chunk's are too
/// large for the host ring). The caller frees it.
pub fn placedZeros(s: mlx.mlx_stream, t: c_int, k: c_int, l: c_int) !Arr {
    var o: Ops = .{ .s = s };
    defer o.deinit();
    const zeros = try o.slot();
    const shape = [_]c_int{ t * k, l };
    try mlx.check(if (mlx.engine.isStream(s)) mlx.engine.zerosOnGpu(zeros, &shape, .float32) else mlx.mlx_zeros(zeros, &shape, 2, .float32, s));
    return o.result(zeros.*);
}

/// `routedExperts`' picks `picks` (flat [T, k] positions; null: every pick in order) landed in `placed` [T·k, L] at their
/// positions, each its weighted output: picks placed in any groups sum (`sumPlaced`) to the one call's bytes.
pub fn placeExperts(gpa: std.mem.Allocator, s: mlx.mlx_stream, placed: *Arr, lat: Arr, weights: Arr, picks: ?[]const u32, rows: []const u32, g: MoeGeometry, gate_up_bank: Arr, down_bank: Arr, tap: ?ExpertTap) !void {
    _ = try runExperts(gpa, s, lat, weights, picks, rows, g, gate_up_bank, down_bank, tap, placed);
}

/// Each row's sum of its k placed picks, [T, L] f32. The caller frees it.
pub fn sumPlaced(s: mlx.mlx_stream, placed: Arr, t: c_int, k: c_int) !Arr {
    var o: Ops = .{ .s = s };
    defer o.deinit();
    return o.result(try o.reduce(try o.reshape(placed, &.{ t, k, mlx.getShape(placed)[1] }), 1, false, false));
}

/// The routed experts' blocks: summed per row when `placed` is null (every pick in order, one block), else landed in
/// `placed`.
fn runExperts(gpa: std.mem.Allocator, s: mlx.mlx_stream, lat: Arr, weights: Arr, picks: ?[]const u32, rows: []const u32, g: MoeGeometry, gate_up_bank: Arr, down_bank: Arr, tap: ?ExpertTap, placed: ?*Arr) !?Arr {
    const k: c_int = g.top_k;
    const t = mlx.getShape(lat)[0];
    const l = mlx.getShape(lat)[1];
    const all: c_int = t * k;
    const p: c_int = @intCast(rows.len);
    var o: Ops = .{ .s = s };
    defer o.deinit();
    const slots = mlx.getShape(gate_up_bank)[0];
    // BF16 products: the operands and the outputs rounded to BF16, the sums f32 (MLX's BF16 product). MLX's CPU gather
    // takes f32 only, so on the CPU stream the BF16 operands multiply widened (exact) and the outputs round as the GPU's.
    const round: mlx.mlx_dtype = if (products == .bf16 and mlx.mlx_array_dtype(gate_up_bank) == .bfloat16) .bfloat16 else .float32;
    const dt: mlx.mlx_dtype = if (mlx.streamIsGpu(s)) round else .float32;
    // Each pick's flat position, its row of `lat` (position / k) and its bank row.
    const pos = try gpa.alloc(u32, rows.len);
    defer gpa.free(pos);
    if (picks) |ps| {
        @memcpy(pos, ps);
    } else {
        for (pos, 0..) |*d, i| d.* = @intCast(i);
    }
    const row_of = try gpa.alloc(u32, rows.len);
    defer gpa.free(row_of);
    for (row_of, pos) |*d, q| d.* = q / @as(u32, @intCast(k));
    const x = try o.reshape(try o.cast(try o.cast(lat, round), dt), &.{ t, 1, l });
    // A BF16 bank on the GPU runs `hy4_expert_mv`: it reads the bank as stored and widens in the register, gather_mm's
    // bytes at either product, each expert's matrix read once for up to `vectors` of its picks; no widened bank copy.
    const mv = mlx.streamIsGpu(s) and mlx.mlx_array_dtype(gate_up_bank) == .bfloat16;
    const gu_bank = try o.reshape(gate_up_bank, &.{ slots, 2 * g.inter, l });
    const dn_bank = try o.reshape(down_bank, &.{ slots, l, g.inter });
    const w_gu = if (mv) gu_bank else try o.transpose(try o.cast(gu_bank, dt), &.{ 0, 2, 1 }); // [S, L, 2I]
    const w_down = if (mv) dn_bank else try o.transpose(try o.cast(dn_bank, dt), &.{ 0, 2, 1 }); // [S, I, L]
    const x_rows = try o.reshape(x, &.{ t, l });
    // A block's picks by bank row (a row's picks in order), and the `lat` row each reads.
    const order = try gpa.alloc(u32, if (mv) @min(rows.len, pick_block) else 0);
    defer gpa.free(order);
    const operand = try gpa.alloc(u32, order.len);
    defer gpa.free(operand);
    const w_flat = try o.reshape(try o.cast(weights, .float32), &.{all});
    // Every pick in order in one block lands as it is; otherwise each block of at most `pick_block` picks lands at its
    // positions in a [T·k, L] sum, so a wide chunk holds one block's planes.
    const direct = placed == null;
    if (direct and (picks != null or p > pick_block)) return error.Hy4PicksNotOneBlock;
    var b0: usize = 0;
    while (b0 < rows.len) : (b0 += pick_block) {
        const b1 = @min(rows.len, b0 + pick_block);
        const pb: c_int = @intCast(b1 - b0);
        var ob: Ops = .{ .s = s };
        defer ob.deinit();
        const lhs = try ob.own(mlx.mlx_array_new_data(row_of[b0..b1].ptr, &[_]c_int{pb}, 1, .uint32));
        const rhs = try ob.own(mlx.mlx_array_new_data(rows[b0..b1].ptr, &[_]c_int{pb}, 1, .uint32));
        const at = try ob.own(mlx.mlx_array_new_data(pos[b0..b1].ptr, &[_]c_int{pb}, 1, .uint32));
        var planned: ?expert_mv.Plan = null;
        defer if (planned) |pl| pl.deinit();
        var dest_plan: ?expert_mv.Plan = null;
        defer if (dest_plan) |pl| pl.deinit();
        const gu_out = if (mv) blk: {
            const ord = order[0 .. b1 - b0];
            for (ord, 0..) |*d, i| d.* = @intCast(i);
            std.mem.sort(u32, ord, rows[b0..b1], struct {
                fn less(bank_rows: []const u32, a: u32, b: u32) bool {
                    return bank_rows[a] < bank_rows[b] or (bank_rows[a] == bank_rows[b] and a < b);
                }
            }.less);
            for (ord, operand[0..ord.len]) |i, *d| d.* = row_of[b0 + i];
            planned = try expert_mv.plan(gpa, rows[b0..b1], ord, operand[0..ord.len], ord);
            dest_plan = try expert_mv.plan(gpa, rows[b0..b1], ord, ord, ord);
            break :blk try ob.own(try expert_mv.product(s, w_gu, x_rows, planned.?, pb));
        } else blk: {
            const slot = try ob.slot();
            try mlx.check(mlx.mlx_gather_mm(slot, x, w_gu, lhs, rhs, false, s));
            break :blk slot.*;
        };
        const gu = try ob.reshape(try ob.cast(try ob.cast(gu_out, round), .float32), &.{ pb, 2 * g.inter }); // [P, 2I]
        const limit = try ob.scalar(g.swiglu_limit, .float32);
        const gate = try ob.binary(.min, try ob.slice(gu, 1, 0, g.inter), limit);
        const up = try ob.binary(.max, try ob.binary(.min, try ob.slice(gu, 1, g.inter, 2 * g.inter), limit), try ob.unary(.negative, limit));
        const act = try ob.cast(try ob.binary(.mul, try ob.silu(gate), up), round);
        if (tap) |et| try et.at.tap.expert_act(et.at.tap.ctx, et.at.layer, act, et.experts[b0..b1]);
        const out = if (mv) try ob.own(try expert_mv.product(s, w_down, try ob.cast(act, dt), dest_plan.?, pb)) else blk: {
            const slot = try ob.slot();
            try mlx.check(mlx.mlx_gather_mm(slot, try ob.reshape(try ob.cast(act, dt), &.{ pb, 1, g.inter }), w_down, .{ .ctx = null }, rhs, false, s));
            break :blk slot.*;
        };
        const w_at = try ob.reshape(try ob.take(w_flat, at, 0), &.{ pb, 1 });
        const weighted = try ob.binary(.mul, try ob.reshape(try ob.cast(try ob.cast(out, round), .float32), &.{ pb, l }), w_at); // [P, L]
        if (direct) return try o.result(try ob.reduce(try ob.reshape(weighted, &.{ t, k, l }), 1, false, false));
        // On the engine each block lands its picks in the sum's own bytes, so no block hands a copy of it up.
        if (mlx.engine.isStream(s)) {
            try mlx.engine.putAlongInPlace(placed.?.*, try ob.broadcast(try ob.reshape(at, &.{ pb, 1 }), &.{ pb, l }), weighted, 0);
            try mlx.check(mlx.mlx_array_eval(placed.?.*));
            continue;
        }
        const put = try ob.slot();
        // The positions broadcast along the row: the engine's put takes indices of the values' shape.
        const at_rows = try ob.broadcast(try ob.reshape(at, &.{ pb, 1 }), &.{ pb, l });
        try mlx.check(mlx.mlx_put_along_axis(put, placed.?.*, at_rows, weighted, 0, s));
        const next = try ob.result(put.*);
        try mlx.check(mlx.mlx_array_eval(next));
        _ = mlx.mlx_array_free(placed.?.*);
        placed.?.* = next;
    }
    return null;
}

// ── attention ───────────────────────────────────────────────────────────────────────────────────────────────────────

/// An MLA layer's tensors, by the release's names: `q_a_proj`, `q_a_layernorm`, `q_b_proj`, `kv_a_proj_with_mqa`,
/// `kv_a_layernorm`, `kv_b_proj`, `linear_gate`, `o_proj`, `post_proj_norm`, `learnable_sink_param` (f32).
pub const Attn = struct { q_a: Lin, q_a_norm: Arr, q_b: Lin, kv_a: Lin, kv_a_norm: Arr, kv_b: Arr, gate: Lin, o: Lin, post_norm: Arr, sinks: Arr };

pub const AttnGeometry = struct {
    heads: c_int,
    nope: c_int,
    rope: c_int,
    v: c_int,
    latent: c_int,
    theta: f32,
    eps: f32,
};

/// `rope(x)` over the trailing `rope` dims of `x` [H, T, d] at positions `pos0..pos0+T`: consecutive pairs
/// (`(x₂ᵢ, x₂ᵢ₊₁)`, ggml `NORM`; transformers' hy_v4_vl `rotate_half`), `inv_freqᵢ = θ^(-2i/rope)`.
pub fn rope(o: *Ops, x: Arr, rope_dims: c_int, theta: f32, pos0: c_int) !Arr {
    return ropeScaled(o, x, rope_dims, theta, pos0, 1);
}

/// `rope` with row t at position `(pos0 + t) · step`: the indexer's group keys sit at their groups' first slots.
pub fn ropeScaled(o: *Ops, x: Arr, rope_dims: c_int, theta: f32, pos0: c_int, step: c_int) !Arr {
    return ropeScaledAs(o, x, rope_dims, theta, pos0, step, .float32);
}

/// Whether a GPU stream rotates through `hy4_rope.tail`'s one launch rather than the slice, rope and concat. A test
/// seam: both give the same bytes.
pub var rope_kernel_on = true;

/// `ropeScaled` cast to `out_dt` (the kernel writes it so).
fn ropeScaledAs(o: *Ops, x: Arr, rope_dims: c_int, theta: f32, pos0: c_int, step: c_int, out_dt: mlx.mlx_dtype) !Arr {
    if (rope_kernel_on and hy4_rope.applies(o.s, x, rope_dims)) return o.own(try hy4_rope.tail(o.s, x, rope_dims, theta, pos0, step, out_dt));
    return o.cast(try ropeChain(o, x, rope_dims, theta, pos0, step), out_dt);
}

fn ropeChain(o: *Ops, x: Arr, rope_dims: c_int, theta: f32, pos0: c_int, step: c_int) !Arr {
    const d = mlx.getShape(x)[2];
    const pass = try o.slice(x, 2, 0, d - rope_dims);
    const rot = try o.contiguous(try o.slice(x, 2, d - rope_dims, d));
    const out = try o.slot();
    try mlx.check(mlx.mlx_fast_rope(out, rot, rope_dims, true, .{ .value = theta, .has_value = true }, @floatFromInt(step), pos0, .{ .ctx = null }, o.s));
    return o.concat(&.{ pass, out.* }, 2);
}

/// What an MLA layer's attention gives: its output [T, D] (after `post_proj_norm`) and `q_resid` [T, q_lora], which
/// the indexer reads.
pub const AttnOut = struct { y: Arr, q_resid: Arr };

/// MLA over the rows `x` [T, D] at positions `pos0..` (`HYV4FlashAttention`): `q = W_qb·RMS(W_qa·x)` split into nope and
/// rope; the latent `[c | k_pe] = W_kva·x`, `c = RMS(c)`, `[k_nope | v] = W_kvb·c` per head; the rope parts rotated;
/// scores `q·k · (nope+rope)^-½` plus `mask` [T, T] (0 or -inf), a learned sink logit per head beside them in the
/// softmax and dropped after; the output gated by `σ(W_gate·x)`, then `W_o` and `post_proj_norm`. Expanded per head
/// as transformers runs it; llama.cpp's absorbed form is the same sum.
pub fn attention(o: *Ops, x: Arr, w: Attn, g: AttnGeometry, pos0: c_int, mask: Arr) !AttnOut {
    const p = try project(o, x, w, g, pos0);
    // The key rows as the cache holds them at its BF16 width.
    return .{ .y = try attend(o, x, p.q, try o.cast(p.c, .bfloat16), try o.cast(p.k_pe, .bfloat16), w, g, mask, null), .q_resid = p.q_resid };
}

/// The new rows' projections: the rotated queries `q` [H, T, qk], the normalized latent `c` [T, latent] and the
/// rotated `k_pe` [T, rope] at the cache's BF16 (a row the cache keeps for every later query), and `q_resid` for the
/// indexer; a decode row also keeps `q_raw`, its queries before the rotation, for the absorb to read beside the rope.
pub const Projected = struct { q: Arr, c: Arr, k_pe: Arr, q_resid: Arr, q_raw: ?Arr = null };

pub fn project(o: *Ops, x: Arr, w: Attn, g: AttnGeometry, pos0: c_int) !Projected {
    return projectWith(o, x, w, g, pos0, .float32, .float32, .float32);
}

/// `project` with the latent `c` at `c_dt` (the cache's BF16 for a caller that only stores it).
pub fn projectAs(o: *Ops, x: Arr, w: Attn, g: AttnGeometry, pos0: c_int, c_dt: mlx.mlx_dtype) !Projected {
    return projectWith(o, x, w, g, pos0, c_dt, .bfloat16, .float32);
}

/// `projectAs` with the q residual written in `q_dt`: BF16 where `W_qb` reads its input at BF16 anyway.
pub fn projectAsQ(o: *Ops, x: Arr, w: Attn, g: AttnGeometry, pos0: c_int, c_dt: mlx.mlx_dtype, q_dt: mlx.mlx_dtype) !Projected {
    return projectWith(o, x, w, g, pos0, c_dt, .bfloat16, q_dt);
}

fn projectWith(o: *Ops, x: Arr, w: Attn, g: AttnGeometry, pos0: c_int, c_dt: mlx.mlx_dtype, k_dt: mlx.mlx_dtype, q_dt: mlx.mlx_dtype) !Projected {
    const t = mlx.getShape(x)[0];
    const qk = g.nope + g.rope;
    const q_resid = try linearNormedAs(o, x, w.q_a, w.q_a_norm, g.eps, q_dt);
    const qb_raw = if (t > 1 and rope_kernel_on and mlx.streamIsGpu(o.s) and storedCheap(q_resid, w.q_b)) try linearStored(o, q_resid, w.q_b) else try linear(o, q_resid, w.q_b);
    const q = try o.transpose(try o.reshape(qb_raw, &.{ t, g.heads, qk }), &.{ 1, 0, 2 }); // [H, T, qk]
    const use_front = k_dt == .bfloat16 and front_kernel_on and norm_kernel_on and rope_kernel_on;
    const ckv = if (t > 1 and use_front and mlx.streamIsGpu(o.s) and storedCheap(x, w.kv_a)) try linearStored(o, x, w.kv_a) else try linear(o, x, w.kv_a); // [T, latent + rope]
    // The latent's norm and its key's rotation in one launch.
    if (use_front and hy4_front.applies(o.s, ckv, w.kv_a_norm, g.latent, g.rope)) {
        const f = try hy4_front.run(o.s, try o.contiguous(ckv), w.kv_a_norm, g.latent, g.rope, g.theta, g.eps, pos0, c_dt);
        const c = o.own(f.c) catch |e| {
            _ = mlx.mlx_array_free(f.k_pe);
            return e;
        };
        const k_pe = try o.own(f.k_pe);
        return .{ .q = try rope(o, q, g.rope, g.theta, pos0), .c = c, .k_pe = k_pe, .q_resid = q_resid, .q_raw = if (t == 1 and rope_kernel_on and hy4_rope.applies(o.s, q, g.rope)) q else null };
    }
    const c = try normAs(o, try o.slice(ckv, 1, 0, g.latent), null, w.kv_a_norm, g.eps, c_dt);
    const k_pe = try o.reshape(try o.slice(ckv, 1, g.latent, g.latent + g.rope), &.{ 1, t, g.rope });
    const k_rot = try o.reshape(try ropeScaledAs(o, k_pe, g.rope, g.theta, pos0, 1, k_dt), &.{ t, g.rope });
    return .{ .q = try rope(o, q, g.rope, g.theta, pos0), .c = c, .k_pe = k_rot, .q_resid = q_resid };
}

/// The new rows' queries `q` [H, T, qk] over the key rows `c` [N, latent] and `k_pe` [N, rope] under `mask` [T, N]:
/// scores `q·k · qk^-½` with `k = [W_k·c | k_pe]`, a learned sink logit per head beside them in the softmax and dropped
/// after, values `W_v·c`; gated by `σ(W_gate·x)`, then `W_o` and `post_proj_norm`. Absorbed, as llama.cpp runs MLA
/// (the same sums as transformers' per-head expansion): the queries go into the latent through `W_k` and the attended
/// latent comes out through `W_v`, so no row of keys is expanded per head.
pub fn attend(o: *Ops, x: Arr, q: Arr, c: Arr, k_pe: Arr, w: Attn, g: AttnGeometry, mask: Arr, at: ?At) !Arr {
    return attendGated(o, .{ .gate = try linearStored(o, x, w.gate), .whole = true }, q, c, k_pe, w, g, mask, at);
}

/// What an attention span reads beside its queries and keys: its rows' gate product as stored (`linearStored` of
/// `W_g`, [T, H·v]), and `kv_b` widened to f32 once for every span of a forward (null: the span widens it, or a
/// one-row span reads it as stored).
/// A span's shared inputs; `whole` when it is the chunk's only span, so `attendGated` runs `W_o` and its norm itself.
/// `o_left`: with `whole`, W_o's launch may leave its rows un-normed or its k-split partials for the iHC mix to finish:
/// the span's result is then what it left, and which one lands here.
/// A span's gated rows written in place: the chunk's block `W_o` reads, and the span's first row in it.
pub const GatedDst = struct { block: Arr, row: c_int };
/// A sparse layer's indexer inputs made once for the chunk's rows from `r0` (`indexPrepare`): the rotated queries, the
/// head weights and the rotated group keys, sliced per span.
pub const IndexIn = struct { r0: c_int, qt: Arr, hw: Arr, k_rot: Arr };
pub const SpanIn = struct { gate: Arr, kv_b32: ?Arr = null, whole: bool = false, o_left: ?*OLeft = null, dst: ?GatedDst = null, index: ?IndexIn = null, rows_as: ?c_int = null, q_raw: ?Arr = null };

/// What a whole span's W_o left for the iHC mix: its rows normed, its k-split partials (`osplit` a row) for
/// `expandMixNormParts`, or its finished rows for `expandMixNormRows`.
pub const OLeft = union(enum) { normed, parts: c_int, rows };

/// `attend` up to `W_o`: the gated attention [T, H·v] as `W_o` reads it (`linearInput`), or past `W_o` and its norm when
/// `in.whole`.
pub fn attendGated(o: *Ops, in: SpanIn, q: Arr, c: Arr, k_pe: Arr, w: Attn, g: AttnGeometry, mask: Arr, at: ?At) !Arr {
    return attendGatedMask(o, in, q, c, k_pe, w, g, .{ .array = mask }, at);
}

/// `attendGated` for a decode row over a full window of keys (the context past the window): every key visible, the
/// score and value products as steel launches.
pub fn attendGatedFull(o: *Ops, in: SpanIn, q: Arr, c: Arr, k_pe: Arr, w: Attn, g: AttnGeometry, p0: c_int, at: ?At) !Arr {
    return attendGatedMask(o, in, q, c, k_pe, w, g, .{ .causal = .{ .p0 = p0, .window = 0, .steel = true } }, at);
}

/// `attendGated` under the causal mask of rows at `p0..` (and only the `window` keys ending at each when `window` > 0),
/// the mask taken inside the softmax launch where it serves, so no mask array is made.
pub fn attendGatedCausal(o: *Ops, in: SpanIn, q: Arr, c: Arr, k_pe: Arr, w: Attn, g: AttnGeometry, p0: c_int, window: c_int, at: ?At) !Arr {
    return attendGatedMask(o, in, q, c, k_pe, w, g, .{ .causal = .{ .p0 = p0, .window = window } }, at);
}

pub fn attendGatedRing(o: *Ops, in: SpanIn, q: Arr, c: Arr, k_pe: Arr, w: Attn, g: AttnGeometry, p0: c_int, window: c_int, ring_p0: c_int, at: ?At) !Arr {
    return attendGatedMask(o, in, q, c, k_pe, w, g, .{ .causal = .{ .p0 = p0, .window = window, .ring_p0 = ring_p0 } }, at);
}

/// The mask an attention core reads: an additive or boolean array, or the causal mask by positions.
pub const MaskSpec = union(enum) {
    array: Arr,
    causal: struct { p0: c_int, window: c_int, ring_p0: ?c_int = null, steel: bool = false },
};

fn freshBuf(o: *Ops, shape: []const c_int, dtype: mlx.mlx_dtype) !Arr {
    const out = try o.slot();
    try mlx.check(mlx.engine.freshArray(out, shape, dtype));
    return out.*;
}

fn mmInto(a: Arr, b: Arr, dst: Arr) !void {
    try mlx.check(mlx.engine.matmulInto(a, b, dst));
}

fn attendGatedMask(o: *Ops, in: SpanIn, q: Arr, c: Arr, k_pe: Arr, w: Attn, g: AttnGeometry, mask_spec: MaskSpec, at: ?At) !Arr {
    const ab = try absorb(o, q, w, g, in.kv_b32, in.rows_as, in.q_raw); // [H, T, latent], [H, T, rope]
    const u_lat = try causalLatent(o, ab.q_lat, ab.q_pe, c, k_pe, w, g, mask_spec);
    return gatedOut(o, in.gate, u_lat, ab.w_v, w, g, at, in.whole, in.o_left, in.dst, in.rows_as);
}

pub fn causalLatent(o: *Ops, q_lat: Arr, q_pe: Arr, c: Arr, k_pe: Arr, w: Attn, g: AttnGeometry, mask_spec: MaskSpec) !Arr {
    const t = mlx.getShape(q_lat)[1];
    const n = mlx.getShape(c)[0];
    const own_stream = mlx.engine.isStream(o.s);
    // A verify block's rows each take the one-row shapes of the score and value products: their bytes are the
    // decode step's (MLX's GEMM picks its kernel by the row count).
    return if (isVerifyBlock(t)) blk: {
        const tu: usize = @intCast(t);
        var u_rows: [verify_rows_max]Arr = undefined;
        switch (mask_spec) {
            .causal => |cm| {
                const use_ring = chain_kernels_on and cm.ring_p0 != null and hy4_chain.ringKeysApplies(o.s, c, k_pe, t, cm.p0, cm.window, cm.ring_p0.?);
                const ring = if (use_ring) rk: {
                    const built = try hy4_chain.ringKeys(o.s, c, k_pe, t, cm.p0, cm.window, cm.ring_p0.?);
                    const rc = o.own(built.c) catch |e| {
                        _ = mlx.mlx_array_free(built.pe);
                        return e;
                    };
                    break :rk .{ .c = rc, .pe = try o.own(built.pe) };
                } else null;
                const c32 = if (use_ring) undefined else try o.cast(c, .float32);
                const pe32 = if (use_ring) undefined else try o.cast(k_pe, .float32);
                const u_buf: ?Arr = if (own_stream) try freshBuf(o, &.{ t, g.heads, g.latent }, .float32) else null;
                for (0..tu) |r| {
                    const ri: c_int = @intCast(r);
                    var c_r: Arr = undefined;
                    var pe_r: Arr = undefined;
                    var nr: c_int = undefined;
                    var p0_r: c_int = undefined;
                    if (ring) |rk| {
                        c_r = try o.reshape(try o.slice(rk.c, 0, ri, ri + 1), &.{ cm.window, g.latent });
                        pe_r = try o.reshape(try o.slice(rk.pe, 0, ri, ri + 1), &.{ cm.window, g.rope });
                        nr = cm.window;
                        p0_r = cm.window - 1;
                    } else {
                        const hi = @min(n, cm.p0 + ri + 1);
                        const lo: c_int = if (cm.window > 0) @max(0, cm.p0 + ri + 1 - cm.window) else 0;
                        c_r = try o.slice(c32, 0, lo, hi);
                        pe_r = try o.slice(pe32, 0, lo, hi);
                        if (cm.ring_p0) |rp0| if (cm.window > 0 and rp0 + ri + 1 > cm.window) {
                            const slot = @mod(rp0 + ri + 1 - cm.window, cm.window);
                            if (slot != 0) {
                                const cut = cm.window - slot;
                                c_r = try o.concat(&.{ try o.slice(c_r, 0, cut, cm.window), try o.slice(c_r, 0, 0, cut) }, 0);
                                pe_r = try o.concat(&.{ try o.slice(pe_r, 0, cut, cm.window), try o.slice(pe_r, 0, 0, cut) }, 0);
                            }
                        };
                        nr = hi - lo;
                        p0_r = if (cm.window > 0) nr - 1 else cm.p0 + ri;
                    }
                    const s1_r = try o.binary(.mm, try o.slice(q_lat, 1, ri, ri + 1), try o.transpose(c_r, &.{ 1, 0 }));
                    const s2_r = try o.binary(.mm, try o.slice(q_pe, 1, ri, ri + 1), try o.transpose(pe_r, &.{ 1, 0 }));
                    const p_r = if (chain_kernels_on and hy4_chain.sinkSoftmaxCausalApplies(o.s, s1_r, s2_r, w.sinks, true))
                        try o.own(try hy4_chain.sinkSoftmaxCausal(o.s, s1_r, s2_r, w.sinks, scaleOf(g), p0_r, 0, true))
                    else
                        try sinkSoftmax(o, try o.binary(.add, try o.binary(.mul, try o.binary(.add, s1_r, s2_r), try o.scalar(scaleOf(g), .float32)), try causalMask(o, 1, nr, p0_r, 0)), w.sinks, g.heads, 1, nr, true);
                    if (u_buf) |ub| {
                        try mmInto(p_r, c_r, try o.reshape(try o.slice(ub, 0, ri, ri + 1), &.{ g.heads, 1, g.latent }));
                    } else {
                        u_rows[r] = try o.binary(.mm, p_r, c_r);
                    }
                }
                if (u_buf) |ub| break :blk try o.transpose(ub, &.{ 1, 0, 2 });
                break :blk try o.concat(u_rows[0..tu], 1);
            },
            .array => |m| {
                const c32 = try o.cast(c, .float32);
                const pe32 = try o.cast(k_pe, .float32);
                const c32_t = try o.transpose(c32, &.{ 1, 0 });
                const pe32_t = try o.transpose(pe32, &.{ 1, 0 });
                if (own_stream) {
                    const s1_t = try freshBuf(o, &.{ t, g.heads, n }, .float32);
                    const s2_t = try freshBuf(o, &.{ t, g.heads, n }, .float32);
                    for (0..tu) |r| {
                        const ri: c_int = @intCast(r);
                        try mmInto(try o.slice(q_lat, 1, ri, ri + 1), c32_t, try o.reshape(try o.slice(s1_t, 0, ri, ri + 1), &.{ g.heads, 1, n }));
                        try mmInto(try o.slice(q_pe, 1, ri, ri + 1), pe32_t, try o.reshape(try o.slice(s2_t, 0, ri, ri + 1), &.{ g.heads, 1, n }));
                    }
                    const p_t = if (chain_kernels_on and hy4_chain.sinkSoftmaxMaskApplies(o.s, s1_t, s2_t, m, w.sinks, false))
                        try o.own(try hy4_chain.sinkSoftmaxMask(o.s, s1_t, s2_t, m, w.sinks, scaleOf(g), false))
                    else inner: {
                        const m3 = if (mlx.getShape(m).len == 2) try o.reshape(m, &.{ t, 1, n }) else m;
                        break :inner try sinkSoftmax(o, try o.binary(.add, try o.binary(.mul, try o.binary(.add, s1_t, s2_t), try o.scalar(scaleOf(g), .float32)), m3), w.sinks, g.heads, t, n, false);
                    };
                    const u_buf = try freshBuf(o, &.{ t, g.heads, g.latent }, .float32);
                    for (0..tu) |r| {
                        const ri: c_int = @intCast(r);
                        try mmInto(try o.reshape(try o.slice(p_t, 0, ri, ri + 1), &.{ g.heads, 1, n }), c32, try o.reshape(try o.slice(u_buf, 0, ri, ri + 1), &.{ g.heads, 1, g.latent }));
                    }
                    break :blk try o.transpose(u_buf, &.{ 1, 0, 2 });
                }
                var s1_rows: [verify_rows_max]Arr = undefined;
                var s2_rows: [verify_rows_max]Arr = undefined;
                for (0..tu) |r| {
                    const ri: c_int = @intCast(r);
                    s1_rows[r] = try o.binary(.mm, try o.slice(q_lat, 1, ri, ri + 1), c32_t);
                    s2_rows[r] = try o.binary(.mm, try o.slice(q_pe, 1, ri, ri + 1), pe32_t);
                }
                const s1 = try o.concat(s1_rows[0..tu], 1);
                const s2 = try o.concat(s2_rows[0..tu], 1);
                const p = if (chain_kernels_on and hy4_chain.sinkSoftmaxMaskApplies(o.s, s1, s2, m, w.sinks, true))
                    try o.own(try hy4_chain.sinkSoftmaxMask(o.s, s1, s2, m, w.sinks, scaleOf(g), true))
                else
                    try sinkSoftmax(o, try o.binary(.add, try o.binary(.mul, try o.binary(.add, s1, s2), try o.scalar(scaleOf(g), .float32)), m), w.sinks, g.heads, t, n, true);
                for (0..tu) |r| {
                    const ri: c_int = @intCast(r);
                    u_rows[r] = try o.binary(.mm, try o.slice(p, 1, ri, ri + 1), c32);
                }
                break :blk try o.concat(u_rows[0..tu], 1);
            },
        }
    } else blk: {
        const c32 = try o.cast(c, .float32);
        const steel = t == 1 and chain_kernels_on and mask_spec == .causal and mask_spec.causal.steel and hy4_chain.steelScoresApplies(o.s, q_lat, q_pe, c32, k_pe);
        var s1: Arr = undefined;
        var s2: Arr = undefined;
        if (steel) {
            const sc = try hy4_chain.steelScores(o.s, q_lat, q_pe, c32, k_pe);
            s1 = try o.own(sc.s1);
            s2 = try o.own(sc.s2);
        } else {
            const pe32 = try o.cast(k_pe, .float32);
            s1 = try o.binary(.mm, q_lat, try o.transpose(c32, &.{ 1, 0 }));
            s2 = try o.binary(.mm, q_pe, try o.transpose(pe32, &.{ 1, 0 }));
        }
        const p = switch (mask_spec) {
            .causal => |cm| if (chain_kernels_on and hy4_chain.sinkSoftmaxCausalApplies(o.s, s1, s2, w.sinks, true))
                try o.own(try hy4_chain.sinkSoftmaxCausal(o.s, s1, s2, w.sinks, scaleOf(g), cm.p0, cm.window, true))
            else
                try sinkSoftmax(o, try o.binary(.add, try o.binary(.mul, try o.binary(.add, s1, s2), try o.scalar(scaleOf(g), .float32)), try causalMask(o, t, n, cm.p0, cm.window)), w.sinks, g.heads, t, n, true),
            .array => |m| if (chain_kernels_on and hy4_chain.sinkSoftmaxMaskApplies(o.s, s1, s2, m, w.sinks, true))
                try o.own(try hy4_chain.sinkSoftmaxMask(o.s, s1, s2, m, w.sinks, scaleOf(g), true))
            else
                try sinkSoftmax(o, try o.binary(.add, try o.binary(.mul, try o.binary(.add, s1, s2), try o.scalar(scaleOf(g), .float32)), m), w.sinks, g.heads, t, n, true),
        };
        if (steel and hy4_chain.steelValuesApplies(o.s, p, c32)) break :blk try o.own(try hy4_chain.steelValues(o.s, p, c32));
        break :blk try o.binary(.mm, p, c32);
    };
}

/// `attendGated` over each row's own key rows: `c_g` [T·S, latent] and `pe_g` [T·S, rope], row-major by query row, and
/// `keep` [T, S] (whether each is one the row reads). A sparse layer past its selection threshold reads its recalled
/// groups' rows this way, so a row's cost does not grow with the context.
pub fn attendRowsGated(o: *Ops, in: SpanIn, q: Arr, c_g_flat: Arr, pe_g_flat: Arr, keep: Arr, w: Attn, g: AttnGeometry, at: ?At) !Arr {
    const ab = try absorb(o, q, w, g, in.kv_b32, in.rows_as, in.q_raw);
    const u_lat = try rowsLatent(o, ab.q_lat, ab.q_pe, c_g_flat, pe_g_flat, keep, w, g);
    return gatedOut(o, in.gate, u_lat, ab.w_v, w, g, at, in.whole, in.o_left, in.dst, in.rows_as);
}

pub fn attendStraddleGated(o: *Ops, in: SpanIn, q: Arr, cut: c_int, c: Arr, k_pe: Arr, p0: c_int, c_g_flat: Arr, pe_g_flat: Arr, keep: Arr, w: Attn, g: AttnGeometry, at: ?At) !Arr {
    const t = mlx.getShape(q)[1];
    const ab = try absorb(o, q, w, g, in.kv_b32, in.rows_as, in.q_raw);
    const u_d = try causalLatent(o, try o.slice(ab.q_lat, 1, 0, cut), try o.slice(ab.q_pe, 1, 0, cut), c, k_pe, w, g, .{ .causal = .{ .p0 = p0, .window = 0 } });
    const u_g = try rowsLatent(o, try o.slice(ab.q_lat, 1, cut, t), try o.slice(ab.q_pe, 1, cut, t), c_g_flat, pe_g_flat, keep, w, g);
    return gatedOut(o, in.gate, try o.concat(&.{ u_d, u_g }, 1), ab.w_v, w, g, at, in.whole, in.o_left, in.dst, in.rows_as);
}

pub fn rowsLatent(o: *Ops, q_lat_h: Arr, q_pe_h: Arr, c_g_flat: Arr, pe_g_flat: Arr, keep: Arr, w: Attn, g: AttnGeometry) !Arr {
    const t = mlx.getShape(q_lat_h)[1];
    const sel = selectedWidth(keep);
    const c_g = try o.reshape(try o.cast(c_g_flat, .float32), &.{ t, sel, g.latent }); // [T, S, latent]
    const pe_g_bf = try o.reshape(pe_g_flat, &.{ t, sel, g.rope });
    const q_lat = try o.transpose(q_lat_h, &.{ 1, 0, 2 }); // [T, H, latent]
    const q_pe = try o.transpose(q_pe_h, &.{ 1, 0, 2 });
    const steel = t == 1 and chain_kernels_on and hy4_chain.steelScoresApplies(o.s, q_lat, q_pe, c_g, pe_g_bf);
    const pe_g = if (steel) pe_g_bf else try o.reshape(try o.cast(pe_g_flat, .float32), &.{ t, sel, g.rope });
    const c_g_t = try o.transpose(c_g, &.{ 0, 2, 1 });
    const pe_g_t = try o.transpose(pe_g, &.{ 0, 2, 1 });
    // A verify block's rows each take the one-row shapes of the score and value products: their bytes are the
    // decode step's (MLX's GEMM picks its kernel by the row count).
    return if (isVerifyBlock(t)) blk: {
        const tu: usize = @intCast(t);
        const own_stream = mlx.engine.isStream(o.s);
        var s1: Arr = undefined;
        var s2: Arr = undefined;
        if (own_stream) {
            s1 = try freshBuf(o, &.{ t, g.heads, sel }, .float32);
            s2 = try freshBuf(o, &.{ t, g.heads, sel }, .float32);
            for (0..tu) |r| {
                const ri: c_int = @intCast(r);
                try mmInto(try o.slice(q_lat, 0, ri, ri + 1), try o.slice(c_g_t, 0, ri, ri + 1), try o.slice(s1, 0, ri, ri + 1));
                try mmInto(try o.slice(q_pe, 0, ri, ri + 1), try o.slice(pe_g_t, 0, ri, ri + 1), try o.slice(s2, 0, ri, ri + 1));
            }
        } else {
            var s1_rows: [verify_rows_max]Arr = undefined;
            var s2_rows: [verify_rows_max]Arr = undefined;
            for (0..tu) |r| {
                const ri: c_int = @intCast(r);
                s1_rows[r] = try o.binary(.mm, try o.slice(q_lat, 0, ri, ri + 1), try o.slice(c_g_t, 0, ri, ri + 1));
                s2_rows[r] = try o.binary(.mm, try o.slice(q_pe, 0, ri, ri + 1), try o.slice(pe_g_t, 0, ri, ri + 1));
            }
            s1 = try o.concat(s1_rows[0..tu], 0);
            s2 = try o.concat(s2_rows[0..tu], 0);
        }
        const p = if (chain_kernels_on and hy4_chain.sinkSoftmaxMaskApplies(o.s, s1, s2, keep, w.sinks, false))
            try o.own(try hy4_chain.sinkSoftmaxMask(o.s, s1, s2, keep, w.sinks, scaleOf(g), false))
        else inner: {
            const scores = try o.binary(.add, s1, s2);
            const mask = try o.reshape(try additive(o, keep), &.{ t, 1, sel });
            break :inner try sinkSoftmax(o, try o.binary(.add, try o.binary(.mul, scores, try o.scalar(scaleOf(g), .float32)), mask), w.sinks, g.heads, t, sel, false);
        };
        if (own_stream) {
            const u_buf = try freshBuf(o, &.{ t, g.heads, g.latent }, .float32);
            for (0..tu) |r| {
                const ri: c_int = @intCast(r);
                try mmInto(try o.slice(p, 0, ri, ri + 1), try o.slice(c_g, 0, ri, ri + 1), try o.slice(u_buf, 0, ri, ri + 1));
            }
            break :blk try o.transpose(u_buf, &.{ 1, 0, 2 });
        }
        var u_rows: [verify_rows_max]Arr = undefined;
        for (0..tu) |r| {
            const ri: c_int = @intCast(r);
            u_rows[r] = try o.binary(.mm, try o.slice(p, 0, ri, ri + 1), try o.slice(c_g, 0, ri, ri + 1));
        }
        break :blk try o.transpose(try o.concat(u_rows[0..tu], 0), &.{ 1, 0, 2 });
    } else blk: {
        var s1: Arr = undefined;
        var s2: Arr = undefined;
        if (steel) {
            const sc = try hy4_chain.steelScores(o.s, q_lat, q_pe, c_g, pe_g_bf);
            s1 = try o.own(sc.s1);
            s2 = try o.own(sc.s2);
        } else {
            s1 = try o.binary(.mm, q_lat, c_g_t);
            s2 = try o.binary(.mm, q_pe, pe_g_t);
        }
        const p = if (chain_kernels_on and hy4_chain.sinkSoftmaxMaskApplies(o.s, s1, s2, keep, w.sinks, false))
            try o.own(try hy4_chain.sinkSoftmaxMask(o.s, s1, s2, keep, w.sinks, scaleOf(g), false))
        else inner: {
            const scores = try o.binary(.add, s1, s2);
            const mask = try o.reshape(try additive(o, keep), &.{ t, 1, sel });
            break :inner try sinkSoftmax(o, try o.binary(.add, try o.binary(.mul, scores, try o.scalar(scaleOf(g), .float32)), mask), w.sinks, g.heads, t, sel, false);
        };
        if (steel and hy4_chain.steelValuesApplies(o.s, p, c_g)) break :blk try o.transpose(try o.own(try hy4_chain.steelValues(o.s, p, c_g)), &.{ 1, 0, 2 });
        break :blk try o.transpose(try o.binary(.mm, p, c_g), &.{ 1, 0, 2 });
    };
}

/// Whether a GPU stream runs a wide span's attention core (scores, sink softmax, value sum) as one launch on the matrix
/// units (`hy4_attn`) rather than as the op chain. A test seam: the two agree to f32 rounding, not byte for byte.
pub var core_kernel_on = true;

/// Whether `hy4_attn` serves a span of `rows` rows of a layer with `w` and `g`: a wide span only, so a decode row and a
/// verify block (2..8 rows) never reach it and keep the chain's bytes.
pub fn coreServes(s: mlx.mlx_stream, w: Attn, g: AttnGeometry, rows: c_int) bool {
    return core_kernel_on and rows > verify_rows_max and hy4_attn.fits(s, g.heads, g.latent, g.rope, w.sinks);
}

/// Where a wide span's keys are, as `hy4_attn` reads them: rows over a range of key rows, rows over their own picks of
/// the layer's rows, or a span that straddles the selection threshold: its first `cut` rows over a range, the rest over
/// their picks.
pub const CoreKeys = union(enum) {
    range: CoreRange,
    picks: CorePicks,
    straddle: struct { cut: c_int, range: CoreRange, picks: CorePicks },
};

/// Row `i` of the span (position `p0 + i` among `keys`) reads `keys` from `max(0, pos - window + 1)` to `pos`, or from
/// the first when `window` is 0.
pub const CoreRange = struct { keys: hy4_attn.Keys, p0: c_int, window: c_int };

/// Row `i` reads `keys` at `rows[i, :]` where `keep[i, :]` (`recalledRows`' rows and keep mask).
pub const CorePicks = struct { keys: hy4_attn.Keys, rows: Arr, keep: Arr };

/// `attendGated` with the core as `hy4_attn`'s launch over `core`'s keys.
pub fn attendCoreGated(o: *Ops, in: SpanIn, q: Arr, core: CoreKeys, w: Attn, g: AttnGeometry, at: ?At) !Arr {
    const ab = try absorb(o, q, w, g, in.kv_b32, in.rows_as, in.q_raw);
    const u_lat = try coreLatent(o, ab, core, w, g);
    return gatedOut(o, in.gate, u_lat, ab.w_v, w, g, at, in.whole, in.o_left, in.dst, in.rows_as);
}

/// The attended latent [H, T, latent] of the absorbed query `ab` over `core`.
fn coreLatent(o: *Ops, ab: Absorbed, core: CoreKeys, w: Attn, g: AttnGeometry) !Arr {
    const q: hy4_attn.Query = .{ .lat = ab.q_lat, .pe = ab.q_pe };
    const t = mlx.getShape(ab.q_lat)[1];
    const scale = scaleOf(g);
    switch (core) {
        inline .range, .picks => |c| if (!hy4_attn.applies(o.s, q, c.keys, w.sinks)) return error.Hy4CoreRefused,
        .straddle => |sg| if (!hy4_attn.applies(o.s, q, sg.range.keys, w.sinks) or !hy4_attn.applies(o.s, q, sg.picks.keys, w.sinks)) return error.Hy4CoreRefused,
    }
    switch (core) {
        .range => |r| return o.own(try hy4_attn.range(o.s, q, r.keys, w.sinks, scale, r.p0, r.window, 0, t)),
        .picks => |p| return o.own(try hy4_attn.picks(o.s, q, p.keys, p.rows, p.keep, w.sinks, scale, 0)),
        .straddle => |sg| {
            const dense = try o.own(try hy4_attn.range(o.s, q, sg.range.keys, w.sinks, scale, sg.range.p0, sg.range.window, 0, sg.cut));
            const picked = try o.own(try hy4_attn.picks(o.s, q, sg.picks.keys, sg.picks.rows, sg.picks.keep, w.sinks, scale, sg.cut));
            return o.concat(&.{ dense, picked }, 1);
        },
    }
}

/// The key rows a row reads past the selection threshold (`recalledRows`): the second axis of its rows or keep mask.
pub fn selectedWidth(rows: Arr) c_int {
    return mlx.getShape(rows)[1];
}

fn scaleOf(g: AttnGeometry) f32 {
    return std.math.pow(f32, @floatFromInt(g.nope + g.rope), -0.5);
}

/// `w_v` is null where a one-row forward reads `kv_b` as stored (`hy4_absorb`): `finish` then does too.
const Absorbed = struct { q_lat: Arr, q_pe: Arr, w_v: ?Arr };

/// `q` [H, T, qk] split: its nope part taken into the latent through `W_k` [H, nope, latent], its rope part kept; and
/// `W_v` [H, v, latent], both from `kv_b` [H·(nope+v), latent] widened to f32 (`kv_b32` when the caller widened it).
fn absorb(o: *Ops, q: Arr, w: Attn, g: AttnGeometry, kv_b32: ?Arr, rows_as: ?c_int, q_raw: ?Arr) !Absorbed {
    if (absorbServes(o.s, w.kv_b, g, mlx.getShape(q)[1])) {
        // The absorb kernel reads the query's nope part in place.
        const q32 = try o.contiguous(try o.cast(q, .float32));
        return .{ .q_lat = try o.own(try hy4_absorb.queryLatent(o.s, q_raw orelse q32, w.kv_b, absorbGeometry(g))), .q_pe = try o.slice(q32, 2, g.nope, g.nope + g.rope), .w_v = null };
    }
    const kvb = try o.reshape(kv_b32 orelse try o.cast(w.kv_b, .float32), &.{ g.heads, g.nope + g.v, g.latent });
    const w_k = try o.slice(kvb, 1, 0, g.nope);
    const w_v = try o.slice(kvb, 1, g.nope, g.nope + g.v);
    const q32 = try o.cast(q, .float32);
    return .{ .q_lat = try mmHeadsAs(o, try o.slice(q32, 2, 0, g.nope), w_k, rows_as), .q_pe = try o.slice(q32, 2, g.nope, g.nope + g.rope), .w_v = w_v };
}

/// Softmax over the last axis (`n` keys) with each head's sink logit beside them, the sink's share dropped after.
/// `logits` is [H, T, n] when `heads_first`, else [T, H, n].
fn sinkSoftmax(o: *Ops, logits: Arr, sinks: Arr, heads: c_int, t: c_int, n: c_int, heads_first: bool) !Arr {
    const shape: [3]c_int = if (heads_first) .{ heads, 1, 1 } else .{ 1, heads, 1 };
    const full: [3]c_int = if (heads_first) .{ heads, t, 1 } else .{ t, heads, 1 };
    const sink = try o.broadcast(try o.reshape(try o.cast(sinks, .float32), &shape), &full);
    return o.slice(try o.softmax(try o.concat(&.{ logits, sink }, 2), 2), 2, 0, n);
}

fn absorbGeometry(g: AttnGeometry) hy4_absorb.Geometry {
    return .{ .heads = g.heads, .nope = g.nope, .v = g.v, .latent = g.latent };
}

pub fn absorbServes(s: mlx.mlx_stream, kv_b: Arr, g: AttnGeometry, rows: c_int) bool {
    return absorb_kernels_on and hy4_absorb.applies(s, kv_b, absorbGeometry(g), rows);
}

/// The attended latent [H, T, latent] out through `W_v`, gated by `σ` of its rows' `gate` (as stored), as `W_o` reads it.
fn gatedOut(o: *Ops, gate_rows: Arr, u_lat: Arr, w_v: ?Arr, w: Attn, g: AttnGeometry, at: ?At, whole: bool, o_left: ?*OLeft, dst: ?GatedDst, rows_as: ?c_int) !Arr {
    const t = mlx.getShape(u_lat)[1];
    // A span that writes into its chunk's block keeps that write; the joined launch returns rows only.
    const joined = dst == null and w_v == null and chain_kernels_on and hy4_absorb.valueOutGatedApplies(o.s, u_lat, w.kv_b, absorbGeometry(g), gate_rows);
    const u = if (joined) u_lat else if (w_v) |wv| try mmHeadsAs(o, u_lat, try o.transpose(wv, &.{ 0, 2, 1 }), rows_as) else try o.own(try hy4_absorb.valueOut(o.s, u_lat, w.kv_b, absorbGeometry(g))); // [H, T, v]
    if (!joined and at == null and !whole) if (dst) |d| if (chain_kernels_on and hy4_chain.gatedIntoApplies(o.s, u, gate_rows, d.block)) {
        try hy4_chain.gatedInto(o.s, u, gate_rows, d.block, d.row);
        return mlx.mlx_array_new();
    };
    const gated = if (joined) try o.own(try hy4_absorb.valueOutGated(o.s, u_lat, w.kv_b, absorbGeometry(g), gate_rows)) else if (chain_kernels_on and hy4_chain.gatedApplies(o.s, u, gate_rows)) try o.own(try hy4_chain.gated(o.s, u, gate_rows)) else blk: {
        const gate = try o.unary(.sigmoid, try o.transpose(try o.reshape(try o.cast(gate_rows, .float32), &.{ t, g.heads, g.v }), &.{ 1, 0, 2 }));
        break :blk try o.reshape(try o.transpose(try o.binary(.mul, u, gate), &.{ 1, 0, 2 }), &.{ t, g.heads * g.v });
    };
    try At.input(at, .o_in, gated);
    // The chunk's only span: W_o and its norm in one launch here, the bytes `output` gives its rows.
    if (whole) {
        if (o_left) |ol| if (w.o.stq) |b| {
            if (o_rows_on and t <= o_rows_max) if (try hy4_stq.applyFinished(o, gated, b)) |y| {
                ol.* = .rows;
                return y;
            };
            if (try hy4_stq.applyParts(o, gated, b)) |p| {
                ol.* = .{ .parts = p.nsplit };
                return p.part;
            }
        };
        return linearNormed(o, gated, w.o, w.post_norm, g.eps);
    }
    return linearInput(o, gated, w.o);
}

/// The gated attention rows (`attendGated`, as `W_o` reads them) through `W_o` and `post_proj_norm`, in f32; with `span`
/// each row as its own attention span's launch would give it (`linearSpans`).
pub fn output(o: *Ops, gated: Arr, w: Attn, g: AttnGeometry, span: ?c_int) !Arr {
    const y = if (span) |r| blk: {
        const s = try linearSpans(o, gated, w.o, r);
        break :blk if (norm_kernel_on and mlx.streamIsGpu(o.s)) s else try o.cast(s, .float32);
    } else try linear(o, gated, w.o);
    return norm(o, y, w.post_norm, g.eps);
}

/// The key rows each of rows `p0..p0+T` reads past the selection threshold: its recalled groups' `ratio` rows each
/// (`recalled` [T, k] group ids, the row's own group last), clamped into `0..N` for the gather, and whether each is one
/// the row reads: at or before its own position, and from an earlier group or the own group's own column. A row with
/// fewer earlier groups than picks picks later ones too, its own among them: read once, not twice.
pub fn recalledRows(o: *Ops, t: c_int, n: c_int, p0: c_int, recalled: Arr, ratio: c_int) !struct { rows: Arr, keep: Arr } {
    if (chain_kernels_on and mlx.streamIsGpu(o.s) and mlx.mlx_array_dtype(recalled) == .float32) {
        const r = try hy4_chain.recalledRows(o.s, recalled, p0, n, ratio);
        return .{ .rows = try o.own(r.rows), .keep = try o.own(r.keep) };
    }
    const k = mlx.getShape(recalled)[1];
    const groups = try o.cast(recalled, .float32);
    const own_group = try o.slice(groups, 1, k - 1, k); // [T, 1]
    const is_own_col = try o.binary(.le, try o.scalar(@floatFromInt(k - 1), .float32), try o.reshape(try arange(o, k), &.{ 1, k }));
    const pick_ok = try o.binary(.lor, try o.binary(.less, groups, own_group), is_own_col); // [T, k]
    const base = try o.binary(.mul, try o.reshape(groups, &.{ t, k, 1 }), try o.scalar(@floatFromInt(ratio), .float32));
    const pos = try o.reshape(try o.binary(.add, base, try o.reshape(try arange(o, ratio), &.{ 1, 1, ratio })), &.{ t, k * ratio }); // [T, S]
    const own = try o.reshape(try arangeFrom(o, p0, t), &.{ t, 1 });
    const pick_rows = try o.reshape(try o.broadcast(try o.reshape(pick_ok, &.{ t, k, 1 }), &.{ t, k, ratio }), &.{ t, k * ratio });
    const keep = try o.binary(.land, try o.binary(.le, pos, own), pick_rows);
    const clamped = try o.binary(.min, pos, try o.scalar(@floatFromInt(n - 1), .float32));
    return .{ .rows = try o.cast(clamped, .uint32), .keep = keep };
}

/// The additive mask [T, N] (0 kept, -inf dropped) of rows at positions `p0..p0+T` over keys at `0..N`: causal, and when
/// `window` > 0 only the `window` keys ending at the query's own (`j > p - window`).
pub fn causalMask(o: *Ops, t: c_int, n: c_int, p0: c_int, window: c_int) !Arr {
    const rows = try o.reshape(try arangeFrom(o, p0, t), &.{ t, 1 });
    const cols = try o.reshape(try arange(o, n), &.{ 1, n });
    var keep = try o.binary(.le, cols, rows);
    if (window > 0) {
        const near = try o.slot();
        try mlx.check(mlx.mlx_greater(near, cols, try o.binary(.sub, rows, try o.scalar(@floatFromInt(window), .float32)), o.s));
        keep = try o.binary(.land, keep, near.*);
    }
    return additive(o, keep);
}

/// Causal ∧ (key j's group, `j / ratio`, is among row i's recalled groups `recalled` [T, k]): the indexer's
/// selection, rows at `p0..p0+T` over keys at `0..N`.
pub fn selectedMask(o: *Ops, t: c_int, n: c_int, p0: c_int, recalled: Arr, ratio: c_int) !Arr {
    const k = mlx.getShape(recalled)[1];
    const groups = try o.reshape(try o.binary(.floor_div, try arange(o, n), try o.scalar(@floatFromInt(ratio), .float32)), &.{ 1, 1, n });
    const eq = try o.slot();
    try mlx.check(mlx.mlx_equal(eq, try o.reshape(try o.cast(recalled, .float32), &.{ t, k, 1 }), groups, o.s));
    const any = try o.slot();
    try mlx.check(mlx.mlx_greater(any, try o.reduce(try o.cast(eq.*, .float32), 1, false, false), try o.scalar(0, .float32), o.s));
    const rows = try o.reshape(try arangeFrom(o, p0, t), &.{ t, 1 });
    const cols = try o.reshape(try arange(o, n), &.{ 1, n });
    return additive(o, try o.binary(.land, any.*, try o.binary(.le, cols, rows)));
}

fn arange(o: *Ops, n: c_int) !Arr {
    return arangeFrom(o, 0, n);
}

fn arangeFrom(o: *Ops, start: c_int, n: c_int) !Arr {
    const out = try o.slot();
    try mlx.check(mlx.mlx_arange(out, @floatFromInt(start), @floatFromInt(start + n), 1, .float32, o.s));
    return out.*;
}

pub fn additive(o: *Ops, keep: Arr) !Arr {
    const out = try o.slot();
    try mlx.check(mlx.mlx_where(out, keep, try o.scalar(0, .float32), try o.scalar(-std.math.inf(f32), .float32), o.s));
    return out.*;
}

// ── the compressed indexer ──────────────────────────────────────────────────────────────────────────────────────────

/// The indexer's tensors (`self_attn.compressed_indexer.*`): `query_projection` [H·d, q_lora], `head_weight_projection`
/// [H, D], `token_key_projection` [d, D], `token_key_norm` [d], the f32 pooling `key_value_projection` and
/// `key_gate_projection` [d, d], `intra_group_ape` [ratio, d], `key_norm` [d].
pub const Indexer = struct { query: Arr, head_weight: Arr, token_key: Arr, token_key_norm: Arr, key_value: Arr, key_gate: Arr, ape: Arr, key_norm: Arr };

pub const IndexerGeometry = struct { heads: c_int, dim: c_int, rope: c_int, ratio: c_int, group_topk: c_int, theta: f32, eps: f32 };

/// The token keys `RMS(W_k·x)` [T, d] of the rows `x`, which the layer keeps until their group closes.
pub fn tokenKeys(o: *Ops, x: Arr, w: Indexer, g: IndexerGeometry) !Arr {
    return norm(o, try linearF32(o, x, w.token_key), w.token_key_norm, g.eps);
}

/// The pooled key of each complete group of `ratio` token keys (`pool_groups`): per group and channel
/// `K = Σ_slot softmax_slot(W_g·k + ape) ⊙ (W_v·k)`, then `RMS` with `key_norm`. [G, d] for the `G = N / ratio`
/// complete groups of `keys` [N, d].
pub fn pool(o: *Ops, keys: Arr, w: Indexer, g: IndexerGeometry) !Arr {
    const n = mlx.getShape(keys)[0];
    const groups = @divTrunc(n, g.ratio);
    const grouped = try o.reshape(try o.slice(keys, 0, 0, groups * g.ratio), &.{ groups, g.ratio, g.dim });
    const values = try mmByRow(o, grouped, try o.transpose(try o.cast(w.key_value, .float32), &.{ 1, 0 }), null);
    const gates = try o.binary(.add, try mmByRow(o, grouped, try o.transpose(try o.cast(w.key_gate, .float32), &.{ 1, 0 }), null), try o.cast(w.ape, .float32));
    const pooled = try o.reduce(try o.binary(.mul, values, try o.softmax(gates, 1)), 1, false, false);
    return norm(o, pooled, w.key_norm, g.eps);
}

/// `pool` over the token keys of the rows `x` themselves.
pub fn pooledKeys(o: *Ops, x: Arr, w: Indexer, g: IndexerGeometry) !Arr {
    return pool(o, try tokenKeys(o, x, w, g), w, g);
}

/// What the indexer gives: `groups` [T, group_topk] (f32 ids) and the `logits` [T, G] they were picked by (-inf at and
/// past each row's own group). ReLU leaves exact-zero scores, so ties are common: a pick is held to the reference's
/// margins (equal logits), never to an index order.
pub const Recall = struct { groups: Arr, logits: Arr };

pub fn recall(o: *Ops, x: Arr, q_resid: Arr, pooled: Arr, w: Indexer, g: IndexerGeometry) !Recall {
    return recallAt(o, x, q_resid, pooled, w, g, 0);
}

/// `recall` for rows at positions `p0..p0+T`.
pub fn recallAt(o: *Ops, x: Arr, q_resid: Arr, pooled: Arr, w: Indexer, g: IndexerGeometry, p0: c_int) !Recall {
    const t = mlx.getShape(x)[0];
    const groups = mlx.getShape(pooled)[0];
    const q_rows = try linearF32(o, q_resid, w.query);
    const scale = std.math.pow(f32, @floatFromInt(g.heads * g.dim), -0.5);
    const take = @min(g.group_topk - 1, groups);
    if (select_kernels_on and mlx.streamIsGpu(o.s) and take > 0) {
        const logits = try indexLogits(o, x, q_rows, pooled, w, g, p0);
        const order = try o.own(try sparse.topkIndices(o.s, logits, @intCast(take)));
        if (chain_kernels_on) return .{ .groups = try o.own(try hy4_chain.recallGroups(o.s, order, p0, g.ratio)), .logits = logits };
        const own = try o.reshape(try o.binary(.floor_div, try arangeFrom(o, p0, t), try o.scalar(@floatFromInt(g.ratio), .float32)), &.{ t, 1 });
        return .{ .groups = try o.concat(&.{ try o.cast(order, .float32), own }, 1), .logits = logits };
    }
    const q = try rope(o, try o.transpose(try o.reshape(q_rows, &.{ t, g.heads, g.dim }), &.{ 1, 0, 2 }), g.rope, g.theta, p0);
    const k = try rotateGroupKeys(o, pooled, g);
    const relu = try o.binary(.max, try o.binary(.mm, q, try o.transpose(try o.cast(k, .float32), &.{ 1, 0 })), try o.scalar(0, .float32)); // [H, T, G]
    const hw = try o.binary(.mul, try linearF32(o, x, w.head_weight), try o.scalar(scale, .float32)); // [T, H]
    const logits = try o.reduce(try o.binary(.mul, relu, try o.reshape(try o.transpose(hw, &.{ 1, 0 }), &.{ g.heads, t, 1 })), 0, false, false); // [T, G]
    const own = try o.reshape(try o.binary(.floor_div, try arangeFrom(o, p0, t), try o.scalar(@floatFromInt(g.ratio), .float32)), &.{ t, 1 });
    const later = try o.slot();
    try mlx.check(mlx.mlx_greater_equal(later, try o.reshape(try arange(o, groups), &.{ 1, groups }), own, o.s));
    const masked = try o.slot();
    try mlx.check(mlx.mlx_where(masked, later.*, try o.scalar(-std.math.inf(f32), .float32), logits, o.s));
    const order = try o.slot();
    try mlx.check(mlx.mlx_argpartition_axis(order, try o.unary(.negative, masked.*), take - 1, -1, o.s));
    return .{ .groups = try o.concat(&.{ try o.cast(try o.slice(order.*, 1, 0, take), .float32), own }, 1), .logits = masked.* };
}

/// `recallAt`'s scores on a GPU stream over the query projection's rows `q_rows` [T, H·d] f32.
fn indexLogits(o: *Ops, x: Arr, q_rows: Arr, pooled: Arr, w: Indexer, g: IndexerGeometry, p0: c_int) !Arr {
    const t = mlx.getShape(x)[0];
    const scale = std.math.pow(f32, @floatFromInt(g.heads * g.dim), -0.5);
    const hw = try o.cast(try linearF32(o, x, w.head_weight), .float32);
    // The query's and the keys' rotations ride the score launch.
    if (index_kernel_on and hy4_index.scoresApply(o.s, q_rows, pooled, g.heads, g.rope))
        return o.own(try hy4_index.scores(o.s, try o.contiguous(q_rows), try o.contiguous(pooled), try o.contiguous(hw), g.heads, g.rope, g.theta, scale, p0, g.ratio));
    const q = try rope(o, try o.transpose(try o.reshape(q_rows, &.{ t, g.heads, g.dim }), &.{ 1, 0, 2 }), g.rope, g.theta, p0);
    // The kernel counts the groups row l sees as (meta0 + l + 1) / ratio: from p0 - 1 that is the row's own group,
    // which the indexer never scores (it is forced in below).
    const qt = try o.contiguous(try o.transpose(try o.cast(q, .float32), &.{ 1, 0, 2 }));
    const k_rot = try rotateGroupKeys(o, pooled, g);
    if (index_kernel_on and hy4_index.wideScoresApply(o.s, qt, k_rot, hw))
        return o.own(try hy4_index.wideScores(o.s, qt, k_rot, hw, scale, p0 - 1, @intCast(g.ratio)));
    return o.own(try sparse.indexScores(o.s, qt, k_rot, hw, scale, p0 - 1, @intCast(g.ratio)));
}

/// Whether the selection of the rows at `p0..p0+t` splits its top-k over the groups closed by then
/// (`sparse.topkSplits`), which takes passes and scratch of its own: a fact of a forward's structure.
pub fn selectionSplits(g: IndexerGeometry, p0: c_int, t: c_int) bool {
    const groups = @divFloor(p0 + t, g.ratio);
    const take = @min(g.group_topk - 1, groups);
    return take > 0 and sparse.topkSplits(@intCast(t), @intCast(groups), @intCast(take)) > 1;
}

/// The key rows [T, S] uint32 rows at positions `p0..` read past the selection over `n` keys, and their keep mask:
/// `recalledRows` of `recallAt`'s groups. On a GPU stream the top-k launch writes them (`hy4_index.pick`).
pub const Recalled = struct { rows: Arr, keep: Arr };

pub fn recalledAt(o: *Ops, x: Arr, q_resid: Arr, pooled: Arr, w: Indexer, g: IndexerGeometry, p0: c_int, n: c_int) !Recalled {
    const t = mlx.getShape(x)[0];
    const take = @min(g.group_topk - 1, mlx.getShape(pooled)[0]);
    if (index_kernel_on and chain_kernels_on and select_kernels_on and mlx.streamIsGpu(o.s) and take > 0) {
        const logits = try indexLogits(o, x, try linearF32(o, q_resid, w.query), pooled, w, g, p0);
        if (hy4_index.pickApplies(o.s, logits, take)) {
            const r = try hy4_index.pick(o.s, logits, take, p0, n, g.ratio);
            _ = try o.own(r.groups);
            return .{ .rows = try o.own(r.rows), .keep = try o.own(r.keep) };
        }
        const order = try o.own(try sparse.topkIndices(o.s, logits, @intCast(take)));
        const sel = try recalledRows(o, t, n, p0, try o.own(try hy4_chain.recallGroups(o.s, order, p0, g.ratio)), g.ratio);
        return .{ .rows = sel.rows, .keep = sel.keep };
    }
    const rec = try recallAt(o, x, q_resid, pooled, w, g, p0);
    const sel = try recalledRows(o, t, n, p0, rec.groups, g.ratio);
    return .{ .rows = sel.rows, .keep = sel.keep };
}

/// The indexer's query and head-weight projections, rope and group keys once for every span of a chunk from `r0`, or
/// null where the spans run their own (`recalledAt`).
pub fn indexPrepare(o: *Ops, x: Arr, q_resid: Arr, pooled: Arr, w: Indexer, g: IndexerGeometry, p0: c_int, r0: c_int, span: c_int) !?IndexIn {
    const t = mlx.getShape(x)[0];
    const take = @min(g.group_topk - 1, mlx.getShape(pooled)[0]);
    if (!spans_share or !index_kernel_on or !chain_kernels_on or !select_kernels_on or !rope_kernel_on) return null;
    if (!mlx.streamIsGpu(o.s) or !mlx.engine.isStream(o.s) or take <= 0 or span <= verify_rows_max or t - r0 <= span) return null;
    const rows = t - r0;
    const xs = if (r0 == 0) x else try o.slice(x, 0, r0, t);
    const qs = if (r0 == 0) q_resid else try o.slice(q_resid, 0, r0, t);
    const q_rows = try linearF32Spans(o, qs, w.query, span);
    const hw = try o.cast(try linearF32Spans(o, xs, w.head_weight, span), .float32);
    const q_in = try o.transpose(try o.reshape(q_rows, &.{ rows, g.heads, g.dim }), &.{ 1, 0, 2 });
    const qt = if (hy4_rope.applies(o.s, q_in, g.rope))
        try o.own(try hy4_rope.tailRows(o.s, q_in, g.rope, g.theta, p0 + r0, 1, .float32))
    else
        try o.contiguous(try o.transpose(try o.cast(try rope(o, q_in, g.rope, g.theta, p0 + r0), .float32), &.{ 1, 0, 2 }));
    const k_rot = try rotateGroupKeys(o, pooled, g);
    return .{ .r0 = r0, .qt = qt, .hw = hw, .k_rot = k_rot };
}

/// `recalledAt` for rows `r0..r1` over the inputs `indexPrepare` made: the same picks, read from the shared inputs.
pub fn recalledPrepared(o: *Ops, idx: IndexIn, r0: c_int, r1: c_int, pooled_groups: c_int, g: IndexerGeometry, p0: c_int, n: c_int) !Recalled {
    const t = r1 - r0;
    const s0 = r0 - idx.r0;
    const s1 = r1 - idx.r0;
    const take = @min(g.group_topk - 1, pooled_groups);
    const scale = std.math.pow(f32, @floatFromInt(g.heads * g.dim), -0.5);
    const qt = try o.slice(idx.qt, 0, s0, s1);
    const hw = try o.slice(idx.hw, 0, s0, s1);
    const logits = if (hy4_index.wideScoresApply(o.s, qt, idx.k_rot, hw))
        try o.own(try hy4_index.wideScores(o.s, qt, idx.k_rot, hw, scale, p0 - 1, @intCast(g.ratio)))
    else
        try o.own(try sparse.indexScores(o.s, qt, idx.k_rot, hw, scale, p0 - 1, @intCast(g.ratio)));
    if (hy4_index.pickApplies(o.s, logits, take)) {
        const r = try hy4_index.pick(o.s, logits, take, p0, n, g.ratio);
        _ = try o.own(r.groups);
        return .{ .rows = try o.own(r.rows), .keep = try o.own(r.keep) };
    }
    const order = try o.own(try sparse.topkIndices(o.s, logits, @intCast(take)));
    const sel = try recalledRows(o, t, n, p0, try o.own(try hy4_chain.recallGroups(o.s, order, p0, g.ratio)), g.ratio);
    return .{ .rows = sel.rows, .keep = sel.keep };
}

/// Whether a GPU stream runs an attention's latent norm and its key's rotation as one launch (`hy4_front`) rather than
/// the norm's and the rope's. A test seam: both give the same bytes.
pub var front_kernel_on = true;

/// Whether a GPU stream scores the indexer's groups through `hy4_index.scores`, the rotations in its launch, rather than
/// the two rope launches and `sparse.indexScores`, and picks them with their key rows in one launch (`hy4_index.pick`).
/// A test seam: both give the same bytes.
pub var index_kernel_on = true;

/// Pooled keys [G, d] rotated at their groups' first positions (`g · ratio`), on the trailing `rope` dims.
fn rotateGroupKeys(o: *Ops, pooled: Arr, g: IndexerGeometry) !Arr {
    const groups = mlx.getShape(pooled)[0];
    return o.reshape(try ropeScaled(o, try o.reshape(pooled, &.{ 1, groups, g.dim }), g.rope, g.theta, 0, g.ratio), &.{ groups, g.dim });
}

// ── Engram ──────────────────────────────────────────────────────────────────────────────────────────────────────────

/// A module's tensors past its table and token map (`engram_layer.*`): the per-head norms (one weight a hash head),
/// and per hyper-connection stream a key projection with bias, key norm, query norm and mix scale; the value
/// projection with bias and `unified_pre_gamma`.
pub const Engram = struct {
    head_norms: []const Arr,
    key_w: []const Arr,
    key_b: []const Arr,
    norm_key: []const Arr,
    norm_query: []const Arr,
    scales: []const Arr,
    value_w: Arr,
    value_b: Arr,
    gamma: Arr,
};

/// The f32 machine epsilon: Engram's norms (llama.cpp; transformers leaves them unset, which torch resolves from the
/// input dtype).
pub const engram_eps: f32 = std.math.floatEps(f32);

/// Engram's delta to the `n` streams `h` [T, n, D] from its `rows` [T, heads, hd] (the table rows of each position's
/// hash heads, as stored, widened), zero where `live` [T] (1 or 0) is 0, an image, video or pad id. Per head
/// `RMSNorm(r_j)`, concatenated to `u`; per stream `k_c = RMSNorm(W_c·u + b_c)`, `d_c = <k_c, RMSNorm(h_c)> / √D`, the
/// gate `σ(sign(d)·√max(|d|, 1e-6))`; `y = γ ⊙ RMS(W_v·u + b_v)`, and the delta `(gate_c + scale_c) ⊙ y`.
pub fn engram(o: *Ops, h: Arr, rows: Arr, live: Arr, w: Engram, at: ?At) !Arr {
    const hs = mlx.getShape(h);
    const t = hs[0];
    const d = hs[2];
    const rs = mlx.getShape(rows);
    var heads: [16]Arr = undefined;
    for (w.head_norms, 0..) |nw, j| {
        const r = try o.reshape(try o.slice(rows, 1, @intCast(j), @intCast(j + 1)), &.{ t, rs[2] });
        heads[j] = try norm(o, r, nw, engram_eps);
    }
    const u = try o.concat(heads[0..w.head_norms.len], 1);
    try At.input(at, .engram_in, u);
    const v = try o.binary(.add, try linearF32(o, u, w.value_w), try o.cast(w.value_b, .float32));
    const y = try o.binary(.mul, try o.binary(.mul, v, try rowRsqrt(o, v, engram_eps)), try o.cast(w.gamma, .float32));
    const inv_sqrt_d = try o.scalar(1.0 / @sqrt(@as(f32, @floatFromInt(d))), .float32);
    var streams: [8]Arr = undefined;
    for (0..w.key_w.len) |c| {
        const k = try norm(o, try o.binary(.add, try linearF32(o, u, w.key_w[c]), try o.cast(w.key_b[c], .float32)), w.norm_key[c], engram_eps);
        const hc = try o.reshape(try o.slice(h, 1, @intCast(c), @intCast(c + 1)), &.{ t, d });
        const q = try norm(o, hc, w.norm_query[c], engram_eps);
        const dot = try o.binary(.mul, try o.reduce(try o.binary(.mul, k, q), -1, false, true), inv_sqrt_d);
        const mag = try unaryOp(o, mlx.mlx_sqrt, try o.binary(.max, try unaryOp(o, mlx.mlx_abs, dot), try o.scalar(1e-6, .float32)));
        const gate = try o.unary(.sigmoid, try o.binary(.mul, mag, try unaryOp(o, mlx.mlx_sign, dot)));
        const delta = try o.binary(.mul, try o.binary(.add, gate, try o.cast(w.scales[c], .float32)), y);
        streams[c] = try o.reshape(delta, &.{ t, 1, d });
    }
    const all = try o.concat(streams[0..w.key_w.len], 1);
    return o.binary(.mul, all, try o.reshape(try o.cast(live, .float32), &.{ t, 1, 1 }));
}

/// The streams after Engram, `h` widened plus `engram(h, rows, live, w)`: on a GPU stream the head norms, the value's
/// bias and square, and the gates' write into the streams are a launch each (`hy4_engram_gate`), the key and query
/// norms `hy4_norm`'s, with the ops' bytes.
pub fn engramAdd(o: *Ops, h: Arr, rows: Arr, live: Arr, w: Engram, at: ?At) !Arr {
    const heads = w.head_norms.len;
    var same = true;
    for (w.head_norms) |nw| same = same and mlx.mlx_array_dtype(nw) == mlx.mlx_array_dtype(w.head_norms[0]);
    if (!chain_kernels_on or !norm_kernel_on or !same or !hy4_engram_gate.applies(o.s, h, rows, heads, w.key_w.len))
        return o.binary(.add, try o.cast(h, .float32), try engram(o, h, rows, live, w, at));
    const t = mlx.getShape(h)[0];
    const d = mlx.getShape(h)[2];
    const u = try o.own(try hy4_engram_gate.headNorms(o.s, rows, w.head_norms, engram_eps));
    try At.input(at, .engram_in, u);
    const vs = try hy4_engram_gate.valueSq(o.s, try linearF32(o, u, w.value_w), w.value_b);
    const v = try o.own(vs.v);
    const vsum = try o.reduce(try o.own(vs.sq), -1, false, true);
    var dsums: [8]Arr = undefined;
    for (0..w.key_w.len) |c| {
        const bias = try o.reshape(w.key_b[c], &.{ 1, d });
        const k = try o.own(try hy4_norm.rows(o.s, try linearF32(o, u, w.key_w[c]), bias, w.norm_key[c], engram_eps, .float32));
        const hc = try o.reshape(try o.slice(h, 1, @intCast(c), @intCast(c + 1)), &.{ t, d });
        const kq = try o.own(try hy4_norm.rowsTimes(o.s, hc, null, k, w.norm_query[c], engram_eps, .float32));
        dsums[c] = try o.reduce(kq, -1, false, true);
    }
    return o.own(try hy4_engram_gate.delta(o.s, h, v, vsum, w.gamma, dsums[0..w.key_w.len], w.scales, try o.cast(live, .float32), engram_eps));
}

fn unaryOp(o: *Ops, comptime f: anytype, x: Arr) !Arr {
    const out = try o.slot();
    try mlx.check(f(out, x, o.s));
    return out.*;
}

// ── tests ───────────────────────────────────────────────────────────────────────────────────────────────────────────

/// `x` rounded to BF16, to nearest even.
fn bf16Round(x: f32) f32 {
    const u: u32 = @bitCast(x);
    return @bitCast((u +% (0x7fff + ((u >> 16) & 1))) & 0xffff0000);
}

/// One BF16 unit in the last place of `x`.
fn bf16Ulp(x: f32) f64 {
    const e: i32 = @as(i32, @intCast((@as(u32, @bitCast(@abs(x))) >> 23) & 0xff)) - 127;
    return std.math.pow(f64, 2, @floatFromInt(@max(e, -126) - 7));
}

/// Banks, a latent, weights and picks for `routedCase`, the bank values already BF16.
const RoutedCase = struct {
    const t = 6;
    const k = 4;
    const l = 64;
    const inter = 32;
    const slots = 4;
    const limit: f32 = 0.5;
    lat: [t * l]f32,
    w: [t * k]f32,
    gu: [slots * 2 * inter * l]f32,
    down: [slots * l * inter]f32,
    rows: [t * k]u32,

    fn init() RoutedCase {
        var prng = std.Random.DefaultPrng.init(0x4879_3462);
        const r = prng.random();
        var c: RoutedCase = undefined;
        for (&c.lat) |*v| v.* = r.floatNorm(f32);
        for (&c.w) |*v| v.* = 0.25 + r.float(f32);
        for (&c.gu) |*v| v.* = bf16Round(0.2 * r.floatNorm(f32));
        for (&c.down) |*v| v.* = bf16Round(0.2 * r.floatNorm(f32));
        for (&c.rows) |*v| v.* = r.uintLessThan(u32, slots);
        return c;
    }

    /// Row-major [t, l]: each pick's clamped SwiGLU in f64, rounded where MLX's BF16 product rounds (its operands and
    /// outputs), weighted and summed over the row's k in f32; `unit` sums each pick's weighted BF16 unit.
    fn reference(c: *const RoutedCase, out: *[t * l]f32, unit: *[t * l]f64) void {
        @memset(out, 0);
        @memset(unit, 0);
        for (0..t) |row| for (0..k) |j| {
            const e = c.rows[row * k + j];
            var h: [inter]f32 = undefined;
            for (0..inter) |i| {
                var gate: f64 = 0;
                var up: f64 = 0;
                for (0..l) |d| {
                    const x: f64 = bf16Round(c.lat[row * l + d]);
                    gate += x * c.gu[(e * 2 * inter + i) * l + d];
                    up += x * c.gu[(e * 2 * inter + inter + i) * l + d];
                }
                const g: f64 = @min(bf16Round(@floatCast(gate)), limit);
                const u: f64 = std.math.clamp(bf16Round(@floatCast(up)), -limit, limit);
                h[i] = bf16Round(@floatCast(g / (1 + @exp(-g)) * u));
            }
            for (0..l) |d| {
                var y: f64 = 0;
                for (0..inter) |i| y += @as(f64, h[i]) * c.down[(e * l + d) * inter + i];
                out[row * l + d] += c.w[row * k + j] * bf16Round(@floatCast(y));
                unit[row * l + d] += c.w[row * k + j] * bf16Ulp(bf16Round(@floatCast(y)));
            }
        };
    }
};

/// `routedExperts` at `products` .bf16 on `s`: the whole call held to the f64 recompute, and the picks placed in three
/// groups giving the whole call's bytes.
fn routedCase(s: mlx.mlx_stream) !void {
    const c = RoutedCase.init();
    const T = RoutedCase;
    var want: [T.t * T.l]f32 = undefined;
    var unit: [T.t * T.l]f64 = undefined;
    c.reference(&want, &unit);
    var o: Ops = .{ .s = s };
    defer o.deinit();
    const lat = try o.own(mlx.mlx_array_new_data(&c.lat, &[_]c_int{ T.t, T.l }, 2, .float32));
    const weights = try o.own(mlx.mlx_array_new_data(&c.w, &[_]c_int{ T.t, T.k }, 2, .float32));
    const gu = try o.cast(try o.own(mlx.mlx_array_new_data(&c.gu, &[_]c_int{ T.slots, 2 * T.inter, T.l }, 3, .float32)), .bfloat16);
    const down = try o.cast(try o.own(mlx.mlx_array_new_data(&c.down, &[_]c_int{ T.slots, T.l, T.inter }, 3, .float32)), .bfloat16);
    const g: MoeGeometry = .{ .top_k = T.k, .inter = T.inter, .scale = 1, .swiglu_limit = T.limit, .eps = 1e-5 };
    const was = products;
    products = .bf16;
    defer products = was;
    const whole = try hostRows(&o, try o.own(try routedExperts(testing.allocator, s, lat, weights, null, &c.rows, g, gu, down, null)));
    // Three groups, as the stream forms them by the cache's slots, each landing at its picks' positions in one sum.
    var picks: [3][T.t * T.k]u32 = undefined;
    var rows: [3][T.t * T.k]u32 = undefined;
    var n: [3]usize = .{ 0, 0, 0 };
    for (c.rows, 0..) |e, i| {
        const gi: usize = e % 3;
        picks[gi][n[gi]] = @intCast(i);
        rows[gi][n[gi]] = e;
        n[gi] += 1;
    }
    var placed = try placedZeros(s, T.t, T.k, T.l);
    defer _ = mlx.mlx_array_free(placed);
    for (0..3) |gi| try placeExperts(testing.allocator, s, &placed, lat, weights, picks[gi][0..n[gi]], rows[gi][0..n[gi]], g, gu, down, null);
    const grouped = try hostRows(&o, try o.own(try sumPlaced(s, placed, T.t, T.k)));
    try testing.expectEqualSlices(u32, @ptrCast(whole), @ptrCast(grouped));
    // Each group summed on its own and the sums added round in another order (126 of the 384 values move): the case
    // tells the two apart.
    {
        var parts: [3]Arr = undefined;
        for (0..3) |gi| parts[gi] = try o.own(try routedExperts(testing.allocator, s, lat, weights, picks[gi][0..n[gi]], rows[gi][0..n[gi]], g, gu, down, null));
        const summed = try hostRows(&o, try o.binary(.add, try o.binary(.add, parts[0], parts[1]), parts[2]));
        var diff: usize = 0;
        for (whole, summed) |x, y| diff += @intFromBool(@as(u32, @bitCast(x)) != @as(u32, @bitCast(y)));
        try testing.expect(diff > 0);
    }
    // Bars (measured: 0 of the 384 values miss, on either stream): a miss is an f32 sum landing across a BF16 rounding
    // the f64 sum does not, so each is within one BF16 unit of each pick's output, and at most 1% of the values miss.
    // Products that skip a BF16 rounding miss most values.
    var missed: usize = 0;
    for (whole, want, 0..) |got, ref, i| {
        if (got == ref) continue;
        missed += 1;
        if (@abs(@as(f64, got) - ref) > unit[i]) {
            std.debug.print("routed experts at {d}: {d} against the f64 recompute's {d}, over {e}\n", .{ i, got, ref, unit[i] });
            return error.OutsideBar;
        }
    }
    if (missed * 100 > whole.len) {
        std.debug.print("routed experts: {d} of {d} values miss the f64 recompute\n", .{ missed, whole.len });
        return error.OutsideBar;
    }
}

fn hostRows(o: *Ops, a: Arr) ![]const f32 {
    const c = try o.contiguous(try o.cast(a, .float32));
    try mlx.check(mlx.mlx_array_eval(c));
    const p = mlx.mlx_array_data_float32(c) orelse return error.HostRead;
    return p[0..mlx.mlx_array_size(c)];
}

const testing = std.testing;

test "hy4 routed experts at BF16 on Sushi Cooler's stream: each pick's clamped SwiGLU over its bank row as stored, held to an f64 recompute rounded where BF16 rounds; three groups placed in one sum give the whole call's bytes" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    try routedCase(es);
}

test "hy4 routed experts at BF16 on the CPU stream, whose gather takes f32 only: the same products as the GPU's, held to the same recompute" {
    const cpu = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(cpu);
    try routedCase(cpu);
}

test "hy4 routed experts at f32 products on Sushi Cooler's stream never widen the BF16 bank: a decode's row and a prefill's eight hold a fraction of its f32 copy, with the bytes of a bank widened whole" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const slots = 64;
    const inter = 64;
    const l = 128;
    const k = 4;
    const t_max = 8;
    var prng = std.Random.DefaultPrng.init(0x4879_6633);
    const r = prng.random();
    const gu_v = try testing.allocator.alloc(f32, slots * 2 * inter * l);
    defer testing.allocator.free(gu_v);
    const down_v = try testing.allocator.alloc(f32, slots * l * inter);
    defer testing.allocator.free(down_v);
    for (gu_v) |*v| v.* = 0.2 * r.floatNorm(f32);
    for (down_v) |*v| v.* = 0.2 * r.floatNorm(f32);
    var lat_v: [t_max * l]f32 = undefined;
    for (&lat_v) |*v| v.* = r.floatNorm(f32);
    var w_v: [t_max * k]f32 = undefined;
    for (&w_v) |*v| v.* = 0.25 + r.float(f32);
    // Repeats within a row and across rows, as a layer's picks have.
    var rows: [t_max * k]u32 = undefined;
    for (&rows) |*v| v.* = r.uintLessThan(u32, 12) * 5;
    var o: Ops = .{ .s = es };
    defer o.deinit();
    const gu = try o.cast(try o.own(mlx.mlx_array_new_data(gu_v.ptr, &[_]c_int{ slots, 2 * inter, l }, 3, .float32)), .bfloat16);
    const down = try o.cast(try o.own(mlx.mlx_array_new_data(down_v.ptr, &[_]c_int{ slots, l, inter }, 3, .float32)), .bfloat16);
    const gu32 = try o.cast(gu, .float32);
    const down32 = try o.cast(down, .float32);
    try mlx.check(mlx.mlx_array_eval(gu32));
    try mlx.check(mlx.mlx_array_eval(down32));
    const g: MoeGeometry = .{ .top_k = k, .inter = inter, .scale = 1, .swiglu_limit = 10, .eps = 1e-5 };
    const products_was = products;
    products = .f32;
    defer products = products_was;
    for ([_]c_int{ 1, t_max }) |t| {
        const n: usize = @intCast(t * k);
        const lat = try o.own(mlx.mlx_array_new_data(&lat_v, &[_]c_int{ t, l }, 2, .float32));
        const weights = try o.own(mlx.mlx_array_new_data(&w_v, &[_]c_int{ t, k }, 2, .float32));
        const whole = try hostRows(&o, try o.own(try routedExperts(testing.allocator, es, lat, weights, null, rows[0..n], g, gu32, down32, null)));
        mlx.engine.close();
        const in_use0 = mlx.engine.poolTotal() - mlx.engine.poolFree();
        mlx.engine.resetPoolPeak(.process);
        const got = try o.own(try routedExperts(testing.allocator, es, lat, weights, null, rows[0..n], g, gu, down, null));
        try mlx.check(mlx.mlx_array_eval(got));
        mlx.engine.close();
        const asked = mlx.engine.poolPeak(.process) -| in_use0;
        try testing.expectEqualSlices(u32, @ptrCast(whole), @ptrCast(try hostRows(&o, got)));
        // The bank widened whole is 6 MB.
        try testing.expect(asked < (slots * 3 * inter * l * 4) / 4);
    }
}

fn ihcValues(o: *Ops, shape: []const c_int, seed: u64, scale: f32) !Arr {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const buf = try testing.allocator.alloc(f32, n);
    defer testing.allocator.free(buf);
    var st = seed;
    for (buf) |*v| {
        st = st *% 6364136223846793005 +% 1442695040888963407;
        v.* = (@as(f32, @floatFromInt(@as(u32, @truncate(st >> 32)))) / 2147483648.0 - 1.0) * scale;
    }
    return o.own(try mlx.newData(buf.ptr, shape, .float32));
}

fn expectNearPeak(got: []const f32, want: []const f32, rel: f32) !void {
    var peak: f32 = 0;
    for (want) |v| peak = @max(peak, @abs(v));
    for (got, want) |g, w| if (@abs(g - w) > rel * peak) return error.OutsideBar;
}

test "hy4 ihc on Sushi Cooler's stream: a mix with its normed collapse is one mix kernel and one norm kernel, the expand one kernel, within f32 rounding of the op chain (the expand its bytes)" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    // The forward runs its f32 products off TF32; the op chain here is held to the same.
    const exact = mlx.engine.exactF32();
    defer exact.end();
    const saved = hc_kernels_on;
    defer hc_kernels_on = saved;
    const T = 3;
    const n = 4;
    const D = 256;
    var o: Ops = .{ .s = es };
    defer o.deinit();
    const h = try ihcValues(&o, &.{ T, n, D }, 1, 2.0);
    const y = try ihcValues(&o, &.{ T, D }, 2, 1.5);
    const w: Mix = .{ .fn_ = try ihcValues(&o, &.{ 2 * n, n * D }, 3, 0.05), .base = try ihcValues(&o, &.{2 * n}, 4, 0.5), .scale = try ihcValues(&o, &.{2}, 5, 1.0) };
    const hw: Mix = .{ .fn_ = try o.slice(w.fn_, 0, 0, n), .base = try o.slice(w.base, 0, 0, n), .scale = try o.slice(w.scale, 0, 0, 1) };
    const norm_w = try ihcValues(&o, &.{D}, 6, 1.0);
    const post = try ihcValues(&o, &.{ T, n }, 7, 2.0);
    var got: [2]struct { post: []const f32, x: []const f32, head: []const f32, expand: []const f32, ops: usize } = undefined;
    for ([_]bool{ true, false }, &got) |on, *g| {
        hc_kernels_on = on;
        var so: Ops = .{ .s = es };
        defer so.deinit();
        const m = try mixNorm(&so, h, w, n, 2.0, 1e-6, 1e-5, norm_w);
        g.ops = so.count;
        const e = try expand(&so, h, post, y);
        g.* = .{ .post = try testing.allocator.dupe(f32, try hostRows(&so, m.post)), .x = try testing.allocator.dupe(f32, try hostRows(&so, m.x)), .head = try testing.allocator.dupe(f32, try hostRows(&so, try headNorm(&so, h, hw, n, 1e-6, 1e-5, norm_w))), .expand = try testing.allocator.dupe(f32, try hostRows(&so, e)), .ops = g.ops };
    }
    defer for (got) |g| for ([_][]const f32{ g.post, g.x, g.head, g.expand }) |b| testing.allocator.free(b);
    try testing.expect(got[0].ops <= 6 and got[1].ops > 15);
    try expectNearPeak(got[0].post, got[1].post, 1e-6);
    try expectNearPeak(got[0].x, got[1].x, 4e-6);
    try expectNearPeak(got[0].head, got[1].head, 4e-6);
    try testing.expectEqualSlices(f32, got[1].expand, got[0].expand);
}

test "hy4 recall on Sushi Cooler's stream: the shared score and top-k kernels give the op chain's logits to f32 rounding over BF16 pooled keys, and a top-k of them, ties to the lower group" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const exact = mlx.engine.exactF32();
    defer exact.end();
    const saved = select_kernels_on;
    defer select_kernels_on = saved;
    const g: IndexerGeometry = .{ .heads = 16, .dim = 128, .rope = 64, .ratio = 4, .group_topk = 8, .theta = 1e7, .eps = 1e-6 };
    const T = 6;
    const G = 40;
    const p0 = 150;
    var o: Ops = .{ .s = es };
    defer o.deinit();
    const x = try ihcValues(&o, &.{ T, 64 }, 31, 1.0);
    const q_resid = try ihcValues(&o, &.{ T, 32 }, 32, 1.0);
    // Pooled keys with repeated rows, so equal scores (and relu zeros) tie.
    const distinct = try o.cast(try ihcValues(&o, &.{ 10, g.dim }, 33, 1.0), .bfloat16);
    const pooled = try o.contiguous(try o.reshape(try o.broadcast(try o.reshape(distinct, &.{ 1, 10, g.dim }), &.{ 4, 10, g.dim }), &.{ G, g.dim }));
    const w: Indexer = .{ .query = try ihcValues(&o, &.{ g.heads * g.dim, 32 }, 34, 0.2), .head_weight = try ihcValues(&o, &.{ g.heads, 64 }, 35, 0.5), .token_key = undefined, .token_key_norm = undefined, .key_value = undefined, .key_gate = undefined, .ape = undefined, .key_norm = undefined };
    var logits: [2][]f32 = undefined;
    var picks: [2][]f32 = undefined;
    var ops: [2]usize = undefined;
    for ([_]bool{ true, false }, 0..) |on, i| {
        select_kernels_on = on;
        var so: Ops = .{ .s = es };
        defer so.deinit();
        const r = try recallAt(&so, x, q_resid, pooled, w, g, p0);
        ops[i] = so.count;
        logits[i] = try testing.allocator.dupe(f32, try hostRows(&so, r.logits));
        picks[i] = try testing.allocator.dupe(f32, try hostRows(&so, r.groups));
    }
    defer for (0..2) |i| {
        testing.allocator.free(logits[i]);
        testing.allocator.free(picks[i]);
    };
    // The kernels replace the chain's score, mask and argpartition arrays ([H, T, G] among them); the query stays.
    try testing.expect(ops[0] + 12 <= ops[1]);
    var peak: f32 = 0;
    for (logits[1]) |v| if (std.math.isFinite(v)) {
        peak = @max(peak, @abs(v));
    };
    for (logits[0], logits[1]) |a, b| {
        try testing.expectEqual(std.math.isFinite(b), std.math.isFinite(a));
        if (std.math.isFinite(b)) try testing.expect(@abs(a - b) <= 1e-5 * peak);
    }
    const k: usize = @intCast(g.group_topk);
    for (0..T) |t| {
        const own: f32 = @floatFromInt(@divFloor(p0 + @as(c_int, @intCast(t)), g.ratio));
        const row = logits[0][(t * G)..][0..G];
        const pr = picks[0][(t * k)..][0..k];
        try testing.expectEqual(own, pr[k - 1]);
        var picked: [G]bool = @splat(false);
        for (pr[0 .. k - 1]) |p| {
            const pi: usize = @intFromFloat(p);
            try testing.expect(!picked[pi]);
            picked[pi] = true;
        }
        for (0..G) |a| if (picked[a]) for (0..G) |b| if (!picked[b]) {
            try testing.expect(row[a] > row[b] or (row[a] == row[b] and a < b) or !std.math.isFinite(row[a]));
        };
    }
}

test "hy4 attention spans: at the release's gate and W_o shapes the whole spans' one launch and a short last span's give each row its own span's launch's bytes, through a trellis bank and a BF16 weight, spans of 256 and 32" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    defer spans_share = true;
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const exact = mlx.engine.exactF32();
    defer exact.end();
    const exl3 = @import("sushi_exl3");
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x4879_5350);
    const rnd = prng.random();
    // The sliding layers' gate and W_o (64 heads), the sparse layers' (32): a chunk past MLX's split-K bound on the
    // sliding W_o, one past the decoded trellis path's minimum on the sparse gate.
    const Case = struct { in: c_int, out: c_int, span: c_int, rows: c_int };
    for ([_]Case{
        .{ .in = 3072, .out = 16384, .span = 256, .rows = 600 },
        .{ .in = 16384, .out = 3072, .span = 256, .rows = 8195 },
        .{ .in = 3072, .out = 8192, .span = 32, .rows = 1100 },
        .{ .in = 8192, .out = 3072, .span = 32, .rows = 4108 },
    }) |c| {
        const in: usize = @intCast(c.in);
        const out: usize = @intCast(c.out);
        var o: Ops = .{ .s = es };
        defer o.deinit();
        const xs = try gpa.alloc(u16, @as(usize, @intCast(c.rows)) * in);
        defer gpa.free(xs);
        for (xs) |*v| v.* = @truncate(@as(u32, @bitCast(rnd.float(f32) * 2 - 1)) >> 16);
        const x = try o.own(mlx.mlx_array_new_data(xs.ptr, &[_]c_int{ c.rows, c.in }, 2, .bfloat16));
        const words = try gpa.alloc(u16, (in / 16) * (out / 16) * 64);
        defer gpa.free(words);
        for (words) |*v| v.* = rnd.int(u16);
        const suh = try gpa.alloc(u16, in);
        defer gpa.free(suh);
        for (suh) |*v| v.* = exl3.format.f32ToF16Bits(0.01 + rnd.float(f32) * 0.05);
        const svh = try gpa.alloc(u16, out);
        defer gpa.free(svh);
        for (svh) |*v| v.* = exl3.format.f32ToF16Bits(0.5 + rnd.float(f32));
        const bank: hy4_stq.Bank = .{
            .w = try o.own(mlx.mlx_array_new_data(words.ptr, &[_]c_int{ 1, @divExact(c.in, 16), @divExact(c.out, 16), 64 }, 4, .uint16)),
            .suh = try o.own(mlx.mlx_array_new_data(suh.ptr, &[_]c_int{ 1, c.in }, 2, .float16)),
            .svh = try o.own(mlx.mlx_array_new_data(svh.ptr, &[_]c_int{ 1, c.out }, 2, .float16)),
            .dec = exl3.kernels.SPEC1_W15,
        };
        const w_bits = try gpa.alloc(u16, out * in);
        defer gpa.free(w_bits);
        for (w_bits) |*v| v.* = @truncate(@as(u32, @bitCast(rnd.floatNorm(f32) * 0.02)) >> 16);
        const dense = Lin.dense(try o.own(mlx.mlx_array_new_data(w_bits.ptr, &[_]c_int{ c.out, c.in }, 2, .bfloat16)));
        for ([_]Lin{ .{ .w = dense.w, .stq = bank }, dense }) |lin| {
            var got: [2]Arr = undefined;
            for (&got, [_]bool{ false, true }) |*g, share| {
                spans_share = share;
                var ob: Ops = .{ .s = es };
                defer ob.deinit();
                g.* = try o.own(try ob.result(try ob.contiguous(try linearSpans(&ob, x, lin, c.span))));
                try mlx.check(mlx.mlx_array_eval(g.*));
            }
            const n = mlx.mlx_array_size(got[0]);
            try testing.expectEqual(n, mlx.mlx_array_size(got[1]));
            const a: [*]const u16 = @ptrCast(mlx.mlx_array_data_bfloat16(got[0]) orelse return error.Unreadable);
            const b: [*]const u16 = @ptrCast(mlx.mlx_array_data_bfloat16(got[1]) orelse return error.Unreadable);
            try testing.expectEqualSlices(u16, a[0..n], b[0..n]);
        }
    }
}

test "hy4 cache-biased route: a decode row's picks move only from experts the cache lacks to experts it holds, each within n of the one it displaced, the weights the unbiased scores of the picks, at the release's 512 experts and top 16 and the tiny's 8 and 2" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const a = testing.allocator;
    const Case = struct { e: usize, k: c_int, n: f32 };
    for ([_]Case{ .{ .e = 512, .k = 16, .n = 0.05 }, .{ .e = 512, .k = 16, .n = 0.5 }, .{ .e = 8, .k = 2, .n = 0.1 }, .{ .e = 8, .k = 2, .n = 8 } }, 0..) |c, ci| {
        var o: Ops = .{ .s = es };
        defer o.deinit();
        const lg = try a.alloc(f32, c.e);
        defer a.free(lg);
        const bs = try a.alloc(f32, c.e);
        defer a.free(bs);
        const map = try a.alloc(i32, c.e);
        defer a.free(map);
        const pen = try a.alloc(f32, c.e);
        defer a.free(pen);
        const eye = try a.alloc(f32, c.e * c.e);
        defer a.free(eye);
        @memset(eye, 0);
        var r = std.Random.DefaultPrng.init(0xb1a5 + ci);
        for (lg, bs, map, 0..) |*l, *b, *m, i| {
            l.* = @floatCast(r.random().floatNorm(f64) * 2);
            b.* = @floatCast(r.random().floatNorm(f64) * 0.05);
            // About half the experts held, at any slot.
            m.* = if (r.random().boolean()) @intCast(i) else -1;
            eye[i * c.e + i] = 1;
        }
        @import("../expert_stream.zig").biasPenalty(map, c.n, pen);
        const ei: c_int = @intCast(c.e);
        // The router's product over the identity is the logits as given.
        const x = try o.own(mlx.mlx_array_new_data(lg.ptr, &[_]c_int{ 1, ei }, 2, .float32));
        const w: Router = .{ .weight = try o.own(mlx.mlx_array_new_data(eye.ptr, &[_]c_int{ ei, ei }, 2, .float32)), .bias = try o.own(mlx.mlx_array_new_data(bs.ptr, &[_]c_int{ei}, 1, .float32)) };
        const exact = try route(&o, x, w, c.k, 3.998);
        const biased = try route(&o, x, try cacheBiased(&o, w, try o.own(mlx.mlx_array_new_data(pen.ptr, &[_]c_int{ei}, 1, .float32))), c.k, 3.998);
        const exact_ids = try hostPicks(a, &o, exact.picks, null);
        defer a.free(exact_ids);
        const ids = try hostPicks(a, &o, biased.picks, null);
        defer a.free(ids);
        const weights = try hostRows(&o, biased.weights);
        const choice = struct {
            fn of(l: []const f32, b: []const f32, e: u32) f64 {
                return 1 / (1 + @exp(-@as(f64, l[e]))) + b[e];
            }
        }.of;
        var moved: usize = 0;
        for (ids) |e| if (std.mem.indexOfScalar(u32, exact_ids, e) == null) {
            moved += 1;
            try testing.expect(map[e] >= 0);
        };
        for (exact_ids) |e| if (std.mem.indexOfScalar(u32, ids, e) == null) {
            try testing.expect(map[e] < 0);
            // Every held expert that displaced it scores at most n below it.
            for (ids) |p| if (std.mem.indexOfScalar(u32, exact_ids, p) == null)
                try testing.expect(choice(lg, bs, p) >= choice(lg, bs, e) - c.n - 1e-6);
        };
        // The penalty moves something at each case's seed, so the bound above is exercised.
        try testing.expect(moved > 0);
        var total: f64 = 0;
        for (ids) |e| total += 1 / (1 + @exp(-@as(f64, lg[e])));
        for (ids, weights) |e, got| try testing.expectApproxEqRel(1 / (1 + @exp(-@as(f64, lg[e]))) / total * 3.998, @as(f64, got), 1e-5);
    }
}

test "hy4 attention verify rows: attendGatedCausal and attendRowsGated over 2 to 8 rows give each row its one-row call's bytes" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const exact = mlx.engine.exactF32();
    defer exact.end();

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x7a12);
    const rnd = prng.random();
    const fill = struct {
        fn f(a: std.mem.Allocator, r: std.Random, n: usize, lo: f32, span: f32) ![]f32 {
            const v = try a.alloc(f32, n);
            for (v) |*e| e.* = lo + r.float(f32) * span;
            return v;
        }
    };

    const g_sparse: AttnGeometry = .{ .heads = 8, .nope = 32, .rope = 16, .v = 32, .latent = 128, .theta = 1e7, .eps = 1e-5 };
    const g_slide: AttnGeometry = .{ .heads = 8, .nope = 32, .rope = 16, .v = 32, .latent = 128, .theta = 1e4, .eps = 1e-5 };

    for ([_]struct { g: AttnGeometry, t: c_int, n: c_int, gathered: bool, window: c_int }{
        .{ .g = g_sparse, .t = 2, .n = 16, .gathered = true, .window = 0 },
        .{ .g = g_sparse, .t = 6, .n = 16, .gathered = true, .window = 0 },
        .{ .g = g_sparse, .t = 8, .n = 16, .gathered = true, .window = 0 },
        .{ .g = g_sparse, .t = 6, .n = 12, .gathered = false, .window = 0 },
        .{ .g = g_slide, .t = 6, .n = 14, .gathered = false, .window = 8 },
    }) |case| {
        var o: Ops = .{ .s = es };
        defer o.deinit();
        const h = case.g.heads;
        const t = case.t;
        const n = case.n;
        const qk = case.g.nope + case.g.rope;
        const hv: usize = @intCast(h * case.g.v);
        const qv = try fill.f(alloc, rnd, @intCast(h * t * qk), -1.0, 2.0);
        const cv = try fill.f(alloc, rnd, @intCast((if (case.gathered) t * n else n) * case.g.latent), -0.5, 1.0);
        const pv = try fill.f(alloc, rnd, @intCast((if (case.gathered) t * n else n) * case.g.rope), -0.5, 1.0);
        const kvbv = try fill.f(alloc, rnd, @intCast(h * (case.g.nope + case.g.v) * case.g.latent), -0.1, 0.2);
        const gatev = try fill.f(alloc, rnd, @intCast(t * h * case.g.v), -1.0, 2.0);
        const sinkv = try fill.f(alloc, rnd, @intCast(h), -1.0, 2.0);

        const q = try o.own(mlx.mlx_array_new_data(qv.ptr, &[_]c_int{ h, t, qk }, 3, .float32));
        const c_arr = try o.cast(try o.own(mlx.mlx_array_new_data(cv.ptr, &[_]c_int{ if (case.gathered) t * n else n, case.g.latent }, 2, .float32)), .bfloat16);
        const pe_arr = try o.cast(try o.own(mlx.mlx_array_new_data(pv.ptr, &[_]c_int{ if (case.gathered) t * n else n, case.g.rope }, 2, .float32)), .bfloat16);
        const kv_b = try o.cast(try o.own(mlx.mlx_array_new_data(kvbv.ptr, &[_]c_int{ h * (case.g.nope + case.g.v), case.g.latent }, 2, .float32)), .bfloat16);
        const gate = try o.cast(try o.own(mlx.mlx_array_new_data(gatev.ptr, &[_]c_int{ t, h * case.g.v }, 2, .float32)), .bfloat16);
        const sinks = try o.own(mlx.mlx_array_new_data(sinkv.ptr, &[_]c_int{h}, 1, .float32));
        var attn: Attn = undefined;
        attn.kv_b = kv_b;
        attn.sinks = sinks;
        attn.o = .{ .w = try o.zeros(&.{ 1, 1 }, .bfloat16) };

        var keep_arr: Arr = undefined;
        if (case.gathered) {
            const kv = try alloc.alloc(bool, @intCast(t * n));
            for (kv, 0..) |*d, i| d.* = (i % 3) != 0;
            keep_arr = try o.own(mlx.mlx_array_new_data(kv.ptr, &[_]c_int{ t, n }, 2, .bool_));
        }

        const p0 = n - t;
        const wide_in: SpanIn = .{ .gate = gate, .kv_b32 = null, .whole = false };
        const wide = try hostRows(&o, if (case.gathered)
            try attendRowsGated(&o, wide_in, q, c_arr, pe_arr, keep_arr, attn, case.g, null)
        else
            try attendGatedCausal(&o, wide_in, q, c_arr, pe_arr, attn, case.g, p0, case.window, null));

        const tu: usize = @intCast(t);
        for (0..tu) |r| {
            const ri: c_int = @intCast(r);
            const q_r = try o.slice(q, 1, ri, ri + 1);
            const in_r: SpanIn = .{ .gate = try o.slice(gate, 0, ri, ri + 1), .kv_b32 = null, .whole = false };
            const one = try hostRows(&o, if (case.gathered)
                try attendRowsGated(&o, in_r, q_r, try o.slice(c_arr, 0, ri * n, (ri + 1) * n), try o.slice(pe_arr, 0, ri * n, (ri + 1) * n), try o.slice(keep_arr, 0, ri, ri + 1), attn, case.g, null)
            else blk: {
                const hi = p0 + ri + 1;
                const lo: c_int = if (case.window > 0) @max(0, hi - case.window) else 0;
                const nr = hi - lo;
                const p0_r = if (case.window > 0) nr - 1 else p0 + ri;
                break :blk try attendGatedCausal(&o, in_r, q_r, try o.slice(c_arr, 0, lo, hi), try o.slice(pe_arr, 0, lo, hi), attn, case.g, p0_r, 0, null);
            });
            try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(one), std.mem.sliceAsBytes(wide[r * hv ..][0..hv]));
        }
        if (!case.gathered and case.window > 0) {
            const ring_p0: c_int = 40;
            const ring_wide = try hostRows(&o, try attendGatedRing(&o, wide_in, q, c_arr, pe_arr, attn, case.g, p0, case.window, ring_p0, null));
            for (0..tu) |r| {
                const ri: c_int = @intCast(r);
                const q_r = try o.slice(q, 1, ri, ri + 1);
                const in_r: SpanIn = .{ .gate = try o.slice(gate, 0, ri, ri + 1), .kv_b32 = null, .whole = false };
                const hi = p0 + ri + 1;
                const lo = hi - case.window;
                var c_r = try o.slice(c_arr, 0, lo, hi);
                var pe_r = try o.slice(pe_arr, 0, lo, hi);
                const slot = @mod(ring_p0 + ri + 1 - case.window, case.window);
                if (slot != 0) {
                    const cut = case.window - slot;
                    c_r = try o.concat(&.{ try o.slice(c_r, 0, cut, case.window), try o.slice(c_r, 0, 0, cut) }, 0);
                    pe_r = try o.concat(&.{ try o.slice(pe_r, 0, cut, case.window), try o.slice(pe_r, 0, 0, cut) }, 0);
                }
                const one = try hostRows(&o, try attendGatedCausal(&o, in_r, q_r, c_r, pe_r, attn, case.g, case.window - 1, 0, null));
                try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(one), std.mem.sliceAsBytes(ring_wide[r * hv ..][0..hv]));
            }
        }
    }
}

test "hy4 linearF32 verify rows: 2 to 8 rows outside hy4_row give each row its one-row call's bytes" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const exact = mlx.engine.exactF32();
    defer exact.end();

    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x7f32);
    const rnd = prng.random();
    for ([_]struct { n: c_int, k: c_int, t: c_int }{
        .{ .n = 128, .k = 64, .t = 6 },
        .{ .n = 4096, .k = 128, .t = 6 },
        .{ .n = 128, .k = 64, .t = 2 },
        .{ .n = 128, .k = 64, .t = 8 },
    }) |case| {
        var o: Ops = .{ .s = es };
        defer o.deinit();
        const nu: usize = @intCast(case.n);
        const ku: usize = @intCast(case.k);
        const tu: usize = @intCast(case.t);
        const xv = try alloc.alloc(f32, tu * ku);
        const wv = try alloc.alloc(f32, nu * ku);
        for (xv) |*v| v.* = rnd.floatNorm(f32);
        for (wv) |*v| v.* = rnd.floatNorm(f32) * 0.1;
        const x = try o.own(mlx.mlx_array_new_data(xv.ptr, &[_]c_int{ case.t, case.k }, 2, .float32));
        const w = try o.cast(try o.own(mlx.mlx_array_new_data(wv.ptr, &[_]c_int{ case.n, case.k }, 2, .float32)), .bfloat16);
        const wide = try hostRows(&o, try linearF32(&o, x, w));
        for (0..tu) |r| {
            const ri: c_int = @intCast(r);
            const one = try hostRows(&o, try linearF32(&o, try o.slice(x, 0, ri, ri + 1), w));
            try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(one), std.mem.sliceAsBytes(wide[r * nu ..][0..nu]));
        }
    }
}

test "hy4 placedZeros: zeros written on the GPU, so a dry recording asks the arena for them and none of the host ring" {
    try mlx.requireGpu();
    const core = @import("../tensor/core.zig");
    const plan = @import("../tensor/plan/plan_api.zig");
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const s = mlx.engine.stream();
    const scope = mlx.engine.enter(s);
    defer scope.end();
    // Real: every element a zero, on a fresh buffer a dirty pool hands back.
    {
        var d: Ops = .{ .s = s };
        defer d.deinit();
        _ = try hostRows(&d, try d.binary(.add, try d.scalar(3, .float32), try d.own(try placedZeros(s, 3, 2, 40))));
    }
    {
        // The threes' buffer is back in the pool at the size the zeros ask for.
        var o: Ops = .{ .s = s };
        defer o.deinit();
        for (try hostRows(&o, try o.own(try placedZeros(s, 3, 2, 40)))) |v| try testing.expectEqual(@as(f32, 0), v);
    }
    var rec = try plan.Recorder.begin(std.testing.allocator, .{ .site = 1 });
    defer rec.deinit();
    var driver: plan.Driver = .{ .gpa = std.testing.allocator };
    defer driver.deinit();
    try core.beginRecording(.{ .dry = .{ .driver = &driver } }, &rec);
    const z = try placedZeros(s, 3, 2, 40);
    _ = mlx.mlx_array_free(z);
    try core.endRecording();
    var asked: u64 = 0;
    for (rec.instances.items) |inst| for (inst.asks.items) |as| {
        try testing.expectEqual(false, as.host);
        asked += as.bytes;
    };
    try testing.expect(asked >= 3 * 2 * 40 * 4);
}

test "hy4 indexer spans: hoisted projections and rope match per-span recalledAt byte for byte with fewer dispatches" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const exact = mlx.engine.exactF32();
    defer exact.end();

    const g: IndexerGeometry = .{ .heads = 16, .dim = 128, .rope = 64, .ratio = 4, .group_topk = 512, .theta = 1e7, .eps = 1e-5 };
    const span: c_int = 32;
    const p0: c_int = 4096;
    const groups: c_int = 1048;
    const q_lora: c_int = 1024;
    const d_model: c_int = 3072;

    for ([_]c_int{ 96, 70 }) |t| {
        const n = p0 + t;
        var o: Ops = .{ .s = es };
        defer o.deinit();
        const x = try ihcValues(&o, &.{ t, d_model }, @intCast(t + 101), 0.5);
        const q_resid = try ihcValues(&o, &.{ t, q_lora }, @intCast(t + 102), 0.5);
        const pooled = try ihcValues(&o, &.{ groups, g.dim }, @intCast(t + 103), 0.5);
        const w: Indexer = .{
            .query = try o.cast(try ihcValues(&o, &.{ g.heads * g.dim, q_lora }, 104, 0.05), .bfloat16),
            .head_weight = try o.cast(try ihcValues(&o, &.{ g.heads, d_model }, 105, 0.05), .bfloat16),
            .token_key = undefined,
            .token_key_norm = undefined,
            .key_value = undefined,
            .key_gate = undefined,
            .ape = undefined,
            .key_norm = undefined,
        };
        for ([_]Arr{ x, q_resid, pooled, w.query, w.head_weight }) |a| try mlx.check(mlx.mlx_array_eval(a));

        const d0 = mlx.engine.stats().dispatches;
        var want_rows: [4]Arr = undefined;
        var want_keep: [4]Arr = undefined;
        var ns: usize = 0;
        var r0: c_int = 0;
        while (r0 < t) : (r0 += span) {
            const r1 = @min(t, r0 + span);
            const sel = try recalledAt(&o, try o.slice(x, 0, r0, r1), try o.slice(q_resid, 0, r0, r1), pooled, w, g, p0 + r0, n);
            want_rows[ns] = sel.rows;
            want_keep[ns] = sel.keep;
            ns += 1;
        }
        const wr = try o.contiguous(try o.concat(want_rows[0..ns], 0));
        const wk = try o.contiguous(try o.concat(want_keep[0..ns], 0));
        try mlx.check(mlx.mlx_array_eval(wr));
        try mlx.check(mlx.mlx_array_eval(wk));
        const old_d = mlx.engine.stats().dispatches - d0;

        const d1 = mlx.engine.stats().dispatches;
        const idx = (try indexPrepare(&o, x, q_resid, pooled, w, g, p0, 0, span)).?;
        var got_rows: [4]Arr = undefined;
        var got_keep: [4]Arr = undefined;
        ns = 0;
        r0 = 0;
        while (r0 < t) : (r0 += span) {
            const r1 = @min(t, r0 + span);
            const sel = try recalledPrepared(&o, idx, r0, r1, groups, g, p0 + r0, n);
            got_rows[ns] = sel.rows;
            got_keep[ns] = sel.keep;
            ns += 1;
        }
        const gr = try o.contiguous(try o.concat(got_rows[0..ns], 0));
        const gk = try o.contiguous(try o.concat(got_keep[0..ns], 0));
        try mlx.check(mlx.mlx_array_eval(gr));
        try mlx.check(mlx.mlx_array_eval(gk));
        const new_d = mlx.engine.stats().dispatches - d1;

        try testing.expect(new_d < old_d);
        const nr = mlx.mlx_array_size(wr);
        try testing.expectEqualSlices(u32, (mlx.mlx_array_data_uint32(wr) orelse return error.HostRead)[0..nr], (mlx.mlx_array_data_uint32(gr) orelse return error.HostRead)[0..nr]);
        try testing.expectEqualSlices(bool, (mlx.mlx_array_data_bool(wk) orelse return error.HostRead)[0..nr], (mlx.mlx_array_data_bool(gk) orelse return error.HostRead)[0..nr]);
    }
}

test "hy4 sparse core spans: 3D matmulRowsAs matches 32-row absorb and valueOut byte for byte with fewer dispatches" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const exact = mlx.engine.exactF32();
    defer exact.end();

    const g: AttnGeometry = .{ .heads = 32, .nope = 192, .rope = 64, .v = 256, .latent = 512, .theta = 1e7, .eps = 1e-5 };
    const span: c_int = 32;
    const n_keys: c_int = 256;
    for ([_]c_int{ 96, 256 }) |t| {
        var o: Ops = .{ .s = es };
        defer o.deinit();
        const q = try ihcValues(&o, &.{ g.heads, t, g.nope + g.rope }, @intCast(t + 201), 0.3);
        const c_keys = try o.cast(try ihcValues(&o, &.{ n_keys, g.latent }, @intCast(t + 202), 0.3), .bfloat16);
        const pe_keys = try o.cast(try ihcValues(&o, &.{ n_keys, g.rope }, @intCast(t + 203), 0.3), .bfloat16);
        const kv_b32 = try ihcValues(&o, &.{ g.heads * (g.nope + g.v), g.latent }, 204, 0.05);
        const gate = try o.cast(try ihcValues(&o, &.{ t, g.heads * g.v }, @intCast(t + 205), 0.5), .bfloat16);
        const sinks = try ihcValues(&o, &.{g.heads}, 206, 0.5);
        var w: Attn = undefined;
        w.kv_b = try o.cast(kv_b32, .bfloat16);
        w.sinks = sinks;
        w.o = .{ .w = try o.zeros(&.{ 1, 1 }, .bfloat16) };
        for ([_]Arr{ q, c_keys, pe_keys, kv_b32, gate, sinks }) |a| try mlx.check(mlx.mlx_array_eval(a));

        const d0 = mlx.engine.stats().dispatches;
        var want_parts: [8]Arr = undefined;
        var ns: usize = 0;
        var r0: c_int = 0;
        while (r0 < t) : (r0 += span) {
            const r1 = @min(t, r0 + span);
            const sin: SpanIn = .{ .gate = try o.slice(gate, 0, r0, r1), .kv_b32 = kv_b32, .whole = false };
            want_parts[ns] = try attendCoreGated(&o, sin, try o.slice(q, 1, r0, r1), .{ .range = .{ .keys = .{ .c = c_keys, .k_pe = pe_keys }, .p0 = r0, .window = 0 } }, w, g, null);
            ns += 1;
        }
        const want = try o.contiguous(try o.concat(want_parts[0..ns], 0));
        try mlx.check(mlx.mlx_array_eval(want));
        const old_d = mlx.engine.stats().dispatches - d0;

        const d1 = mlx.engine.stats().dispatches;
        const wide_in: SpanIn = .{ .gate = gate, .kv_b32 = kv_b32, .whole = false, .rows_as = span };
        const got = try o.contiguous(try attendCoreGated(&o, wide_in, q, .{ .range = .{ .keys = .{ .c = c_keys, .k_pe = pe_keys }, .p0 = 0, .window = 0 } }, w, g, null));
        try mlx.check(mlx.mlx_array_eval(got));
        const new_d = mlx.engine.stats().dispatches - d1;

        try testing.expect(new_d < old_d);
        const n = mlx.mlx_array_size(want);
        const wb: [*]const u16 = @ptrCast(mlx.mlx_array_data_bfloat16(want) orelse return error.HostRead);
        const gb: [*]const u16 = @ptrCast(mlx.mlx_array_data_bfloat16(got) orelse return error.HostRead);
        try testing.expectEqualSlices(u16, wb[0..n], gb[0..n]);
    }
}

test "hy4 attention decode row: the sliding window's and the selected rows' latent products run as a score, a softmax and a value launch, and give the op chain's bytes at the release's 64 and 32 heads" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    defer chain_kernels_on = true;
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const exact = mlx.engine.exactF32();
    defer exact.end();

    const Case = struct { heads: c_int, n: c_int, sparse: bool };
    for ([_]Case{
        .{ .heads = 64, .n = 512, .sparse = false },
        .{ .heads = 32, .n = 2048, .sparse = true },
        .{ .heads = 32, .n = 2048, .sparse = false },
    }, 0..) |case, ci| {
        const g: AttnGeometry = .{ .heads = case.heads, .nope = 192, .rope = 64, .v = 256, .latent = 512, .theta = 1e4, .eps = 1e-5 };
        var o: Ops = .{ .s = es };
        defer o.deinit();
        const seed: u64 = 77 * (ci + 1);
        const q32 = try ihcValues(&o, &.{ case.heads, 1, 256 }, seed + 1, 1.5);
        const q_lat = try ihcValues(&o, &.{ case.heads, 1, 512 }, seed + 2, 1.5);
        const q_pe = try o.slice(q32, 2, 192, 256);
        const keys = try ihcValues(&o, &.{ case.n, 512 }, seed + 3, 2.0);
        const pe = try o.cast(try ihcValues(&o, &.{ case.n, 64 }, seed + 4, 1.0), .bfloat16);
        const sinks = try ihcValues(&o, &.{case.heads}, seed + 5, 1.0);
        var attn: Attn = undefined;
        attn.sinks = sinks;
        const kv = try testing.allocator.alloc(bool, @intCast(case.n));
        defer testing.allocator.free(kv);
        for (kv, 0..) |*d, i| d.* = (i % 5) != 0;
        const keep_arr = try o.own(mlx.mlx_array_new_data(kv.ptr, &[_]c_int{ 1, case.n }, 2, .bool_));
        for ([_]Arr{ q_lat, q_pe, keys, pe, sinks, keep_arr }) |a| try mlx.check(mlx.mlx_array_eval(a));

        var bytes: [2][]const f32 = undefined;
        var launches: [2]u64 = undefined;
        for ([_]bool{ true, false }, 0..) |on, arm| {
            chain_kernels_on = on;
            const before = mlx.engine.stats().dispatches;
            const u = if (case.sparse)
                try rowsLatent(&o, q_lat, q_pe, keys, pe, keep_arr, attn, g)
            else
                try causalLatent(&o, q_lat, q_pe, keys, pe, attn, g, .{ .causal = .{ .p0 = case.n - 1, .window = 0, .steel = true } });
            launches[arm] = mlx.engine.stats().dispatches - before;
            bytes[arm] = try hostRows(&o, u);
        }
        try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(bytes[1]), std.mem.sliceAsBytes(bytes[0]));
        try testing.expect(launches[1] > launches[0]);
        try testing.expectEqual(@as(u64, 3), launches[0]);
    }
}

test "hy4 attention decode row: the absorbed query read from the unrotated queries is the rotated queries' product, one launch, at the release's 64 and 32 heads" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const exact = mlx.engine.exactF32();
    defer exact.end();
    for ([_]c_int{ 64, 32 }, 0..) |heads, ci| {
        const g: AttnGeometry = .{ .heads = heads, .nope = 192, .rope = 64, .v = 256, .latent = 512, .theta = 1e4, .eps = 1e-5 };
        var o: Ops = .{ .s = es };
        defer o.deinit();
        const seed: u64 = 31 * (ci + 1);
        const rows = try ihcValues(&o, &.{ 1, heads * 256 }, seed + 1, 1.5);
        const q_raw = try o.transpose(try o.reshape(rows, &.{ 1, heads, 256 }), &.{ 1, 0, 2 });
        const q = try rope(&o, q_raw, g.rope, g.theta, 77);
        const kv_b = try o.cast(try ihcValues(&o, &.{ heads * 448, 512 }, seed + 2, 0.2), .bfloat16);
        var attn: Attn = undefined;
        attn.kv_b = kv_b;
        for ([_]Arr{ q_raw, q, kv_b }) |a| try mlx.check(mlx.mlx_array_eval(a));
        try testing.expect(absorbServes(es, kv_b, g, 1));
        const before = mlx.engine.stats().dispatches;
        const raw = try absorb(&o, q, attn, g, null, null, q_raw);
        const launches = mlx.engine.stats().dispatches - before;
        const rot = try absorb(&o, q, attn, g, null, null, null);
        try testing.expectEqual(@as(u64, 1), launches);
        try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(try hostRows(&o, rot.q_lat)), std.mem.sliceAsBytes(try hostRows(&o, raw.q_lat)));
        try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(try hostRows(&o, rot.q_pe)), std.mem.sliceAsBytes(try hostRows(&o, raw.q_pe)));
    }
}

test "hy4 attention span into a chunk's block: a one-row span's gated row lands in the block at its row and the span returns none, at the release's 64 and 32 heads" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const exact = mlx.engine.exactF32();
    defer exact.end();
    for ([_]c_int{ 64, 32 }, 0..) |heads, ci| {
        const g: AttnGeometry = .{ .heads = heads, .nope = 192, .rope = 64, .v = 256, .latent = 512, .theta = 1e4, .eps = 1e-5 };
        var o: Ops = .{ .s = es };
        defer o.deinit();
        const seed: u64 = 53 * (ci + 1);
        const u_lat = try ihcValues(&o, &.{ heads, 1, 512 }, seed + 1, 1.5);
        const gate = try ihcValues(&o, &.{ 1, heads * 256 }, seed + 2, 2.0);
        var attn: Attn = undefined;
        attn.kv_b = try o.cast(try ihcValues(&o, &.{ heads * 448, 512 }, seed + 3, 0.2), .bfloat16);
        attn.o = Lin.dense(try o.cast(try ihcValues(&o, &.{ 64, heads * 256 }, seed + 4, 0.1), .bfloat16));
        for ([_]Arr{ u_lat, gate, attn.kv_b }) |a| try mlx.check(mlx.mlx_array_eval(a));
        const want = try o.own(try hy4_chain.gated(es, try o.own(try hy4_absorb.valueOut(es, u_lat, attn.kv_b, absorbGeometry(g))), gate));
        const block = try ihcValues(&o, &.{ 4, heads * 256 }, seed + 5, 1.0);
        try mlx.check(mlx.mlx_array_eval(block));
        const got = try gatedOut(&o, gate, u_lat, null, attn, g, null, false, null, .{ .block = block, .row = 2 }, null);
        try testing.expectEqual(@as(?*anyopaque, null), got.ctx);
        const rows = try hostRows(&o, block);
        const w = try hostRows(&o, want);
        try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(w), std.mem.sliceAsBytes(rows[@intCast(2 * heads * 256)..][0..w.len]));
    }
}
