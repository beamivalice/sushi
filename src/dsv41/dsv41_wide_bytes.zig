//! The bytes a wide forward (more rows than a verify block) holds in the engine's pool at its fullest, tensor by tensor
//! from the shapes the forward makes, each at its pool size class. A layer's intermediates live in its scope to the
//! layer's end, and its attention's and its routed sum's in child scopes of their own, so the fullest point is the end
//! of one layer's attention or of its routed chain, at the layer that holds the most. `Config.forwardCap` bills it.
const sushi_exl3 = @import("sushi_exl3");
const pool_class = @import("../tensor/pool_class.zig");
const cfg_mod = @import("dsv41_config.zig");
const Config = cfg_mod.Config;

/// The arms `c` bills: a config no parse or load gave them is a bug, never this GPU's guess.
fn armsOf(c: *const Config) cfg_mod.WideArms {
    return c.wide_arms orelse @panic("dsv41: a wide forward priced on a config with no arms (Config.wide_arms)");
}

/// The activation dtype's bytes: the streams, the projections and the routed chain's planes are bf16 or f16.
const act: u64 = 2;
const f32b: u64 = 4;
const i32b: u64 = 4;
/// Rows a window of the routed chain's sorted GEMM covers.
const gemm_win: u64 = @intCast(sushi_exl3.kernels.GEMM_WINDOW_ROWS);
/// The pools the fused routed chain binds; a route over more takes the combined chain.
const direct_planes: u64 = sushi_exl3.rate_groups.DIRECT_PLANES;

/// Sums tensors at their pool size class.
const Tally = struct {
    bytes: u64 = 0,

    fn t(self: *Tally, count: u64, len: u64) void {
        self.bytes += count * pool_class.poolBytes(@intCast(len));
    }
};

/// The most the pool holds in use over one wide forward of `rows` rows ending at position `seq`, past what was in
/// use when it began: the widest layer's peak.
pub fn bytes(c: *const Config, rows: u64, seq: u64) u64 {
    // A streamed route reads each rate group from its cache pool and its spill pool: the routing worst case. A resident
    // plan's route reads the cache pools alone.
    return bytesAt(c, rows, seq, (if (c.spill_pools) @as(u64, 2) else 1) * @as(u64, c.rate_groups));
}

/// `bytes` with the routed chain of STQ experts over `pools` pools.
pub fn bytesAt(c: *const Config, rows: u64, seq: u64, pools: u64) u64 {
    const p0 = seq -| rows;
    var most: u64 = 0;
    for (0..c.layers) |li| most = @max(most, layerPeak(c, li, rows, p0, pools));
    if (c.cedSplit()) |split| most = @max(most, cedCompressPeak(c, split, rows, p0));
    return most;
}

/// Layer `li`'s fullest point: its attention's end or its routed chain's, over what the pass and the layer hold.
fn layerPeak(c: *const Config, li: usize, rows: u64, p0: u64, pools: u64) u64 {
    // CED: a chunk wider than the window runs the decoder over its last `window` rows.
    const split = c.cedSplit();
    const L = if (split != null and li >= split.? and rows > c.window) c.window else rows;
    var base: Tally = .{};
    passHeld(&base, c, li, rows, p0);
    if (c.engram.layerAt(li) != null) engramWrite(&base, c, L);
    hcPre(&base, c, L, false);
    const attn = attention(c, li, L, if (L == rows) p0 else p0 + rows - L);
    // The attention's output, the stream it expands into, the next mixes and the router stay to the layer's end.
    var ffn: Tally = .{};
    openGroup(&ffn, c, li);
    ffn.t(1, L * c.dim * act);
    // The window rows `kv`: a ring write's source view keeps them in use until the layer's pass runs.
    ffn.t(1, L * c.head_dim * act);
    hcPre(&ffn, c, L, true);
    router(&ffn, c, L);
    // The layer's last expand, once the routed chain's scope has gone: the routed sum it keeps, the new stream, and
    // the chain's planes where the fused expand does not serve. A wide forward never folds it into the routed chain.
    var last: Tally = .{};
    last.t(1, L * c.dim * act);
    last.t(1, L * c.hc_mult * c.dim * act);
    expandChain(&last, c, L);
    return base.bytes + @max(attn, ffn.bytes + @max(routed(c, L, pools), last.bytes));
}

/// The split layer's compressor over every row of a CED chunk, before the decoder keeps its last rows.
fn cedCompressPeak(c: *const Config, split: usize, rows: u64, p0: u64) u64 {
    var t: Tally = .{};
    passHeld(&t, c, split, rows, p0);
    hcPre(&t, c, rows, false);
    compress(&t, c, split, rows);
    return t.bytes;
}

/// What the pass holds while layer `li` runs: the streams the pipeline keeps in flight, the collapse weights, each
/// Engram layer's rows read so far, the positions and picks the layers share, the state rows the layers so far
/// appended, and the trellis GEMM's row tables.
fn passHeld(t: *Tally, c: *const Config, li: usize, rows: u64, p0: u64) void {
    const hc: u64 = c.hc_mult;
    t.t(@max(1, @min(li, cfg_mod.prefill_pipe_depth)), rows * hc * c.dim * act);
    t.t(1, rows * hc * f32b);
    var engram_read: u64 = 0;
    for (c.engram.layers[0..c.engram.n]) |l| engram_read += @intFromBool(l <= li);
    t.t(engram_read, rows * c.engram.cols() * c.engram.head_dim * f32b);
    // `Shared.index` keeps the first eight distinct position and length arrays: the chunk's, the groups', then two a span.
    var arrays: u64 = 1;
    t.t(1, rows * i32b);
    var grouped = false;
    var indexed = false;
    var ratio: u64 = 1;
    for (0..li + 1) |l| {
        if (c.ratios[l] == 0) continue;
        if (!grouped and c.isKvSource(l)) {
            grouped = true;
            arrays += 1;
            t.t(1, groupsOf(rows, c.ratios[l]) * i32b);
        }
        if (c.isIndexSource(l)) {
            indexed = true;
            ratio = c.ratios[l];
        }
    }
    if (indexed) {
        var a: u64 = 0;
        while (a < rows) : (a += cfg_mod.attn_span) {
            const ls = @min(cfg_mod.attn_span, rows - a);
            if (arrays < 8) {
                t.t(@min(2, 8 - arrays), ls * i32b);
                arrays += 2;
            }
            // Each span's index top-k from the latest index source, kept for the layers after it.
            t.t(1, ls * @min(c.index_topk, (p0 + a + ls) / ratio) * i32b);
        }
    }
    for (0..li) |l| openGroup(t, c, l);
    for (0..li + 1) |l| {
        if (c.ratios[l] == 0) continue;
        const groups = groupsOf(rows, c.ratios[l]);
        if (c.isKvSource(l)) packedRows(t, groups, c.head_dim, 16);
        if (c.isOwner(l)) packedRows(t, groups, c.index_head_dim, 32);
    }
    if (c.trunk_dec != null) {
        // `linear.gemmInputs`: each row's bank and the window table, for one bank and for `wo_a`'s groups.
        const wins = (rows + gemm_win - 1) / gemm_win;
        t.t(1, rows * i32b);
        t.t(2, wins * i32b);
        t.t(1, c.o_groups * rows * i32b);
        t.t(2, c.o_groups * wins * i32b);
    }
}

/// A ratio > 1 compressor's open group, which a wide forward copies out of its projections: `[r - 1, hd]` f32 twice,
/// at the worst phase.
fn openGroup(t: *Tally, c: *const Config, l: usize) void {
    const r: u64 = c.ratios[l];
    if (r > 1 and c.isKvSource(l)) t.t(2, (r - 1) * c.head_dim * f32b);
}

/// FP4 codes `[n, w/2]` and one scale byte per `group`.
fn packedRows(t: *Tally, n: u64, w: u64, group: u64) void {
    t.t(1, n * w / 2);
    t.t(1, n * w / group);
}

/// The gated write of the n-gram rows (`engramApply`): the rows as one bf16 row each, `wkv`'s keys and value and the new
/// stream; the gate kernel's three constants, or the op chain's eight f32 copies of the stream, the f32 value, sixteen
/// per-copy f32 scalars, the sign mask and four constants.
fn engramWrite(t: *Tally, c: *const Config, L: u64) void {
    const hc: u64 = c.hc_mult;
    t.t(1, L * c.engram.cols() * c.engram.head_dim * act);
    t.t(1, L * (hc + 1) * c.dim * act);
    if (armsOf(c).engram_fused) {
        t.t(1, L * hc * c.dim * act);
        t.t(1, 3 * f32b);
        return;
    }
    t.t(8, L * hc * c.dim * f32b);
    t.t(1, L * c.dim * f32b);
    t.t(1, L * hc * c.dim * act);
    t.t(16, L * hc * f32b);
    t.t(1, L * hc);
    t.t(4, 4);
}

/// `hcExpandOps`'s planes beside the stream it makes, where the fused expand does not serve (`WideArms.expand_fused`):
/// the stream's f32 copy, the product, the matmul and their sum `[L,hc,D]`, and the output's f32 row `[L,D]`.
fn expandChain(t: *Tally, c: *const Config, L: u64) void {
    if (armsOf(c).expand_fused) return;
    t.t(4, L * c.hc_mult * c.dim * f32b);
    t.t(1, L * c.dim * f32b);
}

/// A stage's hyper-connection mixes on a wide block: `[L, (2 + hc) hc]` mixes, the split, the normed input, its eps; with
/// `expand`, the stream expanded by the attention's output first, with the expand's chain (`expandChain`), which the
/// layer's scope holds to its end.
fn hcPre(t: *Tally, c: *const Config, L: u64, expand: bool) void {
    const hc: u64 = c.hc_mult;
    if (expand) {
        t.t(1, L * hc * c.dim * act);
        expandChain(t, c, L);
    }
    t.t(1, L * (2 + hc) * hc * f32b);
    t.t(2, L * hc * f32b);
    t.t(1, L * hc * hc * f32b);
    t.t(1, L * c.dim * act);
    t.t(1, 8);
}

/// The router: the f32 input and weight, a tile of `gate_tile_rows` logits at a time and their concat, the picks and
/// their weights.
fn router(t: *Tally, c: *const Config, L: u64) void {
    const E: u64 = c.experts;
    t.t(1, L * c.dim * f32b);
    t.t(1, E * c.dim * f32b);
    const T = cfg_mod.gate_tile_rows;
    var lo: u64 = 0;
    while (lo < L) : (lo += T) t.t(1, @min(T, L - lo) * E * f32b);
    if (L > T) t.t(1, L * E * f32b);
    t.t(2, L * c.top_k * 4);
    t.t(1, 4);
}

/// The attention's own scope at its fullest: the end of its last span's heads, or of its output projections.
fn attention(c: *const Config, li: usize, L: u64, p0: u64) u64 {
    const H: u64 = c.heads;
    const hd: u64 = c.head_dim;
    var t: Tally = .{};
    // `wq_a` and `q_norm`, `wkv` and the rotated window rows, `wq_b` and the rotated queries, the rotation's eps.
    t.t(2, L * c.q_lora * act);
    t.t(2, L * hd * act);
    t.t(2, L * H * hd * act);
    t.t(1, 4);
    compress(&t, c, li, L);
    const n_spans = (L + cfg_mod.attn_span - 1) / cfg_mod.attn_span;
    const last = (n_spans - 1) * cfg_mod.attn_span;
    const grouped = armsOf(c).grouped_heads and fusedHeads(c) and c.rope_dim > 0;
    // Every span's heads stay to the output projections.
    var heads: Tally = .{};
    var span: u64 = 0;
    if (grouped) {
        heads.t(1, L * H * hd * act);
        var start: u64 = 0;
        while (start < L) : (start += cfg_mod.attn_span) {
            span = @max(span, spanBytes(c, li, L, p0, start, true));
        }
    } else {
        var a: u64 = 0;
        while (a < last) : (a += cfg_mod.attn_span) heads.t(1, cfg_mod.attn_span * H * hd * act);
        span = spanBytes(c, li, L, p0, last, false);
    }
    var out: Tally = .{};
    if (!grouped) out.t(1, L * H * hd * act);
    if (!fusedHeads(c)) {
        // The composed arm: the heads concatenated, the tail rotated back, the fp8 round trip, `wo_a`'s rows.
        out.t(1, L * H * hd * act);
    }
    out.t(1, L * c.o_groups * c.o_lora * act);
    out.t(1, L * c.dim * act);
    const last_heads = if (grouped) 0 else (L - last) * H * hd * act;
    return t.bytes + heads.bytes + @max(span, last_heads + out.bytes);
}

/// Whether the wide heads come from the one-pass kernel with its store's epilogue (`attnFusedFits`, `attnEpilogueFits`).
fn fusedHeads(c: *const Config) bool {
    return c.heads % 16 == 0 and c.head_dim % 256 == 0 and (c.heads * c.head_dim / c.o_groups) % 32 == 0;
}

/// The groups a ratio-`r` compressor closes over `L` rows at the worst phase of its cycle: `r - 1` rows pending.
fn groupsOf(L: u64, r: u64) u64 {
    return (L + r - 1) / r;
}

/// The compressor of a KV source over `L` rows at the worst phase of its cycle: the projections joined to the pending
/// rows and the new pending rows copied out, the pooled latents, the index keys an owner publishes, the roped rows, and
/// their FP4 codes.
fn compress(t: *Tally, c: *const Config, li: usize, L: u64) void {
    const r: u64 = c.ratios[li];
    if (r == 0 or !c.isKvSource(li)) return;
    const hd: u64 = c.head_dim;
    const g = groupsOf(L, r);
    if (r == 1) {
        t.t(2, L * hd * act);
    } else {
        t.t(1, L * c.dim * f32b);
        t.t(2, L * hd * f32b);
        t.t(2, (L + r - 1) * hd * f32b);
        openGroup(t, c, li);
        t.t(2, g * r * hd * f32b);
        t.t(1, g * hd * f32b);
        t.t(2, g * hd * act);
    }
    if (c.isOwner(li)) {
        t.t(3, g * c.index_head_dim * act);
        packedRows(t, g, c.index_head_dim, 32);
    }
    t.t(1, g * hd * act);
    packedRows(t, g, hd, 16);
}

/// One span's scope at its heads' end: the window band, the indexer's scores against every key (or one tile of a
/// streamed scan), the picks, and the head kernel's or the composed core's intermediates.
fn spanBytes(c: *const Config, li: usize, L: u64, p0: u64, a: u64, grouped: bool) u64 {
    const H: u64 = c.heads;
    const hd: u64 = c.head_dim;
    const w: u64 = c.window;
    const b = @min(L, a + cfg_mod.attn_span);
    const ls = b - a;
    const lo = if (p0 + a >= w - 1) p0 + a - (w - 1) else 0;
    const wk = p0 + b - lo;
    var t: Tally = .{};
    if (lo < p0) {
        t.t(1, (p0 - lo) * hd * act);
        t.t(1, (p0 - lo) * i32b);
        t.t(1, wk * hd * act);
    }
    t.t(2, wk * i32b);
    t.t(2, ls * i32b);
    t.t(3, ls * wk);
    t.t(1, ls * i32b);
    t.t(3, 4);
    var K: u64 = 0;
    const r: u64 = c.ratios[li];
    if (r > 0) {
        const nb = (p0 + b) / r;
        K = @min(c.index_topk, nb);
        if (c.isIndexSource(li) and nb > 0) {
            const ih: u64 = c.index_heads;
            const ihd: u64 = c.index_head_dim;
            const keys = @min(nb, cfg_mod.index_stream_tile);
            t.t(1, ls * ih * ihd * act);
            t.t(1, ls * ih * ihd * f32b);
            t.t(1, ls * ih * act);
            t.t(1, 4);
            t.t(1, keys * ihd * act);
            t.t(1, ls * ih * ihd * act);
            t.t(2, 12);
            t.t(1, ls * keys * f32b);
            t.t(1, 4);
            if (nb > cfg_mod.index_stream_tile) {
                // The streamed scan's running top-k and its fold: the tile's picks and values, both lists twice as long.
                t.t(4, ls * K * 4);
                t.t(2, ls * 2 * K * 4);
                t.t(2, ls * K * 4);
            }
        }
        if (!fusedHeads(c)) {
            t.t(1, ls * K * hd * act);
            t.t(1, ls * K * i32b);
        }
    }
    if (fusedHeads(c)) {
        t.t(1, 28);
        t.t(2, 4);
        if (!grouped) t.t(1, ls * H * hd * act);
        return t.bytes;
    }
    // `attnCore`: f32 queries and keys, the logits over the window and the picks, the softmax, both products.
    t.t(1, ls * H * hd * f32b);
    t.t(1, wk * hd * f32b);
    t.t(1, ls * H * wk * f32b);
    if (K > 0) {
        t.t(1, ls * K * i32b);
        t.t(2, 4);
        t.t(1, ls * K * hd * act);
        t.t(1, ls * K * hd * f32b);
        t.t(1, ls * H * K * f32b);
        t.t(1, ls * H * (wk + K) * f32b);
        t.t(1, ls * K);
        t.t(1, ls * (wk + K));
    }
    t.t(2, 4);
    t.t(4, ls * H * (wk + K) * f32b);
    t.t(6, ls * H * f32b);
    t.t(if (K > 0) 4 else 2, ls * H * hd * f32b);
    t.t(1, ls * H * hd * act);
    return t.bytes;
}

/// The FFN stage's own scope at its routed chain's fullest.
fn routed(c: *const Config, L: u64, pools: u64) u64 {
    if (c.expert_dec == null) return fp4Routed(c, L);
    const fused = fusedChain(c, L, @min(pools, direct_planes));
    return if (pools > direct_planes) @max(fused, combinedChain(c, L, pools)) else fused;
}

/// `rate_groups.moePrefillSharedFusedFrom` with `p` pools: the sort, the shared expert riding the launches, a gate, an
/// up and a down plane per pool over every sorted row, and the f32 sum.
fn fusedChain(c: *const Config, L: u64, p: u64) u64 {
    const D: u64 = c.dim;
    const I: u64 = c.moe_inter;
    const n = L * c.top_k;
    var t: Tally = .{};
    t.t(1, n * i32b);
    t.t(2, n * 4);
    t.t(1, L * 4);
    t.t(2, L * D * act);
    t.t(2, n * D * act);
    t.t(2, L * I * act);
    windows(&t, c, n, p);
    t.t(2 * p, n * I * act);
    t.t(1, n * I * act);
    t.t(1, L * I * act);
    t.t(1, L * D * act);
    t.t(p, n * D * act);
    t.t(1, L * D * f32b);
    t.t(1, 4);
    return t.bytes;
}

/// `rate_groups.moePrefillFrom` with `p` pools (more than the fused chain binds), after the shared expert's op chain:
/// the gate and up planes each combined, the SwiGLU rows, and the down planes beside their combine.
fn combinedChain(c: *const Config, L: u64, p: u64) u64 {
    const D: u64 = c.dim;
    const I: u64 = c.moe_inter;
    const n = L * c.top_k;
    var t: Tally = .{};
    sharedOps(&t, c, L);
    t.t(1, n * i32b);
    t.t(2, n * 4);
    t.t(2, n * D * act);
    windows(&t, c, n, p);
    t.t(3, n * I * act);
    t.t(p + 1, n * D * act);
    return t.bytes;
}

/// The window table of `p` pools over `n` sorted rows, and each row's inverse and local slot.
fn windows(t: *Tally, c: *const Config, n: u64, p: u64) void {
    t.t(2, (p * ((n + gemm_win - 1) / gemm_win) + c.experts) * 4);
    t.t(2, n * 4);
}

/// The shared expert as ops: gate and up, eight f32 rows of the clamped SwiGLU, its bf16 rows, the down projection.
fn sharedOps(t: *Tally, c: *const Config, L: u64) void {
    const I: u64 = c.moe_inter;
    t.t(2, L * I * act);
    t.t(8, L * I * f32b);
    t.t(1, L * I * act);
    t.t(1, L * c.dim * act);
    t.t(3, 4);
}

/// `moeWide` over FP4 experts beside the shared expert's ops: the slots, the sort and its inverse, the gathered rows,
/// the gate, up, SwiGLU and down rows over every sorted row, and the combined sum.
fn fp4Routed(c: *const Config, L: u64) u64 {
    const D: u64 = c.dim;
    const I: u64 = c.moe_inter;
    const n = L * c.top_k;
    var t: Tally = .{};
    sharedOps(&t, c, L);
    t.t(6, n * 4);
    t.t(1, n * D * act);
    t.t(3, n * I * act);
    t.t(1, n * D * act);
    t.t(1, L * D * act);
    return t.bytes;
}

test "dsv41 wide attention prices one grouped head buffer across prefill spans" {
    const c = try cfg_mod.releaseStqForTest();
    const rows = 8192;
    const input_bytes = 2 * rows * (1280 + 512 + 64 * 512) * 2 + pool_class.page_bytes;
    const head_bytes = rows * 64 * 512 * 2;
    const projection_bytes = rows * (8 * 1024 + 5120) * 2;
    try @import("std").testing.expectEqual(@as(u64, input_bytes + head_bytes + projection_bytes), attention(&c, 0, rows, 0));
}
