//! Hy4-Flash's composed forward held to transformers' on the tiny (`HY4_BF16_TINY`, tests/dump_hy4_fixtures.py),
//! part by part: each piece is fed the oracle's own inputs and its output held to the oracle's, on the CPU stream in
//! f32, within the doc's bars (every trace within 1e-4 of its peak, the logits within 2e-3; ids and picks identical).
const std = @import("std");
const testing = std.testing;
const mlx = @import("../mlx.zig");
const model_mod = @import("../model.zig");
const Ops = @import("../mlx_ops.zig").Ops;
const fwd = @import("hy4_forward.zig");
const hy4_config = @import("hy4_config.zig");
const Arr = mlx.mlx_array;

const trace_bar = 1e-4;

const Tiny = struct {
    pack: model_mod.Weights,
    oracle: model_mod.Weights,
    config: hy4_config.Config,
    cpu: mlx.mlx_stream,

    fn open() !Tiny {
        const dir = std.mem.span(std.c.getenv("HY4_BF16_TINY") orelse return error.SkipZigTest);
        const a = testing.allocator;
        var t: Tiny = .{ .pack = undefined, .oracle = undefined, .config = undefined, .cpu = mlx.mlx_default_cpu_stream_new() };
        // The oracle is transformers' f32 forward: the products are held to it at f32 (`fwd.products`).
        fwd.products = .f32;
        errdefer _ = mlx.mlx_stream_free(t.cpu);
        t.pack = try load(dir, "model.safetensors", t.cpu);
        errdefer t.pack.deinit();
        t.oracle = try load(dir, "oracle.safetensors", t.cpu);
        errdefer t.oracle.deinit();
        const io = std.Io.Threaded.global_single_threaded.io();
        var d = try std.Io.Dir.openDirAbsolute(io, dir, .{});
        defer d.close(io);
        const text = try d.readFileAlloc(io, "config.json", a, .limited(1 << 20));
        defer a.free(text);
        t.config = try hy4_config.parseText(text);
        return t;
    }

    fn close(t: *Tiny) void {
        fwd.products = .bf16;
        t.pack.deinit();
        t.oracle.deinit();
        _ = mlx.mlx_stream_free(t.cpu);
    }

    fn w(t: *const Tiny, comptime fmt: []const u8, args: anytype) !Arr {
        var buf: [160]u8 = undefined;
        const name = try std.fmt.bufPrint(&buf, fmt, args);
        return t.pack.get(name) orelse {
            std.debug.print("the tiny pack has no {s}\n", .{name});
            return error.MissingTensor;
        };
    }

    fn ref(t: *const Tiny, comptime fmt: []const u8, args: anytype) !Arr {
        var buf: [96]u8 = undefined;
        const name = try std.fmt.bufPrint(&buf, fmt, args);
        return t.oracle.get(name) orelse {
            std.debug.print("the oracle has no {s}\n", .{name});
            return error.MissingTensor;
        };
    }

    fn mixOf(t: *const Tiny, layer: usize, comptime which: []const u8) !fwd.Mix {
        return .{
            .fn_ = try t.w("model.layers.{d}." ++ which ++ ".hc_pre.hc_fn", .{layer}),
            .base = try t.w("model.layers.{d}." ++ which ++ ".hc_pre.hc_base", .{layer}),
            .scale = try t.w("model.layers.{d}." ++ which ++ ".hc_pre.hc_scale", .{layer}),
        };
    }
};

fn load(dir: []const u8, name: []const u8, s: mlx.mlx_stream) !model_mod.Weights {
    const path = try std.fmt.allocPrintSentinel(testing.allocator, "{s}/{s}", .{ dir, name }, 0);
    defer testing.allocator.free(path);
    var w = model_mod.Weights.init(testing.allocator);
    errdefer w.deinit();
    try model_mod.loadSafetensorsFile(testing.allocator, &w, path.ptr, s, true);
    return w;
}

/// `got` against `want` as f32 rows: the largest |difference| over `want`'s peak magnitude must be within `bar`.
fn expectNear(label: []const u8, o: *Ops, got: Arr, want: Arr, bar: f64) !void {
    const g = try host(o, got);
    const r = try host(o, want);
    if (g.len != r.len) {
        std.debug.print("{s}: {d} values, the oracle has {d}\n", .{ label, g.len, r.len });
        return error.ShapeMismatch;
    }
    var peak: f64 = 1e-30;
    for (r) |v| peak = @max(peak, @abs(v));
    var worst: f64 = 0;
    var at: usize = 0;
    for (g, r, 0..) |x, y, i| {
        const e = @abs(@as(f64, x) - y);
        if (std.math.isNan(e)) {
            worst = std.math.inf(f64);
            at = i;
            break;
        }
        if (e > worst) {
            worst = e;
            at = i;
        }
    }
    if (worst / peak > bar) {
        std.debug.print("{s}: worst |Δ| {e} at {d} ({d} vs {d}) is {e} of the peak {e}, over {e}\n", .{ label, worst, at, g[at], r[at], worst / peak, peak, bar });
        return error.OutsideBar;
    }
}

fn host(o: *Ops, a: Arr) ![]const f32 {
    const c = try o.contiguous(try o.cast(a, .float32));
    try mlx.check(mlx.mlx_array_eval(c));
    const p = mlx.mlx_array_data_float32(c) orelse return error.HostRead;
    return p[0..mlx.mlx_array_size(c)];
}

test "hy4 oracle: each layer's iHC mixes, the residual write-back and the head's collapse match transformers' (HY4_BF16_TINY)" {
    var t = try Tiny.open();
    defer t.close();
    const c = t.config;
    const n: c_int = @intCast(c.hc_mult);
    for (0..c.layers) |l| {
        var o: Ops = .{ .s = t.cpu };
        defer o.deinit();
        var label: [48]u8 = undefined;
        // The attention mix reads the layer's streams after Engram's delta, where the layer has a module.
        var h = try t.ref("l{d}.in", .{l});
        if (t.oracle.get(try std.fmt.bufPrint(&label, "l{d}.engram", .{l}))) |delta| h = try o.binary(.add, h, delta);
        const a = try fwd.mix(&o, h, try t.mixOf(l, "hc_attn_layer"), n, c.hc_magnitude, c.hc_eps, c.eps);
        try expectNear(try std.fmt.bufPrint(&label, "l{d}.attn_hc.post", .{l}), &o, a.post, try t.ref("l{d}.attn_hc.post", .{l}), trace_bar);
        try expectNear(try std.fmt.bufPrint(&label, "l{d}.attn_hc.x", .{l}), &o, a.x, try t.ref("l{d}.attn_hc.x", .{l}), trace_bar);
        // The FFN mix reads the streams the attention output was written back into, from the oracle's own pieces.
        const mid = try fwd.expand(&o, h, try t.ref("l{d}.attn_hc.post", .{l}), try t.ref("l{d}.attn.out", .{l}));
        const f = try fwd.mix(&o, mid, try t.mixOf(l, "hc_mlp_layer"), n, c.hc_magnitude, c.hc_eps, c.eps);
        try expectNear(try std.fmt.bufPrint(&label, "l{d}.ffn_hc.post", .{l}), &o, f.post, try t.ref("l{d}.ffn_hc.post", .{l}), trace_bar);
        try expectNear(try std.fmt.bufPrint(&label, "l{d}.ffn_hc.x", .{l}), &o, f.x, try t.ref("l{d}.ffn_hc.x", .{l}), trace_bar);
        const out = try fwd.expand(&o, mid, try t.ref("l{d}.ffn_hc.post", .{l}), try t.ref("l{d}.ffn", .{l}));
        try expectNear(try std.fmt.bufPrint(&label, "l{d}.out", .{l}), &o, out, try t.ref("l{d}.out", .{l}), trace_bar);
    }
    var o: Ops = .{ .s = t.cpu };
    defer o.deinit();
    const last = try t.ref("l{d}.out", .{c.layers - 1});
    const hw: fwd.Mix = .{ .fn_ = try t.w("model.hc_head.hc_head_fn", .{}), .base = try t.w("model.hc_head.hc_head_base", .{}), .scale = try t.w("model.hc_head.hc_head_scale", .{}) };
    try expectNear("hc_head", &o, try fwd.head(&o, last, hw, n, c.hc_eps, c.eps), try t.ref("hc_head", .{}), trace_bar);
}

test "hy4 oracle: the dense layer 0 and each latent MoE layer, picks and output, match transformers' (HY4_BF16_TINY)" {
    var t = try Tiny.open();
    defer t.close();
    const c = t.config;
    for (0..c.layers) |l| {
        var o: Ops = .{ .s = t.cpu };
        defer o.deinit();
        var label: [48]u8 = undefined;
        const x = try fwd.norm(&o, try t.ref("l{d}.ffn_hc.x", .{l}), try t.w("model.layers.{d}.post_attention_layernorm.weight", .{l}), c.eps);
        if (l < c.dense_layers) {
            const sw: fwd.Swiglu = .{ .gate = fwd.Lin.dense(try t.w("model.layers.{d}.mlp.gate_proj.weight", .{l})), .up = fwd.Lin.dense(try t.w("model.layers.{d}.mlp.up_proj.weight", .{l})), .down = fwd.Lin.dense(try t.w("model.layers.{d}.mlp.down_proj.weight", .{l})) };
            const y = try fwd.norm(&o, try fwd.swiglu(&o, x, sw, null), try t.w("model.layers.{d}.mlp.post_fc2_norm.weight", .{l}), c.eps);
            try expectNear(try std.fmt.bufPrint(&label, "l{d}.ffn (dense)", .{l}), &o, y, try t.ref("l{d}.ffn", .{l}), trace_bar);
            continue;
        }
        const m: fwd.Moe = .{
            .router = .{ .weight = try t.w("model.layers.{d}.mlp.gate.weight", .{l}), .bias = try t.w("model.layers.{d}.mlp.gate.e_score_correction_bias", .{l}) },
            .gate_up = try t.w("model.layers.{d}.mlp.experts.gate_up_proj", .{l}),
            .down = try t.w("model.layers.{d}.mlp.experts.down_proj", .{l}),
            .lat_down = fwd.Lin.dense(try t.w("model.layers.{d}.mlp.routed_expert_down_proj.weight", .{l})),
            .lat_up = fwd.Lin.dense(try t.w("model.layers.{d}.mlp.routed_expert_up_proj.weight", .{l})),
            .shared = .{ .gate = fwd.Lin.dense(try t.w("model.layers.{d}.mlp.shared_experts.gate_proj.weight", .{l})), .up = fwd.Lin.dense(try t.w("model.layers.{d}.mlp.shared_experts.up_proj.weight", .{l})), .down = fwd.Lin.dense(try t.w("model.layers.{d}.mlp.shared_experts.down_proj.weight", .{l})) },
            .combine = try t.w("model.layers.{d}.mlp.combine_output_norm.weight", .{l}),
        };
        const r = try fwd.moe(testing.allocator, &o, x, m, .{ .top_k = @intCast(c.top_k), .inter = @intCast(c.moe_inter), .scale = c.route_scale, .swiglu_limit = c.swiglu_limit, .eps = c.eps }, null);
        try expectSamePicks(try std.fmt.bufPrint(&label, "l{d}.router.picks", .{l}), &o, r.routed.picks, try t.ref("l{d}.router.picks", .{l}), @intCast(c.top_k));
        try expectNear(try std.fmt.bufPrint(&label, "l{d}.ffn (moe)", .{l}), &o, r.y, try t.ref("l{d}.ffn", .{l}), trace_bar);
    }
}

/// Each row's `k` picks equal the oracle's as a set (the reference's top-k is unsorted).
fn expectSamePicks(label: []const u8, o: *Ops, got: Arr, want: Arr, k: usize) !void {
    const g = try host(o, got);
    const r = try host(o, want);
    if (g.len != r.len) return error.ShapeMismatch;
    var row: usize = 0;
    while (row * k < g.len) : (row += 1) {
        var a: [16]f32 = undefined;
        var b: [16]f32 = undefined;
        @memcpy(a[0..k], g[row * k ..][0..k]);
        @memcpy(b[0..k], r[row * k ..][0..k]);
        std.mem.sort(f32, a[0..k], {}, std.sort.asc(f32));
        std.mem.sort(f32, b[0..k], {}, std.sort.asc(f32));
        if (!std.mem.eql(f32, a[0..k], b[0..k])) {
            std.debug.print("{s}: row {d} picks {any}, the oracle {any}\n", .{ label, row, a[0..k], b[0..k] });
            return error.PicksDiffer;
        }
    }
}

test "hy4 oracle: each layer's attention, and on sparse layers the pooled keys and recalled groups, match transformers' (HY4_BF16_TINY)" {
    var t = try Tiny.open();
    defer t.close();
    const c = t.config;
    for (0..c.layers) |l| {
        var o: Ops = .{ .s = t.cpu };
        defer o.deinit();
        var label: [48]u8 = undefined;
        const x = try fwd.norm(&o, try t.ref("l{d}.attn_hc.x", .{l}), try t.w("model.layers.{d}.input_layernorm.weight", .{l}), c.eps);
        const rows = mlx.getShape(x)[0];
        const sparse = c.kinds[l] == .sparse;
        const p = "model.layers.{d}.self_attn.";
        const w: fwd.Attn = .{
            .q_a = fwd.Lin.dense(try t.w(p ++ "q_a_proj.weight", .{l})),
            .q_a_norm = try t.w(p ++ "q_a_layernorm.weight", .{l}),
            .q_b = fwd.Lin.dense(try t.w(p ++ "q_b_proj.weight", .{l})),
            .kv_a = fwd.Lin.dense(try t.w(p ++ "kv_a_proj_with_mqa.weight", .{l})),
            .kv_a_norm = try t.w(p ++ "kv_a_layernorm.weight", .{l}),
            .kv_b = try t.w(p ++ "kv_b_proj.weight", .{l}),
            .gate = fwd.Lin.dense(try t.w(p ++ "linear_gate.weight", .{l})),
            .o = fwd.Lin.dense(try t.w(p ++ "o_proj.weight", .{l})),
            .post_norm = try t.w(p ++ "post_proj_norm.weight", .{l}),
            .sinks = try t.w(p ++ "learnable_sink_param", .{l}),
        };
        const g: fwd.AttnGeometry = .{
            .heads = @intCast(c.heads(l)),
            .nope = @intCast(c.nope_dim),
            .rope = @intCast(c.rope_dim),
            .v = @intCast(c.v_dim),
            .latent = @intCast(c.kv_lora),
            .theta = @floatCast(if (sparse) c.sparse_rope.theta else c.sliding_rope.theta),
            .eps = c.eps,
        };
        var mask = try fwd.causalMask(&o, rows, rows, 0, if (sparse) 0 else @intCast(c.window));
        if (sparse) {
            const ip = "model.layers.{d}.self_attn.compressed_indexer.";
            const iw: fwd.Indexer = .{
                .query = try t.w(ip ++ "query_projection.weight", .{l}),
                .head_weight = try t.w(ip ++ "head_weight_projection.weight", .{l}),
                .token_key = try t.w(ip ++ "token_key_projection.weight", .{l}),
                .token_key_norm = try t.w(ip ++ "token_key_norm.weight", .{l}),
                .key_value = try t.w(ip ++ "key_value_projection.weight", .{l}),
                .key_gate = try t.w(ip ++ "key_gate_projection.weight", .{l}),
                .ape = try t.w(ip ++ "intra_group_ape", .{l}),
                .key_norm = try t.w(ip ++ "key_norm.weight", .{l}),
            };
            const ig: fwd.IndexerGeometry = .{
                .heads = @intCast(c.indexer.heads),
                .dim = @intCast(c.indexer.head_dim),
                .rope = @intCast(c.rope_dim),
                .ratio = @intCast(c.indexer.group),
                .group_topk = @intCast(c.indexer.group_topk),
                .theta = @floatCast(c.sparse_rope.theta),
                .eps = c.eps,
            };
            const pooled = try fwd.pooledKeys(&o, x, iw, ig);
            try expectNear(try std.fmt.bufPrint(&label, "l{d}.pooled", .{l}), &o, pooled, try t.ref("l{d}.pooled", .{l}), trace_bar);
            const q_resid = try fwd.norm(&o, try fwd.linear(&o, x, w.q_a), w.q_a_norm, c.eps);
            const rec = try fwd.recall(&o, x, q_resid, pooled, iw, ig);
            try expectSameRecall(try std.fmt.bufPrint(&label, "l{d}.recall", .{l}), &o, rec, try t.ref("l{d}.recall", .{l}), c.indexer.group, c.indexer.group_topk);
            // The attention reads the oracle's own recall: a tie the reference broke its own way (an exact-zero
            // score) selects other keys, so our order would move rows the attention is not at fault for.
            mask = try fwd.selectedMask(&o, rows, rows, 0, try t.ref("l{d}.recall", .{l}), @intCast(c.indexer.group));
        }
        const out = try fwd.attention(&o, x, w, g, 0, mask);
        try expectNear(try std.fmt.bufPrint(&label, "l{d}.attn.out", .{l}), &o, out.y, try t.ref("l{d}.attn.out", .{l}), trace_bar);
    }
}

/// Each row's recalled groups at or before its own are the oracle's up to ties: the logits our picks were chosen by
/// equal, as a multiset, the logits of the oracle's picks (ReLU leaves exact-zero scores, and the reference breaks a
/// tie by its own order). A row with fewer earlier groups than slots fills them past its own group, which the causal
/// mask drops whichever they are.
fn expectSameRecall(label: []const u8, o: *Ops, got: fwd.Recall, want: Arr, ratio: u32, k: u32) !void {
    const g = try host(o, got.groups);
    const r = try host(o, want);
    const logits = try host(o, got.logits);
    if (g.len != r.len) return error.ShapeMismatch;
    const rows = g.len / k;
    const groups = logits.len / rows;
    for (0..rows) |row| {
        const own: f32 = @floatFromInt(row / ratio);
        var a: [16]f32 = undefined;
        var b: [16]f32 = undefined;
        var na: usize = 0;
        var nb: usize = 0;
        for (g[row * k ..][0..k]) |v| if (v < own) {
            a[na] = logits[row * groups + @as(usize, @intFromFloat(v))];
            na += 1;
        };
        for (r[row * k ..][0..k]) |v| if (v < own) {
            b[nb] = logits[row * groups + @as(usize, @intFromFloat(v))];
            nb += 1;
        };
        std.mem.sort(f32, a[0..na], {}, std.sort.asc(f32));
        std.mem.sort(f32, b[0..nb], {}, std.sort.asc(f32));
        if (!std.mem.eql(f32, a[0..na], b[0..nb])) {
            std.debug.print("{s}: row {d} picks groups scored {any}, the oracle's {any}\n", .{ label, row, a[0..na], b[0..nb] });
            return error.RecallDiffers;
        }
    }
}

test "hy4 oracle: Engram's row ids, from the shared hash, and its delta match transformers' with llama.cpp's choices (HY4_BF16_TINY)" {
    const engram_mod = @import("../engram.zig");
    var t = try Tiny.open();
    defer t.close();
    const c = t.config;
    const e = c.engram;
    const a = testing.allocator;
    var o: Ops = .{ .s = t.cpu };
    defer o.deinit();
    const layer: usize = e.layers[0];
    // The token map: compressed ids, an id past the map hashing as itself (llama.cpp).
    const map = try host(&o, try t.w("model.layers.{d}.engram_layer.lookup_table", .{layer}));
    var c_max: u32 = 0;
    for (map) |v| c_max = @max(c_max, @as(u32, @intFromFloat(v)));
    const ids = try host(&o, try t.ref("ids", .{}));
    const compressed = try a.alloc(u32, ids.len);
    defer a.free(compressed);
    for (ids, compressed) |v, *dst| {
        const id: u32 = @intFromFloat(v);
        dst.* = if (id < map.len) @intFromFloat(map[id]) else id;
    }
    var geo: engram_mod.Geometry = .{ .n = 1, .seed = e.seed, .max_ngram = e.max_ngram, .vocab_size = e.vocab_size, .heads = e.heads, .head_dim = e.head_dim, .pad_id = e.pad_id, .compressed_vocab = c_max + 1 };
    geo.layers[0] = e.layers[0];
    geo.hash_ids[0] = e.hash_ids[0];
    geo.rows[0] = e.rows[0];
    const hash = try engram_mod.Hash.init(geo, @intFromFloat(map[e.pad_id]));
    const cols = hash.cols();
    const rows_out = try a.alloc(u32, ids.len * cols);
    defer a.free(rows_out);
    hash.rows(&.{}, compressed, rows_out);
    // Held to the oracle's ids, which carry no head offsets.
    const want = try host(&o, try t.ref("l{d}.engram.ids", .{layer}));
    for (0..ids.len) |i| for (0..cols) |col| {
        const got = rows_out[i * cols + col] - @as(u32, @intCast(hash.offsets[0][col]));
        const exp: u32 = @intFromFloat(want[i * cols + col]);
        if (got != exp) {
            std.debug.print("engram ids: position {d} head {d} is {d}, the oracle's {d}\n", .{ i, col, got, exp });
            return error.EngramIdsDiffer;
        }
    };
    // The rows as stored, then the delta, zero at the pad id.
    const idx = try o.own(mlx.mlx_array_new_data(rows_out.ptr, &[_]c_int{@intCast(rows_out.len)}, 1, .uint32));
    const table = try t.w("model.layers.{d}.engram_layer.multi_head_embedding.embedding.weight", .{layer});
    const rows = try o.reshape(try o.take(table, idx, 0), &.{ @intCast(ids.len), @intCast(cols), @intCast(e.head_dim) });
    const live_host = try a.alloc(f32, ids.len);
    defer a.free(live_host);
    for (ids, live_host) |v, *dst| dst.* = if (@as(u32, @intFromFloat(v)) == e.pad_id) 0 else 1;
    const live = try o.own(mlx.mlx_array_new_data(live_host.ptr, &[_]c_int{@intCast(ids.len)}, 1, .float32));
    var bufs: [6][8]Arr = undefined;
    const n: usize = c.hc_mult;
    const p = "model.layers.{d}.engram_layer.";
    for (0..cols) |j| bufs[0][j] = try t.w(p ++ "lookup_per_head_norms.{d}.weight", .{ layer, j });
    for (0..n) |ch| {
        bufs[1][ch] = try t.w(p ++ "key_projs.{d}.weight", .{ layer, ch });
        bufs[2][ch] = try t.w(p ++ "key_projs.{d}.bias", .{ layer, ch });
        bufs[3][ch] = try t.w(p ++ "norm_key.{d}.weight", .{ layer, ch });
        bufs[4][ch] = try t.w(p ++ "norm_query.{d}.weight", .{ layer, ch });
        bufs[5][ch] = try t.w(p ++ "unified_per_hc_scales.{d}", .{ layer, ch });
    }
    const w: fwd.Engram = .{
        .head_norms = bufs[0][0..cols],
        .key_w = bufs[1][0..n],
        .key_b = bufs[2][0..n],
        .norm_key = bufs[3][0..n],
        .norm_query = bufs[4][0..n],
        .scales = bufs[5][0..n],
        .value_w = try t.w(p ++ "value_proj.weight", .{layer}),
        .value_b = try t.w(p ++ "value_proj.bias", .{layer}),
        .gamma = try t.w(p ++ "unified_pre_gamma", .{layer}),
    };
    const delta = try fwd.engram(&o, try t.ref("l{d}.in", .{layer}), rows, live, w, null);
    try expectNear("l1.engram", &o, delta, try t.ref("l{d}.engram", .{layer}), trace_bar);
}

test "hy4 oracle: the whole model prefills the prompt and decodes greedily as transformers does (HY4_BF16_TINY)" {
    const hy4_model = @import("hy4_model.zig");
    var t = try Tiny.open();
    defer t.close();
    const m = try hy4_model.Model.init(testing.allocator, t.config, &t.pack, t.cpu);
    defer m.deinit();
    // transformers keeps the latent in full precision: the oracle is held at the BF16 width.
    m.req().kv = .dense;
    var o: Ops = .{ .s = t.cpu };
    defer o.deinit();
    const ids_f = try host(&o, try t.ref("ids", .{}));
    const ids = try testing.allocator.alloc(u32, ids_f.len);
    defer testing.allocator.free(ids);
    for (ids_f, ids) |v, *d| d.* = @intFromFloat(v);
    const prompt_len = 40;
    const all_logits = try t.ref("logits", .{});
    const vocab = mlx.getShape(all_logits)[1];
    const prefill = try o.own(try m.forward(ids[0..prompt_len]));
    try expectNear("prefill logits", &o, prefill, try o.slice(all_logits, 0, 0, prompt_len), 2e-3);
    var label: [32]u8 = undefined;
    // Each decode step feeds the token the oracle picked; ours must pick the same.
    for (prompt_len..ids.len) |i| {
        const step = try o.own(try m.forward(ids[i .. i + 1]));
        try expectNear(try std.fmt.bufPrint(&label, "decode logits at {d}", .{i}), &o, step, try o.slice(all_logits, 0, @intCast(i), @intCast(i + 1)), 2e-3);
        if (i + 1 < ids.len) {
            const row = try host(&o, step);
            var best: usize = 0;
            for (row, 0..) |v, j| if (v > row[best]) {
                best = j;
            };
            _ = vocab;
            try testing.expectEqual(ids[i + 1], @as(u32, @intCast(best)));
        }
    }
    // The prompt's last row picks the first decode token too.
    const last = try host(&o, try o.slice(prefill, 0, prompt_len - 1, prompt_len));
    var best: usize = 0;
    for (last, 0..) |v, j| if (v > last[best]) {
        best = j;
    };
    try testing.expectEqual(ids[prompt_len], @as(u32, @intCast(best)));
}

test "hy4 oracle: attention in spans of 7 rows, edges inside groups, the window and the selection threshold, gives the prompt's logits (HY4_BF16_TINY)" {
    const hy4_model = @import("hy4_model.zig");
    var t = try Tiny.open();
    defer t.close();
    var o: Ops = .{ .s = t.cpu };
    defer o.deinit();
    const ids_f = try host(&o, try t.ref("ids", .{}));
    var ids: [40]u32 = undefined;
    for (ids_f[0..40], &ids) |v, *d| d.* = @intFromFloat(v);
    const m = try hy4_model.Model.init(testing.allocator, t.config, &t.pack, t.cpu);
    defer m.deinit();
    // transformers keeps the latent in full precision: the oracle is held at the BF16 width.
    m.req().kv = .dense;
    m.span_rows = 7;
    const got = try o.own(try m.forward(&ids));
    try expectNear("spans of 7", &o, got, try o.slice(try t.ref("logits", .{}), 0, 0, 40), 2e-3);
}

test "hy4 oracle: a prefill in chunks of 1 (a decode row writes the ring first), 7 (groups, the window and the ring's wrap split mid-chunk) and 16 (wider than the window) gives the whole prompt's logits (HY4_BF16_TINY)" {
    const hy4_model = @import("hy4_model.zig");
    var t = try Tiny.open();
    defer t.close();
    var o: Ops = .{ .s = t.cpu };
    defer o.deinit();
    const ids_f = try host(&o, try t.ref("ids", .{}));
    var ids: [64]u32 = undefined;
    for (ids_f[0..40], ids[0..40]) |v, *d| d.* = @intFromFloat(v);
    const want = try t.ref("logits", .{});
    // The tiny's groups are 4 rows, its window 8: a chunk of 3 runs the paths a chunk of 7 does.
    for ([_]usize{ 1, 7, 16 }) |chunk| {
        const m = try hy4_model.Model.init(testing.allocator, t.config, &t.pack, t.cpu);
        defer m.deinit();
        // transformers keeps the latent in full precision: the oracle is held at the BF16 width.
        m.req().kv = .dense;
        var at: usize = 0;
        var label: [48]u8 = undefined;
        while (at < 40) {
            const end = @min(at + chunk, 40);
            const got = try o.own(try m.forward(ids[at..end]));
            try expectNear(try std.fmt.bufPrint(&label, "chunk {d} rows {d}..{d}", .{ chunk, at, end }), &o, got, try o.slice(want, 0, @intCast(at), @intCast(end)), 2e-3);
            at = end;
        }
    }
}

test "hy4 cooler: the whole model on Sushi Cooler's stream prefills and decodes as transformers does, past the window with each sliding row written into its ring in place (HY4_BF16_TINY)" {
    coolerWhole() catch |e| {
        if (e == error.SkipZigTest) return e;
        var said: [512]u8 = undefined;
        std.debug.print("\nthe Cooler run failed ({t}): {s}\n", .{ e, mlx.takeError(&said) orelse "(no message)" });
        if (@errorReturnTrace()) |trace| std.debug.dumpStackTrace(trace.*);
        return e;
    };
}

fn coolerWhole() !void {
    const hy4_model = @import("hy4_model.zig");
    try mlx.requireGpu();
    var t = try Tiny.open();
    defer t.close();
    // The oracle as host rows, read before the engine's scope: no MLX stream may run inside it.
    const want = try rawF32(try t.ref("logits", .{}));
    const ids_f = try rawF32(try t.ref("ids", .{}));
    var ids: [64]u32 = undefined;
    for (ids_f, ids[0..ids_f.len]) |v, *d| d.* = @intFromFloat(v);
    const vocab: usize = @intCast(mlx.getShape(try t.ref("logits", .{}))[1]);
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const m = try hy4_model.Model.init(testing.allocator, t.config, &t.pack, es);
    defer m.deinit();
    // transformers keeps the latent in full precision: the oracle is held at the BF16 width.
    m.req().kv = .dense;
    var eo: Ops = .{ .s = es };
    defer eo.deinit();
    const prefill = try host(&eo, try eo.own(try m.forward(ids[0..40])));
    try expectNearRows("cooler prefill logits", prefill, want[0 .. 40 * vocab], 2e-3);
    var label: [40]u8 = undefined;
    const core = @import("../tensor/core.zig");
    const before = core.stats();
    for (40..ids_f.len) |i| {
        const step = try host(&eo, try eo.own(try m.forward(ids[i .. i + 1])));
        try expectNearRows(try std.fmt.bufPrint(&label, "cooler decode logits at {d}", .{i}), step, want[i * vocab ..][0..vocab], 2e-3);
    }
    // Each decode step writes its row into every sliding layer's ring (the latent and the rope rows, at the BF16
    // width), in place: a held view of the ring would turn each into a copy of the window.
    const sliding = t.config.layers - t.config.sparseCount();
    const after = core.stats();
    try testing.expectEqual(@as(u64, 0), after.updates_copied - before.updates_copied);
    try testing.expect(after.updates_in_place - before.updates_in_place >= 2 * sliding * (ids_f.len - 40));
}

/// The tiny on Sushi Cooler's stream at the 8-bit width, past the selection threshold: the state the cells below read.
fn coolerPrefilled(t: *Tiny, es: mlx.mlx_stream, eo: *Ops, ids: []const u32) !*@import("hy4_model.zig").Model {
    const m = try @import("hy4_model.zig").Model.init(testing.allocator, t.config, &t.pack, es);
    errdefer m.deinit();
    m.req().kv = .engine_default;
    var at: usize = 0;
    while (at < ids.len) : (at += 64) _ = try eo.own(try m.forwardReadout(ids[at..@min(at + 64, ids.len)], .none));
    return m;
}

test "hy4 state: a decode row appends its row into each sparse layer's store in place, no buffer copied, and a store's buffers change only when it grows (HY4_BF16_TINY)" {
    try mlx.requireGpu();
    var t = try Tiny.open();
    defer t.close();
    fwd.products = .bf16;
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    var eo: Ops = .{ .s = es };
    defer eo.deinit();
    var ids: [316]u32 = undefined;
    for (&ids, 0..) |*d, i| d.* = @intCast((i * 7919) % 120000);
    const m = try coolerPrefilled(&t, es, &eo, ids[0..300]);
    defer m.deinit();
    const core = @import("../tensor/core.zig");
    const Track = struct { ptr: usize = 0, cap: usize = 0, moves: usize = 0, grows: usize = 0 };
    var track: [hy4_config.max_layers]Track = @splat(.{});
    const before = core.stats();
    for (300..316) |i| {
        _ = try eo.own(try m.forwardReadout(ids[i .. i + 1], .last));
        mlx.engine.close();
        for (m.req().state[0..t.config.layers], &track) |*st, *tr| {
            const r = st.rows orelse continue;
            const ptr = @intFromPtr(core.storePointer(r.buf(0)));
            if (tr.ptr != 0 and ptr != tr.ptr) tr.moves += 1;
            if (tr.cap != 0 and r.cap != tr.cap) tr.grows += 1;
            tr.ptr = ptr;
            tr.cap = r.cap;
        }
    }
    const after = core.stats();
    try testing.expectEqual(@as(u64, 0), after.updates_copied - before.updates_copied);
    for (m.req().state[0..t.config.layers], track) |st, tr| if (st.rows != null) {
        try testing.expectEqual(@as(usize, 316), st.rows.?.used);
        try testing.expect(tr.moves <= tr.grows and tr.grows <= 1);
    };
}

test "hy4 state: a read of a sparse layer's rows still pending when a 6-row block appends sees the rows it was taken over, the block's write going to a copy (HY4_BF16_TINY)" {
    try mlx.requireGpu();
    var t = try Tiny.open();
    defer t.close();
    fwd.products = .bf16;
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    var eo: Ops = .{ .s = es };
    defer eo.deinit();
    var ids: [106]u32 = undefined;
    for (&ids, 0..) |*d, i| d.* = @intCast((i * 7919) % 120000);
    const m = try coolerPrefilled(&t, es, &eo, ids[0..100]);
    defer m.deinit();
    const l = for (m.req().state[0..t.config.layers], 0..) |st, li| {
        if (st.rows != null) break li;
    } else return error.NoSparseLayer;
    // Room for the block: the write lands in the buffer the read holds, not in a grown one.
    try m.req().state[l].rows.?.reserve(es);
    const buf = m.req().state[l].rows.?.buf(0);
    const idx_v = [_]u32{ 99, 0, 42, 7 };
    const idx = try eo.own(mlx.mlx_array_new_data(&idx_v, &[_]c_int{4}, 1, .uint32));
    var ro: Ops = .{ .s = es };
    const want_rows = try testing.allocator.dupe(f32, try hostOf(&ro, try ro.slice(buf, 0, 0, 100)));
    defer testing.allocator.free(want_rows);
    const want_take = try testing.allocator.dupe(f32, try hostOf(&ro, try ro.take(buf, idx, 0)));
    defer testing.allocator.free(want_take);
    ro.deinit();
    // The reads, left pending across the block's append.
    var po: Ops = .{ .s = es };
    defer po.deinit();
    const pending_rows = try po.cast(try po.slice(buf, 0, 0, 100), .float32);
    const pending_take = try po.cast(try po.take(buf, idx, 0), .float32);
    const core = @import("../tensor/core.zig");
    const before = core.stats();
    _ = try eo.own(try m.forwardReadout(ids[100..106], .all));
    mlx.engine.close();
    try testing.expect(core.stats().updates_copied > before.updates_copied);
    try testing.expectEqualSlices(f32, want_rows, try hostOf(&po, pending_rows));
    try testing.expectEqualSlices(f32, want_take, try hostOf(&po, pending_take));
}

test "hy4 model: refused on MLX's GPU stream, where its f32 products could take TF32 tiles (HY4_BF16_TINY)" {
    const hy4_model = @import("hy4_model.zig");
    try mlx.requireGpu();
    var t = try Tiny.open();
    defer t.close();
    const gs = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(gs);
    try testing.expectError(error.Hy4StreamNotCooler, hy4_model.Model.init(testing.allocator, t.config, &t.pack, gs));
}

test "hy4 layer-major: two windows run a layer at a time (embedStreams, engramRowsAfter, passLayer, releaseLayer, readoutStreams) give each window's own forwardReadout bytes (HY4_BF16_TINY)" {
    const hy4_model = @import("hy4_model.zig");
    var t = try Tiny.open();
    defer t.close();
    var o: Ops = .{ .s = t.cpu };
    defer o.deinit();
    const ids_f = try host(&o, try t.ref("ids", .{}));
    var ids: [40]u32 = undefined;
    for (ids_f[0..40], &ids) |v, *d| d.* = @intFromFloat(v);
    const windows = [2][]const u32{ ids[0..20], ids[20..40] };
    const m = try hy4_model.Model.init(testing.allocator, t.config, &t.pack, t.cpu);
    defer m.deinit();
    m.req().kv = .dense;
    var want: [2][]const f32 = undefined;
    for (windows, &want) |w, *d| {
        m.reset();
        d.* = try host(&o, try o.own(try m.forwardReadout(w, .all)));
    }
    m.reset();
    var hs: [2]Arr = undefined;
    var rows: [2]?hy4_model.Model.EngramRows = undefined;
    for (windows, &hs, &rows) |w, *h, *r| {
        h.* = try o.own(try m.embedStreams(w));
        r.* = try m.engramRowsAfter(&.{}, w);
    }
    defer for (rows) |r| if (r) |x| x.deinit(testing.allocator);
    for (0..t.config.layers) |l| for (windows, &hs, rows) |w, *h, r| {
        const tw: c_int = @intCast(w.len);
        h.* = try o.own(try m.passLayer(l, h.*, r, tw, tw, 0));
        m.releaseLayer(l);
    };
    try testing.expectEqual(@as(usize, 0), m.req().pos);
    for (hs, want) |h, wnt| {
        const got = try host(&o, try o.own(try m.readoutStreams(h, .all)));
        try testing.expectEqualSlices(u32, @ptrCast(wnt), @ptrCast(got));
    }
}

/// A tap that only reads: each call counted, by layer and kind, with the rows it was given, and the MLX calls its own
/// reads make.
const CountingTap = struct {
    const kinds = std.enums.values(fwd.TapKind).len;
    own_calls: u64 = 0,
    calls: [hy4_config.max_layers + 1][kinds]usize = @splat(@splat(0)),
    rows: [hy4_config.max_layers + 1][kinds]usize = @splat(@splat(0)),
    routed: [hy4_config.max_layers]usize = @splat(0),
    picked: [hy4_config.max_layers]u64 = @splat(0),
    act_rows: [hy4_config.max_layers]usize = @splat(0),
    acted: [hy4_config.max_layers]u64 = @splat(0),

    fn input(ctx: *anyopaque, layer: usize, kind: fwd.TapKind, x: Arr) anyerror!void {
        const self: *CountingTap = @ptrCast(@alignCast(ctx));
        const c0 = mlx.calls.load(.monotonic);
        defer self.own_calls += mlx.calls.load(.monotonic) - c0;
        self.calls[layer][@backingInt(kind)] += 1;
        self.rows[layer][@backingInt(kind)] += @intCast(mlx.getShape(x)[0]);
    }

    fn routes(ctx: *anyopaque, layer: usize, picks: []const u32, weights: Arr) anyerror!void {
        const self: *CountingTap = @ptrCast(@alignCast(ctx));
        const c0 = mlx.calls.load(.monotonic);
        defer self.own_calls += mlx.calls.load(.monotonic) - c0;
        self.routed[layer] += 1;
        try testing.expectEqual(picks.len, mlx.mlx_array_size(weights));
        for (picks) |e| self.picked[layer] += e + 1;
    }

    fn expertAct(ctx: *anyopaque, layer: usize, act: Arr, experts: []const u32) anyerror!void {
        const self: *CountingTap = @ptrCast(@alignCast(ctx));
        const c0 = mlx.calls.load(.monotonic);
        defer self.own_calls += mlx.calls.load(.monotonic) - c0;
        // The tiny's products run at f32 (`Tiny.open`).
        try testing.expectEqual(mlx.mlx_dtype.float32, mlx.mlx_array_dtype(act));
        try testing.expectEqual(@as(c_int, @intCast(experts.len)), mlx.getShape(act)[0]);
        self.act_rows[layer] += experts.len;
        for (experts) |e| self.acted[layer] += e + 1;
    }
};

test "hy4 tap: called at every site, each layer's rows once (o_in by span, expert_act by pick), the experts' picks the router's; a reading tap adds no MLX call and leaves the logits' bytes (HY4_BF16_TINY)" {
    const hy4_model = @import("hy4_model.zig");
    var t = try Tiny.open();
    defer t.close();
    var o: Ops = .{ .s = t.cpu };
    defer o.deinit();
    const ids_f = try host(&o, try t.ref("ids", .{}));
    var ids: [40]u32 = undefined;
    for (ids_f[0..40], &ids) |v, *d| d.* = @intFromFloat(v);
    const m = try hy4_model.Model.init(testing.allocator, t.config, &t.pack, t.cpu);
    defer m.deinit();
    m.req().kv = .dense;
    // Spans of 16 rows: the 40-row prompt's o_in comes in three calls a layer.
    m.span_rows = 16;
    const c0 = mlx.calls.load(.monotonic);
    const plain = try host(&o, try o.own(try m.forwardReadout(&ids, .all)));
    const plain_calls = mlx.calls.load(.monotonic) - c0;
    m.reset();
    var sink: CountingTap = .{};
    const tap: fwd.Tap = .{ .ctx = &sink, .input = CountingTap.input, .routes = CountingTap.routes, .expert_act = CountingTap.expertAct };
    m.tap = &tap;
    const c1 = mlx.calls.load(.monotonic);
    const tapped = try host(&o, try o.own(try m.forwardReadout(&ids, .all)));
    try testing.expectEqual(plain_calls, mlx.calls.load(.monotonic) - c1 - sink.own_calls);
    try testing.expectEqualSlices(u32, @ptrCast(plain), @ptrCast(tapped));
    const c = t.config;
    const T: usize = ids.len;
    for (0..c.layers) |l| {
        const moe = l >= c.dense_layers;
        var engram = false;
        for (c.engram.layers[0..c.engram.n]) |e| engram = engram or e == l;
        for (std.enums.values(fwd.TapKind)) |kind| {
            const calls: usize = switch (kind) {
                .attn_in, .q_lora, .kv_latent, .ffn_in, .shared_act => 1,
                .o_in => 3,
                .expert_in, .lat_up_in => @intFromBool(moe),
                .engram_in => @intFromBool(engram),
                .head_in => 0,
            };
            try testing.expectEqual(calls, sink.calls[l][@backingInt(kind)]);
            try testing.expectEqual(if (calls > 0) T else 0, sink.rows[l][@backingInt(kind)]);
        }
        try testing.expectEqual(@as(usize, @intFromBool(moe)), sink.routed[l]);
        try testing.expectEqual(if (moe) T * c.top_k else 0, sink.act_rows[l]);
        try testing.expectEqual(sink.picked[l], sink.acted[l]);
    }
    try testing.expectEqual(@as(usize, 1), sink.calls[c.layers][@backingInt(fwd.TapKind.head_in)]);
    try testing.expectEqual(T, sink.rows[c.layers][@backingInt(fwd.TapKind.head_in)]);

    // In chunks that read out nothing, then one row: the head's input is still every row's, and the row read out is
    // the untapped forward's.
    m.reset();
    sink = .{};
    _ = mlx.mlx_array_free(try m.forwardReadout(ids[0..16], .none));
    const last = try host(&o, try o.own(try m.forwardReadout(ids[16..], .last)));
    try testing.expectEqual(@as(usize, 2), sink.calls[c.layers][@backingInt(fwd.TapKind.head_in)]);
    try testing.expectEqual(T, sink.rows[c.layers][@backingInt(fwd.TapKind.head_in)]);
    m.reset();
    m.tap = null;
    _ = mlx.mlx_array_free(try m.forwardReadout(ids[0..16], .none));
    const untapped = try host(&o, try o.own(try m.forwardReadout(ids[16..], .last)));
    try testing.expectEqualSlices(u32, @ptrCast(untapped), @ptrCast(last));
}

test "hy4 prefix: a restore point taken mid-prompt, stored or live, after the request ran past it (rings overwritten, groups closed), prefills the rest and decodes byte for byte as cold (HY4_BF16_TINY)" {
    const hy4_model = @import("hy4_model.zig");
    const prefix = @import("hy4_prefix.zig");
    try mlx.requireGpu();
    var t = try Tiny.open();
    defer t.close();
    const ids_f = try rawF32(try t.ref("ids", .{}));
    var ids: [64]u32 = undefined;
    for (ids_f, ids[0..ids_f.len]) |v, *d| d.* = @intFromFloat(v);
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const m = try hy4_model.Model.init(testing.allocator, t.config, &t.pack, es);
    defer m.deinit();
    var eo: Ops = .{ .s = es };
    defer eo.deinit();
    // Cold: the prompt's rest after its first 21 rows (the point sits mid-group and mid-window), then a decode step.
    m.req().kv = .engine_default;
    _ = try eo.own(try m.forward(ids[0..21]));
    const cold_rest = try host(&eo, try eo.own(try m.forward(ids[21..40])));
    const cold_step = try host(&eo, try eo.own(try m.forward(ids[40..41])));
    // The point at 21, then the request runs on: every ring wraps past it and groups close.
    m.reset();
    m.req().kv = .engine_default;
    _ = try eo.own(try m.forward(ids[0..21]));
    var cp = try prefix.capture(testing.allocator, m, 21);
    defer cp.deinit(testing.allocator);
    _ = try eo.own(try m.forward(ids[21..40]));
    _ = try eo.own(try m.forward(ids[40..44]));
    try prefix.restore(m, &cp, ids[0..44]);
    try testing.expectEqual(@as(usize, 21), m.req().pos);
    try testing.expectEqualSlices(f32, cold_rest, try host(&eo, try eo.own(try m.forward(ids[21..40]))));
    try testing.expectEqualSlices(f32, cold_step, try host(&eo, try eo.own(try m.forward(ids[40..41]))));
    // The live record's point (what a next turn rolls back to without a stored restore) restores the same state.
    m.reset();
    m.req().kv = .engine_default;
    _ = try eo.own(try m.forward(ids[0..21]));
    try m.keepLivePoint();
    _ = try eo.own(try m.forward(ids[21..44]));
    try m.rollbackPoint(&m.req().live_points.items[m.req().live_points.items.len - 1], 41);
    try testing.expectEqualSlices(f32, cold_rest, try host(&eo, try eo.own(try m.forward(ids[21..40]))));
    try testing.expectEqualSlices(f32, cold_step, try host(&eo, try eo.own(try m.forward(ids[40..41]))));
}

/// An evaluated, contiguous f32 array's values, read with no op.
fn rawF32(a: Arr) ![]const f32 {
    if (mlx.mlx_array_dtype(a) != .float32) {
        const cpu = mlx.mlx_default_cpu_stream_new();
        defer _ = mlx.mlx_stream_free(cpu);
        var c = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&c, a, .float32, cpu));
        try mlx.check(mlx.mlx_array_eval(c));
        // Kept for the test's life: the tiny's few non-f32 arrays (ids) are small.
        return (mlx.mlx_array_data_float32(c) orelse return error.HostRead)[0..mlx.mlx_array_size(c)];
    }
    try mlx.check(mlx.mlx_array_eval(a));
    return (mlx.mlx_array_data_float32(a) orelse return error.HostRead)[0..mlx.mlx_array_size(a)];
}

fn expectNearRows(label: []const u8, got: []const f32, want: []const f32, bar: f64) !void {
    if (got.len != want.len) {
        std.debug.print("{s}: {d} values, the oracle has {d}\n", .{ label, got.len, want.len });
        return error.ShapeMismatch;
    }
    var peak: f64 = 1e-30;
    for (want) |v| peak = @max(peak, @abs(v));
    var worst: f64 = 0;
    var at: usize = 0;
    for (got, want, 0..) |x, y, i| {
        const e = @abs(@as(f64, x) - y);
        if (!(e <= worst)) {
            worst = e;
            at = i;
        }
    }
    if (!(worst / peak <= bar)) {
        std.debug.print("{s}: worst |Δ| {e} at {d} ({d} vs {d}) is {e} of the peak {e}, over {e}\n", .{ label, worst, at, got[at], want[at], worst / peak, peak, bar });
        return error.OutsideBar;
    }
}

test "hy4 served: the tiny loaded through the shared loader answers the oracle's logits through the trunk's forward (HY4_BF16_TINY)" {
    try servedTinySaid(false);
}

test "hy4 served: with its experts streamed from SSD through a cache of 4 slots a layer and no union workspace (the prefill's 8 experts routed in groups that fit), the tiny answers the same logits (HY4_BF16_TINY)" {
    try servedTinySaid(true);
}

fn servedTinySaid(streamed: bool) !void {
    servedTiny(streamed) catch |e| {
        var said: [512]u8 = undefined;
        std.debug.print("\nserved: {t}: {s}\n", .{ e, mlx.takeError(&said) orelse "(no message)" });
        if (@errorReturnTrace()) |tr| std.debug.dumpStackTrace(tr.*);
        return e;
    };
}

fn servedTiny(streamed: bool) !void {
    const dir = std.mem.span(std.c.getenv("HY4_BF16_TINY") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const Trunk = @import("../trunk.zig").Trunk;
    // The oracle, read as host rows before the load: no MLX stream runs inside the engine's.
    var t = try Tiny.open();
    defer t.close();
    const want = try rawF32(try t.ref("logits", .{}));
    const ids_f = try rawF32(try t.ref("ids", .{}));
    const vocab: usize = @intCast(mlx.getShape(try t.ref("logits", .{}))[1]);
    var ids: [64]u32 = undefined;
    for (ids_f, ids[0..ids_f.len]) |v, *d| d.* = @intFromFloat(v);
    // Every Hy4 load is a Cooler load: the engine on, as a served load has it.
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    var config = try model_mod.parseConfig(io, a, dir);
    defer config.deinit(a);
    config.expert_streaming = streamed;
    // Engram's tables are read by row from the checkpoint on every load: the model reads them from here.
    config.expert_source_dir = try a.dupe(u8, dir);
    if (streamed) {
        const per = (3 * @as(u64, config.moe_intermediate_size) * config.expert_hidden) * 2;
        // 4 of 8 experts a layer: the 8 decode steps evict and refill.
        config.expert_cache_bytes = per * 4 * config.num_hidden_layers;
    }
    var weights = try model_mod.loadWeightsForConfig(io, a, dir, &config, false);
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    try testing.expectEqual(!streamed, weights.get("model.layers.1.mlp.experts.gate_up_proj") != null);
    // No load holds Engram's table: it is read by row from the shard (the tiny's module sits at layer 1).
    try testing.expect(weights.get("model.layers.1.engram_layer.multi_head_embedding.embedding.weight") == null);
    // The bill's resident bytes are the bytes the load holds.
    var held: u64 = 0;
    var it = weights.map.iterator();
    while (it.next()) |e| held += mlx.mlx_array_size(e.value_ptr.*) * mlx.mlx_array_itemsize(e.value_ptr.*);
    try testing.expectEqual(held, (try @import("hy4_load.zig").residentLoadBytes(io, a, dir, &config, false, false)).?);
    var xfm = try Trunk.init(io, a, config, &weights);
    defer xfm.deinit();
    try testing.expectEqual(streamed, xfm.expert_stream != null);
    var o: Ops = .{ .s = xfm.s };
    defer o.deinit();
    var label: [40]u8 = undefined;
    var at: usize = 0;
    // The prompt in one forward, then each decode step: every forward gives its last row's logits.
    const steps = [_]usize{ 40, 1, 1, 1, 1, 1, 1, 1, 1 };
    for (steps) |rows| {
        const arr = try o.own(mlx.mlx_array_new_data(&ids[at], &[_]c_int{ 1, @intCast(rows) }, 2, .uint32));
        const logits = try host(&o, try o.own(try xfm.forward(arr)));
        at += rows;
        try expectNearRows(try std.fmt.bufPrint(&label, "served logits at {d}", .{at - 1}), logits, want[(at - 1) * vocab ..][0..vocab], 2e-3);
    }
    try testing.expectEqual(@as(usize, 48), xfm.cache.step);
    // The banks came off SSD, more than once over: the cache held fewer experts than the steps picked.
    if (xfm.expert_stream) |engine| {
        try testing.expectEqual(@as(usize, 0), engine.union_slabs.len);
        try testing.expectEqual(@as(u64, 0), engine.plan.workspace_bytes);
        const per: u64 = (3 * @as(u64, config.moe_intermediate_size) * config.expert_hidden) * 2;
        try testing.expect(engine.fill_bytes_total > per * config.num_experts);
    }
}

/// A streamed load of the tiny on Sushi Cooler with `slots` experts a layer in its cache.
const StreamedTiny = struct {
    config: model_mod.ModelConfig,
    weights: model_mod.Weights,
    xfm: @import("../trunk.zig").Trunk,

    fn open(self: *StreamedTiny, dir: []const u8, slots: u64) !void {
        const a = testing.allocator;
        const io = std.Io.Threaded.global_single_threaded.io();
        self.config = try model_mod.parseConfig(io, a, dir);
        errdefer self.config.deinit(a);
        self.config.expert_streaming = true;
        self.config.expert_source_dir = try a.dupe(u8, dir);
        self.config.expert_cache_bytes = (3 * @as(u64, self.config.moe_intermediate_size) * self.config.expert_hidden) * 2 * slots * self.config.num_hidden_layers;
        self.weights = try model_mod.loadWeightsForConfig(io, a, dir, &self.config, false);
        errdefer self.weights.deinit();
        model_mod.resolveWeightPrefix(&self.config, &self.weights);
        self.xfm = try @import("../trunk.zig").Trunk.init(io, a, self.config, &self.weights);
    }

    fn close(self: *StreamedTiny) void {
        self.xfm.deinit();
        self.weights.deinit();
        self.config.deinit(testing.allocator);
    }
};

test "hy4 pinned: a layer whose experts are all pinned runs its picks as one group and reads no SSD; window-major through groups of 4 slots gives its bytes (HY4_BF16_TINY)" {
    const dir = std.mem.span(std.c.getenv("HY4_BF16_TINY") orelse return error.SkipZigTest);
    var t = try Tiny.open();
    defer t.close();
    // The served forward's products.
    fwd.products = .bf16;
    const ids_f = try rawF32(try t.ref("ids", .{}));
    var ids: [40]u32 = undefined;
    for (ids_f[0..40], &ids) |v, *d| d.* = @intFromFloat(v);
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    var got: [2][]f32 = undefined;
    defer for (got) |g| testing.allocator.free(g);
    // Groups: a cache of 4 slots a layer, window-major; pinned: all 8 experts a layer, a layer at a time.
    for ([_]u64{ 4, 8 }, &got) |slots, *out| {
        var st: StreamedTiny = undefined;
        try st.open(dir, slots);
        defer st.close();
        const m = st.xfm.arch_state.hy4flash.model orelse return error.Hy4ModelMissing;
        const engine = st.xfm.expert_stream.?;
        m.req().kv = .dense;
        var o: Ops = .{ .s = st.xfm.s };
        defer o.deinit();
        engine.beginForward();
        const logits = if (slots == 4) try o.own(try m.forwardReadout(&ids, .last)) else blk: {
            var hs = try o.own(try m.embedStreams(&ids));
            const rows = try m.engramRowsAfter(&.{}, &ids);
            defer if (rows) |r| r.deinit(testing.allocator);
            for (0..t.config.layers) |l| {
                const moe = l >= t.config.dense_layers;
                if (moe) try m.pin(@intCast(l));
                defer m.unpin();
                const filled = engine.fill_bytes_total;
                hs = try o.own(try m.passLayer(l, hs, rows, ids.len, ids.len, 0));
                try testing.expectEqual(filled, engine.fill_bytes_total);
            }
            break :blk try o.own(try m.readoutStreams(hs, .last));
        };
        engine.finishForward(ids.len);
        out.* = try testing.allocator.dupe(f32, try rawF32(logits));
    }
    try testing.expectEqualSlices(u32, @ptrCast(got[0]), @ptrCast(got[1]));
}

test "hy4 oracle: a prefill read out as none extends the state, and the next row's last readout is the oracle's (HY4_BF16_TINY)" {
    const hy4_model = @import("hy4_model.zig");
    var t = try Tiny.open();
    defer t.close();
    var o: Ops = .{ .s = t.cpu };
    defer o.deinit();
    const ids_f = try host(&o, try t.ref("ids", .{}));
    var ids: [40]u32 = undefined;
    for (ids_f[0..40], &ids) |v, *d| d.* = @intFromFloat(v);
    const m = try hy4_model.Model.init(testing.allocator, t.config, &t.pack, t.cpu);
    defer m.deinit();
    m.req().kv = .dense;
    _ = mlx.mlx_array_free(try m.forwardReadout(ids[0..39], .none));
    const got = try o.own(try m.forwardReadout(ids[39..40], .last));
    try expectNear("row 39 after a none prefill", &o, got, try o.slice(try t.ref("logits", .{}), 0, 39, 40), 2e-3);
}

fn hostOf(o: *Ops, a: Arr) ![]const f32 {
    const c = try o.contiguous(try o.cast(a, .float32));
    try mlx.check(mlx.mlx_array_eval(c));
    return mlx.mlx_array_data_float32(c).?[0..mlx.mlx_array_size(c)];
}

test "hy4 verify: a 6-row verify block rolled back to each accepted count (0 to 5) and continued gives the serial run's logits, its rows, rings, open groups and Engram history as the serial run left them (HY4_BF16_TINY)" {
    const hy4_model = @import("hy4_model.zig");
    try mlx.requireGpu();
    var t = try Tiny.open();
    defer t.close();
    fwd.products = .bf16;
    const ids_f = try rawF32(try t.ref("ids", .{}));
    var ids: [64]u32 = undefined;
    for (ids_f, ids[0..ids_f.len]) |v, *d| d.* = @intFromFloat(v);
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    var eo: Ops = .{ .s = es };
    defer eo.deinit();
    // The serial run: rows 40..47 one at a time after the prefill (the window of 8 and groups of 4 cross inside).
    var serial: [8][]const f32 = undefined;
    {
        const m = try hy4_model.Model.init(testing.allocator, t.config, &t.pack, es);
        defer m.deinit();
        _ = try eo.own(try m.forwardReadout(ids[0..40], .none));
        for (0..8) |r| serial[r] = try hostOf(&eo, try eo.own(try m.forwardReadout(ids[40 + r .. 41 + r], .last)));
    }
    const m = try hy4_model.Model.init(testing.allocator, t.config, &t.pack, es);
    defer m.deinit();
    _ = try eo.own(try m.forwardReadout(ids[0..40], .none));
    for (0..6) |accepted| {
        var mk = try m.mark();
        defer m.dropMark(&mk);
        // The block: the pending token and 5 drafts, every row read out.
        const lg = try eo.own(try m.forwardReadout(ids[40..46], .all));
        for (0..accepted + 1) |r| try expectRowNear(try hostOf(&eo, try eo.slice(lg, 0, @intCast(r), @intCast(r + 1))), serial[r]);
        // The verdict keeps the pending token and `accepted` drafts; the next row continues from there.
        try m.rollbackTo(&mk, ids[40 .. 41 + accepted]);
        try testing.expectEqual(@as(usize, 41 + accepted), m.req().pos);
        try testing.expectEqual(@as(usize, 41 + accepted), m.req().history.items.len);
        try expectRowNear(try hostOf(&eo, try eo.own(try m.forwardReadout(ids[41 + accepted .. 42 + accepted], .last))), serial[1 + accepted]);
        // Back to the round's start for the next count.
        var none: [0]u32 = .{};
        try m.rollbackTo(&mk, &none);
    }
}

test "hy4 verify: every verify width 2..8 at p0 from threshold - 8 to threshold + 1 matches its one-row steps (HY4_BF16_TINY)" {
    const hy4_model = @import("hy4_model.zig");
    try mlx.requireGpu();
    var t = try Tiny.open();
    defer t.close();
    fwd.products = .bf16;
    const ids_f = try rawF32(try t.ref("ids", .{}));
    var ids: [64]u32 = undefined;
    for (ids_f, ids[0..ids_f.len]) |v, *d| d.* = @intFromFloat(v);
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const m = try hy4_model.Model.init(testing.allocator, t.config, &t.pack, es);
    defer m.deinit();
    const max_w = 8;
    const thr: usize = t.config.indexer.group_topk * t.config.indexer.group;
    for (thr - 8..thr + 2) |p0| {
        var eo: Ops = .{ .s = es };
        defer eo.deinit();
        var serial: [max_w][]const f32 = undefined;
        m.reset();
        _ = try eo.own(try m.forwardReadout(ids[0..p0], .none));
        for (0..max_w) |r| serial[r] = try hostOf(&eo, try eo.own(try m.forwardReadout(ids[p0 + r .. p0 + r + 1], .last)));
        for (2..max_w + 1) |width| {
            var wo: Ops = .{ .s = es };
            defer wo.deinit();
            m.reset();
            _ = try wo.own(try m.forwardReadout(ids[0..p0], .none));
            const lg = try wo.own(try m.forwardReadout(ids[p0 .. p0 + width], .all));
            for (0..width) |r| {
                errdefer std.debug.print("hy4 bf16 width probe: p0={d} width={d} row={d}\n", .{ p0, width, r });
                try expectRowNear(try hostOf(&wo, try wo.slice(lg, 0, @intCast(r), @intCast(r + 1))), serial[r]);
            }
        }
    }
}

/// A row against the serial run's: rows computed in a block differ from single rows by f32 reassociation alone.
fn expectRowNear(got: []const f32, want: []const f32) !void {
    var worst: f32 = 0;
    var peak: f32 = 0;
    var ag: usize = 0;
    var aw: usize = 0;
    for (got, want, 0..) |x, y, i| {
        worst = @max(worst, @abs(x - y));
        peak = @max(peak, @abs(y));
        if (x > got[ag]) ag = i;
        if (y > want[aw]) aw = i;
    }
    if (worst > 1e-5 * peak or ag != aw) {
        std.debug.print("verify row: worst |Δ| {e:.3} of the peak {e:.3}; argmax {d} vs {d}\n", .{ worst, peak, ag, aw });
        return error.OutsideBar;
    }
}

/// A forward of `ids[at..at + width]` after a prefill of `ids[0..at]` run on the op chain, its logits and the core's
/// launches it made.
fn afterChain(m: *@import("hy4_model.zig").Model, eo: *Ops, ids: []const u32, at: usize, width: usize) !struct { rows: []const f32, launches: u64 } {
    const hy4_attn = @import("hy4_attn.zig");
    m.reset();
    fwd.core_kernel_on = false;
    _ = try eo.own(try m.forwardReadout(ids[0..at], .none));
    fwd.core_kernel_on = true;
    const before = hy4_attn.launched;
    const rows = try hostOf(eo, try eo.own(try m.forwardReadout(ids[at .. at + width], .all)));
    return .{ .rows = rows, .launches = hy4_attn.launched - before };
}

test "hy4 core: a decode row and every verify block of 2..8 rows reach no core launch and give the op chain's bytes, and a wide chunk runs the core and gives the chain's logits within the bar (HY4_BF16_TINY)" {
    const hy4_model = @import("hy4_model.zig");
    const hy4_attn = @import("hy4_attn.zig");
    try mlx.requireGpu();
    var t = try Tiny.open();
    defer t.close();
    fwd.products = .bf16;
    const ids_f = try rawF32(try t.ref("ids", .{}));
    var ids: [64]u32 = undefined;
    for (ids_f, ids[0..ids_f.len]) |v, *d| d.* = @intFromFloat(v);
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    defer fwd.core_kernel_on = true;
    const m = try hy4_model.Model.init(testing.allocator, t.config, &t.pack, es);
    defer m.deinit();
    m.req().kv = .engine_default;
    var eo: Ops = .{ .s = es };
    defer eo.deinit();
    const thr: usize = t.config.indexer.group_topk * t.config.indexer.group;
    // A block after a prefill on the chain: the core is on for it, and never launches.
    for ([_]usize{ thr - 1, thr + 1, 30 }) |at| for (1..9) |width| {
        errdefer std.debug.print("hy4 core: at {d} width {d}\n", .{ at, width });
        const on = try afterChain(m, &eo, &ids, at, width);
        try testing.expectEqual(@as(u64, 0), on.launches);
        m.reset();
        fwd.core_kernel_on = false;
        _ = try eo.own(try m.forwardReadout(ids[0..at], .none));
        const off = try hostOf(&eo, try eo.own(try m.forwardReadout(ids[at .. at + width], .all)));
        fwd.core_kernel_on = true;
        try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(off), std.mem.sliceAsBytes(on.rows));
    };
    // A chunk of 40 rows (past the selection and the window): the core runs on each layer, and its logits are the chain's
    // to the bar, at f32 products and the BF16 latent (BF16 products and kv8 codes turn the sums' order into flipped
    // roundings, codes and routing picks).
    fwd.products = .f32;
    m.reset();
    m.req().kv = .dense;
    fwd.core_kernel_on = false;
    const chain = try eo.own(try m.forwardReadout(ids[0..40], .all));
    const chain_rows = try hostOf(&eo, chain);
    m.reset();
    fwd.core_kernel_on = true;
    const before = hy4_attn.launched;
    const core_rows = try hostOf(&eo, try eo.own(try m.forwardReadout(ids[0..40], .all)));
    try testing.expect(hy4_attn.launched - before >= t.config.layers);
    if (std.c.getenv("SUSHI_HY4_ATTN_TRACE") != null) {
        var worst: f64 = 0;
        var peak: f64 = 1e-30;
        for (core_rows, chain_rows) |x, y| {
            worst = @max(worst, @abs(@as(f64, x) - y));
            peak = @max(peak, @abs(@as(f64, y)));
        }
        std.debug.print("hy4 core: 40-row prefill logits, max |d| {e} of the peak {e}: {e}\n", .{ worst, peak, worst / peak });
    }
    try expectNearRows("core prefill", core_rows, chain_rows, core_bar);
}

/// How far the core's prefill logits may sit from the chain's, over the logits' peak: the attention's f32 sums in a
/// different order through the tiny's six layers, 4.0e-4 measured. The chain's own two orders (spans of 7 rows and of
/// 256) sit 4.3e-4 apart.
const core_bar = 1e-3;

/// The tiny's model on the engine stream with a hot cache beside it: the turns a live record is published from and
/// the next turn's lookup (`hy4_prefix.rollbackLive`).
const LiveRig = struct {
    const prefix = @import("hy4_prefix.zig");
    const pc = @import("../prefix_cache.zig");

    t: Tiny,
    ids: [64]u32,
    m: *@import("hy4_model.zig").Model,
    eo: Ops,
    hc: pc.HotPrefixCache,
    es: mlx.mlx_stream,

    fn forward(r: *LiveRig, ids: []const u32) !void {
        _ = try r.eo.own(try r.m.forward(ids));
    }

    /// The logits of `ids`' rows after the state's.
    fn rows(r: *LiveRig, ids: []const u32) ![]const f32 {
        return host(&r.eo, try r.eo.own(try r.m.forward(ids)));
    }

    /// A cold request: the arch's forward at position 0 starts the state over.
    fn cold(r: *LiveRig) void {
        r.m.reset();
        r.m.dropLive();
        r.m.req().kv = .engine_default;
    }

    /// A finished request's record, as the scheduler publishes it.
    fn publish(r: *LiveRig, tokens: []const u32, has_tools: bool, media: pc.MediaStamp) void {
        prefix.liveOf(&r.hc).publish(r.m, tokens, has_tools, media);
    }

    fn lookup(r: *LiveRig, prompt: []const u32, has_tools: bool, rule: ?*const pc.MediaRule) !?pc.LookupResult {
        const KVCache = @import("../kv_cache.zig").KVCache;
        var target = try KVCache.initWithConfig(testing.allocator, r.t.config.layers, .engine_default);
        defer target.deinit();
        var off: usize = 0;
        const limits: @import("../kv_disk_cache.zig").MediaLimits = if (rule) |x| x.limits() else .{};
        return prefix.rollbackLive(&r.hc, r.m, &target, &off, r.es, prompt, has_tools, limits, null);
    }
};

fn withLiveRig(comptime body: fn (*LiveRig) anyerror!void) !void {
    try mlx.requireGpu();
    var r: LiveRig = undefined;
    r.t = try Tiny.open();
    defer r.t.close();
    const ids_f = try rawF32(try r.t.ref("ids", .{}));
    for (ids_f, r.ids[0..ids_f.len]) |v, *d| d.* = @intFromFloat(v);
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    r.es = mlx.engine.stream();
    const scope = mlx.engine.enter(r.es);
    defer scope.end();
    r.m = try @import("hy4_model.zig").Model.init(testing.allocator, r.t.config, &r.t.pack, r.es);
    defer r.m.deinit();
    r.eo = .{ .s = r.es };
    defer r.eo.deinit();
    r.hc = LiveRig.pc.HotPrefixCache.init(testing.allocator, 4);
    defer r.hc.deinit();
    r.m.req().kv = .engine_default;
    try body(&r);
}

test "hy4 live: a request that ran on the model after the last publish and never published is not continued in place as the last turn's state, and one that started the state over leaves the record nothing (HY4_BF16_TINY)" {
    try withLiveRig(struct {
        fn run(r: *LiveRig) !void {
            const c_prompt = r.ids[0..30];
            try r.forward(c_prompt[0..10]);
            const cold = try r.rows(c_prompt[10..]);
            // Turn A publishes with a point at 10; request B rolls back to it, runs its own 10 rows and is cancelled.
            r.cold();
            try r.forward(r.ids[0..10]);
            try r.m.keepLivePoint();
            try r.forward(r.ids[10..20]);
            r.publish(r.ids[0..20], false, .{});
            var b_prompt: [21]u32 = undefined;
            @memcpy(b_prompt[0..10], r.ids[0..10]);
            @memcpy(b_prompt[10..], r.ids[30..41]);
            try testing.expectEqual(@as(usize, 10), (try r.lookup(&b_prompt, false, null)).?.matched);
            try r.forward(b_prompt[10..20]);
            // C shares A's 20 tokens: it rolls back to A's point, not onto B's rows.
            try testing.expectEqual(@as(usize, 10), (try r.lookup(c_prompt, false, null)).?.matched);
            try testing.expectEqualSlices(f32, cold, try r.rows(c_prompt[10..]));
            // A request that started the state over leaves A's point no rows to stand on: the next turn misses.
            r.publish(c_prompt, false, .{});
            r.cold();
            try r.forward(r.ids[30..40]);
            try testing.expectEqual(@as(?LiveRig.pc.LookupResult, null), try r.lookup(c_prompt, false, null));
        }
    }.run);
}

test "hy4 live: a next turn at another KV width than the live state's is not continued in place, nor from a live point (HY4_BF16_TINY)" {
    try withLiveRig(struct {
        fn run(r: *LiveRig) !void {
            r.cold();
            r.m.req().kv = .dense;
            try r.forward(r.ids[0..10]);
            try r.m.keepLivePoint();
            try r.forward(r.ids[10..20]);
            r.publish(r.ids[0..20], false, .{});
            try testing.expectEqual(@as(?LiveRig.pc.LookupResult, null), try r.lookup(r.ids[0..30], false, null));
        }
    }.run);
}

test "hy4 live: with no point to roll back to, a request that ran after the last publish leaves the next turn a miss (HY4_BF16_TINY)" {
    try withLiveRig(struct {
        fn run(r: *LiveRig) !void {
            r.cold();
            try r.forward(r.ids[0..20]);
            r.publish(r.ids[0..20], false, .{});
            r.cold();
            try r.forward(r.ids[30..40]);
            try testing.expectEqual(@as(?LiveRig.pc.LookupResult, null), try r.lookup(r.ids[0..30], false, null));
        }
    }.run);
}

test "hy4 live: a next turn whose image differs from the last turn's (same placeholder ids) is held to the text before the first media row, and the same image continues in place (HY4_BF16_TINY)" {
    try withLiveRig(struct {
        fn run(r: *LiveRig) !void {
            // An image's placeholder rows at 5..9, whose pixels are turn A's: points at 4 (text) and 8 (inside the image).
            var prompt: [24]u32 = undefined;
            @memcpy(&prompt, r.ids[0..24]);
            for (prompt[5..9]) |*d| d.* = r.t.config.image_id;
            try r.forward(prompt[0..4]);
            const cold = try r.rows(prompt[4..]);
            r.cold();
            try r.forward(prompt[0..4]);
            try r.m.keepLivePoint();
            try r.forward(prompt[4..8]);
            try r.m.keepLivePoint();
            try r.forward(prompt[8..20]);
            r.publish(prompt[0..20], false, .{ .key = 1, .start = 5 });
            // Another image under the same ids: the rows past the media start are A's pixels.
            const other: LiveRig.pc.MediaRule = .{ .vision_key = 2, .media_start = 5, .chain = &.{} };
            const res = (try r.lookup(&prompt, false, &other)).?;
            try testing.expectEqual(@as(usize, 4), res.matched);
            try testing.expectEqualSlices(f32, cold, try r.rows(prompt[4..]));
            // The same image: the record's state continues where it stands (the lookup above rolled it back, so the turn runs again).
            r.cold();
            try r.forward(prompt[0..4]);
            try r.m.keepLivePoint();
            try r.forward(prompt[4..20]);
            r.publish(prompt[0..20], false, .{ .key = 1, .start = 5 });
            const same: LiveRig.pc.MediaRule = .{ .vision_key = 1, .media_start = 5, .chain = &.{} };
            try testing.expectEqual(@as(usize, 20), (try r.lookup(&prompt, false, &same)).?.matched);
        }
    }.run);
}

test "hy4 live: a next turn whose tools differ from the last turn's does not continue its state (HY4_BF16_TINY)" {
    try withLiveRig(struct {
        fn run(r: *LiveRig) !void {
            r.cold();
            try r.forward(r.ids[0..10]);
            try r.m.keepLivePoint();
            try r.forward(r.ids[10..20]);
            r.publish(r.ids[0..20], false, .{});
            try testing.expectEqual(@as(?LiveRig.pc.LookupResult, null), try r.lookup(r.ids[0..30], true, null));
            // The request starts the state over: the record is let go (to SSD, when there is a tier).
            try testing.expectEqual(@as(?LiveRig.pc.LookupResult, null), try r.lookup(r.ids[0..30], false, null));
        }
    }.run);
}

test "hy4 live: a restore point of a request that never published does not ride into the next request's record (HY4_BF16_TINY)" {
    try withLiveRig(struct {
        fn run(r: *LiveRig) !void {
            const prefix = LiveRig.prefix;
            r.cold();
            try r.forward(r.ids[0..10]);
            try r.m.keepLivePoint();
            try r.forward(r.ids[10..20]);
            r.publish(r.ids[0..20], false, .{});
            // B rolls back to A's point, takes one of its own at 15 and is cancelled; C rolls back to A's point.
            var b_prompt: [21]u32 = undefined;
            @memcpy(b_prompt[0..10], r.ids[0..10]);
            @memcpy(b_prompt[10..], r.ids[30..41]);
            try testing.expectEqual(@as(usize, 10), (try r.lookup(&b_prompt, false, null)).?.matched);
            try r.forward(b_prompt[10..15]);
            try r.m.keepLivePoint();
            try testing.expectEqual(@as(usize, 10), (try r.lookup(r.ids[0..30], false, null)).?.matched);
            try testing.expectEqual(@as(usize, 0), r.m.req().live_points.items.len);
            // D restores a stored point: the request keeps that point alone, not the ones taken before.
            r.cold();
            try r.forward(r.ids[0..10]);
            var cp = try prefix.capture(testing.allocator, r.m, 10);
            defer cp.deinit(testing.allocator);
            try r.m.keepLivePoint();
            try r.forward(r.ids[10..12]);
            try r.m.keepLivePoint();
            try prefix.restore(r.m, &cp, r.ids[0..30]);
            try testing.expectEqual(@as(usize, 1), r.m.req().live_points.items.len);
            try testing.expectEqual(@as(usize, 10), r.m.req().live_points.items[0].pos);
        }
    }.run);
}

test "hy4 live: a forward that fails mid-layer leaves the model no state, and the next turn does not continue the last record in place (HY4_BF16_TINY)" {
    try withLiveRig(struct {
        fn run(r: *LiveRig) !void {
            try r.forward(r.ids[0..20]);
            const cold = try r.rows(r.ids[20..30]);
            var fired_at: u64 = 0;
            // The fault lands in a later call each round until it is inside the forward.
            for ([_]u64{ 2000, 1000, 500, 250 }) |k| {
                r.cold();
                try r.forward(r.ids[0..20]);
                r.publish(r.ids[0..20], false, .{});
                mlx.fault.arm(k);
                const out = r.m.forward(r.ids[20..30]);
                const fired = mlx.fault.didFire();
                mlx.fault.disarm();
                mlx.dropLatchedErrorUnless(false);
                if (out) |a| {
                    _ = mlx.mlx_array_free(a);
                    continue;
                } else |_| {}
                if (fired) {
                    fired_at = k;
                    break;
                }
            }
            try testing.expect(fired_at != 0);
            try testing.expectEqual(@as(usize, 0), r.m.req().pos);
            for (r.m.req().state[0..r.t.config.layers]) |st| try testing.expect(st.k_pe == null and st.rows == null);
            try testing.expectEqual(@as(?LiveRig.pc.LookupResult, null), try r.lookup(r.ids[0..30], false, null));
            r.cold();
            try r.forward(r.ids[0..20]);
            try testing.expectEqualSlices(f32, cold, try r.rows(r.ids[20..30]));
        }
    }.run);
}

test "hy4 prefix: a restore point taken just after image ids (dead to Engram, read as the pad id) prefills the rest byte for byte as cold, its history the cold run's (HY4_BF16_TINY)" {
    const hy4_model = @import("hy4_model.zig");
    const prefix = @import("hy4_prefix.zig");
    try mlx.requireGpu();
    var t = try Tiny.open();
    defer t.close();
    const ids_f = try rawF32(try t.ref("ids", .{}));
    var ids: [64]u32 = undefined;
    for (ids_f, ids[0..ids_f.len]) |v, *d| d.* = @intFromFloat(v);
    try testing.expect(t.config.engram.n > 0 and t.config.image_id != 0);
    for (ids[18..21]) |*d| d.* = t.config.image_id;
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const m = try hy4_model.Model.init(testing.allocator, t.config, &t.pack, es);
    defer m.deinit();
    var eo: Ops = .{ .s = es };
    defer eo.deinit();
    m.req().kv = .engine_default;
    _ = try eo.own(try m.forward(ids[0..21]));
    const cold_history = try testing.allocator.dupe(u32, m.req().history.items);
    defer testing.allocator.free(cold_history);
    const cold = try host(&eo, try eo.own(try m.forward(ids[21..30])));
    // A stored point, then a live one, each restored over the prompt that ran past it.
    m.reset();
    m.req().kv = .engine_default;
    _ = try eo.own(try m.forward(ids[0..21]));
    var cp = try prefix.capture(testing.allocator, m, 21);
    defer cp.deinit(testing.allocator);
    _ = try eo.own(try m.forward(ids[21..30]));
    try prefix.restore(m, &cp, ids[0..30]);
    try testing.expectEqualSlices(u32, cold_history, m.req().history.items);
    try testing.expectEqualSlices(f32, cold, try host(&eo, try eo.own(try m.forward(ids[21..30]))));
    m.reset();
    m.req().kv = .engine_default;
    _ = try eo.own(try m.forward(ids[0..21]));
    try m.keepLivePoint();
    _ = try eo.own(try m.forward(ids[21..30]));
    try m.rollbackPoint(&m.req().live_points.items[m.req().live_points.items.len - 1], 41);
    try testing.expectEqualSlices(u32, cold_history, m.req().history.items);
    try testing.expectEqualSlices(f32, cold, try host(&eo, try eo.own(try m.forward(ids[21..30]))));
}

/// The launch settings a served prefill keeps restore points under: RAM retention on, the stride and cap's defaults.
const ServedPoints = struct {
    const state = @import("../server_state.zig");
    saved: struct { u32, u32, u32, bool },

    fn set() ServedPoints {
        const was: ServedPoints = .{ .saved = .{ state.ssm_checkpoint_stride, state.ssm_checkpoint_max, state.prefix_cache_capacity, state.prefix_cache_ram_enabled } };
        state.ssm_checkpoint_stride = state.SSM_CHECKPOINT_STRIDE_DEFAULT;
        state.ssm_checkpoint_max = 16;
        state.prefix_cache_capacity = 32;
        state.prefix_cache_ram_enabled = true;
        return was;
    }

    fn restore(self: ServedPoints) void {
        state.ssm_checkpoint_stride, state.ssm_checkpoint_max, state.prefix_cache_capacity, state.prefix_cache_ram_enabled = self.saved;
    }
};

test "hy4 prefix: a stored point taken where the request keeps its live point shares the live point's small state: the pool holds one copy of it (HY4_BF16_TINY)" {
    const hy4_model = @import("hy4_model.zig");
    const hy4_prefix = @import("hy4_prefix.zig");
    try mlx.requireGpu();
    var t = try Tiny.open();
    defer t.close();
    const a = testing.allocator;
    fwd.products = .bf16;
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const ids = [_]u32{ 11, 22, 33, 44, 55, 66, 77, 88, 99, 111, 222, 333, 444, 555, 666, 777, 888, 999, 1111, 2222, 3333 };
    const m = try hy4_model.Model.init(a, t.config, &t.pack, es);
    defer m.deinit();
    _ = mlx.mlx_array_free(try m.forwardReadout(&ids, .none));
    // The live point alone, then the live point and the stored point: the stored one adds no copy of its own.
    var held: [2]u64 = undefined;
    for (&held, 0..) |*h, i| {
        mlx.engine.close();
        mlx.engine.trim();
        const in_use0 = mlx.engine.poolTotal() - mlx.engine.poolFree();
        if (i == 0) try m.keepLivePoint() else {
            var cp = try hy4_prefix.captureKept(a, m, m.req().pos);
            defer cp.deinit(a);
            mlx.engine.close();
            h.* = (mlx.engine.poolTotal() - mlx.engine.poolFree()) -| in_use0;
            continue;
        }
        mlx.engine.close();
        h.* = (mlx.engine.poolTotal() - mlx.engine.poolFree()) -| in_use0;
        m.dropLive();
    }
    errdefer std.debug.print("hy4 prefix: a live point holds {d} B, a live and a stored point {d} B\n", .{ held[0], held[1] });
    try testing.expect(held[0] > 0 and held[1] < held[0] + held[0] / 2);
}

/// A served-like prefill: chunks of `w` over `ids`, a restore point at every chunk end kept as
/// `generate.capturePrefillCheckpoint` keeps it (a live point beside the stored one), thinned as the chunk loop thins.
fn prefillKeepingPoints(a: std.mem.Allocator, m: anytype, ids: []const u32, w: usize, cps: *std.ArrayList(@import("../ssm_cache.zig").SSMCheckpoint), peak: *u64, in_use0: u64) !void {
    const hy4_prefix = @import("hy4_prefix.zig");
    var at: usize = 0;
    while (at < ids.len) : (at += w) {
        _ = mlx.mlx_array_free(try m.forwardReadout(ids[at..@min(at + w, ids.len)], .none));
        mlx.engine.close();
        peak.* = @max(peak.*, mlx.engine.poolPeak(.process) -| in_use0);
        try m.keepLivePoint();
        // As `hy4_prefix.captureCheckpoint` with no write-through: the stored point is its position alone.
        var cp = try hy4_prefix.capture(a, m, m.req().pos);
        cp.release();
        try cps.append(a, cp);
        // The chunk loop's retention for an arch with no request of its own: the newest, up to the arch's cap.
        @import("../prefill_plan.zig").thinHeldCheckpoints(a, cps, @import("hy4_prefix.zig").checkpointMax(16), .oldest);
        mlx.engine.close();
        peak.* = @max(peak.*, (mlx.engine.poolTotal() - mlx.engine.poolFree()) -| in_use0);
    }
}

test "hy4 bill: a prefill that keeps its restore points at every chunk end leaves no more than its state and its points (HY4_BF16_TINY)" {
    const hy4_model = @import("hy4_model.zig");
    const B = @import("hy4_ledger.zig");
    try mlx.requireGpu();
    var t = try Tiny.open();
    defer t.close();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var config = try model_mod.parseConfig(io, a, std.mem.span(std.c.getenv("HY4_BF16_TINY").?));
    defer config.deinit(a);
    const served = ServedPoints.set();
    defer served.restore();
    fwd.products = .bf16;
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const len = 65536;
    const w = 512;
    const ids = try a.alloc(u32, len);
    defer a.free(ids);
    for (ids, 0..) |*d, i| d.* = @intCast((i * 7919) % 120000);
    const m = try hy4_model.Model.init(a, t.config, &t.pack, es);
    defer m.deinit();
    var cps: std.ArrayList(@import("../ssm_cache.zig").SSMCheckpoint) = .empty;
    defer {
        for (cps.items) |*cp| cp.deinit(a);
        cps.deinit(a);
    }
    mlx.engine.close();
    mlx.engine.trim();
    const in_use0 = mlx.engine.poolTotal() - mlx.engine.poolFree();
    mlx.engine.resetPoolPeak(.process);
    var peak: u64 = 0;
    try prefillKeepingPoints(a, m, ids, w, &cps, &peak, in_use0);
    // What stands at the end is the terms' own: the state and the points, with no forward's scratch beside them.
    const held = (mlx.engine.poolTotal() - mlx.engine.poolFree()) -| in_use0;
    const state = B.stateBytes(&t.config, 8, len);
    const terms = state + @min(state, B.checkpointCap(&config, w)) + B.pointsBytes(&t.config, 8, false);
    errdefer std.debug.print("[hy4-bill] {d} points held, in use at the end {d} KB against the state and points {d} KB; peak in use {d} KB\n", .{ cps.items.len, held >> 10, terms >> 10, peak >> 10 });
    try testing.expect(held <= terms);
}

test "hy4 bill: the points a model holds in RAM stay under the RAM tier's cap at the release geometry, at 8 and 16 bits, whatever the context" {
    const B = @import("hy4_ledger.zig");
    const prefix = @import("hy4_prefix.zig");
    const c = try @import("hy4_config.zig").parseText(@import("hy4_config.zig").release_config);
    for ([_]u64{ 8, 16 }) |kv| for ([_]bool{ false, true }) |drafts| {
        try testing.expect(B.pointsBytes(&c, kv, drafts) < prefix.ram_tier_max_bytes);
    };
}

test "hy4 prefix: the points a long prefill and its live record keep hold their small state alone, the same bytes at any length (HY4_BF16_TINY)" {
    const hy4_model = @import("hy4_model.zig");
    const B = @import("hy4_ledger.zig");
    const prefix = @import("hy4_prefix.zig");
    try mlx.requireGpu();
    var t = try Tiny.open();
    defer t.close();
    fwd.products = .bf16;
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const ids = try testing.allocator.alloc(u32, 16384);
    defer testing.allocator.free(ids);
    for (ids, 0..) |*d, i| d.* = @intCast((i * 7919) % 120000);
    var held: [2]u64 = undefined;
    for ([_]usize{ 8192, 16384 }, &held) |len, *h| {
        const m = try hy4_model.Model.init(testing.allocator, t.config, &t.pack, es);
        defer m.deinit();
        var live: prefix.Live = .{};
        defer live.deinit(testing.allocator);
        // Four turns of a point a chunk, each continuing the last; the first three publish.
        var at: usize = 0;
        for (0..4) |turn| {
            while (at < (turn + 1) * len / 4) : (at += 512) {
                _ = mlx.mlx_array_free(try m.forwardReadout(ids[at .. at + 512], .none));
                try m.keepLivePoint();
            }
            if (turn < 3) live.publish(m, ids[0..at], false, .{});
        }
        // One more row: the newest point's rings are its own too.
        _ = mlx.mlx_array_free(try m.forwardReadout(ids[at - 1 .. at], .none));
        h.* = try prefix.pointsHeldBytes(m);
        const points = live.points.items.len + m.req().live_points.items.len;
        errdefer std.debug.print("[hy4-points] {d} points at {d} hold {d} B; a point is billed {d} B\n", .{ points, len, h.*, B.pointBytes(&t.config, 8) });
        try testing.expectEqual(prefix.record_point_cap + hy4_model.Model.live_point_cap, points);
        try testing.expect(h.* <= points * B.pointBytes(&t.config, 8));
    }
    try testing.expectEqual(held[0], held[1]);
}

test "hy4 bill: the state a prefill leaves holds exactly its arrays' bytes, which the bill covers with its stores' slack, at 8 and 16 bits, and reserved for a drafting request the bytes the model counts as held, which its bill covers (HY4_BF16_TINY)" {
    const hy4_model = @import("hy4_model.zig");
    const B = @import("hy4_ledger.zig");
    try mlx.requireGpu();
    var t = try Tiny.open();
    defer t.close();
    const a = testing.allocator;
    fwd.products = .bf16;
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const ids = try a.alloc(u32, 8193);
    defer a.free(ids);
    for (ids, 0..) |*d, i| d.* = @intCast((i * 7919) % 120000);
    for ([_]bool{ false, true }) |served| for ([_]u64{ 8, 16 }) |kv| for ([_]usize{ 7, 1000, 4096, 4097, 8193 }) |len| {
        const m = try hy4_model.Model.init(a, t.config, &t.pack, es);
        defer m.deinit();
        m.req().kv = if (kv == 8) .engine_default else .dense;
        mlx.engine.close();
        mlx.engine.trim();
        const in_use0 = mlx.engine.poolTotal() - mlx.engine.poolFree();
        // As a served drafting request runs: its state reserved at its last row, the handover and picks kept.
        m.keep_hidden = served;
        if (served) try m.reserveFor(len);
        var at: usize = 0;
        while (at < len) : (at += 512) _ = mlx.mlx_array_free(try m.forwardReadout(ids[at..@min(at + 512, len)], .none));
        mlx.engine.close();
        mlx.engine.trim();
        const held = (mlx.engine.poolTotal() - mlx.engine.poolFree()) -| in_use0;
        var arrays: [hy4_model.max_state_arrays]Arr = undefined;
        var own: u64 = 0;
        for (m.req().state[0..t.config.layers]) |*st| for (hy4_model.stateArrays(st, &arrays)) |arr| {
            own += @import("../tensor/pool_class.zig").poolBytes(@intCast(@import("../ssm_cache.zig").arrayBytes(arr)));
        };
        const state_billed = B.stateBytes(&t.config, kv, len);
        if (served) {
            const counted = @import("hy4_prefix.zig").heldBytes(m, null);
            const billed = state_billed + B.pointsBytes(&t.config, kv, true) + B.pickBytes(&t.config) + B.hiddenBytes(&t.config) + B.markBytes(&t.config, kv);
            errdefer std.debug.print("[hy4-bill] served kv{d} state at {d}: {d} B held, {d} B counted, {d} B billed\n", .{ kv, len, held, counted, billed });
            try testing.expectEqual(held, counted);
            try testing.expect(held <= billed);
            continue;
        }
        errdefer std.debug.print("[hy4-bill] kv{d} state at {d}: {d} B held, its arrays {d} B, {d} B billed\n", .{ kv, len, held, own, state_billed });
        try testing.expectEqual(own, held);
        try testing.expect(held <= state_billed);
    };
}

/// How many of the request's state arrays are not evaluated: a graph, which holds the inputs it reads until something
/// reads it.
fn lazyStateArrays(m: anytype, layers: usize) usize {
    var lazy: usize = 0;
    var arrays: [@import("hy4_model.zig").max_state_arrays]Arr = undefined;
    for (m.req().state[0..layers]) |*st| for (@import("hy4_model.zig").stateArrays(st, &arrays)) |arr| {
        var avail = false;
        _ = mlx._mlx_array_is_available(&avail, arr);
        if (!avail) lazy += 1;
    };
    return lazy;
}

test "hy4 state: a forward leaves every state array evaluated, whatever it reads out, and takes no more waits for the GPU (HY4_BF16_TINY)" {
    const hy4_model = @import("hy4_model.zig");
    try mlx.requireGpu();
    var t = try Tiny.open();
    defer t.close();
    fwd.products = .bf16;
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    var cfg = t.config;
    // A threshold of 4096 keys, so a 2,047-row prompt reads every key and no pooled one (the release's is 2,048).
    cfg.indexer.group_topk = 1024;
    const ids = try testing.allocator.alloc(u32, 2048);
    defer testing.allocator.free(ids);
    for (ids, 0..) |*d, i| d.* = @intCast((i * 7919 + 3) % 120000);
    const m = try hy4_model.Model.init(testing.allocator, cfg, &t.pack, es);
    defer m.deinit();
    m.req().kv = .engine_default;
    var eo: Ops = .{ .s = es };
    defer eo.deinit();
    // Prefill chunks read out nothing, a decode row reads its logits: the rows the ring wraps in and the groups close in.
    const steps = [_]struct { rows: usize, readout: hy4_model.Model.Readout }{
        .{ .rows = 7, .readout = .none }, .{ .rows = 2000, .readout = .none }, .{ .rows = 1, .readout = .last }, .{ .rows = 38, .readout = .last }, .{ .rows = 1, .readout = .last },
    };
    var at: usize = 0;
    for (steps) |step| {
        const logits = try m.forwardReadout(ids[at .. at + step.rows], step.readout);
        at += step.rows;
        if (step.readout == .last) _ = try host(&eo, try eo.own(logits)) else _ = mlx.mlx_array_free(logits);
        mlx.engine.close();
        errdefer std.debug.print("[hy4-state] {d} rows read out as {s}: {d} lazy state arrays\n", .{ step.rows, @tagName(step.readout), lazyStateArrays(m, cfg.layers) });
        try testing.expectEqual(@as(usize, 0), lazyStateArrays(m, cfg.layers));
    }
    // One decode row's waits: the state's evaluation rides the forward's own, it adds none.
    const before = mlx.engine.stats().waits;
    _ = try host(&eo, try eo.own(try m.forwardReadout(ids[at .. at + 1], .last)));
    mlx.engine.close();
    const waits = mlx.engine.stats().waits - before;
    // As counted before the state rode the forward (a prefill's lazy arrays were evaluated by the next forward's reads).
    try testing.expectEqual(@as(u64, 6), waits);
}

/// The state a served prefill leaves at its end (`captureCheckpoint`): the model's state, a live point and an entry's
/// point on the same buffers, the record published, nothing decoded since.
const HeldRig = struct {
    m: *@import("hy4_model.zig").Model,
    cp: @import("../ssm_cache.zig").SSMCheckpoint,
    live: @import("hy4_prefix.zig").Live = .{},

    fn build(t: *Tiny, es: mlx.mlx_stream, ids: []const u32) !HeldRig {
        const hy4_model = @import("hy4_model.zig");
        const hy4_prefix = @import("hy4_prefix.zig");
        const m = try hy4_model.Model.init(testing.allocator, t.config, &t.pack, es);
        errdefer m.deinit();
        var at: usize = 0;
        while (at < ids.len) : (at += 512) _ = mlx.mlx_array_free(try m.forwardReadout(ids[at..@min(at + 512, ids.len)], .none));
        try m.keepLivePoint();
        var rig: HeldRig = .{ .m = m, .cp = try hy4_prefix.capture(testing.allocator, m, m.req().pos) };
        rig.live.publish(m, ids, false, .{});
        mlx.engine.close();
        mlx.engine.trim();
        return rig;
    }

    fn deinit(self: *HeldRig) void {
        self.live.deinit(testing.allocator);
        self.cp.deinit(testing.allocator);
        self.m.deinit();
    }

    fn inUse() u64 {
        mlx.engine.close();
        mlx.engine.trim();
        return mlx.engine.poolTotal() - mlx.engine.poolFree();
    }
};

test "hy4 prefix: the bytes the model holds beside an entry's point are the ones its own release frees, and the entry's point frees only its own (HY4_BF16_TINY)" {
    try mlx.requireGpu();
    var t = try Tiny.open();
    defer t.close();
    fwd.products = .bf16;
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const ids = try testing.allocator.alloc(u32, 4096);
    defer testing.allocator.free(ids);
    for (ids, 0..) |*d, i| d.* = @intCast((i * 7919) % 120000);
    const idle = HeldRig.inUse();
    // The entry's point freed (as an eviction frees it): what stays is the model's, state and records, shared buffers too.
    {
        var rig = try HeldRig.build(&t, es, ids);
        defer rig.deinit();
        const all = try heldOf(&rig, false);
        const before = HeldRig.inUse();
        rig.cp.deinit(testing.allocator);
        rig.cp = .{ .pos = 0, .layers = &.{} };
        const after = HeldRig.inUse();
        errdefer std.debug.print("[hy4-held] the entry's point frees {d} B; the model holds {d} B and the pool {d} B past idle\n", .{ before - after, all, after - idle });
        // Beside them the load's own: the placeholder a stored point's absent arrays share (`Model.placeholder`).
        try testing.expectEqual(after - idle, all + @import("../tensor/pool_class.zig").poolBytes(1));
    }
    // The model's state and records let go: what the entry's point alone holds goes, nothing it shares with a held one.
    {
        var rig = try HeldRig.build(&t, es, ids);
        defer rig.deinit();
        const apart = try heldOf(&rig, true);
        const before = HeldRig.inUse();
        rig.m.reset();
        rig.m.dropLive();
        rig.live.clear(rig.m);
        const after = HeldRig.inUse();
        errdefer std.debug.print("[hy4-held] releasing the model's state and records frees {d} B; it holds {d} B beside the entry's point\n", .{ before - after, apart });
        try testing.expectEqual(before - after, apart);
    }
}

fn heldOf(rig: *HeldRig, entry_counted: bool) !u64 {
    const hy4_prefix = @import("hy4_prefix.zig");
    var held: hy4_prefix.Held = .{ .gpa = testing.allocator };
    defer held.deinit();
    if (entry_counted) try held.see(&.{rig.cp});
    try held.addModel(rig.m);
    return held.bytes;
}

test "hy4 prefix: admission credits no Hy4 entry, whose newest points share buffers with the live record" {
    const bills = @import("../bills.zig");
    const a = std.testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var config = try model_mod.parseConfigText(io, a, "", @import("hy4_config.zig").release_config);
    defer config.deinit(a);
    // An eviction frees less than an entry's bytes while the record stands, so none is credited: that is the premise
    // of billing an entry its whole points. Arming the evict-to-admit pass for Hy4 needs the sharing solved first.
    const bill = bills.creditedAdmissionBill(&config, 10 << 30, 1 << 30, 5 << 30, 4 << 30, 4096);
    try testing.expectEqual(@as(u64, 0), bill.evictable);
    try testing.expectEqual(@as(u64, 0), bill.reclaimable);
}

/// The tiny on Sushi Cooler's stream beside an SSD-only cache whose Hy4 split cuts row segments of `span` tokens, served
/// as the scheduler serves it: a lookup, a prefill whose chunks take points the write-through stores, decode, a publish.
const SsdRig = struct {
    const prefix = @import("hy4_prefix.zig");
    const pc = @import("../prefix_cache.zig");
    const kv_disk_cache = @import("../kv_disk_cache.zig");
    const KVCache = @import("../kv_cache.zig").KVCache;

    m: *@import("hy4_model.zig").Model,
    hc: pc.HotPrefixCache,
    es: mlx.mlx_stream,
    eo: *Ops,
    /// SSD bytes the last turn's write-through and flush staged.
    written: u64 = 0,

    fn cache(io: std.Io, base: []const u8, c: *const hy4_config.Config, span: u32) !pc.HotPrefixCache {
        var hc = pc.HotPrefixCache.initWithMem(testing.allocator, 0, 0);
        errdefer hc.deinit();
        hc.ssd_first = true;
        hc.hybrid = true;
        hc.cp_thin = .min_span_recency;
        hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, io, base, "hy4-ssd", 0, 128);
        hc.disk.?.ssd_first = true;
        hc.disk.?.cp_thin = .min_span_recency;
        hc.disk.?.ssm_max_per_entry = kv_disk_cache.SSM_DISK_MAX_PER_ENTRY;
        hc.disk.?.armTestSpace(256 << 30, 1 << 40);
        hc.disk.?.enableBackgroundWriter();
        hc.disk.?.points_hold_state = true;
        try prefix.setRowSpan(testing.allocator, c, &hc.disk.?, span, 8);
        // A Cooler load's cache keeps its arrays on the engine (`arch.prefix.arraysOn`).
        hc.arrays_on = mlx.engine.stream();
        hc.disk.?.arrays_on = hc.arrays_on;
        return hc;
    }

    /// One request over `prompt`: its lookup, its prefill in chunks of `chunk` with a point at each chunk end, and `n`
    /// greedy rows; the logits of every row it ran past the restore, in `out`. Returns where the lookup left the state.
    fn turn(r: *SsdRig, prompt: []const u32, chunk: usize, n: usize, out: *std.ArrayList(f32), tokens: *std.ArrayList(u32)) !usize {
        const d = &r.hc.disk.?;
        const before = d.point_bytes_written;
        var target = try KVCache.initWithConfig(testing.allocator, r.m.config.layers, .engine_default);
        defer target.deinit();
        var off: usize = 0;
        const res = try r.hc.lookupImpl(&target, &off, @import("../arch/prefix.zig").recurrent("hy4flash", r.m), r.es, prompt, false, 0, null, &.{}, null, null, null, false);
        if (res.matched == 0) {
            r.m.reset();
            r.m.req().kv = .engine_default;
        }
        var cps: std.ArrayList(@import("../ssm_cache.zig").SSMCheckpoint) = .empty;
        defer {
            for (cps.items) |*cp| cp.deinit(testing.allocator);
            cps.deinit(testing.allocator);
        }
        var at = res.matched;
        var last: []const f32 = &.{};
        while (at < prompt.len) {
            // Chunk ends on the chunk grid, as the chunk loop cuts a restored prompt's.
            const end = @min((at / chunk + 1) * chunk, prompt.len);
            last = try host(r.eo, try r.eo.own(try r.m.forwardReadout(prompt[at..end], .last)));
            try out.appendSlice(testing.allocator, last);
            at = end;
            // As `generate.capturePrefillCheckpoint` with a write-through armed.
            try r.m.keepLivePoint();
            var cp = try prefix.capture(testing.allocator, r.m, at);
            if (at < kv_disk_cache.MIN_PERSIST_TOKENS) cp.release();
            try cps.append(testing.allocator, cp);
            var kv = try KVCache.initWithConfig(testing.allocator, r.m.config.layers, .engine_default);
            defer kv.deinit();
            kv.step = at;
            prefix.writeThrough(&r.hc, r.m, &kv, prompt[0..at], false, .{}, null, cps.items, r.es);
            d.drainWriter();
        }
        tokens.clearRetainingCapacity();
        try tokens.appendSlice(testing.allocator, prompt);
        for (0..n) |_| {
            var best: usize = 0;
            for (last, 0..) |x, i| if (x > last[best]) {
                best = i;
            };
            try tokens.append(testing.allocator, @intCast(best));
            // The last sampled token is not forwarded: the state stands one short of the tokens, as served.
            if (tokens.items.len == prompt.len + n) break;
            last = try host(r.eo, try r.eo.own(try r.m.forwardReadout(tokens.items[tokens.items.len - 1 ..], .last)));
            try out.appendSlice(testing.allocator, last);
        }
        prefix.publishLive(&r.hc, r.m, tokens.items, false, .{});
        d.drainWriter();
        r.written = d.point_bytes_written - before;
        return res.matched;
    }
};

fn withSsdRig(base: []const u8, span: u32, comptime body: fn (*SsdRig, *Tiny, []const u32, []const u8) anyerror!void) !void {
    try mlx.requireGpu();
    var t = try Tiny.open();
    defer t.close();
    fwd.products = .bf16;
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const ids = try testing.allocator.alloc(u32, 4096);
    defer testing.allocator.free(ids);
    for (ids, 0..) |*d, i| d.* = @intCast((i * 7919 + 13) % 120000);
    var eo: Ops = .{ .s = es };
    defer eo.deinit();
    var r: SsdRig = .{ .m = try @import("hy4_model.zig").Model.init(testing.allocator, t.config, &t.pack, es), .hc = try SsdRig.cache(std.testing.io, base, &t.config, span), .es = es, .eo = &eo };
    defer r.m.deinit();
    defer r.hc.deinit();
    r.m.req().kv = .engine_default;
    try body(&r, &t, ids, base);
}

test "hy4 SSD tier: a conversation another one's cold request interrupts restores its last turn from SSD at that turn's own point, the uninterrupted turn's logits, and writes each row once; a restart restores the same (HY4_BF16_TINY)" {
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(std.testing.io, &buf)];
    try withSsdRig(base, 512, struct {
        fn run(r: *SsdRig, t: *Tiny, ids: []const u32, dir: []const u8) !void {
            const B = @import("hy4_ledger.zig");
            const a = testing.allocator;
            var tokens: std.ArrayList(u32) = .empty;
            defer tokens.deinit(a);
            var want: std.ArrayList(f32) = .empty;
            defer want.deinit(a);
            var got: std.ArrayList(f32) = .empty;
            defer got.deinit(a);
            var scratch: std.ArrayList(f32) = .empty;
            defer scratch.deinit(a);
            // Turn 1 of conversation A: 1,300 tokens in chunks of 256 (points on the 512 grid go to SSD), 8 rows.
            _ = try r.turn(ids[0..1300], 256, 8, &scratch, &tokens);
            const turn1 = r.written;
            const row_bytes = B.stateBytesPerToken(&t.config, 8);
            const point_bytes = B.pointBytes(&t.config, 8) + (64 << 10);
            // The grid points' rows went to two segments once; their files hold the small state alone.
            const e = &r.hc.disk.?.entries.items[0];
            try testing.expectEqual(@as(usize, 2), e.grown.len);
            for (e.ssm_positions, e.ssm_bytes) |p, b| if (p % 512 == 0) try testing.expect(b <= point_bytes);
            try testing.expect(turn1 <= 1024 * row_bytes + 2 * (512 * row_bytes / 8 + point_bytes));
            // Turn 2 repeats turn 1's tokens and adds 300; the uninterrupted state runs it from where it stands.
            const vocab: usize = t.config.vocab;
            var a2: [1308 + 300]u32 = undefined;
            @memcpy(a2[0..1308], tokens.items);
            @memcpy(a2[1308..], ids[2000..2300]);
            {
                var ref = try @import("hy4_model.zig").Model.init(a, t.config, &t.pack, r.es);
                defer ref.deinit();
                ref.req().kv = .engine_default;
                var at: usize = 0;
                // The same rows as the turns above: the prompt on the chunk grid, then the 7 forwarded replies.
                while (at < 1300) : (at = @min((at / 256 + 1) * 256, 1300)) _ = try r.eo.own(try ref.forwardReadout(ids[at..@min((at / 256 + 1) * 256, 1300)], .last));
                for (1300..1307) |i| _ = try r.eo.own(try ref.forwardReadout(a2[i .. i + 1], .last));
                at = 1307;
                while (at < a2.len) {
                    const end = @min((at / 256 + 1) * 256, a2.len);
                    try want.appendSlice(a, try host(r.eo, try r.eo.own(try ref.forwardReadout(a2[at..end], .last))));
                    at = end;
                }
            }
            // Conversation B's cold request takes the state: A's record reaches SSD first.
            _ = try r.turn(ids[3000..3700], 256, 4, &scratch, &tokens);
            // A's next turn restores A's own point behind its reply from SSD and runs what the uninterrupted state runs.
            const matched = try r.turn(&a2, 256, 1, &got, &tokens);
            try testing.expectEqual(@as(usize, 1307), matched);
            try testing.expect(r.hc.last_restored_disk_id != null);
            try testing.expectEqualSlices(f32, want.items, got.items[0..want.items.len]);
            // Turn 2 stores the rows past the restore, not the history again: one segment and its grid point.
            if (std.c.getenv("SUSHI_HY4_SSD_TRACE") != null) std.debug.print("[hy4-ssd] turn 1 wrote {d} B, turn 2 {d} B; a row {d} B, a point's small state {d} B\n", .{ turn1, r.written, row_bytes, B.pointBytes(&t.config, 8) });
            try testing.expect(r.written <= (a2.len - 1024) * row_bytes + 2 * (512 * row_bytes / 8 + point_bytes));
            // A restart: a fresh tier on the same directory restores turn 2's grid point and the same logits past it.
            r.hc.deinit();
            r.hc = try SsdRig.cache(std.testing.io, dir, &t.config, 512);
            r.m.reset();
            got.clearRetainingCapacity();
            try testing.expectEqual(@as(usize, 1536), try r.turn(&a2, 256, 1, &got, &tokens));
            try testing.expectEqualSlices(f32, want.items[want.items.len - vocab ..], got.items[0..vocab]);
        }
    }.run);
}

test "hy4 prefix: rolling back a long turn to a short point trims row stores to the request bill and continues identically (HY4_BF16_TINY)" {
    const hy4_model = @import("hy4_model.zig");
    const B = @import("hy4_ledger.zig");
    const prefix = @import("hy4_prefix.zig");
    try mlx.requireGpu();
    var t = try Tiny.open();
    defer t.close();
    const a = testing.allocator;
    fwd.products = .bf16;
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const ids = try a.alloc(u32, 8192);
    defer a.free(ids);
    for (ids, 0..) |*d, i| d.* = @intCast((i * 7919 + 5) % 120000);
    var p2: [80]u32 = undefined;
    @memcpy(p2[0..64], ids[0..64]);
    for (p2[64..], 0..) |*d, i| d.* = @intCast((i * 3571 + 99) % 120000);
    const next_tok = [_]u32{4242};
    var eo: Ops = .{ .s = es };
    defer eo.deinit();
    const want_append, const want_decode = blk: {
        const ref = try hy4_model.Model.init(a, t.config, &t.pack, es);
        defer ref.deinit();
        ref.req().kv = .engine_default;
        _ = mlx.mlx_array_free(try ref.forwardReadout(p2[0..64], .none));
        const wa = try host(&eo, try eo.own(try ref.forwardReadout(p2[64..80], .last)));
        const wd = try host(&eo, try eo.own(try ref.forwardReadout(&next_tok, .last)));
        break :blk .{ wa, wd };
    };
    const m = try hy4_model.Model.init(a, t.config, &t.pack, es);
    defer m.deinit();
    m.req().kv = .engine_default;
    var hc = @import("../prefix_cache.zig").HotPrefixCache.init(a, 4);
    defer hc.deinit();
    mlx.engine.close();
    mlx.engine.trim();
    const in_use0 = mlx.engine.poolTotal() - mlx.engine.poolFree();
    _ = mlx.mlx_array_free(try m.forwardReadout(ids[0..64], .none));
    try m.keepLivePoint();
    var at: usize = 64;
    while (at < 8192) : (at += 512) {
        _ = mlx.mlx_array_free(try m.forwardReadout(ids[at..@min(at + 512, 8192)], .none));
    }
    prefix.publishLive(&hc, m, ids[0..8192], false, .{});
    var target = try @import("../kv_cache.zig").KVCache.initWithConfig(a, t.config.layers, .engine_default);
    defer target.deinit();
    var off: usize = 0;
    const res = prefix.rollbackLive(&hc, m, &target, &off, es, &p2, false, .{}, null) orelse return error.TestExpectedMatch;
    try testing.expectEqual(@as(usize, 64), res.matched);
    const g: usize = t.config.indexer.group;
    var max_cap: usize = 0;
    var max_pooled_cap: usize = 0;
    for (m.req().state[0..t.config.layers]) |st| {
        if (st.rows) |r| max_cap = @max(max_cap, r.cap);
        if (st.pooled) |r| max_pooled_cap = @max(max_pooled_cap, r.cap);
    }
    const got_append = blk: {
        var step_ops: Ops = .{ .s = es };
        defer step_ops.deinit();
        break :blk try a.dupe(f32, try host(&step_ops, try step_ops.own(try m.forwardReadout(p2[64..80], .last))));
    };
    defer a.free(got_append);
    mlx.engine.close();
    mlx.engine.trim();
    const held = (mlx.engine.poolTotal() - mlx.engine.poolFree()) -| in_use0;
    const state_billed = B.stateBytes(&t.config, 8, 80);
    const with_points = state_billed + B.pointsBytes(&t.config, 8, false);
    errdefer std.debug.print("[hy4-rollback] cap={d} (bill {d}), pooled cap={d} (bill {d}), held={d} B, state+ramTier={d} B\n", .{ max_cap, B.capRows(80), max_pooled_cap, B.capRows(80 / g), held, with_points });
    try testing.expect(max_cap <= B.capRows(80));
    try testing.expect(max_pooled_cap <= B.capRows(80 / g));
    try testing.expect(held <= with_points);
    try testing.expectEqualSlices(f32, want_append, got_append);
    const got_decode = try host(&eo, try eo.own(try m.forwardReadout(&next_tok, .last)));
    try testing.expectEqualSlices(f32, want_decode, got_decode);
}

test "hy4 prefix: RAM+SSD mode flushes live restore points to SSD on handover and restores cold logits (HY4_BF16_TINY)" {
    var tmp = std.testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(std.testing.io, &buf)];
    try withSsdRig(base, 512, struct {
        fn run(r: *SsdRig, t: *Tiny, ids: []const u32, _: []const u8) !void {
            const a = testing.allocator;
            const want = blk: {
                var ref = try @import("hy4_model.zig").Model.init(a, t.config, &t.pack, r.es);
                defer ref.deinit();
                ref.req().kv = .engine_default;
                _ = try r.eo.own(try ref.forwardReadout(ids[0..600], .none));
                break :blk try host(r.eo, try r.eo.own(try ref.forwardReadout(ids[600..610], .last)));
            };
            r.hc.ssd_first = false;
            r.hc.disk.?.ssd_first = false;
            r.hc.max_entries = 4;
            r.hc.ram_enabled = true;
            _ = try r.eo.own(try r.m.forwardReadout(ids[0..600], .none));
            try r.m.keepLivePoint();
            var cp = try SsdRig.prefix.capture(a, r.m, 600);
            defer cp.deinit(a);
            cp.release();
            SsdRig.prefix.publishLive(&r.hc, r.m, ids[0..600], false, .{});
            SsdRig.prefix.handover(&r.hc, r.m, r.es);
            r.hc.disk.?.drainWriter();
            r.m.reset();
            var target = try @import("../kv_cache.zig").KVCache.initWithConfig(a, t.config.layers, .engine_default);
            defer target.deinit();
            var off: usize = 0;
            const res = try r.hc.lookupImpl(&target, &off, @import("../arch/prefix.zig").recurrent("hy4flash", r.m), r.es, ids[0..610], false, 0, null, &.{}, null, null, null, false);
            try testing.expectEqual(@as(usize, 600), res.matched);
            const got = try host(r.eo, try r.eo.own(try r.m.forwardReadout(ids[600..610], .last)));
            try testing.expectEqualSlices(f32, want, got);
        }
    }.run);
}

test "hy4 model: a rollback to a mark taken with no hidden state clears last_hidden and continues identically (HY4_BF16_TINY)" {
    const hy4_model = @import("hy4_model.zig");
    try mlx.requireGpu();
    var t = try Tiny.open();
    defer t.close();
    const a = testing.allocator;
    fwd.products = .bf16;
    const ids_f = try rawF32(try t.ref("ids", .{}));
    var ids: [64]u32 = undefined;
    for (ids_f, ids[0..ids_f.len]) |v, *d| d.* = @intFromFloat(v);
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    var eo: Ops = .{ .s = es };
    defer eo.deinit();
    const want = blk: {
        const ref = try hy4_model.Model.init(a, t.config, &t.pack, es);
        defer ref.deinit();
        ref.keep_hidden = true;
        break :blk try host(&eo, try eo.own(try ref.forwardReadout(ids[0..10], .last)));
    };
    const m = try hy4_model.Model.init(a, t.config, &t.pack, es);
    defer m.deinit();
    m.keep_hidden = true;
    var mk = try m.mark();
    defer m.dropMark(&mk);
    try testing.expect(!mk.hidden);
    _ = mlx.mlx_array_free(try m.forwardReadout(ids[0..10], .last));
    try testing.expect(m.req().last_hidden != null);
    try m.rollbackTo(&mk, &.{});
    try testing.expect(m.req().last_hidden == null);
    const got = try host(&eo, try eo.own(try m.forwardReadout(ids[0..10], .last)));
    try testing.expectEqualSlices(f32, want, got);
}

test "hy4 bill: a load's preflight prices its explicit context on the room the weights leave" {
    const scheduler = @import("../scheduler.zig");
    const a = std.testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var config = try model_mod.parseConfigText(io, a, "", hy4_config.release_config);
    defer config.deinit(a);
    const Seen = struct {
        var budget: u64 = 0;
        fn bill(_: *const model_mod.ModelConfig, room: u64) ?@import("../model_load.zig").LoadServingBill {
            budget = room;
            return null;
        }
    };
    const was = scheduler.load_serving_bill;
    defer scheduler.load_serving_bill = was;
    scheduler.load_serving_bill = &Seen.bill;
    const weights: u64 = 80 << 30;
    const avail: u64 = 100 << 30;
    _ = @import("../model_load.zig").preflightServing(&config, weights, false, .{ .bytes = avail, .free = avail });
    try testing.expectEqual(avail - weights, Seen.budget);
}

const ChunkFlags = struct {
    launch: @import("../settings_resolve.zig").Launch,

    fn save() ChunkFlags {
        return .{ .launch = @import("../settings_resolve.zig").launch };
    }
    fn restore(self: ChunkFlags) void {
        @import("../settings_resolve.zig").launch = self.launch;
    }
};

fn boundsOf(pick: @import("../bills.zig").ChunkBoundsPick) !@import("../bills.zig").ChunkBounds {
    return switch (pick) {
        .bounds => |b| b,
        else => error.TestExpectedBounds,
    };
}

fn refusedOf(pick: @import("../bills.zig").ChunkBoundsPick) ![]const u8 {
    return switch (pick) {
        .refused => |line| line,
        else => error.TestExpectedRefusal,
    };
}

test "hy4 bill: with no bound named, the max is 4096, and a given min wider than it raises it" {
    const bills = @import("../bills.zig");
    const sr = @import("../settings_resolve.zig");
    const saved = ChunkFlags.save();
    defer saved.restore();
    sr.launch = .{};
    var config = model_mod.ModelConfig{ .model_type = "hy_v4_flash" };
    var buf: [256]u8 = undefined;
    try testing.expectEqual(bills.ChunkBounds{ .min = 512, .max = 4096 }, try boundsOf(bills.prefillChunkBounds(&buf, &config)));
    // A given min above the model's default max raises the max to it (no max was named).
    sr.launch.prefill_chunk_min = 8192;
    try testing.expectEqual(bills.ChunkBounds{ .min = 8192, .max = 8192, .max_given = true, .min_given = true, .raised_by_min = true }, try boundsOf(bills.prefillChunkBounds(&buf, &config)));
    // Above a GIVEN max it is the launch error by name.
    sr.launch = .{ .prefill_chunk_max = 4096, .prefill_chunk_min = 8192 };
    try testing.expectEqualStrings("--prefill-chunk-min 8192 is above --prefill-chunk-max 4096", try refusedOf(bills.prefillChunkBounds(&buf, &config)));
    // A bound-free Hy4 load takes the 8192 base it always had, for the checkpoint stride and the bills.
    sr.launch = .{};
    try testing.expectEqual(@as(u32, 8192), bills.checkpointStrideBase(&config));
    try testing.expectEqual(@as(u32, 8192), @import("../model_load.zig").ssdHoldStride(&config, 256));
    // A named max moves both, so a 2048-grid point is on the SSD tier's hold grid.
    sr.launch = .{ .prefill_chunk_max = 2048 };
    try testing.expectEqual(@as(u32, 2048), bills.checkpointStrideBase(&config));
    try testing.expectEqual(@as(u32, 2048), @import("../model_load.zig").ssdHoldStride(&config, 256));
    sr.launch = .{ .prefill_chunk_max = 8192 };
    try testing.expectEqual(bills.ChunkBounds{ .min = 512, .max = 8192, .max_given = true }, try boundsOf(bills.prefillChunkBounds(&buf, &config)));
}

test "hy4 bill: a load's chunk bounds default to 512 and 4096; a flag outranks the file; a bound other than 512, 1024, 2048, 4096 or 8192 rows, from any source, or a min above the max is refused by name" {
    const bills = @import("../bills.zig");
    const sr = @import("../settings_resolve.zig");
    const saved = ChunkFlags.save();
    defer saved.restore();
    sr.launch = .{};
    var config = model_mod.ModelConfig{ .model_type = "hy_v4_flash" };
    var buf: [256]u8 = undefined;
    try testing.expectEqual(bills.ChunkBounds{ .min = 512, .max = 4096 }, try boundsOf(bills.prefillChunkBounds(&buf, &config)));
    config.prefill_chunk_max_override = 2048;
    config.prefill_chunk_min_override = 1024;
    try testing.expectEqual(bills.ChunkBounds{ .min = 1024, .max = 2048, .max_given = true, .min_given = true }, try boundsOf(bills.prefillChunkBounds(&buf, &config)));
    sr.launch.prefill_chunk_max = 4096;
    sr.launch.prefill_chunk_min = 4096;
    try testing.expectEqual(bills.ChunkBounds{ .min = 4096, .max = 4096, .max_given = true, .min_given = true }, try boundsOf(bills.prefillChunkBounds(&buf, &config)));
    // A flag min under the file's max still binds the max the load takes.
    sr.launch = .{ .prefill_chunk_min = 2048 };
    try testing.expectEqual(bills.ChunkBounds{ .min = 2048, .max = 2048, .max_given = true, .min_given = true }, try boundsOf(bills.prefillChunkBounds(&buf, &config)));
    config.prefill_chunk_min_override = null;
    config.prefill_chunk_max_override = null;
    sr.launch = .{ .prefill_chunk_max = 3000 };
    try testing.expectEqualStrings("--prefill-chunk-max 3000 is not a prefill chunk this model runs: 512, 1024, 2048, 4096 or 8192 rows", try refusedOf(bills.prefillChunkBounds(&buf, &config)));
    sr.launch = .{ .prefill_chunk_min = 256 };
    try testing.expectEqualStrings("--prefill-chunk-min 256 is not a prefill chunk this model runs: 512, 1024, 2048, 4096 or 8192 rows", try refusedOf(bills.prefillChunkBounds(&buf, &config)));
    sr.launch = .{};
    config.prefill_chunk_max_override = 9000;
    try testing.expectEqualStrings("model-settings.json prefill_chunk_max 9000 is not a prefill chunk this model runs: 512, 1024, 2048, 4096 or 8192 rows", try refusedOf(bills.prefillChunkBounds(&buf, &config)));
    config.prefill_chunk_max_override = 4096;
    config.prefill_chunk_min_override = 1000;
    try testing.expectEqualStrings("model-settings.json prefill_chunk_min 1000 is not a prefill chunk this model runs: 512, 1024, 2048, 4096 or 8192 rows", try refusedOf(bills.prefillChunkBounds(&buf, &config)));
    config.prefill_chunk_min_override = null;
    config.prefill_chunk_max_override = 1024;
    sr.launch.prefill_chunk_min = 2048;
    try testing.expectEqualStrings("--prefill-chunk-min 2048 is above model-settings.json prefill_chunk_max 1024", try refusedOf(bills.prefillChunkBounds(&buf, &config)));
}

test "hy4 bill: a load whose chunk is fixed runs and prices exactly that chunk: no per-request chooser, no adaptive width, no merged tail" {
    const bills = @import("../bills.zig");
    const prefill_plan = @import("../prefill_plan.zig");
    const sr = @import("../settings_resolve.zig");
    const saved = ChunkFlags.save();
    defer saved.restore();
    sr.launch = .{};
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var config = try model_mod.parseConfigText(io, a, "", hy4_config.release_config);
    defer config.deinit(a);
    // Unfixed, the load keeps its per-request chooser (a load that took no chunk at load).
    try testing.expect(config.perRequestPrefillChunk());
    config.static_prefill_chunk = 2048;
    config.pinned_prefill_chunk = 2048;
    try testing.expect(!config.perRequestPrefillChunk());
    try testing.expect(!bills.perRequestPrefillChunkEnabled(&config));
    try testing.expect(!bills.adaptivePrefillChunkEnabled(&config));
    try testing.expectEqual(@as(u32, 2048), bills.requestPrefillPick(&config, 100_000, 64, 8, 1 << 40, false, .{}));
    try testing.expectEqual(@as(u32, 2048), bills.requestPrefillPick(&config, 100, 64, 8, 0, false, .{}));
    try testing.expectEqual(@as(u32, 2048), bills.widestServedRung(&config));
    try testing.expectEqual(@as(u64, 2048), bills.sushiContextWidth(&config, 65536));
    // A prompt one short tail past the chunk runs the chunk, then the tail: no forward is wider than 2048 rows.
    try testing.expectEqual(@as(u64, 2048), bills.requestForwardRows(&config, 2048 + 300, 2048, 0));
    try testing.expectEqual(@as(u64, 300), bills.requestForwardRows(&config, 300, 2048, 0));
    try testing.expectEqual(@as(usize, 2048), prefill_plan.chunkEndFor(true, 0, 2048 + 300, 2048, false, 0, 0, false));
    try testing.expectEqual(@as(usize, 2048 + 300), prefill_plan.chunkEndFor(true, 2048, 2048 + 300, 2048, false, 0, 0, false));
    try testing.expectEqual(@as(usize, 2048 + 300), prefill_plan.chunkEndFor(false, 0, 2048 + 300, 2048, false, 0, 0, false));
}

test "hy4 bill: a load's fixed chunk is the width its admission bills at every prompt length, the forward's own (HY4 release config)" {
    const bills = @import("../bills.zig");
    const prefill_plan = @import("../prefill_plan.zig");
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var config = try model_mod.parseConfigText(io, a, "", hy4_config.release_config);
    defer config.deinit(a);
    config.static_prefill_chunk = 8192;
    config.pinned_prefill_chunk = 8192;
    for ([_]usize{ 4096, 16_384, 131_072, 1_000_000 }) |seq| {
        const bill = bills.prefillAdmissionBill(&config, null, seq, 64, null, false, null, .{});
        errdefer std.debug.print("[hy4-bill] a prompt of {d}: billed at {d} rows\n", .{ seq, bill.chunk });
        try testing.expectEqual(@as(u64, 8192), bill.chunk);
        try testing.expectEqual(@as(usize, 8192), prefill_plan.prefillWidth(&config, seq, 8192));
    }
}
