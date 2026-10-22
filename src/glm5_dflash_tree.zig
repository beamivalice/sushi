//! GLM draft-tree construction and exact target-decision acceptance.
//! Lattice and best-first policy adapted from mlx-serve/TensorFold (MIT; see NOTICE).
const std = @import("std");
const mlx = @import("mlx.zig");
const Selector = @import("dflash.zig").Selector;

test "GLM DFlash tree rejects siblings and caps before state commit" {
    const tokens = [_]u32{ 10, 11, 12, 13, 14 };
    const parents = [_]i32{ -1, 0, 0, 2, 3 };
    const targets = [_]u32{ 12, 99, 13, 14, 15 };
    const result = try accept(&tokens, &parents, &targets, 3, &.{});
    try std.testing.expectEqualSlices(u32, &.{ 0, 2, 3 }, result.rows[0..result.count]);
    try std.testing.expectEqual(@as(?u32, 14), result.pending);
    const stopped = try accept(&tokens, &parents, &targets, 5, &.{13});
    try std.testing.expectEqualSlices(u32, &.{ 0, 2, 3 }, stopped.rows[0..stopped.count]);
    try std.testing.expect(stopped.stopped);
    try std.testing.expect(stopped.pending == null);
    try std.testing.expectError(error.InvalidGlmDraftTree, accept(&tokens, &.{ -1, 3, 0, 2, 3 }, &targets, 5, &.{}));
}

test "GLM DFlash tree preserves target pending token after zero draft acceptance" {
    const result = try accept(&.{ 10, 11 }, &.{ -1, 0 }, &.{ 20, 21 }, 2, &.{});
    try std.testing.expectEqual(@as(usize, 1), result.count);
    try std.testing.expectEqual(@as(?u32, 20), result.pending);
    try std.testing.expectError(error.InvalidGlmDraftBudget, accept(&.{10}, &.{-1}, &.{20}, 0, &.{}));
}

pub const Lattice = struct {
    m: usize,
    k: usize,
    cands: []i32,
    unary: []f32,
    e0: []f32,
    e: []f32,

    pub fn deinit(self: *Lattice, allocator: std.mem.Allocator) void {
        allocator.free(self.cands);
        if (self.unary.len > 0) allocator.free(self.unary);
        if (self.e0.len > 0) allocator.free(self.e0);
        if (self.e.len > 0) allocator.free(self.e);
    }
};

pub fn lattice(
    allocator: std.mem.Allocator,
    sel: *const Selector,
    top_k: u32,
    blk_hidden: mlx.mlx_array,
    draft_logits: mlx.mlx_array,
    anchor_id: u32,
    s: mlx.mlx_stream,
) !Lattice {
    const dl_shape = mlx.getShape(draft_logits);
    const hs = mlx.getShape(blk_hidden);
    const ps = mlx.getShape(sel.pred_codebook);
    if (dl_shape.len != 3 or dl_shape[0] != 1 or dl_shape[1] < 1 or dl_shape[1] > 15 or dl_shape[2] < 1 or top_k == 0 or top_k > 64 or hs.len != 3 or hs[0] != 1 or hs[1] != dl_shape[1] + 1 or ps.len != 2 or ps[0] < dl_shape[2] or anchor_id >= ps[0] or !std.mem.eql(c_int, ps, mlx.getShape(sel.succ_codebook))) return error.InvalidGlmDraftLattice;
    const m: usize = @intCast(dl_shape[1]);
    const vocab: c_int = dl_shape[2];
    const k: usize = @min(@as(usize, top_k), @as(usize, @intCast(vocab)));
    const hsh = mlx.getShape(blk_hidden);
    std.debug.assert(hsh[1] == dl_shape[1] + 1); // anchor row present on hidden

    // ── Candidates + unary logits ──
    var cands_i32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cands_i32);
    var unary_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(unary_f32);
    const fused = try @import("glm5_dflash_topk.zig").apply(s, draft_logits, @intCast(k));
    if (fused) |picked| {
        _ = mlx.mlx_array_free(cands_i32);
        cands_i32 = picked.ids;
        _ = mlx.mlx_array_free(unary_f32);
        unary_f32 = picked.unary;
    } else {
        var part = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(part);
        try mlx.check(mlx.mlx_argpartition_axis(&part, draft_logits, vocab - @as(c_int, @intCast(k)), 2, s));
        var cands_raw = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cands_raw);
        const start = [_]c_int{ 0, 0, vocab - @as(c_int, @intCast(k)) };
        const stop = [_]c_int{ 1, @intCast(m), vocab };
        const strides = [_]c_int{ 1, 1, 1 };
        try mlx.check(mlx.mlx_slice(&cands_raw, part, &start, 3, &stop, 3, &strides, 3, s));
        try mlx.check(mlx.mlx_astype(&cands_i32, cands_raw, .int32, s));
        var unary = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(unary);
        try mlx.check(mlx.mlx_take_along_axis(&unary, draft_logits, cands_i32, 2, s));
        try mlx.check(mlx.mlx_astype(&unary_f32, unary, .float32, s));
    }

    // ── Hidden projection over the draft rows ──
    var hidden_rows = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hidden_rows);
    {
        const start = [_]c_int{ 0, 1, 0 };
        const stop = [_]c_int{ 1, hsh[1], hsh[2] };
        const strides = [_]c_int{ 1, 1, 1 };
        try mlx.check(mlx.mlx_slice(&hidden_rows, blk_hidden, &start, 3, &stop, 3, &strides, 3, s));
    }
    const hp = try sel.hidden_projection.apply(hidden_rows, s);
    defer _ = mlx.mlx_array_free(hp);
    const rank: c_int = mlx.getShape(hp)[2];

    // ── Codebook rows for every candidate ──
    var cands_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cands_flat);
    {
        const flat_shape = [_]c_int{@intCast(m * k)};
        try mlx.check(mlx.mlx_reshape(&cands_flat, cands_i32, &flat_shape, 1, s));
    }
    var succ_rows = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(succ_rows);
    var pred_rows = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(pred_rows);
    {
        var succ_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(succ_flat);
        try mlx.check(mlx.mlx_take_axis(&succ_flat, sel.succ_codebook, cands_flat, 0, s));
        var pred_flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pred_flat);
        try mlx.check(mlx.mlx_take_axis(&pred_flat, sel.pred_codebook, cands_flat, 0, s));
        const rows_shape = [_]c_int{ @intCast(m), @intCast(k), rank };
        try mlx.check(mlx.mlx_reshape(&succ_rows, succ_flat, &rows_shape, 3, s));
        try mlx.check(mlx.mlx_reshape(&pred_rows, pred_flat, &rows_shape, 3, s));
    }

    // ── Edge scores: anchor row [k] + pairwise [m-1, k, k] ──
    var e0_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(e0_f32);
    {
        const aid: i32 = @intCast(anchor_id);
        const a_shape = [_]c_int{1};
        const aid_arr = mlx.mlx_array_new_data(&aid, &a_shape, 1, .int32);
        defer _ = mlx.mlx_array_free(aid_arr);
        var anchor_row = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(anchor_row);
        try mlx.check(mlx.mlx_take_axis(&anchor_row, sel.pred_codebook, aid_arr, 0, s)); // [1, rank]
        var h0 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(h0);
        {
            var cut = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(cut);
            const start = [_]c_int{ 0, 0, 0 };
            const stop = [_]c_int{ 1, 1, rank };
            const strides = [_]c_int{ 1, 1, 1 };
            try mlx.check(mlx.mlx_slice(&cut, hp, &start, 3, &stop, 3, &strides, 3, s));
            const h0_shape = [_]c_int{ 1, rank };
            try mlx.check(mlx.mlx_reshape(&h0, cut, &h0_shape, 2, s));
        }
        var ah = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ah);
        try mlx.check(mlx.mlx_multiply(&ah, anchor_row, h0, s)); // [1, rank]
        var succ0_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(succ0_t);
        {
            var succ0 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(succ0);
            const start = [_]c_int{ 0, 0, 0 };
            const stop = [_]c_int{ 1, @intCast(k), rank };
            const strides = [_]c_int{ 1, 1, 1 };
            try mlx.check(mlx.mlx_slice(&succ0, succ_rows, &start, 3, &stop, 3, &strides, 3, s));
            var succ0_2d = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(succ0_2d);
            const s2 = [_]c_int{ @intCast(k), rank };
            try mlx.check(mlx.mlx_reshape(&succ0_2d, succ0, &s2, 2, s));
            const perm = [_]c_int{ 1, 0 };
            try mlx.check(mlx.mlx_transpose_axes(&succ0_t, succ0_2d, &perm, 2, s));
        }
        var e0 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(e0);
        try mlx.check(mlx.mlx_matmul(&e0, ah, succ0_t, s)); // [1, k]
        try mlx.check(mlx.mlx_astype(&e0_f32, e0, .float32, s));
    }
    var e_f32: mlx.mlx_array = .{ .ctx = null };
    defer if (e_f32.ctx != null) {
        _ = mlx.mlx_array_free(e_f32);
    };
    if (m > 1) {
        // A = pred_rows[0..m-1] ⊙ H[1..m]  → [m-1, k, rank]
        var pred_head = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pred_head);
        {
            const start = [_]c_int{ 0, 0, 0 };
            const stop = [_]c_int{ @intCast(m - 1), @intCast(k), rank };
            const strides = [_]c_int{ 1, 1, 1 };
            try mlx.check(mlx.mlx_slice(&pred_head, pred_rows, &start, 3, &stop, 3, &strides, 3, s));
        }
        var h_tail = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(h_tail);
        {
            var cut = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(cut);
            const start = [_]c_int{ 0, 1, 0 };
            const stop = [_]c_int{ 1, @intCast(m), rank };
            const strides = [_]c_int{ 1, 1, 1 };
            try mlx.check(mlx.mlx_slice(&cut, hp, &start, 3, &stop, 3, &strides, 3, s));
            const t_shape = [_]c_int{ @intCast(m - 1), 1, rank };
            try mlx.check(mlx.mlx_reshape(&h_tail, cut, &t_shape, 3, s));
        }
        var a_mat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(a_mat);
        try mlx.check(mlx.mlx_multiply(&a_mat, pred_head, h_tail, s));
        var succ_tail_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(succ_tail_t);
        {
            var succ_tail = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(succ_tail);
            const start = [_]c_int{ 1, 0, 0 };
            const stop = [_]c_int{ @intCast(m), @intCast(k), rank };
            const strides = [_]c_int{ 1, 1, 1 };
            try mlx.check(mlx.mlx_slice(&succ_tail, succ_rows, &start, 3, &stop, 3, &strides, 3, s));
            const perm = [_]c_int{ 0, 2, 1 };
            try mlx.check(mlx.mlx_transpose_axes(&succ_tail_t, succ_tail, &perm, 3, s));
        }
        var e = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(e);
        try mlx.check(mlx.mlx_matmul(&e, a_mat, succ_tail_t, s)); // [m-1, k, k]
        e_f32 = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&e_f32, e, .float32, s));
    }

    // ── ONE batched eval, then the host trace ──
    {
        const eval_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(eval_vec);
        _ = mlx.mlx_vector_array_append_value(eval_vec, cands_i32);
        _ = mlx.mlx_vector_array_append_value(eval_vec, unary_f32);
        _ = mlx.mlx_vector_array_append_value(eval_vec, e0_f32);
        if (e_f32.ctx != null) _ = mlx.mlx_vector_array_append_value(eval_vec, e_f32);
        try mlx.check(mlx.mlx_eval(eval_vec));
    }
    const cand_data = mlx.mlx_array_data_int32(cands_i32) orelse return error.MlxArrayDataNull;
    const unary_data = mlx.mlx_array_data_float32(unary_f32) orelse return error.MlxArrayDataNull;
    const e0_data = mlx.mlx_array_data_float32(e0_f32) orelse return error.MlxArrayDataNull;
    var lat = Lattice{ .m = m, .k = k, .cands = try allocator.dupe(i32, cand_data[0 .. m * k]), .unary = &.{}, .e0 = &.{}, .e = &.{} };
    errdefer lat.deinit(allocator);
    lat.unary = try allocator.dupe(f32, unary_data[0 .. m * k]);
    lat.e0 = try allocator.dupe(f32, e0_data[0..k]);
    if (e_f32.ctx != null) {
        const e_data = mlx.mlx_array_data_float32(e_f32) orelse return error.MlxArrayDataNull;
        lat.e = try allocator.dupe(f32, e_data[0 .. (m - 1) * k * k]);
    }
    return lat;
}

pub const TreeParams = struct {
    max_nodes: usize,
    children: usize = 4,
    tau: f32 = 1.5,
    edge_w: f32 = 0.6,
    temperature: f32 = 1.0,
    /// `[m, k]` Gumbel noise the verify rows draw at the candidates, weighted
    /// into the scores (null for a greedy target).
    noise: ?[]const f32 = null,
    noise_w: f32 = 0.7,
};

/// Nodes in the order taken (a parent always before its children): `tokens`,
/// `parents` (node index, -1 = under the anchor), `depth` (0 = position 0) and `cond`, the
/// selector's probability of the node among its siblings' candidates.
pub const DraftTree = struct {
    tokens: []u32,
    parents: []i32,
    depth: []u32,
    cond: []f32,

    pub fn deinit(self: *DraftTree, allocator: std.mem.Allocator) void {
        allocator.free(self.tokens);
        allocator.free(self.parents);
        allocator.free(self.depth);
        allocator.free(self.cond);
    }

    /// Keeps the nodes at most `rows` deep (the root is row 1) and then the first `keep` of them in the order taken; the
    /// survivors still follow their parents.
    pub fn cut(self: *DraftTree, allocator: std.mem.Allocator, rows: usize, keep: usize) !void {
        var kept: usize = 0;
        var moved: [16]i32 = undefined;
        for (0..self.tokens.len) |i| {
            if (self.depth[i] + 2 > rows or kept >= keep) continue;
            moved[i] = @intCast(kept);
            self.tokens[kept] = self.tokens[i];
            self.depth[kept] = self.depth[i];
            self.cond[kept] = self.cond[i];
            self.parents[kept] = if (self.parents[i] < 0) -1 else moved[@intCast(self.parents[i])];
            kept += 1;
        }
        self.tokens = try allocator.realloc(self.tokens, kept);
        self.parents = try allocator.realloc(self.parents, kept);
        self.depth = try allocator.realloc(self.depth, kept);
        self.cond = try allocator.realloc(self.cond, kept);
    }
};

/// The best-first tree with its nodes renumbered depth-first.
pub fn bestFirstTree(allocator: std.mem.Allocator, lat: *const Lattice, p: TreeParams) !DraftTree {
    var tree = try bestFirstNodes(allocator, lat, p);
    errdefer tree.deinit(allocator);
    try preorder(allocator, &tree);
    return tree;
}

/// The nodes in the order taken: node n is the n-th likeliest, so any prefix is the best tree of that size.
pub fn bestFirstNodes(allocator: std.mem.Allocator, lat: *const Lattice, p: TreeParams) !DraftTree {
    try validateLattice(lat, p);
    const k = lat.k;
    const Item = struct { value: f32, parent: i32, depth: u32, cand: u32, step: f32 };
    var queue: std.ArrayList(Item) = .empty;
    defer queue.deinit(allocator);
    var tree = DraftTree{ .tokens = &.{}, .parents = &.{}, .depth = &.{}, .cond = &.{} };
    errdefer tree.deinit(allocator);
    tree.tokens = try allocator.alloc(u32, p.max_nodes);
    tree.parents = try allocator.alloc(i32, p.max_nodes);
    tree.depth = try allocator.alloc(u32, p.max_nodes);
    tree.cond = try allocator.alloc(f32, p.max_nodes);
    var scores: [64]f32 = undefined;
    std.debug.assert(k <= scores.len);

    // Children of a node (or the anchor, cand == null) at `depth`, pushed as log-softmax + parent value.
    const Push = struct {
        fn run(alloc: std.mem.Allocator, q: *std.ArrayList(Item), l: *const Lattice, sc: []f32, pp: TreeParams, parent: i32, parent_cand: ?u32, depth: u32, base: f32) !void {
            const kk = l.k;
            const t = @max(pp.temperature, 1e-6);
            var mx: f32 = -std.math.inf(f32);
            for (sc, 0..) |*v, j| {
                const edge = if (parent_cand) |a| l.e[(@as(usize, depth) - 1) * kk * kk + a * kk + j] else l.e0[j];
                var raw = (l.unary[@as(usize, depth) * kk + j] + pp.edge_w * edge) / t;
                if (pp.noise) |nz| raw += pp.noise_w * nz[@as(usize, depth) * kk + j];
                v.* = raw / pp.tau;
                mx = @max(mx, v.*);
            }
            var total: f32 = 0;
            for (sc) |v| total += @exp(v - mx);
            const lse = mx + @log(total);
            var taken: [64]bool = @splat(false);
            for (0..@min(pp.children, kk)) |_| {
                var best: usize = 0;
                var best_v: f32 = -std.math.inf(f32);
                for (sc, 0..) |v, j| if (!taken[j] and v > best_v) {
                    best_v = v;
                    best = j;
                };
                taken[best] = true;
                try q.append(alloc, .{ .value = base + best_v - lse, .parent = parent, .depth = depth, .cand = @intCast(best), .step = @exp(best_v - lse) });
            }
        }
    };
    try Push.run(allocator, &queue, lat, scores[0..k], p, -1, null, 0, 0);
    var n: usize = 0;
    while (n < p.max_nodes and queue.items.len > 0) {
        var bi: usize = 0;
        for (queue.items, 0..) |it, i| if (it.value > queue.items[bi].value) {
            bi = i;
        };
        const it = queue.swapRemove(bi);
        tree.tokens[n] = @intCast(lat.cands[@as(usize, it.depth) * k + it.cand]);
        tree.parents[n] = it.parent;
        tree.depth[n] = it.depth;
        tree.cond[n] = it.step;
        if (it.depth + 1 < lat.m) try Push.run(allocator, &queue, lat, scores[0..k], p, @intCast(n), it.cand, it.depth + 1, it.value);
        n += 1;
    }
    if (n < p.max_nodes) {
        tree.tokens = try allocator.realloc(tree.tokens, n);
        tree.parents = try allocator.realloc(tree.parents, n);
        tree.depth = try allocator.realloc(tree.depth, n);
        tree.cond = try allocator.realloc(tree.cond, n);
    }
    return tree;
}

/// Renumber the nodes depth-first, each node's children in the order they
/// were taken (best first): the likeliest path lands on consecutive rows, so
/// a round that keeps it moves no KV rows.
pub fn preorder(allocator: std.mem.Allocator, t: *DraftTree) !void {
    const n = t.tokens.len;
    if (n == 0) return;
    const order = try allocator.alloc(usize, n);
    defer allocator.free(order);
    const new_index = try allocator.alloc(i32, n);
    defer allocator.free(new_index);
    var stack: std.ArrayList(i32) = .empty;
    defer stack.deinit(allocator);
    var out: usize = 0;
    // Roots (parent -1) in taken order, visited depth-first.
    var root_i: usize = n;
    while (root_i > 0) {
        root_i -= 1;
        if (t.parents[root_i] < 0) try stack.append(allocator, @intCast(root_i));
    }
    while (stack.pop()) |node| {
        order[out] = @intCast(node);
        new_index[@intCast(node)] = @intCast(out);
        out += 1;
        var c: usize = n;
        while (c > 0) {
            c -= 1;
            if (t.parents[c] == node) try stack.append(allocator, @intCast(c));
        }
    }
    const tokens = try allocator.dupe(u32, t.tokens);
    defer allocator.free(tokens);
    const parents = try allocator.dupe(i32, t.parents);
    defer allocator.free(parents);
    const depth = try allocator.dupe(u32, t.depth);
    defer allocator.free(depth);
    const cond = try allocator.dupe(f32, t.cond);
    defer allocator.free(cond);
    for (order, 0..) |old, i| {
        t.tokens[i] = tokens[old];
        t.depth[i] = depth[old];
        t.cond[i] = cond[old];
        t.parents[i] = if (parents[old] < 0) -1 else new_index[@intCast(parents[old])];
    }
}

fn validateLattice(lat: *const Lattice, p: TreeParams) !void {
    if (lat.m == 0 or lat.m > 15 or lat.k == 0 or lat.k > 64 or p.max_nodes == 0 or p.max_nodes > 15 or p.children == 0 or
        lat.cands.len != lat.m * lat.k or lat.unary.len != lat.cands.len or lat.e0.len != lat.k or lat.e.len != (lat.m - 1) * lat.k * lat.k or
        !std.math.isFinite(p.tau) or p.tau <= 0 or !std.math.isFinite(p.temperature) or p.temperature <= 0 or !std.math.isFinite(p.edge_w)) return error.InvalidGlmDraftLattice;
    for (lat.cands) |c| if (c < 0) return error.InvalidGlmDraftLattice;
    for (lat.unary) |v| if (!std.math.isFinite(v)) return error.InvalidGlmDraftLattice;
    for (lat.e0) |v| if (!std.math.isFinite(v)) return error.InvalidGlmDraftLattice;
    for (lat.e) |v| if (!std.math.isFinite(v)) return error.InvalidGlmDraftLattice;
    if (p.noise) |noise| {
        if (noise.len != lat.cands.len or !std.math.isFinite(p.noise_w)) return error.InvalidGlmDraftLattice;
        for (noise) |v| if (!std.math.isFinite(v)) return error.InvalidGlmDraftLattice;
    }
}

pub fn validate(tokens: []const u32, parents: []const i32) !void {
    if (tokens.len == 0 or tokens.len > 16 or tokens.len != parents.len or parents[0] != -1) return error.InvalidGlmDraftTree;
    for (parents[1..], 1..) |parent, row| {
        if (parent < 0 or parent >= row) return error.InvalidGlmDraftTree;
        for (1..row) |other| if (parents[other] == parent and tokens[other] == tokens[row]) return error.InvalidGlmDraftTree;
    }
}

pub const Accepted = struct {
    rows: [16]u32 = undefined,
    count: usize = 0,
    pending: ?u32 = null,
    stopped: bool = false,
};

/// Target decisions select one ancestry path; the root is already a pending target token.
pub fn accept(tokens: []const u32, parents: []const i32, targets: []const u32, budget: usize, eos: []const u32) !Accepted {
    try validate(tokens, parents);
    if (targets.len != tokens.len) return error.InvalidGlmDraftTree;
    if (budget == 0) return error.InvalidGlmDraftBudget;
    var result = Accepted{};
    var row: usize = 0;
    while (true) {
        result.rows[result.count] = @intCast(row);
        result.count += 1;
        if (std.mem.indexOfScalar(u32, eos, tokens[row]) != null) {
            result.stopped = true;
            result.pending = null;
            return result;
        }
        result.pending = targets[row];
        if (result.count == budget) return result;
        var child: ?usize = null;
        for (1..tokens.len) |candidate| {
            if (parents[candidate] == row and tokens[candidate] == targets[row]) {
                child = candidate;
                break;
            }
        }
        row = child orelse return result;
    }
}

/// Draw from the target distribution only along the visited ancestry path.
/// A matching proposal reuses its already-verified state; a missing child
/// returns that same target draw as the pending correction. Unvisited nodes
/// consume no RNG draws, so proposal breadth cannot change the sample stream.
pub fn sampledTargets(tokens: []const u32, parents: []const i32, targets: []u32, budget: usize, eos: []const u32, ctx: *anyopaque, draw: *const fn (*anyopaque, usize) anyerror!u32) !usize {
    try validate(tokens, parents);
    if (targets.len != tokens.len or budget == 0) return error.InvalidGlmDraftBudget;
    var row: usize = 0;
    var draws: usize = 0;
    while (draws < budget and std.mem.indexOfScalar(u32, eos, tokens[row]) == null) {
        const token = try draw(ctx, row);
        targets[row] = token;
        draws += 1;
        if (draws >= budget) break;
        var child: ?usize = null;
        for (1..tokens.len) |candidate| {
            if (parents[candidate] == row and tokens[candidate] == token) {
                child = candidate;
                break;
            }
        }
        row = child orelse break;
    }
    return draws;
}

test "GLM DFlash best-first tree visits likely ancestry before siblings" {
    const a = std.testing.allocator;
    var lat = Lattice{ .m = 2, .k = 2, .cands = try a.dupe(i32, &.{ 11, 12, 13, 14 }), .unary = try a.dupe(f32, &.{ 5, 0, 5, 0 }), .e0 = try a.dupe(f32, &.{ 0, 0 }), .e = try a.dupe(f32, &.{ 0, 0, 0, 0 }) };
    defer lat.deinit(a);
    var tree = try bestFirstTree(a, &lat, .{ .max_nodes = 3 });
    defer tree.deinit(a);
    try std.testing.expectEqualSlices(u32, &.{ 11, 13, 12 }, tree.tokens);
    try std.testing.expectEqualSlices(i32, &.{ -1, 0, -1 }, tree.parents);
    lat.unary[0] = std.math.nan(f32);
    try std.testing.expectError(error.InvalidGlmDraftLattice, bestFirstTree(a, &lat, .{ .max_nodes = 3 }));
}


fn fixtureLattice(cands: *[7 * 4]i32, unary: *[7 * 4]f32, anchor: *[4]f32, edges: *[6 * 4 * 4]f32) Lattice {
    for (cands, unary, 0..) |*id, *v, i| {
        id.* = @intCast(100 + i);
        v.* = @as(f32, @floatFromInt((i * 7) % 13)) / 8 - 0.5;
    }
    for (anchor, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 3)) / 4;
    for (edges, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i * 3) % 17)) / 8 - 0.75;
    return .{ .m = 7, .k = 4, .cands = cands, .unary = unary, .e0 = anchor, .e = edges };
}

test "GLM best-first nodes in the order taken: any prefix is the best tree of that size, with its selector probabilities" {
    const a = std.testing.allocator;
    var cands: [7 * 4]i32 = undefined;
    var unary: [7 * 4]f32 = undefined;
    var anchor: [4]f32 = undefined;
    var edges: [6 * 4 * 4]f32 = undefined;
    const lat = fixtureLattice(&cands, &unary, &anchor, &edges);
    var full = try bestFirstNodes(a, &lat, .{ .max_nodes = 7 });
    defer full.deinit(a);
    for (1..7) |n| {
        var small = try bestFirstNodes(a, &lat, .{ .max_nodes = n });
        defer small.deinit(a);
        try std.testing.expectEqualSlices(u32, full.tokens[0..n], small.tokens);
        try std.testing.expectEqualSlices(i32, full.parents[0..n], small.parents);
        try std.testing.expectEqualSlices(f32, full.cond[0..n], small.cond);
    }
    var roots: f32 = 0;
    for (full.cond, full.parents) |c, parent| {
        try std.testing.expect(c > 0 and c <= 1);
        if (parent < 0) roots += c;
    }
    try std.testing.expect(roots <= 1.0001);
}

test "GLM draft tree cut keeps the rows that fit, then the likeliest nodes, and the survivors follow their parents" {
    const a = std.testing.allocator;
    var cands: [7 * 4]i32 = undefined;
    var unary: [7 * 4]f32 = undefined;
    var anchor: [4]f32 = undefined;
    var edges: [6 * 4 * 4]f32 = undefined;
    const lat = fixtureLattice(&cands, &unary, &anchor, &edges);
    var whole = try bestFirstNodes(a, &lat, .{ .max_nodes = 7 });
    defer whole.deinit(a);
    var three = try bestFirstNodes(a, &lat, .{ .max_nodes = 3 });
    defer three.deinit(a);
    var kept = try bestFirstNodes(a, &lat, .{ .max_nodes = 7 });
    defer kept.deinit(a);
    try kept.cut(a, 8, 3);
    try std.testing.expectEqualSlices(u32, three.tokens, kept.tokens);
    try std.testing.expectEqualSlices(i32, three.parents, kept.parents);
    var shallow = try bestFirstNodes(a, &lat, .{ .max_nodes = 7 });
    defer shallow.deinit(a);
    try shallow.cut(a, 3, 7);
    for (shallow.depth, shallow.parents, 0..) |depth, parent, i| {
        try std.testing.expect(depth <= 1);
        try std.testing.expect(parent < @as(i32, @intCast(i)));
    }
    try std.testing.expect(shallow.tokens.len >= 1 and shallow.tokens.len <= whole.tokens.len);
    var none = try bestFirstNodes(a, &lat, .{ .max_nodes = 7 });
    defer none.deinit(a);
    try none.cut(a, 1, 7);
    try std.testing.expectEqual(@as(usize, 0), none.tokens.len);
}

test "GLM sampled tree visits only the target-selected path and preserves categorical probabilities" {
    const Fixture = struct {
        first: usize,
        second: usize,
        visited: [4]usize = undefined,
        count: usize = 0,
        fn draw(raw: *anyopaque, row: usize) !u32 {
            const self: *@This() = @ptrCast(@alignCast(raw));
            self.visited[self.count] = row;
            self.count += 1;
            return switch (row) {
                0 => ([_]u32{ 20, 30, 30, 50 })[self.first],
                1 => ([_]u32{ 40, 40, 60 })[self.second],
                2 => 70,
                3 => 80,
                else => error.UnexpectedRow,
            };
        }
    };
    const tokens = [_]u32{ 10, 20, 30, 40 };
    const parents = [_]i32{ -1, 0, 0, 1 };
    var mass: [4]usize = @splat(0);
    for (0..4) |first| for (0..3) |second| {
        var f = Fixture{ .first = first, .second = second };
        var targets = [_]u32{ 999, 999, 999, 999 };
        const draws = try sampledTargets(&tokens, &parents, &targets, 8, &.{}, &f, Fixture.draw);
        const kept = try accept(&tokens, &parents, &targets, 8, &.{});
        try std.testing.expectEqual(f.count, draws);
        try std.testing.expectEqual(kept.count, draws);
        switch (kept.pending.?) {
            50 => { mass[0] += 1; try std.testing.expectEqualSlices(usize, &.{0}, f.visited[0..f.count]); },
            60 => { mass[1] += 1; try std.testing.expectEqualSlices(usize, &.{ 0, 1 }, f.visited[0..f.count]); },
            80 => { mass[2] += 1; try std.testing.expectEqualSlices(usize, &.{ 0, 1, 3 }, f.visited[0..f.count]); },
            70 => { mass[3] += 1; try std.testing.expectEqualSlices(usize, &.{ 0, 2 }, f.visited[0..f.count]); },
            else => return error.UnexpectedPending,
        }
    };
    // Same path probabilities as serial draws: 1/4, (1/4)(1/3), (1/4)(2/3), 1/2.
    try std.testing.expectEqualSlices(usize, &.{ 3, 1, 2, 6 }, &mass);
}

test "GLM sampled tree stops RNG at output budget and EOS" {
    const Fixture = struct {
        calls: usize = 0,
        fn draw(raw: *anyopaque, row: usize) !u32 {
            const self: *@This() = @ptrCast(@alignCast(raw));
            self.calls += 1;
            return ([_]u32{ 20, 40, 80 })[row];
        }
    };
    const tokens = [_]u32{ 10, 20, 40 };
    const parents = [_]i32{ -1, 0, 1 };
    var targets = [_]u32{ 999, 999, 999 };
    var f = Fixture{};
    try std.testing.expectEqual(@as(usize, 1), try sampledTargets(&tokens, &parents, &targets, 1, &.{}, &f, Fixture.draw));
    try std.testing.expectEqual(@as(usize, 1), f.calls);
    const limited = try accept(&tokens, &parents, &targets, 1, &.{});
    try std.testing.expectEqual(@as(usize, 1), limited.count);
    try std.testing.expectEqual(@as(?u32, 20), limited.pending);
    f.calls = 0;
    try std.testing.expectEqual(@as(usize, 2), try sampledTargets(&tokens, &parents, &targets, 8, &.{40}, &f, Fixture.draw));
    const stopped = try accept(&tokens, &parents, &targets, 8, &.{40});
    try std.testing.expect(stopped.stopped and stopped.pending == null);
    try std.testing.expectEqual(@as(usize, 2), f.calls);
}
