//! Hy4-Flash's vision tower and merger (part 1v; docs/hy4-flash-arch.md#vision-tower-and-merger): a 16 x 16 patch
//! embedding with a learned position table resampled to the grid, pre-norm blocks (LayerNorm with bias, full MHA
//! with biases and no rotary, a tanh-GELU MLP), then RMSNorm, a 2 x 2 stride-2 convolution, tanh GELU, a 1 x 1
//! convolution, and an image newline closing every merged row. The tower is served as stored: BF16, or with its
//! matrices as MLX affine codes where the pack's `vision_quant` says so (docs/pack-format.md#hy4-stq).
const std = @import("std");
const mlx = @import("../mlx.zig");
const Ops = @import("../mlx_ops.zig").Ops;
const model_mod = @import("../model.zig");
const hy4_config = @import("hy4_config.zig");
const fwd = @import("hy4_forward.zig");
const log = @import("../log.zig");
const io_util = @import("../io_util.zig");
const plan = @import("../tensor/plan/plan_api.zig");
const Arr = mlx.mlx_array;

const eps: f32 = 1e-5;
pub const attn_query_rows: u32 = 512;

/// The engine pool's bytes an encode of a `grid_h` x `grid_w` image holds beyond the resident tower, attention by
/// `attn_rows` query rows; the request's bill adds `bills.BILL_MARGIN_BYTES`. The pool keeps every buffer it made and
/// reuses one only at its own size class, so it holds, at each class, the most buffers of it ever in use at once:
/// `Replay` walks the encode's buffers in `Tower.forward`'s order.
pub fn encodeHeldBytes(v: hy4_config.Vision, attn_rows: u64, grid_h: u64, grid_w: u64) !u64 {
    const n = grid_h * grid_w;
    if (n == 0) return 0;
    const d: u64 = v.hidden;
    const i: u64 = v.inter;
    const h: u64 = v.heads;
    const hd = d / h;
    const feat: u64 = 3 * @as(u64, v.patch) * v.patch;
    const cells = n / (@as(u64, v.merge) * v.merge);
    const dm = d * v.merge * v.merge;
    const out: u64 = v.out_hidden;
    var r: Replay = .{ .quant = v.quant };
    // The embedding: the patches' linear, the four position taps, their sum; the stream x stays.
    var e = r.mark();
    r.affine(n, feat, d);
    r.make(16 * n, 2); // the taps' rows and weights
    r.make(8 * n * d, 1);
    r.make(16 * n * d, 2);
    r.make(4 * n * d, 1);
    r.make(4 * n * d, 1); // x
    r.release(&e);
    // A block: everything it makes stays until its scope ends, but each chunk's three score sheets.
    var b = r.mark();
    r.make(4 * d, 2);
    r.make(4 * n * d, 1);
    for (0..3) |_| try r.linear(n, d, d);
    r.make(4, 1);
    const rows = @min(attn_rows, n);
    const chunks = (n + rows - 1) / rows;
    for (0..chunks) |c| {
        const rc = @min(rows, n - c * rows);
        r.make(4 * h * rc * n, 3);
        r.free(4 * h * rc * n, 3);
        r.make(4 * h * rc * hd, 1);
    }
    if (chunks > 1) r.make(4 * n * d, 1);
    r.make(4 * n * d, 1); // the heads joined
    try r.linear(n, d, d);
    r.make(4 * n * d, 1); // y
    r.make(4 * d, 2);
    r.make(4 * n * d, 1);
    try r.linear(n, d, i);
    r.gelu(n * i);
    try r.linear(n, i, d);
    r.make(4 * n * d, 1); // the block's output, the next x
    r.release(&b);
    r.drop(4 * n * d); // the block's input x
    // The merger, in the forward's scope beside x.
    r.make(4 * d, 1);
    r.make(4 * n * d, 2); // the norm, the cells' gather
    try r.linear(cells, 4 * d, dm);
    r.gelu(cells * dm);
    try r.linear(cells, dm, out);
    r.make(4 * out, 1);
    r.make(4 * (grid_h / v.merge) * (grid_w / v.merge + 1) * out, 1);
    return r.held();
}

/// The pool's buffers over an encode, by size class: how many are in use, and the most ever in use at once.
const Replay = struct {
    const pool = @import("../tensor/pool_class.zig");
    /// The tower's matrices as affine codes (`vision_quant`), or BF16.
    quant: ?hy4_config.VisionQuant = null,
    now: [pool.classes]u32 = @splat(0),
    most: [pool.classes]u32 = @splat(0),
    /// What the open scope made, to free when it ends.
    made: [64]struct { bytes: u64, k: u32 } = undefined,
    count: usize = 0,

    fn make(r: *Replay, bytes: u64, k: u32) void {
        const c = pool.classOf(bytes);
        r.now[c] += k;
        r.most[c] = @max(r.most[c], r.now[c]);
        r.made[r.count] = .{ .bytes = bytes, .k = k };
        r.count += 1;
    }

    fn free(r: *Replay, bytes: u64, k: u32) void {
        r.now[pool.classOf(bytes)] -= k;
        r.count -= 1;
    }

    /// Frees a buffer made before the open scope.
    fn drop(r: *Replay, bytes: u64) void {
        r.now[pool.classOf(bytes)] -= 1;
    }

    fn mark(r: *const Replay) usize {
        return r.count;
    }

    /// Frees what the scope since `at` made, but its last buffer: the scope's result.
    fn release(r: *Replay, at: *usize) void {
        const last = r.made[r.count - 1];
        while (r.count > at.*) {
            const m = r.made[r.count - 1];
            r.free(m.bytes, m.k);
        }
        r.now[pool.classOf(last.bytes)] += last.k;
        r.made[r.count] = last;
        r.count += 1;
    }

    /// A bf16 linear with its bias over `rows` rows (`affine`): the input as bf16, the product, it in f32, the bias in
    /// f32, the sum.
    fn affine(r: *Replay, rows: u64, in: u64, out: u64) void {
        r.make(2 * rows * in, 1);
        r.make(2 * rows * out, 1);
        r.make(4 * rows * out, 1);
        r.make(4 * out, 1);
        r.make(4 * rows * out, 1);
    }

    /// A block or merger matrix with its bias: affine codes where `quant` stores it so (`hy4_census.visionAffine`), a
    /// product that also makes split-K's partials, which its pass frees only when handed over; else BF16.
    fn linear(r: *Replay, rows: u64, in: u64, out: u64) !void {
        const q = r.quant orelse return r.affine(rows, in, out);
        if (in % q.group != 0) return r.affine(rows, in, out);
        r.make(2 * rows * in, 1);
        r.make(2 * rows * out, 1);
        const parts = @import("../tensor/quantized.zig").partialsBytesMost(@intCast(rows), @intCast(out), @intCast(in), q.group, 2);
        if (parts > 0) r.make(parts, 1);
        r.make(4 * rows * out, 1);
        r.make(4 * out, 1);
        r.make(4 * rows * out, 1);
    }

    /// `geluTanh` over `len` f32 values: seven of its size and two scalars.
    fn gelu(r: *Replay, len: u64) void {
        r.make(4 * len, 7);
        r.make(4, 2);
    }

    fn held(r: *const Replay) u64 {
        var sum: u64 = 0;
        for (r.most, 0..) |k, c| sum += @as(u64, k) * pool.classBytes(c);
        return sum;
    }
};

const Block = struct {
    norm1_w: Arr,
    norm1_b: Arr,
    q: Lin,
    q_b: Arr,
    k: Lin,
    k_b: Arr,
    v: Lin,
    v_b: Arr,
    o: Lin,
    o_b: Arr,
    norm2_w: Arr,
    norm2_b: Arr,
    fc1: Lin,
    fc1_b: Arr,
    fc2: Lin,
    fc2_b: Arr,
};

/// A tower matrix as stored: a float weight (a convolution's kept 4-D), or MLX affine codes `[out, in·bits/32]` with
/// their scales and biases `[out, in/group]`.
const Lin = struct {
    w: Arr,
    affine: ?struct { scales: Arr, biases: Arr, bits: u32, group: u32 } = null,
};

pub const Tower = struct {
    gpa: std.mem.Allocator,
    v: hy4_config.Vision,
    /// The table's grid side: its rows past the class token's are `edge` x `edge` positions.
    edge: u32,
    patch: Lin,
    patch_b: Arr,
    pos: Arr,
    blocks: []Block,
    merge_norm: Arr,
    merge: Lin,
    merge_b: Arr,
    out: Lin,
    out_b: Arr,
    newline: Arr,
    /// Query rows one attention step scores at a time: a block holds [heads, rows, N] score sheets, never
    /// [heads, N, N]. A test lowers it to take several chunks on a small image.
    attn_rows: u32 = attn_query_rows,

    pub fn init(gpa: std.mem.Allocator, v: hy4_config.Vision, w: *const model_mod.Weights) !Tower {
        const blocks = try gpa.alloc(Block, v.layers);
        errdefer gpa.free(blocks);
        const l = "vit.layers.{d}.";
        try strayGrids(v, w);
        for (blocks, 0..) |*b, i| b.* = .{
            .norm1_w = try get(w, l ++ "input_layernorm.weight", .{i}),
            .norm1_b = try get(w, l ++ "input_layernorm.bias", .{i}),
            .q = try matrix(v, w, l ++ "self_attn.q_proj.weight", .{i}),
            .q_b = try get(w, l ++ "self_attn.q_proj.bias", .{i}),
            .k = try matrix(v, w, l ++ "self_attn.k_proj.weight", .{i}),
            .k_b = try get(w, l ++ "self_attn.k_proj.bias", .{i}),
            .v = try matrix(v, w, l ++ "self_attn.v_proj.weight", .{i}),
            .v_b = try get(w, l ++ "self_attn.v_proj.bias", .{i}),
            .o = try matrix(v, w, l ++ "self_attn.o_proj.weight", .{i}),
            .o_b = try get(w, l ++ "self_attn.o_proj.bias", .{i}),
            .norm2_w = try get(w, l ++ "post_attention_layernorm.weight", .{i}),
            .norm2_b = try get(w, l ++ "post_attention_layernorm.bias", .{i}),
            .fc1 = try matrix(v, w, l ++ "mlp.dense_h_to_4h.weight", .{i}),
            .fc1_b = try get(w, l ++ "mlp.dense_h_to_4h.bias", .{i}),
            .fc2 = try matrix(v, w, l ++ "mlp.dense_4h_to_h.weight", .{i}),
            .fc2_b = try get(w, l ++ "mlp.dense_4h_to_h.bias", .{i}),
        };
        const pos = try get(w, "vit.embeddings.position_embedding.weight", .{});
        const rows: u64 = @intCast(mlx.getShape(pos)[0]);
        const edge = std.math.sqrt(rows - 1);
        if (edge * edge + 1 != rows) return error.Hy4VisionPositionTable;
        return .{
            .gpa = gpa,
            .v = v,
            .edge = @intCast(edge),
            .patch = try matrix(v, w, "vit.embeddings.patch_embedding.weight", .{}),
            .patch_b = try get(w, "vit.embeddings.patch_embedding.bias", .{}),
            .pos = pos,
            .blocks = blocks,
            .merge_norm = try get(w, "vit.perceive.norm.weight", .{}),
            .merge = try matrix(v, w, "vit.perceive.proj.0.weight", .{}),
            .merge_b = try get(w, "vit.perceive.proj.0.bias", .{}),
            .out = try matrix(v, w, "vit.perceive.proj.2.weight", .{}),
            .out_b = try get(w, "vit.perceive.proj.2.bias", .{}),
            .newline = try get(w, "vit.perceive.image_newline", .{}),
        };
    }

    pub fn deinit(self: *Tower) void {
        self.gpa.free(self.blocks);
    }

    /// The rows an image of `grid_h` x `grid_w` patches gives the prompt: each merged row of `grid_w / merge` and its
    /// newline.
    pub fn rowsOf(self: *const Tower, grid_h: u32, grid_w: u32) u32 {
        return (grid_h / self.v.merge) * (grid_w / self.v.merge + 1);
    }

    /// What a test reads beside the merged rows: the embedding and each block's output, [N, hidden] f32.
    pub const Stages = struct {
        embed: ?Arr = null,
        blocks: []Arr = &.{},
    };

    /// The image's prompt rows [rowsOf, out_hidden] f32 from its patches [N, 3·P·P] (the processor's row-major
    /// patches, each one's values in (C, P, P) order), `grid_h` x `grid_w` of them.
    pub fn forward(self: *const Tower, o: *Ops, patches: Arr, grid_h: u32, grid_w: u32, stages: ?*Stages) !Arr {
        const v = self.v;
        const n: c_int = @intCast(grid_h * grid_w);
        if (grid_h == 0 or grid_w == 0 or grid_h % v.merge != 0 or grid_w % v.merge != 0) return error.Hy4VisionGrid;
        // Each stage runs in a scope of its own and is evaluated before the next: an encode holds the stream and one
        // stage's buffers, and what a stage freed goes back to the pool (`handOver`) for the next.
        var x = blk: {
            var eo: Ops = .{ .s = o.s };
            defer eo.deinit();
            // The patch embedding: the convolution's weight [hidden, 3, P, P] as a linear over each patch's values.
            const e = try eo.binary(.add, try linear(&eo, patches, self.patch), try eo.cast(self.patch_b, .float32));
            break :blk try eo.result(try eo.binary(.add, e, try self.positions(&eo, grid_h, grid_w)));
        };
        {
            errdefer _ = mlx.mlx_array_free(x);
            try mlx.check(mlx.mlx_array_eval(x));
            try handOver();
            if (stages) |s| s.embed = try o.own(try o.result(x));
        }
        for (self.blocks, 0..) |*b, i| {
            const next = blk: {
                errdefer _ = mlx.mlx_array_free(x);
                // Every block asks alike: one frame, its instances at their places in the encode.
                const f = try plan.frame(@src(), .{});
                defer f.end();
                var bo: Ops = .{ .s = o.s };
                defer bo.deinit();
                const a = try bo.layerNorm(x, try bo.cast(b.norm1_w, .float32), try bo.cast(b.norm1_b, .float32), eps);
                const y = try bo.binary(.add, x, try self.attention(&bo, a, b, n));
                const m = try bo.layerNorm(y, try bo.cast(b.norm2_w, .float32), try bo.cast(b.norm2_b, .float32), eps);
                const h = try geluTanh(&bo, try affine(&bo, m, b.fc1, b.fc1_b));
                const out = try bo.result(try bo.binary(.add, y, try affine(&bo, h, b.fc2, b.fc2_b)));
                errdefer _ = mlx.mlx_array_free(out);
                try mlx.check(mlx.mlx_array_eval(out));
                break :blk out;
            };
            _ = mlx.mlx_array_free(x);
            x = next;
            errdefer _ = mlx.mlx_array_free(x);
            try handOver();
            if (stages) |s| if (i < s.blocks.len) {
                s.blocks[i] = try o.own(try o.result(x));
            };
        }
        return self.merger(o, try o.own(x), grid_h, grid_w);
    }

    /// `forward` of the processor's patches as host bytes (f32 [N, 3·P·P], `grid_h` x `grid_w` of them): their upload
    /// is the encode's own host ask, so a planned encode asks the pool for nothing.
    pub fn encodeHost(self: *const Tower, o: *Ops, pixels: []const u8, grid_h: u32, grid_w: u32) !Arr {
        const n: usize = @as(usize, grid_h) * grid_w;
        const feat: usize = 3 * @as(usize, self.v.patch) * self.v.patch;
        if (pixels.len != n * feat * @sizeOf(f32)) return error.Hy4VisionPatches;
        const patches = blk: {
            const scope = mlx.engine.enter(o.s);
            defer scope.end();
            break :blk try mlx.newData(pixels.ptr, &[_]c_int{ @intCast(n), @intCast(feat) }, .float32);
        };
        defer _ = mlx.mlx_array_free(patches);
        return self.forward(o, patches, grid_h, grid_w, null);
    }

    /// Full attention over the image's patches, 16 heads of `hidden / heads`, scaled by `head_dim^-1/2`.
    fn attention(self: *const Tower, o: *Ops, x: Arr, b: *const Block, n: c_int) !Arr {
        const heads: c_int = @intCast(self.v.heads);
        const hd: c_int = @intCast(self.v.hidden / self.v.heads);
        const q = try splitHeads(o, try affine(o, x, b.q, b.q_b), n, heads, hd);
        const k = try splitHeads(o, try affine(o, x, b.k, b.k_b), n, heads, hd);
        const vv = try splitHeads(o, try affine(o, x, b.v, b.v_b), n, heads, hd);
        const scale = try o.scalar(1.0 / @sqrt(@as(f32, @floatFromInt(hd))), .float32);
        const kt = try o.transpose(k, &.{ 0, 2, 1 });
        // A row's scores, softmax and context are the whole sheet's: the reductions run along keys, never queries.
        const rows: c_int = @intCast(@min(self.attn_rows, @as(u32, @intCast(n))));
        const chunks: usize = @intCast(@divFloor(n + rows - 1, rows));
        const outs = try self.gpa.alloc(Arr, chunks);
        defer self.gpa.free(outs);
        for (outs, 0..) |*out, ci| {
            const r0: c_int = @as(c_int, @intCast(ci)) * rows;
            const it = try plan.each(@src(), .{});
            defer it.end();
            var co: Ops = .{ .s = o.s };
            defer co.deinit();
            out.* = try o.own(try co.result(try chunkContext(&co, q, kt, vv, scale, r0, @min(n, r0 + rows)))); // [H, rows, hd]
            try mlx.check(mlx.mlx_array_eval(out.*));
            try handOver();
        }
        const ctx = if (chunks == 1) outs[0] else try o.concat(outs, 1); // [H, N, hd]
        const flat = try o.reshape(try o.transpose(ctx, &.{ 1, 0, 2 }), &.{ n, heads * hd });
        return affine(o, flat, b.o, b.o_b);
    }

    /// Patch counts whose one-block runs between them reach every pipeline an encode picks by its size: the products'
    /// vector kernels, split-K and whole tiles, a row count aligned and not.
    const warm_patches = [_]u32{ 4, 16, 60, 64, 256, 1020, 1024 };
    /// Keys from which a block's attention takes the looped softmax (past 4,096) and the context product's other K
    /// step: made by shape alone (`warmContext`), not by an encode that wide.
    const context_keys: u32 = 8192;

    /// A Cooler load's tower, before the load seals: one block at a grid of each of `warm_patches`, then the attention
    /// over `context_keys`, so an image of any size compiles nothing inside a request. A manifest of the tower's key
    /// that holds the warm makes its pipelines instead.
    pub fn warmEngine(self: *const Tower) !void {
        if (self.blocks.len == 0) return;
        const v = self.v;
        var key_buf: [128]u8 = undefined;
        const key = std.fmt.bufPrint(&key_buf, "hy4/{s}/{s}/{d}.{d}.{d}.{d}.{d}.{d}/w1", .{
            if (self.blocks[0].q.affine != null) "q8" else "bf16", @tagName(fwd.products), v.hidden, v.heads, v.inter, v.merge, v.out_hidden, self.attn_rows,
        }) catch return error.Hy4VisionKey;
        if (mlx.engine.pipeline_registry.takeMarker(key)) {
            log.info("[vision] engine tower: pipelines made from the manifest, no warm\n", .{});
            return;
        }
        const making = @import("../arch/cooler.zig").making(true).?;
        defer making.end();
        var clock = io_util.Stopwatch.init(std.Io.Threaded.global_single_threaded.io());
        const made_before = mlx.engine.pipelinesMade().count;
        var one = self.*;
        one.blocks = self.blocks[0..1];
        const es = mlx.engine.stream();
        const feat: usize = 3 * @as(usize, v.patch) * v.patch;
        for (warm_patches) |n| {
            // A grid of whole merge cells: `merge` patches wide.
            const cells = n / (v.merge * v.merge);
            const zeros = try self.gpa.alloc(f32, n * feat);
            defer self.gpa.free(zeros);
            @memset(zeros, 0);
            // The engine's own array, made between forwards: the pool's cap rises by it until the encode's replaces it.
            mlx.engine.room(&.{zeros.len * @sizeOf(f32)});
            const patches = blk: {
                const scope = mlx.engine.enter(es);
                defer scope.end();
                break :blk try mlx.newData(zeros.ptr, &[_]c_int{ @intCast(n), @intCast(feat) }, .float32);
            };
            defer _ = mlx.mlx_array_free(patches);
            var o: Ops = .{ .s = es };
            defer o.deinit();
            const rows = try one.forward(&o, patches, cells * v.merge, v.merge, null);
            try mlx.check(mlx.mlx_array_eval(rows));
        }
        try self.warmContext(context_keys);
        mlx.engine.trim();
        log.info("[vision] engine tower warmed: {d} pipelines, {d} ms\n", .{ mlx.engine.pipelinesMade().count - made_before, clock.read() / std.time.ns_per_ms });
        mlx.engine.pipeline_registry.recordMarker("vision tower", key);
        @import("../arch/cooler_pipelines.zig").rewriteLate();
    }

    /// A few query rows' attention over `keys` patches, on operands laid out as `attention` makes them (`splitHeads` of
    /// f32 projections): the kernels a block over that many keys takes, made without its image.
    fn warmContext(self: *const Tower, keys: u32) !void {
        const heads: c_int = @intCast(self.v.heads);
        const hd: c_int = @intCast(self.v.hidden / self.v.heads);
        const n: c_int = @intCast(keys);
        const es = mlx.engine.stream();
        const width: usize = self.v.hidden;
        mlx.engine.room(&.{ 4 * @as(usize, keys) * width, 4 * @as(usize, keys) * width, 4 * @as(usize, keys) * width, 4 * @as(usize, @intCast(heads)) * 4 * keys });
        const scope = mlx.engine.enter(es);
        defer scope.end();
        var o: Ops = .{ .s = es };
        defer o.deinit();
        var parts: [3]Arr = undefined;
        for (&parts) |*part| {
            const zeros = try o.slot();
            try mlx.check(mlx.mlx_zeros(zeros, &[_]c_int{ n, @intCast(width) }, 2, .float32, es));
            part.* = try splitHeads(&o, zeros.*, n, heads, hd);
        }
        const kt = try o.transpose(parts[1], &.{ 0, 2, 1 });
        const scale = try o.scalar(1.0 / @sqrt(@as(f32, @floatFromInt(hd))), .float32);
        const ctx = try chunkContext(&o, parts[0], kt, parts[2], scale, 0, 4);
        try mlx.check(mlx.mlx_array_eval(ctx));
    }

    /// The learned table resampled to the grid as torch's bilinear `interpolate` with `scale_factor (n + 0.1) / edge`
    /// and `align_corners` false: each grid position blends four table rows (past the class token's row 0).
    fn positions(self: *const Tower, o: *Ops, grid_h: u32, grid_w: u32) !Arr {
        const a = self.gpa;
        const n: usize = grid_h * grid_w;
        const idx = try a.alloc(u32, 4 * n);
        defer a.free(idx);
        const wts = try a.alloc(f32, 4 * n);
        defer a.free(wts);
        const ys = try axis(a, grid_h, self.edge);
        defer a.free(ys);
        const xs = try axis(a, grid_w, self.edge);
        defer a.free(xs);
        for (0..grid_h) |i| for (0..grid_w) |j| {
            const p = i * grid_w + j;
            const y = ys[i];
            const x = xs[j];
            const corners = [4][2]u32{ .{ y.lo, x.lo }, .{ y.lo, x.hi }, .{ y.hi, x.lo }, .{ y.hi, x.hi } };
            const weights = [4]f32{ (1 - y.t) * (1 - x.t), (1 - y.t) * x.t, y.t * (1 - x.t), y.t * x.t };
            for (corners, weights, 0..) |c, wt, k| {
                idx[k * n + p] = 1 + c[0] * self.edge + c[1];
                wts[k * n + p] = wt;
            }
        };
        const ni: c_int = @intCast(n);
        const ids = try o.own(mlx.mlx_array_new_data(idx.ptr, &[_]c_int{4 * ni}, 1, .uint32));
        const ws = try o.own(mlx.mlx_array_new_data(wts.ptr, &[_]c_int{ 4, ni, 1 }, 3, .float32));
        const rows = try o.reshape(try o.cast(try o.take(self.pos, ids, 0), .float32), &.{ 4, ni, @intCast(self.v.hidden) });
        return o.reduce(try o.binary(.mul, rows, ws), 0, false, false);
    }

    /// RMSNorm, the 2 x 2 stride-2 convolution as a linear over each cell's (C, kh, kw) values, tanh GELU, the 1 x 1
    /// convolution, then the newline as the last column of every merged row.
    fn merger(self: *const Tower, o: *Ops, x: Arr, grid_h: u32, grid_w: u32) !Arr {
        const m: c_int = @intCast(self.v.merge);
        const c: c_int = @intCast(self.v.hidden);
        const hb: c_int = @intCast(grid_h / self.v.merge);
        const wb: c_int = @intCast(grid_w / self.v.merge);
        const out: c_int = @intCast(self.v.out_hidden);
        const normed = try o.rms(x, try o.cast(self.merge_norm, .float32), eps);
        const grid = try o.reshape(normed, &.{ hb, m, wb, m, c });
        const cells = try o.reshape(try o.transpose(grid, &.{ 0, 2, 4, 1, 3 }), &.{ hb * wb, c * m * m });
        const conv = try affine(o, cells, self.merge, self.merge_b);
        const proj = try affine(o, try geluTanh(o, conv), self.out, self.out_b);
        const nl = try o.broadcast(try o.reshape(try o.cast(self.newline, .float32), &.{ 1, 1, out }), &.{ hb, 1, out });
        const rows = try o.concat(&.{ try o.reshape(proj, &.{ hb, wb, out }), nl }, 1);
        return o.reshape(rows, &.{ hb * (wb + 1), out });
    }
};

/// `t` [n, heads·hd] as [heads, n, hd]: a view, as the attention reads each head's rows.
fn splitHeads(o: *Ops, t: Arr, n: c_int, heads: c_int, hd: c_int) !Arr {
    return o.transpose(try o.reshape(t, &.{ n, heads, hd }), &.{ 1, 0, 2 });
}

/// Query rows `r0..r1`'s context: their scores over every key (`kt` [H, hd, N]), the softmax along keys, then the sum of
/// the values `vv` [H, N, hd] it weighs: [H, r1 - r0, hd]. A row's reductions run along keys, never queries, so chunks
/// join to the whole sheet's.
fn chunkContext(o: *Ops, q: Arr, kt: Arr, vv: Arr, scale: Arr, r0: c_int, r1: c_int) !Arr {
    const qc = try o.slice(q, 1, r0, r1);
    const scores = try o.binary(.mul, try o.binary(.mm, qc, kt), scale);
    return o.binary(.mm, try o.softmax(scores, 2), vv);
}

/// Hands the engine's open pass over, so the buffers a scope freed serve the next scope.
fn handOver() !void {
    try mlx.engine.submit();
}

const Tap = struct { lo: u32, hi: u32, t: f32 };

/// One axis of the resample: output `i` reads source `(i + 0.5) · edge / (len + 0.1) - 0.5`, clamped at 0, between
/// its floor and the next row (the last row repeats).
fn axis(a: std.mem.Allocator, len: u32, edge: u32) ![]Tap {
    const taps = try a.alloc(Tap, len);
    const scale: f64 = (@as(f64, @floatFromInt(len)) + 0.1) / @as(f64, @floatFromInt(edge));
    for (taps, 0..) |*t, i| {
        const src = @max(0.0, (@as(f64, @floatFromInt(i)) + 0.5) / scale - 0.5);
        const lo: u32 = @min(@as(u32, @intFromFloat(@floor(src))), edge - 1);
        t.* = .{ .lo = lo, .hi = @min(lo + 1, edge - 1), .t = @floatCast(src - @as(f64, @floatFromInt(lo))) };
    }
    return taps;
}

/// A linear by `fwd.products` (BF16 as served, f32 for the oracle), as f32. A float weight [out, ...] reads as
/// [out, in]. Affine codes take the BF16 weight's rows: BF16 in and out at `products` .bf16, f32 at .f32.
fn linear(o: *Ops, x: Arr, lin: Lin) !Arr {
    const q = lin.affine orelse {
        const w = if (mlx.getShape(lin.w).len == 2) lin.w else try o.reshape(lin.w, &.{ mlx.getShape(lin.w)[0], -1 });
        return fwd.linear(o, x, .{ .w = w });
    };
    const out = try o.slot();
    const xin = try o.cast(x, if (fwd.products == .bf16) .bfloat16 else .float32);
    try mlx.check(mlx.mlx_quantized_matmul(out, xin, lin.w, q.scales, q.biases, true, mlx.mlx_optional_int.some(@intCast(q.group)), mlx.mlx_optional_int.some(@intCast(q.bits)), "affine", o.s));
    return o.cast(out.*, .float32);
}

fn affine(o: *Ops, x: Arr, w: Lin, b: Arr) !Arr {
    return o.binary(.add, try linear(o, x, w), try o.cast(b, .float32));
}

/// The tower matrix `fmt` as the pack states it: under `vision_quant`, each `hy4_census.visionAffine` matrix as codes
/// with BF16 grids of the stamped group, every other one a float; with none, every matrix a float. Any other form is
/// `Hy4VisionTensorFormat`.
fn matrix(v: hy4_config.Vision, w: *const model_mod.Weights, comptime fmt: []const u8, args: anytype) !Lin {
    var buf: [160]u8 = undefined;
    const name = try std.fmt.bufPrint(&buf, fmt, args);
    const weight = try get(w, fmt, args);
    const base = name[0 .. name.len - ".weight".len];
    var gb: [2][168]u8 = undefined;
    const scales = w.get(try std.fmt.bufPrint(&gb[0], "{s}.scales", .{base}));
    const biases = w.get(try std.fmt.bufPrint(&gb[1], "{s}.biases", .{base}));
    const dtype = mlx.mlx_array_dtype(weight);
    const shape = @import("hy4_census.zig").visionAffine(v, name);
    const q = v.quant orelse {
        if (scales != null or biases != null) return refuse(name, "affine grids, and the config has no vision_quant");
        if (dtype != .bfloat16 and dtype != .float32) return refuse(name, "neither BF16 nor f32");
        return .{ .w = weight };
    };
    const m = shape orelse {
        if (scales != null or biases != null) return refuse(name, "affine grids on a matrix vision_quant keeps a float");
        if (dtype != .bfloat16 and dtype != .float32) return refuse(name, "neither BF16 nor f32");
        return .{ .w = weight };
    };
    const s = scales orelse return refuse(name, "vision_quant stores it affine, and it has no scales");
    const b = biases orelse return refuse(name, "vision_quant stores it affine, and it has no biases");
    const grid = [2]c_int{ @intCast(m[0]), @intCast(m[1] / q.group) };
    if (dtype != .uint32 or !std.mem.eql(c_int, mlx.getShape(weight), &.{ @intCast(m[0]), @intCast(m[1] * q.bits / 32) }) or
        mlx.mlx_array_dtype(s) != .bfloat16 or mlx.mlx_array_dtype(b) != .bfloat16 or
        !std.mem.eql(c_int, mlx.getShape(s), &grid) or !std.mem.eql(c_int, mlx.getShape(b), &grid))
        return refuse(name, "not the affine codes and BF16 grids vision_quant states");
    return .{ .w = weight, .affine = .{ .scales = s, .biases = b, .bits = q.bits, .group = q.group } };
}

/// An affine grid beside no tower matrix the pack stores as codes.
fn strayGrids(v: hy4_config.Vision, w: *const model_mod.Weights) !void {
    var it = w.map.keyIterator();
    while (it.next()) |key| {
        const name = key.*;
        if (!std.mem.startsWith(u8, name, "vit.")) continue;
        const base = for ([_][]const u8{ ".scales", ".biases" }) |suffix| {
            if (std.mem.endsWith(u8, name, suffix)) break name[0 .. name.len - suffix.len];
        } else continue;
        var buf: [168]u8 = undefined;
        const weight = std.fmt.bufPrint(&buf, "{s}.weight", .{base}) catch return refuse(name, "an affine grid of no tower matrix");
        if (v.quant == null or @import("hy4_census.zig").visionAffine(v, weight) == null) return refuse(name, "an affine grid of no matrix vision_quant stores affine");
    }
}

fn refuse(name: []const u8, why: []const u8) error{Hy4VisionTensorFormat} {
    if (!@import("builtin").is_test) std.log.err("hy_v4_flash: vision tensor {s}: {s}", .{ name, why });
    return error.Hy4VisionTensorFormat;
}

/// `0.5 x (1 + tanh(√(2/π) (x + 0.044715 x³)))`, written `x · σ(2 √(2/π) (x + 0.044715 x³))`.
fn geluTanh(o: *Ops, x: Arr) !Arr {
    const x3 = try o.binary(.mul, try o.binary(.mul, x, x), x);
    const inner = try o.binary(.add, x, try o.binary(.mul, x3, try o.scalar(0.044715, .float32)));
    const z = try o.binary(.mul, inner, try o.scalar(2.0 * @sqrt(2.0 / std.math.pi), .float32));
    return o.binary(.mul, x, try o.unary(.sigmoid, z));
}

fn get(w: *const model_mod.Weights, comptime fmt: []const u8, args: anytype) !Arr {
    var buf: [160]u8 = undefined;
    const name = try std.fmt.bufPrint(&buf, fmt, args);
    return w.get(name) orelse {
        std.log.err("hy_v4_flash: the pack has no vision tensor {s}", .{name});
        return error.Hy4TensorMissing;
    };
}
