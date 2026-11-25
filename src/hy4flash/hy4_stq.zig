//! Hy4-Flash's STQ trunk linears (part 1w, docs/pack-format.md#hy4-stq): a linear an STQ pack stores as a trellis bank
//! under `trunk_quant` (`<base>.{trellis,suh,svh}` in place of `<base>.weight`), read through the shared trunk kernels
//! (`trunk_stq`, as DeepSeek's and Qwen's trellis trunks). Any trunk rate the codebook table carries (K4, K5, K6).
const std = @import("std");
const mlx = @import("../mlx.zig");
const Ops = @import("../mlx_ops.zig").Ops;
const model_mod = @import("../model.zig");
const exl3 = @import("sushi_exl3");
const trunk_stq = @import("../trunk_stq.zig");
const Arr = mlx.mlx_array;

/// What an STQ pack's `config.json` stamps (docs/pack-format.md#hy4-stq): how its trunk banks and its routed experts
/// decode, and whether its codes are random (`--synthetic-codes`: the format is real, the weights are not). A BF16
/// checkpoint has neither block. Read here, not in the MLX-free `hy4_config`.
pub const Quant = struct {
    trunk: ?exl3.format.Decode = null,
    expert: ?exl3.format.Decode = null,
    synthetic: bool = false,
};

/// The blocks at `config.json`'s top level. A block naming a codebook without a table (EXL3's) is refused by name.
pub fn parseQuant(root: std.json.ObjectMap) !Quant {
    var q: Quant = .{};
    if (root.get("trunk_quant")) |b| q.trunk = try stqDecode(b);
    if (root.get("expert_quant")) |b| q.expert = try stqDecode(b);
    if (root.get("synthetic")) |v| q.synthetic = v == .bool and v.bool;
    return q;
}

/// `parseQuant` over a config.json's text.
pub fn parseQuantText(gpa: std.mem.Allocator, text: []const u8) !Quant {
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.UnsupportedHy4Config;
    return parseQuant(parsed.value.object);
}

/// `parseQuant` over the `config.json` in `dir`.
pub fn readQuant(io: std.Io, gpa: std.mem.Allocator, dir: []const u8) !Quant {
    var d = try std.Io.Dir.openDirAbsolute(io, dir, .{});
    defer d.close(io);
    const text = try d.readFileAlloc(io, "config.json", gpa, .limited(1 << 20));
    defer gpa.free(text);
    return parseQuantText(gpa, text);
}

fn stqDecode(block: std.json.Value) !exl3.format.Decode {
    const spec = try exl3.parseQuantBlock(block);
    if (!spec.codebook.hasTable()) {
        std.log.err("hy_v4_flash: a quant block names {s}, not an STQ codebook", .{@tagName(spec.codebook)});
        return error.UnsupportedHy4Config;
    }
    return .{ .codebook = spec.codebook, .window = spec.window, .table = spec.table };
}

/// A one-bank trellis linear: `w` U16 `[1, in/16, out/16, n]`, `suh` F16 `[1, in]`, `svh` F16 `[1, out]`.
pub const Bank = struct {
    w: Arr,
    suh: Arr,
    svh: Arr,
    dec: exl3.format.Decode,

    pub fn inDim(self: Bank) c_int {
        return mlx.getShape(self.w)[1] * 16;
    }

    pub fn outDim(self: Bank) c_int {
        return mlx.getShape(self.w)[2] * 16;
    }
};

/// A trunk linear `[out, in]`: the stored weight (BF16 in the release), or a trellis bank (`stq`; `w` is then unused).
pub const Lin = struct {
    w: Arr,
    stq: ?Bank = null,

    pub fn dense(w: Arr) Lin {
        return .{ .w = w };
    }
};

/// Whether `apply` rounds its rows and outputs inside the launch (`trunk_stq.matmulThruBf16`) rather than by cast
/// dispatches around it. A test seam: both give the same bytes.
pub var thru_bf16_on = true;

/// Hy4's cap on a trunk bank decoded to f16 for one wide forward (128 MiB of f16): the sliding layers' gate and `W_o`
/// and layer 0's dense FFN pass it; the head does not.
pub const decoded_max_weights: usize = 1 << 26;

/// Whether a wide chunk's rows through the bank can take `applyInner`: past the multi-row arms, as wide as the bank,
/// its output in whole 128-column tiles.
pub fn wideInnerApplies(x: Arr, b: Bank) bool {
    const sh = mlx.getShape(x);
    return exl3.linear.force_arm == null and sh.len == 2 and sh[0] > @max(exl3.project_rows.NAX_ROWS_MAX, exl3.project_rows.PROJECT_ROWS_MAX) and
        sh[1] == b.inDim() and @rem(b.outDim(), 128) == 0;
}

/// The bank's GEMM of rows already prepared (`prepareIndexed`'s f16 rows), before its finish: decoded to f16 and
/// multiplied where that serves, in the sorted GEMM's K order (the fused arm where MLX would split K), else the sorted
/// GEMM.
pub fn applyPreparedInner(o: *Ops, prepared: Arr, b: Bank) !Arr {
    const sh = mlx.getShape(prepared);
    const rows = sh[0];
    const in = sh[1];
    const out = b.outDim();
    const weights: usize = @as(usize, @intCast(in)) * @as(usize, @intCast(out));
    const exact_order = exl3.trunk_gemm.denseOrderMatches(rows, in, out);
    const use_decoded = exl3.trunk_gemm.decodedCanServe() and exl3.linear.force_arm == null and
        rows >= exl3.trunk_gemm.DECODED_MIN_ROWS and weights <= decoded_max_weights and
        (exact_order or mlx.engine.isStream(o.s));
    try b.dec.requireTable();
    exl3.kernels.setDecodeParams(b.dec);
    const inputs = try exl3.linear.gemmInputs(1, @intCast(rows), exl3.kernels.gemmWindowRows());
    if (use_decoded) {
        exl3.trunk_gemm.decoded_calls += 1;
        const w = try exl3.trunk_gemm.decodeDense(o.s, b.w);
        defer _ = mlx.mlx_array_free(w);
        var wt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wt);
        try mlx.check(mlx.mlx_transpose(&wt, w, o.s));
        var mm = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(mm);
        if (exact_order) {
            try mlx.check(mlx.mlx_matmul(&mm, prepared, wt, o.s));
        } else {
            const arm_m: usize = @max(1025, @as(usize, @intCast(@divTrunc(in, 3))) + 1);
            try mlx.check(mlx.engine.matmulRowsAs(&mm, prepared, wt, arm_m));
        }
        return o.own(mm);
    }
    return o.own(try exl3.kernels.innerGemmSortedTable(o.s, prepared, b.w, inputs.slots, exl3.kernels.gemmWindowRows(), true, inputs.tab));
}

/// `applyPreparedInner` of `x` prepared for the bank: a launch that reads its unfinished rows finishes them.
pub fn applyInner(o: *Ops, x: Arr, b: Bank) !Arr {
    try b.dec.requireTable();
    exl3.kernels.setDecodeParams(b.dec);
    const xb = try o.cast(x, .bfloat16);
    const rows = mlx.getShape(xb)[0];
    const inputs = try exl3.linear.gemmInputs(1, @intCast(rows), exl3.kernels.gemmWindowRows());
    const prepared = try exl3.kernels.prepareIndexed(o.s, xb, b.suh, inputs.slots);
    defer _ = mlx.mlx_array_free(prepared);
    return applyPreparedInner(o, prepared, b);
}

/// `applyPreparedInner` and its finish, BF16 rows.
pub fn applyPreparedStored(o: *Ops, prepared: Arr, b: Bank) !Arr {
    const inner = try applyPreparedInner(o, prepared, b);
    const inputs = try exl3.linear.gemmInputs(1, @intCast(mlx.getShape(prepared)[0]), exl3.kernels.gemmWindowRows());
    return try o.own(try exl3.kernels.finishIndexedAs(o.s, inner, b.svh, inputs.slots, .bfloat16));
}

/// A wide chunk through a bank the shared decoded arm refuses (past `trunk_gemm.DECODED_MAX_WEIGHTS`, or a K order
/// only the fused arm keeps): `applyPreparedStored` of its prepared rows, the sorted GEMM's bytes. Null where the
/// shared arms serve.
fn applyWideDecoded(o: *Ops, x: Arr, b: Bank) !?Arr {
    if (!exl3.trunk_gemm.decodedCanServe() or exl3.linear.force_arm != null) return null;
    const sh = mlx.getShape(x);
    if (sh.len != 2) return null;
    const rows = sh[0];
    const in = sh[1];
    if (rows < exl3.trunk_gemm.DECODED_MIN_ROWS or in != b.inDim()) return null;
    const out = b.outDim();
    const weights: usize = @as(usize, @intCast(in)) * @as(usize, @intCast(out));
    if (weights > decoded_max_weights) return null;
    const exact_order = exl3.trunk_gemm.denseOrderMatches(rows, in, out);
    if (weights <= exl3.trunk_gemm.DECODED_MAX_WEIGHTS and exact_order) return null;
    if (!exact_order and !mlx.engine.isStream(o.s)) return null;
    try b.dec.requireTable();
    exl3.kernels.setDecodeParams(b.dec);
    const xb = try o.cast(x, .bfloat16);
    const inputs = try exl3.linear.gemmInputs(1, @intCast(rows), exl3.kernels.gemmWindowRows());
    const prepared = try exl3.kernels.prepareIndexed(o.s, xb, b.suh, inputs.slots);
    defer _ = mlx.mlx_array_free(prepared);
    return try applyPreparedStored(o, prepared, b);
}

/// `x · Wᵀ` through the bank as f32: the rows rounded to BF16 (the kernels' input, as the BF16 forward rounds them), the
/// kernel's sums, the result rounded to BF16 and widened.
pub fn apply(o: *Ops, x: Arr, b: Bank) !Arr {
    if (try applyWideDecoded(o, x, b)) |y| return o.cast(y, .float32);
    if (thru_bf16_on) return o.own(try trunk_stq.matmulThruBf16(o.s, x, b.w, b.suh, b.svh, b.dec, false));
    const y = try o.own(try trunk_stq.matmul(o.s, try o.cast(x, .bfloat16), b.w, b.suh, b.svh, b.dec, false));
    return o.cast(y, .float32);
}

/// `apply` before its widening: the kernel's BF16 rows, f32 rows read through bf16 in the launch where it serves.
pub fn applyStored(o: *Ops, x: Arr, b: Bank) !Arr {
    if (try applyWideDecoded(o, x, b)) |y| return y;
    if (thru_bf16_on and mlx.mlx_array_dtype(x) == .float32) return o.own(try trunk_stq.matmulThruBf16Stored(o.s, x, b.w, b.suh, b.svh, b.dec));
    return o.own(try trunk_stq.matmul(o.s, try o.cast(x, .bfloat16), b.w, b.suh, b.svh, b.dec, false));
}

/// `RMSNorm_w(apply(x, b))` (`fwd.norm`'s f32 norm) with the bank's finish and the norm in one dispatch, `x` [T, in];
/// null where one launch does not serve it (`thru_bf16_on` off, rows past the multi-row arm).
pub fn applyNormed(o: *Ops, x: Arr, b: Bank, w: Arr, eps: f32) !?Arr {
    if (!thru_bf16_on or mlx.getShape(x).len != 2) return null;
    // A one-element array: a 0-d input reaches a kernel as a scalar, not the `eps[0]` it reads.
    const e = try o.own(try mlx.newData(&[_]f32{eps}, &.{1}, .float32));
    return if (try trunk_stq.matmulThruBf16Normed(o.s, x, b.w, b.suh, b.svh, b.dec, w, e)) |y| try o.own(y) else null;
}

/// `apply(x, b)` of `x` [T, in] f32 left as its launch's k-split partials (`o` owns them), for a launch that finishes
/// them where it reads them; null where `apply`'s launch folds its finish or does not serve the rows.
pub fn applyParts(o: *Ops, x: Arr, b: Bank) !?trunk_stq.Parts {
    if (!thru_bf16_on) return null;
    var p = (try trunk_stq.matmulThruBf16Parts(o.s, x, b.w, b.suh, b.svh, b.dec)) orelse return null;
    p.part = try o.own(p.part);
    return p;
}

/// `apply(x, b)` of `x` [T, in] f32 in one launch that finishes its own partials (`trunk_stq.matmulThruBf16Counted`),
/// on a stream with a counter page; null where `applyParts` is null or the stream has no page.
pub fn applyFinished(o: *Ops, x: Arr, b: Bank) !?Arr {
    if (!thru_bf16_on) return null;
    const y = (try trunk_stq.matmulThruBf16Counted(o.s, x, b.w, b.suh, b.svh, b.dec)) orelse return null;
    finished_launches += 1;
    return try o.own(y);
}

/// `silu(apply(x, gate)) · apply(x, up)` of `x` [T, in] f32, two banks of one shape and decode, in one launch that
/// finishes both (`trunk_stq.matmulThruBf16Swiglu`): the bytes of their launches and `hy4_chain.swiglu`. Null where it
/// does not serve.
pub fn applySwiglu(o: *Ops, x: Arr, gate: Bank, up: Bank) !?Arr {
    if (!thru_bf16_on or !std.meta.eql(gate.dec, up.dec)) return null;
    const y = (try trunk_stq.matmulThruBf16Swiglu(o.s, x, .{ .w = gate.w, .s = gate.suh, .b = gate.svh }, .{ .w = up.w, .s = up.suh, .b = up.svh }, gate.dec)) orelse return null;
    return try o.own(y);
}

/// `applyFinished` launches made since the process began.
pub var finished_launches: u64 = 0;

/// `<base>` as a trellis bank when the pack stores `<base>.trellis` (its views go to `owned`), else null. A bank in a
/// pack without `trunk_quant` (`dec` null), off the 128 grid or at a rate the table lacks is refused by name.
pub fn load(w: *const model_mod.Weights, base: []const u8, dec: ?exl3.format.Decode, owned: *std.ArrayList(Arr), gpa: std.mem.Allocator, s: mlx.mlx_stream) !?Bank {
    const t = (try trunk_stq.load(w, base, dec, owned, gpa, s)) orelse return null;
    return .{ .w = t.w, .suh = t.s, .svh = t.b, .dec = dec.? };
}

/// The bank decoded to a dense BF16 `[out, in]`, by the identity's rows through the kernel: for a weight the forward
/// reads by slices (absorbed MLA's `kv_b`), not by products. The caller owns the result.
pub fn decodeDense(gpa: std.mem.Allocator, s: mlx.mlx_stream, b: Bank) !Arr {
    const in: usize = @intCast(b.inDim());
    const eye = try gpa.alloc(f32, in * in);
    defer gpa.free(eye);
    @memset(eye, 0);
    for (0..in) |i| eye[i * in + i] = 1;
    var o: Ops = .{ .s = s };
    defer o.deinit();
    const x = try o.own(mlx.mlx_array_new_data(eye.ptr, &[_]c_int{ @intCast(in), @intCast(in) }, 2, .float32));
    // Row i of the product is column i of W: [in, out] = Wᵀ, exact in BF16 (one term a sum).
    const wt = try o.own(try trunk_stq.matmul(s, try o.cast(x, .bfloat16), b.w, b.suh, b.svh, b.dec, false));
    const out = try o.result(try o.contiguous(try o.cast(try o.transpose(wt, &.{ 1, 0 }), .bfloat16)));
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_array_eval(out));
    return out;
}

/// The routed experts of MoE layer `l` as the pack stores them (`model.layers.{l}.mlp.switch_mlp.{gate,up,down}_proj.gN`,
/// the trellis `trellis_pm`, pair-major), rate groups in order, as one tile-major ragged bank the clamped chain reads, or
/// null for a layer without them (a BF16 pack). The storage is owned regions (`owned_weights.raggedOwned`), as every
/// engine-resident STQ load's. The pack's tensors become views of that storage, so the layer's bytes stay resident once.
pub fn loadExperts(w: *model_mod.Weights, l: usize, s: mlx.mlx_stream, gpa: std.mem.Allocator) !?exl3.Ragged {
    var banks: [exl3.group_layout.max_groups]exl3.Bank = undefined;
    var n: usize = 0;
    while (n < banks.len) : (n += 1) {
        var projs: [3]exl3.Proj = undefined;
        var found: usize = 0;
        for ([_][]const u8{ "gate", "up", "down" }, &projs) |proj, *p| {
            var parts: [3]Arr = undefined;
            for ([_][]const u8{ "trellis_pm", "suh", "svh" }, &parts) |leaf, *d| {
                var buf: [128]u8 = undefined;
                const name = try std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.switch_mlp.{s}_proj.g{d}.{s}", .{ l, proj, n, leaf });
                if (w.get(name)) |a| {
                    d.* = a;
                    found += 1;
                }
            }
            p.* = .{ .trellis = parts[0], .suh = parts[1], .svh = parts[2] };
        }
        if (found == 0) break;
        if (found != 9) {
            std.log.err("hy_v4_flash: layer {d}'s expert group g{d} lacks a bank or a scale", .{ l, n });
            return error.Hy4ExpertLayout;
        }
        banks[n] = .{ .gate = projs[0], .up = projs[1], .down = projs[2] };
    }
    if (n == 0) return null;
    var rb = try @import("../owned_weights.zig").raggedOwned(s, banks[0..n], .pair_major, null);
    errdefer rb.deinit();
    if (rb.view_count != n) return rb;
    for (0..n) |g| for ([_][]const u8{ "gate", "up", "down" }, 0..) |proj, pi| for ([_][]const u8{ "trellis_pm", "suh", "svh" }, 0..) |leaf, li| {
        var buf: [128]u8 = undefined;
        const name = try std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.switch_mlp.{s}_proj.g{d}.{s}", .{ l, proj, g, leaf });
        const entry = w.map.getPtr(name) orelse continue;
        const p = switch (pi) {
            0 => rb.views[g].gate,
            1 => rb.views[g].up,
            else => rb.views[g].down,
        };
        var view = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(view);
        try mlx.check(mlx.mlx_array_set(&view, switch (li) {
            0 => p.trellis,
            1 => p.suh,
            else => p.svh,
        }));
        _ = mlx.mlx_array_free(entry.*);
        entry.* = view;
    };
    _ = gpa;
    return rb;
}

/// The routed experts over `lat` [T, L] through the clamped trellis chain (`gate ≤ limit`, `|up| ≤ limit`, as the
/// BF16 path), the picks [T, k] indexing `rb`'s experts (the router's rows follow the pack's group order), summed by
/// their weights into [T, L] f32: the decode chain up to `DECODE_ROWS_MAX` rows, the prefill GEMMs wider, both summing
/// into f32 (a caller adding groups' sums rounds once). The rows enter in BF16, as the trellis kernels take them. The
/// caller frees the result.
pub fn routed(s: mlx.mlx_stream, dec: exl3.format.Decode, limit: f32, rb: *const exl3.Ragged, lat: Arr, picks: Arr, weights: Arr) !Arr {
    return routedAs(s, dec, limit, rb, lat, picks, weights, .float32);
}

/// `routed` with its output in `out_dt`: BF16 where the next launch reads BF16, with no cast between.
pub fn routedAs(s: mlx.mlx_stream, dec: exl3.format.Decode, limit: f32, rb: *const exl3.Ragged, lat: Arr, picks: Arr, weights: Arr, out_dt: mlx.mlx_dtype) !Arr {
    var o: Ops = .{ .s = s };
    defer o.deinit();
    const sh = mlx.getShape(picks);
    const t = sh[0];
    const k = sh[1];
    const flat = try o.reshape(try o.cast(picks, .uint32), &.{t * k});
    const scores = try o.reshape(try o.cast(weights, .float32), &.{t * k});
    const x = try o.cast(lat, .bfloat16);
    const lim: c_int = @intFromFloat(limit);
    try dec.requireTable();
    exl3.kernels.setDecodeParams(dec);
    const y = if (@as(usize, @intCast(t)) <= exl3.kernels.DECODE_ROWS_MAX)
        try o.own(try exl3.rate_groups.moeDecodeClamped(s, x, rb, flat, scores, k, .float32, null, null, lim))
    else
        try o.own(try exl3.rate_groups.moePrefillEarlyTo(s, x, rb, flat, scores, k, lim, null, out_dt));
    return o.result(try o.reshape(try o.cast(y, out_dt), &.{ t, mlx.getShape(lat)[1] }));
}

/// The embedding stored as a trellis bank, read by row (docs/pack-format.md#hy4-stq): the bank's `[vocab, dim]` weight is
/// a linear with in = dim, out = vocab, and token t's row is column t of `Wᵀ` [dim, vocab]. The output side's Hadamard
/// mixes 128 consecutive ids, so a token's row needs its 128-id block alone: the block's output tiles over every input
/// tile, decoded on the host as the kernels apply them (`exl3.linear.decodeDense`), with the block's 128 `svh`. The
/// gather owns a host copy of the bank at its stored bytes (the caller drops the pack's tensors) and keeps the f32 `Wᵀ`
/// [dim, 128] of the blocks it decoded last, to `block_cache_bytes`.
pub const EmbedGather = struct {
    words: []u16,
    suh: []u16,
    svh: []u16,
    dim: usize,
    vocab: usize,
    n: u32,
    dec: exl3.format.Decode,
    cache: ?*Blocks = null,

    /// Decoded blocks, the least recently used replaced first. A slot names its block only once its decode is whole.
    const Blocks = struct {
        rows: []f32,
        block: []u32,
        used: []u64,
        tick: u64 = 0,

        const none = std.math.maxInt(u32);

        fn slot(c: *const Blocks, s: usize) []f32 {
            const w = c.rows.len / c.block.len;
            return c.rows[s * w ..][0..w];
        }

        fn touch(c: *Blocks, s: usize) void {
            c.tick += 1;
            c.used[s] = c.tick;
        }

        fn find(c: *Blocks, b: u32) ?[]f32 {
            const s = std.mem.indexOfScalar(u32, c.block, b) orelse return null;
            c.touch(s);
            return c.slot(s);
        }

        fn oldest(c: *const Blocks) usize {
            return std.mem.indexOfMin(u64, c.used);
        }
    };

    /// The slots `cache_bytes` holds at `dim`.
    fn slotsFor(dim: usize, cache_bytes: u64) usize {
        return @intCast(cache_bytes / (dim * 128 * @sizeOf(f32)));
    }

    /// The bytes a gather of `dim` keeps beside its copy of the bank, at `cache_bytes` of decoded blocks.
    pub fn cacheBytes(dim: usize, cache_bytes: u64) u64 {
        const slots = slotsFor(dim, cache_bytes);
        return if (slots == 0) 0 else slots * (dim * 128 * @sizeOf(f32) + @sizeOf(u32) + @sizeOf(u64)) + @sizeOf(Blocks);
    }

    /// Test seam: the blocks `rows` has decoded in this process.
    pub var blocks_decoded: usize = 0;

    /// A host copy of the bank's words and scales.
    pub fn of(gpa: std.mem.Allocator, b: Bank) !EmbedGather {
        return ofWith(gpa, b, block_cache_bytes);
    }

    /// The bytes of decoded blocks a gather keeps.
    pub const block_cache_bytes: u64 = 64 << 20;

    /// `of`, keeping up to `cache_bytes` of decoded blocks.
    pub fn ofWith(gpa: std.mem.Allocator, b: Bank, cache_bytes: u64) !EmbedGather {
        for ([_]Arr{ b.w, b.suh, b.svh }) |a| try mlx.check(mlx.mlx_array_eval(a));
        const sh = mlx.getShape(b.w);
        const dim: usize = @intCast(sh[1] * 16);
        const vocab: usize = @intCast(sh[2] * 16);
        const n: u32 = @intCast(sh[3]);
        try b.dec.requireTable();
        const words_src = (mlx.mlx_array_data_uint16(b.w) orelse return error.HostRead)[0 .. (dim / 16) * (vocab / 16) * n];
        const suh_src: [*]const u16 = @ptrCast(mlx.mlx_array_data_float16(b.suh) orelse return error.HostRead);
        const svh_src: [*]const u16 = @ptrCast(mlx.mlx_array_data_float16(b.svh) orelse return error.HostRead);
        const words = try gpa.dupe(u16, words_src);
        errdefer gpa.free(words);
        const suh = try gpa.dupe(u16, suh_src[0..dim]);
        errdefer gpa.free(suh);
        const svh = try gpa.dupe(u16, svh_src[0..vocab]);
        errdefer gpa.free(svh);
        const slots = slotsFor(dim, cache_bytes);
        const cache: ?*Blocks = if (slots == 0) null else c: {
            const c = try gpa.create(Blocks);
            errdefer gpa.destroy(c);
            const kept = try gpa.alloc(f32, slots * dim * 128);
            errdefer gpa.free(kept);
            const block = try gpa.alloc(u32, slots);
            errdefer gpa.free(block);
            @memset(block, Blocks.none);
            const used = try gpa.alloc(u64, slots);
            @memset(used, 0);
            c.* = .{ .rows = kept, .block = block, .used = used };
            break :c c;
        };
        return .{ .words = words, .suh = suh, .svh = svh, .dim = dim, .vocab = vocab, .n = n, .dec = b.dec, .cache = cache };
    }

    pub fn deinit(self: *EmbedGather, gpa: std.mem.Allocator) void {
        gpa.free(self.words);
        gpa.free(self.suh);
        gpa.free(self.svh);
        if (self.cache) |c| {
            gpa.free(c.rows);
            gpa.free(c.block);
            gpa.free(c.used);
            gpa.destroy(c);
        }
    }

    /// Test seam: the kept blocks emptied, so the next ask of any block decodes it.
    pub fn dropBlocks(self: EmbedGather) void {
        const c = self.cache orelse return;
        @memset(c.block, Blocks.none);
        @memset(c.used, 0);
    }

    /// The bytes the gather holds: the bank's stored bytes, and its decoded blocks with their ids and last uses.
    pub fn bytes(self: EmbedGather) u64 {
        const kept: u64 = if (self.cache) |c| 4 * c.rows.len + 4 * c.block.len + 8 * c.used.len + @sizeOf(Blocks) else 0;
        return 2 * (self.words.len + self.suh.len + self.svh.len) + kept;
    }

    /// Each id's row into `out` [ids.len · dim], f32. An id past the vocabulary is refused. A dry recording reads no
    /// value: its rows are zeros, and no block is decoded on the host.
    pub fn rows(self: EmbedGather, gpa: std.mem.Allocator, ids: []const u32, out: []f32) !void {
        const d = self.dim;
        if (out.len != ids.len * d) return error.BadExl3Shape;
        if (mlx.engine.dryOn()) {
            for (ids) |id| if (id >= self.vocab) return error.TokenOutOfRange;
            return @memset(out, 0);
        }
        const order = try gpa.alloc(u32, ids.len);
        defer gpa.free(order);
        for (order, 0..) |*o, i| o.* = @intCast(i);
        std.mem.sort(u32, order, ids, struct {
            fn lt(c: []const u32, a: u32, b: u32) bool {
                return c[a] < c[b];
            }
        }.lt);
        const ot = self.vocab / 16;
        const sub = try gpa.alloc(u16, (d / 16) * 8 * self.n);
        defer gpa.free(sub);
        const own = try gpa.alloc(f32, if (self.cache == null) d * 128 else 0);
        defer gpa.free(own);
        var i: usize = 0;
        while (i < order.len) {
            const b = ids[order[i]] / 128;
            if (ids[order[i]] >= self.vocab) return error.TokenOutOfRange;
            const blk = (if (self.cache) |c| c.find(b) else null) orelse blk: {
                const s: ?usize = if (self.cache) |c| c.oldest() else null;
                const into = if (s) |k| self.cache.?.slot(k) else own;
                if (s) |k| self.cache.?.block[k] = Blocks.none;
                // The block's 8 output tiles over every input tile, in the bank's own order.
                for (0..d / 16) |tk| @memcpy(sub[tk * 8 * self.n ..][0 .. 8 * self.n], self.words[(tk * ot + b * 8) * self.n ..][0 .. 8 * self.n]);
                try exl3.linear.decodeDense(gpa, sub, self.suh, self.svh[b * 128 ..][0..128], d, 128, self.n, self.dec, into);
                if (@import("builtin").is_test) blocks_decoded += 1;
                if (s) |k| {
                    self.cache.?.block[k] = b;
                    self.cache.?.touch(k);
                }
                break :blk into;
            };
            while (i < order.len and ids[order[i]] / 128 == b) : (i += 1) {
                const row = order[i];
                const col = ids[row] % 128;
                for (0..d) |k| out[row * d + k] = blk[k * 128 + col];
            }
        }
    }
};
