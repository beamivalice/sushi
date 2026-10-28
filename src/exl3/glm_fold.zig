//! GLM serial decode (one row): the affine shared expert's gate/up and down kernels ride the routed lane chain's pair
//! and down launches as extra threadgroups (`fuse.zig`), so a layer's gate/up, shared down, prepare, pair, middle,
//! down and reduce launches become five. Every part keeps its own statements, so the output is the bytes of the
//! separate chain; the finish reduce still adds the shared expert first, as `moeClampedWithShared` does.
const std = @import("std");
const mlx = @import("mlx_host").mlx;
const exl3 = @import("expert_exl3.zig");
const kernels = @import("expert_exl3_kernels.zig");
const rate_groups = @import("rate_groups.zig");
const codebook_tables = @import("codebook_tables.zig");
const fuse = @import("fuse.zig");
const group2 = @import("glm_group2.zig");
const root = @import("root.zig");

pub const Affine = struct { weight: mlx.mlx_array, scales: mlx.mlx_array, biases: mlx.mlx_array };

/// The host's one-token shared expert: its banks, and the text of its two 128-thread kernels for the stored width.
/// `gate_up` (eight rows a threadgroup) reads `x, wg, sg, bg, wu, su, bu, sigmoid` and writes `y`; `down` (sixteen rows)
/// reads `x, w, sc, bi` and writes `y`; both take the int `K`, their input width.
pub const Shared = struct {
    gate: Affine,
    up: Affine,
    down: Affine,
    sigmoid: mlx.mlx_array,
    gate_up_body: []const u8,
    gate_up_header: []const u8,
    gate_up_bits: c_int,
    down_body: []const u8,
    down_header: []const u8,
    down_bits: c_int,
};

const Stage = enum { pair, down };

/// What shapes a fused stage's kernel and launch.
const Key = struct {
    stage: Stage,
    hidden: c_int,
    inter: c_int,
    width: c_int,
    topk: c_int,
    rows: c_int,
    n: u32,
    one_row: bool,
    bits: c_int,
    dec: exl3.Decode,
    shared_first: bool,
};

const Built = struct { kernel: mlx.mlx_fast_metal_kernel, cfg: mlx.mlx_fast_metal_kernel_config };
var stages: std.AutoHashMapUnmanaged(Key, Built) = .empty;
/// Test and bench arm: which part's threadgroups the fused grid lists first.
pub var shared_first: bool = false;
var folded_layers: usize = 0;
var folded_launches: usize = 0;
/// Layers run folded, and the GPU launches they issued; test seams.
pub fn foldedCount() usize {
    return folded_layers;
}
pub fn launchCount() usize {
    return folded_launches;
}

fn ints(alloc: std.mem.Allocator, pairs: anytype) ![]const fuse.IntArg {
    const out = try alloc.alloc(fuse.IntArg, pairs.len);
    inline for (pairs, 0..) |p, i| out[i] = .{ .name = p[0], .value = p[1] };
    return out;
}

fn outs(alloc: std.mem.Allocator, comptime count: usize, dims: [count][]const c_int, dtype: mlx.mlx_dtype) ![]const fuse.Out {
    const out = try alloc.alloc(fuse.Out, count);
    for (dims, out) |d, *o| o.* = .{ .shape = try alloc.dupe(c_int, d), .dtype = dtype };
    return out;
}

/// The routed kernel's template ints: the single-row lane kernels read `NHW`, the group kernels also their slot count,
/// reduction order and tiles per threadgroup.
fn a_ints(alloc: std.mem.Allocator, k: Key, input: c_int, output: c_int) ![]const fuse.IntArg {
    const n: c_int = @intCast(k.n);
    if (k.rows == 1) return ints(alloc, .{ .{ "IDIM", input }, .{ "ODIM", output }, .{ "NHW", n } });
    return ints(alloc, .{ .{ "IDIM", input }, .{ "ODIM", output }, .{ "NSLOTS", k.rows * k.topk }, .{ "NHW", n }, .{ "SERIAL_REDUCTION", 1 }, .{ "TILES", 2 } });
}

fn parts(alloc: std.mem.Allocator, k: Key, sh: Shared, coded_header: []const u8) !struct { a: fuse.Part, b: fuse.Part } {
    const multi = k.rows > 1;
    const slots = k.rows * k.topk;
    switch (k.stage) {
        .pair => {
            const tiles = @divExact(k.inter, 16);
            return .{
                .a = .{
                    .launch = .{ .tgs = .{ if (multi) @divExact(k.inter, 32) else if (k.one_row) @divExact(tiles, 2) else tiles, slots, 2 }, .threads = 128, .ints = try a_ints(alloc, k, k.hidden, k.inter), .outs = try outs(alloc, 2, .{ &.{ slots, k.inter }, &.{ slots, k.inter } }, .float16) },
                    .body = if (multi) (if (k.rows == 3) group2.LANE_THREE_PAIR_SOURCE else group2.LANE_TILED_PAIR_SOURCE) else if (k.one_row) kernels.ONE_ROW_PAIR_SOURCE else kernels.LANE_PAIR_SOURCE,
                    .ins = &.{ "xg", "xu", "tg", "tu", "slots" },
                    .outs = &.{ "yg", "yu" },
                    .header = coded_header,
                },
                .b = .{
                    .launch = .{ .tgs = .{ @divExact(k.width, 8), k.rows, 1 }, .threads = 128, .ints = try ints(alloc, .{ .{ "K", k.hidden }, .{ "N", k.width } }), .outs = try outs(alloc, 1, .{&.{ 1, k.rows, k.width }}, .bfloat16) },
                    .body = sh.gate_up_body,
                    .ins = &.{ "x", "wg", "sg", "bg", "wu", "su", "bu", "sigmoid" },
                    .outs = &.{"y"},
                    .header = sh.gate_up_header,
                },
            };
        },
        .down => {
            const tiles = @divExact(k.hidden, 16);
            return .{
                .a = .{
                    .launch = .{ .tgs = .{ if (multi) @divExact(k.hidden, 32) else if (k.one_row) @divExact(tiles, 2) else tiles, slots, 1 }, .threads = 128, .ints = try a_ints(alloc, k, k.inter, k.hidden), .outs = try outs(alloc, 1, .{&.{ slots, k.hidden }}, .float16) },
                    .body = if (multi) (if (k.rows == 3) group2.LANE_THREE_SOURCE else group2.LANE_TILED_SOURCE) else if (k.one_row) kernels.ONE_ROW_COOP_SOURCE else kernels.DOWN_LANE_COOP_SOURCE,
                    .ins = &.{ "x", "trellis", "slots" },
                    .outs = &.{"y"},
                    .header = coded_header,
                },
                .b = .{
                    .launch = .{ .tgs = .{ @divExact(k.hidden, 16), k.rows, 1 }, .threads = 128, .ints = try ints(alloc, .{ .{ "K", k.width }, .{ "N", k.hidden } }), .outs = try outs(alloc, 1, .{&.{ 1, k.rows, k.hidden }}, .bfloat16) },
                    .body = sh.down_body,
                    .ins = &.{ "x", "w", "sc", "bi" },
                    .outs = &.{"y"},
                    .header = sh.down_header,
                },
            };
        },
    }
}

fn stageFor(k: Key, sh: Shared) !*const Built {
    if (stages.getPtr(k)) |b| return b;
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const p = try parts(alloc, k, sh, try rate_groups.headerOf(k.dec));
    const src = try fuse.source(alloc, p.a, p.b);
    const head = try fuse.header(alloc, p.a, p.b);
    var name: std.ArrayList(u8) = .empty;
    try name.print(alloc, "sushi_glm_fold_{s}_{s}_w{d}", .{ @tagName(k.stage), @tagName(k.dec.codebook), k.dec.window.bits() });
    if (k.dec.codebook.tabledHashMask()) try name.print(alloc, "_t{s}", .{codebook_tables.tables[k.dec.table.?].digest});
    try name.print(alloc, "_a{d}", .{k.bits});
    if (k.rows > 1) try name.print(alloc, "_rows{s}", .{if (k.rows == 3) "3" else "2"});
    if (k.one_row) try name.appendSlice(alloc, "_one");
    try name.append(alloc, 0);
    var slot: ?mlx.mlx_fast_metal_kernel = null;
    const kernel = try kernels.getNamedKernel(&slot, @ptrCast(name.items.ptr), try fuse.insOf(alloc, p.a, p.b), try fuse.outsOf(alloc, p.a, p.b), src, head);
    const built: Built = .{ .kernel = kernel, .cfg = try fuse.config(alloc, p.a.launch, p.b.launch, k.shared_first) };
    try stages.put(std.heap.page_allocator, k, built);
    return stages.getPtr(k).?;
}

fn get(v: mlx.mlx_vector_array, i: usize) !mlx.mlx_array {
    var a = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(a);
    try mlx.check(mlx.mlx_vector_array_get(&a, v, i));
    return a;
}

/// `moeClampedWithShared` with the shared expert `sh` folded in: the routed sum plus the shared expert's bf16 output,
/// in `x`'s shape. Null where the chain or the shared kernels do not serve the inputs; nothing has run then.
pub fn moeClampedSharedFolded(s: mlx.mlx_stream, x: mlx.mlx_array, bank: root.Bank, inds: mlx.mlx_array, scores: mlx.mlx_array, dec: exl3.Decode, limit: c_int, sh: Shared) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s) or !dec.codebook.servesGlmFastPaths() or limit != 10 or mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_ndim(x) != 3) return null;
    const xs = mlx.getShape(x);
    const ix = mlx.getShape(inds);
    if (mlx.mlx_array_strides(x)[2] != 1 or xs[0] != 1 or xs[1] != 1 or ix.len != 3 or ix[0] != 1 or ix[1] != 1 or ix[2] <= 0 or ix[2] > kernels.REDUCE_MAX_TOPK) return null;
    const score_type = mlx.mlx_array_dtype(scores);
    if (!std.mem.eql(c_int, ix, mlx.getShape(scores)) or (score_type != .float32 and score_type != .bfloat16 and score_type != .float16)) return null;
    const id_type = mlx.mlx_array_dtype(inds);
    if (id_type != .uint32 and id_type != .int32) return null;
    const hidden = xs[2];
    const topk = ix[2];
    const gs = mlx.getShape(bank.gate.trellis);
    const us = mlx.getShape(bank.up.trellis);
    const ds = mlx.getShape(bank.down.trellis);
    if (hidden <= 0 or @mod(hidden, 128) != 0 or gs.len != 4 or us.len != 4 or ds.len != 4 or gs[2] <= 0 or gs[2] > std.math.maxInt(c_int) / 16) return null;
    const inter = gs[2] * 16;
    root.validateClampedProjection(bank.gate, gs[0], hidden, inter) catch return null;
    root.validateClampedProjection(bank.up, gs[0], hidden, inter) catch return null;
    root.validateClampedProjection(bank.down, gs[0], inter, hidden) catch return null;
    if (gs[3] != us[3] or gs[3] != ds[3] or !kernels.laneServesRate(gs[3])) return null;
    // The gate/up kernel takes eight rows a threadgroup, the down sixteen.
    const width = mlx.getShape(sh.gate.weight)[0];
    if (width <= 0 or @mod(width, 8) != 0 or mlx.getShape(sh.down.weight)[0] != hidden or @mod(hidden, 16) != 0) return null;
    const rate = exl3.kFromPackedDim(@intCast(gs[3])) orelse return null;

    kernels.setDecodeParams(dec);
    var x2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x2);
    try mlx.check(mlx.mlx_reshape(&x2, x, &[_]c_int{ 1, hidden }, 2, s));
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, inds, &[_]c_int{topk}, 1, s));
    var slots = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(slots);
    try mlx.check(mlx.mlx_astype(&slots, flat, .uint32, s));
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    try mlx.check(mlx.mlx_reshape(&sc, scores, &[_]c_int{topk}, 1, s));

    const prep = try kernels.lanePairPrepare(s, x2, bank.gate.suh, bank.up.suh, slots, null, hidden, topk, topk);
    defer _ = mlx.mlx_array_free(prep[0]);
    defer _ = mlx.mlx_array_free(prep[1]);
    var key: Key = .{ .stage = .pair, .hidden = hidden, .inter = inter, .width = width, .topk = topk, .rows = 1, .n = rate.n, .one_row = kernels.oneRowCoopServes(gs[1], gs[2]), .bits = sh.gate_up_bits, .dec = dec, .shared_first = shared_first };
    const pair = try stageFor(key, sh);
    const pair_ov = try kernels.applyOuts(s, pair.kernel, &.{ prep[0], prep[1], bank.gate.trellis, bank.up.trellis, slots, x, sh.gate.weight, sh.gate.scales, sh.gate.biases, sh.up.weight, sh.up.scales, sh.up.biases, sh.sigmoid }, pair.cfg, 3);
    defer _ = mlx.mlx_vector_array_free(pair_ov);
    const ig = try get(pair_ov, 0);
    defer _ = mlx.mlx_array_free(ig);
    const iu = try get(pair_ov, 1);
    defer _ = mlx.mlx_array_free(iu);
    const mid = try get(pair_ov, 2);
    defer _ = mlx.mlx_array_free(mid);

    const prepared = try kernels.Group2Support.lane_middle(s, ig, iu, bank.gate.svh, bank.up.svh, bank.down.suh, slots, inter, topk, limit);
    defer _ = mlx.mlx_array_free(prepared);

    key.stage = .down;
    key.one_row = kernels.oneRowCoopServes(ds[1], ds[2]);
    key.bits = sh.down_bits;
    const down_stage = try stageFor(key, sh);
    const down_ov = try kernels.applyOuts(s, down_stage.kernel, &.{ prepared, bank.down.trellis, slots, mid, sh.down.weight, sh.down.scales, sh.down.biases }, down_stage.cfg, 2);
    defer _ = mlx.mlx_vector_array_free(down_ov);
    const down = try get(down_ov, 0);
    defer _ = mlx.mlx_array_free(down);
    const shared_out = try get(down_ov, 1);
    defer _ = mlx.mlx_array_free(shared_out);

    folded_layers += 1;
    // Prepare, pair, middle, down, reduce.
    folded_launches += 5;
    const y = try kernels.downFinishReduceWithShared(s, down, bank.down.svh, slots, sc, hidden, 1, topk, .bfloat16, shared_out, null, true);
    defer _ = mlx.mlx_array_free(y);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, y, xs.ptr, xs.len, s));
    return out;
}

/// `glm_group2.moeLayoutShared` (lane layout, serial reduction, grouped down) with the shared expert `sh` folded in, for
/// two to eight rows of `x` [1, rows, hidden]: the routed sum plus the shared expert's bf16 rows, in `x`'s shape. Each
/// shared row is the one-token kernels' bytes, which are the rows kernels'. Null where the chain or the shared kernels do
/// not serve the inputs; nothing has run then.
pub fn moeLayoutSharedFolded(s: mlx.mlx_stream, x: mlx.mlx_array, bank: root.Bank, indices: mlx.mlx_array, scores: mlx.mlx_array, dec: exl3.Decode, sh: Shared) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s) or !dec.codebook.servesGlmFastPaths() or mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_ndim(x) != 3) return null;
    const xs = mlx.getShape(x);
    const ix = mlx.getShape(indices);
    if (mlx.mlx_array_strides(x)[2] != 1 or xs[0] != 1 or xs[1] < 2 or xs[1] > 8 or ix.len != 3 or ix[0] != 1 or ix[1] != xs[1] or ix[2] != 8) return null;
    const score_type = mlx.mlx_array_dtype(scores);
    if (!std.mem.eql(c_int, ix, mlx.getShape(scores)) or (score_type != .float32 and score_type != .bfloat16 and score_type != .float16)) return null;
    const id_type = mlx.mlx_array_dtype(indices);
    if (id_type != .uint32 and id_type != .int32) return null;
    const rows = xs[1];
    const hidden = xs[2];
    const topk = ix[2];
    const nslots = rows * topk;
    const gs = mlx.getShape(bank.gate.trellis);
    const us = mlx.getShape(bank.up.trellis);
    const ds = mlx.getShape(bank.down.trellis);
    if (hidden <= 0 or @mod(hidden, 128) != 0 or gs.len != 4 or us.len != 4 or ds.len != 4 or gs[2] <= 0 or gs[2] > std.math.maxInt(c_int) / 16) return null;
    const inter = gs[2] * 16;
    root.validateClampedProjection(bank.gate, gs[0], hidden, inter) catch return null;
    root.validateClampedProjection(bank.up, gs[0], hidden, inter) catch return null;
    root.validateClampedProjection(bank.down, gs[0], inter, hidden) catch return null;
    if (gs[3] != us[3] or gs[3] != ds[3] or !kernels.laneServesRate(gs[3]) or @mod(gs[2], 8) != 0 or @mod(ds[2], 8) != 0) return null;
    const width = mlx.getShape(sh.gate.weight)[0];
    if (width <= 0 or @mod(width, 8) != 0 or mlx.getShape(sh.down.weight)[0] != hidden or @mod(hidden, 32) != 0) return null;
    const rate = exl3.kFromPackedDim(@intCast(gs[3])) orelse return null;

    kernels.setDecodeParams(dec);
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, x, &[_]c_int{ rows, hidden }, 2, s));
    var ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids);
    try mlx.check(mlx.mlx_reshape(&ids, indices, &[_]c_int{nslots}, 1, s));
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    try mlx.check(mlx.mlx_reshape(&sc, scores, &[_]c_int{nslots}, 1, s));

    const prep = try kernels.lanePairPrepare(s, flat, bank.gate.suh, bank.up.suh, ids, null, hidden, nslots, topk);
    defer _ = mlx.mlx_array_free(prep[0]);
    defer _ = mlx.mlx_array_free(prep[1]);
    var key: Key = .{ .stage = .pair, .hidden = hidden, .inter = inter, .width = width, .topk = topk, .rows = rows, .n = rate.n, .one_row = false, .bits = sh.gate_up_bits, .dec = dec, .shared_first = shared_first };
    const pair = try stageFor(key, sh);
    const pair_ov = try kernels.applyOuts(s, pair.kernel, &.{ prep[0], prep[1], bank.gate.trellis, bank.up.trellis, ids, x, sh.gate.weight, sh.gate.scales, sh.gate.biases, sh.up.weight, sh.up.scales, sh.up.biases, sh.sigmoid }, pair.cfg, 3);
    defer _ = mlx.mlx_vector_array_free(pair_ov);
    const ig = try get(pair_ov, 0);
    defer _ = mlx.mlx_array_free(ig);
    const iu = try get(pair_ov, 1);
    defer _ = mlx.mlx_array_free(iu);
    const mid = try get(pair_ov, 2);
    defer _ = mlx.mlx_array_free(mid);

    const middle = try kernels.Group2Support.lane_middle(s, ig, iu, bank.gate.svh, bank.up.svh, bank.down.suh, ids, inter, nslots, 10);
    defer _ = mlx.mlx_array_free(middle);

    key.stage = .down;
    key.bits = sh.down_bits;
    const down_stage = try stageFor(key, sh);
    const down_ov = try kernels.applyOuts(s, down_stage.kernel, &.{ middle, bank.down.trellis, ids, mid, sh.down.weight, sh.down.scales, sh.down.biases }, down_stage.cfg, 2);
    defer _ = mlx.mlx_vector_array_free(down_ov);
    const down = try get(down_ov, 0);
    defer _ = mlx.mlx_array_free(down);
    const shared_out = try get(down_ov, 1);
    defer _ = mlx.mlx_array_free(shared_out);
    var shared_rows = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(shared_rows);
    try mlx.check(mlx.mlx_reshape(&shared_rows, shared_out, &[_]c_int{ rows, hidden }, 2, s));

    folded_layers += 1;
    folded_launches += 5;
    const y = try kernels.downFinishReduceWithShared(s, down, bank.down.svh, ids, sc, hidden, rows, topk, .bfloat16, shared_rows, null, false);
    defer _ = mlx.mlx_array_free(y);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, y, xs.ptr, xs.len, s));
    return out;
}
