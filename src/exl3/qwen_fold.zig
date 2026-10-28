//! Qwen3.8-Flash-Next decode on a one-bank layer (MCG EXL3): the shared expert's kernels (`qwen_riders.zig`) ride the
//! routed chain's pair and fused-mid down launches as extra threadgroups (`fuse.zig`), and the gated shared add rides
//! the finish reduce, so a layer's shared gate/up, SwiGLU, down, gate logit, pair, down, reduce and sigmoid tail
//! become three launches. Every part keeps its own statements, so the output is the bytes of the separate chain.
const std = @import("std");
const mlx = @import("mlx_host").mlx;
const exl3 = @import("expert_exl3.zig");
const kernels = @import("expert_exl3_kernels.zig");
const rate_groups = @import("rate_groups.zig");
const codebook_tables = @import("codebook_tables.zig");
const fuse = @import("fuse.zig");
const riders = @import("qwen_riders.zig");
const root = @import("root.zig");

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
    group: c_int,
    logit_group: c_int,
    logit: bool,
    layout: kernels.GemvLayout,
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

fn outs(alloc: std.mem.Allocator, comptime count: usize, dims: [count][]const c_int, dtypes: [count]mlx.mlx_dtype) ![]const fuse.Out {
    const out = try alloc.alloc(fuse.Out, count);
    for (dims, dtypes, out) |d, t, *o| o.* = .{ .shape = try alloc.dupe(c_int, d), .dtype = t };
    return out;
}

/// The routed pair launch (`pairGemv`) and the fused-mid down launch (`downGemvFusedMid`), as those functions
/// configure them.
fn routedParts(alloc: std.mem.Allocator, stage: Stage, hidden: c_int, inter: c_int, topk: c_int, rows: c_int, n: u32, layout: kernels.GemvLayout, header: []const u8) !fuse.Part {
    const nslots = rows * topk;
    const nsplit: c_int = @intCast(kernels.pairSplitCountFor(hidden));
    const funnel: c_int = @intFromBool(layout.funnel);
    switch (stage) {
        .pair => return .{
            .launch = .{ .tgs = .{ @divExact(@divExact(inter, 16), layout.tiles), nslots, nsplit }, .threads = 128, .ints = try ints(alloc, .{ .{ "IDIM", hidden }, .{ "ODIM", inter }, .{ "NSPLIT", nsplit }, .{ "NHW", @as(c_int, @intCast(n)) }, .{ "TOPK", topk }, .{ "FUNNEL", funnel }, .{ "OTPT", layout.tiles } }), .outs = try outs(alloc, 2, .{ &.{ nslots * nsplit, inter }, &.{ nslots * nsplit, inter } }, .{ .float32, .float32 }) },
            .body = kernels.PAIR_GEMV_SOURCE,
            .ins = &.{ "x", "suhg", "suhu", "tg", "tu", "slots" },
            .outs = &.{ "yg", "yu" },
            .header = header,
        },
        .down => return .{
            .launch = .{ .tgs = .{ @divExact(@divExact(hidden, 16), layout.tiles), nslots, 1 }, .threads = 128, .ints = try ints(alloc, .{ .{ "IDIM", inter }, .{ "ODIM", hidden }, .{ "NSPLIT", nsplit }, .{ "NHW", @as(c_int, @intCast(n)) }, .{ "FUNNEL", funnel }, .{ "OTPT", layout.tiles } }), .outs = try outs(alloc, 1, .{&.{ nslots, hidden }}, .{.float16}) },
            .body = kernels.DOWN_FUSED_SOURCE,
            .ins = &.{ "ig", "iu", "trellis", "svhg", "svhu", "suhd", "slots" },
            .outs = &.{"y"},
            .header = header,
        },
    }
}

fn stageFor(k: Key) !*const Built {
    if (stages.getPtr(k)) |b| return b;
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const geo: riders.Geometry = .{ .hidden = k.hidden, .width = k.width, .group = k.group, .logit_group = k.logit_group };
    const a = try routedParts(alloc, k.stage, k.hidden, k.inter, k.topk, k.rows, k.n, k.layout, try rate_groups.headerOf(k.dec));
    const b = switch (k.stage) {
        .pair => try riders.gateUpPart(alloc, geo, k.rows, k.logit),
        .down => try riders.downPart(alloc, geo, k.rows),
    };
    const src = try fuse.source(alloc, a, b);
    const head = try fuse.header(alloc, a, b);
    var name: std.ArrayList(u8) = .empty;
    try name.print(alloc, "sushi_qwen_fold_{s}_{s}_w{d}", .{ @tagName(k.stage), @tagName(k.dec.codebook), k.dec.window.bits() });
    if (k.dec.codebook.tabledHashMask()) try name.print(alloc, "_t{s}", .{codebook_tables.tables[k.dec.table.?].digest});
    if (k.logit) try name.appendSlice(alloc, "_logit");
    try name.append(alloc, 0);
    var slot: ?mlx.mlx_fast_metal_kernel = null;
    const kernel = try kernels.getNamedKernel(&slot, @ptrCast(name.items.ptr), try fuse.insOf(alloc, a, b), try fuse.outsOf(alloc, a, b), src, head);
    const built: Built = .{ .kernel = kernel, .cfg = try fuse.config(alloc, a.launch, b.launch, k.shared_first) };
    try stages.put(std.heap.page_allocator, k, built);
    return stages.getPtr(k).?;
}

fn get(v: mlx.mlx_vector_array, i: usize) !mlx.mlx_array {
    var a = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(a);
    try mlx.check(mlx.mlx_vector_array_get(&a, v, i));
    return a;
}

/// The routed sum plus `sigmoid(gate) * shared` of `moeWithGatedShared`, with the shared expert `sh` folded into the
/// routed chain's launches, for one to eight rows of `x` [1, rows, hidden]: bf16, in `x`'s shape. `gate` [rows] is the
/// shared-expert gate's bf16 logit unless `sh.logit` names its bank. Null where the chain or the riders do not serve
/// the inputs; nothing has run then.
pub fn moeGatedSharedFolded(s: mlx.mlx_stream, x: mlx.mlx_array, bank: root.Bank, inds: mlx.mlx_array, scores: mlx.mlx_array, dec: exl3.Decode, sh: riders.Shared, gate: ?mlx.mlx_array) !?mlx.mlx_array {
    if (!mlx.streamIsGpu(s)) return null;
    const route = riders.Routing.of(x, inds, scores) orelse return null;
    const rows = route.rows;
    const topk = route.topk;
    const hidden = mlx.getShape(x)[2];
    const gs = mlx.getShape(bank.gate.trellis);
    const us = mlx.getShape(bank.up.trellis);
    const ds = mlx.getShape(bank.down.trellis);
    if (hidden <= 0 or @mod(hidden, 128) != 0 or gs.len != 4 or us.len != 4 or ds.len != 4 or gs[2] <= 0 or gs[2] > std.math.maxInt(c_int) / 16) return null;
    const inter = gs[2] * 16;
    root.validateClampedProjection(bank.gate, gs[0], hidden, inter) catch return null;
    root.validateClampedProjection(bank.up, gs[0], hidden, inter) catch return null;
    root.validateClampedProjection(bank.down, gs[0], inter, hidden) catch return null;
    if (gs[3] != us[3]) return null;
    const rate = exl3.kFromPackedDim(@intCast(gs[3])) orelse return null;
    const down_rate = exl3.kFromPackedDim(@intCast(ds[3])) orelse return null;
    const geo = riders.geometry(sh, hidden) orelse return null;
    const want_logit = sh.logit != null;
    if (!want_logit) {
        const g = gate orelse return null;
        if (mlx.mlx_array_dtype(g) != .bfloat16 or mlx.mlx_array_size(g) != @as(usize, @intCast(rows))) return null;
    }
    dec.requireTable() catch return null;

    kernels.setDecodeParams(dec);
    const nslots = rows * topk;
    var x2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x2);
    try mlx.check(mlx.mlx_reshape(&x2, x, &[_]c_int{ rows, hidden }, 2, s));
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, inds, &[_]c_int{nslots}, 1, s));
    var slots = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(slots);
    try mlx.check(mlx.mlx_astype(&slots, flat, .uint32, s));
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    try mlx.check(mlx.mlx_reshape(&sc, scores, &[_]c_int{nslots}, 1, s));

    var key: Key = .{ .stage = .pair, .hidden = hidden, .inter = inter, .width = geo.width, .topk = topk, .rows = rows, .n = rate.n, .group = geo.group, .logit_group = geo.logit_group, .logit = want_logit, .layout = kernels.gemvLayout(@divExact(inter, 16)), .dec = dec, .shared_first = shared_first };
    const pair = try stageFor(key);
    var buf: [11]mlx.mlx_array = undefined;
    const rider_in = riders.gateUpInputs(sh, x2, &buf);
    var pair_in: [17]mlx.mlx_array = undefined;
    pair_in[0..6].* = .{ x2, bank.gate.suh, bank.up.suh, bank.gate.trellis, bank.up.trellis, slots };
    @memcpy(pair_in[6..][0..rider_in.len], rider_in);
    const pair_ov = try kernels.applyOuts(s, pair.kernel, pair_in[0 .. 6 + rider_in.len], pair.cfg, if (want_logit) 4 else 3);
    defer _ = mlx.mlx_vector_array_free(pair_ov);
    const ig = try get(pair_ov, 0);
    defer _ = mlx.mlx_array_free(ig);
    const iu = try get(pair_ov, 1);
    defer _ = mlx.mlx_array_free(iu);
    const mid = try get(pair_ov, 2);
    defer _ = mlx.mlx_array_free(mid);
    const logit_out: ?mlx.mlx_array = if (want_logit) try get(pair_ov, 3) else null;
    defer if (logit_out) |l| {
        _ = mlx.mlx_array_free(l);
    };

    key.stage = .down;
    key.n = down_rate.n;
    key.layout = kernels.gemvLayout(@divExact(hidden, 16));
    const down_stage = try stageFor(key);
    const down_ov = try kernels.applyOuts(s, down_stage.kernel, &.{ ig, iu, bank.down.trellis, bank.gate.svh, bank.up.svh, bank.down.suh, slots, mid, sh.down.weight, sh.down.scales, sh.down.biases }, down_stage.cfg, 2);
    defer _ = mlx.mlx_vector_array_free(down_ov);
    const down = try get(down_ov, 0);
    defer _ = mlx.mlx_array_free(down);
    const shared_out = try get(down_ov, 1);
    defer _ = mlx.mlx_array_free(shared_out);

    folded_layers += 1;
    // Pair, down, reduce.
    folded_launches += 3;
    const y = try kernels.downFinishReduceWithShared(s, down, bank.down.svh, slots, sc, hidden, rows, topk, .bfloat16, shared_out, logit_out orelse gate.?, false);
    defer _ = mlx.mlx_array_free(y);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    const xs = mlx.getShape(x);
    try mlx.check(mlx.mlx_reshape(&out, y, xs.ptr, xs.len, s));
    return out;
}
