//! A shared expert stored as trellis banks (a stacked gate/up and a down, the trunk's dense linears) as the parts
//! `fuse.zig` lays beside a routed chain's pair and fused-mid down launches. Each part is the multi-row NAX kernel of
//! `project_rows.zig` with its own k split, so a linear's partial sums are the bytes of its own launch; what its two
//! finish dispatches did moves into the neighbour that reads the partials: the SwiGLU mid is built from the gate/up
//! partials while the down kernel prepares its input, and the finish reduce sums the down partials.
const std = @import("std");
const mlx = @import("mlx_host").mlx;
const exl3 = @import("expert_exl3.zig");
const kernels = @import("expert_exl3_kernels.zig");
const kernel_src = @import("kernel_source.zig");
const project_rows = @import("project_rows.zig");
const fuse = @import("fuse.zig");
const root = @import("root.zig");

/// `gate_up` stacks the gate (columns below the half) and the up; `down` maps the half back to the hidden width.
pub const Banks = struct { gate_up: root.Proj, down: root.Proj, dec: exl3.Decode };

pub const Geo = struct {
    hidden: c_int,
    width: c_int,
    gu: project_rows.RowsGeom,
    dn: project_rows.RowsGeom,
    n_gu: u32,
    n_dn: u32,
    dec: exl3.Decode,
};

fn isF16(a: mlx.mlx_array, size: c_int) bool {
    return a.ctx != null and mlx.mlx_array_dtype(a) == .float16 and mlx.mlx_array_size(a) == @as(usize, @intCast(size));
}

/// What the parts serve for `rows` rows of `hidden`-wide inputs, or null: NAX launches with a k split on both
/// linears, every bank on the 128 grid.
pub fn geometry(b: Banks, hidden: c_int, rows: c_int) ?Geo {
    if (b.gate_up.trellis.ctx == null or b.down.trellis.ctx == null) return null;
    const gs = mlx.getShape(b.gate_up.trellis);
    const ds = mlx.getShape(b.down.trellis);
    if (gs.len != 4 or ds.len != 4 or gs[0] != 1 or ds[0] != 1) return null;
    if (mlx.mlx_array_dtype(b.gate_up.trellis) != .uint16 or mlx.mlx_array_dtype(b.down.trellis) != .uint16) return null;
    const both = gs[2] * 16;
    if (@mod(hidden, 128) != 0 or gs[1] * 16 != hidden or @mod(both, 256) != 0 or ds[2] * 16 != hidden) return null;
    const width = @divExact(both, 2);
    if (ds[1] * 16 != width) return null;
    if (!isF16(b.gate_up.suh, 2 * hidden) or !isF16(b.gate_up.svh, both) or !isF16(b.down.suh, width) or !isF16(b.down.svh, hidden)) return null;
    b.dec.requireTable() catch return null;
    const n_gu = (kernels.packedRate(gs[3]) catch return null).n;
    const n_dn = (kernels.packedRate(ds[3]) catch return null).n;
    const prev = project_rows.rows_exact;
    project_rows.rows_exact = true;
    defer project_rows.rows_exact = prev;
    const gu = project_rows.rowsGeometry(hidden, both, rows, 1);
    const dn = project_rows.rowsGeometry(width, hidden, rows, 1);
    for ([_]project_rows.RowsGeom{ gu, dn }) |g| if (!g.nax or g.kph != 1 or g.nsplit < 2) return null;
    return .{ .hidden = hidden, .width = width, .gu = gu, .dn = dn, .n_gu = n_gu, .n_dn = n_dn, .dec = b.dec };
}

fn codecHeader(dec: exl3.Decode) ![:0]const u8 {
    const prev = kernels.activeDecode();
    kernels.setDecodeParams(dec);
    defer kernels.setDecodeParams(prev);
    return switch (dec.codebook) {
        inline else => |cb| switch (dec.window) {
            inline else => |win| kernel_src.withTableRows(cb, comptime kernel_src.codebookHelpers(cb, win)),
        },
    };
}

fn ints(alloc: std.mem.Allocator, g: project_rows.RowsGeom, in_dim: c_int, out_dim: c_int, bound: c_int, rows: c_int, n: u32, extra: []const fuse.IntArg) ![]const fuse.IntArg {
    const out = try alloc.alloc(fuse.IntArg, 16 + extra.len);
    const base = [_]fuse.IntArg{
        .{ .name = "IDIM", .value = in_dim },
        .{ .name = "ODIM", .value = out_dim },
        .{ .name = "ROWS", .value = rows },
        .{ .name = "GROUPS", .value = 1 },
        .{ .name = "TGRP", .value = g.tgrp },
        .{ .name = "KPH", .value = g.kph },
        .{ .name = "NSPLIT", .value = g.nsplit },
        .{ .name = "CHUNK", .value = g.chunk },
        .{ .name = "FOLD", .value = 0 },
        .{ .name = "NHW", .value = @intCast(n) },
        .{ .name = "SB1", .value = bound },
        .{ .name = "SB2", .value = out_dim },
        .{ .name = "SB3", .value = out_dim },
        .{ .name = "UNR", .value = g.unr },
        .{ .name = "R32", .value = @intFromBool(g.r32) },
        .{ .name = "SW", .value = @intFromBool(g.sw) },
    };
    @memcpy(out[0..16], &base);
    @memcpy(out[16..], extra);
    return out;
}

/// The kernels write partials only, so the output type their folded branch names never stores.
const OUT_TYPE = "typedef bfloat16_t OutT;\n";

const MID_HEADER: [:0]const u8 =
    \\struct mid_row_t {
    \\  bfloat16_t v[4];
    \\  bfloat16_t operator[](uint i) const { return v[i >> 5u]; }
    \\};
    \\static inline mid_row_t mid_row(const device float *part, const device half *svh, const device bfloat16_t *sig, uint r, uint k, uint lane, uint nsplit, uint odim, uint width) {
    \\  float4 g = 0.0f;
    \\  float4 u = 0.0f;
    \\  for (uint s = 0u; s < nsplit; s++) {
    \\    const device float *pg = part + (size_t)(r * nsplit + s) * odim + k + lane;
    \\    g += float4(pg[0], pg[32], pg[64], pg[96]);
    \\    const device float *pu = pg + width;
    \\    u += float4(pu[0], pu[32], pu[64], pu[96]);
    \\  }
    \\  g = exl3_h128(g, lane);
    \\  u = exl3_h128(u, lane);
    \\  const float psc = 0.08838834764831845f;
    \\  mid_row_t m;
    \\  for (uint j = 0u; j < 4u; j++) {
    \\    const bfloat16_t gb = bfloat16_t(g[j] * psc * float(svh[k + lane + 32u * j]));
    \\    const bfloat16_t ub = bfloat16_t(u[j] * psc * float(svh[width + k + lane + 32u * j]));
    \\    const bfloat16_t act = gb * sig[as_type<ushort>(gb)];
    \\    m.v[j] = act * ub;
    \\  }
    \\  return m;
    \\}
    \\
;

/// The down kernel, preparing its input from the gate/up partials instead of reading it.
const DOWN_SOURCE: [:0]const u8 = kernels.metalReplaceFound(
    project_rows.ROWS_NAX_SOURCE,
    "const auto xr = x + (size_t)r * uint(IDIM) + k;",
    "const mid_row_t xr = mid_row(g1, g1svh, sigmoid, r, k, lane, uint(GNSPLIT), uint(GODIM), uint(GWIDTH));",
);

const SOURCE: [:0]const u8 = project_rows.ROWS_NAX_SOURCE;

fn partialOut(alloc: std.mem.Allocator, g: project_rows.RowsGeom, rows: c_int, out_dim: c_int) ![]const fuse.Out {
    const outs = try alloc.alloc(fuse.Out, 1);
    outs[0] = .{ .shape = try alloc.dupe(c_int, &.{ rows * g.nsplit, out_dim }), .dtype = .float32 };
    return outs;
}

/// Threadgroups across the output: one per 128-wide block (the NAX threadgroup is four tile pairs).
fn tiles(out_dim: c_int) c_int {
    return @divExact(out_dim, 128);
}

/// Gate/up partials `[rows * nsplit, 2 * width]` f32 on `x` [rows, hidden]. A caller may add `extra_x` threadgroups
/// along the grid's x axis, which `prefix` (statements ending in an `else`) handles ahead of the kernel's own.
pub fn gateUpPart(alloc: std.mem.Allocator, geo: Geo, rows: c_int, extra_x: c_int, extra_ints: []const fuse.IntArg, prefix: []const u8) !fuse.Part {
    const both = 2 * geo.width;
    const g = geo.gu;
    const body = try std.mem.concat(alloc, u8, &.{ prefix, "{\n", OUT_TYPE, SOURCE, "\n}\n" });
    const head = try std.mem.concat(alloc, u8, &.{ try codecHeader(geo.dec), project_rows.ROWS_NAX_FRAGS });
    return .{
        .launch = .{ .tgs = .{ tiles(both) + extra_x, g.nsplit, 1 }, .threads = g.tgrp * g.kph * 32, .ints = try ints(alloc, g, geo.hidden, both, geo.width, rows, geo.n_gu, extra_ints), .outs = try partialOut(alloc, g, rows, both) },
        .body = body,
        .ins = &project_rows.ROWS_INS,
        .outs = &.{"y"},
        .header = head,
    };
}

/// Down partials `[rows * nsplit, hidden]` f32 from the gate/up partials `g1` of `gateUpPart`.
pub fn downPart(alloc: std.mem.Allocator, geo: Geo, rows: c_int) !fuse.Part {
    const g = geo.dn;
    const body = try std.mem.concat(alloc, u8, &.{ OUT_TYPE, DOWN_SOURCE });
    const head = try std.mem.concat(alloc, u8, &.{ try codecHeader(geo.dec), project_rows.ROWS_NAX_FRAGS, MID_HEADER });
    const extra = [_]fuse.IntArg{ .{ .name = "GNSPLIT", .value = geo.gu.nsplit }, .{ .name = "GODIM", .value = 2 * geo.width }, .{ .name = "GWIDTH", .value = geo.width } };
    return .{
        .launch = .{ .tgs = .{ tiles(geo.hidden), g.nsplit, 1 }, .threads = g.tgrp * g.kph * 32, .ints = try ints(alloc, g, geo.width, geo.hidden, geo.hidden, rows, geo.n_dn, &extra), .outs = try partialOut(alloc, g, rows, geo.hidden) },
        .body = body,
        .ins = &.{ "x", "trellis", "suh", "svh", "g1", "g1svh", "sigmoid" },
        .outs = &.{"y"},
        .header = head,
    };
}

/// The pair launch's inputs of `gateUpPart`.
pub fn gateUpInputs(b: Banks, x: mlx.mlx_array, out: []mlx.mlx_array) []const mlx.mlx_array {
    out[0..4].* = .{ x, b.gate_up.trellis, b.gate_up.suh, b.gate_up.svh };
    return out[0..4];
}

/// The down launch's inputs of `downPart`: `partials` is the gate/up launch's output.
pub fn downInputs(b: Banks, x: mlx.mlx_array, partials: mlx.mlx_array, sigmoid: mlx.mlx_array, out: []mlx.mlx_array) []const mlx.mlx_array {
    out[0..7].* = .{ x, b.down.trellis, b.down.suh, b.down.svh, partials, b.gate_up.svh, sigmoid };
    return out[0..7];
}

/// The routed finish reduce with the shared down finish and the sigmoid-gated add: threadgroup `shv` holds the
/// shared expert's block as `rowsFinish` stored it, summed from the `SNSPLIT` partials by one idle simdgroup.
const REDUCE_SOURCE: [:0]const u8 = blk: {
    const gated = kernels.REDUCE_GATED_SHARED_SOURCE;
    const barrier = "threadgroup_barrier(mem_flags::mem_threadgroup);";
    const finish =
        \\if (sg == (uint(TOPK) > 1u ? 1u : 0u)) {
        \\  float4 sv = 0.0f;
        \\  for (uint si = 0u; si < uint(SNSPLIT); si++) {
        \\    const device float *pp = spart + (size_t)(row * uint(SNSPLIT) + si) * uint(ODIM) + base + lane;
        \\    sv += float4(pp[0], pp[32], pp[64], pp[96]);
        \\  }
        \\  sv = exl3_h128(sv, lane);
        \\  const float psc = 0.08838834764831845f;
        \\  shv[lane] = T(sv.x * psc * float(ssvh[base + lane]));
        \\  shv[lane + 32u] = T(sv.y * psc * float(ssvh[base + lane + 32u]));
        \\  shv[lane + 64u] = T(sv.z * psc * float(ssvh[base + lane + 64u]));
        \\  shv[lane + 96u] = T(sv.w * psc * float(ssvh[base + lane + 96u]));
        \\}
        \\
    ;
    const with_finish = kernels.metalReplaceFound(gated, barrier, finish ++ barrier);
    const r0 = kernels.metalReplaceFound(with_finish, "shared[yb + lane + 32u]", "shv[lane + 32u]");
    const r1 = kernels.metalReplaceFound(r0, "shared[yb + lane + 64u]", "shv[lane + 64u]");
    const r2 = kernels.metalReplaceFound(r1, "shared[yb + lane + 96u]", "shv[lane + 96u]");
    break :blk "threadgroup T shv[128];\n" ++ kernels.metalReplaceFound(r2, "shared[yb + lane]", "shv[lane]");
};

const ReduceKey = struct { out_dim: c_int, rows: c_int, topk: c_int, nsplit: c_int };

var reduce_cfgs: kernel_src.CfgCache(ReduceKey, 16) = .{};
var reduce_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// `y = sum_k score_k * routed_k + sigmoid(gate) * shared`, bf16 [rows, out_dim] (1-D at one row), with the shared
/// expert's down partials `spart` [rows * nsplit, out_dim] finished here through `ssvh`.
pub fn reduce(s: mlx.mlx_stream, inner: mlx.mlx_array, svh: mlx.mlx_array, slots: mlx.mlx_array, scores: mlx.mlx_array, out_dim: c_int, rows: c_int, topk: c_int, spart: mlx.mlx_array, ssvh: mlx.mlx_array, nsplit: c_int, gate: mlx.mlx_array) !mlx.mlx_array {
    if (topk < 1 or topk > kernels.REDUCE_MAX_TOPK) return error.Exl3TopkUnsupported;
    const key = ReduceKey{ .out_dim = out_dim, .rows = rows, .topk = topk, .nsplit = nsplit };
    const cfg = reduce_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const sh = if (rows == 1) [_]c_int{ out_dim, 0 } else [_]c_int{ rows, out_dim };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &sh, if (rows == 1) 1 else 2, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, 32 * topk * @divExact(out_dim, 128), rows, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32 * topk, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "T", .bfloat16));
        inline for (.{ "ODIM", "TOPK", "SNSPLIT" }, .{ out_dim, topk, nsplit }) |name, v|
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, name, v));
        reduce_cfgs.put(key, c);
        break :blk c;
    };
    const ins = [_][*:0]const u8{ "inner", "svh", "slots", "sc", "spart", "ssvh", "gate" };
    const kernel = try kernels.getNamedKernel(&reduce_kernel, "sushi_exl3_down_reduce_shared_trellis", &ins, &.{"y"}, REDUCE_SOURCE, project_rows.ROWS_HELPERS);
    const ov = try kernels.applyOuts(s, kernel, &.{ inner, svh, slots, scores, spart, ssvh, gate }, cfg, 1);
    defer _ = mlx.mlx_vector_array_free(ov);
    var a = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(a);
    try mlx.check(mlx.mlx_vector_array_get(&a, ov, 0));
    return a;
}
