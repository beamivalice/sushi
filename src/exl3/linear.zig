//! Dense trellis linears: a trunk weight stored as a one-bank (or, for a grouped projection, G-bank) EXL3/STQ
//! projection and served by the expert kernels. Up to `PROJECT_ROWS_MAX` rows ride the multi-row kernel (two
//! dispatches, each weight decoded once for every row); wider chunks the sorted GEMM over a window table built on the
//! host from the known row order, so no GPU drain.
const std = @import("std");
const mlx = @import("mlx_host").mlx;
const host_side = @import("host_side.zig");
const exl3 = @import("expert_exl3.zig");
const kernels = @import("expert_exl3_kernels.zig");
const project_rows = @import("project_rows.zig");
const trunk_gemm = @import("trunk_gemm.zig");
const root = @import("root.zig");

/// Tests and the trunk bench: `rows` is the served arm up to `PROJECT_ROWS_MAX`, `gemm` the sorted GEMM at any width,
/// `gemv` the indexed GEMV (one row per bank).
pub const Arm = enum { rows, gemm, gemv };

/// Per-call launch options; the default is a plain serial call.
pub const Opts = struct {
    rows: project_rows.Mode = .{},
    /// The caller's transient bill holds `trunk_gemm.DECODED_SCRATCH_BYTES`; others keep the sorted GEMM.
    decoded_billed: bool = false,
};

/// `x` [rows, in] through the one-bank projection `p` (trellis [1, in/16, out/16, n]): [rows, out] in x's dtype.
pub fn linear(s: mlx.mlx_stream, x: mlx.mlx_array, p: root.Proj, dec: exl3.Decode) !mlx.mlx_array {
    return linearWith(s, x, p, dec, .{});
}

pub fn linearWith(s: mlx.mlx_stream, x: mlx.mlx_array, p: root.Proj, dec: exl3.Decode, opts: Opts) !mlx.mlx_array {
    const rows: usize = @intCast(mlx.getShape(x)[0]);
    return run(s, x, p, dec, 1, rows, opts);
}

/// `x` [G, rows, in] through the G-bank projection `p` (bank g serves x[g]): [G, rows, out] in x's dtype.
pub fn linearGrouped(s: mlx.mlx_stream, x: mlx.mlx_array, p: root.Proj, dec: exl3.Decode) !mlx.mlx_array {
    return linearGroupedWith(s, x, p, dec, .{});
}

pub fn linearGroupedWith(s: mlx.mlx_stream, x: mlx.mlx_array, p: root.Proj, dec: exl3.Decode, opts: Opts) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    if (sh.len != 3) return error.BadExl3Shape;
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, x, &[_]c_int{ sh[0] * sh[1], sh[2] }, 2, s));
    const y = try run(s, flat, p, dec, @intCast(sh[0]), @intCast(sh[1]), opts);
    defer _ = mlx.mlx_array_free(y);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, y, &[_]c_int{ sh[0], sh[1], mlx.getShape(y)[1] }, 3, s));
    return out;
}

/// `x` `[G, rows, in]` through the G-bank `a`, then that result as `[rows, G * mid]` through the one-bank `b`:
/// `linearGroupedWith` then `linearWith`, bit for bit, in one dispatch fewer, because `b`'s launch sums `a`'s k-split
/// partials where it reads its input. Null, with nothing made, when the two are not launches this serves: a row count
/// past one multi-row launch, an `a` whose launch has no finish, a forced arm. With `expand` the result is `b`'s rows
/// expanded into it (`linearExpanded`), or null where `b`'s finish cannot.
pub fn linearPair(s: mlx.mlx_stream, x: mlx.mlx_array, a: root.Proj, b: root.Proj, dec_a: exl3.Decode, dec_b: exl3.Decode, opts: Opts, expand: ?project_rows.Expand) !?mlx.mlx_array {
    if (force_arm != null) return null;
    const xs = mlx.getShape(x);
    const ta = mlx.getShape(a.trellis);
    const tb = mlx.getShape(b.trellis);
    if (xs.len != 3 or ta.len != 4 or tb.len != 4 or ta[0] != xs[0] or tb[0] != 1) return error.BadExl3Shape;
    const groups = xs[0];
    const rows = xs[1];
    const in = xs[2];
    const mid = ta[2] * 16;
    const out = tb[2] * 16;
    const dt = mlx.mlx_array_dtype(x);
    if (ta[1] * 16 != in or tb[1] * 16 != groups * mid or @rem(in, 128) != 0 or @rem(mid, 128) != 0 or @rem(out, 128) != 0) return error.BadExl3Shape;
    if (dt != .bfloat16 and dt != .float16 and dt != .float32) return error.BadExl3Shape;
    if (rows < 1 or rows > project_rows.rowsMax(opts.rows)) return null;
    const ga = project_rows.rowsGeometry(in, mid, rows, groups, opts.rows);
    const gb = project_rows.rowsGeometry(groups * mid, out, rows, 1, opts.rows);
    for ([_]project_rows.RowsGeom{ ga, gb }) |g| if (rows > (if (g.nax) project_rows.NAX_ROWS_MAX else project_rows.PROJECT_ROWS_MAX)) return null;
    if (!project_rows.hasFinish(ga)) return null;
    const lb: project_rows.RowsLaunch = .{ .in_dim = groups * mid, .out_dim = out, .rows = rows, .groups = 1, .n = (try kernels.packedRate(tb[3])).n, .out_dt = dt };
    const fitted = if (expand) |e| project_rows.fitExpand(lb, gb, e) orelse return null else null;
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, x, &[_]c_int{ groups * rows, in }, 2, s));
    try dec_a.requireTable();
    kernels.setDecodeParams(dec_a);
    const part = try project_rows.launchRowsParts(s, flat, a.trellis, a.suh, a.svh, .{ .in_dim = in, .out_dim = mid, .rows = rows, .groups = groups, .n = (try kernels.packedRate(ta[3])).n, .out_dt = dt }, ga);
    defer _ = mlx.mlx_array_free(part);
    try dec_b.requireTable();
    kernels.setDecodeParams(dec_b);
    return try project_rows.launchRowsFrom(s, part, a.svh, .{ .xsplit = ga.nsplit, .xodim = mid, .xdt = dt }, b.trellis, b.suh, b.svh, lb, gb, fitted);
}

/// `linearGroupedWith` whose result leaves as `[rows, G * out]`, bank g's columns from `g * out`: what a transpose and
/// a reshape of it give, without their copy, because the launch's finish writes the rows that way. Null, with nothing
/// made, where no such finish serves: a row count past one multi-row launch, a launch that folds its finish, a forced arm.
pub fn linearGroupedRows(s: mlx.mlx_stream, x: mlx.mlx_array, p: root.Proj, dec: exl3.Decode, opts: Opts) !?mlx.mlx_array {
    if (force_arm != null) return null;
    const xs = mlx.getShape(x);
    const ts = mlx.getShape(p.trellis);
    if (xs.len != 3 or ts.len != 4 or ts[0] != xs[0] or ts[1] * 16 != xs[2]) return error.BadExl3Shape;
    const groups = xs[0];
    const rows = xs[1];
    const out = ts[2] * 16;
    if (@rem(xs[2], 128) != 0 or @rem(out, 128) != 0) return null;
    if (rows < 1 or rows > project_rows.rowsMax(opts.rows)) return null;
    const g = project_rows.rowsGeometry(xs[2], out, rows, groups, opts.rows);
    if (rows > (if (g.nax) project_rows.NAX_ROWS_MAX else project_rows.PROJECT_ROWS_MAX) or !project_rows.hasFinish(g)) return null;
    const l: project_rows.RowsLaunch = .{ .in_dim = xs[2], .out_dim = out, .rows = rows, .groups = groups, .n = (try kernels.packedRate(ts[3])).n, .out_dt = mlx.mlx_array_dtype(x) };
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, x, &[_]c_int{ groups * rows, xs[2] }, 2, s));
    try dec.requireTable();
    kernels.setDecodeParams(dec);
    const part = try project_rows.launchRowsParts(s, flat, p.trellis, p.suh, p.svh, l, g);
    defer _ = mlx.mlx_array_free(part);
    return try project_rows.rowsFinishRowMajor(s, part, p.svh, l, g.nsplit);
}

/// `linearWith` whose rows leave expanded into `expand`'s streams `[rows, hc, out]`: stream i of a row y is
/// `fma(post[i], y, sum over j of fma(comb[j][i], x[j], acc))` in x's dtype, y rounded to it first. The launch's finish
/// does it, so the expand costs no dispatch. Null, with nothing made, where no such finish serves: a row count past
/// one multi-row launch, a launch that folds its finish, a forced arm, streams of another dtype or shape.
pub fn linearExpanded(s: mlx.mlx_stream, x: mlx.mlx_array, p: root.Proj, dec: exl3.Decode, opts: Opts, expand: project_rows.Expand) !?mlx.mlx_array {
    if (force_arm != null) return null;
    const xs = mlx.getShape(x);
    const ts = mlx.getShape(p.trellis);
    if (xs.len != 2 or ts.len != 4 or ts[0] != 1 or ts[1] * 16 != xs[1]) return error.BadExl3Shape;
    const rows = xs[0];
    const out = ts[2] * 16;
    const dt = mlx.mlx_array_dtype(x);
    if (@rem(xs[1], 128) != 0 or @rem(out, 128) != 0) return null;
    if (rows < 1 or rows > project_rows.rowsMax(opts.rows)) return null;
    const g = project_rows.rowsGeometry(xs[1], out, rows, 1, opts.rows);
    if (rows > (if (g.nax) project_rows.NAX_ROWS_MAX else project_rows.PROJECT_ROWS_MAX)) return null;
    const l: project_rows.RowsLaunch = .{ .in_dim = xs[1], .out_dim = out, .rows = rows, .groups = 1, .n = (try kernels.packedRate(ts[3])).n, .out_dt = dt };
    const fitted = project_rows.fitExpand(l, g, expand) orelse return null;
    try dec.requireTable();
    kernels.setDecodeParams(dec);
    const part = try project_rows.launchRowsParts(s, x, p.trellis, p.suh, p.svh, l, g);
    defer _ = mlx.mlx_array_free(part);
    return try project_rows.rowsFinishExpand(s, part, p.svh, l, g.nsplit, fitted);
}

/// The most linears one shared-input launch serves.
pub const SHARED_MAX: usize = 4;

/// Linears that read the same rows, stacked along the output into one bank: trellis `[1, in/16, out/16, n]`, `suh`
/// `[S, in]` (one row per linear), `svh` `[1, out]`; linear s owns columns `cols[s]..cols[s + 1]`, each a multiple of 128.
pub const Shared = struct { p: root.Proj, cols: [SHARED_MAX + 1]c_int, count: usize, geo: ?*const Geo = null };

/// What a one-bank trunk linear's decode-width launches need from its weight, read once when the weight is bound: the
/// shape and rate, and per row count the launch geometry and its pinned configs (for `bf16` inputs, the trunk's).
/// A launch that finds no entry runs as `linearWith` does, reading the shapes off its operands.
pub const Geo = struct {
    in: c_int,
    out: c_int,
    n: u32,
    rows: [@intCast(project_rows.NAX_ROWS_MAX)]?Row = @splat(null),
    /// What `init` was asked, which names the geo in `get`.
    key: Key = .{},

    const Key = struct { shape: [4]c_int = @splat(0), dtype: mlx.mlx_dtype = .uint16, bounds: ?[3]c_int = null, nax: bool = false, exact: bool = true };

    pub const Row = struct { g: project_rows.RowsGeom, pin: project_rows.PinnedCfg };

    /// `trellis` is `[1, in/16, out/16, n]`; `bounds` are the segment bounds of a stacked bank. `mode.exact` picks the
    /// arm a row count runs on, so the geo serves only launches of that `exact` (`linearGeo` declines the others). A
    /// weight the multi-row kernels do not take gets an empty geo.
    pub fn init(trellis: mlx.mlx_array, bounds: ?[3]c_int, mode: project_rows.Mode) !Geo {
        var geo: Geo = .{ .in = 0, .out = 0, .n = 0 };
        const ts = mlx.getShape(trellis);
        if (ts.len == 4) geo.key = .{ .shape = ts[0..4].*, .dtype = mlx.mlx_array_dtype(trellis), .bounds = bounds, .nax = kernels.gemmNaxOn(), .exact = mode.exact };
        if (ts.len != 4 or ts[0] != 1 or mlx.mlx_array_dtype(trellis) != .uint16) return geo;
        const in = ts[1] * 16;
        const out = ts[2] * 16;
        if (@rem(in, 128) != 0 or @rem(out, 128) != 0) return geo;
        const rate = kernels.packedRate(ts[3]) catch return geo;
        geo.in = in;
        geo.out = out;
        geo.n = rate.n;
        for (&geo.rows, 1..) |*slot, rows| {
            const g = project_rows.rowsGeometry(in, out, @intCast(rows), 1, .{ .exact = mode.exact });
            if (rows > (if (g.nax) project_rows.NAX_ROWS_MAX else project_rows.PROJECT_ROWS_MAX)) continue;
            slot.* = .{ .g = g, .pin = try project_rows.pinCfg(.{ .in_dim = in, .out_dim = out, .rows = @intCast(rows), .groups = 1, .n = rate.n, .out_dt = .bfloat16 }, g, bounds) };
        }
        return geo;
    }

    /// The geo of a bank of `trellis`'s shape, made on first ask and kept for the process beside the configs it pins
    /// (banks of one shape share it), so a `Linear` holds a pointer. Not locked: ask on the MLX-owning thread.
    pub fn get(trellis: mlx.mlx_array, bounds: ?[3]c_int, mode: project_rows.Mode) !*const Geo {
        const ts = mlx.getShape(trellis);
        if (ts.len != 4) return &empty_geo;
        const key: Key = .{ .shape = ts[0..4].*, .dtype = mlx.mlx_array_dtype(trellis), .bounds = bounds, .nax = kernels.gemmNaxOn(), .exact = mode.exact };
        for (geos.items) |g| if (std.meta.eql(g.key, key)) return g;
        const made = try std.heap.smp_allocator.create(Geo);
        errdefer std.heap.smp_allocator.destroy(made);
        made.* = try init(trellis, bounds, mode);
        try geos.append(std.heap.smp_allocator, made);
        return made;
    }

    /// The launch of `rows` rows, or null when it is not one the multi-row kernels serve.
    pub fn row(self: *const Geo, rows: c_int) ?Row {
        if (rows < 1 or rows > project_rows.NAX_ROWS_MAX) return null;
        return self.rows[@intCast(rows - 1)];
    }

    pub fn launch(self: *const Geo, rows: c_int) project_rows.RowsLaunch {
        return .{ .in_dim = self.in, .out_dim = self.out, .rows = rows, .groups = 1, .n = self.n, .out_dt = .bfloat16 };
    }
};

var geos: std.ArrayList(*const Geo) = .empty;
const empty_geo: Geo = .{ .in = 0, .out = 0, .n = 0 };

/// `x` [rows, in] in bf16 through the one-bank projection `p` whose `geo` holds the launch, or null when it does not:
/// the rows past the multi-row arm, a `force_arm` seam, a weight `geo` is empty for, a NAX launch after the NAX latch
/// set (the shape-reading path then picks the SIMD arm). [rows, out] otherwise, as `linearWith`.
pub fn linearGeo(s: mlx.mlx_stream, x: mlx.mlx_array, rows: c_int, p: root.Proj, geo: *const Geo, dec: exl3.Decode, opts: Opts) !?mlx.mlx_array {
    if (force_arm != null or opts.rows.exact != geo.key.exact or rows > project_rows.rowsMax(opts.rows)) return null;
    const r = geo.row(rows) orelse return null;
    if (r.g.nax and !kernels.gemmNaxOn()) return null;
    try dec.requireTable();
    kernels.setDecodeParams(dec);
    return try project_rows.launchRows(s, x, p.trellis, p.suh, p.svh, geo.launch(rows), r.g, try project_rows.rowsKernel(r.g), null, r.pin);
}

/// `x` [rows, in] (rows <= `project_rows.rowsMax(opts.rows)`) through every linear of `sh` in one multi-row launch: `outs[s]` is
/// linear s's [rows, width] in x's dtype (the caller frees each). A row holds the bar of its own linear's launch; the
/// sum order may differ from it, since the launch splits k for the stacked width.
pub fn linearShared(s: mlx.mlx_stream, x: mlx.mlx_array, sh: Shared, dec: exl3.Decode, outs: []mlx.mlx_array) !void {
    return linearSharedWith(s, x, sh, dec, outs, .{});
}

pub fn linearSharedWith(s: mlx.mlx_stream, x: mlx.mlx_array, sh: Shared, dec: exl3.Decode, outs: []mlx.mlx_array, opts: Opts) !void {
    try dec.requireTable();
    if (outs.len != sh.count) return error.BadExl3Shape;
    const rows: c_int = mlx.getShape(x)[0];
    kernels.setDecodeParams(dec);
    const total = sh.cols[sh.count];
    var bounds: [3]c_int = @splat(total);
    for (1..sh.count) |i| bounds[i - 1] = sh.cols[i];
    const flat = try project_rows.projectRowsSegmented(s, x, sh.p.trellis, sh.p.suh, sh.p.svh, bounds, opts.rows);
    defer _ = mlx.mlx_array_free(flat);
    try splitShared(s, flat, rows, sh, outs);
}

/// `linearSharedWith` for bf16 `x` [rows, in] whose `sh.geo` holds the launch; false, with nothing made, when it does not.
pub fn linearSharedGeo(s: mlx.mlx_stream, x: mlx.mlx_array, rows: c_int, sh: Shared, dec: exl3.Decode, outs: []mlx.mlx_array, opts: Opts) !bool {
    const geo = sh.geo orelse return false;
    if (opts.rows.exact != geo.key.exact or rows > project_rows.rowsMax(opts.rows)) return false;
    const r = geo.row(rows) orelse return false;
    if (r.g.nax and !kernels.gemmNaxOn()) return false;
    if (outs.len != sh.count) return error.BadExl3Shape;
    try dec.requireTable();
    kernels.setDecodeParams(dec);
    var bounds: [3]c_int = @splat(sh.cols[sh.count]);
    for (1..sh.count) |i| bounds[i - 1] = sh.cols[i];
    const y = try project_rows.launchRows(s, x, sh.p.trellis, sh.p.suh, sh.p.svh, geo.launch(rows), r.g, try project_rows.rowsKernel(r.g), bounds, r.pin);
    defer _ = mlx.mlx_array_free(y);
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, y, &[_]c_int{rows * geo.out}, 1, s));
    try splitShared(s, flat, rows, sh, outs);
    return true;
}

/// `linearSharedWith` whose first linear's output leaves as `mlx_fast_rms_norm(it, norm.w, eps)`: the same bytes, the
/// finish and the norm in one dispatch. False, with nothing made, when the launch is not one this serves.
pub fn linearSharedNormed(s: mlx.mlx_stream, x: mlx.mlx_array, sh: Shared, dec: exl3.Decode, outs: []mlx.mlx_array, opts: Opts, norm: project_rows.Norm) !bool {
    if (outs.len != sh.count) return error.BadExl3Shape;
    const xs = mlx.getShape(x);
    const ts = mlx.getShape(sh.p.trellis);
    if (xs.len != 2 or ts.len != 4 or ts[0] != 1 or ts[1] * 16 != xs[1]) return error.BadExl3Shape;
    const rows = xs[0];
    const out = ts[2] * 16;
    if (force_arm != null or rows < 1 or rows > project_rows.rowsMax(opts.rows)) return false;
    const g = project_rows.rowsGeometry(xs[1], out, rows, 1, opts.rows);
    if (rows > (if (g.nax) project_rows.NAX_ROWS_MAX else project_rows.PROJECT_ROWS_MAX)) return false;
    var bounds: [3]c_int = @splat(sh.cols[sh.count]);
    for (1..sh.count) |i| bounds[i - 1] = sh.cols[i];
    const l: project_rows.RowsLaunch = .{ .in_dim = xs[1], .out_dim = out, .rows = rows, .groups = 1, .n = (try kernels.packedRate(ts[3])).n, .out_dt = mlx.mlx_array_dtype(x) };
    if (!project_rows.normedFits(l, g, bounds, norm)) return false;
    try dec.requireTable();
    kernels.setDecodeParams(dec);
    const pin: ?project_rows.PinnedCfg = if (sh.geo) |geo| (if (l.out_dt == .bfloat16 and opts.rows.exact == geo.key.exact) (if (geo.row(rows)) |r| (if (std.meta.eql(r.g, g)) r.pin else null) else null) else null) else null;
    const y = try project_rows.launchRowsNormed(s, x, sh.p.trellis, sh.p.suh, sh.p.svh, l, g, try project_rows.rowsKernel(g), bounds, pin, norm);
    defer _ = mlx.mlx_array_free(y);
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, y, &[_]c_int{rows * out}, 1, s));
    try splitShared(s, flat, rows, sh, outs);
    return true;
}

fn splitShared(s: mlx.mlx_stream, flat: mlx.mlx_array, rows: c_int, sh: Shared, outs: []mlx.mlx_array) !void {
    var made: usize = 0;
    errdefer for (outs[0..made]) |o| {
        _ = mlx.mlx_array_free(o);
    };
    for (outs, 0..) |*o, i| {
        const c0 = sh.cols[i];
        const c1 = sh.cols[i + 1];
        var part = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(part);
        try mlx.check(mlx.mlx_slice(&part, flat, &[_]c_int{c0 * rows}, 1, &[_]c_int{c1 * rows}, 1, &[_]c_int{1}, 1, s));
        o.* = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(o, part, &[_]c_int{ rows, c1 - c0 }, 2, s));
        made += 1;
    }
}

pub var force_arm: ?Arm = null;

/// `x` [groups * rows, in], bank g's rows contiguous.
fn run(s: mlx.mlx_stream, x: mlx.mlx_array, p: root.Proj, dec: exl3.Decode, groups: usize, rows: usize, opts: Opts) !mlx.mlx_array {
    try dec.requireTable();
    const tsh = mlx.getShape(p.trellis);
    if (tsh.len != 4 or tsh[0] != groups) return error.BadExl3Shape;
    kernels.setDecodeParams(dec);
    if ((force_arm orelse if (rows <= project_rows.rowsMax(opts.rows)) Arm.rows else Arm.gemm) == .rows)
        return project_rows.projectRows(s, x, p.trellis, p.suh, p.svh, opts.rows);
    const inputs = try gemmInputs(groups, rows, kernels.gemmWindowRows());
    const slots = inputs.slots;
    const prepared = try kernels.prepareIndexed(s, x, p.suh, slots);
    defer _ = mlx.mlx_array_free(prepared);
    const inner = if ((force_arm orelse .gemm) == .gemm) blk: {
        const out_dim: c_int = mlx.getShape(p.trellis)[2] * 16;
        if (groups == 1 and trunk_gemm.decodedServes(opts.decoded_billed, @intCast(rows), mlx.getShape(x)[1], out_dim))
            break :blk try trunk_gemm.innerDense(s, prepared, p.trellis);
        break :blk try kernels.innerGemmSortedTable(s, prepared, p.trellis, slots, kernels.gemmWindowRows(), true, inputs.tab);
    } else try kernels.indexedGemvCoopF16(s, prepared, p.trellis, slots);
    defer _ = mlx.mlx_array_free(inner);
    return kernels.finishIndexedAs(s, inner, p.svh, slots, mlx.mlx_array_dtype(x));
}

/// What a GEMM-arm call of `groups` banks of `rows` rows needs: each row's bank id, and the window table over them.
pub const GemmInputs = struct { slots: mlx.mlx_array, tab: kernels.WindowTable };

const GemmKey = struct { groups: usize, rows: usize, win: c_int };
const gemm_cache_len = 8;
const GemmRing = struct {
    entries: [gemm_cache_len]?struct { key: GemmKey, inputs: GemmInputs } = @splat(null),
    next: usize = 0,
};
/// One ring for each side of the host (`host_side.making`): an array belongs to the side that made it, which alone
/// uses it and frees it.
var gemm_cache: [2]GemmRing = .{ .{}, .{} };

/// Frees the inputs side `side` cached (`host_side.of`): a host whose engine no model holds any more gives them back.
pub fn dropGemmInputs(side: u1) void {
    const ring = &gemm_cache[side];
    for (&ring.entries) |*e| {
        if (e.*) |old| for ([_]mlx.mlx_array{ old.inputs.slots, old.inputs.tab.starts, old.inputs.tab.nlives }) |a| {
            _ = mlx.mlx_array_free(a);
        };
        e.* = null;
    }
    ring.next = 0;
}

/// The inputs depend on `(groups, rows, win)` alone, and a prefill chunk's trellis linears all share them, so they are
/// built and uploaded once. The handles are the cache's: valid until a later call evicts them, never freed by the caller.
pub fn gemmInputs(groups: usize, rows: usize, win: c_int) !GemmInputs {
    const key = GemmKey{ .groups = groups, .rows = rows, .win = win };
    const ring = &gemm_cache[host_side.making()];
    for (ring.entries) |e| if (e) |hit| if (std.meta.eql(hit.key, key)) return hit.inputs;
    const alloc = std.heap.page_allocator;
    const ids = try alloc.alloc(u32, groups * rows);
    defer alloc.free(ids);
    for (ids, 0..) |*v, i| v.* = @intCast(i / rows);
    const slots = try host_side.made(mlx.mlx_array_new_data(ids.ptr, &[_]c_int{@intCast(ids.len)}, 1, .uint32));
    errdefer _ = mlx.mlx_array_free(slots);
    const tab = try windowTable(groups, rows, win);
    const at = ring.next;
    if (ring.entries[at]) |old| for ([_]mlx.mlx_array{ old.inputs.slots, old.inputs.tab.starts, old.inputs.tab.nlives }) |a| {
        _ = mlx.mlx_array_free(a);
    };
    ring.entries[at] = .{ .key = key, .inputs = .{ .slots = slots, .tab = tab } };
    ring.next = (at + 1) % gemm_cache_len;
    return ring.entries[at].?.inputs;
}

/// Windows of at most `win` rows that never straddle two banks.
pub fn windowTable(groups: usize, rows: usize, win: c_int) !kernels.WindowTable {
    const w: usize = @intCast(win);
    const per = (rows + w - 1) / w;
    const alloc = std.heap.page_allocator;
    const starts = try alloc.alloc(u32, groups * per);
    defer alloc.free(starts);
    const lives = try alloc.alloc(u32, groups * per);
    defer alloc.free(lives);
    for (0..groups) |g| for (0..per) |i| {
        starts[g * per + i] = @intCast(g * rows + i * w);
        lives[g * per + i] = @intCast(@min(w, rows - i * w));
    };
    const n: c_int = @intCast(groups * per);
    const st = try host_side.made(mlx.mlx_array_new_data(starts.ptr, &[_]c_int{n}, 1, .uint32));
    errdefer _ = mlx.mlx_array_free(st);
    return .{ .starts = st, .nlives = try host_side.made(mlx.mlx_array_new_data(lives.ptr, &[_]c_int{n}, 1, .uint32)), .nwin = n };
}

/// One bank's weight as the kernels apply it, `W^T = diag(suh) H M H diag(svh)` in f32 `[in, out]` (M the decoded
/// trellis): the dense twin a test runs beside the kernels.
pub fn decodeDense(alloc: std.mem.Allocator, words: []const u16, suh: []const u16, svh: []const u16, in: usize, out: usize, n: u32, dec: exl3.Decode, dst: []f32) !void {
    if (dst.len != in * out or words.len != (in / 16) * (out / 16) * n) return error.BadExl3Shape;
    var tile: [exl3.TILE_VALUES]u16 = undefined;
    const ot = out / 16;
    for (0..in / 16) |tk| for (0..ot) |tn| {
        exl3.decodeTile(words[(tk * ot + tn) * n ..][0..n], .{ .n = n }, dec, &tile);
        for (0..16) |r| for (0..16) |c| {
            dst[(tk * 16 + r) * out + tn * 16 + c] = exl3.f16BitsToF32(tile[r * 16 + c]);
        };
    };
    const col = try alloc.alloc(f32, exl3.HAD_DIM);
    defer alloc.free(col);
    var b: usize = 0;
    while (b < in) : (b += exl3.HAD_DIM) for (0..out) |j| {
        for (col, 0..) |*v, i| v.* = dst[(b + i) * out + j];
        exl3.hadamard128(col[0..exl3.HAD_DIM]);
        for (col, 0..) |v, i| dst[(b + i) * out + j] = v;
    };
    for (0..in) |i| {
        const row = dst[i * out ..][0..out];
        var o: usize = 0;
        while (o < out) : (o += exl3.HAD_DIM) exl3.hadamard128(row[o..][0..exl3.HAD_DIM]);
        const su = exl3.f16BitsToF32(suh[i]);
        for (row, svh) |*v, sv| v.* *= su * exl3.f16BitsToF32(sv);
    }
}

// ---- Tests ----

const Host = struct {
    words: []u16,
    suh: []u16,
    svh: []u16,
    p: root.Proj,

    fn init(alloc: std.mem.Allocator, rnd: std.Random, groups: usize, in: usize, out: usize, n: u32) !Host {
        const words = try alloc.alloc(u16, groups * (in / 16) * (out / 16) * n);
        for (words) |*v| v.* = @truncate(rnd.int(u32));
        const suh = try alloc.alloc(u16, groups * in);
        for (suh) |*v| v.* = exl3.f32ToF16Bits((if (rnd.boolean()) @as(f32, 1) else -1) * (0.002 + rnd.float(f32) * 0.02));
        const svh = try alloc.alloc(u16, groups * out);
        for (svh) |*v| v.* = exl3.f32ToF16Bits((if (rnd.boolean()) @as(f32, 1) else -1) * (0.5 + rnd.float(f32)));
        const g: c_int = @intCast(groups);
        return .{ .words = words, .suh = suh, .svh = svh, .p = .{
            .trellis = mlx.mlx_array_new_data(words.ptr, &[_]c_int{ g, @intCast(in / 16), @intCast(out / 16), @intCast(n) }, 4, .uint16),
            .suh = mlx.mlx_array_new_data(suh.ptr, &[_]c_int{ g, @intCast(in) }, 2, .float16),
            .svh = mlx.mlx_array_new_data(svh.ptr, &[_]c_int{ g, @intCast(out) }, 2, .float16),
        } };
    }

    fn deinit(self: *Host) void {
        for ([_]mlx.mlx_array{ self.p.trellis, self.p.suh, self.p.svh }) |a| _ = mlx.mlx_array_free(a);
    }

    /// Bank g's projection of `x` in f32 from the host tile decode.
    fn project(self: *const Host, alloc: std.mem.Allocator, g: usize, x: []const f32, in: usize, out: usize, n: u32, dec: exl3.Decode, dst: []f32) !void {
        const sc = try alloc.alloc(f32, in);
        defer alloc.free(sc);
        for (x, self.suh[g * in ..][0..in], sc) |v, sb, *o| o.* = v * exl3.f16BitsToF32(sb);
        var b: usize = 0;
        while (b < in) : (b += exl3.HAD_DIM) exl3.hadamard128(sc[b..][0..exl3.HAD_DIM]);
        @memset(dst, 0);
        var tile: [exl3.TILE_VALUES]u16 = undefined;
        const ot = out / 16;
        const tr = self.words[g * (in / 16) * ot * n ..];
        for (0..in / 16) |tk| for (0..ot) |tn| {
            exl3.decodeTile(tr[(tk * ot + tn) * n ..][0..n], .{ .n = n }, dec, &tile);
            for (0..16) |r| for (0..16) |c| {
                dst[tn * 16 + c] += sc[tk * 16 + r] * exl3.f16BitsToF32(tile[r * 16 + c]);
            };
        };
        var o: usize = 0;
        while (o < out) : (o += exl3.HAD_DIM) exl3.hadamard128(dst[o..][0..exl3.HAD_DIM]);
        for (dst, self.svh[g * out ..][0..out]) |*v, sb| v.* *= exl3.f16BitsToF32(sb);
    }
};

test "exl3 linear: the dense twin of a bank multiplies as the host projection does" {
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x7a1);
    const rnd = prng.random();
    const in: usize = 256;
    const out: usize = 384;
    const dec = exl3.Decode.tabled(.spec1, .w15);
    var h = try Host.init(alloc, rnd, 1, in, out, 48);
    defer h.deinit();
    const w = try alloc.alloc(f32, in * out);
    try decodeDense(alloc, h.words, h.suh, h.svh, in, out, 48, dec, w);
    const x = try alloc.alloc(f32, in);
    for (x) |*v| v.* = rnd.float(f32) * 2 - 1;
    const want = try alloc.alloc(f32, out);
    try h.project(alloc, 0, x, in, out, 48, dec, want);
    const got = try alloc.alloc(f32, out);
    @memset(got, 0);
    for (0..in) |i| for (0..out) |j| {
        got[j] += x[i] * w[i * out + j];
    };
    try t.expect(relRms(got, want) < 1e-5);
}

fn relRms(got: []const f32, want: []const f32) f64 {
    var ss: f64 = 0;
    var ref: f64 = 0;
    for (got, want) |a, b| {
        ss += (@as(f64, a) - b) * (@as(f64, a) - b);
        ref += @as(f64, b) * b;
    }
    return @sqrt(ss / @max(ref, 1e-30));
}

fn hostF32(s: mlx.mlx_stream, a: mlx.mlx_array, out: *mlx.mlx_array) ![]const f32 {
    try mlx.check(mlx.mlx_astype(out, a, .float32, s));
    try mlx.check(mlx.mlx_array_eval(out.*));
    return (mlx.mlx_array_data_float32(out.*) orelse return error.Unreadable)[0..mlx.mlx_array_size(out.*)];
}

/// `v` rounded to bf16 (nearest even), as an activation of the served dtype carries it.
fn bf16Bits(v: f32) u16 {
    const b: u32 = @bitCast(v);
    return @intCast((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}

fn bf16ToF32(b: u16) f32 {
    return @bitCast(@as(u32, b) << 16);
}

/// Worst measured over the cases below is 4.1e-4 (f16 stores on the prepared input and the output, f32 sums), past the
/// output dtype's own rounding.
const LINEAR_BAR: f64 = 1.5e-3;

fn roundTo(dt: mlx.mlx_dtype, v: f32) f32 {
    return switch (dt) {
        .float16 => exl3.f16BitsToF32(exl3.f32ToF16Bits(v)),
        .bfloat16 => bf16ToF32(bf16Bits(v)),
        else => v,
    };
}

test "exl3 linear: one-bank and grouped trellis linears hold the f32-truth bar on every arm, every STQ codebook and activation dtype" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    defer kernels.setDecodeParams(.mul1);
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x11ea);
    const rnd = prng.random();
    const cases = [_]struct { dec: exl3.Decode, n: u32 }{
        .{ .dec = exl3.Decode.tabled(.spec1, .w15), .n = 64 },
        .{ .dec = exl3.Decode.tabled(.spec1, .w15), .n = 96 },
        .{ .dec = .{ .codebook = .pair1, .window = .w16 }, .n = 32 },
        .{ .dec = .{ .codebook = .mcg, .window = .w16 }, .n = 48 },
    };
    const in: usize = 384;
    const out: usize = 256;
    var failed: usize = 0;
    for (cases) |cs| for ([_]usize{ 1, 2, 8 }) |groups| for ([_]usize{ 1, 3, 8, 70 }) |rows| for ([_]Arm{ .rows, .gemm, .gemv }) |arm| for ([_]mlx.mlx_dtype{ .float16, .bfloat16, .float32 }) |dt| {
        if ((arm == .gemv and rows > 3) or (arm == .rows and rows > project_rows.PROJECT_ROWS_MAX)) continue;
        if (groups == 8 and rows > 8) continue;
        force_arm = arm;
        defer force_arm = null;
        var h = try Host.init(alloc, rnd, groups, in, out, cs.n);
        defer h.deinit();
        const xs = try alloc.alloc(f32, groups * rows * in);
        for (xs) |*v| v.* = (rnd.float(f32) * 2 - 1) * 30;
        const xb = try alloc.alloc(u16, xs.len);
        if (dt != .float32) for (xb, xs) |*b, *v| {
            b.* = if (dt == .float16) exl3.f32ToF16Bits(v.*) else bf16Bits(v.*);
            v.* = if (dt == .float16) exl3.f16BitsToF32(b.*) else bf16ToF32(b.*);
        };
        const x = mlx.mlx_array_new_data(if (dt == .float32) @ptrCast(xs.ptr) else @ptrCast(xb.ptr), &[_]c_int{ @intCast(groups), @intCast(rows), @intCast(in) }, 3, dt);
        defer _ = mlx.mlx_array_free(x);
        const y = if (groups == 1) blk: {
            var x2 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x2);
            try mlx.check(mlx.mlx_reshape(&x2, x, &[_]c_int{ @intCast(rows), @intCast(in) }, 2, s));
            break :blk try linear(s, x2, h.p, cs.dec);
        } else try linearGrouped(s, x, h.p, cs.dec);
        defer _ = mlx.mlx_array_free(y);
        try t.expectEqual(groups * rows * out, mlx.mlx_array_size(y));
        var cy = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cy);
        const got = try hostF32(s, y, &cy);
        const want = try alloc.alloc(f32, got.len);
        for (0..groups) |g| for (0..rows) |r| {
            const at = g * rows + r;
            try h.project(alloc, g, xs[at * in ..][0..in], in, out, cs.n, cs.dec, want[at * out ..][0..out]);
        };
        for (got) |v| try t.expect(std.math.isFinite(v));
        const err = relRms(got, want);
        // The output is stored in x's dtype: its own rounding of the truth adds to the kernel's bar.
        const stored = try alloc.alloc(f32, want.len);
        for (stored, want) |*o, v| o.* = roundTo(dt, v);
        const floor = relRms(stored, want);
        if (err >= @sqrt(LINEAR_BAR * LINEAR_BAR + floor * floor)) {
            std.debug.print("linear {s} n{d} groups {d} rows {d} arm {s} {s}: relRms {d}\n", .{ @tagName(cs.dec.codebook), cs.n, groups, rows, @tagName(arm), @tagName(dt), err });
            failed += 1;
        }
    };
    try t.expectEqual(@as(usize, 0), failed);
}

test "exl3 linear: bf16 or f32 rows past the f16 range stay finite and hold the bar on the multi-row kernel, folded or split" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    defer kernels.setDecodeParams(.mul1);
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x0f16);
    const rnd = prng.random();
    const dec = exl3.Decode.tabled(.spec1, .w15);
    try dec.requireTable();
    // 512 -> 256 splits k and finishes in a second dispatch; 256 -> 40960 folds the finish into the launch on NAX.
    for ([_][2]usize{ .{ 512, 256 }, .{ 256, 40960 } }) |shape| {
        const in = shape[0];
        const out = shape[1];
        var h = try Host.init(alloc, rnd, 1, in, out, 64);
        defer h.deinit();
        for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dt| for ([_]usize{ 1, 5 }) |rows| {
            // Inputs near 3e5 against suh near 0.01 put outputs past 65504, the f16 ceiling, and keep the prepared
            // input (f16 in threadgroup memory) inside it.
            const xs = try alloc.alloc(f32, rows * in);
            for (xs) |*v| v.* = (rnd.float(f32) * 2 - 1) * 3e5;
            const xb = try alloc.alloc(u16, xs.len);
            if (dt == .bfloat16) for (xb, xs) |*b, *v| {
                b.* = bf16Bits(v.*);
                v.* = bf16ToF32(b.*);
            };
            const x = mlx.mlx_array_new_data(if (dt == .float32) @ptrCast(xs.ptr) else @ptrCast(xb.ptr), &[_]c_int{ @intCast(rows), @intCast(in) }, 2, dt);
            defer _ = mlx.mlx_array_free(x);
            const y = try linear(s, x, h.p, dec);
            defer _ = mlx.mlx_array_free(y);
            try t.expectEqual(dt, mlx.mlx_array_dtype(y));
            var cy = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(cy);
            const got = try hostF32(s, y, &cy);
            const want = try alloc.alloc(f32, got.len);
            for (0..rows) |r| try h.project(alloc, 0, xs[r * in ..][0..in], in, out, 64, dec, want[r * out ..][0..out]);
            var peak: f32 = 0;
            for (want) |v| peak = @max(peak, @abs(v));
            try t.expect(peak > 65504);
            for (got) |v| try t.expect(std.math.isFinite(v));
            const stored = try alloc.alloc(f32, want.len);
            for (stored, want) |*o, v| o.* = roundTo(dt, v);
            try t.expect(relRms(got, want) < @sqrt(LINEAR_BAR * LINEAR_BAR + relRms(stored, want) * relRms(stored, want)));
        };
    }
}

test "exl3 linear: linears sharing their input in one launch each hold the f32-truth bar of their own projection" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    defer kernels.setDecodeParams(.mul1);
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x5ea1);
    const rnd = prng.random();
    const dec = exl3.Decode.tabled(.spec1, .w15);
    const in: usize = 512;
    // wq_a | wkv | the compressor's wkv at DeepSeek's proportions, and two equal linears (wkv | wgate).
    for ([_][]const usize{ &.{ 640, 256, 256 }, &.{ 256, 256 }, &.{ 384, 128, 256, 128 } }) |widths| {
        var hosts: [SHARED_MAX]Host = undefined;
        var cols: [SHARED_MAX + 1]c_int = @splat(0);
        for (widths, 0..) |w, i| {
            hosts[i] = try Host.init(alloc, rnd, 1, in, w, 64);
            cols[i + 1] = cols[i] + @as(c_int, @intCast(w));
        }
        defer for (hosts[0..widths.len]) |*h| h.deinit();
        var tr: [SHARED_MAX]mlx.mlx_array = undefined;
        var su: [SHARED_MAX]mlx.mlx_array = undefined;
        var sv: [SHARED_MAX]mlx.mlx_array = undefined;
        for (hosts[0..widths.len], 0..) |h, i| {
            tr[i] = h.p.trellis;
            su[i] = h.p.suh;
            sv[i] = h.p.svh;
        }
        var cat: [3]mlx.mlx_array = .{ mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new() };
        defer for (cat) |a| {
            _ = mlx.mlx_array_free(a);
        };
        for ([_][]mlx.mlx_array{ tr[0..widths.len], su[0..widths.len], sv[0..widths.len] }, [_]c_int{ 2, 0, 1 }, 0..) |parts, axis, k| {
            const v = mlx.mlx_vector_array_new_data(parts.ptr, parts.len);
            defer _ = mlx.mlx_vector_array_free(v);
            try mlx.check(mlx.mlx_concatenate_axis(&cat[k], v, axis, s));
        }
        const sh: Shared = .{ .p = .{ .trellis = cat[0], .suh = cat[1], .svh = cat[2] }, .cols = cols, .count = widths.len };
        for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dt| for ([_]usize{ 1, 2, 5, 8 }) |rows| {
            const xs = try alloc.alloc(f32, rows * in);
            for (xs) |*v| v.* = (rnd.float(f32) * 2 - 1) * 30;
            const xb = try alloc.alloc(u16, xs.len);
            if (dt == .bfloat16) for (xb, xs) |*b, *v| {
                b.* = bf16Bits(v.*);
                v.* = bf16ToF32(b.*);
            };
            const x = mlx.mlx_array_new_data(if (dt == .float32) @ptrCast(xs.ptr) else @ptrCast(xb.ptr), &[_]c_int{ @intCast(rows), @intCast(in) }, 2, dt);
            defer _ = mlx.mlx_array_free(x);
            var outs: [SHARED_MAX]mlx.mlx_array = undefined;
            try linearShared(s, x, sh, dec, outs[0..widths.len]);
            defer for (outs[0..widths.len]) |o| {
                _ = mlx.mlx_array_free(o);
            };
            for (widths, 0..) |w, i| {
                try t.expectEqual(dt, mlx.mlx_array_dtype(outs[i]));
                try t.expectEqualSlices(c_int, &.{ @intCast(rows), @intCast(w) }, mlx.getShape(outs[i]));
                var cy = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(cy);
                const got = try hostF32(s, outs[i], &cy);
                const want = try alloc.alloc(f32, got.len);
                for (0..rows) |r| try hosts[i].project(alloc, 0, xs[r * in ..][0..in], in, w, 64, dec, want[r * w ..][0..w]);
                for (got) |v| try t.expect(std.math.isFinite(v));
                const stored = try alloc.alloc(f32, want.len);
                for (stored, want) |*o, v| o.* = roundTo(dt, v);
                const floor = relRms(stored, want);
                try t.expect(relRms(got, want) < @sqrt(LINEAR_BAR * LINEAR_BAR + floor * floor));
            }
        };
    }
}

test "exl3 linear: the multi-row kernel gives a row the same bytes at widths 2 to 8, on one bank and on a grouped bank" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    defer kernels.setDecodeParams(.mul1);
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0xbea7);
    const rnd = prng.random();
    const in: usize = 256;
    const out: usize = 384;
    const dec = exl3.Decode.tabled(.spec1, .w15);
    for ([_]usize{ 1, 2 }) |groups| {
        var h = try Host.init(alloc, rnd, groups, in, out, 64);
        defer h.deinit();
        const xb = try alloc.alloc(u16, groups * 8 * in);
        for (xb) |*b| b.* = exl3.f32ToF16Bits((rnd.float(f32) * 2 - 1) * 4);
        // Group g's 8 rows are x[g], and the narrower runs take its first `width` rows.
        const x = mlx.mlx_array_new_data(xb.ptr, &[_]c_int{ @intCast(groups), 8, @intCast(in) }, 3, .float16);
        defer _ = mlx.mlx_array_free(x);
        const all = try linearGrouped(s, x, h.p, dec);
        defer _ = mlx.mlx_array_free(all);
        var ca = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ca);
        const want = try hostF32(s, all, &ca);
        for ([_]c_int{ 2, 3, 5 }) |width| {
            var part = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(part);
            try mlx.check(mlx.mlx_slice(&part, x, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ @intCast(groups), width, @intCast(in) }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
            const y = try linearGrouped(s, part, h.p, dec);
            defer _ = mlx.mlx_array_free(y);
            var cy = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(cy);
            const got = try hostF32(s, y, &cy);
            for (0..groups) |g| {
                const w: usize = @intCast(width);
                try t.expectEqualSlices(u8, std.mem.sliceAsBytes(want[g * 8 * out ..][0 .. w * out]), std.mem.sliceAsBytes(got[g * w * out ..][0 .. w * out]));
            }
        }
    }
}

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

const exact: Opts = .{ .rows = .{ .exact = true } };

test "exl3 linear: with exact rows a row has the same bytes at widths 1 to 8 at Flash-Next's trunk shapes, stacked, on the machine's own arm and on SIMD" {
    try mlx.requireGpu();
    // The arm follows the hardware, never the row count.
    for (1..9) |rows| try std.testing.expectEqual(kernels.gemmNaxOn(), project_rows.rowsGeometry(2560, 12288, @intCast(rows), 1, exact.rows).nax);
    try exactRowsCase();
    _ = setenv("SUSHI_FORCE_GPU_FAMILY_FALLBACK", "1", 1);
    defer _ = unsetenv("SUSHI_FORCE_GPU_FAMILY_FALLBACK");
    for (1..9) |rows| try std.testing.expect(!project_rows.rowsGeometry(2560, 12288, @intCast(rows), 1, exact.rows).nax);
    try exactRowsCase();
}

/// Rows 1 to 8 of an exact call equal the same row alone, on one bank, and on three linears stacked into one launch.
fn exactRowsCase() !void {
    const t = std.testing;
    const s = mlx.gpuStream();
    defer kernels.setDecodeParams(.mul1);
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0xe4ac7);
    const rnd = prng.random();
    const dec = exl3.Decode.tabled(.spec1, .w15);
    // The 12288-wide output, a deep input and a narrow one each split k differently.
    const shapes = [_][3]usize{ .{ 2560, 12288, 48 }, .{ 2560, 12288, 96 }, .{ 2560, 10240, 64 }, .{ 2560, 6144, 64 }, .{ 6144, 2560, 48 }, .{ 2560, 640, 32 }, .{ 256, 131072, 32 } };
    for (shapes) |sh| {
        var h = try Host.init(alloc, rnd, 1, sh[0], sh[1], @intCast(sh[2]));
        defer h.deinit();
        const xb = try alloc.alloc(u16, 8 * sh[0]);
        for (xb) |*b| b.* = bf16Bits((rnd.float(f32) * 2 - 1) * 4);
        const x = mlx.mlx_array_new_data(xb.ptr, &[_]c_int{ 8, @intCast(sh[0]) }, 2, .bfloat16);
        defer _ = mlx.mlx_array_free(x);
        var solo: [8][]const f32 = undefined;
        var solo_arrays: [8]mlx.mlx_array = undefined;
        var made: usize = 0;
        defer for (solo_arrays[0..made]) |a| {
            _ = mlx.mlx_array_free(a);
        };
        for (0..8) |r| {
            var row = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(row);
            try mlx.check(mlx.mlx_slice(&row, x, &[_]c_int{ @intCast(r), 0 }, 2, &[_]c_int{ @intCast(r + 1), @intCast(sh[0]) }, 2, &[_]c_int{ 1, 1 }, 2, s));
            const y = try linearWith(s, row, h.p, dec, exact);
            solo_arrays[made] = mlx.mlx_array_new();
            made += 1;
            solo[r] = try hostF32(s, y, &solo_arrays[made - 1]);
            _ = mlx.mlx_array_free(y);
        }
        for (2..9) |width| {
            var part = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(part);
            try mlx.check(mlx.mlx_slice(&part, x, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ @intCast(width), @intCast(sh[0]) }, 2, &[_]c_int{ 1, 1 }, 2, s));
            const y = try linearWith(s, part, h.p, dec, exact);
            defer _ = mlx.mlx_array_free(y);
            var cy = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(cy);
            const got = try hostF32(s, y, &cy);
            for (0..width) |r| try t.expectEqualSlices(u8, std.mem.sliceAsBytes(solo[r]), std.mem.sliceAsBytes(got[r * sh[1] ..][0..sh[1]]));
        }
    }

    // Stacked: three linears that share an input, each row of the one launch the row alone in the same launch.
    const in: usize = 2560;
    const widths = [3]usize{ 1280, 512, 512 };
    var h = try Host.init(alloc, rnd, 1, in, widths[0] + widths[1] + widths[2], 48);
    defer h.deinit();
    var tiled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tiled);
    try mlx.check(mlx.mlx_tile(&tiled, h.p.suh, &[_]c_int{ 3, 1 }, 2, s));
    const stacked = Shared{ .p = .{ .trellis = h.p.trellis, .suh = tiled, .svh = h.p.svh }, .cols = .{ 0, 1280, 1792, 2304, 0 }, .count = 3 };
    const xb = try alloc.alloc(u16, 8 * in);
    for (xb) |*b| b.* = bf16Bits((rnd.float(f32) * 2 - 1) * 4);
    const x = mlx.mlx_array_new_data(xb.ptr, &[_]c_int{ 8, @intCast(in) }, 2, .bfloat16);
    defer _ = mlx.mlx_array_free(x);
    var want: [3][8][]const f32 = undefined;
    var want_arrays: [24]mlx.mlx_array = undefined;
    var wm: usize = 0;
    defer for (want_arrays[0..wm]) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (0..8) |r| {
        var row = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(row);
        try mlx.check(mlx.mlx_slice(&row, x, &[_]c_int{ @intCast(r), 0 }, 2, &[_]c_int{ @intCast(r + 1), @intCast(in) }, 2, &[_]c_int{ 1, 1 }, 2, s));
        var outs: [3]mlx.mlx_array = undefined;
        try linearSharedWith(s, row, stacked, dec, &outs, exact);
        for (outs, 0..) |o, i| {
            want_arrays[wm] = mlx.mlx_array_new();
            wm += 1;
            want[i][r] = try hostF32(s, o, &want_arrays[wm - 1]);
            _ = mlx.mlx_array_free(o);
        }
    }
    for (2..9) |width| {
        var part = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(part);
        try mlx.check(mlx.mlx_slice(&part, x, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ @intCast(width), @intCast(in) }, 2, &[_]c_int{ 1, 1 }, 2, s));
        var outs: [3]mlx.mlx_array = undefined;
        try linearSharedWith(s, part, stacked, dec, &outs, exact);
        for (outs, 0..) |o, i| {
            defer _ = mlx.mlx_array_free(o);
            var co = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(co);
            const got = try hostF32(s, o, &co);
            for (0..width) |r| try t.expectEqualSlices(u8, std.mem.sliceAsBytes(want[i][r]), std.mem.sliceAsBytes(got[r * widths[i] ..][0..widths[i]]));
        }
    }
}

test "exl3 linear: with exact rows a row has the same bytes at widths 1 to 6 at DeepSeek's trunk shapes, and the lm_head keeps its serial launch's" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    defer kernels.setDecodeParams(.mul1);
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x0ea5);
    const rnd = prng.random();
    const dec = exl3.Decode.tabled(.spec1, .w15);
    const shapes = [_]struct { in: usize, out: usize, n: u32, groups: usize }{
        .{ .in = 5120, .out = 1792, .n = 64, .groups = 1 },
        .{ .in = 1280, .out = 32768, .n = 64, .groups = 1 },
        .{ .in = 4096, .out = 1024, .n = 64, .groups = 8 },
        .{ .in = 8192, .out = 5120, .n = 64, .groups = 1 },
        .{ .in = 5120, .out = 129280, .n = 96, .groups = 1 },
    };
    for (shapes) |sh| {
        var h = try Host.init(alloc, rnd, sh.groups, sh.in, sh.out, sh.n);
        defer h.deinit();
        const xb = try alloc.alloc(u16, sh.groups * 6 * sh.in);
        for (xb) |*b| b.* = bf16Bits((rnd.float(f32) * 2 - 1) * 4);
        const x = mlx.mlx_array_new_data(xb.ptr, &[_]c_int{ @intCast(sh.groups), 6, @intCast(sh.in) }, 3, .bfloat16);
        defer _ = mlx.mlx_array_free(x);
        var one_row: []const f32 = &.{};
        for ([_]c_int{ 1, 2, 3, 6 }) |width| {
            var part = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(part);
            try mlx.check(mlx.mlx_slice(&part, x, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ @intCast(sh.groups), width, @intCast(sh.in) }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
            const y = try linearGroupedWith(s, part, h.p, dec, exact);
            defer _ = mlx.mlx_array_free(y);
            var cy = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(cy);
            const got = try hostF32(s, y, &cy);
            if (width == 1) {
                one_row = try alloc.dupe(f32, got);
                if (sh.out == 129280) {
                    const serial = try linearGrouped(s, part, h.p, dec);
                    defer _ = mlx.mlx_array_free(serial);
                    var cs = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(cs);
                    try t.expectEqualSlices(u8, std.mem.sliceAsBytes(one_row), std.mem.sliceAsBytes(try hostF32(s, serial, &cs)));
                }
                continue;
            }
            const w: usize = @intCast(width);
            for (0..sh.groups) |g| try t.expectEqualSlices(u8, std.mem.sliceAsBytes(one_row[g * sh.out ..][0..sh.out]), std.mem.sliceAsBytes(got[g * w * sh.out ..][0..sh.out]));
        }
    }
}

test "exl3 linear: a pair whose second launch sums the first's partials gives the two launches' bytes at DeepSeek's wo shapes and small ones, 1 to 16 rows, on the machine's arm and SIMD, MLX's stream and the engine's" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    defer mlx.engine.enabled = was;
    for ([_]bool{ false, true }) |on_engine| {
        mlx.engine.enabled = on_engine;
        try pairCase(if (on_engine) mlx.engine.stream() else mlx.gpuStream());
    }
    mlx.engine.enabled = false;
    _ = setenv("SUSHI_FORCE_GPU_FAMILY_FALLBACK", "1", 1);
    defer _ = unsetenv("SUSHI_FORCE_GPU_FAMILY_FALLBACK");
    try std.testing.expect(!kernels.gemmNaxOn());
    try pairCase(mlx.gpuStream());
}

test "exl3 linear: a stacked launch whose first linear leaves rms-normed gives the launch's, its finish's and MLX's rms norm's bytes, 1 to 8 rows, on the machine's arm and SIMD, MLX's stream and the engine's" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    defer mlx.engine.enabled = was;
    for ([_]bool{ false, true }) |on_engine| {
        mlx.engine.enabled = on_engine;
        try normedCase(if (on_engine) mlx.engine.stream() else mlx.gpuStream());
    }
    mlx.engine.enabled = false;
    _ = setenv("SUSHI_FORCE_GPU_FAMILY_FALLBACK", "1", 1);
    defer _ = unsetenv("SUSHI_FORCE_GPU_FAMILY_FALLBACK");
    try std.testing.expect(!kernels.gemmNaxOn());
    try normedCase(mlx.gpuStream());
}

test "exl3 linear: a launch whose finish expands its rows into streams gives the host expand of the launch's bytes, alone and as a pair's second, 1 to 8 rows, on the machine's arm and SIMD, MLX's stream and the engine's" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    defer mlx.engine.enabled = was;
    for ([_]bool{ false, true }) |on_engine| {
        mlx.engine.enabled = on_engine;
        try expandedCase(if (on_engine) mlx.engine.stream() else mlx.gpuStream());
    }
    mlx.engine.enabled = false;
    _ = setenv("SUSHI_FORCE_GPU_FAMILY_FALLBACK", "1", 1);
    defer _ = unsetenv("SUSHI_FORCE_GPU_FAMILY_FALLBACK");
    try std.testing.expect(!kernels.gemmNaxOn());
    try expandedCase(mlx.gpuStream());
}

/// The oracle is the host's f32 fma chain over the plain launch's rows, rounded to the dtype.
fn expandedCase(s: mlx.mlx_stream) !void {
    const t = std.testing;
    defer kernels.setDecodeParams(.mul1);
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0xe8a1);
    const rnd = prng.random();
    const dec = exl3.Decode.tabled(.spec1, .w15);
    // DeepSeek's wo_a and wo_b with its four streams, then small ones with two and three.
    const shapes = [_]struct { groups: usize, in: usize, mid: usize, out: usize, n: u32, hc: usize }{
        .{ .groups = 8, .in = 4096, .mid = 1024, .out = 5120, .n = 64, .hc = 4 },
        .{ .groups = 2, .in = 256, .mid = 256, .out = 384, .n = 64, .hc = 2 },
        .{ .groups = 1, .in = 384, .mid = 256, .out = 128, .n = 96, .hc = 3 },
    };
    var served: usize = 0;
    for (shapes) |sh| {
        var ha = try Host.init(alloc, rnd, sh.groups, sh.in, sh.mid, sh.n);
        defer ha.deinit();
        var hb = try Host.init(alloc, rnd, 1, sh.groups * sh.mid, sh.out, sh.n);
        defer hb.deinit();
        for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dt| for ([_]Opts{ .{}, exact, .{ .rows = .{ .exact = true, .wide = true } } }) |opts| for ([_]usize{ 1, 3, 8 }) |rows| {
            const hc = sh.hc;
            const nx = sh.groups * rows * sh.in;
            const xf = try alloc.alloc(f32, nx);
            const xb = try alloc.alloc(u16, nx);
            for (xf, xb) |*f, *bits| {
                f.* = (rnd.float(f32) * 2 - 1) * 4;
                bits.* = bf16Bits(f.*);
            }
            const x = mlx.mlx_array_new_data(if (dt == .float32) @ptrCast(xf.ptr) else @ptrCast(xb.ptr), &[_]c_int{ @intCast(sh.groups), @intCast(rows), @intCast(sh.in) }, 3, dt);
            defer _ = mlx.mlx_array_free(x);
            const ns = rows * hc * sh.out;
            const sf = try alloc.alloc(f32, ns);
            const sb = try alloc.alloc(u16, ns);
            for (sf, sb) |*f, *bits| {
                bits.* = bf16Bits((rnd.float(f32) * 2 - 1) * 3);
                if (dt == .float32) f.* = (rnd.float(f32) * 2 - 1) * 3 else f.* = bf16ToF32(bits.*);
            }
            const streams = mlx.mlx_array_new_data(if (dt == .float32) @ptrCast(sf.ptr) else @ptrCast(sb.ptr), &[_]c_int{ @intCast(rows), @intCast(hc), @intCast(sh.out) }, 3, dt);
            defer _ = mlx.mlx_array_free(streams);
            const postv = try alloc.alloc(f32, rows * hc);
            for (postv) |*v| v.* = rnd.float(f32) * 2;
            const combv = try alloc.alloc(f32, rows * hc * hc);
            for (combv) |*v| v.* = rnd.float(f32);
            const post = mlx.mlx_array_new_data(postv.ptr, &[_]c_int{ @intCast(rows), @intCast(hc) }, 2, .float32);
            defer _ = mlx.mlx_array_free(post);
            const comb = mlx.mlx_array_new_data(combv.ptr, &[_]c_int{ @intCast(rows), @intCast(hc), @intCast(hc) }, 3, .float32);
            defer _ = mlx.mlx_array_free(comb);
            const e: project_rows.Expand = .{ .x = streams, .post = post, .comb = comb };

            const mid = try linearGroupedWith(s, x, ha.p, dec, opts);
            defer _ = mlx.mlx_array_free(mid);
            var turned = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(turned);
            try mlx.check(mlx.mlx_transpose_axes(&turned, mid, &[_]c_int{ 1, 0, 2 }, 3, s));
            var flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(flat);
            try mlx.check(mlx.mlx_reshape(&flat, turned, &[_]c_int{ @intCast(rows), @intCast(sh.groups * sh.mid) }, 2, s));
            const plain = try linearWith(s, flat, hb.p, dec, opts);
            defer _ = mlx.mlx_array_free(plain);
            var cp = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(cp);
            const y = try hostF32(s, plain, &cp);
            const want = try alloc.alloc(f32, ns);
            for (0..rows) |r| for (0..hc) |i| for (0..sh.out) |d| {
                var acc: f32 = 0;
                for (0..hc) |j| acc = @mulAdd(f32, combv[(r * hc + j) * hc + i], sf[(r * hc + j) * sh.out + d], acc);
                const v = @mulAdd(f32, postv[r * hc + i], y[r * sh.out + d], acc);
                want[(r * hc + i) * sh.out + d] = if (dt == .float32) v else bf16ToF32(bf16Bits(v));
            };

            const alone = (try linearExpanded(s, flat, hb.p, dec, opts, e)) orelse return error.ExpandNotServed;
            defer _ = mlx.mlx_array_free(alone);
            var ca = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ca);
            try t.expectEqual(dt, mlx.mlx_array_dtype(alone));
            try t.expectEqualSlices(c_int, &[_]c_int{ @intCast(rows), @intCast(hc), @intCast(sh.out) }, mlx.getShape(alone));
            try t.expectEqualSlices(u8, std.mem.sliceAsBytes(want), std.mem.sliceAsBytes(try hostF32(s, alone, &ca)));

            const paired = (try linearPair(s, x, ha.p, hb.p, dec, dec, opts, e)) orelse return error.ExpandNotServed;
            defer _ = mlx.mlx_array_free(paired);
            var cq = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(cq);
            try t.expectEqualSlices(u8, std.mem.sliceAsBytes(want), std.mem.sliceAsBytes(try hostF32(s, paired, &cq)));
            served += 1;
        };
    }
    try t.expectEqual(@as(usize, 3 * 2 * 3 * 3), served);
}

/// f32 activations show every ulp of the norm's sums, which bf16 outputs round away; bf16 is the served dtype.
fn normedCase(s: mlx.mlx_stream) !void {
    const t = std.testing;
    defer kernels.setDecodeParams(.mul1);
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x4a55);
    const rnd = prng.random();
    const dec = exl3.Decode.tabled(.spec1, .w15);
    // DeepSeek's wq_a, wkv and a compressor's wkv on one input; then two linears, then one alone.
    const shapes = [_]struct { in: usize, widths: []const usize, n: u32 }{
        .{ .in = 5120, .widths = &.{ 1280, 512 }, .n = 64 },
        .{ .in = 5120, .widths = &.{ 1280, 512, 512 }, .n = 64 },
        .{ .in = 384, .widths = &.{ 256, 128 }, .n = 96 },
        .{ .in = 256, .widths = &.{384}, .n = 64 },
    };
    var served: usize = 0;
    for (shapes) |sh| {
        var total: usize = 0;
        var cols: [SHARED_MAX + 1]c_int = @splat(0);
        for (sh.widths, 0..) |w, i| {
            total += w;
            cols[i + 1] = @intCast(total);
        }
        var h = try Host.init(alloc, rnd, 1, sh.in, total, sh.n);
        defer h.deinit();
        const suh_rows = try alloc.alloc(u16, sh.widths.len * sh.in);
        for (0..sh.widths.len) |i| @memcpy(suh_rows[i * sh.in ..][0..sh.in], h.suh);
        const tiled = mlx.mlx_array_new_data(suh_rows.ptr, &[_]c_int{ @intCast(sh.widths.len), @intCast(sh.in) }, 2, .float16);
        defer _ = mlx.mlx_array_free(tiled);
        const stacked = Shared{ .p = .{ .trellis = h.p.trellis, .suh = tiled, .svh = h.p.svh }, .cols = cols, .count = sh.widths.len };
        const eps_v = [_]f32{ 1e-6, 0 };
        const eps = mlx.mlx_array_new_data(&eps_v, &[_]c_int{2}, 1, .float32);
        defer _ = mlx.mlx_array_free(eps);
        for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dt| {
            const wf = try alloc.alloc(f32, sh.widths[0]);
            const wb = try alloc.alloc(u16, sh.widths[0]);
            for (wf, wb) |*f, *bits| {
                f.* = 0.5 + rnd.float(f32);
                bits.* = bf16Bits(f.*);
            }
            const w = mlx.mlx_array_new_data(if (dt == .float32) @ptrCast(wf.ptr) else @ptrCast(wb.ptr), &[_]c_int{@intCast(sh.widths[0])}, 1, dt);
            defer _ = mlx.mlx_array_free(w);
            for ([_]Opts{ .{}, exact, .{ .rows = .{ .exact = true, .wide = true } } }) |opts| for ([_]usize{ 1, 3, 8 }) |rows| for ([_]f32{ 4, 1e-3 }) |amp| {
                const xf = try alloc.alloc(f32, rows * sh.in);
                const xb = try alloc.alloc(u16, rows * sh.in);
                for (xf, xb) |*f, *bits| {
                    f.* = (rnd.float(f32) * 2 - 1) * amp;
                    bits.* = bf16Bits(f.*);
                }
                const x = mlx.mlx_array_new_data(if (dt == .float32) @ptrCast(xf.ptr) else @ptrCast(xb.ptr), &[_]c_int{ @intCast(rows), @intCast(sh.in) }, 2, dt);
                defer _ = mlx.mlx_array_free(x);
                var want: [SHARED_MAX]mlx.mlx_array = undefined;
                try linearSharedWith(s, x, stacked, dec, want[0..sh.widths.len], opts);
                defer for (want[0..sh.widths.len]) |o| {
                    _ = mlx.mlx_array_free(o);
                };
                var normed = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(normed);
                try mlx.check(mlx.mlx_fast_rms_norm(&normed, want[0], w, 1e-6, s));
                var got: [SHARED_MAX]mlx.mlx_array = undefined;
                try t.expect(try linearSharedNormed(s, x, stacked, dec, got[0..sh.widths.len], opts, .{ .w = w, .eps = eps, .cols = @intCast(sh.widths[0]) }));
                defer for (got[0..sh.widths.len]) |o| {
                    _ = mlx.mlx_array_free(o);
                };
                for (0..sh.widths.len) |i| {
                    var cw = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(cw);
                    var cg = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(cg);
                    const wv = try alloc.dupe(f32, try hostF32(s, if (i == 0) normed else want[i], &cw));
                    try t.expectEqual(dt, mlx.mlx_array_dtype(got[i]));
                    try t.expectEqualSlices(u8, std.mem.sliceAsBytes(wv), std.mem.sliceAsBytes(try hostF32(s, got[i], &cg)));
                }
                served += 1;
            };
        }
    }
    try t.expectEqual(@as(usize, 4 * 2 * 3 * 3 * 2), served);
}

/// Every pair `linearPair` serves equals the grouped launch, its finish and the one-bank launch; past one multi-row
/// launch it declines.
fn pairCase(s: mlx.mlx_stream) !void {
    const t = std.testing;
    defer kernels.setDecodeParams(.mul1);
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x9a17);
    const rnd = prng.random();
    const dec = exl3.Decode.tabled(.spec1, .w15);
    // DeepSeek's wo_a and wo_b, then a two-bank and a one-bank first launch.
    const shapes = [_]struct { groups: usize, in: usize, mid: usize, out: usize, n: u32 }{
        .{ .groups = 8, .in = 4096, .mid = 1024, .out = 5120, .n = 64 },
        .{ .groups = 2, .in = 256, .mid = 256, .out = 384, .n = 64 },
        .{ .groups = 1, .in = 384, .mid = 256, .out = 128, .n = 96 },
    };
    for (shapes) |sh| {
        var ha = try Host.init(alloc, rnd, sh.groups, sh.in, sh.mid, sh.n);
        defer ha.deinit();
        var hb = try Host.init(alloc, rnd, 1, sh.groups * sh.mid, sh.out, sh.n);
        defer hb.deinit();
        for ([_]Opts{ .{}, exact, .{ .rows = .{ .exact = true, .wide = true } } }) |opts| for ([_]usize{ 1, 3, 8, 16 }) |rows| {
            const xb = try alloc.alloc(u16, sh.groups * rows * sh.in);
            for (xb) |*v| v.* = bf16Bits((rnd.float(f32) * 2 - 1) * 4);
            const x = mlx.mlx_array_new_data(xb.ptr, &[_]c_int{ @intCast(sh.groups), @intCast(rows), @intCast(sh.in) }, 3, .bfloat16);
            defer _ = mlx.mlx_array_free(x);
            const got = try linearPair(s, x, ha.p, hb.p, dec, dec, opts, null);
            defer if (got) |g| {
                _ = mlx.mlx_array_free(g);
            };
            if (rows > project_rows.rowsMax(opts.rows)) {
                try t.expect(got == null);
                continue;
            }
            const mid = try linearGroupedWith(s, x, ha.p, dec, opts);
            defer _ = mlx.mlx_array_free(mid);
            var turned = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(turned);
            try mlx.check(mlx.mlx_transpose_axes(&turned, mid, &[_]c_int{ 1, 0, 2 }, 3, s));
            var flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(flat);
            try mlx.check(mlx.mlx_reshape(&flat, turned, &[_]c_int{ @intCast(rows), @intCast(sh.groups * sh.mid) }, 2, s));
            // The first launch alone, its banks side by side in each row, is that transposed copy.
            const side = (try linearGroupedRows(s, x, ha.p, dec, opts)) orelse return error.RowMajorNotServed;
            defer _ = mlx.mlx_array_free(side);
            var cs = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(cs);
            var cf = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(cf);
            try t.expectEqualSlices(c_int, mlx.getShape(flat), mlx.getShape(side));
            const fv = try alloc.dupe(f32, try hostF32(s, flat, &cf));
            try t.expectEqualSlices(u8, std.mem.sliceAsBytes(fv), std.mem.sliceAsBytes(try hostF32(s, side, &cs)));
            const want = try linearWith(s, flat, hb.p, dec, opts);
            defer _ = mlx.mlx_array_free(want);
            var cw = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(cw);
            var cg = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(cg);
            const w = try hostF32(s, want, &cw);
            try t.expect(got != null);
            try t.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(got.?));
            try t.expectEqualSlices(u8, std.mem.sliceAsBytes(w), std.mem.sliceAsBytes(try hostF32(s, got.?, &cg)));
        };
    }
}

fn u32Values(a: mlx.mlx_array) ![]const u32 {
    try mlx.check(mlx.mlx_array_eval(a));
    return (mlx.mlx_array_data_uint32(a) orelse return error.Unreadable)[0..mlx.mlx_array_size(a)];
}

test "exl3 linear: the GEMM inputs of one groups, rows and window are built once, and equal the bank ids and window table built directly" {
    try mlx.requireGpu();
    const t = std.testing;
    const first = try gemmInputs(3, 70, 32);
    const again = try gemmInputs(3, 70, 32);
    try t.expectEqual(first.slots.ctx, again.slots.ctx);
    try t.expectEqual(first.tab.starts.ctx, again.tab.starts.ctx);
    const direct = try windowTable(3, 70, 32);
    defer _ = mlx.mlx_array_free(direct.starts);
    defer _ = mlx.mlx_array_free(direct.nlives);
    try t.expectEqual(direct.nwin, first.tab.nwin);
    try t.expectEqualSlices(u32, try u32Values(direct.starts), try u32Values(first.tab.starts));
    try t.expectEqualSlices(u32, try u32Values(direct.nlives), try u32Values(first.tab.nlives));
    const ids = try u32Values(first.slots);
    try t.expectEqual(@as(usize, 210), ids.len);
    for (ids, 0..) |v, i| try t.expectEqual(@as(u32, @intCast(i / 70)), v);
    // Past the cache's length the oldest entries go and a fresh build still equals the direct one.
    for (1..gemm_cache_len + 2) |k| _ = try gemmInputs(1, 40 + k, 32);
    const rebuilt = try gemmInputs(3, 70, 32);
    try t.expectEqualSlices(u32, try u32Values(direct.starts), try u32Values(rebuilt.tab.starts));
}
