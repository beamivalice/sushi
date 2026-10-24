//! Dense trellis linears: a trunk weight stored as a one-bank (or, for a grouped projection, G-bank) EXL3/STQ
//! projection and served by the expert kernels. Up to `PROJECT_ROWS_MAX` rows ride the multi-row kernel (two
//! dispatches, each weight decoded once for every row); wider chunks the sorted GEMM over a window table built on the
//! host from the known row order, so no GPU drain.
const std = @import("std");
const mlx = @import("mlx_host").mlx;
const io_util = @import("mlx_host").io_util;
const exl3 = @import("expert_exl3.zig");
const kernels = @import("expert_exl3_kernels.zig");
const root = @import("root.zig");

/// Tests and the timing: `rows` is the served arm up to `PROJECT_ROWS_MAX`, `gemm` the sorted GEMM at any width,
/// `gemv` the indexed GEMV (one row per bank).
const Arm = enum { rows, gemm, gemv };

/// `x` [rows, in] through the one-bank projection `p` (trellis [1, in/16, out/16, n]): [rows, out] in x's dtype.
pub fn linear(s: mlx.mlx_stream, x: mlx.mlx_array, p: root.Proj, dec: exl3.Decode) !mlx.mlx_array {
    const rows: usize = @intCast(mlx.getShape(x)[0]);
    return run(s, x, p, dec, 1, rows);
}

/// `x` [G, rows, in] through the G-bank projection `p` (bank g serves x[g]): [G, rows, out] in x's dtype.
pub fn linearGrouped(s: mlx.mlx_stream, x: mlx.mlx_array, p: root.Proj, dec: exl3.Decode) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    if (sh.len != 3) return error.BadExl3Shape;
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, x, &[_]c_int{ sh[0] * sh[1], sh[2] }, 2, s));
    const y = try run(s, flat, p, dec, @intCast(sh[0]), @intCast(sh[1]));
    defer _ = mlx.mlx_array_free(y);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, y, &[_]c_int{ sh[0], sh[1], mlx.getShape(y)[1] }, 3, s));
    return out;
}

/// The most linears one shared-input launch serves.
pub const SHARED_MAX: usize = 4;

/// Linears that read the same rows, stacked along the output into one bank: trellis `[1, in/16, out/16, n]`, `suh`
/// `[S, in]` (one row per linear), `svh` `[1, out]`; linear s owns columns `cols[s]..cols[s + 1]`, each a multiple of 128.
pub const Shared = struct { p: root.Proj, cols: [SHARED_MAX + 1]c_int, count: usize };

/// `x` [rows, in] (rows <= `kernels.PROJECT_ROWS_MAX`) through every linear of `sh` in one multi-row launch: `outs[s]` is
/// linear s's [rows, width] in x's dtype (the caller frees each). A row holds the bar of its own linear's launch; the
/// sum order may differ from it, since the launch splits k for the stacked width.
pub fn linearShared(s: mlx.mlx_stream, x: mlx.mlx_array, sh: Shared, dec: exl3.Decode, outs: []mlx.mlx_array) !void {
    try dec.requireTable();
    if (outs.len != sh.count) return error.BadExl3Shape;
    const rows: c_int = mlx.getShape(x)[0];
    kernels.setDecodeParams(dec);
    const total = sh.cols[sh.count];
    var bounds: [3]c_int = @splat(total);
    for (1..sh.count) |i| bounds[i - 1] = sh.cols[i];
    const flat = try kernels.projectRowsSegmented(s, x, sh.p.trellis, sh.p.suh, sh.p.svh, bounds);
    defer _ = mlx.mlx_array_free(flat);
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

var force_arm: ?Arm = null;

/// `x` [groups * rows, in], bank g's rows contiguous.
fn run(s: mlx.mlx_stream, x: mlx.mlx_array, p: root.Proj, dec: exl3.Decode, groups: usize, rows: usize) !mlx.mlx_array {
    try dec.requireTable();
    const tsh = mlx.getShape(p.trellis);
    if (tsh.len != 4 or tsh[0] != groups) return error.BadExl3Shape;
    kernels.setDecodeParams(dec);
    if ((force_arm orelse if (rows <= kernels.PROJECT_ROWS_MAX) Arm.rows else Arm.gemm) == .rows)
        return kernels.projectRows(s, x, p.trellis, p.suh, p.svh);
    const alloc = std.heap.page_allocator;
    const ids = try alloc.alloc(u32, groups * rows);
    defer alloc.free(ids);
    for (ids, 0..) |*v, i| v.* = @intCast(i / rows);
    const slots = mlx.mlx_array_new_data(ids.ptr, &[_]c_int{@intCast(ids.len)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(slots);
    const prepared = try kernels.prepareIndexed(s, x, p.suh, slots);
    defer _ = mlx.mlx_array_free(prepared);
    const inner = if ((force_arm orelse .gemm) == .gemm) blk: {
        const tab = try windowTable(groups, rows, kernels.gemmWindowRows());
        defer _ = mlx.mlx_array_free(tab.starts);
        defer _ = mlx.mlx_array_free(tab.nlives);
        break :blk try kernels.innerGemmSortedTable(s, prepared, p.trellis, slots, kernels.gemmWindowRows(), true, tab);
    } else try kernels.indexedGemvCoopF16(s, prepared, p.trellis, slots);
    defer _ = mlx.mlx_array_free(inner);
    return kernels.finishIndexedAs(s, inner, p.svh, slots, mlx.mlx_array_dtype(x));
}

/// Windows of at most `win` rows that never straddle two banks.
fn windowTable(groups: usize, rows: usize, win: c_int) !kernels.WindowTable {
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
    const st = mlx.mlx_array_new_data(starts.ptr, &[_]c_int{n}, 1, .uint32);
    errdefer _ = mlx.mlx_array_free(st);
    return .{ .starts = st, .nlives = mlx.mlx_array_new_data(lives.ptr, &[_]c_int{n}, 1, .uint32), .nwin = n };
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
        if ((arm == .gemv and rows > 3) or (arm == .rows and rows > kernels.PROJECT_ROWS_MAX)) continue;
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

const bench = @import("bench_util.zig");

/// DeepSeek-V4.1-Flash's trunk linears as the STQ pack stores them (`trunk_quant` n64, the head n96).
const DsvShape = struct { name: []const u8, out: usize, in: usize, n: u32, groups: usize = 1 };
const DSV_SHAPES = [_]DsvShape{
    .{ .name = "wq_a", .out = 1280, .in = 5120, .n = 64 },
    .{ .name = "wq_b", .out = 32768, .in = 1280, .n = 64 },
    .{ .name = "wkv", .out = 512, .in = 5120, .n = 64 },
    .{ .name = "wo_a", .out = 1024, .in = 4096, .n = 64, .groups = 8 },
    .{ .name = "wo_b", .out = 5120, .in = 8192, .n = 64 },
    .{ .name = "shared_w1", .out = 2304, .in = 5120, .n = 64 },
    .{ .name = "shared_w2", .out = 5120, .in = 2304, .n = 64 },
    .{ .name = "comp_wkv", .out = 512, .in = 5120, .n = 64 },
    .{ .name = "idx_wq_b", .out = 4096, .in = 1280, .n = 64 },
    .{ .name = "idx_wk", .out = 128, .in = 512, .n = 64 },
    .{ .name = "head", .out = 129280, .in = 5120, .n = 96 },
};

const DsvArm = enum { stq_gemm, stq_gemv, stq_rows, mxfp8, affine4 };

const DsvCtx = struct {
    h: *const Host,
    g: usize,
    rows: usize,
    in: c_int,
    wq8: mlx.mlx_array,
    sq8: mlx.mlx_array,
    w4: mlx.mlx_array,
    s4: mlx.mlx_array,
    dec: exl3.Decode,
    arm: DsvArm,
};

fn dsvCall(c: DsvCtx, st: mlx.mlx_stream, _: usize, x: mlx.mlx_array) anyerror!mlx.mlx_array {
    var o = mlx.mlx_array_new();
    switch (c.arm) {
        .stq_gemm, .stq_gemv, .stq_rows => {
            force_arm = switch (c.arm) {
                .stq_gemm => .gemm,
                .stq_gemv => .gemv,
                else => .rows,
            };
            defer force_arm = null;
            if (c.g > 1) return linearGrouped(st, x, c.h.p, c.dec);
            var x2 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x2);
            try mlx.check(mlx.mlx_reshape(&x2, x, &[_]c_int{ @intCast(c.rows), c.in }, 2, st));
            return linear(st, x2, c.h.p, c.dec);
        },
        .mxfp8 => try mlx.check(mlx.mlx_quantized_matmul(&o, x, c.wq8, c.sq8, .{ .ctx = null }, true, mlx.mlx_optional_int.some(32), mlx.mlx_optional_int.some(8), "mxfp8", st)),
        .affine4 => try mlx.check(mlx.mlx_quantized_matmul(&o, x, c.w4, c.s4, c.s4, true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", st)),
    }
    return o;
}

// DeepSeek-V4.1-Flash trunk shapes, one chain of dependent calls per arm (launch latency counts), the arms
// interleaved in one process: the STQ GEMM (served), the STQ indexed GEMV, the multi-row kernel (one bank, 1..8
// rows), and MLX's mxfp8 (the release's own trunk) and affine-4 on the same rows. Prints only under
// SUSHI_DSV41_STQ_UBENCH; `_STEPS` and `_ROUNDS` size the chains.
test "exl3 linear timing at DeepSeek-V4.1 trunk shapes (SUSHI_DSV41_STQ_UBENCH=1)" {
    if (!bench.diagOn("SUSHI_DSV41_STQ_UBENCH")) return error.SkipZigTest;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    defer kernels.setDecodeParams(.mul1);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0xd5);
    const rnd = prng.random();
    const only = std.mem.span(std.c.getenv("SUSHI_DSV41_STQ_UBENCH").?);
    const steps = bench.envUsize("SUSHI_UBENCH_STEPS", 32);
    const rounds = bench.envUsize("SUSHI_UBENCH_ROUNDS", 11);
    for (DSV_SHAPES) |sh| {
        if (!std.mem.eql(u8, only, "1")) {
            var picked = false;
            var tokens = std.mem.tokenizeScalar(u8, only, ',');
            while (tokens.next()) |tok| picked = picked or std.mem.indexOf(u8, sh.name, tok) != null;
            if (!picked) continue;
        }
        const dec = exl3.Decode.tabled(.spec1, .w15);
        var h = try Host.init(alloc, rnd, sh.groups, sh.in, sh.out, sh.n);
        defer h.deinit();
        const G: c_int = @intCast(sh.groups);
        const per_out: c_int = @intCast(sh.out);
        const in: c_int = @intCast(sh.in);
        const codes = try alloc.alloc(u32, sh.groups * sh.out * sh.in / 4);
        for (codes) |*v| v.* = rnd.int(u32) & 0x3f3f3f3f;
        const e8 = try alloc.alloc(u8, sh.groups * sh.out * sh.in / 32);
        for (e8) |*v| v.* = 120 + rnd.uintLessThan(u8, 8);
        const wq8 = mlx.mlx_array_new_data(codes.ptr, &[_]c_int{ G, per_out, @divExact(in, 4) }, 3, .uint32);
        defer _ = mlx.mlx_array_free(wq8);
        const sq8 = mlx.mlx_array_new_data(e8.ptr, &[_]c_int{ G, per_out, @divExact(in, 32) }, 3, .uint8);
        defer _ = mlx.mlx_array_free(sq8);
        const codes4 = try alloc.alloc(u32, sh.groups * sh.out * sh.in / 8);
        for (codes4) |*v| v.* = rnd.int(u32);
        const w4 = mlx.mlx_array_new_data(codes4.ptr, &[_]c_int{ G, per_out, @divExact(in, 8) }, 3, .uint32);
        defer _ = mlx.mlx_array_free(w4);
        var sc4 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sc4);
        try mlx.check(mlx.mlx_full(&sc4, &[_]c_int{ G, per_out, @divExact(in, 64) }, 3, mlx.mlx_array_new_float(0.01), .bfloat16, s));
        const bytes: f64 = @as(f64, @floatFromInt(sh.groups * (sh.in / 16) * (sh.out / 16) * sh.n)) * 2;
        for ([_]usize{ 1, 5, 8, 256 }) |rows| {
            if (sh.groups > 1 and rows == 256) continue;
            const xs = try alloc.alloc(u16, sh.groups * rows * sh.in);
            for (xs) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) - 0.5);
            const x16 = mlx.mlx_array_new_data(xs.ptr, &[_]c_int{ G, @intCast(rows), in }, 3, .float16);
            defer _ = mlx.mlx_array_free(x16);
            var xb = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(xb);
            try mlx.check(mlx.mlx_astype(&xb, x16, .bfloat16, s));
            try mlx.check(mlx.mlx_array_eval(xb));
            const base: DsvCtx = .{ .h = &h, .g = sh.groups, .rows = rows, .in = in, .wq8 = wq8, .sq8 = sq8, .w4 = w4, .s4 = sc4, .dec = dec, .arm = .stq_gemm };
            const arms = [_]DsvArm{ .stq_gemm, .stq_gemv, .stq_rows, .mxfp8, .affine4 };
            const runs = struct {
                fn on(arm: DsvArm, r: usize) bool {
                    return switch (arm) {
                        .stq_rows => r <= kernels.PROJECT_ROWS_MAX,
                        .stq_gemv => r <= 5,
                        else => true,
                    };
                }
            }.on;
            const times = try alloc.alloc([]u64, arms.len + 1);
            for (times) |*t| t.* = try alloc.alloc(u64, rounds);
            for (0..rounds + 1) |round| for (0..arms.len + 1) |j| {
                const k = (round + j) % (arms.len + 1);
                var ns: u64 = undefined;
                if (k == arms.len) {
                    ns = try bench.chainNs(s, x16, steps, {}, bench.linkOnly);
                } else {
                    var c = base;
                    c.arm = arms[k];
                    if (!runs(c.arm, rows)) continue;
                    ns = try bench.chainNs(s, if (c.arm == .mxfp8 or c.arm == .affine4) xb else x16, steps, c, dsvCall);
                }
                if (round > 0) times[k][round - 1] = ns;
            };
            const floor = bench.fastest(times[arms.len]) / 1e3 / @as(f64, @floatFromInt(steps));
            var line: [384]u8 = undefined;
            var w = std.Io.Writer.fixed(&line);
            for (arms, 0..) |arm, k| {
                if (!runs(arm, rows)) {
                    w.print(" {s} -", .{@tagName(arm)}) catch {};
                    continue;
                }
                const us = bench.fastest(times[k]) / 1e3 / @as(f64, @floatFromInt(steps)) - floor;
                w.print(" {s} {d:.1}us", .{ @tagName(arm), us }) catch {};
                if (arm == .stq_gemm or arm == .stq_gemv or arm == .stq_rows) w.print("({d:.0}GB/s)", .{bench.gbps(bytes, us)}) catch {};
            }
            std.debug.print("dsv41-stq {s} G{d} [{d}x{d}] n{d} rows {d} ({d:.1} MB):{s}\n", .{ sh.name, sh.groups, sh.out, sh.in, sh.n, rows, bytes / 1e6, w.buffered() });
        }
    }
}

const SharedCtx = struct { qa: *const Host, kv: *const Host, sh: Shared, dec: exl3.Decode, fused: bool };

/// One step: DeepSeek's wq_a and wkv of the same rows, in one launch or two; a one-element add waits on both outputs.
fn sharedStep(c: SharedCtx, st: mlx.mlx_stream, _: usize, x: mlx.mlx_array) anyerror!mlx.mlx_array {
    var a = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a);
    var b = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b);
    if (c.fused) {
        var outs: [2]mlx.mlx_array = undefined;
        try linearShared(st, x, c.sh, c.dec, &outs);
        a = outs[0];
        b = outs[1];
    } else {
        a = try linear(st, x, c.qa.p, c.dec);
        b = try linear(st, x, c.kv.p, c.dec);
    }
    var a1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a1);
    var b1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(b1);
    try mlx.check(mlx.mlx_slice(&a1, a, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ 1, 1 }, 2, &[_]c_int{ 1, 1 }, 2, st));
    try mlx.check(mlx.mlx_slice(&b1, b, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ 1, 1 }, 2, &[_]c_int{ 1, 1 }, 2, st));
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_add(&y, a1, b1, st));
    return y;
}

// DeepSeek's attention-input linears (wq_a 5120->1280, wkv 5120->512) as one stacked launch against two, chained like a
// forward, interleaved. Prints only under SUSHI_DSV41_STQ_UBENCH (`1` or a filter naming `shared`).
test "exl3 linear timing: DeepSeek's wq_a and wkv in one launch against two (SUSHI_DSV41_STQ_UBENCH)" {
    if (!bench.diagOn("SUSHI_DSV41_STQ_UBENCH")) return error.SkipZigTest;
    const only = std.mem.span(std.c.getenv("SUSHI_DSV41_STQ_UBENCH").?);
    if (!std.mem.eql(u8, only, "1") and std.mem.indexOf(u8, only, "shared") == null) return error.SkipZigTest;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    defer kernels.setDecodeParams(.mul1);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x5a);
    const rnd = prng.random();
    const dec = exl3.Decode.tabled(.spec1, .w15);
    var qa = try Host.init(alloc, rnd, 1, 5120, 1280, 64);
    defer qa.deinit();
    var kv = try Host.init(alloc, rnd, 1, 5120, 512, 64);
    defer kv.deinit();
    var cat: [3]mlx.mlx_array = .{ mlx.mlx_array_new(), mlx.mlx_array_new(), mlx.mlx_array_new() };
    defer for (cat) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for ([_][2]mlx.mlx_array{ .{ qa.p.trellis, kv.p.trellis }, .{ qa.p.suh, kv.p.suh }, .{ qa.p.svh, kv.p.svh } }, [_]c_int{ 2, 0, 1 }, 0..) |parts, axis, k| {
        const v = mlx.mlx_vector_array_new_data(&parts, 2);
        defer _ = mlx.mlx_vector_array_free(v);
        try mlx.check(mlx.mlx_concatenate_axis(&cat[k], v, axis, s));
        try mlx.check(mlx.mlx_array_eval(cat[k]));
    }
    var cols: [SHARED_MAX + 1]c_int = @splat(0);
    cols[1] = 1280;
    cols[2] = 1792;
    const sh: Shared = .{ .p = .{ .trellis = cat[0], .suh = cat[1], .svh = cat[2] }, .cols = cols, .count = 2 };
    const steps = bench.envUsize("SUSHI_UBENCH_STEPS", 32);
    const rounds = bench.envUsize("SUSHI_UBENCH_ROUNDS", 11);
    for ([_]c_int{ 1, 5 }) |rows| {
        var x = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x);
        try mlx.check(mlx.mlx_zeros(&x, &[_]c_int{ rows, 5120 }, 2, .bfloat16, s));
        var times: [3][]u64 = undefined;
        for (&times) |*t| t.* = try alloc.alloc(u64, rounds);
        for (0..rounds + 1) |round| for (0..3) |j| {
            const k = (round + j) % 3;
            const ns = if (k == 2) try bench.chainNs(s, x, steps, {}, bench.linkOnly) else try bench.chainNs(s, x, steps, SharedCtx{ .qa = &qa, .kv = &kv, .sh = sh, .dec = dec, .fused = k == 1 }, sharedStep);
            if (round > 0) times[k][round - 1] = ns;
        };
        const per = @as(f64, @floatFromInt(steps)) * 1e3;
        const floor = bench.fastest(times[2]) / per;
        std.debug.print("dsv41-stq wq_a|wkv rows {d}: two launches {d:.1} us, one stacked launch {d:.1} us (link floor removed)\n", .{ rows, bench.fastest(times[0]) / per - floor, bench.fastest(times[1]) / per - floor });
    }
}
