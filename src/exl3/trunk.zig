//! Dense trunk projections stored as STQ trellis banks, carried in the slots of an affine triple: `w` is the trellis
//! `[1, in/16, out/16, n]` (U16), `s` the input scales `[1, in]` and `b` the output scales `[1, out]` (F16). A weight
//! struct built for `(w, s, b)` triples therefore serves a mixed pack unchanged. The MLX math lives here, free of any
//! weight map, so a host with its own loader serves a trunk bank through the same code. The pack side is
//! `docs/pack-format.md#qwen4-stq`: `<base>.{trellis,suh,svh}` under a config `trunk_quant` block.
const std = @import("std");
const mlx = @import("mlx_host").mlx;
const format = @import("expert_exl3.zig");
const kernels = @import("expert_exl3_kernels.zig");
const project_rows = @import("project_rows.zig");
const linear = @import("linear.zig");

pub const Triple = struct { w: mlx.mlx_array, s: mlx.mlx_array, b: mlx.mlx_array };

/// Is `w` a trellis bank rather than an affine or dense weight? No other weight is U16.
pub fn isStq(w: mlx.mlx_array) bool {
    return w.ctx != null and mlx.mlx_array_dtype(w) == .uint16;
}

/// Output width of the trellis bank `w`.
pub fn outDim(w: mlx.mlx_array) usize {
    return @as(usize, @intCast(mlx.getShape(w)[2])) * 16;
}

/// The rows of `[..., in]`, from the shape already read rather than another FFI size call.
fn leadingRows(sh: []const c_int) c_int {
    var rows: c_int = 1;
    for (sh[0 .. sh.len - 1]) |d| rows *= d;
    return rows;
}

/// Up to eight rows take one arm per machine at every width (`exact`), so a row beside others (a batched tick, a
/// verify block) rounds as it does alone; a spec verify (`wide`) keeps that arm up to 16 rows. The bill of a trunk
/// launch holds the decoded path's scratch.
fn launchOpts(wide: bool) linear.Opts {
    return .{ .rows = .{ .exact = true, .wide = wide }, .decoded_billed = true };
}

/// `x` `[..., in]` through the bank, in x's dtype, each row rounding as it does alone; `wide` marks a spec verify.
pub fn matmul(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, suh: mlx.mlx_array, svh: mlx.mlx_array, dec: format.Decode, wide: bool) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const in = sh[sh.len - 1];
    const rows = leadingRows(sh);
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, x, &[_]c_int{ rows, in }, 2, s));
    const y = try linear.linearWith(s, flat, .{ .trellis = w, .suh = suh, .svh = svh }, dec, launchOpts(wide));
    defer _ = mlx.mlx_array_free(y);
    var shape: [8]c_int = undefined;
    @memcpy(shape[0..sh.len], sh);
    shape[sh.len - 1] = @intCast(outDim(w));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, y, &shape, sh.len, s));
    return out;
}

/// `matmul` for a bank whose launches `geo` holds (`linear.Geo.get` on its trellis, read once at load): the weight's
/// shapes and the launch's configs are not read again, so a bf16 `x` costs its own shape query and a dtype query.
/// A call `geo` does not serve (rows past the multi-row arm, another dtype) is `matmul`.
pub fn matmulGeo(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, suh: mlx.mlx_array, svh: mlx.mlx_array, geo: *const linear.Geo, dec: format.Decode, wide: bool) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const in = sh[sh.len - 1];
    const rows = leadingRows(sh);
    if (in != geo.in or mlx.mlx_array_dtype(x) != .bfloat16) return matmul(s, x, w, suh, svh, dec, wide);
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, x, &[_]c_int{ rows, in }, 2, s));
    const y = (try linear.linearGeo(s, flat, rows, .{ .trellis = w, .suh = suh, .svh = svh }, geo, dec, launchOpts(wide))) orelse return matmul(s, x, w, suh, svh, dec, wide);
    defer _ = mlx.mlx_array_free(y);
    var shape: [8]c_int = undefined;
    @memcpy(shape[0..sh.len], sh);
    shape[sh.len - 1] = geo.out;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, y, &shape, sh.len, s));
    return out;
}

/// The most rows one stacked launch serves outside a wide verify; a wider chunk runs each projection alone.
pub const SHARED_ROWS_MAX: usize = @intCast(project_rows.PROJECT_ROWS_MAX);

/// Linears that read the same rows, stacked along the output into one bank (`linear.Shared`).
pub const Shared = linear.Shared;

/// Stacks trellis projections that read the same input into one bank and points each at its view of it, so one launch
/// serves them all at decode widths (`sharedProject`) while a wider chunk still runs each alone. Null, with nothing
/// changed, unless every part is a trellis bank of the same input width and rate, its width a multiple of 128. The
/// arrays the parts held before are freed from `owned` (the caller drops what else holds them).
pub fn fuse(parts: []const *Triple, owned: *std.ArrayList(mlx.mlx_array), allocator: std.mem.Allocator, s: mlx.mlx_stream) !?Shared {
    if (parts.len < 2 or parts.len > linear.SHARED_MAX) return null;
    // A copy: the shape lives in the array.
    if (!isStq(parts[0].w) or mlx.getShape(parts[0].w).len != 4) return null;
    const t0: [4]c_int = mlx.getShape(parts[0].w)[0..4].*;
    var cols: [linear.SHARED_MAX + 1]c_int = @splat(0);
    for (parts, 0..) |part, i| {
        if (!isStq(part.w) or mlx.getShape(part.w).len != 4) return null;
        const t = mlx.getShape(part.w);
        if (t[0] != 1 or t[1] != t0[1] or t[3] != t0[3] or @rem(t[2] * 16, 128) != 0) return null;
        cols[i + 1] = cols[i] + t[2] * 16;
    }
    var pieces: [3][linear.SHARED_MAX]mlx.mlx_array = undefined;
    for (parts, 0..) |part, i| pieces[0][i] = part.w;
    for (parts, 0..) |part, i| pieces[1][i] = part.s;
    for (parts, 0..) |part, i| pieces[2][i] = part.b;
    var cat: [3]mlx.mlx_array = @splat(.{ .ctx = null });
    var made: usize = 0;
    errdefer for (cat[0..made]) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (&cat, 0..) |*slot, k| {
        const v = mlx.mlx_vector_array_new_data(&pieces[k], parts.len);
        defer _ = mlx.mlx_vector_array_free(v);
        slot.* = mlx.mlx_array_new();
        made += 1;
        try mlx.check(mlx.mlx_concatenate_axis(slot, v, @as(c_int, if (k == 0) 2 else if (k == 1) 0 else 1), s));
    }
    const ev = mlx.mlx_vector_array_new_data(&cat, 3);
    defer _ = mlx.mlx_vector_array_free(ev);
    try mlx.check(mlx.mlx_eval(ev));
    var bounds: [3]c_int = @splat(cols[parts.len]);
    for (1..parts.len) |i| bounds[i - 1] = cols[i];
    const geo = try linear.Geo.get(cat[0], bounds, .{ .exact = true });
    // Views first, so a failure leaves the parts as they were.
    var views: [linear.SHARED_MAX]Triple = undefined;
    var viewed: usize = 0;
    errdefer for (views[0..viewed]) |v| for ([_]mlx.mlx_array{ v.w, v.s, v.b }) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (parts, 0..) |_, i| {
        const c0 = cols[i];
        const c1 = cols[i + 1];
        views[i] = .{ .w = mlx.mlx_array_new(), .s = mlx.mlx_array_new(), .b = mlx.mlx_array_new() };
        viewed += 1;
        const one = [_]c_int{ 1, 1, 1, 1 };
        try mlx.check(mlx.mlx_slice(&views[i].w, cat[0], &[_]c_int{ 0, 0, @divExact(c0, 16), 0 }, 4, &[_]c_int{ 1, t0[1], @divExact(c1, 16), t0[3] }, 4, &one, 4, s));
        try mlx.check(mlx.mlx_slice(&views[i].s, cat[1], &[_]c_int{ @intCast(i), 0 }, 2, &[_]c_int{ @intCast(i + 1), t0[1] * 16 }, 2, &one, 2, s));
        try mlx.check(mlx.mlx_slice(&views[i].b, cat[2], &[_]c_int{ 0, c0 }, 2, &[_]c_int{ 1, c1 }, 2, &one, 2, s));
    }
    try owned.ensureUnusedCapacity(allocator, 3 + 3 * parts.len);
    for (parts, views[0..parts.len]) |part, stacked| {
        for ([_]mlx.mlx_array{ part.w, part.s, part.b }) |old| disown(owned, old);
        part.* = stacked;
        owned.appendSliceAssumeCapacity(&.{ stacked.w, stacked.s, stacked.b });
    }
    owned.appendSliceAssumeCapacity(&cat);
    return .{ .p = .{ .trellis = cat[0], .suh = cat[1], .svh = cat[2] }, .cols = cols, .count = parts.len, .geo = geo };
}

/// Frees `a` from `owned`, if it holds it.
fn disown(owned: *std.ArrayList(mlx.mlx_array), a: mlx.mlx_array) void {
    for (owned.items, 0..) |held, i| if (held.ctx == a.ctx) {
        _ = mlx.mlx_array_free(owned.swapRemove(i));
        return;
    };
}

/// `x` `[..., in]` through every linear of `sh` in one launch: `outs[i]` is linear i's `[..., width]` in x's dtype,
/// owned by the caller. False, with nothing made, when x has more rows than a multi-row launch serves (`wide`: a spec
/// verify).
pub fn sharedProject(s: mlx.mlx_stream, x: mlx.mlx_array, sh: Shared, dec: format.Decode, outs: []mlx.mlx_array, wide: bool) !bool {
    const sx = mlx.getShape(x);
    const in = sx[sx.len - 1];
    const rows = leadingRows(sx);
    const opts = launchOpts(wide);
    if (rows > project_rows.rowsMax(opts.rows)) return false;
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, x, &[_]c_int{ rows, in }, 2, s));
    // The geo's launches are bf16's; another dtype reads its shapes as `linearSharedWith` does.
    const geo_ok = if (sh.geo) |g| in == g.in and mlx.mlx_array_dtype(x) == .bfloat16 else false;
    if (!geo_ok or !try linear.linearSharedGeo(s, flat, rows, sh, dec, outs, opts)) try linear.linearSharedWith(s, flat, sh, dec, outs, opts);
    errdefer for (outs) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (outs, 0..) |*o, i| {
        var shape: [8]c_int = undefined;
        @memcpy(shape[0..sx.len], sx);
        shape[sx.len - 1] = sh.cols[i + 1] - sh.cols[i];
        var r = mlx.mlx_array_new();
        mlx.check(mlx.mlx_reshape(&r, o.*, &shape, sx.len, s)) catch |e| {
            _ = mlx.mlx_array_free(r);
            return e;
        };
        _ = mlx.mlx_array_free(o.*);
        o.* = r;
    }
    return true;
}

/// Test seam: a random trellis bank and its host decode.
pub const TestBank = struct {
    words: []u16,
    suh: []u16,
    svh: []u16,
    in: usize,
    out: usize,
    n: u32,

    pub fn init(alloc: std.mem.Allocator, rnd: std.Random, in: usize, out: usize, n: u32) !TestBank {
        const words = try alloc.alloc(u16, (in / 16) * (out / 16) * n);
        for (words) |*v| v.* = @truncate(rnd.int(u32));
        const suh = try alloc.alloc(u16, in);
        for (suh) |*v| v.* = format.f32ToF16Bits((if (rnd.boolean()) @as(f32, 1) else -1) * (0.002 + rnd.float(f32) * 0.02));
        const svh = try alloc.alloc(u16, out);
        for (svh) |*v| v.* = format.f32ToF16Bits((if (rnd.boolean()) @as(f32, 1) else -1) * (0.5 + rnd.float(f32)));
        return .{ .words = words, .suh = suh, .svh = svh, .in = in, .out = out, .n = n };
    }

    /// The stored tensors: trellis `[in/16, out/16, n]`, `suh [in]`, `svh [out]`.
    pub fn tensors(self: *const TestBank) [3]mlx.mlx_array {
        return .{
            mlx.mlx_array_new_data(self.words.ptr, &[_]c_int{ @intCast(self.in / 16), @intCast(self.out / 16), @intCast(self.n) }, 3, .uint16),
            mlx.mlx_array_new_data(self.suh.ptr, &[_]c_int{@intCast(self.in)}, 1, .float16),
            mlx.mlx_array_new_data(self.svh.ptr, &[_]c_int{@intCast(self.out)}, 1, .float16),
        };
    }

    /// The triple a loader hands the weight structs (the caller frees each array).
    pub fn triple(self: *const TestBank, s: mlx.mlx_stream) !Triple {
        const raw = self.tensors();
        defer for (raw) |a| {
            _ = mlx.mlx_array_free(a);
        };
        var t: Triple = .{ .w = mlx.mlx_array_new(), .s = mlx.mlx_array_new(), .b = mlx.mlx_array_new() };
        try mlx.check(mlx.mlx_reshape(&t.w, raw[0], &[_]c_int{ 1, @intCast(self.in / 16), @intCast(self.out / 16), @intCast(self.n) }, 4, s));
        try mlx.check(mlx.mlx_reshape(&t.s, raw[1], &[_]c_int{ 1, @intCast(self.in) }, 2, s));
        try mlx.check(mlx.mlx_reshape(&t.b, raw[2], &[_]c_int{ 1, @intCast(self.out) }, 2, s));
        return t;
    }

    /// W^T `[in, out]` in f32 as the kernels apply it.
    pub fn dense(self: *const TestBank, alloc: std.mem.Allocator, dec: format.Decode) ![]f32 {
        const w = try alloc.alloc(f32, self.in * self.out);
        try linear.decodeDense(alloc, self.words, self.suh, self.svh, self.in, self.out, self.n, dec, w);
        return w;
    }

    /// The dense twin of the bank as the affine loader stores a plain weight: bf16 `[in, out]`, scales and biases empty.
    pub fn twin(self: *const TestBank, alloc: std.mem.Allocator, dec: format.Decode, s: mlx.mlx_stream) !Triple {
        const w = try self.dense(alloc, dec);
        const f = mlx.mlx_array_new_data(w.ptr, &[_]c_int{ @intCast(self.in), @intCast(self.out) }, 2, .float32);
        defer _ = mlx.mlx_array_free(f);
        var t: Triple = .{ .w = mlx.mlx_array_new(), .s = mlx.mlx_array_new(), .b = mlx.mlx_array_new() };
        try mlx.check(mlx.mlx_astype(&t.w, f, .bfloat16, s));
        return t;
    }
};

// ---- Tests ----

const testing = std.testing;

fn bf16Bits(v: f32) u16 {
    const b: u32 = @bitCast(v);
    return @intCast((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}

fn bf16ToF32(b: u16) f32 {
    return @bitCast(@as(u32, b) << 16);
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

/// The kernels' bar over the f32 truth (`exl3/linear.zig`) plus a bf16 store's own rounding.
const MATMUL_BAR: f64 = 1.5e-3 + 4e-3;

test "exl3 trunk: a trellis triple multiplies as its host-decoded dense twin does, at every row count" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x57a1);
    const rnd = prng.random();
    const dec = format.Decode.tabled(.spec1, .w15);
    const in: usize = 256;
    const out: usize = 384;
    const bank = try TestBank.init(alloc, rnd, in, out, 64);
    const twin = try bank.dense(alloc, dec);
    const t = try bank.triple(s);
    defer for ([_]mlx.mlx_array{ t.w, t.s, t.b }) |a| {
        _ = mlx.mlx_array_free(a);
    };
    try testing.expect(isStq(t.w));
    try testing.expectEqual(out, outDim(t.w));
    for ([_]usize{ 1, 2, 5, 8, 40 }) |rows| {
        const xs = try alloc.alloc(f32, rows * in);
        const xb = try alloc.alloc(u16, xs.len);
        for (xs, xb) |*v, *b| {
            b.* = bf16Bits((rnd.float(f32) * 2 - 1) * 30);
            v.* = bf16ToF32(b.*);
        }
        const x = mlx.mlx_array_new_data(xb.ptr, &[_]c_int{ 1, @intCast(rows), @intCast(in) }, 3, .bfloat16);
        defer _ = mlx.mlx_array_free(x);
        const y = try matmul(s, x, t.w, t.s, t.b, dec, false);
        defer _ = mlx.mlx_array_free(y);
        try testing.expectEqualSlices(c_int, &.{ 1, @intCast(rows), @intCast(out) }, mlx.getShape(y));
        try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(y));
        var y32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(y32);
        try mlx.check(mlx.mlx_astype(&y32, y, .float32, s));
        try mlx.check(mlx.mlx_array_eval(y32));
        const got = (mlx.mlx_array_data_float32(y32) orelse return error.Unreadable)[0 .. rows * out];
        const want = try alloc.alloc(f32, rows * out);
        @memset(want, 0);
        for (0..rows) |r| for (0..in) |i| for (0..out) |j| {
            want[r * out + j] += xs[r * in + i] * twin[i * out + j];
        };
        try testing.expect(relRms(got, want) < MATMUL_BAR);
    }
}

test "exl3 trunk: a row beside others rounds as it does alone, so batched ticks and verify blocks equal serial ones" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x57a4);
    const rnd = prng.random();
    const dec = format.Decode.tabled(.spec1, .w15);
    const bank = try TestBank.init(alloc, rnd, 2560, 12288, 48);
    const t = try bank.triple(s);
    defer for ([_]mlx.mlx_array{ t.w, t.s, t.b }) |a| {
        _ = mlx.mlx_array_free(a);
    };
    const rows = 5;
    const xb = try alloc.alloc(u16, rows * 2560);
    for (xb) |*b| b.* = bf16Bits((rnd.float(f32) * 2 - 1) * 3);
    const x = mlx.mlx_array_new_data(xb.ptr, &[_]c_int{ 1, rows, 2560 }, 3, .bfloat16);
    defer _ = mlx.mlx_array_free(x);
    const block = try matmul(s, x, t.w, t.s, t.b, dec, false);
    defer _ = mlx.mlx_array_free(block);
    for (0..rows) |r| {
        var xr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xr);
        try mlx.check(mlx.mlx_slice(&xr, x, &[_]c_int{ 0, @intCast(r), 0 }, 3, &[_]c_int{ 1, @intCast(r + 1), 2560 }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        const alone = try matmul(s, xr, t.w, t.s, t.b, dec, false);
        defer _ = mlx.mlx_array_free(alone);
        var want = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(want);
        try mlx.check(mlx.mlx_slice(&want, block, &[_]c_int{ 0, @intCast(r), 0 }, 3, &[_]c_int{ 1, @intCast(r + 1), 12288 }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        var eq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(eq);
        try mlx.check(mlx.mlx_array_equal(&eq, alone, want, false, s));
        var same = false;
        try mlx.check(mlx.mlx_array_item_bool(&same, eq));
        try testing.expect(same);
    }
}

fn bitsEqual(s: mlx.mlx_stream, a: mlx.mlx_array, b: mlx.mlx_array) !bool {
    var eq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eq);
    try mlx.check(mlx.mlx_array_equal(&eq, a, b, false, s));
    var same = false;
    try mlx.check(mlx.mlx_array_item_bool(&same, eq));
    return same;
}

test "exl3 trunk: stacked banks serve each linear as its own, and a view run alone is the bank it replaced" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x57a5);
    const rnd = prng.random();
    const dec = format.Decode.tabled(.spec1, .w15);
    const widths = [3]usize{ 256, 128, 128 };
    var twins: [3][]f32 = undefined;
    var orig: [3]Triple = undefined;
    var parts: [3]Triple = undefined;
    for (widths, &twins, &orig, &parts) |w, *twin, *o, *p| {
        const bank = try TestBank.init(alloc, rnd, 256, w, 48);
        twin.* = try bank.dense(alloc, dec);
        o.* = try bank.triple(s);
        p.* = o.*;
    }
    defer for (orig) |o| for ([_]mlx.mlx_array{ o.w, o.s, o.b }) |a| {
        _ = mlx.mlx_array_free(a);
    };
    var owned: std.ArrayList(mlx.mlx_array) = .empty;
    defer {
        for (owned.items) |a| _ = mlx.mlx_array_free(a);
        owned.deinit(testing.allocator);
    }
    const sh = (try fuse(&.{ &parts[0], &parts[1], &parts[2] }, &owned, testing.allocator, s)).?;
    try testing.expectEqual(@as(usize, 3), sh.count);
    try testing.expectEqual(@as(c_int, 512), sh.cols[3]);

    // A view run alone gives the bytes of the bank it replaced, in a decode tick and in a prefill chunk.
    for ([_]c_int{ 1, 20 }) |rows| for (orig, parts) |o, p| {
        const xb = try alloc.alloc(u16, @as(usize, @intCast(rows)) * 256);
        for (xb) |*b| b.* = bf16Bits((rnd.float(f32) * 2 - 1) * 3);
        const x = mlx.mlx_array_new_data(xb.ptr, &[_]c_int{ 1, rows, 256 }, 3, .bfloat16);
        defer _ = mlx.mlx_array_free(x);
        const want = try matmul(s, x, o.w, o.s, o.b, dec, false);
        defer _ = mlx.mlx_array_free(want);
        const got = try matmul(s, x, p.w, p.s, p.b, dec, false);
        defer _ = mlx.mlx_array_free(got);
        try testing.expect(try bitsEqual(s, got, want));
    };

    // The stacked launch holds each linear's bar, and a row of it is the row alone.
    var solo: [3]mlx.mlx_array = undefined;
    var xb8: [8 * 256]u16 = undefined;
    for (&xb8) |*b| b.* = bf16Bits((rnd.float(f32) * 2 - 1) * 3);
    for (1..SHARED_ROWS_MAX + 1) |rows| {
        const x = mlx.mlx_array_new_data(&xb8, &[_]c_int{ 1, @intCast(rows), 256 }, 3, .bfloat16);
        defer _ = mlx.mlx_array_free(x);
        var outs: [3]mlx.mlx_array = undefined;
        try testing.expect(try sharedProject(s, x, sh, dec, &outs, false));
        defer for (outs) |a| {
            _ = mlx.mlx_array_free(a);
        };
        for (outs, widths, twins, 0..) |o, w, twin, i| {
            try testing.expectEqualSlices(c_int, &.{ 1, @intCast(rows), @intCast(w) }, mlx.getShape(o));
            var o32 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(o32);
            try mlx.check(mlx.mlx_astype(&o32, o, .float32, s));
            try mlx.check(mlx.mlx_array_eval(o32));
            const got = (mlx.mlx_array_data_float32(o32) orelse return error.Unreadable)[0 .. rows * w];
            const want = try alloc.alloc(f32, rows * w);
            @memset(want, 0);
            for (0..rows) |r| for (0..256) |k| for (0..w) |j| {
                want[r * w + j] += bf16ToF32(xb8[r * 256 + k]) * twin[k * w + j];
            };
            try testing.expect(relRms(got, want) < MATMUL_BAR);
            if (rows == 1) {
                solo[i] = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_array_set(&solo[i], o));
            } else {
                var first = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(first);
                try mlx.check(mlx.mlx_slice(&first, o, &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ 1, 1, @intCast(w) }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
                try testing.expect(try bitsEqual(s, first, solo[i]));
            }
        }
    }
    defer for (solo) |a| {
        _ = mlx.mlx_array_free(a);
    };
    var zeros: [9 * 256]u16 = @splat(0);
    const wide = mlx.mlx_array_new_data(&zeros, &[_]c_int{ 1, 9, 256 }, 3, .bfloat16);
    defer _ = mlx.mlx_array_free(wide);
    var none: [3]mlx.mlx_array = undefined;
    try testing.expect(!try sharedProject(s, wide, sh, dec, &none, false));
}

test "exl3 trunk: a 9- to 16-row spec verify keeps each row's decode tick bytes, alone and stacked; a prefill chunk keeps the GEMM" {
    try mlx.requireGpu();
    if (!kernels.gemmNaxOn()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x57a8);
    const rnd = prng.random();
    const dec = format.Decode.tabled(.spec1, .w15);
    const t = try (try TestBank.init(alloc, rnd, 2560, 1024, 96)).triple(s);
    var parts: [2]Triple = undefined;
    for (&parts, [_]usize{ 512, 256 }) |*p, w| p.* = try (try TestBank.init(alloc, rnd, 2560, w, 96)).triple(s);
    var owned: std.ArrayList(mlx.mlx_array) = .empty;
    defer {
        for (owned.items) |a| _ = mlx.mlx_array_free(a);
        owned.deinit(testing.allocator);
    }
    defer for ([_]mlx.mlx_array{ t.w, t.s, t.b }) |a| {
        _ = mlx.mlx_array_free(a);
    };
    const orig = parts;
    defer for (orig) |o| for ([_]mlx.mlx_array{ o.w, o.s, o.b }) |arr| {
        _ = mlx.mlx_array_free(arr);
    };
    const sh = (try fuse(&.{ &parts[0], &parts[1] }, &owned, testing.allocator, s)).?;
    const max: usize = @intCast(project_rows.NAX_ROWS_MAX);
    const xb = try alloc.alloc(u16, max * 2560);
    for (xb) |*b| b.* = bf16Bits((rnd.float(f32) * 2 - 1) * 3);
    const xs = mlx.mlx_array_new_data(xb.ptr, &[_]c_int{ 1, @intCast(max), 2560 }, 3, .bfloat16);
    defer _ = mlx.mlx_array_free(xs);

    var pool: std.ArrayList(mlx.mlx_array) = .empty;
    defer {
        for (pool.items) |a| _ = mlx.mlx_array_free(a);
        pool.deinit(testing.allocator);
    }
    const rowsOf = struct {
        fn f(st: mlx.mlx_stream, a: mlx.mlx_array, r0: usize, r1: usize, list: *std.ArrayList(mlx.mlx_array)) !mlx.mlx_array {
            const ash = mlx.getShape(a);
            var out = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(&out, a, &[_]c_int{ 0, @intCast(r0), 0 }, 3, &[_]c_int{ 1, @intCast(r1), ash[2] }, 3, &[_]c_int{ 1, 1, 1 }, 3, st));
            try list.append(testing.allocator, out);
            return out;
        }
    }.f;
    // Serial decode ticks: one row each, the stacked launch and the plain one.
    var alone: [16][3]mlx.mlx_array = undefined;
    for (0..max) |r| {
        const xr = try rowsOf(s, xs, r, r + 1, &pool);
        alone[r][0] = try matmul(s, xr, t.w, t.s, t.b, dec, false);
        try pool.append(testing.allocator, alone[r][0]);
        var outs: [2]mlx.mlx_array = undefined;
        try testing.expect(try sharedProject(s, xr, sh, dec, &outs, false));
        try pool.appendSlice(testing.allocator, &outs);
        alone[r][1] = outs[0];
        alone[r][2] = outs[1];
    }

    for (9..max + 1) |rows| {
        const x = try rowsOf(s, xs, 0, rows, &pool);
        const block = try matmul(s, x, t.w, t.s, t.b, dec, true);
        try pool.append(testing.allocator, block);
        var outs: [2]mlx.mlx_array = undefined;
        try testing.expect(try sharedProject(s, x, sh, dec, &outs, true));
        try pool.appendSlice(testing.allocator, &outs);
        for (0..rows) |r| for ([_]mlx.mlx_array{ block, outs[0], outs[1] }, 0..) |y, i| {
            try testing.expect(try bitsEqual(s, try rowsOf(s, y, r, r + 1, &pool), alone[r][i]));
        };
    }

    var none: [2]mlx.mlx_array = undefined;
    try testing.expect(!try sharedProject(s, try rowsOf(s, xs, 0, 9, &pool), sh, dec, &none, false));
}

test "exl3 trunk: a launch from the weight's geo gives the bytes of the launch that reads its shapes, alone and stacked" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x57b1);
    const rnd = prng.random();
    const dec = format.Decode.tabled(.spec1, .w15);
    const rates = [_]u32{ 48, 96 };
    for (rates) |n| {
        const t = try (try TestBank.init(alloc, rnd, 2560, 1024, n)).triple(s);
        var parts: [2]Triple = undefined;
        for (&parts, [_]usize{ 512, 256 }) |*p, w| p.* = try (try TestBank.init(alloc, rnd, 2560, w, n)).triple(s);
        var owned: std.ArrayList(mlx.mlx_array) = .empty;
        defer {
            for (owned.items) |a| _ = mlx.mlx_array_free(a);
            owned.deinit(testing.allocator);
        }
        defer for ([_]mlx.mlx_array{ t.w, t.s, t.b }) |a| {
            _ = mlx.mlx_array_free(a);
        };
        const orig = parts;
        defer for (orig) |o| for ([_]mlx.mlx_array{ o.w, o.s, o.b }) |arr| {
            _ = mlx.mlx_array_free(arr);
        };
        const sh = (try fuse(&.{ &parts[0], &parts[1] }, &owned, testing.allocator, s)).?;
        try testing.expect(sh.geo != null);
        const geo = try linear.Geo.init(t.w, null, .{ .exact = true });
        try testing.expectEqual(@as(c_int, 2560), geo.in);
        try testing.expectEqual(@as(c_int, 1024), geo.out);
        const max: usize = if (kernels.gemmNaxOn()) @intCast(project_rows.NAX_ROWS_MAX) else @intCast(project_rows.PROJECT_ROWS_MAX);
        const xb = try alloc.alloc(u16, (max + 4) * 2560);
        for (xb) |*b| b.* = bf16Bits((rnd.float(f32) * 2 - 1) * 3);
        for (1..max + 5) |rows| {
            const x = mlx.mlx_array_new_data(xb.ptr, &[_]c_int{ 1, @intCast(rows), 2560 }, 3, .bfloat16);
            defer _ = mlx.mlx_array_free(x);
            for ([_]bool{ false, true }) |wide| {
                const want = try matmul(s, x, t.w, t.s, t.b, dec, wide);
                defer _ = mlx.mlx_array_free(want);
                const got = try matmulGeo(s, x, t.w, t.s, t.b, &geo, dec, wide);
                defer _ = mlx.mlx_array_free(got);
                try testing.expect(try bitsEqual(s, got, want));
                var a: [2]mlx.mlx_array = undefined;
                var b: [2]mlx.mlx_array = undefined;
                var bare = sh;
                bare.geo = null;
                const served = try sharedProject(s, x, bare, dec, &a, wide);
                try testing.expectEqual(served, try sharedProject(s, x, sh, dec, &b, wide));
                if (!served) continue;
                defer for (a ++ b) |arr| {
                    _ = mlx.mlx_array_free(arr);
                };
                for (a, b) |ya, yb| try testing.expect(try bitsEqual(s, ya, yb));
            }
        }
    }
}

test "exl3 trunk: a geo bound with NAX on runs the SIMD arm once the NAX latch is set" {
    try mlx.requireGpu();
    if (!kernels.gemmNaxOn()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x57b3);
    const rnd = prng.random();
    const dec = format.Decode.tabled(.spec1, .w15);
    const t = try (try TestBank.init(alloc, rnd, 2560, 1024, 48)).triple(s);
    defer for ([_]mlx.mlx_array{ t.w, t.s, t.b }) |a| {
        _ = mlx.mlx_array_free(a);
    };
    var parts: [2]Triple = undefined;
    for (&parts, [_]usize{ 512, 256 }) |*p, w| p.* = try (try TestBank.init(alloc, rnd, 2560, w, 48)).triple(s);
    const orig = parts;
    defer for (orig) |o| for ([_]mlx.mlx_array{ o.w, o.s, o.b }) |arr| {
        _ = mlx.mlx_array_free(arr);
    };
    var owned: std.ArrayList(mlx.mlx_array) = .empty;
    defer {
        for (owned.items) |a| _ = mlx.mlx_array_free(a);
        owned.deinit(testing.allocator);
    }
    const sh = (try fuse(&.{ &parts[0], &parts[1] }, &owned, testing.allocator, s)).?;
    const geo = try linear.Geo.init(t.w, null, .{ .exact = true });
    const xb = try alloc.alloc(u16, 8 * 2560);
    for (xb) |*b| b.* = bf16Bits((rnd.float(f32) * 2 - 1) * 3);

    const previous = kernels.gemm_nax_failed;
    defer kernels.gemm_nax_failed = previous;
    kernels.gemm_nax_failed = true;
    for (1..9) |rows| {
        const x = mlx.mlx_array_new_data(xb.ptr, &[_]c_int{ 1, @intCast(rows), 2560 }, 3, .bfloat16);
        defer _ = mlx.mlx_array_free(x);
        const want = try matmul(s, x, t.w, t.s, t.b, dec, false);
        defer _ = mlx.mlx_array_free(want);
        const got = try matmulGeo(s, x, t.w, t.s, t.b, &geo, dec, false);
        defer _ = mlx.mlx_array_free(got);
        try testing.expect(try bitsEqual(s, got, want));
        var a: [2]mlx.mlx_array = undefined;
        var b: [2]mlx.mlx_array = undefined;
        var bare = sh;
        bare.geo = null;
        const served = try sharedProject(s, x, bare, dec, &a, false);
        try testing.expectEqual(served, try sharedProject(s, x, sh, dec, &b, false));
        if (!served) continue;
        defer for (a ++ b) |arr| {
            _ = mlx.mlx_array_free(arr);
        };
        for (a, b) |ya, yb| try testing.expect(try bitsEqual(s, ya, yb));
    }
}

test "exl3 trunk: banks of one shape share one geo, and a geo made after the NAX latch holds no NAX launch" {
    try mlx.requireGpu();
    if (!kernels.gemmNaxOn()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x57b4);
    const rnd = prng.random();
    var t: [2]Triple = undefined;
    for (&t) |*x| x.* = try (try TestBank.init(alloc, rnd, 2560, 1152, 48)).triple(s);
    defer for (t) |x| for ([_]mlx.mlx_array{ x.w, x.s, x.b }) |a| {
        _ = mlx.mlx_array_free(a);
    };
    try testing.expectEqual(try linear.Geo.get(t[0].w, null, .{ .exact = true }), try linear.Geo.get(t[1].w, null, .{ .exact = true }));
    try testing.expect(try linear.Geo.get(t[0].w, null, .{ .exact = true }) != try linear.Geo.get(t[0].w, .{ 512, 1024, 1152 }, .{ .exact = true }));
    const before = try linear.Geo.get(t[0].w, null, .{ .exact = true });
    const previous = kernels.gemm_nax_failed;
    defer kernels.gemm_nax_failed = previous;
    kernels.gemm_nax_failed = true;
    const after = try linear.Geo.get(t[0].w, null, .{ .exact = true });
    try testing.expect(before != after);
    for (after.rows) |r| if (r) |row| try testing.expect(!row.g.nax);
}

test "exl3 trunk: a geo made for the non-exact mode binds the SIMD one-row launch and serves only that mode" {
    try mlx.requireGpu();
    if (!kernels.gemmNaxOn()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x57b5);
    const rnd = prng.random();
    const dec = format.Decode.tabled(.spec1, .w15);
    const t = try (try TestBank.init(alloc, rnd, 2560, 1024, 48)).triple(s);
    defer for ([_]mlx.mlx_array{ t.w, t.s, t.b }) |a| {
        _ = mlx.mlx_array_free(a);
    };
    const p: @import("root.zig").Proj = .{ .trellis = t.w, .suh = t.s, .svh = t.b };
    const simd = try linear.Geo.init(t.w, null, .{});
    const exact = try linear.Geo.init(t.w, null, .{ .exact = true });
    try testing.expect(!simd.row(1).?.g.nax and simd.row(2).?.g.nax);
    try testing.expect(exact.row(1).?.g.nax);
    try testing.expect(try linear.Geo.get(t.w, null, .{}) != try linear.Geo.get(t.w, null, .{ .exact = true }));
    try testing.expectEqual(try linear.Geo.get(t.w, null, .{}), try linear.Geo.get(t.w, null, .{ .wide = true }));
    const xb = try alloc.alloc(u16, 8 * 2560);
    for (xb) |*b| b.* = bf16Bits((rnd.float(f32) * 2 - 1) * 3);
    for (1..9) |rows| {
        const x = mlx.mlx_array_new_data(xb.ptr, &[_]c_int{ @intCast(rows), 2560 }, 2, .bfloat16);
        defer _ = mlx.mlx_array_free(x);
        const want = try linear.linearWith(s, x, p, dec, .{});
        defer _ = mlx.mlx_array_free(want);
        const got = (try linear.linearGeo(s, x, @intCast(rows), p, &simd, dec, .{})) orelse return error.TestExpectedLaunch;
        defer _ = mlx.mlx_array_free(got);
        try testing.expect(try bitsEqual(s, got, want));
        try testing.expect((try linear.linearGeo(s, x, @intCast(rows), p, &simd, dec, .{ .rows = .{ .exact = true } })) == null);
        try testing.expect((try linear.linearGeo(s, x, @intCast(rows), p, &exact, dec, .{})) == null);
    }
}

test "exl3 trunk: a launch of another dtype runs as it did, whatever the geo holds" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x57b2);
    const rnd = prng.random();
    const dec = format.Decode.tabled(.spec1, .w15);
    const t = try (try TestBank.init(alloc, rnd, 256, 384, 64)).triple(s);
    defer for ([_]mlx.mlx_array{ t.w, t.s, t.b }) |a| {
        _ = mlx.mlx_array_free(a);
    };
    const geo = try linear.Geo.init(t.w, null, .{ .exact = true });
    const xb = try alloc.alloc(u16, 3 * 256);
    for (xb) |*b| b.* = format.f32ToF16Bits(rnd.float(f32) * 2 - 1);
    const x = mlx.mlx_array_new_data(xb.ptr, &[_]c_int{ 1, 3, 256 }, 3, .float16);
    defer _ = mlx.mlx_array_free(x);
    const want = try matmul(s, x, t.w, t.s, t.b, dec, false);
    defer _ = mlx.mlx_array_free(want);
    const got = try matmulGeo(s, x, t.w, t.s, t.b, &geo, dec, false);
    defer _ = mlx.mlx_array_free(got);
    try testing.expectEqual(mlx.mlx_dtype.float16, mlx.mlx_array_dtype(got));
    try testing.expect(try bitsEqual(s, got, want));
}

test "exl3 trunk: fuse leaves linears it cannot stack as they were" {
    const s = mlx.gpuStream();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x57a6);
    const rnd = prng.random();
    var owned: std.ArrayList(mlx.mlx_array) = .empty;
    defer owned.deinit(testing.allocator);
    const a = try (try TestBank.init(alloc, rnd, 256, 256, 48)).triple(s);
    var b = try (try TestBank.init(alloc, rnd, 256, 128, 64)).triple(s); // another rate
    var c = try (try TestBank.init(alloc, rnd, 256, 48, 48)).triple(s); // off the 128 grid
    var d: Triple = .{ .w = mlx.mlx_array_new(), .s = mlx.mlx_array_new(), .b = mlx.mlx_array_new() }; // affine or dense
    var a_mut = a;
    defer for ([_]Triple{ a, b, c, d }) |t| for ([_]mlx.mlx_array{ t.w, t.s, t.b }) |arr| {
        _ = mlx.mlx_array_free(arr);
    };
    try testing.expectEqual(@as(?Shared, null), try fuse(&.{ &a_mut, &b }, &owned, testing.allocator, s));
    try testing.expectEqual(@as(?Shared, null), try fuse(&.{ &a_mut, &c }, &owned, testing.allocator, s));
    try testing.expectEqual(@as(?Shared, null), try fuse(&.{ &a_mut, &d }, &owned, testing.allocator, s));
    try testing.expectEqual(@as(?Shared, null), try fuse(&.{&a_mut}, &owned, testing.allocator, s));
    try testing.expectEqual(@as(usize, 0), owned.items.len);
    try testing.expectEqual(a.w.ctx, a_mut.w.ctx);
}
