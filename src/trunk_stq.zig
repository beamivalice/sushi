//! Dense trunk projections stored as STQ trellis banks, carried in the slots of an affine triple: `w` is the trellis
//! `[1, in/16, out/16, n]` (U16), `s` the input scales `[1, in]` and `b` the output scales `[1, out]` (F16). A weight
//! struct built for `(w, s, b)` triples therefore serves a mixed pack unchanged, and `Transformer.qmatmul` tells the two
//! apart by `w`'s dtype. The pack side is `docs/pack-format.md#qwen4-stq`: `<base>.{trellis,suh,svh}` under a config
//! `trunk_quant` block.
const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const model_mod = @import("model.zig");
const exl3 = @import("sushi_exl3");

pub const Triple = struct { w: mlx.mlx_array, s: mlx.mlx_array, b: mlx.mlx_array };

/// Is `w` a trellis bank rather than an affine or dense weight? No other weight is U16.
pub fn isStq(w: mlx.mlx_array) bool {
    return w.ctx != null and mlx.mlx_array_dtype(w) == .uint16;
}

/// Output width of the trellis bank `w`.
pub fn outDim(w: mlx.mlx_array) usize {
    return @as(usize, @intCast(mlx.getShape(w)[2])) * 16;
}

/// Up to eight rows take one arm per machine at every width (`rows_exact`), so a row beside others (a batched tick, a
/// verify block) rounds as it does alone. The flag is read when a launch is built, and DeepSeek sets it around its own;
/// returns the value to restore.
fn pinExactRows() bool {
    const prev = exl3.project_rows.rows_exact;
    exl3.project_rows.rows_exact = true;
    return prev;
}

/// `x` `[..., in]` through the bank, in x's dtype, each row rounding as it does alone.
pub fn matmul(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, suh: mlx.mlx_array, svh: mlx.mlx_array, dec: exl3.format.Decode) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const in = sh[sh.len - 1];
    const rows: c_int = @intCast(@divExact(mlx.mlx_array_size(x), @as(usize, @intCast(in))));
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, x, &[_]c_int{ rows, in }, 2, s));
    const prev = pinExactRows();
    defer exl3.project_rows.rows_exact = prev;
    const y = try exl3.linear.linear(s, flat, .{ .trellis = w, .suh = suh, .svh = svh }, dec);
    defer _ = mlx.mlx_array_free(y);
    var shape: [8]c_int = undefined;
    @memcpy(shape[0..sh.len], sh);
    shape[sh.len - 1] = @intCast(outDim(w));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, y, &shape, sh.len, s));
    return out;
}

/// The most rows one stacked launch serves; a wider chunk runs each projection alone.
pub const SHARED_ROWS_MAX: usize = @intCast(exl3.project_rows.PROJECT_ROWS_MAX);

/// Linears that read the same rows, stacked along the output into one bank (`exl3.linear.Shared`).
pub const Shared = exl3.linear.Shared;

/// Stacks trellis projections that read the same input into one bank and points each at its view of it, so one launch
/// serves them all at decode widths (`sharedProject`) while a wider chunk still runs each alone. Null, with nothing
/// changed, unless every part is a trellis bank of the same input width and rate, its width a multiple of 128. The
/// arrays the parts held before are freed from `owned` (the caller drops what else holds them).
pub fn fuse(parts: []const *Triple, owned: *std.ArrayList(mlx.mlx_array), allocator: std.mem.Allocator, s: mlx.mlx_stream) !?Shared {
    if (parts.len < 2 or parts.len > exl3.linear.SHARED_MAX) return null;
    // A copy: the shape lives in the array.
    if (!isStq(parts[0].w) or mlx.getShape(parts[0].w).len != 4) return null;
    const t0: [4]c_int = mlx.getShape(parts[0].w)[0..4].*;
    var cols: [exl3.linear.SHARED_MAX + 1]c_int = @splat(0);
    for (parts, 0..) |part, i| {
        if (!isStq(part.w) or mlx.getShape(part.w).len != 4) return null;
        const t = mlx.getShape(part.w);
        if (t[0] != 1 or t[1] != t0[1] or t[3] != t0[3] or @rem(t[2] * 16, 128) != 0) return null;
        cols[i + 1] = cols[i] + t[2] * 16;
    }
    var pieces: [3][exl3.linear.SHARED_MAX]mlx.mlx_array = undefined;
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
    // Views first, so a failure leaves the parts as they were.
    var views: [exl3.linear.SHARED_MAX]Triple = undefined;
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
    return .{ .p = .{ .trellis = cat[0], .suh = cat[1], .svh = cat[2] }, .cols = cols, .count = parts.len };
}

/// Frees `a` from `owned`, if it holds it.
fn disown(owned: *std.ArrayList(mlx.mlx_array), a: mlx.mlx_array) void {
    for (owned.items, 0..) |held, i| if (held.ctx == a.ctx) {
        _ = mlx.mlx_array_free(owned.swapRemove(i));
        return;
    };
}

/// `x` `[..., in]` through every linear of `sh` in one launch: `outs[i]` is linear i's `[..., width]` in x's dtype,
/// owned by the caller. False, with nothing made, when x has more than `SHARED_ROWS_MAX` rows.
pub fn sharedProject(s: mlx.mlx_stream, x: mlx.mlx_array, sh: Shared, dec: exl3.format.Decode, outs: []mlx.mlx_array) !bool {
    const sx = mlx.getShape(x);
    const in = sx[sx.len - 1];
    const rows: c_int = @intCast(@divExact(mlx.mlx_array_size(x), @as(usize, @intCast(in))));
    if (rows > SHARED_ROWS_MAX) return false;
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, x, &[_]c_int{ rows, in }, 2, s));
    const prev = pinExactRows();
    defer exl3.project_rows.rows_exact = prev;
    try exl3.linear.linearShared(s, flat, sh, dec, outs);
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

/// Drops `<base>.{trellis,suh,svh}` from `weights`, for a bank stacked into another array.
pub fn drop(weights: *model_mod.Weights, base: []const u8) void {
    inline for (.{ "trellis", "suh", "svh" }) |suffix| {
        var buf: [160]u8 = undefined;
        const name = std.fmt.bufPrint(&buf, "{s}." ++ suffix, .{base}) catch return;
        if (weights.take(name)) |a| _ = mlx.mlx_array_free(a);
    }
}

/// `<base>.{trellis,suh,svh}` as a `Triple` whose arrays `owned` takes, or null when the pack stores `<base>` another way.
/// A bank is stored one-bank, `[in/16, out/16, n]` with `[in]` and `[out]` scales, both sides on the 128 grid.
pub fn load(
    weights: *const model_mod.Weights,
    base: []const u8,
    dec: ?exl3.format.Decode,
    owned: *std.ArrayList(mlx.mlx_array),
    allocator: std.mem.Allocator,
    s: mlx.mlx_stream,
) !?Triple {
    var names: [3][160]u8 = undefined;
    const tname = std.fmt.bufPrint(&names[0], "{s}.trellis", .{base}) catch return error.NameTooLong;
    const uname = std.fmt.bufPrint(&names[1], "{s}.suh", .{base}) catch return error.NameTooLong;
    const vname = std.fmt.bufPrint(&names[2], "{s}.svh", .{base}) catch return error.NameTooLong;
    const tr = weights.get(tname) orelse return null;
    if (dec == null) {
        log.err("{s} is a trellis bank and config.json has no trunk_quant\n", .{base});
        return error.TrunkQuantMissing;
    }
    const su = weights.get(uname) orelse return missing(uname);
    const sv = weights.get(vname) orelse return missing(vname);
    const tsh = mlx.getShape(tr);
    const ok = mlx.mlx_array_dtype(tr) == .uint16 and tsh.len == 3 and mlx.mlx_array_dtype(su) == .float16 and
        mlx.mlx_array_dtype(sv) == .float16 and exl3.format.kFromPackedDim(@intCast(tsh[2])) != null;
    const in: c_int = if (tsh.len == 3) tsh[0] * 16 else 0;
    const out: c_int = if (tsh.len == 3) tsh[1] * 16 else 0;
    if (!ok or @rem(in, 128) != 0 or @rem(out, 128) != 0 or mlx.mlx_array_size(su) != @as(usize, @intCast(in)) or
        mlx.mlx_array_size(sv) != @as(usize, @intCast(out)))
    {
        log.err("{s} is not a one-bank trellis (U16 [in/16, out/16, n], F16 scales [in] and [out], both on the 128 grid)\n", .{base});
        return error.TrunkStqLayout;
    }
    const bank_dec = dec.?;
    if (bank_dec.codebook.tabledHashMask() and bank_dec.hashMask(.{ .n = @intCast(tsh[2]) }) == null) {
        log.err("{s} is a rate the trunk_quant codebook table has no row for\n", .{base});
        return error.Exl3CodebookTableRateMissing;
    }
    return .{
        .w = try view(owned, allocator, tr, &.{ 1, tsh[0], tsh[1], tsh[2] }, s),
        .s = try view(owned, allocator, su, &.{ 1, in }, s),
        .b = try view(owned, allocator, sv, &.{ 1, out }, s),
    };
}

fn missing(name: []const u8) error{TrunkStqLayout} {
    log.err("MISSING WEIGHT: {s}\n", .{name});
    return error.TrunkStqLayout;
}

fn view(owned: *std.ArrayList(mlx.mlx_array), allocator: std.mem.Allocator, a: mlx.mlx_array, shape: []const c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&o, a, shape.ptr, shape.len, s));
    errdefer _ = mlx.mlx_array_free(o);
    try owned.append(allocator, o);
    return o;
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
        for (suh) |*v| v.* = exl3.format.f32ToF16Bits((if (rnd.boolean()) @as(f32, 1) else -1) * (0.002 + rnd.float(f32) * 0.02));
        const svh = try alloc.alloc(u16, out);
        for (svh) |*v| v.* = exl3.format.f32ToF16Bits((if (rnd.boolean()) @as(f32, 1) else -1) * (0.5 + rnd.float(f32)));
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
    pub fn dense(self: *const TestBank, alloc: std.mem.Allocator, dec: exl3.format.Decode) ![]f32 {
        const w = try alloc.alloc(f32, self.in * self.out);
        try exl3.linear.decodeDense(alloc, self.words, self.suh, self.svh, self.in, self.out, self.n, dec, w);
        return w;
    }

    /// The dense twin of the bank as the affine loader stores a plain weight: bf16 `[in, out]`, scales and biases empty.
    pub fn twin(self: *const TestBank, alloc: std.mem.Allocator, dec: exl3.format.Decode, s: mlx.mlx_stream) !Triple {
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

test "trunk_stq: a trellis triple multiplies as its host-decoded dense twin does, at every row count" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x57a1);
    const rnd = prng.random();
    const dec = exl3.format.Decode.tabled(.spec1, .w15);
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
        const y = try matmul(s, x, t.w, t.s, t.b, dec);
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

fn putBank(weights: *model_mod.Weights, base: []const u8, bank: *const TestBank) !void {
    const t = bank.tensors();
    inline for (.{ "trellis", "suh", "svh" }, 0..) |suffix, i| {
        const key = try std.fmt.allocPrint(weights.allocator, "{s}." ++ suffix, .{base});
        try weights.map.put(key, t[i]);
    }
}

test "trunk_stq: load views a one-bank trellis as a triple and leaves every other tensor to the affine loader" {
    const s = mlx.gpuStream();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x57a2);
    const bank = try TestBank.init(alloc, prng.random(), 256, 384, 48);
    var weights = model_mod.Weights.init(testing.allocator);
    defer weights.deinit();
    try putBank(&weights, "layers.0.self_attn.q_proj", &bank);
    var owned: std.ArrayList(mlx.mlx_array) = .empty;
    defer {
        for (owned.items) |a| _ = mlx.mlx_array_free(a);
        owned.deinit(testing.allocator);
    }
    const dec = exl3.format.Decode.tabled(.spec1, .w15);
    try testing.expectEqual(@as(?Triple, null), try load(&weights, "layers.0.self_attn.k_proj", dec, &owned, testing.allocator, s));
    const t = (try load(&weights, "layers.0.self_attn.q_proj", dec, &owned, testing.allocator, s)).?;
    try testing.expectEqual(@as(usize, 3), owned.items.len);
    try testing.expect(isStq(t.w));
    try testing.expectEqualSlices(c_int, &.{ 1, 16, 24, 48 }, mlx.getShape(t.w));
    try testing.expectEqualSlices(c_int, &.{ 1, 256 }, mlx.getShape(t.s));
    try testing.expectEqualSlices(c_int, &.{ 1, 384 }, mlx.getShape(t.b));
    try testing.expectEqual(@as(usize, 384), outDim(t.w));
}

test "trunk_stq: load refuses a trellis it cannot serve by name" {
    const s = mlx.gpuStream();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x57a3);
    const rnd = prng.random();
    var owned: std.ArrayList(mlx.mlx_array) = .empty;
    defer owned.deinit(testing.allocator);
    const dec = exl3.format.Decode.tabled(.spec1, .w15);

    var good = model_mod.Weights.init(testing.allocator);
    defer good.deinit();
    try putBank(&good, "w", &try TestBank.init(alloc, rnd, 256, 384, 48));
    try testing.expectError(error.TrunkQuantMissing, load(&good, "w", null, &owned, testing.allocator, s));

    // Off the 128 grid on the output side (a 48-row GDN gate).
    var off = model_mod.Weights.init(testing.allocator);
    defer off.deinit();
    try putBank(&off, "w", &try TestBank.init(alloc, rnd, 256, 48, 48));
    try testing.expectError(error.TrunkStqLayout, load(&off, "w", dec, &owned, testing.allocator, s));

    // A scale tensor missing.
    var torn = model_mod.Weights.init(testing.allocator);
    defer torn.deinit();
    try putBank(&torn, "w", &try TestBank.init(alloc, rnd, 256, 384, 48));
    const gone = torn.map.fetchRemove("w.svh").?;
    _ = mlx.mlx_array_free(gone.value);
    testing.allocator.free(gone.key);
    try testing.expectError(error.TrunkStqLayout, load(&torn, "w", dec, &owned, testing.allocator, s));

    // A rate no trellis tile carries.
    var bad_rate = model_mod.Weights.init(testing.allocator);
    defer bad_rate.deinit();
    try putBank(&bad_rate, "w", &try TestBank.init(alloc, rnd, 256, 384, 7));
    try testing.expectError(error.TrunkStqLayout, load(&bad_rate, "w", dec, &owned, testing.allocator, s));
    try testing.expectEqual(@as(usize, 0), owned.items.len);
}

test "trunk_stq: a row beside others rounds as it does alone, so batched ticks and verify blocks equal serial ones" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x57a4);
    const rnd = prng.random();
    const dec = exl3.format.Decode.tabled(.spec1, .w15);
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
    const block = try matmul(s, x, t.w, t.s, t.b, dec);
    defer _ = mlx.mlx_array_free(block);
    for (0..rows) |r| {
        var xr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xr);
        try mlx.check(mlx.mlx_slice(&xr, x, &[_]c_int{ 0, @intCast(r), 0 }, 3, &[_]c_int{ 1, @intCast(r + 1), 2560 }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
        const alone = try matmul(s, xr, t.w, t.s, t.b, dec);
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
    // The pin is scoped: a caller that runs with the flag off finds it off.
    try testing.expect(!exl3.project_rows.rows_exact);
}

fn bitsEqual(s: mlx.mlx_stream, a: mlx.mlx_array, b: mlx.mlx_array) !bool {
    var eq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eq);
    try mlx.check(mlx.mlx_array_equal(&eq, a, b, false, s));
    var same = false;
    try mlx.check(mlx.mlx_array_item_bool(&same, eq));
    return same;
}

test "trunk_stq: stacked banks serve each linear as its own, and a view run alone is the bank it replaced" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x57a5);
    const rnd = prng.random();
    const dec = exl3.format.Decode.tabled(.spec1, .w15);
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
        const want = try matmul(s, x, o.w, o.s, o.b, dec);
        defer _ = mlx.mlx_array_free(want);
        const got = try matmul(s, x, p.w, p.s, p.b, dec);
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
        try testing.expect(try sharedProject(s, x, sh, dec, &outs));
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
    try testing.expect(!try sharedProject(s, wide, sh, dec, &none));
}

test "trunk_stq: fuse leaves linears it cannot stack as they were" {
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

test "trunk_stq: load refuses a bank at a rate the codebook table has no row for" {
    const s = mlx.gpuStream();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var prng = std.Random.DefaultPrng.init(0x57a7);
    var weights = model_mod.Weights.init(testing.allocator);
    defer weights.deinit();
    // STD1 stops at K4: a K6 bank under it has no pair to decode with.
    try putBank(&weights, "w", &try TestBank.init(arena.allocator(), prng.random(), 256, 384, 96));
    var owned: std.ArrayList(mlx.mlx_array) = .empty;
    defer owned.deinit(testing.allocator);
    try testing.expectError(error.Exl3CodebookTableRateMissing, load(&weights, "w", exl3.format.Decode.tabled(.std1, .w15), &owned, testing.allocator, s));
    try testing.expect((try load(&weights, "w", exl3.format.Decode.tabled(.spec1, .w15), &owned, testing.allocator, s)) != null);
    try testing.expectEqual(@as(usize, 3), owned.items.len);
    for (owned.items) |a| _ = mlx.mlx_array_free(a);
}
