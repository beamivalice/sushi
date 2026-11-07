//! Test helpers for the f32-truth bars: a bank decoded once to a dense f32 twin on the host, a clamped SwiGLU expert
//! and a routed sum in f64 against which the grouped chains are scored by relative RMS.
const std = @import("std");
const mlx = @import("mlx_host").mlx;
const kernels = @import("expert_exl3_kernels.zig");
const exl3 = @import("expert_exl3.zig");
const rate_groups = @import("rate_groups.zig");
const linear = @import("linear.zig");
const root = @import("root.zig");

pub const LIMIT: c_int = 10;

pub fn relRms(got: []const f32, want: []const f64) f64 {
    var ss: f64 = 0;
    var ref: f64 = 0;
    for (got, want) |a, b| {
        ss += (@as(f64, a) - b) * (@as(f64, a) - b);
        ref += b * b;
    }
    return @sqrt(ss / @max(ref, 1e-30));
}

pub fn evalWords(a: mlx.mlx_array) ![]const u16 {
    try mlx.check(mlx.mlx_array_eval(a));
    return (mlx.mlx_array_data_uint16(a) orelse return error.Unreadable)[0..mlx.mlx_array_size(a)];
}

/// An f16 array's bit patterns.
pub fn evalHalves(a: mlx.mlx_array) ![]const u16 {
    try mlx.check(mlx.mlx_array_eval(a));
    const p = mlx.mlx_array_data_float16(a) orelse return error.Unreadable;
    return @as([*]const u16, @ptrCast(p))[0..mlx.mlx_array_size(a)];
}

pub fn toF32(s: mlx.mlx_stream, a: mlx.mlx_array, out: *mlx.mlx_array) ![]const f32 {
    try mlx.check(mlx.mlx_astype(out, a, .float32, s));
    try mlx.check(mlx.mlx_array_eval(out.*));
    return (mlx.mlx_array_data_float32(out.*) orelse return error.Unreadable)[0..mlx.mlx_array_size(out.*)];
}

/// `x` [rows, in] bf16 and the f32 values it holds: unit-scale rows with a few outlier channels, as a normed residual.
pub fn activations(alloc: std.mem.Allocator, rnd: std.Random, rows: usize, in: usize, scale: f32) !struct { x: mlx.mlx_array, vals: []f32 } {
    const bits = try alloc.alloc(u16, rows * in);
    const vals = try alloc.alloc(f32, rows * in);
    for (bits, vals, 0..) |*b, *v, i| {
        var f = (rnd.floatNorm(f32)) * scale;
        if (i % in % 997 == 3) f *= 30;
        const u: u32 = @bitCast(f);
        b.* = @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
        v.* = @bitCast(@as(u32, b.*) << 16);
    }
    return .{ .x = mlx.mlx_array_new_data(bits.ptr, &[_]c_int{ @intCast(rows), @intCast(in) }, 2, .bfloat16), .vals = vals };
}

/// y = x . W over a dense f32 `[in, out]` in f64.
pub fn matvec(w: []const f32, x: []const f32, out: usize, dst: []f64) void {
    @memset(dst, 0);
    for (x, 0..) |xv, i| {
        if (xv == 0) continue;
        for (w[i * out ..][0..out], dst) |wv, *d| d.* += @as(f64, xv) * wv;
    }
}

/// An expert's dense gate, up and down (f32 `[in, out]` each), decoded once.
pub const Dense = struct {
    w: [3][]f32,

    pub fn of(alloc: std.mem.Allocator, bank: root.Bank, e: usize, hidden: usize, inter: usize, n: [3]u32, dec: exl3.Decode) !Dense {
        var d: Dense = undefined;
        for ([_]root.Proj{ bank.gate, bank.up, bank.down }, 0..) |p, pi| {
            const in = if (pi == 2) inter else hidden;
            const out = if (pi == 2) hidden else inter;
            const tr = try evalWords(p.trellis);
            const su = try evalHalves(p.suh);
            const sv = try evalHalves(p.svh);
            const per = (in / 16) * (out / 16) * n[pi];
            d.w[pi] = try alloc.alloc(f32, in * out);
            try linear.decodeDense(alloc, tr[e * per ..][0..per], su[e * in ..][0..in], sv[e * out ..][0..out], in, out, n[pi], dec, d.w[pi]);
        }
        return d;
    }

    /// The clamped SwiGLU expert on `x` [hidden] in f64: the reference for one routed slot.
    pub fn run(self: Dense, alloc: std.mem.Allocator, x: []const f32, hidden: usize, inter: usize, limit: f64, dst: []f64) !void {
        const g = try alloc.alloc(f64, inter);
        const u = try alloc.alloc(f64, inter);
        matvec(self.w[0], x, inter, g);
        matvec(self.w[1], x, inter, u);
        const act = try alloc.alloc(f32, inter);
        for (act, g, u) |*a, gv, uv| {
            const gc = if (limit > 0) @min(gv, limit) else gv;
            const uc = if (limit > 0) std.math.clamp(uv, -limit, limit) else uv;
            a.* = @floatCast(gc / (1.0 + @exp(-gc)) * uc);
        }
        matvec(self.w[2], act, hidden, dst);
    }
};

pub fn reshaped(s: mlx.mlx_stream, a: mlx.mlx_array, shape: []const c_int) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, a, shape.ptr, shape.len, s));
    return out;
}

/// Routed sum of `rows` rows over experts (flat global ids, `topk` per row) weighted `wts`, through the grouped chain
/// at decode or prefill width, against the f64 truth: returns the relative RMS.
pub fn routedErr(
    alloc: std.mem.Allocator,
    s: mlx.mlx_stream,
    rb: *const rate_groups.Ragged,
    dense: []const Dense,
    rnd: std.Random,
    hidden: usize,
    inter: usize,
    rows: usize,
    topk: usize,
    x_scale: f32,
) !f64 {
    const act = try activations(alloc, rnd, rows, hidden, x_scale);
    defer _ = mlx.mlx_array_free(act.x);
    const ids = try alloc.alloc(u32, rows * topk);
    const wts = try alloc.alloc(f32, rows * topk);
    for (0..rows) |r| {
        var k: usize = 0;
        while (k < topk) {
            const pick = rnd.uintLessThan(u32, rb.experts);
            if (std.mem.indexOfScalar(u32, ids[r * topk ..][0..k], pick) != null) continue;
            ids[r * topk + k] = pick;
            wts[r * topk + k] = 0.05 + rnd.float(f32) * 0.25;
            k += 1;
        }
    }
    const want = try alloc.alloc(f64, rows * hidden);
    @memset(want, 0);
    const tmp = try alloc.alloc(f64, hidden);
    for (0..rows) |r| for (0..topk) |k| {
        try dense[ids[r * topk + k]].run(alloc, act.vals[r * hidden ..][0..hidden], hidden, inter, LIMIT, tmp);
        for (want[r * hidden ..][0..hidden], tmp) |*o, v| o.* += wts[r * topk + k] * v;
    };
    const slots = mlx.mlx_array_new_data(ids.ptr, &[_]c_int{@intCast(rows * topk)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(slots);
    const scores = mlx.mlx_array_new_data(wts.ptr, &[_]c_int{@intCast(rows * topk)}, 1, .float32);
    defer _ = mlx.mlx_array_free(scores);
    const y = if (rows <= kernels.DECODE_ROWS_MAX)
        try rate_groups.moeDecodeClamped(s, act.x, rb, slots, scores, @intCast(topk), .float32, null, null, LIMIT)
    else
        try rate_groups.moePrefillClamped(s, act.x, rb, slots, scores, @intCast(topk), LIMIT);
    defer _ = mlx.mlx_array_free(y);
    var cy = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cy);
    const got = try toF32(s, y, &cy);
    for (got) |v| try std.testing.expect(std.math.isFinite(v));
    return relRms(got, want);
}
