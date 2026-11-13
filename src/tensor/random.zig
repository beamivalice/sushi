//! MLX's random numbers on the engine: `RandomBits::eval_gpu` (mlx/backend/metal/primitives.cpp) over the compiled
//! `rbits` kernels, and the op chains of mlx/random.cpp above it (uniform from bits, Gumbel, categorical), composed op
//! for op so the bytes are MLX's for a key. Ported from MLX (MIT, see NOTICE).
//!
//! The engine keeps no key sequence and never asks MLX for a key or for bits: a draw takes the key it is given, and the
//! engine's own keys are `drawKey`, a pure function of a request's root and the position a draw predicts.

const std = @import("std");
const core = @import("core.zig");
const ops = @import("ops.zig");
const scan = @import("scan.zig");
const sort = @import("sort.zig");
const stock = @import("stock.zig");

const Tensor = core.Tensor;
const Dtype = core.Dtype;
const Error = core.Error;

fn scalar(comptime T: type, value: T, dtype: Dtype) Error!*Tensor {
    const v = value;
    return core.fromHost(std.mem.asBytes(&v), dtype, &.{});
}

fn bitsType(width: c_int) Error!Dtype {
    return switch (width) {
        4 => .uint32,
        2 => .uint16,
        1 => .uint8,
        else => error.TensorDtype,
    };
}

/// `random::bits`: `shape` of unsigned integers of `width` bytes from `k` (uint32 `[2]`).
pub fn bits(shape: []const c_int, width: c_int, k: *Tensor) Error!*Tensor {
    const dtype = try bitsType(width);
    if (k.dtype != .uint32 or k.ndim != 1 or k.shape[0] != 2) return error.TensorShape;
    const out = try core.fresh(dtype, shape);
    errdefer core.release(out);
    if (out.size() == 0) return out;
    const keys = try ops.contiguous(k);
    defer core.release(keys);
    const bytes_per_key: u64 = out.nbytes();
    const per_key = (bytes_per_key + 3) / 4;
    const half = per_key / 2;
    const odd = per_key % 2;
    const pipeline = try stock.kernel("rbitsc");
    const p = try core.pass();
    p.pipeline(pipeline);
    core.bind(p, keys, 0);
    core.bindOut(p, out, 1);
    p.constant(odd == 1, 2);
    p.constant(bytes_per_key, 3);
    core.dispatch(p, .{ .x = 1, .y = half + odd }, .{ .x = 1, .y = @min(half + odd, 256) });
    return out;
}

/// The key of the draw that predicts generated position `position` of a request whose root is `root` (its seed, or a
/// number the OS gave it). For one root the map from position to key is one-to-one: `x * odd` and `x + c` are
/// bijections of u64 and `mix`, splitmix64's finalizer, is one too. Two roots share a key only where
/// `mix(r1) + p1 * g == mix(r2) + p2 * g`, which for distinct roots and positions below 2^32 is a chance of about 2^-31.
pub fn drawKey(root: u64, position: u64) [2]u32 {
    const k = mix(mix(root ^ root_domain) +% position *% position_step);
    return .{ @truncate(k), @truncate(k >> 32) };
}

const root_domain: u64 = 0x5EED_D2A5_7DEC_0DE5;
const position_step: u64 = 0x9E37_79B9_7F4A_7C15;

fn mix(x: u64) u64 {
    var z = x;
    z = (z ^ (z >> 30)) *% 0xBF58_476D_1CE4_E5B9;
    z = (z ^ (z >> 27)) *% 0x94D0_49BB_1331_11EB;
    return z ^ (z >> 31);
}

fn fill(value: f32, dtype: Dtype) Error!*Tensor {
    const as_f32 = try scalar(f32, value, .float32);
    defer core.release(as_f32);
    return ops.astype(as_f32, dtype);
}

/// The largest value below one in `dtype`, as a float32 scalar (`random::below_one`).
fn upperBound(dtype: Dtype) Error!*Tensor {
    return switch (dtype) {
        .float32 => scalar(f32, std.math.nextAfter(f32, 1.0, 0.0), .float32),
        .float16 => scalar(f32, @as(f16, @bitCast(@as(u16, 0x3BFF))), .float32),
        .bfloat16 => scalar(f32, @as(f32, @bitCast(@as(u32, 0x3F7F) << 16)), .float32),
        else => error.TensorDtype,
    };
}

/// `random::uniform`: values in `[low, high)` of `shape`, built from the bits as MLX builds them.
pub fn uniform(low: *Tensor, high: *Tensor, shape: []const c_int, dtype: Dtype, key: *Tensor) Error!*Tensor {
    const lo = try ops.astype(low, dtype);
    defer core.release(lo);
    const hi = try ops.astype(high, dtype);
    defer core.release(hi);
    const range = try ops.binary(.subtract, hi, lo);
    defer core.release(range);
    if (!std.mem.eql(c_int, range.dims(), shape) and range.size() != 1) return error.TensorShape;
    const upper = try upperBound(dtype);
    defer core.release(upper);
    const maxval = try scalar(f32, @floatFromInt(std.math.maxInt(u32)), .float32);
    defer core.release(maxval);
    const raw = try bits(shape, 4, key);
    defer core.release(raw);
    const raw_f32 = try ops.astype(raw, .float32);
    defer core.release(raw_f32);
    const unit = try ops.binary(.divide, raw_f32, maxval);
    defer core.release(unit);
    const capped = try ops.binary(.minimum, unit, upper);
    defer core.release(capped);
    const drawn = try ops.astype(capped, dtype);
    defer core.release(drawn);
    const scaled = try ops.binary(.multiply, range, drawn);
    defer core.release(scaled);
    return ops.binary(.add, scaled, lo);
}

fn uniformUnit(shape: []const c_int, dtype: Dtype, key: *Tensor) Error!*Tensor {
    const zero = try fill(0, dtype);
    defer core.release(zero);
    const one = try fill(1, dtype);
    defer core.release(one);
    return uniform(zero, one, shape, dtype, key);
}

/// `-log(-log(uniform))`.
fn gumbel(shape: []const c_int, key: *Tensor) Error!*Tensor {
    const u = try uniformUnit(shape, .float32, key);
    defer core.release(u);
    const a = try ops.unary(.log, u);
    defer core.release(a);
    const b = try ops.unary(.negative, a);
    defer core.release(b);
    const c = try ops.unary(.log, b);
    defer core.release(c);
    return ops.unary(.negative, c);
}

/// `random::categorical` over `axis`: one draw per row of the other axes. A single row is drawn by inverse CDF, as MLX
/// does; a block of rows by the Gumbel-max trick.
pub fn categorical(logits: *Tensor, axis_in: c_int, key: *Tensor) Error!*Tensor {
    if (logits.ndim == 0) return error.TensorShape;
    const axis: usize = @intCast(if (axis_in < 0) axis_in + @as(c_int, logits.ndim) else axis_in);
    if (axis >= logits.ndim) return error.TensorShape;
    switch (logits.dtype) {
        .float16, .bfloat16, .float32 => {},
        else => return error.TensorDtype,
    }
    var rest: [core.max_dims]c_int = undefined;
    var n_rest: usize = 0;
    for (logits.dims(), 0..) |d, i| {
        if (i == axis) continue;
        rest[n_rest] = d;
        n_rest += 1;
    }
    const n: usize = @intCast(logits.shape[axis]);
    if (n > 0 and logits.size() == n and n_rest > 0) return inverseCdf(logits, rest[0..n_rest], key);

    const g = try gumbel(logits.dims(), key);
    defer core.release(g);
    const wide = try ops.astype(logits, .float32);
    defer core.release(wide);
    const noisy = try ops.binary(.add, g, wide);
    defer core.release(noisy);
    return ops.argmax(noisy, @intCast(axis), false);
}

/// `categorical_inverse_cdf`: `shape` draws from the one row `logits`.
fn inverseCdf(logits: *Tensor, shape: []const c_int, key: *Tensor) Error!*Tensor {
    const n: c_int = @intCast(logits.size());
    const flat = try ops.reshape(logits, &.{n});
    defer core.release(flat);
    const x = try ops.astype(flat, .float32);
    defer core.release(x);
    const m = try ops.reduce(.max, x, 0, false);
    defer core.release(m);
    const inf = try scalar(f32, std.math.inf(f32), .float32);
    defer core.release(inf);
    const ninf = try scalar(f32, -std.math.inf(f32), .float32);
    defer core.release(ninf);
    const pos = try ops.binary(.equal, m, inf);
    defer core.release(pos);
    const neg = try ops.binary(.equal, m, ninf);
    defer core.release(neg);
    const is_inf = try ops.binary(.logical_or, pos, neg);
    defer core.release(is_inf);

    const at_max = try ops.binary(.equal, x, m);
    defer core.release(at_max);
    const ties = try ops.astype(at_max, .float32);
    defer core.release(ties);
    const shifted = try ops.binary(.subtract, x, m);
    defer core.release(shifted);
    const ex = try ops.unary(.exp, shifted);
    defer core.release(ex);
    const w = try ops.where(is_inf, ties, ex);
    defer core.release(w);

    const cdf = try scan.cumsum(w, 0, false, false);
    defer core.release(cdf);
    const u = try uniformUnit(shape, .float32, key);
    defer core.release(u);
    const total = try ops.reduce(.sum, w, 0, false);
    defer core.release(total);
    const target = try ops.binary(.multiply, u, total);
    defer core.release(target);
    const below = try sort.searchSorted(cdf, target, true);
    defer core.release(below);
    const one = try scalar(u32, 1, .uint32);
    defer core.release(one);
    return ops.binary(.subtract, below, one);
}
