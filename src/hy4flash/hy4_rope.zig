//! Hy4's rope in one launch: the leading dims of each row copied, the trailing `rope` dims rotated in consecutive pairs,
//! in place of `fwd.ropeScaled`'s slice, MLX's rope over a contiguous copy of the tail and the concat. Each rotated pair
//! is MLX's `rope_single` / `rope` arithmetic (`rope.metal`: `inv_freq = exp2(-(i / half) · log2 θ)`, the position
//! times the scale, `fast::cos` and `fast::sin`, the same two products and sum), so a row is the chain's bytes; the
//! output is written at the dtype a caller casts it to next.
const std = @import("std");
const mlx = @import("../mlx.zig");
const Arr = mlx.mlx_array;
const kcfg = @import("hy4_kcfg.zig");

pub const source: [:0]const u8 =
    \\uint j = thread_position_in_grid.x;
    \\uint t = thread_position_in_grid.y;
    \\uint a = thread_position_in_grid.z;
    \\const device T* row = x + (size_t)a * (size_t)x_strides[0] + (size_t)t * (size_t)x_strides[1];
    \\device OT* dst = out + (ROW_MAJOR ? ((size_t)t * (size_t)threads_per_grid.z + (size_t)a) : ((size_t)a * (size_t)threads_per_grid.y + (size_t)t)) * (size_t)(NOPE + 2 * HALF);
    \\size_t s2 = (size_t)x_strides[2];
    \\if (j < uint(NOPE)) {
    \\  dst[j] = static_cast<OT>(row[(size_t)j * s2]);
    \\  return;
    \\}
    \\uint i = j - uint(NOPE);
    \\float d = static_cast<float>(i) / static_cast<float>(HALF);
    \\float inv_freq = metal::exp2(-d * as_type<float>(BASE));
    \\float L = as_type<float>(SCALE) * static_cast<float>(PER_ROW ? pos[t] : (pos[0] + int(t)));
    \\float theta = L * inv_freq;
    \\float costheta = metal::fast::cos(theta);
    \\float sintheta = metal::fast::sin(theta);
    \\float x1 = static_cast<float>(row[(size_t)(NOPE + 2 * i) * s2]);
    \\float x2 = static_cast<float>(row[(size_t)(NOPE + 2 * i + 1) * s2]);
    \\float rx1 = x1 * costheta - x2 * sintheta;
    \\float rx2 = x1 * sintheta + x2 * costheta;
    \\dst[NOPE + 2 * i] = static_cast<OT>(rx1);
    \\dst[NOPE + 2 * i + 1] = static_cast<OT>(rx2);
;

var kernel: ?mlx.mlx_fast_metal_kernel = null;

fn getKernel() !mlx.mlx_fast_metal_kernel {
    if (kernel) |k| return k;
    const ins = [_][*:0]const u8{ "x", "pos" };
    const outs = [_][*:0]const u8{"out"};
    const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    // The input is read through its strides: a view (a slice of the latent projection, a transposed query) is no copy.
    // The row count is the grid's, never a template value: a chunk of any width takes the pipeline warmed once.
    const k = mlx.mlx_fast_metal_kernel_new("sushi_hy4_rope_tail", iv, ov, source, "", false, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    kernel = k;
    return k;
}

/// Whether `tail` takes `x` [A, T, D] f32: a GPU stream and an even rope width within the row.
pub fn applies(s: mlx.mlx_stream, x: Arr, rope_dims: c_int) bool {
    const sh = mlx.getShape(x);
    const dt = mlx.mlx_array_dtype(x);
    return mlx.streamIsGpu(s) and sh.len == 3 and (dt == .float32 or dt == .bfloat16) and @rem(rope_dims, 2) == 0 and
        rope_dims > 0 and rope_dims <= sh[2];
}

const Key = struct { a: c_int, t: c_int, nope: c_int, half: c_int, in_dt: mlx.mlx_dtype, out_dt: mlx.mlx_dtype, base: c_int, scale: c_int, row_major: bool = false, per_row: bool = false };
var cfgs: kcfg.Cache(Key) = .{};

fn build(k: Key) !kcfg.Config {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const out_sh = if (k.row_major) [3]c_int{ k.t, k.a, k.nope + 2 * k.half } else [3]c_int{ k.a, k.t, k.nope + 2 * k.half };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &out_sh, 3, k.out_dt));
    const width = k.nope + k.half;
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, width, k.t, k.a));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, @min(width, 256), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", k.in_dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "OT", k.out_dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NOPE", k.nope));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "HALF", k.half));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "BASE", k.base));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "SCALE", k.scale));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "ROW_MAJOR", k.row_major));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "PER_ROW", k.per_row));
    return cfg;
}

/// `x` [A, T, D] with its trailing `rope_dims` rotated, row t at position `(pos0 + t) · step`, as `out_dt` [A, T, D]
/// (contiguous). The caller frees the result.
pub fn tail(s: mlx.mlx_stream, x: Arr, rope_dims: c_int, theta: f32, pos0: c_int, step: c_int, out_dt: mlx.mlx_dtype) !Arr {
    const pv = [_]i32{pos0};
    return tailWith(s, x, rope_dims, theta, &pv, step, out_dt, false, false);
}

/// `tail` over rows laid out row-major [T, H, d] (each row's heads together), in `out_dt`.
pub fn tailRows(s: mlx.mlx_stream, x: Arr, rope_dims: c_int, theta: f32, pos0: c_int, step: c_int, out_dt: mlx.mlx_dtype) !Arr {
    const pv = [_]i32{pos0};
    return tailWith(s, x, rope_dims, theta, &pv, step, out_dt, true, false);
}

/// `tail` with row `t` rotated at its own `positions[t] · step`, as `out_dt` [A, T, D] (contiguous).
pub fn tailPerRow(s: mlx.mlx_stream, x: Arr, rope_dims: c_int, theta: f32, positions: []const c_int, step: c_int, out_dt: mlx.mlx_dtype) !Arr {
    return tailWith(s, x, rope_dims, theta, positions, step, out_dt, false, true);
}

/// `tailPerRow` laid out row-major [T, A, D], in `out_dt`.
pub fn tailRowsPerRow(s: mlx.mlx_stream, x: Arr, rope_dims: c_int, theta: f32, positions: []const c_int, step: c_int, out_dt: mlx.mlx_dtype) !Arr {
    return tailWith(s, x, rope_dims, theta, positions, step, out_dt, true, true);
}

fn tailWith(s: mlx.mlx_stream, x: Arr, rope_dims: c_int, theta: f32, positions: []const c_int, step: c_int, out_dt: mlx.mlx_dtype, row_major: bool, per_row: bool) !Arr {
    const sh = mlx.getShape(x);
    if (per_row and positions.len != @as(usize, @intCast(sh[1]))) return error.Hy4RowsPositionsLen;
    const half = @divExact(rope_dims, 2);
    const k = try getKernel();
    // The base and scale as MLX's rope takes them (log2 θ in f32, the step as a float), as the bits of template ints.
    const cfg = try cfgs.get(.{ .a = sh[0], .t = sh[1], .nope = sh[2] - rope_dims, .half = half, .in_dt = mlx.mlx_array_dtype(x), .out_dt = out_dt, .base = kcfg.bits(std.math.log2(theta)), .scale = kcfg.bits(@floatFromInt(step)), .row_major = row_major, .per_row = per_row }, build);
    const pos = try mlx.newData(positions.ptr, &.{@intCast(positions.len)}, .int32);
    defer _ = mlx.mlx_array_free(pos);
    const ins = [_]Arr{ x, pos };
    const iv = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, k, iv, cfg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, ov, 0));
    return out;
}

// ── tests ────────────────────────────────────────────────────────────────────────────────────────

const testing = std.testing;
const Ops = @import("../mlx_ops.zig").Ops;
const fwd = @import("hy4_forward.zig");

fn hostBytes(o: *Ops, a: Arr) ![]const u8 {
    const c = try o.contiguous(a);
    try mlx.check(mlx.mlx_array_eval(c));
    const n = mlx.mlx_array_size(c);
    return switch (mlx.mlx_array_dtype(c)) {
        .float32 => std.mem.sliceAsBytes((mlx.mlx_array_data_float32(c) orelse return error.HostRead)[0..n]),
        else => std.mem.sliceAsBytes((mlx.mlx_array_data_bfloat16(c) orelse return error.HostRead)[0..n]),
    };
}

/// The kernel against the chain it replaces (`fwd.ropeScaled` with the kernel off, then the cast), over `x` as a view.
fn ropeCase(o: *Ops, x: Arr, rope_dims: c_int, theta: f32, pos0: c_int, step: c_int, out_dt: mlx.mlx_dtype) !void {
    fwd.rope_kernel_on = false;
    const want = try o.cast(try fwd.ropeScaled(o, x, rope_dims, theta, pos0, step), out_dt);
    const want_bytes = try hostBytes(o, want);
    try testing.expect(applies(o.s, x, rope_dims));
    try mlx.check(mlx.mlx_array_eval(x));
    const before = mlx.engine.stats().dispatches;
    const got = try o.own(try tail(o.s, x, rope_dims, theta, pos0, step, out_dt));
    try mlx.check(mlx.mlx_array_eval(got));
    try testing.expectEqual(@as(u64, 1), mlx.engine.stats().dispatches - before);
    try testing.expectEqual(out_dt, mlx.mlx_array_dtype(got));
    try testing.expectEqualSlices(c_int, mlx.getShape(x), mlx.getShape(got));
    try testing.expectEqualSlices(u8, want_bytes, try hostBytes(o, got));
}

fn randomRows(o: *Ops, shape: []const c_int, seed: u64) !Arr {
    var n: usize = 1;
    for (shape) |d| n *= @intCast(d);
    const v = try testing.allocator.alloc(f32, n);
    defer testing.allocator.free(v);
    var r = std.Random.DefaultPrng.init(seed);
    for (v) |*d| d.* = @floatCast(r.random().floatNorm(f64) * 3);
    return o.own(mlx.mlx_array_new_data(v.ptr, shape.ptr, @intCast(shape.len), .float32));
}

test "hy4 rope: one launch gives the bytes of the slice, MLX's rope over the tail and the concat, f32 and BF16 out, at the release's query, key, indexer and group-key shapes, decode rows and a chunk, far positions" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    defer fwd.rope_kernel_on = true;
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    var o: Ops = .{ .s = es };
    defer o.deinit();
    for ([_]c_int{ 70, 300_001, 1_048_000 }) |pos| {
        for ([_]c_int{ 1, 5 }) |t| {
            // The query [H, T, 256] as the projection gives it: [T, H, 256] transposed.
            const q = try o.transpose(try randomRows(&o, &.{ t, 64, 256 }, @intCast(pos)), &.{ 1, 0, 2 });
            try ropeCase(&o, q, 64, 1e4, pos, 1, .float32);
            try ropeCase(&o, q, 64, 1e7, pos, 1, .float32);
            // The key's rope part: a slice of the latent projection [T, 576].
            const ckv = try randomRows(&o, &.{ t, 576 }, @intCast(pos + 1));
            const k_pe = try o.reshape(try o.slice(ckv, 1, 512, 576), &.{ 1, t, 64 });
            try ropeCase(&o, k_pe, 64, 1e4, pos, 1, .bfloat16);
            try ropeCase(&o, k_pe, 64, 1e7, pos, 1, .float32);
            // The indexer's query [16, T, 128].
            const iq = try o.transpose(try randomRows(&o, &.{ t, 16, 128 }, @intCast(pos + 2)), &.{ 1, 0, 2 });
            try ropeCase(&o, iq, 64, 1e7, pos, 1, .float32);
        }
    }
    // The pooled group keys [1, G, 128] at their groups' first slots (step 4), and the tiny's widths.
    try ropeCase(&o, try randomRows(&o, &.{ 1, 4099, 128 }, 9), 64, 1e7, 0, 4, .float32);
    try ropeCase(&o, try o.transpose(try randomRows(&o, &.{ 3, 8, 48 }, 10), &.{ 1, 0, 2 }), 16, 1e4, 37, 1, .float32);
    try ropeCase(&o, try randomRows(&o, &.{ 1, 3, 16 }, 11), 16, 1e7, 37, 1, .bfloat16);
}

test "hy4 rope: a chunk of a new width takes the pipeline its first launch made" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    var o: Ops = .{ .s = es };
    defer o.deinit();
    const first = try o.own(try tail(es, try randomRows(&o, &.{ 4, 1, 48 }, 1), 16, 1e4, 3, 1, .float32));
    try mlx.check(mlx.mlx_array_eval(first));
    const before = mlx.engine.pipelinesMade().count;
    for ([_]c_int{ 2, 77, 300 }) |t| {
        const y = try o.own(try tail(es, try randomRows(&o, &.{ 4, t, 48 }, @intCast(t)), 16, 1e4, 3, 1, .float32));
        try mlx.check(mlx.mlx_array_eval(y));
    }
    try testing.expectEqual(before, mlx.engine.pipelinesMade().count);
}

test "hy4 rope: bf16 input widens in register and matches f32 input byte for byte in one launch" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    var o: Ops = .{ .s = es };
    defer o.deinit();
    const q_bf = try o.transpose(try o.cast(try randomRows(&o, &.{ 16, 32, 256 }, 77), .bfloat16), &.{ 1, 0, 2 });
    const q_f32 = try o.cast(q_bf, .float32);
    const want = try hostBytes(&o, try o.own(try tail(es, q_f32, 64, 1e4, 4096, 1, .float32)));
    try testing.expect(applies(es, q_bf, 64));
    try mlx.check(mlx.mlx_array_eval(q_bf));
    const before = mlx.engine.stats().dispatches;
    const got = try o.own(try tail(es, q_bf, 64, 1e4, 4096, 1, .float32));
    try mlx.check(mlx.mlx_array_eval(got));
    try testing.expectEqual(@as(u64, 1), mlx.engine.stats().dispatches - before);
    try testing.expectEqualSlices(u8, want, try hostBytes(&o, got));
}

test "hy4 rope: per-row positions match per-row single launches byte for byte (full and rope-only, heads-first and row-major)" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    var o: Ops = .{ .s = es };
    defer o.deinit();
    const positions = [_]c_int{ 0, 511, 512, 4097, 2048, 100_003, 7, 65_535 };
    const t: c_int = positions.len;
    const raw = try randomRows(&o, &.{ t, 64, 256 }, 123);
    const q = try o.transpose(raw, &.{ 1, 0, 2 }); // [H, T, 256] strided view
    const q_tail = try o.slice(q, 2, 192, 256); // [H, T, 64] strided view
    try mlx.check(mlx.mlx_array_eval(raw));
    const b1 = mlx.engine.stats().dispatches;
    const got_hf = try o.own(try tailPerRow(es, q, 64, 1e4, &positions, 1, .float32));
    const got_rm = try o.own(try tailRowsPerRow(es, q_tail, 64, 1e4, &positions, 1, .float32));
    try mlx.check(mlx.mlx_array_eval(got_hf));
    try mlx.check(mlx.mlx_array_eval(got_rm));
    try testing.expectEqual(@as(u64, 2), mlx.engine.stats().dispatches - b1);
    for (positions, 0..) |pos, r| {
        const ri: c_int = @intCast(r);
        const want = try o.own(try tail(es, try o.slice(q, 1, ri, ri + 1), 64, 1e4, pos, 1, .float32));
        try testing.expectEqualSlices(u8, try hostBytes(&o, want), try hostBytes(&o, try o.slice(got_hf, 1, ri, ri + 1)));
        const want_pe = try o.transpose(try o.slice(want, 2, 192, 256), &.{ 1, 0, 2 });
        try testing.expectEqualSlices(u8, try hostBytes(&o, want_pe), try hostBytes(&o, try o.slice(got_rm, 0, ri, ri + 1)));
    }
}
