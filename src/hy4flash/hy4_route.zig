//! The router (`fwd.route`) of one to eight rows (a decode step, a verify block) past its logits in one launch, a
//! threadgroup a row, each row's bytes its one-row launch's: the Sigmoid, the bias add, the top `k` of the
//! biased scores in the order the engine's stable argsort of their negation gives (larger first, equal ones by lower
//! index, NaN last, by NaNs' index), the picked scores summed in `row_reduce_small`'s order (each value added to the running total),
//! the 1e-20, the division and the scale, each in the ops' f32 arithmetic with no contraction. With a slot map (a
//! resident stream's), the picks' slots are written too, as the take over the map and the casts gave them.
const std = @import("std");
const mlx = @import("../mlx.zig");
const Arr = mlx.mlx_array;
const kcfg = @import("hy4_kcfg.zig");

/// The most experts one threadgroup ranks.
pub const max_experts = 1024;
/// The most rows one launch routes (a verify block).
pub const max_rows = 8;
/// The most picks a row sums in the order `row_reduce_small` takes them one by one (wider sums split across lanes).
pub const max_k = 32;

// Each expert's place is a 64-bit key, ascending in the sort's order: the negated biased score as an order-keeping
// uint (-0 as +0, a NaN past every number), then the index. The keys are a total order, so any exact top `k` of them is
// the full sort's. Selection path (257 to 512 experts, `k` up to 16): each simdgroup sorts its 32 keys by shuffles, a
// simdgroup merges four such lists' 16 best into one, and one simdgroup merges those four, all by shuffles, with two
// barriers between the phases. Otherwise one bitonic sort of the threadgroup's keys (`N`, a power of two of at least
// 32): strides under 32 by simd shuffles, the rest through threadgroup memory.
/// A threadgroup barrier statement.
const bar = "threadgroup_barrier(mem_flags::mem_threadgroup);\n";

/// The simdgroup helpers: ascending bitonic sort of 32 keys, the merge of a 32-lane bitonic sequence, and the 16 best of
/// four sorted lists of 16 (`lists`, one after another), left in lanes 0 to 15 ascending. Every shuffle runs on all 32
/// lanes: a lane a shuffle reads must be active.
pub const header =
    \\inline ulong sg_shuffle(ulong v, ushort l) {
    \\  return (ulong(metal::simd_shuffle(uint(v >> 32), l)) << 32) | ulong(metal::simd_shuffle(uint(v), l));
    \\}
    \\inline ulong sg_xor(ulong v, ushort j) {
    \\  return (ulong(metal::simd_shuffle_xor(uint(v >> 32), j)) << 32) | ulong(metal::simd_shuffle_xor(uint(v), j));
    \\}
    \\inline ulong sg_sort32(ulong key, uint lane) {
    \\  #pragma clang loop unroll(full)
    \\  for (uint k = 2; k <= 32; k <<= 1) {
    \\    #pragma clang loop unroll(full)
    \\    for (uint j = k >> 1; j > 0; j >>= 1) {
    \\      ulong other = sg_xor(key, ushort(j));
    \\      key = (((lane & j) == 0) == ((lane & k) == 0)) ? metal::min(key, other) : metal::max(key, other);
    \\    }
    \\  }
    \\  return key;
    \\}
    \\inline ulong sg_merge32(ulong key, uint lane) {
    \\  #pragma clang loop unroll(full)
    \\  for (uint j = 16; j > 0; j >>= 1) {
    \\    ulong other = sg_xor(key, ushort(j));
    \\    key = ((lane & j) == 0) ? metal::min(key, other) : metal::max(key, other);
    \\  }
    \\  return key;
    \\}
    \\inline ulong sg_best16_of4(threadgroup const ulong* lists, uint lane) {
    \\  ulong a = sg_merge32(lane < 16 ? lists[lane] : lists[16 + (31 - lane)], lane);
    \\  ulong b = sg_merge32(lane < 16 ? lists[32 + lane] : lists[48 + (31 - lane)], lane);
    \\  ulong b_rev = sg_shuffle(b, ushort(31 - lane));
    \\  return sg_merge32(lane < 16 ? a : b_rev, lane);
    \\}
;

const head =
    \\#pragma clang fp contract(off)
    \\uint i = thread_position_in_threadgroup.x;
    \\uint row = threadgroup_position_in_grid.y;
    \\uint lane = i & 31;
    \\#ifdef SUSHI_METALLIB
    \\threadgroup float sc[1024];
    \\threadgroup ulong keys[FAST ? 1 : 1024];
    \\threadgroup ulong cand[FAST ? 256 : 1];
    \\threadgroup ulong cand2[FAST ? 64 : 1];
    \\threadgroup uint picked_id[FAST ? 1 : 32];
    \\#else
    \\threadgroup float sc[E];
    \\threadgroup ulong keys[(N > 32 && !FAST) ? N : 1];
    \\threadgroup ulong cand[FAST ? 256 : 1];
    \\threadgroup ulong cand2[FAST ? 64 : 1];
    \\threadgroup uint picked_id[FAST ? 1 : K];
    \\#endif
    \\threadgroup float total[1];
    \\ulong key = ~0ul;
    \\if (i < uint(E)) {
    \\  float x = logits[size_t(row) * E + i];
    \\  auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
    \\  float s = (x < 0) ? y : 1 - y;
    \\  sc[i] = s;
    \\  float a = -(s + static_cast<float>(bias[i]));
    \\  uint u = as_type<uint>(a == 0.0f ? 0.0f : a);
    \\  uint hi = metal::isnan(a) ? 0xFFFFFFFFu : ((u & 0x80000000u) ? ~u : (u | 0x80000000u));
    \\  key = (ulong(hi) << 32) | ulong(i);
    \\}
    \\if (FAST) {
    \\
;

/// The selection path's phases; a barrier joins each to the next and none sits inside one.
const fast_phases = [_][]const u8{
    \\key = sg_sort32(key, lane);
    \\if (lane < 16) cand[(i >> 5) * 16 + lane] = key;
    \\
    ,
    \\if ((i >> 5) < 4) {
    \\  ulong best = sg_best16_of4(cand + (i >> 5) * 64, lane);
    \\  if (lane < 16) cand2[(i >> 5) * 16 + lane] = best;
    \\}
    \\
    ,
    \\if ((i >> 5) == 0) {
    \\  uint e = uint(sg_best16_of4(cand2, lane));
    \\  float t = 0;
    \\  for (int r = 0; r < K; ++r) t = sc[metal::simd_shuffle(e, ushort(r))] + t;
    \\  float tot = t + 1e-20f;
    \\  if (lane < uint(K)) {
    \\    size_t out_i = size_t(row) * K + lane;
    \\    picks[out_i] = e;
    \\    float q = sc[e] / tot;
    \\    weights[out_i] = q * as_type<float>(SCALE);
    \\    if (MAP) slots[out_i] = static_cast<uint>(map[e]);
    \\  }
    \\}
    \\
};

/// The barriers of the selection path: one between each pair of phases.
pub const fast_barriers: u32 = fast_phases.len - 1;

const fast_body = blk: {
    var out: []const u8 = "";
    for (fast_phases, 0..) |ph, n| {
        if (std.mem.indexOf(u8, ph, "threadgroup_barrier") != null) @compileError("a selection phase holds a barrier");
        out = out ++ (if (n > 0) bar else "") ++ ph;
    }
    break :blk out;
};

const full_sort =
    \\} else {
    \\for (uint k = 2; k <= uint(N); k <<= 1) {
    \\  for (uint j = k >> 1; j > 0; j >>= 1) {
    \\    ulong other;
    \\    if (j < 32) {
    \\      other = sg_xor(key, ushort(j));
    \\    } else {
    \\      keys[i] = key;
    \\      threadgroup_barrier(mem_flags::mem_threadgroup);
    \\      other = keys[i ^ j];
    \\      threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    bool low = (i & j) == 0;
    \\    bool up = (i & k) == 0;
    \\    key = (low == up) ? metal::min(key, other) : metal::max(key, other);
    \\  }
    \\}
    \\if (i < uint(K)) picked_id[i] = uint(key);
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if (i == 0) {
    \\  float t = 0;
    \\  for (int r = 0; r < K; ++r) t = sc[picked_id[r]] + t;
    \\  total[0] = t + 1e-20f;
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if (i < uint(K)) {
    \\  uint e = picked_id[i];
    \\  size_t out_i = size_t(row) * K + i;
    \\  picks[out_i] = e;
    \\  float q = sc[e] / total[0];
    \\  weights[out_i] = q * as_type<float>(SCALE);
    \\  if (MAP) slots[out_i] = static_cast<uint>(map[e]);
    \\}
    \\}
    \\
;

pub const source: [:0]const u8 = head ++ fast_body ++ full_sort;

var kernel: ?mlx.mlx_fast_metal_kernel = null;

fn getKernel() !mlx.mlx_fast_metal_kernel {
    if (kernel) |k| return k;
    const ins = [_][*:0]const u8{ "logits", "bias", "map" };
    const outs = [_][*:0]const u8{ "picks", "weights", "slots" };
    const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new("sushi_hy4_route", iv, ov, source, header, true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    kernel = k;
    return k;
}

/// Whether `pick` takes a router's `logits` [T, E] f32: a GPU stream, 1 ≤ T ≤ `max_rows`, `k` ≤ E ≤ `max_experts`.
pub fn applies(s: mlx.mlx_stream, logits: Arr, bias: Arr, k: c_int) bool {
    const sh = mlx.getShape(logits);
    const bd = mlx.mlx_array_dtype(bias);
    return mlx.streamIsGpu(s) and sh.len == 2 and sh[0] >= 1 and mlx.mlx_array_dtype(logits) == .float32 and
        k >= 1 and k <= max_k and k <= sh[1] and sh[1] <= max_experts and mlx.mlx_array_size(bias) == @as(usize, @intCast(sh[1])) and
        (bd == .float32 or bd == .bfloat16);
}

const Key = struct { t: c_int, e: c_int, k: c_int, map: bool, scale: c_int, fast: bool };
var cfgs: kcfg.Cache(Key) = .{};

fn build(key: Key) !kcfg.Config {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    for ([_]mlx.mlx_dtype{ .uint32, .float32, .uint32 }) |dt| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ key.t, key.k }, 2, dt));
    // One thread a key, the sort's width a power of two and at least a simdgroup.
    const tg: c_int = @intCast(@max(32, std.math.ceilPowerOfTwoAssert(u32, @intCast(key.e))));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, tg, key.t, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, tg, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "E", key.e));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "N", tg));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "K", key.k));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "MAP", key.map));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "FAST", key.fast));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "SCALE", key.scale));
    return cfg;
}

pub const Picked = struct { picks: Arr, weights: Arr, slots: ?Arr };

/// Barriers the full bitonic sort of `n` keys (a power of two, at least 32) costs: two a stride of 32 or more, and the two
/// after the picks.
pub fn sortBarriers(n: u32) u32 {
    var b: u32 = 2;
    var k: u32 = 2;
    while (k <= n) : (k <<= 1) {
        var j = k >> 1;
        while (j > 0) : (j >>= 1) b += if (j >= 32) 2 else 0;
    }
    return b;
}

/// Whether a row's top `k` of `e` experts is selected without the full sort: 257 to 512 experts, `k` of at most 16.
pub fn fastPath(e: c_int, k: c_int) bool {
    return e > 256 and e <= 512 and k <= 16;
}

/// Each row's picks [T, k] uint32 and weights [T, k] f32, and with `map` [E] (int32) the picks' slots [T, k] uint32.
/// The caller frees them.
pub fn pick(s: mlx.mlx_stream, logits: Arr, bias: Arr, k: c_int, scale: f32, map: ?Arr) !Picked {
    const sh = mlx.getShape(logits);
    return pickWith(s, logits, bias, k, scale, map, fastPath(sh[1], k));
}

/// `pick` with the selection path chosen by the caller (a test's full-sort reference); the bytes are the same either way.
fn pickWith(s: mlx.mlx_stream, logits: Arr, bias: Arr, k: c_int, scale: f32, map: ?Arr, fast: bool) !Picked {
    const sh = mlx.getShape(logits);
    const t = sh[0];
    const e = sh[1];
    const kern = try getKernel();
    const cfg = try cfgs.get(.{ .t = t, .e = e, .k = k, .map = map != null, .scale = kcfg.bits(scale), .fast = fast }, build);
    const ins = [_]Arr{ logits, bias, map orelse bias };
    const iv = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, kern, iv, cfg, s));
    var out: [3]Arr = undefined;
    for (&out, 0..) |*o, i| {
        o.* = mlx.mlx_array_new();
        mlx.check(mlx.mlx_vector_array_get(o, ov, i)) catch |err| {
            for (out[0 .. i + 1]) |prev| _ = mlx.mlx_array_free(prev);
            return err;
        };
    }
    if (map == null) _ = mlx.mlx_array_free(out[2]);
    return .{ .picks = out[0], .weights = out[1], .slots = if (map != null) out[2] else null };
}

// ── tests ────────────────────────────────────────────────────────────────────────────────────────

const testing = std.testing;
const Ops = @import("../mlx_ops.zig").Ops;

fn hostBytes(o: *Ops, a: Arr) ![]const u8 {
    const c = try o.contiguous(a);
    try mlx.check(mlx.mlx_array_eval(c));
    const n = mlx.mlx_array_size(c);
    return switch (mlx.mlx_array_dtype(c)) {
        .float32 => std.mem.sliceAsBytes((mlx.mlx_array_data_float32(c) orelse return error.HostRead)[0..n]),
        .uint32 => std.mem.sliceAsBytes((mlx.mlx_array_data_uint32(c) orelse return error.HostRead)[0..n]),
        .int32 => std.mem.sliceAsBytes((mlx.mlx_array_data_int32(c) orelse return error.HostRead)[0..n]),
        else => error.UnexpectedDtype,
    };
}

test "hy4 route: a decode row's picks, weights and slots in one launch are the ops' bytes, at the release's 512 experts and top 16 and the tiny's 8 and 2, ties among the biased scores broken as the stable sort breaks them" {
    try mlx.requireGpu();
    const fwd = @import("hy4_forward.zig");
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    defer fwd.chain_kernels_on = true;
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const Case = struct { e: c_int, k: c_int, ties: bool };
    for ([_]Case{ .{ .e = 512, .k = 16, .ties = false }, .{ .e = 512, .k = 16, .ties = true }, .{ .e = 8, .k = 2, .ties = false }, .{ .e = 8, .k = 2, .ties = true } }, 0..) |c, ci| {
        var o: Ops = .{ .s = es };
        defer o.deinit();
        const n: usize = @intCast(c.e);
        const lg = try testing.allocator.alloc(f32, n);
        defer testing.allocator.free(lg);
        const bs = try testing.allocator.alloc(f32, n);
        defer testing.allocator.free(bs);
        const mp = try testing.allocator.alloc(i32, n);
        defer testing.allocator.free(mp);
        var r = std.Random.DefaultPrng.init(ci + 1);
        for (lg, bs, mp, 0..) |*l, *b, *m, i| {
            // Ties: a few distinct values repeated, so equal biased scores meet across the top k.
            l.* = if (c.ties) @floatFromInt(@as(i32, @intCast(i % 5)) - 2) else @floatCast(r.random().floatNorm(f64) * 2);
            b.* = if (c.ties) 0 else @floatCast(r.random().floatNorm(f64) * 0.1);
            m.* = @intCast((i * 37 + 11) % n);
        }
        const logits = try o.own(mlx.mlx_array_new_data(lg.ptr, &[_]c_int{ 1, c.e }, 2, .float32));
        const bias = try o.own(mlx.mlx_array_new_data(bs.ptr, &[_]c_int{c.e}, 1, .float32));
        const map = try o.own(mlx.mlx_array_new_data(mp.ptr, &[_]c_int{c.e}, 1, .int32));
        // The chain past the logits, as `fwd.route` and the resident route run it.
        const scores = try o.unary(.sigmoid, logits);
        const choice = try o.binary(.add, scores, try o.cast(bias, .float32));
        const order = try o.slot();
        try mlx.check(mlx.mlx_argpartition_axis(order, try o.unary(.negative, choice), c.k - 1, -1, es));
        const want_picks = try o.slice(order.*, 1, 0, c.k);
        const picked = try o.slot();
        try mlx.check(mlx.mlx_take_along_axis(picked, scores, want_picks, -1, es));
        const total = try o.binary(.add, try o.reduce(picked.*, -1, false, true), try o.scalar(1e-20, .float32));
        const want_w = try o.binary(.mul, try o.binary(.div, picked.*, total), try o.scalar(3.998, .float32));
        const want_slots = try o.cast(try o.take(map, try o.cast(want_picks, .int32), 0), .uint32);
        const wp = try hostBytes(&o, want_picks);
        const ww = try hostBytes(&o, want_w);
        const wsl = try hostBytes(&o, want_slots);
        try testing.expect(applies(es, logits, bias, c.k));
        const before = mlx.engine.stats().dispatches;
        const got = try pick(es, logits, bias, c.k, 3.998, map);
        _ = try o.own(got.picks);
        _ = try o.own(got.weights);
        _ = try o.own(got.slots.?);
        try mlx.check(mlx.mlx_array_eval(got.weights));
        try testing.expectEqual(@as(u64, 1), mlx.engine.stats().dispatches - before);
        try testing.expectEqualSlices(u8, wp, try hostBytes(&o, got.picks));
        try testing.expectEqualSlices(u8, ww, try hostBytes(&o, got.weights));
        try testing.expectEqualSlices(u8, wsl, try hostBytes(&o, got.slots.?));
    }
}

test "hy4 route: a NaN score ranks after every number and NaNs by index, so the picks are the stable sort's when the k reach past the numbers" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    // Fewer numbers than k: the last picks are NaNs, lowest index first.
    const Case = struct { e: c_int, k: c_int, numbers: usize };
    for ([_]Case{ .{ .e = 512, .k = 16, .numbers = 8 }, .{ .e = 8, .k = 4, .numbers = 2 } }, 0..) |c, ci| {
        var o: Ops = .{ .s = es };
        defer o.deinit();
        const n: usize = @intCast(c.e);
        const lg = try testing.allocator.alloc(f32, n);
        defer testing.allocator.free(lg);
        const bs = try testing.allocator.alloc(f32, n);
        defer testing.allocator.free(bs);
        var r = std.Random.DefaultPrng.init(ci + 7);
        for (lg, bs, 0..) |*l, *b, i| {
            l.* = if (i % (n / c.numbers) == n / c.numbers - 1) @floatCast(r.random().floatNorm(f64) * 2) else std.math.nan(f32);
            b.* = 0;
        }
        const logits = try o.own(mlx.mlx_array_new_data(lg.ptr, &[_]c_int{ 1, c.e }, 2, .float32));
        const bias = try o.own(mlx.mlx_array_new_data(bs.ptr, &[_]c_int{c.e}, 1, .float32));
        const order = try o.slot();
        try mlx.check(mlx.mlx_argsort_axis(order, try o.unary(.negative, try o.binary(.add, try o.unary(.sigmoid, logits), bias)), -1, es));
        const want = try hostBytes(&o, try o.slice(order.*, 1, 0, c.k));
        const got = try pick(es, logits, bias, c.k, 3.998, null);
        _ = try o.own(got.picks);
        _ = try o.own(got.weights);
        const got_ids: []align(1) const u32 = std.mem.bytesAsSlice(u32, try hostBytes(&o, got.picks));
        const want_ids: []align(1) const u32 = std.mem.bytesAsSlice(u32, want);
        for (want_ids, got_ids) |w, g| try testing.expectEqual(w, g);
    }
}

test "hy4 route: 1 to 8 rows at 512 experts and top 16 (with and without slot map, ties and NaN) match the ops in one launch" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const e: c_int = 512;
    const k: c_int = 16;
    const eu: usize = @intCast(e);
    for (1..9) |tu| {
        const t: c_int = @intCast(tu);
        for ([_]bool{ false, true }) |with_map| {
            var o: Ops = .{ .s = es };
            defer o.deinit();
            const lg = try testing.allocator.alloc(f32, tu * eu);
            defer testing.allocator.free(lg);
            const bs = try testing.allocator.alloc(f32, eu);
            defer testing.allocator.free(bs);
            const mp = try testing.allocator.alloc(i32, eu);
            defer testing.allocator.free(mp);
            var r = std.Random.DefaultPrng.init(tu * 17 + @as(usize, @intFromBool(with_map)) + 3);
            for (lg) |*l| l.* = @floatCast(r.random().floatNorm(f64) * 2);
            for (bs, mp, 0..) |*b, *m, i| {
                b.* = @floatCast(r.random().floatNorm(f64) * 0.1);
                m.* = @intCast((i * 37 + 11) % eu);
            }
            const logits = try o.own(mlx.mlx_array_new_data(lg.ptr, &[_]c_int{ t, e }, 2, .float32));
            const bias = try o.own(mlx.mlx_array_new_data(bs.ptr, &[_]c_int{e}, 1, .float32));
            const map: ?Arr = if (with_map) try o.own(mlx.mlx_array_new_data(mp.ptr, &[_]c_int{e}, 1, .int32)) else null;
            const scores = try o.unary(.sigmoid, logits);
            const choice = try o.binary(.add, scores, try o.cast(bias, .float32));
            const order = try o.slot();
            try mlx.check(mlx.mlx_argpartition_axis(order, try o.unary(.negative, choice), k - 1, -1, es));
            const want_picks = try o.slice(order.*, 1, 0, k);
            const picked = try o.slot();
            try mlx.check(mlx.mlx_take_along_axis(picked, scores, want_picks, -1, es));
            const total = try o.binary(.add, try o.reduce(picked.*, -1, false, true), try o.scalar(1e-20, .float32));
            const want_w = try o.binary(.mul, try o.binary(.div, picked.*, total), try o.scalar(3.998, .float32));
            const wp = try hostBytes(&o, want_picks);
            const ww = try hostBytes(&o, want_w);
            const wsl: ?[]const u8 = if (map) |m| try hostBytes(&o, try o.cast(try o.take(m, try o.cast(want_picks, .int32), 0), .uint32)) else null;
            try testing.expect(applies(es, logits, bias, k));
            const before = mlx.engine.stats().dispatches;
            const got = try pick(es, logits, bias, k, 3.998, map);
            _ = try o.own(got.picks);
            _ = try o.own(got.weights);
            if (got.slots) |sl| _ = try o.own(sl);
            try mlx.check(mlx.mlx_array_eval(got.weights));
            try testing.expectEqual(@as(u64, 1), mlx.engine.stats().dispatches - before);
            try testing.expectEqualSlices(u8, wp, try hostBytes(&o, got.picks));
            try testing.expectEqualSlices(u8, ww, try hostBytes(&o, got.weights));
            if (wsl) |ws| {
                try testing.expectEqualSlices(u8, ws, try hostBytes(&o, got.slots.?));
            } else {
                try testing.expect(got.slots == null);
            }
        }
    }
    for ([_]bool{ false, true }) |nan_case| {
        const t: c_int = 6;
        const tu: usize = 6;
        var o: Ops = .{ .s = es };
        defer o.deinit();
        const lg = try testing.allocator.alloc(f32, tu * eu);
        defer testing.allocator.free(lg);
        const bs = try testing.allocator.alloc(f32, eu);
        defer testing.allocator.free(bs);
        const mp = try testing.allocator.alloc(i32, eu);
        defer testing.allocator.free(mp);
        var r = std.Random.DefaultPrng.init(if (nan_case) 99 else 55);
        for (0..tu) |row| {
            for (0..eu) |i| {
                if (nan_case) {
                    const stride = eu / (row + 3);
                    lg[row * eu + i] = if (i % stride == stride - 1) @floatCast(r.random().floatNorm(f64) * 2) else std.math.nan(f32);
                } else {
                    lg[row * eu + i] = @floatFromInt(@as(i32, @intCast((i + row) % 5)) - 2);
                }
            }
        }
        for (bs, mp, 0..) |*b, *m, i| {
            b.* = 0;
            m.* = @intCast((i * 37 + 11) % eu);
        }
        const logits = try o.own(mlx.mlx_array_new_data(lg.ptr, &[_]c_int{ t, e }, 2, .float32));
        const bias = try o.own(mlx.mlx_array_new_data(bs.ptr, &[_]c_int{e}, 1, .float32));
        const map = try o.own(mlx.mlx_array_new_data(mp.ptr, &[_]c_int{e}, 1, .int32));
        const scores = try o.unary(.sigmoid, logits);
        const choice = try o.binary(.add, scores, bias);
        const order = try o.slot();
        if (nan_case) {
            try mlx.check(mlx.mlx_argsort_axis(order, try o.unary(.negative, choice), -1, es));
        } else {
            try mlx.check(mlx.mlx_argpartition_axis(order, try o.unary(.negative, choice), k - 1, -1, es));
        }
        const want_picks = try o.slice(order.*, 1, 0, k);
        const wp = try hostBytes(&o, want_picks);
        const wsl = try hostBytes(&o, try o.cast(try o.take(map, try o.cast(want_picks, .int32), 0), .uint32));
        try testing.expect(applies(es, logits, bias, k));
        const before = mlx.engine.stats().dispatches;
        const got = try pick(es, logits, bias, k, 3.998, map);
        _ = try o.own(got.picks);
        _ = try o.own(got.weights);
        _ = try o.own(got.slots.?);
        try mlx.check(mlx.mlx_array_eval(got.weights));
        try testing.expectEqual(@as(u64, 1), mlx.engine.stats().dispatches - before);
        try testing.expectEqualSlices(u8, wp, try hostBytes(&o, got.picks));
        try testing.expectEqualSlices(u8, wsl, try hostBytes(&o, got.slots.?));
    }
}

test "hy4 route: applies bounds k to 32 where the sequential sum matches row_reduce_small" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();

    const a = testing.allocator;
    const t: c_int = 4;
    const e: c_int = 512;
    const tu: usize = @intCast(t);
    const eu: usize = @intCast(e);
    var o: Ops = .{ .s = es };
    defer o.deinit();
    const lg = try a.alloc(f32, tu * eu);
    defer a.free(lg);
    const bs = try a.alloc(f32, eu);
    defer a.free(bs);
    var r = std.Random.DefaultPrng.init(0x3233);
    for (lg) |*v| v.* = @floatCast(r.random().floatNorm(f64) * 2.0);
    for (bs) |*v| v.* = @floatCast(r.random().floatNorm(f64) * 0.1);
    const logits = try o.own(mlx.mlx_array_new_data(lg.ptr, &[_]c_int{ t, e }, 2, .float32));
    const bias = try o.own(mlx.mlx_array_new_data(bs.ptr, &[_]c_int{e}, 1, .float32));

    try testing.expect(applies(es, logits, bias, 32));
    try testing.expect(!applies(es, logits, bias, 33));
    try testing.expect(!applies(es, logits, bias, e));

    const scores = try o.unary(.sigmoid, logits);
    const choice = try o.binary(.add, scores, bias);
    const order = try o.slot();
    try mlx.check(mlx.mlx_argsort_axis(order, try o.unary(.negative, choice), -1, es));
    const want_picks = try o.slice(order.*, 1, 0, 32);
    const picked = try o.slot();
    try mlx.check(mlx.mlx_take_along_axis(picked, scores, want_picks, -1, es));
    const total = try o.binary(.add, try o.reduce(picked.*, -1, false, true), try o.scalar(1e-20, .float32));
    const want_w = try o.binary(.mul, try o.binary(.div, picked.*, total), try o.scalar(3.998, .float32));
    const got = try pick(es, logits, bias, 32, 3.998, null);
    _ = try o.own(got.picks);
    _ = try o.own(got.weights);
    try testing.expectEqualSlices(u8, try hostBytes(&o, want_picks), try hostBytes(&o, got.picks));
    try testing.expectEqualSlices(u8, try hostBytes(&o, want_w), try hostBytes(&o, got.weights));
}

test "hy4 route: the selection path takes the barriers of two where the full sort of 512 keys takes 22" {
    try testing.expectEqual(@as(u32, 22), sortBarriers(512));
    try testing.expectEqual(@as(u32, 2), fast_barriers);
    try testing.expect(fast_barriers < sortBarriers(512));
    try testing.expect(fastPath(512, 16) and fastPath(257, 1) and !fastPath(256, 16) and !fastPath(513, 16) and !fastPath(512, 17) and !fastPath(8, 2));
}

/// Fills a row's logits and the bias by pattern: the top of the biased scores lands in one simdgroup, on a tie, past NaN,
/// at an infinity or at a zero's sign, where a selection that drops or misorders a key shows.
fn fillPattern(p: usize, row: usize, rng: *std.Random.DefaultPrng, lg: []f32, bs: []f32) void {
    const nan = std.math.nan(f32);
    const inf = std.math.inf(f32);
    const specials = [_]f32{ nan, inf, -inf, 0, -0.0, 1, 1, 0.5 };
    const bias_specials = [_]f32{ 0, -0.0, inf, -inf, nan, 0.25, 0 };
    for (lg, bs, 0..) |*l, *b, i| {
        b.* = 0;
        l.* = switch (p) {
            0 => @floatCast(rng.random().floatNorm(f64) * 2),
            1 => @floatFromInt(@as(i32, @intCast((i + row) % 5)) - 2),
            2 => 0.5,
            3 => specials[(i * 7 + row) % specials.len],
            // Eight numbers among NaNs: the picks past the numbers are NaNs by index.
            4 => if (i % 64 == 63 - row) @floatCast(rng.random().floatNorm(f64) * 2) else nan,
            // Sixteen best all in one simdgroup (96..127), the rest worse.
            5 => if (i >= 96 and i < 128) 5 + @as(f32, @floatFromInt(i % 11)) * 0.01 else @floatCast(rng.random().floatNorm(f64)),
            // Sixteen best in one simdgroup beside a second simdgroup's equal ones: ties across the lists.
            6 => if (i >= 96 and i < 128) 5 else if (i >= 480) 5 else -3,
            // Descending by index, the best in simdgroup 0, then ascending in the next row's.
            7 => if (row % 2 == 0) -@as(f32, @floatFromInt(i)) * 0.01 else @as(f32, @floatFromInt(i)) * 0.01,
            // Best sixteen spread unevenly: nine in simdgroup 0, seven in simdgroup 15.
            8 => if (i < 9) 4 + @as(f32, @floatFromInt(i)) * 0.1 else if (i >= 480 and i < 487) 4.05 + @as(f32, @floatFromInt(i - 480)) * 0.1 else @floatCast(rng.random().floatNorm(f64)),
            else => @floatCast(rng.random().floatNorm(f64) * 3),
        };
        if (p == 3) b.* = bias_specials[(i * 3 + row) % bias_specials.len];
        if (p == 2) b.* = if (i % 3 == 0) 0.1 else 0;
        if (p == 9) b.* = @floatCast(rng.random().floatNorm(f64) * 0.5);
    }
}

test "hy4 route: the selection path's picks, weights and slots are the full sort's bytes at 512 experts and top 16, 1 and 8 rows, with and without the map, over ties, NaN, signed zeros, infinities and clustered tops" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const e: c_int = 512;
    const k: c_int = 16;
    const eu: usize = 512;
    try testing.expect(fastPath(e, k));
    for (0..10) |p| {
        for ([_]usize{ 1, 8 }) |tu| {
            for ([_]bool{ false, true }) |with_map| {
                for ([_]bool{ false, true }) |bf16_bias| {
                    var o: Ops = .{ .s = es };
                    defer o.deinit();
                    const lg = try testing.allocator.alloc(f32, tu * eu);
                    defer testing.allocator.free(lg);
                    const bs = try testing.allocator.alloc(f32, eu);
                    defer testing.allocator.free(bs);
                    const mp = try testing.allocator.alloc(i32, eu);
                    defer testing.allocator.free(mp);
                    var r = std.Random.DefaultPrng.init(p * 101 + tu * 7 + @as(usize, @intFromBool(with_map)));
                    for (0..tu) |row| fillPattern(p, row, &r, lg[row * eu ..][0..eu], bs);
                    for (mp, 0..) |*m, i| m.* = @intCast((i * 37 + 11) % eu);
                    const logits = try o.own(mlx.mlx_array_new_data(lg.ptr, &[_]c_int{ @intCast(tu), e }, 2, .float32));
                    const bias32 = try o.own(mlx.mlx_array_new_data(bs.ptr, &[_]c_int{e}, 1, .float32));
                    const bias = if (bf16_bias) try o.cast(bias32, .bfloat16) else bias32;
                    const map: ?Arr = if (with_map) try o.own(mlx.mlx_array_new_data(mp.ptr, &[_]c_int{e}, 1, .int32)) else null;
                    try testing.expect(applies(es, logits, bias, k));
                    const want = try pickWith(es, logits, bias, k, 3.998, map, false);
                    _ = try o.own(want.picks);
                    _ = try o.own(want.weights);
                    if (want.slots) |sl| _ = try o.own(sl);
                    const before = mlx.engine.stats().dispatches;
                    const got = try pickWith(es, logits, bias, k, 3.998, map, true);
                    _ = try o.own(got.picks);
                    _ = try o.own(got.weights);
                    if (got.slots) |sl| _ = try o.own(sl);
                    try mlx.check(mlx.mlx_array_eval(got.weights));
                    try testing.expectEqual(@as(u64, 1), mlx.engine.stats().dispatches - before);
                    try testing.expectEqualSlices(u8, try hostBytes(&o, want.picks), try hostBytes(&o, got.picks));
                    try testing.expectEqualSlices(u8, try hostBytes(&o, want.weights), try hostBytes(&o, got.weights));
                    if (want.slots) |ws| try testing.expectEqualSlices(u8, try hostBytes(&o, ws), try hostBytes(&o, got.slots.?));
                }
            }
        }
    }
}

test "hy4 route: wide prefill rows past max_rows run in one launch and match the ops byte for byte" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    const e: c_int = 512;
    const k: c_int = 16;
    const tu: usize = 64;
    const eu: usize = 512;
    var o: Ops = .{ .s = es };
    defer o.deinit();
    const lg = try testing.allocator.alloc(f32, tu * eu);
    defer testing.allocator.free(lg);
    const bs = try testing.allocator.alloc(f32, eu);
    defer testing.allocator.free(bs);
    const mp = try testing.allocator.alloc(i32, eu);
    defer testing.allocator.free(mp);
    var r = std.Random.DefaultPrng.init(0x52007e);
    for (lg) |*l| l.* = @floatCast(r.random().floatNorm(f64) * 2);
    for (bs, mp, 0..) |*b, *m, i| {
        b.* = @floatCast(r.random().floatNorm(f64) * 0.1);
        m.* = @intCast((i * 37 + 11) % eu);
    }
    const logits = try o.own(mlx.mlx_array_new_data(lg.ptr, &[_]c_int{ @intCast(tu), e }, 2, .float32));
    const bias = try o.own(mlx.mlx_array_new_data(bs.ptr, &[_]c_int{e}, 1, .float32));
    const map = try o.own(mlx.mlx_array_new_data(mp.ptr, &[_]c_int{e}, 1, .int32));
    const scores = try o.unary(.sigmoid, logits);
    const choice = try o.binary(.add, scores, bias);
    const order = try o.slot();
    try mlx.check(mlx.mlx_argpartition_axis(order, try o.unary(.negative, choice), k - 1, -1, es));
    const want_picks = try o.slice(order.*, 1, 0, k);
    const picked = try o.slot();
    try mlx.check(mlx.mlx_take_along_axis(picked, scores, want_picks, -1, es));
    const total = try o.binary(.add, try o.reduce(picked.*, -1, false, true), try o.scalar(1e-20, .float32));
    const want_w = try o.binary(.mul, try o.binary(.div, picked.*, total), try o.scalar(3.998, .float32));
    const want_slots = try o.cast(try o.take(map, try o.cast(want_picks, .int32), 0), .uint32);
    const wp = try hostBytes(&o, want_picks);
    const ww = try hostBytes(&o, want_w);
    const wsl = try hostBytes(&o, want_slots);
    try testing.expect(applies(es, logits, bias, k));
    const before = mlx.engine.stats().dispatches;
    const got = try pick(es, logits, bias, k, 3.998, map);
    _ = try o.own(got.picks);
    _ = try o.own(got.weights);
    _ = try o.own(got.slots.?);
    try mlx.check(mlx.mlx_array_eval(got.weights));
    try testing.expectEqual(@as(u64, 1), mlx.engine.stats().dispatches - before);
    try testing.expectEqualSlices(u8, wp, try hostBytes(&o, got.picks));
    try testing.expectEqualSlices(u8, ww, try hostBytes(&o, got.weights));
    try testing.expectEqualSlices(u8, wsl, try hostBytes(&o, got.slots.?));
}
