//! Qwen3.8-Flash-Next's shared expert as two 128-thread kernels for `fuse.zig`: gate, up and SwiGLU (with the
//! shared-expert gate logit in one more threadgroup), and the down projection. Each projection keeps the order MLX's
//! one-row affine8 qmv gives it (`qmv_fast` where K and N allow, `qmv` otherwise), which is also what the verify rows
//! kernels give every row; a grid row is an input row, so every row is the serial kernel's bytes.
const std = @import("std");
const mlx = @import("mlx_host").mlx;
const fuse = @import("fuse.zig");
const kernels = @import("expert_exl3_kernels.zig");

pub const Affine = struct { weight: mlx.mlx_array, scales: mlx.mlx_array, biases: mlx.mlx_array };

/// The shared expert's banks (affine8, bf16 scales and biases), the bf16 sigmoid table, and the shared-expert gate's
/// one-output bank when the gate logit rides the gate/up launch (null: the caller passes it to the reduce).
pub const Shared = struct {
    gate: Affine,
    up: Affine,
    down: Affine,
    sigmoid: mlx.mlx_array,
    logit: ?Affine = null,
};

pub const Geometry = struct { hidden: c_int, width: c_int, group: c_int, logit_group: c_int };

pub const HEADER: [:0]const u8 =
    \\#include <metal_simdgroup>
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\template <int K, int GS, int FST, int RPS>
    \\inline void qwen_qmv_rows(const device uint8_t* w, const device bfloat16_t* sc, const device bfloat16_t* bi, const device bfloat16_t* x, uint lane, thread float* result) {
    \\  constexpr int VPT = FST != 0 ? 8 : 4;
    \\  constexpr int BLOCK = VPT * 32;
    \\  constexpr int GROUPS = K / GS;
    \\  const device uint8_t* ws = w + lane * VPT;
    \\  const device bfloat16_t* sp = sc + lane / (GS / VPT);
    \\  const device bfloat16_t* bp = bi + lane / (GS / VPT);
    \\  const device bfloat16_t* xp = x + lane * VPT;
    \\  int k = 0;
    \\  for (; k < (FST != 0 ? K : K - BLOCK); k += BLOCK) {
    \\    float xt[VPT];
    \\    float sums = 0.0f;
    \\    for (int i = 0; i < VPT; ++i) { sums += xp[i]; xt[i] = xp[i]; }
    \\    for (int row = 0; row < RPS; ++row) {
    \\      uchar codes[VPT];
    \\      for (int i = 0; i < VPT; ++i) codes[i] = ws[row * K + i];
    \\      const float scale = sp[row * GROUPS];
    \\      const float bias = bp[row * GROUPS];
    \\      float accum = 0.0f;
    \\      for (int i = 0; i < VPT; ++i) accum += xt[i] * codes[i];
    \\      result[row] += scale * accum + sums * bias;
    \\    }
    \\    ws += BLOCK;
    \\    sp += BLOCK / GS;
    \\    bp += BLOCK / GS;
    \\    xp += BLOCK;
    \\  }
    \\  if (FST == 0) {
    \\    const int remaining = clamp(K - k - int(lane) * VPT, 0, VPT);
    \\    if (remaining > 0) {
    \\      float xt[VPT];
    \\      float sums = 0.0f;
    \\      for (int i = 0; i < remaining; ++i) { sums += xp[i]; xt[i] = xp[i]; }
    \\      for (int row = 0; row < RPS; ++row) {
    \\        uchar codes[VPT];
    \\        for (int i = 0; i < remaining; ++i) codes[i] = ws[row * K + i];
    \\        const float scale = sp[row * GROUPS];
    \\        const float bias = bp[row * GROUPS];
    \\        float accum = 0.0f;
    \\        for (int i = 0; i < remaining; ++i) accum += xt[i] * codes[i];
    \\        result[row] += scale * accum + sums * bias;
    \\      }
    \\    }
    \\  }
    \\}
;

const GATE_UP_OPEN: [:0]const u8 =
    \\const uint lane = thread_index_in_simdgroup;
    \\const int row = int(threadgroup_position_in_grid.y);
    \\const int tgx = int(threadgroup_position_in_grid.x);
    \\const int sgi = int(simdgroup_index_in_threadgroup);
    \\const device bfloat16_t* xr = x + size_t(row) * K;
    \\
;

/// The threadgroup past the gate/up ones computes the shared-expert gate logit of its row into `gl[row]`.
const LOGIT_BRANCH: [:0]const u8 =
    \\if (tgx == N / 8) {
    \\  if (sgi == 0) {
    \\    float result[1] = {0.0f};
    \\    qwen_qmv_rows<K, LGS, 0, 1>((const device uint8_t*)lw, ls, lb, xr, lane, result);
    \\    const float v = simd_sum(result[0]);
    \\    if (lane == 0) gl[row] = bfloat16_t(v);
    \\  }
    \\} else
;

/// Eight rows a threadgroup (two row groups of four, a gate and an up simdgroup each); `x` is [rows, K] and `y`
/// [rows, N].
const GATE_UP_BLOCK: [:0]const u8 =
    \\{
    \\  const int which = sgi & 1;
    \\  const int part = (sgi >> 1) * 8;
    \\  const int r0 = tgx * 8 + (sgi >> 1) * 4;
    \\  const device uint8_t* wp = (const device uint8_t*)(which == 0 ? wg : wu);
    \\  const device bfloat16_t* sp = which == 0 ? sg : su;
    \\  const device bfloat16_t* bp = which == 0 ? bg : bu;
    \\  float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    \\  qwen_qmv_rows<K, GS, FAST, 4>(wp + size_t(r0) * K, sp + r0 * (K / GS), bp + r0 * (K / GS), xr, lane, result);
    \\  threadgroup float parts[16];
    \\  for (int r = 0; r < 4; ++r) {
    \\    const float v = simd_sum(result[r]);
    \\    if (lane == 0) parts[part + which * 4 + r] = float(bfloat16_t(v));
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  if (which == 0 && lane < 4) {
    \\    const bfloat16_t g = bfloat16_t(parts[part + lane]);
    \\    const bfloat16_t u = bfloat16_t(parts[part + 4 + lane]);
    \\    const bfloat16_t act = g * sigmoid[as_type<ushort>(g)];
    \\    y[row * N + r0 + lane] = act * u;
    \\  }
    \\}
;

const GATE_UP: [:0]const u8 = GATE_UP_OPEN ++ GATE_UP_BLOCK;
const GATE_UP_LOGIT: [:0]const u8 = GATE_UP_OPEN ++ LOGIT_BRANCH ++ GATE_UP_BLOCK;

/// Sixteen rows a threadgroup, a simdgroup of four each; `x` is [rows, K], `y` [rows, N].
const DOWN: [:0]const u8 =
    \\const uint lane = thread_index_in_simdgroup;
    \\const int row = int(threadgroup_position_in_grid.y);
    \\const int r0 = int(threadgroup_position_in_grid.x) * 16 + int(simdgroup_index_in_threadgroup) * 4;
    \\float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    \\qwen_qmv_rows<K, GS, FAST, 4>((const device uint8_t*)w + size_t(r0) * K, sc + r0 * (K / GS), bi + r0 * (K / GS), x + size_t(row) * K, lane, result);
    \\for (int r = 0; r < 4; ++r) {
    \\  const float v = simd_sum(result[r]);
    \\  if (lane == 0) y[row * N + r0 + r] = bfloat16_t(v);
    \\}
;

fn ints(alloc: std.mem.Allocator, pairs: anytype) ![]const fuse.IntArg {
    const out = try alloc.alloc(fuse.IntArg, pairs.len);
    inline for (pairs, 0..) |p, i| out[i] = .{ .name = p[0], .value = p[1] };
    return out;
}

fn bf16Out(alloc: std.mem.Allocator, dims: []const c_int) !fuse.Out {
    return .{ .shape = try alloc.dupe(c_int, dims), .dtype = .bfloat16 };
}

fn isBf16(a: mlx.mlx_array) bool {
    return a.ctx != null and mlx.mlx_array_dtype(a) == .bfloat16;
}

/// The group size of an affine8 bank `[n, k / 4]` U32 with `[n, k / group]` bf16 scales and biases, or null.
fn affineGroup(a: Affine, k: c_int, n: c_int) ?c_int {
    if (a.weight.ctx == null or mlx.mlx_array_dtype(a.weight) != .uint32 or !isBf16(a.scales) or !isBf16(a.biases)) return null;
    const w = mlx.getShape(a.weight);
    const sc = mlx.getShape(a.scales);
    if (w.len != 2 or sc.len != 2 or w[0] != n or w[1] != @divExact(k, 4) or sc[0] != n or sc[1] <= 0 or !std.mem.eql(c_int, sc, mlx.getShape(a.biases))) return null;
    if (@mod(k, sc[1]) != 0) return null;
    const group = @divExact(k, sc[1]);
    return if (group == 64 or group == 128) group else null;
}

/// The shapes the riders serve for hidden width `hidden`, or null: affine8 g64 or g128 throughout, K a multiple of 128
/// and at least 256, the gate/up width a multiple of 128 and the hidden width of 16.
pub fn geometry(sh: Shared, hidden: c_int) ?Geometry {
    if (!isBf16(sh.sigmoid) or mlx.mlx_array_size(sh.sigmoid) != 65536) return null;
    if (hidden < 256 or @mod(hidden, 128) != 0 or sh.gate.weight.ctx == null) return null;
    const gw = mlx.getShape(sh.gate.weight);
    if (gw.len != 2 or gw[0] < 128 or @mod(gw[0], 128) != 0) return null;
    const width = gw[0];
    const g = affineGroup(sh.gate, hidden, width) orelse return null;
    const u = affineGroup(sh.up, hidden, width) orelse return null;
    const d = affineGroup(sh.down, width, hidden) orelse return null;
    if (g != u or g != d) return null;
    var logit_group: c_int = 64;
    if (sh.logit) |l| logit_group = affineGroup(l, hidden, 1) orelse return null;
    return .{ .hidden = hidden, .width = width, .group = g, .logit_group = logit_group };
}

pub fn gateUpPart(alloc: std.mem.Allocator, geo: Geometry, rows: c_int, with_logit: bool) !fuse.Part {
    const fast: c_int = @intFromBool(@mod(geo.hidden, 256) == 0);
    const outs = try alloc.alloc(fuse.Out, if (with_logit) 2 else 1);
    outs[0] = try bf16Out(alloc, &.{ rows, geo.width });
    if (with_logit) outs[1] = try bf16Out(alloc, &.{rows});
    return .{
        .launch = .{ .tgs = .{ @divExact(geo.width, 8) + @intFromBool(with_logit), rows, 1 }, .threads = 128, .ints = try ints(alloc, .{ .{ "K", geo.hidden }, .{ "N", geo.width }, .{ "GS", geo.group }, .{ "FAST", fast }, .{ "LGS", geo.logit_group } }), .outs = outs },
        .body = if (with_logit) GATE_UP_LOGIT else GATE_UP,
        .ins = if (with_logit) &.{ "x", "wg", "sg", "bg", "wu", "su", "bu", "sigmoid", "lw", "ls", "lb" } else &.{ "x", "wg", "sg", "bg", "wu", "su", "bu", "sigmoid" },
        .outs = if (with_logit) &.{ "y", "gl" } else &.{"y"},
        .header = HEADER,
    };
}

pub fn downPart(alloc: std.mem.Allocator, geo: Geometry, rows: c_int) !fuse.Part {
    const fast: c_int = @intFromBool(@mod(geo.width, 256) == 0);
    const outs = try alloc.alloc(fuse.Out, 1);
    outs[0] = try bf16Out(alloc, &.{ rows, geo.hidden });
    return .{
        .launch = .{ .tgs = .{ @divExact(geo.hidden, 16), rows, 1 }, .threads = 128, .ints = try ints(alloc, .{ .{ "K", geo.width }, .{ "N", geo.hidden }, .{ "GS", geo.group }, .{ "FAST", fast } }), .outs = outs },
        .body = DOWN,
        .ins = &.{ "x", "w", "sc", "bi" },
        .outs = &.{"y"},
        .header = HEADER,
    };
}

/// The rider inputs of the gate/up launch, in `gateUpPart`'s order.
pub fn gateUpInputs(sh: Shared, x: mlx.mlx_array, buf: *[11]mlx.mlx_array) []const mlx.mlx_array {
    buf[0..8].* = .{ x, sh.gate.weight, sh.gate.scales, sh.gate.biases, sh.up.weight, sh.up.scales, sh.up.biases, sh.sigmoid };
    if (sh.logit) |l| {
        buf[8..11].* = .{ l.weight, l.scales, l.biases };
        return buf[0..11];
    }
    return buf[0..8];
}

/// The routed ids and scores of `rows` rows of `x` flattened as the chain reads them, or null where the shapes or
/// dtypes are not the fold's.
pub const Routing = struct {
    rows: c_int,
    topk: c_int,

    pub fn of(x: mlx.mlx_array, inds: mlx.mlx_array, scores: mlx.mlx_array) ?Routing {
        if (mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_ndim(x) != 3 or mlx.mlx_array_strides(x)[2] != 1) return null;
        const xs = mlx.getShape(x);
        const ix = mlx.getShape(inds);
        if (xs[0] != 1 or xs[1] < 1 or xs[1] > max_rows or ix.len != 3 or ix[0] != 1 or ix[1] != xs[1] or ix[2] <= 0 or ix[2] > kernels.REDUCE_MAX_TOPK) return null;
        const score_type = mlx.mlx_array_dtype(scores);
        if (!std.mem.eql(c_int, ix, mlx.getShape(scores)) or (score_type != .float32 and score_type != .bfloat16 and score_type != .float16)) return null;
        const id_type = mlx.mlx_array_dtype(inds);
        if (id_type != .uint32 and id_type != .int32) return null;
        return .{ .rows = xs[1], .topk = ix[2] };
    }
};

/// The widest row count the riders serve: the verify and batched decode widths.
pub const max_rows: c_int = 8;
