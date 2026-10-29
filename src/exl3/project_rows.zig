const exl3 = @import("expert_exl3.zig");
const kernel_src = @import("kernel_source.zig");
const mlx = @import("mlx_host").mlx;
const std = @import("std");
const kernels = @import("expert_exl3_kernels.zig");

/// The suh + H128 (or svh + H128) butterfly of one 128-block over a simdgroup, unscaled: lane l holds
/// elements l, l+32, l+64, l+96, in `PREPARE_SOURCE`'s arithmetic order.
pub const ROWS_HELPERS: [:0]const u8 =
    \\static inline size_t exl3_seg_at(uint r, uint col, uint rows, uint odim, uint b1, uint b2, uint b3) {
    \\  const uint lo = col < b1 ? 0u : col < b2 ? b1 : col < b3 ? b2 : b3;
    \\  const uint hi = col < b1 ? b1 : col < b2 ? b2 : col < b3 ? b3 : odim;
    \\  return (size_t)lo * rows + (size_t)r * (hi - lo) + (col - lo);
    \\}
    \\static inline void exl3_store(device float *p, float v) { *p = v; }
    \\static inline void exl3_store(device half *p, float v) { *p = half(v); }
    \\static inline void exl3_store(device bfloat16_t *p, float v) { *p = bfloat16_t(v); }
    \\static inline float4 exl3_h128(float4 v, uint lane) {
    \\  for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
    \\    const float4 p = simd_shuffle_xor(v, bit);
    \\    v = (lane & bit) == 0u ? v + p : p - v;
    \\  }
    \\  const float s0 = v.x + v.y;
    \\  const float s1 = v.x - v.y;
    \\  const float s2 = v.z + v.w;
    \\  const float s3 = v.z - v.w;
    \\  return float4(s0 + s2, s1 + s3, s0 - s2, s1 - s3);
    \\}
    \\
;

/// The multi-row kernels' k walk: KT k-tiles from tk0, CHUNK halves per row at a time, each chunk first
/// prepared (suh + H128) for every row into `prep`, element e of a 128-block at offset `PREP_SLOT`.
const ROWS_CHUNK_OPEN: []const u8 =
    \\for (uint c0 = tk0; c0 < tk0 + KT; c0 += CT) {
    \\  const uint c1 = min(c0 + CT, tk0 + KT);
    \\  const uint nb = (c1 - c0) / 8u;
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  for (uint item = sg; item < uint(ROWS) * nb; item += SGS) {
    \\    const uint r = item / nb;
    \\    const uint b = (item - r * nb) * 128u;
    \\    const uint k = c0 * TILE + b;
    \\    const auto xr = x + (size_t)r * uint(IDIM) + k;
    \\    const float4 v = exl3_h128(float4(
    \\      float(xr[lane]) * float(suh[k + lane]),
    \\      float(xr[lane + 32u]) * float(suh[k + lane + 32u]),
    \\      float(xr[lane + 64u]) * float(suh[k + lane + 64u]),
    \\      float(xr[lane + 96u]) * float(suh[k + lane + 96u])), lane);
    \\    threadgroup half *pw = prep + r * uint(CHUNK) + b + PREP_SLOT;
    \\    pw[0] = half(v.x * psc);
    \\    pw[32] = half(v.y * psc);
    \\    pw[64] = half(v.z * psc);
    \\    pw[96] = half(v.w * psc);
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
;

/// The fold: row r's 128 output sums, KPARTS partials each in `tgm`, through the output H128 + svh.
const ROWS_FOLD: []const u8 =
    \\  for (uint r = sg; r < uint(ROWS); r += SGS) {
    \\    float4 v = 0.0f;
    \\    for (uint k = 0u; k < KPARTS; k++) {
    \\      const threadgroup float *pr = tgm + (r * KPARTS + k) * 128u + lane;
    \\      v += float4(pr[0], pr[32], pr[64], pr[96]);
    \\    }
    \\    v = exl3_h128(v, lane);
    \\    const uint ob = ot0 * TILE + lane;
    \\    const auto yr = y + exl3_seg_at(r, ob, uint(ROWS), uint(ODIM), uint(SB1), uint(SB2), uint(SB3));
    \\    yr[0] = OutT(v.x * psc * float(svh[ob]));
    \\    yr[32] = OutT(v.y * psc * float(svh[ob + 32u]));
    \\    yr[64] = OutT(v.z * psc * float(svh[ob + 64u]));
    \\    yr[96] = OutT(v.w * psc * float(svh[ob + 96u]));
    \\  }
    \\
;

const ROWS_HEAD: []const u8 =
    \\constexpr uint TILE = 16u;
    \\constexpr uint N = uint(NHW);
    \\constexpr uint PACKED_W = N / 2u;
    \\constexpr uint OT = uint(ODIM) / TILE;
    \\constexpr uint KT = uint(IDIM) / uint(NSPLIT) / TILE;
    \\constexpr uint CT = uint(CHUNK) / TILE;
    \\constexpr uint PREP_F = uint(ROWS) * uint(CHUNK) / 2u;
    \\threadgroup float tgm[PREP_F > PART_F ? PREP_F : PART_F];
    \\threadgroup half *prep = (threadgroup half *)tgm;
    \\const uint ot0 = uint(threadgroup_position_in_grid.x) * TGT;
    \\const uint split = uint(threadgroup_position_in_grid.y);
    \\const uint sg = uint(simdgroup_index_in_threadgroup);
    \\const uint lane = uint(thread_index_in_simdgroup);
    \\const uint tk0 = split * KT;
    \\const float psc = 0.08838834764831845f;
    \\if constexpr (GROUPS > 1) {
    \\  const size_t grp = threadgroup_position_in_grid.z;
    \\  x += grp * uint(ROWS) * uint(IDIM);
    \\  trellis += grp * (uint(IDIM) / TILE) * OT * N;
    \\  suh += grp * uint(IDIM);
    \\  svh += grp * uint(ODIM);
    \\  y += grp * (FOLD ? uint(ROWS) : uint(ROWS) * uint(NSPLIT)) * uint(ODIM);
    \\}
    \\if constexpr (uint(SB1) < uint(ODIM)) {
    \\  const uint c0 = ot0 * TILE;
    \\  suh += (uint(c0 >= uint(SB1)) + uint(c0 >= uint(SB2)) + uint(c0 >= uint(SB3))) * uint(IDIM);
    \\}
    \\
;

/// `projectIndexed` over ROWS activation rows of one dense matrix: a lane decodes each of its weights
/// once and feeds every row. A threadgroup owns TGRP output tile pairs, with KPH simdgroups per pair
/// splitting its k range; it prepares that range for every row in threadgroup memory, in GEMV lane
/// order. With one 128-wide output block per threadgroup and no k split (FOLD) it also applies the output
/// H128 + svh; otherwise each k split stores f32 partials that `ROWS_FINISH_SOURCE` sums.
pub const ROWS_SOURCE: [:0]const u8 = kernels.metalReplaceFound(ROWS_BODY, "PREP_SLOT", kernels.LANE_ORDER_SLOT);

/// The k walk of a chunk: each k-tile's weights decode once, into registers, and every row multiplies them. A row
/// loop runs at most `ROWS_LOOP_MAX` rows: Metal compiles a longer one 2 to 3x slower (the sums leave registers).
const ROWS_LOOP_MAX: usize = 4;

pub const ROWS_K_LOOP: []const u8 = rowsKLoop();

fn rowsKLoop() []const u8 {
    comptime {
        var s: []const u8 =
            \\  for (uint tk = c0 + kph; tk < c1; tk += 2u * uint(KPH)) {
            \\    exl3_lane_bits<N> merged[2][2];
            \\    for (uint u = 0u; u < 2u; u++) {
            \\      for (uint o = 0u; o < 2u; o++) {
            \\        merged[u][o] = exl3_lane<N>(wp + u * uint(KPH) * OT * PACKED_W + o * PACKED_W, lane);
            \\      }
            \\    }
            \\    for (uint u = 0u; u < 2u; u++) {
            \\      float2 wd[2][4];
            \\      for (uint o = 0u; o < 2u; o++) {
            \\        for (uint p = 0u; p < 4u; p++) {
            \\          const uint2 cw = uint2(exl3_lane_word(merged[u][o], sh[p * 2u]), exl3_lane_word(merged[u][o], sh[p * 2u + 1u])) & uint2(0xffffu);
            \\          wd[o][p] = exl3_decode2(cw);
            \\        }
            \\      }
            \\
        ;
        var r0: usize = 0;
        while (r0 < @as(usize, @intCast(PROJECT_ROWS_MAX))) : (r0 += ROWS_LOOP_MAX) {
            s = s ++ std.fmt.comptimePrint(
                \\      if constexpr (uint(ROWS) > {0d}u) {{
                \\        constexpr uint RN = uint(ROWS) - {0d}u < {1d}u ? uint(ROWS) - {0d}u : {1d}u;
                \\        for (uint r = 0u; r < RN; r++) {{
                \\          const float4 xr = float4(*((const threadgroup half4 *)(pp + ({0d}u + r) * uint(CHUNK) + u * uint(KPH) * TILE)));
                \\          for (uint o = 0u; o < 2u; o++) {{
                \\            for (uint p = 0u; p < 4u; p++) {{
                \\              const float2 i2 = (p & 1u) != 0u ? xr.zw : xr.xy;
                \\              acc[{0d}u + r][o][p >> 1u] = fma(i2.y, wd[o][p].y, fma(i2.x, wd[o][p].x, acc[{0d}u + r][o][p >> 1u]));
                \\            }}
                \\          }}
                \\        }}
                \\      }}
                \\
            , .{ r0, ROWS_LOOP_MAX });
        }
        return s ++
            \\    }
            \\    wp += 2u * uint(KPH) * OT * PACKED_W;
            \\    pp += 2u * uint(KPH) * TILE;
            \\  }
            \\
        ;
    }
}

pub const ROWS_BODY: [:0]const u8 =
    \\constexpr uint SGS = uint(TGRP) * uint(KPH);
    \\constexpr uint KPARTS = uint(KPH);
    \\constexpr uint TGT = 2u * uint(TGRP);
    \\constexpr uint PART_F = uint(ROWS) * KPARTS * TGT * 16u;
    \\
++ ROWS_HEAD ++
    \\const uint lid = uint(thread_index_in_threadgroup);
    \\const uint tgp = sg % uint(TGRP);
    \\const uint kph = sg / uint(TGRP);
    \\constexpr uint sh[8] = {exl3_lane_sh(N, 0u), exl3_lane_sh(N, 1u), exl3_lane_sh(N, 2u), exl3_lane_sh(N, 3u), exl3_lane_sh(N, 4u), exl3_lane_sh(N, 5u), exl3_lane_sh(N, 6u), exl3_lane_sh(N, 7u)};
    \\float acc[ROWS][2][2] = {};
    \\const device uint *wt = (const device uint *)trellis + (size_t)(ot0 + 2u * tgp) * PACKED_W;
    \\
++ ROWS_CHUNK_OPEN ++
    \\  const device uint *wp = wt + (size_t)(c0 + kph) * OT * PACKED_W;
    \\  const threadgroup half *pp = prep + kph * TILE + (lane & 3u) * 4u;
++ ROWS_K_LOOP ++
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\for (uint r = 0u; r < uint(ROWS); r++) {
    \\  for (uint o = 0u; o < 2u; o++) {
    \\    float lo = acc[r][o][0];
    \\    float hi = acc[r][o][1];
    \\    lo += simd_shuffle_xor(lo, 1u);
    \\    hi += simd_shuffle_xor(hi, 1u);
    \\    lo += simd_shuffle_xor(lo, 2u);
    \\    hi += simd_shuffle_xor(hi, 2u);
    \\    if ((lane & 3u) == 0u) {
    \\      threadgroup float *pr = tgm + ((r * KPARTS + kph) * TGT + 2u * tgp + o) * 16u + (lane >> 2u);
    \\      pr[0] = exl3_out(lo);
    \\      pr[8] = exl3_out(hi);
    \\    }
    \\  }
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if constexpr (FOLD) {
    \\
++ ROWS_FOLD ++
    \\} else {
    \\  for (uint i = lid; i < uint(ROWS) * TGT * 16u; i += SGS * 32u) {
    \\    const uint r = i / (TGT * 16u);
    \\    const uint c = i - r * TGT * 16u;
    \\    float sum = 0.0f;
    \\    for (uint k = 0u; k < KPARTS; k++) sum += tgm[(r * KPARTS + k) * TGT * 16u + c];
    \\    exl3_store(y + (size_t)(r * uint(NSPLIT) + split) * uint(ODIM) + ot0 * TILE + c, sum);
    \\  }
    \\}
;

/// `ROWS_SOURCE` on the matrix units: a simdgroup owns 32 outputs, decodes them into `matmul2d` weight
/// fragments (the NAX GEMM's) and multiplies every k-tile with a 16-row tile holding the prepared rows
/// (rows past ROWS repeat the last row; their sums are never stored), so a row costs no ALU of its own.
/// A threadgroup owns one 128-wide output block; FOLD as in `ROWS_SOURCE`.
pub const ROWS_NAX_SOURCE: [:0]const u8 = kernels.metalReplaceFound(ROWS_NAX_BODY, "PREP_SLOT", "lane");

const ROWS_NAX_BODY: [:0]const u8 =
    \\constexpr uint SGS = 4u * uint(KPH);
    \\constexpr uint KPARTS = uint(KPH);
    \\constexpr uint TGT = 8u;
    \\constexpr uint PART_F = uint(ROWS) * KPARTS * 128u;
    \\
++ ROWS_HEAD ++
    \\constexpr auto desc = matmul2d_descriptor(16, 32, 16, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
    \\matmul2d<desc, execution_simdgroup> op;
    \\auto left = op.get_left_input_cooperative_tensor<half, half, float>();
    \\auto right = op.get_right_input_cooperative_tensor<half, half, float>();
    \\auto destination = op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>, metal::remove_addrspace_t<decltype(right)>, float>();
    \\nax_zero(destination);
    \\const short2 origin = nax_origin(lane);
    \\const uint xrow = min(uint(origin.y), uint(ROWS) - 1u);
    \\const uint lid = uint(thread_index_in_threadgroup);
    \\const uint col0 = sg & 3u;
    \\const uint kslice = sg >> 2u;
    \\const device uint *wcol = (const device uint *)trellis + ((size_t)tk0 * OT + ot0 + 2u * col0) * PACKED_W;
    \\
++ ROWS_CHUNK_OPEN ++
    \\  const threadgroup half *pl = prep + xrow * uint(CHUNK) + uint(origin.x);
    \\  for (uint tk = c0 + (kslice + uint(KPH) - (c0 - tk0) % uint(KPH)) % uint(KPH); tk < c1; tk += uint(KPH)) {
    \\    const device uint *wp = wcol + (size_t)(tk - tk0) * OT * PACKED_W;
    \\    const nfrag w0 = nax_wfrag_k<N>(wp, lane);
    \\    const nfrag w1 = nax_wfrag_k<N>(wp + PACKED_W, lane);
    \\    for (short s = 0; s < 8; s++) {
    \\      right[s] = w0[s];
    \\      right[8 + s] = w1[s];
    \\    }
    \\    const half4 v = *((const threadgroup half4 *)(pl + (tk - c0) * TILE));
    \\    for (uint c = 0u; c < 4u; c++) {
    \\      left[c] = v[c];
    \\      left[4 + c] = half(0.0h);
    \\    }
    \\    op.run(left, right, destination);
    \\  }
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if (uint(origin.y) < uint(ROWS)) {
    \\  for (uint ct = 0u; ct < 2u; ct++) {
    \\    for (uint c = 0u; c < 4u; c++) {
    \\      const uint col = col0 * 32u + ct * 16u + uint(origin.x) + c;
    \\      if constexpr (FOLD || KPARTS > 1u) tgm[(uint(origin.y) * KPARTS + kslice) * 128u + col] = destination[ct * 8u + c];
    \\      else exl3_store(y + (size_t)(uint(origin.y) * uint(NSPLIT) + split) * uint(ODIM) + ot0 * TILE + col, float(destination[ct * 8u + c]));
    \\    }
    \\  }
    \\}
    \\if constexpr (FOLD) {
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
++ ROWS_FOLD ++
    \\} else if constexpr (KPARTS > 1u) {
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  for (uint i = lid; i < uint(ROWS) * 128u; i += SGS * 32u) {
    \\    const uint r = i / 128u;
    \\    const uint c = i - r * 128u;
    \\    float sum = 0.0f;
    \\    for (uint k = 0u; k < KPARTS; k++) sum += tgm[(r * KPARTS + k) * 128u + c];
    \\    exl3_store(y + (size_t)(r * uint(NSPLIT) + split) * uint(ODIM) + ot0 * TILE + c, sum);
    \\  }
    \\}
;

/// Sums a row's k-split partials, then applies the output H128 + svh.
const ROWS_FINISH_SOURCE: [:0]const u8 =
    \\const uint lane = uint(thread_index_in_simdgroup);
    \\const uint b = uint(threadgroup_position_in_grid.x) * 128u + lane;
    \\const uint r = uint(threadgroup_position_in_grid.z) * uint(ROWS) + uint(threadgroup_position_in_grid.y);
    \\const uint g = uint(threadgroup_position_in_grid.z);
    \\float4 v = 0.0f;
    \\for (uint s = 0u; s < uint(NSPLIT); s++) {
    \\  const device float *p = part + (size_t)(r * uint(NSPLIT) + s) * uint(ODIM) + b;
    \\  v += float4(p[0], p[32], p[64], p[96]);
    \\}
    \\v = exl3_h128(v, lane);
    \\const float psc = 0.08838834764831845f;
    \\device OutT *yr = y + (size_t)g * uint(ROWS) * uint(ODIM) + exl3_seg_at(uint(threadgroup_position_in_grid.y), b, uint(ROWS), uint(ODIM), uint(SB1), uint(SB2), uint(SB3));
    \\yr[0] = OutT(v.x * psc * float(svh[g * uint(ODIM) + b]));
    \\yr[32] = OutT(v.y * psc * float(svh[g * uint(ODIM) + b + 32u]));
    \\yr[64] = OutT(v.z * psc * float(svh[g * uint(ODIM) + b + 64u]));
    \\yr[96] = OutT(v.w * psc * float(svh[g * uint(ODIM) + b + 96u]));
;

pub const PROJECT_ROWS_MAX: c_int = 8;

/// Threadgroup memory a launch stages its prepared rows in: an occupancy bound.
const ROWS_TG_BYTES: c_int = 8192;

/// One multi-row launch: `tgrp` output tile pairs per threadgroup, `kph` simdgroups per pair over its k
/// range, `nsplit` threadgroups over the k dim, `chunk` halves per row staged at a time. `nax` launches
/// `ROWS_NAX_SOURCE`, whose threadgroup is always tgrp 4, kph 1.
pub const RowsGeom = struct { tgrp: c_int, kph: c_int, nsplit: c_int, chunk: c_int, nax: bool = false };

const RowsKey = struct { in_dim: c_int, out_dim: c_int, rows: c_int, groups: c_int, n: u32, out_dt: mlx.mlx_dtype, bounds: [3]c_int, geom: RowsGeom };

const RowsFinishKey = struct { out_dim: c_int, rows: c_int, groups: c_int, nsplit: c_int, out_dt: mlx.mlx_dtype, bounds: [3]c_int };

/// DeepSeek's trunk alone holds a dozen shapes at each of a decode and a verify width, so a smaller cache thrashes.
var rows_cfgs: kernel_src.CfgCache(RowsKey, 64) = .{};

var rows_finish_cfgs: kernel_src.CfgCache(RowsFinishKey, 64) = .{};

var rows_kernel: kernel_src.KernelSlots = kernel_src.no_kernels;

var rows_nax_kernel: kernel_src.KernelSlots = kernel_src.no_kernels;

var rows_finish_kernel: ?mlx.mlx_fast_metal_kernel = null;

pub const ROWS_INS = [_][*:0]const u8{ "x", "trellis", "suh", "svh" };

pub const ROWS_NAX_FRAGS: [:0]const u8 = kernels.GEMM_NAX_INCLUDES ++ kernels.GEMM_NAX_FRAGS ++ ROWS_HELPERS;

pub fn rowsKernel(g: RowsGeom) !mlx.mlx_fast_metal_kernel {
    if (g.nax) return kernel_src.codebookKernelWith(&rows_nax_kernel, "sushi_exl3_rows_nax", &ROWS_INS, &.{"y"}, ROWS_NAX_SOURCE, ROWS_NAX_FRAGS);
    return kernel_src.codebookKernelWith(&rows_kernel, "sushi_exl3_rows", &ROWS_INS, &.{"y"}, ROWS_SOURCE, ROWS_HELPERS);
}

fn rowsFolds(g: RowsGeom) bool {
    return g.tgrp == 4 and g.nsplit == 1;
}

fn rowsChunk(in_dim: c_int, rows: c_int, nsplit: c_int) c_int {
    return @min(@divExact(in_dim, nsplit), @max(128, @divFloor(@divFloor(ROWS_TG_BYTES / 2, rows), 128) * 128));
}

/// Picked on dependent chains of calls: a threadgroup walking a whole k range is
/// latency-bound in a forward, so k splits into ranges of at most 512 over at least 320 threadgroups; only an
/// lm_head-wide output fills the GPU with whole 128-wide blocks and folds the finish.
pub fn rowsGeometry(in_dim: c_int, out_dim: c_int, rows: c_int, groups: c_int) RowsGeom {
    const nax = if (rows_exact) !isLmHead(in_dim, out_dim, groups) else rows >= 2;
    return rowsGeometryFor(in_dim, out_dim, rows, groups, nax and kernels.gemmNaxOn());
}

/// A row has the same bytes at every width: the NAX arm at one row too, and for the lm_head, where the SIMD rows arm is
/// the faster one up to 5 rows and equals its one-row launch bit for bit, the SIMD arm at every width.
pub var rows_exact: bool = false;

fn isLmHead(in_dim: c_int, out_dim: c_int, groups: c_int) bool {
    return groups == 1 and in_dim == 5120 and out_dim == 129280;
}

/// DeepSeek's trunk linears on the NAX arm: `kph` simdgroups split a threadgroup's k range and sum in threadgroup
/// memory, so a launch needs fewer k-split partials (none, and no finish dispatch, at `nsplit` 1). The geometry never
/// depends on the row count, which keeps a row's bytes equal at every width.
const NaxTuned = struct { in_dim: c_int, out_dim: c_int, groups: c_int, kph: c_int, nsplit: c_int };

const NAX_TUNED = [_]NaxTuned{
    .{ .in_dim = 5120, .out_dim = 1792, .groups = 1, .kph = 4, .nsplit = 5 },
    .{ .in_dim = 1280, .out_dim = 32768, .groups = 1, .kph = 4, .nsplit = 1 },
    .{ .in_dim = 4096, .out_dim = 1024, .groups = 8, .kph = 4, .nsplit = 8 },
    .{ .in_dim = 8192, .out_dim = 5120, .groups = 1, .kph = 2, .nsplit = 4 },
    .{ .in_dim = 5120, .out_dim = 129280, .groups = 1, .kph = 1, .nsplit = 2 },
};

fn rowsGeometryFor(in_dim: c_int, out_dim: c_int, rows: c_int, groups: c_int, nax: bool) RowsGeom {
    if (nax) for (NAX_TUNED) |t| {
        if (t.in_dim == in_dim and t.out_dim == out_dim and t.groups == groups)
            return .{ .tgrp = 4, .kph = t.kph, .nsplit = t.nsplit, .chunk = rowsChunk(in_dim, rows, t.nsplit), .nax = true };
    };
    if (@divExact(out_dim, 128) * groups >= 1024) return .{ .tgrp = 4, .kph = if (nax) 1 else 2, .nsplit = 1, .chunk = rowsChunk(in_dim, rows, 1), .nax = nax };
    const blocks = @divExact(in_dim, 128);
    const tgs = groups * @divExact(out_dim, @as(c_int, if (nax) 128 else 32));
    var nsplit: c_int = 1;
    while (nsplit < blocks and (@rem(blocks, nsplit) != 0 or @divExact(in_dim, nsplit) > 512 or tgs * nsplit < 320)) nsplit += 1;
    if (nax) return .{ .tgrp = 4, .kph = 1, .nsplit = nsplit, .chunk = rowsChunk(in_dim, rows, nsplit), .nax = true };
    // From two rows a threadgroup spans two output tile pairs, which prepare each row's k range once for both; the sums are
    // unchanged, and 256 threads stay inside what M1 and M2 grant a kernel at full register use (384).
    return .{ .tgrp = if (rows >= 2) 2 else 1, .kph = 4, .nsplit = nsplit, .chunk = rowsChunk(in_dim, rows, nsplit) };
}

/// `y[r] = svh * H128(W . H128(suh * x[r]))` for every row of `x` (`[rows, in]` f16, bf16 or f32, rows 1..8), over one
/// trellis `[in/16, out/16, n]` (or a one-matrix bank), its `suh [in]` and `svh [out]`: `projectIndexed`'s
/// numbers, each weight decoded once for all rows. Returns `[rows, out]` in x's dtype, rounded once from the f32 sum. A bank `[G, in/16, out/16, n]` takes
/// `x` `[G * rows, in]`, group g's rows contiguous, `suh [G, in]` and `svh [G, out]`, and returns `[G * rows, out]`.
pub fn projectRows(s: mlx.mlx_stream, x: mlx.mlx_array, trellis: mlx.mlx_array, suh: mlx.mlx_array, svh: mlx.mlx_array) !mlx.mlx_array {
    const xs = mlx.getShape(x);
    const ts = mlx.getShape(trellis);
    if (xs.len != 2 or ts.len < 3) return error.BadExl3Shape;
    const groups: c_int = if (ts.len == 4) ts[0] else 1;
    const g = rowsGeometry(xs[1], ts[ts.len - 2] * 16, @divExact(xs[0], groups), groups);
    return projectRowsWith(s, x, trellis, suh, svh, g, try rowsKernel(g), null);
}

/// `projectRows` over linears that share their input, stacked along the output (one trellis `[1, in/16, out/16, n]`,
/// `suh [S, in]` one row per linear, `svh [out]`): linear s owns the columns from `bounds[s - 1]` (0 for s = 0) to
/// `bounds[s]` (`out` past the last), every bound a multiple of 128. Returns `[rows * out]`, each linear's
/// `[rows, width]` contiguous at `start * rows`.
pub fn projectRowsSegmented(s: mlx.mlx_stream, x: mlx.mlx_array, trellis: mlx.mlx_array, suh: mlx.mlx_array, svh: mlx.mlx_array, bounds: [3]c_int) !mlx.mlx_array {
    const xs = mlx.getShape(x);
    const ts = mlx.getShape(trellis);
    if (xs.len != 2 or ts.len != 4 or ts[0] != 1) return error.BadExl3Shape;
    const out = ts[2] * 16;
    var segs: usize = 1;
    var prev: c_int = 0;
    for (bounds) |b| {
        if (b < prev or b > out or @rem(b, 128) != 0) return error.BadExl3Shape;
        if (b > prev and b < out) segs += 1;
        prev = b;
    }
    if (mlx.mlx_array_size(suh) != segs * @as(usize, @intCast(xs[1]))) return error.BadExl3Shape;
    const g = rowsGeometry(xs[1], out, xs[0], 1);
    const y = try projectRowsWith(s, x, trellis, suh, svh, g, try rowsKernel(g), bounds);
    defer _ = mlx.mlx_array_free(y);
    var flat = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, y, &[_]c_int{xs[0] * out}, 1, s));
    return flat;
}

pub fn projectRowsWith(s: mlx.mlx_stream, x: mlx.mlx_array, trellis: mlx.mlx_array, suh: mlx.mlx_array, svh: mlx.mlx_array, g: RowsGeom, kernel: mlx.mlx_fast_metal_kernel, seg_bounds: ?[3]c_int) !mlx.mlx_array {
    const xs = mlx.getShape(x);
    const ts = mlx.getShape(trellis);
    if (xs.len != 2 or !(ts.len == 3 or ts.len == 4)) return error.BadExl3Shape;
    const groups: c_int = if (ts.len == 4) ts[0] else 1;
    const rows = @divExact(xs[0], groups);
    const in_dim = xs[1];
    const out_tiles = ts[ts.len - 2];
    if (rows < 1 or rows > PROJECT_ROWS_MAX or @rem(in_dim, 128) != 0 or ts[ts.len - 3] * 16 != in_dim or @rem(out_tiles, 8) != 0) return error.BadExl3Shape;
    switch (mlx.mlx_array_dtype(x)) {
        .float16, .bfloat16, .float32 => {},
        else => return error.BadExl3Shape,
    }
    if (mlx.mlx_array_dtype(trellis) != .uint16) return error.BadExl3Shape;
    const out_dim = out_tiles * 16;
    const out_dt = mlx.mlx_array_dtype(x);
    const rate = try kernels.packedRate(ts[ts.len - 1]);
    const fold = rowsFolds(g);
    const bounds = seg_bounds orelse @as([3]c_int, @splat(out_dim));
    const key = RowsKey{ .in_dim = in_dim, .out_dim = out_dim, .rows = rows, .groups = groups, .n = rate.n, .out_dt = out_dt, .bounds = bounds, .geom = g };
    const cfg = rows_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const sgs = g.tgrp * g.kph;
        const out_shape = [_]c_int{ groups * if (fold) rows else rows * g.nsplit, out_dim };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &out_shape, 2, if (fold) out_dt else .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "OutT", out_dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, @divExact(out_tiles, 2 * g.tgrp) * sgs * 32, g.nsplit, groups));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, sgs * 32, 1, 1));
        inline for (.{ "IDIM", "ODIM", "ROWS", "GROUPS", "TGRP", "KPH", "NSPLIT", "CHUNK", "FOLD" }, .{ in_dim, out_dim, rows, groups, g.tgrp, g.kph, g.nsplit, g.chunk, @as(c_int, @intFromBool(fold)) }) |name, v|
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, name, v));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "NHW", @intCast(rate.n)));
        inline for (.{ "SB1", "SB2", "SB3" }, 0..) |name, i| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, name, bounds[i]));
        rows_cfgs.put(key, c);
        break :blk c;
    };
    const ov = try kernels.applyOuts(s, kernel, &.{ x, trellis, suh, svh }, cfg, 1);
    defer _ = mlx.mlx_vector_array_free(ov);
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, ov, 0));
    if (fold) return y;
    defer _ = mlx.mlx_array_free(y);
    return rowsFinish(s, y, svh, out_dim, rows, groups, g.nsplit, out_dt, bounds);
}

fn rowsFinish(s: mlx.mlx_stream, part: mlx.mlx_array, svh: mlx.mlx_array, out_dim: c_int, rows: c_int, groups: c_int, nsplit: c_int, out_dt: mlx.mlx_dtype, bounds: [3]c_int) !mlx.mlx_array {
    const key = RowsFinishKey{ .out_dim = out_dim, .rows = rows, .groups = groups, .nsplit = nsplit, .out_dt = out_dt, .bounds = bounds };
    const cfg = rows_finish_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &.{ groups * rows, out_dim }, 2, out_dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "OutT", out_dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, @divExact(out_dim, 128) * 32, rows, groups));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ODIM", out_dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ROWS", rows));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "NSPLIT", nsplit));
        inline for (.{ "SB1", "SB2", "SB3" }, 0..) |name, i| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, name, bounds[i]));
        rows_finish_cfgs.put(key, c);
        break :blk c;
    };
    const kernel = try kernels.getNamedKernel(&rows_finish_kernel, "sushi_exl3_rows_finish", &.{ "part", "svh" }, &.{"y"}, ROWS_FINISH_SOURCE, ROWS_HELPERS);
    const ov = try kernels.applyOuts(s, kernel, &.{ part, svh }, cfg, 1);
    defer _ = mlx.mlx_vector_array_free(ov);
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, ov, 0));
    return y;
}

fn rowsVerdict(c: kernels.RowsCase) !?kernels.Exl3GemmParityFail {
    const st = try kernels.rowsParityStats(c);
    return kernels.exl3GemmParityVerdict(st.finite, st.kern_max, st.rms_kern, st.rms_comp, st.ceiling);
}

/// A random bank of `groups` matrices with up to `PROJECT_ROWS_MAX` activation rows each, run through a chosen multi-row
/// kernel; the output is the f16 bits, `[groups * rows, out]`.
const RowsRig = struct {
    in: usize,
    out: usize,
    groups: usize,
    n: u32,
    tr: []u16,
    suh: []u16,
    svh: []u16,
    xh: []u16,

    fn init(alloc: std.mem.Allocator, seed: u64, in: usize, out: usize, groups: usize, n: u32) !RowsRig {
        var prng = std.Random.DefaultPrng.init(seed);
        const rnd = prng.random();
        const rate: exl3.Rate = .{ .n = n };
        const tr = try alloc.alloc(u16, groups * (in / 16) * (out / 16) * rate.halfwords());
        rnd.bytes(std.mem.sliceAsBytes(tr));
        const suh = try alloc.alloc(u16, groups * in);
        const svh = try alloc.alloc(u16, groups * out);
        for ([2][]u16{ suh, svh }) |v| for (v) |*e| {
            const mag = std.math.exp2(rnd.float(f32) * 4 - 2);
            e.* = exl3.f32ToF16Bits(if (rnd.boolean()) mag else -mag);
        };
        const xh = try alloc.alloc(u16, groups * @as(usize, @intCast(PROJECT_ROWS_MAX)) * in);
        for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
        return .{ .in = in, .out = out, .groups = groups, .n = n, .tr = tr, .suh = suh, .svh = svh, .xh = xh };
    }

    /// Rows `first..first + rows` of every group through `kernel` under `geom`.
    fn run(self: RowsRig, alloc: std.mem.Allocator, first: usize, rows: usize, geom: RowsGeom, kernel: mlx.mlx_fast_metal_kernel) ![]const u16 {
        const s = mlx.gpuStream();
        const pool: usize = @intCast(PROJECT_ROWS_MAX);
        const xs = try alloc.alloc(u16, self.groups * rows * self.in);
        for (0..self.groups) |g| @memcpy(xs[g * rows * self.in ..][0 .. rows * self.in], self.xh[(g * pool + first) * self.in ..][0 .. rows * self.in]);
        const rate: exl3.Rate = .{ .n = self.n };
        const x = mlx.mlx_array_new_data(xs.ptr, &[_]c_int{ @intCast(self.groups * rows), @intCast(self.in) }, 2, .float16);
        defer _ = mlx.mlx_array_free(x);
        const tr = mlx.mlx_array_new_data(self.tr.ptr, &[_]c_int{ @intCast(self.groups), @intCast(self.in / 16), @intCast(self.out / 16), @intCast(rate.halfwords()) }, 4, .uint16);
        defer _ = mlx.mlx_array_free(tr);
        const suh = mlx.mlx_array_new_data(self.suh.ptr, &[_]c_int{ @intCast(self.groups), @intCast(self.in) }, 2, .float16);
        defer _ = mlx.mlx_array_free(suh);
        const svh = mlx.mlx_array_new_data(self.svh.ptr, &[_]c_int{ @intCast(self.groups), @intCast(self.out) }, 2, .float16);
        defer _ = mlx.mlx_array_free(svh);
        const y = try projectRowsWith(s, x, tr, suh, svh, geom, kernel, null);
        defer _ = mlx.mlx_array_free(y);
        var yc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(yc);
        const got = try kernels.evalF16(s, y, &yc);
        return alloc.dupe(u16, @as([*]const u16, @ptrCast(got))[0 .. self.groups * rows * self.out]);
    }
};

const SIMD_ROWS_DECODES = [_]struct { dec: exl3.Decode, n: u32 }{
    .{ .dec = exl3.Decode.tabled(.spec1, .w15), .n = 64 },
    .{ .dec = .{ .codebook = .pair1, .window = .w16 }, .n = 32 },
    .{ .dec = .{ .codebook = .mcg, .window = .w15 }, .n = 48 },
};

fn rowsBaselineKernel() !mlx.mlx_fast_metal_kernel {
    return kernel_src.codebookKernelWith(&kernels.rows_baseline_kernel, "sushi_exl3_rows_baseline", &ROWS_INS, &.{"y"}, kernels.ROWS_BASELINE_SOURCE, ROWS_HELPERS);
}

test "exl3 multi-row projection matches projectIndexed per row against the f64 projection" {
    if (!mlx.streamIsGpu(mlx.gpuStream())) return error.SkipZigTest;
    const shapes = [_][2]usize{ .{ 512, 640 }, .{ 6144, 384 } };
    for (0..kernels.PARITY_SEEDS) |i| {
        for ([_]u32{ 40, 64, 112 }) |n| {
            for (shapes) |sh| {
                for (1..5) |rows| {
                    try kernels.reportGemmParity(try kernels.rowsParityStats(.{ .n = n, .out_dim = sh[0], .in_dim = sh[1], .rows = rows, .seed = 1301 + i }));
                }
            }
        }
    }
}

test "exl3 multi-row projection holds the bar on every launch geometry and at eight rows" {
    if (!mlx.streamIsGpu(mlx.gpuStream())) return error.SkipZigTest;
    const geoms = [_]RowsGeom{
        .{ .tgrp = 4, .kph = 4, .nsplit = 1, .chunk = 256 },
        .{ .tgrp = 4, .kph = 2, .nsplit = 1, .chunk = 768 },
        .{ .tgrp = 4, .kph = 1, .nsplit = 2, .chunk = 384 },
        .{ .tgrp = 1, .kph = 4, .nsplit = 3, .chunk = 128 },
        .{ .tgrp = 2, .kph = 1, .nsplit = 6, .chunk = 128 },
        .{ .tgrp = 4, .kph = 1, .nsplit = 1, .chunk = 256, .nax = true },
        .{ .tgrp = 4, .kph = 1, .nsplit = 3, .chunk = 256, .nax = true },
        .{ .tgrp = 4, .kph = 4, .nsplit = 1, .chunk = 256, .nax = true },
        .{ .tgrp = 4, .kph = 2, .nsplit = 3, .chunk = 256, .nax = true },
        .{ .tgrp = 4, .kph = 3, .nsplit = 2, .chunk = 384, .nax = true },
        .{ .tgrp = 4, .kph = 8, .nsplit = 1, .chunk = 256, .nax = true },
    };
    for (geoms, 0..) |g, gi| {
        if (g.nax and !kernels.gemmNaxOn()) continue;
        for ([_]usize{ 1, 3, 8 }) |rows| {
            for ([_]u32{ 48, 96, 128 }) |n| {
                try kernels.reportGemmParity(try kernels.rowsParityStats(.{ .n = n, .out_dim = 1024, .in_dim = 768, .rows = rows, .seed = 1401 + gi, .geom = g }));
            }
        }
    }
}

test "exl3 multi-row projection holds the bar at every admitted rate on both kernels" {
    if (!mlx.streamIsGpu(mlx.gpuStream())) return error.SkipZigTest;
    var n: u32 = exl3.Rate.min_n;
    while (n <= exl3.Rate.max_n) : (n += 2) {
        try kernels.reportGemmParity(try kernels.rowsParityStats(.{ .n = n, .out_dim = 256, .in_dim = 256, .rows = 3, .seed = 1601 + n, .geom = .{ .tgrp = 1, .kph = 4, .nsplit = 2, .chunk = 128 } }));
    }
}

test "exl3 the multi-row parity bar convicts a wrong row and a wrong window" {
    if (!mlx.streamIsGpu(mlx.gpuStream())) return error.SkipZigTest;
    const t = std.testing;
    for ([_][2]usize{ .{ 512, 640 }, .{ 6144, 384 } }) |sh| {
        for (2..5) |rows| {
            try t.expect(try rowsVerdict(.{ .n = 112, .out_dim = sh[0], .in_dim = sh[1], .rows = rows, .seed = 1501, .mutant_rows = true }) != null);
        }
        try t.expect(try rowsVerdict(.{ .n = 112, .out_dim = sh[0], .in_dim = sh[1], .rows = 3, .seed = 1503, .mutate = .window }) != null);
    }
}

// A SIMD multi-row launch computes each row exactly as a one-row launch does: the k split and the order of
// every row's sums do not depend on the row count, so a verify row on a machine without NAX equals its serial step.
test "exl3 SIMD multi-row: every row equals the one-row launch bit for bit" {
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    defer kernels.setDecodeParams(.mul1);
    for (SIMD_ROWS_DECODES) |d| {
        try d.dec.requireTable();
        kernels.setDecodeParams(d.dec);
        for ([_][3]usize{ .{ 1024, 512, 1 }, .{ 384, 256, 2 }, .{ 2304, 640, 1 } }) |sh| {
            const rig = try RowsRig.init(alloc, 0x51d0 + sh[0], sh[0], sh[1], sh[2], d.n);
            const g1 = rowsGeometryFor(@intCast(sh[0]), @intCast(sh[1]), 1, @intCast(sh[2]), false);
            for (1..@as(usize, @intCast(PROJECT_ROWS_MAX)) + 1) |rows| {
                const g = rowsGeometryFor(@intCast(sh[0]), @intCast(sh[1]), @intCast(rows), @intCast(sh[2]), false);
                const many = try rig.run(alloc, 0, rows, g, try rowsKernel(g));
                for (0..rows) |r| {
                    const one = try rig.run(alloc, r, 1, g1, try rowsKernel(g1));
                    for (0..sh[2]) |grp| try std.testing.expectEqualSlices(u16, one[grp * sh[1] ..][0..sh[1]], many[(grp * rows + r) * sh[1] ..][0..sh[1]]);
                }
            }
        }
    }
}

// A NAX multi-row launch sums a row in an order set by its k slicing alone, so a verify row has the same bytes at every
// width from two up; the served geometry may not read the row count.
test "exl3 NAX multi-row: a row has the same bytes at widths 2 to 8 under every k slicing" {
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s) or !kernels.gemmNaxOn()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    defer kernels.setDecodeParams(.mul1);
    var failed: usize = 0;
    for (SIMD_ROWS_DECODES) |d| {
        try d.dec.requireTable();
        kernels.setDecodeParams(d.dec);
        for ([_][3]usize{ .{ 1280, 512, 1 }, .{ 1024, 256, 2 } }) |sh| {
            const rig = try RowsRig.init(alloc, 0x4a58 + sh[0], sh[0], sh[1], sh[2], d.n);
            const in: c_int = @intCast(sh[0]);
            var served: [2]RowsGeom = undefined;
            for ([_]c_int{ 2, 8 }, &served) |rows, *g| g.* = rowsGeometry(in, @intCast(sh[1]), rows, @intCast(sh[2]));
            try std.testing.expectEqual(served[0].kph, served[1].kph);
            try std.testing.expectEqual(served[0].nsplit, served[1].nsplit);
            const blocks = @divExact(in, 128);
            for ([_]c_int{ 1, 2, 3, 4, 8 }) |kph| for ([_]c_int{ 1, 2, 4, 5 }) |nsplit| {
                if (@rem(blocks, nsplit) != 0) continue;
                const at = RowsGeom{ .tgrp = 4, .kph = kph, .nsplit = nsplit, .chunk = rowsChunk(in, 8, nsplit), .nax = true };
                const ref = try rig.run(alloc, 0, 8, at, try rowsKernel(at));
                for (2..8) |rows| {
                    const g = RowsGeom{ .tgrp = 4, .kph = kph, .nsplit = nsplit, .chunk = rowsChunk(in, @intCast(rows), nsplit), .nax = true };
                    const many = try rig.run(alloc, 0, rows, g, try rowsKernel(g));
                    for (0..sh[2]) |grp| for (0..rows) |r| {
                        if (std.mem.eql(u16, ref[(grp * 8 + r) * sh[1] ..][0..sh[1]], many[(grp * rows + r) * sh[1] ..][0..sh[1]])) continue;
                        std.debug.print("NAX rows differ: n{d} {d}x{d} g{d} kph {d} nsplit {d} rows {d} row {d}\n", .{ d.n, sh[0], sh[1], sh[2], kph, nsplit, rows, r });
                        failed += 1;
                    };
                }
            };
        }
    }
    try std.testing.expectEqual(@as(usize, 0), failed);
}

test "exl3 SIMD multi-row: the split row loops and the launch policy leave every output byte unchanged" {
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    defer kernels.setDecodeParams(.mul1);
    const geoms = [_]RowsGeom{
        .{ .tgrp = 4, .kph = 4, .nsplit = 1, .chunk = 256 },
        .{ .tgrp = 4, .kph = 2, .nsplit = 1, .chunk = 768 },
        .{ .tgrp = 4, .kph = 1, .nsplit = 2, .chunk = 384 },
        .{ .tgrp = 1, .kph = 4, .nsplit = 3, .chunk = 128 },
        .{ .tgrp = 2, .kph = 4, .nsplit = 2, .chunk = 384 },
        .{ .tgrp = 2, .kph = 1, .nsplit = 6, .chunk = 128 },
    };
    for (SIMD_ROWS_DECODES) |d| {
        try d.dec.requireTable();
        kernels.setDecodeParams(d.dec);
        for ([_]usize{ 1, 2 }) |groups| {
            const rig = try RowsRig.init(alloc, 0x5b17 + groups, 768, 1024, groups, d.n);
            for (geoms) |g| for ([_]usize{ 1, 3, 5, 8 }) |rows| {
                const want = try rig.run(alloc, 0, rows, g, try rowsBaselineKernel());
                const got = try rig.run(alloc, 0, rows, g, try rowsKernel(g));
                try std.testing.expectEqualSlices(u16, want, got);
            };
        }
        for ([_][3]usize{ .{ 1024, 512, 1 }, .{ 2304, 640, 1 }, .{ 4096, 1024, 8 }, .{ 128, 131072, 1 } }) |sh| {
            const rig = try RowsRig.init(alloc, 0x5b99 + sh[0], sh[0], sh[1], sh[2], d.n);
            for (1..@as(usize, @intCast(PROJECT_ROWS_MAX)) + 1) |rows| {
                const g = rowsGeometryFor(@intCast(sh[0]), @intCast(sh[1]), @intCast(rows), @intCast(sh[2]), false);
                var before = g;
                if (g.nsplit > 1) before.tgrp = 1;
                const want = try rig.run(alloc, 0, rows, before, try rowsBaselineKernel());
                const got = try rig.run(alloc, 0, rows, g, try rowsKernel(g));
                try std.testing.expectEqualSlices(u16, want, got);
            }
        }
    }
}

test "exl3 multi-row projection is one dispatch on an lm_head-wide output and two on a narrower one" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    kernels.setDecodeParams(.{ .codebook = .mcg, .window = .w15 });
    defer kernels.setDecodeParams(.mul1);
    for ([_][3]c_int{ .{ 131072, 256, 1 }, .{ 6144, 2560, 2 }, .{ 512, 2560, 2 } }) |c| {
        const words = try t.allocator.alloc(u16, @intCast(@divExact(c[0], 16) * @divExact(c[1], 16) * 112));
        defer t.allocator.free(words);
        @memset(words, 0x5a5a);
        const ones = try t.allocator.alloc(u16, @intCast(@max(c[0], c[1]) * 4));
        defer t.allocator.free(ones);
        @memset(ones, exl3.f32ToF16Bits(1.0));
        const x = mlx.mlx_array_new_data(ones.ptr, &[_]c_int{ 4, c[1] }, 2, .float16);
        defer _ = mlx.mlx_array_free(x);
        const tr = mlx.mlx_array_new_data(words.ptr, &[_]c_int{ @divExact(c[1], 16), @divExact(c[0], 16), 112 }, 3, .uint16);
        defer _ = mlx.mlx_array_free(tr);
        const suh = mlx.mlx_array_new_data(ones.ptr, &[_]c_int{c[1]}, 1, .float16);
        defer _ = mlx.mlx_array_free(suh);
        const svh = mlx.mlx_array_new_data(ones.ptr, &[_]c_int{c[0]}, 1, .float16);
        defer _ = mlx.mlx_array_free(svh);
        kernels.resetFusedDispatchCount();
        const y = try projectRows(s, x, tr, suh, svh);
        defer _ = mlx.mlx_array_free(y);
        try t.expectEqual(@as(u32, @intCast(c[2])), kernels.fusedDispatchCount());
        try t.expectEqualSlices(c_int, &.{ 4, c[0] }, mlx.getShape(y));
    }
}
