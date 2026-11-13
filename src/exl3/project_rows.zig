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

/// A chunk's read of row r's 128-block at k: the input times suh, through the input H128.
const ROWS_X_PLAIN: []const u8 =
    \\    const auto xr = x + (size_t)r * uint(IDIM) + k;
    \\    const float4 v = exl3_h128(float4(
    \\      float(xr[lane]) * float(suh[k + lane]),
    \\      float(xr[lane + 32u]) * float(suh[k + lane + 32u]),
    \\      float(xr[lane + 64u]) * float(suh[k + lane + 64u]),
    \\      float(xr[lane + 96u]) * float(suh[k + lane + 96u])), lane);
    \\
;

/// `ROWS_X_PLAIN` where `x` holds the k-split partials of the launch before this one (`[XG * ROWS * XSPLIT, XODIM]` f32,
/// that launch's bank g making input columns `g * XODIM` on): the block is first finished as `ROWS_FINISH_SOURCE`
/// finishes it, in its order, and rounded to that launch's output type `XT`, so the prepared row is the same bytes.
const ROWS_X_PENDING: []const u8 =
    \\    static_assert(GROUPS == 1, "a launch that reads partials has one bank");
    \\    const uint xg = k / uint(XODIM);
    \\    const uint xc = k - xg * uint(XODIM) + lane;
    \\    float4 xv = 0.0f;
    \\    for (uint xs = 0u; xs < uint(XSPLIT); xs++) {
    \\      const device float *xp = x + ((size_t)(xg * uint(ROWS) + r) * uint(XSPLIT) + xs) * uint(XODIM) + xc;
    \\      xv += float4(xp[0], xp[32], xp[64], xp[96]);
    \\    }
    \\    xv = exl3_h128(xv, lane);
    \\    const auto xw = xsvh + (size_t)xg * uint(XODIM) + xc;
    \\    xv = float4(float(XT(xv.x * psc * float(xw[0]))), float(XT(xv.y * psc * float(xw[32]))), float(XT(xv.z * psc * float(xw[64]))), float(XT(xv.w * psc * float(xw[96]))));
    \\    const float4 v = exl3_h128(float4(
    \\      xv.x * float(suh[k + lane]),
    \\      xv.y * float(suh[k + lane + 32u]),
    \\      xv.z * float(suh[k + lane + 64u]),
    \\      xv.w * float(suh[k + lane + 96u])), lane);
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
    \\
++ ROWS_X_PLAIN ++
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
/// A threadgroup owns one 128-wide output block; FOLD as in `ROWS_SOURCE`. UNR k-tiles' fragments are read
/// before their MMAs run, in the one-tile walk's order; R32 reads them on 32-bit funnels where they fit.
pub const ROWS_NAX_SOURCE: [:0]const u8 = kernels.metalReplaceFound(ROWS_NAX_BODY, "PREP_SLOT", "lane");

const ROWS_NAX_BODY: [:0]const u8 =
    \\static_assert(UNR == 1 || KPH == 1, "an unrolled walk steps whole chunks of one slice");
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
    \\const uint xrow8 = min(uint(origin.y) + 8u, uint(ROWS) - 1u);
    \\const uint lid = uint(thread_index_in_threadgroup);
    \\const uint col0 = sg & 3u;
    \\const uint kslice = sg >> 2u;
    \\const device uint *wcol = (const device uint *)trellis + ((size_t)tk0 * OT + ot0 + 2u * col0) * PACKED_W;
    \\
++ ROWS_CHUNK_OPEN ++
    \\  const threadgroup half *pl = prep + xrow * uint(CHUNK) + uint(origin.x);
    \\  const threadgroup half *pl8 = prep + xrow8 * uint(CHUNK) + uint(origin.x);
    \\  for (uint tk = c0 + (kslice + uint(KPH) - (c0 - tk0) % uint(KPH)) % uint(KPH); tk < c1; tk += uint(UNR) * uint(KPH)) {
    \\    nfrag wf[UNR][2];
    \\    for (uint u = 0u; u < uint(UNR); u++) {
    \\      const device uint *wp = wcol + (size_t)(tk + u * uint(KPH) - tk0) * OT * PACKED_W;
    \\      wf[u][0] = rows_wfrag<N, R32, SW>(wp, lane);
    \\      wf[u][1] = rows_wfrag<N, R32, SW>(wp + PACKED_W, lane);
    \\    }
    \\    for (uint u = 0u; u < uint(UNR); u++) {
    \\      for (short s = 0; s < 8; s++) {
    \\        right[s] = wf[u][0][s];
    \\        right[8 + s] = wf[u][1][s];
    \\      }
    \\      const half4 v = *((const threadgroup half4 *)(pl + (tk + u * uint(KPH) - c0) * TILE));
    \\      const half4 v8 = ROWS > 8 ? *((const threadgroup half4 *)(pl8 + (tk + u * uint(KPH) - c0) * TILE)) : half4(0.0h);
    \\      for (uint c = 0u; c < 4u; c++) {
    \\        left[c] = v[c];
    \\        left[4 + c] = v8[c];
    \\      }
    \\      op.run(left, right, destination);
    \\    }
    \\  }
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\for (uint h = 0u; h < (ROWS > 8 ? 2u : 1u); h++) {
    \\  const uint row = uint(origin.y) + 8u * h;
    \\  if (row >= uint(ROWS)) continue;
    \\  for (uint ct = 0u; ct < 2u; ct++) {
    \\    for (uint c = 0u; c < 4u; c++) {
    \\      const uint col = col0 * 32u + ct * 16u + uint(origin.x) + c;
    \\      if constexpr (FOLD || KPARTS > 1u) tgm[(row * KPARTS + kslice) * 128u + col] = destination[ct * 8u + 4u * h + c];
    \\      else exl3_store(y + (size_t)(row * uint(NSPLIT) + split) * uint(ODIM) + ot0 * TILE + col, float(destination[ct * 8u + 4u * h + c]));
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
/// Where a finished block of a bank's row lands: bank-major rows, segmented.
const ROWS_FINISH_AT =
    \\device OutT *yr = y + (size_t)g * uint(ROWS) * uint(ODIM) + exl3_seg_at(uint(threadgroup_position_in_grid.y), b, uint(ROWS), uint(ODIM), uint(SB1), uint(SB2), uint(SB3));
;

/// The same block of an unsegmented launch with the banks side by side in each row: `[rows, groups * out]`.
const ROWS_FINISH_AT_ROW_MAJOR =
    \\device OutT *yr = y + ((size_t)threadgroup_position_in_grid.y * uint(GRP) + g) * uint(ODIM) + b;
    \\
;

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
++ ROWS_FINISH_AT ++
    \\yr[0] = OutT(v.x * psc * float(svh[g * uint(ODIM) + b]));
    \\yr[32] = OutT(v.y * psc * float(svh[g * uint(ODIM) + b + 32u]));
    \\yr[64] = OutT(v.z * psc * float(svh[g * uint(ODIM) + b + 64u]));
    \\yr[96] = OutT(v.w * psc * float(svh[g * uint(ODIM) + b + 96u]));
;

const ROWS_FINISH_ROW_MAJOR_SOURCE: [:0]const u8 = kernels.metalReplaceFound(ROWS_FINISH_SOURCE, ROWS_FINISH_AT, ROWS_FINISH_AT_ROW_MAJOR);

/// `ROWS_FINISH_SOURCE` of a one-bank launch whose row `y` leaves as HC streams: stream i is
/// `fma(post[i], y, sum over j of fma(comb[j][i], x[j], acc))` rounded to OutT, `y` rounded to OutT first.
const ROWS_FINISH_EXPAND_SOURCE: [:0]const u8 =
    \\#pragma clang fp contract(off)
    \\const uint lane = uint(thread_index_in_simdgroup);
    \\const uint b = uint(threadgroup_position_in_grid.x) * 128u + lane;
    \\const uint r = uint(threadgroup_position_in_grid.y);
    \\float4 v = 0.0f;
    \\for (uint s = 0u; s < uint(NSPLIT); s++) {
    \\  const device float *p = part + (size_t)(r * uint(NSPLIT) + s) * uint(ODIM) + b;
    \\  v += float4(p[0], p[32], p[64], p[96]);
    \\}
    \\v = exl3_h128(v, lane);
    \\const float psc = 0.08838834764831845f;
    \\const size_t row = (size_t)r * uint(HC);
    \\for (uint q = 0u; q < 4u; q++) {
    \\  const uint col = b + 32u * q;
    \\  const float yv = float(OutT(v[q] * psc * float(svh[col])));
    \\  float sv[HC];
    \\  for (uint j = 0u; j < uint(HC); j++) sv[j] = float(x[(row + j) * uint(ODIM) + col]);
    \\  for (uint i = 0u; i < uint(HC); i++) {
    \\    float acc = 0.0f;
    \\    for (uint j = 0u; j < uint(HC); j++) acc = metal::fma(comb[(row + j) * uint(HC) + i], sv[j], acc);
    \\    y[(row + i) * uint(ODIM) + col] = OutT(metal::fma(post[row + i], yv, acc));
    \\  }
    \\}
;

/// `ROWS_FINISH_SOURCE` for a stacked launch whose first NORM columns leave rms-normed with `w`, one threadgroup a
/// row. The norm is MLX's `rms_single_row` on the finished row, sum for sum: a thread squares four neighbours with a
/// fused multiply-add (as MLX's compiled kernel does: f32 outputs show the ulp), a simdgroup sums its 128 columns,
/// simdgroup 0 the simdgroups', and an output is `w * OutT(x * inv)`.
const ROWS_FINISH_RMS_SOURCE: [:0]const u8 =
    \\#pragma clang fp contract(off)
    \\const uint lane = uint(thread_index_in_simdgroup);
    \\const uint sg = uint(simdgroup_index_in_threadgroup);
    \\const uint r = uint(threadgroup_position_in_grid.y);
    \\threadgroup float fin[NORM];
    \\threadgroup float sums[32];
    \\threadgroup float inv_mean[1];
    \\const uint b = sg * 128u + lane;
    \\const bool normed = sg * 128u < uint(NORM);
    \\float4 v = 0.0f;
    \\for (uint s = 0u; s < uint(NSPLIT); s++) {
    \\  const device float *p = part + (size_t)(r * uint(NSPLIT) + s) * uint(ODIM) + b;
    \\  v += float4(p[0], p[32], p[64], p[96]);
    \\}
    \\v = exl3_h128(v, lane);
    \\const float psc = 0.08838834764831845f;
    \\const OutT o0 = OutT(v.x * psc * float(svh[b]));
    \\const OutT o1 = OutT(v.y * psc * float(svh[b + 32u]));
    \\const OutT o2 = OutT(v.z * psc * float(svh[b + 64u]));
    \\const OutT o3 = OutT(v.w * psc * float(svh[b + 96u]));
    \\if (normed) {
    \\  fin[b] = float(o0);
    \\  fin[b + 32u] = float(o1);
    \\  fin[b + 64u] = float(o2);
    \\  fin[b + 96u] = float(o3);
    \\} else {
    \\  device OutT *yr = y + exl3_seg_at(r, b, uint(ROWS), uint(ODIM), uint(SB1), uint(SB2), uint(SB3));
    \\  yr[0] = o0;
    \\  yr[32] = o1;
    \\  yr[64] = o2;
    \\  yr[96] = o3;
    \\}
    \\if (sg == 0u) sums[lane] = 0.0f;
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\float tx[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    \\float acc = 0.0f;
    \\if (normed) {
    \\  for (int i = 0; i < 4; i++) {
    \\    tx[i] = fin[sg * 128u + lane * 4u + uint(i)];
    \\    acc = metal::fma(tx[i], tx[i], acc);
    \\  }
    \\}
    \\acc = simd_sum(acc);
    \\if (lane == 0u && normed) sums[sg] = acc;
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if (sg == 0u) {
    \\  acc = simd_sum(sums[lane]);
    \\  if (lane == 0u) inv_mean[0] = metal::precise::rsqrt(acc / uint(NORM) + eps[0]);
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if (normed) {
    \\  const uint c0 = sg * 128u + lane * 4u;
    \\  device OutT *yr = y + exl3_seg_at(r, c0, uint(ROWS), uint(ODIM), uint(SB1), uint(SB2), uint(SB3));
    \\  for (int i = 0; i < 4; i++) yr[i] = w[c0 + uint(i)] * OutT(tx[i] * inv_mean[0]);
    \\}
;

pub const PROJECT_ROWS_MAX: c_int = 8;

/// The NAX arm runs one 16-row MMA tile whatever the row count, so it serves up to 16 rows on the same sums.
pub const NAX_ROWS_MAX: c_int = 16;

/// How a multi-row launch is built, per call.
pub const Mode = struct {
    /// A row has the same bytes at every width: the NAX arm at one row too, and for the lm_head, where the SIMD rows
    /// arm is the faster one up to 5 rows and equals its one-row launch bit for bit, the SIMD arm at every width.
    exact: bool = false,
    /// A spec verify's trunk launches: with `exact` on a NAX machine, rows past `PROJECT_ROWS_MAX` stay on the NAX arm
    /// instead of the sorted GEMM, so each still has its decode tick's bytes. Prefill chunks keep the GEMM.
    wide: bool = false,
};

/// The most rows a multi-row launch serves in `mode`.
pub fn rowsMax(mode: Mode) c_int {
    return if (mode.wide and mode.exact and kernels.gemmNaxOn()) NAX_ROWS_MAX else PROJECT_ROWS_MAX;
}

/// Threadgroup memory a launch stages its prepared rows in: an occupancy bound.
const ROWS_TG_BYTES: c_int = 8192;

/// One multi-row launch: `tgrp` output tile pairs per threadgroup, `kph` simdgroups per pair over its k
/// range, `nsplit` threadgroups over the k dim, `chunk` halves per row staged at a time. `nax` launches
/// `ROWS_NAX_SOURCE`, whose threadgroup is always tgrp 4, kph 1; its `unr` (kph 1 only), `r32` and `sw` (n96: the four
/// words a lane's windows span, loaded once) change how the fragments are read, never a byte.
pub const RowsGeom = struct { tgrp: c_int, kph: c_int, nsplit: c_int, chunk: c_int, nax: bool = false, unr: c_int = 1, r32: bool = false, sw: bool = false };

/// What a launch reads in place of finished input rows: the partials of the launch before it, `xsplit` k-split parts a
/// row of `xodim` columns a bank, finished to `xdt`. `xsplit` 0: plain rows.
pub const Pending = struct { xsplit: c_int = 0, xodim: c_int = 0, xdt: mlx.mlx_dtype = .float32 };

const RowsKey = struct { in_dim: c_int, out_dim: c_int, rows: c_int, groups: c_int, n: u32, out_dt: mlx.mlx_dtype, bounds: [3]c_int, geom: RowsGeom, pend: Pending = .{} };

const RowsFinishKey = struct { out_dim: c_int, rows: c_int, groups: c_int, nsplit: c_int, out_dt: mlx.mlx_dtype, bounds: [3]c_int };

/// DeepSeek's trunk alone holds a dozen shapes at each of a decode and a verify width, so a smaller cache thrashes.
var rows_cfgs: kernel_src.CfgCache(RowsKey, 64) = .{};

var rows_finish_cfgs: kernel_src.CfgCache(RowsFinishKey, 64) = .{};

var rows_kernel: kernel_src.KernelSlots = kernel_src.no_kernels;

var rows_nax_kernel: kernel_src.KernelSlots = kernel_src.no_kernels;

var rows_pend_kernel: kernel_src.KernelSlots = kernel_src.no_kernels;

var rows_nax_pend_kernel: kernel_src.KernelSlots = kernel_src.no_kernels;

var rows_finish_kernel: ?mlx.mlx_fast_metal_kernel = null;

const RowsFinishRmsKey = struct { out_dim: c_int, rows: c_int, nsplit: c_int, out_dt: mlx.mlx_dtype, bounds: [3]c_int, norm: c_int };

/// A stacked shape at each of its decode widths.
var rows_finish_rms_cfgs: kernel_src.CfgCache(RowsFinishRmsKey, 32) = .{};

var rows_finish_rms_kernel: ?mlx.mlx_fast_metal_kernel = null;

const RowsFinishExpandKey = struct { out_dim: c_int, rows: c_int, nsplit: c_int, out_dt: mlx.mlx_dtype, hc: c_int };

var rows_finish_expand_cfgs: kernel_src.CfgCache(RowsFinishExpandKey, 16) = .{};

var rows_finish_expand_kernel: ?mlx.mlx_fast_metal_kernel = null;

var rows_finish_row_major_cfgs: kernel_src.CfgCache(RowsFinishKey, 16) = .{};

var rows_finish_row_major_kernel: ?mlx.mlx_fast_metal_kernel = null;

pub const ROWS_INS = [_][*:0]const u8{ "x", "trellis", "suh", "svh" };

pub const ROWS_NAX_FRAGS: [:0]const u8 = kernels.GEMM_NAX_INCLUDES ++ kernels.GEMM_NAX_FRAGS ++ ROWS_HELPERS ++ W32_FRAGS;

/// `nax_wfrag_k` on two 32-bit funnels (`nax_win2`) where a quad's two early codewords lie in the 32 bits ending 16
/// before its last bit and its two late ones in the last 32 (n62 to n102): the same codewords, no 64-bit shift. n64 keeps
/// `nax_wfrag`, whose two-word read measured faster there.
const W32_FRAGS: [:0]const u8 =
    \\static inline constexpr bool nax_w32_fits(uint N) {
    \\  for (uint b = 0u; b < 2u; b++) {
    \\    const uint last = nax_end(N, b, 5u);
    \\    if (last - nax_end(N, b, 1u) < 16u || last - nax_end(N, b, 0u) > 32u || last - nax_end(N, b, 4u) > 16u) return false;
    \\  }
    \\  return true;
    \\}
    \\static inline constexpr bool rows_w32(uint N, uint R32) { return R32 != 0u && N != 64u && nax_w32_fits(N); }
    \\static inline uint rows_funnel(uint late, uint early, uint s) { return uint(((ulong(early) << 32u) | ulong(late)) >> s); }
    \\template<uint N, uint R32, uint SW>
    \\static inline nfrag rows_wfrag(const device uint *words, uint lane) {
    \\  if constexpr (rows_w32(N, R32)) {
    \\    constexpr uint W = N / 2u;
    \\    const uint b = (lane >> 3u) & 1u;
    \\    const uint end = 8u * N * (lane >> 4u) + N * (lane & 7u) + (nax_end(N, 1u, 5u) - nax_end(N, 0u, 5u)) * b + nax_end(N, 0u, 5u);
    \\    const uint s0 = nax_sh(N, b, 0u) - 16u;
    \\    const uint s1 = nax_sh(N, b, 1u) - 16u;
    \\    const uint s4 = nax_sh(N, b, 4u);
    \\    uint2 lo, hi;
    \\    if constexpr (SW != 0u && N == 96u) {
    \\      const uint l = (end - 1u) >> 5u;
    \\      const uint s = (0u - end) & 31u;
    \\      const uint sp = (s + 16u) & 31u;
    \\      const uint wm = words[l == 1u ? W - 1u : l - 2u];
    \\      const uint w0 = words[l - 1u];
    \\      const uint w1 = words[l];
    \\      const uint w2 = words[l + 1u];
    \\      lo = uint2(rows_funnel(w1, w0, s), rows_funnel(w0, wm, sp));
    \\      hi = uint2(rows_funnel(w2, w1, sp), rows_funnel(w2, w1, s));
    \\    } else {
    \\      lo = nax_win2(words, end, W);
    \\      hi = nax_win2(words, end + W, W);
    \\    }
    \\    const half2 p0 = exl3_pairh(uint2((lo.y >> s0) & 0xffffu, (lo.y >> s1) & 0xffffu));
    \\    const half2 p1 = exl3_pairh(uint2((lo.x >> s4) & 0xffffu, lo.x & 0xffffu));
    \\    const half2 p2 = exl3_pairh(uint2((hi.y >> s0) & 0xffffu, (hi.y >> s1) & 0xffffu));
    \\    const half2 p3 = exl3_pairh(uint2((hi.x >> s4) & 0xffffu, hi.x & 0xffffu));
    \\    return nfrag(p0.x, p0.y, p2.x, p2.y, p1.x, p1.y, p3.x, p3.y);
    \\  } else {
    \\    return nax_wfrag_k<N>(words, lane);
    \\  }
    \\}
    \\
;

const ROWS_PEND_INS = ROWS_INS ++ [_][*:0]const u8{"xsvh"};

const ROWS_PEND_SOURCE: [:0]const u8 = kernels.metalReplaceFound(ROWS_SOURCE, ROWS_X_PLAIN, ROWS_X_PENDING);

const ROWS_NAX_PEND_SOURCE: [:0]const u8 = kernels.metalReplaceFound(ROWS_NAX_SOURCE, ROWS_X_PLAIN, ROWS_X_PENDING);

/// `rowsKernel` for a launch whose input is another launch's partials (`Pending`).
pub fn rowsPendKernel(g: RowsGeom) !mlx.mlx_fast_metal_kernel {
    if (g.nax) return kernel_src.codebookKernelWith(&rows_nax_pend_kernel, "sushi_exl3_rows_nax_pend", &ROWS_PEND_INS, &.{"y"}, ROWS_NAX_PEND_SOURCE, ROWS_NAX_FRAGS);
    return kernel_src.codebookKernelWith(&rows_pend_kernel, "sushi_exl3_rows_pend", &ROWS_PEND_INS, &.{"y"}, ROWS_PEND_SOURCE, ROWS_HELPERS);
}

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
pub fn rowsGeometry(in_dim: c_int, out_dim: c_int, rows: c_int, groups: c_int, mode: Mode) RowsGeom {
    const nax = if (mode.exact) !isLmHead(in_dim, out_dim, groups) else rows >= 2;
    return rowsGeometryFor(in_dim, out_dim, rows, groups, nax and kernels.gemmNaxOn());
}

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

/// The NAX arm walks two k-tiles' fragments ahead of their MMAs at one simdgroup per pair, on 32-bit funnels: a one-row
/// trunk launch is bound by the fragment reads and decode, not the MMA.
fn rowsGeometryFor(in_dim: c_int, out_dim: c_int, rows: c_int, groups: c_int, nax: bool) RowsGeom {
    var g = rowsLaunch(in_dim, out_dim, rows, groups, nax);
    if (g.nax) {
        g.r32 = true;
        g.sw = true;
        if (g.kph == 1) g.unr = 2;
    }
    return g;
}

fn rowsLaunch(in_dim: c_int, out_dim: c_int, rows: c_int, groups: c_int, nax: bool) RowsGeom {
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

/// `y[r] = svh * H128(W . H128(suh * x[r]))` for every row of `x` (`[rows, in]` f16, bf16 or f32, rows 1 to `rowsMax(mode)`), over one
/// trellis `[in/16, out/16, n]` (or a one-matrix bank), its `suh [in]` and `svh [out]`: `projectIndexed`'s
/// numbers, each weight decoded once for all rows. Returns `[rows, out]` in x's dtype, rounded once from the f32 sum. A bank `[G, in/16, out/16, n]` takes
/// `x` `[G * rows, in]`, group g's rows contiguous, `suh [G, in]` and `svh [G, out]`, and returns `[G * rows, out]`.
pub fn projectRows(s: mlx.mlx_stream, x: mlx.mlx_array, trellis: mlx.mlx_array, suh: mlx.mlx_array, svh: mlx.mlx_array, mode: Mode) !mlx.mlx_array {
    const xs = mlx.getShape(x);
    const ts = mlx.getShape(trellis);
    if (xs.len != 2 or ts.len < 3) return error.BadExl3Shape;
    const groups: c_int = if (ts.len == 4) ts[0] else 1;
    const rows = @divExact(xs[0], groups);
    const out = ts[ts.len - 2] * 16;
    const g = rowsGeometry(xs[1], out, rows, groups, mode);
    if (g.nax or groups != 1 or rows <= PROJECT_ROWS_MAX or rows > rowsMax(mode)) return projectRowsWith(s, x, trellis, suh, svh, g, try rowsKernel(g), null);
    // A SIMD launch holds `PROJECT_ROWS_MAX` rows and sums a row the same at every count: a wider call goes in halves.
    const half = @divFloor(rows + 1, 2);
    var parts: [2]mlx.mlx_array = @splat(.{});
    defer for (parts) |part| {
        _ = mlx.mlx_array_free(part);
    };
    for (&parts, [_][2]c_int{ .{ 0, half }, .{ half, rows } }) |*part, span| {
        var xp: mlx.mlx_array = .{};
        defer _ = mlx.mlx_array_free(xp);
        try mlx.check(mlx.mlx_slice(&xp, x, &[_]c_int{ span[0], 0 }, 2, &[_]c_int{ span[1], xs[1] }, 2, &[_]c_int{ 1, 1 }, 2, s));
        const gp = rowsGeometry(xs[1], out, span[1] - span[0], 1, mode);
        part.* = try projectRowsWith(s, xp, trellis, suh, svh, gp, try rowsKernel(gp), null);
    }
    const joined = mlx.mlx_vector_array_new_data(&parts, parts.len);
    defer _ = mlx.mlx_vector_array_free(joined);
    var y: mlx.mlx_array = .{};
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_concatenate_axis(&y, joined, 0, s));
    return y;
}

/// `projectRows` over linears that share their input, stacked along the output (one trellis `[1, in/16, out/16, n]`,
/// `suh [S, in]` one row per linear, `svh [out]`): linear s owns the columns from `bounds[s - 1]` (0 for s = 0) to
/// `bounds[s]` (`out` past the last), every bound a multiple of 128. Returns `[rows * out]`, each linear's
/// `[rows, width]` contiguous at `start * rows`.
pub fn projectRowsSegmented(s: mlx.mlx_stream, x: mlx.mlx_array, trellis: mlx.mlx_array, suh: mlx.mlx_array, svh: mlx.mlx_array, bounds: [3]c_int, mode: Mode) !mlx.mlx_array {
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
    const g = rowsGeometry(xs[1], out, xs[0], 1, mode);
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
    if (rows < 1 or rows > (if (g.nax) NAX_ROWS_MAX else PROJECT_ROWS_MAX) or @rem(in_dim, 128) != 0 or ts[ts.len - 3] * 16 != in_dim or @rem(out_tiles, 8) != 0) return error.BadExl3Shape;
    switch (mlx.mlx_array_dtype(x)) {
        .float16, .bfloat16, .float32 => {},
        else => return error.BadExl3Shape,
    }
    if (mlx.mlx_array_dtype(trellis) != .uint16) return error.BadExl3Shape;
    const rate = try kernels.packedRate(ts[ts.len - 1]);
    return launchRows(s, x, trellis, suh, svh, .{ .in_dim = in_dim, .out_dim = out_tiles * 16, .rows = rows, .groups = groups, .n = rate.n, .out_dt = mlx.mlx_array_dtype(x) }, g, kernel, seg_bounds, null);
}

/// What one multi-row launch reads off its operands: their shapes, the rate and `x`'s dtype.
pub const RowsLaunch = struct { in_dim: c_int, out_dim: c_int, rows: c_int, groups: c_int, n: u32, out_dt: mlx.mlx_dtype };

/// The configs of a launch, kept for good so a call looks neither up: the kernel's and, when its k split needs a
/// finish, the finish kernel's.
pub const PinnedCfg = struct { cfg: mlx.mlx_fast_metal_kernel_config, finish: ?mlx.mlx_fast_metal_kernel_config };

var pinned: std.ArrayList(struct { key: RowsKey, cfgs: PinnedCfg }) = .empty;

/// `l`'s configs under `g`, built on first ask and never freed: one set per distinct launch, which the model's shapes
/// bound. For a load to call, not a forward.
pub fn pinCfg(l: RowsLaunch, g: RowsGeom, seg_bounds: ?[3]c_int) !PinnedCfg {
    const bounds = seg_bounds orelse @as([3]c_int, @splat(l.out_dim));
    const key = RowsKey{ .in_dim = l.in_dim, .out_dim = l.out_dim, .rows = l.rows, .groups = l.groups, .n = l.n, .out_dt = l.out_dt, .bounds = bounds, .geom = g };
    for (pinned.items) |p| if (std.meta.eql(p.key, key)) return p.cfgs;
    const cfg = try buildRowsCfg(l, g, bounds, .{});
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const finish: ?mlx.mlx_fast_metal_kernel_config = if (rowsFolds(g)) null else try buildFinishCfg(l, g.nsplit, bounds);
    try pinned.append(std.heap.page_allocator, .{ .key = key, .cfgs = .{ .cfg = cfg, .finish = finish } });
    return pinned.items[pinned.items.len - 1].cfgs;
}

fn buildRowsCfg(l: RowsLaunch, g: RowsGeom, bounds: [3]c_int, pend: Pending) !mlx.mlx_fast_metal_kernel_config {
    const fold = rowsFolds(g);
    const c = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    const sgs = g.tgrp * g.kph;
    const out_shape = [_]c_int{ l.groups * if (fold) l.rows else l.rows * g.nsplit, l.out_dim };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &out_shape, 2, if (fold) l.out_dt else .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "OutT", l.out_dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, @divExact(@divExact(l.out_dim, 16), 2 * g.tgrp) * sgs * 32, g.nsplit, l.groups));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, sgs * 32, 1, 1));
    inline for (.{ "IDIM", "ODIM", "ROWS", "GROUPS", "TGRP", "KPH", "NSPLIT", "CHUNK", "FOLD" }, .{ l.in_dim, l.out_dim, l.rows, l.groups, g.tgrp, g.kph, g.nsplit, g.chunk, @as(c_int, @intFromBool(fold)) }) |name, v|
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, name, v));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "NHW", @intCast(l.n)));
    if (g.nax) {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "UNR", g.unr));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "R32", @intFromBool(g.r32)));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "SW", @intFromBool(g.sw)));
    }
    inline for (.{ "SB1", "SB2", "SB3" }, 0..) |name, i| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, name, bounds[i]));
    if (pend.xsplit > 0) {
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "XSPLIT", pend.xsplit));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "XODIM", pend.xodim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "XT", pend.xdt));
    }
    return c;
}

/// `projectRowsWith` for operands whose shapes the caller holds already (`l`), with configs it may have pinned.
pub fn launchRows(s: mlx.mlx_stream, x: mlx.mlx_array, trellis: mlx.mlx_array, suh: mlx.mlx_array, svh: mlx.mlx_array, l: RowsLaunch, g: RowsGeom, kernel: mlx.mlx_fast_metal_kernel, seg_bounds: ?[3]c_int, pin: ?PinnedCfg) !mlx.mlx_array {
    const fold = rowsFolds(g);
    const bounds = seg_bounds orelse @as([3]c_int, @splat(l.out_dim));
    const key = RowsKey{ .in_dim = l.in_dim, .out_dim = l.out_dim, .rows = l.rows, .groups = l.groups, .n = l.n, .out_dt = l.out_dt, .bounds = bounds, .geom = g };
    const cfg = if (pin) |p| p.cfg else rows_cfgs.get(key) orelse blk: {
        const c = try buildRowsCfg(l, g, bounds, .{});
        rows_cfgs.put(key, c);
        break :blk c;
    };
    const ov = try kernels.applyOuts(s, kernel, &.{ x, trellis, suh, svh }, cfg, 1);
    defer _ = mlx.mlx_vector_array_free(ov);
    // One owner: the fold hands `y` out and leaves the defer an empty handle.
    var y = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, ov, 0));
    if (fold) {
        const whole = y;
        y = mlx.mlx_array_new();
        return whole;
    }
    return rowsFinish(s, y, svh, l, g.nsplit, bounds, if (pin) |p| p.finish else null);
}

/// The first columns of a stacked launch's output to leave rms-normed: `w` `[cols]` in the output's dtype, `eps` an
/// f32 array whose first value is the norm's epsilon.
pub const Norm = struct { w: mlx.mlx_array, eps: mlx.mlx_array, cols: c_int };

/// Whether `launchRowsNormed` serves `l` under `g` with `bounds` and `norm`: a one-bank launch with a finish, its row
/// one threadgroup wide, the normed columns its first segment, in `w`'s dtype.
pub fn normedFits(l: RowsLaunch, g: RowsGeom, bounds: [3]c_int, norm: Norm) bool {
    return l.groups == 1 and !rowsFolds(g) and l.out_dim <= 4096 and norm.cols >= 128 and @rem(norm.cols, 128) == 0 and norm.cols <= bounds[0] and
        mlx.mlx_array_dtype(norm.w) == l.out_dt and mlx.mlx_array_size(norm.w) == @as(usize, @intCast(norm.cols)) and norm.cols == @min(bounds[0], l.out_dim);
}

/// `launchRows` of a stacked launch with `mlx_fast_rms_norm(first segment, norm.w, eps)` in place of the first
/// segment: the bytes of the launch, its finish and the norm, in one dispatch fewer.
pub fn launchRowsNormed(s: mlx.mlx_stream, x: mlx.mlx_array, trellis: mlx.mlx_array, suh: mlx.mlx_array, svh: mlx.mlx_array, l: RowsLaunch, g: RowsGeom, kernel: mlx.mlx_fast_metal_kernel, bounds: [3]c_int, pin: ?PinnedCfg, norm: Norm) !mlx.mlx_array {
    if (!normedFits(l, g, bounds, norm)) return error.BadExl3Shape;
    const key = RowsKey{ .in_dim = l.in_dim, .out_dim = l.out_dim, .rows = l.rows, .groups = 1, .n = l.n, .out_dt = l.out_dt, .bounds = bounds, .geom = g };
    const cfg = if (pin) |p| p.cfg else rows_cfgs.get(key) orelse blk: {
        const c = try buildRowsCfg(l, g, bounds, .{});
        rows_cfgs.put(key, c);
        break :blk c;
    };
    const pv = try kernels.applyOuts(s, kernel, &.{ x, trellis, suh, svh }, cfg, 1);
    defer _ = mlx.mlx_vector_array_free(pv);
    var part = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(part);
    try mlx.check(mlx.mlx_vector_array_get(&part, pv, 0));
    const fkey = RowsFinishRmsKey{ .out_dim = l.out_dim, .rows = l.rows, .nsplit = g.nsplit, .out_dt = l.out_dt, .bounds = bounds, .norm = norm.cols };
    const fcfg = rows_finish_rms_cfgs.get(fkey) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const threads = @divExact(l.out_dim, 4);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &.{ l.rows, l.out_dim }, 2, l.out_dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "OutT", l.out_dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, threads, l.rows, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, threads, 1, 1));
        inline for (.{ "ODIM", "ROWS", "NSPLIT", "NORM" }, .{ l.out_dim, l.rows, g.nsplit, norm.cols }) |name, v| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, name, v));
        inline for (.{ "SB1", "SB2", "SB3" }, 0..) |name, i| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, name, bounds[i]));
        rows_finish_rms_cfgs.put(fkey, c);
        break :blk c;
    };
    const fk = try kernels.getNamedKernel(&rows_finish_rms_kernel, "sushi_exl3_rows_finish_rms", &.{ "part", "svh", "w", "eps" }, &.{"y"}, ROWS_FINISH_RMS_SOURCE, ROWS_HELPERS);
    const ov = try kernels.applyOuts(s, fk, &.{ part, svh, norm.w, norm.eps }, fcfg, 1);
    defer _ = mlx.mlx_vector_array_free(ov);
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, ov, 0));
    return y;
}

/// `launchRows` without its finish, for a launch that has one (`g` does not fold): the k-split partials
/// `[groups * rows * nsplit, out]` in f32, which `launchRowsFrom` reads.
pub fn launchRowsParts(s: mlx.mlx_stream, x: mlx.mlx_array, trellis: mlx.mlx_array, suh: mlx.mlx_array, svh: mlx.mlx_array, l: RowsLaunch, g: RowsGeom) !mlx.mlx_array {
    if (rowsFolds(g)) return error.BadExl3Shape;
    const bounds: [3]c_int = @splat(l.out_dim);
    const key = RowsKey{ .in_dim = l.in_dim, .out_dim = l.out_dim, .rows = l.rows, .groups = l.groups, .n = l.n, .out_dt = l.out_dt, .bounds = bounds, .geom = g };
    const cfg = rows_cfgs.get(key) orelse blk: {
        const c = try buildRowsCfg(l, g, bounds, .{});
        rows_cfgs.put(key, c);
        break :blk c;
    };
    const ov = try kernels.applyOuts(s, try rowsKernel(g), &.{ x, trellis, suh, svh }, cfg, 1);
    defer _ = mlx.mlx_vector_array_free(ov);
    var part = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(part);
    try mlx.check(mlx.mlx_vector_array_get(&part, ov, 0));
    return part;
}

/// `launchRows` whose input rows are `part`, the partials `launchRowsParts` made under `pend` (`xsvh` that launch's
/// svh): input column `g * pend.xodim + c` of row r is that launch's finished `[g, r, c]`. The bytes of the two launches
/// with a finish between them, one dispatch fewer. With `expand` (fitted to `l` and `g`) the finish expands the rows into it.
pub fn launchRowsFrom(s: mlx.mlx_stream, part: mlx.mlx_array, xsvh: mlx.mlx_array, pend: Pending, trellis: mlx.mlx_array, suh: mlx.mlx_array, svh: mlx.mlx_array, l: RowsLaunch, g: RowsGeom, expand: ?Fitted) !mlx.mlx_array {
    if (pend.xsplit < 1 or l.groups != 1 or @rem(pend.xodim, 128) != 0 or @rem(l.in_dim, pend.xodim) != 0) return error.BadExl3Shape;
    const fold = rowsFolds(g);
    const bounds: [3]c_int = @splat(l.out_dim);
    const key = RowsKey{ .in_dim = l.in_dim, .out_dim = l.out_dim, .rows = l.rows, .groups = 1, .n = l.n, .out_dt = l.out_dt, .bounds = bounds, .geom = g, .pend = pend };
    const cfg = rows_cfgs.get(key) orelse blk: {
        const c = try buildRowsCfg(l, g, bounds, pend);
        rows_cfgs.put(key, c);
        break :blk c;
    };
    const ov = try kernels.applyOuts(s, try rowsPendKernel(g), &.{ part, trellis, suh, svh, xsvh }, cfg, 1);
    defer _ = mlx.mlx_vector_array_free(ov);
    var y = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, ov, 0));
    if (fold) {
        const whole = y;
        y = mlx.mlx_array_new();
        return whole;
    }
    if (expand) |e| return rowsFinishExpand(s, y, svh, l, g.nsplit, e);
    return rowsFinish(s, y, svh, l, g.nsplit, bounds, null);
}

/// The streams a one-bank launch's row is expanded into by its finish: `x` `[rows, hc, out]` in the output's dtype,
/// `post` `[rows, hc]` and `comb` `[rows, hc, hc]` in f32. The launch then gives `[rows, hc, out]`.
pub const Expand = struct { x: mlx.mlx_array, post: mlx.mlx_array, comb: mlx.mlx_array };

/// Most streams a finish expands into.
pub const EXPAND_MAX_HC: c_int = 8;

/// The partials `part` of a bank launch that has a finish (`launchRowsParts`), finished with the banks side by side in
/// each row: `[rows, groups * out]`, the transpose of the bank-major rows without its copy.
pub fn rowsFinishRowMajor(s: mlx.mlx_stream, part: mlx.mlx_array, svh: mlx.mlx_array, l: RowsLaunch, nsplit: c_int) !mlx.mlx_array {
    const bounds: [3]c_int = @splat(l.out_dim);
    const key = RowsFinishKey{ .out_dim = l.out_dim, .rows = l.rows, .groups = l.groups, .nsplit = nsplit, .out_dt = l.out_dt, .bounds = bounds };
    const cfg = rows_finish_row_major_cfgs.get(key) orelse blk: {
        const c = try buildFinishCfg(l, nsplit, bounds);
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "GRP", l.groups));
        rows_finish_row_major_cfgs.put(key, c);
        break :blk c;
    };
    const kernel = try kernels.getNamedKernel(&rows_finish_row_major_kernel, "sushi_exl3_rows_finish_row_major", &.{ "part", "svh" }, &.{"y"}, ROWS_FINISH_ROW_MAJOR_SOURCE, ROWS_HELPERS);
    const ov = try kernels.applyOuts(s, kernel, &.{ part, svh }, cfg, 1);
    defer _ = mlx.mlx_vector_array_free(ov);
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_vector_array_get(&flat, ov, 0));
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_reshape(&y, flat, &[_]c_int{ l.rows, l.groups * l.out_dim }, 2, s));
    return y;
}

/// An expand a launch's finish serves; only `fitExpand` makes one.
pub const Fitted = struct { e: Expand, hc: c_int };

/// `e` as the finish of `l` under `g` expands into it, or null where it cannot: a one-bank launch with a finish, in
/// the streams' dtype and shape.
pub fn fitExpand(l: RowsLaunch, g: RowsGeom, e: Expand) ?Fitted {
    const xs = mlx.getShape(e.x);
    if (l.groups != 1 or rowsFolds(g) or xs.len != 3 or xs[0] != l.rows or xs[1] < 1 or xs[1] > EXPAND_MAX_HC or xs[2] != l.out_dim or mlx.mlx_array_dtype(e.x) != l.out_dt) return null;
    return .{ .e = e, .hc = xs[1] };
}

/// The partials `part` of the launch `f` was fitted to, finished into its streams.
pub fn rowsFinishExpand(s: mlx.mlx_stream, part: mlx.mlx_array, svh: mlx.mlx_array, l: RowsLaunch, nsplit: c_int, f: Fitted) !mlx.mlx_array {
    const hc = f.hc;
    const e = f.e;
    const key = RowsFinishExpandKey{ .out_dim = l.out_dim, .rows = l.rows, .nsplit = nsplit, .out_dt = l.out_dt, .hc = hc };
    const cfg = rows_finish_expand_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &.{ l.rows, hc, l.out_dim }, 3, l.out_dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "OutT", l.out_dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, @divExact(l.out_dim, 128) * 32, l.rows, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ODIM", l.out_dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "NSPLIT", nsplit));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "HC", hc));
        rows_finish_expand_cfgs.put(key, c);
        break :blk c;
    };
    const kernel = try kernels.getNamedKernel(&rows_finish_expand_kernel, "sushi_exl3_rows_finish_expand", &.{ "part", "svh", "x", "post", "comb" }, &.{"y"}, ROWS_FINISH_EXPAND_SOURCE, ROWS_HELPERS);
    const ov = try kernels.applyOuts(s, kernel, &.{ part, svh, e.x, e.post, e.comb }, cfg, 1);
    defer _ = mlx.mlx_vector_array_free(ov);
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, ov, 0));
    return y;
}

/// Whether `g`'s launch leaves partials a finish sums (`launchRowsParts` serves it).
pub fn hasFinish(g: RowsGeom) bool {
    return !rowsFolds(g);
}

fn buildFinishCfg(l: RowsLaunch, nsplit: c_int, bounds: [3]c_int) !mlx.mlx_fast_metal_kernel_config {
    const c = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &.{ l.groups * l.rows, l.out_dim }, 2, l.out_dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "OutT", l.out_dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, @divExact(l.out_dim, 128) * 32, l.rows, l.groups));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ODIM", l.out_dim));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ROWS", l.rows));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "NSPLIT", nsplit));
    inline for (.{ "SB1", "SB2", "SB3" }, 0..) |name, i| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, name, bounds[i]));
    return c;
}

fn rowsFinish(s: mlx.mlx_stream, part: mlx.mlx_array, svh: mlx.mlx_array, l: RowsLaunch, nsplit: c_int, bounds: [3]c_int, pin: ?mlx.mlx_fast_metal_kernel_config) !mlx.mlx_array {
    const key = RowsFinishKey{ .out_dim = l.out_dim, .rows = l.rows, .groups = l.groups, .nsplit = nsplit, .out_dt = l.out_dt, .bounds = bounds };
    const cfg = pin orelse rows_finish_cfgs.get(key) orelse blk: {
        const c = try buildFinishCfg(l, nsplit, bounds);
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
        const xh = try alloc.alloc(u16, groups * @as(usize, @intCast(NAX_ROWS_MAX)) * in);
        for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
        return .{ .in = in, .out = out, .groups = groups, .n = n, .tr = tr, .suh = suh, .svh = svh, .xh = xh };
    }

    /// Rows `first..first + rows` of every group through `kernel` under `geom`.
    fn run(self: RowsRig, alloc: std.mem.Allocator, first: usize, rows: usize, geom: RowsGeom, kernel: mlx.mlx_fast_metal_kernel) ![]const u16 {
        const s = mlx.gpuStream();
        const pool: usize = @intCast(NAX_ROWS_MAX);
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

    /// Rows `first..first + rows` of a one-bank rig through `projectRows` in `mode`, on the stream `s`.
    fn project(self: RowsRig, alloc: std.mem.Allocator, s: mlx.mlx_stream, first: usize, rows: usize, mode: Mode) ![]const u16 {
        var held: [4]mlx.mlx_array = undefined;
        {
            const host = mlx.engine.enter(s);
            defer host.end();
            held = .{
                mlx.mlx_array_new_data(self.xh[first * self.in ..].ptr, &[_]c_int{ @intCast(rows), @intCast(self.in) }, 2, .float16),
                mlx.mlx_array_new_data(self.tr.ptr, &[_]c_int{ 1, @intCast(self.in / 16), @intCast(self.out / 16), @intCast(self.n) }, 4, .uint16),
                mlx.mlx_array_new_data(self.suh.ptr, &[_]c_int{ 1, @intCast(self.in) }, 2, .float16),
                mlx.mlx_array_new_data(self.svh.ptr, &[_]c_int{ 1, @intCast(self.out) }, 2, .float16),
            };
        }
        defer for (held) |h| {
            _ = mlx.mlx_array_free(h);
        };
        const y = try projectRows(s, held[0], held[1], held[2], held[3], mode);
        defer _ = mlx.mlx_array_free(y);
        var yc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(yc);
        const got = try kernels.evalF16(s, y, &yc);
        return alloc.dupe(u16, @as([*]const u16, @ptrCast(got))[0 .. rows * self.out]);
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

// In exact mode the lm_head keeps the SIMD arm, whose launch holds eight rows, while a wide verify brings up to
// sixteen: its head rows go eight or fewer at a time, each still its one-row launch.
test "exl3 multi-row mode: at the lm_head's shape a wide exact verify of 9 to 16 rows gives each row its one-row launch's bits, on MLX's stream and on the engine's" {
    try mlx.requireGpu();
    if (rowsMax(.{ .exact = true, .wide = true }) < NAX_ROWS_MAX) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    defer kernels.setDecodeParams(.mul1);
    const d = SIMD_ROWS_DECODES[1];
    try d.dec.requireTable();
    kernels.setDecodeParams(d.dec);
    const out: usize = 129280;
    const rig = try RowsRig.init(alloc, 0x1ead, 5120, out, 1, d.n);
    try std.testing.expect(!rowsGeometry(5120, 129280, 9, 1, .{ .exact = true, .wide = true }).nax);
    const was = mlx.engine.enabled;
    defer mlx.engine.enabled = was;
    for ([_]bool{ false, true }) |on_engine| {
        mlx.engine.enabled = on_engine;
        const s = if (on_engine) mlx.engine.stream() else mlx.gpuStream();
        const pool: usize = @intCast(NAX_ROWS_MAX);
        const ones = try alloc.alloc(u16, pool * out);
        for (0..pool) |r| @memcpy(ones[r * out ..][0..out], try rig.project(alloc, s, r, 1, .{ .exact = true }));
        for ([_]usize{ 9, 13, 16 }) |rows| {
            const many = try rig.project(alloc, s, 0, rows, .{ .exact = true, .wide = true });
            try std.testing.expectEqualSlices(u16, ones[0 .. rows * out], many);
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
            for ([_]c_int{ 2, 8 }, &served) |rows, *g| g.* = rowsGeometry(in, @intCast(sh[1]), rows, @intCast(sh[2]), .{});
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

// Past eight rows the NAX arm feeds the tile's second eight rows, which were zero; a row's sums never see another row's,
// so a 9- to 16-row verify row keeps the bytes of its one-row decode tick.
test "exl3 NAX multi-row: at 9 to 16 rows every row equals the one-row launch bit for bit" {
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s) or !kernels.gemmNaxOn()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    defer kernels.setDecodeParams(.mul1);
    const exact: Mode = .{ .exact = true };
    const decodes = SIMD_ROWS_DECODES ++ [_]@TypeOf(SIMD_ROWS_DECODES[0]){.{ .dec = exl3.Decode.tabled(.spec1, .w15), .n = 96 }};
    for (decodes) |d| {
        try d.dec.requireTable();
        kernels.setDecodeParams(d.dec);
        for ([_][3]usize{ .{ 1280, 512, 1 }, .{ 1024, 256, 2 }, .{ 131072, 256, 1 }, .{ 1792, 5120, 1 } }) |sh| {
            const rig = try RowsRig.init(alloc, 0x9a16 + sh[0] + d.n, sh[0], sh[1], sh[2], d.n);
            const g1 = rowsGeometry(@intCast(sh[0]), @intCast(sh[1]), 1, @intCast(sh[2]), exact);
            var alone: [@as(usize, @intCast(NAX_ROWS_MAX))][]const u16 = undefined;
            for (&alone, 0..) |*a, r| a.* = try rig.run(alloc, r, 1, g1, try rowsKernel(g1));
            for (9..@as(usize, @intCast(NAX_ROWS_MAX)) + 1) |rows| {
                const g = rowsGeometry(@intCast(sh[0]), @intCast(sh[1]), @intCast(rows), @intCast(sh[2]), exact);
                const many = try rig.run(alloc, 0, rows, g, try rowsKernel(g));
                for (0..sh[2]) |grp| for (0..rows) |r|
                    try std.testing.expectEqualSlices(u16, alone[r][grp * sh[1] ..][0..sh[1]], many[(grp * rows + r) * sh[1] ..][0..sh[1]]);
            }
        }
    }
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
        const y = try projectRows(s, x, tr, suh, svh, .{});
        defer _ = mlx.mlx_array_free(y);
        try t.expectEqual(@as(u32, @intCast(c[2])), kernels.fusedDispatchCount());
        try t.expectEqualSlices(c_int, &.{ 4, c[0] }, mlx.getShape(y));
    }
}

// The served NAX walk reads two k-tiles' fragments ahead of their MMAs, on 32-bit funnels; the MMAs run in the one-tile
// walk's order into the same accumulator, so every byte is the one-tile, 64-bit walk's.
test "exl3 NAX multi-row: the unrolled 32-bit walk writes the one-tile walk's bytes" {
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s) or !kernels.gemmNaxOn()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    defer kernels.setDecodeParams(.mul1);
    for ([_]exl3.Decode{ exl3.Decode.tabled(.spec1, .w15), .{ .codebook = .mcg, .window = .w15 } }) |dec| for ([_]u32{ 48, 64, 96 }) |n| {
        try dec.requireTable();
        kernels.setDecodeParams(dec);
        for ([_][3]usize{ .{ 640, 2560, 1 }, .{ 1280, 1024, 2 }, .{ 1408, 384, 1 } }) |sh| {
            const rig = try RowsRig.init(alloc, 0x6a11 + sh[0] + n, sh[0], sh[1], sh[2], n);
            const in: c_int = @intCast(sh[0]);
            for ([_]c_int{ 1, 2, 5, 11 }) |nsplit| {
                if (@rem(@divExact(in, 128), nsplit) != 0) continue;
                for ([_]c_int{ 1, 3, 8 }) |rows| for ([_]c_int{ rowsChunk(in, rows, nsplit), 128 }) |chunk| {
                    const one = RowsGeom{ .tgrp = 4, .kph = 1, .nsplit = nsplit, .chunk = chunk, .nax = true };
                    const want = try rig.run(alloc, 0, @intCast(rows), one, try rowsKernel(one));
                    for ([_]c_int{ 1, 2, 4 }) |unr| for ([_]bool{ false, true }) |r32| for ([_]bool{ false, true }) |sw| {
                        if (sw and (!r32 or unr == 4 or chunk != rowsChunk(in, rows, nsplit))) continue;
                        var g = one;
                        g.unr = unr;
                        g.r32 = r32;
                        g.sw = sw;
                        try std.testing.expectEqualSlices(u16, want, try rig.run(alloc, 0, @intCast(rows), g, try rowsKernel(g)));
                    };
                };
            }
        }
    };
}

// The 32-bit funnel reader takes the 64-bit reader's codewords at every rate it serves (n62 to n102); another rate keeps
// the 64-bit reader. Sliced k (kph > 1) keeps the one-tile walk and takes the 32-bit reader alone.
test "exl3 NAX multi-row: the 32-bit funnel reader writes the 64-bit reader's bytes at every rate" {
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s) or !kernels.gemmNaxOn()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    defer kernels.setDecodeParams(.mul1);
    for ([_]exl3.Decode{ exl3.Decode.tabled(.spec1, .w15), .{ .codebook = .mcg, .window = .w15 }, .{ .codebook = .pair1, .window = .w16 } }) |dec| {
        try dec.requireTable();
        kernels.setDecodeParams(dec);
        var n: u32 = 56;
        while (n <= 112) : (n += 2) {
            if (dec.codebook.tabledHashMask() and dec.hashMask(.{ .n = n }) == null) continue;
            if (dec.codebook == .pair1 and exl3.pair1Pair(.{ .n = n }) == null) continue;
            const rig = try RowsRig.init(alloc, 0x8c41 + n, 1280, 1024, 1, n);
            for ([_]c_int{ 1, 4 }) |kph| for ([_]c_int{ 1, 3 }) |rows| for ([_]bool{ false, true }) |sw| {
                const one = RowsGeom{ .tgrp = 4, .kph = kph, .nsplit = 2, .chunk = rowsChunk(1280, rows, 2), .nax = true };
                const want = try rig.run(alloc, 0, @intCast(rows), one, try rowsKernel(one));
                var g = one;
                g.r32 = true;
                g.sw = sw;
                try std.testing.expectEqualSlices(u16, want, try rig.run(alloc, 0, @intCast(rows), g, try rowsKernel(g)));
            };
        }
    }
}

test "exl3 multi-row mode: exact puts every width but the lm_head's on the NAX arm, and wide lifts the row ceiling only with exact on NAX" {
    try mlx.requireGpu();
    const nax = kernels.gemmNaxOn();
    try std.testing.expectEqual(PROJECT_ROWS_MAX, rowsMax(.{}));
    try std.testing.expect(!rowsGeometry(2560, 12288, 1, 1, .{}).nax);
    try std.testing.expectEqual(nax, rowsGeometry(2560, 12288, 2, 1, .{}).nax);
    try std.testing.expectEqual(PROJECT_ROWS_MAX, rowsMax(.{ .wide = true }));
    try std.testing.expectEqual(if (nax) NAX_ROWS_MAX else PROJECT_ROWS_MAX, rowsMax(.{ .exact = true, .wide = true }));
    try std.testing.expectEqual(nax, rowsGeometry(2560, 12288, 1, 1, .{ .exact = true }).nax);
    try std.testing.expect(!rowsGeometry(5120, 129280, 1, 1, .{ .exact = true }).nax);
    try std.testing.expectEqual(PROJECT_ROWS_MAX, rowsMax(.{ .exact = true }));
}
