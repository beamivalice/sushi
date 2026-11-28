const device = @import("device.zig");
const exl3 = @import("expert_exl3.zig");
const kernel_src = @import("kernel_source.zig");
const mlx = @import("mlx_host").mlx;
const host_side = @import("host_side.zig");
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
    \\template <typename OutT>
    \\static inline OutT exl3_rope_lane(OutT a, uint ch, uint lane, float pf, const device float *freqs, int HD, int RD) {
    \\  #pragma clang fp contract(off)
    \\  const uint HEAD = uint(HD - RD);
    \\  if (ch < HEAD) return a;
    \\  const float xv = float(a);
    \\  const float partner = metal::simd_shuffle_xor(xv, ushort(1));
    \\  const float x1 = (lane & 1u) == 0u ? xv : partner;
    \\  const float x2 = (lane & 1u) == 0u ? partner : xv;
    \\  const float theta = pf * freqs[(ch - HEAD) >> 1u];
    \\  const float cs = metal::precise::cos(theta), sn = metal::precise::sin(theta);
    \\  return OutT((lane & 1u) == 0u ? x1 * cs - x2 * sn : x1 * sn + x2 * cs);
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

/// `ROWS_X_PLAIN` over f32 rows, each element rounded to bf16 as it is read (`Mode.thru_bf16`).
const ROWS_X_THRU: []const u8 =
    \\    const auto xr = x + (size_t)r * uint(IDIM) + k;
    \\    const float4 v = exl3_h128(float4(
    \\      float(bfloat16_t(xr[lane])) * float(suh[k + lane]),
    \\      float(bfloat16_t(xr[lane + 32u])) * float(suh[k + lane + 32u]),
    \\      float(bfloat16_t(xr[lane + 64u])) * float(suh[k + lane + 64u]),
    \\      float(bfloat16_t(xr[lane + 96u])) * float(suh[k + lane + 96u])), lane);
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

/// `ROWS_FOLD` with in-register tail RoPE on each `HD`-channel head's last `RD` channels (zero DRAM roundtrip).
const ROWS_FOLD_ROPE: []const u8 =
    \\  for (uint r = sg; r < uint(ROWS); r += SGS) {
    \\    float4 v = 0.0f;
    \\    for (uint k = 0u; k < KPARTS; k++) {
    \\      const threadgroup float *pr = tgm + (r * KPARTS + k) * 128u + lane;
    \\      v += float4(pr[0], pr[32], pr[64], pr[96]);
    \\    }
    \\    v = exl3_h128(v, lane);
    \\    const uint ob = ot0 * TILE + lane;
    \\    const uint ch = (ot0 * TILE) % uint(HD) + lane;
    \\    const float pf = float(pos[r]);
    \\    const auto yr = y + exl3_seg_at(r, ob, uint(ROWS), uint(ODIM), uint(SB1), uint(SB2), uint(SB3));
    \\    yr[0] = exl3_rope_lane<OutT>(OutT(v.x * psc * float(svh[ob])), ch, lane, pf, freqs, HD, RD);
    \\    yr[32] = exl3_rope_lane<OutT>(OutT(v.y * psc * float(svh[ob + 32u])), ch + 32u, lane, pf, freqs, HD, RD);
    \\    yr[64] = exl3_rope_lane<OutT>(OutT(v.z * psc * float(svh[ob + 64u])), ch + 64u, lane, pf, freqs, HD, RD);
    \\    yr[96] = exl3_rope_lane<OutT>(OutT(v.w * psc * float(svh[ob + 96u])), ch + 96u, lane, pf, freqs, HD, RD);
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
    \\    #pragma clang loop unroll(full)
    \\    for (uint u = 0u; u < uint(UNR); u++) {
    \\      const device uint *wp = wcol + (size_t)(tk + u * uint(KPH) - tk0) * OT * PACKED_W;
    \\      wf[u][0] = rows_wfrag<N, R32, SW>(wp, lane);
    \\      wf[u][1] = rows_wfrag<N, R32, SW>(wp + PACKED_W, lane);
    \\    }
    \\    #pragma clang loop unroll(full)
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
    \\      else exl3_store(y + (size_t)(row * uint(NSPLIT) + split) * uint(ODIM) + ot0 * TILE + col, exl3_out(float(destination[ct * 8u + 4u * h + c])));
    \\    }
    \\  }
    \\}
    \\if constexpr (FOLD) {
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  const float psc = exl3_out(0.08838834764831845f);
    \\
++ ROWS_FOLD ++
    \\} else if constexpr (KPARTS > 1u) {
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  for (uint i = lid; i < uint(ROWS) * 128u; i += SGS * 32u) {
    \\    const uint r = i / 128u;
    \\    const uint c = i - r * 128u;
    \\    float sum = 0.0f;
    \\    for (uint k = 0u; k < KPARTS; k++) sum += tgm[(r * KPARTS + k) * 128u + c];
    \\    exl3_store(y + (size_t)(r * uint(NSPLIT) + split) * uint(ODIM) + ot0 * TILE + c, exl3_out(sum));
    \\  }
    \\}
;

const ROWS_PREPPED_HEAD: []const u8 =
    \\constexpr uint TILE = 16u;
    \\constexpr uint N = uint(NHW);
    \\constexpr uint PACKED_W = N / 2u;
    \\constexpr uint OT = uint(ODIM) / TILE;
    \\constexpr uint KT = uint(IDIM) / uint(NSPLIT) / TILE;
    \\threadgroup float tgm[(FOLD || KPARTS > 1u) ? PART_F : 1u];
    \\const uint ot0 = uint(threadgroup_position_in_grid.x) * TGT;
    \\const uint split = uint(threadgroup_position_in_grid.y);
    \\const uint sg = uint(simdgroup_index_in_threadgroup);
    \\const uint lane = uint(thread_index_in_simdgroup);
    \\const uint tk0 = split * KT;
    \\
;

/// `ROWS_NAX_SOURCE` reading already Hadamard-prepared `[ROWS, IDIM]` f16 rows directly in linear `lane` order
/// (no `suh`, no input H128, no threadgroup prep buffer).
pub const ROWS_NAX_PREPPED_SOURCE: [:0]const u8 =
    \\static_assert(UNR == 1 || KPH == 1, "an unrolled walk steps whole chunks of one slice");
    \\constexpr uint SGS = 4u * uint(KPH);
    \\constexpr uint KPARTS = uint(KPH);
    \\constexpr uint TGT = 8u;
    \\constexpr uint PART_F = uint(ROWS) * KPARTS * 128u;
    \\
++ ROWS_PREPPED_HEAD ++
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
    \\const device half *pl = (const device half *)x + (size_t)xrow * uint(IDIM) + uint(origin.x);
    \\const device half *pl8 = (const device half *)x + (size_t)xrow8 * uint(IDIM) + uint(origin.x);
    \\for (uint tk = tk0 + kslice; tk < tk0 + KT; tk += uint(UNR) * uint(KPH)) {
    \\  nfrag wf[UNR][2];
    \\  for (uint u = 0u; u < uint(UNR); u++) {
    \\    const device uint *wp = wcol + (size_t)(tk + u * uint(KPH) - tk0) * OT * PACKED_W;
    \\    wf[u][0] = rows_wfrag<N, R32, SW>(wp, lane);
    \\    wf[u][1] = rows_wfrag<N, R32, SW>(wp + PACKED_W, lane);
    \\  }
    \\  for (uint u = 0u; u < uint(UNR); u++) {
    \\    for (short s = 0; s < 8; s++) {
    \\      right[s] = wf[u][0][s];
    \\      right[8 + s] = wf[u][1][s];
    \\    }
    \\    const half4 v = *((const device half4 *)(pl + (tk + u * uint(KPH)) * TILE));
    \\    const half4 v8 = ROWS > 8 ? *((const device half4 *)(pl8 + (tk + u * uint(KPH)) * TILE)) : half4(0.0h);
    \\    for (uint c = 0u; c < 4u; c++) {
    \\      left[c] = v[c];
    \\      left[4 + c] = v8[c];
    \\    }
    \\    op.run(left, right, destination);
    \\  }
    \\}
    \\for (uint h = 0u; h < (ROWS > 8 ? 2u : 1u); h++) {
    \\  const uint row = uint(origin.y) + 8u * h;
    \\  if (row >= uint(ROWS)) continue;
    \\  for (uint ct = 0u; ct < 2u; ct++) {
    \\    for (uint c = 0u; c < 4u; c++) {
    \\      const uint col = col0 * 32u + ct * 16u + uint(origin.x) + c;
    \\      if constexpr (FOLD || KPARTS > 1u) tgm[(row * KPARTS + kslice) * 128u + col] = destination[ct * 8u + 4u * h + c];
    \\      else exl3_store(y + (size_t)(row * uint(NSPLIT) + split) * uint(ODIM) + ot0 * TILE + col, exl3_out(float(destination[ct * 8u + 4u * h + c])));
    \\    }
    \\  }
    \\}
    \\if constexpr (FOLD) {
    \\  const float psc = exl3_out(0.08838834764831845f);
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
    \\    exl3_store(y + (size_t)(r * uint(NSPLIT) + split) * uint(ODIM) + ot0 * TILE + c, exl3_out(sum));
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

pub const ROWS_FINISH_SOURCE: [:0]const u8 =
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

/// `ROWS_FINISH_SOURCE` storing each output as its OutT (bf16) rounding into f32 rows (`Mode.thru_bf16`).
pub const ROWS_FINISH_THRU_SOURCE: [:0]const u8 = kernels.metalReplaceFound(ROWS_FINISH_SOURCE, "device OutT *yr", "device float *yr");

pub const ROWS_FINISH_ROW_MAJOR_SOURCE: [:0]const u8 = kernels.metalReplaceFound(ROWS_FINISH_SOURCE, ROWS_FINISH_AT, ROWS_FINISH_AT_ROW_MAJOR);

/// `ROWS_FINISH_SOURCE` for a one-bank query projection whose `HD`-channel heads leave tail-rotated by `RD`.
pub const ROWS_FINISH_ROPE_SOURCE: [:0]const u8 =
    \\const uint lane = uint(thread_index_in_simdgroup);
    \\const uint tg = uint(threadgroup_position_in_grid.x);
    \\const uint b = tg * 128u + lane;
    \\const uint r = uint(threadgroup_position_in_grid.y);
    \\float4 v = 0.0f;
    \\for (uint s = 0u; s < uint(NSPLIT); s++) {
    \\  const device float *p = part + (size_t)(r * uint(NSPLIT) + s) * uint(ODIM) + b;
    \\  v += float4(p[0], p[32], p[64], p[96]);
    \\}
    \\v = exl3_h128(v, lane);
    \\const float psc = 0.08838834764831845f;
    \\const uint ch = (tg * 128u) % uint(HD) + lane;
    \\const float pf = float(pos[r]);
    \\device OutT *yr = y + (size_t)r * uint(ODIM) + b;
    \\yr[0] = exl3_rope_lane<OutT>(OutT(v.x * psc * float(svh[b])), ch, lane, pf, freqs, HD, RD);
    \\yr[32] = exl3_rope_lane<OutT>(OutT(v.y * psc * float(svh[b + 32u])), ch + 32u, lane, pf, freqs, HD, RD);
    \\yr[64] = exl3_rope_lane<OutT>(OutT(v.z * psc * float(svh[b + 64u])), ch + 64u, lane, pf, freqs, HD, RD);
    \\yr[96] = exl3_rope_lane<OutT>(OutT(v.w * psc * float(svh[b + 96u])), ch + 96u, lane, pf, freqs, HD, RD);
;

/// `ROWS_FINISH_SOURCE` of a one-bank launch whose row `y` leaves as HC streams: stream i is
/// `fma(post[i], y, sum over j of fma(comb[j][i], x[j], acc))` rounded to OutT, `y` rounded to OutT first.
pub const ROWS_FINISH_EXPAND_SOURCE: [:0]const u8 =
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
pub const ROWS_FINISH_RMS_SOURCE: [:0]const u8 =
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

pub const ROWS_RING_HELPERS: [:0]const u8 = ROWS_HELPERS ++
    \\static inline float exl3_round_scale(float v) {
    \\  uint b = as_type<uint>(v);
    \\  int e = int((b >> 23) & 0xFFu) - 127 + ((b & 0x7FFFFFu) != 0u ? 1 : 0);
    \\  return as_type<float>(uint(clamp(e + 127, 1, 254)) << 23);
    \\}
    \\static inline float exl3_round_e4m3(float v) {
    \\  if (v == 0.0f) return 0.0f;
    \\  int e = max(int((as_type<uint>(v) >> 23) & 0xFFu) - 127, -6);
    \\  float step = as_type<float>(uint(e - 3 + 127) << 23);
    \\  return min(rint(v / step) * step, 448.0f);
    \\}
    \\static inline float exl3_qat_scale0(float amax) {
    \\  return exl3_round_scale(max(amax, 1e-4f) * (1.0f / 448.0f));
    \\}
    \\static inline float exl3_qat_round0(float v, float s) {
    \\  const float t = clamp(precise::divide(v, s), -448.0f, 448.0f);
    \\  return copysign(exl3_round_e4m3(fabs(t)), t) * s;
    \\}
;

/// `ROWS_FINISH_RMS_SOURCE` for a 2-segment `[NORM, KV_D]` attention input (`wq_a` + `wkv`) that also applies
/// `kv_norm` + tail RoPE + optional fp8 QAT on the `KV_D` segment in the idle simdgroups and copies unchanged rows
/// of `ring` into `rout` in parallel threadgroups, eliminating the `ropeKvRing` barrier and DRAM roundtrip.
const ROWS_FINISH_RMS_RING_BODY: [:0]const u8 =
    \\threadgroup float fin[NORM];
    \\threadgroup float kfin[KV_D];
    \\threadgroup float sums[32];
    \\threadgroup float ksums[32];
    \\threadgroup float inv_mean[1];
    \\threadgroup float kinv[1];
    \\const uint b = sg * 128u + lane;
    \\const bool normed = sg * 128u < uint(NORM);
    \\constexpr uint Q_SGS = uint(NORM) / 128u;
    \\const uint ksg = sg - Q_SGS;
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
    \\  const uint kb = ksg * 128u + lane;
    \\  kfin[kb] = float(o0);
    \\  kfin[kb + 32u] = float(o1);
    \\  kfin[kb + 64u] = float(o2);
    \\  kfin[kb + 96u] = float(o3);
    \\}
    \\if (sg == 0u) {
    \\  sums[lane] = 0.0f;
    \\  ksums[lane] = 0.0f;
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\float tx[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    \\float acc = 0.0f;
    \\if (normed) {
    \\  for (int i = 0; i < 4; i++) {
    \\    tx[i] = fin[sg * 128u + lane * 4u + uint(i)];
    \\    acc = metal::fma(tx[i], tx[i], acc);
    \\  }
    \\  acc = simd_sum(acc);
    \\  if (lane == 0u) sums[sg] = acc;
    \\} else {
    \\  for (int i = 0; i < 4; i++) {
    \\    tx[i] = kfin[ksg * 128u + lane * 4u + uint(i)];
    \\    acc = metal::fma(tx[i], tx[i], acc);
    \\  }
    \\  acc = simd_sum(acc);
    \\  if (lane == 0u) ksums[ksg] = acc;
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if (sg == 0u) {
    \\  const float s = simd_sum(sums[lane]);
    \\  if (lane == 0u) inv_mean[0] = metal::precise::rsqrt(s / uint(NORM) + eps[0]);
    \\} else if (sg == Q_SGS) {
    \\  const float s = simd_sum(ksums[lane]);
    \\  if (lane == 0u) kinv[0] = metal::precise::rsqrt(s / float(KV_D) + eps[0]);
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if (normed) {
    \\  const uint c0 = sg * 128u + lane * 4u;
    \\  device OutT *qr = qout + (size_t)r * uint(NORM) + c0;
    \\  for (int i = 0; i < 4; i++) qr[i] = w[c0 + uint(i)] * OutT(tx[i] * inv_mean[0]);
    \\} else {
    \\  const uint kt = ksg * 32u + lane;
    \\  const float inv = kinv[0];
    \\  float kv[4];
    \\  for (int i = 0; i < 4; i++) kv[i] = float(kw[kt * 4u + uint(i)] * OutT(tx[i] * inv));
    \\  constexpr uint KHEAD = uint(KV_D - RD);
    \\  for (int i = 0; i < 4; i += 2) {
    \\    const uint idx = kt * 4u + uint(i);
    \\    if (idx >= KHEAD) {
    \\      const float theta = float(pos[r]) * freqs[(idx - KHEAD) >> 1u];
    \\      const float cs = metal::precise::cos(theta), sn = metal::precise::sin(theta);
    \\      const float x1 = kv[i], x2 = kv[i + 1];
    \\      kv[i] = float(OutT(x1 * cs - x2 * sn));
    \\      kv[i + 1] = float(OutT(x1 * sn + x2 * cs));
    \\    }
    \\  }
    \\  if (QAT != 0) {
    \\    float amax = 0.0f;
    \\    for (int i = 0; i < 4; i++) amax = metal::max(amax, metal::abs(kv[i]));
    \\    for (ushort m = 4; m > 0; m >>= 1) amax = metal::max(amax, metal::simd_shuffle_xor(amax, m));
    \\    const float sc = exl3_qat_scale0(amax);
    \\    for (int i = 0; i < 4; i++) kv[i] = exl3_qat_round0(kv[i], sc);
    \\  }
    \\  device OutT *dst = rout + (size_t)(int(pos[r]) % cap) * uint(KV_D) + kt * 4u;
    \\  for (int i = 0; i < 4; i++) dst[i] = OutT(kv[i]);
    \\}
;

pub const ROWS_FINISH_RMS_RING_SOURCE: [:0]const u8 =
    \\#pragma clang fp contract(off)
    \\const uint tgid = uint(threadgroup_position_in_grid.x);
    \\const int cap = ring_shape[0];
    \\if (tgid >= uint(ROWS)) {
    \\  constexpr uint U = uint(KV_D) * sizeof(OutT) / 16u;
    \\  const uint u = (tgid - uint(ROWS)) * uint(threads_per_threadgroup.x) + uint(thread_position_in_threadgroup.x);
    \\  if (u >= uint(cap) * U) return;
    \\  if ((int(u / U) + cap - int(pos[0]) % cap) % cap < int(ROWS)) return;
    \\  ((device metal::uint4 *)rout)[u] = ((const device metal::uint4 *)ring)[u];
    \\  return;
    \\}
    \\const uint lane = uint(thread_index_in_simdgroup);
    \\const uint sg = uint(simdgroup_index_in_threadgroup);
    \\const uint r = tgid;
    \\
++ ROWS_FINISH_RMS_RING_BODY;

/// In-place variant of `ROWS_FINISH_RMS_RING_SOURCE`: writes `ROWS` rows directly into `rout` (the owned KV ring)
/// with no `ring` read and no copy threadgroups.
pub const ROWS_FINISH_RMS_RING_IP_SOURCE: [:0]const u8 =
    \\#pragma clang fp contract(off)
    \\const uint r = uint(threadgroup_position_in_grid.x);
    \\const int cap = int(CAP);
    \\const uint lane = uint(thread_index_in_simdgroup);
    \\const uint sg = uint(simdgroup_index_in_threadgroup);
    \\
++ ROWS_FINISH_RMS_RING_BODY;

pub const ROWS_NAX_ROPE_SOURCE: [:0]const u8 = kernels.metalReplaceFound(ROWS_NAX_SOURCE, ROWS_FOLD, ROWS_FOLD_ROPE);

/// `ROWS_FINISH_RMS_SOURCE` of a launch through bf16 (`Mode.thru_bf16`): each finished output rounded to bf16 and widened,
/// the norm in f32 over `w` widened as stored, the rows f32 (OutT): the through launch, then MLX's f32 rms norm.
pub const ROWS_FINISH_RMS_THRU_SOURCE: [:0]const u8 = blk: {
    var t: []const u8 = ROWS_FINISH_RMS_SOURCE;
    for ([_][]const u8{ "o0", "o1", "o2", "o3" }) |o| t = kernels.metalReplaceFound(t, "const OutT " ++ o ++ " = OutT(", "const float " ++ o ++ " = THRU_BF16(");
    t = kernels.metalReplaceFound(t, "yr[i] = w[c0 + uint(i)] * OutT(tx[i] * inv_mean[0]);", "yr[i] = float(w[c0 + uint(i)]) * (tx[i] * inv_mean[0]);");
    break :blk kernels.metalReplaceFound(t, "#pragma clang fp contract(off)\n", "#pragma clang fp contract(off)\n#define THRU_BF16(v) float(bfloat16_t(v))\n");
};

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
    /// `x` arrives f32 and each element is rounded to bf16 where the launch reads it; each output is stored f32 as its
    /// bf16 rounding: the bytes of casting the rows to bf16, the bf16 launch and casting its rows back, two dispatches fewer.
    thru_bf16: bool = false,
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

const RowsKey = struct { in_dim: c_int, out_dim: c_int, rows: c_int, groups: c_int, n: u32, out_dt: mlx.mlx_dtype, bounds: [3]c_int, geom: RowsGeom, pend: Pending = .{}, thru: bool = false };

const RowsFinishKey = struct { out_dim: c_int, rows: c_int, groups: c_int, nsplit: c_int, out_dt: mlx.mlx_dtype, bounds: [3]c_int, thru: bool = false };

/// DeepSeek's trunk alone holds a dozen shapes at each of a decode and a verify width, so a smaller cache thrashes.
var rows_cfgs: kernel_src.CfgCache(RowsKey, 64) = .{};

var rows_finish_cfgs: kernel_src.CfgCache(RowsFinishKey, 64) = .{};

var rows_kernel: kernel_src.KernelSlots = kernel_src.no_kernels;

var rows_nax_kernel: kernel_src.KernelSlots = kernel_src.no_kernels;

var rows_pend_kernel: kernel_src.KernelSlots = kernel_src.no_kernels;

var rows_nax_pend_kernel: kernel_src.KernelSlots = kernel_src.no_kernels;

var rows_finish_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// Joined rows of a batch (`trunk.joinedRows`): a NAX launch of any row total runs one of two pipelines, compiled at
/// eight rows or the cap's and their chunk, its kernel reading the row count off its input, so a total no warm met
/// compiles nothing. On the NAX
/// arm the k walk never depends on the row count (`rowsLaunch`), so a row keeps its bits.
pub var dyn_rows: bool = false;

var rows_nax_dyn_kernel: kernel_src.KernelSlots = kernel_src.no_kernels;

var rows_finish_dyn_kernel: ?mlx.mlx_fast_metal_kernel = null;

var rows_dyn_cfgs: kernel_src.CfgCache(RowsKey, 64) = .{};

var rows_finish_dyn_cfgs: kernel_src.CfgCache(RowsFinishKey, 64) = .{};

/// `src` with its row count `RN`, read at run time as `rn`; the threadgroup arrays keep the compiled ROWS (the cap).
fn dynRowsSource(comptime src: []const u8, comptime rn: []const u8) [:0]const u8 {
    comptime {
        var t: []const u8 = kernels.metalReplaceFound(src, "uint(ROWS)", "RN");
        // Past eight rows is decided at compile time on an eight-row pipeline, at run time on the cap's.
        if (std.mem.indexOf(u8, t, "ROWS > 8") != null) t = kernels.metalReplaceFound(t, "ROWS > 8", "(ROWS > 8 && RN > 8u)");
        for ([_][]const u8{ "PREP_F", "PART_F" }) |f| {
            if (std.mem.indexOf(u8, t, "constexpr uint " ++ f ++ " = RN") != null)
                t = kernels.metalReplaceFound(t, "constexpr uint " ++ f ++ " = RN", "constexpr uint " ++ f ++ " = uint(ROWS)");
        }
        return "const uint RN = " ++ rn ++ ";\n" ++ t;
    }
}

pub const ROWS_NAX_DYN_SOURCE: [:0]const u8 = dynRowsSource(ROWS_NAX_SOURCE, "uint(x_shape[0]) / uint(GROUPS)");

pub const ROWS_FINISH_DYN_SOURCE: [:0]const u8 = dynRowsSource(ROWS_FINISH_SOURCE, "uint(threadgroups_per_grid.y)");

/// The rows a dynamic-rows launch of `rows` compiles at: eight or the cap, so a small total keeps a wide chunk.
fn dynCap(rows: c_int) c_int {
    return if (rows <= 8) 8 else NAX_ROWS_MAX;
}

/// The geometry a dynamic-rows launch compiles at: `g`'s, staged at its cap's chunk.
fn dynGeom(l: RowsLaunch, g: RowsGeom) RowsGeom {
    var d = g;
    d.chunk = rowsChunk(l.in_dim, dynCap(l.rows), g.nsplit);
    return d;
}

/// How many template sets a dynamic-rows launch of this shape compiles over every row total: one a cap.
fn dynTemplates(in_dim: c_int, out_dim: c_int, groups: c_int, n: u32, mode: Mode, expected: NaxSplit) !usize {
    const Set = struct { cap: c_int, geom: RowsGeom };
    var seen: [@intCast(NAX_ROWS_MAX)]Set = undefined;
    var count: usize = 0;
    for (1..@as(usize, @intCast(NAX_ROWS_MAX)) + 1) |rows| {
        const l: RowsLaunch = .{ .in_dim = in_dim, .out_dim = out_dim, .rows = @intCast(rows), .groups = groups, .n = n, .out_dt = .bfloat16 };
        const d: Set = .{ .cap = dynCap(l.rows), .geom = dynGeom(l, rowsGeometry(in_dim, out_dim, @intCast(rows), groups, @intCast(l.n), mode)) };
        try std.testing.expectEqual(expected, NaxSplit{ .kph = d.geom.kph, .nsplit = d.geom.nsplit });
        for (seen[0..count]) |x| {
            if (std.meta.eql(x, d)) break;
        } else {
            seen[count] = d;
            count += 1;
        }
    }
    return count;
}

fn launchRowsDyn(s: mlx.mlx_stream, x: mlx.mlx_array, trellis: mlx.mlx_array, suh: mlx.mlx_array, svh: mlx.mlx_array, l: RowsLaunch, g: RowsGeom, bounds: [3]c_int) !mlx.mlx_array {
    const d = dynGeom(l, g);
    const key = RowsKey{ .in_dim = l.in_dim, .out_dim = l.out_dim, .rows = l.rows, .groups = l.groups, .n = l.n, .out_dt = l.out_dt, .bounds = bounds, .geom = d };
    const cfg = rows_dyn_cfgs.get(key) orelse blk: {
        const c = try buildRowsCfg(l, d, bounds, .{}, dynCap(l.rows));
        rows_dyn_cfgs.put(key, c);
        break :blk c;
    };
    const kernel = try kernel_src.codebookKernelWith(&rows_nax_dyn_kernel, "sushi_exl3_rows_nax_dyn", &ROWS_INS, &.{"y"}, ROWS_NAX_DYN_SOURCE, ROWS_NAX_FRAGS);
    const ov = try kernels.applyOuts(s, kernel, &.{ x, trellis, suh, svh }, cfg, 1);
    defer _ = mlx.mlx_vector_array_free(ov);
    var y = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, ov, 0));
    if (rowsFolds(d)) {
        const whole = y;
        y = mlx.mlx_array_new();
        return whole;
    }
    const fkey = RowsFinishKey{ .out_dim = l.out_dim, .rows = l.rows, .groups = l.groups, .nsplit = d.nsplit, .out_dt = l.out_dt, .bounds = bounds };
    const fcfg = rows_finish_dyn_cfgs.get(fkey) orelse blk: {
        const c = try buildFinishCfg(l, d.nsplit, bounds, dynCap(l.rows));
        rows_finish_dyn_cfgs.put(fkey, c);
        break :blk c;
    };
    const fk = try kernels.getNamedKernel(&rows_finish_dyn_kernel, "sushi_exl3_rows_finish_dyn", &.{ "part", "svh" }, &.{"y"}, ROWS_FINISH_DYN_SOURCE, ROWS_HELPERS);
    const fv = try kernels.applyOuts(s, fk, &.{ y, svh }, fcfg, 1);
    defer _ = mlx.mlx_vector_array_free(fv);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, fv, 0));
    return out;
}

const RowsFinishRmsKey = struct { out_dim: c_int, rows: c_int, nsplit: c_int, out_dt: mlx.mlx_dtype, bounds: [3]c_int, norm: c_int, thru: bool = false };

/// A stacked shape at each of its decode widths.
var rows_finish_rms_cfgs: kernel_src.CfgCache(RowsFinishRmsKey, 32) = .{};

var rows_finish_rms_kernel: ?mlx.mlx_fast_metal_kernel = null;

var rows_finish_rms_thru_kernel: ?mlx.mlx_fast_metal_kernel = null;

const RowsFinishRmsRingKey = struct { out_dim: c_int, rows: c_int, nsplit: c_int, out_dt: mlx.mlx_dtype, norm: c_int, kv_dim: c_int, rd: c_int, qat: bool, cap: c_int };

var rows_finish_rms_ring_cfgs: kernel_src.CfgCache(RowsFinishRmsRingKey, 32) = .{};

var rows_finish_rms_ring_kernel: ?mlx.mlx_fast_metal_kernel = null;

var rows_finish_rms_ring_ip_cfgs: kernel_src.CfgCache(RowsFinishRmsRingKey, 32) = .{};

var rows_finish_rms_ring_ip_kernel: ?mlx.mlx_fast_metal_kernel = null;

const RowsNaxRopeKey = struct { base: RowsKey, hd: c_int, rd: c_int };

var rows_nax_rope_cfgs: kernel_src.CfgCache(RowsNaxRopeKey, 32) = .{};

var rows_nax_rope_kernel: kernel_src.KernelSlots = kernel_src.no_kernels;

const RowsFinishRopeKey = struct { out_dim: c_int, rows: c_int, nsplit: c_int, out_dt: mlx.mlx_dtype, hd: c_int, rd: c_int };

var rows_finish_rope_cfgs: kernel_src.CfgCache(RowsFinishRopeKey, 32) = .{};

var rows_finish_rope_kernel: ?mlx.mlx_fast_metal_kernel = null;

const ROWS_ROPE_INS = [_][*:0]const u8{ "x", "trellis", "suh", "svh", "freqs", "pos" };

const RowsFinishExpandKey = struct { out_dim: c_int, rows: c_int, nsplit: c_int, out_dt: mlx.mlx_dtype, hc: c_int };

var rows_finish_expand_cfgs: kernel_src.CfgCache(RowsFinishExpandKey, 16) = .{};

var rows_finish_expand_kernel: ?mlx.mlx_fast_metal_kernel = null;

var rows_finish_row_major_cfgs: kernel_src.CfgCache(RowsFinishKey, 16) = .{};

var rows_finish_row_major_kernel: ?mlx.mlx_fast_metal_kernel = null;

pub const ROWS_INS = [_][*:0]const u8{ "x", "trellis", "suh", "svh" };

pub const ROWS_NAX_FRAGS: [:0]const u8 = kernels.GEMM_NAX_INCLUDES ++ kernels.GEMM_NAX_FRAGS ++ ROWS_HELPERS ++ W32_FRAGS;

/// `rows_wfrag` delegates to `nax_wfrag_k`, which now carries all branchless funnel fast paths across n16..n128.
const W32_FRAGS: [:0]const u8 =
    \\template<uint N, uint R32, uint SW>
    \\__attribute__((always_inline)) static inline nfrag rows_wfrag(const device uint *words, uint lane) {
    \\  return nax_wfrag_k<N>(words, lane);
    \\}
    \\
;

const ROWS_PEND_INS = ROWS_INS ++ [_][*:0]const u8{"xsvh"};

pub const ROWS_PEND_SOURCE: [:0]const u8 = kernels.metalReplaceFound(ROWS_SOURCE, ROWS_X_PLAIN, ROWS_X_PENDING);

pub const ROWS_NAX_PEND_SOURCE: [:0]const u8 = kernels.metalReplaceFound(ROWS_NAX_SOURCE, ROWS_X_PLAIN, ROWS_X_PENDING);

/// `rowsKernel` for a launch whose input is another launch's partials (`Pending`).
pub fn rowsPendKernel(g: RowsGeom) !mlx.mlx_fast_metal_kernel {
    if (g.nax) return kernel_src.codebookKernelWith(&rows_nax_pend_kernel, "sushi_exl3_rows_nax_pend", &ROWS_PEND_INS, &.{"y"}, ROWS_NAX_PEND_SOURCE, ROWS_NAX_FRAGS);
    return kernel_src.codebookKernelWith(&rows_pend_kernel, "sushi_exl3_rows_pend", &ROWS_PEND_INS, &.{"y"}, ROWS_PEND_SOURCE, ROWS_HELPERS);
}

pub const ROWS_THRU_SOURCE: [:0]const u8 = kernels.metalReplaceFound(ROWS_SOURCE, ROWS_X_PLAIN, ROWS_X_THRU);

pub const ROWS_NAX_THRU_SOURCE: [:0]const u8 = kernels.metalReplaceFound(ROWS_NAX_SOURCE, ROWS_X_PLAIN, ROWS_X_THRU);

var rows_thru_kernel: kernel_src.KernelSlots = kernel_src.no_kernels;

var rows_nax_thru_kernel: kernel_src.KernelSlots = kernel_src.no_kernels;

var rows_finish_thru_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// `rowsKernel` for a launch through bf16 (`Mode.thru_bf16`): f32 rows rounded where they are read. A folding launch
/// stores its outputs as OutT, bf16, into its f32 rows.
pub fn rowsThruKernel(g: RowsGeom) !mlx.mlx_fast_metal_kernel {
    if (g.nax) return kernel_src.codebookKernelWith(&rows_nax_thru_kernel, "sushi_exl3_rows_nax_thru", &ROWS_INS, &.{"y"}, ROWS_NAX_THRU_SOURCE, ROWS_NAX_FRAGS);
    return kernel_src.codebookKernelWith(&rows_thru_kernel, "sushi_exl3_rows_thru", &ROWS_INS, &.{"y"}, ROWS_THRU_SOURCE, ROWS_HELPERS);
}

/// A counted launch's stores: a partial by a relaxed device atomic, which the block's last threadgroup reads back; a
/// finished output as its output's type (`exl3_fin`), or by an atomic where the launch's last reads it back. The
/// SwiGLU is MLX's ops': its Sigmoid functor, then `(g · σ(g)) · u` with no fma contraction.
pub const COUNTED_STORE: [:0]const u8 =
    \\static inline void exl3_store_part(device float *p, float v) { metal::atomic_store_explicit((device metal::atomic_float *)p, v, metal::memory_order_relaxed); }
    \\template <typename T, typename V> static inline void exl3_fin(device T *p, V v) { *p = T(v); }
    \\template <typename V> static inline void exl3_fin_atomic(device float *p, V v) { exl3_store_part(p, float(v)); }
    \\static inline float exl3_sigmoid(float x) {
    \\  auto y = 1 / (1 + metal::precise::exp(metal::abs(x)));
    \\  return (x < 0) ? y : 1 - y;
    \\}
    \\static inline float exl3_swiglu(float g, float u) {
    \\  #pragma clang fp contract(off)
    \\  const float a = g * exl3_sigmoid(g);
    \\  return a * u;
    \\}
    \\
;

/// A counted launch's count-in: every threadgroup counts itself in at its 128-wide output block's counter in the
/// counter page `ctr`, fenced at device scope either side; all but the block's last of `ARRIVALS` return.
const COUNTED_ENTER: [:0]const u8 =
    \\  constexpr uint BT = 128u / (TGT * TILE);
    \\  threadgroup bool t_last[1];
    \\  const uint blk = uint(threadgroup_position_in_grid.x) / BT;
    \\  metal::atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst, metal::thread_scope_device);
    \\  threadgroup_barrier(mem_flags::mem_device);
    \\  if (lid == 0u) {
    \\    metal::atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst, metal::thread_scope_device);
    \\    t_last[0] = metal::atomic_fetch_add_explicit((device metal::atomic_uint *)ctr + blk, 1u, metal::memory_order_relaxed) == ARRIVALS - 1u;
    \\    metal::atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst, metal::thread_scope_device);
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  if (!t_last[0]) return;
    \\  metal::atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst, metal::thread_scope_device);
    \\
;

/// The block's partials at `PB` [ROWS · NSPLIT, ODIM] summed into `tgm` [ROWS, 128] as `ROWS_FINISH_SOURCE` sums
/// them: a thread a column, its parts in order. A thread loads the parts of all its columns before it sums any, while
/// they fit 96 registers.
const COUNTED_SUM: [:0]const u8 =
    \\  {
    \\    constexpr uint TT = SGS * 32u;
    \\    constexpr uint PER = (uint(ROWS) * 128u + TT - 1u) / TT;
    \\    constexpr uint BATCH = PER * uint(NSPLIT) <= 96u ? PER : 1u;
    \\    for (uint k0 = 0u; k0 < PER; k0 += BATCH) {
    \\      float pv[BATCH][NSPLIT];
    \\      #pragma clang loop unroll(full)
    \\      for (uint k = 0u; k < BATCH; k++) {
    \\        const uint i = lid + (k0 + k) * TT;
    \\        if (i < uint(ROWS) * 128u) {
    \\          const uint r = i / 128u;
    \\          device metal::atomic_float *p = (device metal::atomic_float *)PB + (size_t)r * uint(NSPLIT) * uint(ODIM) + blk * 128u + (i - r * 128u);
    \\          #pragma clang loop unroll(full)
    \\          for (uint s = 0u; s < uint(NSPLIT); s++) pv[k][s] = metal::atomic_load_explicit(p + (size_t)s * uint(ODIM), metal::memory_order_relaxed);
    \\        }
    \\      }
    \\      #pragma clang loop unroll(full)
    \\      for (uint k = 0u; k < BATCH; k++) {
    \\        const uint i = lid + (k0 + k) * TT;
    \\        if (i < uint(ROWS) * 128u) {
    \\          float acc = 0.0f;
    \\          #pragma clang loop unroll(full)
    \\          for (uint s = 0u; s < uint(NSPLIT); s++) acc += pv[k][s];
    \\          tgm[i] = acc;
    \\        }
    \\      }
    \\    }
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  const uint b = blk * 128u + lane;
    \\
;

/// A one-bank block's finish into `fin` [ROWS, ODIM]: a simdgroup a row takes the block's H128 and `psc · svh`, each
/// output rounded to OutT, as `ROWS_FINISH_SOURCE` finishes it; the block's counter goes back to zero.
const COUNTED_FIN: [:0]const u8 =
    \\  for (uint r = sg; r < uint(ROWS); r += SGS) {
    \\    const threadgroup float *pr = tgm + r * 128u + lane;
    \\    const float4 v = exl3_h128(float4(pr[0], pr[32], pr[64], pr[96]), lane);
    \\    const auto yr = fin + (size_t)r * uint(ODIM) + b;
    \\    exl3_fin(yr, OutT(v.x * psc * float(svh[b])));
    \\    exl3_fin(yr + 32u, OutT(v.y * psc * float(svh[b + 32u])));
    \\    exl3_fin(yr + 64u, OutT(v.z * psc * float(svh[b + 64u])));
    \\    exl3_fin(yr + 96u, OutT(v.w * psc * float(svh[b + 96u])));
    \\  }
    \\  if (lid == 0u) metal::atomic_store_explicit((device metal::atomic_uint *)ctr + blk, 0u, metal::memory_order_relaxed);
    \\
;

/// After its block, the launch's last block to finish (its counter `ctr[ODIM / 128]`) rms-norms each finished row
/// over `w` with `eps[0]` as `ROWS_FINISH_RMS_THRU_SOURCE` does: a virtual simdgroup a (row, 128-column block) squares
/// four neighbours a lane by fma and takes their `simd_sum`, a simdgroup a row the blocks' sums, `rsqrt` of the mean
/// plus eps; an output is `w · (x · inv)`. Every row at once: a simdgroup loads all its blocks before it sums any.
const COUNTED_NORM: [:0]const u8 =
    \\  {
    \\    #pragma clang fp contract(off)
    \\    constexpr uint NB = uint(ODIM) / 128u;
    \\    constexpr uint PAIRS = uint(ROWS) * NB;
    \\    constexpr uint PP = (PAIRS + SGS - 1u) / SGS;
    \\    static_assert(NB <= 32u, "a row's block sums are one simdgroup's lanes");
    \\    threadgroup float t_sums[uint(ROWS) * 32u];
    \\    threadgroup float t_inv[ROWS];
    \\    metal::atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst, metal::thread_scope_device);
    \\    threadgroup_barrier(mem_flags::mem_device);
    \\    if (lid == 0u) {
    \\      metal::atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst, metal::thread_scope_device);
    \\      t_last[0] = metal::atomic_fetch_add_explicit((device metal::atomic_uint *)ctr + NB, 1u, metal::memory_order_relaxed) == NB - 1u;
    \\      metal::atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst, metal::thread_scope_device);
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    if (!t_last[0]) return;
    \\    metal::atomic_thread_fence(metal::mem_flags::mem_device, metal::memory_order_seq_cst, metal::thread_scope_device);
    \\    for (uint i = lid; i < uint(ROWS) * 32u; i += SGS * 32u) t_sums[i] = 0.0f;
    \\    float tx[PP][4];
    \\    #pragma clang loop unroll(full)
    \\    for (uint k = 0u; k < PP; k++) {
    \\      const uint p = sg + k * SGS;
    \\      if (p < PAIRS) {
    \\        device metal::atomic_float *fp = (device metal::atomic_float *)fin + (size_t)(p / NB) * uint(ODIM) + (p % NB) * 128u + lane * 4u;
    \\        for (int i = 0; i < 4; i++) tx[k][i] = metal::atomic_load_explicit(fp + i, metal::memory_order_relaxed);
    \\      }
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    #pragma clang loop unroll(full)
    \\    for (uint k = 0u; k < PP; k++) {
    \\      const uint p = sg + k * SGS;
    \\      if (p < PAIRS) {
    \\        float acc = 0.0f;
    \\        for (int i = 0; i < 4; i++) acc = metal::fma(tx[k][i], tx[k][i], acc);
    \\        acc = simd_sum(acc);
    \\        if (lane == 0u) t_sums[(p / NB) * 32u + p % NB] = acc;
    \\      }
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint r = sg; r < uint(ROWS); r += SGS) {
    \\      const float acc = simd_sum(t_sums[r * 32u + lane]);
    \\      if (lane == 0u) t_inv[r] = metal::precise::rsqrt(acc / uint(ODIM) + eps[0]);
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    #pragma clang loop unroll(full)
    \\    for (uint k = 0u; k < PP; k++) {
    \\      const uint p = sg + k * SGS;
    \\      if (p < PAIRS) {
    \\        const uint c0 = (p % NB) * 128u + lane * 4u;
    \\        device float *fr = fin + (size_t)(p / NB) * uint(ODIM) + c0;
    \\        for (int i = 0; i < 4; i++) fr[i] = float(w[c0 + uint(i)]) * (tx[k][i] * t_inv[p / NB]);
    \\      }
    \\    }
    \\    if (lid == 0u) metal::atomic_store_explicit((device metal::atomic_uint *)ctr + NB, 0u, metal::memory_order_relaxed);
    \\  }
    \\
;

/// A gate and up pair's block (`launchRowsPair`): its last of both banks' threadgroups finishes the gate's block, then
/// the up's, each as `ROWS_FINISH_THRU_SOURCE` does, and stores `exl3_swiglu` of the two into `fin`.
const COUNTED_PAIR_FIN: [:0]const u8 =
    \\  constexpr uint RPS = (uint(ROWS) + SGS - 1u) / SGS;
    \\  float4 gv[RPS];
    \\  for (uint h = 0u; h < 2u; h++) {
++ kernels.metalReplaceFound(COUNTED_SUM, "PB", "(y0 + (size_t)h * uint(ROWS) * uint(NSPLIT) * uint(ODIM))") ++
    \\    const auto sv = h == 0u ? svh : svh2;
    \\    #pragma clang loop unroll(full)
    \\    for (uint q = 0u; q < RPS; q++) {
    \\      const uint r = sg + q * SGS;
    \\      if (r < uint(ROWS)) {
    \\        const threadgroup float *pr = tgm + r * 128u + lane;
    \\        const float4 v = exl3_h128(float4(pr[0], pr[32], pr[64], pr[96]), lane);
    \\        const float4 o = float4(float(OutT(v.x * psc * float(sv[b]))), float(OutT(v.y * psc * float(sv[b + 32u]))), float(OutT(v.z * psc * float(sv[b + 64u]))), float(OutT(v.w * psc * float(sv[b + 96u]))));
    \\        if (h == 0u) {
    \\          gv[q] = o;
    \\        } else {
    \\          device float *yr = fin + (size_t)r * uint(ODIM) + b;
    \\          yr[0] = exl3_swiglu(gv[q].x, o.x);
    \\          yr[32] = exl3_swiglu(gv[q].y, o.y);
    \\          yr[64] = exl3_swiglu(gv[q].z, o.z);
    \\          yr[96] = exl3_swiglu(gv[q].w, o.w);
    \\        }
    \\      }
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  }
    \\  if (lid == 0u) metal::atomic_store_explicit((device metal::atomic_uint *)ctr + blk, 0u, metal::memory_order_relaxed);
    \\
;

/// The tail a counted launch appends to its rows kernel, `fin` its finished rows.
const COUNTED_TAIL: [:0]const u8 = "{\n  static_assert(PART_F >= uint(ROWS) * 128u, \"the finish sums a block's rows in tgm\");\n" ++
    kernels.metalReplaceFound(COUNTED_ENTER, "ARRIVALS", "BT * uint(NSPLIT)") ++ kernels.metalReplaceFound(COUNTED_SUM, "PB", "y") ++ COUNTED_FIN ++ "}\n";

const COUNTED_NORM_TAIL: [:0]const u8 = "{\n  static_assert(PART_F >= uint(ROWS) * 128u, \"the finish sums a block's rows in tgm\");\n" ++
    kernels.metalReplaceFound(COUNTED_ENTER, "ARRIVALS", "BT * uint(NSPLIT)") ++ kernels.metalReplaceFound(COUNTED_SUM, "PB", "y") ++
    kernels.metalReplaceFound(COUNTED_FIN, "exl3_fin(", "exl3_fin_atomic(") ++ COUNTED_NORM ++ "}\n";

const COUNTED_PAIR_TAIL: [:0]const u8 = "{\n  static_assert(PART_F >= uint(ROWS) * 128u, \"the finish sums a block's rows in tgm\");\n  static_assert(GROUPS == 2, \"a pair is two banks\");\n" ++
    kernels.metalReplaceFound(COUNTED_ENTER, "ARRIVALS", "2u * BT * uint(NSPLIT)") ++ COUNTED_PAIR_FIN ++ "}\n";

/// A rows source whose partials leave by `exl3_store_part`.
fn countedRows(comptime src: [:0]const u8) [:0]const u8 {
    return kernels.metalReplaceFound(src, "exl3_store(y + ", "exl3_store_part(y + ");
}

/// `countedRows` of a pair: bank 1's threadgroups (grid z) read `trellis2` and `suh2` over the one input and store
/// their partials past bank 0's; `y0` keeps where bank 0's start.
fn pairRows(comptime src: [:0]const u8) [:0]const u8 {
    return kernels.metalReplaceFound(countedRows(src),
        \\if constexpr (GROUPS > 1) {
        \\  const size_t grp = threadgroup_position_in_grid.z;
        \\  x += grp * uint(ROWS) * uint(IDIM);
        \\  trellis += grp * (uint(IDIM) / TILE) * OT * N;
        \\  suh += grp * uint(IDIM);
        \\  svh += grp * uint(ODIM);
        \\
    ,
        \\device float *const y0 = y;
        \\if constexpr (GROUPS > 1) {
        \\  const size_t grp = threadgroup_position_in_grid.z;
        \\  if (grp != 0u) {
        \\    trellis = trellis2;
        \\    suh = suh2;
        \\  }
        \\
    );
}

pub const ROWS_THRU_COUNTED_SOURCE: [:0]const u8 = countedRows(ROWS_THRU_SOURCE) ++ COUNTED_TAIL;
pub const ROWS_NAX_THRU_COUNTED_SOURCE: [:0]const u8 = countedRows(ROWS_NAX_THRU_SOURCE) ++ COUNTED_TAIL;
pub const ROWS_THRU_COUNTED_RMS_SOURCE: [:0]const u8 = countedRows(ROWS_THRU_SOURCE) ++ COUNTED_NORM_TAIL;
pub const ROWS_NAX_THRU_COUNTED_RMS_SOURCE: [:0]const u8 = countedRows(ROWS_NAX_THRU_SOURCE) ++ COUNTED_NORM_TAIL;
pub const ROWS_THRU_COUNTED_PAIR_SOURCE: [:0]const u8 = pairRows(ROWS_THRU_SOURCE) ++ COUNTED_PAIR_TAIL;
pub const ROWS_NAX_THRU_COUNTED_PAIR_SOURCE: [:0]const u8 = pairRows(ROWS_NAX_THRU_SOURCE) ++ COUNTED_PAIR_TAIL;

/// A counted launch's kernels: [SIMD, NAX] of the one-bank, the normed and the pair's.
var rows_counted_kernels: [3][2]kernel_src.KernelSlots = @splat(@splat(kernel_src.no_kernels));

/// The most rows a trunk launch finishes its own partials for (`trunk`): a block's last threadgroup finishes every row
/// alone, a tail that past three rows outweighs the finish launch it saves (hy4-flash-arch.md).
pub const COUNTED_ROWS_MAX: c_int = 3;

/// `COUNTED_ROWS_MAX` for a launch leaving `c`: a normed launch's last block norms every row of the output alone, a
/// tail that past one row outweighs the finish it saves (hy4-flash-arch.md).
pub fn countedRowsMax(c: Counted) c_int {
    return if (c == .normed) 1 else COUNTED_ROWS_MAX;
}

/// What a counted launch's finisher leaves (`launchRowsCounted`).
pub const Counted = union(enum) {
    /// f32 rows of each output's bf16 rounding, as `rows_finish_thru` stores them.
    thru,
    /// bf16 rows, as `rows_finish` stores the same launch's over the rows cast to bf16.
    stored,
    /// `thru`'s rows rms-normed over `Norm.w` (the whole row), as `rows_finish_rms_thru` leaves them.
    normed: Norm,
};

const CountedKey = struct { rows: RowsKey, kind: enum { thru, stored, normed, pair } };

var rows_counted_cfgs: kernel_src.CfgCache(CountedKey, 64) = .{};

const ROWS_COUNTED_OUTS = [_][*:0]const u8{ "y", "fin", "ctr" };

/// Whether `launchRowsCounted` serves `l` under `g` on stream `s` leaving `c`: a one-bank through launch that leaves
/// partials, on a stream whose counter page holds a counter for each 128-wide output block (and, normed, one more).
pub fn countedFits(s: mlx.mlx_stream, l: RowsLaunch, g: RowsGeom, c: Counted) bool {
    const blocks = @divFloor(l.out_dim, 128) + @intFromBool(c == .normed);
    const fits = l.thru and l.groups == 1 and !rowsFolds(g) and @rem(l.out_dim, 128) == 0 and host_side.counters(s) >= blocks;
    return fits and switch (c) {
        .normed => |n| normedFits(l, g, @splat(l.out_dim), n),
        else => true,
    };
}

/// Whether `launchRowsPair` serves a gate and up pair of one shape `l` (one bank's launch) under `g` on stream `s`:
/// as `countedFits` serves one bank's.
pub fn pairFits(s: mlx.mlx_stream, l: RowsLaunch, g: RowsGeom) bool {
    return countedFits(s, l, g, .thru);
}

fn countedCfg(key: CountedKey, l: RowsLaunch, g: RowsGeom, fin_dt: mlx.mlx_dtype) !mlx.mlx_fast_metal_kernel_config {
    if (rows_counted_cfgs.get(key)) |c| return c;
    const c = try buildRowsCfg(l, g, key.rows.bounds, .{}, null);
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &.{ l.rows, l.out_dim }, 2, fin_dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &.{host_side.counter_page_words}, 1, .uint32));
    rows_counted_cfgs.put(key, c);
    return c;
}

/// The counted launch of `kernel` over `inputs`: its finished rows, or null, with nothing launched, where the stream
/// has no page.
fn applyCountedRows(s: mlx.mlx_stream, kernel: mlx.mlx_fast_metal_kernel, inputs: []const mlx.mlx_array, cfg: mlx.mlx_fast_metal_kernel_config) !?mlx.mlx_array {
    const iv = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(host_side.applyCounted(&ov, kernel, iv, cfg, s) orelse return null);
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, ov, 1));
    return y;
}

/// `launchRows` of a through launch (`countedFits`) whose finish rides it: the k-split partials leave by device
/// atomics and each output block's last threadgroup to count in finishes it, so the rows `[rows, out]` (f32, or bf16
/// when `stored`) are the bytes of the launch and its finish (and, `normed`, its rms norm) in one dispatch. Null,
/// with nothing launched, where the stream has no page.
pub fn launchRowsCounted(s: mlx.mlx_stream, x: mlx.mlx_array, trellis: mlx.mlx_array, suh: mlx.mlx_array, svh: mlx.mlx_array, l: RowsLaunch, g: RowsGeom, c: Counted) !?mlx.mlx_array {
    if (!countedFits(s, l, g, c)) return error.BadExl3Shape;
    const rk = RowsKey{ .in_dim = l.in_dim, .out_dim = l.out_dim, .rows = l.rows, .groups = 1, .n = l.n, .out_dt = l.out_dt, .bounds = @splat(l.out_dim), .geom = g, .thru = true };
    const nax: usize = @intFromBool(g.nax);
    switch (c) {
        .thru, .stored => {
            const cfg = try countedCfg(.{ .rows = rk, .kind = if (c == .thru) .thru else .stored }, l, g, if (c == .thru) .float32 else .bfloat16);
            const kernel = if (g.nax)
                try kernel_src.codebookKernelWith(&rows_counted_kernels[0][nax], "sushi_exl3_rows_nax_thru_counted", &ROWS_INS, &ROWS_COUNTED_OUTS, ROWS_NAX_THRU_COUNTED_SOURCE, ROWS_NAX_FRAGS ++ COUNTED_STORE)
            else
                try kernel_src.codebookKernelWith(&rows_counted_kernels[0][nax], "sushi_exl3_rows_thru_counted", &ROWS_INS, &ROWS_COUNTED_OUTS, ROWS_THRU_COUNTED_SOURCE, ROWS_HELPERS ++ COUNTED_STORE);
            return applyCountedRows(s, kernel, &.{ x, trellis, suh, svh }, cfg);
        },
        .normed => |n| {
            const cfg = try countedCfg(.{ .rows = rk, .kind = .normed }, l, g, .float32);
            const ins = ROWS_INS ++ [_][*:0]const u8{ "w", "eps" };
            const kernel = if (g.nax)
                try kernel_src.codebookKernelWith(&rows_counted_kernels[1][nax], "sushi_exl3_rows_nax_thru_counted_rms", &ins, &ROWS_COUNTED_OUTS, ROWS_NAX_THRU_COUNTED_RMS_SOURCE, ROWS_NAX_FRAGS ++ COUNTED_STORE)
            else
                try kernel_src.codebookKernelWith(&rows_counted_kernels[1][nax], "sushi_exl3_rows_thru_counted_rms", &ins, &ROWS_COUNTED_OUTS, ROWS_THRU_COUNTED_RMS_SOURCE, ROWS_HELPERS ++ COUNTED_STORE);
            return applyCountedRows(s, kernel, &.{ x, trellis, suh, svh, n.w, n.eps }, cfg);
        },
    }
}

/// A gate bank `a` and an up bank `b` of one shape over the same f32 rows `x` in one counted launch (`pairFits`, `l`
/// one bank's): each bank's threadgroups run its own launch's sums and the last of a block's to count in finishes both
/// as `rows_finish_thru` does, leaving `silu(gate) · up` `[rows, out]` f32 as MLX's Sigmoid and Multiply give it over
/// the two finished rows. Null, with nothing launched, where the stream has no page.
pub fn launchRowsPair(s: mlx.mlx_stream, x: mlx.mlx_array, a: [3]mlx.mlx_array, b: [3]mlx.mlx_array, l: RowsLaunch, g: RowsGeom) !?mlx.mlx_array {
    if (!pairFits(s, l, g)) return error.BadExl3Shape;
    var lp = l;
    lp.groups = 2;
    const rk = RowsKey{ .in_dim = l.in_dim, .out_dim = l.out_dim, .rows = l.rows, .groups = 2, .n = l.n, .out_dt = l.out_dt, .bounds = @splat(l.out_dim), .geom = g, .thru = true };
    const cfg = try countedCfg(.{ .rows = rk, .kind = .pair }, lp, g, .float32);
    const ins = ROWS_INS ++ [_][*:0]const u8{ "trellis2", "suh2", "svh2" };
    const nax: usize = @intFromBool(g.nax);
    const kernel = if (g.nax)
        try kernel_src.codebookKernelWith(&rows_counted_kernels[2][nax], "sushi_exl3_rows_nax_thru_counted_pair", &ins, &ROWS_COUNTED_OUTS, ROWS_NAX_THRU_COUNTED_PAIR_SOURCE, ROWS_NAX_FRAGS ++ COUNTED_STORE)
    else
        try kernel_src.codebookKernelWith(&rows_counted_kernels[2][nax], "sushi_exl3_rows_thru_counted_pair", &ins, &ROWS_COUNTED_OUTS, ROWS_THRU_COUNTED_PAIR_SOURCE, ROWS_HELPERS ++ COUNTED_STORE);
    return applyCountedRows(s, kernel, &.{ x, a[0], a[1], a[2], b[0], b[1], b[2] }, cfg);
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
/// A bench arm: null follows the mode, a value forces the NAX arm on or off at every width.
pub var nax_force: ?bool = null;

pub fn rowsGeometry(in_dim: c_int, out_dim: c_int, rows: c_int, groups: c_int, n: c_int, mode: Mode) RowsGeom {
    const nax = nax_force orelse if (mode.exact) !isLmHead(in_dim, out_dim, groups) else rows >= 2;
    return rowsGeometryFor(in_dim, out_dim, rows, groups, n, nax and kernels.gemmNaxOn());
}

fn isLmHead(in_dim: c_int, out_dim: c_int, groups: c_int) bool {
    return groups == 1 and in_dim == 5120 and out_dim == 129280;
}

/// DeepSeek's trunk linears on the NAX arm: `kph` simdgroups split a threadgroup's k range and sum in threadgroup
/// memory, so a launch needs fewer k-split partials (none, and no finish dispatch, at `nsplit` 1). The geometry never
/// depends on the row count, which keeps a row's bytes equal at every width.
const NaxSplit = struct { kph: c_int, nsplit: c_int };
/// `ultra_k5` / `ultra_k6`: the split a K5 (`n == 80`) or K6 (`n == 96`) bank takes on the widest desktop class (an M5
/// Ultra); every other box and rate keeps `kph`/`nsplit`. A different split sums in another order, so the bytes differ by GPU class.
const NaxTuned = struct { in_dim: c_int, out_dim: c_int, groups: c_int, kph: c_int, nsplit: c_int, ultra_k5: ?NaxSplit = null, ultra_k6: ?NaxSplit = null };

const NAX_TUNED = [_]NaxTuned{
    .{ .in_dim = 5120, .out_dim = 1792, .groups = 1, .kph = 4, .nsplit = 5, .ultra_k5 = .{ .kph = 1, .nsplit = 20 } },
    .{ .in_dim = 1280, .out_dim = 32768, .groups = 1, .kph = 4, .nsplit = 1, .ultra_k5 = .{ .kph = 2, .nsplit = 1 } },
    .{ .in_dim = 4096, .out_dim = 1024, .groups = 8, .kph = 4, .nsplit = 8, .ultra_k5 = .{ .kph = 4, .nsplit = 2 } },
    .{ .in_dim = 8192, .out_dim = 5120, .groups = 1, .kph = 2, .nsplit = 4, .ultra_k5 = .{ .kph = 1, .nsplit = 8 } },
    .{ .in_dim = 6144, .out_dim = 25600, .groups = 1, .kph = 2, .nsplit = 4, .ultra_k5 = .{ .kph = 2, .nsplit = 1 } },
    .{ .in_dim = 5120, .out_dim = 129280, .groups = 1, .kph = 1, .nsplit = 2, .ultra_k6 = .{ .kph = 1, .nsplit = 1 } },
};

/// Hy4-Flash K6 (`n == 96`) attention output projections on M5 Ultra (`class == .d`): cuts k-split partials on
/// `o` (32 -> 8) and `o_sparse` (16 -> 8), speeding up both `rows_nax` and `hy4_hc_expand_parts` while keeping
/// `nsplit > 1` for `applyParts`/`applyFinished`.
const NAX_ULTRA_K6 = [_]NaxTuned{
    .{ .in_dim = 16384, .out_dim = 3072, .groups = 1, .kph = 4, .nsplit = 8 },
    .{ .in_dim = 8192, .out_dim = 3072, .groups = 1, .kph = 4, .nsplit = 8 },
};

/// The NAX arm walks two k-tiles' fragments ahead of their MMAs at one simdgroup per pair, on 32-bit funnels: a one-row
/// trunk launch is bound by the fragment reads and decode, not the MMA.
fn rowsGeometryFor(in_dim: c_int, out_dim: c_int, rows: c_int, groups: c_int, n: c_int, nax: bool) RowsGeom {
    var g = rowsLaunch(in_dim, out_dim, rows, groups, n, nax);
    if (g.nax) {
        g.r32 = true;
        g.sw = true;
        if (g.kph == 1) g.unr = 2;
    }
    return g;
}

/// A bench arm: forces the NAX arm's `kph` and `nsplit` at every shape.
pub var nax_geom_force: ?struct { kph: c_int, nsplit: c_int } = null;

fn rowsLaunch(in_dim: c_int, out_dim: c_int, rows: c_int, groups: c_int, n: c_int, nax: bool) RowsGeom {
    if (nax) if (nax_geom_force) |f| return .{ .tgrp = 4, .kph = f.kph, .nsplit = f.nsplit, .chunk = rowsChunk(in_dim, rows, f.nsplit), .nax = true };
    if (nax) for (NAX_TUNED) |t| {
        if (t.in_dim == in_dim and t.out_dim == out_dim and t.groups == groups) {
            const base: NaxSplit = .{ .kph = t.kph, .nsplit = t.nsplit };
            const g = if (device.current().class == .d)
                (if (n == 80) t.ultra_k5 else if (n == 96) t.ultra_k6 else null) orelse base
            else
                base;
            return .{ .tgrp = 4, .kph = g.kph, .nsplit = g.nsplit, .chunk = rowsChunk(in_dim, rows, g.nsplit), .nax = true };
        }
    };
    if (nax and n == 96 and device.current().class == .d) for (NAX_ULTRA_K6) |t| {
        if (t.in_dim == in_dim and t.out_dim == out_dim and t.groups == groups) {
            return .{ .tgrp = 4, .kph = t.kph, .nsplit = t.nsplit, .chunk = rowsChunk(in_dim, rows, t.nsplit), .nax = true };
        }
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
    const g = rowsGeometry(xs[1], out, rows, groups, ts[ts.len - 1], mode);
    if (g.nax or groups != 1 or rows <= PROJECT_ROWS_MAX or rows > rowsMax(mode)) return projectRowsThru(s, x, trellis, suh, svh, g, null, mode.thru_bf16);
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
        const gp = rowsGeometry(xs[1], out, span[1] - span[0], 1, ts[ts.len - 1], mode);
        part.* = try projectRowsThru(s, xp, trellis, suh, svh, gp, null, mode.thru_bf16);
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
    const g = rowsGeometry(xs[1], out, xs[0], 1, ts[3], mode);
    const y = try projectRowsWith(s, x, trellis, suh, svh, g, try rowsKernel(g), bounds);
    defer _ = mlx.mlx_array_free(y);
    var flat = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, y, &[_]c_int{xs[0] * out}, 1, s));
    return flat;
}

pub fn projectRowsWith(s: mlx.mlx_stream, x: mlx.mlx_array, trellis: mlx.mlx_array, suh: mlx.mlx_array, svh: mlx.mlx_array, g: RowsGeom, kernel: mlx.mlx_fast_metal_kernel, seg_bounds: ?[3]c_int) !mlx.mlx_array {
    return projectRowsKernel(s, x, trellis, suh, svh, g, kernel, seg_bounds, false);
}

/// `projectRowsWith` on `g`'s own kernel, through bf16 when `thru` (`Mode.thru_bf16`).
fn projectRowsThru(s: mlx.mlx_stream, x: mlx.mlx_array, trellis: mlx.mlx_array, suh: mlx.mlx_array, svh: mlx.mlx_array, g: RowsGeom, seg_bounds: ?[3]c_int, thru: bool) !mlx.mlx_array {
    return projectRowsKernel(s, x, trellis, suh, svh, g, if (thru) try rowsThruKernel(g) else try rowsKernel(g), seg_bounds, thru);
}

fn projectRowsKernel(s: mlx.mlx_stream, x: mlx.mlx_array, trellis: mlx.mlx_array, suh: mlx.mlx_array, svh: mlx.mlx_array, g: RowsGeom, kernel: mlx.mlx_fast_metal_kernel, seg_bounds: ?[3]c_int, thru: bool) !mlx.mlx_array {
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
    return launchRows(s, x, trellis, suh, svh, .{ .in_dim = in_dim, .out_dim = out_tiles * 16, .rows = rows, .groups = groups, .n = rate.n, .out_dt = if (thru) .float32 else mlx.mlx_array_dtype(x), .thru = thru }, g, kernel, seg_bounds, null);
}

/// What one multi-row launch reads off its operands: their shapes, the rate and `x`'s dtype.
/// `thru`: f32 rows read through bf16 and outputs stored f32 as their bf16 rounding (`Mode.thru_bf16`; `out_dt` f32).
pub const RowsLaunch = struct { in_dim: c_int, out_dim: c_int, rows: c_int, groups: c_int, n: u32, out_dt: mlx.mlx_dtype, thru: bool = false };

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
    const cfg = try buildRowsCfg(l, g, bounds, .{}, null);
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const finish: ?mlx.mlx_fast_metal_kernel_config = if (rowsFolds(g)) null else try buildFinishCfg(l, g.nsplit, bounds, null);
    try pinned.append(std.heap.page_allocator, .{ .key = key, .cfgs = .{ .cfg = cfg, .finish = finish } });
    return pinned.items[pinned.items.len - 1].cfgs;
}

/// `tmpl_rows`: the ROWS the kernel is compiled at (null: `l.rows`); a dynamic-rows kernel takes the cap's.
fn buildRowsCfg(l: RowsLaunch, g: RowsGeom, bounds: [3]c_int, pend: Pending, tmpl_rows: ?c_int) !mlx.mlx_fast_metal_kernel_config {
    const fold = rowsFolds(g);
    const c = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    const sgs = g.tgrp * g.kph;
    const out_shape = [_]c_int{ l.groups * if (fold) l.rows else l.rows * g.nsplit, l.out_dim };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &out_shape, 2, if (fold) l.out_dt else .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "OutT", if (l.thru) .bfloat16 else l.out_dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, @divExact(@divExact(l.out_dim, 16), 2 * g.tgrp) * sgs * 32, g.nsplit, l.groups));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, sgs * 32, 1, 1));
    inline for (.{ "IDIM", "ODIM", "ROWS", "GROUPS", "TGRP", "KPH", "NSPLIT", "CHUNK", "FOLD" }, .{ l.in_dim, l.out_dim, tmpl_rows orelse l.rows, l.groups, g.tgrp, g.kph, g.nsplit, g.chunk, @as(c_int, @intFromBool(fold)) }) |name, v|
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
    if (dyn_rows and g.nax and !l.thru) return launchRowsDyn(s, x, trellis, suh, svh, l, g, bounds);
    const key = RowsKey{ .in_dim = l.in_dim, .out_dim = l.out_dim, .rows = l.rows, .groups = l.groups, .n = l.n, .out_dt = l.out_dt, .bounds = bounds, .geom = g, .thru = l.thru };
    const cfg = if (pin) |p| p.cfg else rows_cfgs.get(key) orelse blk: {
        const c = try buildRowsCfg(l, g, bounds, .{}, null);
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
    const wdt = mlx.mlx_array_dtype(norm.w);
    const w_ok = if (l.thru) wdt == .bfloat16 or wdt == .float32 else wdt == l.out_dt;
    return l.groups == 1 and !rowsFolds(g) and l.out_dim <= 4096 and norm.cols >= 128 and @rem(norm.cols, 128) == 0 and norm.cols <= bounds[0] and
        w_ok and mlx.mlx_array_size(norm.w) == @as(usize, @intCast(norm.cols)) and norm.cols == @min(bounds[0], l.out_dim);
}

/// `launchRows` of a stacked launch with `mlx_fast_rms_norm(first segment, norm.w, eps)` in place of the first
/// segment: the bytes of the launch, its finish and the norm, in one dispatch fewer.
pub fn launchRowsNormed(s: mlx.mlx_stream, x: mlx.mlx_array, trellis: mlx.mlx_array, suh: mlx.mlx_array, svh: mlx.mlx_array, l: RowsLaunch, g: RowsGeom, kernel: mlx.mlx_fast_metal_kernel, bounds: [3]c_int, pin: ?PinnedCfg, norm: Norm) !mlx.mlx_array {
    if (!normedFits(l, g, bounds, norm)) return error.BadExl3Shape;
    const key = RowsKey{ .in_dim = l.in_dim, .out_dim = l.out_dim, .rows = l.rows, .groups = 1, .n = l.n, .out_dt = l.out_dt, .bounds = bounds, .geom = g, .thru = l.thru };
    const cfg = if (pin) |p| p.cfg else rows_cfgs.get(key) orelse blk: {
        const c = try buildRowsCfg(l, g, bounds, .{}, null);
        rows_cfgs.put(key, c);
        break :blk c;
    };
    const pv = try kernels.applyOuts(s, kernel, &.{ x, trellis, suh, svh }, cfg, 1);
    defer _ = mlx.mlx_vector_array_free(pv);
    var part = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(part);
    try mlx.check(mlx.mlx_vector_array_get(&part, pv, 0));
    const fkey = RowsFinishRmsKey{ .out_dim = l.out_dim, .rows = l.rows, .nsplit = g.nsplit, .out_dt = l.out_dt, .bounds = bounds, .norm = norm.cols, .thru = l.thru };
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
    const fk = if (l.thru)
        try kernels.getNamedKernel(&rows_finish_rms_thru_kernel, "sushi_exl3_rows_finish_rms_thru", &.{ "part", "svh", "w", "eps" }, &.{"y"}, ROWS_FINISH_RMS_THRU_SOURCE, ROWS_HELPERS)
    else
        try kernels.getNamedKernel(&rows_finish_rms_kernel, "sushi_exl3_rows_finish_rms", &.{ "part", "svh", "w", "eps" }, &.{"y"}, ROWS_FINISH_RMS_SOURCE, ROWS_HELPERS);
    const ov = try kernels.applyOuts(s, fk, &.{ part, svh, norm.w, norm.eps }, fcfg, 1);
    defer _ = mlx.mlx_vector_array_free(ov);
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, ov, 0));
    return y;
}

/// Tail-RoPE parameters for a query projection (`wq_b`): `hd` head dim (multiple of 128), `rd` rotated tail (multiple
/// of 32), `freqs` `[rd/2]` f32, `pos` `[rows]` i32.
pub const Rope = struct { freqs: mlx.mlx_array, pos: mlx.mlx_array, hd: c_int, rd: c_int };

/// Ring-update parameters for `launchRowsNormedRing`: norms + tail-rotates (+ optional fp8 QAT) the `kv` segment
/// right after `norm.cols` and writes it into `ring` `[cap, kv_dim]` in the same finish dispatch.
/// When `qout_in_place` is set on the Cooler engine stream, writes `qout` and updates `ring` in place with zero copy.
pub const KvRing = struct { kw: mlx.mlx_array, freqs: mlx.mlx_array, pos: mlx.mlx_array, ring: mlx.mlx_array, cap: c_int, kv_dim: c_int, rd: c_int, qat: bool, qout_in_place: mlx.mlx_array = .{} };

pub const NormedRingOut = struct { q: mlx.mlx_array, ring: mlx.mlx_array, in_place: bool = false };

pub fn normedRingFits(l: RowsLaunch, g: RowsGeom, bounds: [3]c_int, norm: Norm, kr: KvRing) bool {
    if (l.thru or !normedFits(l, g, bounds, norm)) return false;
    if (l.out_dim != norm.cols + kr.kv_dim or kr.kv_dim < 128 or @rem(kr.kv_dim, 128) != 0) return false;
    if (kr.rd <= 0 or @rem(kr.rd, 4) != 0 or kr.rd > kr.kv_dim or kr.cap < l.rows) return false;
    if (mlx.mlx_array_dtype(kr.kw) != l.out_dt or mlx.mlx_array_dtype(kr.ring) != l.out_dt) return false;
    if (mlx.mlx_array_dtype(kr.freqs) != .float32 or mlx.mlx_array_dtype(kr.pos) != .int32) return false;
    const elem_bytes: c_int = if (l.out_dt == .float32) 4 else 2;
    return @rem(kr.kv_dim * elem_bytes, 16) == 0;
}

fn launchRowsNormedRingPart(s: mlx.mlx_stream, x: mlx.mlx_array, trellis: mlx.mlx_array, suh: mlx.mlx_array, svh: mlx.mlx_array, l: RowsLaunch, g: RowsGeom, kernel: mlx.mlx_fast_metal_kernel, bounds: [3]c_int, pin: ?PinnedCfg, norm: Norm, kr: KvRing) !mlx.mlx_array {
    if (!normedRingFits(l, g, bounds, norm, kr)) return error.BadExl3Shape;
    const key = RowsKey{ .in_dim = l.in_dim, .out_dim = l.out_dim, .rows = l.rows, .groups = 1, .n = l.n, .out_dt = l.out_dt, .bounds = bounds, .geom = g, .thru = l.thru };
    const cfg = if (pin) |p| p.cfg else rows_cfgs.get(key) orelse blk: {
        const c = try buildRowsCfg(l, g, bounds, .{}, null);
        rows_cfgs.put(key, c);
        break :blk c;
    };
    const pv = try kernels.applyOuts(s, kernel, &.{ x, trellis, suh, svh }, cfg, 1);
    defer _ = mlx.mlx_vector_array_free(pv);
    var part = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(part);
    try mlx.check(mlx.mlx_vector_array_get(&part, pv, 0));
    return part;
}

/// `launchRowsNormed` for a 2-segment `[q_lora, kv_row]` attention input that writes the normed query `[rows, norm.cols]`
/// and the updated KV ring `[cap, kv_dim]` in one finish dispatch, eliminating `ropeKvRing`.
pub fn launchRowsNormedRing(s: mlx.mlx_stream, x: mlx.mlx_array, trellis: mlx.mlx_array, suh: mlx.mlx_array, svh: mlx.mlx_array, l: RowsLaunch, g: RowsGeom, kernel: mlx.mlx_fast_metal_kernel, bounds: [3]c_int, pin: ?PinnedCfg, norm: Norm, kr: KvRing) !NormedRingOut {
    const part = try launchRowsNormedRingPart(s, x, trellis, suh, svh, l, g, kernel, bounds, pin, norm, kr);
    defer _ = mlx.mlx_array_free(part);
    const fkey = RowsFinishRmsRingKey{ .out_dim = l.out_dim, .rows = l.rows, .nsplit = g.nsplit, .out_dt = l.out_dt, .norm = norm.cols, .kv_dim = kr.kv_dim, .rd = kr.rd, .qat = kr.qat, .cap = kr.cap };
    if (kr.qout_in_place.ctx != null) {
        const fcfg_ip = rows_finish_rms_ring_ip_cfgs.get(fkey) orelse blk: {
            const c = mlx.mlx_fast_metal_kernel_config_new();
            errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
            const threads = @divExact(l.out_dim, 4);
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &.{ l.rows, norm.cols }, 2, l.out_dt));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &.{ kr.cap, kr.kv_dim }, 2, l.out_dt));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "OutT", l.out_dt));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, l.rows * threads, 1, 1));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, threads, 1, 1));
            inline for (.{ "ODIM", "ROWS", "NSPLIT", "NORM", "KV_D", "RD", "QAT", "CAP" }, .{ l.out_dim, l.rows, g.nsplit, norm.cols, kr.kv_dim, kr.rd, @as(c_int, @intFromBool(kr.qat)), kr.cap }) |name, v|
                try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, name, v));
            rows_finish_rms_ring_ip_cfgs.put(fkey, c);
            break :blk c;
        };
        const fk_ip = try kernels.getNamedKernel(&rows_finish_rms_ring_ip_kernel, "sushi_exl3_rows_finish_rms_ring_ip", &.{ "part", "svh", "w", "eps", "kw", "freqs", "pos" }, &.{ "qout", "rout" }, ROWS_FINISH_RMS_RING_IP_SOURCE, ROWS_RING_HELPERS);
        const ins = [_]mlx.mlx_array{ part, svh, norm.w, norm.eps, kr.kw, kr.freqs, kr.pos };
        const iv = mlx.mlx_vector_array_new_data(&ins, ins.len);
        defer _ = mlx.mlx_vector_array_free(iv);
        if (host_side.applyInPlace(fk_ip, iv, fcfg_ip, s, &.{ kr.qout_in_place, kr.ring })) |rc| {
            try mlx.check(rc);
            return .{ .q = kr.qout_in_place, .ring = kr.ring, .in_place = true };
        }
    }
    const fcfg = rows_finish_rms_ring_cfgs.get(fkey) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const threads = @divExact(l.out_dim, 4);
        const elem_bytes: c_int = if (l.out_dt == .float32) 4 else 2;
        const row_vecs = @divExact(kr.kv_dim * elem_bytes, 16);
        const ctgs = @divFloor(kr.cap * row_vecs + threads - 1, threads);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &.{ l.rows, norm.cols }, 2, l.out_dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &.{ kr.cap, kr.kv_dim }, 2, l.out_dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "OutT", l.out_dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, (l.rows + ctgs) * threads, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, threads, 1, 1));
        inline for (.{ "ODIM", "ROWS", "NSPLIT", "NORM", "KV_D", "RD", "QAT" }, .{ l.out_dim, l.rows, g.nsplit, norm.cols, kr.kv_dim, kr.rd, @as(c_int, @intFromBool(kr.qat)) }) |name, v|
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, name, v));
        rows_finish_rms_ring_cfgs.put(fkey, c);
        break :blk c;
    };
    const fk = try kernels.getNamedKernel(&rows_finish_rms_ring_kernel, "sushi_exl3_rows_finish_rms_ring", &.{ "part", "svh", "w", "eps", "kw", "freqs", "pos", "ring" }, &.{ "qout", "rout" }, ROWS_FINISH_RMS_RING_SOURCE, ROWS_RING_HELPERS);
    const ov = try kernels.applyOuts(s, fk, &.{ part, svh, norm.w, norm.eps, kr.kw, kr.freqs, kr.pos, kr.ring }, fcfg, 2);
    defer _ = mlx.mlx_vector_array_free(ov);
    var qout = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(qout);
    var rout = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(rout);
    try mlx.check(mlx.mlx_vector_array_get(&qout, ov, 0));
    try mlx.check(mlx.mlx_vector_array_get(&rout, ov, 1));
    return .{ .q = qout, .ring = rout, .in_place = false };
}

pub fn ropeFits(l: RowsLaunch, g: RowsGeom, rp: Rope) bool {
    if (l.groups != 1 or l.thru or dyn_rows) return false;
    if (rp.hd < 128 or @rem(rp.hd, 128) != 0 or @rem(l.out_dim, rp.hd) != 0) return false;
    if (rp.rd <= 0 or @rem(rp.rd, 32) != 0 or rp.rd > rp.hd) return false;
    if (mlx.mlx_array_dtype(rp.freqs) != .float32 or mlx.mlx_array_dtype(rp.pos) != .int32) return false;
    return if (rowsFolds(g)) g.nax else true;
}

/// `launchRows` with tail RoPE folded into the output store (in `ROWS_NAX_ROPE_SOURCE` when folded, or in
/// `ROWS_FINISH_ROPE_SOURCE` when k-split), eliminating the full-query DRAM roundtrip in `ropeKvRing`.
pub fn launchRowsRope(s: mlx.mlx_stream, x: mlx.mlx_array, trellis: mlx.mlx_array, suh: mlx.mlx_array, svh: mlx.mlx_array, l: RowsLaunch, g: RowsGeom, rp: Rope) !mlx.mlx_array {
    if (!ropeFits(l, g, rp)) return error.BadExl3Shape;
    const bounds: [3]c_int = @splat(l.out_dim);
    if (rowsFolds(g)) {
        const key = RowsNaxRopeKey{ .base = .{ .in_dim = l.in_dim, .out_dim = l.out_dim, .rows = l.rows, .groups = 1, .n = l.n, .out_dt = l.out_dt, .bounds = bounds, .geom = g, .thru = l.thru }, .hd = rp.hd, .rd = rp.rd };
        const cfg = rows_nax_rope_cfgs.get(key) orelse blk: {
            const c = try buildRowsCfg(l, g, bounds, .{}, null);
            errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "HD", rp.hd));
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "RD", rp.rd));
            rows_nax_rope_cfgs.put(key, c);
            break :blk c;
        };
        const kernel = try kernel_src.codebookKernelWith(&rows_nax_rope_kernel, "sushi_exl3_rows_nax_rope", &ROWS_ROPE_INS, &.{"y"}, ROWS_NAX_ROPE_SOURCE, ROWS_NAX_FRAGS);
        const ov = try kernels.applyOuts(s, kernel, &.{ x, trellis, suh, svh, rp.freqs, rp.pos }, cfg, 1);
        defer _ = mlx.mlx_vector_array_free(ov);
        var y = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(y);
        try mlx.check(mlx.mlx_vector_array_get(&y, ov, 0));
        return y;
    }
    const part = try launchRowsParts(s, x, trellis, suh, svh, l, g);
    defer _ = mlx.mlx_array_free(part);
    const fkey = RowsFinishRopeKey{ .out_dim = l.out_dim, .rows = l.rows, .nsplit = g.nsplit, .out_dt = l.out_dt, .hd = rp.hd, .rd = rp.rd };
    const fcfg = rows_finish_rope_cfgs.get(fkey) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &.{ l.rows, l.out_dim }, 2, l.out_dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "OutT", l.out_dt));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, @divExact(l.out_dim, 128) * 32, l.rows, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32, 1, 1));
        inline for (.{ "ODIM", "NSPLIT", "HD", "RD" }, .{ l.out_dim, g.nsplit, rp.hd, rp.rd }) |name, v|
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, name, v));
        rows_finish_rope_cfgs.put(fkey, c);
        break :blk c;
    };
    const fk = try kernels.getNamedKernel(&rows_finish_rope_kernel, "sushi_exl3_rows_finish_rope", &.{ "part", "svh", "freqs", "pos" }, &.{"y"}, ROWS_FINISH_ROPE_SOURCE, ROWS_HELPERS);
    const ov = try kernels.applyOuts(s, fk, &.{ part, svh, rp.freqs, rp.pos }, fcfg, 1);
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
    const key = RowsKey{ .in_dim = l.in_dim, .out_dim = l.out_dim, .rows = l.rows, .groups = l.groups, .n = l.n, .out_dt = l.out_dt, .bounds = bounds, .geom = g, .thru = l.thru };
    const cfg = rows_cfgs.get(key) orelse blk: {
        const c = try buildRowsCfg(l, g, bounds, .{}, null);
        rows_cfgs.put(key, c);
        break :blk c;
    };
    const ov = try kernels.applyOuts(s, if (l.thru) try rowsThruKernel(g) else try rowsKernel(g), &.{ x, trellis, suh, svh }, cfg, 1);
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
        const c = try buildRowsCfg(l, g, bounds, pend, null);
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
        const c = try buildFinishCfg(l, nsplit, bounds, null);
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

fn buildFinishCfg(l: RowsLaunch, nsplit: c_int, bounds: [3]c_int, tmpl_rows: ?c_int) !mlx.mlx_fast_metal_kernel_config {
    const c = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &.{ l.groups * l.rows, l.out_dim }, 2, l.out_dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "OutT", if (l.thru) .bfloat16 else l.out_dt));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, @divExact(l.out_dim, 128) * 32, l.rows, l.groups));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ODIM", l.out_dim));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ROWS", tmpl_rows orelse l.rows));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "NSPLIT", nsplit));
    inline for (.{ "SB1", "SB2", "SB3" }, 0..) |name, i| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, name, bounds[i]));
    return c;
}

fn rowsFinish(s: mlx.mlx_stream, part: mlx.mlx_array, svh: mlx.mlx_array, l: RowsLaunch, nsplit: c_int, bounds: [3]c_int, pin: ?mlx.mlx_fast_metal_kernel_config) !mlx.mlx_array {
    const key = RowsFinishKey{ .out_dim = l.out_dim, .rows = l.rows, .groups = l.groups, .nsplit = nsplit, .out_dt = l.out_dt, .bounds = bounds, .thru = l.thru };
    const cfg = pin orelse rows_finish_cfgs.get(key) orelse blk: {
        const c = try buildFinishCfg(l, nsplit, bounds, null);
        rows_finish_cfgs.put(key, c);
        break :blk c;
    };
    const kernel = if (l.thru)
        try kernels.getNamedKernel(&rows_finish_thru_kernel, "sushi_exl3_rows_finish_thru", &.{ "part", "svh" }, &.{"y"}, ROWS_FINISH_THRU_SOURCE, ROWS_HELPERS)
    else
        try kernels.getNamedKernel(&rows_finish_kernel, "sushi_exl3_rows_finish", &.{ "part", "svh" }, &.{"y"}, ROWS_FINISH_SOURCE, ROWS_HELPERS);
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
        const x = try host_side.made(mlx.mlx_array_new_data(xs.ptr, &[_]c_int{ @intCast(self.groups * rows), @intCast(self.in) }, 2, .float16));
        defer _ = mlx.mlx_array_free(x);
        const tr = try host_side.made(mlx.mlx_array_new_data(self.tr.ptr, &[_]c_int{ @intCast(self.groups), @intCast(self.in / 16), @intCast(self.out / 16), @intCast(rate.halfwords()) }, 4, .uint16));
        defer _ = mlx.mlx_array_free(tr);
        const suh = try host_side.made(mlx.mlx_array_new_data(self.suh.ptr, &[_]c_int{ @intCast(self.groups), @intCast(self.in) }, 2, .float16));
        defer _ = mlx.mlx_array_free(suh);
        const svh = try host_side.made(mlx.mlx_array_new_data(self.svh.ptr, &[_]c_int{ @intCast(self.groups), @intCast(self.out) }, 2, .float16));
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
                try host_side.made(mlx.mlx_array_new_data(self.xh[first * self.in ..].ptr, &[_]c_int{ @intCast(rows), @intCast(self.in) }, 2, .float16)),
                try host_side.made(mlx.mlx_array_new_data(self.tr.ptr, &[_]c_int{ 1, @intCast(self.in / 16), @intCast(self.out / 16), @intCast(self.n) }, 4, .uint16)),
                try host_side.made(mlx.mlx_array_new_data(self.suh.ptr, &[_]c_int{ 1, @intCast(self.in) }, 2, .float16)),
                try host_side.made(mlx.mlx_array_new_data(self.svh.ptr, &[_]c_int{ 1, @intCast(self.out) }, 2, .float16)),
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

test "exl3 multi-row projection matches projectIndexed per row against the f64 projection" {
    if (!mlx.streamIsGpu(mlx.gpuStream())) return error.SkipZigTest;
    const shapes = [_][2]usize{ .{ 512, 640 }, .{ 6144, 384 } };
    // One row takes the one-row arm, three the multi-row arm.
    for ([_]u32{ 40, 64, 112 }) |n| {
        for (shapes) |sh| {
            for ([_]usize{ 1, 3 }) |rows| {
                try kernels.reportGemmParity(try kernels.rowsParityStats(.{ .n = n, .out_dim = sh[0], .in_dim = sh[1], .rows = rows, .seed = 1301 }));
            }
        }
    }
}

test "exl3 multi-row projection holds the bar on every launch geometry and at eight rows" {
    if (!mlx.streamIsGpu(mlx.gpuStream())) return error.SkipZigTest;
    // The launch classes `rowsGeometryFor` serves: SIMD folded (lm_head-wide), one-row and multi-row k splits; NAX at one
    // simdgroup per pair folded and split, and at sliced k folded and split, each with its served fragment reader.
    const geoms = [_]RowsGeom{
        .{ .tgrp = 4, .kph = 2, .nsplit = 1, .chunk = 384 },
        .{ .tgrp = 1, .kph = 4, .nsplit = 3, .chunk = 128 },
        .{ .tgrp = 2, .kph = 4, .nsplit = 3, .chunk = 128 },
        .{ .tgrp = 4, .kph = 1, .nsplit = 1, .chunk = 0, .nax = true, .unr = 2, .r32 = true, .sw = true },
        .{ .tgrp = 4, .kph = 1, .nsplit = 3, .chunk = 0, .nax = true, .unr = 2, .r32 = true, .sw = true },
        .{ .tgrp = 4, .kph = 4, .nsplit = 1, .chunk = 0, .nax = true, .r32 = true, .sw = true },
        .{ .tgrp = 4, .kph = 2, .nsplit = 3, .chunk = 0, .nax = true, .r32 = true, .sw = true },
    };
    for (geoms, 0..) |g0, gi| {
        if (g0.nax and !kernels.gemmNaxOn()) continue;
        for ([_]usize{ 1, 8 }) |rows| {
            var g = g0;
            if (g.nax) g.chunk = rowsChunk(384, @intCast(rows), g.nsplit);
            // n48 takes the 64-bit reader, n96 the 32-bit one with its four-word read.
            for ([_]u32{ 48, 96 }) |n| {
                try kernels.reportGemmParity(try kernels.rowsParityStats(.{ .n = n, .out_dim = 512, .in_dim = 384, .rows = rows, .seed = 1401 + gi, .geom = g }));
            }
        }
    }
}

test "exl3 multi-row projection holds the bar at every admitted rate on both kernels" {
    if (!mlx.streamIsGpu(mlx.gpuStream())) return error.SkipZigTest;
    const nax = rowsGeometryFor(256, 256, 3, 1, 0, true);
    for ([_]u32{ 16, 24, 32, 42, 48, 64, 80, 96, 128 }) |n| {
        try kernels.reportGemmParity(try kernels.rowsParityStats(.{ .n = n, .out_dim = 256, .in_dim = 256, .rows = 3, .seed = 1601 + n, .geom = .{ .tgrp = 1, .kph = 4, .nsplit = 2, .chunk = 128 } }));
        if (kernels.gemmNaxOn()) try kernels.reportGemmParity(try kernels.rowsParityStats(.{ .n = n, .out_dim = 256, .in_dim = 256, .rows = 3, .seed = 1601 + n, .geom = nax }));
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
    // A two-bank launch; linear.zig's exact-rows test holds one-bank launches under the SIMD profile.
    const d = SIMD_ROWS_DECODES[2];
    kernels.setDecodeParams(d.dec);
    const sh = [3]usize{ 384, 256, 2 };
    const rig = try RowsRig.init(alloc, 0x51d0 + sh[0], sh[0], sh[1], sh[2], d.n);
    const g1 = rowsGeometryFor(@intCast(sh[0]), @intCast(sh[1]), 1, @intCast(sh[2]), @intCast(d.n), false);
    var alone: [@as(usize, @intCast(PROJECT_ROWS_MAX))][]const u16 = undefined;
    for (&alone, 0..) |*a, r| a.* = try rig.run(alloc, r, 1, g1, try rowsKernel(g1));
    for ([_]usize{ 2, @intCast(PROJECT_ROWS_MAX) }) |rows| {
        const g = rowsGeometryFor(@intCast(sh[0]), @intCast(sh[1]), @intCast(rows), @intCast(sh[2]), @intCast(d.n), false);
        const many = try rig.run(alloc, 0, rows, g, try rowsKernel(g));
        for (0..rows) |r| {
            for (0..sh[2]) |grp| try std.testing.expectEqualSlices(u16, alone[r][grp * sh[1] ..][0..sh[1]], many[(grp * rows + r) * sh[1] ..][0..sh[1]]);
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
    try std.testing.expect(!rowsGeometry(5120, 129280, 9, 1, 64, .{ .exact = true, .wide = true }).nax);
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
    const dec = exl3.Decode.tabled(.spec1, .w15);
    const n: u32 = 96;
    try dec.requireTable();
    kernels.setDecodeParams(dec);
    // A bank and a deep k split; trunk.zig's wide verify test holds a one-bank launch.
    for ([_][3]usize{ .{ 1024, 256, 2 }, .{ 131072, 256, 1 } }) |sh| {
        const rig = try RowsRig.init(alloc, 0x9a16 + sh[0] + n, sh[0], sh[1], sh[2], n);
        const g1 = rowsGeometry(@intCast(sh[0]), @intCast(sh[1]), 1, @intCast(sh[2]), @intCast(n), exact);
        var alone: [@as(usize, @intCast(NAX_ROWS_MAX))][]const u16 = undefined;
        for (&alone, 0..) |*a, r| a.* = try rig.run(alloc, r, 1, g1, try rowsKernel(g1));
        // 9 and 16 rows stage the two chunk widths `rowsChunk` gives past eight.
        for ([_]usize{ 9, @intCast(NAX_ROWS_MAX) }) |rows| {
            const g = rowsGeometry(@intCast(sh[0]), @intCast(sh[1]), @intCast(rows), @intCast(sh[2]), @intCast(n), exact);
            const many = try rig.run(alloc, 0, rows, g, try rowsKernel(g));
            for (0..sh[2]) |grp| for (0..rows) |r|
                try std.testing.expectEqualSlices(u16, alone[r][grp * sh[1] ..][0..sh[1]], many[(grp * rows + r) * sh[1] ..][0..sh[1]]);
        }
    }
}

test "exl3 NAX joined rows: one launch at every row total gives each row its one-row launch's bits, folded and k-split" {
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s) or !kernels.gemmNaxOn()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    defer kernels.setDecodeParams(.mul1);
    const exact: Mode = .{ .exact = true };
    const dec = exl3.Decode.tabled(.spec1, .w15);
    const n: u32 = 96;
    try dec.requireTable();
    kernels.setDecodeParams(dec);
    defer dyn_rows = false;
    // A bank on the folded launch and a deep k split through the finish.
    for ([_][3]usize{ .{ 1024, 256, 2 }, .{ 131072, 256, 1 } }) |sh| {
        const rig = try RowsRig.init(alloc, 0x70a1 + sh[0] + n, sh[0], sh[1], sh[2], n);
        const g1 = rowsGeometry(@intCast(sh[0]), @intCast(sh[1]), 1, @intCast(sh[2]), @intCast(n), exact);
        dyn_rows = false;
        var alone: [@as(usize, @intCast(NAX_ROWS_MAX))][]const u16 = undefined;
        for (&alone, 0..) |*a, r| a.* = try rig.run(alloc, r, 1, g1, try rowsKernel(g1));
        dyn_rows = true;
        for (1..@as(usize, @intCast(NAX_ROWS_MAX)) + 1) |rows| {
            const g = rowsGeometry(@intCast(sh[0]), @intCast(sh[1]), @intCast(rows), @intCast(sh[2]), @intCast(n), exact);
            const many = try rig.run(alloc, 0, rows, g, try rowsKernel(g));
            for (0..sh[2]) |grp| for (0..rows) |r| {
                errdefer std.debug.print("joined rows: {d}x{d} bank of {d}, {d} rows, row {d} of group {d}\n", .{ sh[0], sh[1], sh[2], rows, r, grp });
                try std.testing.expectEqualSlices(u16, alone[r][grp * sh[1] ..][0..sh[1]], many[(grp * rows + r) * sh[1] ..][0..sh[1]]);
            };
        }
        // Every total ran one template set a cap: eight rows' and the cap's.
        try std.testing.expectEqual(@as(usize, 2), try dynTemplates(@intCast(sh[0]), @intCast(sh[1]), @intCast(sh[2]), n, exact, .{ .kph = g1.kph, .nsplit = g1.nsplit }));
    }
    const profile = device.withProfile(device.Profile.from("g17d", .{ .major = 26, .minor = 2 }, false));
    defer profile.end();
    try std.testing.expectEqual(@as(usize, 2), try dynTemplates(8192, 5120, 1, 80, exact, .{ .kph = 1, .nsplit = 8 }));
    try std.testing.expectEqual(@as(usize, 2), try dynTemplates(8192, 5120, 1, 64, exact, .{ .kph = 2, .nsplit = 4 }));
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

test "exl3 multi-row mode: exact puts every width but the lm_head's on the NAX arm, and wide lifts the row ceiling only with exact on NAX" {
    try mlx.requireGpu();
    const nax = kernels.gemmNaxOn();
    try std.testing.expectEqual(PROJECT_ROWS_MAX, rowsMax(.{}));
    try std.testing.expect(!rowsGeometry(2560, 12288, 1, 1, 64, .{}).nax);
    try std.testing.expectEqual(nax, rowsGeometry(2560, 12288, 2, 1, 64, .{}).nax);
    try std.testing.expectEqual(PROJECT_ROWS_MAX, rowsMax(.{ .wide = true }));
    try std.testing.expectEqual(if (nax) NAX_ROWS_MAX else PROJECT_ROWS_MAX, rowsMax(.{ .exact = true, .wide = true }));
    try std.testing.expectEqual(nax, rowsGeometry(2560, 12288, 1, 1, 64, .{ .exact = true }).nax);
    try std.testing.expect(!rowsGeometry(5120, 129280, 1, 1, 64, .{ .exact = true }).nax);
    try std.testing.expectEqual(PROJECT_ROWS_MAX, rowsMax(.{ .exact = true }));
}
