const std = @import("std");
const mlx = @import("mlx_host").mlx;
const log = @import("mlx_host").log;
const io_util = @import("mlx_host").io_util;
const exl3 = @import("expert_exl3.zig");

var ubench_mute: bool = false;
var ubench_env: ?bool = null;
var pair_splits_force: ?u32 = null;
var swiglu_maxabs_env: ?bool = null;
var swiglu_maxabs_dumped: u32 = 0;

fn diagEnvValueOn(raw: ?[*:0]const u8) bool {
    const v = raw orelse return false;
    return v[0] != 0 and v[0] != '0';
}

fn pairSplitCount() u32 {
    if (pair_splits_force) |v| return v;
    return 2;
}

/// A pair-GEMV threadgroup prepares its own slice of x, so the split must cut
/// the hidden dim on a 128-wide Hadamard block boundary.
fn pairSplitCountFor(in_dim: c_int) u32 {
    const n = pairSplitCount();
    if (@rem(in_dim, @as(c_int, @intCast(n)) * 128) != 0) return 1;
    return n;
}

pub fn setPairSplitsForTest(n: ?u32) void {
    pair_splits_force = n;
}

/// Routes the decode GEMVs through the generic window reader, one tile per
/// threadgroup: the byte reference the lane funnel's layout is held to.
var funnel_off_for_test: bool = false;

/// Every rate below K4 reads its weights through funnels whose shifts follow from
/// n; K4 keeps its own packed branch, the same reads at a word-aligned rate.
fn funnelReads(n: u32) bool {
    return n != 64;
}

/// How a decode GEMV threadgroup (`FUNNEL`, `OTPT`) reads the bank. The lane
/// funnel carries two output tiles per threadgroup, sharing index math, inputs
/// and the prepare; the other readers carry one.
const GemvLayout = struct { funnel: bool, tiles: c_int };

fn gemvLayout(n: u32, out_tiles: c_int) GemvLayout {
    const funnel = !funnel_off_for_test and funnelReads(n);
    return .{ .funnel = funnel, .tiles = if (funnel and @rem(out_tiles, 2) == 0) 2 else 1 };
}

fn addGemvLayout(cfg: mlx.mlx_fast_metal_kernel_config, layout: GemvLayout) !void {
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "FUNNEL", @intFromBool(layout.funnel)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "OTPT", layout.tiles));
}

/// Slots one expert-grouped threadgroup leads: a wider group's accumulators spill.
const DECODE_GROUP_MEMBERS: c_int = 2;
/// The grouped kernels find an expert's slots with one 64-bit ballot mask.
const DECODE_GROUP_MAX_SLOTS: c_int = 64;
/// Routes a multi-row block through the single-slot kernels: the grouped kernels' byte reference.
var grouped_off_for_test: bool = false;

fn addGroup(cfg: mlx.mlx_fast_metal_kernel_config, group: c_int, nslots: c_int) !void {
    if (group == 0) return;
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "GROUP", group));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NSLOTS", nslots));
}

fn logGroupedEngaged(group: c_int) void {
    if (group == 0 or grouped_engaged) return;
    grouped_engaged = true;
    log.info("[exl3-decode] expert-grouped rows engaged group={d}\n", .{group});
}

fn exl3UbenchOn() bool {
    if (ubench_mute) return false;
    if (ubench_env) |v| return v;
    const v = diagEnvValueOn(std.c.getenv("SUSHI_EXL3_LAYER_UBENCH"));
    ubench_env = v;
    return v;
}

fn benchPrint(comptime fmt: []const u8, args: anytype) void {
    if (!exl3UbenchOn()) return;
    std.debug.print(fmt, args);
}

fn ubenchEval(a: mlx.mlx_array, name: []const u8) !void {
    if (!exl3UbenchOn()) return;
    const io = std.Io.Threaded.global_single_threaded.io();
    var sw = io_util.Stopwatch.init(io);
    try mlx.check(mlx.mlx_array_eval(a));
    const ns = sw.read();
    benchPrint("[exl3-ubench] {s} {d:.3} ms\n", .{ name, @as(f64, @floatFromInt(ns)) / 1e6 });
    log.info("[exl3-ubench] {s} {d:.3} ms\n", .{ name, @as(f64, @floatFromInt(ns)) / 1e6 });
}

fn swigluMaxabsOn() bool {
    if (swiglu_maxabs_env) |v| return v;
    const v = diagEnvValueOn(std.c.getenv("SUSHI_EXL3_SWIGLU_MAXABS"));
    swiglu_maxabs_env = v;
    return v;
}

fn dumpAbsMax(s: mlx.mlx_stream, a: mlx.mlx_array, name: []const u8) !void {
    var ab = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ab);
    try mlx.check(mlx.mlx_abs(&ab, a, s));
    var mx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mx);
    try mlx.check(mlx.mlx_max(&mx, ab, false, s));
    try mlx.check(mlx.mlx_array_eval(mx));
    var v: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&v, mx));
    log.info("[exl3-maxabs] {s} {d:.6}\n", .{ name, v });
}

pub const DECODE_ROWS_MAX: usize = 16;

/// The trellis rate a packed last dim carries: n halfwords per 256-weight tile,
/// K = n/16. Every kernel keys on n, never on an integer K.
fn packedRate(last: c_int) !exl3.Rate {
    if (last < 0) return error.BadExl3Shape;
    return exl3.kFromPackedDim(@intCast(last)) orelse error.BadExl3Shape;
}

pub fn usesPrefillArm(rows: usize) bool {
    return rows > DECODE_ROWS_MAX;
}

pub const UNION_EXPERTS: usize = 512;

pub fn unionUnique(eids: []const u32) u32 {
    var seen: [UNION_EXPERTS]u8 = @splat(0);
    var n: u32 = 0;
    for (eids) |e| {
        if (e >= UNION_EXPERTS) continue;
        if (seen[e] == 0) {
            seen[e] = 1;
            n += 1;
        }
    }
    return n;
}

pub fn unionMultiplicity(eids: []const u32, counts: []u32) u32 {
    var unique: u32 = 0;
    for (eids) |e| {
        if (e >= counts.len) continue;
        if (counts[e] == 0) unique += 1;
        counts[e] += 1;
    }
    return unique;
}

var union_hist_env: ?bool = null;

fn unionHistOn() bool {
    if (union_hist_env) |v| return v;
    const v = diagEnvValueOn(std.c.getenv("SUSHI_EXL3_UNION_HIST"));
    union_hist_env = v;
    return v;
}

pub fn dumpUnionHist(slots_u: mlx.mlx_array, S: usize, K: usize) !void {
    if (!unionHistOn()) return;
    if (S < 2 or K == 0) return;
    // The caller swallows this error, so the latch this eval may raise is ours
    // to drop: left standing it becomes the next decode tick's `MlxFailure`.
    const had_error = mlx.errorPending();
    mlx.check(mlx.mlx_array_eval(slots_u)) catch |e| {
        mlx.dropLatchedErrorUnless(had_error);
        return e;
    };
    const n = S * K;
    const ptr = mlx.mlx_array_data_uint32(slots_u) orelse return;
    const slice = ptr[0..n];
    var counts: [UNION_EXPERTS]u32 = @splat(0);
    const unique = unionMultiplicity(slice, &counts);
    var shared: u32 = 0;
    var max_m: u32 = 0;
    for (counts) |c| {
        if (c >= 2) shared += 1;
        if (c > max_m) max_m = c;
    }
    log.info("[exl3-union] S={d} K={d} assignments={d} unique={d} shared={d} max_mult={d}\n", .{ S, K, n, unique, shared, max_m });
    if (S <= DECODE_ROWS_MAX) return;
    // A prefill layer's per-expert row counts, the input `SUSHI_EXL3_GEMM_COUNTS` replays.
    var buf: [UNION_EXPERTS * 6]u8 = undefined;
    var used: usize = 0;
    for (counts) |c| used += (std.fmt.bufPrint(buf[used..], "{d},", .{c}) catch break).len;
    log.info("[exl3-counts] {s}\n", .{buf[0..used]});
}

/// Rows a routed-expert call sees: the product of every leading dim, never the
/// activation width.
pub fn rowsOfShape(shape: []const c_int) usize {
    if (shape.len < 2) return 1;
    var n: usize = 1;
    for (shape[0 .. shape.len - 1]) |d| n *= @intCast(@max(d, 0));
    return n;
}

pub const RunTable = struct {
    start: []u32,
    len: []u32,
    eid: []u32,
    n: u32,
};

pub fn buildRuns(alloc: std.mem.Allocator, experts: []const u32) !RunTable {
    if (experts.len == 0) return .{ .start = &.{}, .len = &.{}, .eid = &.{}, .n = 0 };
    var n: u32 = 1;
    var i: usize = 1;
    while (i < experts.len) : (i += 1) {
        if (experts[i] != experts[i - 1]) n += 1;
    }
    const start = try alloc.alloc(u32, n);
    const len = try alloc.alloc(u32, n);
    const eid = try alloc.alloc(u32, n);
    var r: u32 = 0;
    start[0] = 0;
    eid[0] = experts[0];
    i = 1;
    while (i < experts.len) : (i += 1) {
        if (experts[i] != experts[i - 1]) {
            len[r] = @intCast(i - start[r]);
            r += 1;
            start[r] = @intCast(i);
            eid[r] = experts[i];
        }
    }
    len[r] = @intCast(experts.len - start[r]);
    return .{ .start = start, .len = len, .eid = eid, .n = n };
}

const GEMV_SOURCE: [:0]const u8 =
    \\uint gid = uint(thread_position_in_grid.x);
    \\if (gid >= uint(ODIM)) return;
    \\constexpr uint TILE = 16u;
    \\constexpr uint N = uint(NHW);
    \\constexpr uint PACKED_HW = N;
    \\constexpr uint word_count = N / 2u;
    \\constexpr uint IN_TILES = uint(IDIM) / TILE;
    \\constexpr uint OUT_TILES = uint(ODIM) / TILE;
    \\const uint tn = gid / TILE;
    \\const uint local = gid % TILE;
    \\float acc = 0.0f;
    \\for (uint tk = 0u; tk < IN_TILES; tk++) {
    \\  const device ushort* tile = trellis + (tk * OUT_TILES + tn) * PACKED_HW;
    \\  ushort cw[256];
    \\  for (uint th = 0u; th < 128u; th++) {
    \\    const exl3_win wv = exl3_pair_window(th * 2u, N);
    \\    const uint a = uint(tile[wv.i0 * 2u]) | (uint(tile[wv.i0 * 2u + 1u]) << 16u);
    \\    const uint b = uint(tile[wv.i1 * 2u]) | (uint(tile[wv.i1 * 2u + 1u]) << 16u);
    \\    const ulong merged = (ulong(a) << 32) | ulong(b);
    \\    const uint funnel = uint(merged >> wv.sh);
    \\    cw[th * 2u] = ushort((funnel >> wv.fresh) & 0xffffu);
    \\    cw[th * 2u + 1u] = ushort(funnel & 0xffffu);
    \\  }
    \\  for (uint slot = 0u; slot < 256u; slot++) {
    \\    const uint lane = slot / 8u;
    \\    const uint s = slot % 8u;
    \\    const uint row0 = (lane & 3u) * 2u;
    \\    const uint col0 = lane >> 2u;
    \\    uint pos;
    \\    switch (s) {
    \\      case 0u: pos = row0 * 16u + col0; break;
    \\      case 1u: pos = (row0 + 1u) * 16u + col0; break;
    \\      case 2u: pos = (row0 + 8u) * 16u + col0; break;
    \\      case 3u: pos = (row0 + 9u) * 16u + col0; break;
    \\      case 4u: pos = row0 * 16u + col0 + 8u; break;
    \\      case 5u: pos = (row0 + 1u) * 16u + col0 + 8u; break;
    \\      case 6u: pos = (row0 + 8u) * 16u + col0 + 8u; break;
    \\      default: pos = (row0 + 9u) * 16u + col0 + 8u; break;
    \\    }
    \\    if ((pos % 16u) != local) continue;
    \\    const float w = exl3_decode1(uint(cw[slot]));
    \\    acc += float(x[tk * TILE + (pos / 16u)]) * w;
    \\  }
    \\}
    \\y[gid] = half(acc);
;

const GEMM_SORTED_SOURCE: [:0]const u8 =
    \\threadgroup float partial[2 * 256];
    \\uint ot = uint(threadgroup_position_in_grid.x);
    \\uint win = uint(threadgroup_position_in_grid.y);
    \\uint sg = uint(simdgroup_index_in_threadgroup);
    \\uint lane = uint(thread_index_in_simdgroup);
    \\uint lid = uint(thread_index_in_threadgroup);
    \\constexpr uint TILE = 16u;
    \\constexpr uint N = uint(NHW);
    \\constexpr uint PACKED_HW = N;
    \\constexpr uint PACKED_W = N / 2u;
    \\constexpr uint IT = uint(IDIM) / TILE;
    \\constexpr uint OT = uint(ODIM) / TILE;
    \\const uint start = wstarts[win];
    \\const uint n = wnlive[win];
    \\if (n == 0u || n > uint(WIN)) return;
    \\const uint pg = sg >> 1u;
    \\const uint th = sg & 1u;
    \\const uint first = th * 128u + lane * 4u;
    \\uint pos[4];
    \\uint irow[4];
    \\for (uint s = 0u; s < 4u; s++) {
    \\  const uint src = first + s;
    \\  const uint ln = src >> 3u;
    \\  const uint sl = src & 7u;
    \\  const uint prow = (ln & 3u) * 2u;
    \\  const uint col0 = ln >> 2u;
    \\  uint p;
    \\  switch (sl) {
    \\    case 0u: p = prow * 16u + col0; break;
    \\    case 1u: p = (prow + 1u) * 16u + col0; break;
    \\    case 2u: p = (prow + 8u) * 16u + col0; break;
    \\    case 3u: p = (prow + 9u) * 16u + col0; break;
    \\    case 4u: p = prow * 16u + col0 + 8u; break;
    \\    case 5u: p = (prow + 1u) * 16u + col0 + 8u; break;
    \\    case 6u: p = (prow + 8u) * 16u + col0 + 8u; break;
    \\    default: p = (prow + 9u) * 16u + col0 + 8u; break;
    \\  }
    \\  pos[s] = p;
    \\  irow[s] = p >> 4u;
    \\}
    \\uint row = start;
    \\const uint end = start + n;
    \\while (row < end) {
    \\const uint run0 = row;
    \\const uint eid = uint(eids[row]);
    \\uint run_end = row + 1u;
    \\while (run_end < end && uint(eids[run_end]) == eid) run_end++;
    \\const uint nlive = run_end - row;
    \\float4 acc[8][4];
    \\for (uint g = 0u; g < 8u; g++) {
    \\  acc[g][0] = float4(0.0f);
    \\  acc[g][1] = float4(0.0f);
    \\  acc[g][2] = float4(0.0f);
    \\  acc[g][3] = float4(0.0f);
    \\}
    \\  for (uint tk = pg; tk < IT; tk += 2u) {
    \\    const device uint* words = (const device uint*)(trellis + ((((size_t)eid * (size_t)IT + tk) * (size_t)OT + ot) * PACKED_HW));
    \\    const exl3_win wlo_v = exl3_pair_window(first, N);
    \\    const exl3_win whi_v = exl3_pair_window(first + 2u, N);
    \\    const ulong m0 = ((ulong)words[wlo_v.i0] << 32) | (ulong)words[wlo_v.i1];
    \\    const uint f0 = uint(m0 >> wlo_v.sh);
    \\    const ulong m1 = ((ulong)words[whi_v.i0] << 32) | (ulong)words[whi_v.i1];
    \\    const uint f1 = uint(m1 >> whi_v.sh);
    \\    const uint2 lo = uint2((f0 >> wlo_v.fresh) & 0xffffu, f0 & 0xffffu);
    \\    const uint2 hi = uint2((f1 >> whi_v.fresh) & 0xffffu, f1 & 0xffffu);
    \\    const float2 wlo = exl3_decode2(lo);
    \\    const float2 whi = exl3_decode2(hi);
    \\    const float4 wt = float4(wlo.x, wlo.y, whi.x, whi.y);
    \\    const uint ib = tk * TILE;
    \\    for (uint g = 0u; g < 8u; g++) {
    \\      if (g * 4u >= nlive) break;
    \\      const uint base_r = run0 + g * 4u;
    \\      for (uint s = 0u; s < 4u; s++) {
    \\        const size_t col = (size_t)(ib + irow[s]);
    \\        float4 a = float4(0.0f);
    \\        a.x = float(x[(size_t)base_r * (size_t)(IDIM) + col]);
    \\        if (g * 4u + 1u < nlive) a.y = float(x[(size_t)(base_r + 1u) * (size_t)(IDIM) + col]);
    \\        if (g * 4u + 2u < nlive) a.z = float(x[(size_t)(base_r + 2u) * (size_t)(IDIM) + col]);
    \\        if (g * 4u + 3u < nlive) a.w = float(x[(size_t)(base_r + 3u) * (size_t)(IDIM) + col]);
    \\        acc[g][s] = fma(a, float4(wt[s]), acc[g][s]);
    \\      }
    \\    }
    \\  }
    \\  for (uint g = 0u; g < 8u; g++) {
    \\    for (uint r = 0u; r < 4u; r++) {
    \\      const uint rr = g * 4u + r;
    \\      threadgroup_barrier(mem_flags::mem_threadgroup);
    \\      for (uint s = 0u; s < 4u; s++) {
    \\        partial[pg * 256u + pos[s]] = acc[g][s][r];
    \\      }
    \\      threadgroup_barrier(mem_flags::mem_threadgroup);
    \\      if (lid < 16u && rr < nlive) {
    \\        float sum = 0.0f;
    \\        for (uint pr = 0u; pr < 16u; pr++) {
    \\          const uint p = pr * 16u + lid;
    \\          sum += partial[p] + partial[256u + p];
    \\        }
    \\        y[(size_t)(run0 + rr) * (size_t)(ODIM) + ot * TILE + lid] = half(sum);
    \\      }
    \\    }
    \\  }
    \\  row = run_end;
    \\}
;
/// The simdgroup-matrix body's 8x8 fragment readers. A lane at fragment coordinate (fm, fn)
/// owns tile slot group g = 4*fm + fn/2: its pair j, slots (8g+2j, 8g+2j+1), is W^T block
/// (n half j>>1, k half j&1) at (fm, fn) and (fm, fn+1).
const GEMM_SIMDMAT_FRAGS: [:0]const u8 =
    \\#define SMAT_UNROLL _Pragma("clang loop unroll(full)")
    \\static inline half2 smat_pair(uint f, uint s0, uint s1) {
    \\  return exl3_pairh(uint2((f >> s0) & 0xffffu, (f >> s1) & 0xffffu));
    \\}
    \\// The group's weights 0..L come from one 32-bit funnel ending at weight L, the rest from
    \\// one ending at weight 7: L is the widest split whose first codeword still fits.
    \\static inline constexpr uint smat_split(uint N) {
    \\  return exl3_end(N, 7u) - exl3_end(N, 0u) <= 16u ? 7u : exl3_end(N, 5u) - exl3_end(N, 0u) <= 16u ? 5u : 3u;
    \\}
    \\template<uint N>
    \\static inline void smat_group(const device uint *words, uint g, thread half2 *p) {
    \\  if (N == 64u) {
    \\    const ulong m = ((ulong)words[(g + 31u) & 31u] << 32u) | (ulong)words[g];
    \\    SMAT_UNROLL for (uint j = 0u; j < 4u; j++) p[j] = smat_pair(uint(m >> (24u - 8u * j)), 4u, 0u);
    \\  } else if (exl3_end(N, 3u) - exl3_end(N, 0u) > 16u || N / 2u - exl3_end(N, 4u) > 16u) {
    \\    constexpr uint W = N / 2u;
    \\    const ulong lo = exl3_window(words, W * g + exl3_end(N, 3u), W, true);
    \\    const ulong hi = exl3_window(words, W * g + W, W, true);
    \\    SMAT_UNROLL for (uint j = 0u; j < 4u; j++) {
    \\      const ulong bits = j < 2u ? lo : hi;
    \\      const uint end = j < 2u ? exl3_end(N, 3u) : W;
    \\      p[j] = exl3_pairh(uint2(uint(bits >> (end - exl3_end(N, 2u * j))) & 0xffffu, uint(bits >> (end - exl3_end(N, 2u * j + 1u))) & 0xffffu));
    \\    }
    \\  } else {
    \\    constexpr uint W = N / 2u;
    \\    constexpr uint L = smat_split(N);
    \\    const uint lo = uint(exl3_window(words, W * g + exl3_end(N, L), W, false));
    \\    const uint hi = uint(exl3_window(words, W * g + W, W, false));
    \\    constexpr uint E = exl3_end(N, L);
    \\    p[0] = smat_pair(lo, E - exl3_end(N, 0u), E - exl3_end(N, 1u));
    \\    p[1] = smat_pair(lo, E - exl3_end(N, 2u), E - exl3_end(N, 3u));
    \\    p[2] = L >= 5u ? smat_pair(lo, E - exl3_end(N, 4u), E - exl3_end(N, 5u)) : smat_pair(hi, W - exl3_end(N, 4u), W - exl3_end(N, 5u));
    \\    p[3] = L == 7u ? smat_pair(lo, E - exl3_end(N, 6u), 0u) : smat_pair(hi, W - exl3_end(N, 6u), 0u);
    \\  }
    \\}
;
/// A data-dependent block count spills the accumulators, so use compile-time WIN.
/// Short runs repeat their last row; padded rows are never stored.
const GEMM_SIMDMAT_SOURCE: [:0]const u8 =
    \\uint win = uint(threadgroup_position_in_grid.y);
    \\uint sg = uint(simdgroup_index_in_threadgroup);
    \\ushort lane = ushort(thread_index_in_simdgroup);
    \\constexpr uint TILE = 16u;
    \\constexpr uint N = uint(NHW);
    \\constexpr uint PACKED_W = N / 2u;
    \\constexpr uint IT = uint(IDIM) / TILE;
    \\constexpr uint OT = uint(ODIM) / TILE;
    \\const uint start = wstarts[win];
    \\const uint n = wnlive[win];
    \\if (n == 0u || n > uint(WIN)) return;
    \\const uint col0 = uint(threadgroup_position_in_grid.x) * 128u + sg * 32u;
    \\const ushort qid = lane >> 2;
    \\const ushort fm = (qid & 4) + ((lane >> 1) & 3);
    \\const ushort fn = (qid & 2) * 2 + (lane & 1) * 2;
    \\const uint g = uint(fm) * 4u + uint(fn >> 1);
    \\constexpr uint MB = (uint(WIN) + 7u) / 8u;
    \\uint row = start;
    \\const uint end = start + n;
    \\while (row < end) {
    \\const uint run0 = row;
    \\const uint eid = uint(eids[row]);
    \\uint run_end = row + 1u;
    \\while (run_end < end && uint(eids[run_end]) == eid) run_end++;
    \\const uint nlive = run_end - row;
    \\size_t xo[4][2];
    \\SMAT_UNROLL for (uint mb = 0u; mb < MB; mb++) {
    \\  xo[mb][0] = (size_t)(run0 + min(mb * 8u + uint(fn), nlive - 1u)) * (size_t)(IDIM);
    \\  xo[mb][1] = (size_t)(run0 + min(mb * 8u + uint(fn) + 1u, nlive - 1u)) * (size_t)(IDIM);
    \\}
    \\simdgroup_matrix<float, 8, 8> acc[4][4];
    \\SMAT_UNROLL for (uint nb = 0u; nb < 4u; nb++) {
    \\  SMAT_UNROLL for (uint mb = 0u; mb < MB; mb++) acc[nb][mb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    \\}
    \\const device uint *tiles = (const device uint *)(trellis + (size_t)eid * (size_t)IT * (size_t)OT * (size_t)N) + (col0 / TILE) * PACKED_W;
    \\for (uint tk = 0u; tk < IT; tk++) {
    \\  const uint kc = tk * TILE + uint(fm);
    \\  simdgroup_matrix<half, 8, 8> b[2][4];
    \\  SMAT_UNROLL for (uint mb = 0u; mb < MB; mb++) {
    \\    SMAT_UNROLL for (uint kb = 0u; kb < 2u; kb++) {
    \\      b[kb][mb].thread_elements()[0] = x[xo[mb][0] + kc + kb * 8u];
    \\      b[kb][mb].thread_elements()[1] = x[xo[mb][1] + kc + kb * 8u];
    \\    }
    \\  }
    \\  const device uint *words = tiles + (size_t)tk * (size_t)OT * (size_t)PACKED_W;
    \\  simdgroup_matrix<half, 8, 8> a[4][2];
    \\  SMAT_UNROLL for (uint t = 0u; t < 2u; t++) {
    \\    half2 p[4];
    \\    smat_group<N>(words + t * PACKED_W, g, p);
    \\    SMAT_UNROLL for (uint j = 0u; j < 4u; j++) {
    \\      a[2u * t + (j >> 1u)][j & 1u].thread_elements()[0] = p[j].x;
    \\      a[2u * t + (j >> 1u)][j & 1u].thread_elements()[1] = p[j].y;
    \\    }
    \\  }
    \\  SMAT_UNROLL for (uint mb = 0u; mb < MB; mb++) {
    \\    SMAT_UNROLL for (uint kb = 0u; kb < 2u; kb++) {
    \\      SMAT_UNROLL for (uint nb = 0u; nb < 4u; nb++) simdgroup_multiply_accumulate(acc[nb][mb], a[nb][kb], b[kb][mb], acc[nb][mb]);
    \\    }
    \\  }
    \\}
    \\SMAT_UNROLL for (uint nb = 0u; nb < 4u; nb++) {
    \\  const size_t oc = (size_t)(col0 + nb * 8u + uint(fm));
    \\  SMAT_UNROLL for (uint mb = 0u; mb < MB; mb++) {
    \\    const uint m = mb * 8u + uint(fn);
    \\    if (m < nlive) y[(size_t)(run0 + m) * (size_t)(ODIM) + oc] = half(acc[nb][mb].thread_elements()[0]);
    \\    if (m + 1u < nlive) y[(size_t)(run0 + m + 1u) * (size_t)(ODIM) + oc] = half(acc[nb][mb].thread_elements()[1]);
    \\  }
    \\}
    \\row = run_end;
    \\}
;
const GEMM_NAX_INCLUDES: [:0]const u8 =
    \\#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
    \\using namespace metal;
    \\using namespace mpp::tensor_ops;
    \\
;
const GEMM_NAX_FRAGS: [:0]const u8 =
    \\using nfrag = vec<half, 8>;
    \\static inline nfrag nax_wfrag(const device uint *words, uint lane) {
    \\  const uint source0 = (lane & 16u) + ((lane & 7u) << 1u);
    \\  const uint2 current = *(const device uint2 *)(words + source0);
    \\  const uint previous = words[(source0 + 31u) & 31u];
    \\  const uint slot = ((lane >> 3u) & 1u) * 2u;
    \\  const ulong w0 = ((ulong)previous << 32) | (ulong)current.x;
    \\  const ulong w1 = ((ulong)current.x << 32) | (ulong)current.y;
    \\  const uint s0 = 28u - slot * 4u;
    \\  const uint s1 = 28u - (slot + 4u) * 4u;
    \\  const half2 p00 = exl3_pairh(uint2(uint(w0 >> s0) & 0xffffu, uint(w0 >> (s0 - 4u)) & 0xffffu));
    \\  const half2 p01 = exl3_pairh(uint2(uint(w1 >> s0) & 0xffffu, uint(w1 >> (s0 - 4u)) & 0xffffu));
    \\  const half2 p10 = exl3_pairh(uint2(uint(w0 >> s1) & 0xffffu, uint(w0 >> (s1 - 4u)) & 0xffffu));
    \\  const half2 p11 = exl3_pairh(uint2(uint(w1 >> s1) & 0xffffu, uint(w1 >> (s1 - 4u)) & 0xffffu));
    \\  return nfrag(p00.x, p00.y, p01.x, p01.y, p10.x, p10.y, p11.x, p11.y);
    \\}
    \\// A lane's weights are 16m + 2b + {0, 1, 4, 5} and the same + 8: one funnel per quad, ending at
    \\// its last weight. 16 weights span N whole bits, so the shifts depend on b alone.
    \\static inline constexpr uint nax_end(uint N, uint b, uint j) { return ((2u * b + j + 1u) * N) >> 4u; }
    \\// Past 32 bits (n > 50) a quad's funnel reads 64 bits from three words.
    \\static inline constexpr bool nax_quads_fit(uint N) {
    \\  return nax_end(N, 0u, 5u) - nax_end(N, 0u, 0u) <= 16u && nax_end(N, 1u, 5u) - nax_end(N, 1u, 0u) <= 16u;
    \\}
    \\static inline uint nax_sh(uint N, uint b, uint j) {
    \\  const uint s0 = nax_end(N, 0u, 5u) - nax_end(N, 0u, j);
    \\  return s0 + b * (nax_end(N, 1u, 5u) - nax_end(N, 1u, j) - s0);
    \\}
    \\template<typename F>
    \\static inline nfrag nax_quads(F lo, F hi, uint s0, uint s1, uint s4) {
    \\  const half2 p0 = exl3_pairh(uint2(uint(lo >> s0) & 0xffffu, uint(lo >> s1) & 0xffffu));
    \\  const half2 p1 = exl3_pairh(uint2(uint(lo >> s4) & 0xffffu, uint(lo) & 0xffffu));
    \\  const half2 p2 = exl3_pairh(uint2(uint(hi >> s0) & 0xffffu, uint(hi >> s1) & 0xffffu));
    \\  const half2 p3 = exl3_pairh(uint2(uint(hi >> s4) & 0xffffu, uint(hi) & 0xffffu));
    \\  return nfrag(p0.x, p0.y, p2.x, p2.y, p1.x, p1.y, p3.x, p3.y);
    \\}
    \\template<uint N>
    \\static inline nfrag nax_wfrag_k(const device uint *words, uint lane) {
    \\  if (N == 64u) return nax_wfrag(words, lane);
    \\  constexpr uint W = N / 2u;
    \\  const uint end = 8u * N * (lane >> 4u) + N * (lane & 7u) + (nax_end(N, 1u, 5u) - nax_end(N, 0u, 5u)) * ((lane >> 3u) & 1u) + nax_end(N, 0u, 5u);
    \\  const uint b = (lane >> 3u) & 1u;
    \\  const uint s0 = nax_sh(N, b, 0u);
    \\  const uint s1 = nax_sh(N, b, 1u);
    \\  const uint s4 = nax_sh(N, b, 4u);
    \\  if (nax_quads_fit(N)) return nax_quads(uint(exl3_window(words, end, W, false)), uint(exl3_window(words, end + W, W, false)), s0, s1, s4);
    \\  return nax_quads(exl3_window(words, end, W, true), exl3_window(words, end + W, W, true), s0, s1, s4);
    \\}
    \\static inline short2 nax_origin(uint lane) {
    \\  const short qid = short(lane >> 2u);
    \\  return short2(short(((qid & 2) | short(lane & 1u)) * 4), short((qid & 4) | short((lane >> 1u) & 3u)));
    \\}
    \\template <typename D>
    \\static inline void nax_zero(thread D &dst) {
    \\  for (uint s = 0u; s < dst.get_capacity(); s++) dst[s] = 0.0f;
    \\}
    \\// `rem` is the block's live row count: the 16-row tile is padded with zeros
    \\// so a short run still runs the full mma. The half4 read is 8-byte aligned
    \\// because IDIM is a multiple of 16 and origin.x a multiple of 4.
    \\template <typename L, typename X>
    \\static inline void nax_left(thread L &left, const device X *x, uint row, uint idim, uint kbase, short2 origin, uint rem) {
    \\  using x4 = vec<X, 4>;
    \\  const size_t r0 = (size_t)(row + uint(origin.y)) * (size_t)idim + kbase + uint(origin.x);
    \\  const size_t r1 = r0 + 8u * (size_t)idim;
    \\  const x4 v0 = (uint(origin.y) < rem) ? *(const device x4 *)(x + r0) : x4(0);
    \\  const x4 v1 = (uint(origin.y) + 8u < rem) ? *(const device x4 *)(x + r1) : x4(0);
    \\  for (uint c = 0u; c < 4u; c++) {
    \\    left[c] = v0[c];
    \\    left[4 + c] = v1[c];
    \\  }
    \\}
    \\// Element offset of row `row`, clamped to the last input row, at the lane's column.
    \\static inline size_t nax_row(uint row, uint last, uint idim, short2 origin) {
    \\  return (size_t)min(row, last) * (size_t)idim + uint(origin.x);
    \\}
    \\template <typename L, typename X>
    \\static inline void nax_left_rows(thread L &left, const device X *p0, const device X *p1) {
    \\  using x4 = vec<X, 4>;
    \\  const x4 v0 = *(const device x4 *)p0;
    \\  const x4 v1 = *(const device x4 *)p1;
    \\  for (uint c = 0u; c < 4u; c++) {
    \\    left[c] = v0[c];
    \\    left[4 + c] = v1[c];
    \\  }
    \\}
    \\template <typename D, typename Y>
    \\static inline void nax_store(const thread D &dst, device Y *y, uint row, uint odim, uint obase, short2 origin, uint rem) {
    \\  const size_t o0 = (size_t)(row + uint(origin.y)) * (size_t)odim;
    \\  const size_t o1 = o0 + 8u * (size_t)odim;
    \\  const bool l0 = uint(origin.y) < rem;
    \\  const bool l1 = uint(origin.y) + 8u < rem;
    \\  for (uint ct = 0u; ct < 2u; ct++) {
    \\    for (uint c = 0u; c < 4u; c++) {
    \\      const uint col = obase + ct * 16u + uint(origin.x) + c;
    \\      if (col >= odim) continue;
    \\      if (l0) y[o0 + col] = Y(dst[ct * 8u + c]);
    \\      if (l1) y[o1 + col] = Y(dst[ct * 8u + 4u + c]);
    \\    }
    \\  }
    \\}
;

/// The branch-guarded NAX body GEMM_NAX_SOURCE must reproduce byte for byte (tests only).
const GEMM_NAX_REFERENCE_SOURCE: [:0]const u8 =
    \\uint win = uint(threadgroup_position_in_grid.y);
    \\uint sg = uint(simdgroup_index_in_threadgroup);
    \\uint lane = uint(thread_index_in_simdgroup);
    \\constexpr uint TILE = 16u;
    \\constexpr uint N = uint(NHW);
    \\constexpr uint PACKED_HW = N;
    \\constexpr uint PACKED_W = N / 2u;
    \\constexpr uint IT = uint(IDIM) / TILE;
    \\constexpr uint OT = uint(ODIM) / TILE;
    \\const uint start = wstarts[win];
    \\const uint n = wnlive[win];
    \\if (n == 0u || n > uint(WIN)) return;
    \\constexpr auto desc = matmul2d_descriptor(16, 32, 16, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
    \\matmul2d<desc, execution_simdgroup> op;
    \\auto left = op.get_left_input_cooperative_tensor<half, half, float>();
    \\auto right = op.get_right_input_cooperative_tensor<half, half, float>();
    \\auto destination = op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>, metal::remove_addrspace_t<decltype(right)>, float>();
    \\auto dest_hi = op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>, metal::remove_addrspace_t<decltype(right)>, float>();
    \\const short2 origin = nax_origin(lane);
    \\const uint output_base = uint(threadgroup_position_in_grid.x) * 128u + sg * 32u;
    \\uint row = start;
    \\const uint end = start + n;
    \\while (row < end) {
    \\const uint run0 = row;
    \\const uint eid = uint(eids[row]);
    \\uint run_end = row + 1u;
    \\while (run_end < end && uint(eids[run_end]) == eid) run_end++;
    \\const uint nlive = run_end - row;
    \\const uint n_hi = (nlive > TILE) ? (nlive - TILE) : 0u;
    \\nax_zero(destination);
    \\if (n_hi > 0u) nax_zero(dest_hi);
    \\const device uint *trellis_e = (const device uint *)(trellis + ((size_t)eid * (size_t)IT * (size_t)OT) * PACKED_HW);
    \\for (uint tk = 0u; tk < IT; tk++) {
    \\  const uint kbase = tk * TILE;
    \\  const device uint *words0 = trellis_e + ((size_t)tk * (size_t)OT + output_base / TILE) * PACKED_W;
    \\  const nfrag w0 = nax_wfrag_k<N>(words0, lane);
    \\  const nfrag w1 = nax_wfrag_k<N>(words0 + PACKED_W, lane);
    \\  for (short s = 0; s < 8; s++) {
    \\    right[s] = w0[s];
    \\    right[8 + s] = w1[s];
    \\  }
    \\  nax_left(left, x, run0, uint(IDIM), kbase, origin, nlive);
    \\  op.run(left, right, destination);
    \\  if (n_hi > 0u) {
    \\    nax_left(left, x, run0 + TILE, uint(IDIM), kbase, origin, n_hi);
    \\    op.run(left, right, dest_hi);
    \\  }
    \\}
    \\nax_store(destination, y, run0, uint(ODIM), output_base, origin, nlive);
    \\if (n_hi > 0u) nax_store(dest_hi, y, run0 + TILE, uint(ODIM), output_base, origin, n_hi);
    \\row = run_end;
    \\}
;

/// The NAX body. A lane's x rows are clamped into the input, so a padded row reads a live
/// neighbour whose product is never stored and the loads carry no bounds branch; the k loop is
/// unswitched on the window's second 16-row block. Bytes equal GEMM_NAX_REFERENCE_SOURCE's.
const GEMM_NAX_SOURCE: [:0]const u8 = blk: {
    const decode_lo =
        \\  const nfrag w0 = nax_wfrag_k<N>(words, lane);
        \\  const nfrag w1 = nax_wfrag_k<N>(words + PACKED_W, lane);
        \\  for (short s = 0; s < 8; s++) {
        \\    right[s] = w0[s];
        \\    right[8 + s] = w1[s];
        \\  }
        \\  nax_left_rows(left, xl0 + tk * TILE, xl1 + tk * TILE);
        \\  op.run(left, right, destination);
        \\
    ;
    const open_loop =
        \\#pragma clang loop unroll_count(2)
        \\for (uint tk = 0u; tk < IT; tk++) {
        \\  const device uint *words = wp;
        \\  wp += (size_t)OT * PACKED_W;
        \\
    ;
    break :blk
    \\uint win = uint(threadgroup_position_in_grid.y);
    \\uint sg = uint(simdgroup_index_in_threadgroup);
    \\uint lane = uint(thread_index_in_simdgroup);
    \\constexpr uint TILE = 16u;
    \\constexpr uint N = uint(NHW);
    \\constexpr uint PACKED_HW = N;
    \\constexpr uint PACKED_W = N / 2u;
    \\constexpr uint IT = uint(IDIM) / TILE;
    \\constexpr uint OT = uint(ODIM) / TILE;
    \\const uint start = wstarts[win];
    \\const uint n = wnlive[win];
    \\if (n == 0u || n > uint(WIN)) return;
    \\constexpr auto desc = matmul2d_descriptor(16, 32, 16, false, true, true, matmul2d_descriptor::mode::multiply_accumulate);
    \\matmul2d<desc, execution_simdgroup> op;
    \\auto left = op.get_left_input_cooperative_tensor<half, half, float>();
    \\auto right = op.get_right_input_cooperative_tensor<half, half, float>();
    \\auto destination = op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>, metal::remove_addrspace_t<decltype(right)>, float>();
    \\auto dest_hi = op.get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(left)>, metal::remove_addrspace_t<decltype(right)>, float>();
    \\const short2 origin = nax_origin(lane);
    \\const uint output_base = uint(threadgroup_position_in_grid.x) * 128u + sg * 32u;
    \\const uint last_row = uint(x_shape[0]) - 1u;
    \\uint row = start;
    \\const uint end = start + n;
    \\while (row < end) {
    \\const uint run0 = row;
    \\const uint eid = uint(eids[row]);
    \\uint run_end = row + 1u;
    \\while (run_end < end && uint(eids[run_end]) == eid) run_end++;
    \\const uint nlive = run_end - row;
    \\const uint n_hi = (nlive > TILE) ? (nlive - TILE) : 0u;
    \\nax_zero(destination);
    \\if (n_hi > 0u) nax_zero(dest_hi);
    \\const device uint *trellis_e = (const device uint *)(trellis + ((size_t)eid * (size_t)IT * (size_t)OT) * PACKED_HW);
    \\const device auto *xl0 = x + nax_row(run0 + uint(origin.y), last_row, uint(IDIM), origin);
    \\const device auto *xl1 = x + nax_row(run0 + uint(origin.y) + 8u, last_row, uint(IDIM), origin);
    \\const device auto *xh0 = x + nax_row(run0 + TILE + uint(origin.y), last_row, uint(IDIM), origin);
    \\const device auto *xh1 = x + nax_row(run0 + TILE + uint(origin.y) + 8u, last_row, uint(IDIM), origin);
    \\const device uint *wp = trellis_e + (size_t)(output_base / TILE) * PACKED_W;
    \\if (n_hi > 0u) {
    \\
    ++ open_loop ++ decode_lo ++
        \\  nax_left_rows(left, xh0 + tk * TILE, xh1 + tk * TILE);
        \\  op.run(left, right, dest_hi);
        \\}
        \\} else {
        \\
    ++ open_loop ++ decode_lo ++
        \\}
        \\}
        \\nax_store(destination, y, run0, uint(ODIM), output_base, origin, nlive);
        \\if (n_hi > 0u) nax_store(dest_hi, y, run0 + TILE, uint(ODIM), output_base, origin, n_hi);
        \\row = run_end;
        \\}
    ;
};

/// Test and prefill-ubench seam: dispatch GEMM_NAX_REFERENCE_SOURCE instead.
pub var nax_reference_override: bool = false;
var gemm_nax_ref_kernel: KernelSlots = no_kernels;

const TOKEN_PREPARE_SOURCE: [:0]const u8 =
    \\uint block = uint(threadgroup_position_in_grid.x);
    \\uint slot = uint(threadgroup_position_in_grid.y);
    \\ushort lane = thread_index_in_simdgroup;
    \\const uint eid = uint(slots[slot]);
    \\const uint orig = uint(order[slot]);
    \\const uint row = orig / uint(TOPK);
    \\const uint base = block * 128u;
    \\const size_t xb = (size_t)row * (size_t)(IDIM) + base;
    \\const size_t yb = (size_t)slot * (size_t)(IDIM) + base;
    \\const size_t sb = (size_t)eid * (size_t)(IDIM) + base;
    \\float4 v = float4(
    \\  float(x[xb + lane]) * float(suh[sb + lane]),
    \\  float(x[xb + lane + 32u]) * float(suh[sb + lane + 32u]),
    \\  float(x[xb + lane + 64u]) * float(suh[sb + lane + 64u]),
    \\  float(x[xb + lane + 96u]) * float(suh[sb + lane + 96u]));
    \\for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
    \\  const float p0 = simd_shuffle_xor(v.x, bit);
    \\  const float p1 = simd_shuffle_xor(v.y, bit);
    \\  const float p2 = simd_shuffle_xor(v.z, bit);
    \\  const float p3 = simd_shuffle_xor(v.w, bit);
    \\  const bool lower = (lane & bit) == 0u;
    \\  v.x = lower ? v.x + p0 : p0 - v.x;
    \\  v.y = lower ? v.y + p1 : p1 - v.y;
    \\  v.z = lower ? v.z + p2 : p2 - v.z;
    \\  v.w = lower ? v.w + p3 : p3 - v.w;
    \\}
    \\const float s0 = v.x + v.y;
    \\const float s1 = v.x - v.y;
    \\const float s2 = v.z + v.w;
    \\const float s3 = v.z - v.w;
    \\const float sc = 0.08838834764831845f;
    \\y[yb + lane] = half((s0 + s2) * sc);
    \\y[yb + lane + 32u] = half((s1 + s3) * sc);
    \\y[yb + lane + 64u] = half((s0 - s2) * sc);
    \\y[yb + lane + 96u] = half((s1 - s3) * sc);
;

const TOKEN_PAIR_PREPARE_SOURCE: [:0]const u8 =
    \\uint block = uint(threadgroup_position_in_grid.x);
    \\uint slot = uint(threadgroup_position_in_grid.y);
    \\ushort lane = thread_index_in_simdgroup;
    \\const uint eid = uint(slots[slot]);
    \\const uint orig = uint(order[slot]);
    \\const uint row = orig / uint(TOPK);
    \\const uint base = block * 128u;
    \\const size_t xb = (size_t)row * (size_t)(IDIM) + base;
    \\const size_t yb = (size_t)slot * (size_t)(IDIM) + base;
    \\const size_t sb = (size_t)eid * (size_t)(IDIM) + base;
    \\const float x0 = float(x[xb + lane]);
    \\const float x1 = float(x[xb + lane + 32u]);
    \\const float x2 = float(x[xb + lane + 64u]);
    \\const float x3 = float(x[xb + lane + 96u]);
    \\float4 v = float4(
    \\  x0 * float(suhg[sb + lane]),
    \\  x1 * float(suhg[sb + lane + 32u]),
    \\  x2 * float(suhg[sb + lane + 64u]),
    \\  x3 * float(suhg[sb + lane + 96u]));
    \\for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
    \\  const float p0 = simd_shuffle_xor(v.x, bit);
    \\  const float p1 = simd_shuffle_xor(v.y, bit);
    \\  const float p2 = simd_shuffle_xor(v.z, bit);
    \\  const float p3 = simd_shuffle_xor(v.w, bit);
    \\  const bool lower = (lane & bit) == 0u;
    \\  v.x = lower ? v.x + p0 : p0 - v.x;
    \\  v.y = lower ? v.y + p1 : p1 - v.y;
    \\  v.z = lower ? v.z + p2 : p2 - v.z;
    \\  v.w = lower ? v.w + p3 : p3 - v.w;
    \\}
    \\const float sc = 0.08838834764831845f;
    \\float s0 = v.x + v.y;
    \\float s1 = v.x - v.y;
    \\float s2 = v.z + v.w;
    \\float s3 = v.z - v.w;
    \\yg[yb + lane] = half((s0 + s2) * sc);
    \\yg[yb + lane + 32u] = half((s1 + s3) * sc);
    \\yg[yb + lane + 64u] = half((s0 - s2) * sc);
    \\yg[yb + lane + 96u] = half((s1 - s3) * sc);
    \\v = float4(
    \\  x0 * float(suhu[sb + lane]),
    \\  x1 * float(suhu[sb + lane + 32u]),
    \\  x2 * float(suhu[sb + lane + 64u]),
    \\  x3 * float(suhu[sb + lane + 96u]));
    \\for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
    \\  const float p0 = simd_shuffle_xor(v.x, bit);
    \\  const float p1 = simd_shuffle_xor(v.y, bit);
    \\  const float p2 = simd_shuffle_xor(v.z, bit);
    \\  const float p3 = simd_shuffle_xor(v.w, bit);
    \\  const bool lower = (lane & bit) == 0u;
    \\  v.x = lower ? v.x + p0 : p0 - v.x;
    \\  v.y = lower ? v.y + p1 : p1 - v.y;
    \\  v.z = lower ? v.z + p2 : p2 - v.z;
    \\  v.w = lower ? v.w + p3 : p3 - v.w;
    \\}
    \\s0 = v.x + v.y;
    \\s1 = v.x - v.y;
    \\s2 = v.z + v.w;
    \\s3 = v.z - v.w;
    \\yu[yb + lane] = half((s0 + s2) * sc);
    \\yu[yb + lane + 32u] = half((s1 + s3) * sc);
    \\yu[yb + lane + 64u] = half((s0 - s2) * sc);
    \\yu[yb + lane + 96u] = half((s1 - s3) * sc);
;

const TOKEN_SCATTER_SOURCE: [:0]const u8 =
    \\uint col = uint(thread_position_in_grid.x);
    \\uint slot = uint(thread_position_in_grid.y);
    \\if (col >= uint(DIM)) return;
    \\const uint orig = uint(order[slot]);
    \\y[(size_t)orig * (size_t)(DIM) + col] = x[(size_t)slot * (size_t)(DIM) + col];
;

const TOKEN_REDUCE_SOURCE: [:0]const u8 =
    \\uint col = uint(thread_position_in_grid.x);
    \\uint row = uint(thread_position_in_grid.y);
    \\if (col >= uint(ODIM)) return;
    \\half acc = half(0.0f);
    \\for (uint k = 0u; k < uint(TOPK); k++) {
    \\  const uint orig = row * uint(TOPK) + k;
    \\  const uint si = uint(inv[orig]);
    \\  const half p = half(float(d[(size_t)si * (size_t)(ODIM) + col]) * float(half(sc[orig])));
    \\  acc += p;
    \\}
    \\y[(size_t)row * (size_t)(ODIM) + col] = acc;
;

const PREPARE_SOURCE: [:0]const u8 =
    \\uint block = uint(threadgroup_position_in_grid.x);
    \\uint slot = uint(threadgroup_position_in_grid.y);
    \\ushort lane = thread_index_in_simdgroup;
    \\const uint eid = uint(slots[slot]);
    \\const uint base = block * 128u;
    \\const size_t xb = (size_t)slot * (size_t)(IDIM) + base;
    \\const size_t sb = (size_t)eid * (size_t)(IDIM) + base;
    \\float4 v = float4(
    \\  float(x[xb + lane]) * float(suh[sb + lane]),
    \\  float(x[xb + lane + 32u]) * float(suh[sb + lane + 32u]),
    \\  float(x[xb + lane + 64u]) * float(suh[sb + lane + 64u]),
    \\  float(x[xb + lane + 96u]) * float(suh[sb + lane + 96u]));
    \\for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
    \\  const float p0 = simd_shuffle_xor(v.x, bit);
    \\  const float p1 = simd_shuffle_xor(v.y, bit);
    \\  const float p2 = simd_shuffle_xor(v.z, bit);
    \\  const float p3 = simd_shuffle_xor(v.w, bit);
    \\  const bool lower = (lane & bit) == 0u;
    \\  v.x = lower ? v.x + p0 : p0 - v.x;
    \\  v.y = lower ? v.y + p1 : p1 - v.y;
    \\  v.z = lower ? v.z + p2 : p2 - v.z;
    \\  v.w = lower ? v.w + p3 : p3 - v.w;
    \\}
    \\const float s0 = v.x + v.y;
    \\const float s1 = v.x - v.y;
    \\const float s2 = v.z + v.w;
    \\const float s3 = v.z - v.w;
    \\const float sc = 0.08838834764831845f;
    \\y[xb + lane] = half((s0 + s2) * sc);
    \\y[xb + lane + 32u] = half((s1 + s3) * sc);
    \\y[xb + lane + 64u] = half((s0 - s2) * sc);
    \\y[xb + lane + 96u] = half((s1 - s3) * sc);
;

const FINISH_SOURCE: [:0]const u8 =
    \\uint block = uint(threadgroup_position_in_grid.x);
    \\uint slot = uint(threadgroup_position_in_grid.y);
    \\ushort lane = thread_index_in_simdgroup;
    \\const uint eid = uint(slots[slot]);
    \\const uint base = block * 128u;
    \\const size_t xb = (size_t)slot * (size_t)(ODIM) + base;
    \\const size_t sb = (size_t)eid * (size_t)(ODIM) + base;
    \\float4 v = float4(
    \\  float(inner[xb + lane]),
    \\  float(inner[xb + lane + 32u]),
    \\  float(inner[xb + lane + 64u]),
    \\  float(inner[xb + lane + 96u]));
    \\for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
    \\  const float p0 = simd_shuffle_xor(v.x, bit);
    \\  const float p1 = simd_shuffle_xor(v.y, bit);
    \\  const float p2 = simd_shuffle_xor(v.z, bit);
    \\  const float p3 = simd_shuffle_xor(v.w, bit);
    \\  const bool lower = (lane & bit) == 0u;
    \\  v.x = lower ? v.x + p0 : p0 - v.x;
    \\  v.y = lower ? v.y + p1 : p1 - v.y;
    \\  v.z = lower ? v.z + p2 : p2 - v.z;
    \\  v.w = lower ? v.w + p3 : p3 - v.w;
    \\}
    \\const float s0 = v.x + v.y;
    \\const float s1 = v.x - v.y;
    \\const float s2 = v.z + v.w;
    \\const float s3 = v.z - v.w;
    \\const float sc = 0.08838834764831845f;
    \\y[xb + lane] = half((s0 + s2) * sc * float(svh[sb + lane]));
    \\y[xb + lane + 32u] = half((s1 + s3) * sc * float(svh[sb + lane + 32u]));
    \\y[xb + lane + 64u] = half((s0 - s2) * sc * float(svh[sb + lane + 64u]));
    \\y[xb + lane + 96u] = half((s1 - s3) * sc * float(svh[sb + lane + 96u]));
;

var gemm_sorted_kernel: KernelSlots = no_kernels;
var gemm_simdmat_kernel: KernelSlots = no_kernels;
/// Test seam: false sends a non-NAX GEMM to the scalar SIMD body.
var gemm_simdmat_force: ?bool = null;
var gemm_simdmat_engaged: bool = false;
var prepare_kernel: ?mlx.mlx_fast_metal_kernel = null;
var finish_kernel: ?mlx.mlx_fast_metal_kernel = null;
var gemv_kernel: KernelSlots = no_kernels;
var gemv_engaged: bool = false;

/// How the next dispatch decodes: the pack's codebook and codeword window.
/// A weight kernel is built per (codebook, window), so the slots below are
/// indexed by both; the model's own pair is asserted at each dispatch entry,
/// since several packs can be resident.
var active_decode: exl3.Decode = .mul1;
pub fn setDecodeParams(dec: exl3.Decode) void {
    active_decode = dec;
}
const N_CB = exl3.Codebook.count;
const N_WIN = exl3.Window.count;
const KernelSlots = [N_CB][N_WIN]?mlx.mlx_fast_metal_kernel;
const no_kernels: KernelSlots = @splat(@splat(null));
fn cbIndex(comptime cb: exl3.Codebook) usize {
    return @backingInt(cb);
}
fn winSuffix(comptime win: exl3.Window) [:0]const u8 {
    return comptime if (win == .w16) "" else std.fmt.comptimePrint("_w{d}", .{win.bits()});
}
fn cbSuffix(comptime cb: exl3.Codebook) [:0]const u8 {
    return switch (cb) {
        .mul1 => "",
        .mcg => "_mcg",
    };
}

/// The one decode every weight kernel calls, two codewords to two halves. A
/// pack whose search hashed a narrower window masks the sliding window down to
/// it first; w16 masks nothing and emits the source it always did.
fn codebookHelpers(comptime cb: exl3.Codebook, comptime win: exl3.Window) [:0]const u8 {
    const body: [:0]const u8 = comptime switch (cb) {
        .mul1 =>
        \\  const uint2 mixed = cw * uint2(0x83DCD12Du);
        \\  const uint2 pair_sums = (mixed & uint2(0x00FF00FFu)) + ((mixed >> uint2(8u)) & uint2(0x00FF00FFu));
        \\  const uint2 byte_sum = uint2(0x6400u) + (pair_sums & uint2(0xFFFFu)) + (pair_sums >> uint2(16u));
        \\  const half2 h = as_type<half2>(ushort2(byte_sum & uint2(0xFFFFu)));
        \\  return fma(h, as_type<half2>(ushort2(0x1EEEu)), as_type<half2>(ushort2(0xC931u)));
        ,
        .mcg =>
        \\  const uint2 r = ((cw * uint2(0xCBAC1FEDu)) & uint2(0x8FFF8FFFu)) ^ uint2(0x3B603B60u);
        \\  const half4 h = as_type<half4>(r);
        \\  return half2(h.x + h.y, h.z + h.w);
        ,
    };
    const mask: [:0]const u8 = comptime if (win == .w16) "" else std.fmt.comptimePrint("  cw &= uint2(0x{X}u);\n", .{win.mask()});
    const pair = comptime "static inline half2 exl3_pairh(uint2 cw) {\n" ++ mask ++ body ++ "\n}\n";
    return comptime pair ++
        \\static inline float2 exl3_decode2(uint2 cw) { return float2(exl3_pairh(cw)); }
        \\static inline float exl3_decode1(uint cw) { return exl3_decode2(uint2(cw, 0u)).x; }
        \\// Eight weights span N/2 whole bits, so weight j of an eight-weight group
        \\// ends exl3_end(N, j) bits past the group's first bit.
        \\static inline constexpr uint exl3_end(uint N, uint j) { return ((j + 1u) * N) >> 4u; }
        \\// The 64 stream bits ending at bit `end` of a tile of `nwords` words (the tile
        \\// wraps). Two words carry only the last 32 + end % 32 of them; `full` reads a third.
        \\static inline ulong exl3_window(const device uint *words, uint end, uint nwords, bool full) {
        \\  const uint last = (end - 1u) >> 5u;
        \\  const uint prev = last == 0u ? nwords - 1u : last - 1u;
        \\  const uint s = (0u - end) & 31u;
        \\  ulong bits = (((ulong)words[prev] << 32u) | (ulong)words[last]) >> s;
        \\  if (full) bits |= ((ulong)words[prev == 0u ? nwords - 1u : prev - 1u] << 32u) << (32u - s);
        \\  return bits;
        \\}
        \\// Lane l reads weights 8l..8l+7 from the bits ending at its group's last bit; weight
        \\// j's codeword ends exl3_lane_sh(N, j) bits before that. The least nonzero end % 32 is
        \\// W's lowest set bit, so a longer first codeword takes the third word.
        \\static inline constexpr uint exl3_lane_sh(uint N, uint j) { return N / 2u - exl3_end(N, j); }
        \\template<bool Wide> struct exl3_lane_type {
        \\  using type = ulong;
        \\  static inline type read(const device uint *words, uint end, uint W, bool full) {
        \\    return exl3_window(words, end, W, full);
        \\  }
        \\};
        \\template<> struct exl3_lane_type<true> {
        \\  using type = ulong2;
        \\  static inline type read(const device uint *words, uint end, uint W, bool full) {
        \\    return ulong2(exl3_window(words, end, W, full), exl3_window(words, end - 32u, W, true));
        \\  }
        \\};
        \\template<uint N> using exl3_lane_bits = typename exl3_lane_type<(exl3_lane_sh(N, 0u) + 16u > 64u)>::type;
        \\static inline uint exl3_lane_word(ulong bits, uint sh) { return uint(bits >> sh); }
        \\static inline uint exl3_lane_word(ulong2 bits, uint sh) {
        \\  return sh <= 48u ? uint(bits.x >> sh) : uint(bits.y >> (sh - 32u));
        \\}
        \\template<uint N>
        \\static inline exl3_lane_bits<N> exl3_lane(const device uint *words, uint lane) {
        \\  constexpr uint W = N / 2u;
        \\  constexpr uint low = W & (0u - W);
        \\  const uint end = W * lane + W;
        \\  return exl3_lane_type<(exl3_lane_sh(N, 0u) + 16u > 64u)>::read(words, end, W, exl3_lane_sh(N, 0u) + 16u > 32u + (low < 32u ? low : 32u));
        \\}
        \\// Weight t's 16-bit codeword is the window ending at floor((t+1)*K), with
        \\// K = N/16; the pair (t0, t0+1) shares one 32-bit funnel read.
        \\struct exl3_win { uint i0; uint i1; uint sh; uint fresh; };
        \\static inline exl3_win exl3_pair_window(uint t0, uint N) {
        \\  const uint e0 = ((t0 + 1u) * N) >> 4u;
        \\  const uint e1 = ((t0 + 2u) * N) >> 4u;
        \\  const uint b0 = e0 + 16u * N - 16u;
        \\  const uint b2 = e1 + 16u * N;
        \\  const uint j0 = b0 >> 5u;
        \\  const uint j1 = (b2 - 1u) >> 5u;
        \\  exl3_win w;
        \\  w.i0 = j0 % (N >> 1u);
        \\  w.i1 = j1 % (N >> 1u);
        \\  w.sh = (j1 + 1u) * 32u - b2;
        \\  w.fresh = e1 - e0;
        \\  return w;
        \\}
        \\
    ;
}

fn naxHeader(comptime cb: exl3.Codebook, comptime win: exl3.Window) [:0]const u8 {
    return comptime GEMM_NAX_INCLUDES ++ codebookHelpers(cb, win) ++ GEMM_NAX_FRAGS;
}

/// A weight kernel under the active decode parameters: its own slot, name and
/// header.
fn codebookKernel(slots: *KernelSlots, comptime base: [:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [:0]const u8) !mlx.mlx_fast_metal_kernel {
    return codebookKernelWith(slots, base, ins, outs, source, "");
}

/// `codebookKernel` with helpers of its own appended to the codebook header.
fn codebookKernelWith(slots: *KernelSlots, comptime base: [:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [:0]const u8, comptime frags: [:0]const u8) !mlx.mlx_fast_metal_kernel {
    switch (active_decode.codebook) {
        inline else => |cb| switch (active_decode.window) {
            inline else => |win| return getNamedKernel(
                &slots[cbIndex(cb)][win.index()],
                comptime base ++ cbSuffix(cb) ++ winSuffix(win),
                ins,
                outs,
                source,
                comptime codebookHelpers(cb, win) ++ frags,
            ),
        },
    }
}

const GemvKey = struct { in_dim: c_int, out_dim: c_int, n: u32 };

fn CfgCache(comptime Key: type, comptime CAP: usize) type {
    return struct {
        const Self = @This();
        keys: [CAP]Key = @splat(std.mem.zeroes(Key)),
        cfgs: [CAP]?mlx.mlx_fast_metal_kernel_config = @splat(null),
        used: [CAP]u64 = @splat(0),
        tick: u64 = 0,

        fn get(self: *Self, key: Key) ?mlx.mlx_fast_metal_kernel_config {
            for (self.cfgs, 0..) |c, i| {
                if (c != null and std.meta.eql(self.keys[i], key)) {
                    self.tick += 1;
                    self.used[i] = self.tick;
                    return c.?;
                }
            }
            return null;
        }

        fn put(self: *Self, key: Key, cfg: mlx.mlx_fast_metal_kernel_config) void {
            var victim: usize = 0;
            var oldest: u64 = std.math.maxInt(u64);
            for (self.cfgs, 0..) |c, i| {
                if (c == null) {
                    victim = i;
                    break;
                }
                if (self.used[i] < oldest) {
                    oldest = self.used[i];
                    victim = i;
                }
            }
            if (self.cfgs[victim]) |old| _ = mlx.mlx_fast_metal_kernel_config_free(old);
            self.cfgs[victim] = cfg;
            self.keys[victim] = key;
            self.tick += 1;
            self.used[victim] = self.tick;
        }
    };
}

const IndexedKey = struct { in_dim: c_int, out_dim: c_int, topk: c_int, n: u32 };
const UnaryKey = struct { dim: c_int, topk: c_int };
const GemmSortedKey = struct { in_dim: c_int, out_dim: c_int, rows: c_int, win: c_int, n: u32 };

const GEMM_WINDOW_ROWS: c_int = 32;

/// Maximum run supported by all sorted-GEMM bodies; larger runs leave tail rows unwritten.
const GEMM_WINDOW_MAX_ROWS: c_int = 32;

var gemm_win_cached: ?c_int = null;
var gemm_align_cached: ?bool = null;

/// null = not a window these kernels run, so the default stands.
fn resolveGemmWindowRows(raw: ?[]const u8) ?c_int {
    const v = raw orelse return null;
    const n = std.fmt.parseInt(c_int, v, 10) catch return null;
    if (n < 1 or n > GEMM_WINDOW_MAX_ROWS) return null;
    return n;
}

fn gemmWindowRows() c_int {
    if (gemm_win_cached) |v| return v;
    const v = blk: {
        const p = std.c.getenv("SUSHI_EXL3_GEMM_WIN") orelse break :blk GEMM_WINDOW_ROWS;
        const raw = std.mem.span(p);
        if (resolveGemmWindowRows(raw)) |n| break :blk n;
        log.warn("[exl3] SUSHI_EXL3_GEMM_WIN={s} is not a window in 1..{d}; keeping {d}\n", .{ raw, GEMM_WINDOW_MAX_ROWS, GEMM_WINDOW_ROWS });
        break :blk GEMM_WINDOW_ROWS;
    };
    gemm_win_cached = v;
    return v;
}

fn gemmWindowAligned() bool {
    if (gemm_align_cached) |v| return v;
    var on = true;
    if (std.c.getenv("SUSHI_EXL3_WIN_ALIGN")) |p| {
        const v = std.mem.span(p);
        if (v.len > 0 and v[0] == '0') on = false;
    }
    gemm_align_cached = on;
    return on;
}
var indexed_coop_cfgs: CfgCache(IndexedKey, 8) = .{};
var prepare_cfgs: CfgCache(UnaryKey, 8) = .{};
var finish_cfgs: CfgCache(UnaryKey, 8) = .{};
var gemm_sorted_cfgs: CfgCache(GemmSortedKey, 8) = .{};
var gemm_simdmat_cfgs: CfgCache(GemmSortedKey, 8) = .{};
var gemm_nax_cfgs: CfgCache(GemmSortedKey, 8) = .{};
var gemm_nax_kernel: KernelSlots = no_kernels;
var gemm_nax_failed: bool = false;
var gemm_nax_cached: ?bool = null;
const PairPrepKey = struct { in_dim: c_int, nslots: c_int, topk: c_int };
const PairGemvKey = struct { in_dim: c_int, out_dim: c_int, nslots: c_int, nsplit: c_int, topk: c_int, n: u32, layout: GemvLayout, group: c_int };
const DownFusedKey = struct { in_dim: c_int, out_dim: c_int, nslots: c_int, nsplit: c_int, n: u32, layout: GemvLayout, group: c_int = 0 };
const MidKey = struct { dim: c_int, nslots: c_int, limit: c_int = 0 };
const ReduceKey = struct { out_dim: c_int, rows: c_int, topk: c_int };
const DecodeReduceKey = struct { out_dim: c_int, rows: c_int, topk: c_int, dtype: mlx.mlx_dtype };
var pair_gemv_cfgs: CfgCache(PairGemvKey, 8) = .{};
var mid_cfgs: CfgCache(MidKey, 8) = .{};
var reduce_cfgs: CfgCache(DecodeReduceKey, 8) = .{};
var down_fused_cfgs: CfgCache(DownFusedKey, 8) = .{};
var token_prep_cfgs: CfgCache(PairPrepKey, 8) = .{};
var token_pair_prep_cfgs: CfgCache(PairPrepKey, 8) = .{};
var token_reduce_cfgs: CfgCache(ReduceKey, 8) = .{};
const ScatterKey = struct { dim: c_int, nslots: c_int };
var token_scatter_cfgs: CfgCache(ScatterKey, 8) = .{};
var token_prepare_kernel: ?mlx.mlx_fast_metal_kernel = null;
var token_pair_prepare_kernel: ?mlx.mlx_fast_metal_kernel = null;
var token_pair_unsorted_kernel: ?mlx.mlx_fast_metal_kernel = null;
var token_scatter_kernel: ?mlx.mlx_fast_metal_kernel = null;
var token_reduce_kernel: ?mlx.mlx_fast_metal_kernel = null;
var fused_dispatches: u32 = 0;
var apply_host_ns: u64 = 0;
var apply_host_n: u32 = 0;
var apply_host_layers: u32 = 0;
var apply_host_dumps: u32 = 0;
var apply_ubench_env: ?bool = null;

fn applyUbenchOn() bool {
    if (apply_ubench_env) |v| return v;
    const v = diagEnvValueOn(std.c.getenv("SUSHI_DECODE_TICK_UBENCH"));
    apply_ubench_env = v;
    return v;
}

pub fn resetFusedDispatchCount() void {
    fused_dispatches = 0;
}

pub fn fusedDispatchCount() u32 {
    return fused_dispatches;
}

fn gpuArch(buf: []u8) ?[]const u8 {
    var dev = mlx.mlx_device{ .ctx = null };
    if (mlx.mlx_get_default_device(&dev) != 0) return null;
    var info = mlx.mlx_device_info_new();
    defer _ = mlx.mlx_device_info_free(info);
    if (mlx.mlx_device_info_get(&info, dev) != 0) return null;
    var cstr: [*:0]const u8 = undefined;
    if (mlx.mlx_device_info_get_string(&cstr, info, "architecture") != 0) return null;
    const arch = std.mem.span(cstr);
    if (arch.len == 0 or arch.len > buf.len) return null;
    @memcpy(buf[0..arch.len], arch);
    return buf[0..arch.len];
}

fn gemmNaxOn() bool {
    return !gemm_nax_failed and gemmNaxAvailable();
}

// Capability is independent of the dispatch failure latch: tests must still
// probe the real kernel after an earlier dispatch fell back to SIMD.
fn gemmNaxAvailable() bool {
    if (std.c.getenv("SUSHI_FORCE_GPU_FAMILY_FALLBACK")) |p| {
        const v = std.mem.span(p);
        if (v.len > 0 and v[0] == '1') return false;
    }
    if (gemm_nax_cached) |v| return v;
    var buf: [128]u8 = undefined;
    const arch = gpuArch(&buf) orelse {
        gemm_nax_cached = false;
        return false;
    };
    var i: usize = 0;
    while (i + 2 < arch.len) : (i += 1) {
        const a = arch[i] | 32;
        const b = arch[i + 1] | 32;
        if (a == 'g' and b == '1' and arch[i + 2] >= '7' and arch[i + 2] <= '9') {
            gemm_nax_cached = true;
            return true;
        }
    }
    gemm_nax_cached = false;
    return false;
}

fn getGemmNaxKernel() !mlx.mlx_fast_metal_kernel {
    @setEvalBranchQuota(20_000);
    const ref = nax_reference_override;
    const slots = if (ref) &gemm_nax_ref_kernel else &gemm_nax_kernel;
    switch (active_decode.codebook) {
        inline else => |cb| switch (active_decode.window) {
            inline else => |win| {
                if (slots[cbIndex(cb)][win.index()]) |k| return k;
                const kernel = (if (ref)
                    buildNaxGemmKernel(GEMM_NAX_REFERENCE_SOURCE, naxHeader(cb, win), comptime "sushi_exl3_k4_gemm_nax_ref" ++ cbSuffix(cb) ++ winSuffix(win))
                else
                    buildNaxGemmKernel(GEMM_NAX_SOURCE, naxHeader(cb, win), comptime "sushi_exl3_k4_gemm_nax" ++ cbSuffix(cb) ++ winSuffix(win))) orelse {
                    gemm_nax_failed = true;
                    log.info("[exl3-gemm] NAX arm declined (kernel probe failed); the sorted GEMM serves prefill\n", .{});
                    return error.MetalKernelCompileFailed;
                };
                slots[cbIndex(cb)][win.index()] = kernel;
                return kernel;
            },
        },
    }
}

/// Metal JIT-compiles a kernel at its first EVAL, not at apply, so a source the
/// toolchain rejects is proven here on a one-tile problem before any prefill
/// depends on it. null = the arm is unusable; the latch it raised is dropped.
fn buildNaxGemmKernel(source: [:0]const u8, header: [:0]const u8, name: [*:0]const u8) ?mlx.mlx_fast_metal_kernel {
    const input_names = [_][*:0]const u8{ "x", "trellis", "eids", "wstarts", "wnlive" };
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(name, in_vec, out_vec, source, header, true, false);
    if (kernel.ctx == null) return null;
    if (probeNaxGemm(kernel)) return kernel;
    _ = mlx.mlx_fast_metal_kernel_free(kernel);
    return null;
}

fn probeNaxGemm(kernel: mlx.mlx_fast_metal_kernel) bool {
    const s = mlx.gpuStream();
    const had_error = mlx.errorPending();
    defer mlx.dropLatchedErrorUnless(had_error);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const in_dim: c_int = 16;
    const out_dim: c_int = 128;
    if (mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, out_dim }, 2, .float16) != 0) return false;
    if (mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 128, 1, 1) != 0) return false;
    if (mlx.mlx_fast_metal_kernel_config_set_grid(cfg, out_dim, 1, 1) != 0) return false;
    if (mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "IDIM", in_dim) != 0) return false;
    if (mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ODIM", out_dim) != 0) return false;
    if (mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "WIN", gemmWindowRows()) != 0) return false;
    if (mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NHW", 64) != 0) return false;
    const xh = std.mem.zeroes([16]u16);
    const x = mlx.mlx_array_new_data(&xh, &[_]c_int{ 1, in_dim }, 2, .float16);
    defer _ = mlx.mlx_array_free(x);
    const th = std.mem.zeroes([8 * 64]u16);
    const trellis = mlx.mlx_array_new_data(&th, &[_]c_int{ 1, 1, 8, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(trellis);
    const zero = [_]u32{0};
    const one = [_]u32{1};
    const eids = mlx.mlx_array_new_data(&zero, &[_]c_int{1}, 1, .uint32);
    defer _ = mlx.mlx_array_free(eids);
    const starts = mlx.mlx_array_new_data(&zero, &[_]c_int{1}, 1, .uint32);
    defer _ = mlx.mlx_array_free(starts);
    const nlive = mlx.mlx_array_new_data(&one, &[_]c_int{1}, 1, .uint32);
    defer _ = mlx.mlx_array_free(nlive);
    const inputs = [_]mlx.mlx_array{ x, trellis, eids, starts, nlive };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs);
    if (mlx.mlx_fast_metal_kernel_apply(&outputs, kernel, inputs_vec, cfg, s) != 0) return false;
    if (mlx.mlx_vector_array_size(outputs) != 1) return false;
    var out = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(out);
    if (mlx.mlx_vector_array_get(&out, outputs, 0) != 0) return false;
    if (mlx.mlx_array_eval(out) != 0) return false;
    return !mlx.errorPending();
}

fn getGemmSortedKernel() !mlx.mlx_fast_metal_kernel {
    const ins = [_][*:0]const u8{ "x", "trellis", "eids", "wstarts", "wnlive" };
    const outs = [_][*:0]const u8{"y"};
    return codebookKernel(&gemm_sorted_kernel, "sushi_exl3_k4_gemm_sorted", &ins, &outs, GEMM_SORTED_SOURCE);
}

fn getGemmSimdmatKernel() !mlx.mlx_fast_metal_kernel {
    const ins = [_][*:0]const u8{ "x", "trellis", "eids", "wstarts", "wnlive" };
    const outs = [_][*:0]const u8{"y"};
    return codebookKernelWith(&gemm_simdmat_kernel, "sushi_exl3_k4_gemm_simdmat", &ins, &outs, GEMM_SIMDMAT_SOURCE, GEMM_SIMDMAT_FRAGS);
}

fn sortedGemmCfg(cache: *CfgCache(GemmSortedKey, 8), key: GemmSortedKey) !mlx.mlx_fast_metal_kernel_config {
    if (cache.get(key)) |c| return c;
    const c = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{ key.rows, key.out_dim }, 2, .float16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 128, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "IDIM", key.in_dim));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ODIM", key.out_dim));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "WIN", key.win));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "NHW", @intCast(key.n)));
    cache.put(key, c);
    return c;
}

fn applySortedGemm(s: mlx.mlx_stream, kernel: mlx.mlx_fast_metal_kernel, cfg: mlx.mlx_fast_metal_kernel_config, x: mlx.mlx_array, trellis: mlx.mlx_array, eids: mlx.mlx_array, tab: WindowTable) !mlx.mlx_array {
    const inputs = [_]mlx.mlx_array{ x, trellis, eids, tab.starts, tab.nlives };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, cfg, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 0));
    return out;
}

const WindowTable = struct { starts: mlx.mlx_array, nlives: mlx.mlx_array, nwin: c_int };

var gemm_sel_logged: bool = false;

fn logGemmSelector(win: c_int, aligned: bool, nwin: c_int, n: c_int) void {
    if (n < 2048) return;
    if (gemm_sel_logged) return;
    gemm_sel_logged = true;
    log.info("[exl3-gemm] win={d} aligned={d} nwin={d} mixed={d} n={d}\n", .{
        win,
        @intFromBool(aligned),
        nwin,
        @as(u32, if (aligned) 0 else 1),
        n,
    });
}

fn buildWindowTable(s: mlx.mlx_stream, eids: mlx.mlx_array, n: c_int, win: c_int) !WindowTable {
    return buildWindowTableHost(s, eids, n, win);
}

fn gemmWindowTable(s: mlx.mlx_stream, eids: mlx.mlx_array, n: c_int, win: c_int, aligned: bool) !WindowTable {
    return if (aligned) buildWindowTable(s, eids, n, win) else buildStrideTable(s, n, win);
}

fn buildWindowTableHost(s: mlx.mlx_stream, eids: mlx.mlx_array, n: c_int, win: c_int) !WindowTable {
    const ids = try std.heap.page_allocator.alloc(u32, @intCast(n));
    defer std.heap.page_allocator.free(ids);
    var contig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(contig);
    try mlx.check(mlx.mlx_contiguous(&contig, eids, false, s));
    try mlx.check(mlx.mlx_array_eval(contig));
    switch (mlx.mlx_array_dtype(contig)) {
        .uint32 => {
            const p = mlx.mlx_array_data_uint32(contig) orelse return error.F16Unreadable;
            @memcpy(ids, p[0..ids.len]);
        },
        .int32 => {
            const p = mlx.mlx_array_data_int32(contig) orelse return error.F16Unreadable;
            for (ids, 0..) |*d, i| d.* = @intCast(p[i]);
        },
        else => return error.BadExl3Shape,
    }
    const runs = try buildRuns(std.heap.page_allocator, ids);
    defer std.heap.page_allocator.free(runs.start);
    defer std.heap.page_allocator.free(runs.len);
    defer std.heap.page_allocator.free(runs.eid);
    const w: u32 = @intCast(win);
    var nwin_u: u32 = 0;
    var r: u32 = 0;
    while (r < runs.n) : (r += 1) {
        nwin_u += (runs.len[r] + w - 1) / w;
    }
    const sh = try std.heap.page_allocator.alloc(u32, nwin_u);
    defer std.heap.page_allocator.free(sh);
    const lh = try std.heap.page_allocator.alloc(u32, nwin_u);
    defer std.heap.page_allocator.free(lh);
    var k: u32 = 0;
    r = 0;
    while (r < runs.n) : (r += 1) {
        var off: u32 = 0;
        while (off < runs.len[r]) {
            const live = @min(w, runs.len[r] - off);
            sh[k] = runs.start[r] + off;
            lh[k] = live;
            k += 1;
            off += live;
        }
    }
    const starts_raw = mlx.mlx_array_new_data(sh.ptr, &[_]c_int{@intCast(nwin_u)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(starts_raw);
    const nlives_raw = mlx.mlx_array_new_data(lh.ptr, &[_]c_int{@intCast(nwin_u)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(nlives_raw);
    var starts = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(starts);
    var nlives = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(nlives);
    try mlx.check(mlx.mlx_contiguous(&starts, starts_raw, false, s));
    try mlx.check(mlx.mlx_contiguous(&nlives, nlives_raw, false, s));
    try mlx.check(mlx.mlx_array_eval(starts));
    try mlx.check(mlx.mlx_array_eval(nlives));
    if (exl3UbenchOn()) {
        benchPrint("[exl3-ubench] win_table_host nwin={d} n={d} win={d} eval=eids\n", .{ nwin_u, n, win });
    }
    return .{ .starts = starts, .nlives = nlives, .nwin = @intCast(nwin_u) };
}

fn buildStrideTable(s: mlx.mlx_stream, n: c_int, win: c_int) !WindowTable {
    const w: u32 = @intCast(win);
    const nn: u32 = @intCast(n);
    const nwin_u = (nn + w - 1) / w;
    const sh = try std.heap.page_allocator.alloc(u32, nwin_u);
    defer std.heap.page_allocator.free(sh);
    const lh = try std.heap.page_allocator.alloc(u32, nwin_u);
    defer std.heap.page_allocator.free(lh);
    var i: u32 = 0;
    while (i < nwin_u) : (i += 1) {
        const st = i * w;
        sh[i] = st;
        lh[i] = @min(w, nn - st);
    }
    const starts_raw = mlx.mlx_array_new_data(sh.ptr, &[_]c_int{@intCast(nwin_u)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(starts_raw);
    const nlives_raw = mlx.mlx_array_new_data(lh.ptr, &[_]c_int{@intCast(nwin_u)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(nlives_raw);
    var starts = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(starts);
    var nlives = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(nlives);
    try mlx.check(mlx.mlx_contiguous(&starts, starts_raw, false, s));
    try mlx.check(mlx.mlx_contiguous(&nlives, nlives_raw, false, s));
    try mlx.check(mlx.mlx_array_eval(starts));
    try mlx.check(mlx.mlx_array_eval(nlives));
    if (exl3UbenchOn()) {
        benchPrint("[exl3-ubench] win_table_stride nwin={d} n={d} win={d}\n", .{ nwin_u, n, win });
    }
    return .{ .starts = starts, .nlives = nlives, .nwin = @intCast(nwin_u) };
}

fn windowStats(ids: []const u32, win: u32, aligned: bool) struct { nwin: u32, mixed: u32, decodes: u32 } {
    if (aligned) {
        const runs = buildRuns(std.heap.page_allocator, ids) catch return .{ .nwin = 0, .mixed = 0, .decodes = 0 };
        defer std.heap.page_allocator.free(runs.start);
        defer std.heap.page_allocator.free(runs.len);
        defer std.heap.page_allocator.free(runs.eid);
        var nwin: u32 = 0;
        var r: u32 = 0;
        while (r < runs.n) : (r += 1) {
            nwin += (runs.len[r] + win - 1) / win;
        }
        return .{ .nwin = nwin, .mixed = 0, .decodes = nwin };
    }
    const nwin = (@as(u32, @intCast(ids.len)) + win - 1) / win;
    var mixed: u32 = 0;
    var decodes: u32 = 0;
    var w: u32 = 0;
    while (w < nwin) : (w += 1) {
        const st = w * win;
        const nlive = @min(win, @as(u32, @intCast(ids.len)) - st);
        var runs_here: u32 = 1;
        var i: u32 = 1;
        while (i < nlive) : (i += 1) {
            if (ids[st + i] != ids[st + i - 1]) runs_here += 1;
        }
        decodes += runs_here;
        if (runs_here > 1) mixed += 1;
    }
    return .{ .nwin = nwin, .mixed = mixed, .decodes = decodes };
}

pub fn innerGemmSorted(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    trellis: mlx.mlx_array,
    eids: mlx.mlx_array,
) !mlx.mlx_array {
    return innerGemmSortedWinAlign(s, x, trellis, eids, gemmWindowRows(), gemmWindowAligned());
}

fn innerGemmSortedWin(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    trellis: mlx.mlx_array,
    eids: mlx.mlx_array,
    win: c_int,
) !mlx.mlx_array {
    return innerGemmSortedWinAlign(s, x, trellis, eids, win, true);
}

fn innerGemmSortedWinAlign(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    trellis: mlx.mlx_array,
    eids: mlx.mlx_array,
    win: c_int,
    aligned: bool,
) !mlx.mlx_array {
    const xsh = mlx.getShape(x);
    if (xsh.len != 2) return error.BadExl3Shape;
    if (win <= 0 or win > GEMM_WINDOW_MAX_ROWS) return error.BadExl3Shape;
    const tab = try gemmWindowTable(s, eids, xsh[0], win, aligned);
    defer _ = mlx.mlx_array_free(tab.starts);
    defer _ = mlx.mlx_array_free(tab.nlives);
    return innerGemmSortedTable(s, x, trellis, eids, win, aligned, tab);
}

/// The window table depends only on the sorted slots, so a layer's three
/// projections share one: each build is a host eval that drains the GPU.
fn innerGemmSortedTable(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    trellis: mlx.mlx_array,
    eids: mlx.mlx_array,
    win: c_int,
    aligned: bool,
    tab: WindowTable,
) !mlx.mlx_array {
    const xsh = mlx.getShape(x);
    const tsh = mlx.getShape(trellis);
    if (xsh.len != 2 or tsh.len != 4) return error.BadExl3Shape;
    if (win <= 0 or win > GEMM_WINDOW_MAX_ROWS) return error.BadExl3Shape;
    const n = xsh[0];
    const in_dim = xsh[1];
    const out_dim = tsh[2] * 16;
    const out_tiles = tsh[2];
    const rate = try packedRate(tsh[3]);
    if (tsh[1] * 16 != in_dim) return error.BadExl3Shape;
    if (tab.nwin <= 0) return error.BadExl3Shape;
    logGemmSelector(win, aligned, tab.nwin, n);
    const key = GemmSortedKey{ .in_dim = in_dim, .out_dim = out_dim, .rows = n, .win = win, .n = rate.n };
    if (gemmNaxOn() and @rem(out_dim, 128) == 0) {
        // The fallback below answers a NAX build or dispatch that failed, so the
        // failure's latch is ours to drop: left standing it becomes the next
        // decode tick's `MlxFailure`.
        const had_error = mlx.errorPending();
        if (getGemmNaxKernel()) |nk| {
            const ncfg = try sortedGemmCfg(&gemm_nax_cfgs, key);
            try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(ncfg, out_dim, tab.nwin, 1));
            const ninputs = [_]mlx.mlx_array{ x, trellis, eids, tab.starts, tab.nlives };
            const ninputs_vec = mlx.mlx_vector_array_new_data(&ninputs, ninputs.len);
            defer _ = mlx.mlx_vector_array_free(ninputs_vec);
            var noutputs = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(noutputs);
            if (mlx.mlx_fast_metal_kernel_apply(&noutputs, nk, ninputs_vec, ncfg, s) == 0 and mlx.mlx_vector_array_size(noutputs) == 1) {
                var nout = mlx.mlx_array_new();
                errdefer _ = mlx.mlx_array_free(nout);
                try mlx.check(mlx.mlx_vector_array_get(&nout, noutputs, 0));
                logFunnel(funnelReads(rate.n), rate.n, .nax, mlx.mlx_array_dtype(x));
                return nout;
            }
            gemm_nax_failed = true;
        } else |_| {
            gemm_nax_failed = true;
        }
        mlx.dropLatchedErrorUnless(had_error);
    }
    // x feeds `simdgroup_matrix<half>` as stored, so only f16 activations take the matrix body.
    if ((gemm_simdmat_force orelse true) and @rem(out_dim, 128) == 0 and mlx.mlx_array_dtype(x) == .float16) {
        const cfg = try sortedGemmCfg(&gemm_simdmat_cfgs, key);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, out_dim, tab.nwin, 1));
        const out = try applySortedGemm(s, try getGemmSimdmatKernel(), cfg, x, trellis, eids, tab);
        if (!gemm_simdmat_engaged) {
            gemm_simdmat_engaged = true;
            log.info("[exl3-gemm] simdgroup-matrix body engaged n={d}\n", .{rate.n});
        }
        logFunnel(funnelReads(rate.n), rate.n, .simdmat, mlx.mlx_array_dtype(x));
        return out;
    }
    const cfg = try sortedGemmCfg(&gemm_sorted_cfgs, key);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, out_tiles * 128, tab.nwin, 1));
    return applySortedGemm(s, try getGemmSortedKernel(), cfg, x, trellis, eids, tab);
}

fn getPrepareKernel() !mlx.mlx_fast_metal_kernel {
    if (prepare_kernel) |k| return k;
    const input_names = [_][*:0]const u8{ "x", "suh", "slots" };
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "sushi_exl3_prepare_h128",
        in_vec,
        out_vec,
        PREPARE_SOURCE,
        "",
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    prepare_kernel = kernel;
    return kernel;
}

fn getFinishKernel() !mlx.mlx_fast_metal_kernel {
    if (finish_kernel) |k| return k;
    const input_names = [_][*:0]const u8{ "inner", "svh", "slots" };
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "sushi_exl3_finish_h128",
        in_vec,
        out_vec,
        FINISH_SOURCE,
        "",
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    finish_kernel = kernel;
    return kernel;
}

fn applyUnary(
    s: mlx.mlx_stream,
    kernel: mlx.mlx_fast_metal_kernel,
    inputs: []const mlx.mlx_array,
    cfg: mlx.mlx_fast_metal_kernel_config,
) !mlx.mlx_array {
    const inputs_vec = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, cfg, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 0));
    return out;
}

pub fn prepareIndexed(s: mlx.mlx_stream, x_in: mlx.mlx_array, suh: mlx.mlx_array, slots: mlx.mlx_array) !mlx.mlx_array {
    const topk = mlx.getShape(slots)[0];
    const xsh = mlx.getShape(x_in);
    const in_dim = xsh[xsh.len - 1];
    // The kernel reads `topk` rows of x, so a rank-2 x must already carry them.
    if (xsh.len > 2 or (xsh.len == 2 and xsh[0] != topk)) return error.BadExl3Shape;
    var x = x_in;
    var owned = false;
    if (xsh.len == 1) {
        var b = mlx.mlx_array_new();
        const shape = [_]c_int{ topk, in_dim };
        try mlx.check(mlx.mlx_broadcast_to(&b, x_in, &shape, 2, s));
        x = b;
        owned = true;
    }
    defer if (owned) {
        _ = mlx.mlx_array_free(x);
    };
    const blocks = @divExact(in_dim, 128);
    const key = UnaryKey{ .dim = in_dim, .topk = topk };
    const cfg = prepare_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const out_shape = [_]c_int{ topk, in_dim };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &out_shape, 2, .float16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, 32 * blocks, topk, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "IDIM", in_dim));
        prepare_cfgs.put(key, c);
        break :blk c;
    };
    return applyUnary(s, try getPrepareKernel(), &.{ x, suh, slots }, cfg);
}

pub fn finishIndexed(s: mlx.mlx_stream, inner: mlx.mlx_array, svh: mlx.mlx_array, slots: mlx.mlx_array) !mlx.mlx_array {
    const sh = mlx.getShape(inner);
    const topk = sh[0];
    const out_dim = sh[1];
    const blocks = @divExact(out_dim, 128);
    const key = UnaryKey{ .dim = out_dim, .topk = topk };
    const cfg = finish_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const out_shape = [_]c_int{ topk, out_dim };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &out_shape, 2, .float16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, 32 * blocks, topk, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ODIM", out_dim));
        finish_cfgs.put(key, c);
        break :blk c;
    };
    return applyUnary(s, try getFinishKernel(), &.{ inner, svh, slots }, cfg);
}

fn getGemvKernel() !mlx.mlx_fast_metal_kernel {
    const ins = [_][*:0]const u8{ "x", "trellis" };
    const outs = [_][*:0]const u8{"y"};
    return codebookKernel(&gemv_kernel, "sushi_exl3_k4_mcg_gemv", &ins, &outs, GEMV_SOURCE);
}

var inner_gemv_cfgs: CfgCache(GemvKey, 8) = .{};

fn gemvConfig(in_dim: c_int, out_dim: c_int, rate: exl3.Rate) !mlx.mlx_fast_metal_kernel_config {
    const key = GemvKey{ .in_dim = in_dim, .out_dim = out_dim, .n = rate.n };
    if (inner_gemv_cfgs.get(key)) |c| return c;
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const out_shape = [_]c_int{out_dim};
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &out_shape, 1, .float16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, out_dim, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "IDIM", in_dim));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ODIM", out_dim));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NHW", @intCast(rate.n)));
    inner_gemv_cfgs.put(key, cfg);
    return cfg;
}

const INDEXED_COOP_SOURCE: [:0]const u8 =
    \\threadgroup float partial[4 * 256];
    \\uint ot = uint(threadgroup_position_in_grid.x);
    \\uint slot = uint(threadgroup_position_in_grid.y);
    \\uint split = uint(threadgroup_position_in_grid.z);
    \\uint sg = uint(simdgroup_index_in_threadgroup);
    \\uint lane = uint(thread_index_in_simdgroup);
    \\uint lid = uint(thread_index_in_threadgroup);
    \\constexpr uint TILE = 16u;
    \\constexpr uint N = uint(NHW);
    \\constexpr uint PACKED_HW = N;
    \\constexpr uint PACKED_W = N / 2u;
    \\constexpr uint IT = uint(IDIM) / TILE;
    \\constexpr uint OT = uint(ODIM) / TILE;
    \\constexpr uint SGS = 4u;
    \\constexpr uint SPLITS = 1u;
    \\const uint eid = uint(slots[slot]);
    \\const uint tiles_per_split = (IT + SPLITS - 1u) / SPLITS;
    \\const uint tk0 = split * tiles_per_split;
    \\const uint tk1 = min(tk0 + tiles_per_split, IT);
    \\const uint prow = (lane & 3u) * 2u;
    \\const uint pcol = lane >> 2u;
    \\uint pos[8];
    \\pos[0] = prow * 16u + pcol;
    \\pos[1] = (prow + 1u) * 16u + pcol;
    \\pos[2] = (prow + 8u) * 16u + pcol;
    \\pos[3] = (prow + 9u) * 16u + pcol;
    \\pos[4] = prow * 16u + pcol + 8u;
    \\pos[5] = (prow + 1u) * 16u + pcol + 8u;
    \\pos[6] = (prow + 8u) * 16u + pcol + 8u;
    \\pos[7] = (prow + 9u) * 16u + pcol + 8u;
    \\const uint row0 = pos[0] >> 4u;
    \\const uint row1 = pos[1] >> 4u;
    \\const uint row2 = pos[2] >> 4u;
    \\const uint row3 = pos[3] >> 4u;
    \\float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    \\const size_t xb = (size_t)slot * (size_t)(IDIM);
    \\const size_t expert_stride = (size_t)IT * (size_t)OT * (size_t)PACKED_HW;
    \\const device uint* trellis_e = (const device uint*)(trellis + (size_t)eid * expert_stride);
    \\if (N == 64u) {
    \\for (uint tk = tk0 + sg; tk < tk1; tk += SGS) {
    \\  const device uint* words = trellis_e + ((size_t)tk * (size_t)OT + ot) * 32u;
    \\  const ulong merged = ((ulong)words[(lane + 31u) & 31u] << 32) | (ulong)words[lane];
    \\  const float in0 = float(x[xb + tk * TILE + row0]);
    \\  const float in1 = float(x[xb + tk * TILE + row1]);
    \\  const float in2 = float(x[xb + tk * TILE + row2]);
    \\  const float in3 = float(x[xb + tk * TILE + row3]);
    \\  const uint sh[8] = {28u, 24u, 20u, 16u, 12u, 8u, 4u, 0u};
    \\  const float ins[8] = {in0, in1, in2, in3, in0, in1, in2, in3};
    \\  for (uint p = 0u; p < 4u; p++) {
    \\    const uint2 cw = uint2(uint(merged >> sh[p * 2u]), uint(merged >> sh[p * 2u + 1u])) & uint2(0xffffu);
    \\    const float2 w = exl3_decode2(cw);
    \\    acc[p * 2u] = fma(ins[p * 2u], w.x, acc[p * 2u]);
    \\    acc[p * 2u + 1u] = fma(ins[p * 2u + 1u], w.y, acc[p * 2u + 1u]);
    \\  }
    \\}
    \\} else {
    \\  for (uint tk = tk0 + sg; tk < tk1; tk += SGS) {
    \\    const device uint* words = trellis_e + ((size_t)tk * (size_t)OT + ot) * PACKED_W;
    \\    const auto merged = exl3_lane<N>(words, lane);
    \\    const float in0 = float(x[xb + tk * TILE + row0]);
    \\    const float in1 = float(x[xb + tk * TILE + row1]);
    \\    const float in2 = float(x[xb + tk * TILE + row2]);
    \\    const float in3 = float(x[xb + tk * TILE + row3]);
    \\    const float ins[8] = {in0, in1, in2, in3, in0, in1, in2, in3};
    \\    for (uint p = 0u; p < 4u; p++) {
    \\      const uint2 cw = uint2(exl3_lane_word(merged, exl3_lane_sh(N, p * 2u)), exl3_lane_word(merged, exl3_lane_sh(N, p * 2u + 1u))) & uint2(0xffffu);
    \\      const float2 w = exl3_decode2(cw);
    \\      acc[p * 2u] = fma(ins[p * 2u], w.x, acc[p * 2u]);
    \\      acc[p * 2u + 1u] = fma(ins[p * 2u + 1u], w.y, acc[p * 2u + 1u]);
    \\    }
    \\  }
    \\}
    \\for (uint si = 0u; si < 8u; si++) {
    \\  partial[sg * 256u + pos[si]] = acc[si];
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if (lid < 16u) {
    \\  float sum = 0.0f;
    \\  for (uint r = 0u; r < 16u; r++) {
    \\    const uint p = r * 16u + lid;
    \\    for (uint g = 0u; g < SGS; g++) {
    \\      sum += partial[g * 256u + p];
    \\    }
    \\  }
    \\  y[(size_t)slot * (size_t)(ODIM) + ot * TILE + lid] = half(sum);
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
;

var indexed_coop_kernel: KernelSlots = no_kernels;

fn getIndexedCoopKernel() !mlx.mlx_fast_metal_kernel {
    const ins = [_][*:0]const u8{ "x", "trellis", "slots" };
    const outs = [_][*:0]const u8{"y"};
    return codebookKernel(&indexed_coop_kernel, "sushi_exl3_k4_mul1_gemv_indexed", &ins, &outs, INDEXED_COOP_SOURCE);
}

const INDEXED_PAIR_COOP_SOURCE: [:0]const u8 = blk: {
    const split = "uint split = uint(threadgroup_position_in_grid.z);";
    const at = std.mem.indexOf(u8, INDEXED_COOP_SOURCE, split).?;
    const pointers =
        \\const bool upper = threadgroup_position_in_grid.z != 0u;
        \\const device half* x = upper ? xu : xg;
        \\const device ushort* trellis = upper ? tu : tg;
        \\device half* y = upper ? yu : yg;
        \\
    ;
    // Grid Z selects a projection, never a K split. Keep the original loop
    // and four-simdgroup reduction exactly as the standalone kernel.
    break :blk pointers ++ INDEXED_COOP_SOURCE[0..at] ++ "uint split = 0u;" ++ INDEXED_COOP_SOURCE[at + split.len ..];
};
var indexed_pair_coop_kernel: KernelSlots = no_kernels;
var indexed_pair_coop_cfgs: CfgCache(IndexedKey, 8) = .{};
var indexed_pair_coop_engaged: bool = false;

fn indexedPairCoopF16(s: mlx.mlx_stream, xg: mlx.mlx_array, xu: mlx.mlx_array, tg: mlx.mlx_array, tu: mlx.mlx_array, slots: mlx.mlx_array) !?[2]mlx.mlx_array {
    const xs = mlx.getShape(xg);
    const ts = mlx.getShape(tg);
    const ss = mlx.getShape(slots);
    if (!mlx.streamIsGpu(s) or xs.len != 2 or ts.len != 4 or ss.len != 1 or
        !std.mem.eql(c_int, xs, mlx.getShape(xu)) or !std.mem.eql(c_int, ts, mlx.getShape(tu)) or
        xs[0] <= 0 or xs[1] <= 0 or ts[0] <= 0 or ts[1] <= 0 or ts[2] <= 0 or
        ts[1] > std.math.maxInt(c_int) / 16 or ts[2] > std.math.maxInt(c_int) / 128 or
        xs[0] != ss[0] or ts[1] * 16 != xs[1] or
        mlx.mlx_array_dtype(xg) != .float16 or mlx.mlx_array_dtype(xu) != .float16 or
        mlx.mlx_array_dtype(tg) != .uint16 or mlx.mlx_array_dtype(tu) != .uint16 or
        (mlx.mlx_array_dtype(slots) != .uint32 and mlx.mlx_array_dtype(slots) != .int32)) return null;
    const rate = try packedRate(ts[3]);
    const output = ts[2] * 16;
    const key = IndexedKey{ .in_dim = xs[1], .out_dim = output, .topk = ss[0], .n = rate.n };
    const cfg = indexed_pair_coop_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const shape = [_]c_int{ ss[0], output };
        for (0..2) |_| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &shape, 2, .float16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, ts[2] * 128, ss[0], 2));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 128, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "IDIM", xs[1]));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ODIM", output));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "NHW", @intCast(rate.n)));
        indexed_pair_coop_cfgs.put(key, c);
        break :blk c;
    };
    const kernel = try codebookKernel(&indexed_pair_coop_kernel, "sushi_exl3_indexed_pair_coop", &.{ "xg", "xu", "tg", "tu", "slots" }, &.{ "yg", "yu" }, INDEXED_PAIR_COOP_SOURCE);
    const outputs = try applyOuts(s, kernel, &.{ xg, xu, tg, tu, slots }, cfg, 2);
    defer _ = mlx.mlx_vector_array_free(outputs);
    var result = [2]mlx.mlx_array{ mlx.mlx_array_new(), mlx.mlx_array_new() };
    errdefer for (result) |value| {
        _ = mlx.mlx_array_free(value);
    };
    for (&result, 0..) |*value, i| try mlx.check(mlx.mlx_vector_array_get(value, outputs, i));
    if (!indexed_pair_coop_engaged) {
        indexed_pair_coop_engaged = true;
        if (!@import("builtin").is_test) log.info("[exl3-decode] paired cooperative gate/up engaged inner=f16 k_splits=1\n", .{});
    }
    return result;
}

pub fn indexedGemvCoopF16(s: mlx.mlx_stream, x: mlx.mlx_array, trellis: mlx.mlx_array, slots: mlx.mlx_array) !mlx.mlx_array {
    const xsh = mlx.getShape(x);
    const tsh = mlx.getShape(trellis);
    const ssh = mlx.getShape(slots);
    if ((xsh.len != 1 and xsh.len != 2) or tsh.len != 4 or ssh.len != 1) return error.BadExl3Shape;
    const in_dim = xsh[xsh.len - 1];
    const out_dim = tsh[2] * 16;
    const out_tiles = tsh[2];
    const topk = ssh[0];
    const rate = try packedRate(tsh[3]);
    if (tsh[1] * 16 != in_dim) return error.BadExl3Shape;
    const key = IndexedKey{ .in_dim = in_dim, .out_dim = out_dim, .topk = topk, .n = rate.n };
    const cfg = indexed_coop_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const out_shape = [_]c_int{ topk, out_dim };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &out_shape, 2, .float16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, out_tiles * 128, topk, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 128, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "IDIM", in_dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ODIM", out_dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "NHW", @intCast(rate.n)));
        indexed_coop_cfgs.put(key, c);
        break :blk c;
    };
    const inputs_arr = [_]mlx.mlx_array{ x, trellis, slots };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    const kernel = try getIndexedCoopKernel();
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, cfg, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 0));
    return out;
}

const DOWN_FUSED_SOURCE: [:0]const u8 =
    \\threadgroup float partial[4 * 16 * uint(OTPT)];
    \\threadgroup half prepared[uint(IDIM)];
    \\const uint ot = uint(threadgroup_position_in_grid.x) * uint(OTPT);
    \\uint slot = uint(threadgroup_position_in_grid.y);
    \\uint split = uint(threadgroup_position_in_grid.z);
    \\uint sg = uint(simdgroup_index_in_threadgroup);
    \\uint lane = uint(thread_index_in_simdgroup);
    \\uint lid = uint(thread_index_in_threadgroup);
    \\constexpr uint TILE = 16u;
    \\constexpr uint N = uint(NHW);
    \\constexpr uint PACKED_HW = N;
    \\constexpr uint PACKED_W = N / 2u;
    \\constexpr uint IT = uint(IDIM) / TILE;
    \\constexpr uint OT = uint(ODIM) / TILE;
    \\constexpr uint SGS = 4u;
    \\constexpr uint SPLITS = 1u;
    \\const uint eid = uint(slots[slot]);
    \\const float sc = 0.08838834764831845f;
    \\const uint nblocks = uint(IDIM) / 128u;
    \\for (uint block = sg; block < nblocks; block += SGS) {
    \\  const uint base = block * 128u;
    \\  const size_t sb = (size_t)eid * (size_t)(IDIM) + base;
    \\  float4 v = float4(0.0f, 0.0f, 0.0f, 0.0f);
    \\  for (uint sp = 0u; sp < uint(NSPLIT); sp++) {
    \\    const size_t xb = ((size_t)slot * uint(NSPLIT) + sp) * (size_t)(IDIM) + base;
    \\    v += float4(float(ig[xb + lane]), float(ig[xb + lane + 32u]), float(ig[xb + lane + 64u]), float(ig[xb + lane + 96u]));
    \\  }
    \\  for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
    \\    const float p0 = simd_shuffle_xor(v.x, bit);
    \\    const float p1 = simd_shuffle_xor(v.y, bit);
    \\    const float p2 = simd_shuffle_xor(v.z, bit);
    \\    const float p3 = simd_shuffle_xor(v.w, bit);
    \\    const bool lower = (lane & bit) == 0u;
    \\    v.x = lower ? v.x + p0 : p0 - v.x;
    \\    v.y = lower ? v.y + p1 : p1 - v.y;
    \\    v.z = lower ? v.z + p2 : p2 - v.z;
    \\    v.w = lower ? v.w + p3 : p3 - v.w;
    \\  }
    \\  float s0 = v.x + v.y;
    \\  float s1 = v.x - v.y;
    \\  float s2 = v.z + v.w;
    \\  float s3 = v.z - v.w;
    \\  const float g0 = (s0 + s2) * sc * float(svhg[sb + lane]);
    \\  const float g1 = (s1 + s3) * sc * float(svhg[sb + lane + 32u]);
    \\  const float g2 = (s0 - s2) * sc * float(svhg[sb + lane + 64u]);
    \\  const float g3 = (s1 - s3) * sc * float(svhg[sb + lane + 96u]);
    \\  v = float4(0.0f, 0.0f, 0.0f, 0.0f);
    \\  for (uint sp = 0u; sp < uint(NSPLIT); sp++) {
    \\    const size_t xbu = ((size_t)slot * uint(NSPLIT) + sp) * (size_t)(IDIM) + base;
    \\    v += float4(float(iu[xbu + lane]), float(iu[xbu + lane + 32u]), float(iu[xbu + lane + 64u]), float(iu[xbu + lane + 96u]));
    \\  }
    \\  for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
    \\    const float p0 = simd_shuffle_xor(v.x, bit);
    \\    const float p1 = simd_shuffle_xor(v.y, bit);
    \\    const float p2 = simd_shuffle_xor(v.z, bit);
    \\    const float p3 = simd_shuffle_xor(v.w, bit);
    \\    const bool lower = (lane & bit) == 0u;
    \\    v.x = lower ? v.x + p0 : p0 - v.x;
    \\    v.y = lower ? v.y + p1 : p1 - v.y;
    \\    v.z = lower ? v.z + p2 : p2 - v.z;
    \\    v.w = lower ? v.w + p3 : p3 - v.w;
    \\  }
    \\  s0 = v.x + v.y;
    \\  s1 = v.x - v.y;
    \\  s2 = v.z + v.w;
    \\  s3 = v.z - v.w;
    \\  const float u0 = (s0 + s2) * sc * float(svhu[sb + lane]);
    \\  const float u1 = (s1 + s3) * sc * float(svhu[sb + lane + 32u]);
    \\  const float u2 = (s0 - s2) * sc * float(svhu[sb + lane + 64u]);
    \\  const float u3 = (s1 - s3) * sc * float(svhu[sb + lane + 96u]);
    \\  const float ysig0 = 1 / (1 + exp(abs(g0)));
    \\  const float ysig1 = 1 / (1 + exp(abs(g1)));
    \\  const float ysig2 = 1 / (1 + exp(abs(g2)));
    \\  const float ysig3 = 1 / (1 + exp(abs(g3)));
    \\  const float sig0 = (g0 < 0) ? ysig0 : 1 - ysig0;
    \\  const float sig1 = (g1 < 0) ? ysig1 : 1 - ysig1;
    \\  const float sig2 = (g2 < 0) ? ysig2 : 1 - ysig2;
    \\  const float sig3 = (g3 < 0) ? ysig3 : 1 - ysig3;
    \\  const float silu0 = g0 * sig0;
    \\  const float silu1 = g1 * sig1;
    \\  const float silu2 = g2 * sig2;
    \\  const float silu3 = g3 * sig3;
    \\  const float h0 = silu0 * u0;
    \\  const float h1 = silu1 * u1;
    \\  const float h2 = silu2 * u2;
    \\  const float h3 = silu3 * u3;
    \\  v = float4(h0 * float(suhd[sb + lane]), h1 * float(suhd[sb + lane + 32u]), h2 * float(suhd[sb + lane + 64u]), h3 * float(suhd[sb + lane + 96u]));
    \\  for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
    \\    const float p0 = simd_shuffle_xor(v.x, bit);
    \\    const float p1 = simd_shuffle_xor(v.y, bit);
    \\    const float p2 = simd_shuffle_xor(v.z, bit);
    \\    const float p3 = simd_shuffle_xor(v.w, bit);
    \\    const bool lower = (lane & bit) == 0u;
    \\    v.x = lower ? v.x + p0 : p0 - v.x;
    \\    v.y = lower ? v.y + p1 : p1 - v.y;
    \\    v.z = lower ? v.z + p2 : p2 - v.z;
    \\    v.w = lower ? v.w + p3 : p3 - v.w;
    \\  }
    \\  s0 = v.x + v.y;
    \\  s1 = v.x - v.y;
    \\  s2 = v.z + v.w;
    \\  s3 = v.z - v.w;
    \\  prepared[base + lane] = half((s0 + s2) * sc);
    \\  prepared[base + lane + 32u] = half((s1 + s3) * sc);
    \\  prepared[base + lane + 64u] = half((s0 - s2) * sc);
    \\  prepared[base + lane + 96u] = half((s1 - s3) * sc);
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\const uint tiles_per_split = (IT + SPLITS - 1u) / SPLITS;
    \\const uint tk0 = split * tiles_per_split;
    \\const uint tk1 = min(tk0 + tiles_per_split, IT);
    \\const uint prow = (lane & 3u) * 2u;
    \\const uint pcol = lane >> 2u;
    \\const uint row0 = prow;
    \\const uint row1 = prow + 1u;
    \\const uint row2 = prow + 8u;
    \\const uint row3 = prow + 9u;
    \\float acc[OTPT][8] = {};
    \\const device uint* trellis_e = (const device uint*)(trellis + ((size_t)eid * (size_t)IT * (size_t)OT) * PACKED_HW);
    \\if (N == 64u) {
    \\for (uint tk = tk0 + sg; tk < tk1; tk += SGS) {
    \\  const device uint* words = trellis_e + ((size_t)tk * (size_t)OT + ot) * 32u;
    \\  const ulong merged = ((ulong)words[(lane + 31u) & 31u] << 32) | (ulong)words[lane];
    \\  const float in0 = float(prepared[tk * TILE + row0]);
    \\  const float in1 = float(prepared[tk * TILE + row1]);
    \\  const float in2 = float(prepared[tk * TILE + row2]);
    \\  const float in3 = float(prepared[tk * TILE + row3]);
    \\  const uint sh[8] = {28u, 24u, 20u, 16u, 12u, 8u, 4u, 0u};
    \\  const float ins[8] = {in0, in1, in2, in3, in0, in1, in2, in3};
    \\  for (uint p = 0u; p < 4u; p++) {
    \\    const uint2 cw = uint2(uint(merged >> sh[p * 2u]), uint(merged >> sh[p * 2u + 1u])) & uint2(0xffffu);
    \\    const float2 w = exl3_decode2(cw);
    \\    acc[0][p * 2u] = fma(ins[p * 2u], w.x, acc[0][p * 2u]);
    \\    acc[0][p * 2u + 1u] = fma(ins[p * 2u + 1u], w.y, acc[0][p * 2u + 1u]);
    \\  }
    \\}
    \\}
    \\else if (FUNNEL) {
    \\  // Both k-tiles' words load before either decodes; a k range is whole H128
    \\  // blocks (8 tiles), so the second k-tile of an iteration is always in range.
    \\  const device uint *wp = trellis_e + ((size_t)(tk0 + sg) * (size_t)OT + ot) * PACKED_W;
    \\  const threadgroup half *pp = prepared + (tk0 + sg) * TILE;
    \\  const uint sh[8] = {exl3_lane_sh(N, 0u), exl3_lane_sh(N, 1u), exl3_lane_sh(N, 2u), exl3_lane_sh(N, 3u), exl3_lane_sh(N, 4u), exl3_lane_sh(N, 5u), exl3_lane_sh(N, 6u), exl3_lane_sh(N, 7u)};
    \\  for (uint tk = tk0 + sg; tk < tk1; tk += 2u * SGS) {
    \\    exl3_lane_bits<N> merged[2][OTPT];
    \\    for (uint u = 0u; u < 2u; u++) {
    \\      for (uint o = 0u; o < uint(OTPT); o++) {
    \\        const device uint *words = wp + u * SGS * OT * PACKED_W + o * PACKED_W;
    \\        merged[u][o] = exl3_lane<N>(words, lane);
    \\      }
    \\    }
    \\    for (uint u = 0u; u < 2u; u++) {
    \\      const float in0 = float(pp[u * SGS * TILE + row0]);
    \\      const float in1 = float(pp[u * SGS * TILE + row1]);
    \\      const float in2 = float(pp[u * SGS * TILE + row2]);
    \\      const float in3 = float(pp[u * SGS * TILE + row3]);
    \\      const float ins[8] = {in0, in1, in2, in3, in0, in1, in2, in3};
    \\      for (uint o = 0u; o < uint(OTPT); o++) {
    \\        for (uint p = 0u; p < 4u; p++) {
    \\          const uint2 cw = uint2(exl3_lane_word(merged[u][o], sh[p * 2u]), exl3_lane_word(merged[u][o], sh[p * 2u + 1u])) & uint2(0xffffu);
    \\          const float2 w = exl3_decode2(cw);
    \\          acc[o][p * 2u] = fma(ins[p * 2u], w.x, acc[o][p * 2u]);
    \\          acc[o][p * 2u + 1u] = fma(ins[p * 2u + 1u], w.y, acc[o][p * 2u + 1u]);
    \\        }
    \\      }
    \\    }
    \\    wp += 2u * SGS * OT * PACKED_W;
    \\    pp += 2u * SGS * TILE;
    \\  }
    \\} else {
    \\  uint w0[4];
    \\  uint w1[4];
    \\  uint shv[4];
    \\  uint frv[4];
    \\  for (uint p = 0u; p < 4u; p++) {
    \\    const exl3_win wv = exl3_pair_window(lane * 8u + p * 2u, N);
    \\    w0[p] = wv.i0;
    \\    w1[p] = wv.i1;
    \\    shv[p] = wv.sh;
    \\    frv[p] = wv.fresh;
    \\  }
    \\  for (uint tk = tk0 + sg; tk < tk1; tk += SGS) {
    \\    const device uint* words = trellis_e + ((size_t)tk * (size_t)OT + ot) * PACKED_W;
    \\    const float in0 = float(prepared[tk * TILE + row0]);
    \\    const float in1 = float(prepared[tk * TILE + row1]);
    \\    const float in2 = float(prepared[tk * TILE + row2]);
    \\    const float in3 = float(prepared[tk * TILE + row3]);
    \\    const float ins[8] = {in0, in1, in2, in3, in0, in1, in2, in3};
    \\    for (uint p = 0u; p < 4u; p++) {
    \\      const ulong merged = ((ulong)words[w0[p]] << 32) | (ulong)words[w1[p]];
    \\      const uint funnel = uint(merged >> shv[p]);
    \\      const uint2 cw = uint2((funnel >> frv[p]) & 0xffffu, funnel & 0xffffu);
    \\      const float2 w = exl3_decode2(cw);
    \\      acc[0][p * 2u] = fma(ins[p * 2u], w.x, acc[0][p * 2u]);
    \\      acc[0][p * 2u + 1u] = fma(ins[p * 2u + 1u], w.y, acc[0][p * 2u + 1u]);
    \\    }
    \\  }
    \\}
    \\for (uint o = 0u; o < uint(OTPT); o++) {
    \\  float clo = (acc[o][0] + acc[o][1]) + (acc[o][2] + acc[o][3]);
    \\  float chi = (acc[o][4] + acc[o][5]) + (acc[o][6] + acc[o][7]);
    \\  clo += simd_shuffle_xor(clo, 1u);
    \\  chi += simd_shuffle_xor(chi, 1u);
    \\  clo += simd_shuffle_xor(clo, 2u);
    \\  chi += simd_shuffle_xor(chi, 2u);
    \\  if ((lane & 3u) == 0u) {
    \\    partial[(o * SGS + sg) * 16u + pcol] = clo;
    \\    partial[(o * SGS + sg) * 16u + pcol + 8u] = chi;
    \\  }
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if (lid < 16u * uint(OTPT)) {
    \\  const uint o = lid >> 4u;
    \\  float sum = 0.0f;
    \\  for (uint g = 0u; g < SGS; g++) {
    \\    sum += partial[(o * SGS + g) * 16u + (lid & 15u)];
    \\  }
    \\  y[(size_t)slot * (size_t)(ODIM) + (ot + o) * TILE + (lid & 15u)] = half(sum);
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
;

pub fn downGemvFusedMid(
    s: mlx.mlx_stream,
    ig: mlx.mlx_array,
    iu: mlx.mlx_array,
    trellis: mlx.mlx_array,
    svhg: mlx.mlx_array,
    svhu: mlx.mlx_array,
    suhd: mlx.mlx_array,
    slots: mlx.mlx_array,
    in_dim: c_int,
    out_dim: c_int,
    nslots: c_int,
) !mlx.mlx_array {
    const tsh = mlx.getShape(trellis);
    const rate = try packedRate(tsh[tsh.len - 1]);
    const out_tiles = @divExact(out_dim, 16);
    const nsplit: c_int = @intCast(pairSplitCountFor(out_dim));
    const layout = gemvLayout(rate.n, out_tiles);
    const key = DownFusedKey{ .in_dim = in_dim, .out_dim = out_dim, .nslots = nslots, .nsplit = nsplit, .n = rate.n, .layout = layout };
    const cfg = down_fused_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const sh = [_]c_int{ nslots, out_dim };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &sh, 2, .float16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, @divExact(out_tiles, layout.tiles) * 128, nslots, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 128, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "IDIM", in_dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ODIM", out_dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "NSPLIT", nsplit));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "NHW", @intCast(rate.n)));
        try addGemvLayout(c, layout);
        down_fused_cfgs.put(key, c);
        break :blk c;
    };
    const ins = [_][*:0]const u8{ "ig", "iu", "trellis", "svhg", "svhu", "suhd", "slots" };
    const outs = [_][*:0]const u8{"y"};
    const kernel = try codebookKernel(&down_fused_kernel, "sushi_exl3_k4_down_fused", &ins, &outs, DOWN_FUSED_SOURCE);
    const ov = try applyOuts(s, kernel, &.{ ig, iu, trellis, svhg, svhu, suhd, slots }, cfg, 1);
    logFunnel(layout.funnel, rate.n, .fused_mid_down, mlx.mlx_array_dtype(ig));
    defer _ = mlx.mlx_vector_array_free(ov);
    var a = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(a);
    try mlx.check(mlx.mlx_vector_array_get(&a, ov, 0));
    return a;
}

pub fn projectIndexed(s: mlx.mlx_stream, x: mlx.mlx_array, trellis: mlx.mlx_array, suh: mlx.mlx_array, svh: mlx.mlx_array, slots: mlx.mlx_array) !mlx.mlx_array {
    const prepared = try prepareIndexed(s, x, suh, slots);
    defer _ = mlx.mlx_array_free(prepared);
    const inner = try indexedGemvCoopF16(s, prepared, trellis, slots);
    defer _ = mlx.mlx_array_free(inner);
    return finishIndexed(s, inner, svh, slots);
}

/// The suh scale + H128 Hadamard that feeds this GEMV is applied here, over the
/// k range this threadgroup owns, rather than in a dispatch of its own: the tile
/// loop then reads its operands from threadgroup memory instead of device.
const PAIR_GEMV_SOURCE: [:0]const u8 =
    \\threadgroup float partial[4 * 16 * uint(OTPT)];
    \\const uint ot = uint(threadgroup_position_in_grid.x) * uint(OTPT);
    \\uint slot = uint(threadgroup_position_in_grid.y);
    \\uint split = uint(threadgroup_position_in_grid.z);
    \\uint sg = uint(simdgroup_index_in_threadgroup);
    \\uint lane = uint(thread_index_in_simdgroup);
    \\uint lid = uint(thread_index_in_threadgroup);
    \\constexpr uint TILE = 16u;
    \\constexpr uint N = uint(NHW);
    \\constexpr uint PACKED_HW = N;
    \\constexpr uint PACKED_W = N / 2u;
    \\constexpr uint IT = uint(IDIM) / TILE;
    \\constexpr uint OT = uint(ODIM) / TILE;
    \\constexpr uint SGS = 4u;
    \\const uint tiles_per_split = (IT + uint(NSPLIT) - 1u) / uint(NSPLIT);
    \\const uint tk0 = split * tiles_per_split;
    \\const uint tk1 = min(tk0 + tiles_per_split, IT);
    \\const uint eid = uint(slots[slot]);
    \\constexpr uint KSPAN = uint(IDIM) / uint(NSPLIT);
    \\threadgroup half prepared[2u * KSPAN];
    \\{
    \\  const uint xrow = slot / uint(TOPK);
    \\  const float psc = 0.08838834764831845f;
    \\  for (uint b = sg; b < KSPAN / 128u; b += SGS) {
    \\    const uint pbase = tk0 * TILE + b * 128u;
    \\    const size_t xr = (size_t)xrow * (size_t)(IDIM) + pbase;
    \\    const size_t sr = (size_t)eid * (size_t)(IDIM) + pbase;
    \\    const float x0 = float(x[xr + lane]);
    \\    const float x1 = float(x[xr + lane + 32u]);
    \\    const float x2 = float(x[xr + lane + 64u]);
    \\    const float x3 = float(x[xr + lane + 96u]);
    \\    for (uint pj = 0u; pj < 2u; pj++) {
    \\      const device half *suh = (pj == 0u) ? suhg : suhu;
    \\      float4 v = float4(
    \\        x0 * float(suh[sr + lane]),
    \\        x1 * float(suh[sr + lane + 32u]),
    \\        x2 * float(suh[sr + lane + 64u]),
    \\        x3 * float(suh[sr + lane + 96u]));
    \\      for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
    \\        const float p0 = simd_shuffle_xor(v.x, bit);
    \\        const float p1 = simd_shuffle_xor(v.y, bit);
    \\        const float p2 = simd_shuffle_xor(v.z, bit);
    \\        const float p3 = simd_shuffle_xor(v.w, bit);
    \\        const bool lower = (lane & bit) == 0u;
    \\        v.x = lower ? v.x + p0 : p0 - v.x;
    \\        v.y = lower ? v.y + p1 : p1 - v.y;
    \\        v.z = lower ? v.z + p2 : p2 - v.z;
    \\        v.w = lower ? v.w + p3 : p3 - v.w;
    \\      }
    \\      const float s0 = v.x + v.y;
    \\      const float s1 = v.x - v.y;
    \\      const float s2 = v.z + v.w;
    \\      const float s3 = v.z - v.w;
    \\      const uint pw = pj * KSPAN + b * 128u;
    \\      prepared[pw + lane] = half((s0 + s2) * psc);
    \\      prepared[pw + lane + 32u] = half((s1 + s3) * psc);
    \\      prepared[pw + lane + 64u] = half((s0 - s2) * psc);
    \\      prepared[pw + lane + 96u] = half((s1 - s3) * psc);
    \\    }
    \\  }
    \\}
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\const uint prow = (lane & 3u) * 2u;
    \\const uint pcol = lane >> 2u;
    \\const uint row0 = prow;
    \\const uint row1 = prow + 1u;
    \\const uint row2 = prow + 8u;
    \\const uint row3 = prow + 9u;
    \\for (uint proj = 0u; proj < 2u; proj++) {
    \\  const uint xb = proj * KSPAN;
    \\  const device ushort *trellis = (proj == 0u) ? tg : tu;
    \\  device float *y = (proj == 0u) ? yg : yu;
    \\  float acc[OTPT][8] = {};
    \\  const device uint *trellis_e = (const device uint *)(trellis + ((size_t)eid * (size_t)IT * (size_t)OT) * PACKED_HW);
    \\  if (N == 64u) {
    \\  for (uint tk = tk0 + sg; tk < tk1; tk += SGS) {
    \\    const device uint *words = trellis_e + ((size_t)tk * (size_t)OT + ot) * 32u;
    \\    const ulong merged = ((ulong)words[(lane + 31u) & 31u] << 32) | (ulong)words[lane];
    \\    const float in0 = float(prepared[xb + (tk - tk0) * TILE + row0]);
    \\    const float in1 = float(prepared[xb + (tk - tk0) * TILE + row1]);
    \\    const float in2 = float(prepared[xb + (tk - tk0) * TILE + row2]);
    \\    const float in3 = float(prepared[xb + (tk - tk0) * TILE + row3]);
    \\    const uint sh[8] = {28u, 24u, 20u, 16u, 12u, 8u, 4u, 0u};
    \\    const float ins[8] = {in0, in1, in2, in3, in0, in1, in2, in3};
    \\    for (uint p = 0u; p < 4u; p++) {
    \\      const uint2 cw = uint2(uint(merged >> sh[p * 2u]), uint(merged >> sh[p * 2u + 1u])) & uint2(0xffffu);
    \\      const float2 w = exl3_decode2(cw);
    \\      acc[0][p * 2u] = fma(ins[p * 2u], w.x, acc[0][p * 2u]);
    \\      acc[0][p * 2u + 1u] = fma(ins[p * 2u + 1u], w.y, acc[0][p * 2u + 1u]);
    \\    }
    \\  }
    \\  }
    \\  else if (FUNNEL) {
    \\    // Both k-tiles' words load before either decodes; a k range is whole H128
    \\    // blocks (8 tiles), so the second k-tile of an iteration is always in range.
    \\    const device uint *wp = trellis_e + ((size_t)(tk0 + sg) * (size_t)OT + ot) * PACKED_W;
    \\    const threadgroup half *pp = prepared + xb + sg * TILE;
    \\    const uint sh[8] = {exl3_lane_sh(N, 0u), exl3_lane_sh(N, 1u), exl3_lane_sh(N, 2u), exl3_lane_sh(N, 3u), exl3_lane_sh(N, 4u), exl3_lane_sh(N, 5u), exl3_lane_sh(N, 6u), exl3_lane_sh(N, 7u)};
    \\    for (uint tk = tk0 + sg; tk < tk1; tk += 2u * SGS) {
    \\      exl3_lane_bits<N> merged[2][OTPT];
    \\      for (uint u = 0u; u < 2u; u++) {
    \\        for (uint o = 0u; o < uint(OTPT); o++) {
    \\          const device uint *words = wp + u * SGS * OT * PACKED_W + o * PACKED_W;
    \\          merged[u][o] = exl3_lane<N>(words, lane);
    \\        }
    \\      }
    \\      for (uint u = 0u; u < 2u; u++) {
    \\        const float in0 = float(pp[u * SGS * TILE + row0]);
    \\        const float in1 = float(pp[u * SGS * TILE + row1]);
    \\        const float in2 = float(pp[u * SGS * TILE + row2]);
    \\        const float in3 = float(pp[u * SGS * TILE + row3]);
    \\        const float ins[8] = {in0, in1, in2, in3, in0, in1, in2, in3};
    \\        for (uint o = 0u; o < uint(OTPT); o++) {
    \\          for (uint p = 0u; p < 4u; p++) {
    \\            const uint2 cw = uint2(exl3_lane_word(merged[u][o], sh[p * 2u]), exl3_lane_word(merged[u][o], sh[p * 2u + 1u])) & uint2(0xffffu);
    \\            const float2 w = exl3_decode2(cw);
    \\            acc[o][p * 2u] = fma(ins[p * 2u], w.x, acc[o][p * 2u]);
    \\            acc[o][p * 2u + 1u] = fma(ins[p * 2u + 1u], w.y, acc[o][p * 2u + 1u]);
    \\          }
    \\        }
    \\      }
    \\      wp += 2u * SGS * OT * PACKED_W;
    \\      pp += 2u * SGS * TILE;
    \\    }
    \\  } else {
    \\    uint w0[4];
    \\    uint w1[4];
    \\    uint shv[4];
    \\    uint frv[4];
    \\    for (uint p = 0u; p < 4u; p++) {
    \\      const exl3_win wv = exl3_pair_window(lane * 8u + p * 2u, N);
    \\      w0[p] = wv.i0;
    \\      w1[p] = wv.i1;
    \\      shv[p] = wv.sh;
    \\      frv[p] = wv.fresh;
    \\    }
    \\    for (uint tk = tk0 + sg; tk < tk1; tk += SGS) {
    \\      const device uint *words = trellis_e + ((size_t)tk * (size_t)OT + ot) * PACKED_W;
    \\      const float in0 = float(prepared[xb + (tk - tk0) * TILE + row0]);
    \\      const float in1 = float(prepared[xb + (tk - tk0) * TILE + row1]);
    \\      const float in2 = float(prepared[xb + (tk - tk0) * TILE + row2]);
    \\      const float in3 = float(prepared[xb + (tk - tk0) * TILE + row3]);
    \\      const float ins[8] = {in0, in1, in2, in3, in0, in1, in2, in3};
    \\      for (uint p = 0u; p < 4u; p++) {
    \\        const ulong merged = ((ulong)words[w0[p]] << 32) | (ulong)words[w1[p]];
    \\        const uint funnel = uint(merged >> shv[p]);
    \\        const uint2 cw = uint2((funnel >> frv[p]) & 0xffffu, funnel & 0xffffu);
    \\        const float2 w = exl3_decode2(cw);
    \\        acc[0][p * 2u] = fma(ins[p * 2u], w.x, acc[0][p * 2u]);
    \\        acc[0][p * 2u + 1u] = fma(ins[p * 2u + 1u], w.y, acc[0][p * 2u + 1u]);
    \\      }
    \\    }
    \\  }
    \\  for (uint o = 0u; o < uint(OTPT); o++) {
    \\    float clo = (acc[o][0] + acc[o][1]) + (acc[o][2] + acc[o][3]);
    \\    float chi = (acc[o][4] + acc[o][5]) + (acc[o][6] + acc[o][7]);
    \\    clo += simd_shuffle_xor(clo, 1u);
    \\    chi += simd_shuffle_xor(chi, 1u);
    \\    clo += simd_shuffle_xor(clo, 2u);
    \\    chi += simd_shuffle_xor(chi, 2u);
    \\    if ((lane & 3u) == 0u) {
    \\      partial[(o * SGS + sg) * 16u + pcol] = clo;
    \\      partial[(o * SGS + sg) * 16u + pcol + 8u] = chi;
    \\    }
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\  if (lid < 16u * uint(OTPT)) {
    \\    const uint o = lid >> 4u;
    \\    float sum = 0.0f;
    \\    for (uint g = 0u; g < SGS; g++) {
    \\      sum += partial[(o * SGS + g) * 16u + (lid & 15u)];
    \\    }
    \\    y[(size_t)(slot * uint(NSPLIT) + split) * (size_t)(ODIM) + (ot + o) * TILE + (lid & 15u)] = sum;
    \\  }
    \\  threadgroup_barrier(mem_flags::mem_threadgroup);
    \\}
;

const MID_SOURCE: [:0]const u8 =
    \\uint block = uint(threadgroup_position_in_grid.x);
    \\uint slot = uint(threadgroup_position_in_grid.y);
    \\ushort lane = thread_index_in_simdgroup;
    \\const uint eid = uint(slots[slot]);
    \\const uint base = block * 128u;
    \\const size_t xb = (size_t)slot * (size_t)(ODIM) + base;
    \\const size_t sb = (size_t)eid * (size_t)(ODIM) + base;
    \\const float sc = 0.08838834764831845f;
    \\float4 v = float4(float(ig[xb + lane]), float(ig[xb + lane + 32u]), float(ig[xb + lane + 64u]), float(ig[xb + lane + 96u]));
    \\for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
    \\  const float p0 = simd_shuffle_xor(v.x, bit);
    \\  const float p1 = simd_shuffle_xor(v.y, bit);
    \\  const float p2 = simd_shuffle_xor(v.z, bit);
    \\  const float p3 = simd_shuffle_xor(v.w, bit);
    \\  const bool lower = (lane & bit) == 0u;
    \\  v.x = lower ? v.x + p0 : p0 - v.x;
    \\  v.y = lower ? v.y + p1 : p1 - v.y;
    \\  v.z = lower ? v.z + p2 : p2 - v.z;
    \\  v.w = lower ? v.w + p3 : p3 - v.w;
    \\}
    \\float s0 = v.x + v.y;
    \\float s1 = v.x - v.y;
    \\float s2 = v.z + v.w;
    \\float s3 = v.z - v.w;
    \\float g0 = (s0 + s2) * sc * float(svhg[sb + lane]);
    \\float g1 = (s1 + s3) * sc * float(svhg[sb + lane + 32u]);
    \\float g2 = (s0 - s2) * sc * float(svhg[sb + lane + 64u]);
    \\float g3 = (s1 - s3) * sc * float(svhg[sb + lane + 96u]);
    \\v = float4(float(iu[xb + lane]), float(iu[xb + lane + 32u]), float(iu[xb + lane + 64u]), float(iu[xb + lane + 96u]));
    \\for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
    \\  const float p0 = simd_shuffle_xor(v.x, bit);
    \\  const float p1 = simd_shuffle_xor(v.y, bit);
    \\  const float p2 = simd_shuffle_xor(v.z, bit);
    \\  const float p3 = simd_shuffle_xor(v.w, bit);
    \\  const bool lower = (lane & bit) == 0u;
    \\  v.x = lower ? v.x + p0 : p0 - v.x;
    \\  v.y = lower ? v.y + p1 : p1 - v.y;
    \\  v.z = lower ? v.z + p2 : p2 - v.z;
    \\  v.w = lower ? v.w + p3 : p3 - v.w;
    \\}
    \\s0 = v.x + v.y;
    \\s1 = v.x - v.y;
    \\s2 = v.z + v.w;
    \\s3 = v.z - v.w;
    \\float u0 = (s0 + s2) * sc * float(svhu[sb + lane]);
    \\float u1 = (s1 + s3) * sc * float(svhu[sb + lane + 32u]);
    \\float u2 = (s0 - s2) * sc * float(svhu[sb + lane + 64u]);
    \\float u3 = (s1 - s3) * sc * float(svhu[sb + lane + 96u]);
    \\if constexpr (CLAMP_LIMIT > 0) {
    \\  g0 = min(g0, float(CLAMP_LIMIT)); g1 = min(g1, float(CLAMP_LIMIT));
    \\  g2 = min(g2, float(CLAMP_LIMIT)); g3 = min(g3, float(CLAMP_LIMIT));
    \\  u0 = clamp(u0, -float(CLAMP_LIMIT), float(CLAMP_LIMIT));
    \\  u1 = clamp(u1, -float(CLAMP_LIMIT), float(CLAMP_LIMIT));
    \\  u2 = clamp(u2, -float(CLAMP_LIMIT), float(CLAMP_LIMIT));
    \\  u3 = clamp(u3, -float(CLAMP_LIMIT), float(CLAMP_LIMIT));
    \\}
    \\const float ysig0 = 1 / (1 + exp(abs(g0)));
    \\const float ysig1 = 1 / (1 + exp(abs(g1)));
    \\const float ysig2 = 1 / (1 + exp(abs(g2)));
    \\const float ysig3 = 1 / (1 + exp(abs(g3)));
    \\const float sig0 = (g0 < 0) ? ysig0 : 1 - ysig0;
    \\const float sig1 = (g1 < 0) ? ysig1 : 1 - ysig1;
    \\const float sig2 = (g2 < 0) ? ysig2 : 1 - ysig2;
    \\const float sig3 = (g3 < 0) ? ysig3 : 1 - ysig3;
    \\const float silu0 = g0 * sig0;
    \\const float silu1 = g1 * sig1;
    \\const float silu2 = g2 * sig2;
    \\const float silu3 = g3 * sig3;
    \\const float h0 = silu0 * u0;
    \\const float h1 = silu1 * u1;
    \\const float h2 = silu2 * u2;
    \\const float h3 = silu3 * u3;
    \\v = float4(h0 * float(suhd[sb + lane]), h1 * float(suhd[sb + lane + 32u]), h2 * float(suhd[sb + lane + 64u]), h3 * float(suhd[sb + lane + 96u]));
    \\for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
    \\  const float p0 = simd_shuffle_xor(v.x, bit);
    \\  const float p1 = simd_shuffle_xor(v.y, bit);
    \\  const float p2 = simd_shuffle_xor(v.z, bit);
    \\  const float p3 = simd_shuffle_xor(v.w, bit);
    \\  const bool lower = (lane & bit) == 0u;
    \\  v.x = lower ? v.x + p0 : p0 - v.x;
    \\  v.y = lower ? v.y + p1 : p1 - v.y;
    \\  v.z = lower ? v.z + p2 : p2 - v.z;
    \\  v.w = lower ? v.w + p3 : p3 - v.w;
    \\}
    \\s0 = v.x + v.y;
    \\s1 = v.x - v.y;
    \\s2 = v.z + v.w;
    \\s3 = v.z - v.w;
    \\yd[xb + lane] = half((s0 + s2) * sc);
    \\yd[xb + lane + 32u] = half((s1 + s3) * sc);
    \\yd[xb + lane + 64u] = half((s0 - s2) * sc);
    \\yd[xb + lane + 96u] = half((s1 - s3) * sc);
;

const REDUCE_SOURCE: [:0]const u8 =
    \\threadgroup float vals[uint(TOPK) * 128u];
    \\uint block = uint(threadgroup_position_in_grid.x);
    \\uint row = uint(threadgroup_position_in_grid.y);
    \\uint sg = uint(simdgroup_index_in_threadgroup);
    \\ushort lane = thread_index_in_simdgroup;
    \\const uint base = block * 128u;
    \\const uint slot = row * uint(TOPK) + sg;
    \\const uint eid = uint(slots[slot]);
    \\const size_t xb = (size_t)slot * (size_t)(ODIM) + base;
    \\const size_t sb = (size_t)eid * (size_t)(ODIM) + base;
    \\const float scv = 0.08838834764831845f;
    \\float4 v = float4(float(inner[xb + lane]), float(inner[xb + lane + 32u]), float(inner[xb + lane + 64u]), float(inner[xb + lane + 96u]));
    \\for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
    \\  const float p0 = simd_shuffle_xor(v.x, bit);
    \\  const float p1 = simd_shuffle_xor(v.y, bit);
    \\  const float p2 = simd_shuffle_xor(v.z, bit);
    \\  const float p3 = simd_shuffle_xor(v.w, bit);
    \\  const bool lower = (lane & bit) == 0u;
    \\  v.x = lower ? v.x + p0 : p0 - v.x;
    \\  v.y = lower ? v.y + p1 : p1 - v.y;
    \\  v.z = lower ? v.z + p2 : p2 - v.z;
    \\  v.w = lower ? v.w + p3 : p3 - v.w;
    \\}
    \\const float s0 = v.x + v.y;
    \\const float s1 = v.x - v.y;
    \\const float s2 = v.z + v.w;
    \\const float s3 = v.z - v.w;
    \\vals[sg * 128u + lane] = (s0 + s2) * scv * float(svh[sb + lane]);
    \\vals[sg * 128u + lane + 32u] = (s1 + s3) * scv * float(svh[sb + lane + 32u]);
    \\vals[sg * 128u + lane + 64u] = (s0 - s2) * scv * float(svh[sb + lane + 64u]);
    \\vals[sg * 128u + lane + 96u] = (s1 - s3) * scv * float(svh[sb + lane + 96u]);
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\if (sg == 0u) {
    \\  float a0 = 0.0f;
    \\  float a1 = 0.0f;
    \\  float a2 = 0.0f;
    \\  float a3 = 0.0f;
    \\  for (uint k = 0u; k < uint(TOPK); k++) {
    \\    const float w = float(sc[row * uint(TOPK) + k]);
    \\    a0 += vals[k * 128u + lane] * w;
    \\    a1 += vals[k * 128u + lane + 32u] * w;
    \\    a2 += vals[k * 128u + lane + 64u] * w;
    \\    a3 += vals[k * 128u + lane + 96u] * w;
    \\  }
    \\  const size_t yb = (size_t)row * (size_t)(ODIM) + base;
    \\  y[yb + lane] = T(a0);
    \\  y[yb + lane + 32u] = T(a1);
    \\  y[yb + lane + 64u] = T(a2);
    \\  y[yb + lane + 96u] = T(a3);
    \\}
;

const REDUCE_SHARED_SOURCE: [:0]const u8 = blk: {
    const stores =
        \\  y[yb + lane] = T(a0);
        \\  y[yb + lane + 32u] = T(a1);
        \\  y[yb + lane + 64u] = T(a2);
        \\  y[yb + lane + 96u] = T(a3);
    ;
    const at = std.mem.indexOf(u8, REDUCE_SOURCE, stores).?;
    // Preserve the separate routed store's rounding before the shared addition.
    break :blk REDUCE_SOURCE[0..at] ++
        \\  y[yb + lane] = T(float(T(a0)) + float(shared[yb + lane]));
        \\  y[yb + lane + 32u] = T(float(T(a1)) + float(shared[yb + lane + 32u]));
        \\  y[yb + lane + 64u] = T(float(T(a2)) + float(shared[yb + lane + 64u]));
        \\  y[yb + lane + 96u] = T(float(T(a3)) + float(shared[yb + lane + 96u]));
    ++ REDUCE_SOURCE[at + stores.len ..];
};

const FunnelArm = enum { pair, fused_mid_down, prepared_down, nax, simdmat };
var funnel_engaged: [5]bool = @splat(false);

fn logFunnel(on: bool, n: u32, arm: FunnelArm, dtype: mlx.mlx_dtype) void {
    if (!on or funnel_engaged[@backingInt(arm)]) return;
    funnel_engaged[@backingInt(arm)] = true;
    log.info("[exl3] n{d} funnel engaged arm={s} codebook={s} dtype={s} window={d}\n", .{ n, @tagName(arm), @tagName(active_decode.codebook), @tagName(dtype), active_decode.window.bits() });
}

var pair_gemv_kernel: KernelSlots = no_kernels;
var pair_gemv_grouped_kernel: KernelSlots = no_kernels;
var mid_kernel: ?mlx.mlx_fast_metal_kernel = null;
var reduce_kernel: ?mlx.mlx_fast_metal_kernel = null;
var reduce_shared_kernel: ?mlx.mlx_fast_metal_kernel = null;
var down_fused_kernel: KernelSlots = no_kernels;

fn getNamedKernel(slot: *?mlx.mlx_fast_metal_kernel, name: [*:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [:0]const u8, header: [:0]const u8) !mlx.mlx_fast_metal_kernel {
    if (slot.*) |k| return k;
    const in_vec = mlx.mlx_vector_string_new_data(ins.ptr, ins.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(outs.ptr, outs.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(name, in_vec, out_vec, source, header.ptr, true, false);
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    slot.* = kernel;
    return kernel;
}

fn applyOuts(s: mlx.mlx_stream, kernel: mlx.mlx_fast_metal_kernel, inputs: []const mlx.mlx_array, cfg: mlx.mlx_fast_metal_kernel_config, n_out: usize) !mlx.mlx_vector_array {
    fused_dispatches += 1;
    const inputs_vec = mlx.mlx_vector_array_new_data(inputs.ptr, inputs.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    var outputs_vec = mlx.mlx_vector_array_new();
    errdefer _ = mlx.mlx_vector_array_free(outputs_vec);
    const host_on = applyUbenchOn();
    const io = std.Io.Threaded.global_single_threaded.io();
    var sw = if (host_on) io_util.Stopwatch.init(io) else undefined;
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, cfg, s));
    if (host_on) {
        apply_host_ns += sw.read();
        apply_host_n += 1;
    }
    if (mlx.mlx_vector_array_size(outputs_vec) != n_out) return error.MetalKernelBadOutputCount;
    return outputs_vec;
}

/// `group_ask` > 0 asks for the expert-grouped kernel; only the lane funnel has one.
fn pairGemvConfig(in_dim: c_int, out_dim: c_int, nslots: c_int, topk: c_int, rate: exl3.Rate, layout: GemvLayout, group: c_int) !mlx.mlx_fast_metal_kernel_config {
    const out_tiles = @divExact(out_dim, 16);
    const nsplit: c_int = @intCast(pairSplitCountFor(in_dim));
    const key = PairGemvKey{ .in_dim = in_dim, .out_dim = out_dim, .nslots = nslots, .nsplit = nsplit, .topk = topk, .n = rate.n, .layout = layout, .group = group };
    if (pair_gemv_cfgs.get(key)) |c| return c;
    const c = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
    const sh = [_]c_int{ nslots * nsplit, out_dim };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &sh, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &sh, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, @divExact(out_tiles, layout.tiles) * 128, nslots, nsplit));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 128, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "IDIM", in_dim));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ODIM", out_dim));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "NSPLIT", nsplit));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "NHW", @intCast(rate.n)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "TOPK", topk));
    try addGemvLayout(c, layout);
    try addGroup(c, group, nslots);
    pair_gemv_cfgs.put(key, c);
    return c;
}

pub fn pairGemv(s: mlx.mlx_stream, x: mlx.mlx_array, suhg: mlx.mlx_array, suhu: mlx.mlx_array, tg: mlx.mlx_array, tu: mlx.mlx_array, slots: mlx.mlx_array, in_dim: c_int, out_dim: c_int, nslots: c_int, topk: c_int, group_ask: c_int) !struct { mlx.mlx_array, mlx.mlx_array } {
    const tsh = mlx.getShape(tg);
    const ush = mlx.getShape(tu);
    const rate = try packedRate(tsh[tsh.len - 1]);
    if (ush[ush.len - 1] != tsh[tsh.len - 1]) return error.BadExl3Shape;
    const layout = gemvLayout(rate.n, @divExact(out_dim, 16));
    const group: c_int = if (layout.funnel) group_ask else 0;
    const cfg = try pairGemvConfig(in_dim, out_dim, nslots, topk, rate, layout, group);
    const ins = [_][*:0]const u8{ "x", "suhg", "suhu", "tg", "tu", "slots" };
    const outs = [_][*:0]const u8{ "yg", "yu" };
    const kernel = if (group > 0)
        try codebookKernel(&pair_gemv_grouped_kernel, "sushi_exl3_pair_gemv_grouped", &ins, &outs, PAIR_GEMV_GROUPED_SOURCE)
    else
        try codebookKernel(&pair_gemv_kernel, "sushi_exl3_pair_gemv", &ins, &outs, PAIR_GEMV_SOURCE);
    logGroupedEngaged(group);
    const ov = try applyOuts(s, kernel, &.{ x, suhg, suhu, tg, tu, slots }, cfg, 2);
    logFunnel(layout.funnel, rate.n, .pair, mlx.mlx_array_dtype(x));
    defer _ = mlx.mlx_vector_array_free(ov);
    var a = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(a);
    var b = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(b);
    try mlx.check(mlx.mlx_vector_array_get(&a, ov, 0));
    try mlx.check(mlx.mlx_vector_array_get(&b, ov, 1));
    return .{ a, b };
}

fn midSwigluPrep(s: mlx.mlx_stream, ig: mlx.mlx_array, iu: mlx.mlx_array, svhg: mlx.mlx_array, svhu: mlx.mlx_array, suhd: mlx.mlx_array, slots: mlx.mlx_array, dim: c_int, nslots: c_int) !mlx.mlx_array {
    return midSwigluPrepWithLimit(s, ig, iu, svhg, svhu, suhd, slots, dim, nslots, 0);
}

fn midSwigluPrepWithLimit(s: mlx.mlx_stream, ig: mlx.mlx_array, iu: mlx.mlx_array, svhg: mlx.mlx_array, svhu: mlx.mlx_array, suhd: mlx.mlx_array, slots: mlx.mlx_array, dim: c_int, nslots: c_int, limit: c_int) !mlx.mlx_array {
    const key = MidKey{ .dim = dim, .nslots = nslots, .limit = limit };
    const cfg = mid_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const sh = [_]c_int{ nslots, dim };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &sh, 2, .float16));
        const blocks = @divExact(dim, 128);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, 32 * blocks, nslots, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ODIM", dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "CLAMP_LIMIT", limit));
        mid_cfgs.put(key, c);
        break :blk c;
    };
    const ins = [_][*:0]const u8{ "ig", "iu", "svhg", "svhu", "suhd", "slots" };
    const outs = [_][*:0]const u8{"yd"};
    const kernel = try getNamedKernel(&mid_kernel, "sushi_exl3_mid_swiglu", &ins, &outs, MID_SOURCE, "");
    const ov = try applyOuts(s, kernel, &.{ ig, iu, svhg, svhu, suhd, slots }, cfg, 1);
    defer _ = mlx.mlx_vector_array_free(ov);
    var a = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(a);
    try mlx.check(mlx.mlx_vector_array_get(&a, ov, 0));
    return a;
}

/// One simdgroup per (row, k) slot, so the threadgroup cannot hold more slots
/// than Metal allows threads.
pub const REDUCE_MAX_TOPK: c_int = 32;

pub fn downFinishReduce(s: mlx.mlx_stream, inner: mlx.mlx_array, svh: mlx.mlx_array, slots: mlx.mlx_array, scores: mlx.mlx_array, out_dim: c_int, rows: c_int, topk: c_int, out_dtype: mlx.mlx_dtype) !mlx.mlx_array {
    return downFinishReduceWithShared(s, inner, svh, slots, scores, out_dim, rows, topk, out_dtype, null);
}

fn downFinishReduceWithShared(s: mlx.mlx_stream, inner: mlx.mlx_array, svh: mlx.mlx_array, slots: mlx.mlx_array, scores: mlx.mlx_array, out_dim: c_int, rows: c_int, topk: c_int, out_dtype: mlx.mlx_dtype, shared: ?mlx.mlx_array) !mlx.mlx_array {
    if (topk < 1 or topk > REDUCE_MAX_TOPK) return error.Exl3TopkUnsupported;
    const key = DecodeReduceKey{ .out_dim = out_dim, .rows = rows, .topk = topk, .dtype = out_dtype };
    const cfg = reduce_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const sh = if (rows == 1) [_]c_int{out_dim} ++ [_]c_int{0} else [_]c_int{ rows, out_dim };
        const ndim: usize = if (rows == 1) 1 else 2;
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &sh, ndim, out_dtype));
        const blocks = @divExact(out_dim, 128);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, 32 * topk * blocks, rows, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32 * topk, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "T", out_dtype));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ODIM", out_dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "TOPK", topk));
        reduce_cfgs.put(key, c);
        break :blk c;
    };
    const ins = [_][*:0]const u8{ "inner", "svh", "slots", "sc" };
    const outs = [_][*:0]const u8{"y"};
    const ov = if (shared) |value| blk: {
        const shared_ins = ins ++ [_][*:0]const u8{"shared"};
        const kernel = try getNamedKernel(&reduce_shared_kernel, "sushi_exl3_down_reduce_shared", &shared_ins, &outs, REDUCE_SHARED_SOURCE, "");
        break :blk try applyOuts(s, kernel, &.{ inner, svh, slots, scores, value }, cfg, 1);
    } else blk: {
        const kernel = try getNamedKernel(&reduce_kernel, "sushi_exl3_down_reduce", &ins, &outs, REDUCE_SOURCE, "");
        break :blk try applyOuts(s, kernel, &.{ inner, svh, slots, scores }, cfg, 1);
    };
    defer _ = mlx.mlx_vector_array_free(ov);
    var a = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(a);
    try mlx.check(mlx.mlx_vector_array_get(&a, ov, 0));
    return a;
}

pub fn downFinishReduceShared(s: mlx.mlx_stream, inner: mlx.mlx_array, svh: mlx.mlx_array, slots: mlx.mlx_array, scores: mlx.mlx_array, out_dim: c_int, rows: c_int, topk: c_int, out_dtype: mlx.mlx_dtype, shared: mlx.mlx_array) !mlx.mlx_array {
    return downFinishReduceWithShared(s, inner, svh, slots, scores, out_dim, rows, topk, out_dtype, shared);
}

fn prepareFromTokens(s: mlx.mlx_stream, x: mlx.mlx_array, suh: mlx.mlx_array, slots: mlx.mlx_array, order: mlx.mlx_array, in_dim: c_int, nslots: c_int, topk: c_int) !mlx.mlx_array {
    const key = PairPrepKey{ .in_dim = in_dim, .nslots = nslots, .topk = topk };
    const cfg = token_prep_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const sh = [_]c_int{ nslots, in_dim };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &sh, 2, .float16));
        const blocks = @divExact(in_dim, 128);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, 32 * blocks, nslots, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "IDIM", in_dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "TOPK", topk));
        token_prep_cfgs.put(key, c);
        break :blk c;
    };
    const ins = [_][*:0]const u8{ "x", "suh", "slots", "order" };
    const outs = [_][*:0]const u8{"y"};
    const kernel = try getNamedKernel(&token_prepare_kernel, "sushi_exl3_token_prepare", &ins, &outs, TOKEN_PREPARE_SOURCE, "");
    const ov = try applyOuts(s, kernel, &.{ x, suh, slots, order }, cfg, 1);
    defer _ = mlx.mlx_vector_array_free(ov);
    var a = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(a);
    try mlx.check(mlx.mlx_vector_array_get(&a, ov, 0));
    return a;
}

fn tokenReduce(s: mlx.mlx_stream, d: mlx.mlx_array, inv: mlx.mlx_array, scores: mlx.mlx_array, out_dim: c_int, rows: c_int, topk: c_int) !mlx.mlx_array {
    const key = ReduceKey{ .out_dim = out_dim, .rows = rows, .topk = topk };
    const cfg = token_reduce_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const sh = [_]c_int{ rows, out_dim };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &sh, 2, .float16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, out_dim, rows, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ODIM", out_dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "TOPK", topk));
        token_reduce_cfgs.put(key, c);
        break :blk c;
    };
    const ins = [_][*:0]const u8{ "d", "inv", "sc" };
    const outs = [_][*:0]const u8{"y"};
    const kernel = try getNamedKernel(&token_reduce_kernel, "sushi_exl3_token_reduce", &ins, &outs, TOKEN_REDUCE_SOURCE, "");
    const ov = try applyOuts(s, kernel, &.{ d, inv, scores }, cfg, 1);
    defer _ = mlx.mlx_vector_array_free(ov);
    var a = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(a);
    try mlx.check(mlx.mlx_vector_array_get(&a, ov, 0));
    return a;
}

fn pairPrepareFromTokens(s: mlx.mlx_stream, x: mlx.mlx_array, suhg: mlx.mlx_array, suhu: mlx.mlx_array, slots: mlx.mlx_array, order: ?mlx.mlx_array, in_dim: c_int, nslots: c_int, topk: c_int) !struct { mlx.mlx_array, mlx.mlx_array } {
    const key = PairPrepKey{ .in_dim = in_dim, .nslots = nslots, .topk = topk };
    const cfg = token_pair_prep_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const sh = [_]c_int{ nslots, in_dim };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &sh, 2, .float16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &sh, 2, .float16));
        const blocks = @divExact(in_dim, 128);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, 32 * blocks, nslots, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "IDIM", in_dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "TOPK", topk));
        token_pair_prep_cfgs.put(key, c);
        break :blk c;
    };
    const ins = [_][*:0]const u8{ "x", "suhg", "suhu", "slots", "order" };
    const outs = [_][*:0]const u8{ "yg", "yu" };
    const kernel = if (order != null)
        try getNamedKernel(&token_pair_prepare_kernel, "sushi_exl3_token_pair_prepare", &ins, &outs, TOKEN_PAIR_PREPARE_SOURCE, "")
    else
        try getNamedKernel(&token_pair_unsorted_kernel, "sushi_exl3_token_pair_unsorted", ins[0..4], &outs, metalReplaceFound(TOKEN_PAIR_PREPARE_SOURCE, "const uint orig = uint(order[slot]);", "const uint orig = slot;"), "");
    const ov = if (order) |sorted|
        try applyOuts(s, kernel, &.{ x, suhg, suhu, slots, sorted }, cfg, 2)
    else
        try applyOuts(s, kernel, &.{ x, suhg, suhu, slots }, cfg, 2);
    defer _ = mlx.mlx_vector_array_free(ov);
    var a = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(a);
    var b = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(b);
    try mlx.check(mlx.mlx_vector_array_get(&a, ov, 0));
    try mlx.check(mlx.mlx_vector_array_get(&b, ov, 1));
    return .{ a, b };
}

fn scatterSorted(s: mlx.mlx_stream, x: mlx.mlx_array, order: mlx.mlx_array, dim: c_int, nslots: c_int) !mlx.mlx_array {
    const key = ScatterKey{ .dim = dim, .nslots = nslots };
    const cfg = token_scatter_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const sh = [_]c_int{ nslots, dim };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &sh, 2, .float16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, dim, nslots, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "DIM", dim));
        token_scatter_cfgs.put(key, c);
        break :blk c;
    };
    const ins = [_][*:0]const u8{ "x", "order" };
    const outs = [_][*:0]const u8{"y"};
    const kernel = try getNamedKernel(&token_scatter_kernel, "sushi_exl3_token_scatter", &ins, &outs, TOKEN_SCATTER_SOURCE, "");
    const ov = try applyOuts(s, kernel, &.{ x, order }, cfg, 1);
    defer _ = mlx.mlx_vector_array_free(ov);
    var a = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(a);
    try mlx.check(mlx.mlx_vector_array_get(&a, ov, 0));
    return a;
}

pub fn moeSwigluFused(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    gate_t: mlx.mlx_array,
    gate_suh: mlx.mlx_array,
    gate_svh: mlx.mlx_array,
    up_t: mlx.mlx_array,
    up_suh: mlx.mlx_array,
    up_svh: mlx.mlx_array,
    down_t: mlx.mlx_array,
    down_suh: mlx.mlx_array,
    down_svh: mlx.mlx_array,
    slots: mlx.mlx_array,
    scores: mlx.mlx_array,
    out_dtype: mlx.mlx_dtype,
) !mlx.mlx_array {
    return moeSwigluFusedWithShared(s, x, gate_t, gate_suh, gate_svh, up_t, up_suh, up_svh, down_t, down_suh, down_svh, slots, scores, out_dtype, null);
}

pub fn moeSwigluFusedWithShared(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    gate_t: mlx.mlx_array,
    gate_suh: mlx.mlx_array,
    gate_svh: mlx.mlx_array,
    up_t: mlx.mlx_array,
    up_suh: mlx.mlx_array,
    up_svh: mlx.mlx_array,
    down_t: mlx.mlx_array,
    down_suh: mlx.mlx_array,
    down_svh: mlx.mlx_array,
    slots: mlx.mlx_array,
    scores: mlx.mlx_array,
    out_dtype: mlx.mlx_dtype,
    shared: ?mlx.mlx_array,
) !mlx.mlx_array {
    const xsh = mlx.getShape(x);
    const ssh = mlx.getShape(slots);
    const tsh = mlx.getShape(gate_t);
    const nslots = ssh[0];
    const hidden: c_int = if (xsh.len == 1) xsh[0] else xsh[xsh.len - 1];
    const rows: c_int = if (xsh.len == 1) 1 else xsh[0];
    const topk = @divExact(nslots, rows);
    const inter = tsh[2] * 16;
    const prepared = preparedMidOn(hidden, inter, tsh[0], topk, rows, out_dtype) and mlx.mlx_array_dtype(x) == .bfloat16;
    // Only MiMo's verify rows share enough experts for grouping to pay.
    const group: c_int = if (prepared and rows >= 2 and nslots <= DECODE_GROUP_MAX_SLOTS and !grouped_off_for_test) DECODE_GROUP_MEMBERS else 0;
    // A one-row pair keeps preparing its own K span: the extra prepare dispatch costs it more than it saves.
    const inners = if (prepared and rows >= 2) blk: {
        const prep = try pairPrepare(s, x, gate_suh, up_suh, slots, hidden, nslots, topk);
        defer _ = mlx.mlx_array_free(prep);
        break :blk try pairGemvPrepared(s, prep, gate_t, up_t, slots, hidden, inter, nslots, topk, group);
    } else try pairGemv(s, x, gate_suh, up_suh, gate_t, up_t, slots, hidden, inter, nslots, topk, group);
    defer _ = mlx.mlx_array_free(inners[0]);
    defer _ = mlx.mlx_array_free(inners[1]);
    try ubenchEval(inners[0], "pair_gemv");
    if (exl3UbenchOn()) try mlx.check(mlx.mlx_array_eval(inners[1]));
    const maxabs = swigluMaxabsOn() and swiglu_maxabs_dumped < 96;
    if (maxabs) {
        try dumpAbsMax(s, inners[0], "ig");
        try dumpAbsMax(s, inners[1], "iu");
    }
    const down_inner = if (prepared) blk: {
        const y = try downGemvPreparedMid(s, inners[0], inners[1], down_t, gate_svh, up_svh, down_suh, slots, inter, hidden, nslots, group);
        if (!prepared_mid_engaged) {
            prepared_mid_engaged = true;
            log.info("[exl3-decode] prepared mid engaged dtype=bfloat16 middle=f16 after down-input Hadamard\n", .{});
        }
        break :blk y;
    } else try downGemvFusedMid(s, inners[0], inners[1], down_t, gate_svh, up_svh, down_suh, slots, inter, hidden, nslots);
    defer _ = mlx.mlx_array_free(down_inner);
    try ubenchEval(down_inner, "down_gemv");
    // The SwiGLU product and the down inner plane are the f16 stores that can
    // saturate: a non-finite here is the mid overflow, not a decode fault.
    if (maxabs) {
        try dumpAbsMax(s, down_inner, "down_inner");
        swiglu_maxabs_dumped += 1;
    }
    const out = try downFinishReduceWithShared(s, down_inner, down_svh, slots, scores, hidden, rows, topk, out_dtype, shared);
    errdefer _ = mlx.mlx_array_free(out);
    try ubenchEval(out, "reduce");
    if (applyUbenchOn()) {
        apply_host_layers += 1;
        if (apply_host_layers == 48) {
            if (apply_host_dumps < 8) {
                const ms = @as(f64, @floatFromInt(apply_host_ns)) / 1e6;
                const n: f64 = @floatFromInt(@max(apply_host_n, 1));
                log.info("[exl3-apply] host {d:.3} ms n={d} us/apply={d:.1}\n", .{
                    ms,
                    apply_host_n,
                    (ms * 1e3) / n,
                });
                apply_host_dumps += 1;
            }
            apply_host_ns = 0;
            apply_host_n = 0;
            apply_host_layers = 0;
            if (apply_host_dumps >= 8) apply_ubench_env = false;
        }
    }
    return out;
}

pub fn moeSwigluIndexed(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    gate_t: mlx.mlx_array,
    gate_suh: mlx.mlx_array,
    gate_svh: mlx.mlx_array,
    up_t: mlx.mlx_array,
    up_suh: mlx.mlx_array,
    up_svh: mlx.mlx_array,
    down_t: mlx.mlx_array,
    down_suh: mlx.mlx_array,
    down_svh: mlx.mlx_array,
    slots: mlx.mlx_array,
    scores: mlx.mlx_array,
) !mlx.mlx_array {
    const g = try projectIndexed(s, x, gate_t, gate_suh, gate_svh, slots);
    defer _ = mlx.mlx_array_free(g);
    const u = try projectIndexed(s, x, up_t, up_suh, up_svh, slots);
    defer _ = mlx.mlx_array_free(u);
    var sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sig);
    try mlx.check(mlx.mlx_sigmoid(&sig, g, s));
    var silu = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(silu);
    try mlx.check(mlx.mlx_multiply(&silu, g, sig, s));
    var h = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(h);
    try mlx.check(mlx.mlx_multiply(&h, silu, u, s));
    const d = try projectIndexed(s, h, down_t, down_suh, down_svh, slots);
    defer _ = mlx.mlx_array_free(d);
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    try mlx.check(mlx.mlx_astype(&sc, scores, .float16, s));
    var sc2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc2);
    try mlx.check(mlx.mlx_expand_dims(&sc2, sc, -1, s));
    var weighted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(weighted);
    try mlx.check(mlx.mlx_multiply(&weighted, d, sc2, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_sum_axis(&out, weighted, 0, false, s));
    return out;
}

fn repeatRows(s: mlx.mlx_stream, x: mlx.mlx_array, rows: c_int, topk: c_int) !mlx.mlx_array {
    var ar = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ar);
    try mlx.check(mlx.mlx_arange(&ar, 0, @floatFromInt(rows), 1, .int32, s));
    var col = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(col);
    try mlx.check(mlx.mlx_reshape(&col, ar, &[_]c_int{ rows, 1 }, 2, s));
    var wide = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wide);
    const shape = [_]c_int{ rows, topk };
    try mlx.check(mlx.mlx_broadcast_to(&wide, col, &shape, 2, s));
    var idx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(idx);
    try mlx.check(mlx.mlx_reshape(&idx, wide, &[_]c_int{rows * topk}, 1, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_take_axis(&out, x, idx, 0, s));
    return out;
}

fn projectSortedWithRuns(
    s: mlx.mlx_stream,
    x_sorted: mlx.mlx_array,
    trellis: mlx.mlx_array,
    suh: mlx.mlx_array,
    svh: mlx.mlx_array,
    slots_sorted: mlx.mlx_array,
) !mlx.mlx_array {
    const prepared = try prepareIndexed(s, x_sorted, suh, slots_sorted);
    defer _ = mlx.mlx_array_free(prepared);
    const inner = try innerGemmSorted(s, prepared, trellis, slots_sorted);
    defer _ = mlx.mlx_array_free(inner);
    return finishIndexed(s, inner, svh, slots_sorted);
}

pub fn moePrefill(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    gate_t: mlx.mlx_array,
    gate_suh: mlx.mlx_array,
    gate_svh: mlx.mlx_array,
    up_t: mlx.mlx_array,
    up_suh: mlx.mlx_array,
    up_svh: mlx.mlx_array,
    down_t: mlx.mlx_array,
    down_suh: mlx.mlx_array,
    down_svh: mlx.mlx_array,
    slots: mlx.mlx_array,
    scores: mlx.mlx_array,
    topk: c_int,
) !mlx.mlx_array {
    const xsh = mlx.getShape(x);
    const rows = xsh[0];
    const hidden = xsh[1];
    const nslots = rows * topk;
    var order = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(order);
    try mlx.check(mlx.mlx_argsort_axis(&order, slots, 0, s));
    var order_i = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(order_i);
    try mlx.check(mlx.mlx_astype(&order_i, order, .int32, s));
    var sorted_slots = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sorted_slots);
    try mlx.check(mlx.mlx_take_axis(&sorted_slots, slots, order, 0, s));
    try ubenchEval(sorted_slots, "sort");
    const prep = try pairPrepareFromTokens(s, x, gate_suh, up_suh, sorted_slots, order_i, hidden, nslots, topk);
    defer _ = mlx.mlx_array_free(prep[0]);
    defer _ = mlx.mlx_array_free(prep[1]);
    try ubenchEval(prep[0], "token_prepare");
    if (exl3UbenchOn()) try mlx.check(mlx.mlx_array_eval(prep[1]));
    const win = gemmWindowRows();
    const aligned = gemmWindowAligned();
    const gt = mlx.getShape(gate_t);
    const optimized = mimoPrefillOn(hidden, gt[2] * 16, gt[0], topk) and aligned;
    const metadata: ?MimoWindowTable = if (optimized) try buildMimoWindowTable(s, sorted_slots, order_i, nslots, win, gt[0]) else null;
    defer if (metadata) |m| {
        _ = mlx.mlx_array_free(m.inverse);
    };
    const tab = if (metadata) |m| m.table else try gemmWindowTable(s, sorted_slots, nslots, win, aligned);
    defer _ = mlx.mlx_array_free(tab.starts);
    defer _ = mlx.mlx_array_free(tab.nlives);
    const g_inner = try innerGemmSortedTable(s, prep[0], gate_t, sorted_slots, win, aligned, tab);
    defer _ = mlx.mlx_array_free(g_inner);
    try ubenchEval(g_inner, "gemm_gate");
    const u_inner = try innerGemmSortedTable(s, prep[1], up_t, sorted_slots, win, aligned, tab);
    defer _ = mlx.mlx_array_free(u_inner);
    try ubenchEval(u_inner, "gemm_up");
    const down_x = try midSwigluPrep(s, g_inner, u_inner, gate_svh, up_svh, down_suh, sorted_slots, mlx.getShape(g_inner)[1], nslots);
    defer _ = mlx.mlx_array_free(down_x);
    try ubenchEval(down_x, "mid");
    const d_inner = try innerGemmSortedTable(s, down_x, down_t, sorted_slots, win, aligned, tab);
    defer _ = mlx.mlx_array_free(d_inner);
    try ubenchEval(d_inner, "gemm_down");
    if (metadata) |m| {
        const out = try finishMimoSorted(s, d_inner, m.inverse, down_svh, slots, scores, hidden, rows, topk, mlx.mlx_array_dtype(x));
        try ubenchEval(out, "token_reduce");
        return out;
    }
    const d_unsorted = try scatterSorted(s, d_inner, order_i, hidden, nslots);
    defer _ = mlx.mlx_array_free(d_unsorted);
    const out = try downFinishReduce(s, d_unsorted, down_svh, slots, scores, hidden, rows, topk, mlx.mlx_array_dtype(x));
    try ubenchEval(out, "token_reduce");
    return out;
}

fn clampedWindowTable(s: mlx.mlx_stream, ids: mlx.mlx_array, order: mlx.mlx_array, n: c_int, win: c_int, experts: c_int, aligned: bool) !MimoWindowTable {
    if (aligned and experts > 0 and experts <= MIMO_WINDOW_MAX_EXPERTS)
        return buildMimoWindowTable(s, ids, order, n, win, experts);
    return .{ .table = try gemmWindowTable(s, ids, n, win, aligned), .inverse = .{ .ctx = null } };
}

pub fn moeSwigluClamped(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    gate_t: mlx.mlx_array,
    gate_suh: mlx.mlx_array,
    gate_svh: mlx.mlx_array,
    up_t: mlx.mlx_array,
    up_suh: mlx.mlx_array,
    up_svh: mlx.mlx_array,
    down_t: mlx.mlx_array,
    down_suh: mlx.mlx_array,
    down_svh: mlx.mlx_array,
    slots: mlx.mlx_array,
    scores: mlx.mlx_array,
    topk: c_int,
    limit: c_int,
    out_dtype: mlx.mlx_dtype,
) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const rows = sh[0];
    const hidden = sh[1];
    const nslots = rows * topk;
    if (rows <= DECODE_ROWS_MAX) {
        const pair = (try laneClampedPair(s, x, gate_t, up_t, gate_suh, up_suh, slots, hidden, rows, topk)) orelse blk: {
            const prep = try pairPrepareFromTokens(s, x, gate_suh, up_suh, slots, null, hidden, nslots, topk);
            const prep_g = prep[0];
            const prep_u = prep[1];
            defer _ = mlx.mlx_array_free(prep_g);
            defer _ = mlx.mlx_array_free(prep_u);
            break :blk (try indexedPairCoopF16(s, prep_g, prep_u, gate_t, up_t, slots)) orelse fallback: {
                const gate = try indexedGemvCoopF16(s, prep_g, gate_t, slots);
                errdefer _ = mlx.mlx_array_free(gate);
                break :fallback [2]mlx.mlx_array{ gate, try indexedGemvCoopF16(s, prep_u, up_t, slots) };
            };
        };
        const gate = pair[0];
        const up = pair[1];
        defer for (pair) |value| {
            _ = mlx.mlx_array_free(value);
        };
        const down = blk: {
            const gs = mlx.getShape(gate_t);
            const us = mlx.getShape(up_t);
            const ds = mlx.getShape(down_t);
            if (out_dtype == .bfloat16 and active_decode.codebook == .mcg and
                gs[3] == us[3] and gs[3] == ds[3] and laneServesRate(gs[3]))
            {
                const prepared = try downLanePrepare(s, gate, up, gate_svh, up_svh, down_suh, slots, mlx.getShape(gate)[1], nslots, limit);
                defer _ = mlx.mlx_array_free(prepared);
                lane_decode_dispatches += 1;
                break :blk try downLaneCoop(s, prepared, down_t, slots);
            }
            const prepared = try midSwigluPrepWithLimit(s, gate, up, gate_svh, up_svh, down_suh, slots, mlx.getShape(gate)[1], nslots, limit);
            defer _ = mlx.mlx_array_free(prepared);
            break :blk try indexedGemvCoopF16(s, prepared, down_t, slots);
        };
        defer _ = mlx.mlx_array_free(down);
        return downFinishReduce(s, down, down_svh, slots, scores, hidden, rows, topk, out_dtype);
    }
    var order = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(order);
    try mlx.check(mlx.mlx_argsort_axis(&order, slots, 0, s));
    var order_i = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(order_i);
    try mlx.check(mlx.mlx_astype(&order_i, order, .int32, s));
    var sorted_slots = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sorted_slots);
    try mlx.check(mlx.mlx_take_axis(&sorted_slots, slots, order, 0, s));
    const prep = try pairPrepareFromTokens(s, x, gate_suh, up_suh, sorted_slots, order_i, hidden, nslots, topk);
    defer _ = mlx.mlx_array_free(prep[0]);
    defer _ = mlx.mlx_array_free(prep[1]);
    const win = gemmWindowRows();
    const aligned = gemmWindowAligned();
    const metadata = try clampedWindowTable(s, sorted_slots, order_i, nslots, win, mlx.getShape(gate_t)[0], aligned);
    defer if (metadata.inverse.ctx != null) {
        _ = mlx.mlx_array_free(metadata.inverse);
    };
    const tab = metadata.table;
    defer _ = mlx.mlx_array_free(tab.starts);
    defer _ = mlx.mlx_array_free(tab.nlives);
    const gate = try innerGemmSortedTable(s, prep[0], gate_t, sorted_slots, win, aligned, tab);
    defer _ = mlx.mlx_array_free(gate);
    const up = try innerGemmSortedTable(s, prep[1], up_t, sorted_slots, win, aligned, tab);
    defer _ = mlx.mlx_array_free(up);
    const prepared = try midSwigluPrepWithLimit(s, gate, up, gate_svh, up_svh, down_suh, sorted_slots, mlx.getShape(gate)[1], nslots, limit);
    defer _ = mlx.mlx_array_free(prepared);
    const down = try innerGemmSortedTable(s, prepared, down_t, sorted_slots, win, aligned, tab);
    defer _ = mlx.mlx_array_free(down);
    if (metadata.inverse.ctx != null)
        return finishMimoSorted(s, down, metadata.inverse, down_svh, slots, scores, hidden, rows, topk, out_dtype);
    const original_order = try scatterSorted(s, down, order_i, hidden, nslots);
    defer _ = mlx.mlx_array_free(original_order);
    return downFinishReduce(s, original_order, down_svh, slots, scores, hidden, rows, topk, out_dtype);
}

pub fn prefillDecodeGatherMm(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    gate_pub: mlx.mlx_array,
    up_pub: mlx.mlx_array,
    down_pub: mlx.mlx_array,
    slots: mlx.mlx_array,
    scores: mlx.mlx_array,
    rows: c_int,
    hidden: c_int,
    inter: c_int,
    topk: c_int,
) !mlx.mlx_array {
    const no_idx = mlx.mlx_array{ .ctx = null };
    var x4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x4);
    try mlx.check(mlx.mlx_reshape(&x4, x, &[_]c_int{ rows, 1, 1, hidden }, 4, s));
    var g4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g4);
    try mlx.check(mlx.mlx_gather_mm(&g4, x4, gate_pub, no_idx, slots, false, s));
    var up4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(up4);
    try mlx.check(mlx.mlx_gather_mm(&up4, x4, up_pub, no_idx, slots, false, s));
    var g = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g);
    try mlx.check(mlx.mlx_squeeze(&g, g4, s));
    var u = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(u);
    try mlx.check(mlx.mlx_squeeze(&u, up4, s));
    var sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sig);
    try mlx.check(mlx.mlx_sigmoid(&sig, g, s));
    var silu = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(silu);
    try mlx.check(mlx.mlx_multiply(&silu, g, sig, s));
    var h = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(h);
    try mlx.check(mlx.mlx_multiply(&h, silu, u, s));
    var h4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(h4);
    try mlx.check(mlx.mlx_reshape(&h4, h, &[_]c_int{ rows, topk, 1, inter }, 4, s));
    var d4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(d4);
    try mlx.check(mlx.mlx_gather_mm(&d4, h4, down_pub, no_idx, slots, false, s));
    var d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(d);
    try mlx.check(mlx.mlx_squeeze(&d, d4, s));
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    try mlx.check(mlx.mlx_astype(&sc, scores, mlx.mlx_array_dtype(d), s));
    var sc3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc3);
    try mlx.check(mlx.mlx_reshape(&sc3, sc, &[_]c_int{ rows, topk, 1 }, 3, s));
    var weighted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(weighted);
    try mlx.check(mlx.mlx_multiply(&weighted, d, sc3, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_sum_axis(&out, weighted, 1, false, s));
    return out;
}

pub fn innerGemvF16(s: mlx.mlx_stream, x: mlx.mlx_array, trellis: mlx.mlx_array) !mlx.mlx_array {
    const xsh = mlx.getShape(x);
    const tsh = mlx.getShape(trellis);
    if (xsh.len != 1 or tsh.len != 3) return error.BadExl3Shape;
    const in_dim = xsh[0];
    const out_dim = tsh[1] * 16;
    const rate = try packedRate(tsh[2]);
    if (tsh[0] * 16 != in_dim) return error.BadExl3Shape;
    const cfg = try gemvConfig(in_dim, out_dim, rate);
    const inputs_arr = [_]mlx.mlx_array{ x, trellis };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    const kernel = try getGemvKernel();
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, cfg, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 0));
    if (!gemv_engaged) {
        gemv_engaged = true;
        log.info("[expert-exl3] engaged in={d} out={d}\n", .{ in_dim, out_dim });
    }
    return out;
}

pub fn moeSwigluHost(
    alloc: std.mem.Allocator,
    x: []const f32,
    gate_t: []const u16,
    gate_suh: []const u16,
    gate_svh: []const u16,
    up_t: []const u16,
    up_suh: []const u16,
    up_svh: []const u16,
    down_t: []const u16,
    down_suh: []const u16,
    down_svh: []const u16,
    slots: []const u32,
    weights: []const f32,
    hidden: usize,
    inter: usize,
    packed_n: usize,
    in_tiles_h: usize,
    out_tiles_i: usize,
    dec: exl3.Decode,
) ![]f32 {
    const kbits = exl3.kFromPackedDim(packed_n) orelse return error.BadExl3Shape;
    const topk = slots.len;
    const y = try alloc.alloc(f32, hidden);
    @memset(y, 0);
    const transformed = try alloc.alloc(f32, hidden);
    defer alloc.free(transformed);
    const inner = try alloc.alloc(f32, @max(hidden, inter));
    defer alloc.free(inner);
    const gate_y = try alloc.alloc(f32, inter);
    defer alloc.free(gate_y);
    const up_y = try alloc.alloc(f32, inter);
    defer alloc.free(up_y);
    const h = try alloc.alloc(f32, inter);
    defer alloc.free(h);
    const down_y = try alloc.alloc(f32, hidden);
    defer alloc.free(down_y);
    const tstride_gu = in_tiles_h * out_tiles_i * packed_n;
    const tstride_d = out_tiles_i * in_tiles_h * packed_n;
    for (0..topk) |k| {
        const e = slots[k];
        const g_off = e * tstride_gu;
        const u_off = e * tstride_gu;
        const d_off = e * tstride_d;
        exl3.project(x, gate_t[g_off..][0..tstride_gu], gate_suh[e * hidden ..][0..hidden], gate_svh[e * inter ..][0..inter], hidden, inter, kbits, dec, transformed, inner[0..inter], gate_y);
        exl3.project(x, up_t[u_off..][0..tstride_gu], up_suh[e * hidden ..][0..hidden], up_svh[e * inter ..][0..inter], hidden, inter, kbits, dec, transformed, inner[0..inter], up_y);
        for (0..inter) |i| {
            const g = gate_y[i];
            h[i] = (g / (1.0 + @exp(-g))) * up_y[i];
        }
        exl3.project(h, down_t[d_off..][0..tstride_d], down_suh[e * inter ..][0..inter], down_svh[e * hidden ..][0..hidden], inter, hidden, kbits, dec, inner[0..inter], transformed, down_y);
        const w = weights[k];
        for (0..hidden) |i| y[i] += w * down_y[i];
    }
    return y;
}

test "exl3 packedRate reads n from the last dim and refuses outside K1..K8" {
    const t = std.testing;
    try t.expectEqual(@as(u32, 32), (try packedRate(32)).n);
    try t.expectEqual(@as(u32, 40), (try packedRate(40)).n);
    try t.expectEqual(@as(u32, 44), (try packedRate(44)).n);
    try t.expectEqual(@as(u32, 48), (try packedRate(48)).n);
    try t.expectEqual(@as(u32, 64), (try packedRate(64)).n);
    try t.expectError(error.BadExl3Shape, packedRate(14));
    try t.expectError(error.BadExl3Shape, packedRate(41));
    try t.expectError(error.BadExl3Shape, packedRate(130));
}

/// The fixture is `align(2)` so the u16 view below is a cast the caller has
/// already paid for: a byte-aligned blob is a compile error, not a Debug panic.
fn metalInnerGemvFixture(fixture: []align(2) const u8, rate: exl3.Rate, dec: exl3.Decode) !void {
    setDecodeParams(dec);
    defer setDecodeParams(.mul1);
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const header_len = std.mem.readInt(u64, fixture[0..8], .little);
    const header = fixture[8 .. 8 + header_len];
    const data = fixture[8 + header_len ..];
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header, .{});
    defer parsed.deinit();
    const trellis_meta = parsed.value.object.get("trellis").?.object;
    const t_off: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[0].integer);
    const t_end: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[1].integer);
    const trellis_bits: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, data[t_off..t_end]));
    var x: [128]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(11);
    const rnd = prng.random();
    for (&x) |*v| v.* = rnd.float(f32) * 2 - 1;
    const xf16 = try alloc.alloc(u16, 128);
    for (x, xf16) |v, *b| b.* = exl3.f32ToF16Bits(v);
    const xf = try alloc.alloc(f32, 128);
    for (xf16, xf) |b, *v| v.* = exl3.f16BitsToF32(b);
    const x_arr = mlx.mlx_array_new_data(xf16.ptr, &[_]c_int{128}, 1, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr_arr = mlx.mlx_array_new_data(trellis_bits.ptr, &[_]c_int{ 8, 8, @intCast(rate.halfwords()) }, 3, .uint16);
    defer _ = mlx.mlx_array_free(tr_arr);
    const got = try innerGemvF16(s, x_arr, tr_arr);
    defer _ = mlx.mlx_array_free(got);
    var contig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(contig);
    try mlx.check(mlx.mlx_contiguous(&contig, got, false, s));
    try mlx.check(mlx.mlx_array_eval(contig));
    const src = mlx.mlx_array_data_float16(contig) orelse return error.F16Unreadable;
    var host: [128]f32 = undefined;
    exl3.innerGemv(trellis_bits, xf, 128, 128, rate, dec, &host);
    for (0..128) |i| {
        const bits: u16 = @bitCast(src[i]);
        try t.expectEqual(exl3.f32ToF16Bits(host[i]), bits);
    }
}

test "exl3 K3 Metal inner GEMV matches the host tile decode" {
    try metalInnerGemvFixture(exl3.fixtures.k3, exl3.Rate.fromK(3), .mul1);
}

test "exl3 K2 Metal inner GEMV matches the host tile decode" {
    try metalInnerGemvFixture(exl3.fixtures.k2, exl3.Rate.fromK(2), .mul1);
}

test "exl3 K2.5 MCG Metal inner GEMV matches the host tile decode" {
    try metalInnerGemvFixture(exl3.fixtures.k2p5_mcg, .{ .n = 40 }, .mcg);
}

test "exl3 K3 MCG Metal inner GEMV matches the host tile decode" {
    try metalInnerGemvFixture(exl3.fixtures.k3_mcg, .{ .n = 48 }, .mcg);
}

// A w16 bitstream is a valid w12 bitstream — only the value each window decodes
// to changes — so the existing fixtures are the narrowed packs too, scored
// against the host reference under the same width.
test "exl3 Metal inner GEMV matches the host tile decode at a narrowed codeword window" {
    try metalInnerGemvFixture(exl3.fixtures.k2p5_mcg, .{ .n = 40 }, .{ .codebook = .mcg, .window = .w12 });
    try metalInnerGemvFixture(exl3.fixtures.k2p5_mcg, .{ .n = 40 }, .{ .codebook = .mcg, .window = .w14 });
    try metalInnerGemvFixture(exl3.fixtures.k4, exl3.Rate.fromK(4), .{ .codebook = .mul1, .window = .w12 });
}

fn indexedParityStats(rate: exl3.Rate, in_dim: usize, out_dim: usize, e: usize, topk: usize, seed: u64, dec: exl3.Decode, mutate: Exl3Mutation) !Exl3GemmParityStats {
    setDecodeParams(dec);
    defer setDecodeParams(.mul1);
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const packed_n = rate.halfwords();
    const in_tiles = in_dim / 16;
    const out_tiles = out_dim / 16;
    const tile_n = in_tiles * out_tiles * packed_n;
    const stacked = try alloc.alloc(u16, e * tile_n);
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    for (stacked) |*v| v.* = @truncate(rnd.int(u32));
    const xh = try alloc.alloc(u16, topk * in_dim);
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
    const slots_h = try alloc.alloc(u32, topk);
    for (slots_h, 0..) |*v, i| v.* = @intCast(i % e);
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(topk), @intCast(in_dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr_arr = mlx.mlx_array_new_data(stacked.ptr, &[_]c_int{ @intCast(e), @intCast(in_tiles), @intCast(out_tiles), @intCast(packed_n) }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr_arr);
    const slots = mlx.mlx_array_new_data(slots_h.ptr, &[_]c_int{@intCast(topk)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(slots);
    const got = try indexedGemvCoopF16(s, x_arr, tr_arr, slots);
    defer _ = mlx.mlx_array_free(got);
    var contig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(contig);
    try mlx.check(mlx.mlx_contiguous(&contig, got, false, s));
    try mlx.check(mlx.mlx_array_eval(contig));
    const src = mlx.mlx_array_data_float16(contig) orelse return error.F16Unreadable;
    if (mutate == .codeword) stacked[tile_n / 2] ^= 0x40;
    return measureInnerGemmParity(alloc, s, src[0 .. topk * out_dim], xh, slots_h, stacked, in_dim, out_dim, rate, mutatedDecode(dec, mutate));
}

fn indexedParity(rate: exl3.Rate, in_dim: usize, out_dim: usize, e: usize, topk: usize, seed: u64, dec: exl3.Decode) !void {
    try reportGemmParity(try indexedParityStats(rate, in_dim, out_dim, e, topk, seed, dec, .none));
}

test "exl3 K3 cooperative indexed GEMV matches host MUL1 tile decode" {
    try indexedParity(exl3.Rate.fromK(3), 128, 128, 4, 10, 17, .mul1);
}

test "exl3 K2 cooperative indexed GEMV matches host MUL1 tile decode" {
    try indexedParity(exl3.Rate.fromK(2), 128, 128, 4, 10, 19, .mul1);
}

test "exl3 K3 Metal inner GEMV matches host on production shape" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const in_dim: usize = 2560;
    const out_dim: usize = 640;
    const k: u32 = 3;
    const packed_n = exl3.packedHalfwords(k);
    const in_tiles = in_dim / 16;
    const out_tiles = out_dim / 16;
    const trellis = try alloc.alloc(u16, in_tiles * out_tiles * packed_n);
    var prng = std.Random.DefaultPrng.init(23);
    const rnd = prng.random();
    for (trellis) |*v| v.* = @truncate(rnd.int(u32));
    const xf16 = try alloc.alloc(u16, in_dim);
    for (xf16) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
    const x_arr = mlx.mlx_array_new_data(xf16.ptr, &[_]c_int{@intCast(in_dim)}, 1, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr_arr = mlx.mlx_array_new_data(trellis.ptr, &[_]c_int{ @intCast(in_tiles), @intCast(out_tiles), @intCast(packed_n) }, 3, .uint16);
    defer _ = mlx.mlx_array_free(tr_arr);
    const got = try innerGemvF16(s, x_arr, tr_arr);
    defer _ = mlx.mlx_array_free(got);
    var contig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(contig);
    try mlx.check(mlx.mlx_contiguous(&contig, got, false, s));
    try mlx.check(mlx.mlx_array_eval(contig));
    const src = mlx.mlx_array_data_float16(contig) orelse return error.F16Unreadable;
    try expectInnerGemmParity(alloc, s, src[0..out_dim], xf16, &.{0}, trellis, in_dim, out_dim, exl3.Rate.fromK(k), .mul1);
}

test "exl3 K3 cooperative indexed GEMV matches host MUL1 on production shape" {
    try indexedParity(exl3.Rate.fromK(3), 2560, 640, 4, 10, 23, .mul1);
}

test "exl3 K2 cooperative indexed GEMV matches host MUL1 on production shape" {
    try indexedParity(exl3.Rate.fromK(2), 2560, 640, 4, 10, 29, .mul1);
}

test "exl3 K4 Metal inner GEMV matches the host tile decode" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const fixture = exl3.fixtures.k4;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const header_len = std.mem.readInt(u64, fixture[0..8], .little);
    const header = fixture[8 .. 8 + header_len];
    const data = fixture[8 + header_len ..];
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header, .{});
    defer parsed.deinit();
    const trellis_meta = parsed.value.object.get("trellis").?.object;
    const inner_meta = parsed.value.object.get("inner").?.object;
    const t_off: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[0].integer);
    const t_end: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[1].integer);
    const i_off: usize = @intCast(inner_meta.get("data_offsets").?.array.items[0].integer);
    const i_end: usize = @intCast(inner_meta.get("data_offsets").?.array.items[1].integer);
    const trellis_bits = std.mem.bytesAsSlice(u16, data[t_off..t_end]);
    const inner_bits = std.mem.bytesAsSlice(u16, data[i_off..i_end]);
    var x: [128]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(11);
    const rnd = prng.random();
    for (&x) |*v| v.* = rnd.float(f32) * 2 - 1;
    const xf16 = try alloc.alloc(u16, 128);
    for (x, xf16) |v, *b| b.* = exl3.f32ToF16Bits(v);
    const x_arr = mlx.mlx_array_new_data(xf16.ptr, &[_]c_int{128}, 1, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr_arr = mlx.mlx_array_new_data(trellis_bits.ptr, &[_]c_int{ 8, 8, 64 }, 3, .uint16);
    defer _ = mlx.mlx_array_free(tr_arr);
    const got = try innerGemvF16(s, x_arr, tr_arr);
    defer _ = mlx.mlx_array_free(got);
    var contig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(contig);
    try mlx.check(mlx.mlx_contiguous(&contig, got, false, s));
    try mlx.check(mlx.mlx_array_eval(contig));
    const src = mlx.mlx_array_data_float16(contig) orelse return error.F16Unreadable;
    var host: [128]f32 = undefined;
    @memset(&host, 0);
    for (0..128) |o| {
        var acc: f32 = 0;
        for (0..128) |i| acc += exl3.f16BitsToF32(xf16[i]) * exl3.f16BitsToF32(inner_bits[i * 128 + o]);
        host[o] = exl3.f16BitsToF32(exl3.f32ToF16Bits(acc));
    }
    for (0..128) |i| {
        const bits: u16 = @bitCast(src[i]);
        try t.expectEqual(exl3.f32ToF16Bits(host[i]), bits);
    }
}

test "exl3 verify group union unique vs assignment count" {
    const t = std.testing;
    var eids: [20]u32 = undefined;
    var i: usize = 0;
    while (i < 10) : (i += 1) eids[i] = @intCast(i);
    i = 0;
    while (i < 10) : (i += 1) eids[10 + i] = @intCast(100 + i);
    try t.expectEqual(@as(u32, 20), unionUnique(&eids));
    i = 0;
    while (i < 10) : (i += 1) eids[10 + i] = @intCast(i);
    try t.expectEqual(@as(u32, 10), unionUnique(&eids));
    i = 0;
    while (i < 10) : (i += 1) eids[10 + i] = @intCast(7 + i);
    try t.expectEqual(@as(u32, 17), unionUnique(&eids));
    var counts: [512]u32 = @splat(0);
    const u = unionMultiplicity(&eids, &counts);
    try t.expectEqual(@as(u32, 17), u);
    try t.expectEqual(@as(u32, 1), counts[0]);
    try t.expectEqual(@as(u32, 2), counts[7]);
    try t.expectEqual(@as(u32, 2), counts[9]);
    try t.expectEqual(@as(u32, 1), counts[16]);
}

test "exl3 buildRuns groups sorted expert ids" {
    const t = std.testing;
    const ids = [_]u32{ 3, 3, 3, 7, 7, 1 };
    const runs = try buildRuns(t.allocator, &ids);
    defer t.allocator.free(runs.start);
    defer t.allocator.free(runs.len);
    defer t.allocator.free(runs.eid);
    try t.expectEqual(@as(u32, 3), runs.n);
    try t.expectEqual(@as(u32, 0), runs.start[0]);
    try t.expectEqual(@as(u32, 3), runs.len[0]);
    try t.expectEqual(@as(u32, 3), runs.eid[0]);
    try t.expectEqual(@as(u32, 3), runs.start[1]);
    try t.expectEqual(@as(u32, 2), runs.len[1]);
    try t.expectEqual(@as(u32, 7), runs.eid[1]);
    try t.expectEqual(@as(u32, 5), runs.start[2]);
    try t.expectEqual(@as(u32, 1), runs.len[2]);
    try t.expectEqual(@as(u32, 1), runs.eid[2]);
}

test "exl3 row count is the leading dims, not the activation width" {
    const t = std.testing;
    try t.expectEqual(@as(usize, 1), rowsOfShape(&[_]c_int{2560}));
    try t.expectEqual(@as(usize, 2), rowsOfShape(&[_]c_int{ 2, 2560 }));
    try t.expectEqual(@as(usize, 6), rowsOfShape(&[_]c_int{ 2, 3, 2560 }));
    try t.expect(!usesPrefillArm(rowsOfShape(&[_]c_int{ 16, 1, 2560 })));
    try t.expect(usesPrefillArm(rowsOfShape(&[_]c_int{ 17, 1, 2560 })));
    try t.expect(!usesPrefillArm(rowsOfShape(&[_]c_int{ 4, 2560 })));
}

test "exl3 prefill arm is used only above 16 rows" {
    const t = std.testing;
    try t.expect(!usesPrefillArm(1));
    try t.expect(!usesPrefillArm(2));
    try t.expect(!usesPrefillArm(16));
    try t.expect(usesPrefillArm(17));
    try t.expect(usesPrefillArm(512));
}

test "exl3 512-row prefill: decode-to-f16 gather_mm vs rows kernel" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const fixture = exl3.fixtures.k4;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const header_len = std.mem.readInt(u64, fixture[0..8], .little);
    const header = fixture[8 .. 8 + header_len];
    const data = fixture[8 + header_len ..];
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header, .{});
    defer parsed.deinit();
    const trellis_meta = parsed.value.object.get("trellis").?.object;
    const suh_meta = parsed.value.object.get("suh").?.object;
    const svh_meta = parsed.value.object.get("svh").?.object;
    const pub_meta = parsed.value.object.get("public").?.object;
    const t0: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[0].integer);
    const t1: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[1].integer);
    const s0: usize = @intCast(suh_meta.get("data_offsets").?.array.items[0].integer);
    const s1: usize = @intCast(suh_meta.get("data_offsets").?.array.items[1].integer);
    const v0: usize = @intCast(svh_meta.get("data_offsets").?.array.items[0].integer);
    const v1: usize = @intCast(svh_meta.get("data_offsets").?.array.items[1].integer);
    const p0: usize = @intCast(pub_meta.get("data_offsets").?.array.items[0].integer);
    const p1: usize = @intCast(pub_meta.get("data_offsets").?.array.items[1].integer);
    const trellis_bits = std.mem.bytesAsSlice(u16, data[t0..t1]);
    const suh_bits = std.mem.bytesAsSlice(u16, data[s0..s1]);
    const svh_bits = std.mem.bytesAsSlice(u16, data[v0..v1]);
    const pub_bits = std.mem.bytesAsSlice(u16, data[p0..p1]);
    const E: c_int = 512;
    const topk: c_int = 10;
    const dim: c_int = 128;
    const stacked_t = try alloc.alloc(u16, @intCast(E * 8 * 8 * 64));
    const stacked_suh = try alloc.alloc(u16, @intCast(E * dim));
    const stacked_svh = try alloc.alloc(u16, @intCast(E * dim));
    const stacked_pub = try alloc.alloc(u16, @intCast(E * dim * dim));
    var e_i: c_int = 0;
    while (e_i < E) : (e_i += 1) {
        const tb: usize = @intCast(e_i);
        @memcpy(stacked_t[tb * trellis_bits.len ..][0..trellis_bits.len], trellis_bits);
        @memcpy(stacked_suh[tb * suh_bits.len ..][0..suh_bits.len], suh_bits);
        @memcpy(stacked_svh[tb * svh_bits.len ..][0..svh_bits.len], svh_bits);
        @memcpy(stacked_pub[tb * pub_bits.len ..][0..pub_bits.len], pub_bits);
    }
    const tr = mlx.mlx_array_new_data(stacked_t.ptr, &[_]c_int{ E, 8, 8, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr);
    const suh = mlx.mlx_array_new_data(stacked_suh.ptr, &[_]c_int{ E, dim }, 2, .float16);
    defer _ = mlx.mlx_array_free(suh);
    const svh = mlx.mlx_array_new_data(stacked_svh.ptr, &[_]c_int{ E, dim }, 2, .float16);
    defer _ = mlx.mlx_array_free(svh);
    const pub_a = mlx.mlx_array_new_data(stacked_pub.ptr, &[_]c_int{ E, dim, dim }, 3, .float16);
    defer _ = mlx.mlx_array_free(pub_a);
    var w_oi = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_oi);
    try mlx.check(mlx.mlx_transpose_axes(&w_oi, pub_a, &[_]c_int{ 0, 2, 1 }, 3, s));
    var w_c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_c);
    try mlx.check(mlx.mlx_contiguous(&w_c, w_oi, false, s));
    var triple = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(triple);
    try mlx.check(mlx.mlx_quantize(&triple, w_c, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{ .ctx = null }, s));
    var wq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wq);
    var wsc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wsc);
    var wbi = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wbi);
    try mlx.check(mlx.mlx_vector_array_get(&wq, triple, 0));
    try mlx.check(mlx.mlx_vector_array_get(&wsc, triple, 1));
    try mlx.check(mlx.mlx_vector_array_get(&wbi, triple, 2));
    var prng = std.Random.DefaultPrng.init(3);
    const rnd = prng.random();
    for ([2]c_int{ 512, 2048 }) |R| {
        const xh = try alloc.alloc(u16, @intCast(R * dim));
        for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
        const slots_h = try alloc.alloc(u32, @intCast(R * topk));
        for (slots_h) |*v| v.* = rnd.uintLessThan(u32, @intCast(E));
        var max_e: u32 = 0;
        for (slots_h) |v| if (v > max_e) {
            max_e = v;
        };
        try t.expect(max_e >= 400);
        const scores_h = try alloc.alloc(f32, @intCast(R * topk));
        for (scores_h) |*v| v.* = 0.5;
        const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ R, dim }, 2, .float16);
        defer _ = mlx.mlx_array_free(x_arr);
        const slots = mlx.mlx_array_new_data(slots_h.ptr, &[_]c_int{R * topk}, 1, .uint32);
        defer _ = mlx.mlx_array_free(slots);
        const scores = mlx.mlx_array_new_data(scores_h.ptr, &[_]c_int{R * topk}, 1, .float32);
        defer _ = mlx.mlx_array_free(scores);
        const warm = try moePrefill(s, x_arr, tr, suh, svh, tr, suh, svh, tr, suh, svh, slots, scores, topk);
        try mlx.check(mlx.mlx_array_eval(warm));
        _ = mlx.mlx_array_free(warm);
        var t_rows = io_util.Stopwatch.init(t.io);
        var it: usize = 0;
        while (it < 8) : (it += 1) {
            const rows_out = try moePrefill(s, x_arr, tr, suh, svh, tr, suh, svh, tr, suh, svh, slots, scores, topk);
            try mlx.check(mlx.mlx_array_eval(rows_out));
            _ = mlx.mlx_array_free(rows_out);
        }
        const rows_ns = t_rows.read() / 8;
        const n_tok: c_int = R * topk;
        const xr = try repeatRows(s, x_arr, R, topk);
        defer _ = mlx.mlx_array_free(xr);
        var xrep = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xrep);
        try mlx.check(mlx.mlx_reshape(&xrep, xr, &[_]c_int{ n_tok, 1, dim }, 3, s));
        var slots_i = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(slots_i);
        try mlx.check(mlx.mlx_astype(&slots_i, slots, .int32, s));
        var order = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(order);
        try mlx.check(mlx.mlx_argsort_axis(&order, slots_i, 0, s));
        var sorted = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sorted);
        try mlx.check(mlx.mlx_take_axis(&sorted, slots_i, order, 0, s));
        const no_idx = mlx.mlx_array{ .ctx = null };
        var qmm = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(qmm);
        try mlx.check(mlx.mlx_gather_qmm(&qmm, xrep, wq, wsc, wbi, no_idx, sorted, true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", true, s));
        try mlx.check(mlx.mlx_array_eval(qmm));
        var t_q = io_util.Stopwatch.init(t.io);
        it = 0;
        while (it < 8) : (it += 1) {
            var qmm2 = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_gather_qmm(&qmm2, xrep, wq, wsc, wbi, no_idx, sorted, true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", true, s));
            try mlx.check(mlx.mlx_array_eval(qmm2));
            _ = mlx.mlx_array_free(qmm2);
        }
        const qmm_ns = t_q.read() / 8;
        const ratio_x100: u64 = if (qmm_ns == 0) 0 else (rows_ns * 100) / (qmm_ns * 3);
        benchPrint("exl3 C={d} E=512 H=128 topk=10: sorted-gemm-layer {d} us  affine-gather_qmm-one-proj {d} us  ratio-vs-3x-qmm {d}/100\n", .{
            R,
            rows_ns / 1000,
            qmm_ns / 1000,
            ratio_x100,
        });
        try t.expect(qmm_ns > 0);
    }
}

test "exl3 512-row production-shape sorted gemm vs affine gather_qmm" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    if (mlx.maxRecommendedWorkingSet() < 16 << 30) return error.SkipZigTest; // production-shape banks: GBs a CI runner lacks
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const E: c_int = 512;
    const topk: c_int = 10;
    const H: c_int = 2560;
    const I: c_int = 640;
    const tr_n: usize = @intCast(E * (H / 16) * (I / 16) * 64);
    const tr_g = try alloc.alloc(u16, tr_n);
    const tr_d = try alloc.alloc(u16, @intCast(E * (I / 16) * (H / 16) * 64));
    const suh_g = try alloc.alloc(u16, @intCast(E * H));
    const svh_g = try alloc.alloc(u16, @intCast(E * I));
    const suh_d = try alloc.alloc(u16, @intCast(E * I));
    const svh_d = try alloc.alloc(u16, @intCast(E * H));
    var prng = std.Random.DefaultPrng.init(5);
    const rnd = prng.random();
    for (tr_g) |*v| v.* = @truncate(rnd.int(u32));
    for (tr_d) |*v| v.* = @truncate(rnd.int(u32));
    for (suh_g) |*v| v.* = exl3.f32ToF16Bits(1.0);
    for (svh_g) |*v| v.* = exl3.f32ToF16Bits(1.0);
    for (suh_d) |*v| v.* = exl3.f32ToF16Bits(1.0);
    for (svh_d) |*v| v.* = exl3.f32ToF16Bits(1.0);
    const trg = mlx.mlx_array_new_data(tr_g.ptr, &[_]c_int{ E, H / 16, I / 16, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(trg);
    const trd = mlx.mlx_array_new_data(tr_d.ptr, &[_]c_int{ E, I / 16, H / 16, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(trd);
    const sugh = mlx.mlx_array_new_data(suh_g.ptr, &[_]c_int{ E, H }, 2, .float16);
    defer _ = mlx.mlx_array_free(sugh);
    const svgi = mlx.mlx_array_new_data(svh_g.ptr, &[_]c_int{ E, I }, 2, .float16);
    defer _ = mlx.mlx_array_free(svgi);
    const sudi = mlx.mlx_array_new_data(suh_d.ptr, &[_]c_int{ E, I }, 2, .float16);
    defer _ = mlx.mlx_array_free(sudi);
    const svdh = mlx.mlx_array_new_data(svh_d.ptr, &[_]c_int{ E, H }, 2, .float16);
    defer _ = mlx.mlx_array_free(svdh);
    var dense = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dense);
    try mlx.check(mlx.mlx_random_normal(&dense, &[_]c_int{ E, I, H }, 3, .float16, 0, 1, .{ .ctx = null }, s));
    var w_c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_c);
    try mlx.check(mlx.mlx_contiguous(&w_c, dense, false, s));
    var triple = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(triple);
    try mlx.check(mlx.mlx_quantize(&triple, w_c, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{ .ctx = null }, s));
    var wq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wq);
    var wsc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wsc);
    var wbi = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wbi);
    try mlx.check(mlx.mlx_vector_array_get(&wq, triple, 0));
    try mlx.check(mlx.mlx_vector_array_get(&wsc, triple, 1));
    try mlx.check(mlx.mlx_vector_array_get(&wbi, triple, 2));
    for ([2]c_int{ 512, 2048 }) |R| {
        const xh = try alloc.alloc(u16, @intCast(R * H));
        const slots_h = try alloc.alloc(u32, @intCast(R * topk));
        const scores_h = try alloc.alloc(f32, @intCast(R * topk));
        for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 0.1);
        for (slots_h) |*v| v.* = rnd.uintLessThan(u32, @intCast(E));
        var max_e: u32 = 0;
        for (slots_h) |v| if (v > max_e) {
            max_e = v;
        };
        try t.expect(max_e >= 400);
        for (scores_h) |*v| v.* = 0.5;
        const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ R, H }, 2, .float16);
        defer _ = mlx.mlx_array_free(x_arr);
        const slots = mlx.mlx_array_new_data(slots_h.ptr, &[_]c_int{R * topk}, 1, .uint32);
        defer _ = mlx.mlx_array_free(slots);
        const scores = mlx.mlx_array_new_data(scores_h.ptr, &[_]c_int{R * topk}, 1, .float32);
        defer _ = mlx.mlx_array_free(scores);
        const warm = try moePrefill(s, x_arr, trg, sugh, svgi, trg, sugh, svgi, trd, sudi, svdh, slots, scores, topk);
        try mlx.check(mlx.mlx_array_eval(warm));
        _ = mlx.mlx_array_free(warm);
        var t_g = io_util.Stopwatch.init(t.io);
        var it: usize = 0;
        while (it < 3) : (it += 1) {
            const out = try moePrefill(s, x_arr, trg, sugh, svgi, trg, sugh, svgi, trd, sudi, svdh, slots, scores, topk);
            try mlx.check(mlx.mlx_array_eval(out));
            _ = mlx.mlx_array_free(out);
        }
        const gemm_ns = t_g.read() / 3;
        const xr = try repeatRows(s, x_arr, R, topk);
        defer _ = mlx.mlx_array_free(xr);
        var xrep = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(xrep);
        try mlx.check(mlx.mlx_reshape(&xrep, xr, &[_]c_int{ R * topk, 1, H }, 3, s));
        var slots_i = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(slots_i);
        try mlx.check(mlx.mlx_astype(&slots_i, slots, .int32, s));
        var order = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(order);
        try mlx.check(mlx.mlx_argsort_axis(&order, slots_i, 0, s));
        var sorted = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sorted);
        try mlx.check(mlx.mlx_take_axis(&sorted, slots_i, order, 0, s));
        const no_idx = mlx.mlx_array{ .ctx = null };
        var q0 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q0);
        try mlx.check(mlx.mlx_gather_qmm(&q0, xrep, wq, wsc, wbi, no_idx, sorted, true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", true, s));
        try mlx.check(mlx.mlx_array_eval(q0));
        var t_q = io_util.Stopwatch.init(t.io);
        it = 0;
        while (it < 3) : (it += 1) {
            var q = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_gather_qmm(&q, xrep, wq, wsc, wbi, no_idx, sorted, true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", true, s));
            try mlx.check(mlx.mlx_array_eval(q));
            _ = mlx.mlx_array_free(q);
        }
        const qmm_ns = t_q.read() / 3;
        const ratio_x100: u64 = if (qmm_ns == 0) 0 else (gemm_ns * 100) / (qmm_ns * 3);
        benchPrint("exl3 C={d} E=512 H=2560 I=640 topk=10: sorted-gemm-layer {d} us  affine-gather_qmm-one-proj {d} us  ratio-vs-3x-qmm {d}/100\n", .{
            R,
            gemm_ns / 1000,
            qmm_ns / 1000,
            ratio_x100,
        });
        try t.expect(qmm_ns > 0);
    }
}

test "exl3 K4 cooperative indexed GEMV matches host MUL1 tile decode" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const fixture = exl3.fixtures.k4;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const header_len = std.mem.readInt(u64, fixture[0..8], .little);
    const header = fixture[8 .. 8 + header_len];
    const data = fixture[8 + header_len ..];
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header, .{});
    defer parsed.deinit();
    const trellis_meta = parsed.value.object.get("trellis").?.object;
    const t_off: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[0].integer);
    const t_end: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[1].integer);
    const trellis_bits = std.mem.bytesAsSlice(u16, data[t_off..t_end]);
    const E: usize = 4;
    const dim: usize = 128;
    const topk: usize = 10;
    const tile_n = 8 * 8 * 64;
    const stacked = try alloc.alloc(u16, E * tile_n);
    for (0..E) |e| @memcpy(stacked[e * tile_n ..][0..tile_n], trellis_bits);
    var prng = std.Random.DefaultPrng.init(17);
    const rnd = prng.random();
    const xh = try alloc.alloc(u16, topk * dim);
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
    const slots_h = try alloc.alloc(u32, topk);
    for (slots_h, 0..) |*v, i| v.* = @intCast(i % E);
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(topk), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr_arr = mlx.mlx_array_new_data(stacked.ptr, &[_]c_int{ @intCast(E), 8, 8, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr_arr);
    const slots = mlx.mlx_array_new_data(slots_h.ptr, &[_]c_int{@intCast(topk)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(slots);
    const got = try indexedGemvCoopF16(s, x_arr, tr_arr, slots);
    defer _ = mlx.mlx_array_free(got);
    var contig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(contig);
    try mlx.check(mlx.mlx_contiguous(&contig, got, false, s));
    try mlx.check(mlx.mlx_array_eval(contig));
    const src = mlx.mlx_array_data_float16(contig) orelse return error.F16Unreadable;
    try expectInnerGemmParity(alloc, s, src[0 .. topk * dim], xh, slots_h, stacked, dim, dim, exl3.Rate.fromK(4), .mul1);
}

/// Two renderings of the same chain agree to `bar` in relative RMS. The chains
/// differ in where they round to f16, so the bar is an envelope, never bytes.
fn expectRelRms(got: []const f16, want: []const f16, bar: f64) !void {
    var ss: f64 = 0;
    var ref: f64 = 0;
    for (got, want) |g, w| {
        const a: f64 = @floatCast(g);
        const b: f64 = @floatCast(w);
        if (!std.math.isFinite(a) or !std.math.isFinite(b)) return error.TestExpectedEqual;
        ss += (a - b) * (a - b);
        ref += b * b;
    }
    const rel = @sqrt(ss / @max(ref, 1e-20));
    if (rel < bar) return;
    std.debug.print("exl3 rel_rms {d:.6} over bar {d:.6}\n", .{ rel, bar });
    return error.TestExpectedEqual;
}

/// What a GEMM/GEMV arm's output may differ from the exact dot product of the
/// same decoded weights.
///
/// A trellis dot product cancels: the result lands about four orders below
/// sum|w_i x_i|, so a bar written against the RESULT measures noise and passes
/// or fails on the seed. Every bar here is relative to the SUMMANDS, which is
/// the scale the arithmetic runs at.
const Exl3GemmParity = struct {
    /// f16 unit roundoff. An arm rounds its result to f16 exactly once and
    /// |result| <= sum|w_i x_i|, so that store costs at most U_F16 of the sum.
    const U_F16: f64 = 0x1p-11;
    /// f32 unit roundoff. Every arm accumulates its products in f32 planes, so
    /// a `depth`-long accumulation costs at most depth * U_F32 of the sum.
    const U_F32: f64 = 0x1p-24;
    /// How much of the composite's RMS error the arm may carry.
    const RMS_FACTOR: f64 = 3.0;

    /// Ceiling on one element's error as a fraction of sum|w_i x_i|.
    fn elemCeiling(depth: usize) f64 {
        return U_F16 + @as(f64, @floatFromInt(depth)) * U_F32;
    }
};

/// A defect injected between the arm and its reference, so the bar is shown to
/// FAIL on a wrong arm and not only to pass on a right one. `.none` is the
/// real test; the others make the arm's decode disagree with the reference's
/// exactly as a miscoded kernel would.
const Exl3Mutation = enum {
    none,
    /// The arm reads the codeword window the pack names, the reference reads
    /// another — the same defect as a kernel built for the wrong mask.
    window,
    /// One bit of one trellis halfword, which moves the weights whose sliding
    /// window covers it.
    codeword,
};

fn mutatedDecode(dec: exl3.Decode, mutate: Exl3Mutation) exl3.Decode {
    if (mutate != .window) return dec;
    return .{ .codebook = dec.codebook, .window = if (dec.window == .w16) .w15 else .w16 };
}

/// Why a GEMM parity check failed, or null when it passed. Pure, so the
/// decision is unit-testable without a GPU.
const Exl3GemmParityFail = enum { nonfinite, gross_element, systematic };

fn exl3GemmParityVerdict(
    finite: bool,
    kern_max: f64,
    rms_kern: f64,
    rms_comp: f64,
    ceiling: f64,
) ?Exl3GemmParityFail {
    if (!finite) return .nonfinite;
    if (kern_max > ceiling) return .gross_element;
    // A zero composite RMS means f16 represents every truth exactly on this
    // data; the arm then owes the same, which is the strictest reading.
    if (rms_kern > Exl3GemmParity.RMS_FACTOR * rms_comp) return .systematic;
    return null;
}

const Exl3GemmParityStats = struct {
    n: usize = 0,
    finite: bool = true,
    kern_max: f64 = 0,
    comp_max: f64 = 0,
    rms_kern: f64 = 0,
    rms_comp: f64 = 0,
    ceiling: f64 = 0,
};

/// Every expert's trellis decoded into a dense `[E, in_dim, out_dim]` f16
/// bank — the one decode both the truth and the composite read.
fn dequantStacked(
    alloc: std.mem.Allocator,
    stacked: []const u16,
    in_dim: usize,
    out_dim: usize,
    rate: exl3.Rate,
    dec: exl3.Decode,
) ![]u16 {
    const in_tiles = in_dim / 16;
    const out_tiles = out_dim / 16;
    const packed_n = rate.halfwords();
    const tile_n = in_tiles * out_tiles * packed_n;
    const w = try alloc.alloc(u16, (stacked.len / tile_n) * in_dim * out_dim);
    var tile_w: [exl3.TILE_VALUES]u16 = undefined;
    for (0..stacked.len / tile_n) |e| {
        const trellis = stacked[e * tile_n ..][0..tile_n];
        for (0..in_tiles) |tk| {
            for (0..out_tiles) |tn| {
                exl3.decodeTile(trellis[(tk * out_tiles + tn) * packed_n ..][0..packed_n], rate, dec, &tile_w);
                for (0..16) |r| {
                    const row = w[e * in_dim * out_dim + (tk * 16 + r) * out_dim ..];
                    for (0..16) |c| row[tn * 16 + c] = tile_w[r * 16 + c];
                }
            }
        }
    }
    return w;
}

/// Score one arm's output and mlx's own f16 matmul over the same decoded
/// weights, both against the exact dot product of those weights. Never
/// kernel-vs-kernel: the composite supplies a SCALE for the aggregate, never
/// a per-element answer.
fn measureInnerGemmParity(
    alloc: std.mem.Allocator,
    s: mlx.mlx_stream,
    got: []const f16,
    xh: []const u16,
    eids: []const u32,
    stacked: []const u16,
    in_dim: usize,
    out_dim: usize,
    rate: exl3.Rate,
    dec: exl3.Decode,
) !Exl3GemmParityStats {
    const w = try dequantStacked(alloc, stacked, in_dim, out_dim, rate, dec);
    return measureInnerGemmParityOn(alloc, s, got, xh, eids, w, in_dim, out_dim);
}

/// The same verdict against a weight bank the caller already holds — a
/// reference decode, where the bar must not be our own decoder's output.
fn measureInnerGemmParityOn(
    alloc: std.mem.Allocator,
    s: mlx.mlx_stream,
    got: []const f16,
    xh: []const u16,
    eids: []const u32,
    w: []const u16,
    in_dim: usize,
    out_dim: usize,
) !Exl3GemmParityStats {
    const rows = eids.len;
    const w_arr = mlx.mlx_array_new_data(w.ptr, &[_]c_int{ @intCast(w.len / (in_dim * out_dim)), @intCast(in_dim), @intCast(out_dim) }, 3, .float16);
    defer _ = mlx.mlx_array_free(w_arr);
    const eid_arr = mlx.mlx_array_new_data(eids.ptr, &[_]c_int{@intCast(rows)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(eid_arr);
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(rows), 1, @intCast(in_dim) }, 3, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    var w_sel = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_sel);
    try mlx.check(mlx.mlx_take_axis(&w_sel, w_arr, eid_arr, 0, s));
    var comp = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(comp);
    try mlx.check(mlx.mlx_matmul(&comp, x_arr, w_sel, s));
    var comp_c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(comp_c);
    try mlx.check(mlx.mlx_contiguous(&comp_c, comp, false, s));
    try mlx.check(mlx.mlx_array_eval(comp_c));
    const comp_h = mlx.mlx_array_data_float16(comp_c) orelse return error.F16Unreadable;

    const truth = try alloc.alloc(f64, out_dim);
    const amag = try alloc.alloc(f64, out_dim);
    var st = Exl3GemmParityStats{ .n = rows * out_dim, .ceiling = Exl3GemmParity.elemCeiling(in_dim) };
    var se_k: f64 = 0;
    var se_c: f64 = 0;
    for (0..rows) |r| {
        @memset(truth, 0);
        @memset(amag, 0);
        const wb = w[@as(usize, eids[r]) * in_dim * out_dim ..];
        for (0..in_dim) |k| {
            // An f16 product is exact in f64, so this sum IS the dot product.
            const xv: f64 = exl3.f16BitsToF32(xh[r * in_dim + k]);
            const wrow = wb[k * out_dim ..][0..out_dim];
            for (0..out_dim) |o| {
                const p = xv * @as(f64, exl3.f16BitsToF32(wrow[o]));
                truth[o] += p;
                amag[o] += @abs(p);
            }
        }
        for (0..out_dim) |o| {
            const g: f64 = @floatCast(got[r * out_dim + o]);
            const c: f64 = @floatCast(comp_h[r * out_dim + o]);
            if (!std.math.isFinite(g) or !std.math.isFinite(c) or !std.math.isFinite(truth[o])) {
                st.finite = false;
                return st;
            }
            const denom = if (amag[o] > 0) amag[o] else 1.0;
            const ek = @abs(g - truth[o]) / denom;
            const ec = @abs(c - truth[o]) / denom;
            st.kern_max = @max(st.kern_max, ek);
            st.comp_max = @max(st.comp_max, ec);
            se_k += ek * ek;
            se_c += ec * ec;
        }
    }
    const n: f64 = @floatFromInt(rows * out_dim);
    st.rms_kern = @sqrt(se_k / n);
    st.rms_comp = @sqrt(se_c / n);
    return st;
}

/// The verdict on measured stats, with one failure line naming both sides and
/// the bars.
fn reportGemmParity(st: Exl3GemmParityStats) !void {
    if (exl3GemmParityVerdict(st.finite, st.kern_max, st.rms_kern, st.rms_comp, st.ceiling)) |why| {
        std.debug.print(
            "exl3 GEMM parity FAIL ({s}): n={d} kernel max={d:.7} rms={d:.7}; composite max={d:.7} rms={d:.7}; " ++
                "bars: max<={d:.7} rms<={d:.1}x\n",
            .{ @tagName(why), st.n, st.kern_max, st.rms_kern, st.comp_max, st.rms_comp, st.ceiling, Exl3GemmParity.RMS_FACTOR },
        );
        return error.TestExpectedApproxEq;
    }
}

fn expectInnerGemmParity(
    alloc: std.mem.Allocator,
    s: mlx.mlx_stream,
    got: []const f16,
    xh: []const u16,
    eids: []const u32,
    stacked: []const u16,
    in_dim: usize,
    out_dim: usize,
    rate: exl3.Rate,
    dec: exl3.Decode,
) !void {
    try reportGemmParity(try measureInnerGemmParity(alloc, s, got, xh, eids, stacked, in_dim, out_dim, rate, dec));
}

test "exl3GemmParityVerdict: cancellation noise passes, a wrong weight or a systematic drift does not" {
    const ceiling = Exl3GemmParity.elemCeiling(128);
    // The arm's own rounding: a worst element inside the f16 store's share of
    // the bar, an RMS that matches the composite's.
    try std.testing.expect(exl3GemmParityVerdict(true, 2.19e-4, 2.65e-5, 2.65e-5, ceiling) == null);
    // One decoded weight wrong moves a single element far past the ceiling
    // while the RMS barely stirs: the element bar is the one that sees it.
    try std.testing.expectEqual(
        Exl3GemmParityFail.gross_element,
        exl3GemmParityVerdict(true, 8.0e-3, 2.7e-5, 2.65e-5, ceiling).?,
    );
    // Noisier everywhere without one gross element: the RMS ratio sees it.
    try std.testing.expectEqual(
        Exl3GemmParityFail.systematic,
        exl3GemmParityVerdict(true, 4.0e-4, 1.0e-4, 2.65e-5, ceiling).?,
    );
    try std.testing.expectEqual(
        Exl3GemmParityFail.nonfinite,
        exl3GemmParityVerdict(false, 0, 0, 0, ceiling).?,
    );
    // An arm better than the composite is never a failure.
    try std.testing.expect(exl3GemmParityVerdict(true, 1.0e-5, 1.0e-6, 2.65e-5, ceiling) == null);
    // The ceiling follows the accumulation depth.
    try std.testing.expect(Exl3GemmParity.elemCeiling(2560) > ceiling);
}

test "exl3 K4 cooperative indexed GEMV matches host MUL1 on production shape" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const E: usize = 4;
    const topk: usize = 10;
    const in_dim: usize = 2560;
    const out_dim: usize = 640;
    const in_tiles = in_dim / 16;
    const out_tiles = out_dim / 16;
    const tile_n = in_tiles * out_tiles * 64;
    const stacked = try alloc.alloc(u16, E * tile_n);
    var prng = std.Random.DefaultPrng.init(23);
    const rnd = prng.random();
    for (stacked) |*v| v.* = @truncate(rnd.int(u32));
    const xh = try alloc.alloc(u16, topk * in_dim);
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
    const slots_h = try alloc.alloc(u32, topk);
    for (slots_h, 0..) |*v, i| v.* = @intCast(i % E);
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(topk), @intCast(in_dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr_arr = mlx.mlx_array_new_data(stacked.ptr, &[_]c_int{ @intCast(E), @intCast(in_tiles), @intCast(out_tiles), 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr_arr);
    const slots = mlx.mlx_array_new_data(slots_h.ptr, &[_]c_int{@intCast(topk)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(slots);
    const got = try indexedGemvCoopF16(s, x_arr, tr_arr, slots);
    defer _ = mlx.mlx_array_free(got);
    var contig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(contig);
    try mlx.check(mlx.mlx_contiguous(&contig, got, false, s));
    try mlx.check(mlx.mlx_array_eval(contig));
    const src = mlx.mlx_array_data_float16(contig) orelse return error.F16Unreadable;
    try expectInnerGemmParity(alloc, s, src[0 .. topk * out_dim], xh, slots_h, stacked, in_dim, out_dim, exl3.Rate.fromK(4), .mul1);
}

test "exl3 K4 cooperative indexed GEMV runs at production shape" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const E: usize = 16;
    const topk: usize = 10;
    const in_dim: usize = 2560;
    const out_dim: usize = 640;
    const in_tiles = in_dim / 16;
    const out_tiles = out_dim / 16;
    const tile_n = in_tiles * out_tiles * 64;
    const stacked = try alloc.alloc(u16, E * tile_n);
    var prng = std.Random.DefaultPrng.init(29);
    const rnd = prng.random();
    for (stacked) |*v| v.* = @truncate(rnd.int(u32));
    const xh = try alloc.alloc(u16, topk * in_dim);
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 0.1);
    const slots_h = try alloc.alloc(u32, topk);
    for (slots_h, 0..) |*v, i| v.* = @intCast(i % E);
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(topk), @intCast(in_dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr_arr = mlx.mlx_array_new_data(stacked.ptr, &[_]c_int{ @intCast(E), @intCast(in_tiles), @intCast(out_tiles), 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr_arr);
    const slots = mlx.mlx_array_new_data(slots_h.ptr, &[_]c_int{@intCast(topk)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(slots);
    const warm_new = try indexedGemvCoopF16(s, x_arr, tr_arr, slots);
    try mlx.check(mlx.mlx_array_eval(warm_new));
    _ = mlx.mlx_array_free(warm_new);
    var new_ns: u64 = 0;
    var it: usize = 0;
    while (it < 10) : (it += 1) {
        var sw = io_util.Stopwatch.init(t.io);
        const b = try indexedGemvCoopF16(s, x_arr, tr_arr, slots);
        try mlx.check(mlx.mlx_array_eval(b));
        new_ns += sw.read();
        _ = mlx.mlx_array_free(b);
    }
    new_ns /= 10;
    const packed3 = exl3.packedHalfwords(3);
    const tile3 = in_tiles * out_tiles * packed3;
    const stacked3 = try alloc.alloc(u16, E * tile3);
    for (stacked3) |*v| v.* = @truncate(rnd.int(u32));
    const tr3 = mlx.mlx_array_new_data(stacked3.ptr, &[_]c_int{ @intCast(E), @intCast(in_tiles), @intCast(out_tiles), @intCast(packed3) }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr3);
    const warm3 = try indexedGemvCoopF16(s, x_arr, tr3, slots);
    try mlx.check(mlx.mlx_array_eval(warm3));
    _ = mlx.mlx_array_free(warm3);
    var k3_ns: u64 = 0;
    var k4_ns: u64 = 0;
    it = 0;
    while (it < 10) : (it += 1) {
        var sw4 = io_util.Stopwatch.init(t.io);
        const b4 = try indexedGemvCoopF16(s, x_arr, tr_arr, slots);
        try mlx.check(mlx.mlx_array_eval(b4));
        k4_ns += sw4.read();
        _ = mlx.mlx_array_free(b4);
        var sw3 = io_util.Stopwatch.init(t.io);
        const b3 = try indexedGemvCoopF16(s, x_arr, tr3, slots);
        try mlx.check(mlx.mlx_array_eval(b3));
        k3_ns += sw3.read();
        _ = mlx.mlx_array_free(b3);
    }
    k3_ns /= 10;
    k4_ns /= 10;
    benchPrint("exl3 indexed GEMV H=2560 I=640 topk=10: K4 {d} us K3 {d} us\n", .{
        k4_ns / 1000,
        k3_ns / 1000,
    });
    try t.expect(new_ns > 0);
    try t.expect(k4_ns > 0);
    benchPrint("[exl3-k3-timing] k3 {d} us k4 {d} us ratio {d:.3}\n", .{ k3_ns / 1000, k4_ns / 1000, @as(f64, @floatFromInt(k3_ns)) / @as(f64, @floatFromInt(@max(k4_ns, 1))) });
    try t.expect(k3_ns < k4_ns * 3);
}

fn layerUbench(dec: exl3.Decode) !void {
    setDecodeParams(dec);
    defer setDecodeParams(.mul1);
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    defer {
        ubench_mute = false;
    }
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const E: usize = 16;
    const topk: usize = 10;
    const in_dim: usize = 2560;
    const out_dim: usize = 640;
    const in_tiles = in_dim / 16;
    const out_tiles = out_dim / 16;
    const g_n = in_tiles * out_tiles * 64;
    const d_n = out_tiles * in_tiles * 64;
    const tr_g = try alloc.alloc(u16, E * g_n);
    const tr_d = try alloc.alloc(u16, E * d_n);
    const suh_g = try alloc.alloc(u16, E * in_dim);
    const svh_g = try alloc.alloc(u16, E * out_dim);
    const suh_d = try alloc.alloc(u16, E * out_dim);
    const svh_d = try alloc.alloc(u16, E * in_dim);
    var prng = std.Random.DefaultPrng.init(71);
    const rnd = prng.random();
    for (tr_g) |*v| v.* = @truncate(rnd.int(u32));
    for (tr_d) |*v| v.* = @truncate(rnd.int(u32));
    for (suh_g) |*v| v.* = exl3.f32ToF16Bits(1.0);
    for (svh_g) |*v| v.* = exl3.f32ToF16Bits(1.0);
    for (suh_d) |*v| v.* = exl3.f32ToF16Bits(1.0);
    for (svh_d) |*v| v.* = exl3.f32ToF16Bits(1.0);
    const trg = mlx.mlx_array_new_data(tr_g.ptr, &[_]c_int{ @intCast(E), @intCast(in_tiles), @intCast(out_tiles), 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(trg);
    const trd = mlx.mlx_array_new_data(tr_d.ptr, &[_]c_int{ @intCast(E), @intCast(out_tiles), @intCast(in_tiles), 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(trd);
    const sugh = mlx.mlx_array_new_data(suh_g.ptr, &[_]c_int{ @intCast(E), @intCast(in_dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(sugh);
    const svgi = mlx.mlx_array_new_data(svh_g.ptr, &[_]c_int{ @intCast(E), @intCast(out_dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(svgi);
    const sudi = mlx.mlx_array_new_data(suh_d.ptr, &[_]c_int{ @intCast(E), @intCast(out_dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(sudi);
    const svdh = mlx.mlx_array_new_data(svh_d.ptr, &[_]c_int{ @intCast(E), @intCast(in_dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(svdh);
    const x1h = try alloc.alloc(u16, in_dim);
    for (x1h) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 0.1);
    const sl1 = try alloc.alloc(u32, topk);
    const sc1 = try alloc.alloc(f32, topk);
    for (sl1, 0..) |*v, i| v.* = @intCast(i % E);
    for (sc1) |*v| v.* = 0.1;
    const x1 = mlx.mlx_array_new_data(x1h.ptr, &[_]c_int{@intCast(in_dim)}, 1, .float16);
    defer _ = mlx.mlx_array_free(x1);
    const slots1 = mlx.mlx_array_new_data(sl1.ptr, &[_]c_int{@intCast(topk)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(slots1);
    const scores1 = mlx.mlx_array_new_data(sc1.ptr, &[_]c_int{@intCast(topk)}, 1, .float32);
    defer _ = mlx.mlx_array_free(scores1);
    ubench_mute = true;
    const warm1 = try moeSwigluFused(s, x1, trg, sugh, svgi, trg, sugh, svgi, trd, sudi, svdh, slots1, scores1, .float16);
    try mlx.check(mlx.mlx_array_eval(warm1));
    _ = mlx.mlx_array_free(warm1);
    ubench_mute = false;
    benchPrint("exl3-ubench codebook={s} rows=1\n", .{@tagName(dec.codebook)});
    const y1 = try moeSwigluFused(s, x1, trg, sugh, svgi, trg, sugh, svgi, trd, sudi, svdh, slots1, scores1, .float16);
    try mlx.check(mlx.mlx_array_eval(y1));
    _ = mlx.mlx_array_free(y1);
    const R: usize = 512;
    const xnh = try alloc.alloc(u16, R * in_dim);
    for (xnh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 0.1);
    const sln = try alloc.alloc(u32, R * topk);
    const scn = try alloc.alloc(f32, R * topk);
    for (sln, 0..) |*v, i| v.* = @intCast(i % E);
    for (scn) |*v| v.* = 0.1;
    const xn = mlx.mlx_array_new_data(xnh.ptr, &[_]c_int{ @intCast(R), @intCast(in_dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(xn);
    const slotsn = mlx.mlx_array_new_data(sln.ptr, &[_]c_int{@intCast(R * topk)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(slotsn);
    const scoresn = mlx.mlx_array_new_data(scn.ptr, &[_]c_int{@intCast(R * topk)}, 1, .float32);
    defer _ = mlx.mlx_array_free(scoresn);
    ubench_mute = true;
    const warmn = try moePrefill(s, xn, trg, sugh, svgi, trg, sugh, svgi, trd, sudi, svdh, slotsn, scoresn, @intCast(topk));
    try mlx.check(mlx.mlx_array_eval(warmn));
    _ = mlx.mlx_array_free(warmn);
    ubench_mute = false;
    benchPrint("exl3-ubench codebook={s} rows=512\n", .{@tagName(dec.codebook)});
    const yn = try moePrefill(s, xn, trg, sugh, svgi, trg, sugh, svgi, trd, sudi, svdh, slotsn, scoresn, @intCast(topk));
    try mlx.check(mlx.mlx_array_eval(yn));
    _ = mlx.mlx_array_free(yn);
    for ([_]usize{ 2, 3 }) |Rsmall| {
        const xsh = try alloc.alloc(u16, Rsmall * in_dim);
        for (xsh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 0.1);
        const sls = try alloc.alloc(u32, Rsmall * topk);
        const scs = try alloc.alloc(f32, Rsmall * topk);
        for (sls, 0..) |*v, i| v.* = @intCast(i % E);
        for (scs) |*v| v.* = 0.1;
        const xs = mlx.mlx_array_new_data(xsh.ptr, &[_]c_int{ @intCast(Rsmall), @intCast(in_dim) }, 2, .float16);
        defer _ = mlx.mlx_array_free(xs);
        const slotss = mlx.mlx_array_new_data(sls.ptr, &[_]c_int{@intCast(Rsmall * topk)}, 1, .uint32);
        defer _ = mlx.mlx_array_free(slotss);
        const scoress = mlx.mlx_array_new_data(scs.ptr, &[_]c_int{@intCast(Rsmall * topk)}, 1, .float32);
        defer _ = mlx.mlx_array_free(scoress);
        ubench_mute = true;
        const warms = try moeSwigluFused(s, xs, trg, sugh, svgi, trg, sugh, svgi, trd, sudi, svdh, slotss, scoress, .float16);
        try mlx.check(mlx.mlx_array_eval(warms));
        _ = mlx.mlx_array_free(warms);
        ubench_mute = false;
        benchPrint("exl3-ubench codebook={s} rows={d}\n", .{ @tagName(dec.codebook), Rsmall });
        const ys = try moeSwigluFused(s, xs, trg, sugh, svgi, trg, sugh, svgi, trd, sudi, svdh, slotss, scoress, .float16);
        try mlx.check(mlx.mlx_array_eval(ys));
        _ = mlx.mlx_array_free(ys);
    }
}

test "exl3 layer ubench production shape rows=1 and 512 per codebook" {
    // With the ubench on, the two codebooks interleave over several rounds so
    // GPU clock ramp lands on both; read medians per kernel, never one shot.
    const rounds: usize = if (exl3UbenchOn()) 5 else 1;
    for (0..rounds) |_| {
        for ([_]exl3.Decode{ .mul1, .mcg }) |dec| try layerUbench(dec);
    }
}

test "exl3 fused decode chain matches indexed SwiGLU on one row" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const fixture = exl3.fixtures.k4;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const header_len = std.mem.readInt(u64, fixture[0..8], .little);
    const header = fixture[8 .. 8 + header_len];
    const data = fixture[8 + header_len ..];
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header, .{});
    defer parsed.deinit();
    const trellis_meta = parsed.value.object.get("trellis").?.object;
    const suh_meta = parsed.value.object.get("suh").?.object;
    const svh_meta = parsed.value.object.get("svh").?.object;
    const t0: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[0].integer);
    const t1: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[1].integer);
    const s0: usize = @intCast(suh_meta.get("data_offsets").?.array.items[0].integer);
    const s1: usize = @intCast(suh_meta.get("data_offsets").?.array.items[1].integer);
    const v0: usize = @intCast(svh_meta.get("data_offsets").?.array.items[0].integer);
    const v1: usize = @intCast(svh_meta.get("data_offsets").?.array.items[1].integer);
    const trellis_bits = std.mem.bytesAsSlice(u16, data[t0..t1]);
    const suh_bits = std.mem.bytesAsSlice(u16, data[s0..s1]);
    const svh_bits = std.mem.bytesAsSlice(u16, data[v0..v1]);
    const E: usize = 4;
    const dim: usize = 128;
    const topk: usize = 4;
    const tile_n = 8 * 8 * 64;
    const stacked_t = try alloc.alloc(u16, E * tile_n);
    const stacked_suh = try alloc.alloc(u16, E * dim);
    const stacked_svh = try alloc.alloc(u16, E * dim);
    for (0..E) |e| {
        @memcpy(stacked_t[e * tile_n ..][0..tile_n], trellis_bits);
        @memcpy(stacked_suh[e * dim ..][0..dim], suh_bits);
        @memcpy(stacked_svh[e * dim ..][0..dim], svh_bits);
    }
    var prng = std.Random.DefaultPrng.init(41);
    const rnd = prng.random();
    const xh = try alloc.alloc(u16, dim);
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
    const slots_h = try alloc.alloc(u32, topk);
    const scores_h = try alloc.alloc(f32, topk);
    for (slots_h, 0..) |*v, i| v.* = @intCast(i % E);
    for (scores_h) |*v| v.* = rnd.float(f32);
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{@intCast(dim)}, 1, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr = mlx.mlx_array_new_data(stacked_t.ptr, &[_]c_int{ @intCast(E), 8, 8, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr);
    const suh = mlx.mlx_array_new_data(stacked_suh.ptr, &[_]c_int{ @intCast(E), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(suh);
    const svh = mlx.mlx_array_new_data(stacked_svh.ptr, &[_]c_int{ @intCast(E), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(svh);
    const slots = mlx.mlx_array_new_data(slots_h.ptr, &[_]c_int{@intCast(topk)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(slots);
    const scores = mlx.mlx_array_new_data(scores_h.ptr, &[_]c_int{@intCast(topk)}, 1, .float32);
    defer _ = mlx.mlx_array_free(scores);
    resetFusedDispatchCount();
    pair_splits_force = 1;
    const fused = try moeSwigluFused(s, x_arr, tr, suh, svh, tr, suh, svh, tr, suh, svh, slots, scores, .float16);
    defer _ = mlx.mlx_array_free(fused);
    const n_disp = fusedDispatchCount();
    const old = try moeSwigluIndexed(s, x_arr, tr, suh, svh, tr, suh, svh, tr, suh, svh, slots, scores);
    defer _ = mlx.mlx_array_free(old);
    var c_f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c_f);
    var c_o = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c_o);
    try mlx.check(mlx.mlx_contiguous(&c_f, fused, false, s));
    try mlx.check(mlx.mlx_contiguous(&c_o, old, false, s));
    try mlx.check(mlx.mlx_array_eval(c_f));
    try mlx.check(mlx.mlx_array_eval(c_o));
    const sf = mlx.mlx_array_data_float16(c_f) orelse return error.F16Unreadable;
    const so = mlx.mlx_array_data_float16(c_o) orelse return error.F16Unreadable;
    try expectRelRms(sf[0..dim], so[0..dim], 0.01);
    // The decode chain is three dispatches: pair GEMV (prepare inlined), fused mid+down, finish reduce.
    try t.expectEqual(@as(u32, 3), n_disp);
    const fused_bf = try moeSwigluFused(s, x_arr, tr, suh, svh, tr, suh, svh, tr, suh, svh, slots, scores, .bfloat16);
    defer _ = mlx.mlx_array_free(fused_bf);
    try t.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(fused_bf));
    pair_splits_force = 2;
    defer {
        pair_splits_force = null;
    }
    const fused2 = try moeSwigluFused(s, x_arr, tr, suh, svh, tr, suh, svh, tr, suh, svh, slots, scores, .float16);
    defer _ = mlx.mlx_array_free(fused2);
    var c2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c2);
    try mlx.check(mlx.mlx_contiguous(&c2, fused2, false, s));
    try mlx.check(mlx.mlx_array_eval(c2));
    const s2 = mlx.mlx_array_data_float16(c2) orelse return error.F16Unreadable;
    try expectRelRms(s2[0..dim], sf[0..dim], 0.01);
}

test "exl3 fused decode chain rows match N solo calls" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const fixture = exl3.fixtures.k4;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const header_len = std.mem.readInt(u64, fixture[0..8], .little);
    const header = fixture[8 .. 8 + header_len];
    const data = fixture[8 + header_len ..];
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header, .{});
    defer parsed.deinit();
    const trellis_meta = parsed.value.object.get("trellis").?.object;
    const suh_meta = parsed.value.object.get("suh").?.object;
    const svh_meta = parsed.value.object.get("svh").?.object;
    const t0: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[0].integer);
    const t1: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[1].integer);
    const s0: usize = @intCast(suh_meta.get("data_offsets").?.array.items[0].integer);
    const s1: usize = @intCast(suh_meta.get("data_offsets").?.array.items[1].integer);
    const v0: usize = @intCast(svh_meta.get("data_offsets").?.array.items[0].integer);
    const v1: usize = @intCast(svh_meta.get("data_offsets").?.array.items[1].integer);
    const trellis_bits = std.mem.bytesAsSlice(u16, data[t0..t1]);
    const suh_bits = std.mem.bytesAsSlice(u16, data[s0..s1]);
    const svh_bits = std.mem.bytesAsSlice(u16, data[v0..v1]);
    const E: usize = 4;
    const dim: usize = 128;
    const topk: usize = 4;
    const rows: usize = 8;
    const tile_n = 8 * 8 * 64;
    const stacked_t = try alloc.alloc(u16, E * tile_n);
    const stacked_suh = try alloc.alloc(u16, E * dim);
    const stacked_svh = try alloc.alloc(u16, E * dim);
    for (0..E) |e| {
        @memcpy(stacked_t[e * tile_n ..][0..tile_n], trellis_bits);
        @memcpy(stacked_suh[e * dim ..][0..dim], suh_bits);
        @memcpy(stacked_svh[e * dim ..][0..dim], svh_bits);
    }
    var prng = std.Random.DefaultPrng.init(43);
    const rnd = prng.random();
    const xh = try alloc.alloc(u16, rows * dim);
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
    const slots_h = try alloc.alloc(u32, rows * topk);
    const scores_h = try alloc.alloc(f32, rows * topk);
    for (slots_h, 0..) |*v, i| v.* = @intCast(i % E);
    for (scores_h) |*v| v.* = rnd.float(f32);
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(rows), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr = mlx.mlx_array_new_data(stacked_t.ptr, &[_]c_int{ @intCast(E), 8, 8, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr);
    const suh = mlx.mlx_array_new_data(stacked_suh.ptr, &[_]c_int{ @intCast(E), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(suh);
    const svh = mlx.mlx_array_new_data(stacked_svh.ptr, &[_]c_int{ @intCast(E), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(svh);
    const slots = mlx.mlx_array_new_data(slots_h.ptr, &[_]c_int{@intCast(rows * topk)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(slots);
    const scores = mlx.mlx_array_new_data(scores_h.ptr, &[_]c_int{@intCast(rows * topk)}, 1, .float32);
    defer _ = mlx.mlx_array_free(scores);
    pair_splits_force = 1;
    defer {
        pair_splits_force = null;
    }
    const fused = try moeSwigluFused(s, x_arr, tr, suh, svh, tr, suh, svh, tr, suh, svh, slots, scores, .float16);
    defer _ = mlx.mlx_array_free(fused);
    var c_f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c_f);
    try mlx.check(mlx.mlx_contiguous(&c_f, fused, false, s));
    try mlx.check(mlx.mlx_array_eval(c_f));
    const sf = mlx.mlx_array_data_float16(c_f) orelse return error.F16Unreadable;
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        const x1 = mlx.mlx_array_new_data(xh[r * dim ..][0..dim].ptr, &[_]c_int{@intCast(dim)}, 1, .float16);
        defer _ = mlx.mlx_array_free(x1);
        const sl = mlx.mlx_array_new_data(slots_h[r * topk ..][0..topk].ptr, &[_]c_int{@intCast(topk)}, 1, .uint32);
        defer _ = mlx.mlx_array_free(sl);
        const sc = mlx.mlx_array_new_data(scores_h[r * topk ..][0..topk].ptr, &[_]c_int{@intCast(topk)}, 1, .float32);
        defer _ = mlx.mlx_array_free(sc);
        const solo = try moeSwigluIndexed(s, x1, tr, suh, svh, tr, suh, svh, tr, suh, svh, sl, sc);
        defer _ = mlx.mlx_array_free(solo);
        var c_s = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(c_s);
        try mlx.check(mlx.mlx_contiguous(&c_s, solo, false, s));
        try mlx.check(mlx.mlx_array_eval(c_s));
        const ss = mlx.mlx_array_data_float16(c_s) orelse return error.F16Unreadable;
        try expectRelRms(sf[r * dim ..][0..dim], ss[0..dim], 0.01);
    }
}

test "exl3 pair split clamps to one where it would cut a Hadamard block" {
    const t = std.testing;
    setPairSplitsForTest(4);
    defer setPairSplitsForTest(null);
    try t.expectEqual(@as(u32, 4), pairSplitCountFor(2048));
    try t.expectEqual(@as(u32, 1), pairSplitCountFor(1280));
    try t.expectEqual(@as(u32, 1), pairSplitCountFor(128));
}

test "exl3 split-2 decode chain matches split-1 on a k range the GEMV stages itself" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const E: usize = 4;
    const topk: usize = 2;
    const dim: usize = 256;
    const inter: usize = 128;
    const gt = try alloc.alloc(u16, E * (dim / 16) * (inter / 16) * 64);
    const dt = try alloc.alloc(u16, E * (inter / 16) * (dim / 16) * 64);
    var prng = std.Random.DefaultPrng.init(9);
    const rnd = prng.random();
    for (gt) |*v| v.* = @truncate(rnd.int(u32));
    for (dt) |*v| v.* = @truncate(rnd.int(u32));
    const suh_in = try alloc.alloc(u16, E * dim);
    const svh_in = try alloc.alloc(u16, E * inter);
    const suh_d = try alloc.alloc(u16, E * inter);
    const svh_d = try alloc.alloc(u16, E * dim);
    for (suh_in) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) + 0.5);
    for (svh_in) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) + 0.5);
    for (suh_d) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) + 0.5);
    for (svh_d) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) + 0.5);
    const gta = mlx.mlx_array_new_data(gt.ptr, &[_]c_int{ E, dim / 16, inter / 16, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(gta);
    const dta = mlx.mlx_array_new_data(dt.ptr, &[_]c_int{ E, inter / 16, dim / 16, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(dta);
    const suh_in_a = mlx.mlx_array_new_data(suh_in.ptr, &[_]c_int{ E, dim }, 2, .float16);
    defer _ = mlx.mlx_array_free(suh_in_a);
    const svh_in_a = mlx.mlx_array_new_data(svh_in.ptr, &[_]c_int{ E, inter }, 2, .float16);
    defer _ = mlx.mlx_array_free(svh_in_a);
    const suh_d_a = mlx.mlx_array_new_data(suh_d.ptr, &[_]c_int{ E, inter }, 2, .float16);
    defer _ = mlx.mlx_array_free(suh_d_a);
    const svh_d_a = mlx.mlx_array_new_data(svh_d.ptr, &[_]c_int{ E, dim }, 2, .float16);
    defer _ = mlx.mlx_array_free(svh_d_a);
    const xh = try alloc.alloc(u16, dim);
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
    const x = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{dim}, 1, .float16);
    defer _ = mlx.mlx_array_free(x);
    const sl = try alloc.alloc(u32, topk);
    for (sl, 0..) |*v, i| v.* = @intCast(i % E);
    const slots = mlx.mlx_array_new_data(sl.ptr, &[_]c_int{topk}, 1, .uint32);
    defer _ = mlx.mlx_array_free(slots);
    const sc = try alloc.alloc(f32, topk);
    for (sc) |*v| v.* = 0.5;
    const scores = mlx.mlx_array_new_data(sc.ptr, &[_]c_int{topk}, 1, .float32);
    defer _ = mlx.mlx_array_free(scores);
    const solo = blk: {
        setPairSplitsForTest(1);
        defer setPairSplitsForTest(null);
        break :blk try moeSwigluFused(s, x, gta, suh_in_a, svh_in_a, gta, suh_in_a, svh_in_a, dta, suh_d_a, svh_d_a, slots, scores, .float16);
    };
    defer _ = mlx.mlx_array_free(solo);
    const split = blk: {
        setPairSplitsForTest(2);
        defer setPairSplitsForTest(null);
        break :blk try moeSwigluFused(s, x, gta, suh_in_a, svh_in_a, gta, suh_in_a, svh_in_a, dta, suh_d_a, svh_d_a, slots, scores, .float16);
    };
    defer _ = mlx.mlx_array_free(split);
    try mlx.check(mlx.mlx_array_eval(solo));
    try mlx.check(mlx.mlx_array_eval(split));
    const a = mlx.mlx_array_data_float16(solo) orelse return error.F16Unreadable;
    const b = mlx.mlx_array_data_float16(split) orelse return error.F16Unreadable;
    try expectRelRms(b[0..dim], a[0..dim], 0.01);
}

test "exl3 pair GEMV inner planes are f32" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const fixture = exl3.fixtures.k4;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const header_len = std.mem.readInt(u64, fixture[0..8], .little);
    const header = fixture[8 .. 8 + header_len];
    const data = fixture[8 + header_len ..];
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header, .{});
    defer parsed.deinit();
    const trellis_meta = parsed.value.object.get("trellis").?.object;
    const suh_meta = parsed.value.object.get("suh").?.object;
    const t0: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[0].integer);
    const t1: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[1].integer);
    const s0: usize = @intCast(suh_meta.get("data_offsets").?.array.items[0].integer);
    const s1: usize = @intCast(suh_meta.get("data_offsets").?.array.items[1].integer);
    const trellis_bits = std.mem.bytesAsSlice(u16, data[t0..t1]);
    const suh_bits = std.mem.bytesAsSlice(u16, data[s0..s1]);
    const E: usize = 4;
    const dim: usize = 128;
    const topk: usize = 4;
    const tile_n = 8 * 8 * 64;
    const stacked_t = try alloc.alloc(u16, E * tile_n);
    const stacked_suh = try alloc.alloc(u16, E * dim);
    for (0..E) |e| {
        @memcpy(stacked_t[e * tile_n ..][0..tile_n], trellis_bits);
        @memcpy(stacked_suh[e * dim ..][0..dim], suh_bits);
    }
    var prng = std.Random.DefaultPrng.init(47);
    const rnd = prng.random();
    const xh = try alloc.alloc(u16, dim);
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
    const slots_h = try alloc.alloc(u32, topk);
    for (slots_h, 0..) |*v, i| v.* = @intCast(i % E);
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{@intCast(dim)}, 1, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr = mlx.mlx_array_new_data(stacked_t.ptr, &[_]c_int{ @intCast(E), 8, 8, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr);
    const suh = mlx.mlx_array_new_data(stacked_suh.ptr, &[_]c_int{ @intCast(E), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(suh);
    const slots = mlx.mlx_array_new_data(slots_h.ptr, &[_]c_int{@intCast(topk)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(slots);
    pair_splits_force = 1;
    defer {
        pair_splits_force = null;
    }
    const inners = try pairGemv(s, x_arr, suh, suh, tr, tr, slots, @intCast(dim), @intCast(dim), @intCast(topk), @intCast(topk), 0);
    defer _ = mlx.mlx_array_free(inners[0]);
    defer _ = mlx.mlx_array_free(inners[1]);
    try mlx.check(mlx.mlx_array_eval(inners[0]));
    try t.expectEqual(mlx.mlx_dtype.float32, mlx.mlx_array_dtype(inners[0]));
    try t.expectEqual(mlx.mlx_dtype.float32, mlx.mlx_array_dtype(inners[1]));
}

test "exl3 fused rows at split-2 match N fused solo" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const fixture = exl3.fixtures.k4;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const header_len = std.mem.readInt(u64, fixture[0..8], .little);
    const header = fixture[8 .. 8 + header_len];
    const data = fixture[8 + header_len ..];
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header, .{});
    defer parsed.deinit();
    const trellis_meta = parsed.value.object.get("trellis").?.object;
    const suh_meta = parsed.value.object.get("suh").?.object;
    const svh_meta = parsed.value.object.get("svh").?.object;
    const t0: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[0].integer);
    const t1: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[1].integer);
    const s0: usize = @intCast(suh_meta.get("data_offsets").?.array.items[0].integer);
    const s1: usize = @intCast(suh_meta.get("data_offsets").?.array.items[1].integer);
    const v0: usize = @intCast(svh_meta.get("data_offsets").?.array.items[0].integer);
    const v1: usize = @intCast(svh_meta.get("data_offsets").?.array.items[1].integer);
    const trellis_bits = std.mem.bytesAsSlice(u16, data[t0..t1]);
    const suh_bits = std.mem.bytesAsSlice(u16, data[s0..s1]);
    const svh_bits = std.mem.bytesAsSlice(u16, data[v0..v1]);
    const E: usize = 4;
    const dim: usize = 128;
    const topk: usize = 4;
    const rows: usize = 8;
    const tile_n = 8 * 8 * 64;
    const stacked_t = try alloc.alloc(u16, E * tile_n);
    const stacked_suh = try alloc.alloc(u16, E * dim);
    const stacked_svh = try alloc.alloc(u16, E * dim);
    for (0..E) |e| {
        @memcpy(stacked_t[e * tile_n ..][0..tile_n], trellis_bits);
        @memcpy(stacked_suh[e * dim ..][0..dim], suh_bits);
        @memcpy(stacked_svh[e * dim ..][0..dim], svh_bits);
    }
    var prng = std.Random.DefaultPrng.init(43);
    const rnd = prng.random();
    const xh = try alloc.alloc(u16, rows * dim);
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
    const slots_h = try alloc.alloc(u32, rows * topk);
    const scores_h = try alloc.alloc(f32, rows * topk);
    for (slots_h, 0..) |*v, i| v.* = @intCast(i % E);
    for (scores_h) |*v| v.* = rnd.float(f32);
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(rows), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr = mlx.mlx_array_new_data(stacked_t.ptr, &[_]c_int{ @intCast(E), 8, 8, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr);
    const suh = mlx.mlx_array_new_data(stacked_suh.ptr, &[_]c_int{ @intCast(E), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(suh);
    const svh = mlx.mlx_array_new_data(stacked_svh.ptr, &[_]c_int{ @intCast(E), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(svh);
    const slots = mlx.mlx_array_new_data(slots_h.ptr, &[_]c_int{@intCast(rows * topk)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(slots);
    const scores = mlx.mlx_array_new_data(scores_h.ptr, &[_]c_int{@intCast(rows * topk)}, 1, .float32);
    defer _ = mlx.mlx_array_free(scores);
    pair_splits_force = 2;
    defer {
        pair_splits_force = null;
    }
    const fused = try moeSwigluFused(s, x_arr, tr, suh, svh, tr, suh, svh, tr, suh, svh, slots, scores, .float16);
    defer _ = mlx.mlx_array_free(fused);
    var c_f = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c_f);
    try mlx.check(mlx.mlx_contiguous(&c_f, fused, false, s));
    try mlx.check(mlx.mlx_array_eval(c_f));
    const sf = mlx.mlx_array_data_float16(c_f) orelse return error.F16Unreadable;
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        const x1 = mlx.mlx_array_new_data(xh[r * dim ..][0..dim].ptr, &[_]c_int{@intCast(dim)}, 1, .float16);
        defer _ = mlx.mlx_array_free(x1);
        const sl = mlx.mlx_array_new_data(slots_h[r * topk ..][0..topk].ptr, &[_]c_int{@intCast(topk)}, 1, .uint32);
        defer _ = mlx.mlx_array_free(sl);
        const sc = mlx.mlx_array_new_data(scores_h[r * topk ..][0..topk].ptr, &[_]c_int{@intCast(topk)}, 1, .float32);
        defer _ = mlx.mlx_array_free(sc);
        const solo = try moeSwigluFused(s, x1, tr, suh, svh, tr, suh, svh, tr, suh, svh, sl, sc, .float16);
        defer _ = mlx.mlx_array_free(solo);
        var c_s = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(c_s);
        try mlx.check(mlx.mlx_contiguous(&c_s, solo, false, s));
        try mlx.check(mlx.mlx_array_eval(c_s));
        const ss = mlx.mlx_array_data_float16(c_s) orelse return error.F16Unreadable;
        for (0..dim) |i| {
            const a: u16 = @bitCast(sf[r * dim + i]);
            const b: u16 = @bitCast(ss[i]);
            try t.expectEqual(b, a);
        }
    }
}

test "exl3 MTP MoE rows stay on the fused decode arm" {
    const t = std.testing;
    try t.expect(!usesPrefillArm(1));
    try t.expect(!usesPrefillArm(4));
    try t.expect(!usesPrefillArm(16));
    try t.expect(usesPrefillArm(17));
}

fn sortedGemmSmallShape(dec: exl3.Decode) !void {
    setDecodeParams(dec);
    defer setDecodeParams(.mul1);
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const fixture = exl3.fixtures.k4;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const header_len = std.mem.readInt(u64, fixture[0..8], .little);
    const header = fixture[8 .. 8 + header_len];
    const data = fixture[8 + header_len ..];
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header, .{});
    defer parsed.deinit();
    const trellis_meta = parsed.value.object.get("trellis").?.object;
    const t0: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[0].integer);
    const t1: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[1].integer);
    const trellis_bits = std.mem.bytesAsSlice(u16, data[t0..t1]);
    const E: usize = 4;
    const dim: usize = 128;
    const n: usize = 8;
    const tile_n = 8 * 8 * 64;
    const stacked = try alloc.alloc(u16, E * tile_n);
    for (0..E) |e| @memcpy(stacked[e * tile_n ..][0..tile_n], trellis_bits);
    var prng = std.Random.DefaultPrng.init(47);
    const rnd = prng.random();
    const xh = try alloc.alloc(u16, n * dim);
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
    const eids = [_]u32{ 0, 0, 0, 0, 2, 2, 1, 1 };
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(n), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr_arr = mlx.mlx_array_new_data(stacked.ptr, &[_]c_int{ @intCast(E), 8, 8, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr_arr);
    const eid_a = mlx.mlx_array_new_data(&eids, &[_]c_int{@intCast(n)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(eid_a);
    const got = try innerGemmSorted(s, x_arr, tr_arr, eid_a);
    defer _ = mlx.mlx_array_free(got);
    var contig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(contig);
    try mlx.check(mlx.mlx_contiguous(&contig, got, false, s));
    try mlx.check(mlx.mlx_array_eval(contig));
    const src = mlx.mlx_array_data_float16(contig) orelse return error.F16Unreadable;
    try expectInnerGemmParity(alloc, s, src[0 .. n * dim], xh, &eids, stacked, dim, dim, exl3.Rate.fromK(4), dec);
}

test "exl3 sorted GEMM matches host MUL1 on small shape" {
    try sortedGemmSmallShape(.mul1);
}

test "exl3 sorted GEMM matches host MCG on small shape" {
    try sortedGemmSmallShape(.mcg);
}

test "exl3 a NAX GEMM source the Metal toolchain rejects is declined at the probe, not at prefill" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    mlx.installErrorHandler();
    try t.expect(!mlx.errorPending());
    try t.expect(buildNaxGemmKernel("this is not metal;", "", "sushi_exl3_probe_bad") == null);
    try t.expect(!mlx.errorPending());
    // Exercise the regression: an earlier failed dispatch may already have
    // latched NAX off, but that must not turn this capability check into a pass.
    if (!gemmNaxAvailable()) return; // the real NAX kernel needs M5-class hardware
    const previous_failure = gemm_nax_failed;
    defer gemm_nax_failed = previous_failure;
    gemm_nax_failed = true;
    try t.expect(gemmNaxAvailable());
    try t.expect(!gemmNaxOn());
    const real = buildNaxGemmKernel(GEMM_NAX_SOURCE, naxHeader(.mul1, .w16), "sushi_exl3_k4_gemm_nax") orelse return error.TestUnexpectedResult;
    _ = mlx.mlx_fast_metal_kernel_free(real);
    try t.expect(!mlx.errorPending());
}

test "exl3 K3 sorted GEMM matches host MUL1 on small shape" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const fixture = exl3.fixtures.k3;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const header_len = std.mem.readInt(u64, fixture[0..8], .little);
    const header = fixture[8 .. 8 + header_len];
    const data = fixture[8 + header_len ..];
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header, .{});
    defer parsed.deinit();
    const trellis_meta = parsed.value.object.get("trellis").?.object;
    const t0: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[0].integer);
    const t1: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[1].integer);
    const trellis_bits = std.mem.bytesAsSlice(u16, data[t0..t1]);
    const E: usize = 4;
    const dim: usize = 128;
    const n: usize = 8;
    const tile_n = 8 * 8 * 48;
    const stacked = try alloc.alloc(u16, E * tile_n);
    for (0..E) |e| @memcpy(stacked[e * tile_n ..][0..tile_n], trellis_bits);
    var prng = std.Random.DefaultPrng.init(47);
    const rnd = prng.random();
    const xh = try alloc.alloc(u16, n * dim);
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
    const eids = [_]u32{ 0, 0, 0, 0, 2, 2, 1, 1 };
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(n), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr_arr = mlx.mlx_array_new_data(stacked.ptr, &[_]c_int{ @intCast(E), 8, 8, 48 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr_arr);
    const eid_a = mlx.mlx_array_new_data(&eids, &[_]c_int{@intCast(n)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(eid_a);
    const got = try innerGemmSorted(s, x_arr, tr_arr, eid_a);
    defer _ = mlx.mlx_array_free(got);
    var contig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(contig);
    try mlx.check(mlx.mlx_contiguous(&contig, got, false, s));
    try mlx.check(mlx.mlx_array_eval(contig));
    const src = mlx.mlx_array_data_float16(contig) orelse return error.F16Unreadable;
    try expectInnerGemmParity(alloc, s, src[0 .. n * dim], xh, &eids, stacked, dim, dim, exl3.Rate.fromK(3), .mul1);
}

test "exl3 K4 sorted GEMM matches host MUL1 when a run half-fills the second block" {
    // 20-row runs put 16 rows in the first destination and 4 in the second, so
    // the second block's activation reads are the predicated ones.
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const fixture = exl3.fixtures.k4;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const header_len = std.mem.readInt(u64, fixture[0..8], .little);
    const header = fixture[8 .. 8 + header_len];
    const data = fixture[8 + header_len ..];
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header, .{});
    defer parsed.deinit();
    const trellis_meta = parsed.value.object.get("trellis").?.object;
    const t0: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[0].integer);
    const t1: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[1].integer);
    const trellis_bits = std.mem.bytesAsSlice(u16, data[t0..t1]);
    const E: usize = 4;
    const dim: usize = 128;
    const n: usize = 40;
    const tile_n = 8 * 8 * 64;
    const stacked = try alloc.alloc(u16, E * tile_n);
    for (0..E) |e| @memcpy(stacked[e * tile_n ..][0..tile_n], trellis_bits);
    var prng = std.Random.DefaultPrng.init(61);
    const rnd = prng.random();
    const xh = try alloc.alloc(u16, n * dim);
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
    var eids: [40]u32 = undefined;
    var i: usize = 0;
    while (i < 20) : (i += 1) eids[i] = 0;
    while (i < n) : (i += 1) eids[i] = 2;
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(n), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr_arr = mlx.mlx_array_new_data(stacked.ptr, &[_]c_int{ @intCast(E), 8, 8, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr_arr);
    const eid_a = mlx.mlx_array_new_data(&eids, &[_]c_int{@intCast(n)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(eid_a);
    const got = try innerGemmSortedWin(s, x_arr, tr_arr, eid_a, 32);
    defer _ = mlx.mlx_array_free(got);
    var contig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(contig);
    try mlx.check(mlx.mlx_contiguous(&contig, got, false, s));
    try mlx.check(mlx.mlx_array_eval(contig));
    const src = mlx.mlx_array_data_float16(contig) orelse return error.F16Unreadable;
    try expectInnerGemmParity(alloc, s, src[0 .. n * dim], xh, &eids, stacked, dim, dim, exl3.Rate.fromK(4), .mul1);
}

test "exl3 K3 sorted GEMM matches host MUL1 on 20-40 row runs" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    if (!gemmNaxOn()) return error.SkipZigTest;
    const fixture = exl3.fixtures.k3;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const header_len = std.mem.readInt(u64, fixture[0..8], .little);
    const header = fixture[8 .. 8 + header_len];
    const data = fixture[8 + header_len ..];
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header, .{});
    defer parsed.deinit();
    const trellis_meta = parsed.value.object.get("trellis").?.object;
    const t0: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[0].integer);
    const t1: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[1].integer);
    const trellis_bits = std.mem.bytesAsSlice(u16, data[t0..t1]);
    const E: usize = 4;
    const dim: usize = 128;
    const n: usize = 52;
    const tile_n = 8 * 8 * 48;
    const stacked = try alloc.alloc(u16, E * tile_n);
    for (0..E) |e| @memcpy(stacked[e * tile_n ..][0..tile_n], trellis_bits);
    var prng = std.Random.DefaultPrng.init(53);
    const rnd = prng.random();
    const xh = try alloc.alloc(u16, n * dim);
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
    var eids: [52]u32 = undefined;
    var i: usize = 0;
    while (i < 20) : (i += 1) eids[i] = 0;
    while (i < n) : (i += 1) eids[i] = 1;
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(n), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr_arr = mlx.mlx_array_new_data(stacked.ptr, &[_]c_int{ @intCast(E), 8, 8, 48 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr_arr);
    const eid_a = mlx.mlx_array_new_data(&eids, &[_]c_int{@intCast(n)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(eid_a);
    const got = try innerGemmSortedWin(s, x_arr, tr_arr, eid_a, 32);
    defer _ = mlx.mlx_array_free(got);
    var contig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(contig);
    try mlx.check(mlx.mlx_contiguous(&contig, got, false, s));
    try mlx.check(mlx.mlx_array_eval(contig));
    const src = mlx.mlx_array_data_float16(contig) orelse return error.F16Unreadable;
    try expectInnerGemmParity(alloc, s, src[0 .. n * dim], xh, &eids, stacked, dim, dim, exl3.Rate.fromK(3), .mul1);
}

test "exl3 sorted GEMM matches host MUL1 with the NAX arm forced off" {
    // On NAX hardware every other sorted-GEMM test takes the NAX arm (out_dim 128), so this bars the non-NAX body there.
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    if (!gemmNaxOn()) return error.SkipZigTest;
    _ = setenv("SUSHI_FORCE_GPU_FAMILY_FALLBACK", "1", 1);
    defer _ = unsetenv("SUSHI_FORCE_GPU_FAMILY_FALLBACK");
    try t.expect(!gemmNaxOn());
    const fixture = exl3.fixtures.k4;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const header_len = std.mem.readInt(u64, fixture[0..8], .little);
    const header = fixture[8 .. 8 + header_len];
    const data = fixture[8 + header_len ..];
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header, .{});
    defer parsed.deinit();
    const trellis_meta = parsed.value.object.get("trellis").?.object;
    const t0: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[0].integer);
    const t1: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[1].integer);
    const trellis_bits = std.mem.bytesAsSlice(u16, data[t0..t1]);
    const E: usize = 4;
    const dim: usize = 128;
    const tile_n = 8 * 8 * 64;
    const stacked = try alloc.alloc(u16, E * tile_n);
    for (0..E) |e| @memcpy(stacked[e * tile_n ..][0..tile_n], trellis_bits);
    const runs = [_]u32{ 7, 32, 5, 18, 40, 10 };
    var n: usize = 0;
    for (runs) |r| n += r;
    const eids = try alloc.alloc(u32, n);
    {
        var off: usize = 0;
        for (runs, 0..) |r, ei| {
            var j: usize = 0;
            while (j < r) : (j += 1) eids[off + j] = @intCast(ei % E);
            off += r;
        }
    }
    var prng = std.Random.DefaultPrng.init(0x5124);
    const rnd = prng.random();
    const xh = try alloc.alloc(u16, n * dim);
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(n), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr_arr = mlx.mlx_array_new_data(stacked.ptr, &[_]c_int{ @intCast(E), 8, 8, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr_arr);
    const eid_a = mlx.mlx_array_new_data(eids.ptr, &[_]c_int{@intCast(n)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(eid_a);
    for ([_]bool{ true, false }) |aligned| {
        const got = try innerGemmSortedWinAlign(s, x_arr, tr_arr, eid_a, 32, aligned);
        defer _ = mlx.mlx_array_free(got);
        var contig = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(contig);
        try mlx.check(mlx.mlx_contiguous(&contig, got, false, s));
        try mlx.check(mlx.mlx_array_eval(contig));
        const src = mlx.mlx_array_data_float16(contig) orelse return error.F16Unreadable;
        errdefer std.debug.print("[exl3-simd-arm] aligned={}\n", .{aligned});
        try expectInnerGemmParity(alloc, s, src[0 .. n * dim], xh, eids, stacked, dim, dim, exl3.Rate.fromK(4), .mul1);
    }
}

test "exl3 NAX K3 GEMM within 3x of K4 at C=2048 and 8192" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    if (!gemmNaxOn()) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const E: usize = 4;
    const in_dim: usize = 2560;
    const out_dim: usize = 640;
    const in_tiles = in_dim / 16;
    const out_tiles = out_dim / 16;
    var prng = std.Random.DefaultPrng.init(71);
    const rnd = prng.random();
    const contexts = [_]c_int{ 2048, 8192 };
    for (contexts) |C| {
        const n: usize = @intCast(C);
        const xh = try alloc.alloc(u16, n * in_dim);
        for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 0.1);
        const eids = try alloc.alloc(u32, n);
        const run = n / E;
        for (eids, 0..) |*v, i| v.* = @intCast(@min(i / run, E - 1));
        const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ C, @intCast(in_dim) }, 2, .float16);
        defer _ = mlx.mlx_array_free(x_arr);
        const eid_a = mlx.mlx_array_new_data(eids.ptr, &[_]c_int{C}, 1, .uint32);
        defer _ = mlx.mlx_array_free(eid_a);
        var k4_ns: u64 = 0;
        var k3_ns: u64 = 0;
        const packed4 = exl3.packedHalfwords(4);
        const packed3 = exl3.packedHalfwords(3);
        const tile4 = in_tiles * out_tiles * packed4;
        const tile3 = in_tiles * out_tiles * packed3;
        const stacked4 = try alloc.alloc(u16, E * tile4);
        const stacked3 = try alloc.alloc(u16, E * tile3);
        for (stacked4) |*v| v.* = @truncate(rnd.int(u32));
        for (stacked3) |*v| v.* = @truncate(rnd.int(u32));
        const tr4 = mlx.mlx_array_new_data(stacked4.ptr, &[_]c_int{ @intCast(E), @intCast(in_tiles), @intCast(out_tiles), @intCast(packed4) }, 4, .uint16);
        defer _ = mlx.mlx_array_free(tr4);
        const tr3 = mlx.mlx_array_new_data(stacked3.ptr, &[_]c_int{ @intCast(E), @intCast(in_tiles), @intCast(out_tiles), @intCast(packed3) }, 4, .uint16);
        defer _ = mlx.mlx_array_free(tr3);
        var warm_i: usize = 0;
        while (warm_i < 3) : (warm_i += 1) {
            inline for (.{ tr4, tr3 }) |tr| {
                const warm = try innerGemmSorted(s, x_arr, tr, eid_a);
                try mlx.check(mlx.mlx_array_eval(warm));
                _ = mlx.mlx_array_free(warm);
            }
        }
        var it: usize = 0;
        while (it < 8) : (it += 1) {
            var sw4 = io_util.Stopwatch.init(t.io);
            const b4 = try innerGemmSorted(s, x_arr, tr4, eid_a);
            try mlx.check(mlx.mlx_array_eval(b4));
            k4_ns += sw4.read();
            _ = mlx.mlx_array_free(b4);
            var sw3 = io_util.Stopwatch.init(t.io);
            const b3 = try innerGemmSorted(s, x_arr, tr3, eid_a);
            try mlx.check(mlx.mlx_array_eval(b3));
            k3_ns += sw3.read();
            _ = mlx.mlx_array_free(b3);
        }
        k4_ns /= 8;
        k3_ns /= 8;
        benchPrint("exl3 NAX one-proj C={d} H=2560 I=640: K4 {d} us K3 {d} us\n", .{
            C,
            k4_ns / 1000,
            k3_ns / 1000,
        });
        try t.expect(k4_ns > 0);
        benchPrint("[exl3-k3-timing] k3 {d} us k4 {d} us ratio {d:.3}\n", .{ k3_ns / 1000, k4_ns / 1000, @as(f64, @floatFromInt(k3_ns)) / @as(f64, @floatFromInt(@max(k4_ns, 1))) });
        try t.expect(k3_ns < k4_ns * 3);
    }
}

test "exl3 sorted GEMM 16-row windows match 4-row per row" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const fixture = exl3.fixtures.k4;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const header_len = std.mem.readInt(u64, fixture[0..8], .little);
    const header = fixture[8 .. 8 + header_len];
    const data = fixture[8 + header_len ..];
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header, .{});
    defer parsed.deinit();
    const trellis_meta = parsed.value.object.get("trellis").?.object;
    const t0: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[0].integer);
    const t1: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[1].integer);
    const trellis_bits = std.mem.bytesAsSlice(u16, data[t0..t1]);
    const E: usize = 4;
    const dim: usize = 128;
    const n: usize = 32;
    const tile_n = 8 * 8 * 64;
    const stacked = try alloc.alloc(u16, E * tile_n);
    for (0..E) |e| @memcpy(stacked[e * tile_n ..][0..tile_n], trellis_bits);
    var prng = std.Random.DefaultPrng.init(61);
    const rnd = prng.random();
    const xh = try alloc.alloc(u16, n * dim);
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
    var eids: [32]u32 = undefined;
    var i: usize = 0;
    while (i < 10) : (i += 1) eids[i] = 0;
    while (i < 13) : (i += 1) eids[i] = 2;
    while (i < 29) : (i += 1) eids[i] = 1;
    while (i < n) : (i += 1) eids[i] = 3;
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(n), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr_arr = mlx.mlx_array_new_data(stacked.ptr, &[_]c_int{ @intCast(E), 8, 8, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr_arr);
    const eid_a = mlx.mlx_array_new_data(&eids, &[_]c_int{@intCast(n)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(eid_a);
    const got4 = try innerGemmSortedWin(s, x_arr, tr_arr, eid_a, 4);
    defer _ = mlx.mlx_array_free(got4);
    const got16 = try innerGemmSortedWin(s, x_arr, tr_arr, eid_a, 16);
    defer _ = mlx.mlx_array_free(got16);
    var c4 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c4);
    var c16 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c16);
    try mlx.check(mlx.mlx_contiguous(&c4, got4, false, s));
    try mlx.check(mlx.mlx_contiguous(&c16, got16, false, s));
    try mlx.check(mlx.mlx_array_eval(c4));
    try mlx.check(mlx.mlx_array_eval(c16));
    const a4 = mlx.mlx_array_data_float16(c4) orelse return error.F16Unreadable;
    const a16 = mlx.mlx_array_data_float16(c16) orelse return error.F16Unreadable;
    for (0..n * dim) |j| {
        const b4: u16 = @bitCast(a4[j]);
        const b16: u16 = @bitCast(a16[j]);
        try t.expectEqual(b4, b16);
    }
}

test "exl3 run-aligned windows match stride per row" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const fixture = exl3.fixtures.k4;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const header_len = std.mem.readInt(u64, fixture[0..8], .little);
    const header = fixture[8 .. 8 + header_len];
    const data = fixture[8 + header_len ..];
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header, .{});
    defer parsed.deinit();
    const trellis_meta = parsed.value.object.get("trellis").?.object;
    const t0: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[0].integer);
    const t1: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[1].integer);
    const trellis_bits = std.mem.bytesAsSlice(u16, data[t0..t1]);
    const E: usize = 4;
    const dim: usize = 128;
    const n: usize = 40;
    const tile_n = 8 * 8 * 64;
    const stacked = try alloc.alloc(u16, E * tile_n);
    for (0..E) |e| @memcpy(stacked[e * tile_n ..][0..tile_n], trellis_bits);
    var prng = std.Random.DefaultPrng.init(101);
    const rnd = prng.random();
    const xh = try alloc.alloc(u16, n * dim);
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
    var eids: [40]u32 = undefined;
    var i: usize = 0;
    while (i < 7) : (i += 1) eids[i] = 0;
    while (i < 23) : (i += 1) eids[i] = 1;
    while (i < 27) : (i += 1) eids[i] = 2;
    while (i < n) : (i += 1) eids[i] = 3;
    const st = windowStats(eids[0..], 16, false);
    const al = windowStats(eids[0..], 16, true);
    try t.expect(st.mixed > 0);
    try t.expectEqual(@as(u32, 0), al.mixed);
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(n), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr_arr = mlx.mlx_array_new_data(stacked.ptr, &[_]c_int{ @intCast(E), 8, 8, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr_arr);
    const eid_a = mlx.mlx_array_new_data(&eids, &[_]c_int{@intCast(n)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(eid_a);
    const wins = [_]c_int{ 16, 32 };
    for (wins) |w| {
        const stride = try innerGemmSortedWinAlign(s, x_arr, tr_arr, eid_a, w, false);
        defer _ = mlx.mlx_array_free(stride);
        const aligned = try innerGemmSortedWinAlign(s, x_arr, tr_arr, eid_a, w, true);
        defer _ = mlx.mlx_array_free(aligned);
        var cs = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cs);
        var ca = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ca);
        try mlx.check(mlx.mlx_contiguous(&cs, stride, false, s));
        try mlx.check(mlx.mlx_contiguous(&ca, aligned, false, s));
        try mlx.check(mlx.mlx_array_eval(cs));
        try mlx.check(mlx.mlx_array_eval(ca));
        const as = mlx.mlx_array_data_float16(cs) orelse return error.F16Unreadable;
        const aa = mlx.mlx_array_data_float16(ca) orelse return error.F16Unreadable;
        for (0..n * dim) |j| {
            const bs: u16 = @bitCast(as[j]);
            const ba: u16 = @bitCast(aa[j]);
            try t.expectEqual(bs, ba);
        }
    }
}

test "exl3 aligned GEMM reuses config across nwin" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const fixture = exl3.fixtures.k4;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const header_len = std.mem.readInt(u64, fixture[0..8], .little);
    const header = fixture[8 .. 8 + header_len];
    const data = fixture[8 + header_len ..];
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header, .{});
    defer parsed.deinit();
    const trellis_meta = parsed.value.object.get("trellis").?.object;
    const t0: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[0].integer);
    const t1: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[1].integer);
    const trellis_bits = std.mem.bytesAsSlice(u16, data[t0..t1]);
    const E: usize = 4;
    const dim: usize = 128;
    const n: usize = 40;
    const tile_n = 8 * 8 * 64;
    const stacked = try alloc.alloc(u16, E * tile_n);
    for (0..E) |e| @memcpy(stacked[e * tile_n ..][0..tile_n], trellis_bits);
    var prng = std.Random.DefaultPrng.init(113);
    const rnd = prng.random();
    const xh = try alloc.alloc(u16, n * dim);
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
    var long_runs: [40]u32 = undefined;
    for (&long_runs) |*v| v.* = 1;
    var short_runs: [40]u32 = undefined;
    for (&short_runs, 0..) |*v, i| v.* = @intCast(i % 4);
    try t.expect(windowStats(long_runs[0..], 32, true).nwin < windowStats(short_runs[0..], 32, true).nwin);
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(n), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr_arr = mlx.mlx_array_new_data(stacked.ptr, &[_]c_int{ @intCast(E), 8, 8, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr_arr);
    const eid_long = mlx.mlx_array_new_data(&long_runs, &[_]c_int{@intCast(n)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(eid_long);
    const eid_short = mlx.mlx_array_new_data(&short_runs, &[_]c_int{@intCast(n)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(eid_short);
    const a_long = try innerGemmSortedWinAlign(s, x_arr, tr_arr, eid_long, 32, true);
    defer _ = mlx.mlx_array_free(a_long);
    try mlx.check(mlx.mlx_array_eval(a_long));
    const stride_short = try innerGemmSortedWinAlign(s, x_arr, tr_arr, eid_short, 32, false);
    defer _ = mlx.mlx_array_free(stride_short);
    const aligned_short = try innerGemmSortedWinAlign(s, x_arr, tr_arr, eid_short, 32, true);
    defer _ = mlx.mlx_array_free(aligned_short);
    var cs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cs);
    var ca = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ca);
    try mlx.check(mlx.mlx_contiguous(&cs, stride_short, false, s));
    try mlx.check(mlx.mlx_contiguous(&ca, aligned_short, false, s));
    try mlx.check(mlx.mlx_array_eval(cs));
    try mlx.check(mlx.mlx_array_eval(ca));
    const as = mlx.mlx_array_data_float16(cs) orelse return error.F16Unreadable;
    const aa = mlx.mlx_array_data_float16(ca) orelse return error.F16Unreadable;
    for (0..n * dim) |j| {
        const bs: u16 = @bitCast(as[j]);
        const ba: u16 = @bitCast(aa[j]);
        try t.expectEqual(bs, ba);
    }
}

test "exl3 sorted GEMM 32-row windows match 16-row per row" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const fixture = exl3.fixtures.k4;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const header_len = std.mem.readInt(u64, fixture[0..8], .little);
    const header = fixture[8 .. 8 + header_len];
    const data = fixture[8 + header_len ..];
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header, .{});
    defer parsed.deinit();
    const trellis_meta = parsed.value.object.get("trellis").?.object;
    const t0: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[0].integer);
    const t1: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[1].integer);
    const trellis_bits = std.mem.bytesAsSlice(u16, data[t0..t1]);
    const E: usize = 4;
    const dim: usize = 128;
    const n: usize = 40;
    const tile_n = 8 * 8 * 64;
    const stacked = try alloc.alloc(u16, E * tile_n);
    for (0..E) |e| @memcpy(stacked[e * tile_n ..][0..tile_n], trellis_bits);
    var prng = std.Random.DefaultPrng.init(97);
    const rnd = prng.random();
    const xh = try alloc.alloc(u16, n * dim);
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
    var eids: [40]u32 = undefined;
    var i: usize = 0;
    while (i < 24) : (i += 1) eids[i] = 1;
    while (i < 30) : (i += 1) eids[i] = 0;
    while (i < n) : (i += 1) eids[i] = 2;
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(n), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr_arr = mlx.mlx_array_new_data(stacked.ptr, &[_]c_int{ @intCast(E), 8, 8, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr_arr);
    const eid_a = mlx.mlx_array_new_data(&eids, &[_]c_int{@intCast(n)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(eid_a);
    const got16 = try innerGemmSortedWin(s, x_arr, tr_arr, eid_a, 16);
    defer _ = mlx.mlx_array_free(got16);
    const got32 = try innerGemmSortedWin(s, x_arr, tr_arr, eid_a, 32);
    defer _ = mlx.mlx_array_free(got32);
    var c16 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c16);
    var c32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c32);
    try mlx.check(mlx.mlx_contiguous(&c16, got16, false, s));
    try mlx.check(mlx.mlx_contiguous(&c32, got32, false, s));
    try mlx.check(mlx.mlx_array_eval(c16));
    try mlx.check(mlx.mlx_array_eval(c32));
    const a16 = mlx.mlx_array_data_float16(c16) orelse return error.F16Unreadable;
    const a32 = mlx.mlx_array_data_float16(c32) orelse return error.F16Unreadable;
    for (0..n * dim) |j| {
        const b16: u16 = @bitCast(a16[j]);
        const b32: u16 = @bitCast(a32[j]);
        try t.expectEqual(b16, b32);
    }
}

test "exl3 window 16 vs 32 production C=2048" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const E: c_int = 512;
    const R: c_int = 2048;
    const topk: c_int = 10;
    const H: c_int = 2560;
    const I: c_int = 640;
    const tr_n: usize = @intCast(E * (H / 16) * (I / 16) * 64);
    const tr_g = try alloc.alloc(u16, tr_n);
    const xh = try alloc.alloc(u16, @intCast(R * H));
    const slots_h = try alloc.alloc(u32, @intCast(R * topk));
    var prng = std.Random.DefaultPrng.init(99);
    const rnd = prng.random();
    for (tr_g) |*v| v.* = @truncate(rnd.int(u32));
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 0.1);
    for (slots_h) |*v| v.* = rnd.uintLessThan(u32, @intCast(E));
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ R, H }, 2, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const slots = mlx.mlx_array_new_data(slots_h.ptr, &[_]c_int{R * topk}, 1, .uint32);
    defer _ = mlx.mlx_array_free(slots);
    const trg = mlx.mlx_array_new_data(tr_g.ptr, &[_]c_int{ E, H / 16, I / 16, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(trg);
    var order = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(order);
    try mlx.check(mlx.mlx_argsort_axis(&order, slots, 0, s));
    var sorted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sorted);
    try mlx.check(mlx.mlx_take_axis(&sorted, slots, order, 0, s));
    const xr = try repeatRows(s, x_arr, R, topk);
    defer _ = mlx.mlx_array_free(xr);
    var sc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc);
    try mlx.check(mlx.mlx_contiguous(&sc, sorted, false, s));
    try mlx.check(mlx.mlx_array_eval(sc));
    const nslots: usize = @intCast(R * topk);
    const ids = try alloc.alloc(u32, nslots);
    const sp = mlx.mlx_array_data_uint32(sc) orelse return error.F16Unreadable;
    @memcpy(ids, sp[0..nslots]);
    const io = std.Io.Threaded.global_single_threaded.io();
    const arms = [_]struct { win: c_int, aligned: bool, name: []const u8 }{
        .{ .win = 16, .aligned = false, .name = "stride-16" },
        .{ .win = 16, .aligned = true, .name = "aligned-16" },
        .{ .win = 32, .aligned = true, .name = "aligned-32" },
        .{ .win = 32, .aligned = false, .name = "stride-32" },
    };
    for (arms) |arm| {
        const st = windowStats(ids, @intCast(arm.win), arm.aligned);
        const warm = try innerGemmSortedWinAlign(s, xr, trg, sorted, arm.win, arm.aligned);
        try mlx.check(mlx.mlx_array_eval(warm));
        _ = mlx.mlx_array_free(warm);
        var sw = io_util.Stopwatch.init(io);
        const got = try innerGemmSortedWinAlign(s, xr, trg, sorted, arm.win, arm.aligned);
        try mlx.check(mlx.mlx_array_eval(got));
        const ns = sw.read();
        _ = mlx.mlx_array_free(got);
        benchPrint("C=2048 {s} {d} us nwin={d} mixed={d} decodes={d}\n", .{
            arm.name, ns / 1000, st.nwin, st.mixed, st.decodes,
        });
        try t.expect(ns > 0);
    }
    const R8: c_int = 8192;
    const xh8 = try alloc.alloc(u16, @intCast(R8 * H));
    const slots8 = try alloc.alloc(u32, @intCast(R8 * topk));
    for (xh8) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 0.1);
    for (slots8) |*v| v.* = rnd.uintLessThan(u32, @intCast(E));
    const x8 = mlx.mlx_array_new_data(xh8.ptr, &[_]c_int{ R8, H }, 2, .float16);
    defer _ = mlx.mlx_array_free(x8);
    const sl8 = mlx.mlx_array_new_data(slots8.ptr, &[_]c_int{R8 * topk}, 1, .uint32);
    defer _ = mlx.mlx_array_free(sl8);
    var order8 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(order8);
    try mlx.check(mlx.mlx_argsort_axis(&order8, sl8, 0, s));
    var sorted8 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sorted8);
    try mlx.check(mlx.mlx_take_axis(&sorted8, sl8, order8, 0, s));
    const xr8 = try repeatRows(s, x8, R8, topk);
    defer _ = mlx.mlx_array_free(xr8);
    var sc8 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sc8);
    try mlx.check(mlx.mlx_contiguous(&sc8, sorted8, false, s));
    try mlx.check(mlx.mlx_array_eval(sc8));
    const n8: usize = @intCast(R8 * topk);
    const ids8 = try alloc.alloc(u32, n8);
    const sp8 = mlx.mlx_array_data_uint32(sc8) orelse return error.F16Unreadable;
    @memcpy(ids8, sp8[0..n8]);
    for (arms) |arm| {
        const st = windowStats(ids8, @intCast(arm.win), arm.aligned);
        const warm = try innerGemmSortedWinAlign(s, xr8, trg, sorted8, arm.win, arm.aligned);
        try mlx.check(mlx.mlx_array_eval(warm));
        _ = mlx.mlx_array_free(warm);
        var sw = io_util.Stopwatch.init(io);
        const got = try innerGemmSortedWinAlign(s, xr8, trg, sorted8, arm.win, arm.aligned);
        try mlx.check(mlx.mlx_array_eval(got));
        const ns = sw.read();
        _ = mlx.mlx_array_free(got);
        benchPrint("C=8192 {s} {d} us nwin={d} mixed={d} decodes={d}\n", .{
            arm.name, ns / 1000, st.nwin, st.mixed, st.decodes,
        });
        try t.expect(ns > 0);
    }
}

test "exl3 moePrefill matches staged sorted chain" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const fixture = exl3.fixtures.k4;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const header_len = std.mem.readInt(u64, fixture[0..8], .little);
    const header = fixture[8 .. 8 + header_len];
    const data = fixture[8 + header_len ..];
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header, .{});
    defer parsed.deinit();
    const trellis_meta = parsed.value.object.get("trellis").?.object;
    const t0: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[0].integer);
    const t1: usize = @intCast(trellis_meta.get("data_offsets").?.array.items[1].integer);
    const suh_meta = parsed.value.object.get("suh").?.object;
    const s0: usize = @intCast(suh_meta.get("data_offsets").?.array.items[0].integer);
    const s1: usize = @intCast(suh_meta.get("data_offsets").?.array.items[1].integer);
    const svh_meta = parsed.value.object.get("svh").?.object;
    const v0: usize = @intCast(svh_meta.get("data_offsets").?.array.items[0].integer);
    const v1: usize = @intCast(svh_meta.get("data_offsets").?.array.items[1].integer);
    const trellis_bits = std.mem.bytesAsSlice(u16, data[t0..t1]);
    const suh_bits = std.mem.bytesAsSlice(u16, data[s0..s1]);
    const svh_bits = std.mem.bytesAsSlice(u16, data[v0..v1]);
    const E: c_int = 4;
    const R: c_int = 8;
    const topk: c_int = 2;
    const dim: c_int = 128;
    const stacked_t = try alloc.alloc(u16, @intCast(E * 8 * 8 * 64));
    const stacked_suh = try alloc.alloc(u16, @intCast(E * dim));
    const stacked_svh = try alloc.alloc(u16, @intCast(E * dim));
    var e_i: c_int = 0;
    while (e_i < E) : (e_i += 1) {
        const tb: usize = @intCast(e_i);
        @memcpy(stacked_t[tb * trellis_bits.len ..][0..trellis_bits.len], trellis_bits);
        @memcpy(stacked_suh[tb * suh_bits.len ..][0..suh_bits.len], suh_bits);
        @memcpy(stacked_svh[tb * svh_bits.len ..][0..svh_bits.len], svh_bits);
    }
    var prng = std.Random.DefaultPrng.init(71);
    const rnd = prng.random();
    const xh = try alloc.alloc(u16, @intCast(R * dim));
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
    const slots_h = try alloc.alloc(u32, @intCast(R * topk));
    for (slots_h) |*v| v.* = rnd.uintLessThan(u32, @intCast(E));
    const scores_h = try alloc.alloc(f32, @intCast(R * topk));
    for (scores_h) |*v| v.* = 0.25 + rnd.float(f32) * 0.5;
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ R, dim }, 2, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const slots = mlx.mlx_array_new_data(slots_h.ptr, &[_]c_int{R * topk}, 1, .uint32);
    defer _ = mlx.mlx_array_free(slots);
    const scores = mlx.mlx_array_new_data(scores_h.ptr, &[_]c_int{R * topk}, 1, .float32);
    defer _ = mlx.mlx_array_free(scores);
    const tr = mlx.mlx_array_new_data(stacked_t.ptr, &[_]c_int{ E, 8, 8, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr);
    const suh = mlx.mlx_array_new_data(stacked_suh.ptr, &[_]c_int{ E, dim }, 2, .float16);
    defer _ = mlx.mlx_array_free(suh);
    const svh = mlx.mlx_array_new_data(stacked_svh.ptr, &[_]c_int{ E, dim }, 2, .float16);
    defer _ = mlx.mlx_array_free(svh);
    const got = try moePrefill(s, x_arr, tr, suh, svh, tr, suh, svh, tr, suh, svh, slots, scores, topk);
    defer _ = mlx.mlx_array_free(got);
    var order = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(order);
    try mlx.check(mlx.mlx_argsort_axis(&order, slots, 0, s));
    var order_u = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(order_u);
    try mlx.check(mlx.mlx_astype(&order_u, order, .uint32, s));
    var sorted_slots = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sorted_slots);
    try mlx.check(mlx.mlx_take_axis(&sorted_slots, slots, order, 0, s));
    const g_prep = try prepareFromTokens(s, x_arr, suh, sorted_slots, order_u, dim, R * topk, topk);
    defer _ = mlx.mlx_array_free(g_prep);
    const u_prep = try prepareFromTokens(s, x_arr, suh, sorted_slots, order_u, dim, R * topk, topk);
    defer _ = mlx.mlx_array_free(u_prep);
    const g_inner = try innerGemmSorted(s, g_prep, tr, sorted_slots);
    defer _ = mlx.mlx_array_free(g_inner);
    const u_inner = try innerGemmSorted(s, u_prep, tr, sorted_slots);
    defer _ = mlx.mlx_array_free(u_inner);
    const g = try finishIndexed(s, g_inner, svh, sorted_slots);
    defer _ = mlx.mlx_array_free(g);
    const u = try finishIndexed(s, u_inner, svh, sorted_slots);
    defer _ = mlx.mlx_array_free(u);
    // The arm holds the SwiGLU product wide, so the staged chain must too:
    // an f16 `silu(g) * u` is the store this bar exists to keep out.
    var g32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(g32);
    try mlx.check(mlx.mlx_astype(&g32, g, .float32, s));
    var u32a = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(u32a);
    try mlx.check(mlx.mlx_astype(&u32a, u, .float32, s));
    var sig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sig);
    try mlx.check(mlx.mlx_sigmoid(&sig, g32, s));
    var silu = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(silu);
    try mlx.check(mlx.mlx_multiply(&silu, g32, sig, s));
    var h32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(h32);
    try mlx.check(mlx.mlx_multiply(&h32, silu, u32a, s));
    var h = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(h);
    try mlx.check(mlx.mlx_astype(&h, h32, .float16, s));
    const d_sorted = try projectSortedWithRuns(s, h, tr, suh, svh, sorted_slots);
    defer _ = mlx.mlx_array_free(d_sorted);
    var inv = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(inv);
    try mlx.check(mlx.mlx_argsort_axis(&inv, order, 0, s));
    var inv_u = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(inv_u);
    try mlx.check(mlx.mlx_astype(&inv_u, inv, .uint32, s));
    const ref = try tokenReduce(s, d_sorted, inv_u, scores, dim, R, topk);
    defer _ = mlx.mlx_array_free(ref);
    var cg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cg);
    var cr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cr);
    try mlx.check(mlx.mlx_contiguous(&cg, got, false, s));
    try mlx.check(mlx.mlx_contiguous(&cr, ref, false, s));
    try mlx.check(mlx.mlx_array_eval(cg));
    try mlx.check(mlx.mlx_array_eval(cr));
    const ag = mlx.mlx_array_data_float16(cg) orelse return error.F16Unreadable;
    const ar = mlx.mlx_array_data_float16(cr) orelse return error.F16Unreadable;
    const n: usize = @intCast(R * dim);
    // The arm carries the whole SwiGLU in registers and rounds once; the staged
    // chain lands every stage in f16. An ULP bar would measure that, not the
    // chain, so the bar is an envelope.
    try expectRelRms(ag[0..n], ar[0..n], 0.005);
}

test "exl3 512-row E=512 topk=10 layer within 2x affine" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    if (mlx.maxRecommendedWorkingSet() < 16 << 30) return error.SkipZigTest; // production-shape banks: GBs a CI runner lacks
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const E: c_int = 512;
    const R: c_int = 512;
    const topk: c_int = 10;
    const H: c_int = 2560;
    const I: c_int = 640;
    const tr_g_n: usize = @intCast(E * (H / 16) * (I / 16) * 64);
    const tr_d_n: usize = @intCast(E * (I / 16) * (H / 16) * 64);
    const tr_g = try alloc.alloc(u16, tr_g_n);
    const tr_d = try alloc.alloc(u16, tr_d_n);
    const suh_g = try alloc.alloc(u16, @intCast(E * H));
    const svh_g = try alloc.alloc(u16, @intCast(E * I));
    const suh_d = try alloc.alloc(u16, @intCast(E * I));
    const svh_d = try alloc.alloc(u16, @intCast(E * H));
    const xh = try alloc.alloc(u16, @intCast(R * H));
    const slots_h = try alloc.alloc(u32, @intCast(R * topk));
    const scores_h = try alloc.alloc(f32, @intCast(R * topk));
    var prng = std.Random.DefaultPrng.init(53);
    const rnd = prng.random();
    for (tr_g) |*v| v.* = @truncate(rnd.int(u32));
    for (tr_d) |*v| v.* = @truncate(rnd.int(u32));
    for (suh_g) |*v| v.* = exl3.f32ToF16Bits(1.0);
    for (svh_g) |*v| v.* = exl3.f32ToF16Bits(1.0);
    for (suh_d) |*v| v.* = exl3.f32ToF16Bits(1.0);
    for (svh_d) |*v| v.* = exl3.f32ToF16Bits(1.0);
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 0.1);
    for (slots_h) |*v| v.* = rnd.uintLessThan(u32, @intCast(E));
    for (scores_h) |*v| v.* = 0.5;
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ R, H }, 2, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const slots = mlx.mlx_array_new_data(slots_h.ptr, &[_]c_int{R * topk}, 1, .uint32);
    defer _ = mlx.mlx_array_free(slots);
    const scores = mlx.mlx_array_new_data(scores_h.ptr, &[_]c_int{R * topk}, 1, .float32);
    defer _ = mlx.mlx_array_free(scores);
    const trg = mlx.mlx_array_new_data(tr_g.ptr, &[_]c_int{ E, H / 16, I / 16, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(trg);
    const trd = mlx.mlx_array_new_data(tr_d.ptr, &[_]c_int{ E, I / 16, H / 16, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(trd);
    const sugh = mlx.mlx_array_new_data(suh_g.ptr, &[_]c_int{ E, H }, 2, .float16);
    defer _ = mlx.mlx_array_free(sugh);
    const svgi = mlx.mlx_array_new_data(svh_g.ptr, &[_]c_int{ E, I }, 2, .float16);
    defer _ = mlx.mlx_array_free(svgi);
    const sudi = mlx.mlx_array_new_data(suh_d.ptr, &[_]c_int{ E, I }, 2, .float16);
    defer _ = mlx.mlx_array_free(sudi);
    const svdh = mlx.mlx_array_new_data(svh_d.ptr, &[_]c_int{ E, H }, 2, .float16);
    defer _ = mlx.mlx_array_free(svdh);
    const warm = try moePrefill(s, x_arr, trg, sugh, svgi, trg, sugh, svgi, trd, sudi, svdh, slots, scores, topk);
    try mlx.check(mlx.mlx_array_eval(warm));
    _ = mlx.mlx_array_free(warm);
    var t_g = io_util.Stopwatch.init(t.io);
    var it: usize = 0;
    while (it < 3) : (it += 1) {
        const out = try moePrefill(s, x_arr, trg, sugh, svgi, trg, sugh, svgi, trd, sudi, svdh, slots, scores, topk);
        try mlx.check(mlx.mlx_array_eval(out));
        _ = mlx.mlx_array_free(out);
    }
    const gemm_ns = t_g.read() / 3;
    var dense_g = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dense_g);
    try mlx.check(mlx.mlx_random_normal(&dense_g, &[_]c_int{ E, I, H }, 3, .float16, 0, 1, .{ .ctx = null }, s));
    var w_cg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_cg);
    try mlx.check(mlx.mlx_contiguous(&w_cg, dense_g, false, s));
    var triple_g = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(triple_g);
    try mlx.check(mlx.mlx_quantize(&triple_g, w_cg, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{ .ctx = null }, s));
    var wqg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wqg);
    var wscg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wscg);
    var wbig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wbig);
    try mlx.check(mlx.mlx_vector_array_get(&wqg, triple_g, 0));
    try mlx.check(mlx.mlx_vector_array_get(&wscg, triple_g, 1));
    try mlx.check(mlx.mlx_vector_array_get(&wbig, triple_g, 2));
    var dense_d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dense_d);
    try mlx.check(mlx.mlx_random_normal(&dense_d, &[_]c_int{ E, H, I }, 3, .float16, 0, 1, .{ .ctx = null }, s));
    var w_cd = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(w_cd);
    try mlx.check(mlx.mlx_contiguous(&w_cd, dense_d, false, s));
    var triple_d = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(triple_d);
    try mlx.check(mlx.mlx_quantize(&triple_d, w_cd, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{ .ctx = null }, s));
    var wqd = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wqd);
    var wscd = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wscd);
    var wbid = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wbid);
    try mlx.check(mlx.mlx_vector_array_get(&wqd, triple_d, 0));
    try mlx.check(mlx.mlx_vector_array_get(&wscd, triple_d, 1));
    try mlx.check(mlx.mlx_vector_array_get(&wbid, triple_d, 2));
    const xr = try repeatRows(s, x_arr, R, topk);
    defer _ = mlx.mlx_array_free(xr);
    var xrep = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xrep);
    try mlx.check(mlx.mlx_reshape(&xrep, xr, &[_]c_int{ R * topk, 1, H }, 3, s));
    var xdi = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xdi);
    try mlx.check(mlx.mlx_random_normal(&xdi, &[_]c_int{ R * topk, 1, I }, 3, .float16, 0, 1, .{ .ctx = null }, s));
    var slots_i = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(slots_i);
    try mlx.check(mlx.mlx_astype(&slots_i, slots, .int32, s));
    var order = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(order);
    try mlx.check(mlx.mlx_argsort_axis(&order, slots_i, 0, s));
    var sorted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sorted);
    try mlx.check(mlx.mlx_take_axis(&sorted, slots_i, order, 0, s));
    const no_idx = mlx.mlx_array{ .ctx = null };
    var q0 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(q0);
    try mlx.check(mlx.mlx_gather_qmm(&q0, xrep, wqg, wscg, wbig, no_idx, sorted, true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", true, s));
    try mlx.check(mlx.mlx_array_eval(q0));
    var t_q = io_util.Stopwatch.init(t.io);
    it = 0;
    while (it < 3) : (it += 1) {
        var qg = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_gather_qmm(&qg, xrep, wqg, wscg, wbig, no_idx, sorted, true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", true, s));
        try mlx.check(mlx.mlx_array_eval(qg));
        _ = mlx.mlx_array_free(qg);
        var qu = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_gather_qmm(&qu, xrep, wqg, wscg, wbig, no_idx, sorted, true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", true, s));
        try mlx.check(mlx.mlx_array_eval(qu));
        _ = mlx.mlx_array_free(qu);
        var qd = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_gather_qmm(&qd, xdi, wqd, wscd, wbid, no_idx, sorted, true, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", true, s));
        try mlx.check(mlx.mlx_array_eval(qd));
        _ = mlx.mlx_array_free(qd);
    }
    const affine_ns = t_q.read() / 3;
    const ratio_x100: u64 = if (affine_ns == 0) 0 else (gemm_ns * 100) / affine_ns;
    benchPrint("exl3 512-row E=512 H=2560 I=640 topk=10: layer {d} us  affine-3x-gather_qmm {d} us  ratio {d}/100\n", .{
        gemm_ns / 1000,
        affine_ns / 1000,
        ratio_x100,
    });
    try t.expect(ratio_x100 <= 200);
}

test "exl3 fused decode chain matches the indexed chain across the top-k range" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const fixture = exl3.fixtures.k4;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const header_len = std.mem.readInt(u64, fixture[0..8], .little);
    const header = fixture[8 .. 8 + header_len];
    const data = fixture[8 + header_len ..];
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header, .{});
    defer parsed.deinit();
    const tm = parsed.value.object.get("trellis").?.object;
    const sm = parsed.value.object.get("suh").?.object;
    const vm = parsed.value.object.get("svh").?.object;
    const t0: usize = @intCast(tm.get("data_offsets").?.array.items[0].integer);
    const t1: usize = @intCast(tm.get("data_offsets").?.array.items[1].integer);
    const s0: usize = @intCast(sm.get("data_offsets").?.array.items[0].integer);
    const s1: usize = @intCast(sm.get("data_offsets").?.array.items[1].integer);
    const v0: usize = @intCast(vm.get("data_offsets").?.array.items[0].integer);
    const v1: usize = @intCast(vm.get("data_offsets").?.array.items[1].integer);
    const trellis_bits = std.mem.bytesAsSlice(u16, data[t0..t1]);
    const suh_bits = std.mem.bytesAsSlice(u16, data[s0..s1]);
    const svh_bits = std.mem.bytesAsSlice(u16, data[v0..v1]);
    const E: usize = 7;
    const dim: usize = 128;
    const tile_n = 8 * 8 * 64;
    const st = try alloc.alloc(u16, E * tile_n);
    const su = try alloc.alloc(u16, E * dim);
    const sv = try alloc.alloc(u16, E * dim);
    for (0..E) |e| {
        @memcpy(st[e * tile_n ..][0..tile_n], trellis_bits);
        @memcpy(su[e * dim ..][0..dim], suh_bits);
        @memcpy(sv[e * dim ..][0..dim], svh_bits);
    }
    const tr = mlx.mlx_array_new_data(st.ptr, &[_]c_int{ @intCast(E), 8, 8, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr);
    const suh = mlx.mlx_array_new_data(su.ptr, &[_]c_int{ @intCast(E), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(suh);
    const svh = mlx.mlx_array_new_data(sv.ptr, &[_]c_int{ @intCast(E), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(svh);
    var prng = std.Random.DefaultPrng.init(131);
    const rnd = prng.random();
    // The reduce bank is per (row, k) slot: a top-k past its width silently read
    // another slot's partial. Bar is 10x the f16 floor these shapes agree at.
    for ([_]usize{ 8, 16, 17, 20, 32 }) |topk| {
        const xh = try alloc.alloc(u16, dim);
        for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
        const sl = try alloc.alloc(u32, topk);
        const sc = try alloc.alloc(f32, topk);
        for (sl) |*v| v.* = rnd.uintLessThan(u32, @intCast(E));
        for (sc) |*v| v.* = rnd.float(f32);
        const xa = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{@intCast(dim)}, 1, .float16);
        defer _ = mlx.mlx_array_free(xa);
        const sa = mlx.mlx_array_new_data(sl.ptr, &[_]c_int{@intCast(topk)}, 1, .uint32);
        defer _ = mlx.mlx_array_free(sa);
        const ca = mlx.mlx_array_new_data(sc.ptr, &[_]c_int{@intCast(topk)}, 1, .float32);
        defer _ = mlx.mlx_array_free(ca);
        const fused = try moeSwigluFused(s, xa, tr, suh, svh, tr, suh, svh, tr, suh, svh, sa, ca, .float16);
        defer _ = mlx.mlx_array_free(fused);
        const ref = try moeSwigluIndexed(s, xa, tr, suh, svh, tr, suh, svh, tr, suh, svh, sa, ca);
        defer _ = mlx.mlx_array_free(ref);
        var cf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cf);
        var cr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cr);
        try mlx.check(mlx.mlx_contiguous(&cf, fused, false, s));
        try mlx.check(mlx.mlx_contiguous(&cr, ref, false, s));
        try mlx.check(mlx.mlx_array_eval(cf));
        try mlx.check(mlx.mlx_array_eval(cr));
        const af = mlx.mlx_array_data_float16(cf) orelse return error.F16Unreadable;
        const ar = mlx.mlx_array_data_float16(cr) orelse return error.F16Unreadable;
        var ss: f64 = 0;
        var refsq: f64 = 0;
        for (0..dim) |j| {
            const a = exl3.f16BitsToF32(@bitCast(af[j]));
            const b = exl3.f16BitsToF32(@bitCast(ar[j]));
            ss += @as(f64, a - b) * @as(f64, a - b);
            refsq += @as(f64, b) * @as(f64, b);
        }
        const rel = @sqrt(ss / @max(refsq, 1e-20));
        if (!(rel < 0.01)) {
            std.debug.print("exl3 topk={d} rel_rms={d:.6}\n", .{ topk, rel });
            return error.TestExpectedEqual;
        }
    }
}

test "exl3 prefill arm matches the fused decode arm across row counts and top-k" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const fixture = exl3.fixtures.k4;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const header_len = std.mem.readInt(u64, fixture[0..8], .little);
    const header = fixture[8 .. 8 + header_len];
    const data = fixture[8 + header_len ..];
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header, .{});
    defer parsed.deinit();
    const tm = parsed.value.object.get("trellis").?.object;
    const sm = parsed.value.object.get("suh").?.object;
    const vm = parsed.value.object.get("svh").?.object;
    const t0: usize = @intCast(tm.get("data_offsets").?.array.items[0].integer);
    const t1: usize = @intCast(tm.get("data_offsets").?.array.items[1].integer);
    const s0: usize = @intCast(sm.get("data_offsets").?.array.items[0].integer);
    const s1: usize = @intCast(sm.get("data_offsets").?.array.items[1].integer);
    const v0: usize = @intCast(vm.get("data_offsets").?.array.items[0].integer);
    const v1: usize = @intCast(vm.get("data_offsets").?.array.items[1].integer);
    const trellis_bits = std.mem.bytesAsSlice(u16, data[t0..t1]);
    const suh_bits = std.mem.bytesAsSlice(u16, data[s0..s1]);
    const svh_bits = std.mem.bytesAsSlice(u16, data[v0..v1]);
    const E: usize = 7;
    const dim: usize = 128;
    const tile_n = 8 * 8 * 64;
    const st = try alloc.alloc(u16, E * tile_n);
    const su = try alloc.alloc(u16, E * dim);
    const sv = try alloc.alloc(u16, E * dim);
    for (0..E) |e| {
        @memcpy(st[e * tile_n ..][0..tile_n], trellis_bits);
        @memcpy(su[e * dim ..][0..dim], suh_bits);
        @memcpy(sv[e * dim ..][0..dim], svh_bits);
    }
    const tr = mlx.mlx_array_new_data(st.ptr, &[_]c_int{ @intCast(E), 8, 8, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr);
    const suh = mlx.mlx_array_new_data(su.ptr, &[_]c_int{ @intCast(E), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(suh);
    const svh = mlx.mlx_array_new_data(sv.ptr, &[_]c_int{ @intCast(E), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(svh);
    var prng = std.Random.DefaultPrng.init(179);
    const rnd = prng.random();
    // Row counts either side of the 16-row window, a tail that does not fill a
    // window, and a run that spans one: the two arms must answer the same rows.
    for ([_][2]usize{ .{ 1, 1 }, .{ 3, 2 }, .{ 5, 7 }, .{ 16, 10 }, .{ 17, 3 }, .{ 31, 5 }, .{ 33, 1 }, .{ 64, 6 } }) |c| {
        const rows = c[0];
        const topk = c[1];
        const xh = try alloc.alloc(u16, rows * dim);
        for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
        const sl = try alloc.alloc(u32, rows * topk);
        const sc = try alloc.alloc(f32, rows * topk);
        for (sl) |*v| v.* = rnd.uintLessThan(u32, @intCast(E));
        for (sc) |*v| v.* = rnd.float(f32);
        const xa = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(rows), @intCast(dim) }, 2, .float16);
        defer _ = mlx.mlx_array_free(xa);
        const sa = mlx.mlx_array_new_data(sl.ptr, &[_]c_int{@intCast(rows * topk)}, 1, .uint32);
        defer _ = mlx.mlx_array_free(sa);
        const ca = mlx.mlx_array_new_data(sc.ptr, &[_]c_int{@intCast(rows * topk)}, 1, .float32);
        defer _ = mlx.mlx_array_free(ca);
        const dec = try moeSwigluFused(s, xa, tr, suh, svh, tr, suh, svh, tr, suh, svh, sa, ca, .float16);
        defer _ = mlx.mlx_array_free(dec);
        const pre = try moePrefill(s, xa, tr, suh, svh, tr, suh, svh, tr, suh, svh, sa, ca, @intCast(topk));
        defer _ = mlx.mlx_array_free(pre);
        var cd = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cd);
        var cp = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cp);
        try mlx.check(mlx.mlx_contiguous(&cd, dec, false, s));
        try mlx.check(mlx.mlx_contiguous(&cp, pre, false, s));
        try mlx.check(mlx.mlx_array_eval(cd));
        try mlx.check(mlx.mlx_array_eval(cp));
        const ad = mlx.mlx_array_data_float16(cd) orelse return error.F16Unreadable;
        const ap = mlx.mlx_array_data_float16(cp) orelse return error.F16Unreadable;
        try expectRelRms(ad[0 .. rows * dim], ap[0 .. rows * dim], 0.01);
    }
}

fn fusedChainMatchesHost(fixture: []const u8, rate: exl3.Rate, dec: exl3.Decode) !void {
    setDecodeParams(dec);
    defer setDecodeParams(.mul1);
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const header_len = std.mem.readInt(u64, fixture[0..8], .little);
    const header = fixture[8 .. 8 + header_len];
    const data = fixture[8 + header_len ..];
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header, .{});
    defer parsed.deinit();
    const tm = parsed.value.object.get("trellis").?.object;
    const sm = parsed.value.object.get("suh").?.object;
    const vm = parsed.value.object.get("svh").?.object;
    const t0: usize = @intCast(tm.get("data_offsets").?.array.items[0].integer);
    const t1: usize = @intCast(tm.get("data_offsets").?.array.items[1].integer);
    const s0: usize = @intCast(sm.get("data_offsets").?.array.items[0].integer);
    const s1: usize = @intCast(sm.get("data_offsets").?.array.items[1].integer);
    const v0: usize = @intCast(vm.get("data_offsets").?.array.items[0].integer);
    const v1: usize = @intCast(vm.get("data_offsets").?.array.items[1].integer);
    const trellis_bits = std.mem.bytesAsSlice(u16, data[t0..t1]);
    const suh_bits = std.mem.bytesAsSlice(u16, data[s0..s1]);
    const svh_bits = std.mem.bytesAsSlice(u16, data[v0..v1]);
    const E: usize = 4;
    const dim: usize = 128;
    const topk: usize = 4;
    const tile_n = 8 * 8 * rate.halfwords();
    const st = try alloc.alloc(u16, E * tile_n);
    const su = try alloc.alloc(u16, E * dim);
    const sv = try alloc.alloc(u16, E * dim);
    for (0..E) |e| {
        @memcpy(st[e * tile_n ..][0..tile_n], trellis_bits);
        @memcpy(su[e * dim ..][0..dim], suh_bits);
        @memcpy(sv[e * dim ..][0..dim], svh_bits);
    }
    var prng = std.Random.DefaultPrng.init(211);
    const rnd = prng.random();
    const xh = try alloc.alloc(u16, dim);
    const xf = try alloc.alloc(f32, dim);
    for (xh, xf) |*b, *v| {
        b.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
        v.* = exl3.f16BitsToF32(b.*);
    }
    const sl = try alloc.alloc(u32, topk);
    const sc = try alloc.alloc(f32, topk);
    for (sl, 0..) |*v, i| v.* = @intCast(i % E);
    for (sc) |*v| v.* = rnd.float(f32);
    const xa = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{@intCast(dim)}, 1, .float16);
    defer _ = mlx.mlx_array_free(xa);
    const tr = mlx.mlx_array_new_data(st.ptr, &[_]c_int{ @intCast(E), 8, 8, @intCast(rate.halfwords()) }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr);
    const suh = mlx.mlx_array_new_data(su.ptr, &[_]c_int{ @intCast(E), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(suh);
    const svh = mlx.mlx_array_new_data(sv.ptr, &[_]c_int{ @intCast(E), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(svh);
    const sa = mlx.mlx_array_new_data(sl.ptr, &[_]c_int{@intCast(topk)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(sa);
    const ca = mlx.mlx_array_new_data(sc.ptr, &[_]c_int{@intCast(topk)}, 1, .float32);
    defer _ = mlx.mlx_array_free(ca);
    const fused = try moeSwigluFused(s, xa, tr, suh, svh, tr, suh, svh, tr, suh, svh, sa, ca, .float16);
    defer _ = mlx.mlx_array_free(fused);
    var cf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cf);
    try mlx.check(mlx.mlx_contiguous(&cf, fused, false, s));
    try mlx.check(mlx.mlx_array_eval(cf));
    const af = mlx.mlx_array_data_float16(cf) orelse return error.F16Unreadable;
    const want = try moeSwigluHost(alloc, xf, st, su, sv, st, su, sv, st, su, sv, sl, sc, dim, dim, rate.halfwords(), 8, 8, dec);
    var ss: f64 = 0;
    var ref: f64 = 0;
    for (0..dim) |i| {
        const a: f64 = @floatCast(af[i]);
        const b: f64 = want[i];
        ss += (a - b) * (a - b);
        ref += b * b;
    }
    const rel = @sqrt(ss / @max(ref, 1e-20));
    if (!(rel < 0.005)) {
        std.debug.print("exl3 host oracle rel_rms={d:.6}\n", .{rel});
        return error.TestExpectedEqual;
    }
}

test "exl3 fused decode chain matches the host SwiGLU reference" {
    try fusedChainMatchesHost(exl3.fixtures.k4, exl3.Rate.fromK(4), .mul1);
}

test "exl3 fused decode chain matches the host SwiGLU reference under MCG" {
    try fusedChainMatchesHost(exl3.fixtures.k4, exl3.Rate.fromK(4), .mcg);
}

test "exl3 fused decode chain matches the host SwiGLU reference at K2.5 MCG" {
    try fusedChainMatchesHost(exl3.fixtures.k2p5_mcg, .{ .n = 40 }, .mcg);
}

test "exl3 fused decode chain matches the host SwiGLU reference at a narrowed codeword window" {
    try fusedChainMatchesHost(exl3.fixtures.k2p5_mcg, .{ .n = 40 }, .{ .codebook = .mcg, .window = .w12 });
    try fusedChainMatchesHost(exl3.fixtures.k2p5_mcg_w12, .{ .n = 40 }, .{ .codebook = .mcg, .window = .w12 });
    try fusedChainMatchesHost(exl3.fixtures.k4, exl3.Rate.fromK(4), .{ .codebook = .mul1, .window = .w14 });
}

test "exl3 fused decode chain matches the host SwiGLU reference below window 12" {
    try fusedChainMatchesHost(exl3.fixtures.k2p5_mcg, .{ .n = 40 }, .{ .codebook = .mcg, .window = .w10 });
}

test "exl3 cooperative indexed GEMV matches host MCG tile decode at K4 K3 K2" {
    for (0..PARITY_SEEDS) |i| {
        try indexedParity(exl3.Rate.fromK(4), 128, 128, 4, 10, 23 + i, .mcg);
        try indexedParity(exl3.Rate.fromK(3), 128, 128, 4, 10, 1201 + i, .mcg);
        try indexedParity(exl3.Rate.fromK(2), 128, 128, 4, 10, 1301 + i, .mcg);
    }
}

test "exl3 cooperative indexed GEMV matches the host tile decode at a fractional rate" {
    for (0..PARITY_SEEDS) |i| {
        try indexedParity(.{ .n = 40 }, 128, 128, 4, 10, 1401 + i, .mcg);
        try indexedParity(.{ .n = 44 }, 128, 128, 4, 10, 1501 + i, .mcg);
    }
    try indexedParity(.{ .n = 40 }, 2560, 640, 4, 10, 43, .mul1);
    try indexedParity(.{ .n = 44 }, 2560, 640, 4, 10, 47, .mul1);
}

/// The w12 fixture through the Metal indexed GEMV, scored against sashimi's
/// own reference decode (the fixture's `inner`) rather than against our host
/// decoder — the one bar that certifies the narrowed-window convention on the
/// GPU end to end.
fn indexedGemvMatchesFixtureInner(fixture: []const u8, rate: exl3.Rate, dec: exl3.Decode) !void {
    setDecodeParams(dec);
    defer setDecodeParams(.mul1);
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const header_len = std.mem.readInt(u64, fixture[0..8], .little);
    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, fixture[8 .. 8 + header_len], .{});
    defer parsed.deinit();
    const data = fixture[8 + header_len ..];
    const span = struct {
        fn of(root: std.json.Value, name: []const u8, blob: []const u8) []align(1) const u16 {
            const off = root.object.get(name).?.object.get("data_offsets").?.array.items;
            const a: usize = @intCast(off[0].integer);
            const b: usize = @intCast(off[1].integer);
            return std.mem.bytesAsSlice(u16, blob[a..b]);
        }
    };
    const trellis_bits = span.of(parsed.value, "trellis", data);
    const inner_bits = span.of(parsed.value, "inner", data);

    const E: usize = 3;
    const dim: usize = 128;
    const rows: usize = 6;
    const tile_n = 8 * 8 * rate.halfwords();
    const stacked = try alloc.alloc(u16, E * tile_n);
    const w = try alloc.alloc(u16, E * dim * dim);
    for (0..E) |e| {
        @memcpy(stacked[e * tile_n ..][0..tile_n], trellis_bits);
        @memcpy(w[e * dim * dim ..][0 .. dim * dim], inner_bits);
    }
    var prng = std.Random.DefaultPrng.init(307);
    const rnd = prng.random();
    const xh = try alloc.alloc(u16, rows * dim);
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
    const eids = try alloc.alloc(u32, rows);
    for (eids, 0..) |*v, i| v.* = @intCast(i % E);

    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(rows), @intCast(dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr_arr = mlx.mlx_array_new_data(stacked.ptr, &[_]c_int{ @intCast(E), 8, 8, @intCast(rate.halfwords()) }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr_arr);
    const slots = mlx.mlx_array_new_data(eids.ptr, &[_]c_int{@intCast(rows)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(slots);
    const got = try indexedGemvCoopF16(s, x_arr, tr_arr, slots);
    defer _ = mlx.mlx_array_free(got);
    var contig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(contig);
    try mlx.check(mlx.mlx_contiguous(&contig, got, false, s));
    try mlx.check(mlx.mlx_array_eval(contig));
    const src = mlx.mlx_array_data_float16(contig) orelse return error.F16Unreadable;
    try reportGemmParity(try measureInnerGemmParityOn(alloc, s, src[0 .. rows * dim], xh, eids, w, dim, dim));
}

test "exl3 indexed GEMV decodes the w12 fixture to sashimi's own inner weights" {
    try indexedGemvMatchesFixtureInner(exl3.fixtures.k2p5_mcg_w12, .{ .n = 40 }, .{ .codebook = .mcg, .window = .w12 });
}

test "exl3 cooperative indexed GEMV matches the host tile decode at a narrowed codeword window" {
    for (0..PARITY_SEEDS) |i| {
        try indexedParity(.{ .n = 40 }, 128, 128, 4, 10, 1601 + i, .{ .codebook = .mcg, .window = .w12 });
        // K4 takes the packed fast branch, which decodes through the same helper.
        try indexedParity(exl3.Rate.fromK(4), 128, 128, 4, 10, 1701 + i, .{ .codebook = .mul1, .window = .w12 });
        try indexedParity(.{ .n = 40 }, 128, 128, 4, 10, 1801 + i, .{ .codebook = .mcg, .window = .w14 });
    }
}

test "exl3 cooperative indexed GEMV matches the host tile decode below window 12" {
    for (0..PARITY_SEEDS) |i| {
        try indexedParity(.{ .n = 40 }, 128, 128, 4, 10, 1901 + i, .{ .codebook = .mcg, .window = .w10 });
    }
}

/// One sorted-GEMM arm (NAX where the shape and silicon allow, else the SIMD
/// body) against the host tile decode, at whatever rate the trellis names.
fn sortedGemmParityStats(rate: exl3.Rate, dec: exl3.Decode, win: c_int, seed: u64, mutate: Exl3Mutation) !Exl3GemmParityStats {
    return sortedGemmParityShape(rate, dec, win, seed, mutate, .{});
}

/// Unequal widths catch a swapped stride a square shape hides; `aligned = false` packs
/// several runs into one window.
const ParityShape = struct { in_dim: usize = 128, out_dim: usize = 128, aligned: bool = true };

fn sortedGemmParityShape(rate: exl3.Rate, dec: exl3.Decode, win: c_int, seed: u64, mutate: Exl3Mutation, shape: ParityShape) !Exl3GemmParityStats {
    setDecodeParams(dec);
    defer setDecodeParams(.mul1);
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const E: usize = 4;
    const in_dim = shape.in_dim;
    const out_dim = shape.out_dim;
    const packed_n = rate.halfwords();
    const tile_n = (in_dim / 16) * (out_dim / 16) * packed_n;
    const stacked = try alloc.alloc(u16, E * tile_n);
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    for (stacked) |*v| v.* = @truncate(rnd.int(u32));
    const runs = [_]u32{ 7, 32, 5, 18, 40, 10 };
    var rows: usize = 0;
    for (runs) |r| rows += r;
    const eids = try alloc.alloc(u32, rows);
    {
        var off: usize = 0;
        for (runs, 0..) |r, ei| {
            var j: usize = 0;
            while (j < r) : (j += 1) eids[off + j] = @intCast(ei % E);
            off += r;
        }
    }
    const xh = try alloc.alloc(u16, rows * in_dim);
    for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 2 - 1);
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(rows), @intCast(in_dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr_arr = mlx.mlx_array_new_data(stacked.ptr, &[_]c_int{ @intCast(E), @intCast(in_dim / 16), @intCast(out_dim / 16), @intCast(packed_n) }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr_arr);
    const eid_a = mlx.mlx_array_new_data(eids.ptr, &[_]c_int{@intCast(rows)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(eid_a);
    const got = try innerGemmSortedWinAlign(s, x_arr, tr_arr, eid_a, win, shape.aligned);
    defer _ = mlx.mlx_array_free(got);
    var contig = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(contig);
    try mlx.check(mlx.mlx_contiguous(&contig, got, false, s));
    try mlx.check(mlx.mlx_array_eval(contig));
    const src = mlx.mlx_array_data_float16(contig) orelse return error.F16Unreadable;
    if (mutate == .codeword) stacked[tile_n / 2] ^= 0x40;
    return measureInnerGemmParity(alloc, s, src[0 .. rows * out_dim], xh, eids, stacked, in_dim, out_dim, rate, mutatedDecode(dec, mutate));
}

fn sortedGemmParity(rate: exl3.Rate, dec: exl3.Decode, win: c_int, seed: u64) !void {
    try reportGemmParity(try sortedGemmParityStats(rate, dec, win, seed, .none));
}

/// Seeds are a sample, not a choice: the bar holds for any trellis, so every
/// case sweeps a fixed run of them rather than one that happened to pass.
const PARITY_SEEDS: usize = 8;

test "exl3 sorted GEMM matches the host tile decode at a fractional rate" {
    for (0..PARITY_SEEDS) |i| {
        try sortedGemmParity(.{ .n = 40 }, .mcg, 32, 101 + i);
        try sortedGemmParity(.{ .n = 44 }, .mul1, 32, 201 + i);
        try sortedGemmParity(.{ .n = 42 }, .{ .codebook = .mcg, .window = .w15 }, 32, 251 + i);
        try sortedGemmParity(.{ .n = 42 }, .{ .codebook = .mcg, .window = .w12 }, 16, 271 + i);
        try sortedGemmParity(exl3.Rate.fromK(4), .mul1, 16, 301 + i);
    }
}

test "exl3 sorted GEMM matches the host tile decode at a fractional rate with the NAX arm off" {
    const t = std.testing;
    if (!mlx.streamIsGpu(mlx.gpuStream())) return error.SkipZigTest;
    if (!gemmNaxOn()) return error.SkipZigTest;
    _ = setenv("SUSHI_FORCE_GPU_FAMILY_FALLBACK", "1", 1);
    defer _ = unsetenv("SUSHI_FORCE_GPU_FAMILY_FALLBACK");
    try t.expect(!gemmNaxOn());
    for (0..PARITY_SEEDS) |i| {
        try sortedGemmParity(.{ .n = 40 }, .mcg, 32, 401 + i);
        try sortedGemmParity(.{ .n = 44 }, .mul1, 16, 501 + i);
    }
}

test "exl3 sorted GEMM matches the host tile decode at a narrowed codeword window" {
    for (0..PARITY_SEEDS) |i| {
        try sortedGemmParity(.{ .n = 40 }, .{ .codebook = .mcg, .window = .w12 }, 32, 601 + i);
        try sortedGemmParity(exl3.Rate.fromK(4), .{ .codebook = .mul1, .window = .w12 }, 16, 701 + i);
        try sortedGemmParity(.{ .n = 40 }, .{ .codebook = .mcg, .window = .w14 }, 16, 801 + i);
    }
}

test "exl3 the GEMM parity bar convicts a wrong window and a wrong codeword" {
    // The bar has to fail on a wrong arm, not only pass on a right one: run
    // the real kernel and give the reference a decode the arm did not use.
    if (!mlx.streamIsGpu(mlx.gpuStream())) return error.SkipZigTest;
    for ([_]Exl3Mutation{ .window, .codeword }) |m| {
        for ([_]exl3.Decode{ .mcg, .{ .codebook = .mul1, .window = .w12 } }) |dec| {
            const st = try sortedGemmParityStats(.{ .n = 40 }, dec, 16, 901, m);
            try std.testing.expect(exl3GemmParityVerdict(st.finite, st.kern_max, st.rms_kern, st.rms_comp, st.ceiling) != null);
        }
        // Again at the production reduction width, where the ceiling is widest.
        const wide = try indexedParityStats(exl3.Rate.fromK(4), 2560, 640, 4, 10, 907, .mul1, m);
        try std.testing.expect(exl3GemmParityVerdict(wide.finite, wide.kern_max, wide.rms_kern, wide.rms_comp, wide.ceiling) != null);
    }
}

test "exl3 sorted GEMM matches the host tile decode at a narrowed window with the NAX arm off" {
    const t = std.testing;
    if (!mlx.streamIsGpu(mlx.gpuStream())) return error.SkipZigTest;
    if (!gemmNaxOn()) return error.SkipZigTest;
    _ = setenv("SUSHI_FORCE_GPU_FAMILY_FALLBACK", "1", 1);
    defer _ = unsetenv("SUSHI_FORCE_GPU_FAMILY_FALLBACK");
    try t.expect(!gemmNaxOn());
    for (0..PARITY_SEEDS) |i| {
        try sortedGemmParity(.{ .n = 40 }, .{ .codebook = .mcg, .window = .w12 }, 32, 1001 + i);
        try sortedGemmParity(exl3.Rate.fromK(4), .{ .codebook = .mul1, .window = .w14 }, 16, 1101 + i);
    }
}

test "exl3 sorted GEMM: the simdgroup-matrix body and the scalar body both match the host tile decode" {
    const t = std.testing;
    if (!mlx.streamIsGpu(mlx.gpuStream())) return error.SkipZigTest;
    var env: FallbackEnv = .{};
    env.force();
    defer env.restore();
    try t.expect(!gemmNaxOn());
    defer gemm_simdmat_force = null;
    const w15: exl3.Decode = .{ .codebook = .mcg, .window = .w15 };
    for ([_]bool{ true, false }) |simdmat| {
        gemm_simdmat_force = simdmat;
        gemm_simdmat_engaged = false;
        funnel_engaged = @splat(false);
        for (0..PARITY_SEEDS) |i| {
            try sortedGemmParity(.{ .n = 36 }, .{ .codebook = .mcg, .window = .w12 }, 32, 2001 + i);
            try sortedGemmParity(.{ .n = 48 }, w15, 32, 1201 + i);
            try sortedGemmParity(exl3.Rate.fromK(4), w15, 32, 1301 + i);
            try sortedGemmParity(.{ .n = 40 }, .{ .codebook = .mcg, .window = .w12 }, 16, 1401 + i);
            try sortedGemmParity(.{ .n = 48 }, .mul1, 16, 1501 + i);
            try sortedGemmParity(.{ .n = 44 }, .mul1, 32, 1601 + i);
            try sortedGemmParity(.{ .n = 42 }, w15, 32, 2301 + i);
            try sortedGemmParity(.{ .n = 42 }, .{ .codebook = .mcg, .window = .w12 }, 16, 2401 + i);
            try reportGemmParity(try sortedGemmParityShape(.{ .n = 48 }, w15, 32, 1701 + i, .none, .{ .aligned = false }));
            try reportGemmParity(try sortedGemmParityShape(.{ .n = 48 }, w15, 32, 1801 + i, .none, .{ .in_dim = 2560, .out_dim = 640 }));
            try reportGemmParity(try sortedGemmParityShape(exl3.Rate.fromK(4), w15, 32, 1901 + i, .none, .{ .in_dim = 640, .out_dim = 2560, .aligned = false }));
        }
        try t.expectEqual(simdmat, gemm_simdmat_engaged);
        try t.expectEqual(simdmat, funnel_engaged[@backingInt(FunnelArm.simdmat)]);
    }
    gemm_simdmat_force = true;
    const mimo: exl3.Decode = .{ .codebook = .mcg, .window = .w12 };
    try reportGemmParity(try sortedGemmParityShape(.{ .n = 36 }, mimo, 32, 2101, .none, .{ .in_dim = 4096, .out_dim = 2048 }));
    try reportGemmParity(try sortedGemmParityShape(.{ .n = 36 }, mimo, 16, 2201, .none, .{ .in_dim = 2048, .out_dim = 4096, .aligned = false }));
}

test "exl3 sorted GEMM body ubench: simdgroup-matrix vs scalar at the served shapes, A B B A" {
    if (!exl3UbenchOn()) return error.SkipZigTest;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var env: FallbackEnv = .{};
    env.force();
    defer env.restore();
    setDecodeParams(.{ .codebook = .mcg, .window = .w15 });
    defer setDecodeParams(.mul1);
    defer gemm_simdmat_force = null;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const E: c_int = 512;
    const BLOCKS = 3;
    var prng = std.Random.DefaultPrng.init(0xab12);
    const rnd = prng.random();
    const io = std.Io.Threaded.global_single_threaded.io();
    for ([_]c_int{ 48, 64 }) |nhw| for ([_][2]c_int{ .{ 2560, 640 }, .{ 640, 2560 } }) |shape| {
        const it = @divExact(shape[0], 16);
        const ot = @divExact(shape[1], 16);
        const tr_h = try alloc.alloc(u16, @intCast(E * it * ot * nhw));
        for (tr_h) |*v| v.* = @truncate(rnd.int(u32));
        const tr = mlx.mlx_array_new_data(tr_h.ptr, &[_]c_int{ E, it, ot, nhw }, 4, .uint16);
        defer _ = mlx.mlx_array_free(tr);
        for ([_]c_int{ 170, 640, 2050, 20480 }) |nslots| {
            const ids = try alloc.alloc(u32, @intCast(nslots));
            for (ids) |*v| v.* = rnd.uintLessThan(u32, @intCast(E));
            std.mem.sort(u32, ids, {}, std.sort.asc(u32));
            const eids = mlx.mlx_array_new_data(ids.ptr, &[_]c_int{nslots}, 1, .uint32);
            defer _ = mlx.mlx_array_free(eids);
            const xh = try alloc.alloc(u16, @intCast(nslots * shape[0]));
            for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 0.1);
            const x = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ nslots, shape[0] }, 2, .float16);
            defer _ = mlx.mlx_array_free(x);
            // Build the window table outside timed calls; it drains the GPU.
            const tab = try gemmWindowTable(s, eids, nslots, 32, true);
            defer _ = mlx.mlx_array_free(tab.starts);
            defer _ = mlx.mlx_array_free(tab.nlives);
            for ([_]bool{ false, true }) |arm| {
                gemm_simdmat_force = arm;
                const warm = try innerGemmSortedTable(s, x, tr, eids, 32, true, tab);
                try mlx.check(mlx.mlx_array_eval(warm));
                _ = mlx.mlx_array_free(warm);
            }
            var ratio: [BLOCKS]f64 = undefined;
            var total: [2]u64 = .{ 0, 0 };
            for (&ratio) |*r| {
                var blk: [2]u64 = .{ 0, 0 };
                for ([_]bool{ false, true, true, false }) |arm| {
                    gemm_simdmat_force = arm;
                    var sw = io_util.Stopwatch.init(io);
                    const got = try innerGemmSortedTable(s, x, tr, eids, 32, true, tab);
                    try mlx.check(mlx.mlx_array_eval(got));
                    blk[@intFromBool(arm)] += sw.read();
                    _ = mlx.mlx_array_free(got);
                }
                r.* = @as(f64, @floatFromInt(blk[0])) / @as(f64, @floatFromInt(blk[1]));
                total[0] += blk[0];
                total[1] += blk[1];
            }
            std.mem.sort(f64, &ratio, {}, std.sort.asc(f64));
            benchPrint("[gemm-body] n{d} {d}->{d} slots={d}: scalar {d} us, simdgroup-matrix {d} us per call; paired A B B A ratio {d:.2}x (blocks {d:.2}..{d:.2})\n", .{
                nhw,                            shape[0],                       shape[1],          nslots,
                total[0] / (2 * BLOCKS * 1000), total[1] / (2 * BLOCKS * 1000), ratio[BLOCKS / 2], ratio[0],
                ratio[BLOCKS - 1],
            });
        }
    };
}

test "exl3 decode and prefill arms agree with the indexed chain at production geometry" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const E: usize = 4;
    const H: usize = 2560;
    const I: usize = 640;
    const topk: usize = 4;
    const gu_n = (H / 16) * (I / 16) * 64;
    const d_n = (I / 16) * (H / 16) * 64;
    const tr_gu = try alloc.alloc(u16, E * gu_n);
    const tr_d = try alloc.alloc(u16, E * d_n);
    const suh_h = try alloc.alloc(u16, E * H);
    const svh_i = try alloc.alloc(u16, E * I);
    const suh_i = try alloc.alloc(u16, E * I);
    const svh_h = try alloc.alloc(u16, E * H);
    var prng = std.Random.DefaultPrng.init(233);
    const rnd = prng.random();
    for (tr_gu) |*v| v.* = @truncate(rnd.int(u32));
    for (tr_d) |*v| v.* = @truncate(rnd.int(u32));
    for (suh_h) |*v| v.* = exl3.f32ToF16Bits(0.5 + rnd.float(f32));
    for (svh_i) |*v| v.* = exl3.f32ToF16Bits(0.5 + rnd.float(f32));
    for (suh_i) |*v| v.* = exl3.f32ToF16Bits(0.5 + rnd.float(f32));
    for (svh_h) |*v| v.* = exl3.f32ToF16Bits(0.5 + rnd.float(f32));
    const tg = mlx.mlx_array_new_data(tr_gu.ptr, &[_]c_int{ @intCast(E), @intCast(H / 16), @intCast(I / 16), 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tg);
    const td = mlx.mlx_array_new_data(tr_d.ptr, &[_]c_int{ @intCast(E), @intCast(I / 16), @intCast(H / 16), 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(td);
    const sgh = mlx.mlx_array_new_data(suh_h.ptr, &[_]c_int{ @intCast(E), @intCast(H) }, 2, .float16);
    defer _ = mlx.mlx_array_free(sgh);
    const vgi = mlx.mlx_array_new_data(svh_i.ptr, &[_]c_int{ @intCast(E), @intCast(I) }, 2, .float16);
    defer _ = mlx.mlx_array_free(vgi);
    const sdi = mlx.mlx_array_new_data(suh_i.ptr, &[_]c_int{ @intCast(E), @intCast(I) }, 2, .float16);
    defer _ = mlx.mlx_array_free(sdi);
    const vdh = mlx.mlx_array_new_data(svh_h.ptr, &[_]c_int{ @intCast(E), @intCast(H) }, 2, .float16);
    defer _ = mlx.mlx_array_free(vdh);
    const sl1 = try alloc.alloc(u32, topk);
    const sc1 = try alloc.alloc(f32, topk);
    for (sl1, 0..) |*v, i| v.* = @intCast(i % E);
    for (sc1) |*v| v.* = 0.2 + rnd.float(f32) * 0.3;
    const x1h = try alloc.alloc(u16, H);
    for (x1h) |*v| v.* = exl3.f32ToF16Bits((rnd.float(f32) * 2 - 1) * 0.05);
    const x1 = mlx.mlx_array_new_data(x1h.ptr, &[_]c_int{@intCast(H)}, 1, .float16);
    defer _ = mlx.mlx_array_free(x1);
    const s1 = mlx.mlx_array_new_data(sl1.ptr, &[_]c_int{@intCast(topk)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(s1);
    const c1 = mlx.mlx_array_new_data(sc1.ptr, &[_]c_int{@intCast(topk)}, 1, .float32);
    defer _ = mlx.mlx_array_free(c1);
    const fused1 = try moeSwigluFused(s, x1, tg, sgh, vgi, tg, sgh, vgi, td, sdi, vdh, s1, c1, .float16);
    defer _ = mlx.mlx_array_free(fused1);
    const ref1 = try moeSwigluIndexed(s, x1, tg, sgh, vgi, tg, sgh, vgi, td, sdi, vdh, s1, c1);
    defer _ = mlx.mlx_array_free(ref1);
    var cf1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cf1);
    var cr1 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cr1);
    try mlx.check(mlx.mlx_contiguous(&cf1, fused1, false, s));
    try mlx.check(mlx.mlx_contiguous(&cr1, ref1, false, s));
    try mlx.check(mlx.mlx_array_eval(cf1));
    try mlx.check(mlx.mlx_array_eval(cr1));
    const af1 = mlx.mlx_array_data_float16(cf1) orelse return error.F16Unreadable;
    const ar1 = mlx.mlx_array_data_float16(cr1) orelse return error.F16Unreadable;
    try expectRelRms(af1[0..H], ar1[0..H], 0.01);
    const rows: usize = 20;
    const xnh = try alloc.alloc(u16, rows * H);
    for (xnh) |*v| v.* = exl3.f32ToF16Bits((rnd.float(f32) * 2 - 1) * 0.05);
    const sln = try alloc.alloc(u32, rows * topk);
    const scn = try alloc.alloc(f32, rows * topk);
    for (sln) |*v| v.* = rnd.uintLessThan(u32, @intCast(E));
    for (scn) |*v| v.* = 0.2 + rnd.float(f32) * 0.3;
    const xn = mlx.mlx_array_new_data(xnh.ptr, &[_]c_int{ @intCast(rows), @intCast(H) }, 2, .float16);
    defer _ = mlx.mlx_array_free(xn);
    const sn = mlx.mlx_array_new_data(sln.ptr, &[_]c_int{@intCast(rows * topk)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(sn);
    const cn = mlx.mlx_array_new_data(scn.ptr, &[_]c_int{@intCast(rows * topk)}, 1, .float32);
    defer _ = mlx.mlx_array_free(cn);
    const dec = try moeSwigluFused(s, xn, tg, sgh, vgi, tg, sgh, vgi, td, sdi, vdh, sn, cn, .float16);
    defer _ = mlx.mlx_array_free(dec);
    const pre = try moePrefill(s, xn, tg, sgh, vgi, tg, sgh, vgi, td, sdi, vdh, sn, cn, @intCast(topk));
    defer _ = mlx.mlx_array_free(pre);
    var cd = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cd);
    var cp = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cp);
    try mlx.check(mlx.mlx_contiguous(&cd, dec, false, s));
    try mlx.check(mlx.mlx_contiguous(&cp, pre, false, s));
    try mlx.check(mlx.mlx_array_eval(cd));
    try mlx.check(mlx.mlx_array_eval(cp));
    const ad = mlx.mlx_array_data_float16(cd) orelse return error.F16Unreadable;
    const ap = mlx.mlx_array_data_float16(cp) orelse return error.F16Unreadable;
    try expectRelRms(ad[0 .. rows * H], ap[0 .. rows * H], 0.01);
}

/// One MiMo-V2.6-Flash MoE layer's shape and routing on the GPU arms: hidden
/// != inter, every expert holding its OWN bank, and a routing that leaves some
/// experts unrouted while giving others more rows than one GEMM window holds.
const MimoMoeCase = struct {
    e: usize,
    hidden: usize,
    inter: usize,
    topk: usize,
    rows: usize,
    rate: exl3.Rate,
    dec: exl3.Decode,
    seed: u64,
    /// |suh_h|, |svh_i|, |suh_i|, |svh_h|. The default is the synthetic 0.9 the
    /// parity cases use; the served pack's own magnitudes are `MIMO_BANKS`.
    banks: [4]f32 = @splat(0.9),
    x_scale: f32 = 0.05,
    /// Raw `[E]`-sliced gate/up/down trellis then suh_h/svh_i/suh_i/svh_h, as
    /// the shards store them. Set, the fixture carries the PACK's own bytes.
    real_blob: ?[*:0]const u8 = null,
};

/// The served w12 pack's own scale magnitudes: |suh| is ~0.01 and |svh| ~1,
/// so the residual's size reaches the arm's f16 planes through the GEMMs.
const MIMO_BANKS = [4]f32{ 0.0126, 1.03, 0.0083, 1.009 };

fn readAll(fd: c_int, dst: []u8) !void {
    var got: usize = 0;
    while (got < dst.len) {
        const n = std.c.read(fd, dst.ptr + got, dst.len - got);
        if (n <= 0) return error.RealBlobShort;
        got += @intCast(n);
    }
}

const MimoMoeFixture = struct {
    arrays: [10]mlx.mlx_array,
    gate_t: []u16,
    up_t: []u16,
    down_t: []u16,
    suh_h: []u16,
    svh_i: []u16,
    suh_i: []u16,
    svh_h: []u16,
    xf: []f32,
    slots: []u32,
    scores: []f32,

    fn deinit(self: *MimoMoeFixture) void {
        for (self.arrays) |a| _ = mlx.mlx_array_free(a);
    }
};

/// Real routing: every row takes `topk` DISTINCT experts from a skewed draw, so
/// a few experts carry runs longer than a window and many carry none.
fn mimoRouting(alloc: std.mem.Allocator, rows: usize, topk: usize, e: usize, rnd: std.Random) ![]u32 {
    const out = try alloc.alloc(u32, rows * topk);
    const hot = @max(topk, e / 8);
    for (0..rows) |r| {
        const row = out[r * topk ..][0..topk];
        var k: usize = 0;
        while (k < topk) {
            const pick: u32 = if (rnd.float(f32) < 0.75)
                rnd.uintLessThan(u32, @intCast(hot))
            else
                rnd.uintLessThan(u32, @intCast(e));
            if (std.mem.indexOfScalar(u32, row[0..k], pick) != null) continue;
            row[k] = pick;
            k += 1;
        }
    }
    return out;
}

fn mimoMoeFixture(alloc: std.mem.Allocator, c: MimoMoeCase) !MimoMoeFixture {
    const n = c.rate.halfwords();
    const ith = c.hidden / 16;
    const iti = c.inter / 16;
    const gu_tile = ith * iti * n;
    var prng = std.Random.DefaultPrng.init(c.seed);
    const rnd = prng.random();
    const gate_t = try alloc.alloc(u16, c.e * gu_tile);
    const up_t = try alloc.alloc(u16, c.e * gu_tile);
    const down_t = try alloc.alloc(u16, c.e * gu_tile);
    for ([_][]u16{ gate_t, up_t, down_t }) |bank| {
        for (bank) |*v| v.* = @truncate(rnd.int(u32));
    }
    var real_fd: ?c_int = null;
    if (c.real_blob) |path| {
        const fd = std.c.open(path, .{ .ACCMODE = .RDONLY });
        if (fd < 0) return error.RealBlobOpen;
        real_fd = fd;
        for ([_][]u16{ gate_t, up_t, down_t }) |bank| try readAll(fd, std.mem.sliceAsBytes(bank));
    }
    const suh_h = try alloc.alloc(u16, c.e * c.hidden);
    const svh_i = try alloc.alloc(u16, c.e * c.inter);
    const suh_i = try alloc.alloc(u16, c.e * c.inter);
    const svh_h = try alloc.alloc(u16, c.e * c.hidden);
    for ([_][]u16{ suh_h, svh_i, suh_i, svh_h }, c.banks) |bank, mag| {
        for (bank) |*v| v.* = exl3.f32ToF16Bits(if (rnd.boolean()) mag else -mag);
    }
    if (real_fd) |fd| {
        for ([_][]u16{ suh_h, svh_i, suh_i, svh_h }) |bank| try readAll(fd, std.mem.sliceAsBytes(bank));
        _ = std.c.close(fd);
    }
    const xh = try alloc.alloc(u16, c.rows * c.hidden);
    const xf = try alloc.alloc(f32, c.rows * c.hidden);
    for (xh, xf) |*b, *v| {
        b.* = exl3.f32ToF16Bits((rnd.float(f32) * 2 - 1) * c.x_scale);
        v.* = exl3.f16BitsToF32(b.*);
    }
    const slots = try mimoRouting(alloc, c.rows, c.topk, c.e, rnd);
    const scores = try alloc.alloc(f32, c.rows * c.topk);
    for (scores) |*v| v.* = 0.05 + rnd.float(f32) * 0.3;
    const ci = struct {
        fn i(v: usize) c_int {
            return @intCast(v);
        }
    }.i;
    return .{
        .arrays = .{
            mlx.mlx_array_new_data(gate_t.ptr, &[_]c_int{ ci(c.e), ci(ith), ci(iti), ci(n) }, 4, .uint16),
            mlx.mlx_array_new_data(up_t.ptr, &[_]c_int{ ci(c.e), ci(ith), ci(iti), ci(n) }, 4, .uint16),
            mlx.mlx_array_new_data(down_t.ptr, &[_]c_int{ ci(c.e), ci(iti), ci(ith), ci(n) }, 4, .uint16),
            mlx.mlx_array_new_data(suh_h.ptr, &[_]c_int{ ci(c.e), ci(c.hidden) }, 2, .float16),
            mlx.mlx_array_new_data(svh_i.ptr, &[_]c_int{ ci(c.e), ci(c.inter) }, 2, .float16),
            mlx.mlx_array_new_data(suh_i.ptr, &[_]c_int{ ci(c.e), ci(c.inter) }, 2, .float16),
            mlx.mlx_array_new_data(svh_h.ptr, &[_]c_int{ ci(c.e), ci(c.hidden) }, 2, .float16),
            mlx.mlx_array_new_data(slots.ptr, &[_]c_int{ci(c.rows * c.topk)}, 1, .uint32),
            mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ ci(c.rows), ci(c.hidden) }, 2, .float16),
            mlx.mlx_array_new_data(scores.ptr, &[_]c_int{ci(c.rows * c.topk)}, 1, .float32),
        },
        .gate_t = gate_t,
        .up_t = up_t,
        .down_t = down_t,
        .suh_h = suh_h,
        .svh_i = svh_i,
        .suh_i = suh_i,
        .svh_h = svh_h,
        .xf = xf,
        .slots = slots,
        .scores = scores,
    };
}

fn mimoPrefillArm(s: mlx.mlx_stream, f: *const MimoMoeFixture, topk: usize) !mlx.mlx_array {
    const a = f.arrays;
    return moePrefill(s, a[8], a[0], a[3], a[4], a[1], a[3], a[4], a[2], a[5], a[6], a[7], a[9], @intCast(topk));
}

fn mimoDecodeArm(s: mlx.mlx_stream, f: *const MimoMoeFixture) !mlx.mlx_array {
    const a = f.arrays;
    return moeSwigluFused(s, a[8], a[0], a[3], a[4], a[1], a[3], a[4], a[2], a[5], a[6], a[7], a[9], .float16);
}

fn evalF16(s: mlx.mlx_stream, a: mlx.mlx_array, out: *mlx.mlx_array) ![*c]const f16 {
    try mlx.check(mlx.mlx_contiguous(out, a, false, s));
    try mlx.check(mlx.mlx_array_eval(out.*));
    return mlx.mlx_array_data_float16(out.*) orelse error.F16Unreadable;
}

/// The prefill arm against the host tile decode of the SAME routing: the only
/// oracle here that shares no kernel with what it scores.
fn mimoPrefillMatchesHost(c: MimoMoeCase) !void {
    setDecodeParams(c.dec);
    defer setDecodeParams(.mul1);
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var f = try mimoMoeFixture(alloc, c);
    defer f.deinit();
    const pre = try mimoPrefillArm(s, &f, c.topk);
    defer _ = mlx.mlx_array_free(pre);
    var cp = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cp);
    const got = try evalF16(s, pre, &cp);
    const want = try alloc.alloc(f16, c.rows * c.hidden);
    for (0..c.rows) |r| {
        const y = try moeSwigluHost(
            alloc,
            f.xf[r * c.hidden ..][0..c.hidden],
            f.gate_t,
            f.suh_h,
            f.svh_i,
            f.up_t,
            f.suh_h,
            f.svh_i,
            f.down_t,
            f.suh_i,
            f.svh_h,
            f.slots[r * c.topk ..][0..c.topk],
            f.scores[r * c.topk ..][0..c.topk],
            c.hidden,
            c.inter,
            c.rate.halfwords(),
            c.hidden / 16,
            c.inter / 16,
            c.dec,
        );
        for (y, 0..) |v, i| want[r * c.hidden + i] = @floatCast(v);
    }
    try expectRelRms(got[0 .. c.rows * c.hidden], want, 0.02);
}

/// The two arms on the same rows. The decode chain is what MiMo answers
/// correctly live, so it is the reference at widths the host oracle cannot reach.
fn mimoArmsAgree(c: MimoMoeCase) !void {
    setDecodeParams(c.dec);
    defer setDecodeParams(.mul1);
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    var f = try mimoMoeFixture(arena.allocator(), c);
    defer f.deinit();
    const pre = try mimoPrefillArm(s, &f, c.topk);
    defer _ = mlx.mlx_array_free(pre);
    const dec = try mimoDecodeArm(s, &f);
    defer _ = mlx.mlx_array_free(dec);
    var cp = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cp);
    var cd = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cd);
    const ap = try evalF16(s, pre, &cp);
    const ad = try evalF16(s, dec, &cd);
    try expectRelRms(ap[0 .. c.rows * c.hidden], ad[0 .. c.rows * c.hidden], 0.02);
}

/// The SwiGLU every EXL3 arm approximates, carried end to end in f32: the
/// stored weights are f16 but nothing between them is. `moeSwigluHost` mirrors
/// the kernels' own f16 stores, so it cannot say whether one of them saturated.
const Exl3F32Peaks = struct { g: f64 = 0, u: f64 = 0, mid: f64 = 0, down_inner: f64 = 0 };

fn exl3SwigluF32(
    alloc: std.mem.Allocator,
    x: []const f32,
    f: *const MimoMoeFixture,
    c: MimoMoeCase,
    slots: []const u32,
    scores: []const f32,
    peaks: *Exl3F32Peaks,
    out: []f32,
) !void {
    const n = c.rate.halfwords();
    const gu_tile = (c.hidden / 16) * (c.inter / 16) * n;
    const t_h = try alloc.alloc(f32, c.hidden);
    const t_i = try alloc.alloc(f32, c.inter);
    const gy = try alloc.alloc(f32, c.inter);
    const uy = try alloc.alloc(f32, c.inter);
    const dy = try alloc.alloc(f32, c.hidden);
    @memset(out, 0);
    const proj = struct {
        fn run(src: []const f32, tr: []const u16, suh: []const u16, svh: []const u16, in_f: usize, out_f: usize, rate: exl3.Rate, dec: exl3.Decode, scratch: []f32, dst: []f32) void {
            for (src, suh, scratch) |v, sb, *d| d.* = v * exl3.f16BitsToF32(sb);
            var b: usize = 0;
            while (b < in_f) : (b += exl3.HAD_DIM) {
                var vec: [exl3.HAD_DIM]f32 = scratch[b..][0..exl3.HAD_DIM].*;
                exl3.hadamard128(&vec);
                @memcpy(scratch[b..][0..exl3.HAD_DIM], &vec);
            }
            @memset(dst, 0);
            var tile: [exl3.TILE_VALUES]u16 = undefined;
            const ot = out_f / 16;
            for (0..in_f / 16) |tk| {
                for (0..ot) |tn| {
                    exl3.decodeTile(tr[(tk * ot + tn) * rate.halfwords() ..][0..rate.halfwords()], rate, dec, &tile);
                    for (0..16) |r| {
                        const xv = scratch[tk * 16 + r];
                        for (0..16) |cc| dst[tn * 16 + cc] += xv * exl3.f16BitsToF32(tile[r * 16 + cc]);
                    }
                }
            }
            var ob: usize = 0;
            while (ob < out_f) : (ob += exl3.HAD_DIM) {
                var vec: [exl3.HAD_DIM]f32 = dst[ob..][0..exl3.HAD_DIM].*;
                exl3.hadamard128(&vec);
                @memcpy(dst[ob..][0..exl3.HAD_DIM], &vec);
            }
            for (dst, svh) |*v, sb| v.* *= exl3.f16BitsToF32(sb);
        }
    }.run;
    for (slots, scores) |e, w| {
        const go = e * gu_tile;
        proj(x, f.gate_t[go..][0..gu_tile], f.suh_h[e * c.hidden ..][0..c.hidden], f.svh_i[e * c.inter ..][0..c.inter], c.hidden, c.inter, c.rate, c.dec, t_h, gy);
        proj(x, f.up_t[go..][0..gu_tile], f.suh_h[e * c.hidden ..][0..c.hidden], f.svh_i[e * c.inter ..][0..c.inter], c.hidden, c.inter, c.rate, c.dec, t_h, uy);
        for (gy, uy) |*g, u| {
            peaks.g = @max(peaks.g, @abs(@as(f64, g.*)));
            peaks.u = @max(peaks.u, @abs(@as(f64, u)));
            g.* = (g.* / (1.0 + @exp(-g.*))) * u;
            peaks.mid = @max(peaks.mid, @abs(@as(f64, g.*)));
        }
        proj(gy, f.down_t[e * gu_tile ..][0..gu_tile], f.suh_i[e * c.inter ..][0..c.inter], f.svh_h[e * c.hidden ..][0..c.hidden], c.inter, c.hidden, c.rate, c.dec, t_i, dy);
        for (dy) |v| peaks.down_inner = @max(peaks.down_inner, @abs(@as(f64, v)));
        for (out, dy) |*o, v| o.* += w * v;
    }
}

/// Both arms against the f32 SwiGLU, at whatever magnitude the case carries.
fn mimoArmMatchesF32(c: MimoMoeCase) !void {
    setDecodeParams(c.dec);
    defer setDecodeParams(.mul1);
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var f = try mimoMoeFixture(alloc, c);
    defer f.deinit();
    const y = if (usesPrefillArm(c.rows))
        try mimoPrefillArm(s, &f, c.topk)
    else
        try mimoDecodeArm(s, &f);
    defer _ = mlx.mlx_array_free(y);
    var cy = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cy);
    const got = try evalF16(s, y, &cy);
    const want = try alloc.alloc(f32, c.hidden);
    var peaks: Exl3F32Peaks = .{};
    var ss: f64 = 0;
    var ref: f64 = 0;
    for (0..c.rows) |r| {
        try exl3SwigluF32(alloc, f.xf[r * c.hidden ..][0..c.hidden], &f, c, f.slots[r * c.topk ..][0..c.topk], f.scores[r * c.topk ..][0..c.topk], &peaks, want);
        for (want, 0..) |w, i| {
            const a: f64 = @floatCast(got[r * c.hidden + i]);
            if (!std.math.isFinite(a)) {
                std.debug.print("exl3 arm went non-finite at row {d}: |silu(g)*u| peaks at {d:.0}\n", .{ r, peaks.mid });
                return error.TestExpectedEqual;
            }
            ss += (a - w) * (a - w);
            ref += @as(f64, w) * @as(f64, w);
        }
    }
    const rel = @sqrt(ss / @max(ref, 1e-20));
    if (rel < 0.01) return;
    std.debug.print("exl3 vs f32 SwiGLU rel_rms {d:.6}; peaks g={d:.0} u={d:.0} mid={d:.0}\n", .{ rel, peaks.g, peaks.u, peaks.mid });
    return error.TestExpectedEqual;
}

// The pack's own scale magnitudes with a residual the size the served model
// carries put `silu(gate) * up` past 65504 while every input, weight and
// output stays ordinary: an f16 plane there turns a whole routed row into inf.
// Synthetic-magnitude parity cases cannot see it — they never leave f16 range.
// Every other parity case drives the arms with synthetic trellis and a single
// scale magnitude; a real shard's suh spans 0.0006..0.17 inside one vector.
// `REAL_BLOB` names a pack slice (E experts of gate/up/down trellis, then
// suh_h/svh_i/suh_i/svh_h) so the Metal arms are scored on the bytes a pack
// actually ships. Absent, there is nothing to read and the case skips.
test "mimo_v2 EXL3 arms match the f32 SwiGLU on a real pack's own bytes" {
    const blob = std.c.getenv("REAL_BLOB") orelse return error.SkipZigTest;
    for ([_]usize{ 1, 33 }) |rows| {
        for ([_]f32{ 0.3, 1.0, 3.0 }) |xs| {
            try mimoArmMatchesF32(.{
                .e = 8,
                .hidden = 4096,
                .inter = 2048,
                .topk = 8,
                .rows = rows,
                .rate = .{ .n = 40 },
                .dec = .{ .codebook = .mcg, .window = .w12 },
                .seed = 4242,
                .x_scale = xs,
                .real_blob = blob,
            });
        }
    }
}

test "exl3 token preparation preserves a subnormal scale on a hot BF16 channel" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var xh: [128]u16 = @splat(0);
    xh[0] = @truncate(@as(u32, @bitCast(@as(f32, 400))) >> 16);
    var scales: [128]u16 = @splat(0x3c00);
    scales[0] = 0x0067;
    const x = mlx.mlx_array_new_data(&xh, &.{ 1, 128 }, 2, .bfloat16);
    defer _ = mlx.mlx_array_free(x);
    const suh = mlx.mlx_array_new_data(&scales, &.{ 1, 128 }, 2, .float16);
    defer _ = mlx.mlx_array_free(suh);
    const zero: u32 = 0;
    const slots = mlx.mlx_array_new_data(&zero, &.{1}, 1, .uint32);
    defer _ = mlx.mlx_array_free(slots);
    const prep = try pairPrepareFromTokens(s, x, suh, suh, slots, slots, 128, 1, 1);
    defer _ = mlx.mlx_array_free(prep[0]);
    defer _ = mlx.mlx_array_free(prep[1]);
    const want = 400.0 * exl3.f16BitsToF32(0x0067) / @sqrt(@as(f32, 128));
    for ([_]mlx.mlx_array{ prep[0], prep[1] }) |a| {
        try mlx.check(mlx.mlx_array_eval(a));
        const got = mlx.mlx_array_data_float16(a) orelse return error.F16Unreadable;
        for (0..128) |i| try t.expectApproxEqAbs(want, @as(f32, @floatCast(got[i])), want * 0.001);
    }
}

test "mimo_v2 EXL3 arms stay finite where the SwiGLU product passes the f16 ceiling" {
    for ([_]usize{ 4, 33 }) |rows| {
        try mimoArmMatchesF32(.{
            .e = 8,
            .hidden = 512,
            .inter = 256,
            .topk = 4,
            .rows = rows,
            .rate = .{ .n = 40 },
            .dec = .{ .codebook = .mcg, .window = .w12 },
            .seed = 9001 + rows,
            .banks = MIMO_BANKS,
            .x_scale = 512,
        });
    }
}

test "mimo_v2 EXL3 prefill rows match the host SwiGLU oracle past one GEMM window" {
    for ([_]usize{ 33, 40, 64, 128 }) |rows| {
        try mimoPrefillMatchesHost(.{
            .e = 64,
            .hidden = 256,
            .inter = 128,
            .topk = 8,
            .rows = rows,
            .rate = .{ .n = 40 },
            .dec = .{ .codebook = .mcg, .window = .w12 },
            .seed = 1301 + rows,
        });
    }
}

test "mimo_v2 EXL3 prefill rows match the fused decode arm at E=256 top-8" {
    for ([_]usize{ 33, 64, 128, 512 }) |rows| {
        try mimoArmsAgree(.{
            .e = 256,
            .hidden = 256,
            .inter = 128,
            .topk = 8,
            .rows = rows,
            .rate = .{ .n = 40 },
            .dec = .{ .codebook = .mcg, .window = .w12 },
            .seed = 1401 + rows,
        });
    }
}

test "mimo_v2 EXL3 prefill rows match the fused decode arm at the served hidden and inter" {
    for ([_]usize{ 33, 64 }) |rows| {
        try mimoArmsAgree(.{
            .e = 16,
            .hidden = 4096,
            .inter = 2048,
            .topk = 8,
            .rows = rows,
            .rate = .{ .n = 40 },
            .dec = .{ .codebook = .mcg, .window = .w12 },
            .seed = 1501 + rows,
        });
    }
}

/// The window geometry the production code resolves, through the levers it
/// reads: the cached answer is dropped so the env is what decides.
fn withGemmWindow(win: ?[*:0]const u8, aligned: bool, c: MimoMoeCase) !void {
    const prev_win = gemm_win_cached;
    const prev_align = gemm_align_cached;
    defer {
        gemm_win_cached = prev_win;
        gemm_align_cached = prev_align;
        _ = unsetenv("SUSHI_EXL3_GEMM_WIN");
        _ = unsetenv("SUSHI_EXL3_WIN_ALIGN");
    }
    gemm_win_cached = null;
    gemm_align_cached = null;
    if (win) |w| _ = setenv("SUSHI_EXL3_GEMM_WIN", w, 1) else _ = unsetenv("SUSHI_EXL3_GEMM_WIN");
    _ = setenv("SUSHI_EXL3_WIN_ALIGN", if (aligned) "1" else "0", 1);
    try mimoArmsAgree(c);
}

test "mimo_v2 EXL3 prefill rows match the fused decode arm on every GEMM arm" {
    const base = MimoMoeCase{
        .e = 256,
        .hidden = 256,
        .inter = 128,
        .topk = 8,
        .rows = 55,
        .rate = .{ .n = 40 },
        .dec = .{ .codebook = .mcg, .window = .w12 },
        .seed = 1601,
    };
    for ([_]bool{ true, false }) |nax_off| {
        if (nax_off) {
            if (!gemmNaxOn()) continue;
            _ = setenv("SUSHI_FORCE_GPU_FAMILY_FALLBACK", "1", 1);
        }
        defer if (nax_off) {
            _ = unsetenv("SUSHI_FORCE_GPU_FAMILY_FALLBACK");
        };
        for ([_]?[*:0]const u8{ null, "16" }) |win| {
            for ([_]bool{ true, false }) |aligned| try withGemmWindow(win, aligned, base);
        }
    }
}

// Every MoE layer of a chunk is built before the chunk's ONE evaluation, and
// each layer's routing gives its GEMM a different window count over the one
// cached kernel config, while a tail-bumped pack gives them different RATES:
// the built dispatches must not read each other's.
test "mimo_v2 EXL3 prefill layers built lazily keep their own window count and rate" {
    const c0 = MimoMoeCase{
        .e = 256,
        .hidden = 256,
        .inter = 128,
        .topk = 8,
        .rows = 55,
        .rate = .{ .n = 40 },
        .dec = .{ .codebook = .mcg, .window = .w12 },
        .seed = 0,
    };
    setDecodeParams(c0.dec);
    defer setDecodeParams(.mul1);
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const L = 48;
    var fs: [L]MimoMoeFixture = undefined;
    var lazy: [L]mlx.mlx_array = undefined;
    for (0..L) |i| {
        var c = c0;
        c.seed = 7000 + i;
        // The tail layers a bumped pack packs wider, built into the same graph.
        if (i + 2 >= L) c.rate = .{ .n = 64 };
        fs[i] = try mimoMoeFixture(alloc, c);
        lazy[i] = try mimoPrefillArm(s, &fs[i], c0.topk);
    }
    defer for (0..L) |i| {
        _ = mlx.mlx_array_free(lazy[i]);
        fs[i].deinit();
    };
    const vec = mlx.mlx_vector_array_new_data(&lazy, L);
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_eval(vec));
    for (0..L) |i| {
        const dec = try mimoDecodeArm(s, &fs[i]);
        defer _ = mlx.mlx_array_free(dec);
        var cp = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cp);
        var cd = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cd);
        const ap = try evalF16(s, lazy[i], &cp);
        const ad = try evalF16(s, dec, &cd);
        try expectRelRms(ap[0 .. c0.rows * c0.hidden], ad[0 .. c0.rows * c0.hidden], 0.02);
    }
}

test "exl3 a window wider than the kernel row capacity refuses" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const n: c_int = 40;
    const dim: c_int = 128;
    const E: c_int = 2;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const xh = try alloc.alloc(u16, @intCast(n * dim));
    @memset(xh, 0);
    const tr = try alloc.alloc(u16, @intCast(E * 8 * 8 * 64));
    @memset(tr, 0);
    var eids: [40]u32 = @splat(0);
    const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ n, dim }, 2, .float16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr_arr = mlx.mlx_array_new_data(tr.ptr, &[_]c_int{ E, 8, 8, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr_arr);
    const eid_a = mlx.mlx_array_new_data(&eids, &[_]c_int{n}, 1, .uint32);
    defer _ = mlx.mlx_array_free(eid_a);
    // 32 rows is what both kernel bodies accumulate; past it their own
    // `n > WIN` guard still admits the window and the extra rows go unwritten.
    try t.expectError(error.BadExl3Shape, innerGemmSortedWin(s, x_arr, tr_arr, eid_a, 33));
    try t.expectError(error.BadExl3Shape, innerGemmSortedWin(s, x_arr, tr_arr, eid_a, 64));
    const ok = try innerGemmSortedWin(s, x_arr, tr_arr, eid_a, 32);
    defer _ = mlx.mlx_array_free(ok);
    try t.expectEqual(@as(c_int, n), mlx.getShape(ok)[0]);
}

test "exl3 the GEMM window selector answers the window it was asked for" {
    const t = std.testing;
    try t.expectEqual(@as(?c_int, 4), resolveGemmWindowRows("4"));
    try t.expectEqual(@as(?c_int, 8), resolveGemmWindowRows("8"));
    try t.expectEqual(@as(?c_int, 16), resolveGemmWindowRows("16"));
    try t.expectEqual(@as(?c_int, 32), resolveGemmWindowRows("32"));
    try t.expectEqual(@as(?c_int, null), resolveGemmWindowRows(null));
    try t.expectEqual(@as(?c_int, null), resolveGemmWindowRows(""));
    try t.expectEqual(@as(?c_int, null), resolveGemmWindowRows("0"));
    try t.expectEqual(@as(?c_int, null), resolveGemmWindowRows("64"));
    try t.expectEqual(@as(?c_int, null), resolveGemmWindowRows("16x"));
}

test "exl3 prepareIndexed refuses a row count that is not the slot count" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const dim: c_int = 128;
    const topk: c_int = 4;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const xh = try alloc.alloc(u16, @intCast(dim));
    @memset(xh, 0);
    const suh = try alloc.alloc(u16, @intCast(2 * dim));
    @memset(suh, 0);
    var slots: [4]u32 = @splat(0);
    // One row spelled [1, dim] is the natural shape for a single activation and
    // the kernel reads `topk` rows of it.
    const x1 = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ 1, dim }, 2, .float16);
    defer _ = mlx.mlx_array_free(x1);
    const suh_a = mlx.mlx_array_new_data(suh.ptr, &[_]c_int{ 2, dim }, 2, .float16);
    defer _ = mlx.mlx_array_free(suh_a);
    const sl = mlx.mlx_array_new_data(&slots, &[_]c_int{topk}, 1, .uint32);
    defer _ = mlx.mlx_array_free(sl);
    try t.expectError(error.BadExl3Shape, prepareIndexed(s, x1, suh_a, sl));
    const x0 = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{dim}, 1, .float16);
    defer _ = mlx.mlx_array_free(x0);
    const ok = try prepareIndexed(s, x0, suh_a, sl);
    defer _ = mlx.mlx_array_free(ok);
    try t.expectEqual(topk, mlx.getShape(ok)[0]);
}

test "exl3 prefill output carries the activation dtype" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const E: c_int = 2;
    const dim: c_int = 128;
    const rows: c_int = 20;
    const topk: c_int = 1;
    const tr = try alloc.alloc(u16, @intCast(E * 8 * 8 * 64));
    var prng = std.Random.DefaultPrng.init(21);
    const rnd = prng.random();
    for (tr) |*v| v.* = @truncate(rnd.int(u32));
    const suh = try alloc.alloc(u16, @intCast(E * dim));
    for (suh) |*v| v.* = exl3.f32ToF16Bits(1.0);
    const svh = try alloc.alloc(u16, @intCast(E * dim));
    for (svh) |*v| v.* = exl3.f32ToF16Bits(1.0);
    // A bf16 activation is what the qwen4 trunk hands the routed experts; an
    // f16 result both double-rounds and saturates at 65504.
    const xb = try alloc.alloc(u16, @intCast(rows * dim));
    for (xb) |*v| v.* = @truncate(@as(u32, @bitCast(@as(f32, 0.5))) >> 16);
    const slots_h = try alloc.alloc(u32, @intCast(rows * topk));
    for (slots_h, 0..) |*v, i| v.* = @intCast(i % @as(usize, @intCast(E)));
    const scores_h = try alloc.alloc(f32, @intCast(rows * topk));
    for (scores_h) |*v| v.* = 1e8;
    const x_arr = mlx.mlx_array_new_data(xb.ptr, &[_]c_int{ rows, dim }, 2, .bfloat16);
    defer _ = mlx.mlx_array_free(x_arr);
    const tr_a = mlx.mlx_array_new_data(tr.ptr, &[_]c_int{ E, 8, 8, 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr_a);
    const suh_a = mlx.mlx_array_new_data(suh.ptr, &[_]c_int{ E, dim }, 2, .float16);
    defer _ = mlx.mlx_array_free(suh_a);
    const svh_a = mlx.mlx_array_new_data(svh.ptr, &[_]c_int{ E, dim }, 2, .float16);
    defer _ = mlx.mlx_array_free(svh_a);
    const sl = mlx.mlx_array_new_data(slots_h.ptr, &[_]c_int{rows * topk}, 1, .uint32);
    defer _ = mlx.mlx_array_free(sl);
    const sc = mlx.mlx_array_new_data(scores_h.ptr, &[_]c_int{rows * topk}, 1, .float32);
    defer _ = mlx.mlx_array_free(sc);
    const out = try moePrefill(s, x_arr, tr_a, suh_a, svh_a, tr_a, suh_a, svh_a, tr_a, suh_a, svh_a, sl, sc, topk);
    defer _ = mlx.mlx_array_free(out);
    try t.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(out));
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_contiguous(&c, out, false, s));
    try mlx.check(mlx.mlx_array_eval(c));
    var f32c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f32c);
    try mlx.check(mlx.mlx_astype(&f32c, c, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f32c));
    const p = mlx.mlx_array_data_float32(f32c) orelse return error.F16Unreadable;
    var finite: usize = 0;
    for (0..@intCast(rows * dim)) |j| {
        if (std.math.isFinite(p[j])) finite += 1;
    }
    try t.expectEqual(@as(usize, @intCast(rows * dim)), finite);
}

test "exl3 the decode reduce folds the scores in f32" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const E: c_int = 2;
    const out_dim: c_int = 128;
    const rows: c_int = 2;
    const topk: c_int = 2;
    const nslots: usize = @intCast(rows * topk);
    const od: usize = @intCast(out_dim);
    var prng = std.Random.DefaultPrng.init(37);
    const rnd = prng.random();
    const inner_h = try alloc.alloc(u16, nslots * od);
    for (inner_h) |*v| v.* = exl3.f32ToF16Bits((rnd.float(f32) * 2 - 1) * 10);
    // svh puts the per-slot partials past the f16 range while the score fold
    // brings the answer back: an f16 bank saturates here, an f32 one does not.
    const svh_h = try alloc.alloc(u16, @intCast(E * out_dim));
    for (svh_h) |*v| v.* = exl3.f32ToF16Bits(60000.0);
    const slots_h = try alloc.alloc(u32, nslots);
    for (slots_h, 0..) |*v, i| v.* = @intCast(i % @as(usize, @intCast(E)));
    const sc_h = try alloc.alloc(f32, nslots);
    for (sc_h) |*v| v.* = 1e-4;
    const inner = mlx.mlx_array_new_data(inner_h.ptr, &[_]c_int{ @intCast(nslots), out_dim }, 2, .float16);
    defer _ = mlx.mlx_array_free(inner);
    const svh = mlx.mlx_array_new_data(svh_h.ptr, &[_]c_int{ E, out_dim }, 2, .float16);
    defer _ = mlx.mlx_array_free(svh);
    const slots = mlx.mlx_array_new_data(slots_h.ptr, &[_]c_int{@intCast(nslots)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(slots);
    const scores = mlx.mlx_array_new_data(sc_h.ptr, &[_]c_int{@intCast(nslots)}, 1, .float32);
    defer _ = mlx.mlx_array_free(scores);
    const got = try downFinishReduce(s, inner, svh, slots, scores, out_dim, rows, topk, .float32);
    defer _ = mlx.mlx_array_free(got);
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_contiguous(&c, got, false, s));
    try mlx.check(mlx.mlx_array_eval(c));
    const gp = mlx.mlx_array_data_float32(c) orelse return error.F16Unreadable;
    const nrows: usize = @intCast(rows);
    const host = try alloc.alloc(f32, nrows * od);
    @memset(host, 0);
    var vec: [128]f32 = undefined;
    const ntopk: usize = @intCast(topk);
    for (0..nrows) |r| {
        for (0..ntopk) |k| {
            const slot = r * ntopk + k;
            for (0..od) |j| vec[j] = exl3.f16BitsToF32(inner_h[slot * od + j]);
            exl3.hadamard128(&vec);
            const eid: usize = slots_h[slot];
            for (0..od) |j| host[r * od + j] += vec[j] * exl3.f16BitsToF32(svh_h[eid * od + j]) * sc_h[slot];
        }
    }
    var worst: f32 = 0;
    for (host, 0..) |h, j| {
        try t.expect(std.math.isFinite(gp[j]));
        const rel = @abs(gp[j] - h) / @max(@abs(h), 1e-6);
        if (rel > worst) worst = rel;
    }
    if (!(worst < 1e-4)) {
        std.debug.print("exl3 reduce worst rel {d:.8}\n", .{worst});
        return error.TestExpectedEqual;
    }
}

test "exl3 a diagnostic env switch set to nothing is off" {
    const t = std.testing;
    try t.expect(!diagEnvValueOn(null));
    try t.expect(!diagEnvValueOn("0"));
    try t.expect(!diagEnvValueOn(""));
    try t.expect(diagEnvValueOn("1"));
}

test "exl3 the host SwiGLU oracle decodes at the K its packed dim names" {
    const t = std.testing;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const dim: usize = 128;
    const tiles: usize = dim / 16;
    const E: usize = 2;
    const k: u32 = 3;
    const packed_n = exl3.packedHalfwords(k);
    const stride = tiles * tiles * packed_n;
    // Sized for K4 so a K4 misread stays inside the buffer and shows as a value.
    const room = tiles * tiles * exl3.packedHalfwords(4);
    var prng = std.Random.DefaultPrng.init(3);
    const rnd = prng.random();
    const banks = try alloc.alloc(u16, 3 * E * room);
    for (banks) |*v| v.* = @truncate(rnd.int(u32));
    const gate_t = banks[0 .. E * room];
    const up_t = banks[E * room .. 2 * E * room];
    const down_t = banks[2 * E * room ..];
    const scales = try alloc.alloc(u16, 6 * E * dim);
    for (scales) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 0.5 + 0.75);
    const gate_suh = scales[0 .. E * dim];
    const gate_svh = scales[E * dim .. 2 * E * dim];
    const up_suh = scales[2 * E * dim .. 3 * E * dim];
    const up_svh = scales[3 * E * dim .. 4 * E * dim];
    const down_suh = scales[4 * E * dim .. 5 * E * dim];
    const down_svh = scales[5 * E * dim ..];
    const x = try alloc.alloc(f32, dim);
    for (x) |*v| v.* = rnd.float(f32) * 2 - 1;
    const slots = [_]u32{ 1, 0 };
    const weights = [_]f32{ 0.625, 0.375 };
    const got = try moeSwigluHost(
        alloc,
        x,
        gate_t,
        gate_suh,
        gate_svh,
        up_t,
        up_suh,
        up_svh,
        down_t,
        down_suh,
        down_svh,
        &slots,
        &weights,
        dim,
        dim,
        packed_n,
        tiles,
        tiles,
        .mul1,
    );
    const want = try alloc.alloc(f32, dim);
    @memset(want, 0);
    const scratch_a = try alloc.alloc(f32, dim);
    const scratch_b = try alloc.alloc(f32, dim);
    const gate_y = try alloc.alloc(f32, dim);
    const up_y = try alloc.alloc(f32, dim);
    const h = try alloc.alloc(f32, dim);
    const down_y = try alloc.alloc(f32, dim);
    for (slots, weights) |e, w| {
        const off = e * stride;
        exl3.project(x, gate_t[off..][0..stride], gate_suh[e * dim ..][0..dim], gate_svh[e * dim ..][0..dim], dim, dim, exl3.Rate.fromK(k), .mul1, scratch_a, scratch_b, gate_y);
        exl3.project(x, up_t[off..][0..stride], up_suh[e * dim ..][0..dim], up_svh[e * dim ..][0..dim], dim, dim, exl3.Rate.fromK(k), .mul1, scratch_a, scratch_b, up_y);
        for (0..dim) |i| {
            const g = gate_y[i];
            h[i] = (g / (1.0 + @exp(-g))) * up_y[i];
        }
        exl3.project(h, down_t[off..][0..stride], down_suh[e * dim ..][0..dim], down_svh[e * dim ..][0..dim], dim, dim, exl3.Rate.fromK(k), .mul1, scratch_a, scratch_b, down_y);
        for (0..dim) |i| want[i] += w * down_y[i];
    }
    try t.expectEqualSlices(f32, want, got);
}

test "exl3 a novel row count does not compile another sorted GEMM" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const dim: usize = 256;
    const E: usize = 2;
    const tiles = dim / 16;
    const packed_n = exl3.packedHalfwords(4);
    const tile_n = tiles * tiles * packed_n;
    const stacked = try alloc.alloc(u16, E * tile_n);
    var prng = std.Random.DefaultPrng.init(5);
    const rnd = prng.random();
    for (stacked) |*v| v.* = @truncate(rnd.int(u32));
    const tr = mlx.mlx_array_new_data(stacked.ptr, &[_]c_int{ @intCast(E), @intCast(tiles), @intCast(tiles), @intCast(packed_n) }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr);
    // Row counts the prefill really sees vary per chunk; only the first shape
    // may pay a kernel compile.
    const counts = [_]usize{ 32, 64, 96 };
    var warm_ns: u64 = 0;
    for (counts, 0..) |n, ci| {
        const xh = try alloc.alloc(u16, n * dim);
        for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 0.5);
        const eids = try alloc.alloc(u32, n);
        for (eids, 0..) |*v, i| v.* = @intCast((i / 16) % E);
        const x_arr = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(n), @intCast(dim) }, 2, .float16);
        defer _ = mlx.mlx_array_free(x_arr);
        const eid_a = mlx.mlx_array_new_data(eids.ptr, &[_]c_int{@intCast(n)}, 1, .uint32);
        defer _ = mlx.mlx_array_free(eid_a);
        var first_ns: u64 = 0;
        for (0..3) |rep| {
            var sw = io_util.Stopwatch.init(t.io);
            const y = try innerGemmSorted(s, x_arr, tr, eid_a);
            try mlx.check(mlx.mlx_array_eval(y));
            const dt = sw.read();
            _ = mlx.mlx_array_free(y);
            if (rep == 0) first_ns = dt else warm_ns = @max(warm_ns, dt);
        }
        if (ci == 0) continue;
        benchPrint("[exl3] novel n={d} first={d} us warm={d} us\n", .{ n, first_ns / 1000, warm_ns / 1000 });
        try t.expect(warm_ns > 0);
        try t.expect(first_ns < warm_ns * 10);
    }
}

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

/// Forces the non-NAX arms for one test and puts back whatever the caller's environment held.
const FallbackEnv = struct {
    prior: [64:0]u8 = @splat(0),
    had: bool = false,
    fn force(self: *FallbackEnv) void {
        if (std.c.getenv("SUSHI_FORCE_GPU_FAMILY_FALLBACK")) |p| {
            const v = std.mem.span(p);
            if (v.len > self.prior.len) @panic("SUSHI_FORCE_GPU_FAMILY_FALLBACK is too long to restore");
            @memcpy(self.prior[0..v.len], v);
            self.prior[v.len] = 0;
            self.had = true;
        }
        _ = setenv("SUSHI_FORCE_GPU_FAMILY_FALLBACK", "1", 1);
    }
    fn restore(self: *FallbackEnv) void {
        if (self.had) _ = setenv("SUSHI_FORCE_GPU_FAMILY_FALLBACK", &self.prior, 1) else _ = unsetenv("SUSHI_FORCE_GPU_FAMILY_FALLBACK");
    }
};

// Codebook A/B at production expert shape: ITER forwards per arm built lazily
// and timed as ONE eval, arms alternated over rounds, medians reported.
// Prints only under SUSHI_EXL3_CODEBOOK_AB (a diagnostic, never a test).
test "exl3 codebook A/B at production shape" {
    if (!diagEnvValueOn(std.c.getenv("SUSHI_EXL3_CODEBOOK_AB"))) return error.SkipZigTest;
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    defer setDecodeParams(.mul1);
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const E: usize = 16;
    const topk: usize = 10;
    const in_dim: usize = 2560;
    const out_dim: usize = 640;
    const in_tiles = in_dim / 16;
    const out_tiles = out_dim / 16;
    const g_n = in_tiles * out_tiles * 64;
    var prng = std.Random.DefaultPrng.init(73);
    const rnd = prng.random();
    const tr_g = try alloc.alloc(u16, E * g_n);
    const tr_d = try alloc.alloc(u16, E * g_n);
    for (tr_g) |*v| v.* = @truncate(rnd.int(u32));
    for (tr_d) |*v| v.* = @truncate(rnd.int(u32));
    const ones_in = try alloc.alloc(u16, E * in_dim);
    const ones_out = try alloc.alloc(u16, E * out_dim);
    @memset(ones_in, exl3.f32ToF16Bits(1.0));
    @memset(ones_out, exl3.f32ToF16Bits(1.0));
    const trg = mlx.mlx_array_new_data(tr_g.ptr, &[_]c_int{ @intCast(E), @intCast(in_tiles), @intCast(out_tiles), 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(trg);
    const trd = mlx.mlx_array_new_data(tr_d.ptr, &[_]c_int{ @intCast(E), @intCast(out_tiles), @intCast(in_tiles), 64 }, 4, .uint16);
    defer _ = mlx.mlx_array_free(trd);
    const su_g = mlx.mlx_array_new_data(ones_in.ptr, &[_]c_int{ @intCast(E), @intCast(in_dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(su_g);
    const sv_g = mlx.mlx_array_new_data(ones_out.ptr, &[_]c_int{ @intCast(E), @intCast(out_dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(sv_g);
    const su_d = mlx.mlx_array_new_data(ones_out.ptr, &[_]c_int{ @intCast(E), @intCast(out_dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(su_d);
    const sv_d = mlx.mlx_array_new_data(ones_in.ptr, &[_]c_int{ @intCast(E), @intCast(in_dim) }, 2, .float16);
    defer _ = mlx.mlx_array_free(sv_d);
    const io = std.Io.Threaded.global_single_threaded.io();
    const arms = [_]exl3.Decode{ .mul1, .mcg };
    const rounds: usize = 7;
    std.debug.print("{s:>5} {s:>10} {s:>10} {s:>9}\n", .{ "rows", "mul1 ms", "mcg ms", "mcg/mul1" });
    for ([_]usize{ 1, 4, 16, 512 }) |R| {
        const iters: usize = if (R >= 64) 3 else 10;
        const xh = try alloc.alloc(u16, R * in_dim);
        for (xh) |*v| v.* = exl3.f32ToF16Bits(rnd.float(f32) * 0.1);
        const sl = try alloc.alloc(u32, R * topk);
        const sc = try alloc.alloc(f32, R * topk);
        for (sl, 0..) |*v, i| v.* = @intCast(i % E);
        @memset(sc, 0.1);
        const x = if (R == 1)
            mlx.mlx_array_new_data(xh.ptr, &[_]c_int{@intCast(in_dim)}, 1, .float16)
        else
            mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(R), @intCast(in_dim) }, 2, .float16);
        defer _ = mlx.mlx_array_free(x);
        const slots = mlx.mlx_array_new_data(sl.ptr, &[_]c_int{@intCast(R * topk)}, 1, .uint32);
        defer _ = mlx.mlx_array_free(slots);
        const scores = mlx.mlx_array_new_data(sc.ptr, &[_]c_int{@intCast(R * topk)}, 1, .float32);
        defer _ = mlx.mlx_array_free(scores);
        var med: [2][rounds]f64 = undefined;
        for (0..rounds) |round| {
            for (arms, 0..) |dec, ai| {
                setDecodeParams(dec);
                const outs = try alloc.alloc(mlx.mlx_array, iters + 1);
                for (outs) |*o| {
                    o.* = if (R > DECODE_ROWS_MAX)
                        try moePrefill(s, x, trg, su_g, sv_g, trg, su_g, sv_g, trd, su_d, sv_d, slots, scores, @intCast(topk))
                    else
                        try moeSwigluFused(s, x, trg, su_g, sv_g, trg, su_g, sv_g, trd, su_d, sv_d, slots, scores, .float16);
                }
                try mlx.check(mlx.mlx_array_eval(outs[0]));
                const vec = mlx.mlx_vector_array_new();
                defer _ = mlx.mlx_vector_array_free(vec);
                for (outs[1..]) |o| _ = mlx.mlx_vector_array_append_value(vec, o);
                var sw = io_util.Stopwatch.init(io);
                try mlx.check(mlx.mlx_eval(vec));
                const ns = sw.read();
                med[ai][round] = @as(f64, @floatFromInt(ns)) / 1e6 / @as(f64, @floatFromInt(iters));
                for (outs) |o| _ = mlx.mlx_array_free(o);
            }
        }
        for (&med) |*m| std.mem.sort(f64, m, {}, std.sort.asc(f64));
        const a = med[0][rounds / 2];
        const b = med[1][rounds / 2];
        std.debug.print("{d:5} {d:10.3} {d:10.3} {d:9.3}\n", .{ R, a, b, b / a });
    }
}

fn n40NaxReaderExact(comptime cb: exl3.Codebook, comptime win: exl3.Window, comptime raw: bool) !void {
    return n40WeightReaderExact(cb, win, raw, false);
}

fn n40WeightReaderExact(comptime cb: exl3.Codebook, comptime win: exl3.Window, comptime raw: bool, comptime lane_reader: bool) !void {
    return weightReaderExact(40, cb, win, raw, if (lane_reader) .lane else .nax);
}

fn weightReaderExact(comptime n: u32, comptime cb: exl3.Codebook, comptime win: exl3.Window, comptime raw: bool, comptime reader: enum { lane, nax, simdmat }) !void {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const source = if (reader == .simdmat)
        \\const uint lane = thread_position_in_grid.x % 32u;
        \\const uint tile = thread_position_in_grid.x / 32u;
        \\half2 p[4];
        \\smat_group<uint(NHW)>((const device uint *)trellis + tile * (uint(NHW) / 2u), lane, p);
        \\for (uint j = 0u; j < 4u; j++) {
        \\  result[tile * 256u + lane * 8u + 2u * j] = as_type<ushort>(p[j].x);
        \\  result[tile * 256u + lane * 8u + 2u * j + 1u] = as_type<ushort>(p[j].y);
        \\}
    else if (reader == .lane)
        \\const uint lane = thread_position_in_grid.x % 32u;
        \\const uint tile = thread_position_in_grid.x / 32u;
        \\const device uint *words = (const device uint *)trellis + tile * (uint(NHW) / 2u);
        \\const auto bits = exl3_lane<uint(NHW)>(words, lane);
        \\for (uint j = 0u; j < 8u; j++) {
        \\  const uint cw = exl3_lane_word(bits, exl3_lane_sh(uint(NHW), j)) & 0xffffu;
        \\  result[tile * 256u + lane * 8u + j] = as_type<ushort>(exl3_pairh(uint2(cw)).x);
        \\}
    else
        \\const uint lane = thread_position_in_grid.x % 32u;
        \\const uint tile = thread_position_in_grid.x / 32u;
        \\const nfrag f = nax_wfrag_k<uint(NHW)>((const device uint *)trellis + tile * (uint(NHW) / 2u), lane);
        \\for (uint j = 0u; j < 8u; j++) result[tile * 256u + lane * 8u + j] = as_type<ushort>(f[j]);
    ;
    const identity =
        \\static inline half2 exl3_raw_pair(uint2 cw) { return as_type<half2>(ushort2(cw)); }
        \\#define exl3_pairh exl3_raw_pair
        \\
    ;
    const header = comptime (if (reader == .simdmat) "" else GEMM_NAX_INCLUDES) ++ codebookHelpers(cb, win) ++ (if (raw) identity else "") ++ (if (reader == .simdmat) GEMM_SIMDMAT_FRAGS else GEMM_NAX_FRAGS);
    var kernel: ?mlx.mlx_fast_metal_kernel = null;
    const k = try getNamedKernel(&kernel, comptime std.fmt.comptimePrint("exl3_n{d}_reader_exact", .{n}) ++ cbSuffix(cb) ++ winSuffix(win) ++ (if (raw) "_raw" else "_weights") ++ "_" ++ @tagName(reader), &.{"trellis"}, &.{"result"}, source, header);
    defer _ = mlx.mlx_fast_metal_kernel_free(k);
    var trellis_data: [128 * n]u16 = undefined;
    var prng = std.Random.DefaultPrng.init(430);
    for (&trellis_data) |*v| v.* = prng.random().int(u16);
    if (std.c.getenv("REAL_BLOB")) |path| {
        const fd = std.c.open(path, .{ .ACCMODE = .RDONLY });
        if (fd < 0) return error.RealBlobOpen;
        defer _ = std.c.close(fd);
        try readAll(fd, std.mem.sliceAsBytes(&trellis_data));
    }
    const tr = mlx.mlx_array_new_data(&trellis_data, &.{ 128, @intCast(n) }, 2, .uint16);
    defer _ = mlx.mlx_array_free(tr);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{ 128, 256 }, 2, .uint32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 128, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 128 * 32, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NHW", @intCast(n)));
    const inputs = mlx.mlx_vector_array_new_data(&.{tr}, 1);
    defer _ = mlx.mlx_vector_array_free(inputs);
    var outputs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs, k, inputs, cfg, s));
    var result = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(result);
    try mlx.check(mlx.mlx_vector_array_get(&result, outputs, 0));
    try mlx.check(mlx.mlx_array_eval(result));
    const got = mlx.mlx_array_data_uint32(result) orelse return error.U16Unreadable;
    for (0..128) |tile| {
        var codes: [256]u16 = undefined;
        exl3.unpackTile(trellis_data[tile * n ..][0..n], .{ .n = n }, &codes);
        for (0..32) |lane| {
            const tau = 64 * (lane >> 4) + ((lane & 7) << 3) + ((lane >> 3) & 1);
            for ([_]usize{ 0, 1, 8, 9, 4, 5, 12, 13 }, 0..) |offset, j| {
                const code = codes[if (reader == .nax) 2 * tau + offset else lane * 8 + j];
                const want = if (raw) code else exl3.decodeCodeword(code & win.mask(), cb);
                try t.expectEqual(want, got[tile * 256 + lane * 8 + j]);
            }
        }
    }
}

test "exl3 n40 NAX codewords and decoded weights are exact" {
    try n40NaxReaderExact(.mcg, .w12, true);
    try n40NaxReaderExact(.mcg, .w12, false);
    try n40NaxReaderExact(.mul1, .w8, false);
    try n40NaxReaderExact(.mul1, .w16, false);
}

test "exl3 n42 funnel readers preserve decoded weights at w15 and a narrowed window" {
    inline for (.{ .lane, .nax, .simdmat }) |reader| {
        inline for (.{ exl3.Window.w15, exl3.Window.w12 }) |win| try weightReaderExact(42, .mcg, win, false, reader);
    }
}

test "exl3 n36 simdgroup reader codewords and decoded weights are exact" {
    try weightReaderExact(36, .mcg, .w12, true, .simdmat);
    inline for (.{ exl3.Codebook.mcg, exl3.Codebook.mul1 }) |cb| {
        inline for (.{ exl3.Window.w8, exl3.Window.w12, exl3.Window.w16 }) |win| {
            try weightReaderExact(36, cb, win, false, .simdmat);
        }
    }
}

fn n40PrefillBf16Truth(seed: u64, win: c_int) !void {
    return n40Bf16Truth(seed, win, 65, false);
}

fn n40Bf16Truth(seed: u64, win: c_int, rows: usize, decode: bool) !void {
    return n40Bf16TruthGeometry(seed, win, rows, decode, 256, 128);
}

fn n40Bf16TruthGeometry(seed: u64, win: c_int, rows: usize, decode: bool, hidden: usize, inter: usize) !void {
    const c = MimoMoeCase{ .e = 8, .hidden = hidden, .inter = inter, .topk = 8, .rows = rows, .rate = .{ .n = 40 }, .dec = .{ .codebook = .mcg, .window = .w12 }, .seed = seed, .banks = MIMO_BANKS, .x_scale = 3 };
    return bf16TruthCase(c, win, decode);
}

fn bf16TruthCase(c: MimoMoeCase, win: c_int, decode: bool) !void {
    if (!gemmNaxOn()) return error.SkipZigTest;
    return bf16TruthCaseAnyArm(c, win, decode);
}

fn bf16TruthCaseAnyArm(c: MimoMoeCase, win: c_int, decode: bool) !void {
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const hidden = c.hidden;
    const inter = c.inter;
    const rows = c.rows;
    const seed = c.seed;
    setDecodeParams(c.dec);
    defer setDecodeParams(.mul1);
    const saved_win = gemm_win_cached;
    gemm_win_cached = win;
    defer gemm_win_cached = saved_win;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var f = try mimoMoeFixture(alloc, c);
    defer f.deinit();
    const xb = try alloc.alloc(u16, f.xf.len);
    for (f.xf, xb) |*v, *b| {
        b.* = @truncate(@as(u32, @bitCast(v.*)) >> 16);
        v.* = @bitCast(@as(u32, b.*) << 16);
    }
    _ = mlx.mlx_array_free(f.arrays[8]);
    f.arrays[8] = mlx.mlx_array_new_data(xb.ptr, &.{ @intCast(rows), @intCast(hidden) }, 2, .bfloat16);
    resetFusedDispatchCount();
    const ar = f.arrays;
    const y = if (decode)
        try moeSwigluFused(s, ar[8], ar[0], ar[3], ar[4], ar[1], ar[3], ar[4], ar[2], ar[5], ar[6], ar[7], ar[9], .bfloat16)
    else
        try mimoPrefillArm(s, &f, c.topk);
    defer _ = mlx.mlx_array_free(y);
    // Two or more rows prepare the pair input once per slot in its own dispatch; one row prepares in the pair.
    if (decode and (prepared_mid_force orelse false)) try std.testing.expectEqual(@as(u32, if (rows >= 2) 5 else 4), fusedDispatchCount());
    try std.testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(y));
    var yf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(yf);
    try mlx.check(mlx.mlx_astype(&yf, y, .float32, s));
    try mlx.check(mlx.mlx_array_eval(yf));
    const got = mlx.mlx_array_data_float32(yf) orelse return error.F32Unreadable;
    var peaks: Exl3F32Peaks = .{};
    var err_new: f64 = 0;
    var err_comp: f64 = 0;
    const truth = try alloc.alloc(f32, c.hidden);
    for (0..c.rows) |r| {
        const slots = f.slots[r * c.topk ..][0..c.topk];
        const scores = f.scores[r * c.topk ..][0..c.topk];
        try exl3SwigluF32(alloc, f.xf[r * c.hidden ..][0..c.hidden], &f, c, slots, scores, &peaks, truth);
        const xr = mlx.mlx_array_new_data(xb[r * c.hidden ..].ptr, &.{@intCast(hidden)}, 1, .bfloat16);
        defer _ = mlx.mlx_array_free(xr);
        const sr = mlx.mlx_array_new_data(slots.ptr, &.{@intCast(c.topk)}, 1, .uint32);
        defer _ = mlx.mlx_array_free(sr);
        const cr = mlx.mlx_array_new_data(scores.ptr, &.{@intCast(c.topk)}, 1, .float32);
        defer _ = mlx.mlx_array_free(cr);
        const a = f.arrays;
        const composite = try moeSwigluIndexed(s, xr, a[0], a[3], a[4], a[1], a[3], a[4], a[2], a[5], a[6], sr, cr);
        defer _ = mlx.mlx_array_free(composite);
        var cb = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cb);
        try mlx.check(mlx.mlx_astype(&cb, composite, .bfloat16, s));
        var cf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cf);
        try mlx.check(mlx.mlx_astype(&cf, cb, .float32, s));
        try mlx.check(mlx.mlx_array_eval(cf));
        const comp = mlx.mlx_array_data_float32(cf) orelse return error.F32Unreadable;
        for (truth, 0..) |v, j| {
            const g: f64 = got[r * c.hidden + j];
            const b: f64 = comp[j];
            try std.testing.expect(std.math.isFinite(g) and std.math.isFinite(b) and std.math.isFinite(v));
            err_new += (g - v) * (g - v);
            err_comp += (b - v) * (b - v);
        }
    }
    if (diagEnvValueOn(std.c.getenv("PERF_REPORT"))) std.debug.print("PREPARED_PARITY n={d} decode={} prepared={} hidden={d} inter={d} rows={d} seed={d} rms={e:.9} composite={e:.9}\n", .{ c.rate.n, decode, prepared_mid_force orelse false, hidden, inter, rows, seed, @sqrt(err_new / @as(f64, @floatFromInt(c.rows * c.hidden))), @sqrt(err_comp / @as(f64, @floatFromInt(c.rows * c.hidden))) });
    if (err_new > err_comp) {
        std.debug.print("n40 bf16 truth seed={d} win={d}: squared error {d:.9} > composite {d:.9}\n", .{ seed, win, err_new, err_comp });
        return error.PrefillWorseThanComposite;
    }
}

test "exl3 n40 NAX BF16 prefill no worse than composite against f32 truth" {
    for ([_]c_int{32}) |win| {
        for (0..3) |seed| try n40PrefillBf16Truth(318 + seed, win);
    }
}

test "exl3 BF16 prefill on the simdgroup-matrix body no worse than composite against f32 truth" {
    if (!mlx.streamIsGpu(mlx.gpuStream())) return error.SkipZigTest;
    var env: FallbackEnv = .{};
    env.force();
    defer env.restore();
    defer mimo_prefill_force = null;
    gemm_simdmat_engaged = false;
    // Both window tables: the host-built one, and MiMo's GPU-built metadata with its sorted finish.
    for ([_]?bool{ null, true }) |mimo_meta| {
        mimo_prefill_force = mimo_meta;
        for (0..PARITY_SEEDS) |seed| {
            for ([_]struct { rate: exl3.Rate, dec: exl3.Decode }{
                .{ .rate = .{ .n = 36 }, .dec = .{ .codebook = .mcg, .window = .w12 } },
                .{ .rate = .{ .n = 40 }, .dec = .{ .codebook = .mcg, .window = .w12 } },
                .{ .rate = .{ .n = 48 }, .dec = .{ .codebook = .mcg, .window = .w15 } },
            }) |arm| {
                try bf16TruthCaseAnyArm(.{ .e = 8, .hidden = 256, .inter = 128, .topk = 8, .rows = 65, .rate = arm.rate, .dec = arm.dec, .seed = 318 + seed, .banks = MIMO_BANKS, .x_scale = 3 }, 32, false);
            }
        }
    }
    try std.testing.expect(gemm_simdmat_engaged);
}

const MimoWindowTable = struct { table: WindowTable, inverse: mlx.mlx_array };

const MIMO_WINDOW_SOURCE: [:0]const u8 =
    \\const uint tid = uint(thread_index_in_threadgroup);
    \\const uint n = uint(count[0]);
    \\const uint capacity = (n + uint(WIN) - 1u) / uint(WIN) + uint(EXPERTS);
    \\threadgroup uint counts[uint(EXPERTS)];
    \\for (uint i = tid; i < capacity; i += uint(EXPERTS)) { starts[i] = 0u; nlives[i] = 0u; }
    \\for (uint i = tid; i < n; i += uint(EXPERTS)) inverse[uint(order[i])] = i;
    \\uint lo = 0u, hi = n;
    \\while (lo < hi) { uint m = (lo + hi) >> 1u; if (uint(eids[m]) < tid) lo = m + 1u; else hi = m; }
    \\const uint first = lo;
    \\hi = n;
    \\while (lo < hi) { uint m = (lo + hi) >> 1u; if (uint(eids[m]) <= tid) lo = m + 1u; else hi = m; }
    \\const uint length = lo - first;
    \\counts[tid] = (length + uint(WIN) - 1u) / uint(WIN);
    \\threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    \\uint offset = 0u;
    \\for (uint e = 0u; e < tid; e++) offset += counts[e];
    \\for (uint i = 0u; i < counts[tid]; i++) {
    \\  starts[offset + i] = first + i * uint(WIN);
    \\  nlives[offset + i] = min(uint(WIN), length - i * uint(WIN));
    \\}
;

const MimoWindowKey = struct { rows: c_int, win: c_int, experts: c_int };
var mimo_window_cfgs: CfgCache(MimoWindowKey, 8) = .{};
var mimo_window_kernel: ?mlx.mlx_fast_metal_kernel = null;
var mimo_window_engaged: bool = false;

/// One thread per expert, in one threadgroup.
const MIMO_WINDOW_MAX_EXPERTS: c_int = 512;

fn buildMimoWindowTable(s: mlx.mlx_stream, eids: mlx.mlx_array, order: mlx.mlx_array, n: c_int, win: c_int, experts: c_int) !MimoWindowTable {
    if (experts < 1 or experts > MIMO_WINDOW_MAX_EXPERTS or win < 1 or n < 1) return error.BadExl3Shape;
    const capacity = @divTrunc(n + win - 1, win) + experts;
    const key = MimoWindowKey{ .rows = n, .win = win, .experts = experts };
    const cfg = mimo_window_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &.{capacity}, 1, .uint32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &.{capacity}, 1, .uint32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &.{n}, 1, .uint32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, experts, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, experts, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "WIN", win));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "EXPERTS", experts));
        mimo_window_cfgs.put(key, c);
        break :blk c;
    };
    const nn: u32 = @intCast(n);
    const count = mlx.mlx_array_new_data(&nn, &.{1}, 1, .uint32);
    defer _ = mlx.mlx_array_free(count);
    const kernel = try getNamedKernel(&mimo_window_kernel, "sushi_exl3_mimo_windows", &.{ "eids", "order", "count" }, &.{ "starts", "nlives", "inverse" }, MIMO_WINDOW_SOURCE, "");
    const outputs = try applyOuts(s, kernel, &.{ eids, order, count }, cfg, 3);
    defer _ = mlx.mlx_vector_array_free(outputs);
    var starts = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(starts);
    var nlives = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(nlives);
    var inverse = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(inverse);
    try mlx.check(mlx.mlx_vector_array_get(&starts, outputs, 0));
    try mlx.check(mlx.mlx_vector_array_get(&nlives, outputs, 1));
    try mlx.check(mlx.mlx_vector_array_get(&inverse, outputs, 2));
    if (!mimo_window_engaged) {
        mimo_window_engaged = true;
        if (!@import("builtin").is_test) log.info("[exl3-prefill] GPU window metadata engaged experts={d} win={d}\n", .{ experts, win });
    }
    return .{ .table = .{ .starts = starts, .nlives = nlives, .nwin = capacity }, .inverse = inverse };
}

test "exl3 MiMo GPU window metadata has a routing-independent capacity" {
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    for ([_]c_int{ 256, MIMO_WINDOW_MAX_EXPERTS }) |experts| try mimoWindowCapacityCase(s, experts);
}

fn mimoWindowCapacityCase(s: mlx.mlx_stream, experts: c_int) !void {
    const ne: usize = @intCast(experts);
    for ([_]usize{ 1, 31, 32, 33, 513, 2049, 16 * 1024 }) |n| {
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        const alloc = arena.allocator();
        const ids = try alloc.alloc(u32, n);
        const ord = try alloc.alloc(u32, n);
        for (ids, ord, 0..) |*id, *o, i| {
            id.* = @intCast(@min((i / 97) * 3, ne - 1));
            o.* = @intCast((i + 7) % n);
        }
        const ea = mlx.mlx_array_new_data(ids.ptr, &.{@intCast(n)}, 1, .uint32);
        defer _ = mlx.mlx_array_free(ea);
        const oa = mlx.mlx_array_new_data(ord.ptr, &.{@intCast(n)}, 1, .uint32);
        defer _ = mlx.mlx_array_free(oa);
        const m = try buildMimoWindowTable(s, ea, oa, @intCast(n), 32, experts);
        defer _ = mlx.mlx_array_free(m.table.starts);
        defer _ = mlx.mlx_array_free(m.table.nlives);
        defer _ = mlx.mlx_array_free(m.inverse);
        try std.testing.expectEqual(@as(c_int, @intCast((n + 31) / 32 + ne)), m.table.nwin);
        try mlx.check(mlx.mlx_array_eval(m.table.starts));
        try mlx.check(mlx.mlx_array_eval(m.table.nlives));
        try mlx.check(mlx.mlx_array_eval(m.inverse));
        const starts = mlx.mlx_array_data_uint32(m.table.starts) orelse return error.Unreadable;
        const lives = mlx.mlx_array_data_uint32(m.table.nlives) orelse return error.Unreadable;
        const inv = mlx.mlx_array_data_uint32(m.inverse) orelse return error.Unreadable;
        var row: usize = 0;
        var wi: usize = 0;
        while (row < n) : (wi += 1) {
            var end = row + 1;
            while (end < n and ids[end] == ids[row] and end - row < 32) : (end += 1) {}
            try std.testing.expectEqual(row, starts[wi]);
            try std.testing.expectEqual(end - row, lives[wi]);
            row = end;
        }
        while (wi < @as(usize, @intCast(m.table.nwin))) : (wi += 1) try std.testing.expectEqual(@as(u32, 0), lives[wi]);
        for (0..n) |i| try std.testing.expectEqual((i + n - 7 % n) % n, inv[i]);
    }
}

const MIMO_REDUCE_SOURCE: [:0]const u8 = blk: {
    const old = "const size_t xb = (size_t)slot * (size_t)(ODIM) + base;";
    const at = std.mem.indexOf(u8, REDUCE_SOURCE, old).?;
    break :blk REDUCE_SOURCE[0..at] ++ "const size_t xb = (size_t)inverse[slot] * (size_t)(ODIM) + base;" ++ REDUCE_SOURCE[at + old.len ..];
};
var mimo_reduce_kernel: ?mlx.mlx_fast_metal_kernel = null;
var mimo_reduce_cfgs: CfgCache(DecodeReduceKey, 8) = .{};
var mimo_reduce_engaged: bool = false;

fn finishMimoSorted(s: mlx.mlx_stream, inner: mlx.mlx_array, inverse: mlx.mlx_array, svh: mlx.mlx_array, slots: mlx.mlx_array, scores: mlx.mlx_array, dim: c_int, rows: c_int, topk: c_int, dtype: mlx.mlx_dtype) !mlx.mlx_array {
    if (topk < 1 or topk > REDUCE_MAX_TOPK) return error.Exl3TopkUnsupported;
    const key = DecodeReduceKey{ .out_dim = dim, .rows = rows, .topk = topk, .dtype = dtype };
    const cfg = mimo_reduce_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &.{ rows, dim }, 2, dtype));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, 32 * topk * @divExact(dim, 128), rows, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32 * topk, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "T", dtype));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ODIM", dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "TOPK", topk));
        mimo_reduce_cfgs.put(key, c);
        break :blk c;
    };
    const kernel = try getNamedKernel(&mimo_reduce_kernel, "sushi_exl3_mimo_sorted_reduce", &.{ "inner", "inverse", "svh", "slots", "sc" }, &.{"y"}, MIMO_REDUCE_SOURCE, "");
    const outputs = try applyOuts(s, kernel, &.{ inner, inverse, svh, slots, scores }, cfg, 1);
    defer _ = mlx.mlx_vector_array_free(outputs);
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, outputs, 0));
    if (!mimo_reduce_engaged) {
        mimo_reduce_engaged = true;
        if (!@import("builtin").is_test) log.info("[exl3-prefill] sorted finish/reduce engaged dtype={s}\n", .{@tagName(dtype)});
    }
    return y;
}

test "exl3 MiMo sorted BF16 finish uses one dispatch and preserves f32 truth error" {
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var prng = std.Random.DefaultPrng.init(871);
    var xh: [16 * 128]u16 = undefined;
    var vh: [8 * 128]u16 = undefined;
    var sc: [16]f32 = undefined;
    var slots: [16]u32 = undefined;
    var inverse: [16]u32 = undefined;
    for (&xh) |*v| v.* = exl3.f32ToF16Bits(prng.random().float(f32) * 2 - 1);
    for (&vh) |*v| v.* = exl3.f32ToF16Bits(prng.random().float(f32));
    for (&sc, &slots, &inverse, 0..) |*v, *e, *iv, i| {
        v.* = prng.random().float(f32);
        e.* = @intCast(i % 8);
        iv.* = @intCast((5 * i + 3) % 16);
    }
    const x = mlx.mlx_array_new_data(&xh, &.{ 16, 128 }, 2, .float16);
    defer _ = mlx.mlx_array_free(x);
    const va = mlx.mlx_array_new_data(&vh, &.{ 8, 128 }, 2, .float16);
    defer _ = mlx.mlx_array_free(va);
    const sa = mlx.mlx_array_new_data(&sc, &.{16}, 1, .float32);
    defer _ = mlx.mlx_array_free(sa);
    const ea = mlx.mlx_array_new_data(&slots, &.{16}, 1, .uint32);
    defer _ = mlx.mlx_array_free(ea);
    const inv = mlx.mlx_array_new_data(&inverse, &.{16}, 1, .uint32);
    defer _ = mlx.mlx_array_free(inv);
    resetFusedDispatchCount();
    const y = try finishMimoSorted(s, x, inv, va, ea, sa, 128, 2, 8, .bfloat16);
    defer _ = mlx.mlx_array_free(y);
    try std.testing.expectEqual(@as(u32, 1), fusedDispatchCount());
    var order_h: [16]u32 = undefined;
    for (inverse, 0..) |v, i| order_h[v] = @intCast(i);
    const order_a = mlx.mlx_array_new_data(&order_h, &.{16}, 1, .uint32);
    defer _ = mlx.mlx_array_free(order_a);
    const unsorted = try scatterSorted(s, x, order_a, 128, 16);
    defer _ = mlx.mlx_array_free(unsorted);
    const composite = try downFinishReduce(s, unsorted, va, ea, sa, 128, 2, 8, .bfloat16);
    defer _ = mlx.mlx_array_free(composite);
    var yf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(yf);
    var cf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cf);
    try mlx.check(mlx.mlx_astype(&yf, y, .float32, s));
    try mlx.check(mlx.mlx_astype(&cf, composite, .float32, s));
    try mlx.check(mlx.mlx_array_eval(yf));
    try mlx.check(mlx.mlx_array_eval(cf));
    const yp = mlx.mlx_array_data_float32(yf) orelse return error.Unreadable;
    const cp = mlx.mlx_array_data_float32(cf) orelse return error.Unreadable;
    var en: f64 = 0;
    var ec: f64 = 0;
    for (0..2) |r| {
        var truth: [128]f32 = @splat(0);
        for (0..8) |k| {
            const slot = r * 8 + k;
            var h: [128]f32 = undefined;
            for (&h, 0..) |*v, j| v.* = exl3.f16BitsToF32(xh[inverse[slot] * 128 + j]);
            exl3.hadamard128(&h);
            for (&truth, h, 0..) |*v, a, j| v.* += a * exl3.f16BitsToF32(vh[slots[slot] * 128 + j]) * sc[slot];
        }
        for (truth, 0..) |v, j| {
            const g: f64 = yp[r * 128 + j];
            const c: f64 = cp[r * 128 + j];
            try std.testing.expect(std.math.isFinite(g) and std.math.isFinite(c));
            en += (g - v) * (g - v);
            ec += (c - v) * (c - v);
        }
    }
    try std.testing.expect(en <= ec);
}

var mimo_prefill_force: ?bool = null;

/// The GPU window metadata and the inverse-indexed finish, at the two served geometries (MiMo, Flash-Next).
fn mimoPrefillOn(hidden: c_int, inter: c_int, experts: c_int, topk: c_int) bool {
    if (mimo_prefill_force) |v| return v;
    if (experts < 1) return false;
    return (hidden == 4096 and inter == 2048 and experts <= 256 and topk == 8) or
        (hidden == 2560 and inter == 640 and experts <= MIMO_WINDOW_MAX_EXPERTS and topk == 10);
}

test "exl3 MiMo prefill metadata and sorted finish preserve BF16 f32-truth bar" {
    mimo_prefill_force = true;
    defer mimo_prefill_force = null;
    for (0..3) |seed| try n40PrefillBf16Truth(318 + seed, 32);
}

test "exl3 prefill GPU metadata keys on the served geometries, never on the rate" {
    try std.testing.expect(mimoPrefillOn(4096, 2048, 256, 8));
    try std.testing.expect(mimoPrefillOn(2560, 640, 512, 10));
    try std.testing.expect(!mimoPrefillOn(2560, 640, 513, 10));
    try std.testing.expect(!mimoPrefillOn(2048, 512, 256, 8));
}

test "exl3 Flash-Next prefill reads the sorted down plane through the inverse, to the byte" {
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const c: MimoMoeCase = .{ .e = 32, .hidden = 2560, .inter = 640, .topk = 10, .rows = 200, .rate = .{ .n = 48 }, .dec = .{ .codebook = .mcg, .window = .w15 }, .seed = 653 };
    setDecodeParams(c.dec);
    defer setDecodeParams(.mul1);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var f = try mimoMoeFixture(arena.allocator(), c);
    defer f.deinit();
    defer mimo_prefill_force = null;
    mimo_prefill_force = false;
    const scattered = try mimoPrefillArm(s, &f, c.topk);
    defer _ = mlx.mlx_array_free(scattered);
    mimo_prefill_force = null;
    mimo_window_engaged = false;
    mimo_reduce_engaged = false;
    const served = try mimoPrefillArm(s, &f, c.topk);
    defer _ = mlx.mlx_array_free(served);
    try std.testing.expect(mimo_window_engaged and mimo_reduce_engaged);
    var ca = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ca);
    var cb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cb);
    const a = try evalF16(s, scattered, &ca);
    const b = try evalF16(s, served, &cb);
    const n = c.rows * c.hidden;
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(a[0..n]), std.mem.sliceAsBytes(b[0..n]));
}

test "exl3 n40 decode lane codewords and decoded weights are exact" {
    try n40WeightReaderExact(.mcg, .w12, true, true);
    try n40WeightReaderExact(.mcg, .w12, false, true);
    try n40WeightReaderExact(.mul1, .w8, false, true);
}

test "exl3 n40 decode BF16 rows no worse than composite against f32 truth" {
    for (1..9) |rows| {
        for (0..3) |seed| try n40Bf16Truth(318 + seed, 32, rows, true);
    }
}

var prepared_mid_force: ?bool = null;

test "exl3 n40 prepared mid adds one dispatch and preserves BF16 f32 truth" {
    prepared_mid_force = true;
    defer prepared_mid_force = null;
    for (1..9) |rows| {
        for (0..3) |seed| try n40Bf16Truth(318 + seed, 32, rows, true);
    }
}

/// `metalReplaceAll` for a rewrite a kernel depends on: a pattern the source no longer holds is a
/// compile error, never a silently unchanged kernel.
fn metalReplaceFound(comptime source: []const u8, comptime old: []const u8, comptime replacement: []const u8) [:0]const u8 {
    @setEvalBranchQuota(1000000);
    if (comptime std.mem.indexOf(u8, source, old) == null) @compileError("kernel rewrite pattern not found: " ++ old);
    return metalReplaceAll(source, old, replacement);
}

fn metalReplaceAll(comptime source: []const u8, comptime old: []const u8, comptime replacement: []const u8) [:0]const u8 {
    @setEvalBranchQuota(200000);
    const at = comptime std.mem.indexOf(u8, source, old) orelse return source ++ "";
    return comptime source[0..at] ++ replacement ++ metalReplaceAll(source[at + old.len ..], old, replacement);
}

const DECODE_MID_SOURCE: [:0]const u8 = blk: {
    @setEvalBranchQuota(200000);
    const start = std.mem.indexOf(u8, DOWN_FUSED_SOURCE, "const float sc =").?;
    const end = start + std.mem.indexOf(u8, DOWN_FUSED_SOURCE[start..], "threadgroup_barrier(mem_flags::mem_threadgroup);").?;
    const head =
        \\const uint slot = uint(threadgroup_position_in_grid.y);
        \\const uint sg = uint(threadgroup_position_in_grid.x);
        \\const uint lane = uint(thread_index_in_simdgroup);
        \\const uint eid = uint(slots[slot]);
        \\constexpr uint SGS = uint(IDIM) / 128u;
    ;
    break :blk head ++ "\n" ++ metalReplaceAll(DOWN_FUSED_SOURCE[start..end], "prepared[base", "prepared[(size_t)slot * uint(IDIM) + base");
};

const DOWN_PREPARED_SOURCE: [:0]const u8 = blk: {
    @setEvalBranchQuota(200000);
    const start = std.mem.indexOf(u8, DOWN_FUSED_SOURCE, "const float sc =").?;
    const barrier = "threadgroup_barrier(mem_flags::mem_threadgroup);";
    const end = start + std.mem.indexOf(u8, DOWN_FUSED_SOURCE[start..], barrier).? + barrier.len;
    const head = metalReplaceAll(DOWN_FUSED_SOURCE[0..start], "threadgroup half prepared[uint(IDIM)];", "");
    const tail = metalReplaceAll(DOWN_FUSED_SOURCE[end..], "const threadgroup half *pp", "const device half *pp");
    break :blk head ++ "const device half *prepared = middle + (size_t)slot * uint(IDIM);\n" ++ tail;
};

/// A multi-row block's slots routed to one expert share a threadgroup: the slot whose rank
/// among that expert's slots is a multiple of `GROUP` leads itself and the next `GROUP - 1`
/// (the other threadgroups return). Each weight is read and decoded once, then fed to every
/// member's accumulators in the single-slot kernel's order, so every slot's bytes equal the
/// lane-funnel kernel's.
const GROUP_MEMBERS_SOURCE =
    \\const uint eid = uint(slots[first]);
    \\ulong same = 0ul;
    \\for (uint b = 0u; b < uint(NSLOTS); b += 32u) {
    \\  const bool hit = b + lane < uint(NSLOTS) && uint(slots[b + lane]) == eid;
    \\  same |= (ulong)static_cast<simd_vote::vote_t>(simd_ballot(hit)) << b;
    \\}
    \\if (popcount(same & ((1ul << first) - 1ul)) % uint(GROUP) != 0u) return;
    \\ulong rest = same >> first;
    \\uint members[GROUP];
    \\uint m = 0u;
    \\for (uint j = 0u; j < uint(GROUP); j++) {
    \\  members[j] = first + (rest != 0ul ? uint(ctz(rest)) : 0u);
    \\  m += rest != 0ul ? 1u : 0u;
    \\  rest &= rest - 1ul;
    \\}
    \\const uint prow = (lane & 3u) * 2u;
    \\const uint pcol = lane >> 2u;
    \\const uint sh[8] = {exl3_lane_sh(N, 0u), exl3_lane_sh(N, 1u), exl3_lane_sh(N, 2u), exl3_lane_sh(N, 3u), exl3_lane_sh(N, 4u), exl3_lane_sh(N, 5u), exl3_lane_sh(N, 6u), exl3_lane_sh(N, 7u)};
    \\
;

/// One funnel iteration (two k-tiles, `OTPT` output tiles) over `mc` members; `member_ptr`
/// points at member j's inputs for this iteration's first k-tile.
fn groupFunnelStep(comptime mc: []const u8, comptime member_ptr: []const u8) []const u8 {
    return "exl3_lane_bits<N> merged[2][OTPT];\n" ++
        \\for (uint u = 0u; u < 2u; u++) {
        \\  for (uint o = 0u; o < uint(OTPT); o++) {
        \\    const device uint *words = wp + u * SGS * OT * PACKED_W + o * PACKED_W;
        \\    merged[u][o] = exl3_lane<N>(words, lane);
        \\  }
        \\}
        \\for (uint u = 0u; u < 2u; u++) {
        \\
    ++ "  float ins[" ++ mc ++ "][4];\n  for (uint j = 0u; j < " ++ mc ++ "; j++) {\n    const auto q = " ++ member_ptr ++ " + u * SGS * TILE;\n" ++
        \\    ins[j][0] = float(q[prow]);
        \\    ins[j][1] = float(q[prow + 1u]);
        \\    ins[j][2] = float(q[prow + 8u]);
        \\    ins[j][3] = float(q[prow + 9u]);
        \\  }
        \\  for (uint o = 0u; o < uint(OTPT); o++) {
        \\    for (uint p = 0u; p < 4u; p++) {
        \\      const uint2 cw = uint2(exl3_lane_word(merged[u][o], sh[p * 2u]), exl3_lane_word(merged[u][o], sh[p * 2u + 1u])) & uint2(0xffffu);
        \\      const float2 w = exl3_decode2(cw);
        \\
    ++ "      for (uint j = 0u; j < " ++ mc ++ "; j++) {\n" ++
        \\        acc[j][o][p * 2u] = fma(ins[j][(p * 2u) & 3u], w.x, acc[j][o][p * 2u]);
        \\        acc[j][o][p * 2u + 1u] = fma(ins[j][(p * 2u + 1u) & 3u], w.y, acc[j][o][p * 2u + 1u]);
        \\      }
        \\    }
        \\  }
        \\}
        \\wp += 2u * SGS * OT * PACKED_W;
        \\
    ;
}

/// The single-slot kernels' tile reduction for each of `mc` members, the first `m` stored.
fn groupEpilogue(comptime mc: []const u8, comptime y_at: []const u8, comptime cast: []const u8) []const u8 {
    return "for (uint j = 0u; j < " ++ mc ++ "; j++) {\n" ++
        \\  for (uint o = 0u; o < uint(OTPT); o++) {
        \\    float clo = (acc[j][o][0] + acc[j][o][1]) + (acc[j][o][2] + acc[j][o][3]);
        \\    float chi = (acc[j][o][4] + acc[j][o][5]) + (acc[j][o][6] + acc[j][o][7]);
        \\    clo += simd_shuffle_xor(clo, 1u);
        \\    chi += simd_shuffle_xor(chi, 1u);
        \\    clo += simd_shuffle_xor(clo, 2u);
        \\    chi += simd_shuffle_xor(chi, 2u);
        \\    if ((lane & 3u) == 0u) {
        \\      partial[((j * uint(OTPT) + o) * SGS + sg) * 16u + pcol] = clo;
        \\      partial[((j * uint(OTPT) + o) * SGS + sg) * 16u + pcol + 8u] = chi;
        \\    }
        \\  }
        \\}
        \\threadgroup_barrier(mem_flags::mem_threadgroup);
        \\if (lid < 16u * uint(OTPT) * m) {
        \\  const uint j = lid / (16u * uint(OTPT));
        \\  const uint o = (lid >> 4u) % uint(OTPT);
        \\  float sum = 0.0f;
        \\  for (uint g = 0u; g < SGS; g++) {
        \\    sum += partial[((j * uint(OTPT) + o) * SGS + g) * 16u + (lid & 15u)];
        \\  }
        \\
    ++ "  " ++ y_at ++ " = " ++ cast ++ "(sum);\n" ++
        \\}
        \\threadgroup_barrier(mem_flags::mem_threadgroup);
        \\
    ;
}

/// A lone member takes a one-member copy of the loop: the `GROUP` copy would spend
/// `GROUP - 1` idle FMAs per weight.
fn groupBySize(comptime body: fn (comptime []const u8) []const u8) []const u8 {
    return "if (m == 1u) {\n" ++ body("1u") ++ "} else {\n" ++ body("uint(GROUP)") ++ "}\n";
}

fn pairGroupedBody(comptime mc: []const u8) []const u8 {
    return
    \\for (uint proj = 0u; proj < 2u; proj++) {
    \\  const device half *suh = (proj == 0u) ? suhg : suhu;
    \\  const device ushort *trellis = (proj == 0u) ? tg : tu;
    \\  device float *y = (proj == 0u) ? yg : yu;
    \\
    ++ "  float acc[" ++ mc ++ "][OTPT][8] = {};\n" ++
        \\  const device uint *trellis_e = (const device uint *)(trellis + ((size_t)eid * (size_t)IT * (size_t)OT) * N);
        \\  const device uint *wp = trellis_e + ((size_t)(tk0 + sg) * (size_t)OT + ot) * PACKED_W;
        \\  for (uint c0 = tk0; c0 < tk1; c0 += CHUNK / TILE) {
        \\    const uint c1 = min(c0 + CHUNK / TILE, tk1);
        \\    const uint nb = (c1 - c0) * TILE / 128u;
        \\    threadgroup_barrier(mem_flags::mem_threadgroup);
        \\
    ++ "    for (uint j = 0u; j < " ++ mc ++ "; j++) {\n" ++
        \\    if (j >= m) break;
        \\    for (uint blk = sg; blk < nb; blk += SGS) {
        \\      const uint pbase = c0 * TILE + blk * 128u;
        \\      const size_t xr = (size_t)(members[j] / uint(TOPK)) * (size_t)(IDIM) + pbase;
        \\      const size_t sr = (size_t)eid * (size_t)(IDIM) + pbase;
        \\      float4 v = float4(
        \\        float(x[xr + lane]) * float(suh[sr + lane]),
        \\        float(x[xr + lane + 32u]) * float(suh[sr + lane + 32u]),
        \\        float(x[xr + lane + 64u]) * float(suh[sr + lane + 64u]),
        \\        float(x[xr + lane + 96u]) * float(suh[sr + lane + 96u]));
        \\      for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
        \\        const float p0 = simd_shuffle_xor(v.x, bit);
        \\        const float p1 = simd_shuffle_xor(v.y, bit);
        \\        const float p2 = simd_shuffle_xor(v.z, bit);
        \\        const float p3 = simd_shuffle_xor(v.w, bit);
        \\        const bool lower = (lane & bit) == 0u;
        \\        v.x = lower ? v.x + p0 : p0 - v.x;
        \\        v.y = lower ? v.y + p1 : p1 - v.y;
        \\        v.z = lower ? v.z + p2 : p2 - v.z;
        \\        v.w = lower ? v.w + p3 : p3 - v.w;
        \\      }
        \\      const float s0 = v.x + v.y;
        \\      const float s1 = v.x - v.y;
        \\      const float s2 = v.z + v.w;
        \\      const float s3 = v.z - v.w;
        \\      const uint pw = j * CHUNK + blk * 128u;
        \\      prepared[pw + lane] = half((s0 + s2) * psc);
        \\      prepared[pw + lane + 32u] = half((s1 + s3) * psc);
        \\      prepared[pw + lane + 64u] = half((s0 - s2) * psc);
        \\      prepared[pw + lane + 96u] = half((s1 - s3) * psc);
        \\    }
        \\    }
        \\    threadgroup_barrier(mem_flags::mem_threadgroup);
        \\    const threadgroup half *pp = prepared + sg * TILE;
        \\    for (uint tk = c0 + sg; tk < c1; tk += 2u * SGS) {
        \\
    ++ groupFunnelStep(mc, "(pp + j * CHUNK)") ++
        \\      pp += 2u * SGS * TILE;
        \\    }
        \\  }
        \\
    ++ groupEpilogue(mc, "y[(size_t)(members[j] * uint(NSPLIT) + split) * (size_t)(ODIM) + (ot + o) * TILE + (lid & 15u)]", "") ++
        \\}
        \\
    ;
}

/// `PAIR_GEMV_SOURCE`'s lane-funnel arm, grouped. Members' prepared inputs are staged one
/// projection and one k chunk at a time, so threadgroup memory stays the single-slot 8 KiB.
const PAIR_GEMV_GROUPED_SOURCE: [:0]const u8 =
    \\constexpr uint TILE = 16u;
    \\constexpr uint N = uint(NHW);
    \\constexpr uint PACKED_W = N / 2u;
    \\constexpr uint IT = uint(IDIM) / TILE;
    \\constexpr uint OT = uint(ODIM) / TILE;
    \\constexpr uint SGS = 4u;
    \\constexpr uint KSPAN = uint(IDIM) / uint(NSPLIT);
    \\constexpr uint CHUNK_FIT = (4096u / uint(GROUP)) / 128u * 128u;
    \\constexpr uint CHUNK = CHUNK_FIT < KSPAN ? CHUNK_FIT : KSPAN;
    \\threadgroup float partial[4 * 16 * uint(OTPT) * uint(GROUP)];
    \\threadgroup half prepared[uint(GROUP) * CHUNK];
    \\const uint ot = uint(threadgroup_position_in_grid.x) * uint(OTPT);
    \\const uint first = uint(threadgroup_position_in_grid.y);
    \\const uint split = uint(threadgroup_position_in_grid.z);
    \\const uint sg = uint(simdgroup_index_in_threadgroup);
    \\const uint lane = uint(thread_index_in_simdgroup);
    \\const uint lid = uint(thread_index_in_threadgroup);
    \\const uint tiles_per_split = (IT + uint(NSPLIT) - 1u) / uint(NSPLIT);
    \\const uint tk0 = split * tiles_per_split;
    \\const uint tk1 = min(tk0 + tiles_per_split, IT);
    \\const float psc = 0.08838834764831845f;
    \\
++ GROUP_MEMBERS_SOURCE ++ groupBySize(pairGroupedBody);

fn downGroupedBody(comptime mc: []const u8) []const u8 {
    return "float acc[" ++ mc ++ "][OTPT][8] = {};\n" ++
        \\const device uint *trellis_e = (const device uint *)(trellis + ((size_t)eid * (size_t)IT * (size_t)OT) * N);
        \\const device uint *wp = trellis_e + ((size_t)sg * (size_t)OT + ot) * PACKED_W;
        \\
    ++ "const device half *pm[" ++ mc ++ "];\nfor (uint j = 0u; j < " ++ mc ++ "; j++) pm[j] = middle + (size_t)members[j] * uint(IDIM) + sg * TILE;\n" ++
        \\for (uint tk = sg; tk < IT; tk += 2u * SGS) {
        \\
    ++ groupFunnelStep(mc, "pm[j]") ++
        "  for (uint j = 0u; j < " ++ mc ++ "; j++) pm[j] += 2u * SGS * TILE;\n}\n" ++
        groupEpilogue(mc, "y[(size_t)members[j] * (size_t)(ODIM) + (ot + o) * TILE + (lid & 15u)]", "half");
}

/// `DOWN_PREPARED_SOURCE`'s lane-funnel arm, grouped; each member reads its own prepared middle.
const DOWN_PREPARED_GROUPED_SOURCE: [:0]const u8 =
    \\constexpr uint TILE = 16u;
    \\constexpr uint N = uint(NHW);
    \\constexpr uint PACKED_W = N / 2u;
    \\constexpr uint IT = uint(IDIM) / TILE;
    \\constexpr uint OT = uint(ODIM) / TILE;
    \\constexpr uint SGS = 4u;
    \\threadgroup float partial[4 * 16 * uint(OTPT) * uint(GROUP)];
    \\const uint ot = uint(threadgroup_position_in_grid.x) * uint(OTPT);
    \\const uint first = uint(threadgroup_position_in_grid.y);
    \\const uint sg = uint(simdgroup_index_in_threadgroup);
    \\const uint lane = uint(thread_index_in_simdgroup);
    \\const uint lid = uint(thread_index_in_threadgroup);
    \\
++ GROUP_MEMBERS_SOURCE ++ groupBySize(downGroupedBody);

/// Where lane `lane`'s value of a 128-block lands in lane order: each 16-row tile keeps rows
/// (2q, 2q+1, 2q+8, 2q+9) at 4q..4q+3, the four a GEMV lane with `lane & 3 == q` reads.
const LANE_ORDER_SLOT = "((lane & 16u) + ((lane & 7u) >> 1u) * 4u + ((lane & 15u) >> 3u) * 2u + (lane & 1u))";

/// A decode GEMV source reading its prepared input as one device half4 per lane and k-tile from a
/// lane-ordered buffer instead of four halves; the values and their order are unchanged.
fn laneQuadReads(comptime src: [:0]const u8) [:0]const u8 {
    @setEvalBranchQuota(1000000);
    var out: [:0]const u8 = src;
    for ([_][2][]const u8{
        .{ "pp[u * SGS * TILE + row", "pp + u * SGS * TILE" },
        .{ "prepared[xb + (tk - tk0) * TILE + row", "prepared + xb + (tk - tk0) * TILE" },
        .{ "prepared[tk * TILE + row", "prepared + tk * TILE" },
    }) |r| {
        if (std.mem.indexOf(u8, out, "const float in0 = float(" ++ r[0] ++ "0]);") == null) continue;
        out = metalReplaceAll(out, "const float in0 = float(" ++ r[0] ++ "0]);", "const float4 in4 = float4(*((const device half4 *)(" ++ r[1] ++ " + (lane & 3u) * 4u)));\nconst float in0 = in4.x;");
        out = metalReplaceFound(out, "const float in1 = float(" ++ r[0] ++ "1]);", "const float in1 = in4.y;");
        out = metalReplaceFound(out, "const float in2 = float(" ++ r[0] ++ "2]);", "const float in2 = in4.z;");
        out = metalReplaceFound(out, "const float in3 = float(" ++ r[0] ++ "3]);", "const float in3 = in4.w;");
    }
    if (std.mem.indexOf(u8, out, "ins[j][0] = float(q[prow]);") != null) {
        out = metalReplaceAll(out, "ins[j][0] = float(q[prow]);", "const float4 in4 = float4(*((const device half4 *)(q + (lane & 3u) * 4u)));\n    ins[j][0] = in4.x;");
        out = metalReplaceFound(out, "ins[j][1] = float(q[prow + 1u]);", "ins[j][1] = in4.y;");
        out = metalReplaceFound(out, "ins[j][2] = float(q[prow + 8u]);", "ins[j][2] = in4.z;");
        out = metalReplaceFound(out, "ins[j][3] = float(q[prow + 9u]);", "ins[j][3] = in4.w;");
    }
    if (std.mem.indexOf(u8, out, "in4") == null) @compileError("no prepared read was rewritten");
    if (std.mem.indexOf(u8, out, "row0]") != null or std.mem.indexOf(u8, out, "q[prow]") != null) @compileError("a prepared read kept the natural layout");
    return out;
}

/// A slot's Hadamard-prepared input for both projections, `[slot][gate|up][IDIM]` f16: the values
/// `PAIR_GEMV_SOURCE` otherwise re-derives in every threadgroup of the slot, in lane order.
const PAIR_PREP_SOURCE: [:0]const u8 = metalReplaceFound(PAIR_PREP_BODY, "LANE_SLOT", LANE_ORDER_SLOT);
const PAIR_PREP_BODY: [:0]const u8 =
    \\const uint block = uint(threadgroup_position_in_grid.x);
    \\const uint slot = uint(threadgroup_position_in_grid.y);
    \\const uint pj = uint(threadgroup_position_in_grid.z);
    \\const uint lane = uint(thread_index_in_simdgroup);
    \\const uint eid = uint(slots[slot]);
    \\const uint pbase = block * 128u;
    \\const size_t xr = (size_t)(slot / uint(TOPK)) * (size_t)(IDIM) + pbase;
    \\const size_t sr = (size_t)eid * (size_t)(IDIM) + pbase;
    \\const float psc = 0.08838834764831845f;
    \\const device half *suh = (pj == 0u) ? suhg : suhu;
    \\float4 v = float4(
    \\  float(x[xr + lane]) * float(suh[sr + lane]),
    \\  float(x[xr + lane + 32u]) * float(suh[sr + lane + 32u]),
    \\  float(x[xr + lane + 64u]) * float(suh[sr + lane + 64u]),
    \\  float(x[xr + lane + 96u]) * float(suh[sr + lane + 96u]));
    \\for (ushort bit = 1u; bit <= 16u; bit <<= 1u) {
    \\  const float p0 = simd_shuffle_xor(v.x, bit);
    \\  const float p1 = simd_shuffle_xor(v.y, bit);
    \\  const float p2 = simd_shuffle_xor(v.z, bit);
    \\  const float p3 = simd_shuffle_xor(v.w, bit);
    \\  const bool lower = (lane & bit) == 0u;
    \\  v.x = lower ? v.x + p0 : p0 - v.x;
    \\  v.y = lower ? v.y + p1 : p1 - v.y;
    \\  v.z = lower ? v.z + p2 : p2 - v.z;
    \\  v.w = lower ? v.w + p3 : p3 - v.w;
    \\}
    \\const float s0 = v.x + v.y;
    \\const float s1 = v.x - v.y;
    \\const float s2 = v.z + v.w;
    \\const float s3 = v.z - v.w;
    \\const size_t pw = ((size_t)slot * 2u + pj) * (size_t)(IDIM) + pbase + LANE_SLOT;
    \\prep[pw] = half((s0 + s2) * psc);
    \\prep[pw + 32u] = half((s1 + s3) * psc);
    \\prep[pw + 64u] = half((s0 - s2) * psc);
    \\prep[pw + 96u] = half((s1 - s3) * psc);
;

/// `PAIR_GEMV_SOURCE` reading its slot's prepared input from device memory instead of
/// preparing its K span in threadgroup memory; every arm reads the same values in the same order.
const PAIR_GEMV_PREPARED_SOURCE: [:0]const u8 = blk: {
    @setEvalBranchQuota(400000);
    const start = std.mem.indexOf(u8, PAIR_GEMV_SOURCE, "threadgroup half prepared[2u * KSPAN];").?;
    const barrier = "threadgroup_barrier(mem_flags::mem_threadgroup);\n";
    const end = start + std.mem.indexOf(u8, PAIR_GEMV_SOURCE[start..], barrier).? + barrier.len;
    const body = PAIR_GEMV_SOURCE[0..start] ++ "const device half *prepared = prep + (size_t)slot * 2u * uint(IDIM) + tk0 * TILE;\n" ++ PAIR_GEMV_SOURCE[end..];
    break :blk laneQuadReads(metalReplaceFound(metalReplaceFound(body, "const uint xb = proj * KSPAN;", "const uint xb = proj * uint(IDIM);"), "const threadgroup half *pp", "const device half *pp"));
};

fn pairPreparedGroupedBody(comptime mc: []const u8) []const u8 {
    return
    \\for (uint proj = 0u; proj < 2u; proj++) {
    \\  const device ushort *trellis = (proj == 0u) ? tg : tu;
    \\  device float *y = (proj == 0u) ? yg : yu;
    \\
    ++ "  float acc[" ++ mc ++ "][OTPT][8] = {};\n" ++
        \\  const device uint *trellis_e = (const device uint *)(trellis + ((size_t)eid * (size_t)IT * (size_t)OT) * N);
        \\  const device uint *wp = trellis_e + ((size_t)(tk0 + sg) * (size_t)OT + ot) * PACKED_W;
        \\
    ++ "  const device half *pm[" ++ mc ++ "];\n  for (uint j = 0u; j < " ++ mc ++ "; j++) pm[j] = prep + ((size_t)members[j] * 2u + proj) * uint(IDIM) + (tk0 + sg) * TILE;\n" ++
        \\  for (uint tk = tk0 + sg; tk < tk1; tk += 2u * SGS) {
        \\
    ++ groupFunnelStep(mc, "pm[j]") ++
        "    for (uint j = 0u; j < " ++ mc ++ "; j++) pm[j] += 2u * SGS * TILE;\n  }\n" ++
        groupEpilogue(mc, "y[(size_t)(members[j] * uint(NSPLIT) + split) * (size_t)(ODIM) + (ot + o) * TILE + (lid & 15u)]", "") ++
        \\}
        \\
    ;
}

/// `PAIR_GEMV_GROUPED_SOURCE` on the prepared input: members read their own prepared rows.
const PAIR_GEMV_PREPARED_GROUPED_SOURCE: [:0]const u8 =
    \\constexpr uint TILE = 16u;
    \\constexpr uint N = uint(NHW);
    \\constexpr uint PACKED_W = N / 2u;
    \\constexpr uint IT = uint(IDIM) / TILE;
    \\constexpr uint OT = uint(ODIM) / TILE;
    \\constexpr uint SGS = 4u;
    \\threadgroup float partial[4 * 16 * uint(OTPT) * uint(GROUP)];
    \\const uint ot = uint(threadgroup_position_in_grid.x) * uint(OTPT);
    \\const uint first = uint(threadgroup_position_in_grid.y);
    \\const uint split = uint(threadgroup_position_in_grid.z);
    \\const uint sg = uint(simdgroup_index_in_threadgroup);
    \\const uint lane = uint(thread_index_in_simdgroup);
    \\const uint lid = uint(thread_index_in_threadgroup);
    \\const uint tiles_per_split = (IT + uint(NSPLIT) - 1u) / uint(NSPLIT);
    \\const uint tk0 = split * tiles_per_split;
    \\const uint tk1 = min(tk0 + tiles_per_split, IT);
    \\
++ laneQuadReads(GROUP_MEMBERS_SOURCE ++ groupBySize(pairPreparedGroupedBody));

const PairPrepKeyD = struct { in_dim: c_int, nslots: c_int, topk: c_int };
var pair_prep_cfgs: CfgCache(PairPrepKeyD, 8) = .{};
var pair_prep_kernel: ?mlx.mlx_fast_metal_kernel = null;
var pair_gemv_prepared_kernel: KernelSlots = no_kernels;
var pair_gemv_prepared_grouped_kernel: KernelSlots = no_kernels;

/// Each slot's gate and up inputs, Hadamard-prepared once: `[nslots * 2, in_dim]` f16.
pub fn pairPrepare(s: mlx.mlx_stream, x: mlx.mlx_array, suhg: mlx.mlx_array, suhu: mlx.mlx_array, slots: mlx.mlx_array, in_dim: c_int, nslots: c_int, topk: c_int) !mlx.mlx_array {
    const key = PairPrepKeyD{ .in_dim = in_dim, .nslots = nslots, .topk = topk };
    const cfg = pair_prep_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &.{ nslots * 2, in_dim }, 2, .float16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, @divExact(in_dim, 128) * 32, nslots, 2));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "IDIM", in_dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "TOPK", topk));
        pair_prep_cfgs.put(key, c);
        break :blk c;
    };
    const kernel = try getNamedKernel(&pair_prep_kernel, "sushi_exl3_pair_prepare", &.{ "x", "suhg", "suhu", "slots" }, &.{"prep"}, PAIR_PREP_SOURCE, "");
    const ov = try applyOuts(s, kernel, &.{ x, suhg, suhu, slots }, cfg, 1);
    defer _ = mlx.mlx_vector_array_free(ov);
    var prep = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(prep);
    try mlx.check(mlx.mlx_vector_array_get(&prep, ov, 0));
    return prep;
}

/// `pairGemv` on `pairPrepare`'s output: the same planes, bit for bit.
pub fn pairGemvPrepared(s: mlx.mlx_stream, prep: mlx.mlx_array, tg: mlx.mlx_array, tu: mlx.mlx_array, slots: mlx.mlx_array, in_dim: c_int, out_dim: c_int, nslots: c_int, topk: c_int, group_ask: c_int) !struct { mlx.mlx_array, mlx.mlx_array } {
    const tsh = mlx.getShape(tg);
    const rate = try packedRate(tsh[tsh.len - 1]);
    if (mlx.getShape(tu)[mlx.getShape(tu).len - 1] != tsh[tsh.len - 1]) return error.BadExl3Shape;
    const layout = gemvLayout(rate.n, @divExact(out_dim, 16));
    const group: c_int = if (layout.funnel) group_ask else 0;
    const cfg = try pairGemvConfig(in_dim, out_dim, nslots, topk, rate, layout, group);
    const ins = [_][*:0]const u8{ "prep", "tg", "tu", "slots" };
    const outs = [_][*:0]const u8{ "yg", "yu" };
    const kernel = if (group > 0)
        try codebookKernel(&pair_gemv_prepared_grouped_kernel, "sushi_exl3_pair_gemv_prepared_grouped", &ins, &outs, PAIR_GEMV_PREPARED_GROUPED_SOURCE)
    else
        try codebookKernel(&pair_gemv_prepared_kernel, "sushi_exl3_pair_gemv_prepared", &ins, &outs, PAIR_GEMV_PREPARED_SOURCE);
    logGroupedEngaged(group);
    const ov = try applyOuts(s, kernel, &.{ prep, tg, tu, slots }, cfg, 2);
    logFunnel(layout.funnel, rate.n, .pair, .float16);
    defer _ = mlx.mlx_vector_array_free(ov);
    var a = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(a);
    var b = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(b);
    try mlx.check(mlx.mlx_vector_array_get(&a, ov, 0));
    try mlx.check(mlx.mlx_vector_array_get(&b, ov, 1));
    return .{ a, b };
}

const DecodeMidKey = struct { dim: c_int, nslots: c_int, nsplit: c_int };
var decode_mid_cfgs: CfgCache(DecodeMidKey, 8) = .{};
var decode_mid_kernel: ?mlx.mlx_fast_metal_kernel = null;
var down_prepared_cfgs: CfgCache(DownFusedKey, 8) = .{};
var down_prepared_kernel: KernelSlots = no_kernels;
var down_prepared_grouped_kernel: KernelSlots = no_kernels;
var prepared_mid_engaged: bool = false;
var grouped_engaged: bool = false;

fn prepareDecodeMid(s: mlx.mlx_stream, ig: mlx.mlx_array, iu: mlx.mlx_array, svhg: mlx.mlx_array, svhu: mlx.mlx_array, suhd: mlx.mlx_array, slots: mlx.mlx_array, dim: c_int, nslots: c_int, nsplit: c_int) !mlx.mlx_array {
    const key = DecodeMidKey{ .dim = dim, .nslots = nslots, .nsplit = nsplit };
    const cfg = decode_mid_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &.{ nslots, dim }, 2, .float16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, @divExact(dim, 128) * 32, nslots, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "IDIM", dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "NSPLIT", nsplit));
        decode_mid_cfgs.put(key, c);
        break :blk c;
    };
    const kernel = try getNamedKernel(&decode_mid_kernel, "sushi_exl3_decode_mid", &.{ "ig", "iu", "svhg", "svhu", "suhd", "slots" }, &.{"prepared"}, DECODE_MID_SOURCE, "");
    const outputs = try applyOuts(s, kernel, &.{ ig, iu, svhg, svhu, suhd, slots }, cfg, 1);
    defer _ = mlx.mlx_vector_array_free(outputs);
    var middle = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(middle);
    try mlx.check(mlx.mlx_vector_array_get(&middle, outputs, 0));
    return middle;
}

fn downGemvPreparedMid(s: mlx.mlx_stream, ig: mlx.mlx_array, iu: mlx.mlx_array, trellis: mlx.mlx_array, svhg: mlx.mlx_array, svhu: mlx.mlx_array, suhd: mlx.mlx_array, slots: mlx.mlx_array, in_dim: c_int, out_dim: c_int, nslots: c_int, group_ask: c_int) !mlx.mlx_array {
    const nsplit: c_int = @intCast(pairSplitCountFor(out_dim));
    const middle = try prepareDecodeMid(s, ig, iu, svhg, svhu, suhd, slots, in_dim, nslots, nsplit);
    defer _ = mlx.mlx_array_free(middle);
    const tsh = mlx.getShape(trellis);
    const rate = try packedRate(tsh[tsh.len - 1]);
    const out_tiles = @divExact(out_dim, 16);
    const layout = gemvLayout(rate.n, out_tiles);
    const group: c_int = if (layout.funnel) group_ask else 0;
    const key = DownFusedKey{ .in_dim = in_dim, .out_dim = out_dim, .nslots = nslots, .nsplit = nsplit, .n = rate.n, .layout = layout, .group = group };
    const cfg = down_prepared_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &.{ nslots, out_dim }, 2, .float16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, @divExact(out_tiles, layout.tiles) * 128, nslots, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 128, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "IDIM", in_dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ODIM", out_dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "NHW", @intCast(rate.n)));
        try addGemvLayout(c, layout);
        try addGroup(c, group, nslots);
        down_prepared_cfgs.put(key, c);
        break :blk c;
    };
    const kernel = if (group > 0)
        try codebookKernel(&down_prepared_grouped_kernel, "sushi_exl3_down_prepared_grouped", &.{ "middle", "trellis", "slots" }, &.{"y"}, DOWN_PREPARED_GROUPED_SOURCE)
    else
        try codebookKernel(&down_prepared_kernel, "sushi_exl3_down_prepared", &.{ "middle", "trellis", "slots" }, &.{"y"}, DOWN_PREPARED_SOURCE);
    const outputs = try applyOuts(s, kernel, &.{ middle, trellis, slots }, cfg, 1);
    logFunnel(layout.funnel, rate.n, .prepared_down, mlx.mlx_array_dtype(middle));
    defer _ = mlx.mlx_vector_array_free(outputs);
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_vector_array_get(&y, outputs, 0));
    return y;
}

fn preparedMidOn(hidden: c_int, inter: c_int, experts: c_int, topk: c_int, rows: c_int, dtype: mlx.mlx_dtype) bool {
    if (prepared_mid_force) |v| return v;
    return hidden == 4096 and inter == 2048 and experts > 0 and experts <= 256 and topk == 8 and rows >= 1 and rows <= 8 and dtype == .bfloat16;
}

test "exl3 prepared mid keys on MiMo geometry, never on the rate, and excludes qwen and unmeasured widths" {
    try std.testing.expect(preparedMidOn(4096, 2048, 256, 8, 1, .bfloat16));
    try std.testing.expect(preparedMidOn(4096, 2048, 256, 8, 8, .bfloat16));
    try std.testing.expect(!preparedMidOn(4096, 2048, 256, 8, 9, .bfloat16));
    try std.testing.expect(!preparedMidOn(4096, 2048, 256, 8, 1, .float16));
    try std.testing.expect(!preparedMidOn(2560, 640, 512, 10, 1, .bfloat16));
}

test "exl3 prepared mid production geometry BF16 f32 truth" {
    if (!diagEnvValueOn(std.c.getenv("PERF_REPORT"))) return error.SkipZigTest;
    prepared_mid_force = true;
    defer prepared_mid_force = null;
    for (1..9) |rows| {
        for (0..3) |seed| try n40Bf16TruthGeometry(318 + seed, 32, rows, true, 4096, 2048);
    }
}

test "exl3 MCG half pairs preserve every codeword at windows 8 through 16" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const source =
        \\const uint i = thread_position_in_grid.x;
        \\result[i] = as_type<uint>(exl3_pairh(uint2(codes[i], codes[65535u - i])));
    ;
    const codes = try t.allocator.alloc(u32, 65536);
    defer t.allocator.free(codes);
    for (codes, 0..) |*v, i| v.* = @intCast(i);
    const input = mlx.mlx_array_new_data(codes.ptr, &.{65536}, 1, .uint32);
    defer _ = mlx.mlx_array_free(input);
    const inputs = mlx.mlx_vector_array_new_data(&.{input}, 1);
    defer _ = mlx.mlx_vector_array_free(inputs);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{65536}, 1, .uint32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 128, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 65536, 1, 1));
    inline for (8..17) |bits| {
        const win = comptime exl3.Window.fromBits(bits).?;
        var slot: ?mlx.mlx_fast_metal_kernel = null;
        const kernel = try getNamedKernel(&slot, comptime "exl3_mcg_half_pairs" ++ winSuffix(win), &.{"codes"}, &.{"result"}, source, comptime codebookHelpers(.mcg, win));
        defer _ = mlx.mlx_fast_metal_kernel_free(kernel);
        var outputs = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(outputs);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs, kernel, inputs, cfg, s));
        var result = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(result);
        try mlx.check(mlx.mlx_vector_array_get(&result, outputs, 0));
        try mlx.check(mlx.mlx_array_eval(result));
        const got = mlx.mlx_array_data_uint32(result) orelse return error.U32Unreadable;
        for (codes, 0..) |cw, i| {
            const lo = exl3.decodeCodeword(@as(u16, @intCast(cw)) & win.mask(), .mcg);
            const hi = exl3.decodeCodeword(@as(u16, @intCast(65535 - cw)) & win.mask(), .mcg);
            try t.expectEqual(@as(u32, lo) | (@as(u32, hi) << 16), got[i]);
        }
    }
}

test "exl3 n48 funnel readers preserve codewords and weights" {
    inline for ([_]exl3.Codebook{.mcg}) |cb| {
        inline for ([_]bool{ false, true }) |lane_reader| {
            try weightReaderExact(48, cb, .w12, true, if (lane_reader) .lane else .nax);
            inline for (8..17) |bits| try weightReaderExact(48, cb, comptime exl3.Window.fromBits(bits).?, false, if (lane_reader) .lane else .nax);
        }
    }
}

fn n48FunnelCase(cb: exl3.Codebook, rows: usize, seed: u64, decode: bool, prepared: ?bool) !void {
    prepared_mid_force = prepared;
    defer prepared_mid_force = null;
    try bf16TruthCase(.{ .e = 16, .hidden = 256, .inter = 128, .topk = 10, .rows = rows, .rate = .{ .n = 48 }, .dec = .{ .codebook = cb, .window = .w12 }, .seed = seed, .banks = MIMO_BANKS, .x_scale = 3 }, 32, decode);
}

test "exl3 the n48 funnel engages for every codebook" {
    for ([_]exl3.Codebook{ .mcg, .mul1 }) |cb| {
        funnel_engaged = @splat(false);
        try n48FunnelCase(cb, 4, 318, true, null);
        try std.testing.expect(funnel_engaged[@backingInt(FunnelArm.pair)]);
    }
}

test "exl3 the n42 funnel engages on the decode and NAX arms and keeps the f32-truth bar" {
    const w15: exl3.Decode = .{ .codebook = .mcg, .window = .w15 };
    for ([_]struct { rows: usize, decode: bool, arms: []const FunnelArm }{
        .{ .rows = 4, .decode = true, .arms = &.{ .pair, .fused_mid_down } },
        .{ .rows = 33, .decode = false, .arms = &.{.nax} },
    }) |k| {
        funnel_engaged = @splat(false);
        for (0..PARITY_SEEDS) |seed| {
            try bf16TruthCase(.{ .e = 16, .hidden = 256, .inter = 128, .topk = 10, .rows = k.rows, .rate = .{ .n = 42 }, .dec = w15, .seed = 318 + seed, .banks = MIMO_BANKS, .x_scale = 3 }, 32, k.decode);
        }
        for (k.arms) |arm| try std.testing.expect(funnel_engaged[@backingInt(arm)]);
    }
}

test "exl3 decode GEMV layout: the lane funnel carries two tiles, every other reader one" {
    const t = std.testing;
    defer setDecodeParams(.mul1);
    setDecodeParams(.{ .codebook = .mcg, .window = .w12 });
    try t.expectEqual(GemvLayout{ .funnel = true, .tiles = 2 }, gemvLayout(40, 128));
    try t.expectEqual(GemvLayout{ .funnel = true, .tiles = 2 }, gemvLayout(48, 40));
    try t.expectEqual(GemvLayout{ .funnel = true, .tiles = 1 }, gemvLayout(40, 45));
    try t.expectEqual(GemvLayout{ .funnel = false, .tiles = 1 }, gemvLayout(64, 128));
    for ([_]u32{ 32, 36, 42, 44, 62 }) |n| try t.expectEqual(GemvLayout{ .funnel = true, .tiles = 2 }, gemvLayout(n, 40));
    setDecodeParams(.{ .codebook = .mul1, .window = .w16 });
    try t.expectEqual(GemvLayout{ .funnel = true, .tiles = 2 }, gemvLayout(40, 128));
    try t.expectEqual(GemvLayout{ .funnel = true, .tiles = 2 }, gemvLayout(42, 40));
    try t.expectEqual(GemvLayout{ .funnel = true, .tiles = 2 }, gemvLayout(48, 40));
    funnel_off_for_test = true;
    defer funnel_off_for_test = false;
    try t.expectEqual(GemvLayout{ .funnel = false, .tiles = 1 }, gemvLayout(40, 128));
}

fn gemvOutBytes(a: mlx.mlx_array) ![]const u8 {
    try mlx.check(mlx.mlx_array_eval(a));
    const n = mlx.mlx_array_size(a);
    return switch (mlx.mlx_array_dtype(a)) {
        .float32 => std.mem.sliceAsBytes((mlx.mlx_array_data_float32(a) orelse return error.F32Unreadable)[0..n]),
        .float16 => std.mem.sliceAsBytes((mlx.mlx_array_data_float16(a) orelse return error.F16Unreadable)[0..n]),
        else => error.UnexpectedDtype,
    };
}

/// The lane funnel's two-tile layout against the generic window reader at one
/// tile per threadgroup: the pair planes and both downs must match to the bit.
/// Both downs read the reference pair planes, so a down mismatch is its own.
fn funnelLayoutBytesMatch(c: MimoMoeCase) !void {
    setDecodeParams(c.dec);
    defer setDecodeParams(.mul1);
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var f = try mimoMoeFixture(arena.allocator(), c);
    defer f.deinit();
    const a = f.arrays;
    const hidden: c_int = @intCast(c.hidden);
    const inter: c_int = @intCast(c.inter);
    const nslots: c_int = @intCast(c.rows * c.topk);
    var outs: [2][4]?mlx.mlx_array = @splat(@splat(null));
    defer for (outs) |arm| for (arm) |o| {
        if (o) |v| _ = mlx.mlx_array_free(v);
    };
    for (0..2) |arm| {
        funnel_off_for_test = arm == 0;
        defer funnel_off_for_test = false;
        funnel_engaged = @splat(false);
        outs[arm][0], outs[arm][1] = try pairGemv(s, a[8], a[3], a[3], a[0], a[1], a[7], hidden, inter, nslots, @intCast(c.topk), 0);
        outs[arm][2] = try downGemvFusedMid(s, outs[0][0].?, outs[0][1].?, a[2], a[4], a[4], a[5], a[7], inter, hidden, nslots);
        outs[arm][3] = try downGemvPreparedMid(s, outs[0][0].?, outs[0][1].?, a[2], a[4], a[4], a[5], a[7], inter, hidden, nslots, 0);
        const funnel = arm == 1 and c.rate.n != 64;
        for ([_]FunnelArm{ .pair, .fused_mid_down, .prepared_down }) |fa| try std.testing.expectEqual(funnel, funnel_engaged[@backingInt(fa)]);
    }
    for (outs[0], outs[1]) |want, got| try std.testing.expectEqualSlices(u8, try gemvOutBytes(want.?), try gemvOutBytes(got.?));
}

test "exl3 decode GEMV funnel layout is bit-identical to the one-tile reader" {
    const cases = [_]struct { n: u32, topk: usize, dec: exl3.Decode }{
        .{ .n = 40, .topk = 8, .dec = .{ .codebook = .mcg, .window = .w12 } },
        .{ .n = 40, .topk = 8, .dec = .{ .codebook = .mul1, .window = .w16 } },
        .{ .n = 48, .topk = 10, .dec = .{ .codebook = .mcg, .window = .w12 } },
        .{ .n = 48, .topk = 10, .dec = .{ .codebook = .mcg, .window = .w15 } },
        .{ .n = 42, .topk = 10, .dec = .{ .codebook = .mcg, .window = .w15 } },
        .{ .n = 42, .topk = 10, .dec = .{ .codebook = .mcg, .window = .w12 } },
    };
    for (cases) |k| {
        for ([_]usize{ 1, 2, 4, 8, 16 }) |rows| {
            try funnelLayoutBytesMatch(.{ .e = 16, .hidden = 1024, .inter = 512, .topk = k.topk, .rows = rows, .rate = .{ .n = k.n }, .dec = k.dec, .seed = 318 + rows, .banks = MIMO_BANKS, .x_scale = 3 });
        }
    }
    try funnelLayoutBytesMatch(.{ .e = 16, .hidden = 1024, .inter = 512, .topk = 10, .rows = 4, .rate = .{ .n = 42 }, .dec = .mul1, .seed = 518, .banks = MIMO_BANKS, .x_scale = 3 });
}

/// The MiMo decode chain at the served hidden and inter (prepared mid, grouped verify
/// rows) on the fast arms against the one-tile generic reader, to the bit.
fn mimoChainBytesMatch(c: MimoMoeCase) !void {
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    setDecodeParams(c.dec);
    defer setDecodeParams(.mul1);
    prepared_mid_force = true;
    defer prepared_mid_force = null;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var f = try mimoMoeFixture(alloc, c);
    defer f.deinit();
    const xb = try alloc.alloc(u16, f.xf.len);
    for (f.xf, xb) |v, *b| b.* = @truncate(@as(u32, @bitCast(v)) >> 16);
    _ = mlx.mlx_array_free(f.arrays[8]);
    f.arrays[8] = mlx.mlx_array_new_data(xb.ptr, &.{ @intCast(c.rows), @intCast(c.hidden) }, 2, .bfloat16);
    const ar = f.arrays;
    var ys: [2]mlx.mlx_array = undefined;
    var cs: [2]mlx.mlx_array = .{ mlx.mlx_array_new(), mlx.mlx_array_new() };
    defer for (cs) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (0..2) |arm| {
        funnel_off_for_test = arm == 0;
        defer funnel_off_for_test = false;
        funnel_engaged = @splat(false);
        grouped_engaged = false;
        ys[arm] = try moeSwigluFused(s, ar[8], ar[0], ar[3], ar[4], ar[1], ar[3], ar[4], ar[2], ar[5], ar[6], ar[7], ar[9], .bfloat16);
        const fast = arm == 1 and c.rate.n != 64;
        try std.testing.expectEqual(fast, funnel_engaged[@backingInt(FunnelArm.pair)]);
        try std.testing.expectEqual(fast, funnel_engaged[@backingInt(FunnelArm.prepared_down)]);
        try std.testing.expectEqual(fast and c.rows >= 2, grouped_engaged);
    }
    defer for (ys) |a| {
        _ = mlx.mlx_array_free(a);
    };
    try std.testing.expectEqualSlices(u8, try bf16Bytes(s, ys[0], &cs[0]), try bf16Bytes(s, ys[1], &cs[1]));
}

// Every rate the format admits takes the fast arms: a rate that falls back to the
// generic reader fails here. K4 (n64) keeps its own packed branch, one tile a threadgroup.
test "exl3 every admitted rate takes the fast arms, bit-identical to the generic reader" {
    const w12: exl3.Decode = .{ .codebook = .mcg, .window = .w12 };
    inline for (exl3.Rate.min_n / 2..exl3.Rate.max_n / 2 + 1) |half| {
        const n: u32 = 2 * half;
        if (n != 64) try weightReaderExact(n, .mcg, .w16, true, .lane);
        inline for (.{ .nax, .simdmat }) |reader| try weightReaderExact(n, .mcg, .w16, true, reader);
        for ([_]usize{ 1, 4 }) |rows| {
            try funnelLayoutBytesMatch(.{ .e = 16, .hidden = 1024, .inter = 512, .topk = 8, .rows = rows, .rate = .{ .n = n }, .dec = w12, .seed = 418 + n + rows, .banks = MIMO_BANKS, .x_scale = 3 });
        }
        try mimoChainBytesMatch(.{ .e = 8, .hidden = 4096, .inter = 2048, .topk = 8, .rows = 4, .rate = .{ .n = n }, .dec = w12, .seed = 618 + n, .banks = MIMO_BANKS, .x_scale = 3 });
        naxBodyBytesMatch(.{ .n = n }, w12, 256, 256, 818 + n) catch |e| if (e != error.SkipZigTest) return e;
        for ([_]bool{ false, true }) |fallback| {
            var env: FallbackEnv = .{};
            if (fallback) env.force();
            defer if (fallback) env.restore();
            funnel_engaged = @splat(false);
            try sortedGemmParity(.{ .n = n }, w12, 32, 718 + n);
            const arm: FunnelArm = if (gemmNaxOn()) .nax else .simdmat;
            try std.testing.expectEqual(n != 64, funnel_engaged[@backingInt(arm)]);
        }
    }
}

/// GLM's clamped decode chain at one (rate, window): both lane kernels engage and the bytes are
/// the staged chain's.
fn glmLaneChainBytesMatch(n: u32, dec: exl3.Decode, rows: usize, seed: u64) !void {
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    setDecodeParams(dec);
    defer setDecodeParams(.mul1);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var f = try mimoMoeFixture(arena.allocator(), .{ .hidden = 256, .inter = 256, .e = 8, .rows = rows, .topk = 8, .rate = .{ .n = n }, .dec = dec, .seed = seed, .banks = .{ 0.125, 0.25, 0.125, 0.25 }, .x_scale = 8 });
    defer f.deinit();
    const a = f.arrays;
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    try mlx.check(mlx.mlx_astype(&x, a[8], .bfloat16, s));
    const old = try clampedSortedReference(s, x, a[0], a[3], a[4], a[1], a[3], a[4], a[2], a[5], a[6], a[7], a[9], 8, 10, .bfloat16);
    defer _ = mlx.mlx_array_free(old);
    const lanes = lane_decode_dispatches;
    const got = try moeSwigluClamped(s, x, a[0], a[3], a[4], a[1], a[3], a[4], a[2], a[5], a[6], a[7], a[9], 8, 10, .bfloat16);
    defer _ = mlx.mlx_array_free(got);
    try std.testing.expectEqual(lanes + 2, lane_decode_dispatches);
    var cs: [2]mlx.mlx_array = .{ mlx.mlx_array_new(), mlx.mlx_array_new() };
    defer for (cs) |c| {
        _ = mlx.mlx_array_free(c);
    };
    try std.testing.expectEqualSlices(u8, try bf16Bytes(s, old, &cs[0]), try bf16Bytes(s, got, &cs[1]));
}

// A window is one mask in the decode helper every arm inlines, so each low rate is held to
// the generic reader's bytes on every arm at every window the format admits.
test "exl3 every K1.5 to K2 rate takes the fast arms at every window, bit-identical to the generic reader" {
    inline for (12..17) |half| {
        const n: u32 = 2 * half;
        inline for (exl3.Window.min_bits..17) |bits| {
            const win = comptime exl3.Window.fromBits(bits).?;
            const dec: exl3.Decode = .{ .codebook = .mcg, .window = win };
            const seed: u64 = 100 * n + bits;
            inline for (.{ .lane, .nax, .simdmat }) |reader| try weightReaderExact(n, .mcg, win, false, reader);
            for ([_]usize{ 1, 4 }) |rows| {
                try funnelLayoutBytesMatch(.{ .e = 16, .hidden = 1024, .inter = 512, .topk = 8, .rows = rows, .rate = .{ .n = n }, .dec = dec, .seed = seed + rows, .banks = MIMO_BANKS, .x_scale = 3 });
                try glmLaneChainBytesMatch(n, dec, rows, seed + rows);
            }
            try mimoChainBytesMatch(.{ .e = 8, .hidden = 4096, .inter = 2048, .topk = 8, .rows = 4, .rate = .{ .n = n }, .dec = dec, .seed = seed, .banks = MIMO_BANKS, .x_scale = 3 });
            naxBodyBytesMatch(.{ .n = n }, dec, 256, 256, seed) catch |e| if (e != error.SkipZigTest) return e;
            for ([_]bool{ false, true }) |fallback| {
                var env: FallbackEnv = .{};
                if (fallback) env.force();
                defer if (fallback) env.restore();
                funnel_engaged = @splat(false);
                try sortedGemmParity(.{ .n = n }, dec, 32, seed);
                try std.testing.expect(funnel_engaged[@backingInt(if (gemmNaxOn()) FunnelArm.nax else FunnelArm.simdmat)]);
            }
        }
    }
}

fn bf16Bytes(s: mlx.mlx_stream, a: mlx.mlx_array, out: *mlx.mlx_array) ![]const u8 {
    try mlx.check(mlx.mlx_contiguous(out, a, false, s));
    try mlx.check(mlx.mlx_array_eval(out.*));
    const p = mlx.mlx_array_data_bfloat16(out.*) orelse return error.Bf16Unreadable;
    return std.mem.sliceAsBytes(p[0..mlx.mlx_array_size(out.*)]);
}

/// A MiMo verify block against each row's own one-row decode tick, on a bank so small that
/// the rows share most experts: every row's bytes must be its tick's (greedy MTP == serial).
fn verifyRowsMatchDecodeTicks(c: MimoMoeCase) !void {
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    setDecodeParams(c.dec);
    defer setDecodeParams(.mul1);
    prepared_mid_force = true;
    defer prepared_mid_force = null;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var f = try mimoMoeFixture(alloc, c);
    defer f.deinit();
    const xb = try alloc.alloc(u16, f.xf.len);
    for (f.xf, xb) |v, *b| b.* = @truncate(@as(u32, @bitCast(v)) >> 16);
    _ = mlx.mlx_array_free(f.arrays[8]);
    f.arrays[8] = mlx.mlx_array_new_data(xb.ptr, &.{ @intCast(c.rows), @intCast(c.hidden) }, 2, .bfloat16);
    const ar = f.arrays;
    grouped_engaged = false;
    const block = try moeSwigluFused(s, ar[8], ar[0], ar[3], ar[4], ar[1], ar[3], ar[4], ar[2], ar[5], ar[6], ar[7], ar[9], .bfloat16);
    defer _ = mlx.mlx_array_free(block);
    try std.testing.expect(grouped_engaged);
    var block_c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(block_c);
    const want = try bf16Bytes(s, block, &block_c);
    const row_bytes = c.hidden * 2;
    const k: c_int = @intCast(c.topk);
    for (0..c.rows) |ri| {
        const r: c_int = @intCast(ri);
        var x_r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x_r);
        try mlx.check(mlx.mlx_slice(&x_r, ar[8], &.{ r, 0 }, 2, &.{ r + 1, @intCast(c.hidden) }, 2, &.{ 1, 1 }, 2, s));
        var slots_r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(slots_r);
        try mlx.check(mlx.mlx_slice(&slots_r, ar[7], &.{r * k}, 1, &.{(r + 1) * k}, 1, &.{1}, 1, s));
        var scores_r = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(scores_r);
        try mlx.check(mlx.mlx_slice(&scores_r, ar[9], &.{r * k}, 1, &.{(r + 1) * k}, 1, &.{1}, 1, s));
        const tick = try moeSwigluFused(s, x_r, ar[0], ar[3], ar[4], ar[1], ar[3], ar[4], ar[2], ar[5], ar[6], slots_r, scores_r, .bfloat16);
        defer _ = mlx.mlx_array_free(tick);
        var tick_c = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(tick_c);
        try std.testing.expectEqualSlices(u8, try bf16Bytes(s, tick, &tick_c), want[ri * row_bytes ..][0..row_bytes]);
    }
}

test "exl3 MiMo verify rows sharing experts are bit-identical to their one-row decode ticks" {
    for ([_]usize{ 2, 3, 4 }) |rows| {
        for (0..PARITY_SEEDS) |seed| {
            try verifyRowsMatchDecodeTicks(.{ .e = 16, .hidden = 1024, .inter = 512, .topk = 8, .rows = rows, .rate = .{ .n = 40 }, .dec = .{ .codebook = .mcg, .window = .w12 }, .seed = 318 + seed, .banks = MIMO_BANKS, .x_scale = 3 });
        }
    }
}

test "exl3 MCG n48 BF16 decode and NAX preserve f32 truth across seeds" {
    for ([_]bool{ false, true }) |prepared| {
        for (1..9) |rows| {
            for (0..PARITY_SEEDS) |seed| try n48FunnelCase(.mcg, rows, 318 + seed, true, prepared);
        }
    }
    for (0..PARITY_SEEDS) |seed| try n48FunnelCase(.mcg, 33, 318 + seed, false, null);
}

test "exl3 MCG K2.5 BF16 decode and NAX preserve f32 truth across seeds" {
    for ([_]bool{ false, true }) |prepared| {
        prepared_mid_force = prepared;
        defer prepared_mid_force = null;
        for (1..9) |rows| {
            for (0..PARITY_SEEDS) |seed| {
                try bf16TruthCase(.{ .e = 16, .hidden = 256, .inter = 128, .topk = 8, .rows = rows, .rate = .{ .n = 40 }, .dec = .{ .codebook = .mcg, .window = .w12 }, .seed = 318 + seed, .banks = MIMO_BANKS, .x_scale = 3 }, 32, true);
            }
        }
    }
    for (0..PARITY_SEEDS) |seed| {
        try bf16TruthCase(.{ .e = 16, .hidden = 256, .inter = 128, .topk = 8, .rows = 33, .rate = .{ .n = 40 }, .dec = .{ .codebook = .mcg, .window = .w12 }, .seed = 318 + seed, .banks = MIMO_BANKS, .x_scale = 3 }, 32, false);
    }
}

test "exl3 Sushi CPU dispatch selects fast layouts for every rate" {
    for (8..65) |half| {
        const n: u32 = @intCast(half * 2);
        try std.testing.expectEqual(n != 64, funnelReads(n));
        try std.testing.expectEqual(GemvLayout{ .funnel = n != 64, .tiles = if (n == 64) 1 else 2 }, gemvLayout(n, 128));
        try std.testing.expectEqual(n, (try packedRate(@intCast(n))).n);
    }
}

/// A copy of `a` that waits on `dep`, so the timed GEMV steps run one after another.
const LANE_LINK_SOURCE: [:0]const u8 =
    \\const uint i = thread_position_in_grid.x;
    \\if (i >= uint(LEN)) return;
    \\out[i] = a[i];
    \\if (i == 0u && float(dep[0]) == 1.5e30f) out[0] = a[1];
;
var lane_link_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// Nanoseconds for `steps` dependent pair + fused-down GEMV steps over one-row fixture `f`.
fn laneChainNs(s: mlx.mlx_stream, f: *const MimoMoeFixture, c: MimoMoeCase, steps: usize) !u64 {
    const a = f.arrays;
    const hidden: c_int = @intCast(c.hidden);
    const inter: c_int = @intCast(c.inter);
    const nslots: c_int = @intCast(c.topk);
    const link = try getNamedKernel(&lane_link_kernel, "sushi_exl3_lane_link", &.{ "a", "dep" }, &.{"out"}, LANE_LINK_SOURCE, "");
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{ 1, hidden }, 2, .float16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, hidden, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "LEN", hidden));
    var x = try applyUnary(s, link, &.{ a[8], a[8] }, cfg);
    defer _ = mlx.mlx_array_free(x);
    for (0..steps) |_| {
        const inner = try pairGemv(s, x, a[3], a[3], a[0], a[1], a[7], hidden, inter, nslots, @intCast(c.topk), 0);
        defer _ = mlx.mlx_array_free(inner[0]);
        defer _ = mlx.mlx_array_free(inner[1]);
        const y = try downGemvFusedMid(s, inner[0], inner[1], a[2], a[4], a[4], a[5], a[7], inter, hidden, nslots);
        defer _ = mlx.mlx_array_free(y);
        const next = try applyUnary(s, link, &.{ a[8], y }, cfg);
        _ = mlx.mlx_array_free(x);
        x = next;
    }
    const io = std.Io.Threaded.global_single_threaded.io();
    var sw = io_util.Stopwatch.init(io);
    try mlx.check(mlx.mlx_array_eval(x));
    return sw.read();
}

// A reader change can keep every byte and still slow a rate (n42 to n62 but n48 read a third word
// per decode lane), so every K1.5 to K4 rate's decode GEMVs are held to a margin over n48's.
test "exl3 every K1.5 to K4 rate decodes within a margin of n48" {
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const dec: exl3.Decode = .{ .codebook = .mcg, .window = .w15 };
    setDecodeParams(dec);
    defer setDecodeParams(.mul1);
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const ref_case: MimoMoeCase = .{ .e = 16, .hidden = 2560, .inter = 640, .topk = 10, .rows = 1, .rate = .{ .n = 48 }, .dec = dec, .seed = 4848 };
    var ref = try mimoMoeFixture(arena.allocator(), ref_case);
    defer ref.deinit();
    var n: u32 = 24;
    while (n <= 64) : (n += 2) {
        if (n == 48) continue;
        var c = ref_case;
        c.rate = .{ .n = n };
        c.seed = 4800 + n;
        var rate_arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer rate_arena.deinit();
        var rate = try mimoMoeFixture(rate_arena.allocator(), c);
        defer rate.deinit();
        // Another GPU job only adds time, so each arm keeps its fastest round. Arms 0 and 1 time the
        // rate, arms 2 and 3 the n48 kernel: when a pair disagrees by more than 10% the box is too
        // busy to read a ratio, and after three tries the test skips instead of failing on contention.
        var attempt: u32 = 0;
        const ratio: f64 = while (attempt < 3) : (attempt += 1) {
            var best: [4]u64 = @splat(std.math.maxInt(u64));
            for (0..9) |round| {
                for (0..4) |j| {
                    const k = (round + j) % 4;
                    const ns = try laneChainNs(s, if (k < 2) &rate else &ref, if (k < 2) c else ref_case, 48);
                    best[k] = @min(best[k], ns);
                }
            }
            if (@max(best[0], best[1]) * 100 > @min(best[0], best[1]) * 110) continue;
            if (@max(best[2], best[3]) * 100 > @min(best[2], best[3]) * 110) continue;
            break @as(f64, @floatFromInt(@min(best[0], best[1]))) / @as(f64, @floatFromInt(@min(best[2], best[3])));
        } else return error.SkipZigTest;
        // n64's packed branch carries one output tile per threadgroup where the funnel carries two.
        const limit: f64 = if (n == 64) 1.8 else 1.4;
        benchPrint("[exl3-lane-margin] n{d} over n48: {d:.3}\n", .{ n, ratio });
        try std.testing.expect(ratio < limit);
    }
}

// Decode GEMV steps and prefill GEMMs of each low rate over n32's, interleaved, at Flash-Next
// geometry. Prints only under SUSHI_EXL3_LOWK_UBENCH (a diagnostic, never a test).
test "exl3 low-rate decode and prefill timing over n32 (SUSHI_EXL3_LOWK_UBENCH=1)" {
    if (!diagEnvValueOn(std.c.getenv("SUSHI_EXL3_LOWK_UBENCH"))) return error.SkipZigTest;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    defer setDecodeParams(.mul1);
    const rates = [_]u32{ 24, 26, 28, 30, 34, 36, 40, 48 };
    const io = std.Io.Threaded.global_single_threaded.io();
    for ([_]exl3.Window{ .w8, .w12, .w15 }) |win| {
        const dec: exl3.Decode = .{ .codebook = .mcg, .window = win };
        setDecodeParams(dec);
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        const ref_case: MimoMoeCase = .{ .e = 16, .hidden = 2560, .inter = 640, .topk = 10, .rows = 1, .rate = .{ .n = 32 }, .dec = dec, .seed = 3232 };
        var ref = try mimoMoeFixture(arena.allocator(), ref_case);
        defer ref.deinit();
        for (rates) |n| {
            var c = ref_case;
            c.rate = .{ .n = n };
            c.seed = 3200 + n;
            var rate_arena = std.heap.ArenaAllocator.init(std.testing.allocator);
            defer rate_arena.deinit();
            var rate = try mimoMoeFixture(rate_arena.allocator(), c);
            defer rate.deinit();
            var ratios: [9]f64 = undefined;
            var step_us: [9]f64 = undefined;
            for (&ratios, &step_us, 0..) |*ratio, *us, round| {
                var ns: [2]u64 = undefined;
                for (0..2) |j| {
                    const k = (round + j) % 2;
                    ns[k] = try laneChainNs(s, if (k == 0) &rate else &ref, if (k == 0) c else ref_case, 48);
                }
                ratio.* = @as(f64, @floatFromInt(ns[0])) / @as(f64, @floatFromInt(ns[1]));
                us.* = @as(f64, @floatFromInt(ns[0])) / 48e3;
            }
            std.mem.sort(f64, &ratios, {}, std.sort.asc(f64));
            std.mem.sort(f64, &step_us, {}, std.sort.asc(f64));
            std.debug.print("[lowk-decode] w{d} n{d} over n32: {d:.3} ({d:.1} us per step)\n", .{ win.bits(), n, ratios[ratios.len / 2], step_us[step_us.len / 2] });
        }
        if (!gemmNaxOn()) continue;
        const E: c_int = 128;
        const run: usize = 40;
        const nslots: usize = @as(usize, @intCast(E)) * run;
        const alloc = arena.allocator();
        const ids = try alloc.alloc(u32, nslots);
        for (ids, 0..) |*v, i| v.* = @intCast(i / run);
        const eids = mlx.mlx_array_new_data(ids.ptr, &[_]c_int{@intCast(nslots)}, 1, .uint32);
        defer _ = mlx.mlx_array_free(eids);
        var prng = std.Random.DefaultPrng.init(0x10c);
        const rnd = prng.random();
        for ([_][2]c_int{ .{ 2560, 640 }, .{ 640, 2560 } }) |shape| {
            const xh = try alloc.alloc(u16, nslots * @as(usize, @intCast(shape[0])));
            for (xh) |*v| v.* = exl3.f32ToF16Bits((rnd.float(f32) - 0.5) * 0.2);
            const x = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(nslots), shape[0] }, 2, .float16);
            defer _ = mlx.mlx_array_free(x);
            const tab = try gemmWindowTable(s, eids, @intCast(nslots), 32, true);
            defer _ = mlx.mlx_array_free(tab.starts);
            defer _ = mlx.mlx_array_free(tab.nlives);
            const it = @divExact(shape[0], 16);
            const ot = @divExact(shape[1], 16);
            var banks: [rates.len + 1]mlx.mlx_array = undefined;
            const all_n = [_]u32{32} ++ rates;
            for (&banks, all_n) |*b, n| {
                const tr_h = try alloc.alloc(u16, @intCast(E * it * ot * @as(c_int, @intCast(n))));
                for (tr_h) |*v| v.* = @truncate(rnd.int(u32));
                b.* = mlx.mlx_array_new_data(tr_h.ptr, &[_]c_int{ E, it, ot, @intCast(n) }, 4, .uint16);
            }
            defer for (banks) |b| {
                _ = mlx.mlx_array_free(b);
            };
            const ROUNDS = 7;
            var t: [rates.len + 1][ROUNDS]u64 = undefined;
            for (0..ROUNDS + 1) |r| {
                for (banks, 0..) |b, bi| {
                    var sw = io_util.Stopwatch.init(io);
                    const got = try innerGemmSortedTable(s, x, b, eids, 32, true, tab);
                    try mlx.check(mlx.mlx_array_eval(got));
                    if (r > 0) t[bi][r - 1] = sw.read();
                    _ = mlx.mlx_array_free(got);
                }
            }
            for (&t) |*row| std.mem.sort(u64, row, {}, std.sort.asc(u64));
            for (rates, 1..) |n, bi| {
                const ratio = @as(f64, @floatFromInt(t[bi][ROUNDS / 2])) / @as(f64, @floatFromInt(t[0][ROUNDS / 2]));
                std.debug.print("[lowk-prefill] w{d} {d}->{d} n{d} over n32: {d:.3} (n32 {d:.1} us)\n", .{ win.bits(), shape[0], shape[1], n, ratio, @as(f64, @floatFromInt(t[0][ROUNDS / 2])) / 1000.0 });
            }
        }
    }
}

test "exl3 Sushi GPU decode and prefill match scalar truth at K1 K1.5 K3 K5 K8" {
    if (!mlx.streamIsGpu(mlx.gpuStream())) return error.SkipZigTest;
    defer prepared_mid_force = null;
    defer gemm_simdmat_force = null;
    defer mimo_prefill_force = null;
    for ([_]u32{ 16, 24, 48, 80, 128 }) |n| {
        for ([_]exl3.Codebook{ .mul1, .mcg }) |cb| {
            const dec: exl3.Decode = .{ .codebook = cb, .window = .w12 };
            for (0..3) |seed| {
                try indexedParity(.{ .n = n }, 256, 128, 8, 4, 913 + seed, dec);
                for ([_]bool{ false, true }) |prepared| {
                    prepared_mid_force = prepared;
                    for ([_]usize{ 1, 4 }) |rows| try bf16TruthCaseAnyArm(.{ .e = 8, .hidden = 256, .inter = 128, .topk = 4, .rows = rows, .rate = .{ .n = n }, .dec = dec, .seed = 913 + seed, .banks = MIMO_BANKS, .x_scale = 3 }, 32, true);
                }
                prepared_mid_force = null;
                for (0..3) |arm| {
                    var env: FallbackEnv = .{};
                    if (arm != 0) env.force();
                    defer if (arm != 0) env.restore();
                    gemm_simdmat_force = arm != 2;
                    funnel_engaged = @splat(false);
                    const expect_nax = arm == 0 and gemmNaxAvailable();
                    try sortedGemmParity(.{ .n = n }, dec, 32, 913 + seed);
                    try std.testing.expectEqual(expect_nax, funnel_engaged[@backingInt(FunnelArm.nax)]);
                    try std.testing.expectEqual(!expect_nax and arm != 2, funnel_engaged[@backingInt(FunnelArm.simdmat)]);
                    for ([_]bool{ false, true }) |metadata| {
                        mimo_prefill_force = metadata;
                        try bf16TruthCaseAnyArm(.{ .e = 8, .hidden = 256, .inter = 128, .topk = 4, .rows = 33, .rate = .{ .n = n }, .dec = dec, .seed = 913 + seed, .banks = MIMO_BANKS, .x_scale = 3 }, 32, false);
                    }
                }
            }
        }
    }
}

test "MiMo EXL3 streaming CPU slab capacities preserve resident kernel arms" {
    const t = std.testing;
    for (1..257) |slots| {
        try t.expect(mimoPrefillOn(4096, 2048, @intCast(slots), 8));
        for (1..9) |rows| try t.expect(preparedMidOn(4096, 2048, @intCast(slots), 8, @intCast(rows), .bfloat16));
    }
    try t.expect(!mimoPrefillOn(4096, 2048, 0, 8));
    try t.expect(!mimoPrefillOn(4096, 2048, 257, 8));
    try t.expect(!preparedMidOn(4096, 2048, 0, 8, 1, .bfloat16));
    try t.expect(!preparedMidOn(4096, 2048, 257, 8, 1, .bfloat16));
    try t.expect(!preparedMidOn(4096, 2048, 16, 8, 9, .bfloat16));
    try t.expect(!preparedMidOn(4096, 2048, 16, 8, 1, .float16));
    try t.expect(mimoPrefillOn(2560, 640, 16, 10));
    try t.expect(!preparedMidOn(2560, 640, 16, 10, 1, .bfloat16));
}

/// The pair GEMV on a prepared input (one prepare dispatch per slot and projection) against
/// the pair GEMV that prepares its own K span per threadgroup: both planes to the bit.
fn pairPreparedBytesMatch(c: MimoMoeCase, group: c_int) !void {
    setDecodeParams(c.dec);
    defer setDecodeParams(.mul1);
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var f = try mimoMoeFixture(arena.allocator(), c);
    defer f.deinit();
    const a = f.arrays;
    const hidden: c_int = @intCast(c.hidden);
    const inter: c_int = @intCast(c.inter);
    const nslots: c_int = @intCast(c.rows * c.topk);
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    try mlx.check(mlx.mlx_astype(&x, a[8], .bfloat16, s));
    const want = try pairGemv(s, x, a[3], a[6], a[0], a[1], a[7], hidden, inter, nslots, @intCast(c.topk), group);
    defer _ = mlx.mlx_array_free(want[0]);
    defer _ = mlx.mlx_array_free(want[1]);
    const prep = try pairPrepare(s, x, a[3], a[6], a[7], hidden, nslots, @intCast(c.topk));
    defer _ = mlx.mlx_array_free(prep);
    const got = try pairGemvPrepared(s, prep, a[0], a[1], a[7], hidden, inter, nslots, @intCast(c.topk), group);
    defer _ = mlx.mlx_array_free(got[0]);
    defer _ = mlx.mlx_array_free(got[1]);
    inline for (.{ want[0], want[1] }, .{ got[0], got[1] }) |w, g| std.testing.expectEqualSlices(u8, try gemvOutBytes(w), try gemvOutBytes(g)) catch |err| {
        std.debug.print("pair prepared n={d} rows={d} group={d}\n", .{ c.rate.n, c.rows, group });
        return err;
    };
}

test "exl3 pair GEMV on a prepared input keeps the self-preparing kernel's bytes (MiMo rates, grouped rows, K4)" {
    for ([_]u32{ 36, 40, 64 }) |n| {
        for ([_]usize{ 1, 2, 3, 4, 8 }) |rows| {
            for ([_]c_int{ 0, DECODE_GROUP_MEMBERS }) |group| {
                if (rows == 1 and group != 0) continue;
                try pairPreparedBytesMatch(.{ .e = 16, .hidden = 1024, .inter = 512, .topk = 8, .rows = rows, .rate = .{ .n = n }, .dec = .{ .codebook = .mcg, .window = .w12 }, .seed = 610 + rows, .banks = MIMO_BANKS, .x_scale = 3 }, group);
            }
        }
    }
    try pairPreparedBytesMatch(.{ .e = 32, .hidden = 4096, .inter = 2048, .topk = 8, .rows = 4, .rate = .{ .n = 36 }, .dec = .{ .codebook = .mcg, .window = .w12 }, .seed = 640, .banks = MIMO_BANKS, .x_scale = 3 }, DECODE_GROUP_MEMBERS);
}

/// The down GEMV on a middle prepared once per slot against the down that prepares its own middle
/// in threadgroup memory, from the same pair planes: the outputs to the bit.
fn downPreparedBytesMatch(c: MimoMoeCase, group: c_int) !void {
    setDecodeParams(c.dec);
    defer setDecodeParams(.mul1);
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var f = try mimoMoeFixture(arena.allocator(), c);
    defer f.deinit();
    const a = f.arrays;
    const hidden: c_int = @intCast(c.hidden);
    const inter: c_int = @intCast(c.inter);
    const nslots: c_int = @intCast(c.rows * c.topk);
    const planes = try pairGemv(s, a[8], a[3], a[6], a[0], a[1], a[7], hidden, inter, nslots, @intCast(c.topk), 0);
    defer _ = mlx.mlx_array_free(planes[0]);
    defer _ = mlx.mlx_array_free(planes[1]);
    const want = try downGemvFusedMid(s, planes[0], planes[1], a[2], a[4], a[5], a[5], a[7], inter, hidden, nslots);
    defer _ = mlx.mlx_array_free(want);
    const got = try downGemvPreparedMid(s, planes[0], planes[1], a[2], a[4], a[5], a[5], a[7], inter, hidden, nslots, group);
    defer _ = mlx.mlx_array_free(got);
    std.testing.expectEqualSlices(u8, try gemvOutBytes(want), try gemvOutBytes(got)) catch |err| {
        std.debug.print("down prepared n={d} rows={d} group={d}\n", .{ c.rate.n, c.rows, group });
        return err;
    };
}

test "exl3 down GEMV on a prepared middle keeps the self-preparing kernel's bytes (MiMo rates, grouped rows, K4)" {
    for ([_]u32{ 36, 40, 64 }) |n| {
        for ([_]usize{ 1, 2, 3, 4, 8 }) |rows| {
            for ([_]c_int{ 0, DECODE_GROUP_MEMBERS }) |group| {
                if (rows == 1 and group != 0) continue;
                try downPreparedBytesMatch(.{ .e = 16, .hidden = 1024, .inter = 512, .topk = 8, .rows = rows, .rate = .{ .n = n }, .dec = .{ .codebook = .mcg, .window = .w12 }, .seed = 710 + rows, .banks = MIMO_BANKS, .x_scale = 3 }, group);
            }
        }
    }
    try downPreparedBytesMatch(.{ .e = 32, .hidden = 4096, .inter = 2048, .topk = 8, .rows = 4, .rate = .{ .n = 36 }, .dec = .{ .codebook = .mcg, .window = .w12 }, .seed = 740, .banks = MIMO_BANKS, .x_scale = 3 }, DECODE_GROUP_MEMBERS);
}

/// Per-expert row counts of one layer: `SUSHI_EXL3_GEMM_COUNTS=<file>` (one layer per line,
/// 256 comma-separated counts), else a skewed synthetic layer summing to 16200.
fn gemmArmCounts(alloc: std.mem.Allocator, out: *std.ArrayList([256]u32)) !void {
    if (std.c.getenv("SUSHI_EXL3_GEMM_COUNTS")) |path| {
        const io = std.Io.Threaded.global_single_threaded.io();
        const text = try std.Io.Dir.cwd().readFileAlloc(io, std.mem.sliceTo(path, 0), alloc, .limited(1 << 20));
        var lines = std.mem.tokenizeScalar(u8, text, '\n');
        while (lines.next()) |line| {
            const at = std.mem.indexOf(u8, line, "] ") orelse continue;
            var c: [256]u32 = @splat(0);
            var it = std.mem.tokenizeScalar(u8, line[at + 2 ..], ',');
            var i: usize = 0;
            while (it.next()) |v| : (i += 1) {
                if (i == 256) break;
                c[i] = std.fmt.parseInt(u32, std.mem.trim(u8, v, " \r"), 10) catch 0;
            }
            try out.append(alloc, c);
        }
        return;
    }
    try out.append(alloc, syntheticArmCounts());
}

/// One synthetic layer of 16200 routed slots, skewed toward low expert ids.
fn syntheticArmCounts() [256]u32 {
    var c: [256]u32 = @splat(0);
    var total: u32 = 0;
    for (0..256) |e| {
        c[e] = @intFromFloat(400.0 * @exp(-@as(f64, @floatFromInt(e)) / 40.0));
        total += c[e];
    }
    c[0] += 16200 -| total;
    return c;
}

test "exl3 the GEMM ubench's synthetic layer routes exactly 16200 slots" {
    var sum: u64 = 0;
    for (syntheticArmCounts()) |c| sum += c;
    try std.testing.expectEqual(@as(u64, 16200), sum);
}

test "exl3 MiMo NAX GEMM served vs reference body at the served geometry, interleaved (SUSHI_EXL3_GEMM_ARMS=1)" {
    if (!diagEnvValueOn(std.c.getenv("SUSHI_EXL3_GEMM_ARMS"))) return error.SkipZigTest;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s) or !gemmNaxOn()) return error.SkipZigTest;
    setDecodeParams(.{ .codebook = .mcg, .window = .w12 });
    defer setDecodeParams(.mul1);
    defer nax_reference_override = false;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var layers: std.ArrayList([256]u32) = .empty;
    try gemmArmCounts(alloc, &layers);
    const pick = [_]usize{ 0, 10, 23, 35, 46 };
    const E: c_int = 256;
    const NHW: c_int = 36;
    const io = std.Io.Threaded.global_single_threaded.io();
    var prng = std.Random.DefaultPrng.init(0x6E33);
    const rnd = prng.random();
    const n_arms = 2;
    const names = [_][]const u8{ "reference", "served" };
    for ([_][2]c_int{ .{ 4096, 2048 }, .{ 2048, 4096 } }) |shape| {
        const it = @divExact(shape[0], 16);
        const ot = @divExact(shape[1], 16);
        const tr_h = try alloc.alloc(u16, @intCast(E * it * ot * NHW));
        for (tr_h) |*v| v.* = @truncate(rnd.int(u32));
        const tr = mlx.mlx_array_new_data(tr_h.ptr, &[_]c_int{ E, it, ot, NHW }, 4, .uint16);
        defer _ = mlx.mlx_array_free(tr);
        for (pick) |li| {
            if (li >= layers.items.len) continue;
            const counts = layers.items[li];
            var n: usize = 0;
            for (counts) |c| n += c;
            const ids = try alloc.alloc(u32, n);
            const ord = try alloc.alloc(u32, n);
            var at: usize = 0;
            for (counts, 0..) |c, e| {
                for (0..c) |_| {
                    ids[at] = @intCast(e);
                    ord[at] = @intCast(at);
                    at += 1;
                }
            }
            const eids = mlx.mlx_array_new_data(ids.ptr, &[_]c_int{@intCast(n)}, 1, .uint32);
            defer _ = mlx.mlx_array_free(eids);
            const order = mlx.mlx_array_new_data(ord.ptr, &[_]c_int{@intCast(n)}, 1, .uint32);
            defer _ = mlx.mlx_array_free(order);
            const xh = try alloc.alloc(u16, n * @as(usize, @intCast(shape[0])));
            for (xh) |*v| v.* = exl3.f32ToF16Bits((rnd.float(f32) - 0.5) * 0.2);
            const x = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(n), shape[0] }, 2, .float16);
            defer _ = mlx.mlx_array_free(x);
            const meta = try buildMimoWindowTable(s, eids, order, @intCast(n), 32, E);
            defer _ = mlx.mlx_array_free(meta.table.starts);
            defer _ = mlx.mlx_array_free(meta.table.nlives);
            defer _ = mlx.mlx_array_free(meta.inverse);
            try mlx.check(mlx.mlx_array_eval(meta.table.starts));
            try mlx.check(mlx.mlx_array_eval(meta.table.nlives));
            var ref: []const u16 = &.{};
            var same: [n_arms]bool = @splat(true);
            for (0..n_arms) |arm| {
                nax_reference_override = arm == 0;
                const got = try innerGemmSortedTable(s, x, tr, eids, 32, true, meta.table);
                defer _ = mlx.mlx_array_free(got);
                try mlx.check(mlx.mlx_array_eval(got));
                const bits: [*]const u16 = @ptrCast(mlx.mlx_array_data_float16(got) orelse return error.F16Unreadable);
                const view = bits[0 .. n * @as(usize, @intCast(shape[1]))];
                if (arm == 0) ref = try alloc.dupe(u16, view) else same[arm] = std.mem.eql(u16, ref, view);
            }
            const ROUNDS = 6;
            var t: [n_arms][2 * ROUNDS]u64 = undefined;
            for (0..ROUNDS) |r| {
                for (0..2 * n_arms) |k| {
                    const arm = if (k < n_arms) k else 2 * n_arms - 1 - k;
                    nax_reference_override = arm == 0;
                    var sw = io_util.Stopwatch.init(io);
                    const got = try innerGemmSortedTable(s, x, tr, eids, 32, true, meta.table);
                    try mlx.check(mlx.mlx_array_eval(got));
                    t[arm][2 * r + @intFromBool(k >= n_arms)] = sw.read();
                    _ = mlx.mlx_array_free(got);
                }
            }
            var med: [n_arms]f64 = undefined;
            for (0..n_arms) |arm| {
                std.mem.sort(u64, &t[arm], {}, std.sort.asc(u64));
                med[arm] = @as(f64, @floatFromInt(t[arm][ROUNDS])) / 1000.0;
            }
            for (0..n_arms) |arm| {
                std.debug.print("[gemm-arms] {d}->{d} layer{d} n={d} {s:<12} {d:>8.1} us  x{d:.3}  identical={}\n", .{
                    shape[0], shape[1], li + 1, n, names[arm], med[arm], med[arm] / med[0], same[arm],
                });
            }
        }
    }
}

/// The served NAX body against GEMM_NAX_REFERENCE_SOURCE on one bank: runs cover a lone row, a
/// half block, both blocks, several windows and the input's last row, on both window tables.
fn naxBodyBytesMatch(rate: exl3.Rate, dec: exl3.Decode, in_dim: c_int, out_dim: c_int, seed: u64) !void {
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s) or !gemmNaxOn()) return error.SkipZigTest;
    setDecodeParams(dec);
    defer setDecodeParams(.mul1);
    defer nax_reference_override = false;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    const runs = [_]u32{ 17, 1, 70, 16, 33, 7, 32, 48, 31 };
    const E: c_int = runs.len;
    const it = @divExact(in_dim, 16);
    const ot = @divExact(out_dim, 16);
    const nhw: c_int = @intCast(rate.n);
    const tr_h = try alloc.alloc(u16, @intCast(E * it * ot * nhw));
    for (tr_h) |*v| v.* = @truncate(rnd.int(u32));
    const tr = mlx.mlx_array_new_data(tr_h.ptr, &[_]c_int{ E, it, ot, nhw }, 4, .uint16);
    defer _ = mlx.mlx_array_free(tr);
    var nslots: usize = 0;
    for (runs) |r| nslots += r;
    const ids = try alloc.alloc(u32, nslots);
    var at: usize = 0;
    for (runs, 0..) |r, e| for (0..r) |_| {
        ids[at] = @intCast(e);
        at += 1;
    };
    const eids = mlx.mlx_array_new_data(ids.ptr, &[_]c_int{@intCast(nslots)}, 1, .uint32);
    defer _ = mlx.mlx_array_free(eids);
    const xh = try alloc.alloc(u16, nslots * @as(usize, @intCast(in_dim)));
    for (xh) |*v| v.* = exl3.f32ToF16Bits((rnd.float(f32) - 0.5) * 6.0);
    const x = mlx.mlx_array_new_data(xh.ptr, &[_]c_int{ @intCast(nslots), in_dim }, 2, .float16);
    defer _ = mlx.mlx_array_free(x);
    for ([_]bool{ true, false }) |aligned| {
        const tab = try gemmWindowTable(s, eids, @intCast(nslots), 32, aligned);
        defer _ = mlx.mlx_array_free(tab.starts);
        defer _ = mlx.mlx_array_free(tab.nlives);
        var outs: [2][]const u16 = undefined;
        for (&outs, [_]bool{ true, false }) |*o, ref| {
            nax_reference_override = ref;
            const got = try innerGemmSortedTable(s, x, tr, eids, 32, aligned, tab);
            defer _ = mlx.mlx_array_free(got);
            try mlx.check(mlx.mlx_array_eval(got));
            const p: [*]const u16 = @ptrCast(mlx.mlx_array_data_float16(got) orelse return error.F16Unreadable);
            o.* = try alloc.dupe(u16, p[0 .. nslots * @as(usize, @intCast(out_dim))]);
        }
        try std.testing.expect(!gemm_nax_failed);
        try std.testing.expectEqualSlices(u16, outs[0], outs[1]);
    }
}

test "exl3 NAX GEMM body writes the reference body's bytes at MiMo and Flash-Next geometry" {
    const mimo: exl3.Decode = .{ .codebook = .mcg, .window = .w12 };
    try naxBodyBytesMatch(.{ .n = 36 }, mimo, 4096, 2048, 71);
    try naxBodyBytesMatch(.{ .n = 36 }, mimo, 2048, 4096, 72);
    try naxBodyBytesMatch(exl3.Rate.fromK(4), mimo, 2048, 4096, 73);
    for ([_]exl3.Rate{ .{ .n = 48 }, .{ .n = 42 }, exl3.Rate.fromK(4) }) |rate| {
        for ([_]exl3.Decode{ .mul1, .{ .codebook = .mcg, .window = .w15 } }) |dec| {
            try naxBodyBytesMatch(rate, dec, 2560, 640, 74 + rate.n);
            try naxBodyBytesMatch(rate, dec, 640, 2560, 75 + rate.n);
        }
    }
}

test "exl3 shared reduction preserves the routed output rounding before addition" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const t = std.testing;
    const s = mlx.gpuStream();
    var prng = std.Random.DefaultPrng.init(1701);
    const rnd = prng.random();
    var ih: [10 * 256]u16 = undefined;
    var vh: [4 * 256]u16 = undefined;
    var sh: [256]f32 = undefined;
    for (&ih) |*v| v.* = exl3.f32ToF16Bits((rnd.float(f32) * 2 - 1) * 3);
    for (&vh) |*v| v.* = exl3.f32ToF16Bits((rnd.float(f32) * 2 - 1) * 2);
    for (&sh) |*v| v.* = (rnd.float(f32) * 2 - 1) * 4;
    const inner = mlx.mlx_array_new_data(&ih, &.{ 10, 256 }, 2, .float16);
    defer _ = mlx.mlx_array_free(inner);
    const svh = mlx.mlx_array_new_data(&vh, &.{ 4, 256 }, 2, .float16);
    defer _ = mlx.mlx_array_free(svh);
    const slots = mlx.mlx_array_new_data(&[_]u32{ 0, 1, 2, 3, 0, 2, 1, 3, 0, 2 }, &.{10}, 1, .uint32);
    defer _ = mlx.mlx_array_free(slots);
    const scores = mlx.mlx_array_new_data(&[_]f32{ 0.03, 0.08, 0.15, 0.2, 0.02, 0.01, 0.09, 0.17, 0.11, 0.14 }, &.{10}, 1, .float32);
    defer _ = mlx.mlx_array_free(scores);
    const shared_f32 = mlx.mlx_array_new_data(&sh, &.{256}, 1, .float32);
    defer _ = mlx.mlx_array_free(shared_f32);
    inline for ([_]mlx.mlx_dtype{ .bfloat16, .float16, .float32 }) |dtype| {
        var shared = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(shared);
        try mlx.check(mlx.mlx_astype(&shared, shared_f32, dtype, s));
        const routed = try downFinishReduce(s, inner, svh, slots, scores, 256, 1, 10, dtype);
        defer _ = mlx.mlx_array_free(routed);
        var expected = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(expected);
        try mlx.check(mlx.mlx_add(&expected, routed, shared, s));
        const got = try downFinishReduceShared(s, inner, svh, slots, scores, 256, 1, 10, dtype, shared);
        defer _ = mlx.mlx_array_free(got);
        var ef = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ef);
        var gf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(gf);
        try mlx.check(mlx.mlx_astype(&ef, expected, .float32, s));
        try mlx.check(mlx.mlx_astype(&gf, got, .float32, s));
        try mlx.check(mlx.mlx_array_eval(ef));
        try mlx.check(mlx.mlx_array_eval(gf));
        const ep = mlx.mlx_array_data_float32(ef).?;
        const gp = mlx.mlx_array_data_float32(gf).?;
        try t.expectEqualSlices(u8, std.mem.sliceAsBytes(ep[0..256]), std.mem.sliceAsBytes(gp[0..256]));
    }
}

// Staged oracle keeps the original projection and reduction ordering.
fn clampedSortedReference(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    gate_t: mlx.mlx_array,
    gate_suh: mlx.mlx_array,
    gate_svh: mlx.mlx_array,
    up_t: mlx.mlx_array,
    up_suh: mlx.mlx_array,
    up_svh: mlx.mlx_array,
    down_t: mlx.mlx_array,
    down_suh: mlx.mlx_array,
    down_svh: mlx.mlx_array,
    slots: mlx.mlx_array,
    scores: mlx.mlx_array,
    topk: c_int,
    limit: c_int,
    out_dtype: mlx.mlx_dtype,
) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const rows = sh[0];
    const hidden = sh[1];
    const nslots = rows * topk;
    const repeated = try repeatRows(s, x, rows, topk);
    defer _ = mlx.mlx_array_free(repeated);
    var order = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(order);
    try mlx.check(mlx.mlx_argsort_axis(&order, slots, 0, s));
    var sorted_slots = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sorted_slots);
    try mlx.check(mlx.mlx_take_axis(&sorted_slots, slots, order, 0, s));
    var sorted_x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sorted_x);
    try mlx.check(mlx.mlx_take_axis(&sorted_x, repeated, order, 0, s));
    const prep_g = try prepareIndexed(s, sorted_x, gate_suh, sorted_slots);
    defer _ = mlx.mlx_array_free(prep_g);
    const prep_u = try prepareIndexed(s, sorted_x, up_suh, sorted_slots);
    defer _ = mlx.mlx_array_free(prep_u);
    const decode = rows <= DECODE_ROWS_MAX;
    const gate = if (decode) try indexedGemvCoopF16(s, prep_g, gate_t, sorted_slots) else try innerGemmSorted(s, prep_g, gate_t, sorted_slots);
    defer _ = mlx.mlx_array_free(gate);
    const up = if (decode) try indexedGemvCoopF16(s, prep_u, up_t, sorted_slots) else try innerGemmSorted(s, prep_u, up_t, sorted_slots);
    defer _ = mlx.mlx_array_free(up);
    const prepared = try midSwigluPrepWithLimit(s, gate, up, gate_svh, up_svh, down_suh, sorted_slots, mlx.getShape(gate)[1], nslots, limit);
    defer _ = mlx.mlx_array_free(prepared);
    const down = if (decode) try indexedGemvCoopF16(s, prepared, down_t, sorted_slots) else try innerGemmSorted(s, prepared, down_t, sorted_slots);
    defer _ = mlx.mlx_array_free(down);
    var inverse = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(inverse);
    try mlx.check(mlx.mlx_argsort_axis(&inverse, order, 0, s));
    var original_order = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(original_order);
    try mlx.check(mlx.mlx_take_axis(&original_order, down, inverse, 0, s));
    return downFinishReduce(s, original_order, down_svh, slots, scores, hidden, rows, topk, out_dtype);
}

test "exl3 clamped routing takes the lane kernels and preserves staged bytes at every admitted rate" {
    const a = std.testing.allocator;
    const s = mlx.gpuStream();
    const dim = 256;
    const experts = 8;
    const topk = 8;
    setDecodeParams(.{ .codebook = .mcg, .window = .w12 });
    defer setDecodeParams(.{ .codebook = .mul1, .window = .w16 });
    var random = std.Random.DefaultPrng.init(89031);
    const rnd = random.random();
    var n: c_int = exl3.Rate.min_n;
    while (n <= exl3.Rate.max_n) : (n += 2) {
        const bits = try a.alloc(u16, experts * 16 * 16 * @as(usize, @intCast(n)));
        defer a.free(bits);
        for (bits) |*v| v.* = rnd.int(u16);
        var scales: [experts * dim]u16 = undefined;
        for (&scales) |*v| v.* = exl3.f32ToF16Bits(0.125 + rnd.float(f32) * 0.125);
        const tr = mlx.mlx_array_new_data(bits.ptr, &[_]c_int{ experts, 16, 16, n }, 4, .uint16);
        defer _ = mlx.mlx_array_free(tr);
        const sc = mlx.mlx_array_new_data(&scales, &[_]c_int{ experts, dim }, 2, .float16);
        defer _ = mlx.mlx_array_free(sc);
        for ([_]usize{ 1, 2, 8, 16, 17 }) |rows| {
            const x = try a.alloc(f32, rows * dim);
            defer a.free(x);
            for (x) |*v| v.* = (rnd.float(f32) - 0.5) * 32;
            const ids = try a.alloc(u32, rows * topk);
            defer a.free(ids);
            const weights = try a.alloc(f32, rows * topk);
            defer a.free(weights);
            for (ids, weights, 0..) |*id, *w, i| {
                id.* = @intCast((experts - 1 - i % experts + i / topk) % experts);
                w.* = 0.0625 * @as(f32, @floatFromInt(1 + i % 5));
            }
            const xf = mlx.mlx_array_new_data(x.ptr, &[_]c_int{ @intCast(rows), dim }, 2, .float32);
            defer _ = mlx.mlx_array_free(xf);
            const sl = mlx.mlx_array_new_data(ids.ptr, &[_]c_int{@intCast(ids.len)}, 1, .uint32);
            defer _ = mlx.mlx_array_free(sl);
            const ws = mlx.mlx_array_new_data(weights.ptr, &[_]c_int{@intCast(weights.len)}, 1, .float32);
            defer _ = mlx.mlx_array_free(ws);
            inline for (.{ mlx.mlx_dtype.float32, mlx.mlx_dtype.bfloat16 }) |dtype| {
                var xa = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(xa);
                try mlx.check(mlx.mlx_astype(&xa, xf, dtype, s));
                const old = try clampedSortedReference(s, xa, tr, sc, sc, tr, sc, sc, tr, sc, sc, sl, ws, topk, 10, dtype);
                defer _ = mlx.mlx_array_free(old);
                const lanes = lane_decode_dispatches;
                const got = try moeSwigluClamped(s, xa, tr, sc, sc, tr, sc, sc, tr, sc, sc, sl, ws, topk, 10, dtype);
                defer _ = mlx.mlx_array_free(got);
                const lane_rows = dtype == .bfloat16 and rows <= DECODE_ROWS_MAX;
                try std.testing.expectEqual(lanes + @as(usize, if (lane_rows) 2 else 0), lane_decode_dispatches);
                var old32 = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(old32);
                var got32 = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(got32);
                try mlx.check(mlx.mlx_astype(&old32, old, .float32, s));
                try mlx.check(mlx.mlx_astype(&got32, got, .float32, s));
                try std.testing.expectEqualSlices(u8, try gemvOutBytes(old32), try gemvOutBytes(got32));
            }
        }
    }
}

test "exl3 clamped routing preserves GLM production width bytes" {
    const s = mlx.gpuStream();
    defer setDecodeParams(.mul1);
    for ([_]exl3.Window{ .w12, .w14 }) |window| for ([_]usize{ 1, 2, 4, 8, 16, 17 }) |rows| {
        const dec = exl3.Decode{ .codebook = .mcg, .window = window };
        setDecodeParams(dec);
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        var f = try mimoMoeFixture(arena.allocator(), .{
            .hidden = 4096,
            .inter = 2048,
            .e = 8,
            .rows = rows,
            .topk = 8,
            .rate = .{ .n = 36 },
            .dec = dec,
            .seed = 621 + rows,
            .banks = .{ 0.125, 0.25, 0.125, 0.25 },
            .x_scale = 8,
        });
        defer f.deinit();
        const a = f.arrays;
        var x = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x);
        try mlx.check(mlx.mlx_astype(&x, a[8], .bfloat16, s));
        const old = try clampedSortedReference(s, x, a[0], a[3], a[4], a[1], a[3], a[4], a[2], a[5], a[6], a[7], a[9], 8, 10, .bfloat16);
        defer _ = mlx.mlx_array_free(old);
        const lanes = lane_decode_dispatches;
        const got = try moeSwigluClamped(s, x, a[0], a[3], a[4], a[1], a[3], a[4], a[2], a[5], a[6], a[7], a[9], 8, 10, .bfloat16);
        defer _ = mlx.mlx_array_free(got);
        // Decode widths take the lane gate/up and lane down kernels at every window.
        try std.testing.expectEqual(lanes + @as(usize, if (rows <= DECODE_ROWS_MAX) 2 else 0), lane_decode_dispatches);
        var old32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(old32);
        var got32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(got32);
        try mlx.check(mlx.mlx_astype(&old32, old, .float32, s));
        try mlx.check(mlx.mlx_astype(&got32, got, .float32, s));
        try std.testing.expectEqualSlices(u8, try gemvOutBytes(old32), try gemvOutBytes(got32));
    };
}

test "exl3 unsorted paired prepare preserves distinct scale planes at GLM widths" {
    const a = std.testing.allocator;
    const s = mlx.gpuStream();
    const experts = 288;
    const topk = 8;
    var random = std.Random.DefaultPrng.init(235124);
    const rnd = random.random();
    for ([_]usize{ 128, 4096 }) |dim| {
        const gate = try a.alloc(u16, experts * dim);
        defer a.free(gate);
        const up = try a.alloc(u16, experts * dim);
        defer a.free(up);
        for (gate, up) |*g, *u| {
            g.* = exl3.f32ToF16Bits((rnd.float(f32) - 0.5) * 2);
            u.* = exl3.f32ToF16Bits((rnd.float(f32) - 0.5) * 3);
        }
        const ga = mlx.mlx_array_new_data(gate.ptr, &[_]c_int{ experts, @intCast(dim) }, 2, .float16);
        defer _ = mlx.mlx_array_free(ga);
        const ua = mlx.mlx_array_new_data(up.ptr, &[_]c_int{ experts, @intCast(dim) }, 2, .float16);
        defer _ = mlx.mlx_array_free(ua);
        for ([_]usize{ 1, 8, 16 }) |rows| {
            const input = try a.alloc(f32, rows * dim);
            defer a.free(input);
            for (input) |*x| x.* = (rnd.float(f32) - 0.5) * 32;
            const ids = try a.alloc(u32, rows * topk);
            defer a.free(ids);
            for (ids, 0..) |*id, i| id.* = if (i % topk == 0) 287 else @intCast((i * 37) % experts);
            const xf = mlx.mlx_array_new_data(input.ptr, &[_]c_int{ @intCast(rows), @intCast(dim) }, 2, .float32);
            defer _ = mlx.mlx_array_free(xf);
            const sl = mlx.mlx_array_new_data(ids.ptr, &[_]c_int{@intCast(ids.len)}, 1, .uint32);
            defer _ = mlx.mlx_array_free(sl);
            inline for (.{ mlx.mlx_dtype.float32, mlx.mlx_dtype.bfloat16 }) |dtype| {
                var x = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(x);
                try mlx.check(mlx.mlx_astype(&x, xf, dtype, s));
                const repeated = try repeatRows(s, x, @intCast(rows), topk);
                defer _ = mlx.mlx_array_free(repeated);
                const oldg = try prepareIndexed(s, repeated, ga, sl);
                defer _ = mlx.mlx_array_free(oldg);
                const oldu = try prepareIndexed(s, repeated, ua, sl);
                defer _ = mlx.mlx_array_free(oldu);
                const new = try pairPrepareFromTokens(s, x, ga, ua, sl, null, @intCast(dim), @intCast(ids.len), topk);
                defer _ = mlx.mlx_array_free(new[0]);
                defer _ = mlx.mlx_array_free(new[1]);
                try std.testing.expectEqualSlices(u8, try gemvOutBytes(oldg), try gemvOutBytes(new[0]));
                try std.testing.expectEqualSlices(u8, try gemvOutBytes(oldu), try gemvOutBytes(new[1]));
            }
        }
    }
}

test "exl3 GLM GPU windows match host metadata with zero padded windows" {
    const a = std.testing.allocator;
    const s = mlx.gpuStream();
    const experts = 288;
    for ([_]usize{ 1, 31, 32, 33, 296, 1024 }) |n| for (0..3) |pattern| {
        const ids = try a.alloc(u32, n);
        defer a.free(ids);
        for (ids, 0..) |*id, i| id.* = @intCast(switch (pattern) {
            0 => (i % 5) * 47,
            1 => i % experts,
            else => if (i % 101 == 0) @as(usize, experts - 1) else 7,
        });
        const raw = mlx.mlx_array_new_data(ids.ptr, &.{@intCast(n)}, 1, .uint32);
        defer _ = mlx.mlx_array_free(raw);
        var order = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(order);
        var sorted = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sorted);
        try mlx.check(mlx.mlx_argsort_axis(&order, raw, 0, s));
        try mlx.check(mlx.mlx_take_axis(&sorted, raw, order, 0, s));
        for ([_]c_int{ 16, 32 }) |win| {
            const host = try buildWindowTableHost(s, sorted, @intCast(n), win);
            defer _ = mlx.mlx_array_free(host.starts);
            defer _ = mlx.mlx_array_free(host.nlives);
            const metadata = try clampedWindowTable(s, sorted, order, @intCast(n), win, experts, true);
            defer _ = mlx.mlx_array_free(metadata.inverse);
            const gpu = metadata.table;
            defer _ = mlx.mlx_array_free(gpu.starts);
            defer _ = mlx.mlx_array_free(gpu.nlives);
            try std.testing.expectEqual(@as(c_int, @intCast((n + @as(usize, @intCast(win)) - 1) / @as(usize, @intCast(win)) + experts)), gpu.nwin);
            try mlx.check(mlx.mlx_array_eval(gpu.starts));
            try mlx.check(mlx.mlx_array_eval(gpu.nlives));
            const starts = mlx.mlx_array_data_uint32(gpu.starts).?[0..@intCast(gpu.nwin)];
            const lives = mlx.mlx_array_data_uint32(gpu.nlives).?[0..@intCast(gpu.nwin)];
            const used: usize = @intCast(host.nwin);
            try std.testing.expectEqualSlices(u32, mlx.mlx_array_data_uint32(host.starts).?[0..used], starts[0..used]);
            try std.testing.expectEqualSlices(u32, mlx.mlx_array_data_uint32(host.nlives).?[0..used], lives[0..used]);
            for (starts[used..], lives[used..]) |begin, live| {
                try std.testing.expectEqual(@as(u32, 0), begin);
                try std.testing.expectEqual(@as(u32, 0), live);
            }
        }
    };
}

fn clampedGpuWindowBytes(hidden: usize, inter: usize, n: u32) !void {
    return clampedWindowBytes(hidden, inter, n, true);
}

fn clampedWindowBytes(hidden: usize, inter: usize, n: u32, aligned: bool) !void {
    const prior = gemm_align_cached;
    gemm_align_cached = aligned;
    defer gemm_align_cached = prior;
    const s = mlx.gpuStream();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const experts = 288;
    const rows = 37;
    const topk = 8;
    var f = try mimoMoeFixture(a, .{
        .hidden = hidden,
        .inter = inter,
        .e = experts,
        .rows = rows,
        .topk = topk,
        .rate = .{ .n = n },
        .dec = .{ .codebook = .mcg, .window = .w12 },
        .seed = 8320 + n,
        .banks = .{ 0.125, 0.25, 0.125, 0.25 },
        .x_scale = 8,
    });
    defer f.deinit();
    const v = f.arrays;
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    try mlx.check(mlx.mlx_astype(&x, v[8], .bfloat16, s));
    for (0..3) |pattern| {
        var ids: [rows * topk]u32 = undefined;
        for (&ids, 0..) |*id, i| id.* = @intCast(switch (pattern) {
            0 => (i % topk) * 35,
            1 => (i * 17) % experts,
            else => if ((i / topk) % 11 == 0) 280 + i % topk else i % topk,
        });
        const slots = mlx.mlx_array_new_data(&ids, &.{rows * topk}, 1, .uint32);
        defer _ = mlx.mlx_array_free(slots);
        const reference = try clampedSortedReference(s, x, v[0], v[3], v[4], v[1], v[3], v[4], v[2], v[5], v[6], slots, v[9], topk, 10, .bfloat16);
        defer _ = mlx.mlx_array_free(reference);
        resetFusedDispatchCount();
        const actual = try moeSwigluClamped(s, x, v[0], v[3], v[4], v[1], v[3], v[4], v[2], v[5], v[6], slots, v[9], topk, 10, .bfloat16);
        defer _ = mlx.mlx_array_free(actual);
        try std.testing.expectEqual(@as(u32, 4), fusedDispatchCount());
        var ref32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ref32);
        var got32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(got32);
        try mlx.check(mlx.mlx_astype(&ref32, reference, .float32, s));
        try mlx.check(mlx.mlx_astype(&got32, actual, .float32, s));
        try std.testing.expectEqualSlices(u8, try gemvOutBytes(ref32), try gemvOutBytes(got32));
    }
}

test "exl3 GLM GPU windows preserve 288 expert bytes at every admitted rate" {
    setDecodeParams(.{ .codebook = .mcg, .window = .w12 });
    defer setDecodeParams(.mul1);
    var n: u32 = exl3.Rate.min_n;
    while (n <= exl3.Rate.max_n) : (n += 2) try clampedGpuWindowBytes(128, 128, n);
}

test "exl3 GLM GPU windows preserve 288 expert production width bytes" {
    setDecodeParams(.{ .codebook = .mcg, .window = .w12 });
    defer setDecodeParams(.mul1);
    try clampedGpuWindowBytes(4096, 2048, 36);
}

test "exl3 paired cooperative projections preserve separate F16 bytes" {
    const a = std.testing.allocator;
    const stream = mlx.gpuStream();
    setDecodeParams(.{ .codebook = .mcg, .window = .w12 });
    defer setDecodeParams(.{ .codebook = .mul1, .window = .w16 });
    var random = std.Random.DefaultPrng.init(563891);
    const rnd = random.random();
    var rate: c_int = exl3.Rate.min_n;
    while (rate <= exl3.Rate.max_n) : (rate += 2) {
        const bits = try a.alloc(u16, 3 * 8 * 16 * @as(usize, @intCast(rate)));
        defer a.free(bits);
        var banks: [2]mlx.mlx_array = undefined;
        var made: usize = 0;
        defer for (banks[0..made]) |v| {
            _ = mlx.mlx_array_free(v);
        };
        for (&banks) |*bank| {
            for (bits) |*v| v.* = rnd.int(u16);
            bank.* = mlx.mlx_array_new_data(bits.ptr, &[_]c_int{ 3, 8, 16, rate }, 4, .uint16);
            made += 1;
        }
        var g: [8 * 128]u16 = undefined;
        var u: [8 * 128]u16 = undefined;
        var ids: [8]u32 = undefined;
        for (&g, &u) |*gv, *uv| {
            gv.* = exl3.f32ToF16Bits((rnd.float(f32) - 0.5) * 8);
            uv.* = exl3.f32ToF16Bits((rnd.float(f32) - 0.5) * 8);
        }
        for (&ids, 0..) |*id, i| id.* = @intCast((i * 7) % 3);
        const xg = mlx.mlx_array_new_data(&g, &[_]c_int{ 8, 128 }, 2, .float16);
        defer _ = mlx.mlx_array_free(xg);
        const xu = mlx.mlx_array_new_data(&u, &[_]c_int{ 8, 128 }, 2, .float16);
        defer _ = mlx.mlx_array_free(xu);
        const slots = mlx.mlx_array_new_data(&ids, &[_]c_int{8}, 1, .uint32);
        defer _ = mlx.mlx_array_free(slots);
        const expected = [_]mlx.mlx_array{ try indexedGemvCoopF16(stream, xg, banks[0], slots), try indexedGemvCoopF16(stream, xu, banks[1], slots) };
        defer for (expected) |v| {
            _ = mlx.mlx_array_free(v);
        };
        resetFusedDispatchCount();
        const got = (try indexedPairCoopF16(stream, xg, xu, banks[0], banks[1], slots)) orelse return error.TestExpectedPairedProjection;
        try std.testing.expectEqual(@as(u32, 1), fusedDispatchCount());
        defer for (got) |v| {
            _ = mlx.mlx_array_free(v);
        };
        for (got, expected) |value, reference| try std.testing.expectEqualSlices(u8, try gemvOutBytes(reference), try gemvOutBytes(value));
    }
}

test "exl3 paired cooperative projections preserve GLM width and mixed rate fallback" {
    const s = mlx.gpuStream();
    setDecodeParams(.{ .codebook = .mcg, .window = .w12 });
    defer setDecodeParams(.mul1);
    for ([_]usize{ 1, 16 }) |rows| {
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        var f = try mimoMoeFixture(a, .{ .hidden = 4096, .inter = 2048, .e = 8, .rows = rows, .topk = 8, .rate = .{ .n = 36 }, .dec = .{ .codebook = .mcg, .window = .w12 }, .seed = 29632 + rows, .banks = .{ 0.125, 0.25, 0.125, 0.25 }, .x_scale = 8 });
        defer f.deinit();
        const v = f.arrays;
        var random = std.Random.DefaultPrng.init(9762 + rows);
        var planes: [2]mlx.mlx_array = undefined;
        var made: usize = 0;
        defer for (planes[0..made]) |value| {
            _ = mlx.mlx_array_free(value);
        };
        for (&planes) |*plane| {
            const data = try a.alloc(u16, rows * 8 * 4096);
            for (data) |*value| value.* = exl3.f32ToF16Bits((random.random().float(f32) - 0.5) * 8);
            plane.* = mlx.mlx_array_new_data(data.ptr, &.{ @intCast(rows * 8), 4096 }, 2, .float16);
            made += 1;
        }
        const pair = (try indexedPairCoopF16(s, planes[0], planes[1], v[0], v[1], v[7])) orelse return error.TestExpectedPairedProjection;
        defer for (pair) |value| {
            _ = mlx.mlx_array_free(value);
        };
        for (pair, planes, [_]mlx.mlx_array{ v[0], v[1] }) |actual, plane, bank| {
            const expected = try indexedGemvCoopF16(s, plane, bank, v[7]);
            defer _ = mlx.mlx_array_free(expected);
            try std.testing.expectEqualSlices(u8, try gemvOutBytes(expected), try gemvOutBytes(actual));
        }
        const mixed_up = mlx.mlx_array_new_data(f.up_t.ptr, &.{ 8, 256, 128, 32 }, 4, .uint16);
        defer _ = mlx.mlx_array_free(mixed_up);
        try std.testing.expect((try indexedPairCoopF16(s, planes[0], planes[1], v[0], mixed_up, v[7])) == null);
        var x = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x);
        try mlx.check(mlx.mlx_astype(&x, v[8], .bfloat16, s));
        const expected = try clampedSortedReference(s, x, v[0], v[3], v[4], mixed_up, v[3], v[4], v[2], v[5], v[6], v[7], v[9], 8, 10, .bfloat16);
        defer _ = mlx.mlx_array_free(expected);
        const actual = try moeSwigluClamped(s, x, v[0], v[3], v[4], mixed_up, v[3], v[4], v[2], v[5], v[6], v[7], v[9], 8, 10, .bfloat16);
        defer _ = mlx.mlx_array_free(actual);
        var ef = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ef);
        var af = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(af);
        try mlx.check(mlx.mlx_astype(&ef, expected, .float32, s));
        try mlx.check(mlx.mlx_astype(&af, actual, .float32, s));
        try std.testing.expectEqualSlices(u8, try gemvOutBytes(ef), try gemvOutBytes(af));
    }
}

test "exl3 GLM sorted finish retains stride fallback bytes" {
    setDecodeParams(.{ .codebook = .mcg, .window = .w12 });
    defer setDecodeParams(.mul1);
    try clampedWindowBytes(128, 128, 36, false);
}

const CLAMPED_MIDDLE_DOWN_SOURCE: [:0]const u8 = blk: {
    @setEvalBranchQuota(100000);
    const mid_begin = std.mem.indexOf(u8, MID_SOURCE, "const float sc =").?;
    const mid_end = std.mem.indexOf(u8, MID_SOURCE, "yd[xb + lane]").?;
    const ot = "uint ot = uint(threadgroup_position_in_grid.x);";
    const coop_begin = std.mem.indexOf(u8, INDEXED_COOP_SOURCE, ot).? + ot.len;
    const xb = "const size_t xb = (size_t)slot * (size_t)(IDIM);";
    const coop_xb = std.mem.indexOf(u8, INDEXED_COOP_SOURCE, xb).?;
    const prefix =
        \\threadgroup float partial[4 * 256];
        \\threadgroup half prepared[uint(IDIM)];
        \\{
        \\ const uint slot = threadgroup_position_in_grid.y;
        \\ const uint eid = uint(slots[slot]);
        \\ const ushort lane = thread_index_in_simdgroup;
        \\ for (uint block = simdgroup_index_in_threadgroup; block < uint(IDIM)/128u; block += 4u) {
        \\  const uint base = block * 128u;
        \\  const size_t xb = (size_t)slot * uint(IDIM) + base;
        \\  const size_t sb = (size_t)eid * uint(IDIM) + base;
        \\
    ;
    const bridge =
        \\prepared[base + lane] = half((s0 + s2) * sc);
        \\prepared[base + lane + 32u] = half((s1 + s3) * sc);
        \\prepared[base + lane + 64u] = half((s0 - s2) * sc);
        \\prepared[base + lane + 96u] = half((s1 - s3) * sc);
        \\ }
        \\}
        \\threadgroup_barrier(mem_flags::mem_threadgroup);
        \\const threadgroup half* x = prepared;
        \\for (uint output_tile = 0; output_tile < uint(OTPT); ++output_tile) {
        \\ uint ot = uint(threadgroup_position_in_grid.x) * uint(OTPT) + output_tile;
        \\
    ;
    break :blk prefix ++ MID_SOURCE[mid_begin..mid_end] ++ bridge ++ INDEXED_COOP_SOURCE[coop_begin..coop_xb] ++
        "const size_t xb = 0u;" ++ INDEXED_COOP_SOURCE[coop_xb + xb.len ..] ++ "\n}\n";
};
const ClampedMiddleKey = struct { input: c_int, output: c_int, slots: c_int, n: u32, limit: c_int, tiles: c_int };
var clamped_middle_cfgs: CfgCache(ClampedMiddleKey, 8) = .{};
var clamped_middle_kernels: KernelSlots = no_kernels;

fn clampedMiddleDownCoop(s: mlx.mlx_stream, ig: mlx.mlx_array, iu: mlx.mlx_array, down: mlx.mlx_array, svhg: mlx.mlx_array, svhu: mlx.mlx_array, suhd: mlx.mlx_array, slots: mlx.mlx_array, limit: c_int, tiles: c_int) !?mlx.mlx_array {
    const x = mlx.getShape(ig);
    const w = mlx.getShape(down);
    const ids = mlx.getShape(slots);
    if (!mlx.streamIsGpu(s) or x.len != 2 or w.len != 4 or ids.len != 1 or
        !std.mem.eql(c_int, x, mlx.getShape(iu)) or x[0] != ids[0] or x[0] < 1 or x[1] < 128 or x[1] > 8192 or @mod(x[1], 128) != 0 or
        w[0] < 1 or w[1] < 1 or w[1] > std.math.maxInt(c_int) / 16 or w[1] * 16 != x[1] or w[2] < 1 or w[2] > std.math.maxInt(c_int) / 128 or (tiles != 1 and tiles != 4 and tiles != 8) or @mod(w[2], tiles) != 0 or limit < 1 or limit > 128 or
        mlx.mlx_array_dtype(ig) != .float16 or mlx.mlx_array_dtype(iu) != .float16 or mlx.mlx_array_dtype(down) != .uint16 or
        (mlx.mlx_array_dtype(slots) != .uint32 and mlx.mlx_array_dtype(slots) != .int32)) return null;
    for ([_]mlx.mlx_array{ svhg, svhu, suhd }) |a|
        if (mlx.mlx_array_dtype(a) != .float16 or !std.mem.eql(c_int, &.{ w[0], x[1] }, mlx.getShape(a))) return null;
    const rate = try packedRate(w[3]);
    const key = ClampedMiddleKey{ .input = x[1], .output = w[2] * 16, .slots = x[0], .n = rate.n, .limit = limit, .tiles = tiles };
    const cfg = clamped_middle_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &.{ x[0], key.output }, 2, .float16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, @divExact(w[2], tiles) * 128, x[0], 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 128, 1, 1));
        inline for (.{ .{ "IDIM", key.input }, .{ "ODIM", key.output }, .{ "NHW", @as(c_int, @intCast(rate.n)) }, .{ "CLAMP_LIMIT", limit }, .{ "OTPT", tiles } }) |p|
            try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, p[0], p[1]));
        clamped_middle_cfgs.put(key, c);
        break :blk c;
    };
    const k = try codebookKernel(&clamped_middle_kernels, "sushi_exl3_clamped_middle_down", &.{ "ig", "iu", "trellis", "svhg", "svhu", "suhd", "slots" }, &.{"y"}, CLAMPED_MIDDLE_DOWN_SOURCE);
    const outputs = try applyOuts(s, k, &.{ ig, iu, down, svhg, svhu, suhd, slots }, cfg, 1);
    defer _ = mlx.mlx_vector_array_free(outputs);
    var result = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(result);
    try mlx.check(mlx.mlx_vector_array_get(&result, outputs, 0));
    return result;
}

fn clampedMiddleDownCase(hidden: usize, inter: usize, rows: usize, rate: u32, tiles: c_int) !void {
    const s = mlx.gpuStream();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try mimoMoeFixture(a, .{ .hidden = hidden, .inter = inter, .e = 8, .rows = rows, .topk = 8, .rate = .{ .n = rate }, .dec = .{ .codebook = .mcg, .window = .w12 }, .seed = 99211 + rate + rows, .banks = .{ 0.125, 0.25, 0.125, 0.25 }, .x_scale = 8 });
    defer f.deinit();
    var prng = std.Random.DefaultPrng.init(7821 + rate);
    var planes: [2]mlx.mlx_array = undefined;
    var made: usize = 0;
    defer for (planes[0..made]) |v| {
        _ = mlx.mlx_array_free(v);
    };
    for (&planes) |*p| {
        const data = try a.alloc(u16, rows * 8 * inter);
        for (data) |*v| v.* = exl3.f32ToF16Bits((prng.random().float(f32) - 0.5) * 128);
        p.* = mlx.mlx_array_new_data(data.ptr, &.{ @intCast(rows * 8), @intCast(inter) }, 2, .float16);
        made += 1;
    }
    const scale = try a.alloc(u16, 8 * inter);
    for (scale) |*v| v.* = exl3.f32ToF16Bits((prng.random().float(f32) - 0.5) * 0.5);
    const up_scale = mlx.mlx_array_new_data(scale.ptr, &.{ 8, @intCast(inter) }, 2, .float16);
    defer _ = mlx.mlx_array_free(up_scale);
    const v = f.arrays;
    const mid = try midSwigluPrepWithLimit(s, planes[0], planes[1], v[4], up_scale, v[5], v[7], @intCast(inter), @intCast(rows * 8), 10);
    defer _ = mlx.mlx_array_free(mid);
    const old = try indexedGemvCoopF16(s, mid, v[2], v[7]);
    defer _ = mlx.mlx_array_free(old);
    const got = (try clampedMiddleDownCoop(s, planes[0], planes[1], v[2], v[4], up_scale, v[5], v[7], 10, tiles)) orelse return error.TestExpectedMiddleDownFusion;
    defer _ = mlx.mlx_array_free(got);
    try std.testing.expectEqualSlices(u8, try gemvOutBytes(old), try gemvOutBytes(got));
}

test "exl3 clamped middle down cooperative fusion preserves F16 bytes" {
    setDecodeParams(.{ .codebook = .mcg, .window = .w12 });
    defer setDecodeParams(.mul1);
    for (0..17) |i| for ([_]c_int{ 4, 8 }) |tiles| try clampedMiddleDownCase(256, 128, 1, @intCast(32 + 2 * i), tiles);
    for ([_]usize{ 1, 2, 4, 8, 16 }) |rows| try clampedMiddleDownCase(4096, 2048, rows, 36, if (rows == 1) 4 else 8);
}

// Lane-ordered half4 gate/up pair; each row keeps the indexed pair's arithmetic.
fn lanePairChain(s: mlx.mlx_stream, v: [10]mlx.mlx_array, x: mlx.mlx_array, hidden: c_int, rows: c_int, candidate: bool) ![2]mlx.mlx_array {
    const prepared = if (candidate) try lanePairPrepare(s, x, v[3], v[4], v[7], null, hidden, rows * 8, 8) else try pairPrepareFromTokens(s, x, v[3], v[4], v[7], null, hidden, rows * 8, 8);
    defer _ = mlx.mlx_array_free(prepared[0]);
    defer _ = mlx.mlx_array_free(prepared[1]);
    return (if (candidate) try lanePairCoop(s, prepared[0], prepared[1], v[0], v[1], v[7]) else try indexedPairCoopF16(s, prepared[0], prepared[1], v[0], v[1], v[7])) orelse error.TestExpectedLanePair;
}
fn lanePairCase(hidden: usize, inter: usize, rows: usize, rate: u32) !void {
    const s = mlx.gpuStream();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var f = try mimoMoeFixture(arena.allocator(), .{ .hidden = hidden, .inter = inter, .e = 8, .rows = rows, .topk = 8, .rate = .{ .n = rate }, .dec = .{ .codebook = .mcg, .window = .w12 }, .seed = 78314 + rows, .banks = .{ 0.125, 0.25, 0.125, 0.25 }, .x_scale = 8 });
    defer f.deinit();
    var x = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x);
    try mlx.check(mlx.mlx_astype(&x, f.arrays[8], .bfloat16, s));
    var up_scale = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(up_scale);
    try mlx.check(mlx.mlx_negative(&up_scale, f.arrays[3], s));
    var banks = f.arrays;
    banks[4] = up_scale;
    const expected = try lanePairChain(s, banks, x, @intCast(hidden), @intCast(rows), false);
    defer for (expected) |v| {
        _ = mlx.mlx_array_free(v);
    };
    const actual = try lanePairChain(s, banks, x, @intCast(hidden), @intCast(rows), true);
    defer for (actual) |v| {
        _ = mlx.mlx_array_free(v);
    };
    for (expected, actual) |want, got| try std.testing.expectEqualSlices(u8, try gemvOutBytes(want), try gemvOutBytes(got));
}
test "exl3 GLM lane half4 pair exact bytes" {
    setDecodeParams(.{ .codebook = .mcg, .window = .w12 });
    defer setDecodeParams(.mul1);
    for (0..17) |i| try lanePairCase(256, 128, 3, @intCast(32 + 2 * i));
    for ([_]usize{ 1, 3, 7, 16 }) |rows| try lanePairCase(4096, 2048, rows, 36);
}

const LANE_PREP_SOURCE: [:0]const u8 = blk: {
    @setEvalBranchQuota(1000000);
    break :blk metalReplaceAll(TOKEN_PAIR_PREPARE_SOURCE, "[yb + lane", "[yb + " ++ LANE_ORDER_SLOT);
};
const LANE_PAIR_SOURCE: [:0]const u8 = blk: {
    @setEvalBranchQuota(1000000);
    var body = INDEXED_PAIR_COOP_SOURCE;
    body = metalReplaceAll(body, "const float in0 = float(x[xb + tk * TILE + row0]);", "const float4 in4 = float4(*((const device half4 *)(x + xb + tk * TILE + (lane & 3u) * 4u)));\nconst float in0 = in4.x;");
    body = metalReplaceAll(body, "const float in1 = float(x[xb + tk * TILE + row1]);", "const float in1 = in4.y;");
    body = metalReplaceAll(body, "const float in2 = float(x[xb + tk * TILE + row2]);", "const float in2 = in4.z;");
    body = metalReplaceAll(body, "const float in3 = float(x[xb + tk * TILE + row3]);", "const float in3 = in4.w;");
    break :blk body;
};
var lane_prep_cfgs: CfgCache(PairPrepKey, 8) = .{};
var lane_prep_sorted: ?mlx.mlx_fast_metal_kernel = null;
var lane_prep_unsorted: ?mlx.mlx_fast_metal_kernel = null;
var lane_pair_cfgs: CfgCache(IndexedKey, 8) = .{};
var lane_pair_kernel: KernelSlots = no_kernels;
pub fn lanePairPrepare(s: mlx.mlx_stream, x: mlx.mlx_array, suhg: mlx.mlx_array, suhu: mlx.mlx_array, slots: mlx.mlx_array, order: ?mlx.mlx_array, in_dim: c_int, nslots: c_int, topk: c_int) !struct { mlx.mlx_array, mlx.mlx_array } {
    const key = PairPrepKey{ .in_dim = in_dim, .nslots = nslots, .topk = topk };
    const cfg = lane_prep_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const sh = [_]c_int{ nslots, in_dim };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &sh, 2, .float16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &sh, 2, .float16));
        const blocks = @divExact(in_dim, 128);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, 32 * blocks, nslots, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "IDIM", in_dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "TOPK", topk));
        lane_prep_cfgs.put(key, c);
        break :blk c;
    };
    const ins = [_][*:0]const u8{ "x", "suhg", "suhu", "slots", "order" };
    const outs = [_][*:0]const u8{ "yg", "yu" };
    const kernel = if (order != null)
        try getNamedKernel(&lane_prep_sorted, "sushi_glm_lane_prepare", &ins, &outs, LANE_PREP_SOURCE, "")
    else
        try getNamedKernel(&lane_prep_unsorted, "sushi_glm_lane_unsorted", ins[0..4], &outs, metalReplaceFound(LANE_PREP_SOURCE, "const uint orig = uint(order[slot]);", "const uint orig = slot;"), "");
    const ov = if (order) |sorted|
        try applyOuts(s, kernel, &.{ x, suhg, suhu, slots, sorted }, cfg, 2)
    else
        try applyOuts(s, kernel, &.{ x, suhg, suhu, slots }, cfg, 2);
    defer _ = mlx.mlx_vector_array_free(ov);
    var a = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(a);
    var b = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(b);
    try mlx.check(mlx.mlx_vector_array_get(&a, ov, 0));
    try mlx.check(mlx.mlx_vector_array_get(&b, ov, 1));
    return .{ a, b };
}
pub fn lanePairCoop(s: mlx.mlx_stream, xg: mlx.mlx_array, xu: mlx.mlx_array, tg: mlx.mlx_array, tu: mlx.mlx_array, slots: mlx.mlx_array) !?[2]mlx.mlx_array {
    const xs = mlx.getShape(xg);
    const ts = mlx.getShape(tg);
    const ss = mlx.getShape(slots);
    if (!mlx.streamIsGpu(s) or xs.len != 2 or ts.len != 4 or ss.len != 1 or
        !std.mem.eql(c_int, xs, mlx.getShape(xu)) or !std.mem.eql(c_int, ts, mlx.getShape(tu)) or
        xs[0] <= 0 or xs[1] <= 0 or ts[0] <= 0 or ts[1] <= 0 or ts[2] <= 0 or
        ts[1] > std.math.maxInt(c_int) / 16 or ts[2] > std.math.maxInt(c_int) / 128 or
        xs[0] != ss[0] or ts[1] * 16 != xs[1] or
        mlx.mlx_array_dtype(xg) != .float16 or mlx.mlx_array_dtype(xu) != .float16 or
        mlx.mlx_array_dtype(tg) != .uint16 or mlx.mlx_array_dtype(tu) != .uint16 or
        (mlx.mlx_array_dtype(slots) != .uint32 and mlx.mlx_array_dtype(slots) != .int32)) return null;
    const rate = try packedRate(ts[3]);
    const output = ts[2] * 16;
    const key = IndexedKey{ .in_dim = xs[1], .out_dim = output, .topk = ss[0], .n = rate.n };
    const cfg = lane_pair_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const shape = [_]c_int{ ss[0], output };
        for (0..2) |_| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &shape, 2, .float16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, ts[2] * 128, ss[0], 2));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 128, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "IDIM", xs[1]));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ODIM", output));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "NHW", @intCast(rate.n)));
        lane_pair_cfgs.put(key, c);
        break :blk c;
    };
    const kernel = try codebookKernel(&lane_pair_kernel, "sushi_glm_lane_pair", &.{ "xg", "xu", "tg", "tu", "slots" }, &.{ "yg", "yu" }, LANE_PAIR_SOURCE);
    const outputs = try applyOuts(s, kernel, &.{ xg, xu, tg, tu, slots }, cfg, 2);
    defer _ = mlx.mlx_vector_array_free(outputs);
    var result = [2]mlx.mlx_array{ mlx.mlx_array_new(), mlx.mlx_array_new() };
    errdefer for (result) |value| {
        _ = mlx.mlx_array_free(value);
    };
    for (&result, 0..) |*value, i| try mlx.check(mlx.mlx_vector_array_get(value, outputs, i));
    return result;
}

// Kernels shared with the DFlash group-of-two expert path in glm_group2.zig.
pub const Group2Support = struct {
    pub const Slots = KernelSlots;
    pub const empty = no_kernels;
    pub const cooperative_source = INDEXED_COOP_SOURCE;
    pub const prepare = pairPrepareFromTokens;
    pub const middle = midSwigluPrepWithLimit;
    pub const lane_middle = downLanePrepare;
    pub const lane_down = downLaneCoop;
    pub const fused_middle_down = clampedMiddleDownCoop;
    pub fn makeKernel(storage: *Slots, comptime name: [:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [:0]const u8) !mlx.mlx_fast_metal_kernel {
        return codebookKernel(storage, name, ins, outs, source);
    }
};

/// NAX source and helpers shared by the GLM prefill grid permutation.
pub const PrefillGridSupport = struct {
    pub const source = GEMM_NAX_SOURCE;
    pub const prepare = pairPrepareFromTokens;
    pub const windows = clampedWindowTable;
    pub const middle = midSwigluPrepWithLimit;
    pub const finish = finishMimoSorted;
    pub fn available() bool {
        if (!gemmNaxOn()) return false;
        const had_error = mlx.errorPending();
        defer mlx.dropLatchedErrorUnless(had_error);
        _ = getGemmNaxKernel() catch return false;
        return true;
    }
    pub fn makeKernel(kernel_source: [:0]const u8, window: exl3.Window) !mlx.mlx_fast_metal_kernel {
        @setEvalBranchQuota(20_000);
        const had_error = mlx.errorPending();
        defer mlx.dropLatchedErrorUnless(had_error);
        switch (window) {
            inline else => |win| return buildNaxGemmKernel(kernel_source, naxHeader(.mcg, win), comptime "sushi_glm_prefill_grid_transpose" ++ winSuffix(win)) orelse error.MetalKernelCompileFailed,
        }
    }
};

var lane_decode_dispatches: usize = 0;
/// The GLM lane kernels read through `exl3_lane`, whose words and shifts follow from n.
fn laneServesRate(n: c_int) bool {
    return n >= 0 and exl3.kFromPackedDim(@intCast(n)) != null;
}
fn laneClampedPair(s: mlx.mlx_stream, x: mlx.mlx_array, tg: mlx.mlx_array, tu: mlx.mlx_array, sg: mlx.mlx_array, su: mlx.mlx_array, slots: mlx.mlx_array, hidden: c_int, rows: c_int, topk: c_int) !?[2]mlx.mlx_array {
    if (!mlx.streamIsGpu(s) or active_decode.codebook != .mcg or mlx.mlx_array_dtype(x) != .bfloat16 or rows < 1 or rows > 16 or hidden < 128 or @mod(hidden, 128) != 0) return null;
    const gs = mlx.getShape(tg);
    if (gs.len != 4 or !std.mem.eql(c_int, gs, mlx.getShape(tu)) or !laneServesRate(gs[3])) return null;
    const prepared = try lanePairPrepare(s, x, sg, su, slots, null, hidden, rows * topk, topk);
    defer _ = mlx.mlx_array_free(prepared[0]);
    defer _ = mlx.mlx_array_free(prepared[1]);
    const pair = try lanePairCoop(s, prepared[0], prepared[1], tg, tu, slots);
    if (pair != null) lane_decode_dispatches += 1;
    return pair;
}

// Lane-ordered half4 reads for the clamped middle and the down projection.
fn downLaneReads(comptime source: [:0]const u8, comptime space: []const u8) [:0]const u8 {
    @setEvalBranchQuota(1000000);
    var body = source;
    body = metalReplaceAll(body, "const float in0 = float(x[xb + tk * TILE + row0]);", "const float4 in4 = float4(*((const " ++ space ++ " half4 *)(x + xb + tk * TILE + (lane & 3u) * 4u)));\nconst float in0 = in4.x;");
    body = metalReplaceAll(body, "const float in1 = float(x[xb + tk * TILE + row1]);", "const float in1 = in4.y;");
    body = metalReplaceAll(body, "const float in2 = float(x[xb + tk * TILE + row2]);", "const float in2 = in4.z;");
    body = metalReplaceAll(body, "const float in3 = float(x[xb + tk * TILE + row3]);", "const float in3 = in4.w;");
    return body;
}
const DOWN_LANE_MID_SOURCE = metalReplaceAll(MID_SOURCE, "yd[xb + lane", "yd[xb + " ++ LANE_ORDER_SLOT);
const DOWN_LANE_COOP_SOURCE = downLaneReads(INDEXED_COOP_SOURCE, "device");
var down_lane_mid_cfgs: CfgCache(MidKey, 8) = .{};
var down_lane_mid_kernel: ?mlx.mlx_fast_metal_kernel = null;
var down_lane_coop_cfgs: CfgCache(IndexedKey, 8) = .{};
var down_lane_coop_kernel: KernelSlots = no_kernels;
fn getDownLaneCoopKernel() !mlx.mlx_fast_metal_kernel {
    return codebookKernel(&down_lane_coop_kernel, "sushi_glm_down_lane_coop", &.{ "x", "trellis", "slots" }, &.{"y"}, DOWN_LANE_COOP_SOURCE);
}
fn downLanePrepare(s: mlx.mlx_stream, ig: mlx.mlx_array, iu: mlx.mlx_array, svhg: mlx.mlx_array, svhu: mlx.mlx_array, suhd: mlx.mlx_array, slots: mlx.mlx_array, dim: c_int, nslots: c_int, limit: c_int) !mlx.mlx_array {
    const key = MidKey{ .dim = dim, .nslots = nslots, .limit = limit };
    const cfg = down_lane_mid_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const sh = [_]c_int{ nslots, dim };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &sh, 2, .float16));
        const blocks = @divExact(dim, 128);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, 32 * blocks, nslots, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 32, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ODIM", dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "CLAMP_LIMIT", limit));
        down_lane_mid_cfgs.put(key, c);
        break :blk c;
    };
    const ins = [_][*:0]const u8{ "ig", "iu", "svhg", "svhu", "suhd", "slots" };
    const outs = [_][*:0]const u8{"yd"};
    const kernel = try getNamedKernel(&down_lane_mid_kernel, "sushi_glm_down_lane_prepare", &ins, &outs, DOWN_LANE_MID_SOURCE, "");
    const ov = try applyOuts(s, kernel, &.{ ig, iu, svhg, svhu, suhd, slots }, cfg, 1);
    defer _ = mlx.mlx_vector_array_free(ov);
    var a = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(a);
    try mlx.check(mlx.mlx_vector_array_get(&a, ov, 0));
    return a;
}
fn downLaneCoop(s: mlx.mlx_stream, x: mlx.mlx_array, trellis: mlx.mlx_array, slots: mlx.mlx_array) !mlx.mlx_array {
    const xsh = mlx.getShape(x);
    const tsh = mlx.getShape(trellis);
    const ssh = mlx.getShape(slots);
    if ((xsh.len != 1 and xsh.len != 2) or tsh.len != 4 or ssh.len != 1) return error.BadExl3Shape;
    const in_dim = xsh[xsh.len - 1];
    const out_dim = tsh[2] * 16;
    const out_tiles = tsh[2];
    const topk = ssh[0];
    const rate = try packedRate(tsh[3]);
    if (tsh[1] * 16 != in_dim) return error.BadExl3Shape;
    const key = IndexedKey{ .in_dim = in_dim, .out_dim = out_dim, .topk = topk, .n = rate.n };
    const cfg = down_lane_coop_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        const out_shape = [_]c_int{ topk, out_dim };
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &out_shape, 2, .float16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, out_tiles * 128, topk, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 128, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "IDIM", in_dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ODIM", out_dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "NHW", @intCast(rate.n)));
        down_lane_coop_cfgs.put(key, c);
        break :blk c;
    };
    const inputs_arr = [_]mlx.mlx_array{ x, trellis, slots };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);
    const kernel = try getDownLaneCoopKernel();
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, cfg, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, outputs_vec, 0));
    return out;
}
fn downLaneCase(hidden: usize, inter: usize, rows: usize, rate: u32, tiles: c_int) !void {
    const s = mlx.gpuStream();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try mimoMoeFixture(a, .{ .hidden = hidden, .inter = inter, .e = 8, .rows = rows, .topk = 8, .rate = .{ .n = rate }, .dec = .{ .codebook = .mcg, .window = .w12 }, .seed = 99211 + rate + rows, .banks = .{ 0.125, 0.25, 0.125, 0.25 }, .x_scale = 8 });
    defer f.deinit();
    var prng = std.Random.DefaultPrng.init(7821 + rate);
    var planes: [2]mlx.mlx_array = undefined;
    var made: usize = 0;
    defer for (planes[0..made]) |v| {
        _ = mlx.mlx_array_free(v);
    };
    for (&planes) |*p| {
        const data = try a.alloc(u16, rows * 8 * inter);
        for (data) |*v| v.* = exl3.f32ToF16Bits((prng.random().float(f32) - 0.5) * 128);
        p.* = mlx.mlx_array_new_data(data.ptr, &.{ @intCast(rows * 8), @intCast(inter) }, 2, .float16);
        made += 1;
    }
    const scale = try a.alloc(u16, 8 * inter);
    for (scale) |*v| v.* = exl3.f32ToF16Bits((prng.random().float(f32) - 0.5) * 0.5);
    const up_scale = mlx.mlx_array_new_data(scale.ptr, &.{ 8, @intCast(inter) }, 2, .float16);
    defer _ = mlx.mlx_array_free(up_scale);
    const v = f.arrays;
    const args = DownLaneArgs{ .s = s, .g = planes[0], .u = planes[1], .v = v, .up_scale = up_scale, .inter = @intCast(inter), .slots = @intCast(rows * 8), .tiles = tiles };
    const expected = try downLaneArm(args, 0);
    defer _ = mlx.mlx_array_free(expected);
    for (1..3) |arm| {
        const actual = try downLaneArm(args, arm);
        defer _ = mlx.mlx_array_free(actual);
        try std.testing.expectEqualSlices(u8, try gemvOutBytes(expected), try gemvOutBytes(actual));
    }
}
const DownLaneArgs = struct { s: mlx.mlx_stream, g: mlx.mlx_array, u: mlx.mlx_array, v: [10]mlx.mlx_array, up_scale: mlx.mlx_array, inter: c_int, slots: c_int, tiles: c_int };
fn downLaneArm(a: DownLaneArgs, arm: usize) !mlx.mlx_array {
    if (arm == 1) return (try clampedMiddleDownCoop(a.s, a.g, a.u, a.v[2], a.v[4], a.up_scale, a.v[5], a.v[7], 10, a.tiles)) orelse error.TestExpectedMiddleDownFusion;
    const mid = if (arm == 2) try downLanePrepare(a.s, a.g, a.u, a.v[4], a.up_scale, a.v[5], a.v[7], a.inter, a.slots, 10) else try midSwigluPrepWithLimit(a.s, a.g, a.u, a.v[4], a.up_scale, a.v[5], a.v[7], a.inter, a.slots, 10);
    defer _ = mlx.mlx_array_free(mid);
    return if (arm == 2) try downLaneCoop(a.s, mid, a.v[2], a.v[7]) else try indexedGemvCoopF16(a.s, mid, a.v[2], a.v[7]);
}
test "exl3 GLM down lane exact staged and fused bytes" {
    setDecodeParams(.{ .codebook = .mcg, .window = .w12 });
    defer setDecodeParams(.mul1);
    for (0..17) |i| try downLaneCase(256, 128, 3, @intCast(32 + 2 * i), 8);
    for ([_]usize{ 1, 4, 8, 16 }) |rows| try downLaneCase(4096, 2048, rows, 36, 8);
}
