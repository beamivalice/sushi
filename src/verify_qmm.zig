const trf = @import("transformer.zig");
const std = @import("std");
const mlx = @import("mlx.zig");
const test_support = @import("test_support.zig");
const testRandWeightBf16 = test_support.testRandWeightBf16;
const log = @import("log.zig");
const io_util_mod = @import("io_util.zig");
const diagEnvOn = io_util_mod.diagEnvOn;
const testing = std.testing;
const nax_gate = @import("nax_gate.zig");
const naxMinMForBits = nax_gate.naxMinMForBits;
const verifyQmmNaxAvailable = nax_gate.verifyQmmNaxAvailable;
const verifyQmmNaxEnabled = nax_gate.verifyQmmNaxEnabled;
const VerifyQmmParity = trf.VerifyQmmParity;
const verifyQmmParityVerdict = trf.verifyQmmParityVerdict;

/// Runtime codegen of the per-M split-K kernel body. The row loads, weight
/// loads, and dequant+FMA chains are emitted as NAMED SCALARS with LITERAL
/// accumulator indices — the ledger's load-bearing constraint: array-indexed
/// rows behind a runtime-ish loop stack-spill (measured 10x at M=6 with the
/// `Vec8 v[MROWS]` form of this same kernel). GS and K_PARTS stay template
/// ints; M is baked into the source (one cached kernel per M, distinct Metal
/// host_names — two specializations sharing a name bind the wrong binary).
/// The activation-FMA lines one output column contributes for a single K
/// value: `wv` is the dequantized weight, `idx` the element inside the Vec8
/// activation group. Shared by both plain-SIMD verify tiles so the two cannot
/// drift.
fn vqmmAccLines(a: std.mem.Allocator, m: usize, j: usize, idx: []const u8) std.mem.Allocator.Error![]const u8 {
    var out: []const u8 = "";
    for (0..m) |r| {
        out = try vqmmCat(a, out, try std.fmt.allocPrint(a, "        acc[{d}] += float(v{d}[{s}]) * wv;\n", .{ j * m + r, r, idx }));
    }
    return out;
}

/// Hoisted weight WORDS for one column, decoded per width.
///
/// One `pack` iteration always covers 8 K values (that is the Vec8 activation
/// load), so its weights are exactly BITS bytes at
/// `w_q + n * K_bytes + pack * BITS`. The loads stay HOISTED above the FMA
/// chains — the tile's whole shape is "issue every column's weight word, then
/// run one sequential chain per column", and leaving the load inside the chain
/// serializes it behind the previous column's arithmetic. Hoisting only the
/// POINTER is not enough: the widened arms must hoist the BYTES.
///
/// Widest load each width's alignment allows. 4-bit keeps its original aligned
/// uint32 verbatim — that width must stay bit-identical, and it is this
/// refactor's tripwire. 8-bit sits on `pack * 8` and takes two uints (not one
/// ulong: 4-byte alignment is what the shipped 4-bit path already assumes of
/// this buffer, and widening that assumption for one fused load buys nothing).
/// 6-bit lands on `pack * 6`, only 2-aligned, so three ushorts re-assembled
/// into two 3-byte words. 5-bit on `pack * 5` is 1-aligned and takes bytes.
fn vqmmColumnLoad(a: std.mem.Allocator, j: usize) std.mem.Allocator.Error![]const u8 {
    return std.fmt.allocPrint(a,
        \\  uint32_t p{d}; ulong r{d}; uint32_t a{d}0; uint32_t a{d}1;
        \\  if constexpr (BITS == 4) {{ p{d} = w_q[(n0 + {d}) * K_by_p + pack]; }}
        \\  else {{
        \\    const device uchar* wp = ((const device uchar*)w_q) + (n0 + {d}) * K_bytes + pack * BITS;
        \\    if constexpr (BITS == 8) {{
        \\      const device uint* wu = (const device uint*)wp;
        \\      r{d} = ulong(wu[0]) | (ulong(wu[1]) << 32);
        \\    }} else if constexpr (BITS == 5) {{
        \\      r{d} = ulong(wp[0]) | (ulong(wp[1]) << 8) | (ulong(wp[2]) << 16) |
        \\             (ulong(wp[3]) << 24) | (ulong(wp[4]) << 32);
        \\    }} else {{
        \\      const device ushort* wh = (const device ushort*)wp;
        \\      uint32_t h0 = uint32_t(wh[0]); uint32_t h1 = uint32_t(wh[1]); uint32_t h2 = uint32_t(wh[2]);
        \\      a{d}0 = h0 | ((h1 & 0xFFu) << 16);
        \\      a{d}1 = (h1 >> 8) | (h2 << 8);
        \\    }}
        \\  }}
        \\
    , .{ j, j, j, j, j, j, j, j, j, j, j });
}

/// Per-column dequant + FMA chain, one arm per supported affine width. The
/// byte-addressed arms are lifted from the NAX m16 tile's validated unpack
/// (4 values per 3 bytes at 6 bits, 8 per 5 bytes at 5, one byte per value at
/// 8) — this is not a fresh reading of MLX's packing.
fn vqmmColumnChain(a: std.mem.Allocator, m: usize, j: usize) std.mem.Allocator.Error![]const u8 {
    const acc_ki = try vqmmAccLines(a, m, j, "ki");
    const acc_hi = try vqmmAccLines(a, m, j, "4 + ki");
    return std.fmt.allocPrint(a,
        \\  {{
        \\    float sj = s{d}; float bj = b{d};
        \\    if constexpr (BITS == 4) {{
        \\      for (int ki = 0; ki < 8; ++ki) {{
        \\        float wv = float((p{d} >> (ki * 4)) & 0xFu) * sj + bj;
        \\{s}      }}
        \\    }} else if constexpr (BITS == 8) {{
        \\      for (int ki = 0; ki < 8; ++ki) {{
        \\        float wv = float(uint32_t((r{d} >> (ki * 8)) & 0xFFul)) * sj + bj;
        \\{s}      }}
        \\    }} else if constexpr (BITS == 5) {{
        \\      for (int ki = 0; ki < 8; ++ki) {{
        \\        float wv = float(uint32_t((r{d} >> (ki * 5)) & 0x1Ful)) * sj + bj;
        \\{s}      }}
        \\    }} else {{
        \\      for (int ki = 0; ki < 4; ++ki) {{
        \\        float wv = float((a{d}0 >> (ki * 6)) & 0x3Fu) * sj + bj;
        \\{s}      }}
        \\      for (int ki = 0; ki < 4; ++ki) {{
        \\        float wv = float((a{d}1 >> (ki * 6)) & 0x3Fu) * sj + bj;
        \\{s}      }}
        \\    }}
        \\  }}
        \\
    , .{ j, j, j, acc_ki, j, acc_ki, j, acc_ki, j, acc_ki, j, acc_hi });
}

fn verifyQmmSource(a: std.mem.Allocator, m: usize, bn: usize) std.mem.Allocator.Error![:0]const u8 {
    const nacc = bn * m;
    var body: []const u8 = "";
    // Row activation loads, all up front.
    for (0..m) |r| {
        body = try vqmmCat(a, body, try std.fmt.allocPrint(a, "  Vec8 v{d} = xv[({d} * K + k_base) / 8];\n", .{ r, r }));
    }
    // Weight handles for the owned columns, then scales/biases.
    for (0..bn) |j| {
        body = try vqmmCat(a, body, try vqmmColumnLoad(a, j));
    }
    for (0..bn) |j| {
        body = try vqmmCat(a, body, try std.fmt.allocPrint(a, "  float s{d} = float(scales[(n0 + {d}) * K_by_gs + gi]); float b{d} = float(biases[(n0 + {d}) * K_by_gs + gi]);\n", .{ j, j, j, j }));
    }
    // One sequential dequant+FMA chain per output column (column-major
    // construction; the interleaved form measurably loses).
    for (0..bn) |j| {
        body = try vqmmCat(a, body, try vqmmColumnChain(a, m, j));
    }
    var acc_init: []const u8 = "";
    for (0..nacc) |i| {
        acc_init = try vqmmCat(a, acc_init, try std.fmt.allocPrint(a, "acc[{d}] = 0.0f; ", .{i}));
    }
    var acc_sum: []const u8 = "";
    for (0..nacc) |i| {
        acc_sum = try vqmmCat(a, acc_sum, try std.fmt.allocPrint(a, "acc[{d}] = simd_sum(acc[{d}]); ", .{ i, i }));
    }
    return std.fmt.allocPrintSentinel(a,
        \\auto part = simdgroup_index_in_threadgroup;
        \\auto lane = thread_index_in_simdgroup;
        \\auto tg_n = threadgroup_position_in_grid.y;
        \\
        \\int K = int(K_size);
        \\int N = int(N_size);
        \\int K_by_p = K / 8;
        \\int K_bytes = (K * BITS) / 8;
        \\int K_by_gs = K / GS;
        \\int per_part = K_by_p / K_PARTS;
        \\int n0 = int(tg_n) * {d};
        \\int p_start = int(part) * per_part;
        \\int p_end = (int(part) == K_PARTS - 1) ? K_by_p : p_start + per_part;
        \\
        \\float acc[{d}];
        \\{s}
        \\
        \\using Vec8 = vec<T, 8>;
        \\const device Vec8 *xv = (const device Vec8*)x;
        \\
        \\for (int pack = p_start + int(lane); pack < p_end; pack += 32) {{
        \\  int k_base = pack * 8;
        \\  int gi = k_base / GS;
        \\{s}
        \\}}
        \\
        \\{s}
        \\
        \\threadgroup float partials[K_PARTS * {d}];
        \\if (lane == 0) {{
        \\  _Pragma("unroll")
        \\  for (int i = 0; i < {d}; ++i) {{
        \\    partials[int(part) * {d} + i] = acc[i];
        \\  }}
        \\}}
        \\threadgroup_barrier(mem_flags::mem_threadgroup);
        \\
        \\if (part == 0) {{
        \\  for (int i = int(lane); i < {d}; i += 32) {{
        \\    float total = 0.0f;
        \\    _Pragma("unroll")
        \\    for (int p = 0; p < K_PARTS; ++p) {{
        \\      total += partials[p * {d} + i];
        \\    }}
        \\    int j = i / {d};
        \\    int row = i - j * {d};
        \\    y[row * N + n0 + j] = T(total);
        \\  }}
        \\}}
    , .{ bn, nacc, acc_init, body, acc_sum, nacc, nacc, nacc, nacc, nacc, m, m }, 0);
}

/// Wide multi-simdgroup tile for HUGE N (the lm_head class): NSG independent
/// simdgroups ride one threadgroup purely to give the scheduler fewer,
/// heavier units — the split-K tiny-tile grid measurably thrashes there
/// (2.1x stock at M=4), while this tile holds 1.18x off the weight-stream
/// floor per the source ledger. Full-K per simdgroup, no barrier.
fn verifyQmmMsgSource(a: std.mem.Allocator, m: usize, bn: usize) std.mem.Allocator.Error![:0]const u8 {
    const nacc = bn * m;
    var body: []const u8 = "";
    for (0..m) |r| {
        body = try vqmmCat(a, body, try std.fmt.allocPrint(a, "  Vec8 v{d} = xv[({d} * K + k_base) / 8];\n", .{ r, r }));
    }
    for (0..bn) |j| {
        body = try vqmmCat(a, body, try vqmmColumnLoad(a, j));
    }
    for (0..bn) |j| {
        body = try vqmmCat(a, body, try std.fmt.allocPrint(a, "  float s{d} = float(scales[(n0 + {d}) * K_by_gs + gi]); float b{d} = float(biases[(n0 + {d}) * K_by_gs + gi]);\n", .{ j, j, j, j }));
    }
    for (0..bn) |j| {
        body = try vqmmCat(a, body, try vqmmColumnChain(a, m, j));
    }
    var acc_init: []const u8 = "";
    for (0..nacc) |i| {
        acc_init = try vqmmCat(a, acc_init, try std.fmt.allocPrint(a, "acc[{d}] = 0.0f; ", .{i}));
    }
    var acc_sum: []const u8 = "";
    for (0..nacc) |i| {
        acc_sum = try vqmmCat(a, acc_sum, try std.fmt.allocPrint(a, "acc[{d}] = simd_sum(acc[{d}]); ", .{ i, i }));
    }
    return std.fmt.allocPrintSentinel(a,
        \\auto sg = simdgroup_index_in_threadgroup;
        \\auto lane = thread_index_in_simdgroup;
        \\auto tg_n = threadgroup_position_in_grid.y;
        \\
        \\int K = int(K_size);
        \\int N = int(N_size);
        \\int K_by_p = K / 8;
        \\int K_bytes = (K * BITS) / 8;
        \\int K_by_gs = K / GS;
        \\int n0 = (int(tg_n) * NSG + int(sg)) * {d};
        \\if (n0 + {d} >= N) {{ return; }}
        \\
        \\float acc[{d}];
        \\{s}
        \\
        \\using Vec8 = vec<T, 8>;
        \\const device Vec8 *xv = (const device Vec8*)x;
        \\
        \\for (int pack = int(lane); pack < K_by_p; pack += 32) {{
        \\  int k_base = pack * 8;
        \\  int gi = k_base / GS;
        \\{s}
        \\}}
        \\
        \\{s}
        \\
        \\if (lane < {d}) {{
        \\  int j = int(lane) / {d};
        \\  int row = int(lane) - j * {d};
        \\  y[row * N + n0 + j] = T(acc[int(lane)]);
        \\}}
    , .{ bn, bn - 1, nacc, acc_init, body, acc_sum, nacc, m, m }, 0);
}

/// msg column-tile width per M: 4 columns through M=6, 2 for M=7 (14
/// accumulators — under the 24 ceiling; covers the depth-6 verify lm_head).
fn vqmmMsgBn(m: c_int) c_int {
    return if (m <= 6) 4 else 2;
}

/// Kernel text is built at runtime into a scratch arena: comptime string work for 24 tiles costs seconds of Sema.
fn vqmmCat(a: std.mem.Allocator, x: []const u8, y: []const u8) std.mem.Allocator.Error![]const u8 {
    return std.mem.concat(a, u8, &.{ x, y });
}

const VQMM_MSG_NAMES = [6][*:0]const u8{
    "sushi_vqmm_msg_m2", "sushi_vqmm_msg_m3", "sushi_vqmm_msg_m4",
    "sushi_vqmm_msg_m5", "sushi_vqmm_msg_m6", "sushi_vqmm_msg_m7",
};

var vqmm_msg_kernels: [6]?mlx.mlx_fast_metal_kernel = @splat(null);

fn getVerifyQmmMsgKernel(m: c_int) !mlx.mlx_fast_metal_kernel {
    if (m < 2 or m > 7) return error.UnsupportedShape;
    const idx: usize = @intCast(m - 2);
    if (vqmm_msg_kernels[idx]) |k| return k;
    const input_names = [_][*:0]const u8{ "x", "w_q", "scales", "biases", "K_size", "N_size" };
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const source = try verifyQmmMsgSource(arena.allocator(), @intCast(m), @intCast(vqmmMsgBn(m)));
    const kernel = mlx.mlx_fast_metal_kernel_new(
        VQMM_MSG_NAMES[idx],
        in_vec,
        out_vec,
        source,
        "",
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    vqmm_msg_kernels[idx] = kernel;
    return kernel;
}

/// Column-tile width per M. A threadgroup owns `bn` output columns and
/// streams ALL M activation rows over its K partition, so the x traffic is
/// `N/bn * K * M * 2` bytes against `N * K * bits/8` of weight — at M=5,
/// K=6656, N=19968, bn=4 that is 332 MB of activation reads for 66 MB of
/// weights. Widening the tile is the direct lever on that ratio; the ceiling
/// is the accumulator budget (`bn * m` live floats per thread) and N's
/// divisibility by bn. Measured per (M, bn) by the width-sweep µbench.
pub const VQMM_BN_CHOICES = [_]c_int{ 2, 4, 8 };

/// Test seam: forces a column tile (0 = per-M default) so the parity test drives every tile.
pub var vqmm_bn_test_override: ?c_int = null;
fn vqmmBnOverride() c_int {
    return vqmm_bn_test_override orelse 0;
}

/// Column tile per M: 4 columns through M=6 (24 accumulators, the measured
/// register ceiling), 2 for M=7 (14) — one deep-verify width past MTPLX's
/// family. M=8 measured a CLIFF at bn=4 (T(7) round 636 ms vs 115 stock —
/// spill/occupancy past 7 live Vec8 row vectors), so 8+ falls back to stock.
///
/// WIDENING THE TILE IS DISPROVEN. The activation-traffic arithmetic above
/// predicts bn=8 should nearly halve the dominant read; measured on 4-bit
/// trunk shapes it is 5-8% SLOWER at every width (M=5: 45.9 ms vs 43.0
/// whole-forward, same direction at M 2/3/4). The x rows are 66 KB and stay
/// cache-resident, so the "traffic" was never leaving the cache — the extra
/// accumulators just cost occupancy. The ceiling below admits the 8-column
/// tile only through the test override.
pub const VQMM_MAX_ACC: c_int = 40;

/// Simdgroups per threadgroup, each owning a K partition. Wide-N shapes
/// already emit N/bn threadgroups, so the grid does not need more
/// parallelism there — 2 keeps the reduction cheap, and the sweep agrees:
/// on 4-bit trunk shapes at M=5 the whole-forward estimate is 42.4 ms at
/// 2, 42.5 at 1, 44.5 at 4, 49.0 at 8.
fn vqmmKParts(N: c_int) c_int {
    return if (N >= 4096) 2 else 4;
}

pub fn vqmmBnFor(m: c_int, N: c_int) c_int {
    const forced = vqmmBnOverride();
    if (forced != 0 and @mod(N, forced) == 0 and forced * m <= VQMM_MAX_ACC) return forced;
    const want: c_int = if (m <= 6) 4 else 2;
    return if (@mod(N, want) == 0) want else 0;
}

fn vqmmSourceIdx(m: c_int, bn: c_int) usize {
    const mi: usize = @intCast(m - 2);
    const bi: usize = switch (bn) {
        2 => 0,
        4 => 1,
        else => 2,
    };
    return mi * VQMM_BN_CHOICES.len + bi;
}

const VQMM_N_KERNELS = 6 * VQMM_BN_CHOICES.len;

const VQMM_NAMES = blk: {
    var out: [VQMM_N_KERNELS][*:0]const u8 = undefined;
    for (2..8) |m| {
        for (VQMM_BN_CHOICES, 0..) |bn, bi| {
            out[(m - 2) * VQMM_BN_CHOICES.len + bi] =
                std.fmt.comptimePrint("sushi_vqmm_ks_m{d}_n{d}", .{ m, bn });
        }
    }
    break :blk out;
};

var vqmm_kernels: [VQMM_N_KERNELS]?mlx.mlx_fast_metal_kernel = @splat(null);

fn getVerifyQmmKernel(m: c_int, bn: c_int) !mlx.mlx_fast_metal_kernel {
    if (m < 2 or m > 7) return error.UnsupportedShape;
    const idx = vqmmSourceIdx(m, bn);
    if (vqmm_kernels[idx]) |k| return k;
    const input_names = [_][*:0]const u8{ "x", "w_q", "scales", "biases", "K_size", "N_size" };
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const source = try verifyQmmSource(arena.allocator(), @intCast(m), @intCast(bn));
    const name: [*:0]const u8 = VQMM_NAMES[idx];
    const kernel = mlx.mlx_fast_metal_kernel_new(
        name,
        in_vec,
        out_vec,
        source,
        "",
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    vqmm_kernels[idx] = kernel;
    return kernel;
}

/// Process-lifetime cache of 0-d int scalars for the kernel's K/N inputs —
/// only a handful of distinct layer geometries exist per model, and the
/// verify path builds ~300 kernel nodes per round (inference thread only,
/// same single-caller discipline as the kernel caches above).
const VqmmScalar = struct { v: c_int, arr: mlx.mlx_array, tick: u64 };
var vqmm_scalar_cache: [16]VqmmScalar = undefined;
var vqmm_scalar_count: usize = 0;
var vqmm_scalar_tick: u64 = 0;

/// Which slot may be evicted: the LEAST RECENTLY USED one, never a fixed slot.
///
/// The old policy always evicted slot 0, and that is a use-after-free. Every
/// caller takes TWO scalars back to back (`K_arr` then `N_arr`) and holds BOTH
/// in one `inputs_arr`. With the cache full, the K call evicts slot 0 and lands
/// there; the N call then frees slot 0 again — which is now K's array — so the
/// caller carried a dangling handle into `mlx_vector_array_new_data`. Observed
/// on a 40-layer GatedDeltaNet + MoE trunk with a 248k
/// vocab (enough distinct (K, N) pairs to overflow 16 slots) as SIGBUS building
/// the vector and SIGSEGV freeing it — one request killed the server, and the
/// surfaces probed after it all read as 0%.
///
/// The old comment claimed eviction was safe because "in-flight lazy graph
/// nodes hold their own refs". True, and irrelevant: the array has not reached
/// a graph node yet, it is still sitting in the caller's local array.
///
/// LRU is correct by construction here — the scalars a call site is still
/// holding are by definition the most recently ticked, so they can never be
/// the minimum while the cache (16) is far larger than the most any one site
/// takes (2).
fn vqmmScalarEvictIndex(entries: []const VqmmScalar) usize {
    var best: usize = 0;
    for (entries, 0..) |e, i| {
        if (e.tick < entries[best].tick) best = i;
    }
    return best;
}

pub fn cachedScalarInt(v: c_int) mlx.mlx_array {
    vqmm_scalar_tick += 1;
    for (vqmm_scalar_cache[0..vqmm_scalar_count]) |*e| {
        if (e.v == v) {
            e.tick = vqmm_scalar_tick;
            return e.arr;
        }
    }
    const arr = mlx.mlx_array_new_int(v);
    if (vqmm_scalar_count < vqmm_scalar_cache.len) {
        vqmm_scalar_cache[vqmm_scalar_count] = .{ .v = v, .arr = arr, .tick = vqmm_scalar_tick };
        vqmm_scalar_count += 1;
    } else {
        const victim = vqmmScalarEvictIndex(vqmm_scalar_cache[0..vqmm_scalar_count]);
        _ = mlx.mlx_array_free(vqmm_scalar_cache[victim].arr);
        vqmm_scalar_cache[victim] = .{ .v = v, .arr = arr, .tick = vqmm_scalar_tick };
    }
    return arr;
}

pub var verify_qmm_override: ?bool = null;
pub fn verifyQmmEnabled() bool {
    return verify_qmm_override orelse true;
}

pub const VqmmLane = enum { none, splitk, msg, nax };

var vqmm_nax_engaged = false;

/// The byte-unpack NAX tile is profitable for wide q5/q6/q8 projections, but
/// loses to stock MLX on the 1024-wide K/V projections. This threshold is the
/// measured split on M5 Max at M=8; q4 keeps its existing full geometry.
/// Qwen3.8-27B 8-bit/g64, forced depth 8, six paired settled boots: 89.81
/// tok/s NAX vs 79.51 stock (+12.96%); every arm asserted its own engagement.
pub fn mixedNaxShapeEnabled(bits: u32, group_size: u32, n: c_int) bool {
    if (bits == 4) return true;
    return (bits == 5 or bits == 6 or bits == 8) and group_size == 64 and n >= 5120;
}

/// Whether the PLAIN-SIMD tiles (split-K / msg) adopt a non-4-bit affine
/// weight. Profitability is a SEPARATE question from "the kernel is correct":
/// the byte-addressed unpack does strictly more ALU per value than the 4-bit
/// nibble path, so the shapes where it beats stock differ from the NAX tile's
/// (`mixedNaxShapeEnabled` exists because that tile LOSES on the 1024-wide K/V
/// projections).
///
/// MEASURED, Qwen3.8-27B on an M4 Max (no NAX lane, so the plain-SIMD tiles are
/// all this machine has), MTP on, decode tok/s from the server's own timings,
/// per-prompt medians over two passes with the CELL order reversed on the
/// second (the first cell of a pass reads ~2% fast, which a control pack that
/// ignores the lever entirely made visible):
///
/// | pack | code | prose | agentish |
/// |---|---|---|---|
/// | 6-bit gs64 | **1.10x** | **1.22x** | **1.17x** |
/// | 8-bit gs64 | 1.03x | 1.04x | 1.01x |
///
/// For scale: killing the 4-bit lane outright on the same box costs
/// 1.09-1.12x, so the 6-bit lane is worth MORE than the 4-bit one it was
/// modelled on — stock's 6-bit qmm has more to give back at these widths.
///
/// So 6-bit is default-on and 8-bit is NOT: 0.7-3.7% is inside the noise this
/// harness can resolve, and a lane that only adds dispatch surface for that is
/// not worth defaulting. 5-bit has the kernel arm and the parity test but no
/// pack on this box to measure, so it stays off for the same reason rather
/// than riding 6-bit's number.
///
/// The N floor is where the whole result lives: forcing every shape on
/// (which adopts the 1024-wide K/V projections too) measured
/// only 1.01-1.05x on the same pack. Excluding them is worth ~10 points.
pub fn mixedPlainShapeEnabled(bits: u32, group_size: u32, n: c_int) bool {
    if (bits == 4) return true;
    if (bits != 6) return false;
    return group_size == 64 and n >= MIXED_PLAIN_MIN_N;
}

/// The N below which the byte-addressed unpack loses to stock — the same split
/// the NAX tile measured, and the reason force-all-shapes gives back most of
/// the win.
pub const MIXED_PLAIN_MIN_N: c_int = 5120;

var vqmm_mixed_plain_engaged: bool = false;

/// Test seam: the parity test drives the plain-SIMD mixed-width arms without
/// touching the process environment (correctness is not a perf question).
pub var vqmm_mixed_plain_test_force: ?bool = null;

pub fn mixedPlainEnabled(bits: u32, group_size: u32, n: c_int) bool {
    if (vqmm_mixed_plain_test_force) |v| return v;
    return mixedPlainShapeEnabled(bits, group_size, n);
}

/// Pure lane selection for the verify-qmm family (hermetically pinned by
/// the NAX dispatch-table test). Shared floors: M 2..16, N >= 512 (tiny-N
/// projections — GDN ba proj, MoE routers — gain nothing over stock qmv
/// while the per-call kernel-node build ~10us is real; ~50 such calls ride
/// every verify round). Lanes:
/// - nax: M >= nax_min_m (default 8, see `naxMinMForBits`) when
///   the M5-class probe is live AND the m16 tile's stricter geometry holds
///   (K % 256 == 0, N % 32 == 0) — the lm_head N=151936 qualifies natively,
///   so no msg variant is needed past M=7.
/// - splitk/msg: the plain-SIMD M 2..7 lanes (K % 64 == 0, N % 4 == 0),
///   byte-identical to the pre-NAX dispatch; msg takes N >= 100000.
/// - none: stock mlx_quantized_matmul.
pub fn vqmmLaneFor(m: c_int, K: c_int, N: c_int, nax_on: bool, nax_min_m: c_int) VqmmLane {
    if (m < 2 or m > 16) return .none;
    if (N < 512) return .none;
    if (nax_on and m >= nax_min_m and @mod(K, 256) == 0 and @mod(N, 32) == 0) return .nax;
    if (m > 7) return .none;
    if (@mod(K, 64) != 0 or @mod(N, 4) != 0) return .none;
    return if (N >= 100000) .msg else .splitk;
}

/// Verify-width qmm dispatch (three lanes; see vqmmLaneFor). Returns the
/// [.., M, N] product when the shape is eligible — split-K (M 2..7),
/// msg wide tile (M 2..7 at huge N), or the NAX m16 tile (M 8..16,
/// M5-class machines only) — null to fall through to stock. 4-bit affine,
/// gs in {32,64,128}, bf16/fp16 activations.
pub fn verifyQmm(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    bi: mlx.mlx_array,
    bits: u32,
    group_size: u32,
) !?mlx.mlx_array {
    if (!verifyQmmEnabled()) return null;
    // The plain-SIMD split-K/msg kernels carry byte-addressed 5/6/8-bit unpack,
    // but mixed-width adoption remains restricted to measured width/shape
    // combinations. The M5 NAX tile additionally handles eligible mixed-bit
    // projections at and above its takeover row.
    if (bits != 4 and bits != 5 and bits != 6 and bits != 8) return null;
    if (group_size != 32 and group_size != 64 and group_size != 128) return null;
    if (sc.ctx == null or bi.ctx == null) return null;
    const xd = mlx.mlx_array_dtype(x);
    if (xd != .bfloat16 and xd != .float16) return null;
    const xsh = mlx.getShape(x);
    if (xsh.len < 2) return null;
    const K: c_int = xsh[xsh.len - 1];
    var m: c_int = 1;
    for (xsh[0 .. xsh.len - 1]) |d| m *= d;
    const wsh = mlx.getShape(w);
    if (wsh.len != 2) return null;
    const N: c_int = wsh[0];
    // Affine packed geometry sanity: one uint32 column carries 32/bits source
    // values (5/6-bit rows are byte-packed but still exposed as uint32 arrays).
    if (@as(i64, wsh[1]) * 32 != @as(i64, K) * @as(i64, bits)) return null;
    const nax_on = verifyQmmNaxAvailable();
    const lane = vqmmLaneFor(m, K, N, nax_on, naxMinMForBits(bits));
    if (bits != 4) switch (lane) {
        .nax => if (!mixedNaxShapeEnabled(bits, group_size, N)) return null,
        // The plain-SIMD tiles now carry the same byte-addressed unpack the
        // NAX tile validated, so 5/6/8-bit no longer fall through to stock on
        // every M1-M4 machine — but adoption is per-shape and measured.
        .splitk, .msg => {
            if (!mixedPlainEnabled(bits, group_size, N)) return null;
            // A lane that silently declines is indistinguishable from one that
            // is merely slow, so every A/B arm has to be readable from its own
            // log rather than from the env it was launched with.
            if (!vqmm_mixed_plain_engaged) {
                vqmm_mixed_plain_engaged = true;
                log.info("[vqmm] plain-SIMD verify lane engaged at {d}-bit: lane={s} M={d} K={d} N={d} gs={d}\n", .{ bits, @tagName(lane), m, K, N, group_size });
            }
        },
        .none => return null,
    };
    switch (lane) {
        .none => return null,
        // NAX m16 tile (M5-class): M 8..16 by default — past the plain-SIMD
        // register cliff. Construction stays strictly behind the probe.
        .nax => return try runVerifyQmmNax(s, x, w, sc, bi, bits, group_size, m, K, N, xd, xsh),
        // Huge-N (lm_head class): the tiny-tile split-K grid thrashes the
        // scheduler there (measured 2.1x stock at M=4) — route through the
        // wide multi-simdgroup tile instead (2-column tile at M=7).
        .msg => return try runVerifyQmmMsg(s, x, w, sc, bi, bits, group_size, m, K, N, xd, xsh),
        .splitk => {},
    }

    // x rides in with its original shape — the kernel indexes the buffer
    // linearly, and a contiguous [.., M, K] has the same layout as [M, K]
    // (ensure_row_contiguous copies the rare non-contiguous case).
    const k_parts: c_int = vqmmKParts(N);
    const bn = vqmmBnFor(m, N);
    if (bn == 0) return null;
    const K_arr = cachedScalarInt(K);
    const N_arr = cachedScalarInt(N);

    const config = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    const y_shape = [_]c_int{ m, N };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &y_shape, 2, xd));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32 * k_parts, @divExact(N, bn), 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32 * k_parts, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", xd));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GS", @intCast(group_size)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BITS", @intCast(bits)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "K_PARTS", k_parts));

    const inputs_arr = [_]mlx.mlx_array{ x, w, sc, bi, K_arr, N_arr };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);

    const kernel = try getVerifyQmmKernel(m, bn);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, config, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var y2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y2);
    try mlx.check(mlx.mlx_vector_array_get(&y2, outputs_vec, 0));

    // Restore the caller's leading shape with N as the last dim.
    var out_shape_buf: [8]c_int = undefined;
    const ndim = xsh.len;
    if (ndim > out_shape_buf.len) return error.UnsupportedShape;
    for (xsh[0 .. ndim - 1], 0..) |d, i| out_shape_buf[i] = d;
    out_shape_buf[ndim - 1] = N;
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_reshape(&y, y2, out_shape_buf[0..ndim].ptr, @intCast(ndim), s));
    return y;
}

const VQMM_MSG_NSG: c_int = 8; // simdgroups per threadgroup (their sweep winner)

fn runVerifyQmmMsg(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    bi: mlx.mlx_array,
    bits: u32,
    group_size: u32,
    m: c_int,
    K: c_int,
    N: c_int,
    xd: mlx.mlx_dtype,
    xsh: []const c_int,
) !?mlx.mlx_array {
    const K_arr = cachedScalarInt(K);
    const N_arr = cachedScalarInt(N);

    const config = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    const y_shape = [_]c_int{ m, N };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &y_shape, 2, xd));
    const cols: c_int = vqmmMsgBn(m) * VQMM_MSG_NSG;
    const tg_count: c_int = @divTrunc(N + cols - 1, cols); // in-kernel n0 guard covers the tail
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 32 * VQMM_MSG_NSG, tg_count, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 32 * VQMM_MSG_NSG, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", xd));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GS", @intCast(group_size)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BITS", @intCast(bits)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "NSG", VQMM_MSG_NSG));

    const inputs_arr = [_]mlx.mlx_array{ x, w, sc, bi, K_arr, N_arr };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);

    const kernel = try getVerifyQmmMsgKernel(m);
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, config, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var y2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y2);
    try mlx.check(mlx.mlx_vector_array_get(&y2, outputs_vec, 0));

    var out_shape_buf: [8]c_int = undefined;
    const ndim = xsh.len;
    if (ndim > out_shape_buf.len) return error.UnsupportedShape;
    for (xsh[0 .. ndim - 1], 0..) |d, i| out_shape_buf[i] = d;
    out_shape_buf[ndim - 1] = N;
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_reshape(&y, y2, out_shape_buf[0..ndim].ptr, @intCast(ndim), s));
    return y;
}

const VQMM_NAX_HEADER: [:0]const u8 =
    \\#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
    \\
;

/// MTPLX's _build_kernel_m16_nax_ktmpl body VERBATIM (nax_verify.py,
/// Apache-2.0; single-brace form of their f-string). The tensor
/// extents/strides encode dflash's working matmul2d convention — treat
/// them as opaque and correct, do not re-derive. Geometry: threadgroup =
/// 256 threads = 8 simdgroups; each simdgroup owns a K/8 chunk and stages
/// a dequantized 16x32 B tile in threadgroup memory (each of its 32 lanes
/// dequants ONE output column's 16 K-values from MLX's packed byte stream per
/// iteration); matmul2d (16x32x16 multiply_accumulate, execution_simdgroup)
/// multiplies the row-padded activation tile against it into a cooperative
/// fp32 accumulator; after the K loop the 8 partial C tiles reduce across
/// the threadgroup. T/GS/KCONST arrive as template args (MLX name-mangles
/// template values into the host_name, so one kernel object serves every
/// specialization — the GDN Dk/Dv precedent).
const VQMM_NAX_SOURCE: [:0]const u8 =
    \\using namespace metal;
    \\using namespace mpp::tensor_ops;
    \\
    \\constexpr int BM = 16;
    \\constexpr int BN = 32;
    \\constexpr int BK = 16;
    \\constexpr int NSG = 8;
    \\constexpr int K = KCONST;
    \\constexpr int K_bytes = K * BITS / 8;
    \\constexpr int K_by_gs = K / GS;
    \\constexpr int K_chunk = K / NSG;
    \\
    \\uint tid = thread_position_in_threadgroup.x;
    \\uint sg_id = simdgroup_index_in_threadgroup;
    \\uint lane = thread_index_in_simdgroup;
    \\uint tg_n = threadgroup_position_in_grid.y;
    \\int N = int(N_size);
    \\int n0 = int(tg_n) * BN;
    \\int k_begin = int(sg_id) * K_chunk;
    \\int k_end = k_begin + K_chunk;
    \\
    \\threadgroup T B_tile[NSG][BK * BN];
    \\threadgroup float partial[NSG][BM * BN];
    \\
    \\constexpr auto desc = matmul2d_descriptor(
    \\    16,
    \\    32,
    \\    16,
    \\    false,
    \\    false,
    \\    false,
    \\    matmul2d_descriptor::mode::multiply_accumulate);
    \\matmul2d<desc, metal::execution_simdgroup> op;
    \\
    \\tensor<device T, dextents<int, 2>, tensor_inline> A(
    \\    (device T*)x,
    \\    dextents<int, 2>{K, BM},
    \\    array<int, 2>{1, K});
    \\tensor<threadgroup T, dextents<int, 2>, tensor_inline> B(
    \\    B_tile[sg_id],
    \\    dextents<int, 2>{BN, BK},
    \\    array<int, 2>{1, BN});
    \\tensor<threadgroup float, dextents<int, 2>, tensor_inline> C(
    \\    partial[sg_id],
    \\    dextents<int, 2>{BN, BM},
    \\    array<int, 2>{1, BN});
    \\
    \\auto ct_c = op.template get_destination_cooperative_tensor<
    \\    tensor<device T, extents<int, 16, 16>, tensor_inline>,
    \\    tensor<threadgroup T, extents<int, 32, 16>, tensor_inline>,
    \\    float>();
    \\_Pragma("unroll")
    \\for (uint16_t i = 0; i < ct_c.get_capacity(); ++i) {
    \\    ct_c[i] = 0.0f;
    \\}
    \\
    \\int n_global = n0 + int(lane);
    \\for (int k0 = k_begin; k0 < k_end; k0 += BK) {
    \\    const device uchar* wp =
    \\        ((const device uchar*)w_q) + n_global * K_bytes + (k0 * BITS) / 8;
    \\    float scale = float(scales[n_global * K_by_gs + (k0 / GS)]);
    \\    float bias = float(biases[n_global * K_by_gs + (k0 / GS)]);
    \\
    \\    if constexpr (BITS == 4) {
    \\        _Pragma("unroll")
    \\        for (int pack = 0; pack < 2; ++pack) {
    \\            uint32_t p =
    \\                uint32_t(wp[pack * 4 + 0]) |
    \\                (uint32_t(wp[pack * 4 + 1]) << 8) |
    \\                (uint32_t(wp[pack * 4 + 2]) << 16) |
    \\                (uint32_t(wp[pack * 4 + 3]) << 24);
    \\            _Pragma("unroll")
    \\            for (int ki = 0; ki < 8; ++ki) {
    \\                uint32_t q = (p >> (ki * 4)) & 0xFu;
    \\                B_tile[sg_id][(pack * 8 + ki) * BN + int(lane)] =
    \\                    T(float(q) * scale + bias);
    \\            }
    \\        }
    \\    } else if constexpr (BITS == 5) {
    \\        _Pragma("unroll")
    \\        for (int pack = 0; pack < 2; ++pack) {
    \\            ulong p =
    \\                ulong(wp[pack * 5 + 0]) |
    \\                (ulong(wp[pack * 5 + 1]) << 8) |
    \\                (ulong(wp[pack * 5 + 2]) << 16) |
    \\                (ulong(wp[pack * 5 + 3]) << 24) |
    \\                (ulong(wp[pack * 5 + 4]) << 32);
    \\            _Pragma("unroll")
    \\            for (int ki = 0; ki < 8; ++ki) {
    \\                uint32_t q = uint32_t((p >> (ki * 5)) & 0x1Ful);
    \\                B_tile[sg_id][(pack * 8 + ki) * BN + int(lane)] =
    \\                    T(float(q) * scale + bias);
    \\            }
    \\        }
    \\    } else if constexpr (BITS == 6) {
    \\        _Pragma("unroll")
    \\        for (int pack = 0; pack < 4; ++pack) {
    \\            uint32_t p =
    \\                uint32_t(wp[pack * 3 + 0]) |
    \\                (uint32_t(wp[pack * 3 + 1]) << 8) |
    \\                (uint32_t(wp[pack * 3 + 2]) << 16);
    \\            _Pragma("unroll")
    \\            for (int ki = 0; ki < 4; ++ki) {
    \\                uint32_t q = (p >> (ki * 6)) & 0x3Fu;
    \\                B_tile[sg_id][(pack * 4 + ki) * BN + int(lane)] =
    \\                    T(float(q) * scale + bias);
    \\            }
    \\        }
    \\    } else {
    \\        static_assert(BITS == 8, "unsupported NAX affine width");
    \\        _Pragma("unroll")
    \\        for (int ki = 0; ki < BK; ++ki) {
    \\            uint32_t q = uint32_t(wp[ki]);
    \\            B_tile[sg_id][ki * BN + int(lane)] =
    \\                T(float(q) * scale + bias);
    \\        }
    \\    }
    \\    simdgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\    auto tA = A.template slice<16, 16>(k0, 0);
    \\    auto tB = B.template slice<32, 16>(0, 0);
    \\    op.run(tA, tB, ct_c);
    \\    simdgroup_barrier(mem_flags::mem_threadgroup);
    \\}
    \\
    \\auto tC = C.template slice<32, 16>(0, 0);
    \\ct_c.store(tC);
    \\threadgroup_barrier(mem_flags::mem_threadgroup);
    \\
    \\for (int off = int(tid); off < BM * BN; off += NSG * 32) {
    \\    float acc01 = partial[0][off] + partial[1][off];
    \\    float acc23 = partial[2][off] + partial[3][off];
    \\    float acc45 = partial[4][off] + partial[5][off];
    \\    float acc67 = partial[6][off] + partial[7][off];
    \\    float acc = (acc01 + acc23) + (acc45 + acc67);
    \\    int row = off / BN;
    \\    int col = off - row * BN;
    \\    y[row * N + n0 + col] = T(acc);
    \\}
;

var vqmm_nax_kernel: ?mlx.mlx_fast_metal_kernel = null;

/// ONLY call where verifyQmmNaxAvailable() — see the never-build rule.
fn getVerifyQmmNaxKernel() !mlx.mlx_fast_metal_kernel {
    if (vqmm_nax_kernel) |k| return k;
    const input_names = [_][*:0]const u8{ "x", "w_q", "scales", "biases", "N_size" };
    const output_names = [_][*:0]const u8{"y"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const kernel = mlx.mlx_fast_metal_kernel_new(
        "sushi_vqmm_nax_m16",
        in_vec,
        out_vec,
        VQMM_NAX_SOURCE,
        VQMM_NAX_HEADER,
        true,
        false,
    );
    if (kernel.ctx == null) return error.MetalKernelCompileFailed;
    vqmm_nax_kernel = kernel;
    return kernel;
}

/// The M-padding half of the NAX host contract, split out so it can be
/// exercised hermetically on non-M5 machines (the scaffolding test wraps
/// it around STOCK qmm — zero pad rows must produce zero output rows):
/// collapse the caller's leading shape to [m, K] (mirrors MTPLX's
/// x.reshape(m, k) so row padding is well-defined for every batch shape),
/// then zero-pad the row axis to the tile's fixed 16. Owned handle.
fn naxPadTo16(s: mlx.mlx_stream, x: mlx.mlx_array, m: c_int, K: c_int, xd: mlx.mlx_dtype) !mlx.mlx_array {
    var x2 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(x2);
    const x2_shape = [_]c_int{ m, K };
    try mlx.check(mlx.mlx_reshape(&x2, x, &x2_shape, 2, s));
    var x16 = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(x16);
    if (m < 16) {
        var zero = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(zero);
        const zdims = [_]c_int{};
        try mlx.check(mlx.mlx_zeros(&zero, &zdims, 0, xd, s));
        const pad_axes = [_]c_int{0};
        const pad_low = [_]c_int{0};
        const pad_high = [_]c_int{16 - m};
        try mlx.check(mlx.mlx_pad(&x16, x2, &pad_axes, 1, &pad_low, 1, &pad_high, 1, zero, "constant", s));
    } else {
        try mlx.check(mlx.mlx_array_set(&x16, x2));
    }
    return x16;
}

/// The slice-back half: first m rows of the [16, N] tile output (pad rows
/// dropped). Owned handle.
fn naxSliceRows(s: mlx.mlx_stream, y16: mlx.mlx_array, m: c_int, N: c_int) !mlx.mlx_array {
    var ym = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(ym);
    if (m < 16) {
        const start = [_]c_int{ 0, 0 };
        const stop = [_]c_int{ m, N };
        const strides = [_]c_int{ 1, 1 };
        try mlx.check(mlx.mlx_slice(&ym, y16, &start, 2, &stop, 2, &strides, 2, s));
    } else {
        try mlx.check(mlx.mlx_array_set(&ym, y16));
    }
    return ym;
}

/// Run the NAX m16 tile: pad the activations to the fixed [16, K] tile
/// ("weight streaming dominates so padded rows are nearly free" — the
/// source ledger), grid (256, N/32), slice y back to the caller's M rows.
fn runVerifyQmmNax(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    w: mlx.mlx_array,
    sc: mlx.mlx_array,
    bi: mlx.mlx_array,
    bits: u32,
    group_size: u32,
    m: c_int,
    K: c_int,
    N: c_int,
    xd: mlx.mlx_dtype,
    xsh: []const c_int,
) !?mlx.mlx_array {
    if (!vqmm_nax_engaged) {
        vqmm_nax_engaged = true;
        log.info("[vqmm] NAX verify lane engaged at {d}-bit: M={d} K={d} N={d} gs={d} min_m={d} (SUSHI_VERIFY_QMM=0 restores stock)\n", .{ bits, m, K, N, group_size, naxMinMForBits(bits) });
    }
    const x16 = try naxPadTo16(s, x, m, K, xd);
    defer _ = mlx.mlx_array_free(x16);

    const N_arr = cachedScalarInt(N);

    const config = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(config);
    const y_shape = [_]c_int{ 16, N };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(config, &y_shape, 2, xd));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(config, 256, @divExact(N, 32), 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(config, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(config, "T", xd));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "BITS", @intCast(bits)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "GS", @intCast(group_size)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(config, "KCONST", K));

    const inputs_arr = [_]mlx.mlx_array{ x16, w, sc, bi, N_arr };
    const inputs_vec = mlx.mlx_vector_array_new_data(&inputs_arr, inputs_arr.len);
    defer _ = mlx.mlx_vector_array_free(inputs_vec);

    const kernel = try getVerifyQmmNaxKernel();
    var outputs_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs_vec);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs_vec, kernel, inputs_vec, config, s));
    if (mlx.mlx_vector_array_size(outputs_vec) != 1) return error.MetalKernelBadOutputCount;
    var y16 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(y16);
    try mlx.check(mlx.mlx_vector_array_get(&y16, outputs_vec, 0));

    // Drop the pad rows, restore the caller's leading shape with N last.
    const ym = try naxSliceRows(s, y16, m, N);
    defer _ = mlx.mlx_array_free(ym);

    var out_shape_buf: [8]c_int = undefined;
    const ndim = xsh.len;
    if (ndim > out_shape_buf.len) return error.UnsupportedShape;
    for (xsh[0 .. ndim - 1], 0..) |d, i| out_shape_buf[i] = d;
    out_shape_buf[ndim - 1] = N;
    var y = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(y);
    try mlx.check(mlx.mlx_reshape(&y, ym, out_shape_buf[0..ndim].ptr, @intCast(ndim), s));
    return y;
}

fn expectVerifyQmmNoWorseThanStock(
    s: mlx.mlx_stream,
    x: mlx.mlx_array,
    wq: mlx.mlx_array,
    wsc: mlx.mlx_array,
    wbi: mlx.mlx_array,
    bits: u32,
    gs: u32,
    wdq_t: mlx.mlx_array,
    got: mlx.mlx_array,
    label: []const u8,
) !void {
    var ref = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ref);
    try mlx.check(mlx.mlx_quantized_matmul(&ref, x, wq, wsc, wbi, true, mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(@intCast(bits)), "affine", s));

    var xf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xf);
    try mlx.check(mlx.mlx_astype(&xf, x, .float32, s));
    var truth = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(truth);
    try mlx.check(mlx.mlx_matmul(&truth, xf, wdq_t, s));

    var tf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tf);
    var rf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rf);
    var gf = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gf);
    try mlx.check(mlx.mlx_astype(&tf, truth, .float32, s));
    try mlx.check(mlx.mlx_astype(&rf, ref, .float32, s));
    try mlx.check(mlx.mlx_astype(&gf, got, .float32, s));
    try mlx.check(mlx.mlx_array_eval(tf));
    try mlx.check(mlx.mlx_array_eval(rf));
    try mlx.check(mlx.mlx_array_eval(gf));
    const td = mlx.mlx_array_data_float32(tf).?;
    const rd = mlx.mlx_array_data_float32(rf).?;
    const gd = mlx.mlx_array_data_float32(gf).?;

    const gsh = mlx.getShape(got);
    const count: usize = @intCast(gsh[1] * gsh[2]);
    var stock_max: f64 = 0;
    var kern_max: f64 = 0;
    var dot_s: f64 = 0;
    var dot_k: f64 = 0;
    var nt: f64 = 0;
    var ns: f64 = 0;
    var nk: f64 = 0;
    // First pass: cosines against the fp32 truth, plus each side's worst
    // element. Neither max is a BAR any more (see VerifyQmmParity) — stock's
    // is reported alongside the kernel's so a failure line shows both.
    for (0..count) |i| {
        const t: f64 = td[i];
        const r: f64 = rd[i];
        const g: f64 = gd[i];
        // Clamped denominator: a near-zero output of a heavily-cancelling dot
        // product must not manufacture a huge ratio from a tiny absolute error.
        const denom = @max(1.0, @abs(t));
        stock_max = @max(stock_max, @abs(r - t) / denom);
        kern_max = @max(kern_max, @abs(g - t) / denom);
        dot_s += r * t;
        dot_k += g * t;
        nt += t * t;
        ns += r * r;
        nk += g * g;
    }
    var se_s: f64 = 0;
    var se_k: f64 = 0;
    for (0..count) |i| {
        const t: f64 = td[i];
        const denom = @max(1.0, @abs(t));
        const es = (@as(f64, rd[i]) - t) / denom;
        const ek = (@as(f64, gd[i]) - t) / denom;
        se_s += es * es;
        se_k += ek * ek;
    }
    const n_f: f64 = @floatFromInt(count);
    const rms_stock = @sqrt(se_s / n_f);
    const rms_kern = @sqrt(se_k / n_f);
    const cos_stock = dot_s / (@sqrt(ns) * @sqrt(nt));
    const cos_kern = dot_k / (@sqrt(nk) * @sqrt(nt));

    if (diagEnvOn("SUSHI_VQMM_PARITY_DEBUG")) {
        std.debug.print(
            "[vqmm-parity] {s:>16}: n={d} max {d:.5}/{d:.5} rms {d:.6}/{d:.6} (ratio {d:.3}) cos {d:.7}/{d:.7}\n",
            .{ label, count, stock_max, kern_max, rms_stock, rms_kern, if (rms_stock > 0) rms_kern / rms_stock else 0, cos_stock, cos_kern },
        );
    }

    if (verifyQmmParityVerdict(kern_max, rms_stock, rms_kern, cos_stock, cos_kern)) |why| {
        std.debug.print(
            "verifyQmm parity FAIL [{s}] ({s}): n={d} kernel max={d:.4} rms={d:.6} cos={d:.6}; " ++
                "stock max={d:.4} rms={d:.6} cos={d:.6}; bars: max<={d:.2} rms<={d:.1}x cos>=stock-{d}\n",
            .{
                label,                      @tagName(why),
                count,                      kern_max,
                rms_kern,                   cos_kern,
                stock_max,                  rms_stock,
                cos_stock,                  VerifyQmmParity.GROSS_CEILING,
                VerifyQmmParity.RMS_FACTOR, VerifyQmmParity.COS_SLACK,
            },
        );
        return error.TestExpectedApproxEq;
    }
}

test "vqmm scalar cache never evicts a scalar the caller is still holding" {
    // The live crash, as a pure decision: every verifyQmm-family call site
    // takes K then N back to back and holds BOTH handles in one `inputs_arr`.
    // The old policy evicted slot 0 unconditionally, so the N call freed the
    // array the K call had just been handed — a dangling entry that killed the
    // server inside mlx_vector_array_new_data / _free.
    var cache: [16]VqmmScalar = undefined;
    for (&cache, 0..) |*e, i| {
        e.* = .{ .v = @intCast(i), .arr = .{ .ctx = null }, .tick = @intCast(i + 1) };
    }

    // Call 1 (K): the cache is full, so something is evicted and K takes its
    // slot with the newest tick.
    const k_slot = vqmmScalarEvictIndex(&cache);
    try std.testing.expectEqual(@as(usize, 0), k_slot); // oldest tick == slot 0 here
    cache[k_slot] = .{ .v = 1000, .arr = .{ .ctx = null }, .tick = 17 };

    // Call 2 (N): the victim must NOT be the slot K just took — the caller is
    // still holding that handle. Under the old always-slot-0 policy this
    // returned 0 and freed K.
    const n_slot = vqmmScalarEvictIndex(&cache);
    try std.testing.expect(n_slot != k_slot);
    cache[n_slot] = .{ .v = 2000, .arr = .{ .ctx = null }, .tick = 18 };

    // And it generalizes: after any run of insertions, the two most recently
    // ticked entries are never the next victim.
    var tick: u64 = 19;
    var round: usize = 0;
    while (round < 40) : (round += 1) {
        const victim = vqmmScalarEvictIndex(&cache);
        var newest: u64 = 0;
        var second: u64 = 0;
        for (cache) |e| {
            if (e.tick > newest) {
                second = newest;
                newest = e.tick;
            } else if (e.tick > second) second = e.tick;
        }
        try std.testing.expect(cache[victim].tick != newest);
        try std.testing.expect(cache[victim].tick != second);
        cache[victim] = .{ .v = @intCast(3000 + round), .arr = .{ .ctx = null }, .tick = tick };
        tick += 1;
    }
}

test "vqmm Metal sources are byte-identical to the shipped text" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var sha = std.crypto.hash.sha2.Sha256.init(.{});
    var bytes: usize = 0;
    for (2..8) |m| {
        for (VQMM_BN_CHOICES) |bn| {
            const src = try verifyQmmSource(arena.allocator(), m, @intCast(bn));
            sha.update(src);
            bytes += src.len;
        }
    }
    for (2..8) |m| {
        const src = try verifyQmmMsgSource(arena.allocator(), m, @intCast(vqmmMsgBn(@intCast(m))));
        sha.update(src);
        bytes += src.len;
    }
    try testing.expectEqual(@as(usize, 312_254), bytes);
    const digest = std.fmt.bytesToHex(sha.finalResult(), .lower);
    try testing.expectEqualStrings("484c91a3c6460e022df1f2a5f1f2db9d8630c50c760722b90cc4821bf66f9aea", &digest);
}

test "verifyQmm: the plain-SIMD tiles match stock qmm at 5/6/8-bit affine too" {
    try mlx.requireGpu();
    // The split-K and msg tiles were 4-bit specializations by construction, so
    // every non-4-bit pack fell through to stock on the whole M1-M4 line — the
    // Qwen3.8-27B 6-bit pack (95.0% top-1 against bf16 vs the 4-bit pack's
    // 83.6%) paid the full lane-off penalty on every MTP round. The byte-
    // addressed unpack lifted from the NAX m16 tile fixes that; this pins it.
    //
    // Same bar as the 4-bit test: no worse than stock against an fp32 dequant
    // reference, never kernel-vs-kernel. Parity is asserted independently of
    // the PROFITABILITY predicate — `vqmm_mixed_plain_test_force` opens the
    // gate so a shape the measurement later declines is still proven correct.
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xB175);
    const rnd = prng.random();

    vqmm_mixed_plain_test_force = true;
    defer vqmm_mixed_plain_test_force = null;
    // The NAX probe must not steal these rows on an M5 box: this test is about
    // the plain-SIMD lanes, and M 2..7 is below the NAX takeover anyway.
    nax_gate.vqmm_nax_probe_override = false;
    defer nax_gate.vqmm_nax_probe_override = null;

    const cases = [_]struct { k: c_int, n: c_int, gs: u32, label: []const u8 }{
        .{ .k = 5120, .n = 17408, .gs = 64, .label = "mlp" }, // MLP gate/up, split-K
        .{ .k = 2048, .n = 151936, .gs = 64, .label = "lm_head" }, // huge N, msg tile
        // The unpack is independent of the group size (gs only indexes the
        // scale/bias pair), but the forced-on A/B arm adopts every gs, so the
        // other two are proven rather than argued.
        .{ .k = 2048, .n = 5120, .gs = 32, .label = "gs32" },
        .{ .k = 2048, .n = 5120, .gs = 128, .label = "gs128" },
    };
    for ([_]u32{ 5, 6, 8 }) |bits| {
        for (cases) |cs| {
            const wb = try testRandWeightBf16(rnd, &[_]c_int{ cs.n, cs.k }, s);
            defer _ = mlx.mlx_array_free(wb);

            var triple = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(triple);
            try mlx.check(mlx.mlx_quantize(&triple, wb, mlx.mlx_optional_int.some(@intCast(cs.gs)), mlx.mlx_optional_int.some(@intCast(bits)), "affine", .{}, s));
            var wq = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wq);
            var wsc = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wsc);
            var wbi = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wbi);
            try mlx.check(mlx.mlx_vector_array_get(&wq, triple, 0));
            try mlx.check(mlx.mlx_vector_array_get(&wsc, triple, 1));
            try mlx.check(mlx.mlx_vector_array_get(&wbi, triple, 2));

            var wdq = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wdq);
            try mlx.check(mlx.mlx_dequantize(&wdq, wq, wsc, wbi, mlx.mlx_optional_int.some(@intCast(cs.gs)), mlx.mlx_optional_int.some(@intCast(bits)), "affine", .{ .ctx = null }, mlx.mlx_optional_dtype{ .has_value = true, .value = .float32 }, s));
            var wdq_t = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wdq_t);
            const t_axes = [_]c_int{ 1, 0 };
            try mlx.check(mlx.mlx_transpose_axes(&wdq_t, wdq, &t_axes, 2, s));

            for ([_]c_int{ 2, 4, 7 }) |m| {
                const xn: usize = @intCast(m * cs.k);
                const xbuf = try allocator.alloc(f32, xn);
                defer allocator.free(xbuf);
                for (xbuf) |*v| v.* = rnd.float(f32) - 0.5;
                const xshape = [_]c_int{ 1, m, cs.k };
                const x32 = mlx.mlx_array_new_data(xbuf.ptr, &xshape, 3, .float32);
                defer _ = mlx.mlx_array_free(x32);
                var x = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(x);
                try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));

                const got_opt = try verifyQmm(s, x, wq, wsc, wbi, bits, cs.gs);
                try testing.expect(got_opt != null);
                const got = got_opt.?;
                defer _ = mlx.mlx_array_free(got);
                const gsh = mlx.getShape(got);
                try testing.expectEqual(m, gsh[1]);
                try testing.expectEqual(cs.n, gsh[2]);
                try expectVerifyQmmNoWorseThanStock(s, x, wq, wsc, wbi, bits, cs.gs, wdq_t, got, cs.label);
            }

            // With the gate closed the same shape falls through to stock — the
            // profitability predicate is what decides adoption, and a lane that
            // cannot be declined is a lane that cannot be A/B'd.
            {
                vqmm_mixed_plain_test_force = false;
                defer vqmm_mixed_plain_test_force = true;
                const xbuf = try allocator.alloc(f32, @intCast(4 * cs.k));
                defer allocator.free(xbuf);
                for (xbuf) |*v| v.* = 0.1;
                const xshape = [_]c_int{ 1, 4, cs.k };
                const x32 = mlx.mlx_array_new_data(xbuf.ptr, &xshape, 3, .float32);
                defer _ = mlx.mlx_array_free(x32);
                var x = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(x);
                try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));
                try testing.expectEqual(@as(?mlx.mlx_array, null), try verifyQmm(s, x, wq, wsc, wbi, bits, cs.gs));
            }
        }
    }
}

test "mixedPlainShapeEnabled: adoption is what was MEASURED, per width and per shape" {
    const t = testing;
    // 4-bit is the original specialization and is always in.
    try t.expect(mixedPlainShapeEnabled(4, 64, 1024));

    // 6-bit is the measured win (1.10-1.22x on the Qwen3.8-27B 6-bit pack).
    try t.expect(mixedPlainShapeEnabled(6, 64, MIXED_PLAIN_MIN_N));
    try t.expect(mixedPlainShapeEnabled(6, 64, 17408));
    // Narrow projections (GDN ba, K/V) are where the byte-addressed unpack
    // loses to stock — the same shape-dependence mixedNaxShapeEnabled encodes,
    // and forcing them on gave back ~10 of the ~20 points.
    try t.expect(!mixedPlainShapeEnabled(6, 64, MIXED_PLAIN_MIN_N - 1));
    try t.expect(!mixedPlainShapeEnabled(6, 64, 1024));
    // Only gs 64 was measured.
    try t.expect(!mixedPlainShapeEnabled(6, 32, 17408));
    try t.expect(!mixedPlainShapeEnabled(6, 128, 17408));

    // 8-bit HAS a correct kernel arm and a parity test, and is still default
    // OFF: measured 1.01-1.04x, inside what this harness resolves. 5-bit has
    // the arm too but no pack on the measuring box, so it does not get to ride
    // 6-bit's number. Both are one env var from on — an unmeasured default is
    // the thing this predicate exists to prevent.
    for ([_]u32{ 5, 8 }) |b| {
        try t.expect(!mixedPlainShapeEnabled(b, 64, 17408));
    }
    // A width with no unpack arm is never adopted, whatever the shape.
    for ([_]u32{ 2, 3, 7 }) |b| try t.expect(!mixedPlainShapeEnabled(b, 64, 17408));
}

test "verifyQmm: split-K + msg + NAX verify-width kernels match stock qmm (4-bit affine)" {
    try mlx.requireGpu();
    // The spec-verify fast path: stock MLX qmm is tuned for M=1 (qmv) and
    // large M (steel); the 2..8-row verify shapes fall in a dead zone.
    // Port of MTPLX's split-K verify kernel family (verify_kernels.py,
    // Apache-2.0) — parity vs mlx_quantized_matmul at real 27B shapes.
    //
    // THE INVARIANT IS "NO LESS ACCURATE THAN STOCK", NOT "AGREES WITH STOCK".
    // Both paths accumulate in fp32 in a DIFFERENT ORDER and then round the
    // result to bf16, so the two outputs legitimately disagree by a few ULPs —
    // and on this data (K=5120 dot products of U(-0.5,0.5) → heavy cancellation)
    // each path already sits up to ~0.036 from the true value on its worst
    // element, purely from that bf16 output rounding. A direct kernel-vs-stock
    // bound therefore pins AGREEMENT tighter than bf16 permits: the original
    // `max_rel <= 0.02` / `cos > 0.99999` thresholds were tuned on one GPU and
    // the CI runner's GPU (different rounding, same correctness) failed them at
    // max_rel 0.0977 / cos 0.999988 — a green-here/red-there test that says
    // nothing about correctness. Measured: kernel-vs-truth equals stock-vs-truth
    // to four decimals on every shape.
    //
    // So both paths are measured against an fp32 DEQUANT REFERENCE (the same
    // 4-bit weights, dequantized, matmul'd in fp32 — the ground truth both are
    // approximating) and the kernel must be no worse than the stock kernel it
    // replaces. That is machine-independent, and a real defect (a partial-sum
    // race, a register spill, bad indexing) blows kernel-vs-truth far past
    // stock-vs-truth while a rounding-order difference cannot.
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x5EED);
    const rnd = prng.random();

    const cases = [_]struct { k: c_int, n: c_int, gs: u32 }{
        .{ .k = 5120, .n = 17408, .gs = 64 }, // MLP gate/up
        .{ .k = 5120, .n = 1024, .gs = 32 }, // small-N above the gate, MTP gs-32 class
        .{ .k = 2048, .n = 5120, .gs = 128 }, // gs-128 arm
    };
    for (cases) |cs| {
        // Random dense weight → 4-bit affine triple.
        const wb = try testRandWeightBf16(rnd, &[_]c_int{ cs.n, cs.k }, s);
        defer _ = mlx.mlx_array_free(wb);
        var triple = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(triple);
        try mlx.check(mlx.mlx_quantize(&triple, wb, mlx.mlx_optional_int.some(@intCast(cs.gs)), mlx.mlx_optional_int.some(4), "affine", .{}, s));
        var wq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wq);
        var wsc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wsc);
        var wbi = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wbi);
        try mlx.check(mlx.mlx_vector_array_get(&wq, triple, 0));
        try mlx.check(mlx.mlx_vector_array_get(&wsc, triple, 1));
        try mlx.check(mlx.mlx_vector_array_get(&wbi, triple, 2));

        // fp32 ground truth operand: the same 4-bit weights, dequantized.
        var wdq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wdq);
        try mlx.check(mlx.mlx_dequantize(&wdq, wq, wsc, wbi, mlx.mlx_optional_int.some(@intCast(cs.gs)), mlx.mlx_optional_int.some(4), "affine", .{ .ctx = null }, mlx.mlx_optional_dtype{ .has_value = true, .value = .float32 }, s));
        var wdq_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wdq_t);
        const t_axes = [_]c_int{ 1, 0 };
        try mlx.check(mlx.mlx_transpose_axes(&wdq_t, wdq, &t_axes, 2, s));

        // Kernel rows: split-K M 2..7 on every machine; the NAX m16 tile rows
        // (M 8..16, the padding path at 8/9/12 and the exact-16 tile) run only
        // where the M5-class probe is live (todo-m5-nax.md §9.3) — the probe
        // self-gates, so plain-SIMD machines skip them.
        const simd_ms = [_]c_int{ 2, 3, 4, 5, 6, 7 };
        const nax_ms = [_]c_int{ 8, 9, 12, 16 };
        const total_ms: usize = if (verifyQmmNaxEnabled()) simd_ms.len + nax_ms.len else simd_ms.len;
        var mi: usize = 0;
        while (mi < total_ms) : (mi += 1) {
            const m = if (mi < simd_ms.len) simd_ms[mi] else nax_ms[mi - simd_ms.len];
            const label: []const u8 = if (mi < simd_ms.len) "split-K" else "nax m16";
            const xn: usize = @intCast(m * cs.k);
            const xbuf = try allocator.alloc(f32, xn);
            defer allocator.free(xbuf);
            for (xbuf) |*v| v.* = rnd.float(f32) - 0.5;
            const xshape = [_]c_int{ 1, m, cs.k };
            const x32 = mlx.mlx_array_new_data(xbuf.ptr, &xshape, 3, .float32);
            defer _ = mlx.mlx_array_free(x32);
            var x = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x);
            try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));

            // Kernel path — must ENGAGE for every listed M, at EVERY column
            // tile the test override can force; null takes the shipped per-M default.
            for ([_]?c_int{null} ++ [_]?c_int{ 2, 4, 8 }) |bn_force| {
                if (bn_force) |bn| {
                    // The NAX tile ignores bn; only the split-K rows sweep.
                    if (mi >= simd_ms.len) continue;
                    if (@mod(cs.n, bn) != 0 or bn * m > VQMM_MAX_ACC) continue;
                }
                vqmm_bn_test_override = bn_force;
                defer vqmm_bn_test_override = null;
                const got_opt = try verifyQmm(s, x, wq, wsc, wbi, 4, cs.gs);
                try testing.expect(got_opt != null);
                const got = got_opt.?;
                defer _ = mlx.mlx_array_free(got);
                const gsh = mlx.getShape(got);
                try testing.expectEqual(@as(c_int, 1), gsh[0]);
                try testing.expectEqual(m, gsh[1]);
                try testing.expectEqual(cs.n, gsh[2]);

                try expectVerifyQmmNoWorseThanStock(s, x, wq, wsc, wbi, 4, cs.gs, wdq_t, got, label);
            }
        }

        // Ineligible widths fall through to stock (null). The NAX probe is
        // FORCED false so this pins the plain-SIMD (M1-M4) dispatch on every
        // machine, an M5 running the suite included: M=1 stays qmv, M 8..16
        // must NOT reach the split-K family (the measured spill cliff past 7
        // live row vectors), and M=17 is past every lane.
        {
            nax_gate.vqmm_nax_probe_override = false;
            defer nax_gate.vqmm_nax_probe_override = null;
            for ([_]c_int{ 1, 8, 16, 17 }) |bad_m| {
                const xn: usize = @intCast(bad_m * cs.k);
                const xbuf = try allocator.alloc(f32, xn);
                defer allocator.free(xbuf);
                for (xbuf) |*v| v.* = 0.1;
                const xshape = [_]c_int{ 1, bad_m, cs.k };
                const x32 = mlx.mlx_array_new_data(xbuf.ptr, &xshape, 3, .float32);
                defer _ = mlx.mlx_array_free(x32);
                var x = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(x);
                try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));
                try testing.expectEqual(@as(?mlx.mlx_array, null), try verifyQmm(s, x, wq, wsc, wbi, 4, cs.gs));
            }
        }

        // M=1 stays stock under the REAL probe too (MTPLX's tile nominally
        // covers M 1..16, but M=1 belongs to stock qmv on every machine).
        {
            const xbuf = try allocator.alloc(f32, @intCast(cs.k));
            defer allocator.free(xbuf);
            for (xbuf) |*v| v.* = 0.1;
            const xshape = [_]c_int{ 1, 1, cs.k };
            const x32 = mlx.mlx_array_new_data(xbuf.ptr, &xshape, 3, .float32);
            defer _ = mlx.mlx_array_free(x32);
            var x = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x);
            try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));
            try testing.expectEqual(@as(?mlx.mlx_array, null), try verifyQmm(s, x, wq, wsc, wbi, 4, cs.gs));
        }
    }

    // Mixed-precision trunk projections: the M5 NAX lane adopts affine
    // 5/6/8-bit weights on the measured wide shape class. Keep this self-gated
    // like the NAX rows above so non-G17 CI never attempts to compile an
    // unavailable MPP kernel.
    if (verifyQmmNaxEnabled()) {
        const k: c_int = 512;
        const n: c_int = 5120;
        const gs: u32 = 64;
        for ([_]u32{ 5, 6, 8 }) |bits| {
            const wb = try testRandWeightBf16(rnd, &[_]c_int{ n, k }, s);
            defer _ = mlx.mlx_array_free(wb);

            var triple = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(triple);
            try mlx.check(mlx.mlx_quantize(
                &triple,
                wb,
                mlx.mlx_optional_int.some(@intCast(gs)),
                mlx.mlx_optional_int.some(@intCast(bits)),
                "affine",
                .{},
                s,
            ));
            var wq = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wq);
            var wsc = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wsc);
            var wbi = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wbi);
            try mlx.check(mlx.mlx_vector_array_get(&wq, triple, 0));
            try mlx.check(mlx.mlx_vector_array_get(&wsc, triple, 1));
            try mlx.check(mlx.mlx_vector_array_get(&wbi, triple, 2));

            var wdq = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wdq);
            try mlx.check(mlx.mlx_dequantize(
                &wdq,
                wq,
                wsc,
                wbi,
                mlx.mlx_optional_int.some(@intCast(gs)),
                mlx.mlx_optional_int.some(@intCast(bits)),
                "affine",
                .{ .ctx = null },
                mlx.mlx_optional_dtype{ .has_value = true, .value = .float32 },
                s,
            ));
            var wdq_t = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(wdq_t);
            const t_axes = [_]c_int{ 1, 0 };
            try mlx.check(mlx.mlx_transpose_axes(&wdq_t, wdq, &t_axes, 2, s));

            const m: c_int = 8;
            const xn: usize = @intCast(m * k);
            const xbuf = try allocator.alloc(f32, xn);
            defer allocator.free(xbuf);
            for (xbuf) |*v| v.* = rnd.float(f32) - 0.5;
            const xshape = [_]c_int{ 1, m, k };
            const x32 = mlx.mlx_array_new_data(xbuf.ptr, &xshape, 3, .float32);
            defer _ = mlx.mlx_array_free(x32);
            var x = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x);
            try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));

            const got_opt = try verifyQmm(s, x, wq, wsc, wbi, bits, gs);
            try testing.expect(got_opt != null);
            const got = got_opt.?;
            defer _ = mlx.mlx_array_free(got);
            try expectVerifyQmmNoWorseThanStock(
                s,
                x,
                wq,
                wsc,
                wbi,
                bits,
                gs,
                wdq_t,
                got,
                switch (bits) {
                    5 => "nax oQe q5",
                    6 => "nax oQe q6",
                    else => "nax q8 candidate",
                },
            );

            // Below the NAX takeover row, default adoption remains the measured
            // width/shape surface: this q6/gs64/N=5120 projection uses the
            // plain-SIMD lane, while q5 still falls through to stock.
            const x6buf = try allocator.alloc(f32, 6 * @as(usize, @intCast(k)));
            defer allocator.free(x6buf);
            for (x6buf) |*v| v.* = 0.1;
            const x6shape = [_]c_int{ 1, 6, k };
            const x6f = mlx.mlx_array_new_data(x6buf.ptr, &x6shape, 3, .float32);
            defer _ = mlx.mlx_array_free(x6f);
            var x6 = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x6);
            try mlx.check(mlx.mlx_astype(&x6, x6f, .bfloat16, s));
            const got6_opt = try verifyQmm(s, x6, wq, wsc, wbi, bits, gs);
            if (bits == 6) {
                try testing.expect(got6_opt != null);
                const got6 = got6_opt.?;
                defer _ = mlx.mlx_array_free(got6);
                try expectVerifyQmmNoWorseThanStock(s, x6, wq, wsc, wbi, bits, gs, wdq_t, got6, "plain-SIMD q6");
            } else {
                try testing.expectEqual(@as(?mlx.mlx_array, null), got6_opt);
            }
        }
    }

    // ── msg wide-tile parity (the huge-N/lm_head arm), called directly with
    // a RAGGED N (1000 % 32 != 0) so the in-kernel n0 guard is exercised. ──
    {
        const mk: c_int = 2048;
        const mn: c_int = 1000;
        const wb = try testRandWeightBf16(rnd, &[_]c_int{ mn, mk }, s);
        defer _ = mlx.mlx_array_free(wb);
        var triple = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(triple);
        try mlx.check(mlx.mlx_quantize(&triple, wb, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{}, s));
        var wq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wq);
        var wsc = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wsc);
        var wbi = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wbi);
        try mlx.check(mlx.mlx_vector_array_get(&wq, triple, 0));
        try mlx.check(mlx.mlx_vector_array_get(&wsc, triple, 1));
        try mlx.check(mlx.mlx_vector_array_get(&wbi, triple, 2));

        var wdq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wdq);
        try mlx.check(mlx.mlx_dequantize(&wdq, wq, wsc, wbi, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(4), "affine", .{ .ctx = null }, mlx.mlx_optional_dtype{ .has_value = true, .value = .float32 }, s));
        var wdq_t = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wdq_t);
        const t_axes = [_]c_int{ 1, 0 };
        try mlx.check(mlx.mlx_transpose_axes(&wdq_t, wdq, &t_axes, 2, s));

        var m: c_int = 2;
        while (m <= 6) : (m += 1) {
            const xn: usize = @intCast(m * mk);
            const xbuf = try allocator.alloc(f32, xn);
            defer allocator.free(xbuf);
            for (xbuf) |*v| v.* = rnd.float(f32) - 0.5;
            const xshape = [_]c_int{ 1, m, mk };
            const x32 = mlx.mlx_array_new_data(xbuf.ptr, &xshape, 3, .float32);
            defer _ = mlx.mlx_array_free(x32);
            var x = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(x);
            try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));

            const xsh = mlx.getShape(x);
            const got = (try runVerifyQmmMsg(s, x, wq, wsc, wbi, 4, 64, m, mk, mn, .bfloat16, xsh)).?;
            defer _ = mlx.mlx_array_free(got);

            try expectVerifyQmmNoWorseThanStock(s, x, wq, wsc, wbi, 4, 64, wdq_t, got, "msg ragged-N");
        }
    }
}

test "vqmmLaneFor: NAX dispatch table (M 8..16 route to the m16 tile only when the probe is live)" {
    // Pure lane selection — todo-m5-nax.md §7. Hermetic on every machine:
    // no kernels are built, only the routing decision is pinned.
    const nax_off = false;
    const nax_on = true;

    // Plain-SIMD machine (probe false): the table is byte-identical to the
    // pre-NAX dispatch — M 8..16 fall through to stock.
    try testing.expectEqual(VqmmLane.none, vqmmLaneFor(1, 5120, 17408, nax_off, 8));
    try testing.expectEqual(VqmmLane.splitk, vqmmLaneFor(2, 5120, 17408, nax_off, 8));
    try testing.expectEqual(VqmmLane.splitk, vqmmLaneFor(7, 5120, 17408, nax_off, 8));
    try testing.expectEqual(VqmmLane.msg, vqmmLaneFor(4, 5120, 151936, nax_off, 8));
    try testing.expectEqual(VqmmLane.none, vqmmLaneFor(8, 5120, 17408, nax_off, 8));
    try testing.expectEqual(VqmmLane.none, vqmmLaneFor(16, 5120, 17408, nax_off, 8));
    // Geometry floors shared by every lane.
    try testing.expectEqual(VqmmLane.none, vqmmLaneFor(4, 5120, 256, nax_off, 8)); // N < 512
    try testing.expectEqual(VqmmLane.none, vqmmLaneFor(4, 100, 17408, nax_off, 8)); // K % 64 != 0
    try testing.expectEqual(VqmmLane.none, vqmmLaneFor(4, 5120, 17409, nax_off, 8)); // N % 4 != 0

    // M5 (probe true), default takeover width 8: M 2..7 KEEP the plain-SIMD
    // lanes (MTPLX's own dispatcher keeps SIMD through M=6 with NAX lit —
    // 16-row padding waste makes SIMD competitive at small M).
    try testing.expectEqual(VqmmLane.splitk, vqmmLaneFor(7, 5120, 17408, nax_on, 8));
    try testing.expectEqual(VqmmLane.msg, vqmmLaneFor(7, 5120, 151936, nax_on, 8));
    // M 8..16 route to the NAX tile when its stricter geometry holds…
    try testing.expectEqual(VqmmLane.nax, vqmmLaneFor(8, 5120, 17408, nax_on, 8));
    try testing.expectEqual(VqmmLane.nax, vqmmLaneFor(16, 5120, 17408, nax_on, 8));
    // …including the lm_head class natively (N=151936 % 32 == 0 — no msg
    // variant needed past M=7).
    try testing.expectEqual(VqmmLane.nax, vqmmLaneFor(9, 5120, 151936, nax_on, 8));
    // Tile-ineligible geometry at M 8..16 stays stock: K % 256, N % 32.
    try testing.expectEqual(VqmmLane.none, vqmmLaneFor(8, 5184, 17408, nax_on, 8)); // 5184 % 256 == 64
    try testing.expectEqual(VqmmLane.none, vqmmLaneFor(8, 5120, 17412, nax_on, 8)); // 17412 % 32 == 4
    try testing.expectEqual(VqmmLane.none, vqmmLaneFor(17, 5120, 17408, nax_on, 8)); // past every lane
    try testing.expectEqual(VqmmLane.none, vqmmLaneFor(1, 5120, 17408, nax_on, 8)); // M=1 stays qmv
    try testing.expectEqual(VqmmLane.none, vqmmLaneFor(8, 5120, 480, nax_on, 8)); // tiny-N floor holds for NAX too

    // A lowered takeover width (5): M 5..7 route to NAX, M 4 keeps split-K,
    // and the tile geometry is still required, else the SIMD lanes serve.
    try testing.expectEqual(VqmmLane.nax, vqmmLaneFor(5, 5120, 17408, nax_on, 5));
    try testing.expectEqual(VqmmLane.nax, vqmmLaneFor(7, 5120, 151936, nax_on, 5));
    try testing.expectEqual(VqmmLane.splitk, vqmmLaneFor(4, 5120, 17408, nax_on, 5));
    try testing.expectEqual(VqmmLane.splitk, vqmmLaneFor(5, 5184, 17408, nax_on, 5)); // K % 256 fails, % 64 holds
}

test "mixedNaxShapeEnabled keeps measured q5/q6/q8 wins and rejects narrow regressions" {
    try testing.expect(mixedNaxShapeEnabled(4, 32, 1024));
    try testing.expect(mixedNaxShapeEnabled(5, 64, 5120));
    try testing.expect(mixedNaxShapeEnabled(6, 64, 5120));
    try testing.expect(mixedNaxShapeEnabled(8, 64, 5120));
    try testing.expect(!mixedNaxShapeEnabled(5, 64, 1024));
    try testing.expect(!mixedNaxShapeEnabled(6, 64, 1024));
    try testing.expect(!mixedNaxShapeEnabled(8, 64, 1024));
    try testing.expect(!mixedNaxShapeEnabled(5, 32, 5120));
    try testing.expect(!mixedNaxShapeEnabled(6, 128, 5120));
    try testing.expect(!mixedNaxShapeEnabled(8, 32, 5120));
}

test "NAX host scaffolding: zero-pad to 16 rows + slice-back are exact (runs off-M5)" {
    // The half of runVerifyQmmNax that CAN execute without G17 hardware —
    // the exact production helpers (naxPadTo16 / naxSliceRows) wrapped
    // around STOCK qmm standing in for the NAX kernel. Pins the mlx_pad
    // axis/value plumbing (pad rows are EXACT zeros in the activation
    // dtype, real rows byte-preserved), the slice bounds, and that zero
    // activation rows produce exactly-zero output rows (so on the M5 the
    // padded tile positions can never bleed into the sliced result).
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xA110C);
    const rnd = prng.random();
    const K: c_int = 512;
    const N: c_int = 640; // % 32 == 0, NAX-shaped
    const gs: u32 = 64;

    const wb = try testRandWeightBf16(rnd, &[_]c_int{ N, K }, s);
    defer _ = mlx.mlx_array_free(wb);
    var triple = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(triple);
    try mlx.check(mlx.mlx_quantize(&triple, wb, mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(4), "affine", .{}, s));
    var wq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wq);
    var wsc = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wsc);
    var wbi = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wbi);
    try mlx.check(mlx.mlx_vector_array_get(&wq, triple, 0));
    try mlx.check(mlx.mlx_vector_array_get(&wsc, triple, 1));
    try mlx.check(mlx.mlx_vector_array_get(&wbi, triple, 2));

    for ([_]c_int{ 9, 16 }) |m| {
        const xn: usize = @intCast(m * K);
        const xbuf = try allocator.alloc(f32, xn);
        defer allocator.free(xbuf);
        for (xbuf) |*v| v.* = rnd.float(f32) - 0.5;
        const xshape = [_]c_int{ 1, m, K };
        const x32 = mlx.mlx_array_new_data(xbuf.ptr, &xshape, 3, .float32);
        defer _ = mlx.mlx_array_free(x32);
        var x = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x);
        try mlx.check(mlx.mlx_astype(&x, x32, .bfloat16, s));

        // Pad half: [1, m, K] → [16, K]; real rows byte-preserved, pad rows 0.
        const x16 = try naxPadTo16(s, x, m, K, .bfloat16);
        defer _ = mlx.mlx_array_free(x16);
        const psh = mlx.getShape(x16);
        try testing.expectEqual(@as(usize, 2), psh.len);
        try testing.expectEqual(@as(c_int, 16), psh[0]);
        try testing.expectEqual(K, psh[1]);
        {
            var xf = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(xf);
            var pf = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(pf);
            try mlx.check(mlx.mlx_astype(&xf, x, .float32, s));
            try mlx.check(mlx.mlx_astype(&pf, x16, .float32, s));
            try mlx.check(mlx.mlx_array_eval(xf));
            try mlx.check(mlx.mlx_array_eval(pf));
            const xd_ = mlx.mlx_array_data_float32(xf).?;
            const pd_ = mlx.mlx_array_data_float32(pf).?;
            for (0..xn) |i| try testing.expectEqual(xd_[i], pd_[i]);
            for (xn..@intCast(16 * K)) |i| try testing.expectEqual(@as(f32, 0.0), pd_[i]);
        }

        // Slice half around stock qmm: zero rows in → exactly-zero rows out,
        // and the slice returns the first m rows byte-identically.
        var y16 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(y16);
        try mlx.check(mlx.mlx_quantized_matmul(&y16, x16, wq, wsc, wbi, true, mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(4), "affine", s));
        const ym = try naxSliceRows(s, y16, m, N);
        defer _ = mlx.mlx_array_free(ym);
        const ysh = mlx.getShape(ym);
        try testing.expectEqual(@as(usize, 2), ysh.len);
        try testing.expectEqual(m, ysh[0]);
        try testing.expectEqual(N, ysh[1]);
        {
            var ff = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ff);
            var mf = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(mf);
            try mlx.check(mlx.mlx_astype(&ff, y16, .float32, s));
            try mlx.check(mlx.mlx_astype(&mf, ym, .float32, s));
            try mlx.check(mlx.mlx_array_eval(ff));
            try mlx.check(mlx.mlx_array_eval(mf));
            const fd = mlx.mlx_array_data_float32(ff).?;
            const md = mlx.mlx_array_data_float32(mf).?;
            const mn: usize = @intCast(m * N);
            for (0..mn) |i| try testing.expectEqual(fd[i], md[i]);
            for (mn..@intCast(16 * N)) |i| try testing.expectEqual(@as(f32, 0.0), @abs(fd[i]));
        }
    }
}
