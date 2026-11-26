const codebook_tables = @import("codebook_tables.zig");
const exl3 = @import("expert_exl3.zig");
const mlx = @import("mlx_host").mlx;
const std = @import("std");
const kernels = @import("expert_exl3_kernels.zig");
const ResearchCodebook = kernels.ResearchCodebook;

const N_CB = exl3.Codebook.count;

const N_RC = std.enums.values(ResearchCodebook).len;

const N_WIN = exl3.Window.count;

const N_TABLES = codebook_tables.tables.len;

const TABLED_ROW0 = 2 * N_CB + N_RC;

/// A SPEC1 kernel carries its table's rows, so each (table, folded or unfolded) pair has its own row after the codebooks'.
pub const SLOT_ROWS = TABLED_ROW0 + 2 * N_TABLES;

pub const KernelSlots = [SLOT_ROWS][N_WIN]?mlx.mlx_fast_metal_kernel;

pub const no_kernels: KernelSlots = @splat(@splat(null));

fn cbIndex(cb: exl3.Codebook) usize {
    return @backingInt(cb);
}

fn tableOf(dec: exl3.Decode) usize {
    return dec.table.?;
}

/// The slot row of codebook `cb` under `active_decode`, `unfolded` for the test seam's scale-in-weights variant.
pub fn slotRow(cb: exl3.Codebook, unfolded: bool) usize {
    if (cb.tabledHashMask()) return TABLED_ROW0 + 2 * tableOf(kernels.active_decode) + @intFromBool(unfolded);
    const row: usize = if (unfolded) N_CB else 0;
    return row + cbIndex(cb);
}

pub fn winSuffix(comptime win: exl3.Window) [:0]const u8 {
    return comptime if (win == .w16) "" else std.fmt.comptimePrint("_w{d}", .{win.bits()});
}

pub fn cbSuffix(comptime cb: exl3.Codebook) [:0]const u8 {
    return switch (cb) {
        .mul1 => "",
        .mcg => "_mcg",
        .alpha => "_alpha",
        .spec1 => "_spec1",
        .std1 => "_std1",
        .pair1 => "_pair1",
    };
}

/// The power of two `exl3_decode2` leaves out of a codebook's weights: a hash-mask codebook's
/// scale, restored by `exl3_out` once per stored sum instead of once per decoded pair.
fn foldScale(comptime cb: exl3.Codebook) u32 {
    return if (cb.isHashMask()) exl3.hash_mask_scale else if (cb == .pair1) PAIR1_SCALE else 1;
}

/// Every PAIR1 row's scale; `pair1Helpers` refuses a row that differs.
const PAIR1_SCALE: u32 = 4096;

/// The one decode every weight kernel calls, two codewords to two halves. A
/// pack whose search hashed a narrower window masks the sliding window down to
/// it first; w16 masks nothing and emits the source it always did.
/// `exl3_pairh` is the weight itself, an MMA operand. `exl3_decode1`/`exl3_decode2` are the weight
/// over `foldScale`, and a kernel that reads them passes every f32 sum it stores through
/// `exl3_out`: a power of two, so its sums and stored bytes are the unfolded ones.
pub fn codebookHelpers(comptime cb: exl3.Codebook, comptime win: exl3.Window) [:0]const u8 {
    return decodeHelpers(cb, win, foldScale(cb));
}

const HELPERS_TAIL: [:0]const u8 =
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
    \\// The first codeword a codebook decodes: a pair codec that reads only odd weights skips weight 0's.
    \\#ifndef EXL3_LANE_FIRST
    \\#define EXL3_LANE_FIRST 0u
    \\#endif
    \\template<bool Wide, bool Funnel = false, uint PairKind = 0u> struct exl3_lane_type {
    \\  using type = ulong;
    \\  static inline type read(const device uint *words, uint end, uint W, bool full) {
    \\    return exl3_window(words, end, W, full);
    \\  }
    \\};
    \\// exl3_window's bits with a third word, built as two 32-bit funnels of adjacent words
    \\// (s < 32): OR-ing the third word into a 64-bit shift costs the ALU-bound decode GEMV far more.
    \\template<> struct exl3_lane_type<false, true, 0u> {
    \\  using type = ulong;
    \\  static inline type read(const device uint *words, uint end, uint W, bool) {
    \\    const uint last = (end - 1u) >> 5u;
    \\    const uint prev = last == 0u ? W - 1u : last - 1u;
    \\    const uint third = prev == 0u ? W - 1u : prev - 1u;
    \\    const uint s = (0u - end) & 31u;
    \\    const uint a = words[last];
    \\    const uint b = words[prev];
    \\    const uint c = words[third];
    \\    return as_type<ulong>(uint2(uint((((ulong)b << 32u) | (ulong)a) >> s), uint((((ulong)c << 32u) | (ulong)b) >> s)));
    \\  }
    \\};
    \\// PAIR1 narrow (N <= 42, EXL3_LANE_FIRST == 1u): all 4 odd codewords lie in bits 0..31 -> single 32-bit uint.
    \\template<> struct exl3_lane_type<false, false, 1u> {
    \\  using type = uint;
    \\  static inline type read(const device uint *words, uint end, uint W, bool) {
    \\    const uint last = (end - 1u) >> 5u;
    \\    const uint prev = last == 0u ? W - 1u : last - 1u;
    \\    const uint s = (0u - end) & 31u;
    \\    return uint((((ulong)words[prev] << 32u) | (ulong)words[last]) >> s);
    \\  }
    \\};
    \\// PAIR1 medium (42 < N <= 108, EXL3_LANE_FIRST == 1u): uint2 keeps sh <= 16u as 32-bit shifts and sh > 16u as 32-bit funnels.
    \\template<> struct exl3_lane_type<false, false, 2u> {
    \\  using type = uint2;
    \\  static inline type read(const device uint *words, uint end, uint W, bool) {
    \\    const uint last = (end - 1u) >> 5u;
    \\    const uint prev = last == 0u ? W - 1u : last - 1u;
    \\    const uint s = (0u - end) & 31u;
    \\    const uint a = words[last];
    \\    const uint b = words[prev];
    \\    return uint2(uint((((ulong)b << 32u) | (ulong)a) >> s), b >> s);
    \\  }
    \\};
    \\template<> struct exl3_lane_type<false, true, 2u> {
    \\  using type = uint2;
    \\  static inline type read(const device uint *words, uint end, uint W, bool) {
    \\    const uint last = (end - 1u) >> 5u;
    \\    const uint prev = last == 0u ? W - 1u : last - 1u;
    \\    const uint third = prev == 0u ? W - 1u : prev - 1u;
    \\    const uint s = (0u - end) & 31u;
    \\    const uint a = words[last];
    \\    const uint b = words[prev];
    \\    const uint c = words[third];
    \\    return uint2(uint((((ulong)b << 32u) | (ulong)a) >> s), uint((((ulong)c << 32u) | (ulong)b) >> s));
    \\  }
    \\};
    \\template<> struct exl3_lane_type<true, false, 0u> {
    \\  using type = ulong2;
    \\  static inline type read(const device uint *words, uint end, uint W, bool full) {
    \\    return ulong2(exl3_window(words, end, W, full), exl3_window(words, end - 32u, W, true));
    \\  }
    \\};
    \\static inline constexpr uint exl3_pair_kind(uint N) {
    \\  return (EXL3_LANE_FIRST == 1u && exl3_lane_sh(N, 0u) + 16u <= 64u)
    \\    ? (exl3_lane_sh(N, 1u) + 16u <= 32u ? 1u : 2u)
    \\    : 0u;
    \\}
    \\template<uint N> using exl3_lane_bits = typename exl3_lane_type<(exl3_lane_sh(N, 0u) + 16u > 64u), false, exl3_pair_kind(N)>::type;
    \\static inline uint exl3_lane_word(uint bits, uint sh) { return bits >> sh; }
    \\static inline uint exl3_lane_word(uint2 bits, uint sh) {
    \\  return sh <= 16u ? (bits.x >> sh) : (sh < 32u ? ((bits.x >> sh) | (bits.y << (32u - sh))) : (bits.y >> (sh - 32u)));
    \\}
    \\static inline uint exl3_lane_word(ulong bits, uint sh) { return uint(bits >> sh); }
    \\static inline uint exl3_lane_word(ulong2 bits, uint sh) {
    \\  return sh <= 48u ? uint(bits.x >> sh) : uint(bits.y >> (sh - 32u));
    \\}
    \\template<uint N>
    \\static inline exl3_lane_bits<N> exl3_lane(const device uint *words, uint lane) {
    \\  constexpr uint W = N / 2u;
    \\  constexpr uint low = W & (0u - W);
    \\  constexpr bool wide = exl3_lane_sh(N, 0u) + 16u > 64u;
    \\  constexpr uint pk = exl3_pair_kind(N);
    \\  constexpr bool full = exl3_lane_sh(N, EXL3_LANE_FIRST) + 16u > 32u + (low < 32u ? low : 32u);
    \\  const uint end = W * lane + W;
    \\  // At K3.5 the W bits under a lane's 32-bit window are the high W bits of the preceding lane's
    \\  // window, so a shuffle stands in for their loads. Every caller runs it with the simdgroup in step.
    \\  if constexpr (N == 56u && EXL3_LANE_FIRST == 1u) {
    \\    const uint low_bits = uint(exl3_window(words, end, W, false));
    \\    const uint prior = simd_shuffle(low_bits, ushort((lane + 31u) & 31u));
    \\    return uint2(low_bits, prior >> (32u - W));
    \\  }
    \\  return exl3_lane_type<wide, !wide && full, pk>::read(words, end, W, full);
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

fn decodeHelpers(comptime cb: exl3.Codebook, comptime win: exl3.Window, comptime fold: u32) [:0]const u8 {
    if (cb == .pair1) return comptime pair1Helpers(win, fold) ++ HELPERS_TAIL;
    if (cb.tabledHashMask()) return comptime tabledHelpers(win, fold) ++ HELPERS_TAIL;
    // The half sum rounds once, as `decodeHashMask` does on the host; its power-of-two rescale is exact.
    const body: [:0]const u8 = comptime if (cb.hashMask()) |hm| std.fmt.comptimePrint(
        \\  const uint2 r = (cw * uint2(0x{X}u)) & uint2(0x{X}u);
        \\  const half4 h = as_type<half4>(r);
        \\  return half2(h.x + h.y, h.z + h.w);
    , .{ hm.mult, hm.mask }) else switch (cb) {
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
        else => unreachable,
    };
    return comptime helpersWithBody("", body, win, if (cb.hashMask()) |hm| hm.scale else null, fold);
}

const ANDONLY_MULT: u32 = 0x84CC1707;
const ANDONLY_MASK: u32 = 0x8FE08FC0;
/// sashimi's KRON_K3 multiplier; each rate's own multiplier costs the same.
const KRON_MULT: u32 = 0x9F56F68B;
const KRON_STD: f64 = 1.2441;

fn kronBits(rc: ResearchCodebook) u5 {
    return switch (rc) {
        .kron8 => 8,
        .kron10, .kron10s, .kron10h => 10,
        .andonly, .kronlog, .kronlogsq => unreachable,
    };
}

/// `KRON_STD * Phi^-1((i + 0.5) / 2^bits)` as half bits, Phi^-1 by AS241 (Python's NormalDist).
fn kronTable(comptime bits: u5) [1 << bits]u16 {
    @setEvalBranchQuota(10_000_000);
    var out: [1 << bits]u16 = undefined;
    for (&out, 0..) |*v, i| {
        const p = (@as(f64, @floatFromInt(i)) + 0.5) / @as(f64, @floatFromInt(1 << bits));
        v.* = exl3.f32ToF16Bits(@floatCast(KRON_STD * invNormal(p)));
    }
    return out;
}

fn invNormal(p: f64) f64 {
    const q = p - 0.5;
    if (@abs(q) <= 0.425) {
        const r = 0.180625 - q * q;
        const num = (((((((2509.0809287301226727 * r + 33430.575583588128105) * r + 67265.770927008700853) * r + 45921.953931549871457) * r + 13731.693765509461125) * r + 1971.5909503065514427) * r + 133.14166789178437745) * r + 3.387132872796366608);
        const den = (((((((5226.495278852545925 * r + 28729.085735721942674) * r + 39307.89580009271061) * r + 21213.794301586595867) * r + 5394.1960214247511077) * r + 687.1870074920579083) * r + 42.313330701600911252) * r + 1.0);
        return q * num / den;
    }
    var r = @sqrt(-@log(if (q <= 0) p else 1.0 - p));
    var x: f64 = undefined;
    if (r <= 5.0) {
        r -= 1.6;
        const num = (((((((7.7454501427834140764e-4 * r + 0.0227238449892691845833) * r + 0.24178072517745061177) * r + 1.27045825245236838258) * r + 3.64784832476320460504) * r + 5.7694972214606914055) * r + 4.6303378461565452959) * r + 1.42343711074968357734);
        const den = (((((((1.05075007164441684324e-9 * r + 5.475938084995344946e-4) * r + 0.0151986665636164571966) * r + 0.14810397642748007459) * r + 0.68976733498510000455) * r + 1.6763848301838038494) * r + 2.05319162663775882187) * r + 1.0);
        x = num / den;
    } else {
        r -= 5.0;
        const num = (((((((2.01033439929228813265e-7 * r + 2.71155556874348757815e-5) * r + 0.0012426609473880784386) * r + 0.026532189526576123093) * r + 0.29656057182850489123) * r + 1.7848265399172913358) * r + 5.4637849111641143699) * r + 6.6579046435011037772);
        const den = (((((((2.04426310338993978564e-15 * r + 1.4215117583164458887e-7) * r + 1.8463183175100546818e-5) * r + 7.868691311456132591e-4) * r + 0.0148753612908506148525) * r + 0.13692988092273580531) * r + 0.59983220655588793769) * r + 1.0);
        x = num / den;
    }
    return if (q < 0) -x else x;
}

fn kronTableDecl(comptime bits: u5) [:0]const u8 {
    @setEvalBranchQuota(10_000_000);
    const table = comptime kronTable(bits);
    comptime var hex: [table.len * 7]u8 = undefined;
    inline for (table, 0..) |v, i| {
        hex[i * 7 ..][0..2].* = "0x".*;
        inline for (0..4) |d| hex[i * 7 + 2 + d] = "0123456789ABCDEF"[(v >> (12 - 4 * d)) & 0xF];
        hex[i * 7 + 6] = ',';
    }
    const text = hex;
    return std.fmt.comptimePrint("constant ushort EXL3_KRON[{d}] = {{", .{table.len}) ++ text ++ "};\n";
}

/// MCG's helpers with a research codebook's decode body (and kron's table ahead of them).
fn researchHelpers(comptime rc: ResearchCodebook, comptime win: exl3.Window) [:0]const u8 {
    @setEvalBranchQuota(10_000_000);
    return comptime switch (rc) {
        .andonly => helpersWithBody("", std.fmt.comptimePrint(
            \\  const uint2 r = (cw * uint2(0x{X}u)) & uint2(0x{X}u);
            \\  const half4 h = as_type<half4>(r);
            \\  return half2(h.x + h.y, h.z + h.w) * half2(4096.0h);
        , .{ ANDONLY_MULT, ANDONLY_MASK }), win, null, 1),
        .kronlog, .kronlogsq => helpersWithBody("", std.fmt.comptimePrint(
            \\  const uint2 x = cw * uint2(0x{X}u);
            \\  const float2 a = float2(x);
            \\  const float2 b = 4294967296.0f - a;
            \\  const float2 y = float2(as_type<int2>(a) - as_type<int2>(b)) * 8.262958317573066e-08f;
            \\
        , .{KRON_MULT}) ++ (if (rc == .kronlog) "  return half2(y);" else "  return half2(y * fma(float2(-0.04f), fabs(y), float2(1.0f)));"), win, null, 1),
        .kron8, .kron10, .kron10s, .kron10h => helpersWithBody(kronTableDecl(kronBits(rc)), std.fmt.comptimePrint(
            \\  const uint2 x = (cw * uint2(0x{X}u)) >> uint2({d}u);
            \\  return half2(as_type<half>(EXL3_KRON[x.x]), as_type<half>(EXL3_KRON[x.y]));
        , .{ KRON_MULT, 32 - @as(u32, kronBits(rc)) }), win, null, 1),
    };
}

/// The scalar truth of `researchHelpers`' decode for one (already masked) codeword.
pub fn researchDecode(cw: u16, rc: ResearchCodebook) u16 {
    switch (rc) {
        .andonly => {
            const r = (@as(u32, cw) *% ANDONLY_MULT) & ANDONLY_MASK;
            const sum = exl3.f32ToF16Bits(exl3.f16BitsToF32(@truncate(r)) + exl3.f16BitsToF32(@truncate(r >> 16)));
            return exl3.f32ToF16Bits(exl3.f16BitsToF32(sum) * 4096.0);
        },
        .kron8 => return (comptime kronTable(8))[(@as(u32, cw) *% KRON_MULT) >> 24],
        .kron10, .kron10s, .kron10h => return (comptime kronTable(10))[(@as(u32, cw) *% KRON_MULT) >> 22],
        .kronlog, .kronlogsq => {
            const a: f32 = @floatFromInt(@as(u32, cw) *% KRON_MULT);
            const b: f32 = 4294967296.0 - a;
            const d = @as(i32, @bitCast(a)) -% @as(i32, @bitCast(b));
            const y = @as(f32, @floatFromInt(d)) * @as(f32, 8.262958317573066e-08);
            return exl3.f32ToF16Bits(if (rc == .kronlog) y else y * @mulAdd(f32, -0.04, @abs(y), 1.0));
        },
    }
}

fn stagedResearch(rc: ResearchCodebook) bool {
    return rc == .kron10s or rc == .kron10h;
}

/// Kernel-body setup of a staged arm: its table copied into threadgroup memory before any decode.
fn stagedPrologue(comptime rc: ResearchCodebook) [:0]const u8 {
    const n: u32 = if (rc == .kron10h) 512 else 1024;
    const src = if (rc == .kron10h) "EXL3_KRON_HALF" else "EXL3_KRON";
    return comptime std.fmt.comptimePrint(
        \\threadgroup ushort EXL3_KRON_TG[{d}];
        \\for (uint exl3_i = uint(thread_index_in_threadgroup); exl3_i < {d}u; exl3_i += uint(threads_per_threadgroup.x)) EXL3_KRON_TG[exl3_i] = {s}[exl3_i];
        \\threadgroup_barrier(mem_flags::mem_threadgroup);
        \\
    , .{ n, n, src });
}

/// After every helper a staged arm sends the kernel bodies' `exl3_decode2` to the staged table, which only
/// the kernel body can see; helpers defined earlier keep the constant-table decode.
fn stagedTail(comptime rc: ResearchCodebook, comptime win: exl3.Window) [:0]const u8 {
    if (!stagedResearch(rc)) return "";
    const mask: [:0]const u8 = comptime if (win == .w16) "" else std.fmt.comptimePrint("  cw &= uint2(0x{X}u);\n", .{win.mask()});
    const lookup: [:0]const u8 = comptime if (rc == .kron10s)
        \\  return half2(as_type<half>(tab[x.x]), as_type<half>(tab[x.y]));
    else
        \\  const uint2 hi = x >> uint2(9u);
        \\  const uint2 j = x ^ (hi * uint2(0x3FFu));
        \\  return as_type<half2>(ushort2(tab[j.x], tab[j.y]) ^ ushort2(hi << uint2(15u)));
    ;
    const half_decl: [:0]const u8 = comptime if (rc == .kron10h) kronHalfTableDecl() else "";
    return comptime half_decl ++ "static inline half2 exl3_pairh_tg(uint2 cw, threadgroup const ushort *tab) {\n" ++ mask ++
        std.fmt.comptimePrint("  const uint2 x = (cw * uint2(0x{X}u)) >> uint2(22u);\n", .{KRON_MULT}) ++ lookup ++
        "\n}\n#define exl3_decode2(cw) float2(exl3_pairh_tg((cw), EXL3_KRON_TG))\n";
}

fn kronHalfTableDecl() [:0]const u8 {
    @setEvalBranchQuota(10_000_000);
    const table = comptime kronTable(10);
    comptime var hex: [512 * 7]u8 = undefined;
    inline for (table[0..512], 0..) |v, i| {
        hex[i * 7 ..][0..2].* = "0x".*;
        inline for (0..4) |d| hex[i * 7 + 2 + d] = "0123456789ABCDEF"[(v >> (12 - 4 * d)) & 0xF];
        hex[i * 7 + 6] = ',';
    }
    const text = hex;
    return "constant ushort EXL3_KRON_HALF[512] = {" ++ text ++ "};\n";
}


/// `hash_scale` is a hash-mask codebook's scale, left out of `body`'s half sums (`fold` == it, or 1 to carry it).
fn helpersWithBody(comptime prefix: [:0]const u8, comptime body: [:0]const u8, comptime win: exl3.Window, comptime hash_scale: ?u32, comptime fold: u32) [:0]const u8 {
    const mask: [:0]const u8 = comptime if (win == .w16) "" else std.fmt.comptimePrint("  cw &= uint2(0x{X}u);\n", .{win.mask()});
    const pair = comptime if (hash_scale) |scale|
        prefix ++ "static inline half2 exl3_unscaled(uint2 cw) {\n" ++ mask ++ body ++ "\n}\n" ++
            std.fmt.comptimePrint("static inline half2 exl3_pairh(uint2 cw) {{ return exl3_unscaled(cw) * half2({d}.0h); }}\n", .{scale})
    else
        prefix ++ "static inline half2 exl3_pairh(uint2 cw) {\n" ++ mask ++ body ++ "\n}\n";
    const decode = comptime if (fold == 1)
        \\static inline float2 exl3_decode2(uint2 cw) { return float2(exl3_pairh(cw)); }
        \\static inline float exl3_out(float v) { return v; }
        \\
    else if (hash_scale != null and fold == hash_scale.?)
        \\static inline float2 exl3_decode2(uint2 cw) { return float2(exl3_unscaled(cw)); }
        \\
    ++ std.fmt.comptimePrint("static inline float exl3_out(float v) {{ return v * {d}.0f; }}\n", .{fold})
    else
        @compileError("a fold is the hash-mask scale or none");
    return comptime pair ++ decode ++
        \\static inline float exl3_decode1(uint cw) { return exl3_decode2(uint2(cw, 0u)).x; }
        \\// A generic reader's weight `slot` of a tile's unpacked codewords.
        \\static inline float exl3_slot(thread const ushort *cw, uint slot) { return exl3_decode1(uint(cw[slot])); }
        \\
    ++ HELPERS_TAIL;
}

/// What the row-driven codebooks (SPEC1, PAIR1) build over their `exl3_unscaled_n<N>`: the weight, the decode that
/// leaves `scale` out or keeps it, `slot` (a generic reader's weight), and the macros that hand every kernel's rate
/// `N` to them, so each row constant is a compile-time immediate of its own kernel instance.
fn rowedHelpers(comptime fold: u32, comptime scale: u32, comptime slot: [:0]const u8) [:0]const u8 {
    const out: [:0]const u8 = comptime if (fold == 1)
        "template<uint N> static inline float2 exl3_decode2_n(uint2 cw) { return float2(exl3_pairh_n<N>(cw)); }\n" ++
            "static inline float exl3_out(float v) { return v; }\n"
    else if (fold == scale)
        "template<uint N> static inline float2 exl3_decode2_n(uint2 cw) { return float2(exl3_unscaled_n<N>(cw)); }\n" ++
            std.fmt.comptimePrint("static inline float exl3_out(float v) {{ return v * {d}.0f; }}\n", .{fold})
    else
        @compileError("a fold is the codebook's scale or none");
    return comptime std.fmt.comptimePrint("template<uint N> static inline half2 exl3_pairh_n(uint2 cw) {{ return exl3_unscaled_n<N>(cw) * half2({d}.0h); }}\n", .{scale}) ++
        out ++ slot ++
        "#define exl3_pairh(cw) exl3_pairh_n<N>((cw))\n" ++
        "#define exl3_decode2(cw) exl3_decode2_n<N>((cw))\n" ++
        "#define exl3_slot(cw, slot) exl3_slot_n<N>((cw), (slot))\n";
}

fn windowMask(comptime win: exl3.Window) [:0]const u8 {
    return if (win == .w16) "" else std.fmt.comptimePrint("  cw &= uint2(0x{X}u);\n", .{win.mask()});
}

/// PAIR1's pair decode: weights 2j and 2j+1 both come from the window of weight 2j+1 (`cw.y`; `cw.x` is
/// never read). A bank's rate is the kernel's `N`, which the macros of `rowedHelpers` look the constants up by in
/// `exl3_pair_row_at`; the row table is generated from `exl3.pair1Pair`, the one definition the host decodes with.
fn pair1Helpers(comptime win: exl3.Window, comptime fold: u32) [:0]const u8 {
    comptime {
        @setEvalBranchQuota(2_000_000);
        var rows: [:0]const u8 = "";
        var n: u32 = exl3.Rate.min_n;
        while (n <= exl3.Rate.max_n) : (n += 2) {
            const p = exl3.pair1Pair(.{ .n = n }) orelse continue;
            if (p.scale != PAIR1_SCALE) @compileError("a PAIR1 row scales by 4096");
            rows = rows ++ std.fmt.comptimePrint("    case {d}u: return {{0x{X}u, {d}u, 0x{X}u, 0x{X}u}};\n", .{ n, p.mult, p.rot, p.mask0, p.mask1 });
        }
        return "#define EXL3_LANE_FIRST 1u\nstruct exl3_pair_row { uint mult; uint rot; uint mask0; uint mask1; };\n" ++
            "static inline constexpr exl3_pair_row exl3_pair_row_at(uint n) {\n  switch (n) {\n" ++ rows ++
            "  }\n  return {0u, 0u, 0u, 0u};\n}\n" ++
            "template<uint N> static inline half2 exl3_unscaled_n(uint2 cw) {\n" ++
            "  constexpr exl3_pair_row r = exl3_pair_row_at(N);\n" ++
            "  static_assert(r.mult != 0u, \"PAIR1 has no pair row at this rate\");\n" ++ windowMask(win) ++
            "  const uint p = cw.y * r.mult;\n" ++
            "  const half4 h = as_type<half4>(uint2(p & r.mask0, rotate(p, r.rot) & r.mask1));\n" ++
            "  return half2(h.x + h.y, h.z + h.w);\n}\n" ++
            rowedHelpers(fold, PAIR1_SCALE,
                "template<uint N> static inline float exl3_slot_n(thread const ushort *cw, uint slot) {\n" ++
                    "  const float2 w = exl3_decode2_n<N>(uint2(0u, uint(cw[slot | 1u])));\n" ++
                    "  return (slot & 1u) != 0u ? w.y : w.x;\n}\n");
    }
}

/// SPEC1's hash-mask decode with the (multiplier, mask) of the bank's rate: `exl3_hash_row_at(N)` is the table's row
/// lookup, which `withTableRows` puts ahead of this text because the rows are the pack's data, not the kernel's.
fn tabledHelpers(comptime win: exl3.Window, comptime fold: u32) [:0]const u8 {
    return comptime "#undef EXL3_LANE_FIRST\ntemplate<uint N> static inline half2 exl3_unscaled_n(uint2 cw) {\n" ++
        "  constexpr exl3_hash_row r = exl3_hash_row_at(N);\n" ++
        "  static_assert(r.mult != 0u, \"the codebook table has no row at this rate\");\n" ++ windowMask(win) ++
        "  const half4 h = as_type<half4>((cw * uint2(r.mult)) & uint2(r.mask));\n" ++
        "  return half2(h.x + h.y, h.z + h.w);\n}\n" ++
        rowedHelpers(fold, exl3.hash_mask_scale,
            "template<uint N> static inline float exl3_slot_n(thread const ushort *cw, uint slot) { return exl3_decode2_n<N>(uint2(uint(cw[slot]), 0u)).x; }\n");
}

/// The row lookup `tabledHelpers` reads, generated from SPEC1 table `which`.
fn tableRowsText(alloc: std.mem.Allocator, which: u8) ![:0]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    const table = &codebook_tables.tables[which];
    try out.print(alloc, "struct exl3_hash_row {{ uint mult; uint mask; }};\nstatic inline constexpr exl3_hash_row exl3_hash_row_at(uint n) {{\n  switch (n) {{\n", .{});
    for (table.rows, 0..) |r, i| try out.print(alloc, "    case {d}u: return {{0x{X}u, 0x{X}u}};\n", .{ table.rateOf(i).n, r[0], r[1] });
    try out.print(alloc, "  }}\n  return {{0u, 0u}};\n}}\n", .{});
    return out.toOwnedSliceSentinel(alloc, 0);
}

/// `header` as a kernel of `cb` under `active_decode` needs it: SPEC1's gets its table's rows ahead of the helpers, on
/// `alloc` (a build's arena: MLX copies a kernel's header).
pub fn withTableRows(alloc: std.mem.Allocator, comptime cb: exl3.Codebook, header: [:0]const u8) ![:0]const u8 {
    if (!cb.tabledHashMask()) return header;
    const rows = try tableRowsText(alloc, kernels.active_decode.table.?);
    defer alloc.free(rows);
    return std.mem.concatWithSentinel(alloc, u8, &.{ rows, header }, 0);
}

/// `getNamedKernel` for a kernel of codebook `cb` under `active_decode`: a SPEC1 kernel carries its table's rows and
/// spells the table in its name, since MLX keys a library by name.
fn tableKernel(slot: *?mlx.mlx_fast_metal_kernel, comptime cb: exl3.Codebook, comptime name: [:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [:0]const u8, comptime header: [:0]const u8) !mlx.mlx_fast_metal_kernel {
    if (slot.*) |k| return k;
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    return kernels.getNamedKernel(slot, (try tableKernelName(alloc, cb, name)).ptr, ins, outs, source, try withTableRows(alloc, cb, header));
}

/// MLX keys a kernel library by name, so a SPEC1 kernel's name spells its table (on `alloc`, a build's arena).
pub fn tableKernelName(alloc: std.mem.Allocator, comptime cb: exl3.Codebook, comptime name: [:0]const u8) ![:0]const u8 {
    if (!cb.tabledHashMask()) return name;
    return std.fmt.allocPrintSentinel(alloc, "{s}_t{s}", .{ name, codebook_tables.tables[tableOf(kernels.active_decode)].digest }, 0);
}

pub fn naxHeader(comptime cb: exl3.Codebook, comptime win: exl3.Window) [:0]const u8 {
    return comptime kernels.GEMM_NAX_INCLUDES ++ codebookHelpers(cb, win) ++ kernels.GEMM_NAX_FRAGS;
}

/// A weight kernel under the active decode parameters: its own slot, name and
/// header.
pub fn codebookKernel(slots: *KernelSlots, comptime base: [:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [:0]const u8) !mlx.mlx_fast_metal_kernel {
    return codebookKernelWith(slots, base, ins, outs, source, "");
}

/// `codebookKernel` with helpers of its own appended to the codebook header.
pub fn codebookKernelWith(slots: *KernelSlots, comptime base: [:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [:0]const u8, comptime frags: [:0]const u8) !mlx.mlx_fast_metal_kernel {
    if (kernels.research_codebook) |research| {
        if (kernels.active_decode.codebook != .mcg) return error.ResearchCodebookNeedsMcg;
        switch (research) {
            inline else => |rc| switch (kernels.active_decode.window) {
                inline else => |win| {
                    const slot = &slots[2 * N_CB + @as(usize, @intFromEnum(rc))][win.index()];
                    if (slot.*) |k| return k;
                    const name = comptime base ++ "_rc_" ++ @tagName(rc) ++ winSuffix(win);
                    const header = comptime researchHelpers(rc, win) ++ frags ++ stagedTail(rc, win);
                    if (comptime !stagedResearch(rc)) return kernels.getNamedKernel(slot, name, ins, outs, source, header);
                    const body = try std.mem.concatWithSentinel(std.heap.page_allocator, u8, &.{ stagedPrologue(rc), source }, 0);
                    defer std.heap.page_allocator.free(body);
                    return kernels.getNamedKernel(slot, name, ins, outs, body, header);
                },
            },
        }
    }
    switch (kernels.active_decode.codebook) {
        inline else => |cb| switch (kernels.active_decode.window) {
            inline else => |win| {
                if (@import("builtin").is_test and foldScale(cb) != 1 and kernels.unfolded_for_test) return tableKernel(
                    &slots[slotRow(cb, true)][win.index()],
                    cb,
                    comptime base ++ cbSuffix(cb) ++ "_unfolded" ++ winSuffix(win),
                    ins,
                    outs,
                    source,
                    comptime decodeHelpers(cb, win, 1) ++ frags,
                );
                return tableKernel(
                    &slots[slotRow(cb, false)][win.index()],
                    cb,
                    comptime base ++ cbSuffix(cb) ++ winSuffix(win),
                    ins,
                    outs,
                    source,
                    comptime codebookHelpers(cb, win) ++ frags,
                );
            },
        },
    }
}

pub fn CfgCache(comptime Key: type, comptime CAP: usize) type {
    return struct {
        const Self = @This();
        keys: [CAP]Key = @splat(std.mem.zeroes(Key)),
        cfgs: [CAP]?mlx.mlx_fast_metal_kernel_config = @splat(null),
        used: [CAP]u64 = @splat(0),
        tick: u64 = 0,

        pub fn get(self: *Self, key: Key) ?mlx.mlx_fast_metal_kernel_config {
            for (self.cfgs, 0..) |c, i| {
                if (c != null and std.meta.eql(self.keys[i], key)) {
                    self.tick += 1;
                    self.used[i] = self.tick;
                    return c.?;
                }
            }
            return null;
        }

        pub fn put(self: *Self, key: Key, cfg: mlx.mlx_fast_metal_kernel_config) void {
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

fn n40NaxReaderExact(comptime cb: exl3.Codebook, comptime win: exl3.Window, comptime raw: bool) !void {
    return n40WeightReaderExact(cb, win, raw, false);
}

fn n40WeightReaderExact(comptime cb: exl3.Codebook, comptime win: exl3.Window, comptime raw: bool, comptime lane_reader: bool) !void {
    return kernels.weightReaderExact(40, cb, win, raw, if (lane_reader) .lane else .nax);
}

// Both helpers, every codeword: the MMA operand `exl3_pairh`, and a GEMV's `exl3_decode2` as its
// f32 sums carry it (the weight over the fold) and as `exl3_out` stores it, each against the host
// decode. Alpha's and spec1's halves are often subnormal; spec1's half sum rounds, alike on both.
/// The host decode of one codeword at the probe rate n40.
fn hostWeight(cw: u16, dec: exl3.Decode) u16 {
    return if (dec.hashMask(.{ .n = 40 })) |hm| exl3.decodeHashMask(cw, hm) else exl3.decodeCodeword(cw, dec.codebook);
}

test "exl3 n40 NAX codewords and decoded weights are exact" {
    try n40NaxReaderExact(.mcg, .w12, true);
    try n40NaxReaderExact(.mcg, .w12, false);
    try n40NaxReaderExact(.mul1, .w8, false);
    try n40NaxReaderExact(.mul1, .w16, false);
}

test "exl3 n42 funnel readers preserve decoded weights at w15 and a narrowed window" {
    inline for (.{ .lane, .nax, .simdmat }) |reader| {
        inline for (.{ exl3.Window.w15, exl3.Window.w12 }) |win| try kernels.weightReaderExact(42, .mcg, win, false, reader);
    }
}

test "exl3 n36 simdgroup reader codewords and decoded weights are exact" {
    try kernels.weightReaderExact(36, .mcg, .w12, true, .simdmat);
    inline for (.{ exl3.Codebook.mcg, exl3.Codebook.mul1 }) |cb| {
        inline for (.{ exl3.Window.w8, exl3.Window.w12, exl3.Window.w16 }) |win| {
            try kernels.weightReaderExact(36, cb, win, false, .simdmat);
        }
    }
}

test "exl3 n40 decode lane codewords and decoded weights are exact" {
    try n40WeightReaderExact(.mcg, .w12, true, true);
    try n40WeightReaderExact(.mcg, .w12, false, true);
    try n40WeightReaderExact(.mul1, .w8, false, true);
}

test "exl3 MCG, spec1 and std1 decode every codeword, masked and unmasked, through both decode helpers" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const source =
        \\constexpr uint N = 40u;
        \\const uint i = thread_position_in_grid.x;
        \\const uint2 cw = uint2(codes[i], codes[65535u - i]);
        \\pairs[i] = as_type<uint>(exl3_pairh(cw));
        \\const float2 w = exl3_decode2(cw);
        \\stored[i] = as_type<uint>(half2(half(exl3_out(w.x)), half(exl3_out(w.y))));
        \\carried[i] = as_type<uint>(w.x);
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
    for (0..3) |_| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{65536}, 1, .uint32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 128, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 65536, 1, 1));
    // A window is one mask, w16 none; a table's rows are data the per-rate reader tests vary.
    inline for (.{ exl3.Codebook.mcg, exl3.Codebook.spec1, exl3.Codebook.std1 }) |cb| inline for (.{ exl3.Window.w12, exl3.Window.w16 }) |win| for (@as([]const ?u8, if (cb == .spec1) &.{kernels.SPEC1_W15.table} else if (cb == .std1) &.{kernels.STD1_W15.table} else &.{null})) |table| {
        const dec: exl3.Decode = .{ .codebook = cb, .window = win, .table = table };
        kernels.setDecodeParams(dec);
        defer kernels.setDecodeParams(.mul1);
        var slot: ?mlx.mlx_fast_metal_kernel = null;
        var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
        defer arena.deinit();
        const kernel = try kernels.getNamedKernel(&slot, (try tableKernelName(arena.allocator(), cb, comptime "exl3_half_pairs" ++ cbSuffix(cb) ++ winSuffix(win))).ptr, &.{"codes"}, &.{ "pairs", "stored", "carried" }, source, try withTableRows(arena.allocator(), cb, comptime codebookHelpers(cb, win)));
        defer _ = mlx.mlx_fast_metal_kernel_free(kernel);
        var outputs = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(outputs);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs, kernel, inputs, cfg, s));
        for (0..3) |o| {
            var result = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(result);
            try mlx.check(mlx.mlx_vector_array_get(&result, outputs, o));
            try mlx.check(mlx.mlx_array_eval(result));
            const got = mlx.mlx_array_data_uint32(result) orelse return error.U32Unreadable;
            for (codes, 0..) |cw, i| {
                const lo = hostWeight(@as(u16, @intCast(cw)) & win.mask(), dec);
                const hi = hostWeight(@as(u16, @intCast(65535 - cw)) & win.mask(), dec);
                const want = if (o < 2)
                    @as(u32, lo) | (@as(u32, hi) << 16)
                else
                    @as(u32, @bitCast(exl3.f16BitsToF32(lo) / @as(f32, @floatFromInt(foldScale(cb)))));
                try t.expectEqual(want, got[i]);
            }
        }
    };
}

test "exl3 research codebooks decode every codeword like their scalar reference" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    defer kernels.setResearchCodebook(null);
    defer kernels.setDecodeParams(.mul1);
    const source =
        \\const uint i = thread_position_in_grid.x;
        \\result[i] = as_type<uint>(half2(exl3_decode2(uint2(codes[i], codes[65535u - i]))));
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
    for (std.enums.values(ResearchCodebook)) |rc| {
        for ([_]exl3.Window{ .w12, .w15 }) |win| {
            kernels.setDecodeParams(.{ .codebook = .mcg, .window = win });
            kernels.setResearchCodebook(rc);
            var slots: KernelSlots = no_kernels;
            const kernel = try codebookKernel(&slots, "exl3_research_pairs", &.{"codes"}, &.{"result"}, source);
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
                const lo = researchDecode(@as(u16, @intCast(cw)) & win.mask(), rc);
                const hi = researchDecode(@as(u16, @intCast(65535 - cw)) & win.mask(), rc);
                try t.expectEqual(@as(u32, lo) | (@as(u32, hi) << 16), got[i]);
            }
        }
    }
}

// PAIR1 at every row of the pack contract: the odd weight's window alone decodes both weights, through the
// MMA operand and through a GEMV's folded sums, bit-identical to the host's `decodePair`.
test "exl3 PAIR1 decodes every codeword at every rate through both decode helpers" {
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    const source =
        \\const uint i = thread_position_in_grid.x;
        \\const uint2 cw = uint2(codes[i], codes[65535u - i]);
        \\pairs[i] = as_type<uint>(exl3_pairh(cw));
        \\const float2 w = exl3_decode2(cw);
        \\stored[i] = as_type<uint>(half2(half(exl3_out(w.x)), half(exl3_out(w.y))));
        \\carried[i] = as_type<uint>(w.x);
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
    for (0..3) |_| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{65536}, 1, .uint32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 128, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 65536, 1, 1));
    inline for (0..21) |row| {
        const n = 24 + 2 * row;
        const pair = exl3.pair1Pair(.{ .n = n }).?;
        var slot: ?mlx.mlx_fast_metal_kernel = null;
        const kernel = try kernels.getNamedKernel(&slot, comptime std.fmt.comptimePrint("exl3_pair1_pairs_n{d}", .{n}), &.{"codes"}, &.{ "pairs", "stored", "carried" }, comptime std.fmt.comptimePrint("constexpr uint N = {d}u;\n", .{n}) ++ source, comptime codebookHelpers(.pair1, .w16));
        defer _ = mlx.mlx_fast_metal_kernel_free(kernel);
        var outputs = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(outputs);
        try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs, kernel, inputs, cfg, s));
        for (0..3) |o| {
            var result = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(result);
            try mlx.check(mlx.mlx_vector_array_get(&result, outputs, o));
            try mlx.check(mlx.mlx_array_eval(result));
            const got = mlx.mlx_array_data_uint32(result) orelse return error.U32Unreadable;
            for (codes, 0..) |_, i| {
                const w = exl3.decodePair(@intCast(65535 - i), pair);
                const want = if (o < 2)
                    @as(u32, w[0]) | (@as(u32, w[1]) << 16)
                else
                    @as(u32, @bitCast(exl3.f16BitsToF32(w[0]) / @as(f32, @floatFromInt(foldScale(.pair1)))));
                try t.expectEqual(want, got[i]);
            }
        }
    }
}

test "exl3 n48 funnel readers preserve codewords and weights" {
    inline for ([_]exl3.Codebook{.mcg}) |cb| {
        inline for ([_]bool{ false, true }) |lane_reader| {
            try kernels.weightReaderExact(48, cb, .w12, true, if (lane_reader) .lane else .nax);
            inline for (.{ exl3.Window.w8, exl3.Window.w16 }) |win| try kernels.weightReaderExact(48, cb, win, false, if (lane_reader) .lane else .nax);
        }
    }
}

// PAIR1's pair reads the odd weight's window alone; every reader hands exl3_pairh a trellis-adjacent
// (even, odd) pair, so each decodes each rate's weights as the host does. The raw readers hold every rate's bits and
// the decode helpers every rate's row: here K1.5, the last two-word and the first three-word lane read of an odd
// first codeword, and K4.
test "exl3 PAIR1 weight readers decode K1.5 to K4 exactly" {
    inline for (.{ 24, 44, 46, 64 }) |n| {
        if (n != 64) try kernels.weightReaderExact(n, .pair1, .w16, false, .lane);
        inline for (.{ .nax, .simdmat }) |reader| try kernels.weightReaderExact(n, .pair1, .w16, false, reader);
    }
}

// The NAX fragment, simdgroup-matrix and decode-lane readers decode spec1 and std1 through the same
// helper as every other codebook; held to the host decode on w15 at K1.5, a three-word lane read and K4 (the per-rate
// table's test holds every rate's own row).
test "exl3 spec1 and std1 weight readers decode K1.5 to K4 exactly at w15" {
    inline for (.{ exl3.Codebook.spec1, exl3.Codebook.std1 }) |cb| inline for (.{ 24, 42, 64 }) |n| {
        if (n != 64) try kernels.weightReaderExact(n, cb, .w15, false, .lane);
        inline for (.{ .nax, .simdmat }) |reader| try kernels.weightReaderExact(n, cb, .w15, false, reader);
    };
}

test "exl3 SPEC1 per-rate table: every reader decodes every rate with its own row" {
    inline for (kernels.PER_RATE_NS) |n| {
        if (n != 64) try kernels.weightReaderExactAt(n, .spec1, .w12, false, .lane, kernels.SPEC1_PER_RATE.table);
        inline for (.{ .nax, .simdmat }) |reader| try kernels.weightReaderExactAt(n, .spec1, .w12, false, reader, kernels.SPEC1_PER_RATE.table);
    }
}

test "exl3 SPEC1 below K1.5 and above K4: every reader decodes every rate with its own row" {
    inline for (.{kernels.SPEC1_PER_RATE}) |dec| inline for (kernels.SPEC1_WIDE_NS) |n| {
        try kernels.weightReaderExactAt(n, .spec1, dec.window, false, .lane, dec.table);
        inline for (.{ .nax, .simdmat }) |reader| try kernels.weightReaderExactAt(n, .spec1, dec.window, false, reader, dec.table);
    };
}

test "exl3 K3.5 PAIR1 lane words match the host across every word and tile boundary" {
    try kernels.weightReaderExact(56, .pair1, .w16, false, .lane);
}

test "exl3 K5 NAX fragments match host SPEC1 weights at both halfword alignments" {
    try kernels.weightReaderExactAt(80, .spec1, .w15, false, .nax, kernels.SPEC1_W15.table);
}

test "exl3 routed prefill fragments match host PAIR1 weights at K2.5 K3 and K3.5" {
    inline for (.{ 40, 48, 56 }) |n| try kernels.weightReaderExact(n, .pair1, .w16, false, .nax);
}
