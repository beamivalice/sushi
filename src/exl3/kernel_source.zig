const codebook_tables = @import("codebook_tables.zig");
const exl3 = @import("expert_exl3.zig");
const mlx = @import("mlx_host").mlx;
const std = @import("std");
const kernels = @import("expert_exl3_kernels.zig");

const N_CB = exl3.Codebook.count;

const N_WIN = exl3.Window.count;

const N_TABLES = codebook_tables.tables.len;

const TABLED_ROW0 = 2 * N_CB;

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
        .spec1 => "_spec1",
        .std1 => "_std1",
        .pair1 => "_pair1",
    };
}

/// The power of two `exl3_decode2` leaves out of a codebook's weights: a hash-mask codebook's
/// scale, restored by `exl3_out` once per stored sum instead of once per decoded pair.
fn foldScale(comptime cb: exl3.Codebook) u32 {
    return if (cb.tabledHashMask()) exl3.hash_mask_scale else if (cb == .pair1) PAIR1_SCALE else 1;
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
    \\template<bool Wide, bool Funnel = false> struct exl3_lane_type {
    \\  using type = ulong;
    \\  static inline type read(const device uint *words, uint end, uint W, bool full) {
    \\    return exl3_window(words, end, W, full);
    \\  }
    \\};
    \\// exl3_window's bits with a third word, built as two 32-bit funnels of adjacent words
    \\// (s < 32): OR-ing the third word into a 64-bit shift costs the ALU-bound decode GEMV far more.
    \\template<> struct exl3_lane_type<false, true> {
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
    \\  constexpr bool wide = exl3_lane_sh(N, 0u) + 16u > 64u;
    \\  constexpr bool full = exl3_lane_sh(N, EXL3_LANE_FIRST) + 16u > 32u + (low < 32u ? low : 32u);
    \\  const uint end = W * lane + W;
    \\  return exl3_lane_type<wide, !wide && full>::read(words, end, W, full);
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
        else => unreachable,
    };
    return comptime helpersWithBody(body, win);
}

fn helpersWithBody(comptime body: [:0]const u8, comptime win: exl3.Window) [:0]const u8 {
    const mask: [:0]const u8 = comptime if (win == .w16) "" else std.fmt.comptimePrint("  cw &= uint2(0x{X}u);\n", .{win.mask()});
    return comptime "static inline half2 exl3_pairh(uint2 cw) {\n" ++ mask ++ body ++ "\n}\n" ++
        \\static inline float2 exl3_decode2(uint2 cw) { return float2(exl3_pairh(cw)); }
        \\static inline float exl3_out(float v) { return v; }
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
    return comptime "template<uint N> static inline half2 exl3_unscaled_n(uint2 cw) {\n" ++
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

/// `header` as a kernel of `cb` under `active_decode` needs it: SPEC1's gets its table's rows ahead of the helpers.
/// Allocated for the life of the process, so call it only where a kernel is built.
pub fn withTableRows(comptime cb: exl3.Codebook, header: [:0]const u8) ![:0]const u8 {
    if (!cb.tabledHashMask()) return header;
    const alloc = std.heap.page_allocator;
    const rows = try tableRowsText(alloc, kernels.active_decode.table.?);
    defer alloc.free(rows);
    return std.mem.concatWithSentinel(alloc, u8, &.{ rows, header }, 0);
}

/// `getNamedKernel` for a kernel of codebook `cb` under `active_decode`: a SPEC1 kernel carries its table's rows and
/// spells the table in its name, since MLX keys a library by name.
fn tableKernel(slot: *?mlx.mlx_fast_metal_kernel, comptime cb: exl3.Codebook, comptime name: [:0]const u8, ins: []const [*:0]const u8, outs: []const [*:0]const u8, source: [:0]const u8, comptime header: [:0]const u8) !mlx.mlx_fast_metal_kernel {
    if (slot.*) |k| return k;
    return kernels.getNamedKernel(slot, (try tableKernelName(cb, name)).ptr, ins, outs, source, try withTableRows(cb, header));
}

/// MLX keys a kernel library by name, so a SPEC1 kernel's name spells its table.
pub fn tableKernelName(comptime cb: exl3.Codebook, comptime name: [:0]const u8) ![:0]const u8 {
    if (!cb.tabledHashMask()) return name;
    return std.fmt.allocPrintSentinel(std.heap.page_allocator, "{s}_t{s}", .{ name, codebook_tables.tables[tableOf(kernels.active_decode)].digest }, 0);
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
// decode. Spec1's halves are often subnormal and its half sum rounds, alike on both.
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

test "exl3 MCG, spec1 and std1 decode every codeword at windows 8 through 16, through both decode helpers" {
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
    inline for (.{ exl3.Codebook.mcg, exl3.Codebook.spec1, exl3.Codebook.std1 }) |cb| inline for (8..17) |bits| for (@as([]const ?u8, if (cb == .spec1) &.{ kernels.SPEC1_W15.table, kernels.SPEC1_PER_RATE.table } else if (cb == .std1) &.{kernels.STD1_W15.table} else &.{null})) |table| {
        const win = comptime exl3.Window.fromBits(bits).?;
        const dec: exl3.Decode = .{ .codebook = cb, .window = win, .table = table };
        kernels.setDecodeParams(dec);
        defer kernels.setDecodeParams(.mul1);
        var slot: ?mlx.mlx_fast_metal_kernel = null;
        const kernel = try kernels.getNamedKernel(&slot, (try tableKernelName(cb, comptime "exl3_half_pairs" ++ cbSuffix(cb) ++ winSuffix(win))).ptr, &.{"codes"}, &.{ "pairs", "stored", "carried" }, source, try withTableRows(cb, comptime codebookHelpers(cb, win)));
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
            inline for (8..17) |bits| try kernels.weightReaderExact(48, cb, comptime exl3.Window.fromBits(bits).?, false, if (lane_reader) .lane else .nax);
        }
    }
}

// PAIR1's pair reads the odd weight's window alone; every reader hands exl3_pairh a trellis-adjacent
// (even, odd) pair, so each decodes each rate's weights as the host does.
test "exl3 PAIR1 weight readers decode every K1.5 to K4 rate exactly" {
    inline for (12..33) |half| {
        const n: u32 = 2 * half;
        if (n != 64) try kernels.weightReaderExact(n, .pair1, .w16, false, .lane);
        inline for (.{ .nax, .simdmat }) |reader| try kernels.weightReaderExact(n, .pair1, .w16, false, reader);
    }
}

// The NAX fragment, simdgroup-matrix and decode-lane readers decode spec1 and std1 through the same
// helper as every other codebook; held to the host decode at each K1.5 to K4 rate on w15.
test "exl3 spec1 and std1 weight readers decode every K1.5 to K4 rate exactly at w15" {
    inline for (.{ exl3.Codebook.spec1, exl3.Codebook.std1 }) |cb| inline for (12..33) |half| {
        const n: u32 = 2 * half;
        if (n != 64) try kernels.weightReaderExact(n, cb, .w15, false, .lane);
        inline for (.{ .nax, .simdmat }) |reader| try kernels.weightReaderExact(n, cb, .w15, false, reader);
    };
}

test "exl3 SPEC1 per-rate table: every reader decodes every rate with its own row, at any window" {
    inline for (.{ exl3.Window.w8, exl3.Window.w12, exl3.Window.w15, exl3.Window.w16 }) |win| inline for (kernels.PER_RATE_NS) |n| {
        if (n != 64) try kernels.weightReaderExactAt(n, .spec1, win, false, .lane, kernels.SPEC1_PER_RATE.table);
        inline for (.{ .nax, .simdmat }) |reader| try kernels.weightReaderExactAt(n, .spec1, win, false, reader, kernels.SPEC1_PER_RATE.table);
    };
}

test "exl3 SPEC1 below K1.5 and above K4: every reader decodes every rate with its own row" {
    inline for (.{ kernels.SPEC1_W15, kernels.SPEC1_PER_RATE }) |dec| inline for (kernels.SPEC1_WIDE_NS) |n| {
        try kernels.weightReaderExactAt(n, .spec1, dec.window, false, .lane, dec.table);
        inline for (.{ .nax, .simdmat }) |reader| try kernels.weightReaderExactAt(n, .spec1, dec.window, false, reader, dec.table);
    };
}
