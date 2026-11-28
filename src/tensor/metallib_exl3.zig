//! EXL3 / STQ precompiled custom kernels catalog (`sushi.metallib`).
//!
//! Single source of truth for ALL Sushi Cooler EXL3 kernels across 7 metallib units:
//!   - `exl3_simd`:      non-codebook utilities + single-rate SIMD GEMMs / decode GEMVs / coop
//!   - `exl3_rows_simd`: `project_rows` SIMD kernels + all 7 `rows_finish*` kernels
//!   - `exl3_rg_simd`:   `rate_groups` mixed-rate decode kernels (unified rate switch + FC tile_major/clamp)
//!   - `exl3_fz_simd`:   fused SIMD kernels (`sushi_exl3_fz_*`, `sushi_rg_fold_*`, `sushi_gated_fold_*` affine)
//!   - `exl3_gemm_nax`:  NAX prefill GEMMs (`gemm_nax[_tm|_ref]`, `trunk_dense_nax`, `prefill_grid_transpose`)
//!   - `exl3_rows_nax`:  `project_rows` NAX kernels (`rows_nax`, `_dyn`, `_pend`, `_thru`, `_thru_counted*`)
//!   - `exl3_fz_nax`:    NAX fused kernels (`sushi_exl3_fz_{pair,down}*_snax*`, `sushi_rg_fold_*_tr_*`, `sushi_gated_fold_*_tr_*`)
//!
//! Codebook parameterization uses a 2-way compile-time structural split (`_scalar` vs `_pair1`,
//! preserving `EXL3_LANE_FIRST = 0` vs `1` register widths and shuffles) combined with Metal
//! `[[function_constant]]` slots for codebook kind, window mask, hash-mask `(mult, mask)` / table id,
//! fold flag, tile-major flag, and SwiGLU clamp limit — eliminating all runtime Metal source splicing
//! while preserving bit-exact arithmetic under `-fno-fast-math`.

const std = @import("std");
const custom = @import("metallib_custom.zig");
const spec = @import("metallib_spec.zig");
const exl3_mod = @import("sushi_exl3");

const format = exl3_mod.format;
const tables = exl3_mod.codebook_tables;
const kernels = exl3_mod.kernels;
const kernel_src = exl3_mod.kernel_source;
const project_rows = exl3_mod.project_rows;
const sorted_windows = exl3_mod.sorted_windows;
const coop_rows2 = exl3_mod.coop_rows2;
const trunk_gemm = exl3_mod.trunk_gemm;
const prefill_grid = exl3_mod.prefill_grid;
const rate_groups = exl3_mod.rate_groups;
const shared_trellis = exl3_mod.shared_trellis;
const riders_gated = exl3_mod.riders_gated;
const fuse = exl3_mod.fuse;

const Dtype = custom.Dtype;
const KernelSpec = custom.KernelSpec;
const ParamSpec = custom.ParamSpec;
const InputSlot = custom.InputSlot;
const OutputSlot = custom.OutputSlot;
const CtVal = custom.CtVal;
const TypeRef = custom.TypeRef;
const RuntimeInput = custom.RuntimeInput;
const RuntimeTarg = custom.RuntimeTarg;
const Resolved = custom.Resolved;

// ── Reserved EXL3 Function-Constant Slot Indices (disjoint from kernel param FCs 0..35, 48..59) ──

const fc_cb_kind: u32 = 36; // 0=mul1, 1=mcg, 2=hash_mask (Part A)
const fc_win_mask: u32 = 37; // uint bitcast to i32 (Part A)
const fc_hash_mult: u32 = 38; // uint bitcast to i32 (Part A single-rate scalar)
const fc_hash_mask: u32 = 39; // uint bitcast to i32 (Part A single-rate scalar)
const fc_table_id: u32 = 40; // 0=alpha, 1=spec1-w15, 2=spec1-w16, 3=std1, 4=spec1-test-per-rate (Part A mixed)
const fc_clamp_limit: u32 = 41; // int >= 0 (rg_mid / rg_down / fz_mid)
const fc_b_cb_kind: u32 = 42; // Part B cb_kind
const fc_b_win_mask: u32 = 43; // Part B win_mask
const fc_b_hash_mult: u32 = 44; // Part B hash_mult
const fc_b_hash_mask: u32 = 45; // Part B hash_mask
const fc_rate_mask_g: u32 = 46; // Active rate-index bitmask for gate/down mixed-rate switch
const fc_rate_mask_u: u32 = 47; // Active rate-index bitmask for up mixed-rate switch

const fc_fold_out: u32 = spec.fc_bool_base + 12; // FC_B12 (index 60)
const fc_tile_major: u32 = spec.fc_bool_base + 13; // FC_B13 (index 61)
const fc_b_fold_out: u32 = spec.fc_bool_base + 14; // FC_B14 (index 62)

// ── Rate Tables ──

/// Every PAIR1 rate (24..64 even = 21 values).
const pair1_rates: [21]i32 = blk: {
    var r: [21]i32 = undefined;
    for (&r, 0..) |*v, i| v.* = @intCast(24 + 2 * i);
    break :blk r;
};

/// Served & tested scalar rates for mixed-rate switches and secondary variants (30 values).
const served_scalar_rates = [_]i32{
    16, 18, 20, 22,
    24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64,
    72, 80, 96, 112, 128,
};

/// Common trunk & test rates for secondary NAX/SIMD rows variants (26 values).
const trunk_scalar_rates = [_]i32{
    16, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64,
    72, 80, 96, 112, 128,
};

fn inRateSlice(slice: []const i32, n: u32) bool {
    for (slice) |r| if (r == @as(i32, @intCast(n))) return true;
    return false;
}

// ── Type Shorthands ──

fn fix(d: Dtype) TypeRef {
    return .{ .fixed = d };
}
fn par(n: []const u8) TypeRef {
    return .{ .param = n };
}
fn inSlot(n: []const u8, ty: TypeRef) InputSlot {
    return .{ .name = n, .ty = ty };
}
fn idxSlot(n: []const u8) InputSlot {
    return .{ .name = n, .ty = fix(.uint32), .allow_i32_as_u32 = true };
}
fn outSlot(n: []const u8, ty: TypeRef) OutputSlot {
    return .{ .name = n, .ty = ty };
}
fn fcInt(n: []const u8) ParamSpec {
    return .{ .name = n, .kind = .fc_int };
}
fn ign(n: []const u8) ParamSpec {
    return .{ .name = n, .kind = .ignored };
}

const floats = [_]Dtype{ .float16, .bfloat16, .float32 };
const bf_f16 = [_]Dtype{ .bfloat16, .float16 };
const bf_f32 = [_]Dtype{ .bfloat16, .float32 };

// ── String Replacement Helpers (Build-Time Only) ──

fn replaceOnceAlloc(a: std.mem.Allocator, src: []const u8, old: []const u8, new: []const u8) ![]u8 {
    const at = std.mem.indexOf(u8, src, old) orelse std.debug.panic("metallib_exl3: missing anchor '{s}'", .{old});
    return std.mem.concat(a, u8, &.{ src[0..at], new, src[at + old.len ..] });
}

fn replaceAllAlloc(a: std.mem.Allocator, src: []const u8, old: []const u8, new: []const u8) ![]u8 {
    if (std.mem.indexOf(u8, src, old) == null) std.debug.panic("metallib_exl3: missing anchor '{s}'", .{old});
    return std.mem.replaceOwned(u8, a, src, old, new);
}

fn replaceOptAlloc(a: std.mem.Allocator, src: []const u8, old: []const u8, new: []const u8) ![]u8 {
    if (std.mem.indexOf(u8, src, old) == null) return a.dupe(u8, src);
    return std.mem.replaceOwned(u8, a, src, old, new);
}

fn isIdentStart(c: u8) bool {
    return std.ascii.isAlphabetic(c) or c == '_';
}
fn isIdent(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '_';
}

/// Same identifier renamer as `src/exl3/fuse.zig:renameB` for build-time fused sources/headers.
fn renamePartB(
    a: std.mem.Allocator,
    src: []const u8,
    io_names: []const []const u8,
    int_names: []const []const u8,
    rewrite_tgp: bool,
) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    var i: usize = 0;
    while (i < src.len) {
        const c = src[i];
        if (std.ascii.isDigit(c)) {
            const start = i;
            while (i < src.len and isIdent(src[i])) i += 1;
            try out.appendSlice(a, src[start..i]);
            continue;
        }
        if (!isIdentStart(c)) {
            try out.append(a, c);
            i += 1;
            continue;
        }
        const start = i;
        while (i < src.len and isIdent(src[i])) i += 1;
        const word = src[start..i];
        const member = start > 0 and src[start - 1] == '.';
        if (member) {
            try out.appendSlice(a, word);
        } else if (rewrite_tgp and std.mem.eql(u8, word, "threadgroup_position_in_grid")) {
            try out.appendSlice(a, "fz_tgp");
        } else if (std.mem.startsWith(u8, word, "exl3_") or std.mem.startsWith(u8, word, "EXL3_")) {
            try out.appendSlice(a, "sx_");
            try out.appendSlice(a, word);
        } else if (for (io_names) |n| {
            if (std.mem.eql(u8, n, word)) break true;
        } else false) {
            try out.appendSlice(a, "s_");
            try out.appendSlice(a, word);
        } else if (for (int_names) |n| {
            if (std.mem.eql(u8, n, word)) break true;
        } else false) {
            try out.appendSlice(a, "S_");
            try out.appendSlice(a, word);
        } else {
            try out.appendSlice(a, word);
        }
    }
    return out.toOwnedSlice(a);
}

// ── Unified FC Codebook Header Emission (Build-Time Only) ──

const CodecHeaderOpts = struct {
    is_pair1: bool,
    mixed_rate: bool = false,
    secondary: bool = false,
};

fn emitPair1RowsTable(out: *std.ArrayList(u8), a: std.mem.Allocator) !void {
    try out.appendSlice(a,
        \\struct exl3_pair_row { uint mult; uint rot; uint mask0; uint mask1; };
        \\__attribute__((always_inline)) static inline constexpr exl3_pair_row exl3_pair_row_at(uint n) {
        \\  switch (n) {
        \\
    );
    var n: u32 = format.Rate.min_n;
    while (n <= format.Rate.max_n) : (n += 2) {
        const p = format.pair1Pair(.{ .n = n }) orelse continue;
        try out.print(a, "    case {d}u: return {{0x{X}u, {d}u, 0x{X}u, 0x{X}u}};\n", .{ n, p.mult, p.rot, p.mask0, p.mask1 });
    }
    try out.appendSlice(a,
        \\  }
        \\  return {0u, 0u, 0u, 0u};
        \\}
        \\
    );
}

fn emitMixedHashTables(out: *std.ArrayList(u8), a: std.mem.Allocator) !void {
    const alpha = format.Codebook.alpha.hashMask().?;
    try out.appendSlice(a,
        \\struct exl3_hash_row { uint mult; uint mask; };
        \\__attribute__((always_inline)) static inline exl3_hash_row exl3_hash_row_at(int tid, uint n) {
        \\  if (tid == 0) {
        \\
    );
    try out.print(a, "    return {{0x{X}u, 0x{X}u}};\n", .{ alpha.mult, alpha.mask });
    try out.appendSlice(a, "  }\n");

    const table_names = [_]struct { id: u32, name: []const u8 }{
        .{ .id = 1, .name = "spec1-w15" },
        .{ .id = 2, .name = "spec1-w16" },
        .{ .id = 3, .name = "std1" },
    };
    for (table_names) |tn| {
        const tbl = tables.lookup(tn.name).?;
        try out.print(a, "  if (tid == {d}) {{\n    switch (n) {{\n", .{tn.id});
        for (tbl.rows, 0..) |r, i| {
            try out.print(a, "      case {d}u: return {{0x{X}u, 0x{X}u}};\n", .{ tbl.rateOf(i).n, r[0], r[1] });
        }
        try out.appendSlice(a, "      default: return {0u, 0u};\n    }\n  }\n");
    }
    // tid == 4: spec1-test-per-rate (digest a2f2fe40) — matches codebook_tables.syntheticPairs().
    try out.appendSlice(a, "  if (tid == 4) {\n    switch (n) {\n");
    const synth_masks = [_]u32{ 0x8FFE8FD0, 0x8FE08FC0, 0x8FF08FE0, 0x8F8E8FD8 };
    for (0..57) |i| {
        const n: u32 = @intCast(16 + 2 * i);
        const mult: u32 = 0x840C1F37 +% n *% 0x2E0B5;
        const mask: u32 = synth_masks[i % synth_masks.len];
        try out.print(a, "      case {d}u: return {{0x{X}u, 0x{X}u}};\n", .{ n, mult, mask });
    }
    try out.appendSlice(a, "      default: return {0u, 0u};\n    }\n  }\n  return {0u, 0u};\n}\n");
}

fn buildCodecHeader(a: std.mem.Allocator, opts: CodecHeaderOpts) ![]u8 {
    var raw: std.ArrayList(u8) = .empty;
    errdefer raw.deinit(a);

    const cb_fc = if (opts.secondary) fc_b_cb_kind else fc_cb_kind;
    const win_fc = if (opts.secondary) fc_b_win_mask else fc_win_mask;
    const mult_fc = if (opts.secondary) fc_b_hash_mult else fc_hash_mult;
    const mask_fc = if (opts.secondary) fc_b_hash_mask else fc_hash_mask;
    const fold_b = (if (opts.secondary) fc_b_fold_out else fc_fold_out) - spec.fc_bool_base;

    if (opts.is_pair1) {
        try raw.appendSlice(a, "#define EXL3_LANE_FIRST 1u\n");
        try emitPair1RowsTable(&raw, a);
        try raw.print(a,
            \\template<uint N> __attribute__((always_inline)) static inline half2 exl3_unscaled_n(uint2 cw) {{
            \\  constexpr exl3_pair_row r = exl3_pair_row_at(N);
            \\  cw &= uint2(as_type<uint>(FC_I{d}));
            \\  const uint p = cw.y * r.mult;
            \\  const half4 h = as_type<half4>(uint2(p & r.mask0, rotate(p, r.rot) & r.mask1));
            \\  return half2(h.x + h.y, h.z + h.w);
            \\}}
            \\template<uint N> __attribute__((always_inline)) static inline half2 exl3_pairh_n(uint2 cw) {{
            \\  return exl3_unscaled_n<N>(cw) * half2(4096.0h);
            \\}}
            \\template<uint N> __attribute__((always_inline)) static inline float2 exl3_decode2_n(uint2 cw) {{
            \\  return FC_B{d} ? float2(exl3_unscaled_n<N>(cw)) : float2(exl3_pairh_n<N>(cw));
            \\}}
            \\__attribute__((always_inline)) static inline float exl3_out(float v) {{
            \\  return FC_B{d} ? (v * 4096.0f) : v;
            \\}}
            \\template<uint N> __attribute__((always_inline)) static inline float exl3_slot_n(thread const ushort *cw, uint slot) {{
            \\  const float2 w = exl3_decode2_n<N>(uint2(0u, uint(cw[slot | 1u])));
            \\  return (slot & 1u) != 0u ? w.y : w.x;
            \\}}
            \\#define exl3_pairh(cw) exl3_pairh_n<N>((cw))
            \\#define exl3_decode2(cw) exl3_decode2_n<N>((cw))
            \\#define exl3_slot(cw, slot) exl3_slot_n<N>((cw), (slot))
            \\
        , .{ win_fc, fold_b, fold_b });
    } else {
        try raw.appendSlice(a, "#define EXL3_LANE_FIRST 0u\n");
        if (opts.mixed_rate) {
            try emitMixedHashTables(&raw, a);
        }
        try raw.print(a,
            \\template<uint N> __attribute__((always_inline)) static inline half2 exl3_unscaled_n(uint2 cw) {{
            \\  cw &= uint2(as_type<uint>(FC_I{d}));
            \\  if (FC_I{d} == 2) {{
            \\
        , .{ win_fc, cb_fc });
        if (opts.mixed_rate) {
            try raw.print(a,
                \\    const exl3_hash_row r = exl3_hash_row_at(FC_I{d}, N);
                \\    const half4 h = as_type<half4>((cw * uint2(r.mult)) & uint2(r.mask));
                \\    return half2(h.x + h.y, h.z + h.w);
                \\
            , .{fc_table_id});
        } else {
            try raw.print(a,
                \\    const half4 h = as_type<half4>((cw * uint2(as_type<uint>(FC_I{d}))) & uint2(as_type<uint>(FC_I{d})));
                \\    return half2(h.x + h.y, h.z + h.w);
                \\
            , .{ mult_fc, mask_fc });
        }
        try raw.print(a,
            \\  }} else if (FC_I{d} == 1) {{
            \\    const uint2 r = ((cw * uint2(0xCBAC1FEDu)) & uint2(0x8FFF8FFFu)) ^ uint2(0x3B603B60u);
            \\    const half4 h = as_type<half4>(r);
            \\    return half2(h.x + h.y, h.z + h.w);
            \\  }} else {{
            \\    const uint2 mixed = cw * uint2(0x83DCD12Du);
            \\    const uint2 pair_sums = (mixed & uint2(0x00FF00FFu)) + ((mixed >> uint2(8u)) & uint2(0x00FF00FFu));
            \\    const uint2 byte_sum = uint2(0x6400u) + (pair_sums & uint2(0xFFFFu)) + (pair_sums >> uint2(16u));
            \\    const half2 h = as_type<half2>(ushort2(byte_sum & uint2(0xFFFFu)));
            \\    return fma(h, as_type<half2>(ushort2(0x1EEEu)), as_type<half2>(ushort2(0xC931u)));
            \\  }}
            \\}}
            \\template<uint N> __attribute__((always_inline)) static inline half2 exl3_pairh_n(uint2 cw) {{
            \\  return (FC_I{d} == 2) ? (exl3_unscaled_n<N>(cw) * half2(4096.0h)) : exl3_unscaled_n<N>(cw);
            \\}}
            \\template<uint N> __attribute__((always_inline)) static inline float2 exl3_decode2_n(uint2 cw) {{
            \\  return (FC_I{d} == 2 && FC_B{d}) ? float2(exl3_unscaled_n<N>(cw)) : float2(exl3_pairh_n<N>(cw));
            \\}}
            \\__attribute__((always_inline)) static inline float exl3_out(float v) {{
            \\  return (FC_I{d} == 2 && FC_B{d}) ? (v * 4096.0f) : v;
            \\}}
            \\template<uint N> __attribute__((always_inline)) static inline float exl3_slot_n(thread const ushort *cw, uint slot) {{
            \\  return exl3_decode2_n<N>(uint2(uint(cw[slot]), 0u)).x;
            \\}}
            \\#define exl3_pairh(cw) exl3_pairh_n<N>((cw))
            \\#define exl3_decode2(cw) exl3_decode2_n<N>((cw))
            \\#define exl3_slot(cw, slot) exl3_slot_n<N>((cw), (slot))
            \\
        , .{ cb_fc, cb_fc, cb_fc, fold_b, cb_fc, fold_b });
    }
    try raw.appendSlice(a, kernel_src.HELPERS_TAIL);

    if (!opts.secondary) return raw.toOwnedSlice(a);
    const base = try raw.toOwnedSlice(a);
    defer a.free(base);
    return renamePartB(a, base, &.{}, &.{}, false);
}

// ── Build-Time Source / Header Builders ──

const hdr_scalar: usize = 0;
const hdr_pair1: usize = 1;
const hdr_scalar_smat: usize = 2;
const hdr_pair1_smat: usize = 3;
const hdr_scalar_nax: usize = 4;
const hdr_pair1_nax: usize = 5;
const hdr_scalar_rows_simd: usize = 6;
const hdr_pair1_rows_simd: usize = 7;
const hdr_scalar_rows_simd_cnt: usize = 8;
const hdr_scalar_rows_nax: usize = 9;
const hdr_pair1_rows_nax: usize = 10;
const hdr_scalar_rows_nax_cnt: usize = 11;
const hdr_rg_scalar: usize = 12;
const hdr_rg_pair1: usize = 13;

fn buildRowsNaxFragsFc(a: std.mem.Allocator) ![]u8 {
    // Replace template<uint N, uint R32, uint SW> rows_wfrag with runtime-arg R32, SW (FCs).
    const old_sig =
        \\template<uint N, uint R32, uint SW>
        \\static inline nfrag rows_wfrag(const device uint *words, uint lane) {
        \\  if constexpr (N == 80u && SW != 0u) return nax_k5_frag<N>(words, lane);
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
        \\      lo = uint2(nax_funnel(w1, w0, s), nax_funnel(w0, wm, sp));
        \\      hi = uint2(nax_funnel(w2, w1, sp), nax_funnel(w2, w1, s));
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
    ;
    const new_sig =
        \\template<uint N>
        \\__attribute__((always_inline)) static inline nfrag rows_wfrag(const device uint *words, uint lane, uint R32, uint SW) {
        \\  if constexpr (N == 80u) { if (SW != 0u) return nax_k5_frag<N>(words, lane); }
        \\  if constexpr (N != 64u && nax_w32_fits(N)) {
        \\    if (R32 != 0u) {
        \\      constexpr uint W = N / 2u;
        \\      const uint b = (lane >> 3u) & 1u;
        \\      const uint end = 8u * N * (lane >> 4u) + N * (lane & 7u) + (nax_end(N, 1u, 5u) - nax_end(N, 0u, 5u)) * b + nax_end(N, 0u, 5u);
        \\      const uint s0 = nax_sh(N, b, 0u) - 16u;
        \\      const uint s1 = nax_sh(N, b, 1u) - 16u;
        \\      const uint s4 = nax_sh(N, b, 4u);
        \\      uint2 lo, hi;
        \\      if constexpr (N == 96u) {
        \\        if (SW != 0u) {
        \\          const uint l = (end - 1u) >> 5u;
        \\          const uint s = (0u - end) & 31u;
        \\          const uint sp = (s + 16u) & 31u;
        \\          const uint wm = words[l == 1u ? W - 1u : l - 2u];
        \\          const uint w0 = words[l - 1u];
        \\          const uint w1 = words[l];
        \\          const uint w2 = words[l + 1u];
        \\          lo = uint2(nax_funnel(w1, w0, s), nax_funnel(w0, wm, sp));
        \\          hi = uint2(nax_funnel(w2, w1, sp), nax_funnel(w2, w1, s));
        \\        } else {
        \\          lo = nax_win2(words, end, W);
        \\          hi = nax_win2(words, end + W, W);
        \\        }
        \\      } else {
        \\        lo = nax_win2(words, end, W);
        \\        hi = nax_win2(words, end + W, W);
        \\      }
        \\      const half2 p0 = exl3_pairh(uint2((lo.y >> s0) & 0xffffu, (lo.y >> s1) & 0xffffu));
        \\      const half2 p1 = exl3_pairh(uint2((lo.x >> s4) & 0xffffu, lo.x & 0xffffu));
        \\      const half2 p2 = exl3_pairh(uint2((hi.y >> s0) & 0xffffu, (hi.y >> s1) & 0xffffu));
        \\      const half2 p3 = exl3_pairh(uint2((hi.x >> s4) & 0xffffu, hi.x & 0xffffu));
        \\      return nfrag(p0.x, p0.y, p2.x, p2.y, p1.x, p1.y, p3.x, p3.y);
        \\    }
        \\  }
        \\  return nax_wfrag_k<N>(words, lane);
        \\}
    ;
    return replaceOnceAlloc(a, project_rows.ROWS_NAX_FRAGS, old_sig, new_sig);
}

fn exl3HeaderFn(a: std.mem.Allocator, arg: usize) ![]const u8 {
    switch (arg) {
        hdr_scalar => return buildCodecHeader(a, .{ .is_pair1 = false }),
        hdr_pair1 => return buildCodecHeader(a, .{ .is_pair1 = true }),
        hdr_scalar_smat => {
            const c = try buildCodecHeader(a, .{ .is_pair1 = false });
            return std.mem.concat(a, u8, &.{ c, kernels.GEMM_SIMDMAT_FRAGS });
        },
        hdr_pair1_smat => {
            const c = try buildCodecHeader(a, .{ .is_pair1 = true });
            return std.mem.concat(a, u8, &.{ c, kernels.GEMM_SIMDMAT_FRAGS });
        },
        hdr_scalar_nax => {
            const c = try buildCodecHeader(a, .{ .is_pair1 = false });
            return std.mem.concat(a, u8, &.{ kernels.GEMM_NAX_INCLUDES, c, kernels.GEMM_NAX_FRAGS });
        },
        hdr_pair1_nax => {
            const c = try buildCodecHeader(a, .{ .is_pair1 = true });
            return std.mem.concat(a, u8, &.{ kernels.GEMM_NAX_INCLUDES, c, kernels.GEMM_NAX_FRAGS });
        },
        hdr_scalar_rows_simd => {
            const c = try buildCodecHeader(a, .{ .is_pair1 = false });
            return std.mem.concat(a, u8, &.{ c, project_rows.ROWS_HELPERS });
        },
        hdr_pair1_rows_simd => {
            const c = try buildCodecHeader(a, .{ .is_pair1 = true });
            return std.mem.concat(a, u8, &.{ c, project_rows.ROWS_HELPERS });
        },
        hdr_scalar_rows_simd_cnt => {
            const c = try buildCodecHeader(a, .{ .is_pair1 = false });
            return std.mem.concat(a, u8, &.{ c, project_rows.ROWS_HELPERS, project_rows.COUNTED_STORE });
        },
        hdr_scalar_rows_nax => {
            const c = try buildCodecHeader(a, .{ .is_pair1 = false });
            const f = try buildRowsNaxFragsFc(a);
            return std.mem.concat(a, u8, &.{ kernels.GEMM_NAX_INCLUDES, c, f });
        },
        hdr_pair1_rows_nax => {
            const c = try buildCodecHeader(a, .{ .is_pair1 = true });
            const f = try buildRowsNaxFragsFc(a);
            return std.mem.concat(a, u8, &.{ kernels.GEMM_NAX_INCLUDES, c, f });
        },
        hdr_scalar_rows_nax_cnt => {
            const c = try buildCodecHeader(a, .{ .is_pair1 = false });
            const f = try buildRowsNaxFragsFc(a);
            return std.mem.concat(a, u8, &.{ kernels.GEMM_NAX_INCLUDES, c, f, project_rows.COUNTED_STORE });
        },
        hdr_rg_scalar => return buildCodecHeader(a, .{ .is_pair1 = false, .mixed_rate = true }),
        hdr_rg_pair1 => return buildCodecHeader(a, .{ .is_pair1 = true, .mixed_rate = true }),
        else => unreachable,
    }
}

/// Converts runtime `CLAMP_LIMIT` / `vals` / `counts` / `prepared` / `seen` / `if constexpr` to FC-safe MSL.
fn fcCleanSource(a: std.mem.Allocator, base: []const u8) ![]u8 {
    var s = try a.dupe(u8, base);
    s = try replaceOptAlloc(a, s, "if constexpr (CLAMP_LIMIT > 0) {", "if (CLAMP_LIMIT > 0) {");
    s = try replaceOptAlloc(a, s, "threadgroup float vals[uint(TOPK) * 128u];", "threadgroup float vals[32u * 128u];");
    s = try replaceOptAlloc(a, s, "threadgroup float vals[(uint(TOPK) + 1u) * 128u];", "threadgroup float vals[32u * 128u];");
    s = try replaceOptAlloc(a, s, "threadgroup uint counts[uint(EXPERTS)];", "threadgroup uint counts[512u];");
    s = try replaceOptAlloc(a, s, "constexpr uint IT = uint(IDIM) / TILE;", "const uint IT = uint(IDIM) / TILE;");
    s = try replaceOptAlloc(a, s, "constexpr uint OT = uint(ODIM) / TILE;", "const uint OT = uint(ODIM) / TILE;");
    s = try replaceOptAlloc(a, s, "constexpr uint OT = uint(ODIM) / 16u;", "const uint OT = uint(ODIM) / 16u;");
    s = try replaceOptAlloc(a, s, "constexpr uint SGS = uint(IDIM) / 128u;", "const uint SGS = uint(IDIM) / 128u;");
    s = try replaceOptAlloc(a, s, "constexpr uint KSPAN = uint(IDIM) / uint(NSPLIT);", "const uint KSPAN = uint(IDIM) / uint(NSPLIT);");
    s = try replaceOptAlloc(a, s, "constexpr uint NW = (uint(NSLOTS) + 31u) / 32u;", "const uint NW = (uint(NSLOTS) + 31u) / 32u;");
    s = try replaceOptAlloc(a, s, "constexpr uint CHUNK_FIT = (4096u / uint(GROUP)) / 128u * 128u;\nconstexpr uint CHUNK = CHUNK_FIT < KSPAN ? CHUNK_FIT : KSPAN;", "const uint CHUNK_FIT = (4096u / uint(GROUP)) / 128u * 128u;\nconst uint CHUNK = CHUNK_FIT < KSPAN ? CHUNK_FIT : KSPAN;");
    s = try replaceOptAlloc(a, s, "threadgroup half prepared[uint(GROUP) * CHUNK];", "threadgroup half prepared[4096];");
    s = try replaceOptAlloc(a, s, "threadgroup half prepared[2u * KSPAN];", "threadgroup half prepared[7680];");
    s = try replaceOptAlloc(a, s, "threadgroup half prepared[uint(IDIM)];", "threadgroup half prepared[4096];");
    s = try replaceOptAlloc(a, s, "threadgroup float partial[SERIAL_REDUCTION?1024:2048];", "threadgroup float partial[2048];");
    s = try replaceOptAlloc(a, s, "threadgroup float partial[SERIAL_REDUCTION?1024:3072];", "threadgroup float partial[3072];");
    s = try replaceOptAlloc(a, s, "if constexpr(SERIAL_REDUCTION)", "if (SERIAL_REDUCTION)");
    s = try replaceOptAlloc(a, s, "threadgroup float fin[NORM];", "threadgroup float fin[4096];");
    s = try replaceOptAlloc(a, s, "threadgroup float kfin[KV_D];", "threadgroup float kfin[2048];");
    s = try replaceOptAlloc(a, s, "constexpr uint U = uint(KV_D) * sizeof(OutT) / 16u;", "const uint U = uint(KV_D) * sizeof(OutT) / 16u;");
    s = try replaceOptAlloc(a, s, "constexpr uint Q_SGS = uint(NORM) / 128u;", "const uint Q_SGS = uint(NORM) / 128u;");
    s = try replaceOptAlloc(a, s, "constexpr uint KHEAD = uint(KV_D - RD);", "const uint KHEAD = uint(KV_D - RD);");
    s = try replaceOptAlloc(a, s, "float sv[HC];", "float sv[8];");
    return s;
}

/// Adds FC_I41 (`EXL3_CLAMP_LIMIT`) SwiGLU clamping to `DECODE_MID_SOURCE` / `DOWN_FUSED_SOURCE`.
fn addFcClamp(a: std.mem.Allocator, base: []const u8) ![]u8 {
    var s = try a.dupe(u8, base);
    for ([_]u8{ 'g', 'u' }) |w| for (0..4) |k| {
        var old_buf: [32]u8 = undefined;
        var new_buf: [32]u8 = undefined;
        const old = std.fmt.bufPrint(&old_buf, "const float {c}{d} = ", .{ w, k }) catch unreachable;
        const new = std.fmt.bufPrint(&new_buf, "float {c}{d} = ", .{ w, k }) catch unreachable;
        s = try replaceOnceAlloc(a, s, old, new);
    };
    const clamp_block =
        \\if (FC_I41 > 0) {
        \\  g0 = min(g0, float(FC_I41)); g1 = min(g1, float(FC_I41));
        \\  g2 = min(g2, float(FC_I41)); g3 = min(g3, float(FC_I41));
        \\  u0 = clamp(u0, -float(FC_I41), float(FC_I41));
        \\  u1 = clamp(u1, -float(FC_I41), float(FC_I41));
        \\  u2 = clamp(u2, -float(FC_I41), float(FC_I41));
        \\  u3 = clamp(u3, -float(FC_I41), float(FC_I41));
        \\}
        \\const float ysig0 =
    ;
    return replaceOnceAlloc(a, s, "const float ysig0 =", clamp_block);
}

/// Transforms `project_rows` SIMD or NAX body to FC-safe MSL (`IDIM, ODIM, ROWS, GROUPS, NSPLIT, CHUNK, FOLD, SB1..3, R32, SW` as FCs).
fn transformRowsSource(a: std.mem.Allocator, base: []const u8, is_nax: bool, is_counted: bool) ![]u8 {
    var s = try a.dupe(u8, base);
    // Common ROWS_HEAD CT->FC conversions
    s = try replaceOnceAlloc(a, s, "constexpr uint OT = uint(ODIM) / TILE;", "const uint OT = uint(ODIM) / TILE;");
    s = try replaceOnceAlloc(a, s, "constexpr uint KT = uint(IDIM) / uint(NSPLIT) / TILE;", "const uint KT = uint(IDIM) / uint(NSPLIT) / TILE;");
    s = try replaceOnceAlloc(a, s, "constexpr uint CT = uint(CHUNK) / TILE;", "const uint CT = uint(CHUNK) / TILE;");
    s = try replaceOnceAlloc(a, s, "constexpr uint PREP_F = uint(ROWS) * uint(CHUNK) / 2u;", "constexpr uint PREP_F = 2048u;");
    s = try replaceAllAlloc(a, s, "if constexpr (GROUPS > 1) {", "if (GROUPS > 1) {");
    s = try replaceOnceAlloc(a, s, "if constexpr (uint(SB1) < uint(ODIM)) {", "if (uint(SB1) < uint(ODIM)) {");
    s = try replaceOptAlloc(a, s, "static_assert(GROUPS == 1, \"a launch that reads partials has one bank\");", "");

    if (is_nax) {
        s = try replaceOnceAlloc(a, s, "constexpr uint PART_F = uint(ROWS) * KPARTS * 128u;", "constexpr uint PART_F = uint(ROW_CAP) * KPARTS * 128u;");
        s = try replaceOnceAlloc(a, s, "rows_wfrag<N, R32, SW>(wp, lane)", "rows_wfrag<N>(wp, lane, uint(R32), uint(SW))");
        s = try replaceOnceAlloc(a, s, "rows_wfrag<N, R32, SW>(wp + PACKED_W, lane)", "rows_wfrag<N>(wp + PACKED_W, lane, uint(R32), uint(SW))");
        s = try replaceOnceAlloc(a, s, "if constexpr (FOLD || KPARTS > 1u) tgm[", "if (FOLD != 0 || KPARTS > 1u) tgm[");
        s = try replaceOnceAlloc(a, s, "if constexpr (FOLD) {", "if (FOLD != 0) {");
        s = try replaceOnceAlloc(a, s, "} else if constexpr (KPARTS > 1u) {", "} else if (KPARTS > 1u) {");
    } else {
        s = try replaceOnceAlloc(a, s, "constexpr uint PART_F = uint(ROWS) * KPARTS * TGT * 16u;", "constexpr uint PART_F = uint(ROW_CAP) * KPARTS * TGT * 16u;");
        s = try replaceOnceAlloc(a, s, "float acc[ROWS][2][2] = {};", "float acc[ROW_CAP][2][2] = {};");
        s = try replaceOnceAlloc(a, s, "if constexpr (uint(ROWS) > 0u) {\n        constexpr uint RN = uint(ROWS) - 0u < 4u ? uint(ROWS) - 0u : 4u;", "if (uint(ROWS) > 0u) {\n        const uint RN = uint(ROWS) < 4u ? uint(ROWS) : 4u;");
        s = try replaceOnceAlloc(a, s, "if constexpr (uint(ROWS) > 4u) {\n        constexpr uint RN = uint(ROWS) - 4u < 4u ? uint(ROWS) - 4u : 4u;", "if constexpr (ROW_CAP > 4) { if (uint(ROWS) > 4u) {\n        const uint RN = uint(ROWS) - 4u < 4u ? uint(ROWS) - 4u : 4u;");
        // Close the extra `{` opened by `if constexpr (ROW_CAP > 4) {` right after the second block.
        s = try replaceOnceAlloc(a, s, "acc[4u + r][o][p >> 1u] = fma(i2.y, wd[o][p].y, fma(i2.x, wd[o][p].x, acc[4u + r][o][p >> 1u]));\n            }\n          }\n        }\n      }", "acc[4u + r][o][p >> 1u] = fma(i2.y, wd[o][p].y, fma(i2.x, wd[o][p].x, acc[4u + r][o][p >> 1u]));\n            }\n          }\n        }\n      } }");
        s = try replaceOnceAlloc(a, s, "if constexpr (FOLD) {", "if (FOLD != 0) {");
    }

    if (is_counted) {
        s = try replaceOptAlloc(a, s, "static_assert(PART_F >= uint(ROWS) * 128u, \"the finish sums a block's rows in tgm\");", "static_assert(PART_F >= 128u, \"tgm holds at least one block\");");
        s = try replaceOptAlloc(a, s, "static_assert(GROUPS == 2, \"a pair is two banks\");", "");
        // Replace COUNTED_SUM block (handles both plain PB=y and pair PB=(y0 + ...))
        const old_sum_prefix =
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
            \\          device metal::atomic_float *p = (device metal::atomic_float *)
        ;
        const old_sum_suffix =
            \\ + (size_t)r * uint(NSPLIT) * uint(ODIM) + blk * 128u + (i - r * 128u);
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
        ;
        const p_start = std.mem.indexOf(u8, s, old_sum_prefix) orelse std.debug.panic("metallib_exl3: missing COUNTED_SUM prefix", .{});
        const pb_start = p_start + old_sum_prefix.len;
        const pb_end = std.mem.indexOfPos(u8, s, pb_start, old_sum_suffix) orelse std.debug.panic("metallib_exl3: missing COUNTED_SUM suffix", .{});
        const pb_expr = s[pb_start..pb_end];
        const new_sum = try std.fmt.allocPrint(a,
            \\    constexpr uint TT = SGS * 32u;
            \\    const uint PER = (uint(ROWS) * 128u + TT - 1u) / TT;
            \\    if (PER <= 3u && uint(NSPLIT) <= 24u && PER * uint(NSPLIT) <= 96u) {{
            \\      float pv[3][24];
            \\      for (uint k = 0u; k < PER; k++) {{
            \\        const uint i = lid + k * TT;
            \\        if (i < uint(ROWS) * 128u) {{
            \\          const uint r = i / 128u;
            \\          device metal::atomic_float *p = (device metal::atomic_float *){s} + (size_t)r * uint(NSPLIT) * uint(ODIM) + blk * 128u + (i - r * 128u);
            \\          for (uint s = 0u; s < uint(NSPLIT); s++) pv[k][s] = metal::atomic_load_explicit(p + (size_t)s * uint(ODIM), metal::memory_order_relaxed);
            \\        }}
            \\      }}
            \\      for (uint k = 0u; k < PER; k++) {{
            \\        const uint i = lid + k * TT;
            \\        if (i < uint(ROWS) * 128u) {{
            \\          float acc = 0.0f;
            \\          for (uint s = 0u; s < uint(NSPLIT); s++) acc += pv[k][s];
            \\          tgm[i] = acc;
            \\        }}
            \\      }}
            \\    }} else {{
            \\      for (uint k = 0u; k < PER; k++) {{
            \\        const uint i = lid + k * TT;
            \\        if (i < uint(ROWS) * 128u) {{
            \\          const uint r = i / 128u;
            \\          device metal::atomic_float *p = (device metal::atomic_float *){s} + (size_t)r * uint(NSPLIT) * uint(ODIM) + blk * 128u + (i - r * 128u);
            \\          float acc = 0.0f;
            \\          for (uint s = 0u; s < uint(NSPLIT); s++) acc += metal::atomic_load_explicit(p + (size_t)s * uint(ODIM), metal::memory_order_relaxed);
            \\          tgm[i] = acc;
            \\        }}
            \\      }}
            \\    }}
        , .{ pb_expr, pb_expr });
        s = try std.mem.concat(a, u8, &.{ s[0..p_start], new_sum, s[pb_end + old_sum_suffix.len ..] });

        // COUNTED_NORM FC conversions (present on _rms)
        s = try replaceOptAlloc(a, s, "constexpr uint NB = uint(ODIM) / 128u;\n    constexpr uint PAIRS = uint(ROWS) * NB;\n    constexpr uint PP = (PAIRS + SGS - 1u) / SGS;\n    static_assert(NB <= 32u, \"a row's block sums are one simdgroup's lanes\");\n    threadgroup float t_sums[uint(ROWS) * 32u];\n    threadgroup float t_inv[ROWS];", "const uint NB = uint(ODIM) / 128u;\n    const uint PAIRS = uint(ROWS) * NB;\n    const uint PP = (PAIRS + SGS - 1u) / SGS;\n    threadgroup float t_sums[uint(ROW_CAP) * 32u];\n    threadgroup float t_inv[ROW_CAP];");
        s = try replaceOptAlloc(a, s, "float tx[PP][4];", "float tx[(uint(ROW_CAP) * 32u + SGS - 1u) / SGS][4];");
        // COUNTED_PAIR_FIN FC conversions (present on _pair)
        s = try replaceOptAlloc(a, s, "constexpr uint RPS = (uint(ROWS) + SGS - 1u) / SGS;\n  float4 gv[RPS];", "const uint RPS = (uint(ROWS) + SGS - 1u) / SGS;\n  float4 gv[(uint(ROW_CAP) + SGS - 1u) / SGS];");
        s = try replaceOptAlloc(a, s, "#pragma clang loop unroll(full)\n", "");
    }
    return s;
}

/// Builds a mixed-rate `rate_groups` kernel body with unrolled `proj = 0u` / `proj = 1u` rate switches,
/// `FC_B13` (`EXL3_TILE_MAJOR`) retile, and `FC_I41` (`EXL3_CLAMP_LIMIT`) SwiGLU clamp.
const RgKind = enum { pair, down, pair_prepared, down_prepared };

fn buildMixedRateBody(a: std.mem.Allocator, kind: RgKind, group: u32, is_pair1: bool) ![]u8 {
    const down = (kind == .down or kind == .down_prepared);
    const rates: []const i32 = if (is_pair1) &pair1_rates else &served_scalar_rates;

    const raw_base: []const u8 = if (group > 0) switch (kind) {
        .pair_prepared => switch (group) {
            2 => comptime kernels.pairPreparedGroupedSource(2),
            4 => comptime kernels.pairPreparedGroupedSource(4),
            8 => comptime kernels.pairPreparedGroupedSource(8),
            16 => comptime kernels.pairPreparedGroupedSource(16),
            else => unreachable,
        },
        .down_prepared => switch (group) {
            2 => comptime kernels.downGroupedSource(2),
            4 => comptime kernels.downGroupedSource(4),
            8 => comptime kernels.downGroupedSource(8),
            16 => comptime kernels.downGroupedSource(16),
            else => unreachable,
        },
        else => unreachable,
    } else switch (kind) {
        .pair => kernels.PAIR_GEMV_SOURCE,
        .down => kernels.DOWN_FUSED_SOURCE,
        .pair_prepared => kernels.PAIR_GEMV_PREPARED_SOURCE,
        .down_prepared => kernels.DOWN_PREPARED_SOURCE,
    };

    var base = try fcCleanSource(a, raw_base);
    if (kind == .down) {
        base = try addFcClamp(a, base);
    }

    const n_decl = if (group > 0)
        "constexpr uint N = uint(NHW);\nconstexpr uint PACKED_W = N / 2u;\n"
    else
        "constexpr uint N = uint(NHW);\nconstexpr uint PACKED_HW = N;\nconstexpr uint PACKED_W = N / 2u;\n";
    const pair_loop = "for (uint proj = 0u; proj < 2u; proj++) {\n";

    const d = std.mem.indexOf(u8, base, n_decl) orelse std.debug.panic("metallib_exl3: rg missing N_DECL", .{});
    const split_anchor = if (group > 0) "const uint prow = " else if (down) "float acc[OTPT][8] = {};" else pair_loop;
    const anchor_pos = std.mem.indexOf(u8, base, split_anchor) orelse std.debug.panic("metallib_exl3: rg missing split anchor", .{});
    const z = if (group > 0 or down) base.len else std.mem.lastIndexOfScalar(u8, base, '}') orelse unreachable;

    const body_slice = if (group > 0 or down) base[anchor_pos..z] else base[anchor_pos + pair_loop.len .. z];
    const old_t = if (group > 0)
        "(const device uint *)(trellis + ((size_t)eid * (size_t)IT * (size_t)OT) * N)"
    else if (down)
        "(const device uint*)(trellis + ((size_t)eid * (size_t)IT * (size_t)OT) * PACKED_HW)"
    else
        "(const device uint *)(trellis + ((size_t)eid * (size_t)IT * (size_t)OT) * PACKED_HW)";
    const new_t = if (down)
        "((const device uint *)trellis + (size_t)meta[eid * 8u + 6u])"
    else
        "((const device uint *)trellis + (size_t)meta[eid * 8u + 4u + proj])";

    var body = try replaceAllAlloc(a, body_slice, old_t, new_t);
    if (group > 0 and !down) {
        body = try replaceAllAlloc(a, body, pair_loop, "{\n");
    }
    // FC_B13 (EXL3_TILE_MAJOR) retiling in-place (constant-folded at PSO creation).
    body = try replaceAllAlloc(a, body, "SGS * OT * PACKED_W", "(FC_B13 ? (SGS * 2u * PACKED_W) : (SGS * OT * PACKED_W))");
    if (std.mem.indexOf(u8, body, "((size_t)(tk0 + sg) * (size_t)OT + ot) * PACKED_W") != null) {
        body = try replaceAllAlloc(
            a,
            body,
            "((size_t)(tk0 + sg) * (size_t)OT + ot) * PACKED_W",
            "(FC_B13 ? ((((size_t)(ot >> 1u) * (size_t)IT + (tk0 + sg)) << 1u) + (ot & 1u)) : ((size_t)(tk0 + sg) * (size_t)OT + ot)) * PACKED_W",
        );
    } else if (std.mem.indexOf(u8, body, "((size_t)sg * (size_t)OT + ot) * PACKED_W") != null) {
        body = try replaceAllAlloc(
            a,
            body,
            "((size_t)sg * (size_t)OT + ot) * PACKED_W",
            "(FC_B13 ? ((((size_t)(ot >> 1u) * (size_t)IT + sg) << 1u) + (ot & 1u)) : ((size_t)sg * (size_t)OT + ot)) * PACKED_W",
        );
    }

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.appendSlice(a, base[0..d]);
    try out.appendSlice(a, base[d + n_decl.len .. anchor_pos]);

    if (down) {
        try out.appendSlice(a, "switch (meta[eid * 8u + 2u]) {\n");
        for (rates, 0..) |r, ri| {
            try out.print(a, "case {d}u: if ((as_type<uint>(FC_I{d}) & (1u << {d}u)) != 0u) {{\nconstexpr uint N = {d}u;\nconstexpr uint PACKED_HW = N;\nconstexpr uint PACKED_W = N / 2u;\n", .{ r, fc_rate_mask_g, ri, r });
            try out.appendSlice(a, body);
            try out.appendSlice(a, "\n} break;\n");
        }
        try out.appendSlice(a, "default: break;\n}\n");
    } else {
        // Explicitly unrolled proj=0 (gate) then proj=1 (up): O(R) cases instead of O(R^2), zero rolled-loop overhead!
        // Each case is guarded by FC_I46/FC_I47 so PSO specialization dead-code-eliminates every inactive rate.
        for ([_]u32{ 0, 1 }) |proj| {
            const mask_fc = if (proj == 0) fc_rate_mask_g else fc_rate_mask_u;
            try out.print(a, "{{\nconstexpr uint proj = {d}u;\nswitch (meta[eid * 8u + {d}u]) {{\n", .{ proj, proj });
            for (rates, 0..) |r, ri| {
                try out.print(a, "case {d}u: if ((as_type<uint>(FC_I{d}) & (1u << {d}u)) != 0u) {{\nconstexpr uint N = {d}u;\nconstexpr uint PACKED_HW = N;\nconstexpr uint PACKED_W = N / 2u;\n", .{ r, mask_fc, ri, r });
                try out.appendSlice(a, body);
                try out.appendSlice(a, "\n} break;\n");
            }
            try out.appendSlice(a, "default: break;\n}\n}\n");
        }
    }
    if (group == 0 and !down) try out.appendSlice(a, base[z + 1 ..]);
    return out.toOwnedSlice(a);
}

/// Wraps a single-rate kernel body (whose `N` is driven by FC `NHW` / `S_NHW`) in an FC rate switch.
/// Hoists the prologue before `N` (`threadgroup` arrays and grid setup) outside the switch so `metal -c`
/// emits one threadgroup allocation instead of one per rate, while Metal specializes the switch at PSO creation.
fn wrapFcRateSwitch(a: std.mem.Allocator, body_in: []const u8, nhw_expr: []const u8, is_pair1: bool, extra_n_decl: []const u8) ![]u8 {
    const rates: []const i32 = if (is_pair1) &pair1_rates else &served_scalar_rates;
    const d = std.mem.indexOf(u8, body_in, extra_n_decl) orelse std.debug.panic("metallib_exl3: rate switch missing N decl '{s}'", .{extra_n_decl});
    const tail = body_in[d + extra_n_decl.len ..];
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.appendSlice(a, body_in[0..d]);
    try out.print(a, "switch (uint({s})) {{\n", .{nhw_expr});
    for (rates) |r| {
        try out.print(a, "case {d}u: {{\nconstexpr uint N = {d}u;\nconstexpr uint PACKED_HW = N;\nconstexpr uint PACKED_W = N / 2u;\n", .{ r, r });
        try out.appendSlice(a, tail);
        try out.appendSlice(a, "\n} break;\n");
    }
    try out.appendSlice(a, "default: break;\n}\n");
    return out.toOwnedSlice(a);
}

fn emitFuseBranch(out: *std.ArrayList(u8), a: std.mem.Allocator, ax: []const u8, ay: []const u8, body: []const u8, is_a: bool) !void {
    try out.print(a, "{{\nconst uint fz_l = {s};\nconst uint3 fz_tgp = uint3(fz_l % uint({s}), (fz_l / uint({s})) % uint({s}), fz_l / uint({s} * {s}));\n", .{
        if (is_a) "(FZ_B_FIRST != 0 ? fz_lin - uint(FZ_NB) : fz_lin)" else "(FZ_B_FIRST != 0 ? fz_lin : fz_lin - uint(FZ_NA))",
        ax, ax, ay, ax, ay,
    });
    try out.appendSlice(a, body);
    try out.appendSlice(a, "\n}\n");
}

fn buildFusedSource(a: std.mem.Allocator, a_body_raw: []const u8, b_body_raw: []const u8, b_io: []const []const u8, b_ints: []const []const u8) ![]u8 {
    const a_body = try std.mem.replaceOwned(u8, a, a_body_raw, "threadgroup_position_in_grid", "fz_tgp");
    const b_body = try renamePartB(a, b_body_raw, b_io, b_ints, true);
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.appendSlice(a, "const uint fz_lin = uint(threadgroup_position_in_grid.x);\n");
    try out.appendSlice(a, "if (FZ_B_FIRST != 0 ? fz_lin >= uint(FZ_NB) : fz_lin < uint(FZ_NA)) ");
    try emitFuseBranch(&out, a, "FZ_AX", "FZ_AY", a_body, true);
    try out.appendSlice(a, "else ");
    try emitFuseBranch(&out, a, "FZ_BX", "FZ_BY", b_body, false);
    return out.toOwnedSlice(a);
}

fn buildRidersHeaderFc(a: std.mem.Allocator) ![]u8 {
    // Convert qmv_rows<K, GS, FST, RPS> to qmv_rows<RPS>(K, GS, FST, ...) with fixed local arrays [8].
    return a.dupe(u8,
        \\template <int RPS>
        \\inline void qmv_rows(int K, int GS, int FST, const device uint8_t* w, const device bfloat16_t* sc, const device bfloat16_t* bi, const device bfloat16_t* x, uint lane, thread float* result) {
        \\  const int VPT = FST != 0 ? 8 : 4;
        \\  const int BLOCK = VPT * 32;
        \\  const int GROUPS = K / GS;
        \\  const device uint8_t* ws = w + lane * VPT;
        \\  const device bfloat16_t* sp = sc + lane / (GS / VPT);
        \\  const device bfloat16_t* bp = bi + lane / (GS / VPT);
        \\  const device bfloat16_t* xp = x + lane * VPT;
        \\  int k = 0;
        \\  for (; k < (FST != 0 ? K : K - BLOCK); k += BLOCK) {
        \\    float xt[8];
        \\    float sums = 0.0f;
        \\    for (int i = 0; i < VPT; ++i) { sums += xp[i]; xt[i] = xp[i]; }
        \\    for (int row = 0; row < RPS; ++row) {
        \\      uchar codes[8];
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
        \\      float xt[8];
        \\      float sums = 0.0f;
        \\      for (int i = 0; i < remaining; ++i) { sums += xp[i]; xt[i] = xp[i]; }
        \\      for (int row = 0; row < RPS; ++row) {
        \\        uchar codes[8];
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
        \\
    );
}

fn buildRidersBodyFc(a: std.mem.Allocator, raw: []const u8) ![]u8 {
    var s = try a.dupe(u8, raw);
    s = try replaceOptAlloc(a, s, "qmv_rows<K, LGS, 0, 1>(", "qmv_rows<1>(K, LGS, 0, ");
    s = try replaceOptAlloc(a, s, "qmv_rows<K, GS, FAST, 4>(", "qmv_rows<4>(K, GS, FAST, ");
    return s;
}

// ── Static Source Function Dispatchers ──

fn srcClean(comptime raw: []const u8) ?custom.SourceFn {
    if (!custom.is_codegen) return null;
    return struct {
        fn f(a: std.mem.Allocator, _: usize) ![]const u8 {
            return fcCleanSource(a, raw);
        }
    }.f;
}

fn srcRgMid(a: std.mem.Allocator, _: usize) ![]const u8 {
    const clean = try fcCleanSource(a, kernels.DECODE_MID_SOURCE);
    return addFcClamp(a, clean);
}

fn srcTokenPairUnsorted(a: std.mem.Allocator, _: usize) ![]const u8 {
    return replaceOnceAlloc(a, kernels.TOKEN_PAIR_PREPARE_SOURCE, "const uint orig = uint(order[slot]);", "const uint orig = slot;");
}

fn srcCombine(a: std.mem.Allocator, chained: usize) ![]const u8 {
    return rate_groups.combineSource(a, rate_groups.COMBINE_PLANES, chained != 0);
}

fn srcFzPrefillPrep(a: std.mem.Allocator, _: usize) ![]const u8 {
    const a_part: fuse.Part = .{
        .launch = .{ .tgs = .{ 1, 1, 1 }, .threads = 32, .ints = &.{ .{ .name = "IDIM", .value = 0 }, .{ .name = "TOPK", .value = 0 } }, .outs = &.{} },
        .body = kernels.TOKEN_PAIR_PREPARE_SOURCE,
        .ins = &.{ "x", "suhg", "suhu", "slots", "order" },
        .outs = &.{ "yg", "yu" },
    };
    const b_part: fuse.Part = .{
        .launch = .{ .tgs = .{ 1, 1, 1 }, .threads = 32, .ints = &.{ .{ .name = "IDIM", .value = 0 }, .{ .name = "TOPK", .value = 0 } }, .outs = &.{} },
        .body = rate_groups.SHARED_PREPARE_SOURCE,
        .ins = &.{ "x", "suhg", "suhu", "slots" },
        .outs = &.{ "yg", "yu" },
    };
    return fuse.sourceRuntime(a, a_part, b_part);
}

fn srcFzPrefillMid(a: std.mem.Allocator, _: usize) ![]const u8 {
    const a_body = try fcCleanSource(a, rate_groups.MID_PLANES_SOURCE);
    const b_body = try fcCleanSource(a, rate_groups.SHARED_MID_SOURCE);
    const a_part: fuse.Part = .{
        .launch = .{ .tgs = .{ 1, 1, 1 }, .threads = 32, .ints = &.{ .{ .name = "ODIM", .value = 0 }, .{ .name = "CLAMP_LIMIT", .value = 0 } }, .outs = &.{} },
        .body = a_body,
        .ins = &rate_groups.MID_PLANES_INS,
        .outs = &.{"yd"},
    };
    const b_part: fuse.Part = .{
        .launch = .{ .tgs = .{ 1, 1, 1 }, .threads = 32, .ints = &.{ .{ .name = "ODIM", .value = 0 }, .{ .name = "CLAMP_LIMIT", .value = 0 } }, .outs = &.{} },
        .body = b_body,
        .ins = &.{ "ig", "iu", "svhg", "svhu", "suhd" },
        .outs = &.{"yd"},
    };
    return fuse.sourceRuntime(a, a_part, b_part);
}

fn srcClampedMidDown(a: std.mem.Allocator, _: usize) ![]const u8 {
    var s = try fcCleanSource(a, kernels.CLAMPED_MIDDLE_DOWN_SOURCE);
    // Clamped middle down allows IDIM up to 8192 (`expert_exl3_kernels.zig:7940`).
    s = try replaceOnceAlloc(a, s, "threadgroup half prepared[4096];", "threadgroup half prepared[8192];");
    return s;
}

fn srcRows(comptime raw: []const u8, comptime is_nax: bool, comptime is_counted: bool) ?custom.SourceFn {
    if (!custom.is_codegen) return null;
    return struct {
        fn f(a: std.mem.Allocator, _: usize) ![]const u8 {
            return transformRowsSource(a, raw, is_nax, is_counted);
        }
    }.f;
}

fn srcRg(comptime kind: RgKind, comptime group: u32, comptime is_pair1: bool) ?custom.SourceFn {
    if (!custom.is_codegen) return null;
    return struct {
        fn f(a: std.mem.Allocator, _: usize) ![]const u8 {
            return buildMixedRateBody(a, kind, group, is_pair1);
        }
    }.f;
}

fn srcFzPrep(a: std.mem.Allocator, _: usize) ![]const u8 {
    return buildFusedSource(
        a,
        kernels.PAIR_PREP_SOURCE,
        kernels.LANE_PREP_UNSORTED_SOURCE,
        &.{ "x", "suhg", "suhu", "slots", "yg", "yu" },
        &.{ "IDIM", "TOPK" },
    );
}

fn srcFzPrepSnax(a: std.mem.Allocator, _: usize) ![]const u8 {
    return buildFusedSource(
        a,
        kernels.PAIR_PREP_SOURCE,
        "if (thread_index_in_threadgroup == 0u && threadgroup_position_in_grid.x == 0u && threadgroup_position_in_grid.y == 0u) { yg[0] = half(0); yu[0] = half(0); }",
        &.{ "x", "suhg", "suhu", "slots", "yg", "yu" },
        &.{},
    );
}

fn srcFzMid(a: std.mem.Allocator, _: usize) ![]const u8 {
    const a_clean = try fcCleanSource(a, kernels.DECODE_MID_SOURCE);
    const a_clamped = try addFcClamp(a, a_clean);
    const b_clean = try fcCleanSource(a, kernels.DOWN_LANE_MID_SOURCE);
    return buildFusedSource(
        a,
        a_clamped,
        b_clean,
        &.{ "ig", "iu", "svhg", "svhu", "suhd", "slots", "yd" },
        &.{ "ODIM", "CLAMP_LIMIT" },
    );
}

fn srcFzMidSnax(a: std.mem.Allocator, _: usize) ![]const u8 {
    const a_clean = try fcCleanSource(a, kernels.DECODE_MID_SOURCE);
    const a_clamped = try addFcClamp(a, a_clean);
    return buildFusedSource(
        a,
        a_clamped,
        a_clamped,
        &.{ "ig", "iu", "svhg", "svhu", "suhd", "slots", "prepared" },
        &.{ "IDIM", "NSPLIT" },
    );
}

fn hdrFzSnaxStage(comptime a_pair1: bool, comptime b_pair1: bool) ?custom.SourceFn {
    if (!custom.is_codegen) return null;
    return struct {
        fn f(a: std.mem.Allocator, _: usize) ![]const u8 {
            const ah = try buildCodecHeader(a, .{ .is_pair1 = a_pair1, .mixed_rate = true, .secondary = false });
            const bh = try buildCodecHeader(a, .{ .is_pair1 = b_pair1, .mixed_rate = false, .secondary = true });
            const frags_fc = try buildRowsNaxFragsFc(a);
            const frags_sx = try renamePartB(a, frags_fc, &.{}, &.{}, false);
            return std.mem.concat(a, u8, &.{ kernels.GEMM_NAX_INCLUDES, ah, "\n", bh, "\n", frags_sx });
        }
    }.f;
}

fn srcFzSnaxStage(comptime is_down: bool, comptime group: u32, comptime a_pair1: bool, comptime b_pair1: bool) ?custom.SourceFn {
    if (!custom.is_codegen) return null;
    return struct {
        fn f(a: std.mem.Allocator, _: usize) ![]const u8 {
            const a_body = try buildMixedRateBody(a, if (is_down) .down_prepared else .pair_prepared, group, a_pair1);
            var tr_body: []u8 = undefined;
            if (!is_down) {
                const rep =
                    \\const bool upper = (threadgroup_position_in_grid.z & 1u) != 0u;
                    \\const uint split = uint(threadgroup_position_in_grid.z) >> 1u;
                    \\const device half *suh = upper ? suhu : suhg;
                    \\const device half *svh = suh;
                    \\const device ushort *trellis = upper ? tu : tg;
                    \\device float *y = upper ? yu : yg;
                ;
                const s0 = try replaceOnceAlloc(a, project_rows.ROWS_NAX_SOURCE, "const uint split = uint(threadgroup_position_in_grid.y);", rep);
                tr_body = try transformRowsSource(a, s0, true, false);
                tr_body = try replaceAllAlloc(a, tr_body, "uint(KPH)", "1u");
                tr_body = try replaceAllAlloc(a, tr_body, "KPH", "1");
                tr_body = try replaceOnceAlloc(a, tr_body, "static_assert(UNR == 1 || 1 == 1, \"an unrolled walk steps whole chunks of one slice\");", "");
                tr_body = try replaceOnceAlloc(a, tr_body, "nfrag wf[UNR][2];", "nfrag wf[2][2];");
                tr_body = try replaceOnceAlloc(a, tr_body, "uint(ROW_CAP)", "16u");
            } else {
                var s = try replaceOnceAlloc(a, project_rows.ROWS_NAX_PREPPED_SOURCE, "const uint split = uint(threadgroup_position_in_grid.y);", "const uint split = uint(threadgroup_position_in_grid.z);");
                s = try replaceOnceAlloc(a, s, "static_assert(UNR == 1 || KPH == 1, \"an unrolled walk steps whole chunks of one slice\");", "");
                s = try replaceOnceAlloc(a, s, "constexpr uint OT = uint(ODIM) / TILE;", "const uint OT = uint(ODIM) / TILE;");
                s = try replaceOnceAlloc(a, s, "constexpr uint KT = uint(IDIM) / uint(NSPLIT) / TILE;", "const uint KT = uint(IDIM) / uint(NSPLIT) / TILE;");
                s = try replaceOnceAlloc(a, s, "constexpr uint PART_F = uint(ROWS) * KPARTS * 128u;", "constexpr uint PART_F = 16u * KPARTS * 128u;");
                s = try replaceOnceAlloc(a, s, "threadgroup float tgm[(FOLD || KPARTS > 1u) ? PART_F : 1u];", "threadgroup float tgm[1u];");
                s = try replaceOnceAlloc(a, s, "rows_wfrag<N, R32, SW>(wp, lane)", "rows_wfrag<N>(wp, lane, uint(R32), uint(SW))");
                s = try replaceOnceAlloc(a, s, "rows_wfrag<N, R32, SW>(wp + PACKED_W, lane)", "rows_wfrag<N>(wp + PACKED_W, lane, uint(R32), uint(SW))");
                s = try replaceOnceAlloc(a, s, "nfrag wf[UNR][2];", "nfrag wf[2][2];");
                s = try replaceOnceAlloc(a, s, "if constexpr (FOLD || KPARTS > 1u) tgm[", "if constexpr (false) tgm[");
                s = try replaceOnceAlloc(a, s, "if constexpr (FOLD) {", "if constexpr (false) {");
                s = try replaceOnceAlloc(a, s, "} else if constexpr (KPARTS > 1u) {", "} else if constexpr (false) {");
                s = try replaceAllAlloc(a, s, "uint(KPH)", "1u");
                s = try replaceOptAlloc(a, s, "KPH", "1");
                tr_body = s;
            }
            const tr_switched = try wrapFcRateSwitch(a, tr_body, "NHW", b_pair1, "constexpr uint N = uint(NHW);\nconstexpr uint PACKED_W = N / 2u;\n");
            const b_full = try std.mem.concat(a, u8, &.{ "{\ntypedef bfloat16_t OutT;\n", tr_switched, "\n}\n" });
            const b_io: []const []const u8 = if (is_down)
                &.{ "x", "trellis", "svh", "y" }
            else
                &.{ "x", "tg", "tu", "suhg", "suhu", "yg", "yu" };
            const b_ints: []const []const u8 = &.{ "IDIM", "ODIM", "ROWS", "GROUPS", "TGRP", "KPH", "NSPLIT", "CHUNK", "FOLD", "NHW", "SB1", "SB2", "SB3", "UNR", "R32", "SW" };
            return buildFusedSource(a, a_body, b_full, b_io, b_ints);
        }
    }.f;
}

const FzBMode = enum { lane, one, shared_rows };

fn hdrFzStage(comptime a_pair1: bool, comptime b_pair1: bool) ?custom.SourceFn {
    if (!custom.is_codegen) return null;
    return struct {
        fn f(a: std.mem.Allocator, _: usize) ![]const u8 {
            const ah = try buildCodecHeader(a, .{ .is_pair1 = a_pair1, .mixed_rate = true, .secondary = false });
            const bh = try buildCodecHeader(a, .{ .is_pair1 = b_pair1, .mixed_rate = false, .secondary = true });
            return std.mem.concat(a, u8, &.{ ah, "\n", bh });
        }
    }.f;
}

fn srcFzStage(comptime is_down: bool, comptime group: u32, comptime b_mode: FzBMode, comptime a_pair1: bool, comptime b_pair1: bool) ?custom.SourceFn {
    if (!custom.is_codegen) return null;
    return struct {
        fn f(a: std.mem.Allocator, _: usize) ![]const u8 {
            const a_body = try buildMixedRateBody(a, if (is_down) .down_prepared else .pair_prepared, group, a_pair1);
            const raw_b: []const u8 = if (is_down) switch (b_mode) {
                .lane => kernels.DOWN_LANE_COOP_SOURCE,
                .one => kernels.ONE_ROW_COOP_SOURCE,
                .shared_rows => kernels.SHARED_ROWS_COOP_SOURCE,
            } else switch (b_mode) {
                .lane => kernels.LANE_PAIR_SOURCE,
                .one => kernels.ONE_ROW_PAIR_SOURCE,
                .shared_rows => kernels.SHARED_ROWS_PAIR_SOURCE,
            };
            var b_clean = try fcCleanSource(a, raw_b);
            if (b_mode == .shared_rows) {
                b_clean = try replaceOnceAlloc(a, b_clean, "constexpr uint row_count = uint(ROWS);", "const uint row_count = uint(ROWS);");
            }
            const n_decl = if (b_mode == .lane)
                "constexpr uint N = uint(NHW);\nconstexpr uint PACKED_HW = N;\nconstexpr uint PACKED_W = N / 2u;\n"
            else
                "constexpr uint N = uint(NHW);\nconstexpr uint PACKED_W = N / 2u;\n";
            const b_switched = try wrapFcRateSwitch(a, b_clean, "NHW", b_pair1, n_decl);
            const b_io: []const []const u8 = if (is_down)
                &.{ "x", "trellis", "slots", "y" }
            else
                &.{ "xg", "xu", "tg", "tu", "slots", "yg", "yu" };
            const b_ints: []const []const u8 = &.{ "IDIM", "ODIM", "NHW", "ROWS", "SHARED_ROWS" };
            return buildFusedSource(a, a_body, b_switched, b_io, b_ints);
        }
    }.f;
}

fn hdrFoldAffine(comptime mixed: bool, comptime a_pair1: bool) ?custom.SourceFn {
    if (!custom.is_codegen) return null;
    return struct {
        fn f(a: std.mem.Allocator, _: usize) ![]const u8 {
            const ah = try buildCodecHeader(a, .{ .is_pair1 = a_pair1, .mixed_rate = mixed, .secondary = false });
            const bh = try buildRidersHeaderFc(a);
            return std.mem.concat(a, u8, &.{ ah, "\n", bh });
        }
    }.f;
}

fn srcFoldAffine(comptime mixed: bool, comptime is_down: bool, comptime with_logit: bool, comptime a_pair1: bool) ?custom.SourceFn {
    if (!custom.is_codegen) return null;
    return struct {
        fn f(a: std.mem.Allocator, _: usize) ![]const u8 {
            const a_body: []const u8 = if (mixed)
                try buildMixedRateBody(a, if (is_down) .down else .pair, 0, a_pair1)
            else blk: {
                const raw = try fcCleanSource(a, if (is_down) kernels.DOWN_FUSED_SOURCE else kernels.PAIR_GEMV_SOURCE);
                break :blk try wrapFcRateSwitch(a, raw, "NHW", a_pair1, "constexpr uint N = uint(NHW);\nconstexpr uint PACKED_HW = N;\nconstexpr uint PACKED_W = N / 2u;\n");
            };
            const raw_b = if (is_down) riders_gated.DOWN else if (with_logit) riders_gated.GATE_UP_LOGIT else riders_gated.GATE_UP;
            const b_body = try buildRidersBodyFc(a, raw_b);
            const b_io: []const []const u8 = if (is_down)
                &.{ "x", "w", "sc", "bi", "y" }
            else if (with_logit)
                &.{ "x", "wg", "sg", "bg", "wu", "su", "bu", "sigmoid", "lw", "ls", "lb", "y", "gl" }
            else
                &.{ "x", "wg", "sg", "bg", "wu", "su", "bu", "sigmoid", "y" };
            const b_ints: []const []const u8 = if (is_down)
                &.{ "K", "N", "GS", "FAST" }
            else
                &.{ "K", "N", "GS", "FAST", "LGS" };
            return buildFusedSource(a, a_body, b_body, b_io, b_ints);
        }
    }.f;
}

fn hdrFoldTrellis(comptime mixed: bool, comptime is_down: bool, comptime with_logit: bool, comptime a_pair1: bool, comptime b_pair1: bool) ?custom.SourceFn {
    if (!custom.is_codegen) return null;
    return struct {
        fn f(a: std.mem.Allocator, _: usize) ![]const u8 {
            const ah = try buildCodecHeader(a, .{ .is_pair1 = a_pair1, .mixed_rate = mixed, .secondary = false });
            const bh_codec = try buildCodecHeader(a, .{ .is_pair1 = b_pair1, .mixed_rate = false, .secondary = true });
            const frags_fc = try buildRowsNaxFragsFc(a);
            const frags_sx = try renamePartB(a, frags_fc, &.{}, &.{}, false);
            var out: std.ArrayList(u8) = .empty;
            errdefer out.deinit(a);
            try out.appendSlice(a, kernels.GEMM_NAX_INCLUDES);
            try out.appendSlice(a, ah);
            try out.appendSlice(a, "\n");
            try out.appendSlice(a, bh_codec);
            try out.appendSlice(a, "\n");
            try out.appendSlice(a, frags_sx);
            if (is_down) {
                const mid_sx = try replaceOptAlloc(a, try renamePartB(a, shared_trellis.MID_HEADER, &.{}, &.{}, false), "bfloat16_t operator[](uint i) const {", "bfloat16_t operator[](uint i) const thread {");
                try out.appendSlice(a, "\n");
                try out.appendSlice(a, mid_sx);
            } else if (with_logit) {
                const rh = try buildRidersHeaderFc(a);
                try out.appendSlice(a, "\n");
                try out.appendSlice(a, rh);
            }
            return out.toOwnedSlice(a);
        }
    }.f;
}

fn srcFoldTrellis(comptime mixed: bool, comptime is_down: bool, comptime with_logit: bool, comptime a_pair1: bool, comptime b_pair1: bool) ?custom.SourceFn {
    if (!custom.is_codegen) return null;
    return struct {
        fn f(a: std.mem.Allocator, _: usize) ![]const u8 {
            const a_body: []const u8 = if (mixed)
                try buildMixedRateBody(a, if (is_down) .down else .pair, 0, a_pair1)
            else blk: {
                const raw = try fcCleanSource(a, if (is_down) kernels.DOWN_FUSED_SOURCE else kernels.PAIR_GEMV_SOURCE);
                break :blk try wrapFcRateSwitch(a, raw, "NHW", a_pair1, "constexpr uint N = uint(NHW);\nconstexpr uint PACKED_HW = N;\nconstexpr uint PACKED_W = N / 2u;\n");
            };
            // Part B (`shared_trellis`): always KPH=1, UNR=2, TGRP=4, FOLD=0, GROUPS=1 (`shared_trellis.zig:50`).
            const raw_tr = if (is_down) shared_trellis.DOWN_SOURCE else project_rows.ROWS_NAX_SOURCE;
            var tr_body = try transformRowsSource(a, raw_tr, true, false);
            tr_body = try replaceAllAlloc(a, tr_body, "uint(KPH)", "1u");
            tr_body = try replaceAllAlloc(a, tr_body, "KPH", "1");
            tr_body = try replaceAllAlloc(a, tr_body, "uint(UNR)", "2u");
            tr_body = try replaceAllAlloc(a, tr_body, "UNR", "2");
            tr_body = try replaceOnceAlloc(a, tr_body, "uint(ROW_CAP)", "16u");
            const tr_switched = try wrapFcRateSwitch(a, tr_body, "NHW", b_pair1, "constexpr uint N = uint(NHW);\nconstexpr uint PACKED_W = N / 2u;\n");

            const b_full: []const u8 = if (!is_down and with_logit) blk: {
                const logit_fc = try buildRidersBodyFc(a, riders_gated.LOGIT_ROW);
                break :blk try std.mem.concat(a, u8, &.{ logit_fc, "{\n", shared_trellis.OUT_TYPE, tr_switched, "\n}\n" });
            } else try std.mem.concat(a, u8, &.{ "{\n", shared_trellis.OUT_TYPE, tr_switched, "\n}\n" });

            const b_io: []const []const u8 = if (is_down)
                &.{ "x", "trellis", "suh", "svh", "g1", "g1svh", "sigmoid", "y" }
            else if (with_logit)
                &.{ "x", "trellis", "suh", "svh", "lw", "ls", "lb", "y", "gl" }
            else
                &.{ "x", "trellis", "suh", "svh", "y" };
            const b_ints: []const []const u8 = if (is_down)
                &.{ "IDIM", "ODIM", "ROWS", "GROUPS", "TGRP", "KPH", "NSPLIT", "CHUNK", "FOLD", "NHW", "SB1", "SB2", "SB3", "UNR", "R32", "SW", "GNSPLIT", "GODIM", "GWIDTH" }
            else if (with_logit)
                &.{ "IDIM", "ODIM", "ROWS", "GROUPS", "TGRP", "KPH", "NSPLIT", "CHUNK", "FOLD", "NHW", "SB1", "SB2", "SB3", "UNR", "R32", "SW", "K", "LGS", "TILES" }
            else
                &.{ "IDIM", "ODIM", "ROWS", "GROUPS", "TGRP", "KPH", "NSPLIT", "CHUNK", "FOLD", "NHW", "SB1", "SB2", "SB3", "UNR", "R32", "SW" };
            return buildFusedSource(a, a_body, b_full, b_io, b_ints);
        }
    }.f;
}

// ── Unit 1: `exl3_simd` Spec Tables ──

fn combineInputSlots(comptime chained: bool) [rate_groups.COMBINE_PLANES + (if (chained) 3 else 2)]InputSlot {
    var out: [rate_groups.COMBINE_PLANES + (if (chained) 3 else 2)]InputSlot = undefined;
    out[0] = inSlot("eids", fix(.uint32));
    out[1] = inSlot("meta", fix(.uint32));
    comptime var k: usize = 2;
    if (chained) {
        out[k] = inSlot("prev", fix(.float16));
        k += 1;
    }
    inline for (0..rate_groups.COMBINE_PLANES) |i| {
        out[k + i] = inSlot(std.fmt.comptimePrint("p{d}", .{i}), fix(.float16));
    }
    return out;
}

const combine_ins_plain = combineInputSlots(false);
const combine_ins_chained = combineInputSlots(true);

fn prefillMidInputSlots() [rate_groups.MID_PLANES_INS.len + 5 + 1]InputSlot {
    var out: [rate_groups.MID_PLANES_INS.len + 5 + 1]InputSlot = undefined;
    var k: usize = 0;
    inline for (0..rate_groups.DIRECT_PLANES) |i| {
        out[k] = inSlot(std.fmt.comptimePrint("ig{d}", .{i}), fix(.float16));
        k += 1;
    }
    inline for (0..rate_groups.DIRECT_PLANES) |i| {
        out[k] = inSlot(std.fmt.comptimePrint("iu{d}", .{i}), fix(.float16));
        k += 1;
    }
    out[k] = inSlot("svhg", fix(.float16));
    out[k + 1] = inSlot("svhu", fix(.float16));
    out[k + 2] = inSlot("suhd", fix(.float16));
    out[k + 3] = inSlot("slots", fix(.uint32));
    out[k + 4] = inSlot("meta", fix(.uint32));
    out[k + 5] = inSlot("s_ig", fix(.float16));
    out[k + 6] = inSlot("s_iu", fix(.float16));
    out[k + 7] = inSlot("s_svhg", fix(.float16));
    out[k + 8] = inSlot("s_svhu", fix(.float16));
    out[k + 9] = inSlot("s_suhd", fix(.float16));
    out[k + 10] = inSlot("fz_grid", fix(.uint32));
    return out;
}

const prefill_mid_ins = prefillMidInputSlots();

fn prefillReduceInputSlots() [rate_groups.REDUCE_PLANES_INS.len]InputSlot {
    var out: [rate_groups.REDUCE_PLANES_INS.len]InputSlot = undefined;
    var k: usize = 0;
    inline for (0..rate_groups.DIRECT_PLANES) |i| {
        out[k] = inSlot(std.fmt.comptimePrint("inner{d}", .{i}), fix(.float16));
        k += 1;
    }
    out[k] = inSlot("svh", fix(.float16));
    out[k + 1] = inSlot("slots", fix(.uint32));
    out[k + 2] = inSlot("sc", fix(.float32));
    out[k + 3] = inSlot("s_inner", fix(.float16));
    out[k + 4] = inSlot("s_svh", fix(.float16));
    out[k + 5] = inSlot("unit", fix(.float32));
    out[k + 6] = inSlot("inverse", fix(.uint32));
    out[k + 7] = inSlot("meta", fix(.uint32));
    return out;
}

const prefill_reduce_ins = prefillReduceInputSlots();

const static_simd_specs = [_]KernelSpec{
    // 1. Prepare / Finish H128
    .{
        .name = "sushi_exl3_prepare_h128",
        .source = kernels.PREPARE_SOURCE,
        .inputs = &.{ inSlot("x", par("X")), inSlot("suh", fix(.float16)), idxSlot("slots") },
        .outputs = &.{outSlot("y", fix(.float16))},
        .params = &.{ .{ .name = "X", .kind = .io_dtype, .dtypes = &floats }, fcInt("IDIM") },
    },
    .{
        .name = "sushi_exl3_finish_h128",
        .source = kernels.FINISH_SOURCE,
        .inputs = &.{ inSlot("inner", fix(.float16)), inSlot("svh", fix(.float16)), idxSlot("slots") },
        .outputs = &.{outSlot("y", par("OutT"))},
        .params = &.{ .{ .name = "OutT", .kind = .ct_dtype, .dtypes = &floats }, fcInt("ODIM") },
    },
    // 2. SwiGLU mid + decode mid + rg_mid + down_lane_prepare
    .{
        .name = "sushi_exl3_mid_swiglu",
        .source_fn = srcClean(kernels.MID_SOURCE),
        .inputs = &.{ inSlot("ig", fix(.float16)), inSlot("iu", fix(.float16)), inSlot("svhg", fix(.float16)), inSlot("svhu", fix(.float16)), inSlot("suhd", fix(.float16)), idxSlot("slots") },
        .outputs = &.{outSlot("yd", fix(.float16))},
        .params = &.{ fcInt("ODIM"), fcInt("CLAMP_LIMIT") },
    },
    .{
        .name = "sushi_exl3_decode_mid",
        .source_fn = srcClean(kernels.DECODE_MID_SOURCE),
        .inputs = &.{ inSlot("ig", fix(.float32)), inSlot("iu", fix(.float32)), inSlot("svhg", fix(.float16)), inSlot("svhu", fix(.float16)), inSlot("suhd", fix(.float16)), idxSlot("slots") },
        .outputs = &.{outSlot("prepared", fix(.float16))},
        .params = &.{ fcInt("IDIM"), fcInt("NSPLIT") },
    },
    .{
        .name = "sushi_exl3_rg_mid",
        .source_fn = custom.codegenFn(srcRgMid),
        .inputs = &.{ inSlot("ig", fix(.float32)), inSlot("iu", fix(.float32)), inSlot("svhg", fix(.float16)), inSlot("svhu", fix(.float16)), inSlot("suhd", fix(.float16)), inSlot("slots", fix(.uint32)) },
        .outputs = &.{outSlot("prepared", fix(.float16))},
        .params = &.{ fcInt("IDIM"), fcInt("NSPLIT") },
    },
    .{
        .name = "sushi_down_lane_prepare",
        .source_fn = srcClean(kernels.DOWN_LANE_MID_SOURCE),
        .inputs = &.{ inSlot("ig", fix(.float16)), inSlot("iu", fix(.float16)), inSlot("svhg", fix(.float16)), inSlot("svhu", fix(.float16)), inSlot("suhd", fix(.float16)), idxSlot("slots") },
        .outputs = &.{outSlot("yd", fix(.float16))},
        .params = &.{ fcInt("ODIM"), fcInt("CLAMP_LIMIT") },
    },
    // 3. Down reduces (4 variants + sorted_inverse + shared_trellis + 3 fused + prefill_planes)
    .{
        .name = "sushi_exl3_down_reduce",
        .source_fn = srcClean(kernels.REDUCE_SOURCE),
        .inputs = &.{ inSlot("inner", fix(.float16)), inSlot("svh", fix(.float16)), idxSlot("slots"), inSlot("sc", par("SC")) },
        .outputs = &.{outSlot("y", par("T"))},
        .params = &.{ .{ .name = "T", .kind = .ct_dtype, .dtypes = &floats }, .{ .name = "SC", .kind = .io_dtype, .dtypes = &floats }, fcInt("ODIM"), fcInt("TOPK") },
    },
    .{
        .name = "sushi_exl3_down_reduce_shared",
        .source_fn = srcClean(kernels.REDUCE_SHARED_SOURCE),
        .inputs = &.{ inSlot("inner", fix(.float16)), inSlot("svh", fix(.float16)), idxSlot("slots"), inSlot("sc", par("SC")), inSlot("shared", par("SH")) },
        .outputs = &.{outSlot("y", par("T"))},
        .params = &.{ .{ .name = "T", .kind = .ct_dtype, .dtypes = &floats }, .{ .name = "SH", .kind = .io_dtype, .dtypes = &floats }, .{ .name = "SC", .kind = .io_dtype, .dtypes = &floats }, fcInt("ODIM"), fcInt("TOPK") },
        .filter = sameOrBfFilter,
    },
    .{
        .name = "sushi_exl3_down_reduce_shared_lead",
        .source_fn = srcClean(kernels.REDUCE_SHARED_SOURCE),
        .inputs = &.{ inSlot("shared", par("SH")), inSlot("inner", fix(.float16)), inSlot("svh", fix(.float16)), idxSlot("slots"), inSlot("sc", par("SC")) },
        .outputs = &.{outSlot("y", par("T"))},
        .params = &.{ .{ .name = "T", .kind = .ct_dtype, .dtypes = &floats }, .{ .name = "SH", .kind = .io_dtype, .dtypes = &floats }, .{ .name = "SC", .kind = .io_dtype, .dtypes = &floats }, fcInt("ODIM"), fcInt("TOPK") },
        .filter = sameOrBfFilter,
    },
    .{
        .name = "sushi_exl3_down_reduce_gated_shared",
        .source_fn = srcClean(kernels.REDUCE_GATED_SHARED_SOURCE),
        .inputs = &.{ inSlot("inner", fix(.float16)), inSlot("svh", fix(.float16)), idxSlot("slots"), inSlot("sc", par("SC")), inSlot("shared", par("T")), inSlot("gate", par("T")) },
        .outputs = &.{outSlot("y", par("T"))},
        .params = &.{ .{ .name = "T", .kind = .ct_dtype, .dtypes = &floats }, .{ .name = "SC", .kind = .io_dtype, .dtypes = &floats }, fcInt("ODIM"), fcInt("TOPK") },
    },
    .{
        .name = "sushi_exl3_sorted_inverse_reduce",
        .source_fn = srcClean(sorted_windows.SORTED_INVERSE_SOURCE),
        .inputs = &.{ inSlot("inner", fix(.float16)), inSlot("inverse", fix(.uint32)), inSlot("svh", fix(.float16)), idxSlot("slots"), inSlot("sc", par("SC")) },
        .outputs = &.{outSlot("y", par("T"))},
        .params = &.{ .{ .name = "T", .kind = .ct_dtype, .dtypes = &floats }, .{ .name = "SC", .kind = .io_dtype, .dtypes = &floats }, fcInt("ODIM"), fcInt("TOPK") },
    },
    .{
        .name = "sushi_exl3_down_reduce_shared_trellis",
        .header = project_rows.ROWS_HELPERS,
        .source_fn = srcClean(shared_trellis.REDUCE_SOURCE),
        .inputs = &.{ inSlot("inner", fix(.float16)), inSlot("svh", fix(.float16)), inSlot("slots", fix(.uint32)), inSlot("sc", par("SC")), inSlot("spart", fix(.float32)), inSlot("ssvh", fix(.float16)), inSlot("gate", par("T")) },
        .outputs = &.{outSlot("y", par("T"))},
        .params = &.{ .{ .name = "T", .kind = .ct_dtype, .dtypes = &bf_f32 }, .{ .name = "SC", .kind = .io_dtype, .dtypes = &floats }, fcInt("ODIM"), fcInt("TOPK"), fcInt("SNSPLIT") },
    },
    .{
        .name = "sushi_exl3_down_reduce_fused_shared",
        .source_fn = srcClean(rate_groups.REDUCE_FUSED_SOURCE),
        .inputs = &.{ inSlot("inner", fix(.float16)), inSlot("svh", fix(.float16)), inSlot("slots", fix(.uint32)), inSlot("sc", fix(.float32)), inSlot("s_inner", par("SI")), inSlot("s_svh", fix(.float16)), inSlot("unit", fix(.float32)) },
        .outputs = &.{outSlot("y", par("T"))},
        .params = &.{ .{ .name = "T", .kind = .ct_dtype, .dtypes = &floats }, .{ .name = "ST", .kind = .ct_dtype, .dtypes = &.{.bfloat16} }, .{ .name = "SI", .kind = .io_dtype, .dtypes = &.{ .float16, .float32 } }, fcInt("ODIM"), fcInt("TOPK"), fcInt("S_DOWN_NSPLIT") },
    },
    .{
        .name = "sushi_exl3_down_reduce_fused_shared_redo",
        .source_fn = srcClean(rate_groups.REDUCE_FUSED_REDO_SOURCE),
        .inputs = &.{ inSlot("inner", fix(.float16)), inSlot("part", fix(.float16)), inSlot("redo", fix(.int32)), inSlot("svh", fix(.float16)), inSlot("slots", fix(.uint32)), inSlot("sc", fix(.float32)), inSlot("s_inner", par("SI")), inSlot("s_svh", fix(.float16)), inSlot("unit", fix(.float32)) },
        .outputs = &.{outSlot("y", par("T"))},
        .params = &.{ .{ .name = "T", .kind = .ct_dtype, .dtypes = &floats }, .{ .name = "ST", .kind = .ct_dtype, .dtypes = &.{.bfloat16} }, .{ .name = "SI", .kind = .io_dtype, .dtypes = &.{ .float16, .float32 } }, fcInt("ODIM"), fcInt("TOPK"), fcInt("S_DOWN_NSPLIT") },
    },
    .{
        .name = "sushi_exl3_down_reduce_fused_expand",
        .source_fn = srcClean(rate_groups.REDUCE_EXPAND_SOURCE),
        .inputs = &.{ inSlot("inner", fix(.float16)), inSlot("svh", fix(.float16)), inSlot("slots", fix(.uint32)), inSlot("sc", fix(.float32)), inSlot("s_inner", par("SI")), inSlot("s_svh", fix(.float16)), inSlot("unit", fix(.float32)), inSlot("x1", par("XT")), inSlot("post", fix(.float32)), inSlot("comb", fix(.float32)) },
        .outputs = &.{outSlot("y", par("XT"))},
        .params = &.{ .{ .name = "T", .kind = .ct_dtype, .dtypes = &bf_f32 }, .{ .name = "ST", .kind = .ct_dtype, .dtypes = &.{.bfloat16} }, .{ .name = "XT", .kind = .ct_dtype, .dtypes = &bf_f32 }, .{ .name = "SI", .kind = .io_dtype, .dtypes = &.{ .float16, .float32 } }, fcInt("ODIM"), fcInt("TOPK"), fcInt("S_DOWN_NSPLIT") },
    },
    .{
        .name = "sushi_exl3_prefill_reduce_fused_shared_planes",
        .source_fn = srcClean(rate_groups.PREFILL_REDUCE_PLANES_SOURCE),
        .inputs = &prefill_reduce_ins,
        .outputs = &.{outSlot("y", fix(.float32))},
        .params = &.{ .{ .name = "T", .kind = .ct_dtype, .dtypes = &.{.bfloat16} }, .{ .name = "ST", .kind = .ct_dtype, .dtypes = &.{.bfloat16} }, fcInt("ODIM"), fcInt("TOPK") },
    },
    // 4. Token / Pair / Lane prepare & scatter
    .{
        .name = "sushi_exl3_token_pair_prepare",
        .source = kernels.TOKEN_PAIR_PREPARE_SOURCE,
        .inputs = &.{ inSlot("x", par("X")), inSlot("suhg", fix(.float16)), inSlot("suhu", fix(.float16)), idxSlot("slots"), idxSlot("order") },
        .outputs = &.{ outSlot("yg", fix(.float16)), outSlot("yu", fix(.float16)) },
        .params = &.{ .{ .name = "X", .kind = .io_dtype, .dtypes = &floats }, fcInt("IDIM"), fcInt("TOPK") },
    },
    .{
        .name = "sushi_exl3_token_pair_unsorted",
        .source_fn = custom.codegenFn(srcTokenPairUnsorted),
        .inputs = &.{ inSlot("x", par("X")), inSlot("suhg", fix(.float16)), inSlot("suhu", fix(.float16)), idxSlot("slots") },
        .outputs = &.{ outSlot("yg", fix(.float16)), outSlot("yu", fix(.float16)) },
        .params = &.{ .{ .name = "X", .kind = .io_dtype, .dtypes = &floats }, fcInt("IDIM"), fcInt("TOPK") },
    },
    .{
        .name = "sushi_exl3_token_scatter",
        .source = kernels.TOKEN_SCATTER_SOURCE,
        .inputs = &.{ inSlot("x", par("X")), idxSlot("order") },
        .outputs = &.{outSlot("y", par("X"))},
        .params = &.{ .{ .name = "X", .kind = .io_dtype, .dtypes = &floats }, fcInt("DIM") },
    },
    .{
        .name = "sushi_exl3_pair_prepare",
        .source = kernels.PAIR_PREP_SOURCE,
        .inputs = &.{ inSlot("x", par("X")), inSlot("suhg", fix(.float16)), inSlot("suhu", fix(.float16)), idxSlot("slots") },
        .outputs = &.{outSlot("prep", fix(.float16))},
        .params = &.{ .{ .name = "X", .kind = .io_dtype, .dtypes = &floats }, fcInt("IDIM"), fcInt("TOPK") },
    },
    .{
        .name = "sushi_lane_prepare",
        .source = kernels.LANE_PREP_SOURCE,
        .inputs = &.{ inSlot("x", par("X")), inSlot("suhg", fix(.float16)), inSlot("suhu", fix(.float16)), idxSlot("slots"), idxSlot("order") },
        .outputs = &.{ outSlot("yg", fix(.float16)), outSlot("yu", fix(.float16)) },
        .params = &.{ .{ .name = "X", .kind = .io_dtype, .dtypes = &floats }, fcInt("IDIM"), fcInt("TOPK") },
    },
    .{
        .name = "sushi_lane_unsorted",
        .source = kernels.LANE_PREP_UNSORTED_SOURCE,
        .inputs = &.{ inSlot("x", par("X")), inSlot("suhg", fix(.float16)), inSlot("suhu", fix(.float16)), idxSlot("slots") },
        .outputs = &.{ outSlot("yg", fix(.float16)), outSlot("yu", fix(.float16)) },
        .params = &.{ .{ .name = "X", .kind = .io_dtype, .dtypes = &floats }, fcInt("IDIM"), fcInt("TOPK") },
    },
    // 5. Window tables & combine & fused prefill stages
    .{
        .name = "sushi_exl3_gpu_windows",
        .source_fn = srcClean(sorted_windows.GPU_WINDOW_SOURCE),
        .inputs = &.{ inSlot("eids", fix(.uint32)), idxSlot("order"), inSlot("count", fix(.uint32)) },
        .outputs = &.{ outSlot("starts", fix(.uint32)), outSlot("nlives", fix(.uint32)), outSlot("inverse", fix(.uint32)) },
        .params = &.{ fcInt("WIN"), fcInt("EXPERTS") },
    },
    .{
        .name = "sushi_exl3_rg_windows",
        .source = rate_groups.GROUP_WINDOW_SOURCE,
        .inputs = &.{ inSlot("eids", fix(.uint32)), idxSlot("order"), inSlot("count", fix(.uint32)), inSlot("meta", fix(.uint32)) },
        .outputs = &.{ outSlot("starts", fix(.uint32)), outSlot("nlives", fix(.uint32)), outSlot("inverse", fix(.uint32)), outSlot("local", fix(.uint32)) },
        .params = &.{ fcInt("WIN"), .{ .name = "MAX_EXPERTS", .kind = .ct_int, .ints = &.{512} } },
    },
    .{
        .name = "sushi_exl3_rg_combine",
        .source_fn = custom.codegenFn(srcCombine),
        .source_arg = 0,
        .inputs = &combine_ins_plain,
        .outputs = &.{outSlot("y", fix(.float16))},
        .params = &.{ fcInt("DIM"), fcInt("FIRST") },
    },
    .{
        .name = "sushi_exl3_rg_combine_chained",
        .source_fn = custom.codegenFn(srcCombine),
        .source_arg = 1,
        .inputs = &combine_ins_chained,
        .outputs = &.{outSlot("y", fix(.float16))},
        .params = &.{ fcInt("DIM"), fcInt("FIRST") },
    },
    .{
        .name = "sushi_exl3_fz_prefill_prep",
        .source_fn = custom.codegenFn(srcFzPrefillPrep),
        .inputs = &.{
            inSlot("x", fix(.bfloat16)), inSlot("suhg", fix(.float16)), inSlot("suhu", fix(.float16)), inSlot("slots", fix(.uint32)), inSlot("order", fix(.int32)),
            inSlot("s_x", fix(.bfloat16)), inSlot("s_suhg", fix(.float16)), inSlot("s_suhu", fix(.float16)), inSlot("s_slots", fix(.uint32)),
            inSlot("fz_grid", fix(.uint32)),
        },
        .outputs = &.{ outSlot("yg", fix(.float16)), outSlot("yu", fix(.float16)), outSlot("s_yg", fix(.float16)), outSlot("s_yu", fix(.float16)) },
        .params = &.{ fcInt("IDIM"), fcInt("TOPK"), fcInt("S_IDIM"), fcInt("S_TOPK") },
    },
    .{
        .name = "sushi_exl3_fz_prefill_mid",
        .source_fn = custom.codegenFn(srcFzPrefillMid),
        .inputs = &prefill_mid_ins,
        .outputs = &.{ outSlot("yd", fix(.float16)), outSlot("s_yd", fix(.float16)) },
        .params = &.{ fcInt("ODIM"), fcInt("CLAMP_LIMIT"), fcInt("S_ODIM"), fcInt("S_CLAMP_LIMIT") },
    },
};

fn sameOrBfFilter(vals: []const CtVal) bool {
    const t = vals[0].dtype;
    const sh = vals[1].dtype;
    return t == sh or sh == .bfloat16;
}

fn pairCodecSpecs(
    comptime base: []const u8,
    comptime nax: bool,
    comptime h_scalar: usize,
    comptime h_pair1: usize,
    comptime sfn: ?custom.SourceFn,
    comptime ins: []const InputSlot,
    comptime outs: []const OutputSlot,
    comptime params_scalar: []const ParamSpec,
    comptime params_pair1: []const ParamSpec,
) [2]KernelSpec {
    return .{
        .{ .name = base ++ "_scalar", .nax = nax, .header_fn = custom.codegenFn(exl3HeaderFn), .source_arg = h_scalar, .source_fn = sfn, .inputs = ins, .outputs = outs, .params = params_scalar },
        .{ .name = base ++ "_pair1", .nax = nax, .header_fn = custom.codegenFn(exl3HeaderFn), .source_arg = h_pair1, .source_fn = sfn, .inputs = ins, .outputs = outs, .params = params_pair1 },
    };
}

fn nhwParams(comptime rates: []const i32, comptime rest: []const ParamSpec) [rest.len + 1]ParamSpec {
    var out: [rest.len + 1]ParamSpec = undefined;
    out[0] = .{ .name = "NHW", .kind = .ct_int, .ints = rates };
    for (rest, 0..) |p, i| out[i + 1] = p;
    return out;
}

const gemm_ins = [_]InputSlot{
    inSlot("x", fix(.float16)),
    inSlot("trellis", fix(.uint16)),
    inSlot("eids", fix(.uint32)),
    inSlot("wstarts", fix(.uint32)),
    inSlot("wnlive", fix(.uint32)),
};
const gemm_outs = [_]OutputSlot{outSlot("y", fix(.float16))};

const gemm_sorted_p_sc = nhwParams(&served_scalar_rates, &.{ fcInt("IDIM"), fcInt("ODIM"), fcInt("WIN") });
const gemm_sorted_p_p1 = nhwParams(&pair1_rates, &.{ fcInt("IDIM"), fcInt("ODIM"), fcInt("WIN") });

const gemm_smat_p_sc = nhwParams(&served_scalar_rates, &.{ .{ .name = "WIN", .kind = .ct_int, .ints = &.{ 16, 32 } }, fcInt("IDIM"), fcInt("ODIM") });
const gemm_smat_p_p1 = nhwParams(&pair1_rates, &.{ .{ .name = "WIN", .kind = .ct_int, .ints = &.{ 16, 32 } }, fcInt("IDIM"), fcInt("ODIM") });

const gemv_idx_p_sc = nhwParams(&served_scalar_rates, &.{ fcInt("IDIM"), fcInt("ODIM") });
const gemv_idx_p_p1 = nhwParams(&pair1_rates, &.{ fcInt("IDIM"), fcInt("ODIM") });

const down_fused_p_sc = nhwParams(&served_scalar_rates, &.{ .{ .name = "OTPT", .kind = .ct_int, .ints = &.{ 1, 2 } }, fcInt("IDIM"), fcInt("ODIM"), fcInt("NSPLIT"), fcInt("FUNNEL") });
const down_fused_p_p1 = nhwParams(&pair1_rates, &.{ .{ .name = "OTPT", .kind = .ct_int, .ints = &.{ 1, 2 } }, fcInt("IDIM"), fcInt("ODIM"), fcInt("NSPLIT"), fcInt("FUNNEL") });

const pair_gemv_p_sc = nhwParams(&served_scalar_rates, &.{ .{ .name = "X", .kind = .io_dtype, .dtypes = &floats }, .{ .name = "OTPT", .kind = .ct_int, .ints = &.{ 1, 2 } }, fcInt("IDIM"), fcInt("ODIM"), fcInt("NSPLIT"), fcInt("TOPK"), fcInt("FUNNEL") });
const pair_gemv_p_p1 = nhwParams(&pair1_rates, &.{ .{ .name = "X", .kind = .io_dtype, .dtypes = &floats }, .{ .name = "OTPT", .kind = .ct_int, .ints = &.{ 1, 2 } }, fcInt("IDIM"), fcInt("ODIM"), fcInt("NSPLIT"), fcInt("TOPK"), fcInt("FUNNEL") });

const pair_prep_gemv_p_sc = nhwParams(&served_scalar_rates, &.{ .{ .name = "OTPT", .kind = .ct_int, .ints = &.{ 1, 2 } }, fcInt("IDIM"), fcInt("ODIM"), fcInt("NSPLIT"), fcInt("TOPK"), fcInt("FUNNEL") });
const pair_prep_gemv_p_p1 = nhwParams(&pair1_rates, &.{ .{ .name = "OTPT", .kind = .ct_int, .ints = &.{ 1, 2 } }, fcInt("IDIM"), fcInt("ODIM"), fcInt("NSPLIT"), fcInt("TOPK"), fcInt("FUNNEL") });

const down_prep_p_sc = nhwParams(&served_scalar_rates, &.{ .{ .name = "OTPT", .kind = .ct_int, .ints = &.{ 1, 2 } }, fcInt("IDIM"), fcInt("ODIM"), fcInt("FUNNEL") });
const down_prep_p_p1 = nhwParams(&pair1_rates, &.{ .{ .name = "OTPT", .kind = .ct_int, .ints = &.{ 1, 2 } }, fcInt("IDIM"), fcInt("ODIM"), fcInt("FUNNEL") });

fn pairGroupedParams(comptime g: i32, comptime rates: []const i32, comptime otpts: []const i32, comptime prepared: bool) [if (prepared) 9 else 10]ParamSpec {
    if (prepared) {
        return .{
            .{ .name = "NHW", .kind = .ct_int, .ints = rates },
            .{ .name = "GROUP", .kind = .ct_int, .ints = &.{g} },
            .{ .name = "OTPT", .kind = .ct_int, .ints = otpts },
            fcInt("IDIM"),
            fcInt("ODIM"),
            fcInt("NSPLIT"),
            fcInt("TOPK"),
            fcInt("FUNNEL"),
            fcInt("NSLOTS"),
        };
    }
    return .{
        .{ .name = "NHW", .kind = .ct_int, .ints = rates },
        .{ .name = "X", .kind = .io_dtype, .dtypes = &bf_f16 },
        .{ .name = "GROUP", .kind = .ct_int, .ints = &.{g} },
        .{ .name = "OTPT", .kind = .ct_int, .ints = otpts },
        fcInt("IDIM"),
        fcInt("ODIM"),
        fcInt("NSPLIT"),
        fcInt("TOPK"),
        fcInt("FUNNEL"),
        fcInt("NSLOTS"),
    };
}

fn downGroupedParams(comptime g: i32, comptime rates: []const i32, comptime otpts: []const i32) [7]ParamSpec {
    return .{
        .{ .name = "NHW", .kind = .ct_int, .ints = rates },
        .{ .name = "GROUP", .kind = .ct_int, .ints = &.{g} },
        .{ .name = "OTPT", .kind = .ct_int, .ints = otpts },
        fcInt("IDIM"),
        fcInt("ODIM"),
        fcInt("FUNNEL"),
        fcInt("NSLOTS"),
    };
}

const pg_g2_sc = pairGroupedParams(2, &served_scalar_rates, &.{ 1, 2 }, false);
const pg_g2_p1 = pairGroupedParams(2, &pair1_rates, &.{ 1, 2 }, false);
const ppg_g2_sc = pairGroupedParams(2, &served_scalar_rates, &.{ 1, 2 }, true);
const ppg_g2_p1 = pairGroupedParams(2, &pair1_rates, &.{ 1, 2 }, true);
const dpg_g2_sc = downGroupedParams(2, &served_scalar_rates, &.{ 1, 2 });
const dpg_g2_p1 = downGroupedParams(2, &pair1_rates, &.{ 1, 2 });

const clamped_md_p_sc = nhwParams(&served_scalar_rates, &.{ .{ .name = "OTPT", .kind = .ct_int, .ints = &.{ 1, 4, 8 } }, fcInt("IDIM"), fcInt("ODIM"), fcInt("CLAMP_LIMIT") });
const clamped_md_p_p1 = nhwParams(&pair1_rates, &.{ .{ .name = "OTPT", .kind = .ct_int, .ints = &.{ 1, 4, 8 } }, fcInt("IDIM"), fcInt("ODIM"), fcInt("CLAMP_LIMIT") });

const coop_t1_p_sc = nhwParams(&served_scalar_rates, &.{ .{ .name = "TILES", .kind = .ct_int, .ints = &.{1} }, fcInt("IDIM"), fcInt("ODIM"), fcInt("NSLOTS"), fcInt("SERIAL_REDUCTION") });
const coop_t1_p_p1 = nhwParams(&pair1_rates, &.{ .{ .name = "TILES", .kind = .ct_int, .ints = &.{1} }, fcInt("IDIM"), fcInt("ODIM"), fcInt("NSLOTS"), fcInt("SERIAL_REDUCTION") });
const coop_t2_p_sc = nhwParams(&served_scalar_rates, &.{ .{ .name = "TILES", .kind = .ct_int, .ints = &.{2} }, fcInt("IDIM"), fcInt("ODIM"), fcInt("NSLOTS"), fcInt("SERIAL_REDUCTION") });
const coop_t2_p_p1 = nhwParams(&pair1_rates, &.{ .{ .name = "TILES", .kind = .ct_int, .ints = &.{2} }, fcInt("IDIM"), fcInt("ODIM"), fcInt("NSLOTS"), fcInt("SERIAL_REDUCTION") });

const single_simd_specs =
    pairCodecSpecs("sushi_exl3_k4_gemm_sorted", false, hdr_scalar, hdr_pair1, srcClean(kernels.GEMM_SORTED_SOURCE), &gemm_ins, &gemm_outs, &gemm_sorted_p_sc, &gemm_sorted_p_p1) ++
    pairCodecSpecs("sushi_exl3_k4_gemm_sorted_tm", false, hdr_scalar, hdr_pair1, srcClean(kernels.GEMM_SORTED_TM_SOURCE), &gemm_ins, &gemm_outs, &gemm_sorted_p_sc, &gemm_sorted_p_p1) ++
    pairCodecSpecs("sushi_exl3_k4_gemm_simdmat", false, hdr_scalar_smat, hdr_pair1_smat, srcClean(kernels.GEMM_SIMDMAT_SOURCE), &gemm_ins, &gemm_outs, &gemm_smat_p_sc, &gemm_smat_p_p1) ++
    pairCodecSpecs("sushi_exl3_k4_gemm_simdmat_tm", false, hdr_scalar_smat, hdr_pair1_smat, srcClean(kernels.GEMM_SIMDMAT_TM_SOURCE), &gemm_ins, &gemm_outs, &gemm_smat_p_sc, &gemm_smat_p_p1) ++
    pairCodecSpecs("sushi_exl3_k4_mul1_gemv_indexed", false, hdr_scalar, hdr_pair1, srcClean(kernels.INDEXED_COOP_SOURCE), &.{ inSlot("x", fix(.float16)), inSlot("trellis", fix(.uint16)), idxSlot("slots") }, &.{outSlot("y", fix(.float16))}, &gemv_idx_p_sc, &gemv_idx_p_p1) ++
    pairCodecSpecs("sushi_exl3_indexed_pair_coop", false, hdr_scalar, hdr_pair1, srcClean(kernels.INDEXED_PAIR_COOP_SOURCE), &.{ inSlot("xg", fix(.float16)), inSlot("xu", fix(.float16)), inSlot("tg", fix(.uint16)), inSlot("tu", fix(.uint16)), idxSlot("slots") }, &.{ outSlot("yg", fix(.float16)), outSlot("yu", fix(.float16)) }, &gemv_idx_p_sc, &gemv_idx_p_p1) ++
    pairCodecSpecs("sushi_exl3_k4_down_fused", false, hdr_scalar, hdr_pair1, srcClean(kernels.DOWN_FUSED_SOURCE), &.{ inSlot("ig", fix(.float32)), inSlot("iu", fix(.float32)), inSlot("trellis", fix(.uint16)), inSlot("svhg", fix(.float16)), inSlot("svhu", fix(.float16)), inSlot("suhd", fix(.float16)), inSlot("slots", fix(.uint32)) }, &.{outSlot("y", fix(.float16))}, &down_fused_p_sc, &down_fused_p_p1) ++
    pairCodecSpecs("sushi_exl3_pair_gemv", false, hdr_scalar, hdr_pair1, srcClean(kernels.PAIR_GEMV_SOURCE), &.{ inSlot("x", par("X")), inSlot("suhg", fix(.float16)), inSlot("suhu", fix(.float16)), inSlot("tg", fix(.uint16)), inSlot("tu", fix(.uint16)), inSlot("slots", fix(.uint32)) }, &.{ outSlot("yg", fix(.float32)), outSlot("yu", fix(.float32)) }, &pair_gemv_p_sc, &pair_gemv_p_p1) ++
    pairCodecSpecs("sushi_exl3_pair_gemv_prepared", false, hdr_scalar, hdr_pair1, srcClean(kernels.PAIR_GEMV_PREPARED_SOURCE), &.{ inSlot("prep", fix(.float16)), inSlot("tg", fix(.uint16)), inSlot("tu", fix(.uint16)), inSlot("slots", fix(.uint32)) }, &.{ outSlot("yg", fix(.float32)), outSlot("yu", fix(.float32)) }, &pair_prep_gemv_p_sc, &pair_prep_gemv_p_p1) ++
    pairCodecSpecs("sushi_exl3_down_prepared", false, hdr_scalar, hdr_pair1, srcClean(kernels.DOWN_PREPARED_SOURCE), &.{ inSlot("middle", fix(.float16)), inSlot("trellis", fix(.uint16)), inSlot("slots", fix(.uint32)) }, &.{outSlot("y", fix(.float16))}, &down_prep_p_sc, &down_prep_p_p1) ++
    pairCodecSpecs("sushi_exl3_pair_gemv_grouped_g2", false, hdr_scalar, hdr_pair1, if (custom.is_codegen) srcClean(kernels.pairGroupedSource(2)) else null, &.{ inSlot("x", par("X")), inSlot("suhg", fix(.float16)), inSlot("suhu", fix(.float16)), inSlot("tg", fix(.uint16)), inSlot("tu", fix(.uint16)), inSlot("slots", fix(.uint32)) }, &.{ outSlot("yg", fix(.float32)), outSlot("yu", fix(.float32)) }, &pg_g2_sc, &pg_g2_p1) ++
    pairCodecSpecs("sushi_exl3_pair_gemv_prepared_grouped_g2", false, hdr_scalar, hdr_pair1, if (custom.is_codegen) srcClean(kernels.pairPreparedGroupedSource(2)) else null, &.{ inSlot("prep", fix(.float16)), inSlot("tg", fix(.uint16)), inSlot("tu", fix(.uint16)), inSlot("slots", fix(.uint32)) }, &.{ outSlot("yg", fix(.float32)), outSlot("yu", fix(.float32)) }, &ppg_g2_sc, &ppg_g2_p1) ++
    pairCodecSpecs("sushi_exl3_down_prepared_grouped_g2", false, hdr_scalar, hdr_pair1, if (custom.is_codegen) srcClean(kernels.downGroupedSource(2)) else null, &.{ inSlot("middle", fix(.float16)), inSlot("trellis", fix(.uint16)), inSlot("slots", fix(.uint32)) }, &.{outSlot("y", fix(.float16))}, &dpg_g2_sc, &dpg_g2_p1) ++
    pairCodecSpecs("sushi_lane_pair", false, hdr_scalar, hdr_pair1, srcClean(kernels.LANE_PAIR_SOURCE), &.{ inSlot("xg", fix(.float16)), inSlot("xu", fix(.float16)), inSlot("tg", fix(.uint16)), inSlot("tu", fix(.uint16)), idxSlot("slots") }, &.{ outSlot("yg", fix(.float16)), outSlot("yu", fix(.float16)) }, &gemv_idx_p_sc, &gemv_idx_p_p1) ++
    pairCodecSpecs("sushi_lane_pair_one", false, hdr_scalar, hdr_pair1, srcClean(kernels.ONE_ROW_PAIR_SOURCE), &.{ inSlot("xg", fix(.float16)), inSlot("xu", fix(.float16)), inSlot("tg", fix(.uint16)), inSlot("tu", fix(.uint16)), idxSlot("slots") }, &.{ outSlot("yg", fix(.float16)), outSlot("yu", fix(.float16)) }, &gemv_idx_p_sc, &gemv_idx_p_p1) ++
    pairCodecSpecs("sushi_down_lane_coop", false, hdr_scalar, hdr_pair1, srcClean(kernels.DOWN_LANE_COOP_SOURCE), &.{ inSlot("x", fix(.float16)), inSlot("trellis", fix(.uint16)), idxSlot("slots") }, &.{outSlot("y", fix(.float16))}, &gemv_idx_p_sc, &gemv_idx_p_p1) ++
    pairCodecSpecs("sushi_down_one", false, hdr_scalar, hdr_pair1, srcClean(kernels.ONE_ROW_COOP_SOURCE), &.{ inSlot("x", fix(.float16)), inSlot("trellis", fix(.uint16)), idxSlot("slots") }, &.{outSlot("y", fix(.float16))}, &gemv_idx_p_sc, &gemv_idx_p_p1) ++
    pairCodecSpecs("sushi_exl3_clamped_middle_down", false, hdr_scalar, hdr_pair1, custom.codegenFn(srcClampedMidDown), &.{ inSlot("ig", fix(.float16)), inSlot("iu", fix(.float16)), inSlot("trellis", fix(.uint16)), inSlot("svhg", fix(.float16)), inSlot("svhu", fix(.float16)), inSlot("suhd", fix(.float16)), inSlot("slots", fix(.uint32)) }, &.{outSlot("y", fix(.float16))}, &clamped_md_p_sc, &clamped_md_p_p1) ++
    pairCodecSpecs("sushi_exl3_group2", false, hdr_scalar, hdr_pair1, srcClean(coop_rows2.SOURCE), &.{ inSlot("x", fix(.float16)), inSlot("trellis", fix(.uint16)), idxSlot("slots") }, &.{outSlot("y", fix(.float16))}, &coop_t1_p_sc, &coop_t1_p_p1) ++
    pairCodecSpecs("sushi_exl3_group2_tile2_lane", false, hdr_scalar, hdr_pair1, srcClean(coop_rows2.LANE_TILED_SOURCE), &.{ inSlot("x", fix(.float16)), inSlot("trellis", fix(.uint16)), idxSlot("slots") }, &.{outSlot("y", fix(.float16))}, &coop_t2_p_sc, &coop_t2_p_p1) ++
    pairCodecSpecs("sushi_exl3_group3_tile2_lane", false, hdr_scalar, hdr_pair1, srcClean(coop_rows2.LANE_THREE_SOURCE), &.{ inSlot("x", fix(.float16)), inSlot("trellis", fix(.uint16)), idxSlot("slots") }, &.{outSlot("y", fix(.float16))}, &coop_t2_p_sc, &coop_t2_p_p1) ++
    pairCodecSpecs("sushi_exl3_pair_group2", false, hdr_scalar, hdr_pair1, srcClean(coop_rows2.PAIR_SOURCE), &.{ inSlot("xg", fix(.float16)), inSlot("xu", fix(.float16)), inSlot("tg", fix(.uint16)), inSlot("tu", fix(.uint16)), idxSlot("slots") }, &.{ outSlot("yg", fix(.float16)), outSlot("yu", fix(.float16)) }, &coop_t1_p_sc, &coop_t1_p_p1) ++
    pairCodecSpecs("sushi_exl3_pair_group2_tile2_lane", false, hdr_scalar, hdr_pair1, srcClean(coop_rows2.LANE_TILED_PAIR_SOURCE), &.{ inSlot("xg", fix(.float16)), inSlot("xu", fix(.float16)), inSlot("tg", fix(.uint16)), inSlot("tu", fix(.uint16)), idxSlot("slots") }, &.{ outSlot("yg", fix(.float16)), outSlot("yu", fix(.float16)) }, &coop_t2_p_sc, &coop_t2_p_p1) ++
    pairCodecSpecs("sushi_exl3_pair_group3_tile2_lane", false, hdr_scalar, hdr_pair1, srcClean(coop_rows2.LANE_THREE_PAIR_SOURCE), &.{ inSlot("xg", fix(.float16)), inSlot("xu", fix(.float16)), inSlot("tg", fix(.uint16)), inSlot("tu", fix(.uint16)), idxSlot("slots") }, &.{ outSlot("yg", fix(.float16)), outSlot("yu", fix(.float16)) }, &coop_t2_p_sc, &coop_t2_p_p1);

// ── Unit 2: `exl3_rows_simd` Spec Tables ──

const rows_finish_specs = [_]KernelSpec{
    .{
        .name = "sushi_exl3_rows_finish",
        .header = project_rows.ROWS_HELPERS,
        .source = project_rows.ROWS_FINISH_SOURCE,
        .inputs = &.{ inSlot("part", fix(.float32)), inSlot("svh", fix(.float16)) },
        .outputs = &.{outSlot("y", par("OutT"))},
        .params = &.{ .{ .name = "OutT", .kind = .ct_dtype, .dtypes = &floats }, fcInt("ODIM"), fcInt("ROWS"), fcInt("NSPLIT"), fcInt("SB1"), fcInt("SB2"), fcInt("SB3") },
    },
    .{
        .name = "sushi_exl3_rows_finish_thru",
        .header = project_rows.ROWS_HELPERS,
        .source = project_rows.ROWS_FINISH_THRU_SOURCE,
        .inputs = &.{ inSlot("part", fix(.float32)), inSlot("svh", fix(.float16)) },
        .outputs = &.{outSlot("y", fix(.float32))},
        .params = &.{ .{ .name = "OutT", .kind = .ct_dtype, .dtypes = &.{.bfloat16} }, fcInt("ODIM"), fcInt("ROWS"), fcInt("NSPLIT"), fcInt("SB1"), fcInt("SB2"), fcInt("SB3") },
    },
    .{
        .name = "sushi_exl3_rows_finish_dyn",
        .header = project_rows.ROWS_HELPERS,
        .source = project_rows.ROWS_FINISH_DYN_SOURCE,
        .inputs = &.{ inSlot("part", fix(.float32)), inSlot("svh", fix(.float16)) },
        .outputs = &.{outSlot("y", par("OutT"))},
        .params = &.{ .{ .name = "OutT", .kind = .ct_dtype, .dtypes = &floats }, fcInt("ODIM"), fcInt("ROWS"), fcInt("NSPLIT"), fcInt("SB1"), fcInt("SB2"), fcInt("SB3") },
    },
    .{
        .name = "sushi_exl3_rows_finish_rms",
        .header = project_rows.ROWS_HELPERS,
        .source_fn = srcClean(project_rows.ROWS_FINISH_RMS_SOURCE),
        .inputs = &.{ inSlot("part", fix(.float32)), inSlot("svh", fix(.float16)), inSlot("w", par("OutT")), inSlot("eps", fix(.float32)) },
        .outputs = &.{outSlot("y", par("OutT"))},
        .params = &.{ .{ .name = "OutT", .kind = .ct_dtype, .dtypes = &floats }, fcInt("ODIM"), fcInt("ROWS"), fcInt("NSPLIT"), fcInt("NORM"), fcInt("SB1"), fcInt("SB2"), fcInt("SB3") },
    },
    .{
        .name = "sushi_exl3_rows_finish_rms_thru",
        .header = project_rows.ROWS_HELPERS,
        .source_fn = srcClean(project_rows.ROWS_FINISH_RMS_THRU_SOURCE),
        .inputs = &.{ inSlot("part", fix(.float32)), inSlot("svh", fix(.float16)), inSlot("w", par("W")), inSlot("eps", fix(.float32)) },
        .outputs = &.{outSlot("y", fix(.float32))},
        .params = &.{ .{ .name = "OutT", .kind = .ct_dtype, .dtypes = &.{.float32} }, .{ .name = "W", .kind = .io_dtype, .dtypes = &bf_f32 }, fcInt("ODIM"), fcInt("ROWS"), fcInt("NSPLIT"), fcInt("NORM"), fcInt("SB1"), fcInt("SB2"), fcInt("SB3") },
    },
    .{
        .name = "sushi_exl3_rows_finish_rms_ring",
        .header = project_rows.ROWS_RING_HELPERS,
        .source_fn = srcClean(project_rows.ROWS_FINISH_RMS_RING_SOURCE),
        .inputs = &.{
            inSlot("part", fix(.float32)),
            inSlot("svh", fix(.float16)),
            inSlot("w", par("OutT")),
            inSlot("eps", fix(.float32)),
            inSlot("kw", par("OutT")),
            inSlot("freqs", fix(.float32)),
            inSlot("pos", fix(.int32)),
            inSlot("ring", par("OutT")),
        },
        .outputs = &.{ outSlot("qout", par("OutT")), outSlot("rout", par("OutT")) },
        .params = &.{ .{ .name = "OutT", .kind = .ct_dtype, .dtypes = &floats }, fcInt("ODIM"), fcInt("ROWS"), fcInt("NSPLIT"), fcInt("NORM"), fcInt("KV_D"), fcInt("RD"), fcInt("QAT") },
    },
    .{
        .name = "sushi_exl3_rows_finish_rms_ring_ip",
        .header = if (custom.is_codegen) project_rows.ROWS_RING_HELPERS else "",
        .source_fn = srcClean(project_rows.ROWS_FINISH_RMS_RING_IP_SOURCE),
        .inputs = &.{
            inSlot("part", fix(.float32)),
            inSlot("svh", fix(.float16)),
            inSlot("w", par("OutT")),
            inSlot("eps", fix(.float32)),
            inSlot("kw", par("OutT")),
            inSlot("freqs", fix(.float32)),
            inSlot("pos", fix(.int32)),
        },
        .outputs = &.{ outSlot("qout", par("OutT")), outSlot("rout", par("OutT")) },
        .params = &.{ .{ .name = "OutT", .kind = .ct_dtype, .dtypes = &floats }, fcInt("ODIM"), fcInt("ROWS"), fcInt("NSPLIT"), fcInt("NORM"), fcInt("KV_D"), fcInt("RD"), fcInt("QAT"), fcInt("CAP") },
    },
    .{
        .name = "sushi_exl3_rows_finish_rope",
        .header = project_rows.ROWS_HELPERS,
        .source = project_rows.ROWS_FINISH_ROPE_SOURCE,
        .inputs = &.{ inSlot("part", fix(.float32)), inSlot("svh", fix(.float16)), inSlot("freqs", fix(.float32)), inSlot("pos", fix(.int32)) },
        .outputs = &.{outSlot("y", par("OutT"))},
        .params = &.{ .{ .name = "OutT", .kind = .ct_dtype, .dtypes = &floats }, fcInt("ODIM"), fcInt("NSPLIT"), fcInt("HD"), fcInt("RD") },
    },
    .{
        .name = "sushi_exl3_rows_finish_row_major",
        .header = project_rows.ROWS_HELPERS,
        .source = project_rows.ROWS_FINISH_ROW_MAJOR_SOURCE,
        .inputs = &.{ inSlot("part", fix(.float32)), inSlot("svh", fix(.float16)) },
        .outputs = &.{outSlot("y", par("OutT"))},
        .params = &.{ .{ .name = "OutT", .kind = .ct_dtype, .dtypes = &floats }, fcInt("ODIM"), fcInt("ROWS"), fcInt("NSPLIT"), fcInt("SB1"), fcInt("SB2"), fcInt("SB3"), fcInt("GRP") },
    },
    .{
        .name = "sushi_exl3_rows_finish_expand",
        .header = project_rows.ROWS_HELPERS,
        .source_fn = srcClean(project_rows.ROWS_FINISH_EXPAND_SOURCE),
        .inputs = &.{ inSlot("part", fix(.float32)), inSlot("svh", fix(.float16)), inSlot("x", par("OutT")), inSlot("post", fix(.float32)), inSlot("comb", fix(.float32)) },
        .outputs = &.{outSlot("y", par("OutT"))},
        .params = &.{ .{ .name = "OutT", .kind = .ct_dtype, .dtypes = &floats }, fcInt("ODIM"), fcInt("NSPLIT"), fcInt("HC") },
    },
};

/// Filter for SIMD `project_rows`: only the 5 served `(ROW_CAP, TGRP, KPH)` geometries
/// `{(1,1,4), (4,2,4), (8,2,4), (4,4,2), (8,4,2)}` and valid `(OutT, Y)` pairs (`Y=f32` when `TGRP<4`, `Y=OutT` when `TGRP==4`).
fn rowsSimdFilter(vals: []const CtVal) bool {
    // Variant order: [OutT, Y, ROW_CAP, TGRP, KPH, NHW]
    const out_t = vals[0].dtype;
    const y_t = vals[1].dtype;
    const cap = vals[2].int;
    const tgrp = vals[3].int;
    const kph = vals[4].int;
    const geom_ok = (cap == 1 and tgrp == 1 and kph == 4) or
        ((cap == 4 or cap == 8) and tgrp == 2 and kph == 4) or
        ((cap == 4 or cap == 8) and tgrp == 4 and kph == 2);
    if (!geom_ok) return false;
    if (tgrp < 4) return y_t == .float32;
    return y_t == out_t;
}

fn rowsSimdPendFilter(vals: []const CtVal) bool {
    // Variant order: [OutT, XT, Y, ROW_CAP, TGRP, KPH, NHW]
    const out_t = vals[0].dtype;
    const xt = vals[1].dtype;
    const y_t = vals[2].dtype;
    const cap = vals[3].int;
    const tgrp = vals[4].int;
    const kph = vals[5].int;
    if (out_t != xt) return false;
    const geom_ok = (cap == 1 and tgrp == 1 and kph == 4) or
        ((cap == 4 or cap == 8) and tgrp == 2 and kph == 4) or
        ((cap == 4 or cap == 8) and tgrp == 4 and kph == 2);
    if (!geom_ok) return false;
    if (tgrp < 4) return y_t == .float32;
    return y_t == out_t;
}

fn rowsSimdThruFilter(vals: []const CtVal) bool {
    // Variant order: [OutT, X, ROW_CAP, TGRP, KPH, NHW]
    const n = vals.len;
    const cap = vals[n - 4].int;
    const tgrp = vals[n - 3].int;
    const kph = vals[n - 2].int;
    return (cap == 1 and tgrp == 1 and kph == 4) or
        ((cap == 4 or cap == 8) and tgrp == 2 and kph == 4) or
        ((cap == 4 or cap == 8) and tgrp == 4 and kph == 2);
}

fn rowsSimdCountedFilter(vals: []const CtVal) bool {
    // Variant order: [OutT, (FIN or W)?, ROW_CAP, TGRP, KPH, NHW]
    const n = vals.len;
    const cap = vals[n - 4].int;
    const tgrp = vals[n - 3].int;
    const kph = vals[n - 2].int;
    return ((cap == 1 and tgrp == 1) or ((cap == 4 or cap == 8) and tgrp == 2)) and kph == 4;
}

const rows_ins = [_]InputSlot{
    inSlot("x", par("OutT")),
    inSlot("trellis", fix(.uint16)),
    inSlot("suh", fix(.float16)),
    inSlot("svh", fix(.float16)),
};
const rows_rope_ins = [_]InputSlot{
    inSlot("x", par("OutT")),
    inSlot("trellis", fix(.uint16)),
    inSlot("suh", fix(.float16)),
    inSlot("svh", fix(.float16)),
    inSlot("freqs", fix(.float32)),
    inSlot("pos", fix(.int32)),
};
const rows_thru_ins = [_]InputSlot{
    inSlot("x", par("X")),
    inSlot("trellis", fix(.uint16)),
    inSlot("suh", fix(.float16)),
    inSlot("svh", fix(.float16)),
};
const rows_cnt_ins = [_]InputSlot{
    inSlot("x", fix(.float32)),
    inSlot("trellis", fix(.uint16)),
    inSlot("suh", fix(.float16)),
    inSlot("svh", fix(.float16)),
};
const rows_pend_ins = [_]InputSlot{
    inSlot("x", fix(.float32)),
    inSlot("trellis", fix(.uint16)),
    inSlot("suh", fix(.float16)),
    inSlot("svh", fix(.float16)),
    inSlot("xsvh", fix(.float16)),
};
const rows_cnt_rms_ins = [_]InputSlot{
    inSlot("x", fix(.float32)),
    inSlot("trellis", fix(.uint16)),
    inSlot("suh", fix(.float16)),
    inSlot("svh", fix(.float16)),
    inSlot("w", par("W")),
    inSlot("eps", fix(.float32)),
};
const rows_cnt_pair_ins = [_]InputSlot{
    inSlot("x", fix(.float32)),
    inSlot("trellis", fix(.uint16)),
    inSlot("suh", fix(.float16)),
    inSlot("svh", fix(.float16)),
    inSlot("trellis2", fix(.uint16)),
    inSlot("suh2", fix(.float16)),
    inSlot("svh2", fix(.float16)),
};

const rows_simd_specs = [_]KernelSpec{
    .{
        .name = "sushi_exl3_rows_scalar",
        .header_fn = custom.codegenFn(exl3HeaderFn),
        .source_arg = hdr_scalar_rows_simd,
        .source_fn = srcRows(project_rows.ROWS_SOURCE, false, false),
        .inputs = &rows_ins,
        .outputs = &.{outSlot("y", par("Y"))},
        .params = &.{
            .{ .name = "OutT", .kind = .ct_dtype, .dtypes = &floats },
            .{ .name = "Y", .kind = .io_dtype, .dtypes = &floats },
            .{ .name = "ROW_CAP", .kind = .ct_int, .ints = &.{ 1, 4, 8 } },
            .{ .name = "TGRP", .kind = .ct_int, .ints = &.{ 1, 2, 4 } },
            .{ .name = "KPH", .kind = .ct_int, .ints = &.{ 2, 4 } },
            .{ .name = "NHW", .kind = .ct_int, .ints = &trunk_scalar_rates },
            fcInt("IDIM"), fcInt("ODIM"), fcInt("ROWS"), fcInt("GROUPS"), fcInt("NSPLIT"), fcInt("CHUNK"), fcInt("FOLD"), fcInt("SB1"), fcInt("SB2"), fcInt("SB3"),
        },
        .filter = rowsSimdFilter,
    },
    .{
        .name = "sushi_exl3_rows_pair1",
        .header_fn = custom.codegenFn(exl3HeaderFn),
        .source_arg = hdr_pair1_rows_simd,
        .source_fn = srcRows(project_rows.ROWS_SOURCE, false, false),
        .inputs = &rows_ins,
        .outputs = &.{outSlot("y", par("Y"))},
        .params = &.{
            .{ .name = "OutT", .kind = .ct_dtype, .dtypes = &bf_f16 },
            .{ .name = "Y", .kind = .io_dtype, .dtypes = &floats },
            .{ .name = "ROW_CAP", .kind = .ct_int, .ints = &.{ 1, 4, 8 } },
            .{ .name = "TGRP", .kind = .ct_int, .ints = &.{ 1, 2, 4 } },
            .{ .name = "KPH", .kind = .ct_int, .ints = &.{ 2, 4 } },
            .{ .name = "NHW", .kind = .ct_int, .ints = &pair1_rates },
            fcInt("IDIM"), fcInt("ODIM"), fcInt("ROWS"), fcInt("GROUPS"), fcInt("NSPLIT"), fcInt("CHUNK"), fcInt("FOLD"), fcInt("SB1"), fcInt("SB2"), fcInt("SB3"),
        },
        .filter = rowsSimdFilter,
    },
    .{
        .name = "sushi_exl3_rows_pend_scalar",
        .header_fn = custom.codegenFn(exl3HeaderFn),
        .source_arg = hdr_scalar_rows_simd,
        .source_fn = srcRows(project_rows.ROWS_PEND_SOURCE, false, false),
        .inputs = &rows_pend_ins,
        .outputs = &.{outSlot("y", par("Y"))},
        .params = &.{
            .{ .name = "OutT", .kind = .ct_dtype, .dtypes = &floats },
            .{ .name = "XT", .kind = .ct_dtype, .dtypes = &floats },
            .{ .name = "Y", .kind = .io_dtype, .dtypes = &floats },
            .{ .name = "ROW_CAP", .kind = .ct_int, .ints = &.{ 1, 4, 8 } },
            .{ .name = "TGRP", .kind = .ct_int, .ints = &.{ 1, 2, 4 } },
            .{ .name = "KPH", .kind = .ct_int, .ints = &.{ 2, 4 } },
            .{ .name = "NHW", .kind = .ct_int, .ints = &.{ 32, 48, 64, 80, 96, 112, 128 } },
            fcInt("IDIM"), fcInt("ODIM"), fcInt("ROWS"), fcInt("GROUPS"), fcInt("NSPLIT"), fcInt("CHUNK"), fcInt("FOLD"), fcInt("SB1"), fcInt("SB2"), fcInt("SB3"), fcInt("XSPLIT"), fcInt("XODIM"),
        },
        .filter = rowsSimdPendFilter,
    },
    .{
        .name = "sushi_exl3_rows_thru_scalar",
        .header_fn = custom.codegenFn(exl3HeaderFn),
        .source_arg = hdr_scalar_rows_simd,
        .source_fn = srcRows(project_rows.ROWS_THRU_SOURCE, false, false),
        .inputs = &rows_thru_ins,
        .outputs = &.{outSlot("y", fix(.float32))},
        .params = &.{
            .{ .name = "OutT", .kind = .ct_dtype, .dtypes = &.{.bfloat16} },
            .{ .name = "X", .kind = .io_dtype, .dtypes = &bf_f32 },
            .{ .name = "ROW_CAP", .kind = .ct_int, .ints = &.{ 1, 4, 8 } },
            .{ .name = "TGRP", .kind = .ct_int, .ints = &.{ 1, 2, 4 } },
            .{ .name = "KPH", .kind = .ct_int, .ints = &.{ 2, 4 } },
            .{ .name = "NHW", .kind = .ct_int, .ints = &trunk_scalar_rates },
            fcInt("IDIM"), fcInt("ODIM"), fcInt("ROWS"), fcInt("GROUPS"), fcInt("NSPLIT"), fcInt("CHUNK"), fcInt("FOLD"), fcInt("SB1"), fcInt("SB2"), fcInt("SB3"),
        },
        .filter = rowsSimdThruFilter,
    },
    .{
        .name = "sushi_exl3_rows_thru_counted_scalar",
        .header_fn = custom.codegenFn(exl3HeaderFn),
        .source_arg = hdr_scalar_rows_simd_cnt,
        .source_fn = srcRows(project_rows.ROWS_THRU_COUNTED_SOURCE, false, true),
        .inputs = &rows_cnt_ins,
        .outputs = &.{ outSlot("y", fix(.float32)), outSlot("fin", par("FIN")), outSlot("ctr", fix(.uint32)) },
        .params = &.{
            .{ .name = "OutT", .kind = .ct_dtype, .dtypes = &.{.bfloat16} },
            .{ .name = "FIN", .kind = .io_dtype, .dtypes = &bf_f32 },
            .{ .name = "ROW_CAP", .kind = .ct_int, .ints = &.{ 1, 4, 8 } },
            .{ .name = "TGRP", .kind = .ct_int, .ints = &.{ 1, 2 } },
            .{ .name = "KPH", .kind = .ct_int, .ints = &.{4} },
            .{ .name = "NHW", .kind = .ct_int, .ints = &trunk_scalar_rates },
            fcInt("IDIM"), fcInt("ODIM"), fcInt("ROWS"), fcInt("GROUPS"), fcInt("NSPLIT"), fcInt("CHUNK"), fcInt("FOLD"), fcInt("SB1"), fcInt("SB2"), fcInt("SB3"),
        },
        .filter = rowsSimdCountedFilter,
    },
    .{
        .name = "sushi_exl3_rows_thru_counted_rms_scalar",
        .header_fn = custom.codegenFn(exl3HeaderFn),
        .source_arg = hdr_scalar_rows_simd_cnt,
        .source_fn = srcRows(project_rows.ROWS_THRU_COUNTED_RMS_SOURCE, false, true),
        .inputs = &rows_cnt_rms_ins,
        .outputs = &.{ outSlot("y", fix(.float32)), outSlot("fin", fix(.float32)), outSlot("ctr", fix(.uint32)) },
        .params = &.{
            .{ .name = "OutT", .kind = .ct_dtype, .dtypes = &.{.bfloat16} },
            .{ .name = "W", .kind = .io_dtype, .dtypes = &bf_f32 },
            .{ .name = "ROW_CAP", .kind = .ct_int, .ints = &.{ 1, 4, 8 } },
            .{ .name = "TGRP", .kind = .ct_int, .ints = &.{ 1, 2 } },
            .{ .name = "KPH", .kind = .ct_int, .ints = &.{4} },
            .{ .name = "NHW", .kind = .ct_int, .ints = &trunk_scalar_rates },
            fcInt("IDIM"), fcInt("ODIM"), fcInt("ROWS"), fcInt("GROUPS"), fcInt("NSPLIT"), fcInt("CHUNK"), fcInt("FOLD"), fcInt("SB1"), fcInt("SB2"), fcInt("SB3"),
        },
        .filter = rowsSimdCountedFilter,
    },
    .{
        .name = "sushi_exl3_rows_thru_counted_pair_scalar",
        .header_fn = custom.codegenFn(exl3HeaderFn),
        .source_arg = hdr_scalar_rows_simd_cnt,
        .source_fn = srcRows(project_rows.ROWS_THRU_COUNTED_PAIR_SOURCE, false, true),
        .inputs = &rows_cnt_pair_ins,
        .outputs = &.{ outSlot("y", fix(.float32)), outSlot("fin", fix(.float32)), outSlot("ctr", fix(.uint32)) },
        .params = &.{
            .{ .name = "OutT", .kind = .ct_dtype, .dtypes = &.{.bfloat16} },
            .{ .name = "ROW_CAP", .kind = .ct_int, .ints = &.{ 1, 4, 8 } },
            .{ .name = "TGRP", .kind = .ct_int, .ints = &.{ 1, 2 } },
            .{ .name = "KPH", .kind = .ct_int, .ints = &.{4} },
            .{ .name = "NHW", .kind = .ct_int, .ints = &trunk_scalar_rates },
            fcInt("IDIM"), fcInt("ODIM"), fcInt("ROWS"), fcInt("GROUPS"), fcInt("NSPLIT"), fcInt("CHUNK"), fcInt("FOLD"), fcInt("SB1"), fcInt("SB2"), fcInt("SB3"),
        },
        .filter = rowsSimdCountedFilter,
    },
};

// ── Unit 3: `exl3_rg_simd` Spec Tables ──

fn rgPairSpec(comptime is_pair1: bool) KernelSpec {
    return .{
        .name = if (is_pair1) "sushi_exl3_rg_pair_pair1" else "sushi_exl3_rg_pair_scalar",
        .header_fn = custom.codegenFn(exl3HeaderFn),
        .source_arg = if (is_pair1) hdr_rg_pair1 else hdr_rg_scalar,
        .source_fn = srcRg(.pair, 0, is_pair1),
        .inputs = &.{ inSlot("x", par("X")), inSlot("suhg", fix(.float16)), inSlot("suhu", fix(.float16)), inSlot("tg", fix(.uint16)), inSlot("tu", fix(.uint16)), inSlot("slots", fix(.uint32)), inSlot("meta", fix(.uint32)) },
        .outputs = &.{ outSlot("yg", fix(.float32)), outSlot("yu", fix(.float32)) },
        .params = &.{ .{ .name = "X", .kind = .io_dtype, .dtypes = &floats }, .{ .name = "OTPT", .kind = .ct_int, .ints = &.{ 1, 2 } }, fcInt("NSPLIT"), fcInt("TOPK"), fcInt("IDIM"), fcInt("ODIM"), fcInt("FUNNEL") },
    };
}

fn rgDownSpec(comptime is_pair1: bool) KernelSpec {
    return .{
        .name = if (is_pair1) "sushi_exl3_rg_down_pair1" else "sushi_exl3_rg_down_scalar",
        .header_fn = custom.codegenFn(exl3HeaderFn),
        .source_arg = if (is_pair1) hdr_rg_pair1 else hdr_rg_scalar,
        .source_fn = srcRg(.down, 0, is_pair1),
        .inputs = &.{ inSlot("ig", fix(.float32)), inSlot("iu", fix(.float32)), inSlot("trellis", fix(.uint16)), inSlot("svhg", fix(.float16)), inSlot("svhu", fix(.float16)), inSlot("suhd", fix(.float16)), inSlot("slots", fix(.uint32)), inSlot("meta", fix(.uint32)) },
        .outputs = &.{outSlot("y", fix(.float16))},
        .params = &.{ .{ .name = "OTPT", .kind = .ct_int, .ints = &.{ 1, 2 } }, fcInt("NSPLIT"), fcInt("IDIM"), fcInt("ODIM"), fcInt("FUNNEL") },
    };
}

fn rgPrepPairSpec(comptime group: u32, comptime is_pair1: bool) KernelSpec {
    const suffix = (if (group == 0) "" else std.fmt.comptimePrint("_g{d}", .{group})) ++ (if (is_pair1) "_pair1" else "_scalar");
    const otpts: []const i32 = if (group <= 2) &.{ 1, 2 } else &.{1};
    return .{
        .name = "sushi_exl3_rg_pair_prepared" ++ suffix,
        .header_fn = custom.codegenFn(exl3HeaderFn),
        .source_arg = if (is_pair1) hdr_rg_pair1 else hdr_rg_scalar,
        .source_fn = srcRg(.pair_prepared, group, is_pair1),
        .inputs = &.{ inSlot("prep", fix(.float16)), inSlot("tg", fix(.uint16)), inSlot("tu", fix(.uint16)), inSlot("slots", fix(.uint32)), inSlot("meta", fix(.uint32)) },
        .outputs = &.{ outSlot("yg", fix(.float32)), outSlot("yu", fix(.float32)) },
        .params = if (group == 0)
            &.{ .{ .name = "OTPT", .kind = .ct_int, .ints = otpts }, fcInt("NSPLIT"), fcInt("TOPK"), fcInt("IDIM"), fcInt("ODIM"), fcInt("FUNNEL") }
        else
            &.{ .{ .name = "GROUP", .kind = .ct_int, .ints = &.{@as(i32, @intCast(group))} }, .{ .name = "OTPT", .kind = .ct_int, .ints = otpts }, fcInt("NSPLIT"), fcInt("TOPK"), fcInt("IDIM"), fcInt("ODIM"), fcInt("FUNNEL"), fcInt("NSLOTS") },
    };
}

fn rgPrepDownSpec(comptime group: u32, comptime is_pair1: bool) KernelSpec {
    const suffix = (if (group == 0) "" else std.fmt.comptimePrint("_g{d}", .{group})) ++ (if (is_pair1) "_pair1" else "_scalar");
    const otpts: []const i32 = if (group <= 2) &.{ 1, 2 } else &.{1};
    return .{
        .name = "sushi_exl3_rg_down_prepared" ++ suffix,
        .header_fn = custom.codegenFn(exl3HeaderFn),
        .source_arg = if (is_pair1) hdr_rg_pair1 else hdr_rg_scalar,
        .source_fn = srcRg(.down_prepared, group, is_pair1),
        .inputs = &.{ inSlot("middle", fix(.float16)), inSlot("trellis", fix(.uint16)), inSlot("slots", fix(.uint32)), inSlot("meta", fix(.uint32)) },
        .outputs = &.{outSlot("y", fix(.float16))},
        .params = if (group == 0)
            &.{ .{ .name = "OTPT", .kind = .ct_int, .ints = otpts }, fcInt("NSPLIT"), fcInt("IDIM"), fcInt("ODIM"), fcInt("FUNNEL") }
        else
            &.{ .{ .name = "GROUP", .kind = .ct_int, .ints = &.{@as(i32, @intCast(group))} }, .{ .name = "OTPT", .kind = .ct_int, .ints = otpts }, fcInt("NSPLIT"), fcInt("IDIM"), fcInt("ODIM"), fcInt("FUNNEL"), fcInt("NSLOTS") },
    };
}

const rg_simd_specs = [_]KernelSpec{
    rgPairSpec(false),        rgPairSpec(true),
    rgDownSpec(false),        rgDownSpec(true),
    rgPrepPairSpec(0, false), rgPrepPairSpec(0, true),
    rgPrepPairSpec(2, false), rgPrepPairSpec(2, true),
    rgPrepPairSpec(4, false), rgPrepPairSpec(4, true),
    rgPrepPairSpec(8, false), rgPrepPairSpec(8, true),
    rgPrepPairSpec(16, false), rgPrepPairSpec(16, true),
    rgPrepDownSpec(0, false), rgPrepDownSpec(0, true),
    rgPrepDownSpec(2, false), rgPrepDownSpec(2, true),
    rgPrepDownSpec(4, false), rgPrepDownSpec(4, true),
    rgPrepDownSpec(8, false), rgPrepDownSpec(8, true),
    rgPrepDownSpec(16, false), rgPrepDownSpec(16, true),
};

// ── Unit 4: `exl3_fz_simd` Spec Tables ──

const fz_grid_params = [_]ParamSpec{
    fcInt("FZ_AX"), fcInt("FZ_AY"), fcInt("FZ_BX"), fcInt("FZ_BY"), fcInt("FZ_NA"), fcInt("FZ_NB"), fcInt("FZ_B_FIRST"),
};

fn fzPairSpec(comptime group: u32, comptime b_mode: FzBMode, comptime a_pair1: bool, comptime b_pair1: bool) KernelSpec {
    const g_s = if (group == 0) "" else std.fmt.comptimePrint("_g{d}", .{group});
    const m_s = switch (b_mode) {
        .lane => "_lane",
        .one => "_one",
        .shared_rows => "_shared_rows",
    };
    const ab_s = (if (a_pair1) "_ap1" else "_asc") ++ (if (b_pair1) "_bp1" else "_bsc");
    const a_p = if (group == 0)
        &[_]ParamSpec{ .{ .name = "OTPT", .kind = .ct_int, .ints = &.{ 1, 2 } }, fcInt("NSPLIT"), fcInt("TOPK"), fcInt("IDIM"), fcInt("ODIM"), fcInt("FUNNEL") }
    else
        &[_]ParamSpec{ .{ .name = "GROUP", .kind = .ct_int, .ints = &.{@as(i32, @intCast(group))} }, .{ .name = "OTPT", .kind = .ct_int, .ints = &.{ 1, 2 } }, fcInt("NSPLIT"), fcInt("TOPK"), fcInt("IDIM"), fcInt("ODIM"), fcInt("FUNNEL"), fcInt("NSLOTS") };
    const b_p = if (b_mode == .shared_rows)
        &[_]ParamSpec{ fcInt("S_IDIM"), fcInt("S_ODIM"), fcInt("S_NHW"), fcInt("S_ROWS"), .{ .name = "S_SHARED_ROWS", .kind = .ct_int, .ints = &.{ 2, 3 } } }
    else
        &[_]ParamSpec{ fcInt("S_IDIM"), fcInt("S_ODIM"), fcInt("S_NHW"), ign("S_ROWS"), ign("S_SHARED_ROWS") };
    return .{
        .name = "sushi_exl3_fz_pair" ++ g_s ++ m_s ++ ab_s,
        .header_fn = hdrFzStage(a_pair1, b_pair1),
        .source_fn = srcFzStage(false, group, b_mode, a_pair1, b_pair1),
        .inputs = &.{
            inSlot("prep", fix(.float16)), inSlot("tg", fix(.uint16)), inSlot("tu", fix(.uint16)), inSlot("slots", fix(.uint32)), inSlot("meta", fix(.uint32)),
            inSlot("s_xg", fix(.float16)), inSlot("s_xu", fix(.float16)), inSlot("s_tg", fix(.uint16)), inSlot("s_tu", fix(.uint16)), inSlot("s_slots", fix(.uint32)),
        },
        .outputs = &.{ outSlot("yg", fix(.float32)), outSlot("yu", fix(.float32)), outSlot("s_yg", fix(.float16)), outSlot("s_yu", fix(.float16)) },
        .params = a_p ++ b_p ++ &fz_grid_params,
    };
}

fn fzDownSpec(comptime group: u32, comptime b_mode: FzBMode, comptime a_pair1: bool, comptime b_pair1: bool) KernelSpec {
    const g_s = if (group == 0) "" else std.fmt.comptimePrint("_g{d}", .{group});
    const m_s = switch (b_mode) {
        .lane => "_lane",
        .one => "_one",
        .shared_rows => "_shared_rows",
    };
    const ab_s = (if (a_pair1) "_ap1" else "_asc") ++ (if (b_pair1) "_bp1" else "_bsc");
    const a_p = if (group == 0)
        &[_]ParamSpec{ .{ .name = "OTPT", .kind = .ct_int, .ints = &.{ 1, 2 } }, fcInt("NSPLIT"), fcInt("IDIM"), fcInt("ODIM"), fcInt("FUNNEL") }
    else
        &[_]ParamSpec{ .{ .name = "GROUP", .kind = .ct_int, .ints = &.{@as(i32, @intCast(group))} }, .{ .name = "OTPT", .kind = .ct_int, .ints = &.{ 1, 2 } }, fcInt("NSPLIT"), fcInt("IDIM"), fcInt("ODIM"), fcInt("FUNNEL"), fcInt("NSLOTS") };
    const b_p = if (b_mode == .shared_rows)
        &[_]ParamSpec{ fcInt("S_IDIM"), fcInt("S_ODIM"), fcInt("S_NHW"), fcInt("S_ROWS"), .{ .name = "S_SHARED_ROWS", .kind = .ct_int, .ints = &.{ 2, 3 } } }
    else
        &[_]ParamSpec{ fcInt("S_IDIM"), fcInt("S_ODIM"), fcInt("S_NHW"), ign("S_ROWS"), ign("S_SHARED_ROWS") };
    return .{
        .name = "sushi_exl3_fz_down" ++ g_s ++ m_s ++ ab_s,
        .header_fn = hdrFzStage(a_pair1, b_pair1),
        .source_fn = srcFzStage(true, group, b_mode, a_pair1, b_pair1),
        .inputs = &.{
            inSlot("middle", fix(.float16)), inSlot("trellis", fix(.uint16)), inSlot("slots", fix(.uint32)), inSlot("meta", fix(.uint32)),
            inSlot("s_x", fix(.float16)), inSlot("s_trellis", fix(.uint16)), inSlot("s_slots", fix(.uint32)),
        },
        .outputs = &.{ outSlot("y", fix(.float16)), outSlot("s_y", fix(.float16)) },
        .params = a_p ++ b_p ++ &fz_grid_params,
    };
}

fn foldAffineSpec(comptime mixed: bool, comptime is_down: bool, comptime with_logit: bool, comptime a_pair1: bool) KernelSpec {
    const nm = (if (mixed) "sushi_rg_fold_" else "sushi_gated_fold_") ++
        (if (is_down) "down" else "pair") ++
        (if (with_logit) "_logit" else "") ++
        (if (a_pair1) "_ap1" else "_asc");

    const a_ins_pair_rg = [_]InputSlot{ inSlot("x", fix(.bfloat16)), inSlot("suhg", fix(.float16)), inSlot("suhu", fix(.float16)), inSlot("tg", fix(.uint16)), inSlot("tu", fix(.uint16)), inSlot("slots", fix(.uint32)), inSlot("meta", fix(.uint32)) };
    const a_ins_pair_gt = [_]InputSlot{ inSlot("x", fix(.bfloat16)), inSlot("suhg", fix(.float16)), inSlot("suhu", fix(.float16)), inSlot("tg", fix(.uint16)), inSlot("tu", fix(.uint16)), inSlot("slots", fix(.uint32)) };
    const a_ins_down_rg = [_]InputSlot{ inSlot("ig", fix(.float32)), inSlot("iu", fix(.float32)), inSlot("trellis", fix(.uint16)), inSlot("svhg", fix(.float16)), inSlot("svhu", fix(.float16)), inSlot("suhd", fix(.float16)), inSlot("slots", fix(.uint32)), inSlot("meta", fix(.uint32)) };
    const a_ins_down_gt = [_]InputSlot{ inSlot("ig", fix(.float32)), inSlot("iu", fix(.float32)), inSlot("trellis", fix(.uint16)), inSlot("svhg", fix(.float16)), inSlot("svhu", fix(.float16)), inSlot("suhd", fix(.float16)), inSlot("slots", fix(.uint32)) };

    const b_ins_gu = [_]InputSlot{ inSlot("s_x", fix(.bfloat16)), inSlot("s_wg", fix(.uint32)), inSlot("s_sg", fix(.bfloat16)), inSlot("s_bg", fix(.bfloat16)), inSlot("s_wu", fix(.uint32)), inSlot("s_su", fix(.bfloat16)), inSlot("s_bu", fix(.bfloat16)), inSlot("s_sigmoid", fix(.bfloat16)) };
    const b_ins_logit = [_]InputSlot{ inSlot("s_lw", fix(.uint32)), inSlot("s_ls", fix(.bfloat16)), inSlot("s_lb", fix(.bfloat16)) };
    const b_ins_dn = [_]InputSlot{ inSlot("s_x", fix(.bfloat16)), inSlot("s_w", fix(.uint32)), inSlot("s_sc", fix(.bfloat16)), inSlot("s_bi", fix(.bfloat16)) };

    const a_p = if (is_down)
        (if (mixed) &[_]ParamSpec{ .{ .name = "OTPT", .kind = .ct_int, .ints = &.{ 1, 2 } }, fcInt("NSPLIT"), fcInt("IDIM"), fcInt("ODIM"), fcInt("FUNNEL") } else &[_]ParamSpec{ .{ .name = "OTPT", .kind = .ct_int, .ints = &.{ 1, 2 } }, fcInt("IDIM"), fcInt("ODIM"), fcInt("NSPLIT"), fcInt("NHW"), fcInt("FUNNEL") })
    else
        (if (mixed) &[_]ParamSpec{ .{ .name = "OTPT", .kind = .ct_int, .ints = &.{ 1, 2 } }, fcInt("NSPLIT"), fcInt("TOPK"), fcInt("IDIM"), fcInt("ODIM"), fcInt("FUNNEL") } else &[_]ParamSpec{ .{ .name = "OTPT", .kind = .ct_int, .ints = &.{ 1, 2 } }, fcInt("IDIM"), fcInt("ODIM"), fcInt("NSPLIT"), fcInt("NHW"), fcInt("TOPK"), fcInt("FUNNEL") });
    const b_p = if (is_down)
        &[_]ParamSpec{ fcInt("S_K"), fcInt("S_N"), fcInt("S_GS"), fcInt("S_FAST") }
    else
        &[_]ParamSpec{ fcInt("S_K"), fcInt("S_N"), fcInt("S_GS"), fcInt("S_FAST"), fcInt("S_LGS") };

    return .{
        .name = nm,
        .header_fn = hdrFoldAffine(mixed, a_pair1),
        .source_fn = srcFoldAffine(mixed, is_down, with_logit, a_pair1),
        .inputs = if (is_down)
            (if (mixed) &a_ins_down_rg else &a_ins_down_gt) ++ &b_ins_dn
        else if (with_logit)
            (if (mixed) &a_ins_pair_rg else &a_ins_pair_gt) ++ &b_ins_gu ++ &b_ins_logit
        else
            (if (mixed) &a_ins_pair_rg else &a_ins_pair_gt) ++ &b_ins_gu,
        .outputs = if (is_down)
            &.{ outSlot("y", fix(.float16)), outSlot("s_y", fix(.bfloat16)) }
        else if (with_logit)
            &.{ outSlot("yg", fix(.float32)), outSlot("yu", fix(.float32)), outSlot("s_y", fix(.bfloat16)), outSlot("s_gl", fix(.bfloat16)) }
        else
            &.{ outSlot("yg", fix(.float32)), outSlot("yu", fix(.float32)), outSlot("s_y", fix(.bfloat16)) },
        .params = a_p ++ b_p ++ &fz_grid_params,
    };
}

const fz_simd_specs = blk: {
    @setEvalBranchQuota(100000);
    break :blk [_]KernelSpec{
    // 1. DSv41 fused prep & mid
    .{
        .name = "sushi_exl3_fz_prep",
        .source_fn = custom.codegenFn(srcFzPrep),
        .inputs = &.{
            inSlot("x", fix(.bfloat16)), inSlot("suhg", fix(.float16)), inSlot("suhu", fix(.float16)), inSlot("slots", fix(.uint32)),
            inSlot("s_x", fix(.bfloat16)), inSlot("s_suhg", fix(.float16)), inSlot("s_suhu", fix(.float16)), inSlot("s_slots", fix(.uint32)),
        },
        .outputs = &.{ outSlot("prep", fix(.float16)), outSlot("s_yg", fix(.float16)), outSlot("s_yu", fix(.float16)) },
        .params = &[_]ParamSpec{ fcInt("IDIM"), fcInt("TOPK"), fcInt("S_IDIM"), fcInt("S_TOPK") } ++ &fz_grid_params,
    },
    .{
        .name = "sushi_exl3_fz_prep_snax",
        .source_fn = custom.codegenFn(srcFzPrepSnax),
        .inputs = &.{
            inSlot("x", fix(.bfloat16)), inSlot("suhg", fix(.float16)), inSlot("suhu", fix(.float16)), inSlot("slots", fix(.uint32)),
            inSlot("s_x", fix(.bfloat16)), inSlot("s_suhg", fix(.float16)), inSlot("s_suhu", fix(.float16)), inSlot("s_slots", fix(.uint32)),
        },
        .outputs = &.{ outSlot("prep", fix(.float16)), outSlot("s_yg", fix(.float16)), outSlot("s_yu", fix(.float16)) },
        .params = &[_]ParamSpec{ fcInt("IDIM"), fcInt("TOPK") } ++ &fz_grid_params,
    },
    .{
        .name = "sushi_exl3_fz_mid",
        .source_fn = custom.codegenFn(srcFzMid),
        .inputs = &.{
            inSlot("ig", fix(.float32)), inSlot("iu", fix(.float32)), inSlot("svhg", fix(.float16)), inSlot("svhu", fix(.float16)), inSlot("suhd", fix(.float16)), inSlot("slots", fix(.uint32)),
            inSlot("s_ig", fix(.float16)), inSlot("s_iu", fix(.float16)), inSlot("s_svhg", fix(.float16)), inSlot("s_svhu", fix(.float16)), inSlot("s_suhd", fix(.float16)), inSlot("s_slots", fix(.uint32)),
        },
        .outputs = &.{ outSlot("prepared", fix(.float16)), outSlot("s_yd", fix(.float16)) },
        .params = &[_]ParamSpec{ fcInt("IDIM"), fcInt("NSPLIT"), fcInt("S_ODIM"), fcInt("S_CLAMP_LIMIT") } ++ &fz_grid_params,
    },
    .{
        .name = "sushi_exl3_fz_mid_snax",
        .source_fn = custom.codegenFn(srcFzMidSnax),
        .inputs = &.{
            inSlot("ig", fix(.float32)), inSlot("iu", fix(.float32)), inSlot("svhg", fix(.float16)), inSlot("svhu", fix(.float16)), inSlot("suhd", fix(.float16)), inSlot("slots", fix(.uint32)),
            inSlot("s_ig", fix(.float32)), inSlot("s_iu", fix(.float32)), inSlot("s_svhg", fix(.float16)), inSlot("s_svhu", fix(.float16)), inSlot("s_suhd", fix(.float16)), inSlot("s_slots", fix(.uint32)),
        },
        .outputs = &.{ outSlot("prepared", fix(.float16)), outSlot("s_prepared", fix(.float16)) },
        .params = &[_]ParamSpec{ fcInt("IDIM"), fcInt("NSPLIT"), fcInt("S_IDIM"), fcInt("S_NSPLIT") } ++ &fz_grid_params,
    },
    // 2. DSv41 fused pair & down (group 0 & 2, lane/one/shared_rows, A={scalar,pair1}, B={scalar,pair1})
    fzPairSpec(0, .lane, false, false),        fzPairSpec(0, .lane, true, false),
    fzPairSpec(0, .one, false, false),         fzPairSpec(0, .one, true, false),
    fzPairSpec(0, .shared_rows, false, false), fzPairSpec(0, .shared_rows, true, false),
    fzPairSpec(2, .lane, false, false),        fzPairSpec(2, .lane, true, false),
    fzPairSpec(2, .shared_rows, false, false), fzPairSpec(2, .shared_rows, true, false),
    fzPairSpec(0, .one, true, true),           fzPairSpec(0, .shared_rows, true, true),
    fzPairSpec(2, .shared_rows, true, true),   fzPairSpec(0, .lane, true, true),
    fzDownSpec(0, .lane, false, false),        fzDownSpec(0, .lane, true, false),
    fzDownSpec(0, .one, false, false),         fzDownSpec(0, .one, true, false),
    fzDownSpec(0, .shared_rows, false, false), fzDownSpec(0, .shared_rows, true, false),
    fzDownSpec(2, .lane, false, false),        fzDownSpec(2, .lane, true, false),
    fzDownSpec(2, .shared_rows, false, false), fzDownSpec(2, .shared_rows, true, false),
    fzDownSpec(0, .one, true, true),           fzDownSpec(0, .shared_rows, true, true),
    fzDownSpec(2, .shared_rows, true, true),   fzDownSpec(0, .lane, true, true),
    // 3. Qwen4 affine folded (rg_fold & gated_fold)
    foldAffineSpec(true, false, false, false),  foldAffineSpec(true, false, false, true),
    foldAffineSpec(true, false, true, false),   foldAffineSpec(true, false, true, true),
    foldAffineSpec(true, true, false, false),   foldAffineSpec(true, true, false, true),
    foldAffineSpec(false, false, false, false), foldAffineSpec(false, false, false, true),
    foldAffineSpec(false, false, true, false),  foldAffineSpec(false, false, true, true),
    foldAffineSpec(false, true, false, false),  foldAffineSpec(false, true, false, true),
};
};

// ── Unit 5: `exl3_gemm_nax` Spec Tables ──

const dense_nax_p_sc = nhwParams(&served_scalar_rates, &.{ fcInt("IDIM"), fcInt("ODIM") });
const dense_nax_p_p1 = nhwParams(&pair1_rates, &.{ fcInt("IDIM"), fcInt("ODIM") });

const gemm_nax_specs =
    pairCodecSpecs("sushi_exl3_k4_gemm_nax", true, hdr_scalar_nax, hdr_pair1_nax, srcClean(kernels.GEMM_NAX_SOURCE), &gemm_ins, &gemm_outs, &gemm_sorted_p_sc, &gemm_sorted_p_p1) ++
    pairCodecSpecs("sushi_exl3_k4_gemm_nax_tm", true, hdr_scalar_nax, hdr_pair1_nax, srcClean(kernels.GEMM_NAX_TM_SOURCE), &gemm_ins, &gemm_outs, &gemm_sorted_p_sc, &gemm_sorted_p_p1) ++
    pairCodecSpecs("sushi_exl3_k4_gemm_nax_ref", true, hdr_scalar_nax, hdr_pair1_nax, srcClean(kernels.GEMM_NAX_REFERENCE_SOURCE), &gemm_ins, &gemm_outs, &gemm_sorted_p_sc, &gemm_sorted_p_p1) ++
    pairCodecSpecs("sushi_exl3_trunk_dense_nax", true, hdr_scalar_nax, hdr_pair1_nax, srcClean(trunk_gemm.DENSE_SOURCE), &.{inSlot("trellis", fix(.uint16))}, &.{outSlot("y", fix(.float16))}, &dense_nax_p_sc, &dense_nax_p_p1) ++
    pairCodecSpecs("sushi_prefill_grid_transpose", true, hdr_scalar_nax, hdr_pair1_nax, srcClean(prefill_grid.SOURCE), &gemm_ins, &gemm_outs, &gemm_sorted_p_sc, &gemm_sorted_p_p1);

// ── Unit 6: `exl3_rows_nax` Spec Tables ──

fn rowsNaxFilter(vals: []const CtVal) bool {
    // Variant order: [OutT, Y, ROW_CAP, KPH, UNR, NHW]
    const out_t = vals[0].dtype;
    const y_t = vals[1].dtype;
    const cap = vals[2].int;
    const kph = vals[3].int;
    const unr = vals[4].int;
    if (y_t != .float32 and y_t != out_t) return false;
    if (unr == 2 and kph != 1) return false;
    if (kph == 1 and cap != 16) return false;
    if (kph == 2 and cap != 16) return false;
    if (kph == 4 and cap != 8 and cap != 16) return false;
    return true;
}

fn rowsNaxDynFilter(vals: []const CtVal) bool {
    // Variant order: [OutT, Y, ROW_CAP, KPH, UNR, NHW]
    const out_t = vals[0].dtype;
    const y_t = vals[1].dtype;
    const cap = vals[2].int;
    const kph = vals[3].int;
    const unr = vals[4].int;
    if (y_t != .float32 and y_t != out_t) return false;
    if ((kph == 1 and unr == 2) or (kph == 2 and unr == 1) or (kph == 4 and unr == 1)) {
        if (kph < 4) return cap == 16;
        return cap == 8 or cap == 16;
    }
    return false;
}

fn rowsNaxPendFilter(vals: []const CtVal) bool {
    // Variant order: [OutT, XT, Y, ROW_CAP, KPH, UNR, NHW]
    const out_t = vals[0].dtype;
    const xt = vals[1].dtype;
    const y_t = vals[2].dtype;
    const cap = vals[3].int;
    const kph = vals[4].int;
    const unr = vals[5].int;
    if (out_t != xt) return false;
    if (y_t != .float32 and y_t != out_t) return false;
    if ((kph == 1 and unr == 2) or (kph == 2 and unr == 1) or (kph == 4 and unr == 1)) {
        if (kph < 4) return cap == 16;
        return cap == 8 or cap == 16;
    }
    return false;
}

fn rowsNaxThruFilter(vals: []const CtVal) bool {
    // Variant order: [OutT, (FIN|W)?, ROW_CAP, KPH, UNR, NHW]
    const n = vals.len;
    const cap = vals[n - 4].int;
    const kph = vals[n - 3].int;
    const unr = vals[n - 2].int;
    if (unr == 2 and kph != 1) return false;
    if (kph < 4) return cap == 16;
    return cap == 8 or cap == 16;
}

fn rowsNaxCountedFilter(vals: []const CtVal) bool {
    const n = vals.len;
    const kph = vals[n - 3].int;
    const unr = vals[n - 2].int;
    return (kph == 1 and unr == 2) or (kph == 2 and unr == 1) or (kph == 4 and unr == 1);
}

const rows_nax_specs = [_]KernelSpec{
    .{
        .name = "sushi_exl3_rows_nax_scalar",
        .nax = true,
        .header_fn = custom.codegenFn(exl3HeaderFn),
        .source_arg = hdr_scalar_rows_nax,
        .source_fn = srcRows(project_rows.ROWS_NAX_SOURCE, true, false),
        .inputs = &rows_ins,
        .outputs = &.{outSlot("y", par("Y"))},
        .params = &.{
            .{ .name = "OutT", .kind = .ct_dtype, .dtypes = &floats },
            .{ .name = "Y", .kind = .io_dtype, .dtypes = &floats },
            .{ .name = "ROW_CAP", .kind = .ct_int, .ints = &.{ 8, 16 } },
            .{ .name = "KPH", .kind = .ct_int, .ints = &.{ 1, 2, 4 } },
            .{ .name = "UNR", .kind = .ct_int, .ints = &.{ 1, 2 } },
            .{ .name = "NHW", .kind = .ct_int, .ints = &trunk_scalar_rates },
            fcInt("IDIM"), fcInt("ODIM"), fcInt("ROWS"), fcInt("GROUPS"), ign("TGRP"), fcInt("NSPLIT"), fcInt("CHUNK"), fcInt("FOLD"), fcInt("R32"), fcInt("SW"), fcInt("SB1"), fcInt("SB2"), fcInt("SB3"),
        },
        .filter = rowsNaxFilter,
    },
    .{
        .name = "sushi_exl3_rows_nax_pair1",
        .nax = true,
        .header_fn = custom.codegenFn(exl3HeaderFn),
        .source_arg = hdr_pair1_rows_nax,
        .source_fn = srcRows(project_rows.ROWS_NAX_SOURCE, true, false),
        .inputs = &rows_ins,
        .outputs = &.{outSlot("y", par("Y"))},
        .params = &.{
            .{ .name = "OutT", .kind = .ct_dtype, .dtypes = &bf_f16 },
            .{ .name = "Y", .kind = .io_dtype, .dtypes = &floats },
            .{ .name = "ROW_CAP", .kind = .ct_int, .ints = &.{16} },
            .{ .name = "KPH", .kind = .ct_int, .ints = &.{1} },
            .{ .name = "UNR", .kind = .ct_int, .ints = &.{2} },
            .{ .name = "NHW", .kind = .ct_int, .ints = &pair1_rates },
            fcInt("IDIM"), fcInt("ODIM"), fcInt("ROWS"), fcInt("GROUPS"), ign("TGRP"), fcInt("NSPLIT"), fcInt("CHUNK"), fcInt("FOLD"), fcInt("R32"), fcInt("SW"), fcInt("SB1"), fcInt("SB2"), fcInt("SB3"),
        },
        .filter = rowsNaxFilter,
    },
    .{
        .name = "sushi_exl3_rows_nax_rope_scalar",
        .nax = true,
        .header_fn = custom.codegenFn(exl3HeaderFn),
        .source_arg = hdr_scalar_rows_nax,
        .source_fn = srcRows(project_rows.ROWS_NAX_ROPE_SOURCE, true, false),
        .inputs = &rows_rope_ins,
        .outputs = &.{outSlot("y", par("OutT"))},
        .params = &.{
            .{ .name = "OutT", .kind = .ct_dtype, .dtypes = &floats },
            .{ .name = "ROW_CAP", .kind = .ct_int, .ints = &.{ 8, 16 } },
            .{ .name = "KPH", .kind = .ct_int, .ints = &.{ 1, 2, 4 } },
            .{ .name = "UNR", .kind = .ct_int, .ints = &.{ 1, 2 } },
            .{ .name = "NHW", .kind = .ct_int, .ints = &trunk_scalar_rates },
            fcInt("IDIM"), fcInt("ODIM"), fcInt("ROWS"), fcInt("GROUPS"), ign("TGRP"), fcInt("NSPLIT"), fcInt("CHUNK"), fcInt("FOLD"), fcInt("R32"), fcInt("SW"), fcInt("SB1"), fcInt("SB2"), fcInt("SB3"), fcInt("HD"), fcInt("RD"),
        },
        .filter = rowsNaxThruFilter,
    },
    .{
        .name = "sushi_exl3_rows_nax_rope_pair1",
        .nax = true,
        .header_fn = custom.codegenFn(exl3HeaderFn),
        .source_arg = hdr_pair1_rows_nax,
        .source_fn = srcRows(project_rows.ROWS_NAX_ROPE_SOURCE, true, false),
        .inputs = &rows_rope_ins,
        .outputs = &.{outSlot("y", par("OutT"))},
        .params = &.{
            .{ .name = "OutT", .kind = .ct_dtype, .dtypes = &bf_f16 },
            .{ .name = "ROW_CAP", .kind = .ct_int, .ints = &.{16} },
            .{ .name = "KPH", .kind = .ct_int, .ints = &.{1} },
            .{ .name = "UNR", .kind = .ct_int, .ints = &.{2} },
            .{ .name = "NHW", .kind = .ct_int, .ints = &pair1_rates },
            fcInt("IDIM"), fcInt("ODIM"), fcInt("ROWS"), fcInt("GROUPS"), ign("TGRP"), fcInt("NSPLIT"), fcInt("CHUNK"), fcInt("FOLD"), fcInt("R32"), fcInt("SW"), fcInt("SB1"), fcInt("SB2"), fcInt("SB3"), fcInt("HD"), fcInt("RD"),
        },
        .filter = rowsNaxThruFilter,
    },
    .{
        .name = "sushi_exl3_rows_nax_dyn_scalar",
        .nax = true,
        .header_fn = custom.codegenFn(exl3HeaderFn),
        .source_arg = hdr_scalar_rows_nax,
        .source_fn = srcRows(project_rows.ROWS_NAX_DYN_SOURCE, true, false),
        .inputs = &rows_ins,
        .outputs = &.{outSlot("y", par("Y"))},
        .params = &.{
            .{ .name = "OutT", .kind = .ct_dtype, .dtypes = &bf_f16 },
            .{ .name = "Y", .kind = .io_dtype, .dtypes = &floats },
            .{ .name = "ROW_CAP", .kind = .ct_int, .ints = &.{ 8, 16 } },
            .{ .name = "KPH", .kind = .ct_int, .ints = &.{ 1, 2, 4 } },
            .{ .name = "UNR", .kind = .ct_int, .ints = &.{ 1, 2 } },
            .{ .name = "NHW", .kind = .ct_int, .ints = &.{ 32, 48, 64, 80, 96, 112, 128 } },
            fcInt("IDIM"), fcInt("ODIM"), fcInt("ROWS"), fcInt("GROUPS"), ign("TGRP"), fcInt("NSPLIT"), fcInt("CHUNK"), fcInt("FOLD"), fcInt("R32"), fcInt("SW"), fcInt("SB1"), fcInt("SB2"), fcInt("SB3"),
        },
        .filter = rowsNaxDynFilter,
    },
    .{
        .name = "sushi_exl3_rows_nax_pend_scalar",
        .nax = true,
        .header_fn = custom.codegenFn(exl3HeaderFn),
        .source_arg = hdr_scalar_rows_nax,
        .source_fn = srcRows(project_rows.ROWS_NAX_PEND_SOURCE, true, false),
        .inputs = &rows_pend_ins,
        .outputs = &.{outSlot("y", par("Y"))},
        .params = &.{
            .{ .name = "OutT", .kind = .ct_dtype, .dtypes = &floats },
            .{ .name = "XT", .kind = .ct_dtype, .dtypes = &floats },
            .{ .name = "Y", .kind = .io_dtype, .dtypes = &floats },
            .{ .name = "ROW_CAP", .kind = .ct_int, .ints = &.{ 8, 16 } },
            .{ .name = "KPH", .kind = .ct_int, .ints = &.{ 1, 2, 4 } },
            .{ .name = "UNR", .kind = .ct_int, .ints = &.{ 1, 2 } },
            .{ .name = "NHW", .kind = .ct_int, .ints = &.{ 32, 48, 64, 80, 96, 112, 128 } },
            fcInt("IDIM"), fcInt("ODIM"), fcInt("ROWS"), fcInt("GROUPS"), ign("TGRP"), fcInt("NSPLIT"), fcInt("CHUNK"), fcInt("FOLD"), fcInt("R32"), fcInt("SW"), fcInt("SB1"), fcInt("SB2"), fcInt("SB3"), fcInt("XSPLIT"), fcInt("XODIM"),
        },
        .filter = rowsNaxPendFilter,
    },
    .{
        .name = "sushi_exl3_rows_nax_thru_scalar",
        .nax = true,
        .header_fn = custom.codegenFn(exl3HeaderFn),
        .source_arg = hdr_scalar_rows_nax,
        .source_fn = srcRows(project_rows.ROWS_NAX_THRU_SOURCE, true, false),
        .inputs = &rows_thru_ins,
        .outputs = &.{outSlot("y", fix(.float32))},
        .params = &.{
            .{ .name = "OutT", .kind = .ct_dtype, .dtypes = &.{.bfloat16} },
            .{ .name = "X", .kind = .io_dtype, .dtypes = &bf_f32 },
            .{ .name = "ROW_CAP", .kind = .ct_int, .ints = &.{ 8, 16 } },
            .{ .name = "KPH", .kind = .ct_int, .ints = &.{ 1, 2, 4 } },
            .{ .name = "UNR", .kind = .ct_int, .ints = &.{ 1, 2 } },
            .{ .name = "NHW", .kind = .ct_int, .ints = &trunk_scalar_rates },
            fcInt("IDIM"), fcInt("ODIM"), fcInt("ROWS"), fcInt("GROUPS"), ign("TGRP"), fcInt("NSPLIT"), fcInt("CHUNK"), fcInt("FOLD"), fcInt("R32"), fcInt("SW"), fcInt("SB1"), fcInt("SB2"), fcInt("SB3"),
        },
        .filter = rowsNaxThruFilter,
    },
    .{
        .name = "sushi_exl3_rows_nax_thru_counted_scalar",
        .nax = true,
        .header_fn = custom.codegenFn(exl3HeaderFn),
        .source_arg = hdr_scalar_rows_nax_cnt,
        .source_fn = srcRows(project_rows.ROWS_NAX_THRU_COUNTED_SOURCE, true, true),
        .inputs = &rows_cnt_ins,
        .outputs = &.{ outSlot("y", fix(.float32)), outSlot("fin", par("FIN")), outSlot("ctr", fix(.uint32)) },
        .params = &.{
            .{ .name = "OutT", .kind = .ct_dtype, .dtypes = &.{.bfloat16} },
            .{ .name = "FIN", .kind = .io_dtype, .dtypes = &bf_f32 },
            .{ .name = "ROW_CAP", .kind = .ct_int, .ints = &.{ 4, 8 } },
            .{ .name = "KPH", .kind = .ct_int, .ints = &.{ 1, 2, 4 } },
            .{ .name = "UNR", .kind = .ct_int, .ints = &.{ 1, 2 } },
            .{ .name = "NHW", .kind = .ct_int, .ints = &trunk_scalar_rates },
            fcInt("IDIM"), fcInt("ODIM"), fcInt("ROWS"), fcInt("GROUPS"), ign("TGRP"), fcInt("NSPLIT"), fcInt("CHUNK"), fcInt("FOLD"), fcInt("R32"), fcInt("SW"), fcInt("SB1"), fcInt("SB2"), fcInt("SB3"),
        },
        .filter = rowsNaxCountedFilter,
    },
    .{
        .name = "sushi_exl3_rows_nax_thru_counted_rms_scalar",
        .nax = true,
        .header_fn = custom.codegenFn(exl3HeaderFn),
        .source_arg = hdr_scalar_rows_nax_cnt,
        .source_fn = srcRows(project_rows.ROWS_NAX_THRU_COUNTED_RMS_SOURCE, true, true),
        .inputs = &rows_cnt_rms_ins,
        .outputs = &.{ outSlot("y", fix(.float32)), outSlot("fin", fix(.float32)), outSlot("ctr", fix(.uint32)) },
        .params = &.{
            .{ .name = "OutT", .kind = .ct_dtype, .dtypes = &.{.bfloat16} },
            .{ .name = "W", .kind = .io_dtype, .dtypes = &bf_f32 },
            .{ .name = "ROW_CAP", .kind = .ct_int, .ints = &.{ 1, 4, 8 } },
            .{ .name = "KPH", .kind = .ct_int, .ints = &.{ 1, 2, 4 } },
            .{ .name = "UNR", .kind = .ct_int, .ints = &.{ 1, 2 } },
            .{ .name = "NHW", .kind = .ct_int, .ints = &trunk_scalar_rates },
            fcInt("IDIM"), fcInt("ODIM"), fcInt("ROWS"), fcInt("GROUPS"), ign("TGRP"), fcInt("NSPLIT"), fcInt("CHUNK"), fcInt("FOLD"), fcInt("R32"), fcInt("SW"), fcInt("SB1"), fcInt("SB2"), fcInt("SB3"),
        },
        .filter = rowsNaxCountedFilter,
    },
    .{
        .name = "sushi_exl3_rows_nax_thru_counted_pair_scalar",
        .nax = true,
        .header_fn = custom.codegenFn(exl3HeaderFn),
        .source_arg = hdr_scalar_rows_nax_cnt,
        .source_fn = srcRows(project_rows.ROWS_NAX_THRU_COUNTED_PAIR_SOURCE, true, true),
        .inputs = &rows_cnt_pair_ins,
        .outputs = &.{ outSlot("y", fix(.float32)), outSlot("fin", fix(.float32)), outSlot("ctr", fix(.uint32)) },
        .params = &.{
            .{ .name = "OutT", .kind = .ct_dtype, .dtypes = &.{.bfloat16} },
            .{ .name = "ROW_CAP", .kind = .ct_int, .ints = &.{ 4, 8 } },
            .{ .name = "KPH", .kind = .ct_int, .ints = &.{ 1, 2, 4 } },
            .{ .name = "UNR", .kind = .ct_int, .ints = &.{ 1, 2 } },
            .{ .name = "NHW", .kind = .ct_int, .ints = &trunk_scalar_rates },
            fcInt("IDIM"), fcInt("ODIM"), fcInt("ROWS"), fcInt("GROUPS"), ign("TGRP"), fcInt("NSPLIT"), fcInt("CHUNK"), fcInt("FOLD"), fcInt("R32"), fcInt("SW"), fcInt("SB1"), fcInt("SB2"), fcInt("SB3"),
        },
        .filter = rowsNaxCountedFilter,
    },
};

// ── Unit 7: `exl3_fz_nax` Spec Tables ──

fn foldTrellisSpec(comptime mixed: bool, comptime is_down: bool, comptime with_logit: bool, comptime a_pair1: bool, comptime b_pair1: bool) KernelSpec {
    const nm = (if (mixed) "sushi_rg_fold_" else "sushi_gated_fold_") ++
        (if (is_down) "down" else "pair") ++
        (if (with_logit) "_logit" else "") ++
        "_tr" ++
        (if (a_pair1) "_ap1" else "_asc") ++
        (if (b_pair1) "_bp1" else "_bsc");

    const a_ins_pair_rg = [_]InputSlot{ inSlot("x", fix(.bfloat16)), inSlot("suhg", fix(.float16)), inSlot("suhu", fix(.float16)), inSlot("tg", fix(.uint16)), inSlot("tu", fix(.uint16)), inSlot("slots", fix(.uint32)), inSlot("meta", fix(.uint32)) };
    const a_ins_pair_gt = [_]InputSlot{ inSlot("x", fix(.bfloat16)), inSlot("suhg", fix(.float16)), inSlot("suhu", fix(.float16)), inSlot("tg", fix(.uint16)), inSlot("tu", fix(.uint16)), inSlot("slots", fix(.uint32)) };
    const a_ins_down_rg = [_]InputSlot{ inSlot("ig", fix(.float32)), inSlot("iu", fix(.float32)), inSlot("trellis", fix(.uint16)), inSlot("svhg", fix(.float16)), inSlot("svhu", fix(.float16)), inSlot("suhd", fix(.float16)), inSlot("slots", fix(.uint32)), inSlot("meta", fix(.uint32)) };
    const a_ins_down_gt = [_]InputSlot{ inSlot("ig", fix(.float32)), inSlot("iu", fix(.float32)), inSlot("trellis", fix(.uint16)), inSlot("svhg", fix(.float16)), inSlot("svhu", fix(.float16)), inSlot("suhd", fix(.float16)), inSlot("slots", fix(.uint32)) };

    const b_ins_gu = [_]InputSlot{ inSlot("s_x", fix(.bfloat16)), inSlot("s_trellis", fix(.uint16)), inSlot("s_suh", fix(.float16)), inSlot("s_svh", fix(.float16)) };
    const b_ins_logit = [_]InputSlot{ inSlot("s_lw", fix(.uint32)), inSlot("s_ls", fix(.bfloat16)), inSlot("s_lb", fix(.bfloat16)) };
    const b_ins_dn = [_]InputSlot{ inSlot("s_x", fix(.bfloat16)), inSlot("s_trellis", fix(.uint16)), inSlot("s_suh", fix(.float16)), inSlot("s_svh", fix(.float16)), inSlot("s_g1", fix(.float32)), inSlot("s_g1svh", fix(.float16)), inSlot("s_sigmoid", fix(.bfloat16)) };

    const a_p = if (is_down)
        (if (mixed) &[_]ParamSpec{ .{ .name = "OTPT", .kind = .ct_int, .ints = &.{ 1, 2 } }, fcInt("NSPLIT"), fcInt("IDIM"), fcInt("ODIM"), fcInt("FUNNEL") } else &[_]ParamSpec{ .{ .name = "OTPT", .kind = .ct_int, .ints = &.{ 1, 2 } }, fcInt("IDIM"), fcInt("ODIM"), fcInt("NSPLIT"), fcInt("NHW"), fcInt("FUNNEL") })
    else
        (if (mixed) &[_]ParamSpec{ .{ .name = "OTPT", .kind = .ct_int, .ints = &.{ 1, 2 } }, fcInt("NSPLIT"), fcInt("TOPK"), fcInt("IDIM"), fcInt("ODIM"), fcInt("FUNNEL") } else &[_]ParamSpec{ .{ .name = "OTPT", .kind = .ct_int, .ints = &.{ 1, 2 } }, fcInt("IDIM"), fcInt("ODIM"), fcInt("NSPLIT"), fcInt("NHW"), fcInt("TOPK"), fcInt("FUNNEL") });

    const b_base = [_]ParamSpec{
        fcInt("S_IDIM"), fcInt("S_ODIM"), fcInt("S_ROWS"), fcInt("S_GROUPS"), ign("S_TGRP"), ign("S_KPH"), fcInt("S_NSPLIT"), fcInt("S_CHUNK"), fcInt("S_FOLD"), fcInt("S_NHW"), fcInt("S_SB1"), fcInt("S_SB2"), fcInt("S_SB3"), ign("S_UNR"), fcInt("S_R32"), fcInt("S_SW"),
    };
    const b_extra = if (is_down)
        &[_]ParamSpec{ fcInt("S_GNSPLIT"), fcInt("S_GODIM"), fcInt("S_GWIDTH") }
    else if (with_logit)
        &[_]ParamSpec{ fcInt("S_K"), fcInt("S_LGS"), fcInt("S_TILES") }
    else
        &[_]ParamSpec{};

    return .{
        .name = nm,
        .nax = true,
        .header_fn = hdrFoldTrellis(mixed, is_down, with_logit, a_pair1, b_pair1),
        .source_fn = srcFoldTrellis(mixed, is_down, with_logit, a_pair1, b_pair1),
        .inputs = if (is_down)
            (if (mixed) &a_ins_down_rg else &a_ins_down_gt) ++ &b_ins_dn
        else if (with_logit)
            (if (mixed) &a_ins_pair_rg else &a_ins_pair_gt) ++ &b_ins_gu ++ &b_ins_logit
        else
            (if (mixed) &a_ins_pair_rg else &a_ins_pair_gt) ++ &b_ins_gu,
        .outputs = if (is_down)
            &.{ outSlot("y", fix(.float16)), outSlot("s_y", fix(.float32)) }
        else if (with_logit)
            &.{ outSlot("yg", fix(.float32)), outSlot("yu", fix(.float32)), outSlot("s_y", fix(.float32)), outSlot("s_gl", fix(.bfloat16)) }
        else
            &.{ outSlot("yg", fix(.float32)), outSlot("yu", fix(.float32)), outSlot("s_y", fix(.float32)) },
        .params = a_p ++ &b_base ++ b_extra ++ &fz_grid_params,
    };
}

fn fzPairSnaxSpec(comptime group: u32, comptime a_pair1: bool, comptime b_pair1: bool) KernelSpec {
    const g_s = if (group == 0) "" else std.fmt.comptimePrint("_g{d}", .{group});
    const ab_s = (if (a_pair1) "_ap1" else "_asc") ++ (if (b_pair1) "_bp1" else "_bsc");
    const a_p = if (group == 0)
        &[_]ParamSpec{ .{ .name = "OTPT", .kind = .ct_int, .ints = &.{ 1, 2 } }, fcInt("NSPLIT"), fcInt("TOPK"), fcInt("IDIM"), fcInt("ODIM"), fcInt("FUNNEL") }
    else
        &[_]ParamSpec{ .{ .name = "GROUP", .kind = .ct_int, .ints = &.{@as(i32, @intCast(group))} }, .{ .name = "OTPT", .kind = .ct_int, .ints = &.{ 1, 2 } }, fcInt("NSPLIT"), fcInt("TOPK"), fcInt("IDIM"), fcInt("ODIM"), fcInt("FUNNEL"), fcInt("NSLOTS") };
    const b_p = [_]ParamSpec{
        fcInt("S_IDIM"), fcInt("S_ODIM"), fcInt("S_ROWS"), fcInt("S_GROUPS"), ign("S_TGRP"), ign("S_KPH"), fcInt("S_NSPLIT"), fcInt("S_CHUNK"), fcInt("S_FOLD"), fcInt("S_NHW"), fcInt("S_SB1"), fcInt("S_SB2"), fcInt("S_SB3"), fcInt("S_UNR"), fcInt("S_R32"), fcInt("S_SW"),
    };
    return .{
        .name = "sushi_exl3_fz_pair" ++ g_s ++ "_snax" ++ ab_s,
        .nax = true,
        .header_fn = hdrFzSnaxStage(a_pair1, b_pair1),
        .source_fn = srcFzSnaxStage(false, group, a_pair1, b_pair1),
        .inputs = &.{
            inSlot("prep", fix(.float16)), inSlot("tg", fix(.uint16)), inSlot("tu", fix(.uint16)), inSlot("slots", fix(.uint32)), inSlot("meta", fix(.uint32)),
            inSlot("s_x", fix(.bfloat16)), inSlot("s_tg", fix(.uint16)), inSlot("s_tu", fix(.uint16)), inSlot("s_suhg", fix(.float16)), inSlot("s_suhu", fix(.float16)),
        },
        .outputs = &.{ outSlot("yg", fix(.float32)), outSlot("yu", fix(.float32)), outSlot("s_yg", fix(.float32)), outSlot("s_yu", fix(.float32)) },
        .params = a_p ++ &b_p ++ &fz_grid_params,
    };
}

fn fzDownSnaxSpec(comptime group: u32, comptime a_pair1: bool, comptime b_pair1: bool) KernelSpec {
    const g_s = if (group == 0) "" else std.fmt.comptimePrint("_g{d}", .{group});
    const ab_s = (if (a_pair1) "_ap1" else "_asc") ++ (if (b_pair1) "_bp1" else "_bsc");
    const a_p = if (group == 0)
        &[_]ParamSpec{ .{ .name = "OTPT", .kind = .ct_int, .ints = &.{ 1, 2 } }, fcInt("NSPLIT"), fcInt("IDIM"), fcInt("ODIM"), fcInt("FUNNEL") }
    else
        &[_]ParamSpec{ .{ .name = "GROUP", .kind = .ct_int, .ints = &.{@as(i32, @intCast(group))} }, .{ .name = "OTPT", .kind = .ct_int, .ints = &.{ 1, 2 } }, fcInt("NSPLIT"), fcInt("IDIM"), fcInt("ODIM"), fcInt("FUNNEL"), fcInt("NSLOTS") };
    const b_p = [_]ParamSpec{
        fcInt("S_IDIM"), fcInt("S_ODIM"), fcInt("S_ROWS"), ign("S_GROUPS"), ign("S_TGRP"), ign("S_KPH"), fcInt("S_NSPLIT"), ign("S_CHUNK"), ign("S_FOLD"), fcInt("S_NHW"), fcInt("S_SB1"), fcInt("S_SB2"), fcInt("S_SB3"), fcInt("S_UNR"), fcInt("S_R32"), fcInt("S_SW"),
    };
    return .{
        .name = "sushi_exl3_fz_down" ++ g_s ++ "_snax" ++ ab_s,
        .nax = true,
        .header_fn = hdrFzSnaxStage(a_pair1, b_pair1),
        .source_fn = srcFzSnaxStage(true, group, a_pair1, b_pair1),
        .inputs = &.{
            inSlot("middle", fix(.float16)), inSlot("trellis", fix(.uint16)), inSlot("slots", fix(.uint32)), inSlot("meta", fix(.uint32)),
            inSlot("s_x", fix(.float16)), inSlot("s_trellis", fix(.uint16)), inSlot("s_svh", fix(.float16)),
        },
        .outputs = &.{ outSlot("y", fix(.float16)), outSlot("s_y", fix(.float32)) },
        .params = a_p ++ &b_p ++ &fz_grid_params,
    };
}

const fz_nax_specs = blk: {
    @setEvalBranchQuota(100000);
    break :blk [_]KernelSpec{
    // DSv41 fused pair & down with multi-row NAX shared expert (group 0 & 2, A={scalar,pair1}, B={scalar,pair1})
    fzPairSnaxSpec(0, false, false), fzPairSnaxSpec(0, true, false), fzPairSnaxSpec(0, false, true), fzPairSnaxSpec(0, true, true),
    fzPairSnaxSpec(2, false, false), fzPairSnaxSpec(2, true, false), fzPairSnaxSpec(2, false, true), fzPairSnaxSpec(2, true, true),
    fzDownSnaxSpec(0, false, false), fzDownSnaxSpec(0, true, false), fzDownSnaxSpec(0, false, true), fzDownSnaxSpec(0, true, true),
    fzDownSnaxSpec(2, false, false), fzDownSnaxSpec(2, true, false), fzDownSnaxSpec(2, false, true), fzDownSnaxSpec(2, true, true),
    // rg_fold with trellis shared expert (B scalar = spec1/mcg/alpha, B pair1 = pair1)
    foldTrellisSpec(true, false, false, false, false),  foldTrellisSpec(true, false, false, true, false),
    foldTrellisSpec(true, false, true, false, false),   foldTrellisSpec(true, false, true, true, false),
    foldTrellisSpec(true, true, false, false, false),   foldTrellisSpec(true, true, false, true, false),
    foldTrellisSpec(true, false, false, true, true),    foldTrellisSpec(true, false, true, true, true),
    foldTrellisSpec(true, true, false, true, true),
    // gated_fold with trellis shared expert
    foldTrellisSpec(false, false, false, false, false), foldTrellisSpec(false, false, false, true, false),
    foldTrellisSpec(false, false, true, false, false),  foldTrellisSpec(false, false, true, true, false),
    foldTrellisSpec(false, true, false, false, false),  foldTrellisSpec(false, true, false, true, false),
    foldTrellisSpec(false, false, false, true, true),   foldTrellisSpec(false, false, true, true, true),
    foldTrellisSpec(false, true, false, true, true),
};
};

// ── Runtime Name Parsing & Resolution ──

const CodecMeta = struct {
    cb: format.Codebook = .mul1,
    win: format.Window = .w16,
    digest: ?[]const u8 = null,
    fold_out: bool = true,
};

fn parseWindowBits(s: []const u8) ?format.Window {
    if (s.len == 0) return null;
    const v = std.fmt.parseInt(u8, s, 10) catch return null;
    return switch (v) {
        8 => .w8,
        9 => .w9,
        10 => .w10,
        11 => .w11,
        12 => .w12,
        13 => .w13,
        14 => .w14,
        15 => .w15,
        16 => .w16,
        else => null,
    };
}

fn parseCbName(s: []const u8) ?format.Codebook {
    if (std.mem.eql(u8, s, "mul1")) return .mul1;
    if (std.mem.eql(u8, s, "mcg")) return .mcg;
    if (std.mem.eql(u8, s, "alpha")) return .alpha;
    if (std.mem.eql(u8, s, "spec1")) return .spec1;
    if (std.mem.eql(u8, s, "std1")) return .std1;
    if (std.mem.eql(u8, s, "pair1")) return .pair1;
    return null;
}

fn isHex8(s: []const u8) bool {
    if (s.len != 8) return false;
    for (s) |c| if (!std.ascii.isHex(c)) return false;
    return true;
}

fn cbKindInt(cb: format.Codebook) i32 {
    return switch (cb) {
        .mul1 => 0,
        .mcg => 1,
        .alpha, .spec1, .std1 => 2,
        .pair1 => 0,
    };
}

const HashMaskPair = struct { mult: u32, mask: u32 };

fn lookupSingleHashMask(meta: CodecMeta, n: u32) ?HashMaskPair {
    switch (meta.cb) {
        .mul1, .mcg => return .{ .mult = 0, .mask = 0 },
        .pair1 => return if (format.pair1Pair(.{ .n = n }) != null) .{ .mult = 0, .mask = 0 } else null,
        .alpha => {
            const hm = format.Codebook.alpha.hashMask().?;
            return .{ .mult = hm.mult, .mask = hm.mask };
        },
        .spec1, .std1 => {
            const dig = meta.digest orelse return null;
            for (&tables.tables) |*tbl| {
                if (std.mem.eql(u8, tbl.digest, dig)) {
                    const row = tbl.row(.{ .n = n }) orelse return null;
                    return .{ .mult = row[0], .mask = row[1] };
                }
            }
            // Fallback for spec1-test-per-rate (`a2f2fe40`) even in non-test builds if ever queried.
            if (std.mem.eql(u8, dig, "a2f2fe40") and n >= 16 and n <= 128 and n % 2 == 0) {
                const synth_masks = [_]u32{ 0x8FFE8FD0, 0x8FE08FC0, 0x8FF08FE0, 0x8F8E8FD8 };
                return .{ .mult = 0x840C1F37 +% n *% 0x2E0B5, .mask = synth_masks[((n - 16) / 2) % synth_masks.len] };
            }
            return null;
        },
    }
}

fn lookupMixedTableId(meta: CodecMeta) ?i32 {
    switch (meta.cb) {
        .mul1, .mcg, .pair1 => return 0,
        .alpha => return 0,
        .spec1, .std1 => {
            const dig = meta.digest orelse return null;
            if (std.mem.eql(u8, dig, "37f72431")) return 1; // spec1-w15
            if (std.mem.eql(u8, dig, "753ffa03")) return 2; // spec1-w16
            if (std.mem.eql(u8, dig, "c5dba171")) return 3; // std1
            if (std.mem.eql(u8, dig, "a2f2fe40")) return 4; // spec1-test-per-rate
            return null;
        },
    }
}

fn findIntTarg(targs: []const RuntimeTarg, name: []const u8) ?i32 {
    for (targs) |ta| {
        if (std.mem.eql(u8, ta.name, name) and ta.value == .int) return ta.value.int;
    }
    return null;
}

fn appendFcInt(out: *Resolved, slot: u32, val: i32) bool {
    if (out.n_ints >= custom.max_fc_ints) return false;
    out.ints[out.n_ints] = .{ .index = slot, .value = val };
    out.n_ints += 1;
    return true;
}

fn appendFcU32(out: *Resolved, slot: u32, val: u32) bool {
    return appendFcInt(out, slot, @bitCast(val));
}

fn appendFcBool(out: *Resolved, slot: u32, val: bool) bool {
    if (out.n_flags >= custom.max_fc_bools) return false;
    out.flags[out.n_flags] = .{ .index = slot, .value = val };
    out.n_flags += 1;
    return true;
}

fn sortAndFinalize(out: *Resolved) bool {
    std.mem.sortUnstable(custom.Flag, out.flags[0..out.n_flags], {}, struct {
        fn lt(_: void, a: custom.Flag, b: custom.Flag) bool {
            return a.index < b.index;
        }
    }.lt);
    std.mem.sortUnstable(custom.IntConstant, out.ints[0..out.n_ints], {}, struct {
        fn lt(_: void, a: custom.IntConstant, b: custom.IntConstant) bool {
            return a.index < b.index;
        }
    }.lt);
    out.finalizeHash() catch return false;
    return true;
}

/// Strips `[cbSuffix][_unfolded][winSuffix][_t<8hex>]` from a single-rate kernel name.
fn parseSingleRateName(name: []const u8) struct { base: []const u8, meta: CodecMeta } {
    var rest = name;
    var meta: CodecMeta = .{};

    // 1. Optional `_t<8hex>`
    if (rest.len > 10 and std.mem.eql(u8, rest[rest.len - 10 .. rest.len - 8], "_t") and isHex8(rest[rest.len - 8 ..])) {
        meta.digest = rest[rest.len - 8 ..];
        rest = rest[0 .. rest.len - 10];
    }
    // 2. Optional `_w<8..15>`
    if (std.mem.lastIndexOf(u8, rest, "_w")) |idx| {
        if (parseWindowBits(rest[idx + 2 ..])) |w| {
            if (w != .w16) {
                meta.win = w;
                rest = rest[0..idx];
            }
        }
    }
    // 3. Optional `_unfolded`
    if (std.mem.endsWith(u8, rest, "_unfolded")) {
        meta.fold_out = false;
        rest = rest[0 .. rest.len - "_unfolded".len];
    }
    // 4. Optional `_mcg` / `_alpha` / `_spec1` / `_std1` / `_pair1`
    const suffixes = [_]struct { s: []const u8, cb: format.Codebook }{
        .{ .s = "_mcg", .cb = .mcg },
        .{ .s = "_alpha", .cb = .alpha },
        .{ .s = "_spec1", .cb = .spec1 },
        .{ .s = "_std1", .cb = .std1 },
        .{ .s = "_pair1", .cb = .pair1 },
    };
    for (suffixes) |sf| {
        if (std.mem.endsWith(u8, rest, sf.s)) {
            meta.cb = sf.cb;
            rest = rest[0 .. rest.len - sf.s.len];
            break;
        }
    }
    return .{ .base = rest, .meta = meta };
}

/// Parses `<cb>_w<win>[_t<8hex>]` prefix from a `_`-delimited token slice; returns consumed byte length.
fn parseCbWinPrefix(s: []const u8, out_meta: *CodecMeta) ?usize {
    const sep1 = std.mem.indexOfScalar(u8, s, '_') orelse return null;
    const cb = parseCbName(s[0..sep1]) orelse return null;
    if (sep1 + 2 >= s.len or s[sep1 + 1] != 'w') return null;
    var end = sep1 + 2;
    while (end < s.len and std.ascii.isDigit(s[end])) end += 1;
    const win = parseWindowBits(s[sep1 + 2 .. end]) orelse return null;
    out_meta.* = .{ .cb = cb, .win = win, .fold_out = true };
    if (end + 10 <= s.len and std.mem.eql(u8, s[end .. end + 2], "_t") and isHex8(s[end + 2 .. end + 10])) {
        out_meta.digest = s[end + 2 .. end + 10];
        end += 10;
    }
    return end;
}

const ClassRateMasks = struct { g: u32, u: u32 };

comptime {
    // FC_I46/FC_I47 encode active rate indices in a 32-bit function constant bitmask.
    std.debug.assert(served_scalar_rates.len <= 32);
    std.debug.assert(pair1_rates.len <= 32);
}

fn rateBit(slice: []const i32, n: u32) ?u32 {
    for (slice, 0..) |r, i| if (r == @as(i32, @intCast(n))) return @as(u32, 1) << @intCast(i);
    return null;
}

/// Verifies every `_<g>x<u>` class token in `s` uses rates covered by the mixed-rate switch and returns
/// bitmasks of active gate/down (`g`) and up (`u`) rate indices for PSO dead-code elimination.
fn parseClassRateMasks(s: []const u8, is_pair1: bool) ?ClassRateMasks {
    const rates: []const i32 = if (is_pair1) &pair1_rates else &served_scalar_rates;
    var it = std.mem.splitScalar(u8, s, '_');
    var saw_any = false;
    var masks: ClassRateMasks = .{ .g = 0, .u = 0 };
    while (it.next()) |tok| {
        if (tok.len == 0) continue;
        const x_pos = std.mem.indexOfScalar(u8, tok, 'x') orelse return null;
        const g = std.fmt.parseInt(u32, tok[0..x_pos], 10) catch return null;
        const u = std.fmt.parseInt(u32, tok[x_pos + 1 ..], 10) catch return null;
        masks.g |= rateBit(rates, g) orelse return null;
        if (u != 0) masks.u |= rateBit(rates, u) orelse return null;
        saw_any = true;
    }
    return if (saw_any) masks else null;
}

fn resolveMixedRg(
    name: []const u8,
    inputs: []const RuntimeInput,
    out_dtypes: []const Dtype,
    targs: []const RuntimeTarg,
    out: *Resolved,
) bool {
    // Format: sushi_exl3_rg_<kind>_<cb>_w<win>[_g<g>][_t<d>]_<classes>[_clamp<L>][_tm]
    var rest = name["sushi_exl3_rg_".len..];
    const kind_prefixes = [_]struct { p: []const u8, k: []const u8 }{
        .{ .p = "pair_prepared_", .k = "pair_prepared" },
        .{ .p = "down_prepared_", .k = "down_prepared" },
        .{ .p = "pair_", .k = "pair" },
        .{ .p = "down_", .k = "down" },
    };
    const kind = for (kind_prefixes) |kp| {
        if (std.mem.startsWith(u8, rest, kp.p)) {
            rest = rest[kp.p.len..];
            break kp.k;
        }
    } else return false;

    // Parse cb_w<win>
    const sep1 = std.mem.indexOfScalar(u8, rest, '_') orelse return false;
    const cb = parseCbName(rest[0..sep1]) orelse return false;
    if (sep1 + 2 >= rest.len or rest[sep1 + 1] != 'w') return false;
    var pos = sep1 + 2;
    while (pos < rest.len and std.ascii.isDigit(rest[pos])) pos += 1;
    const win = parseWindowBits(rest[sep1 + 2 .. pos]) orelse return false;
    rest = rest[pos..];

    var group: u32 = 0;
    if (std.mem.startsWith(u8, rest, "_g")) {
        var gend: usize = 2;
        while (gend < rest.len and std.ascii.isDigit(rest[gend])) gend += 1;
        group = std.fmt.parseInt(u32, rest[2..gend], 10) catch return false;
        rest = rest[gend..];
    }

    var meta: CodecMeta = .{ .cb = cb, .win = win, .fold_out = true };
    if (rest.len >= 10 and std.mem.eql(u8, rest[0..2], "_t") and isHex8(rest[2..10])) {
        meta.digest = rest[2..10];
        rest = rest[10..];
    }

    var tile_major = false;
    if (std.mem.endsWith(u8, rest, "_tm")) {
        tile_major = true;
        rest = rest[0 .. rest.len - 3];
    }
    var clamp_limit: i32 = 0;
    if (std.mem.lastIndexOf(u8, rest, "_clamp")) |cidx| {
        clamp_limit = std.fmt.parseInt(i32, rest[cidx + 6 ..], 10) catch return false;
        rest = rest[0..cidx];
    }
    const rate_masks = parseClassRateMasks(rest, cb == .pair1) orelse return false;

    const table_id = lookupMixedTableId(meta) orelse return false;
    var spec_name_buf: [128]u8 = undefined;
    const spec_name = if (group > 0)
        std.fmt.bufPrint(&spec_name_buf, "sushi_exl3_rg_{s}_g{d}_{s}", .{ kind, group, if (cb == .pair1) "pair1" else "scalar" }) catch return false
    else
        std.fmt.bufPrint(&spec_name_buf, "sushi_exl3_rg_{s}_{s}", .{ kind, if (cb == .pair1) "pair1" else "scalar" }) catch return false;

    if (!custom.resolveSpecs(&rg_simd_specs, spec_name, inputs, out_dtypes, targs, out)) return false;
    if (cb != .pair1) {
        if (!appendFcInt(out, fc_cb_kind, cbKindInt(cb))) return false;
        if (!appendFcInt(out, fc_table_id, table_id)) return false;
    }
    if (!appendFcU32(out, fc_win_mask, win.mask())) return false;
    if (!appendFcBool(out, fc_fold_out, true)) return false;
    if (!appendFcBool(out, fc_tile_major, tile_major)) return false;
    if (!appendFcU32(out, fc_rate_mask_g, rate_masks.g)) return false;
    const is_down_kind = std.mem.eql(u8, kind, "down") or std.mem.eql(u8, kind, "down_prepared");
    if (!is_down_kind) {
        if (!appendFcU32(out, fc_rate_mask_u, rate_masks.u)) return false;
    }
    if (std.mem.eql(u8, kind, "down")) {
        if (!appendFcInt(out, fc_clamp_limit, clamp_limit)) return false;
    }
    return sortAndFinalize(out);
}

fn resolveFusedFz(
    name: []const u8,
    inputs: []const RuntimeInput,
    out_dtypes: []const Dtype,
    targs: []const RuntimeTarg,
    out: *Resolved,
) bool {
    // Format: sushi_exl3_fz_<stage>_<cb>_w<win>[_t<d>]_s<scb>_w<sw>_n<sn>[_t<sd>][_<classes>_g<g>][_clamp<L>][_one|_shared_rows][_tm]
    var rest = name["sushi_exl3_fz_".len..];
    if (std.mem.startsWith(u8, rest, "prep_")) {
        const spec_name = if (std.mem.indexOf(u8, rest, "_snax") != null) "sushi_exl3_fz_prep_snax" else "sushi_exl3_fz_prep";
        return custom.resolveSpecs(&fz_simd_specs, spec_name, inputs, out_dtypes, targs, out);
    }
    if (std.mem.startsWith(u8, rest, "mid_")) {
        const cidx = std.mem.lastIndexOf(u8, rest, "_clamp") orelse return false;
        var lim_end = cidx + 6;
        while (lim_end < rest.len and std.ascii.isDigit(rest[lim_end])) lim_end += 1;
        const limit = std.fmt.parseInt(i32, rest[cidx + 6 .. lim_end], 10) catch return false;
        const spec_name = if (std.mem.indexOf(u8, rest, "_snax") != null) "sushi_exl3_fz_mid_snax" else "sushi_exl3_fz_mid";
        if (!custom.resolveSpecs(&fz_simd_specs, spec_name, inputs, out_dtypes, targs, out)) return false;
        if (!appendFcInt(out, fc_clamp_limit, limit)) return false;
        return sortAndFinalize(out);
    }
    const is_down = if (std.mem.startsWith(u8, rest, "pair_")) blk: {
        rest = rest["pair_".len..];
        break :blk false;
    } else if (std.mem.startsWith(u8, rest, "down_")) blk: {
        rest = rest["down_".len..];
        break :blk true;
    } else return false;

    var a_meta: CodecMeta = .{};
    const a_len = parseCbWinPrefix(rest, &a_meta) orelse return false;
    rest = rest[a_len..];
    if (!std.mem.startsWith(u8, rest, "_s")) return false;
    rest = rest[2..];

    var b_meta: CodecMeta = .{};
    // Parse <scb>_w<sw>_n<sn>[_t<sd>]
    const bu1 = std.mem.indexOfScalar(u8, rest, '_') orelse return false;
    b_meta.cb = parseCbName(rest[0..bu1]) orelse return false;
    if (bu1 + 2 >= rest.len or rest[bu1 + 1] != 'w') return false;
    var bp = bu1 + 2;
    while (bp < rest.len and std.ascii.isDigit(rest[bp])) bp += 1;
    b_meta.win = parseWindowBits(rest[bu1 + 2 .. bp]) orelse return false;
    if (bp + 2 >= rest.len or !std.mem.eql(u8, rest[bp .. bp + 2], "_n")) return false;
    bp += 2;
    const n_start = bp;
    while (bp < rest.len and std.ascii.isDigit(rest[bp])) bp += 1;
    const sn = std.fmt.parseInt(u32, rest[n_start..bp], 10) catch return false;
    if (bp + 10 <= rest.len and std.mem.eql(u8, rest[bp .. bp + 2], "_t") and isHex8(rest[bp + 2 .. bp + 10])) {
        b_meta.digest = rest[bp + 2 .. bp + 10];
        bp += 10;
    }
    rest = rest[bp..];

    var tile_major = false;
    if (std.mem.endsWith(u8, rest, "_tm")) {
        tile_major = true;
        rest = rest[0 .. rest.len - 3];
    }
    var b_mode: []const u8 = "lane";
    if (std.mem.endsWith(u8, rest, "_shared_rows")) {
        b_mode = "shared_rows";
        rest = rest[0 .. rest.len - "_shared_rows".len];
    } else if (std.mem.endsWith(u8, rest, "_one")) {
        b_mode = "one";
        rest = rest[0 .. rest.len - "_one".len];
    } else if (std.mem.endsWith(u8, rest, "_snax")) {
        b_mode = "snax";
        rest = rest[0 .. rest.len - "_snax".len];
    }
    const gidx = std.mem.lastIndexOf(u8, rest, "_g") orelse return false;
    const group = std.fmt.parseInt(u32, rest[gidx + 2 ..], 10) catch return false;
    const classes_s = rest[0..gidx];
    const rate_masks = parseClassRateMasks(classes_s, a_meta.cb == .pair1) orelse return false;
    if (!inRateSlice(if (b_meta.cb == .pair1) &pair1_rates else &served_scalar_rates, sn)) return false;

    const a_tid = lookupMixedTableId(a_meta) orelse return false;
    const b_hm = lookupSingleHashMask(b_meta, sn) orelse return false;

    var spec_buf: [128]u8 = undefined;
    const spec_name = if (group > 0)
        std.fmt.bufPrint(&spec_buf, "sushi_exl3_fz_{s}_g{d}_{s}_{s}_{s}", .{
            if (is_down) "down" else "pair",
            group,
            b_mode,
            if (a_meta.cb == .pair1) "ap1" else "asc",
            if (b_meta.cb == .pair1) "bp1" else "bsc",
        }) catch return false
    else
        std.fmt.bufPrint(&spec_buf, "sushi_exl3_fz_{s}_{s}_{s}_{s}", .{
            if (is_down) "down" else "pair",
            b_mode,
            if (a_meta.cb == .pair1) "ap1" else "asc",
            if (b_meta.cb == .pair1) "bp1" else "bsc",
        }) catch return false;

    const is_snax = std.mem.eql(u8, b_mode, "snax");
    if (!custom.resolveSpecs(if (is_snax) &fz_nax_specs else &fz_simd_specs, spec_name, inputs, out_dtypes, targs, out)) return false;
    if (a_meta.cb != .pair1) {
        if (!appendFcInt(out, fc_cb_kind, cbKindInt(a_meta.cb))) return false;
        if (!appendFcInt(out, fc_table_id, a_tid)) return false;
    }
    if (!appendFcU32(out, fc_win_mask, a_meta.win.mask())) return false;
    if (!appendFcBool(out, fc_fold_out, true)) return false;
    if (!appendFcBool(out, fc_tile_major, tile_major)) return false;
    if (!appendFcU32(out, fc_rate_mask_g, rate_masks.g)) return false;
    if (!is_down) {
        if (!appendFcU32(out, fc_rate_mask_u, rate_masks.u)) return false;
    }

    if (b_meta.cb != .pair1) {
        if (!appendFcInt(out, fc_b_cb_kind, cbKindInt(b_meta.cb))) return false;
        if (!appendFcU32(out, fc_b_hash_mult, b_hm.mult)) return false;
        if (!appendFcU32(out, fc_b_hash_mask, b_hm.mask)) return false;
    }
    if (!appendFcU32(out, fc_b_win_mask, b_meta.win.mask())) return false;
    if (!appendFcBool(out, fc_b_fold_out, true)) return false;
    return sortAndFinalize(out);
}

fn resolveFoldGatedOrRg(
    name: []const u8,
    mixed: bool,
    inputs: []const RuntimeInput,
    out_dtypes: []const Dtype,
    targs: []const RuntimeTarg,
    out: *Resolved,
) bool {
    const prefix = if (mixed) "sushi_rg_fold_" else "sushi_gated_fold_";
    var rest = name[prefix.len..];
    const is_down = if (std.mem.startsWith(u8, rest, "pair_")) blk: {
        rest = rest["pair_".len..];
        break :blk false;
    } else if (std.mem.startsWith(u8, rest, "down_")) blk: {
        rest = rest["down_".len..];
        break :blk true;
    } else return false;

    var a_meta: CodecMeta = .{};
    const a_len = parseCbWinPrefix(rest, &a_meta) orelse return false;
    rest = rest[a_len..];

    // Optional `_tr_<scb>_w<sw>_n<ngu>_<ndn>[_t<sd>]` at the end
    var has_tr = false;
    var b_meta: CodecMeta = .{};
    var b_n: u32 = 0;
    if (std.mem.indexOf(u8, rest, "_tr_")) |tr_idx| {
        has_tr = true;
        var tr_s = rest[tr_idx + 4 ..];
        rest = rest[0..tr_idx];
        const bu1 = std.mem.indexOfScalar(u8, tr_s, '_') orelse return false;
        b_meta.cb = parseCbName(tr_s[0..bu1]) orelse return false;
        if (bu1 + 2 >= tr_s.len or tr_s[bu1 + 1] != 'w') return false;
        var bp = bu1 + 2;
        while (bp < tr_s.len and std.ascii.isDigit(tr_s[bp])) bp += 1;
        b_meta.win = parseWindowBits(tr_s[bu1 + 2 .. bp]) orelse return false;
        if (bp + 2 >= tr_s.len or !std.mem.eql(u8, tr_s[bp .. bp + 2], "_n")) return false;
        bp += 2;
        const g0 = bp;
        while (bp < tr_s.len and std.ascii.isDigit(tr_s[bp])) bp += 1;
        const n_gu = std.fmt.parseInt(u32, tr_s[g0..bp], 10) catch return false;
        if (bp >= tr_s.len or tr_s[bp] != '_') return false;
        bp += 1;
        const d0 = bp;
        while (bp < tr_s.len and std.ascii.isDigit(tr_s[bp])) bp += 1;
        const n_dn = std.fmt.parseInt(u32, tr_s[d0..bp], 10) catch return false;
        b_n = if (is_down) n_dn else n_gu;
        if (bp + 10 <= tr_s.len and std.mem.eql(u8, tr_s[bp .. bp + 2], "_t") and isHex8(tr_s[bp + 2 .. bp + 10])) {
            b_meta.digest = tr_s[bp + 2 .. bp + 10];
            bp += 10;
        }
        if (bp != tr_s.len) return false;
    }

    var tile_major = false;
    if (mixed and std.mem.endsWith(u8, rest, "_tm")) {
        tile_major = true;
        rest = rest[0 .. rest.len - 3];
    }
    var with_logit = false;
    if (std.mem.endsWith(u8, rest, "_logit")) {
        with_logit = true;
        rest = rest[0 .. rest.len - "_logit".len];
    }
    var rate_masks: ClassRateMasks = .{ .g = 0, .u = 0 };
    if (mixed) {
        rate_masks = parseClassRateMasks(rest, a_meta.cb == .pair1) orelse return false;
    } else if (rest.len != 0) {
        return false;
    }

    // Validate uniform Part A NHW is in its switch
    var a_hm_single: HashMaskPair = .{ .mult = 0, .mask = 0 };
    var a_tid: i32 = 0;
    if (mixed) {
        a_tid = lookupMixedTableId(a_meta) orelse return false;
    } else {
        const nhw_i = findIntTarg(targs, "NHW") orelse return false;
        if (nhw_i <= 0) return false;
        const nhw: u32 = @intCast(nhw_i);
        if (!inRateSlice(if (a_meta.cb == .pair1) &pair1_rates else &served_scalar_rates, nhw)) return false;
        a_hm_single = lookupSingleHashMask(a_meta, nhw) orelse return false;
    }

    var spec_buf: [128]u8 = undefined;
    if (has_tr) {
        // Verify shared_trellis fixed NAX geometry (`S_TGRP=4, S_KPH=1, S_UNR=2`) and rate in switch.
        if (findIntTarg(targs, "S_TGRP") != 4 or findIntTarg(targs, "S_KPH") != 1 or findIntTarg(targs, "S_UNR") != 2) return false;
        if (!inRateSlice(if (b_meta.cb == .pair1) &pair1_rates else &served_scalar_rates, b_n)) return false;
        const b_hm = lookupSingleHashMask(b_meta, b_n) orelse return false;
        const spec_name = std.fmt.bufPrint(&spec_buf, "{s}{s}{s}_tr_{s}_{s}", .{
            prefix,
            if (is_down) "down" else "pair",
            if (with_logit) "_logit" else "",
            if (a_meta.cb == .pair1) "ap1" else "asc",
            if (b_meta.cb == .pair1) "bp1" else "bsc",
        }) catch return false;
        if (!custom.resolveSpecs(&fz_nax_specs, spec_name, inputs, out_dtypes, targs, out)) return false;
        if (b_meta.cb != .pair1) {
            if (!appendFcInt(out, fc_b_cb_kind, cbKindInt(b_meta.cb))) return false;
            if (!appendFcU32(out, fc_b_hash_mult, b_hm.mult)) return false;
            if (!appendFcU32(out, fc_b_hash_mask, b_hm.mask)) return false;
        }
        if (!appendFcU32(out, fc_b_win_mask, b_meta.win.mask())) return false;
        if (!appendFcBool(out, fc_b_fold_out, true)) return false;
    } else {
        const spec_name = std.fmt.bufPrint(&spec_buf, "{s}{s}{s}_{s}", .{
            prefix,
            if (is_down) "down" else "pair",
            if (with_logit) "_logit" else "",
            if (a_meta.cb == .pair1) "ap1" else "asc",
        }) catch return false;
        if (!custom.resolveSpecs(&fz_simd_specs, spec_name, inputs, out_dtypes, targs, out)) return false;
    }

    if (a_meta.cb != .pair1) {
        if (!appendFcInt(out, fc_cb_kind, cbKindInt(a_meta.cb))) return false;
        if (mixed) {
            if (!appendFcInt(out, fc_table_id, a_tid)) return false;
        } else {
            if (!appendFcU32(out, fc_hash_mult, a_hm_single.mult)) return false;
            if (!appendFcU32(out, fc_hash_mask, a_hm_single.mask)) return false;
        }
    }
    if (!appendFcU32(out, fc_win_mask, a_meta.win.mask())) return false;
    if (!appendFcBool(out, fc_fold_out, true)) return false;
    if (mixed) {
        if (!appendFcBool(out, fc_tile_major, tile_major)) return false;
        if (!appendFcU32(out, fc_rate_mask_g, rate_masks.g)) return false;
        if (!is_down) {
            if (!appendFcU32(out, fc_rate_mask_u, rate_masks.u)) return false;
        }
        if (is_down) {
            if (!appendFcInt(out, fc_clamp_limit, 0)) return false;
        }
    }
    return sortAndFinalize(out);
}

fn resolveSingleRate(
    name: []const u8,
    inputs: []const RuntimeInput,
    out_dtypes: []const Dtype,
    targs: []const RuntimeTarg,
    out: *Resolved,
) bool {
    const parsed = parseSingleRateName(name);
    const nhw_i = findIntTarg(targs, "NHW") orelse return false;
    if (nhw_i <= 0) return false;
    const nhw: u32 = @intCast(nhw_i);
    const hm = lookupSingleHashMask(parsed.meta, nhw) orelse return false;

    var spec_buf: [128]u8 = undefined;
    const spec_name = std.fmt.bufPrint(&spec_buf, "{s}_{s}", .{ parsed.base, if (parsed.meta.cb == .pair1) "pair1" else "scalar" }) catch return false;

    // For `project_rows` SIMD/NAX, synthesize the `ROW_CAP` CT param from runtime `ROWS` / `TGRP` / `KPH`.
    var aug_targs: [64]RuntimeTarg = undefined;
    var active_targs: []const RuntimeTarg = targs;
    const is_rows_nax = std.mem.startsWith(u8, parsed.base, "sushi_exl3_rows_nax");
    const is_rows_simd = !is_rows_nax and std.mem.startsWith(u8, parsed.base, "sushi_exl3_rows") and !std.mem.startsWith(u8, parsed.base, "sushi_exl3_rows_finish");
    if (is_rows_simd or is_rows_nax) {
        if (targs.len + 1 > aug_targs.len) return false;
        @memcpy(aug_targs[0..targs.len], targs);
        const rows = findIntTarg(targs, "ROWS") orelse return false;
        const cap: i32 = if (is_rows_nax) blk: {
            if (std.mem.indexOf(u8, parsed.base, "_counted") != null) {
                if (rows < 1 or rows > 8) return false;
                if (rows == 1 and std.mem.indexOf(u8, parsed.base, "_rms") != null) break :blk 1;
                break :blk if (rows <= 4) 4 else 8;
            }
            const kph = findIntTarg(targs, "KPH") orelse return false;
            if (rows < 1 or rows > 16) return false;
            if (kph == 4 and rows <= 8) break :blk 8;
            break :blk 16;
        } else blk: {
            const tgrp = findIntTarg(targs, "TGRP") orelse return false;
            if (rows < 1 or rows > 8) return false;
            if (tgrp == 1) {
                if (rows != 1) return false;
                break :blk 1;
            }
            break :blk if (rows <= 4) 4 else 8;
        };
        aug_targs[targs.len] = .{ .name = "ROW_CAP", .value = .{ .int = cap } };
        active_targs = aug_targs[0 .. targs.len + 1];
    }

    const matched = if (is_rows_nax)
        custom.resolveSpecs(&rows_nax_specs, spec_name, inputs, out_dtypes, active_targs, out)
    else if (is_rows_simd)
        custom.resolveSpecs(&rows_simd_specs, spec_name, inputs, out_dtypes, active_targs, out)
    else
        custom.resolveSpecs(&gemm_nax_specs, spec_name, inputs, out_dtypes, active_targs, out) or
            custom.resolveSpecs(&single_simd_specs, spec_name, inputs, out_dtypes, active_targs, out);

    if (!matched) return false;

    if (parsed.meta.cb != .pair1) {
        if (!appendFcInt(out, fc_cb_kind, cbKindInt(parsed.meta.cb))) return false;
        if (!appendFcU32(out, fc_hash_mult, hm.mult)) return false;
        if (!appendFcU32(out, fc_hash_mask, hm.mask)) return false;
    }
    if (!appendFcU32(out, fc_win_mask, parsed.meta.win.mask())) return false;
    if (!appendFcBool(out, fc_fold_out, parsed.meta.fold_out)) return false;
    return sortAndFinalize(out);
}

// ── Public Catalog Interface ──

pub fn resolve(
    name: []const u8,
    inputs: []const RuntimeInput,
    out_dtypes: []const Dtype,
    targs: []const RuntimeTarg,
    out: *Resolved,
) bool {
    // 1. Static non-codebook EXL3 kernels (utility + rows_finish*)
    if (custom.resolveSpecs(&static_simd_specs, name, inputs, out_dtypes, targs, out)) return true;
    if (custom.resolveSpecs(&rows_finish_specs, name, inputs, out_dtypes, targs, out)) return true;

    // 2. Dynamic clamped mid (`sushi_exl3_rg_mid_clamp<L>`)
    if (std.mem.startsWith(u8, name, "sushi_exl3_rg_mid_clamp")) {
        const limit = std.fmt.parseInt(i32, name["sushi_exl3_rg_mid_clamp".len..], 10) catch return false;
        if (!custom.resolveSpecs(&static_simd_specs, "sushi_exl3_rg_mid", inputs, out_dtypes, targs, out)) return false;
        if (!appendFcInt(out, fc_clamp_limit, limit)) return false;
        return sortAndFinalize(out);
    }

    // 3. Mixed-rate standalone (`sushi_exl3_rg_{pair|down|pair_prepared|down_prepared}_...`)
    if (std.mem.startsWith(u8, name, "sushi_exl3_rg_")) {
        return resolveMixedRg(name, inputs, out_dtypes, targs, out);
    }

    // 4. DSv41 fused (`sushi_exl3_fz_{prep|pair|mid|down}_...`)
    if (std.mem.startsWith(u8, name, "sushi_exl3_fz_")) {
        return resolveFusedFz(name, inputs, out_dtypes, targs, out);
    }

    // 5. Qwen4 folded (`sushi_rg_fold_*`, `sushi_gated_fold_*`)
    if (std.mem.startsWith(u8, name, "sushi_rg_fold_")) {
        return resolveFoldGatedOrRg(name, true, inputs, out_dtypes, targs, out);
    }
    if (std.mem.startsWith(u8, name, "sushi_gated_fold_")) {
        return resolveFoldGatedOrRg(name, false, inputs, out_dtypes, targs, out);
    }

    // 6. Single-rate codebook kernels (`sushi_exl3_*`, `sushi_lane_*`, `sushi_down_*`, `sushi_prefill_grid_transpose*`)
    if (std.mem.startsWith(u8, name, "sushi_exl3_") or
        std.mem.startsWith(u8, name, "sushi_lane_") or
        std.mem.startsWith(u8, name, "sushi_down_") or
        std.mem.startsWith(u8, name, "sushi_prefill_grid_transpose"))
    {
        return resolveSingleRate(name, inputs, out_dtypes, targs, out);
    }

    return false;
}

pub fn visitFunctions(ctx: anytype, comptime onName: fn (@TypeOf(ctx), []const u8) anyerror!void) !void {
    try custom.visitSpecs(&static_simd_specs, ctx, onName);
    try custom.visitSpecs(&single_simd_specs, ctx, onName);
    try custom.visitSpecs(&rows_finish_specs, ctx, onName);
    try custom.visitSpecs(&rows_simd_specs, ctx, onName);
    try custom.visitSpecs(&rg_simd_specs, ctx, onName);
    try custom.visitSpecs(&fz_simd_specs, ctx, onName);
    try custom.visitSpecs(&gemm_nax_specs, ctx, onName);
    try custom.visitSpecs(&rows_nax_specs, ctx, onName);
    try custom.visitSpecs(&fz_nax_specs, ctx, onName);
}

pub fn gemmNaxSpecs() []const KernelSpec {
    return &gemm_nax_specs;
}

pub fn emitUnit(out: *std.ArrayList(u8), a: std.mem.Allocator, unit_name: []const u8) !bool {
    const sh = custom.parseShardUnit(unit_name);
    if (std.mem.eql(u8, sh.base, "exl3_simd")) {
        try custom.emitSpecListsSharded(out, a, false, &.{ &static_simd_specs, &single_simd_specs }, .by_inst, sh.idx, sh.count);
        return true;
    }
    if (std.mem.eql(u8, sh.base, "exl3_rows_simd")) {
        try custom.emitSpecListsSharded(out, a, false, &.{ &rows_finish_specs, &rows_simd_specs }, .by_inst, sh.idx, sh.count);
        return true;
    }
    if (std.mem.eql(u8, sh.base, "exl3_rg_simd")) {
        try custom.emitSpecListsSharded(out, a, false, &.{&rg_simd_specs}, .by_inst, sh.idx, sh.count);
        return true;
    }
    if (std.mem.eql(u8, sh.base, "exl3_fz_simd")) {
        try custom.emitSpecListsSharded(out, a, false, &.{&fz_simd_specs}, .by_inst, sh.idx, sh.count);
        return true;
    }
    if (std.mem.eql(u8, sh.base, "exl3_gemm_nax")) {
        try custom.emitSpecListsSharded(out, a, true, &.{&gemm_nax_specs}, .by_spec, sh.idx, sh.count);
        return true;
    }
    if (std.mem.eql(u8, sh.base, "exl3_rows_nax")) {
        try custom.emitSpecListsSharded(out, a, true, &.{&rows_nax_specs}, .by_inst, sh.idx, sh.count);
        return true;
    }
    if (std.mem.eql(u8, sh.base, "exl3_fz_nax")) {
        try custom.emitSpecListsSharded(out, a, true, &.{&fz_nax_specs}, .by_inst, sh.idx, sh.count);
        return true;
    }
    return false;
}
