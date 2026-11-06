//! The prefill GEMM of a one-bank trellis linear (a trunk weight) past the multi-row kernel: the bank decoded once to
//! its f16 weight, then MLX's GEMM. Every row reads the same bank, so one decode serves the whole chunk, where the
//! sorted GEMM decodes the bank again for every 32-row window. MLX's regular NAX GEMM sums each output as the sorted
//! GEMM does (16x32x16 MMAs, k ascending, one f32 accumulator, stored f16), so the inner plane is its bytes.
const std = @import("std");
const mlx = @import("mlx_host").mlx;
const kernels = @import("expert_exl3_kernels.zig");
const kernel_src = @import("kernel_source.zig");

/// Test and bench seam: false keeps every linear on the sorted GEMM.
pub var enabled: bool = true;
/// Inner planes taken through the decoded path, for a test.
pub var decoded_calls: u64 = 0;
/// Set by a caller around the linears whose transient bill holds `DECODED_SCRATCH_BYTES`; others keep the sorted GEMM.
pub var decoded_billed: bool = false;

/// A one-bank trellis `[1, in/16, out/16, n]` decoded to its f16 weight `[out, in]`: the values the GEMMs' fragment
/// reader hands their MMAs, each at its (output, input) place.
const DENSE_SOURCE: [:0]const u8 =
    \\constexpr uint N = uint(NHW);
    \\constexpr uint PACKED_W = N / 2u;
    \\constexpr uint OT = uint(ODIM) / 16u;
    \\const uint lane = uint(thread_index_in_simdgroup);
    \\const uint pair = uint(threadgroup_position_in_grid.x) * 4u + uint(simdgroup_index_in_threadgroup);
    \\if (pair >= OT / 2u) return;
    \\const uint tk = uint(threadgroup_position_in_grid.y);
    \\const device uint *words = (const device uint *)trellis + ((size_t)tk * (size_t)OT + pair * 2u) * PACKED_W;
    \\const nfrag w0 = nax_wfrag_k<N>(words, lane);
    \\const nfrag w1 = nax_wfrag_k<N>(words + PACKED_W, lane);
    \\const short2 o = nax_origin(lane);
    \\const size_t k = (size_t)tk * 16u + uint(o.x);
    \\const size_t n0 = (size_t)pair * 32u + uint(o.y);
    \\*(device half4 *)(y + n0 * IDIM + k) = half4(w0[0], w0[1], w0[2], w0[3]);
    \\*(device half4 *)(y + (n0 + 8u) * IDIM + k) = half4(w0[4], w0[5], w0[6], w0[7]);
    \\*(device half4 *)(y + (n0 + 16u) * IDIM + k) = half4(w1[0], w1[1], w1[2], w1[3]);
    \\*(device half4 *)(y + (n0 + 24u) * IDIM + k) = half4(w1[4], w1[5], w1[6], w1[7]);
;

var dense_slots: kernel_src.KernelSlots = kernel_src.no_kernels;
const DenseKey = struct { in_dim: c_int, out_dim: c_int, n: c_int };
var dense_cfgs: kernel_src.CfgCache(DenseKey, 16) = .{};

/// The bank's weight as f16 `[out, in]` under the active decode parameters.
pub fn decodeDense(s: mlx.mlx_stream, trellis: mlx.mlx_array) !mlx.mlx_array {
    const tsh = mlx.getShape(trellis);
    if (tsh.len != 4 or tsh[0] != 1 or @rem(tsh[2], 2) != 0) return error.BadExl3Shape;
    const in_dim = tsh[1] * 16;
    const out_dim = tsh[2] * 16;
    const rate = try kernels.packedRate(tsh[3]);
    const key = DenseKey{ .in_dim = in_dim, .out_dim = out_dim, .n = @intCast(rate.n) };
    const cfg = dense_cfgs.get(key) orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &[_]c_int{ out_dim, in_dim }, 2, .float16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 128, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, @divFloor(@divExact(tsh[2], 2) + 3, 4) * 128, tsh[1], 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "IDIM", in_dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ODIM", out_dim));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "NHW", @intCast(rate.n)));
        dense_cfgs.put(key, c);
        break :blk c;
    };
    const k = try kernel_src.codebookKernelWith(&dense_slots, "sushi_exl3_trunk_dense_nax", &.{"trellis"}, &.{"y"}, DENSE_SOURCE, kernels.GEMM_NAX_INCLUDES ++ kernels.GEMM_NAX_FRAGS);
    const ov = try kernels.applyOuts(s, k, &.{trellis}, cfg, 1);
    defer _ = mlx.mlx_vector_array_free(ov);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, ov, 0));
    return out;
}

/// Below this many rows the decode pass costs more than the sorted GEMM saves.
pub const DECODED_MIN_ROWS: c_int = 1024;
/// The widest bank decoded whole; a wider one (an lm_head) keeps the sorted GEMM.
pub const DECODED_MAX_WEIGHTS: usize = 1 << 25;
/// The decoded path's scratch at its peak: one bank's f16 weight, and the previous one MLX's cache may still hold.
pub const DECODED_SCRATCH_BYTES: u64 = 2 * DECODED_MAX_WEIGHTS * 2;

/// Whether a one-bank linear of `rows` x `in` -> `out` takes the decoded path.
pub fn decodedServes(rows: c_int, in: c_int, out: c_int) bool {
    return enabled and decoded_billed and rows >= DECODED_MIN_ROWS and @as(usize, @intCast(in)) * @as(usize, @intCast(out)) <= DECODED_MAX_WEIGHTS and
        denseOrderMatches(rows, in, out);
}

/// Whether MLX's matmul of `rows` x `in` by `in` x `out` takes its regular NAX GEMM: k ascending in 16-wide MMAs into
/// one f32 accumulator per output, the sorted GEMM's order. MLX splits K (another sum order) when K is large against
/// the output (`steel_matmul_axpby`, mlx 0.32.3).
pub fn denseOrderMatches(rows: c_int, in: c_int, out: c_int) bool {
    const mx = @max(rows, out);
    return kernels.gemmNaxOn() and !(in >= 3 * mx or (mx <= 1024 and in > 2 * mx));
}

/// `inner`'s bytes by another road: the bank decoded to f16 once, then MLX's GEMM. Only where `denseOrderMatches`.
pub fn innerDense(s: mlx.mlx_stream, x: mlx.mlx_array, trellis: mlx.mlx_array) !mlx.mlx_array {
    decoded_calls += 1;
    const w = try decodeDense(s, trellis);
    defer _ = mlx.mlx_array_free(w);
    var wt = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(wt);
    try mlx.check(mlx.mlx_transpose(&wt, w, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_matmul(&out, x, wt, s));
    return out;
}

test "trunk gemm: the dense decode is the host tile decode, and MLX's GEMM over it writes the sorted GEMM's bytes where the order matches" {
    try mlx.requireGpu();
    const t = std.testing;
    if (!kernels.gemmNaxOn()) return error.SkipZigTest;
    const linear = @import("linear.zig");
    const fmt = @import("expert_exl3.zig");
    const s = mlx.gpuStream();
    defer kernels.setDecodeParams(.mul1);
    var prng = std.Random.DefaultPrng.init(0x7e7);
    const rnd = prng.random();
    const Case = struct { in: usize, out: usize, n: u32, dec: fmt.Decode, rows: []const c_int };
    for ([_]Case{
        .{ .in = 2560, .out = 640, .n = 96, .dec = kernels.SPEC1_W15, .rows = &.{ 1300, 2048, 4065 } },
        .{ .in = 6144, .out = 2560, .n = 96, .dec = kernels.SPEC1_W15, .rows = &.{4096} },
        .{ .in = 640, .out = 2560, .n = 64, .dec = kernels.SPEC1_W15, .rows = &.{ 33, 600 } },
        .{ .in = 512, .out = 384, .n = 48, .dec = .mcg, .rows = &.{ 200, 1000 } },
    }) |c| {
        kernels.setDecodeParams(c.dec);
        const words = try t.allocator.alloc(u16, (c.in / 16) * (c.out / 16) * c.n);
        defer t.allocator.free(words);
        for (words) |*v| v.* = rnd.int(u16);
        const tr = mlx.mlx_array_new_data(words.ptr, &[_]c_int{ 1, @intCast(c.in / 16), @intCast(c.out / 16), @intCast(c.n) }, 4, .uint16);
        defer _ = mlx.mlx_array_free(tr);
        const dense = try decodeDense(s, tr);
        defer _ = mlx.mlx_array_free(dense);
        try mlx.check(mlx.mlx_array_eval(dense));
        const dp: [*]const u16 = @ptrCast(mlx.mlx_array_data_float16(dense) orelse return error.Unreadable);
        var tile: [fmt.TILE_VALUES]u16 = undefined;
        for (0..c.in / 16) |tk| for (0..c.out / 16) |tn| {
            fmt.decodeTile(words[(tk * (c.out / 16) + tn) * c.n ..][0..c.n], .{ .n = c.n }, c.dec, &tile);
            for (0..16) |r| for (0..16) |col| {
                try t.expectEqual(tile[r * 16 + col], dp[(tn * 16 + col) * c.in + tk * 16 + r]);
            };
        };
        for (c.rows) |rows| {
            try t.expect(denseOrderMatches(rows, @intCast(c.in), @intCast(c.out)));
            const xs = try t.allocator.alloc(u16, @as(usize, @intCast(rows)) * c.in);
            defer t.allocator.free(xs);
            for (xs) |*v| v.* = fmt.f32ToF16Bits((rnd.float(f32) * 2 - 1) * 0.5);
            const x = mlx.mlx_array_new_data(xs.ptr, &[_]c_int{ rows, @intCast(c.in) }, 2, .float16);
            defer _ = mlx.mlx_array_free(x);
            const ids = try t.allocator.alloc(u32, @intCast(rows));
            defer t.allocator.free(ids);
            @memset(ids, 0);
            const slots = mlx.mlx_array_new_data(ids.ptr, &[_]c_int{rows}, 1, .uint32);
            defer _ = mlx.mlx_array_free(slots);
            const tab = try linear.windowTable(1, @intCast(rows), kernels.gemmWindowRows());
            defer _ = mlx.mlx_array_free(tab.starts);
            defer _ = mlx.mlx_array_free(tab.nlives);
            const want = try kernels.innerGemmSortedTable(s, x, tr, slots, kernels.gemmWindowRows(), true, tab);
            defer _ = mlx.mlx_array_free(want);
            const got = try innerDense(s, x, tr);
            defer _ = mlx.mlx_array_free(got);
            try mlx.check(mlx.mlx_array_eval(want));
            try mlx.check(mlx.mlx_array_eval(got));
            const len: usize = mlx.mlx_array_size(want);
            const wp: [*]const u16 = @ptrCast(mlx.mlx_array_data_float16(want) orelse return error.Unreadable);
            const gp: [*]const u16 = @ptrCast(mlx.mlx_array_data_float16(got) orelse return error.Unreadable);
            try t.expectEqualSlices(u16, wp[0..len], gp[0..len]);
        }
    }
}

test "trunk gemm: a billed one-bank linear past the decoded width takes the decoded path with the sorted GEMM's bytes" {
    try mlx.requireGpu();
    const t = std.testing;
    if (!kernels.gemmNaxOn()) return error.SkipZigTest;
    const linear = @import("linear.zig");
    const fmt = @import("expert_exl3.zig");
    const s = mlx.gpuStream();
    defer kernels.setDecodeParams(.mul1);
    var prng = std.Random.DefaultPrng.init(0x7e8);
    const rnd = prng.random();
    const in: usize = 2560;
    const out: usize = 512;
    const n: u32 = 96;
    const rows: c_int = DECODED_MIN_ROWS + 3;
    const words = try t.allocator.alloc(u16, (in / 16) * (out / 16) * n);
    defer t.allocator.free(words);
    for (words) |*v| v.* = rnd.int(u16);
    const suh_v = try t.allocator.alloc(u16, in);
    defer t.allocator.free(suh_v);
    for (suh_v) |*v| v.* = fmt.f32ToF16Bits(0.01 + rnd.float(f32) * 0.05);
    const svh_v = try t.allocator.alloc(u16, out);
    defer t.allocator.free(svh_v);
    for (svh_v) |*v| v.* = fmt.f32ToF16Bits(0.5 + rnd.float(f32));
    const xs = try t.allocator.alloc(u16, @as(usize, @intCast(rows)) * in);
    defer t.allocator.free(xs);
    for (xs) |*v| v.* = @truncate(@as(u32, @bitCast(rnd.float(f32) * 2 - 1)) >> 16);
    const p: @import("root.zig").Proj = .{
        .trellis = mlx.mlx_array_new_data(words.ptr, &[_]c_int{ 1, @intCast(in / 16), @intCast(out / 16), @intCast(n) }, 4, .uint16),
        .suh = mlx.mlx_array_new_data(suh_v.ptr, &[_]c_int{ 1, @intCast(in) }, 2, .float16),
        .svh = mlx.mlx_array_new_data(svh_v.ptr, &[_]c_int{ 1, @intCast(out) }, 2, .float16),
    };
    defer for ([_]mlx.mlx_array{ p.trellis, p.suh, p.svh }) |a| {
        _ = mlx.mlx_array_free(a);
    };
    const x = mlx.mlx_array_new_data(xs.ptr, &[_]c_int{ rows, @intCast(in) }, 2, .bfloat16);
    defer _ = mlx.mlx_array_free(x);
    defer decoded_billed = false;
    var got: [2]mlx.mlx_array = undefined;
    for (&got, [_]bool{ false, true }) |*g, billed| {
        decoded_billed = billed;
        const before = decoded_calls;
        g.* = try linear.linear(s, x, p, kernels.SPEC1_W15);
        try mlx.check(mlx.mlx_array_eval(g.*));
        try t.expectEqual(before + @intFromBool(billed), decoded_calls);
    }
    defer for (got) |g| {
        _ = mlx.mlx_array_free(g);
    };
    const len: usize = mlx.mlx_array_size(got[0]);
    var f: [2]mlx.mlx_array = .{ mlx.mlx_array_new(), mlx.mlx_array_new() };
    defer for (f) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (&f, got) |*dst, g| {
        try mlx.check(mlx.mlx_astype(dst, g, .float32, s));
        try mlx.check(mlx.mlx_array_eval(dst.*));
    }
    const a: [*]const u32 = @ptrCast(mlx.mlx_array_data_float32(f[0]) orelse return error.Unreadable);
    const b: [*]const u32 = @ptrCast(mlx.mlx_array_data_float32(f[1]) orelse return error.Unreadable);
    try t.expectEqualSlices(u32, a[0..len], b[0..len]);
}

test "trunk gemm: real pack trunk banks, every shape the gate allows writes the sorted GEMM bytes at 1024, 2048 and 4096 rows" {
    try mlx.requireGpu();
    const t = std.testing;
    const dir = std.mem.span(std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest);
    if (!kernels.gemmNaxOn()) return error.SkipZigTest;
    const linear = @import("linear.zig");
    const fmt = @import("expert_exl3.zig");
    const trace = std.c.getenv("SUSHI_EXL3_PARITY_TRACE") != null;
    const s = mlx.gpuStream();
    kernels.setDecodeParams(kernels.SPEC1_W15);
    defer kernels.setDecodeParams(.mul1);

    // One real bank per (in, out) the trunk layers hold.
    const Bank = struct { in: c_int, out: c_int, trellis: mlx.mlx_array };
    var banks: [16]Bank = undefined;
    var n_banks: usize = 0;
    defer for (banks[0..n_banks]) |b| {
        _ = mlx.mlx_array_free(b.trellis);
    };
    var layer: usize = 0;
    while (true) : (layer += 1) {
        const path = try std.fmt.allocPrintSentinel(t.allocator, "{s}/trunk-stq-L{d:0>2}.safetensors", .{ dir, layer }, 0);
        defer t.allocator.free(path);
        std.Io.Dir.cwd().access(t.io, path, .{}) catch break;
        var map = mlx.mlx_map_string_to_array_new();
        defer _ = mlx.mlx_map_string_to_array_free(map);
        var meta = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(meta);
        try mlx.check(mlx.mlx_load_safetensors(&map, &meta, path.ptr, mlx.mlx_default_cpu_stream_new()));
        const it = mlx.mlx_map_string_to_array_iterator_new(map);
        defer _ = mlx.mlx_map_string_to_array_iterator_free(it);
        while (true) {
            var key: ?[*:0]const u8 = null;
            var value = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(value);
            if (mlx.mlx_map_string_to_array_iterator_next(&key, &value, it) != 0 or key == null) break;
            if (!std.mem.endsWith(u8, std.mem.span(key.?), ".trellis")) continue;
            const sh = mlx.getShape(value);
            if (sh.len != 3 or sh[2] != 96) continue;
            const in = sh[0] * 16;
            const out = sh[1] * 16;
            const seen = for (banks[0..n_banks]) |b| {
                if (b.in == in and b.out == out) break true;
            } else false;
            if (seen or n_banks == banks.len) continue;
            var bank = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_reshape(&bank, value, &[_]c_int{ 1, sh[0], sh[1], sh[2] }, 4, s));
            banks[n_banks] = .{ .in = in, .out = out, .trellis = bank };
            n_banks += 1;
        }
    }
    try t.expect(n_banks >= 5);

    var prng = std.Random.DefaultPrng.init(0x7e9);
    const rnd = prng.random();
    var allowed: usize = 0;
    var refused: usize = 0;
    for (banks[0..n_banks]) |b| for ([_]c_int{ 1024, 2048, 4096 }) |rows| {
        const ok = denseOrderMatches(rows, b.in, b.out);
        if (ok) allowed += 1 else refused += 1;
        const xs = try t.allocator.alloc(u16, @as(usize, @intCast(rows)) * @as(usize, @intCast(b.in)));
        defer t.allocator.free(xs);
        for (xs) |*v| v.* = fmt.f32ToF16Bits((rnd.float(f32) * 2 - 1) * 0.5);
        const x = mlx.mlx_array_new_data(xs.ptr, &[_]c_int{ rows, b.in }, 2, .float16);
        defer _ = mlx.mlx_array_free(x);
        const ids = try t.allocator.alloc(u32, @intCast(rows));
        defer t.allocator.free(ids);
        @memset(ids, 0);
        const slots = mlx.mlx_array_new_data(ids.ptr, &[_]c_int{rows}, 1, .uint32);
        defer _ = mlx.mlx_array_free(slots);
        const tab = try linear.windowTable(1, @intCast(rows), kernels.gemmWindowRows());
        defer _ = mlx.mlx_array_free(tab.starts);
        defer _ = mlx.mlx_array_free(tab.nlives);
        const want = try kernels.innerGemmSortedTable(s, x, b.trellis, slots, kernels.gemmWindowRows(), true, tab);
        defer _ = mlx.mlx_array_free(want);
        const got = try innerDense(s, x, b.trellis);
        defer _ = mlx.mlx_array_free(got);
        try mlx.check(mlx.mlx_array_eval(want));
        try mlx.check(mlx.mlx_array_eval(got));
        const len: usize = mlx.mlx_array_size(want);
        const wp: [*]const u16 = @ptrCast(mlx.mlx_array_data_float16(want) orelse return error.Unreadable);
        const gp: [*]const u16 = @ptrCast(mlx.mlx_array_data_float16(got) orelse return error.Unreadable);
        var differ: usize = 0;
        for (wp[0..len], gp[0..len]) |a, c| differ += @intFromBool(a != c);
        if (trace) std.debug.print("trunk gemm {d}x{d}->{d}: gate {s}, {d} of {d} outputs differ\n", .{ rows, b.in, b.out, if (ok) "allows" else "refuses", differ, len });
        if (ok) try t.expectEqual(@as(usize, 0), differ);
    };
    try t.expect(allowed > 0 and refused > 0);
}
