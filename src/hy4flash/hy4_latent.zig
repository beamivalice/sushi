//! A layer's stored kv8 latent rows read to f32 in one launch: a row range (`range`) or picked rows (`gather`) of
//! the codes, scales and biases, each value `bfloat16_t(scale * code + bias)` as MLX's affine dequantize computes it at
//! bits 8, then widened to f32, so the bytes are those of the dequantize, the take and the cast it replaces.
const std = @import("std");
const mlx = @import("../mlx.zig");
const mlx_pages = @import("../mlx_pages.zig");
const Arr = mlx.mlx_array;
const kcfg = @import("hy4_kcfg.zig");

const source: [:0]const u8 =
    \\uint col_w = thread_position_in_grid.x;
    \\uint out_row = thread_position_in_grid.y;
    \\uint src_row = GATHER ? uint(idx[size_t(out_row) * size_t(idx_strides[0])]) : out_row;
    \\uint32_t word = q[size_t(src_row) * size_t(q_strides[0]) + size_t(col_w) * size_t(q_strides[1])];
    \\uint g = col_w / uint(GS_WORDS);
    \\bfloat16_t scale = scales[size_t(src_row) * size_t(scales_strides[0]) + size_t(g) * size_t(scales_strides[1])];
    \\bfloat16_t bias = biases[size_t(src_row) * size_t(biases_strides[0]) + size_t(g) * size_t(biases_strides[1])];
    \\bfloat16_t v0 = scale * uint8_t(word & 0xffu) + bias;
    \\bfloat16_t v1 = scale * uint8_t((word >> 8) & 0xffu) + bias;
    \\bfloat16_t v2 = scale * uint8_t((word >> 16) & 0xffu) + bias;
    \\bfloat16_t v3 = scale * uint8_t((word >> 24) & 0xffu) + bias;
    \\device float4* dst = reinterpret_cast<device float4*>(out + (size_t(out_row) * size_t(WORDS) + size_t(col_w)) * 4);
    \\*dst = float4(float(v0), float(v1), float(v2), float(v3));
;

var kernel: ?mlx.mlx_fast_metal_kernel = null;

fn getKernel() !mlx.mlx_fast_metal_kernel {
    if (kernel) |k| return k;
    const ins = [_][*:0]const u8{ "q", "scales", "biases", "idx" };
    const outs = [_][*:0]const u8{"out"};
    const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new("sushi_hy4_latent_read", iv, ov, source, "", false, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    kernel = k;
    return k;
}

const Key = struct { gather: bool, rows: c_int, words: c_int, gs_words: c_int, idx_dt: mlx.mlx_dtype };
var cfgs: kcfg.Cache(Key) = .{};

fn build(k: Key) !kcfg.Config {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const tg_x: c_int = @min(k.words, 128);
    const tg_y: c_int = @min(k.rows, @divTrunc(256, tg_x));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ k.rows, k.words * 4 }, 2, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, k.words, k.rows, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, tg_x, tg_y, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, "GATHER", k.gather));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "WORDS", k.words));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "GS_WORDS", k.gs_words));
    return cfg;
}

fn commonApplies(s: mlx.mlx_stream, q: Arr, scales: Arr, biases: Arr, group_size: u32, bits: u8) bool {
    if (!mlx.streamIsGpu(s) or bits != 8 or group_size == 0 or group_size % 4 != 0) return false;
    if (mlx.mlx_array_dtype(q) != .uint32 or mlx.mlx_array_dtype(scales) != .bfloat16 or mlx.mlx_array_dtype(biases) != .bfloat16) return false;
    const qs = mlx.getShape(q);
    const ss = mlx.getShape(scales);
    const bs = mlx.getShape(biases);
    if (qs.len != 2 or ss.len != 2 or bs.len != 2) return false;
    if (qs[0] <= 0 or qs[1] <= 0 or ss[0] != qs[0] or bs[0] != qs[0] or ss[1] != bs[1]) return false;
    const gs_words: c_int = @intCast(group_size / 4);
    return @rem(qs[1], gs_words) == 0 and ss[1] == @divExact(qs[1], gs_words);
}

/// Whether `range` serves the store's parts: kv8, a GPU stream, the group size it was quantized at.
pub fn rangeApplies(s: mlx.mlx_stream, q: Arr, scales: Arr, biases: Arr, group_size: u32, bits: u8) bool {
    return commonApplies(s, q, scales, biases, group_size, bits);
}

/// Whether `gather` serves the parts and the pick list `idx`.
pub fn gatherApplies(s: mlx.mlx_stream, q: Arr, scales: Arr, biases: Arr, idx: Arr, group_size: u32, bits: u8) bool {
    if (!commonApplies(s, q, scales, biases, group_size, bits)) return false;
    const is = mlx.getShape(idx);
    const dt = mlx.mlx_array_dtype(idx);
    return is.len == 1 and is[0] > 0 and (dt == .uint32 or dt == .int32);
}

fn launch(s: mlx.mlx_stream, q: Arr, scales: Arr, biases: Arr, idx: Arr, gather_on: bool, rows: c_int, group_size: u32) !Arr {
    const words = mlx.getShape(q)[1];
    const gs_words: c_int = @intCast(group_size / 4);
    const cfg = try cfgs.get(.{ .gather = gather_on, .rows = rows, .words = words, .gs_words = gs_words, .idx_dt = mlx.mlx_array_dtype(idx) }, build);
    const ins = [_]Arr{ q, scales, biases, idx };
    const iv = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    var ov = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ov);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&ov, try getKernel(), iv, cfg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, ov, 0));
    return out;
}

/// Every stored row of the parts, dequantized and widened: [rows, width] f32. The caller frees it.
pub fn range(s: mlx.mlx_stream, q: Arr, scales: Arr, biases: Arr, group_size: u32) !Arr {
    return launch(s, q, scales, biases, q, false, mlx.getShape(q)[0], group_size);
}

/// The rows `idx` picks, in its order (repeats allowed), dequantized and widened. The caller frees it.
pub fn gather(s: mlx.mlx_stream, q: Arr, scales: Arr, biases: Arr, idx: Arr, group_size: u32) !Arr {
    return launch(s, q, scales, biases, idx, true, mlx.getShape(idx)[0], group_size);
}

/// The most rows one `append` writes (a verify block).
pub const append_rows_max: c_int = 8;

const append_source: [:0]const u8 =
    \\constexpr float eps = 1e-7f;
    \\constexpr float n_bins = 255.0f;
    \\uint tid = thread_position_in_grid.x;
    \\uint row = thread_position_in_grid.y;
    \\uint g = tid >> 5;
    \\uint lane = thread_index_in_simdgroup;
    \\uint start_row = uint(meta[0]);
    \\uint ring_len = uint(meta[1]);
    \\uint dst_row = ring_len > 0u ? ((start_row + row) % ring_len) : (start_row + row);
    \\size_t in_idx = (size_t(row) * size_t(GROUPS) + size_t(g)) * 64 + size_t(lane) * 2;
    \\float w0 = float(c[in_idx]);
    \\float w1 = float(c[in_idx + 1]);
    \\float w_min = min(min(metal::numeric_limits<float>::infinity(), w0), w1);
    \\float w_max = max(max(0.0f, w0), w1);
    \\w_min = simd_min(w_min);
    \\w_max = simd_max(w_max);
    \\float scale = max((w_max - w_min) / n_bins, eps);
    \\bool side = abs(w_min) > abs(w_max);
    \\scale = side ? scale : -scale;
    \\float edge = side ? w_min : w_max;
    \\float q0 = round(edge / scale);
    \\bool at_zero = q0 == 0.0f;
    \\scale = at_zero ? scale : edge / q0;
    \\float bias = at_zero ? 0.0f : edge;
    \\if (lane == 0u) {
    \\  size_t g_out = size_t(dst_row) * size_t(GROUPS) + size_t(g);
    \\  s_dst[g_out] = static_cast<bfloat16_t>(scale);
    \\  b_dst[g_out] = static_cast<bfloat16_t>(bias);
    \\}
    \\uint8_t v0 = uint8_t(min(round((w0 - bias) / scale), n_bins));
    \\uint8_t v1 = uint8_t(min(round((w1 - bias) / scale), n_bins));
    \\uint32_t lo = uint32_t(v0) | (uint32_t(v1) << 8);
    \\uint32_t hi = simd_shuffle_down(lo, 1);
    \\if ((lane & 1u) == 0u) {
    \\  q_dst[size_t(dst_row) * size_t(GROUPS * 16) + size_t(g * 16 + (lane >> 1))] = lo | (hi << 16);
    \\}
    \\for (uint j = tid; j < uint(ROPE); j += uint(GROUPS * 32)) {
    \\  pe_dst[size_t(dst_row) * size_t(ROPE) + size_t(j)] = k_pe[size_t(row) * size_t(ROPE) + size_t(j)];
    \\}
;

var append_kernel: ?mlx.mlx_fast_metal_kernel = null;

fn getAppendKernel() !mlx.mlx_fast_metal_kernel {
    if (append_kernel) |k| return k;
    const ins = [_][*:0]const u8{ "c", "k_pe", "meta" };
    const outs = [_][*:0]const u8{ "q_dst", "s_dst", "b_dst", "pe_dst" };
    const iv = mlx.mlx_vector_string_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_string_free(iv);
    const ov = mlx.mlx_vector_string_new_data(&outs, outs.len);
    defer _ = mlx.mlx_vector_string_free(ov);
    const k = mlx.mlx_fast_metal_kernel_new("sushi_hy4_latent_append", iv, ov, append_source, "", true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    append_kernel = k;
    return k;
}

const AppendKey = struct { rows: c_int, groups: c_int, rope: c_int };
var append_cfgs: kcfg.Cache(AppendKey) = .{};

fn buildAppend(k: AppendKey) !kcfg.Config {
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    errdefer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const gx = k.groups * 32;
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, k.groups * 16 }, 2, .uint32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, k.groups }, 2, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, k.groups }, 2, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ 1, k.rope }, 2, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, gx, k.rows, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, @min(gx, 256), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "GROUPS", k.groups));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "ROPE", k.rope));
    return cfg;
}

/// Whether `append` serves the rows `c` [T, latent] and `k_pe` [T, rope], bf16, into a kv8 store (bits 8, group 64)
/// on the engine's stream, 1..`append_rows_max` rows; `ring_len` is a sliding layer's ring (0: a linear store).
pub fn appendApplies(s: mlx.mlx_stream, c: Arr, k_pe: Arr, group_size: u32, bits: u8, ring_len: c_int) bool {
    if (!mlx.engine.isStream(s) or bits != 8 or group_size != 64) return false;
    if (mlx.mlx_array_dtype(c) != .bfloat16 or mlx.mlx_array_dtype(k_pe) != .bfloat16) return false;
    const cs = mlx.getShape(c);
    const ps = mlx.getShape(k_pe);
    if (cs.len != 2 or ps.len != 2) return false;
    if (cs[0] < 1 or cs[0] > append_rows_max or ps[0] != cs[0]) return false;
    if (cs[1] <= 0 or @rem(cs[1], 64) != 0 or ps[1] <= 0) return false;
    if (ring_len < 0 or (ring_len > 0 and cs[0] > ring_len)) return false;
    return true;
}

/// Quantizes `c` as MLX's affine quantize does (bits 8, group 64) and writes its codes, scales and biases and `k_pe`
/// into `dsts` (the store's parts) from row `start_row`, wrapping at `ring_len` when it is a ring: one launch.
pub fn append(s: mlx.mlx_stream, c: Arr, k_pe: Arr, dsts: []const Arr, start_row: c_int, ring_len: c_int) !void {
    const cs = mlx.getShape(c);
    const groups = @divExact(cs[1], 64);
    const rope = mlx.getShape(k_pe)[1];
    const cfg = try append_cfgs.get(.{ .rows = cs[0], .groups = groups, .rope = rope }, buildAppend);
    const mv = [_]i32{ start_row, ring_len };
    const meta = try mlx.newData(&mv, &.{2}, .int32);
    defer _ = mlx.mlx_array_free(meta);
    const ins = [_]Arr{ c, k_pe, meta };
    const iv = mlx.mlx_vector_array_new_data(&ins, ins.len);
    defer _ = mlx.mlx_vector_array_free(iv);
    try mlx.check(mlx.kernelApplyInPlace(try getAppendKernel(), iv, cfg, s, dsts) orelse return error.MlxError);
}

const testing = std.testing;
const Ops = @import("../mlx_ops.zig").Ops;
const kv_quant = @import("../kv_quant.zig");
const row_store = @import("../row_store.zig");

fn bytesOf(o: *Ops, a: Arr) ![]const u8 {
    const c = try o.contiguous(a);
    try mlx.check(mlx.mlx_array_eval(c));
    return std.mem.sliceAsBytes((mlx.mlx_array_data_float32(c) orelse return error.HostRead)[0..mlx.mlx_array_size(c)]);
}

test "hy4 latent read: kv8 store range and gather match MLX affine dequantize plus f32 cast byte for byte at width 512, 1 and 2048 rows" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();

    const width: c_int = 512;
    const rope: c_int = 64;
    const total: usize = 2048;
    const a = testing.allocator;
    const vals = try a.alloc(f32, total * @as(usize, @intCast(width)));
    defer a.free(vals);
    const pe_vals = try a.alloc(f32, total * @as(usize, @intCast(rope)));
    defer a.free(pe_vals);
    var rng = std.Random.DefaultPrng.init(0x1a7e17);
    for (vals) |*v| v.* = @floatCast(rng.random().floatNorm(f64) * 2.5);
    for (pe_vals) |*v| v.* = @floatCast(rng.random().floatNorm(f64) * 1.0);

    var st = row_store.Store.init(&.{
        .{ .cols = @intCast(kv_quant.packedWords(width, 8)), .dtype = .uint32 },
        .{ .cols = @intCast(@divExact(width, 64)), .dtype = .bfloat16 },
        .{ .cols = @intCast(@divExact(width, 64)), .dtype = .bfloat16 },
        .{ .cols = @intCast(rope), .dtype = .bfloat16 },
    }, 16);
    defer st.deinit();

    {
        var o: Ops = .{ .s = es };
        defer o.deinit();
        const all = try o.own(try mlx.newData(vals.ptr, &[_]c_int{ @intCast(total), width }, .float32));
        const pe = try o.own(try mlx.newData(pe_vals.ptr, &[_]c_int{ @intCast(total), rope }, .float32));
        const cuts = [_]usize{ 0, 1, 17, 513, total };
        for (cuts[0 .. cuts.len - 1], cuts[1..]) |r0, r1| {
            const c_bf = try o.cast(try o.slice(all, 0, @intCast(r0), @intCast(r1)), .bfloat16);
            const pe_bf = try o.cast(try o.slice(pe, 0, @intCast(r0), @intCast(r1)), .bfloat16);
            var q = try kv_quant.quantizeAffine(es, c_bf, 64, 8);
            defer q.deinit();
            try st.append(&.{ q.q, q.scales, q.biases, pe_bf }, es);
        }
    }

    const q_buf = st.buf(0);
    const s_buf = st.buf(1);
    const b_buf = st.buf(2);

    for ([_][2]c_int{ .{ 0, 1 }, .{ 137, 138 }, .{ 0, 2048 }, .{ 19, 531 } }) |rng_pair| {
        var o: Ops = .{ .s = es };
        defer o.deinit();
        const cs = try o.slice(q_buf, 0, rng_pair[0], rng_pair[1]);
        const ss = try o.slice(s_buf, 0, rng_pair[0], rng_pair[1]);
        const bs = try o.slice(b_buf, 0, rng_pair[0], rng_pair[1]);
        const want = try o.cast(try o.own(try kv_quant.dequantizeAffine(es, cs, ss, bs, 64, 8)), .float32);
        try testing.expect(rangeApplies(es, cs, ss, bs, 64, 8));
        const got = try o.own(try range(es, cs, ss, bs, 64));
        try testing.expectEqualSlices(u8, try bytesOf(&o, want), try bytesOf(&o, got));
    }

    for ([_]usize{ 1, 2048 }) |n_pick| {
        var o: Ops = .{ .s = es };
        defer o.deinit();
        const picks = try a.alloc(u32, n_pick);
        defer a.free(picks);
        for (picks, 0..) |*p, i| {
            p.* = if (i % 7 == 0 and i > 0) picks[i - 1] else rng.random().uintLessThan(u32, @intCast(total));
        }
        const idx = try o.own(try mlx.newData(picks.ptr, &[_]c_int{@intCast(n_pick)}, .uint32));
        const want = try o.cast(try o.own(try kv_quant.dequantizeAffine(
            es,
            try o.take(q_buf, idx, 0),
            try o.take(s_buf, idx, 0),
            try o.take(b_buf, idx, 0),
            64,
            8,
        )), .float32);
        try testing.expect(gatherApplies(es, q_buf, s_buf, b_buf, idx, 64, 8));
        const got = try o.own(try gather(es, q_buf, s_buf, b_buf, idx, 64));
        try testing.expectEqualSlices(u8, try bytesOf(&o, want), try bytesOf(&o, got));
    }
}

test "hy4 latent read: range and gather each run in one launch instead of two and five" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();

    const width: c_int = 512;
    const total: usize = 64;
    const a = testing.allocator;
    const vals = try a.alloc(f32, total * @as(usize, @intCast(width)));
    defer a.free(vals);
    for (vals, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.13);

    var o: Ops = .{ .s = es };
    defer o.deinit();
    const all = try o.own(try mlx.newData(vals.ptr, &[_]c_int{ @intCast(total), width }, .float32));
    var q = try kv_quant.quantizeAffine(es, try o.cast(all, .bfloat16), 64, 8);
    defer q.deinit();
    try mlx.check(mlx.mlx_array_eval(q.q));
    try mlx.check(mlx.mlx_array_eval(q.scales));
    try mlx.check(mlx.mlx_array_eval(q.biases));

    const r0 = mlx.engine.stats().dispatches;
    const old_range = try o.cast(try o.own(try kv_quant.dequantizeAffine(es, q.q, q.scales, q.biases, 64, 8)), .float32);
    try mlx.check(mlx.mlx_array_eval(old_range));
    const old_range_d = mlx.engine.stats().dispatches - r0;

    const r1 = mlx.engine.stats().dispatches;
    const new_range = try o.own(try range(es, q.q, q.scales, q.biases, 64));
    try mlx.check(mlx.mlx_array_eval(new_range));
    const new_range_d = mlx.engine.stats().dispatches - r1;

    try testing.expectEqual(@as(u64, 2), old_range_d);
    try testing.expectEqual(@as(u64, 1), new_range_d);

    const picks = [_]u32{ 12, 0, 63, 12, 5, 31, 7, 19 };
    const idx = try o.own(try mlx.newData(&picks, &[_]c_int{picks.len}, .uint32));
    try mlx.check(mlx.mlx_array_eval(idx));

    const g0 = mlx.engine.stats().dispatches;
    const old_gather = try o.cast(try o.own(try kv_quant.dequantizeAffine(
        es,
        try o.take(q.q, idx, 0),
        try o.take(q.scales, idx, 0),
        try o.take(q.biases, idx, 0),
        64,
        8,
    )), .float32);
    try mlx.check(mlx.mlx_array_eval(old_gather));
    const old_gather_d = mlx.engine.stats().dispatches - g0;

    const g1 = mlx.engine.stats().dispatches;
    const new_gather = try o.own(try gather(es, q.q, q.scales, q.biases, idx, 64));
    try mlx.check(mlx.mlx_array_eval(new_gather));
    const new_gather_d = mlx.engine.stats().dispatches - g1;

    try testing.expectEqual(@as(u64, 5), old_gather_d);
    try testing.expectEqual(@as(u64, 1), new_gather_d);
}

fn rawBytesOf(o: *Ops, a: Arr) ![]const u8 {
    const c = try o.contiguous(a);
    try mlx.check(mlx.mlx_array_eval(c));
    const n = mlx.mlx_array_size(c);
    return switch (mlx.mlx_array_dtype(c)) {
        .uint32 => std.mem.sliceAsBytes((mlx.mlx_array_data_uint32(c) orelse return error.HostRead)[0..n]),
        .bfloat16 => std.mem.sliceAsBytes((mlx.mlx_array_data_bfloat16(c) orelse return error.HostRead)[0..n]),
        else => return error.HostRead,
    };
}

test "hy4 latent append: kv8 store append matches MLX affine quantize plus slice writes byte for byte at width 512, 1 and 8 rows" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();

    const width: c_int = 512;
    const rope: c_int = 64;
    const a = testing.allocator;
    const parts = [_]row_store.Part{
        .{ .cols = @intCast(kv_quant.packedWords(width, 8)), .dtype = .uint32 },
        .{ .cols = @intCast(@divExact(width, 64)), .dtype = .bfloat16 },
        .{ .cols = @intCast(@divExact(width, 64)), .dtype = .bfloat16 },
        .{ .cols = @intCast(rope), .dtype = .bfloat16 },
    };

    var rng = std.Random.DefaultPrng.init(0xa99e4d);
    for ([_]usize{ 1, 8 }) |batch| {
        const vals = try a.alloc(f32, batch * @as(usize, @intCast(width)));
        defer a.free(vals);
        const pe_vals = try a.alloc(f32, batch * @as(usize, @intCast(rope)));
        defer a.free(pe_vals);
        for (vals) |*v| v.* = @floatCast(rng.random().floatNorm(f64) * 3.0);
        for (pe_vals) |*v| v.* = @floatCast(rng.random().floatNorm(f64) * 1.5);

        @memset(vals[0..64], 2.25);
        @memset(vals[64..128], -3.5);
        @memset(vals[128..192], 0.0);
        for (vals[192..256], 0..) |*v, i| v.* = if (i % 2 == 0) 0.0 else -0.0;
        for (vals[256..320]) |*v| v.* = @floatCast(rng.random().floatNorm(f64) * 0.1);
        vals[256 + 17] = 512.0;
        for (vals[320..384]) |*v| v.* = -@abs(@as(f32, @floatCast(rng.random().floatNorm(f64) * 4.0))) - 0.5;
        for (vals[384..448]) |*v| v.* = @floatCast(rng.random().floatNorm(f64) * 1.0e4);

        var ref_st = row_store.Store.init(&parts, 16);
        defer ref_st.deinit();
        var got_st = row_store.Store.init(&parts, 16);
        defer got_st.deinit();

        var o: Ops = .{ .s = es };
        defer o.deinit();
        const c_bf = try o.cast(try o.own(try mlx.newData(vals.ptr, &[_]c_int{ @intCast(batch), width }, .float32)), .bfloat16);
        const pe_bf = try o.cast(try o.own(try mlx.newData(pe_vals.ptr, &[_]c_int{ @intCast(batch), rope }, .float32)), .bfloat16);

        for (0..3) |_| {
            var q = try kv_quant.quantizeAffine(es, c_bf, 64, 8);
            defer q.deinit();
            try ref_st.append(&.{ q.q, q.scales, q.biases, pe_bf }, es);

            try testing.expect(appendApplies(es, c_bf, pe_bf, 64, 8, 0));
            try got_st.reserveAppend(batch, es);
            try append(es, c_bf, pe_bf, &.{ got_st.buf(0), got_st.buf(1), got_st.buf(2), got_st.buf(3) }, @intCast(got_st.used), 0);
            got_st.used += batch;
        }

        const used: c_int = @intCast(ref_st.used);
        try testing.expectEqual(ref_st.used, got_st.used);
        for (0..4) |p| {
            const want_slice = try o.slice(ref_st.buf(p), 0, 0, used);
            const got_slice = try o.slice(got_st.buf(p), 0, 0, used);
            try testing.expectEqualSlices(u8, try rawBytesOf(&o, want_slice), try rawBytesOf(&o, got_slice));
        }
    }

    const ring_w: c_int = 8;
    const n_ring: usize = 5;
    const rvals = try a.alloc(f32, n_ring * @as(usize, @intCast(width)));
    defer a.free(rvals);
    const rpe = try a.alloc(f32, n_ring * @as(usize, @intCast(rope)));
    defer a.free(rpe);
    for (rvals) |*v| v.* = @floatCast(rng.random().floatNorm(f64) * 2.0);
    for (rpe) |*v| v.* = @floatCast(rng.random().floatNorm(f64) * 1.0);

    var o: Ops = .{ .s = es };
    defer o.deinit();
    const rc_bf = try o.cast(try o.own(try mlx.newData(rvals.ptr, &[_]c_int{ @intCast(n_ring), width }, .float32)), .bfloat16);
    const rpe_bf = try o.cast(try o.own(try mlx.newData(rpe.ptr, &[_]c_int{ @intCast(n_ring), rope }, .float32)), .bfloat16);
    var rq = try kv_quant.quantizeAffine(es, rc_bf, 64, 8);
    defer rq.deinit();

    var ref_ring: [4]Arr = undefined;
    var got_ring: [4]Arr = undefined;
    var saved_ring: [4]Arr = undefined;
    for (parts, 0..) |pt, i| {
        ref_ring[i] = try o.own((try mlx_pages.ringBuffer(&.{ ring_w, @intCast(pt.cols) }, pt.dtype, es)).array);
        got_ring[i] = try o.own((try mlx_pages.ringBuffer(&.{ ring_w, @intCast(pt.cols) }, pt.dtype, es)).array);
        saved_ring[i] = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&saved_ring[i], got_ring[i]));
        _ = try o.own(saved_ring[i]);
    }

    const start_pos: c_int = 6;
    const src_parts = [_]Arr{ rq.q, rq.scales, rq.biases, rpe_bf };
    for (src_parts, 0..) |src, i| {
        const cols = mlx.getShape(src)[1];
        var cur = ref_ring[i];
        var pos: c_int = 0;
        while (pos < @as(c_int, @intCast(n_ring))) {
            const slot = @mod(start_pos + pos, ring_w);
            const run = @min(@as(c_int, @intCast(n_ring)) - pos, ring_w - slot);
            const sub = try o.slice(src, 0, pos, pos + run);
            const up = try o.slot();
            _ = try mlx_pages.InPlace.adopt(cur, .ring).write(up, sub, &.{ slot, 0 }, &.{ slot + run, cols }, es);
            cur = up.*;
            pos += run;
        }
        ref_ring[i] = cur;
    }

    try testing.expect(appendApplies(es, rc_bf, rpe_bf, 64, 8, ring_w));
    try append(es, rc_bf, rpe_bf, &got_ring, start_pos, ring_w);

    for (0..4) |i| {
        try testing.expectEqualSlices(u8, try rawBytesOf(&o, ref_ring[i]), try rawBytesOf(&o, got_ring[i]));
        for (try rawBytesOf(&o, saved_ring[i])) |b| try testing.expectEqual(@as(u8, 0), b);
    }
}

test "hy4 latent append: a layer append runs in one launch instead of five" {
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();

    const width: c_int = 512;
    const rope: c_int = 64;
    const parts = [_]row_store.Part{
        .{ .cols = @intCast(kv_quant.packedWords(width, 8)), .dtype = .uint32 },
        .{ .cols = @intCast(@divExact(width, 64)), .dtype = .bfloat16 },
        .{ .cols = @intCast(@divExact(width, 64)), .dtype = .bfloat16 },
        .{ .cols = @intCast(rope), .dtype = .bfloat16 },
    };

    var vals: [512]f32 = undefined;
    for (&vals, 0..) |*v, i| v.* = @sin(@as(f32, @floatFromInt(i)) * 0.17);
    var pe_vals: [64]f32 = undefined;
    for (&pe_vals, 0..) |*v, i| v.* = @cos(@as(f32, @floatFromInt(i)) * 0.23);

    var o: Ops = .{ .s = es };
    defer o.deinit();
    const c_bf = try o.cast(try o.own(try mlx.newData(&vals, &[_]c_int{ 1, width }, .float32)), .bfloat16);
    const pe_bf = try o.cast(try o.own(try mlx.newData(&pe_vals, &[_]c_int{ 1, rope }, .float32)), .bfloat16);
    try mlx.check(mlx.mlx_array_eval(c_bf));
    try mlx.check(mlx.mlx_array_eval(pe_bf));

    var old_st = row_store.Store.init(&parts, 16);
    defer old_st.deinit();
    try old_st.reserveAppend(4, es);
    for (0..4) |i| try mlx.check(mlx.mlx_array_eval(old_st.buf(i)));

    var new_st = row_store.Store.init(&parts, 16);
    defer new_st.deinit();
    try new_st.reserveAppend(4, es);
    for (0..4) |i| try mlx.check(mlx.mlx_array_eval(new_st.buf(i)));

    const d0 = mlx.engine.stats().dispatches;
    var q = try kv_quant.quantizeAffine(es, c_bf, 64, 8);
    defer q.deinit();
    try old_st.append(&.{ q.q, q.scales, q.biases, pe_bf }, es);
    for (0..4) |i| try mlx.check(mlx.mlx_array_eval(old_st.buf(i)));
    const old_d = mlx.engine.stats().dispatches - d0;

    const d1 = mlx.engine.stats().dispatches;
    try append(es, c_bf, pe_bf, &.{ new_st.buf(0), new_st.buf(1), new_st.buf(2), new_st.buf(3) }, 0, 0);
    new_st.used += 1;
    for (0..4) |i| try mlx.check(mlx.mlx_array_eval(new_st.buf(i)));
    const new_d = mlx.engine.stats().dispatches - d1;

    try testing.expectEqual(@as(u64, 5), old_d);
    try testing.expectEqual(@as(u64, 1), new_d);
}
