//! Many rows through one expert at a time, with the bytes of MLX's one-row product. `gather_mm` over a `[T, 1, K]`
//! operand runs `gemv_gather` once per pick, so a layer's picks read each expert's matrix once per pick (and, at f32
//! products, over the bank widened to f32). This kernel reads the BF16 bank as stored and widens in the register,
//! which is the f32 product's operand exactly; the rows and the output are BF16 or f32 as the product's. It takes the picks sorted by expert and applies up to `vectors` of an expert's rows to every matrix tile it loads, with
//! `gemv`'s thread layout and its f32 accumulation order (`gemv.h`, `GEMVKernel`: 4 rows per simdgroup, 4 columns a
//! thread, `simd_shuffle_down` over 32 lanes), so each output element is the one `gather_mm` writes.
const std = @import("std");
const mlx = @import("../mlx.zig");
const Arr = mlx.mlx_array;

/// Output rows a threadgroup covers: 4 simdgroups of `gemv`'s 4 rows.
const tile_rows: c_int = 16;

/// Vectors one matrix tile serves.
pub const vectors: usize = 4;

const source: [:0]const u8 =
    \\constexpr int TM = 4;
    \\constexpr int TN = 4;
    \\constexpr int BLOCK = 32 * TN;
    \\uint lane = thread_index_in_simdgroup;
    \\uint sg = simdgroup_index_in_threadgroup;
    \\uint g = threadgroup_position_in_grid.x;
    \\uint row0 = (threadgroup_position_in_grid.y * 4u + sg) * uint(TM);
    \\uint expert = groups[g * 3u];
    \\uint start = groups[g * 3u + 1u];
    \\uint count = groups[g * 3u + 2u];
    \\const device TW* mat = bank + ((size_t)expert * (size_t)NDIM + (size_t)row0) * (size_t)KDIM;
    \\const device T* vec[NV];
    \\for (int v = 0; v < NV; ++v) {
    \\  uint pick = start + (uint(v) < count ? uint(v) : 0u);
    \\  vec[v] = rows + (size_t)picks[pick * 2u] * (size_t)KDIM;
    \\}
    \\float result[NV][TM];
    \\for (int v = 0; v < NV; ++v) for (int tm = 0; tm < TM; ++tm) result[v][tm] = 0.0f;
    \\constexpr int n_iter = KDIM / BLOCK;
    \\int bn = int(lane) * TN;
    \\for (int i = 0; i < n_iter; ++i) {
    \\  float inter[TM][TN];
    \\  for (int tm = 0; tm < TM; ++tm) for (int tn = 0; tn < TN; ++tn) inter[tm][tn] = static_cast<float>(mat[tm * KDIM + bn + tn]);
    \\  for (int v = 0; v < NV; ++v) {
    \\    float coeff[TN];
    \\    for (int tn = 0; tn < TN; ++tn) coeff[tn] = static_cast<float>(vec[v][bn + tn]);
    \\    for (int tm = 0; tm < TM; ++tm) for (int tn = 0; tn < TN; ++tn) result[v][tm] += inter[tm][tn] * coeff[tn];
    \\  }
    \\  bn += BLOCK;
    \\}
    \\if (KDIM - n_iter * BLOCK > 0) {
    \\  float inter[TM][TN];
    \\  for (int tm = 0; tm < TM; ++tm) for (int tn = 0; tn < TN; ++tn) inter[tm][tn] = bn + tn < KDIM ? static_cast<float>(mat[tm * KDIM + bn + tn]) : 0.0f;
    \\  for (int v = 0; v < NV; ++v) {
    \\    float coeff[TN];
    \\    for (int tn = 0; tn < TN; ++tn) coeff[tn] = bn + tn < KDIM ? static_cast<float>(vec[v][bn + tn]) : 0.0f;
    \\    for (int tm = 0; tm < TM; ++tm) for (int tn = 0; tn < TN; ++tn) result[v][tm] += inter[tm][tn] * coeff[tn];
    \\  }
    \\}
    \\for (int v = 0; v < NV; ++v) for (int tm = 0; tm < TM; ++tm) {
    \\  for (ushort sn = 16; sn >= 1; sn >>= 1) result[v][tm] += simd_shuffle_down(result[v][tm], sn);
    \\}
    \\if (lane == 0u) {
    \\  for (int v = 0; v < NV; ++v) if (uint(v) < count) {
    \\    device T* o = out + (size_t)picks[(start + uint(v)) * 2u + 1u] * (size_t)NDIM + (size_t)row0;
    \\    for (int tm = 0; tm < TM; ++tm) o[tm] = static_cast<T>(result[v][tm]);
    \\  }
    \\}
;

var kernel: ?mlx.mlx_fast_metal_kernel = null;

fn getKernel() !mlx.mlx_fast_metal_kernel {
    if (kernel) |k| return k;
    const input_names = [_][*:0]const u8{ "bank", "rows", "groups", "picks" };
    const output_names = [_][*:0]const u8{"out"};
    const in_vec = mlx.mlx_vector_string_new_data(&input_names, input_names.len);
    defer _ = mlx.mlx_vector_string_free(in_vec);
    const out_vec = mlx.mlx_vector_string_new_data(&output_names, output_names.len);
    defer _ = mlx.mlx_vector_string_free(out_vec);
    const k = mlx.mlx_fast_metal_kernel_new("sushi_hy4_expert_mv", in_vec, out_vec, source, "", true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    kernel = k;
    return k;
}

/// The picks of one product, grouped by expert: each group is up to `vectors` picks of one expert, and `picks` holds
/// each one's `[row of the operand, row of the output]`.
pub const Plan = struct {
    groups: Arr,
    picks: Arr,
    n_groups: usize,

    pub fn deinit(self: Plan) void {
        _ = mlx.mlx_array_free(self.groups);
        _ = mlx.mlx_array_free(self.picks);
    }
};

/// Plans the picks `order` (indices into `bank_rows`, equal bank rows adjacent): the `j`-th ordered pick reads operand
/// row `operand[j]` and writes output row `dest[j]`.
pub fn plan(gpa: std.mem.Allocator, bank_rows: []const u32, order: []const u32, operand: []const u32, dest: []const u32) !Plan {
    std.debug.assert(order.len == operand.len and order.len == dest.len);
    var groups: std.ArrayList(u32) = .empty;
    defer groups.deinit(gpa);
    var at: usize = 0;
    while (at < order.len) {
        const e = bank_rows[order[at]];
        var n: usize = 1;
        while (n < vectors and at + n < order.len and bank_rows[order[at + n]] == e) n += 1;
        try groups.appendSlice(gpa, &.{ e, @intCast(at), @intCast(n) });
        at += n;
    }
    const pairs = try gpa.alloc(u32, 2 * order.len);
    defer gpa.free(pairs);
    for (0..order.len) |i| {
        pairs[2 * i] = operand[i];
        pairs[2 * i + 1] = dest[i];
    }
    // The groups follow the picks, a readback: a dry recording plans one a pick, the most there can be.
    const n_groups = try @import("../tensor/plan/plan_api.zig").bound(groups.items.len / 3, order.len);
    try groups.appendNTimes(gpa, 0, 3 * n_groups - groups.items.len);
    return .{
        .groups = mlx.mlx_array_new_data(groups.items.ptr, &[_]c_int{@intCast(groups.items.len)}, 1, .uint32),
        .picks = mlx.mlx_array_new_data(pairs.ptr, &[_]c_int{@intCast(pairs.len)}, 1, .uint32),
        .n_groups = n_groups,
    };
}

/// `out[dest j] = bank[e_j] @ rows[operand j]` for every planned pick: `bank` [S, N, K] BF16, `rows` [R, K] BF16 or f32,
/// `out` [out_rows, N] of the rows' dtype (a row no pick names is unset). N is a multiple of 16.
pub fn product(s: mlx.mlx_stream, bank: Arr, rows: Arr, p: Plan, out_rows: c_int) !Arr {
    const bs = mlx.getShape(bank);
    const dtype = mlx.mlx_array_dtype(rows);
    if (bs.len != 3 or mlx.mlx_array_dtype(bank) != .bfloat16 or (dtype != .bfloat16 and dtype != .float32)) return error.Hy4ExpertMvOperand;
    const n = bs[1];
    const k = bs[2];
    if (@rem(n, tile_rows) != 0 or mlx.getShape(rows)[1] != k) return error.Hy4ExpertMvOperand;
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &[_]c_int{ out_rows, n }, 2, dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32 * @as(c_int, @intCast(p.n_groups)), 4 * @divExact(n, tile_rows), 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 4, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", dtype));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "TW", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NV", @as(c_int, vectors)));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "NDIM", n));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "KDIM", k));
    const inputs = [_]Arr{ bank, rows, p.groups, p.picks };
    const in_vec = mlx.mlx_vector_array_new_data(&inputs, inputs.len);
    defer _ = mlx.mlx_vector_array_free(in_vec);
    var outputs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs, try getKernel(), in_vec, cfg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, outputs, 0));
    return out;
}

// ── tests ────────────────────────────────────────────────────────────────────────────────────────

const testing = std.testing;

fn randomBf16(gpa: std.mem.Allocator, rnd: std.Random, n: usize) ![]u16 {
    const out = try gpa.alloc(u16, n);
    for (out) |*v| v.* = @intCast(@as(u32, @bitCast(@as(f32, @floatCast(rnd.floatNorm(f64) * 0.5)))) >> 16);
    return out;
}

/// The product on random bytes in the shape the layers run, against `gather_mm`, byte for byte.
fn parityCase(s: mlx.mlx_stream, dtype: mlx.mlx_dtype, n: usize, k: usize, slots: usize, picks: usize, seed: u64) !void {
    const gpa = testing.allocator;
    var prng = std.Random.DefaultPrng.init(seed);
    const rnd = prng.random();
    const bank_host = try randomBf16(gpa, rnd, slots * n * k);
    defer gpa.free(bank_host);
    const rows_n = picks / 3 + 1;
    const rows_bits = try randomBf16(gpa, rnd, rows_n * k);
    defer gpa.free(rows_bits);
    const bank = mlx.mlx_array_new_data(bank_host.ptr, &[_]c_int{ @intCast(slots), @intCast(n), @intCast(k) }, 3, .bfloat16);
    defer _ = mlx.mlx_array_free(bank);
    const rows_bf16 = mlx.mlx_array_new_data(rows_bits.ptr, &[_]c_int{ @intCast(rows_n), @intCast(k) }, 2, .bfloat16);
    defer _ = mlx.mlx_array_free(rows_bf16);
    // f32 rows hold more than a BF16's bits, as the f32 products' latent rows do.
    const rows_wide = try gpa.alloc(f32, rows_n * k);
    defer gpa.free(rows_wide);
    for (rows_wide, rows_bits, 0..) |*d, bits, i| d.* = @as(f32, @bitCast(@as(u32, bits) << 16)) * (1.0 + @as(f32, @floatFromInt(i % 7)) * 0.0001234);
    const rows_f32 = mlx.mlx_array_new_data(rows_wide.ptr, &[_]c_int{ @intCast(rows_n), @intCast(k) }, 2, .float32);
    defer _ = mlx.mlx_array_free(rows_f32);
    const rows = if (dtype == .float32) rows_f32 else rows_bf16;
    const e = try gpa.alloc(u32, picks);
    defer gpa.free(e);
    const src = try gpa.alloc(u32, picks);
    defer gpa.free(src);
    for (e, src) |*d, *r| {
        d.* = rnd.uintLessThan(u32, @intCast(slots));
        r.* = rnd.uintLessThan(u32, @intCast(rows_n));
    }
    // The old path: one gemv per pick, `[P, 1, K]` against the transposed bank.
    var o: @import("../mlx_ops.zig").Ops = .{ .s = s };
    defer o.deinit();
    const lhs = try o.own(mlx.mlx_array_new_data(src.ptr, &[_]c_int{@intCast(picks)}, 1, .uint32));
    const rhs = try o.own(mlx.mlx_array_new_data(e.ptr, &[_]c_int{@intCast(picks)}, 1, .uint32));
    const x = try o.reshape(rows, &.{ @intCast(rows_n), 1, @intCast(k) });
    // At f32 the product runs over the bank widened to f32, as the layer's does.
    const w = try o.transpose(try o.cast(bank, dtype), &.{ 0, 2, 1 });
    const want = try o.slot();
    try mlx.check(mlx.mlx_gather_mm(want, x, w, lhs, rhs, false, s));
    const want_flat = try o.reshape(want.*, &.{ @intCast(picks), @intCast(n) });
    try mlx.check(mlx.mlx_array_eval(want_flat));

    const order = try gpa.alloc(u32, picks);
    defer gpa.free(order);
    for (order, 0..) |*d, i| d.* = @intCast(i);
    std.mem.sort(u32, order, e, struct {
        fn less(bank_rows: []const u32, a: u32, b: u32) bool {
            return bank_rows[a] < bank_rows[b] or (bank_rows[a] == bank_rows[b] and a < b);
        }
    }.less);
    const operand = try gpa.alloc(u32, picks);
    defer gpa.free(operand);
    for (order, operand) |i, *d| d.* = src[i];
    const pl = try plan(gpa, e, order, operand, order);
    defer pl.deinit();
    const got = try product(s, bank, rows, pl, @intCast(picks));
    defer _ = mlx.mlx_array_free(got);
    try mlx.check(mlx.mlx_array_eval(got));
    if (dtype == .float32) {
        const a = (mlx.mlx_array_data_float32(want_flat) orelse return error.MlxArrayDataNull)[0 .. picks * n];
        const b = (mlx.mlx_array_data_float32(got) orelse return error.MlxArrayDataNull)[0 .. picks * n];
        try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(a), std.mem.sliceAsBytes(b));
    } else {
        const a = (mlx.mlx_array_data_bfloat16(want_flat) orelse return error.MlxArrayDataNull)[0 .. picks * n];
        const b = (mlx.mlx_array_data_bfloat16(got) orelse return error.MlxArrayDataNull)[0 .. picks * n];
        try testing.expectEqualSlices(u16, a, b);
    }
}

test "hy4 expert mv: the product is gather_mm's bytes, at f32 and BF16 rows, at the layers' shapes and at the tiny's" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    for ([_]mlx.mlx_dtype{ .float32, .bfloat16 }) |dtype| {
        try parityCase(s, dtype, 64, 32, 8, 100, 1);
        try parityCase(s, dtype, 32, 32, 8, 37, 2);
        try parityCase(s, dtype, 3072, 1536, 16, 700, 3);
        try parityCase(s, dtype, 1536, 1536, 16, 700, 4);
    }
}
