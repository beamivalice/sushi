//! Sparse head-packed native NAX SDPA with bounded query batches; FP32 GEMMs without NAX.
const std = @import("std");
const mlx = @import("mlx.zig");
const Ops = @import("glm5_model.zig").Ops;
const latent_store = @import("glm5_latent.zig");
const Latent = latent_store.Latent;
const Arr = mlx.mlx_array;
pub const max_rows: usize = 16;
pub const wide_rows: usize = 32;
/// The FP32 composite's bank and score planes fit `scratch_limit` at this many rows.
pub const composite_rows: usize = 8;
/// Bounds one tile of either width; a pending layer holds at most one.
pub const scratch_limit: usize = 128 * 1024 * 1024;
var calls: usize = 0;
var composite_calls: usize = 0;
pub fn enabled() bool {
    return !@import("glm5_model.zig").reference_numerics;
}
/// Rows per tile on this GPU; only the fused NAX arm takes `wide_rows`.
pub fn tileRows() usize {
    return if (@import("glm5_model.zig").naxArms()) max_rows else composite_rows;
}
pub fn compositeCount() usize {
    return composite_calls;
}
pub fn resetDispatchCount() void {
    calls = 0;
}
pub fn dispatchCount() usize {
    return calls;
}
pub fn transientBudget(chunk: usize, pending_layers: usize) !usize {
    if (!enabled() or chunk <= 8) return 0;
    return std.math.mul(usize, scratch_limit, pending_layers);
}
pub fn temporaryBytes(rows: usize) !usize {
    if (rows == 0 or (rows > max_rows and rows != wide_rows)) return error.UnsupportedGlmPackedAttention;
    // One gathered BF16 KV bank shared by K/V; query, result, zeroing and contiguity
    // copies; the bool mask and index bookkeeping. No [Q,H,K] floating score tensor.
    return std.math.mul(usize, rows, 2051 * 512 * 2 + 64 * 512 * 2 * 4 + 64 * 2051 + 2051 * 9 + 16);
}
/// BF16 and FP32 banks, the mask, FP32 and scaled queries, three FP32 score planes, the FP32/BF16 results.
pub fn compositeBytes(rows: usize) !usize {
    if (rows == 0 or rows > composite_rows) return error.UnsupportedGlmPackedAttention;
    return std.math.mul(usize, rows, 2051 * 512 * 6 + 2051 + 64 * 512 * 4 * 3 + 64 * 2051 * 4 * 3 + 64 * 512 * 2 * 2);
}
const gather_source: [:0]const u8 =
    \\const uint i=thread_position_in_grid.x;
    \\if(i>=uint(ROWS)*2051u*128u) return;
    \\const uint slot=i/128u,d=(i%128u)*4u,row=slot/2051u;
    \\const int id=selected[slot];
    \\const bool valid=id>=0 && uint(id)<uint(length) && uint(id)<=uint(offset)+row;
    \\if(d==0u) mask[slot]=valid;
    \\if(!valid) {for(uint j=0u;j<4u;++j) kv[size_t(slot)*512u+d+j]=OutT(0);return;}
    \\for(uint j=0u;j<4u;++j) kv[size_t(slot)*512u+d+j]=SUSHI_LATENT(cache,uint(id),d+j,512u);
;
var gather_kernels: [latent_store.variants]?mlx.mlx_fast_metal_kernel = @splat(null);
var gather_configs: [wide_rows]?mlx.mlx_fast_metal_kernel_config = @splat(null);
fn gather(ops: *Ops, cache: Latent, ids: Arr, offset: usize, history: usize) !struct { kv: Arr, mask: Arr } {
    const rows = mlx.getShape(ids)[0];
    const index: usize = @intCast(rows - 1);
    const cfg = gather_configs[index] orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &.{ rows, 2051, 512 }, 3, .bfloat16));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &.{ rows, 2051 }, 2, .bool_));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, rows * 2051 * 128, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 256, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "ROWS", rows));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "OutT", .bfloat16));
        gather_configs[index] = c;
        break :blk c;
    };
    const quantized = cache.quantized();
    const bits = cache.bits();
    const kernel = &gather_kernels[latent_store.variant(bits)];
    if (kernel.* == null) {
        const names: []const [*:0]const u8 = if (quantized) &.{ "cache", "cache_scales", "cache_biases", "selected", "offset", "length" } else &.{ "cache", "selected", "offset", "length" };
        const ins = mlx.mlx_vector_string_new_data(names.ptr, names.len);
        defer _ = mlx.mlx_vector_string_free(ins);
        const outs = mlx.mlx_vector_string_new_data(&.{ "kv", "mask" }, 2);
        defer _ = mlx.mlx_vector_string_free(outs);
        // Strides are explicitly admitted below; disable the wrapper's flag-
        // based copies so false row-contiguous flags cannot copy full history.
        const k = mlx.mlx_fast_metal_kernel_new(latent_store.kernelName("sushi_glm_sparse_head_gather", bits), ins, outs, gather_source, latent_store.header(bits), false, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        kernel.* = k;
    }
    const off: u32 = @intCast(offset);
    const len: u32 = @intCast(history);
    const oa = try ops.own(mlx.mlx_array_new_data(&off, &.{}, 0, .uint32));
    const la = try ops.own(mlx.mlx_array_new_data(&len, &.{}, 0, .uint32));
    const iv = if (quantized) mlx.mlx_vector_array_new_data(&.{ cache.data, cache.scales, cache.biases, ids, oa, la }, 6) else mlx.mlx_vector_array_new_data(&.{ cache.data, ids, oa, la }, 4);
    defer _ = mlx.mlx_vector_array_free(iv);
    var outputs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs, kernel.*.?, iv, cfg, ops.s));
    if (mlx.mlx_vector_array_size(outputs) != 2) return error.MetalKernelBadOutputCount;
    const kv = try ops.slot();
    try mlx.check(mlx.mlx_vector_array_get(kv, outputs, 0));
    const mask = try ops.slot();
    try mlx.check(mlx.mlx_vector_array_get(mask, outputs, 1));
    return .{ .kv = kv.*, .mask = mask.* };
}
fn geometry(q: []const c_int, cache: []const c_int, ids: []const c_int, offset: usize, history: usize) bool {
    return q.len == 3 and q[0] > 0 and (q[0] <= max_rows or q[0] == wide_rows) and q[1] == 64 and q[2] == 512 and
        cache.len == 2 and cache[0] >= history and cache[1] == 512 and ids.len == 2 and ids[0] == q[0] and ids[1] == 2051 and
        history > 0 and history <= 1048576 and offset <= history and q[0] <= history - offset;
}
/// Attention of q [R,64,512] over its own gathered bank [R,2051,512] in FP32 GEMMs and a precise
/// softmax, as [R,1,64,512] BF16. Off NAX MLX has no fused D512 SDPA.
pub fn composite(ops: *Ops, q: Arr, kv_bank: Arr, mask: Arr, scale: f32) !Arr {
    return compositeWith(ops, q, kv_bank, mask, scale, .matmul);
}

/// `block_masked` multiplies through MLX's block-masked GEMM, which never takes the NAX TF32 path,
/// so a NAX GPU's products stay FP32 whatever `MLX_ENABLE_TF32` says.
pub const Gemm = enum { matmul, block_masked };

fn gemm(ops: *Ops, a: Arr, b: Arr, how: Gemm) !Arr {
    if (how == .matmul) return ops.binary(.mm, a, b);
    const sa = mlx.getShape(a);
    const sb = mlx.getShape(b);
    const blocks = try ops.ones(&.{ sa[0], @divTrunc(sa[1] + 31, 32), @divTrunc(sb[2] + 31, 32) }, .bool_);
    const out = try ops.slot();
    try mlx.check(mlx.mlx_block_masked_mm(out, a, b, 32, blocks, .{ .ctx = null }, .{ .ctx = null }, ops.s));
    return out.*;
}

pub fn compositeWith(ops: *Ops, q: Arr, kv_bank: Arr, mask: Arr, scale: f32, how: Gemm) !Arr {
    const rows = mlx.getShape(q)[0];
    const kv = try ops.cast(kv_bank, .float32);
    const queries = try ops.binary(.mul, try ops.cast(q, .float32), try ops.scalar(scale, .float32));
    const masked = try ops.slot();
    try mlx.check(mlx.mlx_where(masked, try ops.reshape(mask, &.{ rows, 1, 2051 }), try gemm(ops, queries, try ops.transpose(kv, &.{ 0, 2, 1 }), how), try ops.scalar(-std.math.inf(f32), .float32), ops.s));
    return ops.reshape(try ops.cast(try gemm(ops, try ops.softmax(masked.*, -1), kv, how), .bfloat16), &.{ rows, 1, 64, 512 });
}

test "GLM block-masked composite batching preserves every B1 output bit" {
    const s = mlx.gpuStream();
    for (0..8) |case| {
        var ops = Ops{ .s = s };
        defer ops.deinit();
        const key = try ops.slot();
        try mlx.check(mlx.mlx_random_key(key, 1103 + case));
        const q = try ops.slot();
        try mlx.check(mlx.mlx_random_normal(q, &.{ 3, 64, 512 }, 3, .bfloat16, 0, if (case < 4) 0.05 else 2, key.*, s));
        const bank = try ops.reshape(try ops.own(try latent_store.randomRows(3 * 2051, 512, 1201 + case, s)), &.{ 3, 2051, 512 });
        var masks: [3 * 2051]bool = undefined;
        for (&masks, 0..) |*valid, i| valid.* = case % 2 == 0 or switch (i / 2051) {
            0 => i < 1024,
            1 => i % 7 == 0,
            else => i % 2051 == 2050,
        };
        const mask = try ops.own(mlx.mlx_array_new_data(&masks, &.{ 3, 2051 }, 2, .bool_));
        const batched = try compositeWith(&ops, q.*, bank, mask, 1.0 / 16.0, .block_masked);
        for (0..3) |r| {
            const at: c_int = @intCast(r);
            const one = try compositeWith(&ops, try ops.slice(q.*, 0, at, at + 1), try ops.slice(bank, 0, at, at + 1), try ops.slice(mask, 0, at, at + 1), 1.0 / 16.0, .block_masked);
            try @import("glm5_attention.zig").expectSameBits(one, try ops.slice(batched, 0, at, at + 1));
        }
    }
}
/// Valid IDs must be unique per real query, as guaranteed by IndexPool selection.
pub fn run(ops: *Ops, q: Arr, cache: Latent, selected: Arr, offset: usize, history: usize, scale: f32) !?Arr {
    const fused = @import("glm5_model.zig").naxArms();
    if (!mlx.streamIsGpu(ops.s) or !std.math.isFinite(scale) or scale <= 0) return null;
    for ([_]Arr{ q, cache.data, selected }) |a| if (a.ctx == null) return null;
    // Refuse a cache that would require the generic gather wrapper to copy
    // full history; actual State latent buffers have these contiguous strides.
    if (!cache.rowMajor()) return null;
    if (mlx.mlx_array_dtype(q) != .bfloat16 or cache.dtype() != .bfloat16 or mlx.mlx_array_dtype(selected) != .int32 or
        !geometry(mlx.getShape(q), &.{ cache.rows(), cache.width() }, mlx.getShape(selected), offset, history)) return null;
    const rows = mlx.getShape(q)[0];
    if (try (if (fused) temporaryBytes(@intCast(rows)) else compositeBytes(@intCast(rows))) > scratch_limit) return error.GlmPackedAttentionScratchBudget;
    const is = mlx.mlx_array_strides(selected);
    const ids = if (is[0] == 2051 and is[1] == 1) selected else try ops.contiguous(selected);
    const bank = try gather(ops, cache, ids, offset, history);
    const out = if (fused) blk: {
        const queries = try ops.reshape(q, &.{ rows, 1, 64, 512 });
        const kv = try ops.reshape(bank.kv, &.{ rows, 1, 2051, 512 });
        const mask = try ops.reshape(bank.mask, &.{ rows, 1, 1, 2051 });
        const attended = try ops.slot();
        // Fake Q positions are original heads; array mode has no position bias.
        // Q64/GQA1 selects native full D512 NAX, and force_fused forbids fallback.
        try mlx.check(mlx.mlx_fast_scaled_dot_product_attention(attended, queries, kv, kv, scale, "array", mask, .{ .ctx = null }, true, ops.s));
        break :blk attended.*;
    } else blk: {
        composite_calls += 1;
        if (composite_calls == 1) @import("log.zig").info("[glm-attn] FP32 composite sparse attention engaged\n", .{});
        break :blk try composite(ops, q, bank.kv, bank.mask, scale);
    };
    const populated = try ops.slot();
    try mlx.check(mlx.mlx_any_axes(populated, bank.mask, &.{1}, 1, true, ops.s));
    const safe = try ops.slot();
    try mlx.check(mlx.mlx_where(safe, try ops.reshape(populated.*, &.{ rows, 1, 1, 1 }), out, try ops.scalar(0, .bfloat16), ops.s));
    const result = try ops.reshape(safe.*, &.{ rows, 64, 512 });
    calls += 1;
    return result;
}
test "GLM packed NAX geometry and fixed scratch bound" {
    try std.testing.expect(geometry(&.{ 16, 64, 512 }, &.{ 32768, 512 }, &.{ 16, 2051 }, 32752, 32768));
    try std.testing.expect(!geometry(&.{ 17, 64, 512 }, &.{ 32768, 512 }, &.{ 17, 2051 }, 32751, 32768));
    try std.testing.expect(!geometry(&.{ 16, 64, 512 }, &.{ 32768, 512 }, &.{ 16, 2051 }, 32753, 32768));
    try std.testing.expect(try temporaryBytes(16) <= scratch_limit);
}
test "GLM packed NAX bills every pending layer" {
    try std.testing.expectEqual(scratch_limit, try transientBudget(16, 1));
    try std.testing.expectEqual(scratch_limit * 2, try transientBudget(2048, 2));
    try std.testing.expectEqual(@as(usize, 0), try transientBudget(8, 2));
    try std.testing.expectError(error.Overflow, transientBudget(2048, std.math.maxInt(usize)));
    const model = @import("glm5_model.zig");
    model.reference_numerics = true;
    defer model.reference_numerics = false;
    try std.testing.expectEqual(@as(usize, 0), try transientBudget(2048, 2));
}
test "GLM packed NAX invalid gather cannot poison output with masked nonfinite key zero" {
    var ops = Ops{ .s = mlx.gpuStream() };
    defer ops.deinit();
    var data: [4 * 512]u16 = @splat(0);
    for (0..512) |i| {
        data[i] = if (i % 2 == 0) 0x7fc0 else 0x7f80;
        data[512 + i] = 0x3e80; // exact BF16 0.25
    }
    const cache = Latent{ .data = try ops.own(mlx.mlx_array_new_data(&data, &.{ 4, 512 }, 2, .bfloat16)) };
    var indices: [2 * 2051]i32 = @splat(-1);
    indices[0] = std.math.minInt(i32);
    indices[1] = std.math.maxInt(i32);
    indices[2] = 3; // future at real query position1
    indices[2051 + 100] = 1;
    const selected = try ops.own(mlx.mlx_array_new_data(&indices, &.{ 2, 2051 }, 2, .int32));
    const q = try ops.zeros(&.{ 2, 64, 512 }, .bfloat16);
    const out = (try run(&ops, q, cache, selected, 1, 4, 1.0 / 16.0)) orelse return error.ExpectedPackedNax;
    try mlx.check(mlx.mlx_array_eval(out));
    const bits = mlx.mlx_array_data_bfloat16(out).?;
    for (0..64 * 512) |i| {
        try std.testing.expectEqual(@as(u16, 0), bits[i]);
        try std.testing.expectEqual(@as(u16, 0x3e80), bits[64 * 512 + i]);
    }
}
test "GLM packed NAX preserves head order real-row selection and final ragged slot" {
    var ops = Ops{ .s = mlx.gpuStream() };
    defer ops.deinit();
    var data: [4 * 512]u16 = @splat(0);
    data[0] = 0x3f80; // BF16 +1
    data[512] = 0xbf80; // BF16 -1
    const cache = Latent{ .data = try ops.own(mlx.mlx_array_new_data(&data, &.{ 4, 512 }, 2, .bfloat16)) };
    var query: [2 * 64 * 512]u16 = @splat(0);
    for (0..2) |row| for (0..64) |head| {
        query[(row * 64 + head) * 512] = if (head % 2 == 0) 0x4180 else 0xc180; // +/-16
    };
    const q = try ops.own(mlx.mlx_array_new_data(&query, &.{ 2, 64, 512 }, 3, .bfloat16));
    var indices: [2 * 2051]i32 = @splat(-1);
    indices[0] = 0;
    indices[2050] = 1; // last ragged K position must participate
    indices[2051 + 10] = 0; // a different real query has a different key set
    const selected = try ops.own(mlx.mlx_array_new_data(&indices, &.{ 2, 2051 }, 2, .int32));
    const out = (try run(&ops, q, cache, selected, 1, 4, 1.0 / 16.0)) orelse return error.ExpectedPackedNax;
    try mlx.check(mlx.mlx_array_eval(out));
    const bits = mlx.mlx_array_data_bfloat16(out).?;
    for (0..64) |head| {
        const value: f32 = @bitCast(@as(u32, bits[head * 512]) << 16);
        if (head % 2 == 0) try std.testing.expect(value > 0.7 and value < 0.8) else try std.testing.expect(value < -0.7 and value > -0.8);
        try std.testing.expectEqual(@as(u16, 0x3f80), bits[(64 + head) * 512]);
        for (1..512) |d| {
            try std.testing.expectEqual(@as(u16, 0), bits[head * 512 + d]);
            try std.testing.expectEqual(@as(u16, 0), bits[(64 + head) * 512 + d]);
        }
    }
}

test "GLM packed32 geometry and conservative graph bill" {
    try std.testing.expect(geometry(&.{ 32, 64, 512 }, &.{ 16384, 512 }, &.{ 32, 2051 }, 16352, 16384));
    for ([_]c_int{ 17, 24, 31 }) |rows| try std.testing.expect(!geometry(&.{ rows, 64, 512 }, &.{ 16384, 512 }, &.{ rows, 2051 }, 16352, 16384));
    try std.testing.expect(!geometry(&.{ 32, 64, 512 }, &.{ 16384, 512 }, &.{ 32, 2051 }, 16353, 16384));
    try std.testing.expect(try temporaryBytes(32) <= scratch_limit);
    for ([_]usize{ 17, 24, 31 }) |rows| try std.testing.expectError(error.UnsupportedGlmPackedAttention, temporaryBytes(rows));
}
