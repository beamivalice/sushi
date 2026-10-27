//! Mode-matched native decode attention for B1 and up to eight batched branches.
const std = @import("std");
const mlx = @import("mlx.zig");
const Ops = @import("glm5_model.zig").Ops;
const latent_store = @import("glm5_latent.zig");
const packed_attention = @import("glm5_attention_nax_packed.zig");
const Latent = latent_store.Latent;
const Arr = mlx.mlx_array;
var b1_calls: usize = 0;
var b3_calls: usize = 0;
var logged = false;
pub fn enabled() bool {
    return !@import("glm5_model.zig").reference_numerics;
}
/// Per pending layer: the FP32 composite's bounded branch planes.
pub fn scratchLimit() usize {
    return scratchLimitFor(4);
}
/// Batches past five rows hold a wider FP32 bank; only a verify forms them.
pub fn scratchLimitFor(rows: usize) usize {
    return (if (rows <= 5) @as(usize, 32) else 64) * 1024 * 1024;
}
pub fn supportedConfig(cfg: *const @import("model.zig").ModelConfig, dtype: mlx.mlx_dtype, s: mlx.mlx_stream) bool {
    return cfg.num_attention_heads == 64 and cfg.mla_kv_lora_rank == 512 and cfg.mla_qk_nope_head_dim == 256 and
        cfg.indexer_n_heads == 32 and cfg.indexer_head_dim == 128 and dtype == .bfloat16 and mlx.streamIsGpu(s);
}
pub fn supportedQuery(shape: []const c_int, dtype: mlx.mlx_dtype, s: mlx.mlx_stream) bool {
    return shape.len == 3 and shape[0] > 0 and shape[0] <= 8 and shape[1] == 64 and shape[2] == 512 and
        dtype == .bfloat16 and mlx.streamIsGpu(s);
}
pub fn b1Calls() usize {
    return b1_calls;
}
pub fn b3Calls() usize {
    return b3_calls;
}
pub fn resetCalls() void {
    b1_calls = 0;
    b3_calls = 0;
    logged = false;
}
/// Ancestry rows a branch may read from the verification tape (the gather's `paths` stride).
pub const max_tail = 8;
pub const Branch = struct {
    offset: usize,
    length: usize,
    path: [max_tail]u32,
};
pub fn pathOf(ids: []const u32) [max_tail]u32 {
    var path: [max_tail]u32 = @splat(0);
    @memcpy(path[0..ids.len], ids);
    return path;
}
pub fn temporaryBytes(batch: usize) !usize {
    if (batch == 0 or batch > max_tail) return error.UnsupportedGlmDecodeBatch;
    const bytes = try packed_attention.compositeBytes(batch);
    // Multi-row gathers write the FP32 bank directly, omitting the BF16 intermediate.
    return if (batch >= 2) bytes - batch * 2051 * 512 * 2 else bytes;
}
/// Serial rows run as B1; a verify's wider batch is billed inside `glm5_dflash_memory.limit_bytes`.
pub fn transientBudget(pending_layers: usize) !usize {
    if (!enabled()) return 0;
    return std.math.mul(usize, scratchLimit(), pending_layers);
}
fn geometry(q: []const c_int, prefix: []const c_int, prefix_rows: usize, tape: []const c_int, ids: []const c_int, branches: []const Branch) bool {
    if (q.len != 3 or q[0] < 1 or q[0] > max_tail or q[1] != 64 or q[2] != 512 or branches.len != q[0] or
        prefix.len != 2 or prefix[0] < prefix_rows or prefix[1] != 512 or prefix_rows > 1048576 or
        tape.len != 2 or tape[0] < 1 or tape[0] > max_tail or tape[1] != 512 or
        ids.len != 2 or ids[0] != q[0] or ids[1] != 2051) return false;
    for (branches) |branch| {
        if (branch.length <= prefix_rows or branch.length - prefix_rows > tape[0] or branch.offset != branch.length - 1) return false;
        const depth = branch.length - prefix_rows;
        for (branch.path[0..depth], 0..) |node, i| {
            if (node >= tape[0]) return false;
            for (branch.path[0..i]) |prior| if (node == prior) return false;
        }
    }
    return true;
}
const gather_source: [:0]const u8 =
    \\const uint i=thread_position_in_grid.x;
    \\if(i>=uint(batch_count)*2051u*128u) return;
    \\const uint slot=i/128u,d=(i%128u)*4u,row=slot/2051u;
    \\const int id=selected[slot];
    \\const bool valid=id>=0 && uint(id)<lengths[row] && uint(id)<=offsets[row];
    \\if(d==0u) mask[slot]=valid;
    \\if(!valid) {for(uint j=0u;j<4u;++j) kv[size_t(slot)*512u+d+j]=OutT(0);return;}
    \\if(uint(id)<uint(prefix_length)) {for(uint j=0u;j<4u;++j) kv[size_t(slot)*512u+d+j]=SUSHI_LATENT(prefix,uint(id),d+j,512u);return;}
    \\const device bfloat* values=tape+size_t(paths[row*uint(TAIL)+uint(id)-uint(prefix_length)])*512u;
    \\for(uint j=0u;j<4u;++j) kv[size_t(slot)*512u+d+j]=values[d+j];
;
var gather_kernels: [latent_store.variants]?mlx.mlx_fast_metal_kernel = @splat(null);
var configs: [max_tail]?mlx.mlx_fast_metal_kernel_config = @splat(null);
pub fn run(ops: *Ops, q: Arr, prefix: Latent, prefix_rows: usize, tape: Arr, branches: []const Branch, selected: Arr, scale: f32) !?Arr {
    if (!mlx.streamIsGpu(ops.s) or !std.math.isFinite(scale) or scale <= 0) return null;
    for ([_]Arr{ q, prefix.data, tape, selected }) |a| if (a.ctx == null) return null;
    if (!prefix.rowMajor() or prefix.dtype() != .bfloat16) return null;
    for ([_]Arr{ q, tape }) |a| if (mlx.mlx_array_dtype(a) != .bfloat16) return null;
    if (mlx.mlx_array_dtype(selected) != .int32 or !geometry(mlx.getShape(q), &.{ prefix.rows(), prefix.width() }, prefix_rows, mlx.getShape(tape), mlx.getShape(selected), branches)) return null;
    const ts = mlx.mlx_array_strides(tape);
    if (ts[0] != 512 or ts[1] != 1) return null;
    const qs = mlx.mlx_array_strides(q);
    if (qs[1] != 512 or qs[2] != 1) return null;
    const batch: c_int = @intCast(branches.len);
    if (try temporaryBytes(branches.len) > scratchLimitFor(branches.len)) return error.GlmDecodeBatchScratchBudget;
    var offsets: [max_tail]u32 = undefined;
    var lengths: [max_tail]u32 = undefined;
    var paths: [max_tail * max_tail]u32 = undefined;
    for (branches, 0..) |branch, row| {
        offsets[row] = @intCast(branch.offset);
        lengths[row] = @intCast(branch.length);
        @memcpy(paths[row * max_tail ..][0..max_tail], &branch.path);
    }
    const oa = try ops.own(mlx.mlx_array_new_data(&offsets, &.{batch}, 1, .uint32));
    const la = try ops.own(mlx.mlx_array_new_data(&lengths, &.{batch}, 1, .uint32));
    const pa = try ops.own(mlx.mlx_array_new_data(&paths, &.{ batch, max_tail }, 2, .uint32));
    const base: u32 = @intCast(prefix_rows);
    const ba = try ops.own(mlx.mlx_array_new_data(&base, &.{}, 0, .uint32));
    const batches: u32 = @intCast(branches.len);
    const batches_array = try ops.own(mlx.mlx_array_new_data(&batches, &.{}, 0, .uint32));
    const ids = try ops.contiguous(selected);
    const cfg_index: usize = @intCast(batch - 1);
    // SUSHI_LATENT still returns BF16, including kv8's rounding boundary.
    const bank_dtype: mlx.mlx_dtype = if (batch >= 2) .float32 else .bfloat16;
    const cfg = configs[cfg_index] orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &.{ batch, 2051, 512 }, 3, bank_dtype));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &.{ batch, 2051 }, 2, .bool_));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, batch * 2051 * 128, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 256, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(c, "OutT", bank_dtype));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "TAIL", max_tail));
        configs[cfg_index] = c;
        break :blk c;
    };
    const quantized = prefix.quantized();
    const bits = prefix.bits();
    const kernel = &gather_kernels[latent_store.variant(bits)];
    if (kernel.* == null) {
        const names: []const [*:0]const u8 = if (quantized)
            &.{ "prefix", "prefix_scales", "prefix_biases", "tape", "selected", "offsets", "lengths", "paths", "prefix_length", "batch_count" }
        else
            &.{ "prefix", "tape", "selected", "offsets", "lengths", "paths", "prefix_length", "batch_count" };
        const inputs = mlx.mlx_vector_string_new_data(names.ptr, names.len);
        defer _ = mlx.mlx_vector_string_free(inputs);
        const outputs = mlx.mlx_vector_string_new_data(&.{ "kv", "mask" }, 2);
        defer _ = mlx.mlx_vector_string_free(outputs);
        // Admitted strides avoid wrapper copies of the immutable full prefix.
        const k = mlx.mlx_fast_metal_kernel_new(latent_store.kernelName("sushi_glm_decode_batch_gather", bits), inputs, outputs, gather_source, latent_store.header(bits), false, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        kernel.* = k;
    }
    const inputs = if (quantized)
        mlx.mlx_vector_array_new_data(&.{ prefix.data, prefix.scales, prefix.biases, tape, ids, oa, la, pa, ba, batches_array }, 10)
    else
        mlx.mlx_vector_array_new_data(&.{ prefix.data, tape, ids, oa, la, pa, ba, batches_array }, 8);
    defer _ = mlx.mlx_vector_array_free(inputs);
    var outputs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs, kernel.*.?, inputs, cfg, ops.s));
    const bank = try ops.slot();
    const mask = try ops.slot();
    try mlx.check(mlx.mlx_vector_array_get(bank, outputs, 0));
    try mlx.check(mlx.mlx_vector_array_get(mask, outputs, 1));
    // A NAX GPU runs FP32 GEMMs as TF32 by default; the block-masked GEMM keeps them FP32 there.
    const how: packed_attention.Gemm = if (@import("glm5_model.zig").naxArms()) .block_masked else .matmul;
    if (!logged) {
        @import("log.zig").info("[glm-attn] FP32 composite native decode attention engaged ({s})\n", .{@tagName(how)});
        logged = true;
    }
    // Block-masked GEMM fixes its tiles at 32x32x16 regardless of batch size.
    // Ordinary GEMM changes tiles/split-K with the batch, so retain B1 there.
    const out = if (how == .block_masked and batch >= 2)
        try packed_attention.compositeWith(ops, q, bank.*, mask.*, scale, how)
    else blk: {
        var rows: [max_tail]Arr = undefined;
        for (rows[0..branches.len], 0..) |*row, r| {
            const at: c_int = @intCast(r);
            row.* = try packed_attention.compositeWith(ops, try ops.slice(q, 0, at, at + 1), try ops.slice(bank.*, 0, at, at + 1), try ops.slice(mask.*, 0, at, at + 1), scale, how);
        }
        break :blk if (branches.len == 1) rows[0] else try ops.concat(rows[0..branches.len], 0);
    };
    const populated = try ops.slot();
    try mlx.check(mlx.mlx_any_axes(populated, mask.*, &.{1}, 1, true, ops.s));
    const safe = try ops.slot();
    try mlx.check(mlx.mlx_where(safe, try ops.reshape(populated.*, &.{ batch, 1, 1, 1 }), out, try ops.scalar(0, .bfloat16), ops.s));
    const result = try ops.reshape(safe.*, &.{ batch, 64, 512 });
    if (batch == 1) b1_calls += 1 else if (batch == 3) b3_calls += 1;
    return result;
}
// One serial row reads its whole history, current row included, from the cache. The FP32 bank is written
// directly (SUSHI_LATENT is BF16, so widening is exact), the query is widened and scaled beside it, and a
// sparse row expands its 512 pools to tokens here exactly as `sushi_glm_expand_pools` does.
const row_gather_source: [:0]const u8 =
    \\const uint i=thread_position_in_grid.x;
    \\if(i<32768u) qs[i]=float(q[i])*float(qscale);
    \\if(i>=2051u*128u) return;
    \\const uint slot=i/128u,d=(i%128u)*4u;
    \\const uint last=uint(offset);
    \\int id;
    \\if(POOLED==1) {
    \\ if(slot<2048u) {
    \\  const uint p=selected[slot>>2];
    \\  id=(p+1u)*4u<=last+1u?int(p*4u+(slot&3u)):-1;
    \\ } else {
    \\  const uint rem=(last+1u)%4u;
    \\  id=slot-2048u<rem?int(last+1u-rem+slot-2048u):-1;
    \\ }
    \\} else id=int(selected[slot]);
    \\const bool valid=id>=0 && uint(id)<=last;
    \\if(d==0u) mask[slot]=valid;
    \\for(uint j=0u;j<4u;++j) kv[size_t(slot)*512u+d+j]=valid?float(SUSHI_LATENT(cache,uint(id),d+j,512u)):0.0f;
;
var row_gather_kernels: [2][latent_store.variants]?mlx.mlx_fast_metal_kernel = @splat(@splat(null));
var row_gather_configs: [2]?mlx.mlx_fast_metal_kernel_config = .{ null, null };
var row_calls: usize = 0;
var block_masks: [2]Arr = .{ .{ .ctx = null }, .{ .ctx = null } };
/// The block-masked GEMM wants an all-true block mask per shape; built and settled once.
fn blockMask(slot: usize, s: mlx.mlx_stream) !Arr {
    if (block_masks[slot].ctx == null) {
        var mask = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(mask);
        const shapes = [2][3]c_int{ .{ 1, 2, 65 }, .{ 1, 2, 16 } };
        try mlx.check(mlx.mlx_ones(&mask, &shapes[slot], 3, .bool_, s));
        try mlx.check(mlx.mlx_array_eval(mask));
        block_masks[slot] = mask;
    }
    return block_masks[slot];
}
pub fn rowCalls() usize {
    return row_calls;
}
/// What a serial row attends to: its `decodeSelected` tokens, or the 512 pools they expand from.
pub const RowSelection = union(enum) { ids: Arr, pools: Arr };
/// Whether `runRow` takes this row: the fused arithmetic matches B1's block-masked GEMMs only on a NAX GPU.
pub fn rowSupported(q: Arr, cache: Latent, offset: usize, s: mlx.mlx_stream) bool {
    if (!mlx.streamIsGpu(s) or !@import("glm5_model.zig").naxArms() or q.ctx == null or cache.data.ctx == null) return false;
    if (!cache.rowMajor() or cache.dtype() != .bfloat16 or mlx.mlx_array_dtype(q) != .bfloat16) return false;
    const qs = mlx.getShape(q);
    const strides = mlx.mlx_array_strides(q);
    return qs.len == 3 and qs[0] == 1 and qs[1] == 64 and qs[2] == 512 and strides[1] == 512 and strides[2] == 1 and
        cache.width() == 512 and offset < @as(usize, @intCast(cache.rows()));
}
/// `run` for one serial row (offset = the cache's last row) with the same arithmetic and bytes as B1: the
/// FP32 bank is gathered once instead of widened from a BF16 bank, the tail row comes from the cache itself
/// (SUSHI_LATENT equals MLX's dequantizer) and the block masks are built once. The selection must come
/// from `decodeSelected`, whose first slot is never invalid, so `run`'s all-invalid zeroing is a no-op here.
pub fn runRow(ops: *Ops, q: Arr, cache: Latent, offset: usize, selection: RowSelection, scale: f32) !?Arr {
    if (!rowSupported(q, cache, offset, ops.s) or !std.math.isFinite(scale) or scale <= 0) return null;
    const pooled = selection == .pools;
    const selected = switch (selection) {
        .ids => |a| a,
        .pools => |a| a,
    };
    const ss = mlx.getShape(selected);
    if (selected.ctx == null or ss.len != 2 or ss[0] != 1) return null;
    if (pooled) {
        if (ss[1] != 512 or mlx.mlx_array_dtype(selected) != .uint32) return null;
    } else if (ss[1] != 2051 or mlx.mlx_array_dtype(selected) != .int32) return null;
    const off: u32 = @intCast(offset);
    const oa = try ops.own(mlx.mlx_array_new_data(&off, &.{}, 0, .uint32));
    const qa = try ops.own(mlx.mlx_array_new_data(&scale, &.{}, 0, .float32));
    const ids = try ops.contiguous(selected);
    const mode = @intFromBool(pooled);
    const cfg = row_gather_configs[mode] orelse blk: {
        const c = mlx.mlx_fast_metal_kernel_config_new();
        errdefer _ = mlx.mlx_fast_metal_kernel_config_free(c);
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &.{ 1, 2051, 512 }, 3, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &.{ 1, 2051 }, 2, .bool_));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(c, &.{ 1, 64, 512 }, 3, .float32));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(c, 2051 * 128, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(c, 256, 1, 1));
        try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(c, "POOLED", mode));
        row_gather_configs[mode] = c;
        break :blk c;
    };
    const quantized = cache.quantized();
    const bits = cache.bits();
    const kernel = &row_gather_kernels[mode][latent_store.variant(bits)];
    if (kernel.* == null) {
        const names: []const [*:0]const u8 = if (quantized)
            &.{ "cache", "cache_scales", "cache_biases", "selected", "offset", "q", "qscale" }
        else
            &.{ "cache", "selected", "offset", "q", "qscale" };
        const inputs = mlx.mlx_vector_string_new_data(names.ptr, names.len);
        defer _ = mlx.mlx_vector_string_free(inputs);
        const outputs = mlx.mlx_vector_string_new_data(&.{ "kv", "mask", "qs" }, 3);
        defer _ = mlx.mlx_vector_string_free(outputs);
        const k = mlx.mlx_fast_metal_kernel_new(latent_store.kernelName("sushi_glm_decode_row_gather", bits), inputs, outputs, row_gather_source, latent_store.header(bits), false, false);
        if (k.ctx == null) return error.MetalKernelCompileFailed;
        kernel.* = k;
    }
    const inputs = if (quantized)
        mlx.mlx_vector_array_new_data(&.{ cache.data, cache.scales, cache.biases, ids, oa, q, qa }, 7)
    else
        mlx.mlx_vector_array_new_data(&.{ cache.data, ids, oa, q, qa }, 5);
    defer _ = mlx.mlx_vector_array_free(inputs);
    var outputs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs, kernel.*.?, inputs, cfg, ops.s));
    const bank = try ops.slot();
    const mask = try ops.slot();
    const queries = try ops.slot();
    try mlx.check(mlx.mlx_vector_array_get(bank, outputs, 0));
    try mlx.check(mlx.mlx_vector_array_get(mask, outputs, 1));
    try mlx.check(mlx.mlx_vector_array_get(queries, outputs, 2));
    const scores = try ops.slot();
    try mlx.check(mlx.mlx_block_masked_mm(scores, queries.*, try ops.transpose(bank.*, &.{ 0, 2, 1 }), 32, try blockMask(0, ops.s), .{ .ctx = null }, .{ .ctx = null }, ops.s));
    const masked = try ops.slot();
    try mlx.check(mlx.mlx_where(masked, try ops.reshape(mask.*, &.{ 1, 1, 2051 }), scores.*, try ops.scalar(-std.math.inf(f32), .float32), ops.s));
    const mixed = try ops.slot();
    try mlx.check(mlx.mlx_block_masked_mm(mixed, try ops.softmax(masked.*, -1), bank.*, 32, try blockMask(1, ops.s), .{ .ctx = null }, .{ .ctx = null }, ops.s));
    b1_calls += 1;
    row_calls += 1;
    return try ops.reshape(try ops.cast(mixed.*, .bfloat16), &.{ 1, 64, 512 });
}

test "GLM decode batch geometry ancestry and conservative scratch" {
    const branches = [_]Branch{
        .{ .offset = 16381, .length = 16382, .path = pathOf(&.{0}) },
        .{ .offset = 16382, .length = 16383, .path = pathOf(&.{ 0, 1 }) },
        .{ .offset = 16382, .length = 16383, .path = pathOf(&.{ 0, 2 }) },
    };
    try std.testing.expect(geometry(&.{ 3, 64, 512 }, &.{ 16384, 512 }, 16381, &.{ 3, 512 }, &.{ 3, 2051 }, &branches));
    try std.testing.expect(!geometry(&.{ 0, 64, 512 }, &.{ 16384, 512 }, 16381, &.{ 3, 512 }, &.{ 0, 2051 }, &.{}));
    var bad = branches;
    bad[2].offset += 1;
    try std.testing.expect(!geometry(&.{ 3, 64, 512 }, &.{ 16384, 512 }, 16381, &.{ 3, 512 }, &.{ 3, 2051 }, &bad));
    bad = branches;
    bad[2].path = pathOf(&.{});
    try std.testing.expect(!geometry(&.{ 3, 64, 512 }, &.{ 16384, 512 }, 16381, &.{ 3, 512 }, &.{ 3, 2051 }, &bad));
    try std.testing.expect(try temporaryBytes(3) <= scratchLimit());
    try std.testing.expect(try temporaryBytes(4) <= scratchLimit());
    for (5..max_tail + 1) |rows| {
        try std.testing.expect(try temporaryBytes(rows) <= scratchLimitFor(rows));
        try std.testing.expectEqual(rows > 5, try temporaryBytes(rows) > scratchLimit());
    }
    try std.testing.expectError(error.UnsupportedGlmDecodeBatch, temporaryBytes(max_tail + 1));
    try std.testing.expectEqual(scratchLimit() * 4, try transientBudget(4));
    try std.testing.expect(scratchLimitFor(8) > scratchLimit());
    try std.testing.expectError(error.Overflow, transientBudget(std.math.maxInt(usize)));
    const model = @import("glm5_model.zig");
    model.reference_numerics = true;
    defer model.reference_numerics = false;
    try std.testing.expectEqual(@as(usize, 0), try transientBudget(4));
}

test "GLM kv8 and kv6 decode batches match BF16 and independent B1 branches" {
    for ([_]u8{ latent_store.kv8_bits, 6 }) |bits| for ([_]usize{ 2, 3, 4, 5, 6, 7, 8 }) |rows| for ([_]usize{ 5, 3000 }) |prefix_rows| for ([_]bool{ false, true }) |chain| for ([_]f32{ 0.05, 2 }) |spread| {
        try decodeBatchParity(bits, rows, prefix_rows, chain, spread);
    };
}

fn decodeBatchParity(bits: u8, rows: usize, prefix_rows: usize, chain: bool, spread: f32) !void {
    const s = mlx.gpuStream();
    var ops = Ops{ .s = s };
    defer ops.deinit();
    var q = try latent_store.quantize(try ops.own(try latent_store.randomRows(@intCast(prefix_rows), 512, 61, s)), bits, s);
    defer q.deinit();
    const kv8 = Latent{ .data = q.q, .scales = q.scales, .biases = q.biases };
    const bf16 = Latent{ .data = try ops.own(try kv8.dense(0, @intCast(prefix_rows), s)) };
    const tape = try ops.own(try latent_store.readable(try ops.own(try latent_store.randomRows(@intCast(rows), 512, 62, s)), bits, s));
    var branches: [max_tail]Branch = undefined;
    for (branches[0..rows], 0..) |*branch, row| {
        const depth = if (chain) row + 1 else if (row == 0) @as(usize, 1) else 2;
        branch.* = .{ .offset = prefix_rows + depth - 1, .length = prefix_rows + depth, .path = @splat(0) };
        if (chain) {
            for (0..depth) |j| branch.path[j] = @intCast(j);
        } else if (row > 0) {
            branch.path[1] = @intCast(row);
        }
    }
    var ids: [max_tail * 2051]i32 = undefined;
    for (branches[0..rows], 0..) |branch, row| for (0..2051) |k| {
        ids[row * 2051 + k] = if (k < @min(branch.length, 2051)) @intCast((branch.length -| 2051) + k) else -1;
    };
    const selected = try ops.own(mlx.mlx_array_new_data(&ids, &.{ @intCast(rows), 2051 }, 2, .int32));
    const key = try ops.slot();
    try mlx.check(mlx.mlx_random_key(key, 63));
    const query = try ops.slot();
    try mlx.check(mlx.mlx_random_normal(query, &.{ @intCast(rows), 64, 512 }, 3, .bfloat16, 0, spread, key.*, s));
    resetCalls();
    const got = (try run(&ops, query.*, kv8, prefix_rows, tape, branches[0..rows], selected, 1.0 / 16.0)) orelse return error.ExpectedNativeDecode;
    const want = (try run(&ops, query.*, bf16, prefix_rows, tape, branches[0..rows], selected, 1.0 / 16.0)) orelse return error.ExpectedNativeDecode;
    try @import("glm5_attention.zig").expectSameBits(want, got);
    for (0..rows) |row| {
        // A fresh arena per row: without NAX the batch runs as per-row composites and one arena would overflow at 8 rows.
        var row_ops = Ops{ .s = s };
        defer row_ops.deinit();
        const at: c_int = @intCast(row);
        const one = try row_ops.slice(query.*, 0, at, at + 1);
        const one_ids = try row_ops.slice(selected, 0, at, at + 1);
        const got1 = (try run(&row_ops, one, kv8, prefix_rows, tape, branches[row .. row + 1], one_ids, 1.0 / 16.0)) orelse return error.ExpectedNativeDecode;
        const want1 = (try run(&row_ops, one, bf16, prefix_rows, tape, branches[row .. row + 1], one_ids, 1.0 / 16.0)) orelse return error.ExpectedNativeDecode;
        try @import("glm5_attention.zig").expectSameBits(want1, got1);
        try @import("glm5_attention.zig").expectSameBits(got1, try row_ops.slice(got, 0, at, at + 1));
    }
    try std.testing.expectEqual(@as(usize, if (rows == 3) 2 else 0), b3Calls());
    try std.testing.expectEqual(rows * 2, b1Calls());
}

test "GLM B1 reads a four-row ancestry tail exactly as those rows committed to the prefix" {
    const s = mlx.gpuStream();
    var ops = Ops{ .s = s };
    defer ops.deinit();
    const prefix_rows = 3000;
    const rows = try ops.own(try latent_store.randomRows(prefix_rows + 4, 512, 71, s));
    const committed = Latent{ .data = rows };
    const prefix = Latent{ .data = try ops.slice(rows, 0, 0, prefix_rows) };
    const tail4 = try ops.contiguous(try ops.slice(rows, 0, prefix_rows, prefix_rows + 4));
    const tail1 = try ops.contiguous(try ops.slice(rows, 0, prefix_rows + 3, prefix_rows + 4));
    const length = prefix_rows + 4;
    var ids: [2051]i32 = undefined;
    for (&ids, 0..) |*id, k| id.* = @intCast(length - 2051 + k);
    const selected = try ops.own(mlx.mlx_array_new_data(&ids, &.{ 1, 2051 }, 2, .int32));
    var key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_key(&key, 72));
    const query = try ops.slot();
    try mlx.check(mlx.mlx_random_normal(query, &.{ 1, 64, 512 }, 3, .bfloat16, 0, 0.05, key, s));
    const deep = Branch{ .offset = length - 1, .length = length, .path = pathOf(&.{ 0, 1, 2, 3 }) };
    const flat = Branch{ .offset = length - 1, .length = length, .path = pathOf(&.{}) };
    const got = (try run(&ops, query.*, prefix, prefix_rows, tail4, &.{deep}, selected, 1.0 / 16.0)) orelse return error.ExpectedNativeDecode;
    const want = (try run(&ops, query.*, committed, prefix_rows + 3, tail1, &.{flat}, selected, 1.0 / 16.0)) orelse return error.ExpectedNativeDecode;
    try @import("glm5_attention.zig").expectSameBits(want, got);
}
