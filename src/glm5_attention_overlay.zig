//! Verification-only latent prefix plus at most four ancestry rows.
const std = @import("std");
const mlx = @import("mlx.zig");
const Ops = @import("glm5_model.zig").Ops;
const latent_store = @import("glm5_latent.zig");
const Latent = latent_store.Latent;
const Arr = mlx.mlx_array;

/// Ancestry rows a branch overlays: the native gather's tail bound.
pub const max_rows = @import("glm5_attention_decode_batch.zig").max_tail;

pub fn tailBytes(width: usize, rows: usize, bytes: usize) !usize {
    if (width == 0 or rows == 0 or rows > max_rows or (bytes != 2 and bytes != 4)) return error.InvalidGlmOverlay;
    return std.math.mul(usize, try std.math.mul(usize, rows, width), bytes);
}
/// `tail` holds rows as attention reads them: under kv8 or kv6, their quantize-dequantize round trip.
pub const View = struct {
    prefix: Latent,
    prefix_rows: usize,
    tail: Arr,
    pub fn storage(self: View) Latent {
        return if (self.prefix_rows == 0) .{ .data = self.tail } else self.prefix;
    }
    pub fn length(self: View) usize {
        return self.prefix_rows + @as(usize, @intCast(mlx.getShape(self.tail)[0]));
    }
    pub fn validate(self: View, q: Arr) !void {
        if (self.tail.ctx == null or q.ctx == null or self.prefix_rows > 1048576) return error.InvalidGlmOverlay;
        const qs = mlx.getShape(q);
        const ts = mlx.getShape(self.tail);
        if (qs.len != 3 or qs[0] != 1 or qs[1] < 1 or qs[2] < 1 or ts.len != 2 or ts[0] < 1 or ts[0] > max_rows or ts[1] != qs[2] or
            mlx.mlx_array_dtype(self.tail) != mlx.mlx_array_dtype(q)) return error.InvalidGlmOverlay;
        _ = try tailBytes(@intCast(qs[2]), @intCast(ts[0]), mlx.mlx_array_itemsize(q));
        const qt = mlx.mlx_array_strides(q);
        const tt = mlx.mlx_array_strides(self.tail);
        if (qt[1] != qs[2] or qt[2] != 1 or tt[0] != qs[2] or tt[1] != 1) return error.InvalidGlmOverlay;
        if (self.prefix_rows != 0) {
            const p = self.prefix;
            if (!p.rowMajor() or p.rows() < self.prefix_rows or p.width() != qs[2] or p.dtype() != mlx.mlx_array_dtype(q)) return error.InvalidGlmOverlay;
        }
    }
};
const shader_source: [:0]const u8 =
    \\#pragma clang fp contract(off)
    \\const uint lane=thread_position_in_threadgroup.x;
    \\const uint head=threadgroup_position_in_grid.y;
    \\const uint row=threadgroup_position_in_grid.z/uint(SPLITS);
    \\const uint part=threadgroup_position_in_grid.z%uint(SPLITS);
    \\constexpr uint ITEMS=(uint(D)+31u)/32u;
    \\float query[ITEMS],acc[ITEMS];
    \\for(uint j=0;j<ITEMS;++j) {uint d=lane+j*32u; query[j]=d<uint(D)?float(q[(row*uint(H)+head)*uint(D)+d]):0.0f;acc[j]=0.0f;}
    \\const uint pos=uint(offset)+row;
    \\const uint count=SELECTED?2051u:pos+1u;
    \\const uint chunk=(count+uint(SPLITS)-1u)/uint(SPLITS);
    \\const uint begin=part*chunk,end=min(count,begin+chunk);
    \\float maximum=-INFINITY,denom=0.0f;
    \\for(uint k=begin;k<end;++k) {
    \\  const int token=SELECTED?selected[row*2051u+k]:int(k);
    \\  if(token<0 || uint(token)>pos || uint(token)>=uint(length)) continue;
    \\  const bool cached=uint(token)<uint(prefix_length);
    \\  float values[ITEMS]; float dot=0.0f;
    \\  for(uint j=0;j<ITEMS;++j) {uint d=lane+j*32u;values[j]=d<uint(D)?float(cached?SUSHI_LATENT(cache,uint(token),d,uint(D)):tail[(uint(token)-uint(prefix_length))*uint(D)+d]):0.0f;dot+=query[j]*values[j];}
    \\  dot=simd_sum(dot)*float(scale);
    \\  float next=max(maximum,dot),old=precise::exp(maximum-next),p=precise::exp(dot-next);
    \\  denom=denom*old+p;
    \\  for(uint j=0;j<ITEMS;++j) acc[j]=acc[j]*old+p*values[j];
    \\  maximum=next;
    \\}
    \\const uint base=((row*uint(H)+head)*uint(SPLITS)+part);
;
var kernels: [latent_store.variants]?mlx.mlx_fast_metal_kernel = @splat(null);
pub const Partials = struct {
    partial: Arr,
    stats: Arr,
    pub fn deinit(self: Partials) void {
        _ = mlx.mlx_array_free(self.partial);
        _ = mlx.mlx_array_free(self.stats);
    }
};
pub fn partials(q: Arr, view: View, ids: Arr, offset: Arr, length: Arr, scale: Arr, splits: c_int, selected: bool, s: mlx.mlx_stream) !Partials {
    try view.validate(q);
    var ops = Ops{ .s = s };
    defer ops.deinit();
    const sh = mlx.getShape(q);
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{ sh[0], sh[1], splits, sh[2] }, 4, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &.{ sh[0], sh[1], splits, 2 }, 4, .float32));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, 32, sh[1], sh[0] * splits));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 32, 1, 1));
    for ([_][*:0]const u8{ "H", "D", "SPLITS", "SELECTED" }, [_]c_int{ sh[1], sh[2], splits, @intFromBool(selected) }) |name, value| try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, name, value));
    const cache = view.storage();
    const quantized = cache.quantized();
    const bits = cache.bits();
    const slot = &kernels[latent_store.variant(bits)];
    if (slot.* == null) {
        const names: []const [*:0]const u8 = if (quantized)
            &.{ "q", "cache", "cache_scales", "cache_biases", "tail", "selected", "offset", "length", "scale", "prefix_length" }
        else
            &.{ "q", "cache", "tail", "selected", "offset", "length", "scale", "prefix_length" };
        const ins = mlx.mlx_vector_string_new_data(names.ptr, names.len);
        defer _ = mlx.mlx_vector_string_free(ins);
        const outs = mlx.mlx_vector_string_new_data(&.{ "partial", "stats" }, 2);
        defer _ = mlx.mlx_vector_string_free(outs);
        const kernel = mlx.mlx_fast_metal_kernel_new(latent_store.kernelName("sushi_glm_verify_latent_overlay", bits), ins, outs, shader_source ++ @import("glm5_attention_prefill.zig").partial_tail, latent_store.header(bits), false, false);
        if (kernel.ctx == null) return error.MetalKernelCompileFailed;
        slot.* = kernel;
    }
    const base: u32 = @intCast(view.prefix_rows);
    const ba = try ops.own(mlx.mlx_array_new_data(&base, &.{}, 0, .uint32));
    const inputs = if (quantized)
        mlx.mlx_vector_array_new_data(&.{ q, cache.data, cache.scales, cache.biases, view.tail, ids, offset, length, scale, ba }, 10)
    else
        mlx.mlx_vector_array_new_data(&.{ q, cache.data, view.tail, ids, offset, length, scale, ba }, 8);
    defer _ = mlx.mlx_vector_array_free(inputs);
    var outputs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outputs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outputs, slot.*.?, inputs, cfg, s));
    const pa = try ops.slot();
    const st = try ops.slot();
    try mlx.check(mlx.mlx_vector_array_get(pa, outputs, 0));
    try mlx.check(mlx.mlx_vector_array_get(st, outputs, 1));
    var result = Partials{ .partial = try ops.result(pa.*), .stats = .{ .ctx = null } };
    errdefer _ = mlx.mlx_array_free(result.partial);
    result.stats = try ops.result(st.*);
    return result;
}

test "GLM latent overlay byte bill excludes immutable prefix" {
    try std.testing.expectEqual(@as(usize, 3072), try tailBytes(512, 3, 2));
    try std.testing.expectEqual(@as(usize, 6144), try tailBytes(512, 3, 4));
    try std.testing.expectEqual(@as(usize, 4096), try tailBytes(512, 4, 2));
    try std.testing.expectError(error.InvalidGlmOverlay, tailBytes(512, 5, 2));
}

fn fork(source: *const @import("glm5_attention.zig").State) !@import("glm5_attention.zig").State {
    return source.share();
}
fn exact(a: Arr, b: Arr) !void {
    try std.testing.expectEqualSlices(c_int, mlx.getShape(a), mlx.getShape(b));
    try std.testing.expectEqual(mlx.mlx_array_dtype(a), mlx.mlx_array_dtype(b));
    const count = mlx.mlx_array_size(a);
    if (count == 0) return;
    try mlx.check(mlx.mlx_array_eval(a));
    try mlx.check(mlx.mlx_array_eval(b));
    if (mlx.mlx_array_dtype(a) == .bfloat16) {
        try std.testing.expectEqualSlices(u16, mlx.mlx_array_data_bfloat16(a).?[0..count], mlx.mlx_array_data_bfloat16(b).?[0..count]);
    } else {
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(mlx.mlx_array_data_float32(a).?[0..count]), std.mem.sliceAsBytes(mlx.mlx_array_data_float32(b).?[0..count]));
    }
}
fn normal(ops: *Ops, shape: []const c_int, dtype: mlx.mlx_dtype, seed: u64) !Arr {
    const key = try ops.slot();
    try mlx.check(mlx.mlx_random_key(key, seed));
    const out = try ops.slot();
    try mlx.check(mlx.mlx_random_normal(out, shape.ptr, shape.len, dtype, 0, 0.25, key.*, ops.s));
    return out.*;
}

test "GLM latent overlay branch outputs index state and commit match full append" {
    const attention = @import("glm5_attention.zig");
    const a = std.testing.allocator;
    const s = mlx.gpuStream();
    for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dtype| for ([_]usize{ 1, 2, 3, 252, 253, 254, 255, 256, 2050, 2051, 2052 }) |prefix| {
        var ops = Ops{ .s = s };
        defer ops.deinit();
        var source = attention.State.init();
        defer source.deinit();
        const ape = try ops.zeros(&.{ 4, 8 }, dtype);
        _ = try source.append(try normal(&ops, &.{ @intCast(prefix), 32 }, dtype, 11), try normal(&ops, &.{ @intCast(prefix), 8 }, dtype, 12), try ops.zeros(&.{ @intCast(prefix), 8 }, dtype), ape, s);
        try source.evaluate();
        const latents = try normal(&ops, &.{ 4, 32 }, dtype, 31);
        const keys = try normal(&ops, &.{ 4, 8 }, dtype, 32);
        const gates = try ops.zeros(&.{ 4, 8 }, dtype);
        const queries = try normal(&ops, &.{ 4, 2, 32 }, dtype, 41);
        const iq = try normal(&ops, &.{ 4, 2, 8 }, dtype, 42);
        const iw = try ops.ones(&.{ 4, 2 }, dtype);
        for ([_][]const u32{ &.{0}, &.{ 0, 1 }, &.{ 0, 1, 2 }, &.{ 0, 2 }, &.{ 0, 1, 2, 3 } }) |path| {
            var original = try fork(&source);
            defer original.deinit();
            var virtual = try fork(&source);
            defer virtual.deinit();
            const ids = try ops.own(mlx.mlx_array_new_data(path.ptr, &.{@intCast(path.len)}, 1, .uint32));
            const tail = try ops.take(latents, ids, 0);
            const k = try ops.take(keys, ids, 0);
            const g = try ops.take(gates, ids, 0);
            _ = try original.append(tail, k, g, ape, s);
            _ = try virtual.appendIndexOnly(tail, k, g, ape, s);
            try original.evaluate();
            try virtual.evaluate();
            try std.testing.expectEqual(prefix + path.len, virtual.processed);
            try std.testing.expectEqual(prefix, source.processed);
            for ([_]Arr{ original.tail_keys, original.tail_gates }, [_]Arr{ virtual.tail_keys, virtual.tail_gates }) |left, right| try exact(left, right);
            if (source.pooled.ctx != null) try std.testing.expectEqual(mlx.mlx_array_data_uint8(source.pooled), mlx.mlx_array_data_uint8(virtual.pooled)) else try std.testing.expect(virtual.pooled.ctx == null);
            const pools: c_int = @intCast(virtual.processed / 4);
            const fresh: c_int = if (virtual.pool_tail.ctx == null) 0 else mlx.getShape(virtual.pool_tail)[0];
            if (fresh > 0) {
                const joined = if (pools == fresh) virtual.pool_tail else try ops.concat(&.{ try ops.slice(source.pooled, 0, 0, pools - fresh), virtual.pool_tail }, 0);
                try exact(try ops.contiguous(try ops.slice(original.pooled, 0, 0, pools)), try ops.contiguous(joined));
            } else if (pools > 0) try exact(try ops.contiguous(try ops.slice(original.pooled, 0, 0, pools)), try ops.contiguous(try ops.slice(source.pooled, 0, 0, pools)));
            if (dtype == .bfloat16) try std.testing.expectEqual(mlx.mlx_array_data_bfloat16(source.latent), mlx.mlx_array_data_bfloat16(virtual.latent)) else try std.testing.expectEqual(mlx.mlx_array_data_float32(source.latent), mlx.mlx_array_data_float32(virtual.latent));
            const row: c_int = @intCast(path[path.len - 1]);
            const q = try ops.slice(queries, 0, row, row + 1);
            const index_q = try ops.slice(iq, 0, row, row + 1);
            const weights = try ops.slice(iw, 0, row, row + 1);
            const offset = prefix + path.len - 1;
            const baseline = try ops.own(try attention.attend(&original, q, index_q, weights, offset, 0.25, s));
            const candidate = try ops.own(try attention.attendOverlay(&virtual, q, index_q, weights, offset, 0.25, .{ .prefix = source.latentView(), .prefix_rows = prefix, .tail = tail }, s));
            try exact(baseline, candidate);
            var commit = try fork(&source);
            defer commit.deinit();
            _ = try commit.append(tail, k, g, ape, s);
            try commit.evaluate();
            for (original.arrays(), commit.arrays()) |left, right| if (left.ctx != null) try exact(left, right);
        }
    };
    _ = a;
}

test "GLM kv8 and kv6 latent overlay reads round-tripped tails exactly as the committed append" {
    for ([_]u8{ latent_store.kv8_bits, 6 }) |bits| try overlayTailCase(bits);
}

fn overlayTailCase(bits: u8) !void {
    const attention = @import("glm5_attention.zig");
    const s = mlx.gpuStream();
    const Case = struct { prefix: usize, heads: c_int, width: c_int };
    for ([_]Case{ .{ .prefix = 1, .heads = 2, .width = 64 }, .{ .prefix = 255, .heads = 2, .width = 64 }, .{ .prefix = 2052, .heads = 2, .width = 64 }, .{ .prefix = 2052, .heads = 64, .width = 512 } }) |case| {
        var ops = Ops{ .s = s };
        defer ops.deinit();
        var source = attention.State{ .latent_bits = bits };
        defer source.deinit();
        const ape = try ops.zeros(&.{ 4, 8 }, .bfloat16);
        _ = try source.append(try normal(&ops, &.{ @intCast(case.prefix), case.width }, .bfloat16, 51), try normal(&ops, &.{ @intCast(case.prefix), 8 }, .bfloat16, 52), try ops.zeros(&.{ @intCast(case.prefix), 8 }, .bfloat16), ape, s);
        try source.evaluate();
        const latents = try normal(&ops, &.{ 3, case.width }, .bfloat16, 53);
        const readable = try ops.own(try latent_store.readable(latents, bits, s));
        const keys = try normal(&ops, &.{ 3, 8 }, .bfloat16, 54);
        const gates = try ops.zeros(&.{ 3, 8 }, .bfloat16);
        const queries = try normal(&ops, &.{ 3, case.heads, case.width }, .bfloat16, 55);
        const iq = try normal(&ops, &.{ 3, 2, 8 }, .bfloat16, 56);
        const iw = try ops.ones(&.{ 3, 2 }, .bfloat16);
        for ([_][]const u32{ &.{0}, &.{ 0, 1 }, &.{ 0, 1, 2 }, &.{ 0, 2 } }) |path| {
            var original = try source.share();
            defer original.deinit();
            var virtual = try source.share();
            defer virtual.deinit();
            const ids = try ops.own(mlx.mlx_array_new_data(path.ptr, &.{@intCast(path.len)}, 1, .uint32));
            const k = try ops.take(keys, ids, 0);
            const g = try ops.take(gates, ids, 0);
            _ = try original.append(try ops.take(latents, ids, 0), k, g, ape, s);
            _ = try virtual.appendIndexOnly(try ops.take(latents, ids, 0), k, g, ape, s);
            try std.testing.expect(virtual.latent.ctx != null and virtual.latent_scales.ctx != null);
            const row: c_int = @intCast(path[path.len - 1]);
            const q = try ops.slice(queries, 0, row, row + 1);
            const index_q = try ops.slice(iq, 0, row, row + 1);
            const weights = try ops.slice(iw, 0, row, row + 1);
            const offset = case.prefix + path.len - 1;
            const view = View{ .prefix = source.latentView(), .prefix_rows = case.prefix, .tail = try ops.take(readable, ids, 0) };
            const baseline = try ops.own(try attention.attend(&original, q, index_q, weights, offset, 1.0 / 16.0, s));
            const candidate = try ops.own(try attention.attendOverlay(&virtual, q, index_q, weights, offset, 1.0 / 16.0, view, s));
            try attention.expectSameBits(baseline, candidate);
        }
    }
}

test "GLM overlay branch completing a pool reads a reserved pooled buffer without copying it" {
    const attention = @import("glm5_attention.zig");
    const s = mlx.gpuStream();
    for ([_]u8{ 0, latent_store.kv8_bits, 6 }) |bits| {
        var ops = Ops{ .s = s };
        defer ops.deinit();
        var source = attention.State{ .latent_bits = bits };
        defer source.deinit();
        const prefix = 2050;
        const ape = try ops.zeros(&.{ 4, 128 }, .bfloat16);
        _ = try source.append(try normal(&ops, &.{ prefix, 64 }, .bfloat16, 61), try normal(&ops, &.{ prefix, 128 }, .bfloat16, 62), try ops.zeros(&.{ prefix, 128 }, .bfloat16), ape, s);
        // A whole-context reservation: 64 MiB of pooled keys, far past the 512 committed pools.
        const reserved = try ops.concat(&.{ source.pooled, try ops.zeros(&.{ 262144 - mlx.getShape(source.pooled)[0], 128 }, .bfloat16) }, 0);
        _ = mlx.mlx_array_free(source.pooled);
        source.pooled = try ops.result(reserved);
        try source.evaluate();
        const latents = try normal(&ops, &.{ 2, 64 }, .bfloat16, 63);
        const readable = try ops.own(try latent_store.readable(latents, bits, s));
        const keys = try normal(&ops, &.{ 2, 128 }, .bfloat16, 64);
        const gates = try ops.zeros(&.{ 2, 128 }, .bfloat16);
        const q = try normal(&ops, &.{ 1, 4, 64 }, .bfloat16, 65);
        const iq = try normal(&ops, &.{ 1, 4, 128 }, .bfloat16, 66);
        const iw = try ops.ones(&.{ 1, 4 }, .bfloat16);
        const offset = prefix + 1;
        var committed = try source.share();
        defer committed.deinit();
        _ = try committed.append(latents, keys, gates, ape, s);
        const baseline = try ops.own(try attention.attend(&committed, q, iq, iw, offset, 0.125, s));
        try mlx.check(mlx.mlx_array_eval(baseline));
        try mlx.check(mlx.mlx_array_eval(readable));
        var before: usize = 0;
        try mlx.check(mlx.mlx_get_active_memory(&before));
        try mlx.check(mlx.mlx_reset_peak_memory());
        var branch = try source.share();
        defer branch.deinit();
        _ = try branch.appendIndexOnly(latents, keys, gates, ape, s);
        const view = View{ .prefix = source.latentView(), .prefix_rows = prefix, .tail = readable };
        const candidate = try ops.own(try attention.attendOverlay(&branch, q, iq, iw, offset, 0.125, view, s));
        try mlx.check(mlx.mlx_array_eval(candidate));
        var peak: usize = 0;
        try mlx.check(mlx.mlx_get_peak_memory(&peak));
        try std.testing.expect(peak -| before < 8 * 1024 * 1024);
        try attention.expectSameBits(baseline, candidate);
    }
}
