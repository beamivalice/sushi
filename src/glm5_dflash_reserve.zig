//! Lossless request-owned MLA capacity reservation before speculative cloning.
const std = @import("std");
const mlx = @import("mlx.zig");
const attention = @import("glm5_attention.zig");
const Ops = @import("glm5_model.zig").Ops;
const Arr = mlx.mlx_array;

/// `growth_bytes`: what the state holds more once grown. `peak_bytes`: the most it holds above its start
/// while growing, when the buffer in flight keeps its old rows beside the new buffer and its padding.
pub const Plan = struct { latent_capacity: usize, pool_capacity: usize, growth_bytes: usize, peak_bytes: usize };
fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.GlmReserveOverflow;
}
fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.GlmReserveOverflow;
}
pub fn capacity(rows: usize) !usize {
    const value = try mul((try add(rows, 255)) / 256, 256);
    if (value > std.math.maxInt(c_int)) return error.GlmReserveOverflow;
    return value;
}
fn inFlight(old: usize, target: usize, row_bytes: usize) !usize {
    if (target == old) return 0;
    return mul(try add(target, target - old), row_bytes);
}
/// `latent_row_bytes` is one stored row: BF16 1024, kv8 544 or kv6 416 at width 512 (`glm5_latent.rowBytes`).
pub fn plan(processed: usize, lc: usize, pc: usize, latent_row_bytes: usize, iw: usize, ib: usize, total: usize) !Plan {
    if (latent_row_bytes == 0 or iw == 0 or lc < processed or pc < processed / 4 or total < processed or
        (ib != 2 and ib != 4)) return error.InvalidGlmReserveShape;
    if (lc > std.math.maxInt(c_int) or pc > std.math.maxInt(c_int) or iw > std.math.maxInt(c_int)) return error.GlmReserveOverflow;
    const latent = @max(lc, try capacity(total));
    const pool = @max(pc, try capacity(total / 4));
    const pool_row_bytes = try mul(iw, ib);
    const latent_growth = try mul(latent - lc, latent_row_bytes);
    return .{
        .latent_capacity = latent,
        .pool_capacity = pool,
        .growth_bytes = try add(latent_growth, try mul(pool - pc, pool_row_bytes)),
        .peak_bytes = @max(try inFlight(lc, latent, latent_row_bytes), try add(latent_growth, try inFlight(pc, pool, pool_row_bytes))),
    };
}

/// Peak above the live state when `reserve` grows `count` states with this plan one after another.
pub fn statesPeak(count: usize, p: Plan) !usize {
    if (count == 0) return 0;
    return add(try mul(count - 1, p.growth_bytes), p.peak_bytes);
}
fn supported(a: Arr) bool {
    return a.ctx != null and (mlx.mlx_array_dtype(a) == .bfloat16 or mlx.mlx_array_dtype(a) == .float32);
}
fn statePlan(st: *const attention.State, total: usize) !Plan {
    const view = st.latentView();
    if (!view.rowMajor() or (view.dtype() != .bfloat16 and view.dtype() != .float32) or !supported(st.tail_keys) or !supported(st.tail_gates)) return error.InvalidGlmReserveShape;
    const ts = mlx.getShape(st.tail_keys);
    if (view.rows() <= 0 or view.width() <= 0 or ts.len != 2 or ts[0] < 0 or ts[1] <= 0 or
        @as(usize, @intCast(ts[0])) != st.processed % 4 or
        !std.mem.eql(c_int, ts, mlx.getShape(st.tail_gates)) or
        mlx.mlx_array_dtype(st.tail_keys) != mlx.mlx_array_dtype(st.tail_gates)) return error.InvalidGlmReserveShape;
    var pc: usize = 0;
    if (st.pooled.ctx != null) {
        const ps = mlx.getShape(st.pooled);
        if (!supported(st.pooled) or ps.len != 2 or ps[0] < 0 or ps[1] != ts[1] or
            mlx.mlx_array_dtype(st.pooled) != mlx.mlx_array_dtype(st.tail_keys)) return error.InvalidGlmReserveShape;
        pc = @intCast(ps[0]);
    }
    const row_bytes = if (view.quantized()) @import("glm5_latent.zig").rowBytes(@intCast(view.width()), st.latent_bits) else @as(usize, @intCast(view.width())) * mlx.mlx_array_itemsize(st.latent);
    return plan(st.processed, @intCast(view.rows()), pc, row_bytes, @intCast(ts[1]), mlx.mlx_array_itemsize(st.tail_keys), total);
}
fn grow(buffer: *Arr, rows: usize, width: c_int, dtype: mlx.mlx_dtype, stream: mlx.mlx_stream) !void {
    const old: usize = if (buffer.ctx == null) 0 else @intCast(mlx.getShape(buffer.*)[0]);
    if (rows == old) return;
    var ops = Ops{ .s = stream };
    defer ops.deinit();
    const padding = try ops.zeros(&.{ @intCast(rows - old), width }, dtype);
    const joined = if (old == 0) padding else try ops.concat(&.{ buffer.*, padding }, 0);
    try mlx.check(mlx.mlx_array_eval(joined));
    const result = try ops.result(joined);
    if (buffer.ctx != null) _ = mlx.mlx_array_free(buffer.*);
    buffer.* = result;
}
fn grown(buffer: Arr, rows: usize, stream: mlx.mlx_stream) !Arr {
    var ops = Ops{ .s = stream };
    defer ops.deinit();
    const sh = mlx.getShape(buffer);
    const padding = try ops.zeros(&.{ @intCast(rows - @as(usize, @intCast(sh[0]))), sh[1] }, mlx.mlx_array_dtype(buffer));
    const joined = try ops.concat(&.{ buffer, padding }, 0);
    try mlx.check(mlx.mlx_array_eval(joined));
    return ops.result(joined);
}
/// kv8 codes, scales and biases grow together or not at all.
fn growLatent(st: *attention.State, rows: usize, stream: mlx.mlx_stream) !void {
    if (rows == @as(usize, @intCast(mlx.getShape(st.latent)[0]))) return;
    const fields = [_]*Arr{ &st.latent, &st.latent_scales, &st.latent_biases };
    var parts: [3]Arr = undefined;
    var done: usize = 0;
    errdefer for (parts[0..done]) |p| {
        _ = mlx.mlx_array_free(p);
    };
    for (fields) |field| if (field.ctx != null) {
        parts[done] = try grown(field.*, rows, stream);
        done += 1;
    };
    var next: usize = 0;
    for (fields) |field| if (field.ctx != null) {
        _ = mlx.mlx_array_free(field.*);
        field.* = parts[next];
        next += 1;
    };
}
/// Call on the inference owner after prefill, before taking any request clone. `available_peak_bytes`
/// is what the request may still hold above its live state; returns the peak the growth reaches.
pub fn reserve(request: anytype, total_tokens: usize, available_peak_bytes: usize, stream: mlx.mlx_stream) !usize {
    if (request.failed) return error.InvalidGlmReserveShape;
    var held: usize = 0;
    var bill: usize = 0;
    for (request.layers) |*layer| {
        if (layer.attention.processed == 0) continue;
        if (layer.attention.processed != request.offset) return error.InvalidGlmReserveShape;
        const p = try statePlan(&layer.attention, total_tokens);
        bill = @max(bill, try add(held, p.peak_bytes));
        held = try add(held, p.growth_bytes);
    }
    if (bill > available_peak_bytes) return error.GlmReserveMemoryLimit;
    for (request.layers) |*layer| {
        const st = &layer.attention;
        if (st.processed == 0) continue;
        const p = try statePlan(st, total_tokens);
        const iw = mlx.getShape(st.tail_keys)[1];
        const dtype = mlx.mlx_array_dtype(st.tail_keys);
        try growLatent(st, p.latent_capacity, stream);
        try grow(&st.pooled, p.pool_capacity, iw, dtype, stream);
    }
    return bill;
}

test "GLM reserve ledger admits the 128K verifier" {
    const scratch = @import("glm5_dflash_memory.zig");
    try std.testing.expectError(error.GlmTreeScratchLimit, scratch.plan(131072, 131072, 32768, 512, 128, 64, scratch.overlay_rows + 1, 2));
    const p = try plan(131072, 131072, 32768, 1024, 128, 2, 131072 + 256 + 3);
    try std.testing.expectEqual(@as(usize, 131584), p.latent_capacity);
    try std.testing.expectEqual(@as(usize, 33024), p.pool_capacity);
    try std.testing.expectEqual(@as(usize, 512 * 1024 + 256 * 256), p.growth_bytes);
    try std.testing.expectEqual(@as(usize, (131584 + 512) * 1024), p.peak_bytes);
    try std.testing.expectEqual(@as(usize, 10 * (512 * 1024 + 256 * 256) + (131584 + 512) * 1024), try statesPeak(11, p));
    const admitted = try scratch.plan(131328, p.latent_capacity, p.pool_capacity, 512, 128, 64, 3, 2);
    try std.testing.expectEqual(@as(usize, 3), admitted.branches);
    try std.testing.expect(admitted.live_bytes <= scratch.limit_bytes);
    try std.testing.expectError(error.InvalidGlmReserveShape, plan(4, 3, 1, 1024, 128, 2, 16));
    try std.testing.expectError(error.InvalidGlmReserveShape, plan(4, 4, 1, 1024, 128, 1, 16));
    try std.testing.expectError(error.InvalidGlmReserveShape, plan(4, 4, 1, 1024, 128, 2, 3));
    try std.testing.expectError(error.GlmReserveOverflow, plan(4, 4, 1, std.math.maxInt(usize), 128, 2, 16));
    try std.testing.expectError(error.GlmReserveOverflow, capacity(std.math.maxInt(usize)));
}

fn equal(a: Arr, b: Arr, _: mlx.mlx_stream) !void {
    try std.testing.expectEqual(mlx.mlx_array_dtype(a), mlx.mlx_array_dtype(b));
    try std.testing.expectEqualSlices(c_int, mlx.getShape(a), mlx.getShape(b));
    try mlx.check(mlx.mlx_array_eval(a));
    try mlx.check(mlx.mlx_array_eval(b));
    const n = mlx.mlx_array_size(a);
    if (mlx.mlx_array_dtype(a) == .bfloat16) {
        try std.testing.expectEqualSlices(u16, mlx.mlx_array_data_bfloat16(a).?[0..n], mlx.mlx_array_data_bfloat16(b).?[0..n]);
    } else {
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(mlx.mlx_array_data_float32(a).?[0..n]), std.mem.sliceAsBytes(mlx.mlx_array_data_float32(b).?[0..n]));
    }
}

fn prefixes(a: *const attention.State, b: *const attention.State, ops: *Ops) !void {
    try std.testing.expectEqual(a.processed, b.processed);
    try equal(try ops.slice(a.latent, 0, 0, @intCast(a.processed)), try ops.slice(b.latent, 0, 0, @intCast(b.processed)), ops.s);
    try equal(try ops.slice(a.pooled, 0, 0, @intCast(a.processed / 4)), try ops.slice(b.pooled, 0, 0, @intCast(b.processed / 4)), ops.s);
    try equal(a.tail_keys, b.tail_keys, ops.s);
    try equal(a.tail_gates, b.tail_gates, ops.s);
}

test "GLM reserve preserves cache bits and subsequent append and attention on BF16 and FP32" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    mlx.installErrorHandler();
    const stream = mlx.gpuStream();
    for ([_]mlx.mlx_dtype{ .bfloat16, .float32 }) |dtype| for ([_]usize{ 1023, 2047 }) |n| {
        var ops = Ops{ .s = stream };
        defer ops.deinit();
        const Layer = struct { attention: attention.State = .init() };
        var layers: [2]Layer = @splat(.{});
        defer for (&layers) |*layer| layer.attention.deinit();
        var request = struct { layers: []Layer, offset: usize, failed: bool = false }{ .layers = &layers, .offset = n };
        var baseline = attention.State.init();
        defer baseline.deinit();
        const total = n + 33;
        const data = try std.testing.allocator.alloc(f32, n * 8);
        defer std.testing.allocator.free(data);
        for (data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 31)) / 16 - 1;
        data[0] = -0.0;
        data[7] = -0.0;
        const source = try ops.own(mlx.mlx_array_new_data(data.ptr, &.{ @intCast(n), 8 }, 2, .float32));
        const latent = try ops.cast(source, dtype);
        const keys = try ops.slice(latent, 1, 0, 4);
        const gates = try ops.binary(.mul, keys, try ops.scalar(0.5, dtype));
        const ape = try ops.ones(&.{ 4, 4 }, dtype);
        for (&layers) |*layer| {
            _ = try layer.attention.append(latent, keys, gates, ape, stream);
            try layer.attention.evaluate();
        }
        _ = try baseline.append(latent, keys, gates, ape, stream);
        try baseline.evaluate();
        const p = try statePlan(&layers[0].attention, total);
        const bill = try statesPeak(2, p);
        const old = layers[0].attention.latent.ctx;
        const tail = layers[0].attention.tail_keys.ctx;
        try std.testing.expectError(error.GlmReserveMemoryLimit, reserve(&request, total, bill - 1, stream));
        try std.testing.expect(layers[0].attention.latent.ctx == old);
        const valid = layers[1].attention.pooled;
        layers[1].attention.pooled = try ops.result(try ops.zeros(&.{ @intCast(mlx.getShape(valid)[0]), 4 }, .int32));
        try std.testing.expectError(error.InvalidGlmReserveShape, reserve(&request, total, bill, stream));
        _ = mlx.mlx_array_free(layers[1].attention.pooled);
        layers[1].attention.pooled = valid;
        try std.testing.expect(layers[0].attention.latent.ctx == old);
        try std.testing.expectEqual(bill, try reserve(&request, total, bill, stream));
        try std.testing.expectEqual(@as(c_int, @intCast(p.latent_capacity)), mlx.getShape(layers[0].attention.latent)[0]);
        try std.testing.expectEqual(@as(c_int, @intCast(p.pool_capacity)), mlx.getShape(layers[0].attention.pooled)[0]);
        try std.testing.expect(layers[0].attention.tail_keys.ctx == tail);
        try std.testing.expectEqual(n, request.offset);
        try prefixes(&layers[0].attention, &baseline, &ops);
        const reserved = layers[0].attention.latent.ctx;
        try std.testing.expectEqual(@as(usize, 0), try reserve(&request, total, 0, stream));
        try std.testing.expect(layers[0].attention.latent.ctx == reserved);
        const next = try ops.cast(try ops.slice(source, 0, 7, 13), dtype);
        const next_keys = try ops.slice(next, 1, 0, 4);
        const next_gates = try ops.binary(.mul, next_keys, try ops.scalar(0.5, dtype));
        _ = try layers[0].attention.append(next, next_keys, next_gates, ape, stream);
        _ = try baseline.append(next, next_keys, next_gates, ape, stream);
        try layers[0].attention.evaluate();
        try baseline.evaluate();
        try prefixes(&layers[0].attention, &baseline, &ops);
        const q = try ops.ones(&.{ 1, 2, 8 }, dtype);
        const iq = try ops.ones(&.{ 1, 1, 4 }, dtype);
        const weights = try ops.ones(&.{ 1, 1 }, dtype);
        const got = try ops.own(try attention.attend(&layers[0].attention, q, iq, weights, n + 5, 1.0 / 16.0, stream));
        const want = try ops.own(try attention.attend(&baseline, q, iq, weights, n + 5, 1.0 / 16.0, stream));
        try equal(got, want, stream);
    };
}

test "GLM reserve grows kv8 and kv6 codes scales and biases together and bills 544 and 416-byte rows" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    for ([_]u8{ 8, 6 }) |bits| try reserveQuantizedCase(bits);
}

fn reserveQuantizedCase(bits: u8) !void {
    const latent_store = @import("glm5_latent.zig");
    const row = latent_store.rowBytes(512, bits);
    const stream = mlx.gpuStream();
    var ops = Ops{ .s = stream };
    defer ops.deinit();
    const n = 1023;
    const Layer = struct { attention: attention.State = .{} };
    var layers: [1]Layer = .{.{}};
    layers[0].attention.latent_bits = bits;
    defer layers[0].attention.deinit();
    var request = struct { layers: []Layer, offset: usize, failed: bool = false }{ .layers = &layers, .offset = n };
    var baseline = attention.State{ .latent_bits = bits };
    defer baseline.deinit();
    const source = try ops.own(try latent_store.randomRows(n + 6, 512, 81, stream));
    const keys = try ops.zeros(&.{ n + 6, 4 }, .bfloat16);
    const ape = try ops.zeros(&.{ 4, 4 }, .bfloat16);
    for ([_]*attention.State{ &layers[0].attention, &baseline }) |st| {
        _ = try st.append(try ops.slice(source, 0, 0, n), try ops.slice(keys, 0, 0, n), try ops.slice(keys, 0, 0, n), ape, stream);
        try st.evaluate();
    }
    const total = n + 33;
    const p = try statePlan(&layers[0].attention, total);
    try std.testing.expectEqual(@as(usize, 1280), p.latent_capacity);
    try std.testing.expectEqual(256 * row + 256 * 4 * 2, p.growth_bytes);
    try std.testing.expectEqual((1280 + 256) * row, p.peak_bytes);
    try std.testing.expectError(error.GlmReserveMemoryLimit, reserve(&request, total, p.peak_bytes - 1, stream));
    try std.testing.expectEqual(p.peak_bytes, try reserve(&request, total, p.peak_bytes, stream));
    const view = layers[0].attention.latentView();
    try std.testing.expect(view.rowMajor() and view.rows() == 1280);
    for ([_]Arr{ view.data, view.scales, view.biases }, [_]Arr{ baseline.latent, baseline.latent_scales, baseline.latent_biases }) |grown_part, kept| try attention.expectSameBits(try ops.slice(grown_part, 0, 0, n), try ops.slice(kept, 0, 0, n));
    for ([_]*attention.State{ &layers[0].attention, &baseline }) |st| {
        _ = try st.append(try ops.slice(source, 0, n, n + 6), try ops.slice(keys, 0, n, n + 6), try ops.slice(keys, 0, n, n + 6), ape, stream);
        try st.evaluate();
    }
    const q = try ops.reshape(try ops.own(try latent_store.randomRows(2, 512, 82, stream)), &.{ 1, 2, 512 });
    const iq = try ops.ones(&.{ 1, 1, 4 }, .bfloat16);
    const weights = try ops.ones(&.{ 1, 1 }, .bfloat16);
    try attention.expectSameBits(try ops.own(try attention.attend(&baseline, q, iq, weights, n + 5, 1.0 / 16.0, stream)), try ops.own(try attention.attend(&layers[0].attention, q, iq, weights, n + 5, 1.0 / 16.0, stream)));
}
