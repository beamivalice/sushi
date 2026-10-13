//! GLM prefix-cache state. A restore point P is a multiple of 4 (an IndexPool boundary), so the
//! state there is the KDA conv and recurrence of every linear layer, captured as an
//! `SSMCheckpoint`, plus latent rows [0,P) and pooled rows [0,P/4): a prefix of the entry's
//! `MlaRows`, which serve every checkpoint of the entry.
const std = @import("std");
const mlx = @import("mlx.zig");
const forward = @import("glm5_forward.zig");
const transformer = @import("transformer.zig");
const Arr = mlx.mlx_array;
const KVQuantConfig = @import("kv_quant.zig").KVQuantConfig;
const nil = Arr{ .ctx = null };

pub const pool_size: usize = 4;

/// KDA checkpoints one entry or one prefill keeps: each is 141 MiB of FP32 state.
pub const checkpoint_cap: u32 = 8;

/// KDA states an SSD-only prefill holds in RAM at once: the checkpoint just taken and the one before it,
/// whose write may still be landing. Older ones are on SSD and their RAM copy is released.
pub const checkpoint_ram_held: u32 = 2;

/// KDA checkpoints the RAM tier's hot entry keeps (the newest) when an SSD tier holds the rest.
pub const checkpoint_hot_kept: u32 = 2;

pub fn checkpointMax(global: u32) u32 {
    return if (global == 0) checkpoint_cap else @min(global, checkpoint_cap);
}

/// MLA rows a commit copies: a restore resumes from a checkpoint, so the rows past the newest one are
/// never read.
pub fn commitRows(cps: []const transformer.SSMCheckpoint, offset: usize) usize {
    var newest: usize = 0;
    for (cps) |cp| newest = @max(newest, cp.pos);
    return @min(newest, offset / pool_size * pool_size);
}

/// What one captured token costs: the bytes `MlaRows.capture` copies per row, rounded up.
pub fn rowBytesOf(request: *const forward.Request) u64 {
    var per_row: u64 = 0;
    var per_pool: u64 = 0;
    for (request.layers) |*layer| {
        const st = &layer.attention;
        if (st.processed == 0) continue;
        for ([_]Arr{ st.latent, st.latent_scales, st.latent_biases, st.pooled }, [_]*u64{ &per_row, &per_row, &per_row, &per_pool }) |a, acc| {
            if (a.ctx == null or mlx.mlx_array_size(a) == 0) continue;
            acc.* += @as(u64, mlx.mlx_array_size(a)) * mlx.mlx_array_itemsize(a) / @as(u64, @intCast(mlx.getShape(a)[0]));
        }
    }
    return per_row + (per_pool + pool_size - 1) / pool_size;
}

/// `cps` without those past `len`, whose KDA state nothing can restore; the rest are freed.
pub fn keepThrough(allocator: std.mem.Allocator, cps: []transformer.SSMCheckpoint, len: usize) []transformer.SSMCheckpoint {
    var kept: usize = 0;
    while (kept < cps.len and cps[kept].pos <= len) kept += 1;
    if (kept == cps.len) return cps;
    const shrunk = allocator.dupe(transformer.SSMCheckpoint, cps[0..kept]) catch return cps;
    for (cps[kept..]) |*cp| cp.deinit(allocator);
    allocator.free(cps);
    return shrunk;
}

/// The request's KDA state at its current offset, as owned copies (MLA slots stay empty).
pub fn captureKda(allocator: std.mem.Allocator, request: *const forward.Request, s: mlx.mlx_stream) !transformer.SSMCheckpoint {
    if (request.failed) return error.GlmRequestNeedsReset;
    if (request.offset % pool_size != 0) return error.GlmCheckpointOffPool;
    const layers = try allocator.alloc(transformer.SSMCacheEntrySnapshot, request.layers.len);
    var built: usize = 0;
    errdefer {
        for (layers[0..built]) |*l| transformer.ssmSnapshotDeinit(l);
        allocator.free(layers);
    }
    const evals = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(evals);
    for (request.layers, layers) |*src, *out| {
        out.* = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = src.recurrent.initialized };
        built += 1;
        if (!src.recurrent.initialized) continue;
        for ([_]Arr{ src.recurrent.conv_state, src.recurrent.ssm_state }, [_]*Arr{ &out.conv_state, &out.ssm_state }) |live, dst| {
            const copy = try bitsOwnedCopy(live, s);
            _ = mlx.mlx_array_free(dst.*);
            dst.* = copy;
            try mlx.check(mlx.mlx_vector_array_append_value(evals, copy));
        }
    }
    try mlx.check(mlx.mlx_eval(evals));
    return .{ .pos = request.offset, .layers = layers };
}

/// Installs a checkpoint's KDA state; its arrays are shared, never written.
pub fn restoreKda(request: *forward.Request, cp: *const transformer.SSMCheckpoint) !void {
    if (cp.layers.len != request.layers.len) return error.SsmCheckpointLayerMismatch;
    for (request.layers, cp.layers) |*dst, *src| try transformer.ssmRestore(&dst.recurrent, src);
}

/// The request as it stood at `cp.pos`; a failure leaves it reset.
pub fn restore(request: *forward.Request, rows: *const MlaRows, cp: *const transformer.SSMCheckpoint) !void {
    try rows.restore(request, cp.pos);
    errdefer request.reset();
    try restoreKda(request, cp);
}

/// MLA rows [0, len) of every layer: latent (BF16, or kv8 codes with scales and biases) and
/// pooled index keys, plus empty IndexPool tails of the right width.
pub const MlaRows = struct {
    allocator: std.mem.Allocator,
    layers: []Layer,
    len: usize,
    latent_bits: u8,

    pub const Layer = struct {
        latent: Arr = nil,
        scales: Arr = nil,
        biases: Arr = nil,
        pooled: Arr = nil,
        tail_keys: Arr = nil,
        tail_gates: Arr = nil,

        fn arrays(self: *Layer) [6]*Arr {
            return .{ &self.latent, &self.scales, &self.biases, &self.pooled, &self.tail_keys, &self.tail_gates };
        }
    };

    /// Owned copies of the request's first `len` rows (`len` <= offset, a multiple of 4): a
    /// shared slice would keep the request's whole reservation alive and bill only `len`.
    pub fn capture(allocator: std.mem.Allocator, request: *const forward.Request, len: usize, s: mlx.mlx_stream) !MlaRows {
        if (request.failed) return error.GlmRequestNeedsReset;
        if (len % pool_size != 0 or len > request.offset or len == 0) return error.GlmCheckpointOffPool;
        var out = try empty(allocator, request.layers.len, len, request.latent_bits);
        errdefer out.deinit();
        const evals = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(evals);
        for (request.layers, out.layers) |*src, *dst| {
            const st = &src.attention;
            if (st.processed == 0) continue;
            if (st.processed != request.offset or st.tail_keys.ctx == null or st.tail_gates.ctx == null) return error.InvalidGlmAttentionShape;
            const sources = [_]Arr{ st.latent, st.latent_scales, st.latent_biases, st.pooled };
            const rows = [_]usize{ len, len, len, len / pool_size };
            for (sources, rows, dst.arrays()[0..4]) |source, n, field| {
                if (source.ctx == null or n == 0) continue;
                field.* = try rowsOwned(source, 0, n, s);
                try mlx.check(mlx.mlx_vector_array_append_value(evals, field.*));
            }
            for ([_]Arr{ st.tail_keys, st.tail_gates }, dst.arrays()[4..6]) |source, field| field.* = try rowsOwned(source, 0, 0, s);
        }
        try mlx.check(mlx.mlx_eval(evals));
        return out;
    }

    /// Rows [0, len) of the request's own buffers, shared rather than copied, for the SSD flush.
    /// The share keeps the request's whole reservation alive and makes a later write to it copy,
    /// so it lives from the commit to the flush after the response, while the request is done.
    pub fn shareLive(allocator: std.mem.Allocator, request: *const forward.Request, len: usize, s: mlx.mlx_stream) !MlaRows {
        if (request.failed) return error.GlmRequestNeedsReset;
        if (len % pool_size != 0 or len > request.offset or len == 0) return error.GlmCheckpointOffPool;
        var out = try empty(allocator, request.layers.len, len, request.latent_bits);
        errdefer out.deinit();
        for (request.layers, out.layers) |*src, *dst| {
            const st = &src.attention;
            if (st.processed == 0) continue;
            if (st.processed != request.offset) return error.InvalidGlmAttentionShape;
            const sources = [_]Arr{ st.latent, st.latent_scales, st.latent_biases, st.pooled };
            const rows = [_]usize{ len, len, len, len / pool_size };
            for (sources, rows, dst.arrays()[0..4]) |source, n, field| {
                if (source.ctx == null or n == 0) continue;
                field.* = try rowsView(source, 0, n, s);
            }
        }
        return out;
    }

    fn empty(allocator: std.mem.Allocator, count: usize, len: usize, bits: u8) !MlaRows {
        const layers = try allocator.alloc(Layer, count);
        @memset(layers, .{});
        return .{ .allocator = allocator, .layers = layers, .len = len, .latent_bits = bits };
    }

    pub fn deinit(self: *MlaRows) void {
        self.releaseHandles();
        self.allocator.free(self.layers);
        self.layers = &.{};
    }

    /// Drops every handle, keeping the layer table: the move half of a checkout.
    pub fn releaseHandles(self: *MlaRows) void {
        for (self.layers) |*layer| for (layer.arrays()) |a| {
            if (a.ctx != null) _ = mlx.mlx_array_free(a.*);
            a.* = nil;
        };
    }

    /// A second owner of the same buffers.
    pub fn share(self: *const MlaRows) !MlaRows {
        var out = try empty(self.allocator, self.layers.len, self.len, self.latent_bits);
        errdefer out.deinit();
        for (self.layers, out.layers) |*src, *dst| {
            var from = src.*;
            for (from.arrays(), dst.arrays()) |a, b| if (a.ctx != null) {
                b.* = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_array_set(b, a.*));
            };
        }
        return out;
    }

    pub fn bytes(self: *const MlaRows) u64 {
        var total: u64 = 0;
        for (self.layers) |*layer| {
            var l = layer.*;
            for (l.arrays()) |a| if (a.ctx != null) {
                total += @as(u64, mlx.mlx_array_size(a.*)) * mlx.mlx_array_itemsize(a.*);
            };
        }
        return total;
    }

    /// What one retained token costs; a trimmed copy holds exactly `len * rowBytes()`.
    pub fn rowBytes(self: *const MlaRows) u64 {
        return if (self.len == 0) 0 else self.bytes() / self.len;
    }

    /// Owned copies of the first `len` rows (a multiple of 4).
    pub fn trimmedCopy(self: *const MlaRows, len: usize, s: mlx.mlx_stream) !MlaRows {
        if (len % pool_size != 0 or len > self.len or len == 0) return error.GlmCheckpointOffPool;
        var out = try empty(self.allocator, self.layers.len, len, self.latent_bits);
        errdefer out.deinit();
        const evals = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(evals);
        for (self.layers, out.layers) |*src, *dst| {
            var from = src.*;
            const rows = [_]usize{ len, len, len, len / pool_size, 0, 0 };
            for (from.arrays(), dst.arrays(), rows) |a, b, n| if (a.ctx != null) {
                b.* = try rowsOwned(a.*, 0, n, s);
                try mlx.check(mlx.mlx_vector_array_append_value(evals, b.*));
            };
        }
        try mlx.check(mlx.mlx_eval(evals));
        return out;
    }

    /// Binds rows [0, pos) into a reset request's MLA layers; the first append copies them.
    pub fn restore(self: *const MlaRows, request: *forward.Request, pos: usize) !void {
        if (pos % pool_size != 0 or pos > self.len or pos == 0 or request.layers.len != self.layers.len) return error.GlmCheckpointOffPool;
        request.reset();
        try request.setLatentBits(self.latent_bits);
        errdefer request.reset();
        for (self.layers, request.layers) |*src, *dst| {
            if (src.latent.ctx == null) continue;
            var from = src.*;
            const st = &dst.attention;
            const fields = [_]*Arr{ &st.latent, &st.latent_scales, &st.latent_biases, &st.pooled, &st.tail_keys, &st.tail_gates };
            for (from.arrays(), fields) |a, b| if (a.ctx != null) {
                b.* = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_array_set(b, a.*));
            };
            st.processed = pos;
        }
        request.offset = pos;
    }
};

/// The SSD tier keeps MLA rows as a dense pseudo-cache of two entries per layer: entry 2i holds
/// the latent (BF16, or kv8 codes) as keys and the pooled keys, one pool per four rows, as values;
/// entry 2i+1 holds the kv8 scales and biases. Its quant key keeps the two latent widths apart.
pub fn diskQuant(bits: u8) KVQuantConfig {
    return if (bits == 0) KVQuantConfig.dense else .{ .scheme = .off, .bits = bits, .group_size = 64 };
}

/// Views of `rows` in the disk layout; free with `freeDiskEntries`.
pub fn diskEntries(allocator: std.mem.Allocator, rows: *const MlaRows) ![]transformer.KVCacheEntry {
    const out = try allocator.alloc(transformer.KVCacheEntry, rows.layers.len * 2);
    for (out) |*e| e.* = transformer.newEmptyKVEntry();
    errdefer freeDiskEntries(allocator, out);
    for (rows.layers, 0..) |*layer, i| {
        if (layer.latent.ctx == null) continue;
        const pair = [_][2]Arr{ .{ layer.latent, layer.pooled }, .{ layer.scales, layer.biases } };
        for (pair, out[2 * i ..][0..2]) |arrays, *e| {
            if (arrays[0].ctx == null) continue;
            try asTokenRows(&e.keys, arrays[0], rows.len);
            try asTokenRows(&e.values, arrays[1], rows.len);
            e.offset = rows.len;
            e.initialized = true;
        }
    }
    return out;
}

pub fn freeDiskEntries(allocator: std.mem.Allocator, entries: []transformer.KVCacheEntry) void {
    for (entries) |*e| transformer.resetKVEntry(e);
    allocator.free(entries);
}

/// Takes the rows a disk restore filled into `cache` (the `diskEntries` layout).
pub fn rowsFromDisk(allocator: std.mem.Allocator, cache: *transformer.KVCache, bits: u8) !MlaRows {
    const len = cache.step;
    if (len % pool_size != 0 or len == 0 or cache.entries.len % 2 != 0) return error.GlmCheckpointOffPool;
    var out = try MlaRows.empty(allocator, cache.entries.len / 2, len, bits);
    errdefer out.deinit();
    const s = mlx.gpuStream();
    for (out.layers, 0..) |*layer, i| {
        const main = &cache.entries[2 * i];
        if (!main.initialized) continue;
        const width: c_int = mlx.getShape(main.values)[3] * @as(c_int, pool_size);
        layer.latent = try reshaped(main.keys, &.{ @intCast(len), mlx.getShape(main.keys)[3] }, s);
        layer.pooled = try reshaped(main.values, &.{ @intCast(len / pool_size), width }, s);
        const quant = &cache.entries[2 * i + 1];
        if (quant.initialized) {
            layer.scales = try reshaped(quant.keys, &.{ @intCast(len), mlx.getShape(quant.keys)[3] }, s);
            layer.biases = try reshaped(quant.values, &.{ @intCast(len), mlx.getShape(quant.values)[3] }, s);
        }
        for ([_]*Arr{ &layer.tail_keys, &layer.tail_gates }) |tail| {
            tail.* = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_zeros(tail, &[_]c_int{ 0, width }, 2, mlx.mlx_array_dtype(layer.pooled), s));
        }
    }
    return out;
}

fn asTokenRows(dst: *Arr, src: Arr, len: usize) !void {
    const n: usize = @intCast(mlx.mlx_array_size(src));
    _ = mlx.mlx_array_free(dst.*);
    dst.* = try reshaped(src, &.{ 1, 1, @intCast(len), @intCast(n / len) }, mlx.gpuStream());
}

fn reshaped(src: Arr, shape: []const c_int, s: mlx.mlx_stream) !Arr {
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, src, shape.ptr, shape.len, s));
    return out;
}

/// An owned copy that keeps every bit: adding a float zero turns -0.0 into +0.0, and a restored
/// state carries that into the next chunk.
fn bitsOwnedCopy(x: Arr, s: mlx.mlx_stream) !Arr {
    const dtype = mlx.mlx_array_dtype(x);
    const raw_dtype: mlx.mlx_dtype = switch (dtype) {
        .bfloat16, .float16 => .uint16,
        .float32 => .uint32,
        else => dtype,
    };
    var raw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(raw);
    try mlx.check(mlx.mlx_view(&raw, x, raw_dtype, s));
    const copy = try transformer.materializedOwnedCopy(s, raw);
    if (raw_dtype == dtype) return copy;
    defer _ = mlx.mlx_array_free(copy);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_view(&out, copy, dtype, s));
    return out;
}

/// An owned copy of rows [from, to) on axis 0.
fn rowsOwned(x: Arr, from: usize, to: usize, s: mlx.mlx_stream) !Arr {
    const sliced = try rowsView(x, from, to, s);
    defer _ = mlx.mlx_array_free(sliced);
    return bitsOwnedCopy(sliced, s);
}

/// Rows [from, to) on axis 0, a view of `x`'s buffer.
fn rowsView(x: Arr, from: usize, to: usize, s: mlx.mlx_stream) !Arr {
    const sh = mlx.getShape(x);
    var sliced = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(sliced);
    var start: [4]c_int = @splat(0);
    var stop: [4]c_int = @splat(0);
    const strides: [4]c_int = @splat(1);
    @memcpy(stop[0..sh.len], sh);
    start[0] = @intCast(from);
    stop[0] = @intCast(to);
    try mlx.check(mlx.mlx_slice(&sliced, x, &start, sh.len, &stop, sh.len, &strides, sh.len, s));
    return sliced;
}

const testing = std.testing;
const model = @import("model.zig");

fn testIds(tokens: []const u32) Arr {
    return mlx.mlx_array_new_data(tokens.ptr, &[_]c_int{ 1, @intCast(tokens.len) }, 2, .uint32);
}

pub fn servedRequest(bits: u8) !forward.Request {
    var req = try forward.Request.init(testing.allocator, 4);
    errdefer req.deinit();
    req.dense_prefill = true;
    req.prefill_async = true;
    try req.setLatentBits(bits);
    return req;
}

/// Forwards `tokens` in `chunks`-long pieces from the request's offset; returns the last logits.
pub fn prefill(net: *const forward.Model, req: *forward.Request, tokens: []const u32, chunks: []const usize) !Arr {
    var at: usize = 0;
    var last: Arr = nil;
    errdefer if (last.ctx != null) {
        _ = mlx.mlx_array_free(last);
    };
    for (chunks) |n| {
        const ids = testIds(tokens[at .. at + n]);
        defer _ = mlx.mlx_array_free(ids);
        if (last.ctx != null) _ = mlx.mlx_array_free(last);
        last = try net.forwardLast(req, ids, true);
        at += n;
    }
    return last;
}

/// Every row a forward can read: latent and pooled rows below `processed`, tails, KDA state.
pub fn expectSameLogicalState(a: *forward.Request, b: *forward.Request) !void {
    const s = mlx.gpuStream();
    try testing.expectEqual(a.offset, b.offset);
    for (a.layers, b.layers) |*x, *y| {
        try testing.expectEqual(x.recurrent.initialized, y.recurrent.initialized);
        if (x.recurrent.initialized) {
            try forward.expectArrayBits(x.recurrent.conv_state, y.recurrent.conv_state);
            try forward.expectArrayBits(x.recurrent.ssm_state, y.recurrent.ssm_state);
        }
        const p = x.attention.processed;
        try testing.expectEqual(p, y.attention.processed);
        if (p == 0) continue;
        const pairs = [_][2]Arr{
            .{ x.attention.latent, y.attention.latent },
            .{ x.attention.latent_scales, y.attention.latent_scales },
            .{ x.attention.latent_biases, y.attention.latent_biases },
            .{ x.attention.pooled, y.attention.pooled },
        };
        for (pairs, [_]usize{ p, p, p, p / pool_size }) |pair, rows| {
            try testing.expectEqual(pair[0].ctx == null, pair[1].ctx == null);
            if (pair[0].ctx == null or rows == 0) continue;
            const l = try rowsOwned(pair[0], 0, rows, s);
            defer _ = mlx.mlx_array_free(l);
            const r = try rowsOwned(pair[1], 0, rows, s);
            defer _ = mlx.mlx_array_free(r);
            try forward.expectArrayBits(l, r);
        }
        try forward.expectArrayBits(x.attention.tail_keys, y.attention.tail_keys);
        try forward.expectArrayBits(x.attention.tail_gates, y.attention.tail_gates);
    }
}

test "GLM prefix restore at a chunk-aligned checkpoint continues exactly like a cold prefill, BF16 and kv8" {
    var weights = model.Weights.init(testing.allocator);
    defer weights.deinit();
    const cfg = try forward.nonzeroDecodeFixture(&weights);
    var net = try forward.Model.load(testing.allocator, cfg, &weights, mlx.gpuStream());
    defer net.deinit();
    // The committed turn and the next one share 18 tokens; the stale rows past the checkpoint differ.
    const first = [_]u32{ 1, 2, 3, 0, 2, 2, 1, 3, 0, 1, 1, 2, 3, 3, 0, 2, 1, 3, 0, 0 };
    const next = [_]u32{ 1, 2, 3, 0, 2, 2, 1, 3, 0, 1, 1, 2, 3, 3, 0, 2, 1, 3, 2, 1, 3, 1, 0, 2 };
    for ([_]u8{ 0, 8 }) |bits| {
        var turn = try servedRequest(bits);
        defer turn.deinit();
        _ = mlx.mlx_array_free(try prefill(&net, &turn, &first, &.{8}));
        var cp8 = try captureKda(testing.allocator, &turn, mlx.gpuStream());
        defer cp8.deinit(testing.allocator);
        _ = mlx.mlx_array_free(try prefill(&net, &turn, first[8..], &.{ 8, 4 }));
        var rows = try MlaRows.capture(testing.allocator, &turn, 20, mlx.gpuStream());
        defer rows.deinit();
        turn.reset();

        var cold = try servedRequest(bits);
        defer cold.deinit();
        const want = try prefill(&net, &cold, &next, &.{ 8, 8, 8 });
        defer _ = mlx.mlx_array_free(want);

        var warm = try servedRequest(bits);
        defer warm.deinit();
        try restore(&warm, &rows, &cp8);
        try testing.expectEqual(@as(usize, 8), warm.offset);
        const got = try prefill(&net, &warm, next[8..], &.{ 8, 8 });
        defer _ = mlx.mlx_array_free(got);
        try forward.expectArrayBits(want, got);
        try expectSameLogicalState(&cold, &warm);
    }
}

test "GLM prefix rows bill exactly their rows and trim to a pool boundary" {
    var weights = model.Weights.init(testing.allocator);
    defer weights.deinit();
    const cfg = try forward.nonzeroDecodeFixture(&weights);
    var net = try forward.Model.load(testing.allocator, cfg, &weights, mlx.gpuStream());
    defer net.deinit();
    const tokens = [_]u32{ 1, 2, 3, 0, 2, 2, 1, 3, 0, 1, 1, 2, 3, 3, 0, 2, 1, 3, 0 };
    for ([_]u8{ 0, 8 }) |bits| {
        var req = try servedRequest(bits);
        defer req.deinit();
        _ = mlx.mlx_array_free(try prefill(&net, &req, &tokens, &.{ 16, 3 }));
        try testing.expectError(error.GlmCheckpointOffPool, captureKda(testing.allocator, &req, mlx.gpuStream()));
        try testing.expectError(error.GlmCheckpointOffPool, MlaRows.capture(testing.allocator, &req, 18, mlx.gpuStream()));
        var rows = try MlaRows.capture(testing.allocator, &req, 16, mlx.gpuStream());
        defer rows.deinit();
        const latent = @import("glm5_latent.zig").rowBytes(cfg.mla_kv_lora_rank, bits);
        const pooled = cfg.indexer_head_dim * 2 / pool_size;
        try testing.expectEqual(@as(u64, latent + pooled), rows.rowBytes());
        try testing.expectEqual(rows.rowBytes(), rowBytesOf(&req));
        try testing.expectEqual(16 * rows.rowBytes(), rows.bytes());
        var short = try rows.trimmedCopy(8, mlx.gpuStream());
        defer short.deinit();
        try testing.expectEqual(8 * rows.rowBytes(), short.bytes());
        try testing.expectError(error.GlmCheckpointOffPool, rows.trimmedCopy(6, mlx.gpuStream()));
        var twin = try rows.share();
        defer twin.deinit();
        twin.releaseHandles();
        try testing.expectEqual(@as(u64, 0), twin.bytes());
        try testing.expectEqual(16 * rows.rowBytes(), rows.bytes());
        var kda = try captureKdaAt(&net, bits, tokens[0..12]);
        defer kda.deinit(testing.allocator);
        var target = try servedRequest(bits);
        defer target.deinit();
        try testing.expectError(error.GlmCheckpointOffPool, rows.restore(&target, 18));
        try testing.expectError(error.GlmCheckpointOffPool, restore(&target, &short, &kda));
        try restore(&target, &rows, &kda);
        try testing.expectEqual(@as(usize, 12), target.offset);
    }
}

test "GLM live rows for the SSD flush share the request's buffers and outlive its reset" {
    var weights = model.Weights.init(testing.allocator);
    defer weights.deinit();
    const cfg = try forward.nonzeroDecodeFixture(&weights);
    var net = try forward.Model.load(testing.allocator, cfg, &weights, mlx.gpuStream());
    defer net.deinit();
    const s = mlx.gpuStream();
    const tokens = [_]u32{ 1, 2, 3, 0, 2, 2, 1, 3, 0, 1, 1, 2, 3, 3, 0, 2, 1, 3, 0 };
    for ([_]u8{ 0, 8 }) |bits| {
        var req = try servedRequest(bits);
        defer req.deinit();
        _ = mlx.mlx_array_free(try prefill(&net, &req, &tokens, &.{ 16, 3 }));
        try testing.expectError(error.GlmCheckpointOffPool, MlaRows.shareLive(testing.allocator, &req, 18, s));
        var want = try MlaRows.capture(testing.allocator, &req, 16, s);
        defer want.deinit();
        try mlx.check(mlx.mlx_synchronize(s));
        var before: usize = 0;
        _ = mlx.mlx_get_active_memory(&before);
        var live = try MlaRows.shareLive(testing.allocator, &req, 16, s);
        defer live.deinit();
        for (live.layers) |*layer| for (layer.arrays()[0..4]) |a| if (a.ctx != null) {
            try mlx.check(mlx.mlx_array_eval(a.*));
        };
        // Views of buffers that already existed: nothing was copied.
        var after: usize = 0;
        _ = mlx.mlx_get_active_memory(&after);
        try testing.expectEqual(before, after);
        req.reset();
        try testing.expectEqual(want.bytes(), live.bytes());
        for (want.layers, live.layers) |*a, *b| {
            var x = a.*;
            var y = b.*;
            for (x.arrays()[0..4], y.arrays()[0..4]) |p, q| {
                try testing.expectEqual(p.ctx == null, q.ctx == null);
                if (p.ctx != null) try forward.expectArrayBits(p.*, q.*);
            }
        }
    }
}

test "a commit sheds the checkpoints above the rows it keeps" {
    var weights = model.Weights.init(testing.allocator);
    defer weights.deinit();
    const cfg = try forward.nonzeroDecodeFixture(&weights);
    var net = try forward.Model.load(testing.allocator, cfg, &weights, mlx.gpuStream());
    defer net.deinit();
    const tokens = [_]u32{ 1, 2, 3, 0, 2, 2, 1, 3, 0, 1, 1, 2, 3, 3, 0, 2, 1, 3, 0 };
    const cps = try testing.allocator.alloc(transformer.SSMCheckpoint, 2);
    cps[0] = try captureKdaAt(&net, 8, tokens[0..8]);
    cps[1] = try captureKdaAt(&net, 8, tokens[0..16]);
    const kept = keepThrough(testing.allocator, cps, 12);
    defer {
        for (kept) |*cp| cp.deinit(testing.allocator);
        testing.allocator.free(kept);
    }
    try testing.expectEqual(@as(usize, 1), kept.len);
    try testing.expectEqual(@as(usize, 8), kept[0].pos);
}

fn captureKdaAt(net: *const forward.Model, bits: u8, tokens: []const u32) !transformer.SSMCheckpoint {
    var req = try servedRequest(bits);
    defer req.deinit();
    _ = mlx.mlx_array_free(try prefill(net, &req, tokens, &.{tokens.len}));
    return captureKda(testing.allocator, &req, mlx.gpuStream());
}

test "GLM prefix checkpoints keep signed zeros of the KDA state" {
    var req = try forward.Request.init(testing.allocator, 1);
    defer req.deinit();
    const s = mlx.gpuStream();
    const conv = [_]u16{ 0x8000, 0x3f80, 0x8000, 0x0000 };
    const state = [_]u32{ 0x80000000, 0x3f800000, 0x00000000, 0x80000000 };
    try mlx.check(mlx.mlx_array_set(&req.layers[0].recurrent.conv_state, mlx.mlx_array_new_data(&conv, &[_]c_int{ 1, 1, 4 }, 3, .bfloat16)));
    try mlx.check(mlx.mlx_array_set(&req.layers[0].recurrent.ssm_state, mlx.mlx_array_new_data(&state, &[_]c_int{ 1, 1, 2, 2 }, 4, .float32)));
    req.layers[0].recurrent.initialized = true;
    req.offset = 4;
    var cp = try captureKda(testing.allocator, &req, s);
    defer cp.deinit(testing.allocator);
    try mlx.check(mlx.mlx_array_eval(cp.layers[0].ssm_state));
    try testing.expectEqualSlices(u32, &state, @ptrCast(mlx.mlx_array_data_float32(cp.layers[0].ssm_state).?[0..4]));
    var raw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(raw);
    try mlx.check(mlx.mlx_view(&raw, cp.layers[0].conv_state, .uint16, s));
    try mlx.check(mlx.mlx_array_eval(raw));
    try testing.expectEqualSlices(u16, &conv, mlx.mlx_array_data_uint16(raw).?[0..4]);
}

test "a commit copies MLA rows only through its newest checkpoint" {
    const cp = struct {
        fn at(pos: usize) transformer.SSMCheckpoint {
            return .{ .pos = pos, .layers = &.{} };
        }
    }.at;
    try testing.expectEqual(@as(usize, 0), commitRows(&.{}, 1 << 20));
    try testing.expectEqual(@as(usize, 32), commitRows(&.{cp(32)}, (1 << 20) + 3));
    try testing.expectEqual(@as(usize, 8192), commitRows(&.{ cp(2048), cp(8192), cp(4096) }, 9000));
    // A prefill cancelled below its newest checkpoint cannot copy rows it never forwarded.
    try testing.expectEqual(@as(usize, 100), commitRows(&.{cp(128)}, 103));
}
