//! Layer-major GLM prefill: a batch of independent windows runs layer 0 for every window, then layer 1, and so on,
//! so a streamed layer's experts are read once per batch. Each window keeps its own [1, t] forwards and state.
const std = @import("std");
const mlx = @import("mlx.zig");
const model = @import("model.zig");
const forward = @import("glm5_forward.zig");
const Arr = mlx.mlx_array;

pub const Sink = struct {
    ctx: *anyopaque,
    /// One chunk of one window at a block boundary (0 = layer 0's input), [1, t, hc, hidden] BF16.
    boundary: *const fn (ctx: *anyopaque, window: usize, boundary: usize, rows: Arr) anyerror!void,
    /// The window's logits at its last position, [1, 1, vocab].
    logits: *const fn (ctx: *anyopaque, window: usize, logits: Arr) anyerror!void,
};

pub const Stats = struct { fill_bytes: u64 = 0, fill_ns: u64 = 0, wall_ns: u64 = 0 };

/// Prefills every window in `chunk`-row pieces, layer by layer; a streamed MoE layer is pinned once for the batch.
/// The order of work is the only difference from one window after another: each chunk sees the same input,
/// state and kernels, so every output is the same bytes.
pub fn prefill(a: std.mem.Allocator, net: *const forward.Model, windows: []const []const u32, chunk: usize, sink: Sink) !Stats {
    if (windows.len == 0 or chunk == 0) return error.InvalidGlmLayerMajorBatch;
    const clock_start = @import("expert_stream.zig").Clock.init();
    const engine = if (net.expert_stream) |store| store.engine else null;
    const fill_bytes = if (engine) |e| e.fill_bytes_total else 0;
    const fill_ns = if (engine) |e| e.fill_ns_total else 0;
    var chunks: usize = 0;
    for (windows) |ids| {
        if (ids.len == 0) return error.EmptyPrompt;
        if (ids.len > net.cfg.max_position_embeddings) return error.GlmContextExceeded;
        for (ids) |id| if (id >= net.cfg.vocab_size) return error.InvalidGlmPrompt;
        chunks += (ids.len + chunk - 1) / chunk;
    }
    const requests = try a.alloc(forward.Request, windows.len);
    defer a.free(requests);
    var made: usize = 0;
    defer for (requests[0..made]) |*r| r.deinit();
    while (made < windows.len) : (made += 1) {
        requests[made] = try forward.Request.init(a, net.layers.len);
        requests[made].dense_prefill = true;
    }
    const hs = try a.alloc(Arr, chunks);
    defer a.free(hs);
    @memset(hs, .{ .ctx = null });
    defer for (hs) |h| if (h.ctx != null) {
        _ = mlx.mlx_array_free(h);
    };
    var k: usize = 0;
    for (windows, 0..) |ids, w| {
        var cursor: usize = 0;
        while (cursor < ids.len) : (k += 1) {
            const end = @min(ids.len, cursor + chunk);
            const input = mlx.mlx_array_new_data(ids[cursor..end].ptr, &.{ 1, @intCast(end - cursor) }, 2, .uint32);
            defer _ = mlx.mlx_array_free(input);
            hs[k] = try net.embedStreams(input, null);
            try mlx.check(mlx.mlx_array_eval(hs[k]));
            try sink.boundary(sink.ctx, w, 0, hs[k]);
            cursor = end;
        }
    }
    for (0..net.layers.len) |layer| {
        const store = if (layer >= net.cfg.first_k_dense_replace) net.expert_stream else null;
        if (store) |s| try s.pin(@intCast(layer));
        defer if (store) |s| s.unpin();
        k = 0;
        for (windows, requests, 0..) |ids, *request, w| {
            for (0..(ids.len + chunk - 1) / chunk) |_| {
                const next = try net.prefillLayer(request, layer, hs[k]);
                _ = mlx.mlx_array_free(hs[k]);
                hs[k] = next;
                try sink.boundary(sink.ctx, w, layer + 1, next);
                k += 1;
            }
            request.releaseLayer(layer);
        }
    }
    k = 0;
    for (windows, 0..) |ids, w| {
        k += (ids.len + chunk - 1) / chunk;
        const logits = try net.headLogits(hs[k - 1], true);
        defer _ = mlx.mlx_array_free(logits);
        try mlx.check(mlx.mlx_array_eval(logits));
        try sink.logits(sink.ctx, w, logits);
    }
    var clock = clock_start;
    return .{
        .fill_bytes = if (engine) |e| e.fill_bytes_total - fill_bytes else 0,
        .fill_ns = if (engine) |e| e.fill_ns_total - fill_ns else 0,
        .wall_ns = clock.lap(),
    };
}

const testing = std.testing;
const fixture = @import("glm_stream_fixture.zig");

const Collector = struct {
    a: std.mem.Allocator,
    rows: [][fixture.Tiny.layers + 1]std.ArrayList(u8),
    logits: [][]u8,

    fn init(a: std.mem.Allocator, windows: usize) !Collector {
        const rows = try a.alloc([fixture.Tiny.layers + 1]std.ArrayList(u8), windows);
        for (rows) |*r| r.* = @splat(.empty);
        const logits = try a.alloc([]u8, windows);
        @memset(logits, &.{});
        return .{ .a = a, .rows = rows, .logits = logits };
    }
    fn deinit(self: *Collector) void {
        for (self.rows) |*r| for (r) |*b| b.deinit(self.a);
        for (self.logits) |l| self.a.free(l);
        self.a.free(self.rows);
        self.a.free(self.logits);
    }
    fn sink(self: *Collector) Sink {
        return .{ .ctx = self, .boundary = boundary, .logits = logitsRow };
    }
    fn boundary(ctx: *anyopaque, window: usize, b: usize, rows: Arr) anyerror!void {
        const self: *Collector = @ptrCast(@alignCast(ctx));
        const bytes = try hostBytes(self.a, rows);
        defer self.a.free(bytes);
        try self.rows[window][b].appendSlice(self.a, bytes);
    }
    fn logitsRow(ctx: *anyopaque, window: usize, logits: Arr) anyerror!void {
        const self: *Collector = @ptrCast(@alignCast(ctx));
        if (self.logits[window].len != 0) self.a.free(self.logits[window]);
        self.logits[window] = try hostBytes(self.a, logits);
    }
};

fn hostBytes(a: std.mem.Allocator, x: Arr) ![]u8 {
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_contiguous(&c, x, false, mlx.gpuStream()));
    try mlx.check(mlx.mlx_array_eval(c));
    const n = mlx.mlx_array_size(c) * mlx.mlx_array_itemsize(c);
    const data: [*]const u8 = @ptrCast(mlx.mlx_array_data_uint8(c) orelse return error.MlxArrayDataNull);
    return a.dupe(u8, data[0..n]);
}

/// The native capture's own window-major loop: a fresh request per window, `chunk`-row forwards.
fn windowMajor(net: *const forward.Model, request: *forward.Request, windows: []const []const u32, chunk: usize, out: *Collector) !void {
    const Hook = struct {
        out: *Collector,
        window: usize,
        fn append(ctx: *anyopaque, b: usize, rows: Arr) anyerror!void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            return Collector.boundary(self.out, self.window, b, rows);
        }
    };
    for (windows, 0..) |ids, w| {
        request.reset();
        var hook = Hook{ .out = out, .window = w };
        request.boundaries = .{ .ctx = &hook, .append = Hook.append };
        defer request.boundaries = null;
        var cursor: usize = 0;
        while (cursor < ids.len) {
            const end = @min(ids.len, cursor + chunk);
            const input = mlx.mlx_array_new_data(ids[cursor..end].ptr, &.{ 1, @intCast(end - cursor) }, 2, .uint32);
            defer _ = mlx.mlx_array_free(input);
            const logits = try net.forwardLast(request, input, true);
            defer _ = mlx.mlx_array_free(logits);
            try mlx.check(mlx.mlx_array_eval(logits));
            if (end == ids.len) try Collector.logitsRow(out, w, logits);
            cursor = end;
        }
    }
}

test "GLM layer-major prefill writes window-major's logits and every boundary byte for byte at any batch size" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const base = @import("glm5_model.zig");
    base.reference_numerics = true;
    defer base.reference_numerics = false;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try fixture.writeCheckpoint(a, tmp.dir, 0x5eed);
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = path_buf[0..try tmp.dir.realPath(testing.io, &path_buf)];
    var cfg = try model.parseConfig(testing.io, a, path);
    defer cfg.deinit(a);
    const s = mlx.gpuStream();
    var weights = try @import("glm5_diagnostic.zig").loadWeightsBounded(testing.io, a, path, s, true, std.math.maxInt(u64));
    defer weights.deinit();
    const stream = @import("expert_stream.zig");
    var engine = try stream.Engine.initWithOptions(a, path, cfg.expertGeometry(), 2 * (cfg.num_hidden_layers - cfg.first_k_dense_replace) * try stream.expertBytesFor(a, path, cfg.expertGeometry(), .bf16_individual), s, .{ .layout = .bf16_individual });
    defer engine.deinit();
    var net = try forward.Model.loadStreamed(a, cfg, &weights, s, .{ .engine = &engine, .max_tokens = 64, .max_chunk = 12 });
    defer net.deinit();
    var request = try forward.Request.init(a, cfg.num_hidden_layers);
    defer request.deinit();
    request.dense_prefill = true;
    request.decode_async = false;
    request.prefill_sync_layers = 1;

    var prng = std.Random.DefaultPrng.init(7);
    const lengths = [_]usize{ 5, 13, 12, 3, 10, 7, 11 };
    var storage: [lengths.len][16]u32 = undefined;
    var windows: [lengths.len][]const u32 = undefined;
    for (&storage, &windows, lengths) |*ids, *w, n| {
        for (ids) |*id| id.* = prng.random().uintLessThan(u32, fixture.Tiny.vocab);
        w.* = ids[0..n];
    }
    for ([_]usize{ 4, 12 }) |chunk| {
        var reference = try Collector.init(a, windows.len);
        defer reference.deinit();
        try windowMajor(&net, &request, &windows, chunk, &reference);
        request.reset();
        for (reference.logits) |row| {
            try testing.expectEqual(@as(usize, fixture.Tiny.vocab * 2), row.len);
            try testing.expect(!std.mem.allEqual(u8, row, 0));
        }
        for ([_]usize{ 1, 2, 3, 5 }) |batch| {
            var actual = try Collector.init(a, windows.len);
            defer actual.deinit();
            var first: usize = 0;
            while (first < windows.len) : (first += batch) {
                const end = @min(windows.len, first + batch);
                var shifted = try Collector.init(a, end - first);
                defer shifted.deinit();
                _ = try prefill(a, &net, windows[first..end], chunk, shifted.sink());
                for (first..end) |w| {
                    for (&actual.rows[w], &shifted.rows[w - first]) |*dst, src| try dst.appendSlice(a, src.items);
                    actual.logits[w] = try a.dupe(u8, shifted.logits[w - first]);
                }
            }
            for (reference.rows, actual.rows, reference.logits, actual.logits) |want_rows, got_rows, want, got| {
                try testing.expectEqualSlices(u8, want, got);
                for (want_rows, got_rows) |want_b, got_b| {
                    try testing.expect(want_b.items.len > 0);
                    try testing.expectEqualSlices(u8, want_b.items, got_b.items);
                }
            }
        }
    }
    try testing.expect(net.expert_stream.?.pinned_applies > 0);
}
