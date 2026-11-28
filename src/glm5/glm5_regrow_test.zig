//! A native GLM decode step past the latent's page grid: the cache it grows holds one layer's old copy at once.
const std = @import("std");
const mlx = @import("../mlx.zig");
const model = @import("../model.zig");
const fwd = @import("glm5_forward.zig");
const mlxPeakOver = @import("../test_support.zig").mlxPeakOver;

/// 44 layers hold 11 MLA layers (3, 7, ..., 43), one in each group of four the staged decode evaluates, as GLM-5.3 does.
const tiny_layers: u32 = 44;

const Step = struct {
    net: *const fwd.Model,
    req: *fwd.Request,
    ids: mlx.mlx_array,
    pub fn run(self: @This()) !void {
        const result = try self.net.forwardLast(self.req, self.ids, true);
        _ = mlx.mlx_array_free(result);
    }
};

/// Bytes of each buffer of layer 3's state that `forwardLast` rebuilds on a growth, and their rows.
const Buffers = struct {
    bytes: [4]u64,
    rows: [4]u64,
    fn of(state: *const fwd.LayerState) Buffers {
        const arrays = [_]mlx.mlx_array{ state.attention.latent.array, state.attention.latent_scales.array, state.attention.latent_biases.array, state.attention.pooled.array };
        var out: Buffers = .{ .bytes = @splat(0), .rows = @splat(0) };
        for (arrays, 0..) |a, i| if (a.ctx != null) {
            out.bytes[i] = mlx.mlx_array_size(a) * mlx.mlx_array_itemsize(a);
            out.rows[i] = @intCast(mlx.getShape(a)[0]);
        };
        return out;
    }
};

fn prefill(net: *const fwd.Model, req: *fwd.Request, rows: usize) !void {
    var ids_host: [1024]u32 = undefined;
    for (&ids_host, 0..) |*id, i| id.* = @intCast(i % 4);
    var left = rows;
    while (left > 0) {
        const n: usize = @min(left, ids_host.len);
        const ids = mlx.mlx_array_new_data(&ids_host, &[_]c_int{ 1, @intCast(n) }, 2, .uint32);
        defer _ = mlx.mlx_array_free(ids);
        const out = try net.forwardLast(req, ids, true);
        _ = mlx.mlx_array_free(out);
        left -= n;
    }
}

test "a native GLM decode step past the latent's page grid holds beside the grown cache one buffer's old copy, whatever the staged schedule has in flight" {
    try mlx.requireGpu();
    const a = std.testing.allocator;
    const s = mlx.gpuStream();
    var over: usize = 0;
    var grows: usize = 0;
    var weights = model.Weights.init(a);
    defer weights.deinit();
    const cfg = try fwd.nonzeroDecodeFixtureLayers(&weights, tiny_layers, 16384);
    var net = try fwd.Model.load(a, cfg, &weights, s);
    defer net.deinit();
    const one = mlx.mlx_array_new_data(&[_]u32{1}, &[_]c_int{ 1, 1 }, 2, .uint32);
    defer _ = mlx.mlx_array_free(one);
    // Both ends of the stored latent: BF16 rows grow in 256-row steps, kv8 codes and their scales in 4,096-row steps.
    for ([_]struct { bits: u8, from: usize, steps: usize }{ .{ .bits = 0, .from = 8184, .steps = 16 }, .{ .bits = 8, .from = 8184, .steps = 16 } }) |case| {
        var req = try fwd.Request.init(a, tiny_layers);
        defer req.deinit();
        try req.setLatentBits(case.bits);
        try prefill(&net, &req, case.from);
        const Held = struct { held: u64, grew: u64 };
        const steps = try a.alloc(Held, case.steps);
        defer a.free(steps);
        // The step's own transient: the dearest step that grows no buffer, in this run.
        var transient: u64 = 0;
        for (steps) |*step| {
            const before = Buffers.of(&req.layers[3]);
            const held = try mlxPeakOver(s, Step, .{ .net = &net, .req = &req, .ids = one });
            const after = Buffers.of(&req.layers[3]);
            var widest: u64 = 0;
            for (0..4) |i| if (after.rows[i] != before.rows[i]) {
                widest = @max(widest, after.bytes[i]);
            };
            step.* = .{ .held = held, .grew = widest };
            if (widest == 0) transient = @max(transient, held);
        }
        for (steps, 0..) |step, i| {
            if (step.grew == 0) continue;
            grows += 1;
            // Old copy and zero tail make up the new buffer whole; a buffer at a time holds one.
            if (step.held > step.grew + transient) {
                over += 1;
                std.debug.print("kv{d}: step {d} past row {d} holds {d} B past the grown cache, one buffer's old copy is {d} B and a step's transient {d} B\n", .{ if (case.bits == 0) @as(u8, 16) else case.bits, i, case.from + i, step.held, step.grew, transient });
            }
        }
    }
    try std.testing.expect(grows > 0);
    try std.testing.expectEqual(@as(usize, 0), over);
}
