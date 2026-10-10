//! Offline GLM trunk calibration: per-projection input covariance X^T X, replayed from verified block boundaries.
const std = @import("std");
pub const mlx = @import("mlx.zig");
pub const log = @import("log.zig");
pub const io_util = @import("io_util.zig");
const model = @import("model.zig");
const forward = @import("glm5_forward.zig");
const base = @import("glm5_model.zig");
const native = @import("glm5_diagnostic.zig");
const Arr = mlx.mlx_array;

pub const Collector = struct {
    const Slot = struct { cov: Arr, tokens: u64 };
    allocator: std.mem.Allocator,
    stream: mlx.mlx_stream,
    slots: std.AutoHashMap(usize, Slot),

    pub fn init(allocator: std.mem.Allocator, stream: mlx.mlx_stream) Collector {
        return .{ .allocator = allocator, .stream = stream, .slots = .init(allocator) };
    }

    pub fn deinit(self: *Collector) void {
        var it = self.slots.valueIterator();
        while (it.next()) |slot| _ = mlx.mlx_array_free(slot.cov);
        self.slots.deinit();
    }

    pub fn tap(self: *Collector) base.LinearTap {
        return .{ .ctx = self, .observe = observe };
    }

    /// Writes `<weight name>` -> covariance for every tapped projection of `layer`, plus the shared token count.
    pub fn save(self: *Collector, weights: *const model.Weights, layer_prefix: []const u8, path: []const u8) !void {
        const map = mlx.mlx_map_string_to_array_new();
        defer _ = mlx.mlx_map_string_to_array_free(map);
        var tokens: u32 = 0;
        var named: usize = 0;
        var it = weights.map.iterator();
        var key_buf: [256]u8 = undefined;
        while (it.next()) |entry| {
            const name = entry.key_ptr.*;
            if (!std.mem.startsWith(u8, name, layer_prefix) or !std.mem.endsWith(u8, name, ".weight")) continue;
            const slot = self.slots.get(@intFromPtr(entry.value_ptr.ctx)) orelse continue;
            tokens = @intCast(slot.tokens);
            named += 1;
            try mlx.check(mlx.mlx_map_string_to_array_insert(map, try std.fmt.bufPrintSentinel(&key_buf, "{s}", .{name}, 0), slot.cov));
        }
        if (named != self.slots.count()) return error.TrunkCovUnnamedProjection;
        const count = mlx.mlx_array_new_data(&tokens, &[_]c_int{1}, 1, .uint32);
        defer _ = mlx.mlx_array_free(count);
        try mlx.check(mlx.mlx_map_string_to_array_insert(map, "tokens", count));
        const meta = mlx.mlx_map_string_to_string_new();
        defer _ = mlx.mlx_map_string_to_string_free(meta);
        const path_z = try std.fmt.allocPrintSentinel(self.allocator, "{s}", .{path}, 0);
        defer self.allocator.free(path_z);
        try mlx.check(mlx.mlx_save_safetensors(path_z.ptr, map, meta));
    }

    /// Rows are every position of `x`; the sum stays F32 and is evaluated per call so the graph stays bounded.
    fn observe(ctx: *anyopaque, w: Arr, x: Arr) anyerror!void {
        const self: *Collector = @ptrCast(@alignCast(ctx));
        const width = mlx.getShape(x)[mlx.getShape(x).len - 1];
        var ops = base.Ops{ .s = self.stream };
        defer ops.deinit();
        const rows = try ops.cast(try ops.reshape(x, &.{ -1, width }), .float32);
        const gram = try ops.binary(.mm, try ops.transpose(rows, &.{ 1, 0 }), rows);
        const entry = try self.slots.getOrPut(@intFromPtr(w.ctx));
        const total = if (entry.found_existing) try ops.binary(.add, entry.value_ptr.cov, gram) else gram;
        const kept = try ops.result(total);
        errdefer _ = mlx.mlx_array_free(kept);
        try mlx.check(mlx.mlx_array_eval(kept));
        const tokens: u64 = mlx.mlx_array_size(rows) / @as(usize, @intCast(width));
        if (entry.found_existing) {
            _ = mlx.mlx_array_free(entry.value_ptr.cov);
            entry.value_ptr.* = .{ .cov = kept, .tokens = entry.value_ptr.tokens + tokens };
        } else entry.value_ptr.* = .{ .cov = kept, .tokens = tokens };
    }
};

pub fn main(init: std.process.Init) !void {
    const a = init.gpa;
    const io = init.io;
    var iterator = try std.process.Args.Iterator.initAllocator(init.minimal.args, a);
    defer iterator.deinit();
    _ = iterator.next();
    const pack = iterator.next() orelse return error.ExpectedTeacherPack;
    const index = try std.fmt.parseInt(usize, iterator.next() orelse return error.ExpectedLayer, 10);
    const input_path = iterator.next() orelse return error.ExpectedInputBoundary;
    const out = iterator.next() orelse return error.ExpectedOutput;
    const windows = try std.fmt.parseInt(usize, iterator.next() orelse return error.ExpectedWindows, 10);
    const length = try std.fmt.parseInt(usize, iterator.next() orelse return error.ExpectedWindowLength, 10);
    if (iterator.next() != null or windows == 0 or length == 0 or length > 512) return error.InvalidReplayArguments;
    var cfg = try model.parseConfig(io, a, pack);
    defer cfg.deinit(a);
    if (!cfg.isGlm5() or cfg.hc_count != 4 or cfg.expert_layout != .bf16_individual) return error.InvalidGlmConfig;
    const stream = mlx.gpuStream();
    base.enterTeacher();
    var weights = try native.loadWeightsBounded(io, a, pack, stream, true, 32 << 30);
    defer weights.deinit();
    var replay = try forward.FfnPrefixReplay.load(cfg, &weights, index, stream);
    defer replay.deinit();
    var request = try forward.Request.init(a, cfg.num_hidden_layers);
    defer request.deinit();
    request.dense_prefill = true;
    const path = try a.dupeSentinel(u8, input_path, 0);
    defer a.free(path);
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY, .NOFOLLOW = true }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.MissingInputBoundary;
    defer _ = std.c.close(fd);
    const elements = try std.math.mul(usize, length, @as(usize, cfg.hidden_size) * 4);
    const window_bytes = try std.math.mul(usize, elements, 2);
    var st: std.c.Stat = undefined;
    if (std.c.fstat(fd, &st) != 0 or !std.c.S.ISREG(@intCast(st.mode)) or st.size < try std.math.mul(usize, window_bytes, windows)) return error.InvalidBoundaryLength;
    const raw = try a.alloc(u16, elements);
    defer a.free(raw);
    var collector = Collector.init(a, stream);
    defer collector.deinit();
    base.linear_tap = collector.tap();
    defer base.linear_tap = null;
    for (0..windows) |window| {
        request.reset();
        try @import("expert_io.zig").readExact(fd, std.mem.sliceAsBytes(raw), window * window_bytes);
        for (raw) |v| if (v & 0x7f80 == 0x7f80) return error.NonfiniteBoundary;
        const h = mlx.mlx_array_new_data(raw.ptr, &[_]c_int{ 1, @intCast(length), 4, @intCast(cfg.hidden_size) }, 4, .bfloat16);
        defer _ = mlx.mlx_array_free(h);
        var prefix = try replay.prepare(&request, h);
        defer prefix.deinit();
        var ops = base.Ops{ .s = stream };
        defer ops.deinit();
        const mlp = replay.shared orelse return error.InvalidGlmLayer;
        try mlx.check(mlx.mlx_array_eval(try mlp.apply(&ops, prefix.input, cfg.glm_swiglu_limit)));
        if ((window + 1) % 100 == 0 or window + 1 == windows) std.debug.print("layer {d}: cov window {d}/{d}\n", .{ index, window + 1, windows });
    }
    var buf: [256]u8 = undefined;
    try collector.save(&weights, try std.fmt.bufPrint(&buf, "{s}.layers.{d}.", .{ cfg.weight_prefix, index }), out);
}

test "GLM trunk cov collector sums X^T X per projection across calls" {
    const a = std.testing.allocator;
    const stream = mlx.gpuStream();
    var ops = base.Ops{ .s = stream };
    defer ops.deinit();
    const in: usize = 128;
    var wdata: [4 * 128]u16 = @splat(0x3f80);
    const linear = base.Linear{ .w = try ops.own(mlx.mlx_array_new_data(&wdata, &[_]c_int{ 4, 128 }, 2, .bfloat16)), .input = 128, .output = 4 };
    var collector = Collector.init(a, stream);
    defer collector.deinit();
    base.linear_tap = collector.tap();
    defer base.linear_tap = null;

    var host = try a.alloc(f32, in * in);
    defer a.free(host);
    @memset(host, 0);
    for ([_]usize{ 3, 2 }) |rows| {
        const xs = try a.alloc(f32, rows * in);
        defer a.free(xs);
        for (xs, 0..) |*v, i| v.* = @floatFromInt(@as(i32, @intCast((i * 7 + rows) % 5)) - 2);
        const x = try ops.cast(try ops.own(mlx.mlx_array_new_data(xs.ptr, &[_]c_int{ 1, @intCast(rows), @intCast(in) }, 3, .float32)), .bfloat16);
        _ = try linear.apply(&ops, x);
        for (0..rows) |r| for (0..in) |i| for (0..in) |j| {
            host[i * in + j] += xs[r * in + i] * xs[r * in + j];
        };
    }
    const slot = collector.slots.get(@intFromPtr(linear.w.ctx)) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(u64, 5), slot.tokens);
    try mlx.check(mlx.mlx_array_eval(slot.cov));
    try std.testing.expectEqualSlices(f32, host, mlx.mlx_array_data_float32(slot.cov).?[0 .. in * in]);
}
