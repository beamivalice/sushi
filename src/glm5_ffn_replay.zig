//! Offline GLM frozen-prefix extraction for verified independent-window boundaries.
const std = @import("std");
pub const mlx = @import("mlx.zig");
pub const log = @import("log.zig");
pub const io_util = @import("io_util.zig");
const model = @import("model.zig");
const forward = @import("glm5_forward.zig");
const base = @import("glm5_model.zig");
const native = @import("glm5_diagnostic.zig");

fn append(fd: std.c.fd_t, array: mlx.mlx_array, stream: mlx.mlx_stream) !void {
    var contiguous = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(contiguous);
    try mlx.check(mlx.mlx_contiguous(&contiguous, array, false, stream));
    try mlx.check(mlx.mlx_array_eval(contiguous));
    const n = mlx.mlx_array_size(contiguous);
    const bytes: []const u8 = switch (mlx.mlx_array_dtype(contiguous)) {
        .bfloat16 => blk: {
            const p = mlx.mlx_array_data_bfloat16(contiguous) orelse return error.UnreadablePrefix;
            for (p[0..n]) |v| if (v & 0x7f80 == 0x7f80) return error.NonfinitePrefix;
            break :blk std.mem.sliceAsBytes(p[0..n]);
        },
        .float32 => blk: {
            const p = mlx.mlx_array_data_float32(contiguous) orelse return error.UnreadablePrefix;
            for (p[0..n]) |v| if (!std.math.isFinite(v)) return error.NonfinitePrefix;
            break :blk std.mem.sliceAsBytes(p[0..n]);
        },
        .uint32 => std.mem.sliceAsBytes((mlx.mlx_array_data_uint32(contiguous) orelse return error.UnreadablePrefix)[0..n]),
        else => return error.InvalidPrefixDtype,
    };
    var done: usize = 0;
    while (done < bytes.len) {
        const written = std.c.write(fd, bytes[done..].ptr, bytes.len - done);
        if (written < 0 and std.c._errno().* == @backingInt(std.c.E.INTR)) continue;
        if (written <= 0) return error.PrefixWriteFailed;
        done += @intCast(written);
    }
}

pub fn main(init: std.process.Init) !void {
    const a = init.gpa;
    const io = init.io;
    var iterator = try std.process.Args.Iterator.initAllocator(init.minimal.args, a);
    defer iterator.deinit();
    _ = iterator.next();
    const pack = iterator.next() orelse return error.ExpectedLayerPack;
    const index = try std.fmt.parseInt(usize, iterator.next() orelse return error.ExpectedLayer, 10);
    const input_path = iterator.next() orelse return error.ExpectedInputBoundary;
    const out = iterator.next() orelse return error.ExpectedOutput;
    const windows = try std.fmt.parseInt(usize, iterator.next() orelse return error.ExpectedWindows, 10);
    const length = try std.fmt.parseInt(usize, iterator.next() orelse return error.ExpectedWindowLength, 10);
    if (iterator.next() != null or windows == 0 or length == 0 or length > 512) return error.InvalidReplayArguments;
    var cfg = try model.parseConfig(io, a, pack);
    defer cfg.deinit(a);
    if (!cfg.isGlm5() or cfg.hc_count != 4 or index < cfg.first_k_dense_replace) return error.InvalidGlmConfig;
    const stream = mlx.gpuStream();
    if (cfg.expert_layout == .bf16_individual) base.enterTeacher();
    var weights = try native.loadWeightsBounded(io, a, pack, stream, true, 8 << 30);
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
    if (std.c.fstat(fd, &st) != 0 or !std.c.S.ISREG(@intCast(st.mode)) or st.size != try std.math.mul(usize, window_bytes, windows)) return error.InvalidBoundaryLength;
    try std.Io.Dir.cwd().createDirPath(io, out);
    var fds: [7]std.c.fd_t = @splat(-1);
    defer for (fds) |handle| {
        if (handle >= 0) _ = std.c.close(handle);
    };
    for ([_][]const u8{ "input.bin", "residual.bin", "post.bin", "comb.bin", "shared.bin", "ids.bin", "scores.bin" }, &fds) |name, *handle| {
        const output = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ out, name }, 0);
        defer a.free(output);
        handle.* = std.c.open(output, .{ .ACCMODE = .WRONLY, .CREAT = true, .EXCL = true, .NOFOLLOW = true }, @as(std.c.mode_t, 0o644));
        if (handle.* < 0) return error.PrefixOutputExists;
    }
    const raw = try a.alloc(u16, elements);
    defer a.free(raw);
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
        const routing = try replay.routing(&ops, prefix.input);
        const shared = if (replay.shared) |mlp| try mlp.apply(&ops, prefix.input, cfg.glm_swiglu_limit) else try ops.zeros(&.{ 1, @intCast(length), @intCast(cfg.hidden_size) }, .bfloat16);
        for ([_]mlx.mlx_array{ prefix.input, prefix.residual, prefix.post, prefix.comb, shared, routing.indices, routing.scores }, fds) |array, handle| try append(handle, array, stream);
        for (fds) |handle| if (std.c.fsync(handle) != 0) return error.PrefixSyncFailed;
        std.debug.print("layer {d}: prefix window {d}/{d}\n", .{ index, window + 1, windows });
    }
}
