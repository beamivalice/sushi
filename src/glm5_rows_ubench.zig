//! DIAGNOSTIC (SUSHI_GLM_ROWS_UBENCH=N): at load, N rounds per width B in 1..4 of one plain decode
//! row for each of B requests, two arms interleaved per round: B serial one-row forwards against one
//! grouped verification (`verifyGroups`, one single-row group per request, the DFlash2 capture taps).
//! Then one request's verification of a linear chain of 1..8 rows (the continuation of its window in
//! `_TEXT`, else synthetic ids), also interleaved per round.
//! Requests prefill distinct `_CTX`-token (default 1024) windows of `_TEXT` (else synthetic ids).
const std = @import("std");
const mlx = @import("mlx.zig");
const forward = @import("glm5_forward.zig");
const verifier = @import("glm5_dflash_model.zig");
const adapter = @import("glm5_dflash.zig");
const io_util = @import("io_util.zig");
const log = @import("log.zig");

const widths = 4;
const chain_widths = 8;
const taps = [_]u32{ 5, 14, 24, 33, 42 };

pub fn run(allocator: std.mem.Allocator, target: *forward.Model, latent_bits: u8, source: []const u32, ctx: usize, rounds: usize) !void {
    const io = std.Io.Threaded.global_single_threaded.io();
    var requests: [widths]forward.Request = undefined;
    var made: usize = 0;
    defer for (requests[0..made]) |*r| r.deinit();
    var next: [widths]u32 = undefined;
    for (&requests, 0..) |*request, b| {
        request.* = try forward.Request.initServing(allocator, target.layers.len);
        made += 1;
        try request.setLatentBits(latent_bits);
        const window = try allocator.alloc(u32, ctx);
        defer allocator.free(window);
        for (window, 0..) |*id, i| id.* = if (source.len > 0) source[(b * ctx + i) % source.len] else @intCast(1 + (i * 7919 + b * 104729) % 150000);
        const ids = mlx.mlx_array_new_data(window.ptr, &.{ 1, @intCast(ctx) }, 2, .uint32);
        defer _ = mlx.mlx_array_free(ids);
        const logits = try target.forwardLast(request, ids, true);
        defer _ = mlx.mlx_array_free(logits);
        var best = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(best);
        try mlx.check(mlx.mlx_argmax_axis(&best, logits, -1, false, target.s));
        try mlx.check(mlx.mlx_array_eval(best));
        var token: i32 = 0;
        try mlx.check(mlx.mlx_array_item_int32(&token, best));
        next[b] = @intCast(token);
    }
    var serial_ns: [widths + 1]u64 = @splat(0);
    var rows_ns: [widths + 1]u64 = @splat(0);
    const binding = verifier.bindDefaultSchedule();
    defer binding.restore();
    // Round 0 compiles every shape and is not counted.
    for (0..rounds + 1) |round| for (1..widths + 1) |width| {
        var sw = io_util.Stopwatch.init(io);
        for (0..width) |b| {
            var clone = try adapter.cloneRequest(&requests[b]);
            defer clone.deinit();
            const ids = mlx.mlx_array_new_data(&next[b], &.{ 1, 1 }, 2, .uint32);
            defer _ = mlx.mlx_array_free(ids);
            const logits = try target.forwardLast(&clone, ids, true);
            defer _ = mlx.mlx_array_free(logits);
            try mlx.check(mlx.mlx_array_eval(logits));
        }
        if (round > 0) serial_ns[width] += sw.read();
        var groups: [widths]verifier.Group = undefined;
        for (groups[0..width], 0..) |*g, b| g.* = .{ .request = &requests[b], .tokens = next[b .. b + 1], .parents = &.{-1} };
        var out: [widths]verifier.Verified = undefined;
        sw = io_util.Stopwatch.init(io);
        try verifier.verifyGroups(target, groups[0..width], &taps, .affine_rows_ffn, out[0..width]);
        if (round > 0) rows_ns[width] += sw.read();
        for (out[0..width]) |*v| v.deinit();
    };
    var chain_ns: [chain_widths + 1]u64 = @splat(0);
    for (0..rounds + 1) |round| for (1..chain_widths + 1) |width| {
        var tokens: [chain_widths]u32 = undefined;
        var parents: [chain_widths]i32 = undefined;
        for (0..width) |i| {
            tokens[i] = if (i == 0) next[0] else if (source.len > 0) source[(ctx + i) % source.len] else @intCast(1 + ((ctx + i) * 7919) % 150000);
            parents[i] = @as(i32, @intCast(i)) - 1;
        }
        const group = verifier.Group{ .request = &requests[0], .tokens = tokens[0..width], .parents = parents[0..width] };
        var out: [1]verifier.Verified = undefined;
        const sw = io_util.Stopwatch.init(io);
        try verifier.verifyGroups(target, &.{group}, &taps, .affine_rows_ffn, &out);
        if (round > 0) chain_ns[width] += sw.read();
        defer out[0].deinit();
        if (round == 0 and width > 1) {
            var serial = try adapter.cloneRequest(&requests[0]);
            defer serial.deinit();
            var differ: usize = 0;
            for (0..width) |i| {
                const ids = mlx.mlx_array_new_data(&tokens[i], &.{ 1, 1 }, 2, .uint32);
                defer _ = mlx.mlx_array_free(ids);
                const logits = try target.forwardLast(&serial, ids, true);
                defer _ = mlx.mlx_array_free(logits);
                if (!try sameRow(logits, out[0].logits, i)) differ += 1;
            }
            log.info("[glm-rows-ubench] ctx={d} chain of {d} rows: {d} of {d} rows' logits differ from serial decode\n", .{ ctx, width, differ, width });
        }
    };
    try forkRows(target, &requests[0], next[0], ctx, source);
    try ffnRows(target, rounds);
    const n: f64 = @floatFromInt(@max(rounds, 1));
    for (1..chain_widths + 1) |width| {
        const ms = @as(f64, @floatFromInt(chain_ns[width])) / n / 1e6;
        const before = @as(f64, @floatFromInt(chain_ns[@max(width - 1, 1)])) / n / 1e6;
        log.info("[glm-rows-ubench] ctx={d} chain of {d} rows: verify {d:.2} ms ({d:.2} ms over the row before)\n", .{ ctx, width, ms, if (width == 1) 0 else ms - before });
    }
    var sx: f64 = 0;
    var sy: f64 = 0;
    var sxx: f64 = 0;
    var sxy: f64 = 0;
    for (1..widths + 1) |width| {
        const serial = @as(f64, @floatFromInt(serial_ns[width])) / n / 1e6;
        const rows = @as(f64, @floatFromInt(rows_ns[width])) / n / 1e6;
        const w: f64 = @floatFromInt(width);
        sx += w;
        sy += rows;
        sxx += w * w;
        sxy += w * rows;
        log.info("[glm-rows-ubench] ctx={d} B={d} serial {d:.2} ms ({d:.2} ms/row) grouped {d:.2} ms ({d:.2} ms/row)\n", .{ ctx, width, serial, serial / w, rows, rows / w });
    }
    const k: f64 = widths;
    const per_row = (k * sxy - sx * sy) / (k * sxx - sx * sx);
    log.info("[glm-rows-ubench] grouped fit: fixed {d:.2} ms + {d:.2} ms per row ({d} rounds)\n", .{ (sy - per_row * sx) / k, per_row, rounds });
}

/// Bit-equality of branching 5- and 8-row trees against serial decode of each node's own path.
fn forkRows(target: *forward.Model, request: *forward.Request, first: u32, ctx: usize, source: []const u32) !void {
    const trees = [_][]const i32{ &.{ -1, 0, 0, 1, 1 }, &.{ -1, 0, 0, 1, 1, 2, 2, 3 }, &.{ -1, 0, 1, 0, 3, 4, 0, 6 } };
    for (trees) |parents| {
        var tokens: [chain_widths]u32 = undefined;
        for (parents, 0..) |_, i| tokens[i] = if (i == 0) first else if (source.len > 0) source[(ctx + 3 * i) % source.len] else @intCast(1 + ((ctx + 3 * i) * 7919) % 150000);
        const group = verifier.Group{ .request = request, .tokens = tokens[0..parents.len], .parents = parents };
        var out: [1]verifier.Verified = undefined;
        try verifier.verifyGroups(target, &.{group}, &taps, .affine_rows_ffn, &out);
        defer out[0].deinit();
        var differ: usize = 0;
        for (parents, 0..) |_, node| {
            var path: [chain_widths]u32 = undefined;
            var depth: usize = 0;
            var at: i32 = @intCast(node);
            while (at >= 0) : (at = parents[@intCast(at)]) {
                path[depth] = tokens[@intCast(at)];
                depth += 1;
            }
            var serial = try adapter.cloneRequest(request);
            defer serial.deinit();
            var logits = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(logits);
            while (depth > 0) {
                depth -= 1;
                const ids = mlx.mlx_array_new_data(&path[depth], &.{ 1, 1 }, 2, .uint32);
                defer _ = mlx.mlx_array_free(ids);
                _ = mlx.mlx_array_free(logits);
                logits = try target.forwardLast(&serial, ids, true);
            }
            if (!try sameRow(logits, out[0].logits, node)) differ += 1;
        }
        log.info("[glm-rows-ubench] ctx={d} tree of {d} rows: {d} of {d} rows' logits differ from serial decode\n", .{ ctx, parents.len, differ, parents.len });
    }
}

/// Whether `verified` row `row` is bit for bit the serial row's logits.
fn sameRow(serial: mlx.mlx_array, verified: mlx.mlx_array, row: usize) !bool {
    try mlx.check(mlx.mlx_array_eval(serial));
    try mlx.check(mlx.mlx_array_eval(verified));
    if (mlx.mlx_array_dtype(serial) != mlx.mlx_array_dtype(verified)) return false;
    const bytes = mlx.mlx_array_size(serial) * mlx.mlx_array_itemsize(serial);
    const one = mlx.mlx_array_data_uint8(serial) orelse return error.MlxArrayDataNull;
    const all = mlx.mlx_array_data_uint8(verified) orelse return error.MlxArrayDataNull;
    return std.mem.eql(u8, one[0..bytes], all[row * bytes ..][0..bytes]);
}

/// One routed layer's feed-forward at 1..8 rows, the verifier's own `apply`, each call settled.
fn ffnRows(target: *forward.Model, rounds: usize) !void {
    const io = std.Io.Threaded.global_single_threaded.io();
    const layer = for (target.layers, 0..) |l, i| {
        if (l.ffn == .moe) break i;
    } else return;
    const ffn = @import("glm5_dflash_ffn.zig");
    const reps = @max(rounds * 4, 8);
    for (1..chain_widths + 1) |width| {
        var total: u64 = 0;
        for (0..reps + 2) |rep| {
            var ops = @import("glm5_model.zig").Ops{ .s = target.s };
            defer ops.deinit();
            const key = try ops.slot();
            try mlx.check(mlx.mlx_random_key(key, 77 + rep));
            const x = try ops.slot();
            try mlx.check(mlx.mlx_random_normal(x, &.{ 1, @intCast(width), @intCast(target.cfg.hidden_size) }, 3, .bfloat16, 0, 1, key.*, target.s));
            try mlx.check(mlx.mlx_array_eval(x.*));
            const sw = io_util.Stopwatch.init(io);
            const y = try ffn.apply(target, layer, &ops, x.*);
            try mlx.check(mlx.mlx_array_eval(y));
            if (rep >= 2) total += sw.read();
        }
        log.info("[glm-rows-ubench] routed layer {d} feed-forward at {d} rows: {d:.3} ms\n", .{ layer, width, @as(f64, @floatFromInt(total)) / @as(f64, @floatFromInt(reps)) / 1e6 });
    }
}
