//! The image encode on the arena (`hy4_encode_plan`, docs/bill_redesign.md stage 4c) on the tinies: a served load's
//! tower runs each image as one execution with the pool's bytes and no ask of the pool, and writes the request's Media.
const std = @import("std");
const testing = std.testing;
const mlx = @import("../mlx.zig");
const core = @import("../tensor/core.zig");
const contract = @import("../cooler_contract_test.zig");
const hy4_contract = @import("hy4_contract.zig");
const hy4_held = @import("hy4_held.zig");
const hy4_encode_plan = @import("hy4_encode_plan.zig");
const hy4_media = @import("hy4_media.zig");

/// An image's encode into a media of its own rows, in the load's region when `planned`, else on the pool: its rows'
/// bytes, the pool's asks and the wide executions it took.
fn encodeOnce(s: *hy4_held.Held, pixels: []const u8, gh: u32, gw: u32, planned: bool, out: *std.ArrayList(u8)) !struct { asks: u64, wide: u64 } {
    const m = s.ld.xfm.arch_state.hy4flash.model.?;
    const t = &s.tower.?.tower.?.arch.hy4flash.tower;
    var media = blk: {
        const scope = mlx.engine.enter(mlx.engine.stream());
        defer scope.end();
        break :blk try hy4_encode_plan.makeMedia(t.v, t.rowsOf(gh, gw));
    };
    defer _ = mlx.mlx_array_free(media);
    mlx.engine.close();
    const before = core.stats();
    try hy4_encode_plan.run(if (planned) m.plans.? else null, t, &media, 0, pixels, gh, gw);
    try mlx.check(mlx.mlx_array_eval(media));
    const after = core.stats();
    const data = mlx.mlx_array_data_float32(media) orelse return error.TestNoRows;
    try out.appendSlice(testing.allocator, std.mem.sliceAsBytes(data[0 .. @as(usize, t.rowsOf(gh, gw)) * t.v.out_hidden]));
    return .{ .asks = after.asks - before.asks, .wide = after.wide_executions - before.wide_executions };
}

test "hy4 plan: an image's encode in the region gives the pool's rows in one wide execution and asks the pool nothing, poison and canaries on, at one attention chunk, several, a short last one and the most patches the load accepts (HY4_BF16_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const s = try hy4_held.acquire(.bf16_vision);
    const turn = hy4_contract.openTurn(s.ld.h, &s.ld.config);
    defer turn.end();
    const t = &s.tower.?.tower.?.arch.hy4flash.tower;
    const most = hy4_media.maxImagePatches(hy4_media.preproc(&s.ld.config));
    const m: u32 = t.v.merge;
    // 64 patches (one chunk of 512 query rows), 1,024 (two whole), 640 (a short last one of 128), and the most patches
    // the load accepts as a square and one merge cell wide.
    const side: u32 = @intCast(std.math.sqrt(most) / m * m);
    const grids = [_][2]u32{ .{ 8, 8 }, .{ 32, 32 }, .{ 32, 20 }, .{ side, side }, .{ @intCast(most / m), m } };
    core.arena_check.byte = 0xFF;
    defer core.arena_check.byte = null;
    const hits = core.arena_check.hits;
    var rng = std.Random.DefaultPrng.init(0x1a6e);
    for (grids) |g| {
        const n: usize = @as(usize, g[0]) * g[1];
        const vals = try a.alloc(f32, n * 3 * @as(usize, t.v.patch) * t.v.patch);
        defer a.free(vals);
        for (vals) |*x| x.* = rng.random().floatNorm(f32);
        var pool: std.ArrayList(u8) = .empty;
        defer pool.deinit(a);
        var arena: std.ArrayList(u8) = .empty;
        defer arena.deinit(a);
        const on_pool = try encodeOnce(s, std.mem.sliceAsBytes(vals), g[0], g[1], false, &pool);
        const on_arena = try encodeOnce(s, std.mem.sliceAsBytes(vals), g[0], g[1], true, &arena);
        errdefer std.debug.print("an encode of {d} x {d} patches: {d} wide executions, {d} asks of the pool\n", .{ g[0], g[1], on_arena.wide, on_arena.asks });
        try testing.expectEqual(@as(u64, 0), on_pool.wide);
        try testing.expectEqual(@as(u64, 1), on_arena.wide);
        try testing.expectEqual(@as(u64, 0), on_arena.asks);
        try testing.expectEqualSlices(u8, pool.items, arena.items);
    }
    try testing.expectEqual(hits, core.arena_check.hits);
}

test "hy4 plan: a request of two images encodes each into its Media past the seal asking the closed pool nothing, and its prompt answers as the tower's rows joined do (HY4_BF16_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const s = try hy4_held.acquire(.bf16_vision);
    const tower = &s.tower.?.tower.?;
    const t = &tower.arch.hy4flash.tower;
    const grids = [_][2]u32{ .{ 6, 10 }, .{ 32, 20 } };
    var pixels: [2][]f32 = undefined;
    var rng = std.Random.DefaultPrng.init(0x3ed1a);
    var rows: usize = 0;
    for (grids, &pixels) |g, *px| {
        px.* = try a.alloc(f32, @as(usize, g[0]) * g[1] * 3 * t.v.patch * t.v.patch);
        for (px.*) |*x| x.* = rng.random().floatNorm(f32);
        rows += t.rowsOf(g[0], g[1]);
    }
    defer for (pixels) |px| a.free(px);
    // The tower's rows of each image joined, as the request's rows were before Media.
    const joined = blk: {
        const turn = hy4_contract.openTurn(s.ld.h, &s.ld.config);
        defer turn.end();
        const scope = mlx.engine.enter(mlx.engine.stream());
        defer scope.end();
        var parts: [2]mlx.mlx_array = undefined;
        for (grids, pixels, &parts) |g, px, *part| {
            const host = try mlx.newData(px.ptr, &[_]c_int{ @intCast(@as(usize, g[0]) * g[1]), @intCast(3 * t.v.patch * t.v.patch) }, .float32);
            defer _ = mlx.mlx_array_free(host);
            part.* = try tower.forward(host, g[0], g[1]);
        }
        defer for (parts) |p| {
            _ = mlx.mlx_array_free(p);
        };
        var out = mlx.mlx_array_new();
        const vec = mlx.mlx_vector_array_new_data(&parts, 2);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 1, mlx.engine.stream()));
        try mlx.check(mlx.mlx_array_eval(out));
        break :blk out;
    };
    defer _ = mlx.mlx_array_free(joined);
    // The encode as `vision_queue.runVisionEncode` runs it for a tower that writes Media, the pool closed to a census.
    const media = blk: {
        const turn = @import("../arch/handle.zig").turnFor(s.ld.h, &s.ld.config);
        defer turn.end();
        try core.closePool(.census);
        defer core.closePool(.open) catch {};
        const before = core.stats();
        var md = try tower.makeMedia(rows);
        errdefer _ = mlx.mlx_array_free(md);
        var row0: usize = 0;
        for (grids, pixels) |g, px| {
            try tower.encodeInto(s.ld.xfm, &md, row0, std.mem.sliceAsBytes(px), g[0], g[1]);
            row0 += t.rowsOf(g[0], g[1]);
        }
        try mlx.check(mlx.mlx_array_eval(md));
        const after = core.stats();
        errdefer std.debug.print("the encode asked the closed pool {d} times, {d} B\n", .{ after.closed_asks - before.closed_asks, after.closed_ask_bytes - before.closed_ask_bytes });
        try testing.expectEqual(before.closed_asks, after.closed_asks);
        try testing.expectEqual(before.wide_executions + 2, after.wide_executions);
        break :blk md;
    };
    defer _ = mlx.mlx_array_free(media);
    const len = rows * t.v.out_hidden;
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(mlx.mlx_array_data_float32(joined).?[0..len]), std.mem.sliceAsBytes(mlx.mlx_array_data_float32(media).?[0..len]));
    const ids = try a.alloc(u32, rows + 8);
    defer a.free(ids);
    for (ids, 0..) |*id, i| id.* = if (i >= 3 and i < 3 + rows) s.ld.config.image_token_id else @intCast(1000 + 7 * i);
    var answers: [2]std.ArrayList(u32) = undefined;
    for ([_]mlx.mlx_array{ joined, media }, &answers) |ve, *got| {
        var ctx = s.ld.xfm.defaultCtx();
        ctx.vision_embeddings = ve;
        got.* = try contract.serialWith(s.ld, ids, 6, .{ .temperature = 0 }, .{ .ctx = ctx });
    }
    defer for (&answers) |*got| got.deinit(a);
    try testing.expectEqualSlices(u32, answers[0].items, answers[1].items);
}

test "hy4 plan: an encode that asks past the region its load holds is refused by name, and the engine's latch names the region (HY4_BF16_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const s = try hy4_held.acquire(.bf16_vision);
    const turn = @import("../arch/handle.zig").turnFor(s.ld.h, &s.ld.config);
    defer turn.end();
    const p = s.ld.xfm.arch_state.hy4flash.model.?.plans.?;
    const t = &s.tower.?.tower.?.arch.hy4flash.tower;
    try testing.expect(p.encode.scratch > 0);
    const g = [2]u32{ 32, 32 };
    const vals = try a.alloc(f32, @as(usize, g[0]) * g[1] * 3 * @as(usize, t.v.patch) * t.v.patch);
    defer a.free(vals);
    @memset(vals, 0.25);
    var media = blk: {
        const scope = mlx.engine.enter(mlx.engine.stream());
        defer scope.end();
        break :blk try hy4_encode_plan.makeMedia(t.v, t.rowsOf(g[0], g[1]));
    };
    defer _ = mlx.mlx_array_free(media);
    const held = core.arenaLayout().?;
    try core.arenaRelease();
    var small = held;
    small.g = 64 << 10;
    try core.arenaFor(small);
    try testing.expectError(error.WideFull, hy4_encode_plan.run(p, t, &media, 0, std.mem.sliceAsBytes(vals), g[0], g[1]));
    var said: [512]u8 = undefined;
    try testing.expect(std.mem.indexOf(u8, mlx.takeError(&said) orelse "", "WideFull: a wide execution's scratch reached") != null);
    try testing.expectEqual(@as(u64, 64), core.refused_wide.?.room >> 10);
    try core.arenaRelease();
    try core.arenaFor(held);
}

test "hy4 plan: an image the processor resizes past the most patches the encode is planned for is refused by name at admission; one at the most is not" {
    const a = testing.allocator;
    var config = try @import("../model.zig").parseConfigFromJson(a, @import("hy4_config.zig").release_file);
    defer config.deinit(a);
    const request_media = @import("../request_media.zig");
    const vp = hy4_media.preproc(&config);
    const most = hy4_media.maxImagePatches(vp);
    // One pixel wide: the resize scales it up to its least pixels and keeps a whole cell across, which passes its most.
    const thin = hy4_media.planImage(vp, 1, 5_000_000).?;
    try testing.expect(@as(u64, thin.grid_h) * thin.grid_w > most);
    const square = hy4_media.planImage(vp, 4000, 4000).?;
    try testing.expect(@as(u64, square.grid_h) * square.grid_w <= most);
    const images = [_]@import("../chat.zig").ImageData{
        .{ .pixels = &.{}, .width = square.width, .height = square.height, .grid_h = square.grid_h, .grid_w = square.grid_w },
        .{ .pixels = &.{}, .width = thin.width, .height = thin.height, .grid_h = thin.grid_h, .grid_w = thin.grid_w },
    };
    try testing.expect(request_media.pastPlanRefusal(&config, images[0..1]) == null);
    const fault = request_media.pastPlanRefusal(&config, &images) orelse return error.TestExpectedRefusal;
    try testing.expect(std.mem.startsWith(u8, fault.text(), "image 2 resizes to "));
    try testing.expect(!fault.server_error);
}

test "hy4 bill: a planned encode's request bill holds its Media and its images' host pixels" {
    const a = testing.allocator;
    var config = try @import("../model.zig").parseConfigFromJson(a, @import("hy4_config.zig").release_file);
    defer config.deinit(a);
    const v = config.hy4flash.?.vision.?;
    const pixels: u64 = 9216 * 3 * @as(u64, v.patch) * v.patch * @sizeOf(f32);
    const rows: usize = 2400;
    const media = @import("hy4_encode_plan.zig").mediaBytes(v, rows);
    try testing.expectEqual(media + pixels, @import("hy4_media.zig").mediaBill(&config, 9216, pixels, rows));
}
