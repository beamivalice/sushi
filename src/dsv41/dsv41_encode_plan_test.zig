//! Tests for DeepSeek-V4.1's wide-region vision encode (`dsv41_encode_plan`) and `mediaBill` / `imagePastPlan`
//! Cooler-vs-MLX coexistence.
const std = @import("std");
const testing = std.testing;
const mlx = @import("../mlx.zig");
const model = @import("../model.zig");
const common = @import("../vision_common.zig");
const arch_media = @import("../arch/media.zig");
const request_media = @import("../request_media.zig");
const chat = @import("../chat.zig");
const core = @import("../tensor/core.zig");
const plan = @import("../tensor/plan/plan_api.zig");
const cfg_mod = @import("dsv41_config.zig");
const dsv41_vision = @import("dsv41_vision.zig");
const dsv41_media = @import("dsv41_media.zig");
const dsv41_plan = @import("dsv41_plan.zig");
const dsv41_encode_plan = @import("dsv41_encode_plan.zig");

const test_dim: u32 = 128;

fn makeWeights(v: cfg_mod.Vision, dim: u32) !model.Weights {
    const d: c_int = @intCast(v.hidden);
    const inter: c_int = @intCast(v.inter);
    var w = model.Weights.init(testing.allocator);
    errdefer w.deinit();
    var buf: [96]u8 = undefined;
    var seed: u64 = 1;
    const feat: c_int = @intCast(3 * v.patch * v.patch);
    try common.putRandomBf16(w.allocator, &w.map, "vision.patch_embed.proj.weight", &.{ d, feat }, 1, 0.1, 0);
    try common.putRandomBf16(w.allocator, &w.map, "vision.patch_embed.proj.bias", &.{d}, 2, 0.05, 0);
    for (0..v.layers) |i| {
        const Part = struct { name: []const u8, shape: [2]c_int, rank: u2 };
        const parts = [_]Part{
            .{ .name = "norm1.weight", .shape = .{ d, 0 }, .rank = 1 },
            .{ .name = "norm2.weight", .shape = .{ d, 0 }, .rank = 1 },
            .{ .name = "attn.wqkv.weight", .shape = .{ 3 * d, d }, .rank = 2 },
            .{ .name = "attn.wqkv.bias", .shape = .{ 3 * d, 0 }, .rank = 1 },
            .{ .name = "attn.wo.weight", .shape = .{ d, d }, .rank = 2 },
            .{ .name = "attn.wo.bias", .shape = .{ d, 0 }, .rank = 1 },
            .{ .name = "mlp.w1.weight", .shape = .{ 2 * inter, d }, .rank = 2 },
            .{ .name = "mlp.w2.weight", .shape = .{ d, inter }, .rank = 2 },
        };
        for (parts) |p| {
            seed += 1;
            const norm = std.mem.startsWith(u8, p.name, "norm");
            try common.putRandomBf16(w.allocator, &w.map, try std.fmt.bufPrint(&buf, "vision.blocks.{d}.{s}", .{ i, p.name }), p.shape[0..p.rank], seed, 0.1, if (norm) 1 else 0);
        }
    }
    try common.putRandomBf16(w.allocator, &w.map, "vision.norm.weight", &.{d}, 80, 0.1, 1);
    const out: c_int = @intCast(dim);
    try common.putRandomBf16(w.allocator, &w.map, "aligner.w1.weight", &.{ out, d * @as(c_int, @intCast(v.downsample * v.downsample)) }, 81, 0.1, 0);
    try common.putRandomBf16(w.allocator, &w.map, "aligner.w1.bias", &.{out}, 82, 0.05, 0);
    try common.putRandomBf16(w.allocator, &w.map, "aligner.w2.weight", &.{ out, out }, 83, 0.1, 0);
    try common.putRandomBf16(w.allocator, &w.map, "aligner.w2.bias", &.{out}, 84, 0.05, 0);
    for ([_][]const u8{ "image_start", "image_end", "image_newline" }) |name| {
        try common.putRandomBf16(w.allocator, &w.map, name, &.{out}, 70, 0.1, 0);
    }
    var it = w.map.valueIterator();
    while (it.next()) |a| {
        const shape = mlx.getShape(a.*);
        const ours = mlx.engine.mlx_array_new_data(mlx.mlx_array_data_uint8(a.*).?, shape.ptr, @intCast(shape.len), mlx.mlx_array_dtype(a.*));
        if (ours.ctx == null) return error.OutOfMemory;
        _ = mlx.mlx_array_free(a.*);
        a.* = ours;
    }
    return w;
}

fn encodePoolRef(a: std.mem.Allocator, t: *dsv41_vision.Dsv41Vision, pixels: []const f32, gh: u32, gw: u32) ![]u8 {
    const s = mlx.engine.stream();
    const scope = mlx.engine.enter(s);
    defer scope.end();
    const feat: usize = 3 * @as(usize, t.v.patch) * t.v.patch;
    const patches = try mlx.newData(pixels.ptr, &[_]c_int{ @intCast(@as(usize, gh) * gw), @intCast(feat) }, .float32);
    defer _ = mlx.mlx_array_free(patches);
    const span = try t.encode(patches, gh, gw);
    defer _ = mlx.mlx_array_free(span);
    try mlx.check(mlx.mlx_array_eval(span));
    const n_elems: usize = dsv41_vision.spanRows(gh, gw, t.v.downsample) * t.out_dim;
    const ptr = mlx.mlx_array_data_uint8(span) orelse return error.InvalidTestOutput;
    return try a.dupe(u8, ptr[0 .. n_elems * 2]);
}

test "dsv41 encode plan: wide-region encode matches pool bit-for-bit, asks closed pool nothing, and checks canaries" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const was_enabled = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer mlx.engine.enabled = was_enabled;

    const vis: cfg_mod.Vision = .{
        .layers = 2,
        .hidden = 128,
        .heads = 2,
        .inter = 128,
        .patch = 4,
        .rope_theta = 10000,
        .downsample = 3,
        .max_tokens = 256,
        .min_pixels = 256,
        .max_wh_ratio = 0,
    };
    var cfg: model.ModelConfig = .{ .model_type = "deepseek_v41", .cooler_for_test = true };
    cfg.dsv41 = .{ .dim = test_dim, .vision = vis };

    var w = try makeWeights(vis, test_dim);
    defer w.deinit();
    var tower = try dsv41_vision.Dsv41Vision.init(a, cfg, &w);
    defer tower.deinit();
    defer mlx.engine.seal(false);
    defer mlx.engine.budget(1 << 62) catch {};

    const max_patches: u64 = 640;
    const encode_need = try dsv41_encode_plan.need(&tower, a, max_patches, plan.region.default_alignment, plan.region.test_guard);
    try testing.expect(encode_need.scratch > 0);

    var dummy_plans: dsv41_plan.Plans = .{
        .gpa = a,
        .reach = 1,
        .encode = encode_need,
    };
    defer dummy_plans.deinit();

    const grids = [_][2]u32{
        .{ 6, 6 }, // unpadded single chunk (36 patches)
        .{ 4, 5 }, // padded aligner branch (20 patches)
        .{ 32, 18 }, // > QUERY_CHUNK multi-chunk with short 64-row tail (576 patches)
    };

    var pixels: [3][]f32 = undefined;
    var refs: [3][]u8 = undefined;
    var rng = std.Random.DefaultPrng.init(0xd541e);
    const feat: usize = 3 * @as(usize, vis.patch) * vis.patch;
    var made_px: usize = 0;
    defer for (pixels[0..made_px]) |px| a.free(px);
    var made_refs: usize = 0;
    defer for (refs[0..made_refs]) |rf| a.free(rf);

    for (grids, 0..) |g, idx| {
        const px = try a.alloc(f32, @as(usize, g[0]) * g[1] * feat);
        pixels[idx] = px;
        made_px = idx + 1;
        for (px) |*x| x.* = rng.random().floatNorm(f32);
        refs[idx] = try encodePoolRef(a, &tower, px, g[0], g[1]);
        made_refs = idx + 1;
    }

    try core.arenaFor(.{ .g = encode_need.scratch, .h = encode_need.host, .alignment = plan.region.default_alignment, .guard = plan.region.test_guard });
    dummy_plans.has_arena = true;

    const pool_was = core.poolGate();
    try core.closePool(.refuse);
    defer core.closePool(pool_was) catch {};
    core.arena_check.byte = 0xFF;
    defer core.arena_check.byte = null;
    const hits = core.arena_check.hits;

    // Stacked multi-image Media buffer written at advancing offsets.
    var total_rows: usize = 0;
    for (grids) |g| total_rows += dsv41_vision.spanRows(g[0], g[1], vis.downsample);
    var media = blk: {
        const scope = mlx.engine.enter(mlx.engine.stream());
        defer scope.end();
        break :blk try tower.makeMedia(total_rows);
    };
    defer _ = mlx.mlx_array_free(media);

    var row0: usize = 0;
    const before = core.stats();
    for (grids, pixels) |g, px| {
        try dsv41_encode_plan.run(&dummy_plans, &tower, &media, row0, std.mem.sliceAsBytes(px), g[0], g[1]);
        row0 += dsv41_vision.spanRows(g[0], g[1], vis.downsample);
    }
    try mlx.check(mlx.mlx_array_eval(media));
    const after = core.stats();
    try testing.expectEqual(@as(u64, grids.len), after.wide_executions - before.wide_executions);
    try testing.expectEqual(before.asks, after.asks);
    try testing.expectEqual(before.closed_asks, after.closed_asks);
    try testing.expectEqual(hits, core.arena_check.hits);

    const media_bytes = mlx.mlx_array_data_uint8(media).?[0 .. total_rows * test_dim * 2];
    var off: usize = 0;
    for (refs) |want| {
        try testing.expectEqualSlices(u8, want, media_bytes[off..][0..want.len]);
        off += want.len;
    }
}

test "dsv41 encode plan: ask past measured wide region is refused with WideFull" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const was_enabled = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer mlx.engine.enabled = was_enabled;

    const vis: cfg_mod.Vision = .{
        .layers = 2,
        .hidden = 128,
        .heads = 2,
        .inter = 128,
        .patch = 4,
        .rope_theta = 10000,
        .downsample = 2,
        .max_tokens = 256,
        .min_pixels = 256,
        .max_wh_ratio = 0,
    };
    var cfg: model.ModelConfig = .{ .model_type = "deepseek_v41", .cooler_for_test = true };
    cfg.dsv41 = .{ .dim = test_dim, .vision = vis };

    var w = try makeWeights(vis, test_dim);
    defer w.deinit();
    var tower = try dsv41_vision.Dsv41Vision.init(a, cfg, &w);
    defer tower.deinit();
    defer mlx.engine.seal(false);
    defer mlx.engine.budget(1 << 62) catch {};

    const g = [2]u32{ 16, 16 };
    const feat: usize = 3 * @as(usize, vis.patch) * vis.patch;
    const vals = try a.alloc(f32, @as(usize, g[0]) * g[1] * feat);
    defer a.free(vals);
    @memset(vals, 0.25);

    var media = blk: {
        const scope = mlx.engine.enter(mlx.engine.stream());
        defer scope.end();
        break :blk try tower.makeMedia(dsv41_vision.spanRows(g[0], g[1], vis.downsample));
    };
    defer _ = mlx.mlx_array_free(media);

    try core.arenaFor(.{ .g = 64 << 10, .h = 64 << 10, .alignment = plan.region.default_alignment });
    defer core.arenaRelease() catch {};

    const dummy_plans: dsv41_plan.Plans = .{
        .gpa = a,
        .reach = 1,
    };
    try testing.expectError(error.WideFull, dsv41_encode_plan.run(&dummy_plans, &tower, &media, 0, std.mem.sliceAsBytes(vals), g[0], g[1]));
    var said: [512]u8 = undefined;
    try testing.expect(std.mem.indexOf(u8, mlx.takeError(&said) orelse "", "WideFull:") != null);
    try testing.expectEqual(@as(u64, 64), core.refused_wide.?.room >> 10);
}

test "dsv41 media bill: Cooler returns Media + host input bytes while --ds41-mlx-engine keeps null and skips past-plan refusal" {
    const a = testing.allocator;
    var cfg = try model.parseConfigFromJson(a, cfg_mod.release_config);
    defer cfg.deinit(a);

    const rows: usize = 400;
    const input_bytes: u64 = 128 << 10;

    cfg.cooler_for_test = false;
    try testing.expectEqual(@as(?u64, null), dsv41_media.mediaBill(&cfg, 1000, input_bytes, rows));
    try testing.expectEqual(@as(?u64, null), arch_media.mediaBill(&cfg, 1000, input_bytes, rows));

    cfg.cooler_for_test = true;
    const expected = dsv41_encode_plan.mediaBytes(cfg.dsv41.?.dim, rows) + input_bytes;
    try testing.expectEqual(@as(?u64, expected), dsv41_media.mediaBill(&cfg, 1000, input_bytes, rows));
    try testing.expectEqual(@as(?u64, expected), arch_media.mediaBill(&cfg, 1000, input_bytes, rows));

    const most = arch_media.maxImagePatches(dsv41_media.preproc(&cfg));
    const huge_w: u32 = @intCast(most + 10);
    const images = [_]chat.ImageData{
        .{ .pixels = &.{}, .width = huge_w * 14, .height = 14, .grid_h = 1, .grid_w = huge_w },
    };
    cfg.cooler_for_test = false;
    try testing.expect(request_media.pastPlanRefusal(&cfg, &images) == null);
    cfg.cooler_for_test = true;
    try testing.expect(request_media.pastPlanRefusal(&cfg, &images) != null);
}
