//! Hy4-Flash's late-kernel cells on a Sushi Cooler load: what a served request can still make after the load's warm
//! (`Model.warmKernels`, then `cooler_warm.warmServed`). Each runs alone (`alone_tests`): the pipeline cache is the
//! process's.
const std = @import("std");
const testing = std.testing;
const mlx = @import("../mlx.zig");
const Ops = @import("../mlx_ops.zig").Ops;
const contract = @import("../cooler_contract_test.zig");
const hy4_contract = @import("hy4_contract.zig");
const hy4_model = @import("hy4_model.zig");
const fwd = @import("hy4_forward.zig");

/// Every width the short last span of a chunk takes through the gate and W_o of every layer, at each span a layer
/// splits a chunk by (`linearSpans`): a launch of its own, whatever its rows.
fn spanTails(env: [:0]const u8) !void {
    const dir = std.mem.span(std.c.getenv(env) orelse return error.SkipZigTest);
    var cooler: contract.Load = undefined;
    try cooler.open(hy4_contract, dir, true);
    defer _ = cooler.close();
    const m = cooler.xfm.arch_state.hy4flash.model orelse return error.Hy4ModelMissing;
    const turn = hy4_contract.openTurn(cooler.h, &cooler.config);
    defer turn.end();
    const late = mlx.engine.latePipelines();
    mlx.engine.late_names.clear();
    var first: ?struct { layer: usize, span: c_int, tail: c_int } = null;
    const spans = [_]c_int{ hy4_model.gathered_span, m.span_rows };
    // The gate reads the attention norm's rows in f32; W_o the gated rows, narrow (`linearInput`).
    for (m.layers[0..m.config.layers], 0..) |*layer, l| for ([_]struct { fwd.Lin, mlx.mlx_dtype }{ .{ layer.attn.gate, .float32 }, .{ layer.attn.o, .bfloat16 } }) |pair| for (spans) |span| {
        const lin = pair[0];
        const in: c_int = if (lin.stq) |b| b.inDim() else mlx.getShape(lin.w)[1];
        var tail: c_int = 1;
        while (tail < span) : (tail += 1) {
            const exact = mlx.engine.exactF32();
            defer exact.end();
            var o: Ops = .{ .s = m.s };
            defer o.deinit();
            const x = try o.slot();
            try mlx.check(mlx.mlx_zeros(x, &[_]c_int{ span + tail, in }, 2, pair[1], m.s));
            const y = try o.own(try o.result(try fwd.linearSpans(&o, x.*, lin, span)));
            try mlx.check(mlx.mlx_array_eval(y));
            if (first == null and mlx.engine.latePipelines() != late) first = .{ .layer = l, .span = span, .tail = tail };
        }
    };
    errdefer std.debug.print("hy4 late: from layer {d}, span {d}, tail {d} on, a chunk's last span compiled:\n{s}", .{ first.?.layer, first.?.span, first.?.tail, mlx.engine.late_names.text() });
    try testing.expectEqual(late, mlx.engine.latePipelines());
}

test "hy4 late: after a Cooler load's served warm, a chunk's short last span of any width through the gate and W_o of any layer compiles no kernel (HY4_BF16_TINY)" {
    try spanTails("HY4_BF16_TINY");
}

test "hy4 late: after a Cooler load's served warm, a chunk's short last span of any width through the gate and W_o of any layer compiles no kernel (HY4_STQ_TINY)" {
    try spanTails("HY4_STQ_TINY");
}

// The load's own kernel warm (`Model.warmKernels`) alone, with no served request after it: a chunk's short last span
// launches the gate and W_o of a layer at its own width, and each width a rows arm serves is a pipeline of its own.
test "hy4 late: the kernel warm alone makes every pipeline a chunk's short last span takes through the gate and W_o, at every width a rows arm serves (HY4_STQ_TINY)" {
    const dir = std.mem.span(std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest);
    var cooler: contract.Load = undefined;
    try cooler.openLight(hy4_contract, dir, true);
    defer _ = cooler.close();
    const m = cooler.xfm.arch_state.hy4flash.model orelse return error.Hy4ModelMissing;
    const turn = hy4_contract.openTurn(cooler.h, &cooler.config);
    defer turn.end();
    try m.warmKernels();
    const made = mlx.engine.pipelinesMade().count;
    var first: ?struct { layer: usize, tail: c_int } = null;
    for (m.layers[0..m.config.layers], 0..) |*layer, l| for ([_]struct { fwd.Lin, mlx.mlx_dtype }{ .{ layer.attn.gate, .float32 }, .{ layer.attn.o, .bfloat16 } }) |pair| {
        const bank = pair[0].stq orelse continue;
        var tail: c_int = 1;
        while (!@import("sushi_exl3").trunk.gemmRows(@intCast(tail), true)) : (tail += 1) {
            const exact = mlx.engine.exactF32();
            defer exact.end();
            var o: Ops = .{ .s = m.s };
            defer o.deinit();
            const x = try o.slot();
            try mlx.check(mlx.mlx_zeros(x, &[_]c_int{ tail, bank.inDim() }, 2, pair[1], m.s));
            try mlx.check(mlx.mlx_array_eval(try fwd.linearStored(&o, x.*, pair[0])));
            if (first == null and mlx.engine.pipelinesMade().count != made) first = .{ .layer = l, .tail = tail };
        }
    };
    if (first) |f| std.debug.print("hy4 late: from layer {d}, tail {d} on, a span tail made a pipeline the kernel warm did not\n", .{ f.layer, f.tail });
    try testing.expectEqual(made, mlx.engine.pipelinesMade().count);
}

/// A served image request as the scheduler runs it: the tower encodes the image, then the prompt prefills with the rows
/// spliced at its placeholders and decodes. The grids are the patch counts an image of any size lands on, each a whole
/// number of merge cells: a few patches, a row's worth, whole and not whole 512-row attention chunks, and wide.
fn imageRequests(env: [:0]const u8) !void {
    const a = testing.allocator;
    const dir = std.mem.span(std.c.getenv(env) orelse return error.SkipZigTest);
    var cooler: contract.Load = undefined;
    try cooler.prepare(hy4_contract, dir, true);
    cooler.vision = true;
    try cooler.load(dir);
    var tower = @import("../vision.zig").VisionEncoder.init(a, cooler.config, &cooler.weights) catch |e| {
        cooler.discard();
        return e;
    };
    defer _ = cooler.close();
    defer tower.deinit();
    try cooler.serve();
    const late = mlx.engine.latePipelines();
    mlx.engine.late_names.clear();
    const v = cooler.config.hy4flash.?.vision.?;
    const merge = v.merge;
    const grids = [_][2]u32{ .{ 1, 1 }, .{ 1, 2 }, .{ 2, 2 }, .{ 2, 3 }, .{ 3, 3 }, .{ 4, 4 }, .{ 2, 8 }, .{ 5, 5 }, .{ 8, 8 }, .{ 11, 11 }, .{ 16, 16 }, .{ 6, 40 }, .{ 23, 23 }, .{ 32, 32 }, .{ 33, 33 }, .{ 40, 25 } };
    var first: ?[2]u32 = null;
    for (grids) |g| {
        const gh = g[0] * merge;
        const gw = g[1] * merge;
        const n: usize = @as(usize, gh) * gw;
        const feat: usize = 3 * v.patch * v.patch;
        const rows_n = (gh / merge) * (gw / merge + 1);
        const rows = blk: {
            const patches_host = try a.alloc(f32, n * feat);
            defer a.free(patches_host);
            @memset(patches_host, 0);
            const patches = pat: {
                const host = mlx.engine.enter(tower.s);
                defer host.end();
                break :pat try mlx.newData(patches_host.ptr, &[_]c_int{ @intCast(n), @intCast(feat) }, .float32);
            };
            defer _ = mlx.mlx_array_free(patches);
            break :blk try tower.forwardPatches(patches, gh, gw);
        };
        defer _ = mlx.mlx_array_free(rows);
        const ids = try a.alloc(u32, rows_n + 6);
        defer a.free(ids);
        for (ids, 0..) |*t, i| t.* = if (i >= 3 and i < 3 + rows_n) cooler.config.image_token_id else @intCast(1000 + 7 * i);
        var ctx = cooler.xfm.defaultCtx();
        ctx.vision_embeddings = rows;
        var got = try contract.serialWith(&cooler, ids, 6, .{ .temperature = 0 }, .{ .ctx = ctx });
        got.deinit(a);
        if (first == null and mlx.engine.latePipelines() != late) first = g;
    }
    errdefer std.debug.print("hy4 late: from the grid of {d} x {d} merge cells on, an image request compiled:\n{s}", .{ first.?[0], first.?[1], mlx.engine.late_names.text() });
    try testing.expectEqual(late, mlx.engine.latePipelines());
}

test "hy4 late: after a Cooler load's served warm, an image request's encode and its prompt's prefill compile no kernel at any grid (HY4_BF16_TINY)" {
    try imageRequests("HY4_BF16_TINY");
}

/// An image prompt's prefill at row counts a release grid lands on (up to the widest the warm's text widths cover), its
/// rows (the tower's encode) random: the splice at the image ids runs in the trunk's forward, and on a pack with no
/// tower file too.
fn splicedPrompts(env: [:0]const u8) !void {
    const a = testing.allocator;
    const dir = std.mem.span(std.c.getenv(env) orelse return error.SkipZigTest);
    var cooler: contract.Load = undefined;
    try cooler.open(hy4_contract, dir, true);
    defer _ = cooler.close();
    const dim: usize = cooler.config.hidden_size;
    const late = mlx.engine.latePipelines();
    mlx.engine.late_names.clear();
    var first: ?[2]u32 = null;
    var rng = std.Random.DefaultPrng.init(0x51de);
    for ([_][2]u32{ .{ 1, 1 }, .{ 2, 2 }, .{ 4, 4 }, .{ 2, 8 }, .{ 8, 8 }, .{ 9, 9 }, .{ 10, 10 } }) |g| {
        const rows_n: usize = g[0] * (g[1] + 1);
        const vals = try a.alloc(f32, rows_n * dim);
        defer a.free(vals);
        for (vals) |*v| v.* = rng.random().floatNorm(f32);
        const rows = blk: {
            const host = mlx.engine.enter(cooler.xfm.s);
            defer host.end();
            const r = try mlx.newData(vals.ptr, &[_]c_int{ @intCast(rows_n), @intCast(dim) }, .float32);
            try mlx.check(mlx.mlx_array_eval(r));
            break :blk r;
        };
        defer _ = mlx.mlx_array_free(rows);
        const ids = try a.alloc(u32, rows_n + 6);
        defer a.free(ids);
        for (ids, 0..) |*t, i| t.* = if (i >= 3 and i < 3 + rows_n) cooler.config.image_token_id else @intCast(1000 + 7 * i);
        var ctx = cooler.xfm.defaultCtx();
        ctx.vision_embeddings = rows;
        var got = try contract.serialWith(&cooler, ids, 6, .{ .temperature = 0 }, .{ .ctx = ctx });
        got.deinit(a);
        if (first == null and mlx.engine.latePipelines() != late) first = g;
    }
    errdefer std.debug.print("hy4 late: from the grid of {d} x {d} merge cells on, an image prompt compiled:\n{s}", .{ first.?[0], first.?[1], mlx.engine.late_names.text() });
    try testing.expectEqual(late, mlx.engine.latePipelines());
}

test "hy4 late: after a Cooler load's served warm, an image prompt's splice at its image ids compiles no kernel at the release's row counts (HY4_STQ_TINY)" {
    try splicedPrompts("HY4_STQ_TINY");
}

fn verifyAfterWarm(env: [:0]const u8) !void {
    const dir = std.mem.span(std.c.getenv(env) orelse return error.SkipZigTest);
    var cooler: contract.Load = undefined;
    try cooler.open(hy4_contract, dir, true);
    defer _ = cooler.close();
    const m = cooler.xfm.arch_state.hy4flash.model orelse return error.Hy4ModelMissing;
    const turn = hy4_contract.openTurn(cooler.h, &cooler.config);
    defer turn.end();
    const late = mlx.engine.latePipelines();
    mlx.engine.late_names.clear();

    var ids: [64]u32 = undefined;
    var rng = std.Random.DefaultPrng.init(0xbeef);
    for (&ids) |*t| t.* = rng.random().uintLessThan(u32, @intCast(m.config.vocab - 1)) + 1;

    var first: ?struct { pre: usize, block: usize } = null;
    for ([_]usize{ 5, 7, 8, 15, 20 }) |pre| {
        for (2..9) |block| {
            m.reset();
            const pre_lg = try m.forwardReadout(ids[0..pre], .none);
            _ = mlx.mlx_array_free(pre_lg);
            const blg = try m.forwardReadout(ids[pre .. pre + block], .all);
            defer _ = mlx.mlx_array_free(blg);
            try mlx.check(mlx.mlx_array_eval(blg));
            if (first == null and mlx.engine.latePipelines() != late) first = .{ .pre = pre, .block = block };
        }
    }
    errdefer std.debug.print("hy4 late: from pre={d}, block={d} on, a verify compiled:\n{s}", .{ first.?.pre, first.?.block, mlx.engine.late_names.text() });
    try testing.expectEqual(late, mlx.engine.latePipelines());
}

test "hy4 late: after a Cooler load's served warm, a 2..8-row verify compiles no kernel (HY4_STQ_TINY)" {
    try verifyAfterWarm("HY4_STQ_TINY");
}
