//! Hy4-Flash's late-kernel cells on a Sushi Cooler load: what a served request can still make after the load's warm
//! (`Model.warmKernels`, then `cooler_warm.warmServed`). Each runs alone (`alone_tests`): the pipeline cache is the
//! process's.
const std = @import("std");
const testing = std.testing;
const mlx = @import("../mlx.zig");
const Ops = @import("../mlx_ops.zig").Ops;
const model_mod = @import("../model.zig");
const static_bill = @import("../static_bill.zig");
const contract = @import("../cooler_contract_test.zig");
const hy4_contract = @import("hy4_contract.zig");
const hy4_model = @import("hy4_model.zig");
const fwd = @import("hy4_forward.zig");

const ResidentHy4 = struct {
    pub fn prepare(a: std.mem.Allocator, config: *model_mod.ModelConfig, dir: []const u8) !void {
        try hy4_contract.prepare(a, config, dir);
        config.expert_resident = true;
        config.expert_cache_bytes = std.math.maxInt(u64) / 2;
    }
};

fn endSlot(sv: *contract.Served, slot: *contract.Served.scheduler.Slot) void {
    sv.sch.complete(slot);
    contract.Served.scheduler.drainSlots(&sv.sch, sv.sch.cleanup_queue.items);
    sv.sch.cleanup_queue.clearRetainingCapacity();
}

fn submitPending(sv: *contract.Served, prompt: []const u32, n: u32, mtp: bool) !*contract.Served.scheduler.Slot {
    const head = if (mtp) @import("../generate.zig").MtpHeadRef.fromModel(@import("../arch/handle.zig").trunk(sv.entry.handle.?), true, null) else null;
    const slot = try sv.sch.submit(.{
        .model = sv.entry,
        .prompt_ids = prompt,
        .sampling = .{ .temperature = 0.0 },
        .eos_token_ids = &.{},
        .max_tokens = n,
        .enable_mtp = mtp,
        .mtp = head,
    });
    _ = sv.sch.pending.orderedRemove(0);
    return slot;
}

test "hy4 late: after a Cooler load's served warm, a chunk's short last span and an image request's encode and prompt prefill compile no kernel (HY4_BF16_TINY)" {
    const a = testing.allocator;
    const dir = std.mem.span(std.c.getenv("HY4_BF16_TINY") orelse return error.SkipZigTest);
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

    const m = cooler.xfm.arch_state.hy4flash.model orelse return error.Hy4ModelMissing;
    const late = mlx.engine.latePipelines();
    mlx.engine.late_names.clear();

    var sparse_l: ?usize = null;
    var sliding_l: ?usize = null;
    for (m.layers[0..m.config.layers], 0..) |*layer, l| {
        if (layer.indexer != null and sparse_l == null) sparse_l = l;
        if (layer.indexer == null and sliding_l == null) sliding_l = l;
    }
    const tail_layers = [_]usize{ sparse_l orelse 0, sliding_l orelse 0 };
    const tail_widths = [_]c_int{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 31, 32, 64, 127, 128, 255 };
    const spans = [_]c_int{ hy4_model.gathered_span, m.span_rows };
    var first_tail: ?struct { layer: usize, span: c_int, tail: c_int } = null;
    {
        const turn = hy4_contract.openTurn(cooler.h, &cooler.config);
        defer turn.end();
        for (tail_layers, 0..) |l, li| {
            if (li > 0 and l == tail_layers[0]) continue;
            const layer = &m.layers[l];
            for ([_]struct { fwd.Lin, mlx.mlx_dtype }{ .{ layer.attn.gate, .float32 }, .{ layer.attn.o, .bfloat16 } }) |pair| for (spans) |span| {
                const lin = pair[0];
                const in: c_int = if (lin.stq) |b| b.inDim() else mlx.getShape(lin.w)[1];
                for (tail_widths) |tail| {
                    if (tail >= span) continue;
                    const exact = mlx.engine.exactF32();
                    defer exact.end();
                    var o: Ops = .{ .s = m.s };
                    defer o.deinit();
                    const x = try o.slot();
                    try mlx.check(mlx.mlx_zeros(x, &[_]c_int{ span + tail, in }, 2, pair[1], m.s));
                    const y = try o.own(try o.result(try fwd.linearSpans(&o, x.*, lin, span)));
                    try mlx.check(mlx.mlx_array_eval(y));
                    if (first_tail == null and mlx.engine.latePipelines() != late) first_tail = .{ .layer = l, .span = span, .tail = tail };
                }
            };
        }
    }
    errdefer if (first_tail) |ft| std.debug.print("hy4 late: from layer {d}, span {d}, tail {d} on, a chunk's last span compiled:\n{s}", .{ ft.layer, ft.span, ft.tail, mlx.engine.late_names.text() });

    const v = cooler.config.hy4flash.?.vision.?;
    const merge = v.merge;
    const grids = [_][2]u32{ .{ 1, 1 }, .{ 4, 4 }, .{ 16, 16 }, .{ 23, 23 }, .{ 33, 33 } };
    var first_grid: ?[2]u32 = null;
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
        var got = try contract.serialWith(&cooler, ids, 1, .{ .temperature = 0 }, .{ .ctx = ctx });
        got.deinit(a);
        if (first_grid == null and mlx.engine.latePipelines() != late) first_grid = g;
    }
    errdefer if (first_grid) |fg| std.debug.print("hy4 late: from the grid of {d} x {d} merge cells on, an image request compiled:\n{s}", .{ fg[0], fg[1], mlx.engine.late_names.text() });
    try testing.expectEqual(late, mlx.engine.latePipelines());
}

test "hy4 late: after a Cooler load's served warm, a chunk's short last span, an image splice, a 2..8-row verify and joined ticks compile no kernel (HY4_STQ_TINY)" {
    const a = testing.allocator;
    const dir = std.mem.span(std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest);
    const exl3 = @import("sushi_exl3");
    var cooler: contract.Load = undefined;
    try cooler.open(ResidentHy4, dir, true);
    defer _ = cooler.close();
    defer static_bill.close(&cooler.config);
    const m = cooler.xfm.arch_state.hy4flash.model orelse return error.Hy4ModelMissing;
    const late = mlx.engine.latePipelines();
    mlx.engine.late_names.clear();

    // 1. Short last span tails on layer 0 (`1..rowsMax` + `{rowsMax + 1, span - 1}`).
    var first_tail: ?struct { span: c_int, tail: c_int } = null;
    {
        const turn = hy4_contract.openTurn(cooler.h, &cooler.config);
        defer turn.end();
        const layer = &m.layers[0];
        const spans = [_]c_int{ hy4_model.gathered_span, m.span_rows };
        var rows_max: c_int = 1;
        while (!exl3.trunk.gemmRows(@intCast(rows_max), true)) : (rows_max += 1) {}
        rows_max -= 1;
        for ([_]struct { fwd.Lin, mlx.mlx_dtype }{ .{ layer.attn.gate, .float32 }, .{ layer.attn.o, .bfloat16 } }) |pair| for (spans) |span| {
            const lin = pair[0];
            const in: c_int = if (lin.stq) |b| b.inDim() else mlx.getShape(lin.w)[1];
            var tail: c_int = 1;
            while (tail < span) {
                const exact = mlx.engine.exactF32();
                defer exact.end();
                var o: Ops = .{ .s = m.s };
                defer o.deinit();
                const x = try o.slot();
                try mlx.check(mlx.mlx_zeros(x, &[_]c_int{ span + tail, in }, 2, pair[1], m.s));
                const y = try o.own(try o.result(try fwd.linearSpans(&o, x.*, lin, span)));
                try mlx.check(mlx.mlx_array_eval(y));
                if (first_tail == null and mlx.engine.latePipelines() != late) first_tail = .{ .span = span, .tail = tail };
                if (tail < rows_max) {
                    tail += 1;
                } else if (tail == rows_max) {
                    tail = rows_max + 1;
                } else if (tail < span - 1) {
                    tail = span - 1;
                } else {
                    break;
                }
            }
        };
    }
    errdefer if (first_tail) |ft| std.debug.print("hy4 late: from span {d}, tail {d} on, a chunk's last span compiled:\n{s}", .{ ft.span, ft.tail, mlx.engine.late_names.text() });

    // 2. Spliced image prompts across 3 boundary grids.
    const dim: usize = cooler.config.hidden_size;
    var first_grid: ?[2]u32 = null;
    var rng = std.Random.DefaultPrng.init(0x51de);
    for ([_][2]u32{ .{ 1, 1 }, .{ 4, 4 }, .{ 9, 9 } }) |g| {
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
        var got = try contract.serialWith(&cooler, ids, 1, .{ .temperature = 0 }, .{ .ctx = ctx });
        got.deinit(a);
        if (first_grid == null and mlx.engine.latePipelines() != late) first_grid = g;
    }
    errdefer if (first_grid) |fg| std.debug.print("hy4 late: from the grid of {d} x {d} merge cells on, an image prompt compiled:\n{s}", .{ fg[0], fg[1], mlx.engine.late_names.text() });

    // 3. 2..8-row verify blocks after prefill.
    var first_verify: ?struct { pre: usize, block: usize } = null;
    {
        const turn = hy4_contract.openTurn(cooler.h, &cooler.config);
        defer turn.end();
        defer m.reset();
        var ids: [64]u32 = undefined;
        var vrng = std.Random.DefaultPrng.init(0xbeef);
        for (&ids) |*t| t.* = vrng.random().uintLessThan(u32, @intCast(m.config.vocab - 1)) + 1;
        for ([_]usize{ 5, 8, 20 }) |pre| {
            for (2..9) |block| {
                m.reset();
                const pre_lg = try m.forwardReadout(ids[0..pre], .none);
                _ = mlx.mlx_array_free(pre_lg);
                const blg = try m.forwardReadout(ids[pre .. pre + block], .all);
                defer _ = mlx.mlx_array_free(blg);
                try mlx.check(mlx.mlx_array_eval(blg));
                if (first_verify == null and mlx.engine.latePipelines() != late) first_verify = .{ .pre = pre, .block = block };
            }
        }
    }
    errdefer if (first_verify) |fv| std.debug.print("hy4 late: from pre={d}, block={d} on, a verify compiled:\n{s}", .{ fv.pre, fv.block, mlx.engine.late_names.text() });

    // 4. Joined ticks narrowing across every joined width (moved from hy4_request_test).
    const rows = @import("hy4_plan.zig").joinedRows(&cooler.config);
    if (rows >= 2) {
        const prompt_ids = try hy4_contract.prompt(a, dir);
        defer a.free(prompt_ids);
        var sv: contract.Served = undefined;
        try sv.init(&cooler, dir);
        defer sv.deinit();
        const max_rows = exl3.project_rows.NAX_ROWS_MAX;
        sv.sch.queue_cap = @intCast(rows);
        const mtp = @import("../arch/handle.zig").trunk(sv.entry.handle.?).arch_state.hy4flash.drafter != null;
        const grouped = fwd.grouped_chains_for_test;
        var slots: [max_rows]*contract.Served.scheduler.Slot = undefined;
        var made: usize = 0;
        defer for (0..made) |i| endSlot(&sv, slots[i]);
        for (0..rows) |i| {
            slots[i] = try submitPending(&sv, prompt_ids[i % 4 .. @min(prompt_ids.len, 12 + i % 4)], @intCast(2 + i), mtp);
            made = i + 1;
            try contract.Served.scheduler.runPrefill(&sv.sch, slots[i]);
            try sv.sch.decoding.append(a, slots[i]);
        }
        var ticks: usize = 0;
        while (ticks < 4 * max_rows) : (ticks += 1) {
            var act: [max_rows]*contract.Served.scheduler.Slot = undefined;
            var k: usize = 0;
            for (slots[0..made]) |s| if (s.joinsTick()) {
                act[k] = s;
                k += 1;
            };
            if (k == 0) break;
            try contract.Served.scheduler.runDecodeTick(&sv.sch, act[0..k]);
            contract.Served.scheduler.cullDecoding(&sv.sch);
        }
        for (slots[0..made]) |s| try testing.expect(s.error_code == null);
        try testing.expect(fwd.grouped_chains_for_test > grouped);
    }

    try testing.expectEqual(late, mlx.engine.latePipelines());
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
    const layer = &m.layers[0];
    for ([_]struct { fwd.Lin, mlx.mlx_dtype }{ .{ layer.attn.gate, .float32 }, .{ layer.attn.o, .bfloat16 } }) |pair| {
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
            if (first == null and mlx.engine.pipelinesMade().count != made) first = .{ .layer = 0, .tail = tail };
        }
    }
    if (first) |f| std.debug.print("hy4 late: from layer {d}, tail {d} on, a span tail made a pipeline the kernel warm did not\n", .{ f.layer, f.tail });
    try testing.expectEqual(made, mlx.engine.pipelinesMade().count);
}
