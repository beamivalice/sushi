//! Hy4-Flash past its load's seal (docs/bill_redesign.md, stage 4c): the pool closes for the model's turns (D64), so a
//! served request's work outside the wide phases still on the pool asks it nothing, and gives the tokens it gives with
//! the plans off.
const std = @import("std");
const testing = std.testing;
const mlx = @import("../mlx.zig");
const core = @import("../tensor/core.zig");
const contract = @import("../cooler_contract_test.zig");
const hy4_contract = @import("hy4_contract.zig");
const hy4_held = @import("hy4_held.zig");
const sampler = @import("../sampler.zig");
const Generator = @import("../generate.zig").Generator;
const Tokenizer = @import("../tokenizer.zig").Tokenizer;
const arch_handle = @import("../arch/handle.zig");

const io = std.Io.Threaded.global_single_threaded.io();

/// What a served request said, and the asks of the closed pool it made outside a wide phase.
const Said = struct {
    tokens: std.ArrayList(u32) = .empty,
    closed: u64 = 0,

    fn deinit(s: *Said) void {
        s.tokens.deinit(testing.allocator);
    }
};

/// `prompt`'s `n` tokens through the generator under `sampling` and `opts`, on the load's plans under the turn's closed
/// pool or, `planned` false, on the pool in a wide phase.
fn generated(ld: *contract.Load, prompt: []const u32, n: u32, sampling: sampler.SamplingParams, opts: Generator.InitOptions, planned: bool) !Said {
    const a = testing.allocator;
    const m = ld.xfm.arch_state.hy4flash.model.?;
    const saved = m.plans;
    if (!planned) m.plans = null;
    defer m.plans = saved;
    const turn = arch_handle.turnFor(ld.h, &ld.config);
    defer turn.end();
    defer @import("../arch/sched.zig").endRequest(&ld.config);
    try testing.expectEqual(core.PoolGate.refuse, core.poolGate());
    const wide: ?core.Wide = if (planned) null else core.widePhase();
    defer if (wide) |w| w.end();
    var tok = Tokenizer.initEmpty(a);
    defer tok.deinit();
    try ld.xfm.resetCache();
    const before = core.stats().closed_asks;
    var out: Said = .{};
    errdefer out.deinit();
    var gen = try Generator.initWithOptions(io, a, ld.xfm, &tok, prompt, n, sampling, &.{}, opts);
    defer gen.deinit(a);
    if (opts.drafter_enabled) {
        while (try gen.nextDrafterRound(a)) |r| {
            defer a.free(r.tokens);
            try out.tokens.appendSlice(a, r.tokens);
        }
    } else while (try gen.next(a)) |t| try out.tokens.append(a, t);
    out.closed = core.stats().closed_asks - before;
    return out;
}

fn expectSame(what: []const u8, want: *const Said, got: *const Said) !void {
    errdefer std.debug.print("hy4 close: {s}: {d} tokens planned against {d}, {d} asks of the closed pool\n", .{ what, got.tokens.items.len, want.tokens.items.len, got.closed });
    try testing.expectEqualSlices(u32, want.tokens.items, got.tokens.items);
    try testing.expectEqual(@as(u64, 0), got.closed);
}

/// A conversation's ids: `n` of the fixture's prompt, cycled from `seed`.
fn conversation(a: std.mem.Allocator, ids: []const u32, n: usize, seed: usize) !std.ArrayList(u32) {
    var p: std.ArrayList(u32) = .empty;
    errdefer p.deinit(a);
    for (0..n) |i| try p.append(a, ids[(i * 7 + i / ids.len + seed * 131) % ids.len]);
    return p;
}

test "hy4 close: past the seal, a served image request and a restore from the SSD point behind the last reply ask the closed pool nothing outside a wide phase and give the plans-off tokens (HY4_BF16_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const h = try hy4_held.acquire(.bf16_vision);
    const ld = h.ld;
    const tower = &h.tower.?;
    const ids = try hy4_contract.prompt(a, h.dir);
    defer a.free(ids);
    const m = ld.xfm.arch_state.hy4flash.model.?;
    // An image request: its encode into the request's Media, then its prompt, as `vision_queue` and the slot run them.
    {
        const hy4t = &tower.tower.?.arch.hy4flash;
        const t = &hy4t.tower;
        const g = [2]u32{ 8, 12 };
        const pixels = try a.alloc(f32, @as(usize, g[0]) * g[1] * 3 * t.v.patch * t.v.patch);
        defer a.free(pixels);
        var rng = std.Random.DefaultPrng.init(0xc105e);
        for (pixels) |*x| x.* = rng.random().floatNorm(f32);
        const rows = t.rowsOf(g[0], g[1]);
        const prompt = try a.alloc(u32, rows + 8);
        defer a.free(prompt);
        for (prompt, 0..) |*id, i| id.* = if (i >= 3 and i < 3 + rows) ld.config.image_token_id else ids[i % ids.len];
        var said: [2]Said = undefined;
        for ([_]bool{ false, true }, &said) |planned, *out| {
            const saved = m.plans;
            if (!planned) m.plans = null;
            defer m.plans = saved;
            const turn = arch_handle.turnFor(ld.h, &ld.config);
            defer turn.end();
            const wide: ?core.Wide = if (planned) null else core.widePhase();
            defer if (wide) |w| w.end();
            const before = core.stats().closed_asks;
            var media = blk: {
                const scope = mlx.engine.enter(mlx.engine.stream());
                defer scope.end();
                break :blk try hy4t.makeMedia(rows);
            };
            defer _ = mlx.mlx_array_free(media);
            try hy4t.encodeInto(ld.xfm, &media, 0, std.mem.sliceAsBytes(pixels), g[0], g[1]);
            var ctx = ld.xfm.defaultCtx();
            ctx.vision_embeddings = media;
            const closed = core.stats().closed_asks - before;
            out.* = .{ .tokens = try contract.serialWith(ld, prompt, 8, .{ .temperature = 0 }, .{ .ctx = ctx }) };
            out.closed = core.stats().closed_asks - before;
            _ = closed;
        }
        defer for (&said) |*s| s.deinit();
        try expectSame("image", &said[0], &said[1]);
    }
    // A restore through the scheduler from the SSD point the last reply wrote.
    var first = try conversation(a, ids, 600, 0);
    defer first.deinit(a);
    var said: [2]Said = .{ .{}, .{} };
    defer for (&said) |*s| s.deinit();
    for ([_]bool{ false, true }, &said) |planned, *out| {
        const saved = m.plans;
        if (!planned) m.plans = null;
        defer m.plans = saved;
        var sv: contract.Served = undefined;
        try sv.initTiers(ld, h.dir, 0);
        defer sv.deinit();
        const wide: ?core.Wide = if (planned) null else core.widePhase();
        defer if (wide) |w| w.end();
        var f: u32 = 0;
        var reply = try sv.request(first.items, 8, &f);
        defer reply.deinit(a);
        var next: std.ArrayList(u32) = .empty;
        defer next.deinit(a);
        try next.appendSlice(a, first.items);
        try next.appendSlice(a, reply.items);
        try next.appendSlice(a, ids[0..20]);
        const before = core.stats().closed_asks;
        out.tokens = try sv.request(next.items, 8, &f);
        out.closed = core.stats().closed_asks - before;
        // The reply's end point gives back the whole conversation: the next turn prefills only its new tokens.
        try testing.expectEqual(first.items.len + reply.items.len, @as(usize, sv.reused));
        try testing.expectEqual(sv.reused, f);
    }
    try expectSame("restore from the SSD point", &said[0], &said[1]);
}

test "hy4 close: past the seal, served text requests greedy, sampled, under penalties and in DFly2 rounds ask the closed pool nothing outside a wide phase and give the plans-off tokens (HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const h = try hy4_held.acquire(.stq_openLight);
    const ids = try hy4_contract.prompt(a, h.dir);
    defer a.free(ids);
    const Case = struct { what: []const u8, sampling: sampler.SamplingParams, opts: Generator.InitOptions = .{} };
    const cases = [_]Case{
        .{ .what = "greedy", .sampling = .{ .temperature = 0 } },
        .{ .what = "sampled", .sampling = .{ .temperature = 0.8, .top_k = 40, .top_p = 0.9, .seed = 11 } },
        .{ .what = "penalties", .sampling = .{ .temperature = 0.8, .seed = 11, .repeat_penalty = 1.2, .presence_penalty = 0.4 } },
        .{ .what = "DFly2 greedy", .sampling = .{ .temperature = 0 }, .opts = .{ .drafter_enabled = true } },
        .{ .what = "DFly2 sampled", .sampling = .{ .temperature = 0.8, .seed = 7 }, .opts = .{ .drafter_enabled = true } },
    };
    for (cases) |c| {
        var want = try generated(h.ld, ids[0..24], 16, c.sampling, c.opts, false);
        defer want.deinit();
        var got = try generated(h.ld, ids[0..24], 16, c.sampling, c.opts, true);
        defer got.deinit();
        try expectSame(c.what, &want, &got);
    }
}
