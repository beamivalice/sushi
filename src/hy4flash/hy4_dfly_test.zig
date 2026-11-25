//! DFly2 (part 1x) on the tiny: `HY4_BF16_TINY_STQ` (beamster test-fixtures/hy4-bf16-tiny-stq), whose drafter tensors
//! are random in the release's names and shapes. A random drafter drafts near-misses; what these hold is what any
//! drafter's machinery must. That rounds of it give the serial run's tokens is the served contract's
//! ("cooler contract: a served Hy4 STQ request with DFly2 says its serial tokens greedy and seeded").
const std = @import("std");
const testing = std.testing;
const mlx = @import("../mlx.zig");
const Ops = @import("../mlx_ops.zig").Ops;
const model_mod = @import("../model.zig");
const hy4_config = @import("hy4_config.zig");
const hy4_model = @import("hy4_model.zig");
const hy4_dfly = @import("hy4_dfly.zig");
const Arr = mlx.mlx_array;

const Fixture = struct {
    w: model_mod.Weights,
    config: hy4_config.Config,

    fn open(s: mlx.mlx_stream) !Fixture {
        const dir = std.mem.span(std.c.getenv("HY4_BF16_TINY_STQ") orelse return error.SkipZigTest);
        const a = testing.allocator;
        var f: Fixture = .{ .w = model_mod.Weights.init(a), .config = undefined };
        errdefer f.w.deinit();
        const path = try std.fmt.allocPrintSentinel(a, "{s}/model.safetensors", .{dir}, 0);
        defer a.free(path);
        try model_mod.loadSafetensorsFile(a, &f.w, path.ptr, s, true);
        const io = std.Io.Threaded.global_single_threaded.io();
        var d = try std.Io.Dir.openDirAbsolute(io, dir, .{});
        defer d.close(io);
        const text = try d.readFileAlloc(io, "config.json", a, .limited(1 << 20));
        defer a.free(text);
        f.config = try hy4_config.parseText(text);
        return f;
    }
};

fn engineOn() bool {
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    return was;
}

fn argmaxLast(o: *Ops, logits: Arr) !u32 {
    const t = mlx.getShape(logits)[0];
    const am = try o.slot();
    try mlx.check(mlx.mlx_argmax_axis(am, try o.slice(logits, 0, t - 1, t), 1, false, o.s));
    const v = try o.contiguous(try o.cast(am.*, .uint32));
    try mlx.check(mlx.mlx_array_eval(v));
    return mlx.mlx_array_data_uint32(v).?[0];
}

fn prompt(vocab: u32) [24]u32 {
    var ids: [24]u32 = undefined;
    var r = std.Random.DefaultPrng.init(0xdf1);
    for (&ids) |*d| d.* = r.random().uintLessThan(u32, vocab - 1) + 1;
    return ids;
}

test "hy4 dfly probe: the acceptance probe scores a block at every k-th position whose drafts the greedy run covers, and leaves the model reset (HY4_BF16_TINY_STQ)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const was = engineOn();
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    var f = try Fixture.open(es);
    defer f.w.deinit();
    const ids = prompt(f.config.vocab);
    const m = try hy4_model.Model.init(a, f.config, &f.w, es);
    defer m.deinit();
    var d = try hy4_dfly.Drafter.init(a, m, &f.w);
    defer d.deinit();
    var tally: hy4_dfly.Tally = .{};
    // 16 greedy tokens drafted at 0, 3, .., 15: the blocks at 0 to 9 have their 5 drafts' tokens behind them.
    try hy4_dfly.probe(&d, m, &ids, 16, 3, &tally);
    try testing.expectEqual(@as(usize, 4), tally.blocks);
    try testing.expectEqual(@as(usize, 4), tally.reached[0]);
    try testing.expectEqual(@as(usize, 0), m.pos);
    try testing.expectEqual(@as(usize, 0), m.history.items.len);
}

test "hy4 dfly: a draft whose confidence rows cannot be read to the host leaves nothing allocated" {
    const a = testing.allocator;
    const ids = try a.alloc(u32, 3);
    defer a.free(ids);
    const best = [_]u32{ 7, 8 };
    try testing.expectError(error.HostRead, hy4_dfly.hostDraft(a, ids, &best, null));
}

fn openStq(a: std.mem.Allocator, dir: []const u8, s: mlx.mlx_stream) !struct {
    w: model_mod.Weights,
    config: hy4_config.Config,
    quant: @import("hy4_stq.zig").Quant,
} {
    const hy4_stq = @import("hy4_stq.zig");
    var w = model_mod.Weights.init(a);
    errdefer w.deinit();
    for ([_][]const u8{
        "model-globals",
        "model-trunk-00",
        "model-trunk-01",
        "model-trunk-02",
        "model-trunk-03",
        "model-trunk-04",
        "model-trunk-05",
        "model-layer-01",
        "model-layer-02",
        "model-layer-03",
        "model-layer-04",
        "model-layer-05",
        "model-dfly",
    }) |shard| {
        const path = try std.fmt.allocPrintSentinel(a, "{s}/{s}.safetensors", .{ dir, shard }, 0);
        defer a.free(path);
        try model_mod.loadSafetensorsFile(a, &w, path.ptr, s, true);
    }
    const io = std.Io.Threaded.global_single_threaded.io();
    var d = try std.Io.Dir.openDirAbsolute(io, dir, .{});
    defer d.close(io);
    const text = try d.readFileAlloc(io, "config.json", a, .limited(1 << 20));
    defer a.free(text);
    return .{
        .w = w,
        .config = try hy4_config.parseText(text),
        .quant = try hy4_stq.parseQuantText(a, text),
    };
}

test "hy4 dfly: a draft waits for the GPU once across all slots (HY4_BF16_TINY_STQ, HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const was = engineOn();
    defer @import("../test_support.zig").restoreEngine(was);
    const es = mlx.engine.stream();
    const scope = mlx.engine.enter(es);
    defer scope.end();
    var o: Ops = .{ .s = es };
    defer o.deinit();
    {
        var f = try Fixture.open(es);
        defer f.w.deinit();
        const ids = prompt(f.config.vocab);
        const m = try hy4_model.Model.init(a, f.config, &f.w, es);
        defer m.deinit();
        var d = try hy4_dfly.Drafter.init(a, m, &f.w);
        defer d.deinit();
        const t1 = try argmaxLast(&o, try o.own(try m.forwardReadout(&ids, .last)));
        const before = mlx.engine.stats().waits;
        var b = try d.draft(m, t1);
        defer b.deinit();
        const waits = mlx.engine.stats().waits - before;
        try testing.expectEqual(@as(u64, 1), waits);
    }
    {
        const dir = std.mem.span(std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest);
        var f = try openStq(a, dir, es);
        defer f.w.deinit();
        const ids = prompt(f.config.vocab);
        const m = try hy4_model.Model.initWith(a, f.config, &f.w, es, .{ .dir = dir, .quant = f.quant });
        defer m.deinit();
        var d = try hy4_dfly.Drafter.init(a, m, &f.w);
        defer d.deinit();
        const t1 = try argmaxLast(&o, try o.own(try m.forwardReadout(&ids, .last)));
        const before = mlx.engine.stats().waits;
        var b = try d.draft(m, t1);
        defer b.deinit();
        const waits = mlx.engine.stats().waits - before;
        try testing.expectEqual(@as(u64, 1), waits);
    }
}

test "hy4 dfly: a plain round of a DFly2 request makes no more launches, waits or ring copies than a plain decode step (HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const dir = std.mem.span(std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const contract = @import("../cooler_contract_test.zig");
    const hy4_contract = @import("hy4_contract.zig");
    const arch_handle = @import("../arch/handle.zig");
    const arch_sched = @import("../arch/sched.zig");
    const Generator = @import("../generate.zig").Generator;
    const Tokenizer = @import("../tokenizer.zig").Tokenizer;
    const io = std.Io.Threaded.global_single_threaded.io();
    const ids = try hy4_contract.prompt(a, dir);
    defer a.free(ids);
    const p = ids[0..24];
    const Count = struct { dispatches: u64, waits: u64, updates_copied: u64, pool_bytes_delta: i64, tok: u32 };
    const plain: Count = blk: {
        var ld: contract.Load = undefined;
        try ld.prepare(hy4_contract, dir, true);
        ld.config.stream_mtp_head = false;
        try ld.loadServed(dir);
        defer _ = ld.close();
        const turn = arch_handle.turnFor(ld.h, &ld.config);
        defer turn.end();
        defer arch_sched.endRequest(&ld.config);
        var tok = Tokenizer.initEmpty(a);
        defer tok.deinit();
        try ld.xfm.resetCache();
        var gen = try Generator.initWithOptions(io, a, ld.xfm, &tok, p, 12, .{ .temperature = 0.0 }, &.{}, .{});
        defer gen.deinit(a);
        _ = try gen.next(a);
        _ = try gen.next(a);
        const s0 = mlx.engine.stats();
        var t: u32 = 0;
        for (0..4) |_| t = (try gen.next(a)).?;
        const s1 = mlx.engine.stats();
        break :blk .{
            .dispatches = s1.dispatches - s0.dispatches,
            .waits = s1.waits - s0.waits,
            .updates_copied = s1.updates_copied - s0.updates_copied,
            .pool_bytes_delta = @as(i64, @intCast(s1.pool_bytes)) - @as(i64, @intCast(s0.pool_bytes)),
            .tok = t,
        };
    };
    const dfly: Count = blk: {
        var ld: contract.Load = undefined;
        try ld.open(hy4_contract, dir, true);
        defer _ = ld.close();
        const turn = arch_handle.turnFor(ld.h, &ld.config);
        defer turn.end();
        defer arch_sched.endRequest(&ld.config);
        var tok = Tokenizer.initEmpty(a);
        defer tok.deinit();
        try ld.xfm.resetCache();
        var gen = try Generator.initWithOptions(io, a, ld.xfm, &tok, p, 12, .{ .temperature = 0.0 }, &.{}, .{ .drafter_enabled = true });
        defer gen.deinit(a);
        gen.drafter_policy.gate.mode = .plain;
        gen.drafter_policy.gate.plain_tokens = 4;
        if (try gen.nextDrafterRound(a)) |r| a.free(r.tokens);
        if (try gen.nextDrafterRound(a)) |r| a.free(r.tokens);
        const s0 = mlx.engine.stats();
        var t: u32 = 0;
        for (0..4) |_| {
            const r = (try gen.nextDrafterRound(a)).?;
            t = r.tokens[0];
            a.free(r.tokens);
        }
        try testing.expectEqual(@as(u64, 6), gen.drafter_stats.plain);
        const s1 = mlx.engine.stats();
        break :blk .{
            .dispatches = s1.dispatches - s0.dispatches,
            .waits = s1.waits - s0.waits,
            .updates_copied = s1.updates_copied - s0.updates_copied,
            .pool_bytes_delta = @as(i64, @intCast(s1.pool_bytes)) - @as(i64, @intCast(s0.pool_bytes)),
            .tok = t,
        };
    };
    errdefer std.debug.print("hy4 dfly plain step: plain(dispatches={d}, waits={d}, updates_copied={d}, pool_delta={d}) vs dfly(dispatches={d}, waits={d}, updates_copied={d}, pool_delta={d})\n", .{
        plain.dispatches, plain.waits, plain.updates_copied, plain.pool_bytes_delta,
        dfly.dispatches,  dfly.waits,  dfly.updates_copied,  dfly.pool_bytes_delta,
    });
    try testing.expectEqual(plain.tok, dfly.tok);
    try testing.expect(dfly.waits <= plain.waits);
    try testing.expect(dfly.updates_copied <= plain.updates_copied);
    try testing.expect(dfly.dispatches <= plain.dispatches);
}

test "hy4 dfly: a plain stretch hands over fresh hidden and picks to the retrial draft round (HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const dir = std.mem.span(std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const contract = @import("../cooler_contract_test.zig");
    const hy4_contract = @import("hy4_contract.zig");
    const arch_handle = @import("../arch/handle.zig");
    const arch_sched = @import("../arch/sched.zig");
    const Generator = @import("../generate.zig").Generator;
    const Tokenizer = @import("../tokenizer.zig").Tokenizer;
    const io = std.Io.Threaded.global_single_threaded.io();
    const ids = try hy4_contract.prompt(a, dir);
    defer a.free(ids);
    const p = ids[0..24];
    var ld: contract.Load = undefined;
    try ld.open(hy4_contract, dir, true);
    defer _ = ld.close();
    var want = try contract.serial(&ld, p, 12, .{ .temperature = 0.0 });
    defer want.deinit(a);
    const turn = arch_handle.turnFor(ld.h, &ld.config);
    defer turn.end();
    defer arch_sched.endRequest(&ld.config);
    var tok = Tokenizer.initEmpty(a);
    defer tok.deinit();
    try ld.xfm.resetCache();
    var gen = try Generator.initWithOptions(io, a, ld.xfm, &tok, p, 12, .{ .temperature = 0.0 }, &.{}, .{ .drafter_enabled = true });
    defer gen.deinit(a);
    gen.drafter_policy.gate.mode = .plain;
    gen.drafter_policy.gate.retrial = 6;
    gen.drafter_policy.gate.plain_tokens = 0;
    gen.drafter_policy.gate.ms = 1000;
    gen.drafter_policy.gate.tokens = 1;
    var got: std.ArrayList(u32) = .empty;
    defer got.deinit(a);
    while (try gen.nextDrafterRound(a)) |r| {
        defer a.free(r.tokens);
        try got.appendSlice(a, r.tokens);
    }
    try testing.expectEqualSlices(u32, want.items, got.items);
    try testing.expect(gen.drafter_stats.plain >= 6);
    try testing.expect(gen.drafter_stats.rounds > gen.drafter_stats.plain);
}

test "hy4 dfly: a plain stretch pipelines one step in flight like plain decode (HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const dir = std.mem.span(std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const contract = @import("../cooler_contract_test.zig");
    const hy4_contract = @import("hy4_contract.zig");
    const arch_handle = @import("../arch/handle.zig");
    const arch_sched = @import("../arch/sched.zig");
    const Generator = @import("../generate.zig").Generator;
    const Tokenizer = @import("../tokenizer.zig").Tokenizer;
    const io = std.Io.Threaded.global_single_threaded.io();
    const ids = try hy4_contract.prompt(a, dir);
    defer a.free(ids);
    const p = ids[0..24];
    const Run = struct { waits: u64, submits: u64, in_flight: usize, toks: [4]u32 };
    const plain: Run = blk: {
        var ld: contract.Load = undefined;
        try ld.prepare(hy4_contract, dir, true);
        ld.config.stream_mtp_head = false;
        try ld.loadServed(dir);
        defer _ = ld.close();
        const turn = arch_handle.turnFor(ld.h, &ld.config);
        defer turn.end();
        defer arch_sched.endRequest(&ld.config);
        var tok = Tokenizer.initEmpty(a);
        defer tok.deinit();
        try ld.xfm.resetCache();
        var gen = try Generator.initWithOptions(io, a, ld.xfm, &tok, p, 12, .{ .temperature = 0.0 }, &.{}, .{});
        defer gen.deinit(a);
        _ = try gen.next(a);
        _ = try gen.next(a);
        const s0 = mlx.engine.stats();
        var toks: [4]u32 = undefined;
        var in_flight: usize = 0;
        for (&toks) |*t| {
            t.* = (try gen.next(a)).?;
            if (gen.pending != null and gen.ctx.cache.step == p.len + gen.completion_tokens + 1) in_flight += 1;
        }
        const s1 = mlx.engine.stats();
        break :blk .{ .waits = s1.waits - s0.waits, .submits = s1.submits - s0.submits, .in_flight = in_flight, .toks = toks };
    };
    const dfly: Run = blk: {
        var ld: contract.Load = undefined;
        try ld.open(hy4_contract, dir, true);
        defer _ = ld.close();
        const turn = arch_handle.turnFor(ld.h, &ld.config);
        defer turn.end();
        defer arch_sched.endRequest(&ld.config);
        var tok = Tokenizer.initEmpty(a);
        defer tok.deinit();
        try ld.xfm.resetCache();
        var gen = try Generator.initWithOptions(io, a, ld.xfm, &tok, p, 12, .{ .temperature = 0.0 }, &.{}, .{ .drafter_enabled = true });
        defer gen.deinit(a);
        gen.drafter_policy.gate.mode = .plain;
        gen.drafter_policy.gate.plain_tokens = 4;
        if (try gen.nextDrafterRound(a)) |r| a.free(r.tokens);
        if (try gen.nextDrafterRound(a)) |r| a.free(r.tokens);
        const s0 = mlx.engine.stats();
        var toks: [4]u32 = undefined;
        var in_flight: usize = 0;
        for (&toks) |*t| {
            const r = (try gen.nextDrafterRound(a)).?;
            t.* = r.tokens[0];
            a.free(r.tokens);
            if (gen.pending != null and gen.ctx.cache.step == p.len + gen.completion_tokens + 1) in_flight += 1;
        }
        try testing.expectEqual(@as(u64, 6), gen.drafter_stats.plain);
        const s1 = mlx.engine.stats();
        break :blk .{ .waits = s1.waits - s0.waits, .submits = s1.submits - s0.submits, .in_flight = in_flight, .toks = toks };
    };
    errdefer std.debug.print("hy4 dfly pipeline: plain(waits={d}, submits={d}, in_flight={d}) vs dfly(waits={d}, submits={d}, in_flight={d})\n", .{
        plain.waits, plain.submits, plain.in_flight,
        dfly.waits,  dfly.submits,  dfly.in_flight,
    });
    try testing.expectEqualSlices(u32, &plain.toks, &dfly.toks);
    try testing.expectEqual(@as(usize, 4), plain.in_flight);
    try testing.expectEqual(plain.in_flight, dfly.in_flight);
    try testing.expectEqual(plain.waits, dfly.waits);
    try testing.expect(dfly.submits <= plain.submits);
}

test "hy4 dfly: a resident load keeps DFly2 by default where the pack ships it and drops it under mtp off (HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const dir = std.mem.span(std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const arch = @import("arch.zig");
    const hy4_load = @import("hy4_load.zig");
    const model_load = @import("../model_load.zig");
    const trunk_mod = @import("../trunk.zig");
    var cfg = try model_mod.parseConfig(io, a, dir);
    defer cfg.deinit(a);
    cfg.expert_source_dir = try a.dupe(u8, dir);
    const mtp = model_load.mtpChoiceFor(null, &cfg);
    try testing.expect(mtp.on);
    const billed_on = (try hy4_load.residentLoadBytes(io, a, dir, &cfg, false, true)).?;
    const billed_off = (try hy4_load.residentLoadBytes(io, a, dir, &cfg, false, false)).?;
    try testing.expect(billed_on > billed_off);
    const ships = hy4_load.shipsHead(io, a, dir);
    var load_cfg = cfg;
    load_cfg.stream_mtp_head = (try model_load.residentHead(mtp, ships)).kept;
    try testing.expect(arch.drafterOn(&load_cfg));
    var off_cfg = cfg;
    off_cfg.mtp_override = false;
    off_cfg.stream_mtp_head = (try model_load.residentHead(model_load.mtpChoiceFor(null, &off_cfg), ships)).kept;
    try testing.expect(!arch.drafterOn(&off_cfg));
    var w = try model_mod.loadWeightsForConfig(io, a, dir, &load_cfg, false);
    defer w.deinit();
    var xfm = try trunk_mod.Trunk.init(io, a, load_cfg, &w);
    defer xfm.deinit();
    try testing.expect(xfm.arch_state.hy4flash.drafter != null);
}

test "hy4 dfly: coolerPipelineScope distinguishes DFly2 on vs off (HY4_STQ_TINY)" {
    const dir = std.mem.span(std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const arch = @import("arch.zig");
    var cfg_off = try model_mod.parseConfig(io, a, dir);
    defer cfg_off.deinit(a);
    cfg_off.expert_streaming = true;
    cfg_off.stream_mtp_head = false;
    var cfg_on = cfg_off;
    cfg_on.stream_mtp_head = true;
    const scope_off = arch.coolerPipelineScope(&cfg_off);
    const scope_on = arch.coolerPipelineScope(&cfg_on);
    errdefer std.debug.print("hy4 coolerPipelineScope: off={s} on={s}\n", .{ scope_off, scope_on });
    try testing.expect(!std.mem.eql(u8, &scope_off, &scope_on));
    var res_on = cfg_off;
    res_on.expert_streaming = false;
    res_on.stream_mtp_head = true;
    var res_off = res_on;
    res_off.stream_mtp_head = false;
    try testing.expect(!std.mem.eql(u8, &arch.coolerPipelineScope(&res_off), &arch.coolerPipelineScope(&res_on)));
}

test "hy4 bill: warmNeedBytes and warmRequestBill bill Hy4 DFly2 warm request with drafting on" {
    const bills = @import("../bills.zig");
    const c = try hy4_config.parseText(hy4_config.release_config);
    var mc = @import("../model.zig").ModelConfig{
        .model_type = "hy_v4_flash",
        .hy4flash = c,
        .stream_mtp_head = true,
        .expert_streaming = true,
        .vocab_size = c.vocab,
        .hidden_size = c.dim,
        .num_hidden_layers = c.layers,
    };
    const floor: u64 = bills.PREFILL_CHUNK_LADDER[bills.PREFILL_CHUNK_LADDER.len - 1];
    var plain = mc;
    plain.stream_mtp_head = false;
    const bill_plain = bills.warmRequestBill(&plain, 8, floor, false);
    const bill_off = bills.warmRequestBill(&mc, 8, floor, false);
    const bill_on = bills.warmRequestBill(&mc, 8, floor, true);
    const warm_need = bills.warmNeedBytes(&mc);
    errdefer std.debug.print("hy4 release floor=512: warmRequestBill(no DFly2)={d} MB, (DFly2, mtp=false)={d} MB, (DFly2, mtp=true)={d} MB, warmNeedBytes={d} MB\n", .{
        bill_plain >> 20, bill_off >> 20, bill_on >> 20, warm_need >> 20,
    });
    // A load that drafts holds its drafting blocks for a request that does not draft (D56); a round adds its rows.
    try testing.expect(bill_off > bill_plain);
    try testing.expect(bill_on >= bill_off);
    try testing.expect(warm_need >= bill_on);
}

test "hy4 bill: on a load that drafts, the closed-form terms at a request's reach hold the blocks reserveFor makes for it, drafting or not, at 8 and 16 bits (HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const arch = @import("arch.zig");
    const contract = @import("../cooler_contract_test.zig");
    const hy4_contract = @import("hy4_contract.zig");
    const arch_handle = @import("../arch/handle.zig");
    const arch_sched = @import("../arch/sched.zig");
    const dir = std.mem.span(std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest);
    const seq: u64 = 500;
    const max_tokens: u32 = 100;
    const chunk: u64 = 512;
    for ([_]u64{ 8, 16 }) |kv| {
        var ld: contract.Load = undefined;
        try ld.open(hy4_contract, dir, true);
        defer _ = ld.close();
        const turn = arch_handle.turnFor(ld.h, &ld.config);
        defer turn.end();
        defer arch_sched.endRequest(&ld.config);
        const m = ld.xfm.arch_state.hy4flash.model.?;
        try testing.expect(arch.drafterOn(&ld.config));
        try testing.expect(m.keep_hidden);
        m.reset();
        m.kv = if (kv == 8) .engine_default else .dense;
        const end = seq + max_tokens;
        try m.reserveFor(@intCast(end));
        const held = @import("hy4_prefix.zig").heldBytes(m, null);
        const terms = @import("hy4_ledger.zig").ledgerTerms(&ld.config, kv, chunk).?;
        errdefer std.debug.print("[hy4-bill] kv{d}: reserveFor({d}) holds {d} B, the terms {d} B\n", .{ kv, end, held, terms.at(end) });
        try testing.expect(held <= terms.at(end));
    }
}

