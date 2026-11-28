//! DeepSeek-V4.1 past its load's seal (docs/bill_redesign.md): the pool closes for the model's turns, so served
//! requests outside wide phases ask the pool nothing, reproduce plans-off tokens identically, and run cleanly against
//! the static ledger.
const std = @import("std");
const testing = std.testing;
const mlx = @import("../mlx.zig");
const core = @import("../tensor/core.zig");
const contract = @import("../cooler_contract_test.zig");
const dsv41_contract = @import("dsv41_contract.zig");
const dsv41_state = @import("dsv41_state.zig");
const dsv41_sched = @import("dsv41_sched.zig");
const dsv41_model = @import("dsv41_model.zig");
const static_bill = @import("../static_bill.zig");
const sampler = @import("../sampler.zig");
const Generator = @import("../generate.zig").Generator;
const Tokenizer = @import("../tokenizer.zig").Tokenizer;
const arch_handle = @import("../arch/handle.zig");
const arch_sched = @import("../arch/sched.zig");

const io = std.Io.Threaded.global_single_threaded.io();

const Said = struct {
    tokens: std.ArrayList(u32) = .empty,
    closed: u64 = 0,

    fn deinit(s: *Said) void {
        s.tokens.deinit(testing.allocator);
    }
};

fn generated(ld: *contract.Load, prompt: []const u32, n: u32, sampling: sampler.SamplingParams, opts: Generator.InitOptions, planned: bool) !Said {
    const a = testing.allocator;
    const m = dsv41_state.modelOf(ld.xfm).?;
    const saved = m.plans;
    if (!planned) m.plans = null;
    defer m.plans = saved;
    const turn = arch_handle.turnFor(ld.h, &ld.config);
    defer turn.end();
    defer arch_sched.endRequest(&ld.config);
    try testing.expectEqual(core.PoolGate.refuse, core.poolGate());
    const wide: ?core.Wide = if (planned) null else core.widePhase();
    defer if (wide) |w| w.end();
    var tok = Tokenizer.initEmpty(a);
    defer tok.deinit();
    const own: ?dsv41_model.Lent = if (m.serves_rows and m.bound == null) dsv41_model.lendOwn(m) else null;
    defer if (own) |l| l.end();
    try ld.xfm.resetCache();
    if (dsv41_model.currentState(m) catch null) |st| {
        if (st.reach < prompt.len + n + 16) try dsv41_model.reserveNew(m, prompt.len + n + 16);
    } else try dsv41_model.reserveNew(m, prompt.len + n + 16);
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
    errdefer std.debug.print("dsv41 close: {s}: {d} tokens planned against {d}, {d} asks of the closed pool\n", .{ what, got.tokens.items.len, want.tokens.items.len, got.closed });
    try testing.expectEqualSlices(u32, want.tokens.items, got.tokens.items);
    try testing.expectEqual(@as(u64, 0), got.closed);
}

fn checkCloseFixture(env_name: [:0]const u8) !void {
    try mlx.requireGpu();
    const raw = std.c.getenv(env_name) orelse return error.SkipZigTest;
    const dir = std.mem.span(raw);
    const a = testing.allocator;
    const ids = try dsv41_contract.prompt(a, dir);
    defer a.free(ids);

    var ld: contract.Load = undefined;
    try ld.openLight(dsv41_contract, dir, true);
    defer _ = ld.close();
    const m = dsv41_state.modelOf(ld.xfm).?;
    try testing.expect(ld.config.static_ledger != null);
    try testing.expect(m.plans != null);
    try testing.expectEqual(core.PoolGate.refuse, ld.xfm.cooler_budget.gate);

    const Case = struct { what: []const u8, sampling: sampler.SamplingParams, opts: Generator.InitOptions = .{}, need_dspark: bool = false };
    const cases = [_]Case{
        .{ .what = "greedy", .sampling = .{ .temperature = 0 } },
        .{ .what = "sampled", .sampling = .{ .temperature = 0.8, .top_k = 40, .top_p = 0.9, .seed = 11 } },
        .{ .what = "penalties", .sampling = .{ .temperature = 0.8, .seed = 11, .repeat_penalty = 1.2, .presence_penalty = 0.4 } },
        .{ .what = "DSpark greedy", .sampling = .{ .temperature = 0 }, .opts = .{ .drafter_enabled = true }, .need_dspark = true },
        .{ .what = "DSpark sampled", .sampling = .{ .temperature = 0.8, .seed = 7 }, .opts = .{ .drafter_enabled = true }, .need_dspark = true },
    };
    const prompt = ids[0..@min(ids.len, 24)];
    for (cases) |c| {
        if (c.need_dspark and m.dspark == null) continue;
        var want = try generated(&ld, prompt, 12, c.sampling, c.opts, false);
        defer want.deinit();
        var got = try generated(&ld, prompt, 12, c.sampling, c.opts, true);
        defer got.deinit();
        try expectSame(c.what, &want, &got);
    }
}

test "dsv41 close: past the seal, served STQ text requests greedy, sampled, under penalties and in DSpark rounds ask the closed pool nothing and match plans-off tokens (DSV41_STQ_TINY)" {
    if (!@import("sushi_exl3").kernels.gemmNaxOn()) return error.SkipZigTest;
    try checkCloseFixture("DSV41_STQ_TINY");
}

test "dsv41 close: past the seal, served FP8/MXFP4 text requests ask the closed pool nothing and match plans-off tokens (DSV41_TINY)" {
    try checkCloseFixture("DSV41_TINY");
}

test "dsv41 ledger: static ledger admits, releases at end, clears parked state, and refuses oversized context naming fitting max_tokens (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    if (!@import("sushi_exl3").kernels.gemmNaxOn()) return error.SkipZigTest;
    const raw = std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest;
    const dir = std.mem.span(raw);
    const a = testing.allocator;
    const ids = try dsv41_contract.prompt(a, dir);
    defer a.free(ids);

    var ld: contract.Load = undefined;
    try ld.openLight(dsv41_contract, dir, true);
    defer _ = ld.close();
    const l = ld.config.static_ledger orelse return error.TestUnexpectedResult;

    var sv: contract.Served = undefined;
    try sv.init(&ld, dir);
    defer sv.deinit();

    const prompt = ids[0..@min(ids.len, 24)];
    const need = l.terms.?.at(prompt.len + 8);
    const budget = l.budget();
    defer l.setBudget(budget);

    const before = l.committed();
    const slot = try sv.sch.submit(.{ .model = sv.entry, .prompt_ids = prompt, .sampling = .{ .temperature = 0 }, .eos_token_ids = &.{}, .max_tokens = 8, .deferred_media = null });
    _ = sv.sch.pending.orderedRemove(0);
    try static_bill.admit(slot);
    try testing.expectEqual(need, l.of(.state));
    try testing.expectEqual(before + need, l.committed());
    sv.sch.complete(slot);
    contract.Served.scheduler.drainSlots(&sv.sch, sv.sch.cleanup_queue.items);
    sv.sch.cleanup_queue.clearRetainingCapacity();
    try testing.expectEqual(@as(u64, 0), l.of(.state));

    dsv41_sched.releaseParked(sv.entry);
    try testing.expectEqual(@as(u64, 0), dsv41_sched.parkedStateBytes(sv.entry));

    const asked: u32 = 1_000_000;
    const fit_room = l.terms.?.at(prompt.len + 16);
    l.setBudget(l.committed() -| l.of(.parked) + fit_room);
    const refused = switch (static_bill.check(&ld.config, prompt.len, asked, null)) {
        .refused => |r| r,
        .proceed => return error.TestExpectedRefusal,
    };
    try testing.expect(refused.fitting_max_tokens > 0 and refused.fitting_max_tokens < asked);
    try testing.expect(static_bill.check(&ld.config, prompt.len, refused.fitting_max_tokens, null) == .proceed);
}
