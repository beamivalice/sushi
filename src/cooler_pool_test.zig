//! The Sushi Cooler contract's pool cells: what a request holds against its bill and cap, and a request refused at a
//! pool buffer. The runner and the shared helpers are in `cooler_contract_test.zig`.

const std = @import("std");
const testing = std.testing;
const mlx = @import("mlx.zig");
const model_mod = @import("model.zig");
const arch_handle = @import("arch/handle.zig");
const test_support = @import("test_support.zig");
const arch_sched = @import("arch/sched.zig");
const arch_cooler = @import("arch/cooler.zig");
const prefill_plan = @import("prefill_plan.zig");
const Generator = @import("generate.zig").Generator;
const Tokenizer = @import("tokenizer.zig").Tokenizer;
const SamplingParams = @import("sampler.zig").SamplingParams;
const io = std.Io.Threaded.global_single_threaded.io();
const contract = @import("cooler_contract_test.zig");
const ContractOf = contract.ContractOf;
const Load = contract.Load;
const MixedTick = contract.MixedTick;
const Served = contract.Served;
const Sheet = contract.Sheet;
const inUsePeak = contract.inUsePeak;
const eachArch = contract.eachArch;
const eachFixture = contract.eachFixture;
const eachQwenServed = contract.eachQwenServed;
const eachResidency = contract.eachResidency;
const fixtureChosen = contract.fixtureChosen;
const greedy = contract.greedy;
const qwenTinyServed = contract.qwenTinyServed;
const selection = contract.selection;
const serialWith = contract.serialWith;
const servedOn = contract.servedOn;
const servedRows = contract.servedRows;


test "cooler contract: a Cooler request ends with its pass closed and at most its resting cap free in the pool, and the unload leaves the pool empty" {
    try eachResidency(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            const before = Sheet.now();
            var ld: Load = undefined;
            try ld.open(C, dir, true);
            {
                errdefer _ = ld.close();
                // Lengths rising: each new one takes buffers of new sizes, which a pool without an owner keeps.
                var long: [2000]u32 = undefined;
                for (&long, 0..) |*t, i| t.* = ids[i % ids.len];
                for ([_]usize{ 1, 40, 300, 700, 1500, 2000 }) |len| {
                    var got = try greedy(&ld, long[0..len], 4);
                    got.deinit(testing.allocator);
                    const st = mlx.engine.stats();
                    const rest = arch_cooler.restCap(&ld.config);
                    errdefer std.debug.print("cooler contract: after a request over {d} tokens, {d} dispatches open, {d} B free in the pool beside {d} B asked, resting cap {d} B\n", .{ len, mlx.engine.openDispatches(), st.pool_free, st.pool_asked, rest });
                    try testing.expectEqual(@as(u32, 0), mlx.engine.openDispatches());
                    try testing.expect(st.pool_free <= rest);
                }
            }
            _ = ld.close();
            const st = mlx.engine.stats();
            errdefer std.debug.print("cooler contract: after the unload the pool holds {d} B, {d} B of them free, {d} B in use before the load\n", .{ st.pool_bytes, st.pool_free, before.in_use });
            try testing.expectEqual(@as(u64, 0), st.pool_free);
            // Every array a cache kept for the engine's side goes with its last model (`side_cache.dropAll`): the
            // pool holds only what was in use before the load, nothing when this test runs alone.
            try testing.expectEqual(before.in_use, st.pool_bytes);
        }
    }.run);
}

test "cooler contract: a Cooler prefill chunk, the widest on the load's first request and any width on a later one, grows the engine's pool by no more than the chunk's bill" {
    try eachFixture(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            var ld: Load = undefined;
            try ld.open(C, dir, true);
            defer _ = ld.close();
            const bills = @import("bills.zig");
            const kv = bills.configuredKvQuantFor(&ld.config);
            const top: usize = 8192;
            const long = try a.alloc(u32, top + 1);
            defer a.free(long);
            for (long, 0..) |*t, i| t.* = ids[i % ids.len];
            const cfg = &ld.xfm.config;
            // The widest is the load's first request; a narrower width, then the widest again, run after another ended.
            for ([_]usize{ top, 1024, top }, 0..) |pin, at| {
                const prompt = long[0 .. pin + 1];
                const width = prefill_plan.requestPrefillChunk(cfg.prefillScoreHeadDim(), cfg.num_attention_heads, prompt.len, cfg.has_sliding_window, cfg.isMoe(), cfg.longCtxGated(), pin);
                // The request's own hold on the cache's growth, which a chunk at the prompt's end regrows from full.
                const hold: @import("kv_cache.zig").Hold = .{ .reserve = prefill_plan.reservedPrefillTokens(&ld.config, prompt.len, 1, width), .end = prompt.len + 1 };
                const bits: u64 = if (kv.scheme == .off) 16 else kv.bits;
                // A reserved request's cache is its admission commit's, which this request, never admitted, does not carry.
                const commit: u64 = if (hold.reserve > 0) bills.cacheCommitBytes(&ld.config, prompt.len, 1, bits, width, false) else 0;
                const bill = bills.prefillChunkHeldMax(&ld.config, bits, width, prompt.len, .{ .hold = hold }) +| commit;
                mlx.engine.trim();
                mlx.engine.resetPoolPeak(.process);
                const held = mlx.engine.poolTotal();
                errdefer std.debug.print("cooler contract: request {d}, {d} tokens pinned at {d} rows a chunk ({d} served), bill {d} MB\n", .{ at, prompt.len, pin, width, bill >> 20 });
                var got = try serialWith(&ld, prompt, 1, .{ .temperature = 0.0 }, .{ .pinned_prefill_chunk = pin });
                got.deinit(a);
                const grew = mlx.engine.poolHeldPeak(.process) -| held;
                if (grew > bill) std.debug.print("cooler contract: the pool grew {d} MB\n", .{grew >> 20});
                try testing.expect(grew <= bill);
            }
        }
    }.run);
}

test "cooler contract: a contract load bills a chunk from what its trunk serves, the facts its load recorded past the parse (a probe serving the arm the parse did not guess) included" {
    try eachArch(struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            _ = ids;
            const against = comptime @hasDecl(C, "armsAgainstParse");
            if (against) C.armsAgainstParse(true);
            defer if (against) C.armsAgainstParse(false);
            var ld: Load = undefined;
            try ld.open(C, dir, true);
            defer _ = ld.close();
            const bills = @import("bills.zig");
            const kv = bills.configuredKvQuantFor(&ld.config);
            const bits: u64 = if (kv.scheme == .off) 16 else kv.bits;
            const top: u64 = 8192;
            const room: @import("kv_cache.zig").Room = .{ .hold = .{ .reserve = 0, .end = top + 2 } };
            // The config the cells bill from, given what the trunk recorded at its load as a served load hands it over.
            var recorded = ld.config;
            @import("arch/config.zig").adoptLoaded(&recorded, &ld.xfm.config);
            const billed = bills.prefillChunkHeldMax(&ld.config, bits, top, top + 1, room);
            const served = bills.prefillChunkHeldMax(&recorded, bits, top, top + 1, room);
            errdefer std.debug.print("cooler contract: a {d}-row chunk billed {d} KB from the load's config, {d} KB with what its trunk recorded\n", .{ top, billed >> 10, served >> 10 });
            try testing.expectEqual(served, billed);
        }
    }.run);
}

test "cooler contract: a long reserved request holds no more than its admission bill, prefill and decode" {
    const Q = ContractOf("qwen4");
    if (comptime @hasDecl(Q, "prompt")) return requestWithinBill(Q, false);
    return error.SkipZigTest;
}

test "cooler contract: a long reserved speculative request holds no more than its admission bill, prefill and decode" {
    const Q = ContractOf("qwen4");
    if (comptime @hasDecl(Q, "specOptions")) return requestWithinBill(Q, true);
    return error.SkipZigTest;
}

fn requestWithinBill(comptime Q: type, spec: bool) !void {
    try mlx.requireGpu();
    const was_enabled = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was_enabled);
    var ran = false;
    for ([_][:0]const u8{ "QWEN4_STQ_TINY", "QWEN4_TEST_MODEL" }) |env| {
        if (!fixtureChosen(selection(), env)) continue;
        const dir = std.mem.span(std.c.getenv(env) orelse continue);
        try requestWithinBillOn(Q, dir, spec);
        ran = true;
    }
    if (!ran) return error.SkipZigTest;
}

fn requestWithinBillOn(comptime Q: type, dir: []const u8, spec: bool) !void {
    const a = testing.allocator;
    const ids = try Q.prompt(a, dir);
    defer a.free(ids);
    var ld: Load = undefined;
    try ld.open(Q, dir, true);
    defer _ = ld.close();
    const bills = @import("bills.zig");
    // Past the threshold, so the cache reserves the prompt and the whole reply up front.
    const len: usize = @intCast(@import("kv_cache.zig").KVCache.RESERVE_MIN_TOKENS + 1024);
    const n: u32 = 1024;
    const long = try a.alloc(u32, len);
    defer a.free(long);
    for (long, 0..) |*t, i| t.* = ids[i % ids.len];
    const c = &ld.config;
    const kv = bills.configuredKvQuantFor(c);
    const kv_bits: u64 = if (kv.scheme == .off) 16 else kv.bits;
    const chunk = @import("prefill_plan.zig").requestPrefillChunk(c.prefillScoreHeadDim(), c.num_attention_heads, len, c.has_sliding_window, c.isMoe(), c.longCtxGated(), 0);
    const bill = bills.prefillNeededAtChunk(c, len, n, kv_bits, chunk, .{ .mtp_on = spec });
    const turn = arch_handle.turnFor(ld.h, c);
    defer turn.end();
    defer arch_sched.endRequest(c);
    var tok = Tokenizer.initEmpty(a);
    defer tok.deinit();
    try ld.xfm.resetCache();
    mlx.engine.close();
    mlx.engine.trim();
    const held0 = mlx.engine.poolTotal();
    mlx.engine.resetPoolPeak(.process);
    var mlx0: usize = 0;
    _ = mlx.mlx_get_active_memory(&mlx0);
    _ = mlx.mlx_reset_peak_memory();
    var gen = try Generator.initWithOptions(io, a, ld.xfm, &tok, long, n, .{ .temperature = 0.0 }, &.{}, if (spec) Q.specOptions(ld.xfm, 0) else .{});
    defer gen.deinit(a);
    if (spec) {
        while (try gen.nextMtp(a)) |r| a.free(r.tokens);
    } else while (try gen.next(a)) |_| {}
    mlx.engine.close();
    var mlx_peak: usize = 0;
    _ = mlx.mlx_get_peak_memory(&mlx_peak);
    const held = (mlx.engine.poolHeldPeak(.process) -| held0) + (@as(u64, mlx_peak) -| mlx0);
    if (held > bill) std.debug.print("cooler contract: {s}: a {d}-token request with {d} new tokens (MTP {}) held {d} KB over its {d} KB bill\n", .{ dir, len, n, spec, (held - bill) >> 10, bill >> 10 });
    try testing.expect(held <= bill);
}

test "an MLX Flash-Next prefill with its MTP head holds no more per prompt token than its bill's storage" {
    const Q = ContractOf("qwen4");
    if (comptime !@hasDecl(Q, "specOptions")) return error.SkipZigTest;
    try mlx.requireGpu();
    var ran = false;
    for ([_][:0]const u8{ "QWEN4_STQ_TINY", "QWEN4_TEST_MODEL" }) |env| {
        if (!fixtureChosen(selection(), env)) continue;
        const dir = std.mem.span(std.c.getenv(env) orelse continue);
        try headPrefillWithinStorage(Q, dir);
        ran = true;
    }
    if (!ran) return error.SkipZigTest;
}

/// What a prefill's chunks start on, and what the bill's storage terms price, for two prompt lengths: the growth between
/// them is the hold a prompt token adds, which no fixed part of either reading can hide.
fn headPrefillWithinStorage(comptime Q: type, dir: []const u8) !void {
    const a = testing.allocator;
    const bills = @import("bills.zig");
    const generate = @import("generate.zig");
    const ids = try Q.prompt(a, dir);
    defer a.free(ids);
    var ld: Load = undefined;
    try ld.open(Q, dir, false);
    defer _ = ld.close();
    const c = &ld.config;
    const chunk: usize = prefill_plan.PREFILL_CHUNK_FLOOR;
    const kv = bills.configuredKvQuantFor(c);
    const kv_bits: u64 = if (kv.scheme == .off) 16 else kv.bits;
    var held: [2]u64 = undefined;
    var storage: [2]u64 = undefined;
    const lens = [2]usize{ 16 * 1024 + 1, 48 * 1024 + 1 };
    for (lens, 0..) |len, k| {
        const long = try a.alloc(u32, len);
        defer a.free(long);
        for (long, 0..) |*t, i| t.* = ids[(i * 7919) % ids.len];
        const terms = bills.prefillRequestTerms(c, len, 1, kv_bits, chunk, .{ .mtp_on = true });
        storage[k] = len * bills.kvBytesPerTokenAtBits(c.kvBytesPerToken(), kv_bits) + terms.reserved_kv_bytes + terms.state_bytes +
            terms.checkpoint_bytes + terms.grow_coexist_bytes + terms.qsa_ring_bytes + terms.mtp_head_kv_bytes;
        const turn = arch_handle.turnFor(ld.h, c);
        defer turn.end();
        defer arch_sched.endRequest(c);
        var tok = Tokenizer.initEmpty(a);
        defer tok.deinit();
        try ld.xfm.resetCache();
        _ = mlx.mlx_clear_cache();
        var active0: usize = 0;
        _ = mlx.mlx_get_active_memory(&active0);
        generate.held_between_chunks_for_test = 0;
        var opts = Q.specOptions(ld.xfm, 0);
        opts.pinned_prefill_chunk = chunk;
        var gen = try Generator.initWithOptions(io, a, ld.xfm, &tok, long, 1, .{ .temperature = 0.0 }, &.{}, opts);
        defer gen.deinit(a);
        if (try gen.nextMtp(a)) |r| a.free(r.tokens);
        held[k] = generate.held_between_chunks_for_test -| active0;
    }
    const grew = held[1] -| held[0];
    const billed = storage[1] -| storage[0];
    if (grew > billed) std.debug.print("{s}: from {d} to {d} prompt tokens at {d}-row chunks a chunk started on {d} KB more, the billed storage {d} KB more\n", .{ dir, lens[0], lens[1], chunk, grew >> 10, billed >> 10 });
    try testing.expect(grew <= billed);
}

/// A greedy request's slot, submitted and prefilled as the inference loop runs it (`failPrefill` on a failure).
/// `refuse_first`: its prefill's first pool buffer is refused, as when it meets a full pool.
fn servedAdmit(sv: *Served, prompt: []const u32, n: u32, refuse_first: bool) !*Served.scheduler.Slot {
    const scheduler = Served.scheduler;
    const sch = &sv.sch;
    const slot = try sch.submit(.{ .model = sv.entry, .prompt_ids = prompt, .sampling = .{ .temperature = 0 }, .eos_token_ids = &.{}, .max_tokens = n });
    for (sch.pending.items, 0..) |p, k| if (p == slot) {
        _ = sch.pending.orderedRemove(k);
        break;
    };
    if (refuse_first) mlx.engine.fault.arm(1);
    const prefilled = scheduler.runPrefill(sch, slot);
    mlx.engine.fault.disarm();
    prefilled catch |e| scheduler.failPrefill(slot, e);
    if (!scheduler.postPrefillTerminal(sch, slot)) try sch.decoding.append(testing.allocator, slot);
    return slot;
}

/// One decode tick of the scheduler's decoding slots, as the inference loop runs it (`failTick` on a failure).
fn servedTick(sv: *Served) void {
    const scheduler = Served.scheduler;
    var act: [4]*scheduler.Slot = undefined;
    var k: usize = 0;
    for (sv.sch.decoding.items) |s| if (s.joinsTick() and k < act.len) {
        act[k] = s;
        k += 1;
    };
    if (k > 0) scheduler.runDecodeTick(&sv.sch, act[0..k]) catch |e| scheduler.failTick(act[0..k], e, "[scheduler]");
    scheduler.cullDecoding(&sv.sch);
}

test "cooler contract: a served request refused at its prefill's first pool buffer fails alone as TransientCapExceeded; the request after it and one decoding beside it say their solo tokens" {
    const Body = struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            var ld: Load = undefined;
            try ld.open(C, dir, true);
            defer _ = ld.close();
            var sv: Served = undefined;
            try sv.init(&ld, dir);
            defer sv.deinit();
            var other: [27]u32 = undefined;
            for (&other, 0..) |*t, i| t.* = ids[(i * 7 + 3) % ids.len];
            var refused: [19]u32 = undefined;
            for (&refused, 0..) |*t, i| t.* = ids[(i * 5 + 1) % ids.len];
            var f: u32 = 0;
            var solo = try sv.request(&other, 12, &f);
            defer solo.deinit(a);
            // The refused request arrives before the other one is admitted, then after it has decoded three tokens.
            for ([_]u32{ 0, 3 }) |ticks_before| {
                errdefer std.debug.print("cooler contract: the refused request arrived after {d} ticks of the other\n", .{ticks_before});
                var slots: [2]*Served.scheduler.Slot = undefined;
                var made: usize = 0;
                defer {
                    for (slots[0..made]) |s| sv.sch.complete(s);
                    for (sv.sch.cleanup_queue.items) |s| s.deinit();
                    sv.sch.cleanup_queue.clearRetainingCapacity();
                }
                const other_first = ticks_before > 0;
                if (other_first) {
                    slots[made] = try servedAdmit(&sv, &other, 12, false);
                    made += 1;
                    for (0..ticks_before) |_| servedTick(&sv);
                }
                slots[made] = try servedAdmit(&sv, &refused, 12, true);
                const r = slots[made];
                made += 1;
                if (!other_first) {
                    slots[made] = try servedAdmit(&sv, &other, 12, false);
                    made += 1;
                }
                const o = if (other_first) slots[0] else slots[1];
                while (o.joinsTick()) servedTick(&sv);
                try testing.expectEqualStrings("TransientCapExceeded", r.error_code orelse "served");
                try testing.expect(o.error_code == null);
                try testing.expectEqualSlices(u32, solo.items, o.legacy_gen.?.generated_ids.items);
                try testing.expect(!mlx.errorPending());
            }
        }
    };
    var ran = false;
    inline for (.{ .{ "qwen4", "QWEN4_STQ_TINY" }, .{ "dsv41", "DSV41_STQ_TINY" } }) |arm| {
        if (servedOn(arm[0], &.{arm[1]}, Body.run)) |_| {
            ran = true;
        } else |e| if (e != error.SkipZigTest) return e;
    }
    if (!ran) return error.SkipZigTest;
}

test "cooler contract: a pool refusal in one request of a mixed tick fails that request alone by its name and leaves the others their solo tokens" {
    try qwenTinyServed(struct {
        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            var ld: Load = undefined;
            try ld.open(Q, dir, true);
            defer _ = ld.close();
            var sv: Served = undefined;
            try sv.init(&ld, dir);
            defer sv.deinit();
            var b: [3][64]u32 = undefined;
            var p: [3][]u32 = undefined;
            for (&b, 0..) |*x, i| {
                p[i] = x[0 .. 21 + 9 * i];
                for (p[i], 0..) |*t, k| t.* = ids[(k * (7 + 2 * i) + i + 2) % ids.len];
            }
            const seeded: SamplingParams = .{ .temperature = 0.8, .top_k = 20, .seed = 5 };
            var seen: std.AutoHashMapUnmanaged(u64, void) = .empty;
            defer seen.deinit(testing.allocator);
            // A logprobs request decodes serially and two speculative ones run their own rounds: three forwards a tick.
            try MixedTick.faultSweep(Q, &sv, &.{ .{ .prompt = p[0], .n = 10, .logprobs = 2 }, .{ .prompt = p[1], .n = 10, .mtp = true }, .{ .prompt = p[2], .n = 10, .mtp = true, .sampling = seeded } }, &seen);
            // One of them leaves after two ticks: the sites its commit and drain between the others' ticks add.
            try MixedTick.faultSweep(Q, &sv, &.{ .{ .prompt = p[0], .n = 12, .logprobs = 2 }, .{ .prompt = p[1], .n = 12, .mtp = true, .cancel_at = 2 }, .{ .prompt = p[2], .n = 12, .mtp = true, .sampling = seeded } }, &seen);
        }
    }.run);
}

/// The served width hook (`bills.adaptivePrefillWidth`) and its widen confirmation (`adaptivePrefillWidenStillFits`'s
/// test) on a box whose free memory is what admission read, less what the request has taken from the engine's pool
/// since. Records the width each boundary chose, and the widens it confirmed.
const AdmittedBox = struct {
    cfg: *const @import("model.zig").ModelConfig,
    bits: u64,
    avail: u64,
    in0: u64,
    widths: [64]u32 = @splat(0),
    n: usize = 0,
    widened: u32 = 0,
    /// The cache's room at the boundary the hook was last asked at, which the confirmation prices a widen from.
    room: @import("kv_cache.zig").Room = .{},

    fn call(ctx: *anyopaque, pos: usize, cur: u32, cap: u32, st: *@import("mtp_plan.zig").AdaptiveWidthState) u32 {
        const box: *AdmittedBox = @ptrCast(@alignCast(ctx));
        const used = (mlx.engine.poolTotal() - mlx.engine.poolFree()) -| box.in0;
        box.room = st.room;
        const w = @import("bills.zig").adaptivePrefillWidth(box.cfg, box.bits, pos, box.avail -| used, cur, cap, st);
        if (box.n < box.widths.len) box.widths[box.n] = if (w != 0) w else cur;
        box.n += 1;
        return w;
    }
    fn confirm(ctx: *anyopaque, pos: usize, want: u32) bool {
        const box: *AdmittedBox = @ptrCast(@alignCast(ctx));
        const bills = @import("bills.zig");
        const used = (mlx.engine.poolTotal() - mlx.engine.poolFree()) -| box.in0;
        const fits = bills.prefillChunkHeldMax(box.cfg, box.bits, want, pos, box.room) <= box.avail -| used;
        if (fits) box.widened += 1;
        return fits;
    }
};

test "cooler contract: a cold prompt admitted at its 512-row bill holds no more than that bill through its prefill, under the served width hook" {
    const Q = ContractOf("qwen4");
    if (comptime !@hasDecl(Q, "prompt")) return error.SkipZigTest;
    try mlx.requireGpu();
    const was_enabled = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was_enabled);
    const dir = std.mem.span(std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const bills = @import("bills.zig");
    const ids = try Q.prompt(a, dir);
    defer a.free(ids);
    var ld: Load = undefined;
    try ld.open(Q, dir, true);
    defer _ = ld.close();
    // Long enough that the request's reservation outgrows the cache's floor.
    const len: usize = @intCast(@import("kv_cache.zig").KVCache.RESERVE_MIN_TOKENS + 1024);
    const n: u32 = 16;
    const long = try a.alloc(u32, len);
    defer a.free(long);
    for (long, 0..) |*t, i| t.* = ids[i % ids.len];
    const c = &ld.config;
    const bits = bills.defaultKvBits(c);
    // Admission at the edge: the memory it reads is the 512-row bill, and the width it picks there.
    const bill = bills.prefillNeededAtChunk(c, len, n, bits, 512, .{});
    const picked = bills.chooseRequestPrefillChunk(c, len, n, bits, bill, 0, 0, .{});
    const turn = arch_handle.turnFor(ld.h, c);
    defer turn.end();
    defer arch_sched.endRequest(c);
    var tok = Tokenizer.initEmpty(a);
    defer tok.deinit();
    try ld.xfm.resetCache();
    mlx.engine.close();
    mlx.engine.trim();
    const held0 = mlx.engine.poolTotal();
    var box: AdmittedBox = .{ .cfg = c, .bits = bits, .avail = bill, .in0 = mlx.engine.poolTotal() - mlx.engine.poolFree() };
    mlx.engine.resetPoolPeak(.process);
    var gen = try Generator.initWithOptions(io, a, ld.xfm, &tok, long, n, .{ .temperature = 0.0 }, &.{}, .{
        .pinned_prefill_chunk = picked,
        .adaptive_chunk_width = true,
        .chunk_width_hook = .{ .ctx = &box, .call = AdmittedBox.call, .confirm = AdmittedBox.confirm },
    });
    defer gen.deinit(a);
    _ = try gen.next(a);
    mlx.engine.close();
    const held = mlx.engine.poolHeldPeak(.process) -| held0;
    errdefer std.debug.print("a cold {d}-token prompt admitted at its {d} MB bill (width {d}), the hook's widths {any}, {d} widens confirmed: {d} MB held at the peak\n", .{ len, bill >> 20, picked, box.widths[0..@min(box.n, box.widths.len)], box.widened, held >> 20 });
    try testing.expect(held <= bill);
}

test "cooler contract: a speculative request whose reply ends at a cache step grows its KV and its draft head's KV no further than the rows its bill prices" {
    const Q = ContractOf("qwen4");
    if (comptime !@hasDecl(Q, "specOptions")) return error.SkipZigTest;
    try mlx.requireGpu();
    if (!fixtureChosen(selection(), "QWEN4_STQ_TINY")) return error.SkipZigTest;
    const dir = std.mem.span(std.c.getenv("QWEN4_STQ_TINY") orelse return error.SkipZigTest);
    const was_enabled = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was_enabled);
    const a = testing.allocator;
    const ids = try Q.prompt(a, dir);
    defer a.free(ids);
    var ld: Load = undefined;
    try ld.open(Q, dir, true);
    defer _ = ld.close();
    const bills = @import("bills.zig");
    const depth_bounds = @import("mtp_depth_bounds.zig");
    const saved = depth_bounds.active;
    defer depth_bounds.active = saved;
    depth_bounds.active = .{ .min = 6, .max = 6 };
    const prompt = ids[0..@min(ids.len, 9)];
    const ctx: u64 = bills.getEffectiveContextLength(&ld.config);
    var over: usize = 0;
    // A reply that ends at, or a draft block short of, a 256-row step and a +25% step.
    for ([_]u32{ 256, 512, 1280 }) |end| for ([_]u32{ 0, 1, 3, 6 }) |less| {
        const n: u32 = end - @as(u32, @intCast(prompt.len)) - less;
        const billed = bills.cacheCapacityTokens(&ld.config, prompt.len, n, ld.config.prefillStartWidth(), ctx, bills.speculativeRowsPastEnd(&ld.config, .{ .mtp_on = true }), bills.defaultKvBits(&ld.config));
        const turn = arch_handle.turnFor(ld.h, &ld.config);
        defer turn.end();
        defer arch_sched.endRequest(&ld.config);
        var tok = Tokenizer.initEmpty(a);
        defer tok.deinit();
        try ld.xfm.resetCache();
        var gen = try Generator.initWithOptions(io, a, ld.xfm, &tok, prompt, n, .{ .temperature = 0.0 }, &.{}, Q.specOptions(ld.xfm, 6));
        defer gen.deinit(a);
        var head: usize = 0;
        while (try gen.nextMtp(a)) |r| {
            a.free(r.tokens);
            if (gen.mtp_cache) |*mc| mc.activate();
            head = @max(head, Q.headCacheTokens(ld.xfm));
        }
        const trunk_rows = ld.xfm.cache.residentCapacityTokens();
        if (head > billed or trunk_rows > billed) {
            over += 1;
            std.debug.print("cooler contract: prompt {d} + max_tokens {d}: KV capacity {d} rows, the head's {d}, past the {d} its bill prices\n", .{ prompt.len, n, trunk_rows, head, billed });
        }
    };
    try testing.expectEqual(@as(usize, 0), over);
}

test "cooler contract: a speculative request's growth commit, less its resident KV as admission reads it, never drops below what its KV, histories and draft head still grow" {
    const Q = ContractOf("qwen4");
    if (comptime !@hasDecl(Q, "stateHeldBytes")) return error.SkipZigTest;
    try mlx.requireGpu();
    if (!fixtureChosen(selection(), "QWEN4_STQ_TINY")) return error.SkipZigTest;
    const dir = std.mem.span(std.c.getenv("QWEN4_STQ_TINY") orelse return error.SkipZigTest);
    const was_enabled = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was_enabled);
    const a = testing.allocator;
    const ids = try Q.prompt(a, dir);
    defer a.free(ids);
    var ld: Load = undefined;
    try ld.open(Q, dir, true);
    defer _ = ld.close();
    const bills = @import("bills.zig");
    const outstandingGrowth = @import("slot_admission.zig").outstandingGrowth;
    const depth_bounds = @import("mtp_depth_bounds.zig");
    const saved = depth_bounds.active;
    defer depth_bounds.active = saved;
    depth_bounds.active = .{ .min = 6, .max = 6 };
    const prompt = ids[0..@min(ids.len, 9)];
    const kv = ld.xfm.cache.config;
    const kv_bits: u64 = if (kv.scheme == .off) 16 else kv.bits;
    var under: usize = 0;
    for ([_]u32{ 256, 1280 }) |end| for ([_]u32{ 0, 6 }) |less| {
        const n: u32 = end - @as(u32, @intCast(prompt.len)) - less;
        const commit = bills.admissionCommitBytes(&ld.config, prompt.len, n, kv_bits, ld.config.prefillStartWidth(), true);
        const turn = arch_handle.turnFor(ld.h, &ld.config);
        defer turn.end();
        defer arch_sched.endRequest(&ld.config);
        var tok = Tokenizer.initEmpty(a);
        defer tok.deinit();
        try ld.xfm.resetCache();
        var gen = try Generator.initWithOptions(io, a, ld.xfm, &tok, prompt, n, .{ .temperature = 0.0 }, &.{}, Q.specOptions(ld.xfm, 6));
        defer gen.deinit(a);
        // At each round's end: what admission holds outstanding (the slot side reads the trunk's KV as resident), and
        // everything the request's storage holds.
        var points: std.ArrayList([2]u64) = .empty;
        defer points.deinit(a);
        while (true) {
            if (gen.mtp_cache) |*mc| mc.activate();
            const trunk_kv = ld.xfm.cache.residentBytes();
            try points.append(a, .{ outstandingGrowth(commit, trunk_kv), trunk_kv + Q.stateHeldBytes(ld.xfm) });
            const r = try gen.nextMtp(a) orelse break;
            a.free(r.tokens);
        }
        var peak: u64 = 0;
        var i = points.items.len;
        while (i > 0) {
            i -= 1;
            const p = points.items[i];
            peak = @max(peak, p[1]);
            if (p[0] < peak - p[1]) {
                under += 1;
                std.debug.print("cooler contract: prompt {d} + max_tokens {d}, round {d}: {d} B outstanding, {d} B still to grow\n", .{ prompt.len, n, i, p[0], peak - p[1] });
                break;
            }
        }
    };
    try testing.expectEqual(@as(usize, 0), under);
}

test "cooler contract: a served request, plain or speculative, solo or joined, short or past a chunk, holds no more in use than the exact part of its bill" {
    try eachQwenServed(struct {
        fn run(comptime Q: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            const bills = @import("bills.zig");
            const depth_bounds = @import("mtp_depth_bounds.zig");
            const saved = depth_bounds.active;
            defer depth_bounds.active = saved;
            var ld: Load = undefined;
            try ld.open(Q, dir, true);
            defer _ = ld.close();
            var sv: Served = undefined;
            try sv.init(&ld, dir);
            defer sv.deinit();
            const c = &ld.config;
            const kv = bills.configuredKvQuantFor(c);
            const kv_bits: u64 = if (kv.scheme == .off) 16 else kv.bits;
            const chunk: u64 = c.prefillStartWidth();
            // The unnamed small tensors of a request: well under one recurrent state set.
            const small: u64 = 4 << 20;
            const long = try a.alloc(u32, chunk + 1);
            defer a.free(long);
            for (long, 0..) |*t, i| t.* = ids[(i * 7) % ids.len];
            var other: [9]u32 = undefined;
            for (&other, 0..) |*t, i| t.* = ids[(i * 3 + 1) % ids.len];
            const n: u32 = 64;
            var over: usize = 0;
            const Ask = struct { sv: *Served, prompts: []const []const u32, n: u32, mtp: bool };
            // Depth 1 verifies two rows (the fold), depth 6 seven (the replay).
            for ([_]?u32{ null, 1, 6 }) |depth| for ([_][]const u32{ ids[0..@min(ids.len, 9)], long }) |p| for ([_]usize{ 1, 2 }) |rows| {
                if (rows == 2 and p.len > 9) continue;
                const mtp = depth != null;
                if (depth) |d| depth_bounds.active = .{ .min = d, .max = d };
                defer depth_bounds.active = saved;
                const prompts = [2][]const u32{ p, &other };
                var exact: u64 = 0;
                for (prompts[0..rows]) |q| exact += bills.requestHeldBytes(c, q.len, n, kv_bits, chunk, .{ .mtp_on = mtp });
                const asked = try inUsePeak(Ask{ .sv = &sv, .prompts = prompts[0..rows], .n = n, .mtp = mtp }, struct {
                    fn go(q: Ask) !void {
                        if (q.prompts.len == 1) {
                            var f: u32 = 0;
                            var got = try q.sv.requestWith(q.prompts[0], q.n, .{ .temperature = 0.0 }, q.mtp, &f);
                            got.deinit(testing.allocator);
                        } else {
                            var o: [2]std.ArrayList(u32) = undefined;
                            try servedRows(q.sv, q.prompts[0..2].*, q.n, q.mtp, &o);
                            for (&o) |*x| x.deinit(testing.allocator);
                        }
                    }
                }.go);
                if (asked > exact + small) {
                    over += 1;
                    std.debug.print("cooler contract: {s}: {d} row(s) of a {d}-token prompt + {d}, depth {?d}: {d} KB in use at the peak, {d} KB past the exact bill ({d} KB)\n", .{ dir, rows, p.len, n, depth, asked >> 10, (asked - exact) >> 10, exact >> 10 });
                }
            };
            try testing.expectEqual(@as(usize, 0), over);
        }
    }.run);
}

/// A box whose free memory is `base` plus what the decoder pools gave back, less a resident pack's union workspace (as
/// `dsv41_stream.BorrowProbe`); it counts the pools released and read back between its reads.
const LeaseProbe = struct {
    var engine: ?*@import("expert_stream.zig").Engine = null;
    var base: u64 = 0;
    var seen: usize = 0;
    var released: usize = 0;
    var restored: usize = 0;
    /// Armed at the next read: a decode forward's first read is its first read-back's, so the fault's count starts there.
    var arm_at: u64 = 0;
    /// With `mark`, the fault's count at the first two reads: a decode forward reads before each read-back.
    var mark: bool = false;
    var marks: [2]u64 = .{ 0, 0 };
    var marked: usize = 0;

    fn reset() void {
        seen = 0;
        released = 0;
        restored = 0;
    }

    fn headroom(_: *const model_mod.ModelConfig) @import("bills.zig").Headroom {
        if (arm_at != 0) mlx.engine.fault.arm(arm_at);
        arm_at = 0;
        if (mark and marked < marks.len) {
            marks[marked] = mlx.engine.fault.count();
            marked += 1;
        }
        const e = engine.?;
        var lent: u64 = 0;
        var n: usize = 0;
        for (e.borrow_from..e.layers.len) |li| {
            lent += e.lentBytes(@intCast(li));
            n += @intFromBool(e.isBorrowed(@intCast(li)));
        }
        if (n > seen) released += n - seen;
        if (n < seen) restored += seen - n;
        seen = n;
        const made = if (e.plan.resident and e.grouped_union.len != 0) e.unionBytes() else 0;
        return .{ .live = (base + lent) -| made };
    }
};

const LeaseRun = struct {
    a_tokens: std.ArrayList(u32) = .empty,
    released: usize = 0,
    restored: usize = 0,
    filled: u64 = 0,
    b_ticks: u32 = 0,
    failed: usize = 0,

    fn deinit(self: *LeaseRun) void {
        self.a_tokens.deinit(testing.allocator);
    }
};

/// Request A prefills `a_prompt` under the probe; with `b_spec`, request B (in DSpark rounds or not) decodes joined
/// rows between A's chunks. Then both decode to their end.
fn leaseRun(ld: *Load, dir: []const u8, a_prompt: []const u32, b_prompt: []const u32, b_spec: ?bool) !LeaseRun {
    const scheduler = Served.scheduler;
    const a = testing.allocator;
    var sv: Served = undefined;
    try sv.init(ld, dir);
    defer sv.deinit();
    const sch = &sv.sch;
    const engine = LeaseProbe.engine.?;
    var run: LeaseRun = .{};
    errdefer run.deinit();
    var slots: [2]?*scheduler.Slot = .{ null, null };
    defer {
        for (slots) |s| if (s) |x| sch.complete(x);
        for (sch.cleanup_queue.items) |s| s.deinit();
        sch.cleanup_queue.clearRetainingCapacity();
    }
    if (b_spec) |spec| {
        const head = if (spec) @import("generate.zig").MtpHeadRef.fromModel(arch_handle.trunk(sv.entry.handle.?), true, null) else null;
        const b = try sch.submit(.{ .model = sv.entry, .prompt_ids = b_prompt, .sampling = .{ .temperature = 0.0 }, .eos_token_ids = &.{}, .max_tokens = 400, .enable_mtp = spec, .mtp = head });
        slots[1] = b;
        _ = sch.pending.orderedRemove(0);
        try scheduler.runPrefill(sch, b);
        if (!scheduler.postPrefillTerminal(sch, b)) try sch.decoding.append(a, b);
        var one = [_]*scheduler.Slot{b};
        for (0..2) |_| try scheduler.runDecodeTick(sch, &one);
    }
    const b_before: u32 = if (slots[1]) |b| b.completion_tokens else 0;
    const fill0 = engine.fill_bytes_total;
    LeaseProbe.reset();
    const slot_a = try sch.submit(.{ .model = sv.entry, .prompt_ids = a_prompt, .sampling = .{ .temperature = 0.0 }, .eos_token_ids = &.{}, .max_tokens = 8 });
    slots[0] = slot_a;
    _ = sch.pending.orderedRemove(0);
    scheduler.runPrefill(sch, slot_a) catch |e| slot_a.markError(@errorName(e));
    run.released = LeaseProbe.released;
    run.restored = LeaseProbe.restored;
    run.filled = engine.fill_bytes_total - fill0;
    run.b_ticks = (if (slots[1]) |b| b.completion_tokens else 0) - b_before;
    if (!scheduler.postPrefillTerminal(sch, slot_a)) try sch.decoding.append(a, slot_a);
    var guard: usize = 0;
    while (guard < 2000) : (guard += 1) {
        var act: [2]*scheduler.Slot = undefined;
        var k: usize = 0;
        for (sch.decoding.items) |s| if (s.joinsTick()) {
            act[k] = s;
            k += 1;
        };
        if (k == 0) break;
        scheduler.runDecodeTick(sch, act[0..k]) catch |e| for (act[0..k]) |s| s.markError(@errorName(e));
        scheduler.cullDecoding(sch);
    }
    scheduler.cullDecoding(sch);
    for (slots) |s| if (s) |x| if (x.error_code != null) {
        run.failed += 1;
        var msg: [300]u8 = undefined;
        _ = mlx.takeError(&msg);
    };
    if (slot_a.error_code == null) try run.a_tokens.appendSlice(a, slot_a.legacy_gen.?.generated_ids.items);
    return run;
}

test "cooler contract: a DeepSeek prefill that releases decoder pools keeps them released through another request's joined and DSpark decode ticks between its chunks, and says its solo tokens" {
    try mlx.requireGpu();
    const D = ContractOf("dsv41");
    if (comptime !@hasDecl(D, "prompt")) return error.SkipZigTest;
    const was_enabled = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was_enabled);
    const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const ids = try D.prompt(a, dir);
    defer a.free(ids);
    var ld: Load = undefined;
    try ld.open(D, dir, true);
    defer _ = ld.close();
    if ((arch_sched.rowCap(&ld.config) orelse 1) < 2) return error.SkipZigTest;
    const borrow = @import("dsv41/dsv41_borrow.zig");
    const xfm = arch_handle.trunk(ld.h);
    const engine = xfm.expert_stream.?;
    const m = @import("dsv41/dsv41_state.zig").modelOf(xfm).?;
    // Three 2048-row chunks and the held-back span; the free memory is the second chunk's need less half a pool, so the
    // first chunk fits, the second and third release pools, and a decode step between them finds room to read one back.
    ld.config.pinned_prefill_chunk = 2048;
    const len: usize = 3 * 2048 + prefill_plan.SSM_SNAPSHOT_BACKOFF + 1;
    var long: [len]u32 = undefined;
    for (&long, 0..) |*t, i| t.* = ids[(i * 7 + i / ids.len) % ids.len];
    var short: [40]u32 = undefined;
    for (&short, 0..) |*t, i| t.* = ids[(i * 5 + 2) % ids.len];
    const need = borrow.prefillNeed(&m.cfg, @import("cooler_side.zig").onEngine(&ld.config), 2048, 4096);
    LeaseProbe.engine = engine;
    defer LeaseProbe.engine = null;
    LeaseProbe.base = need - engine.poolBytes(@intCast(engine.layers.len - 1)) / 2;
    borrow.headroom_fn = &LeaseProbe.headroom;
    defer borrow.headroom_fn = null;
    var solo = try leaseRun(&ld, dir, &long, &short, null);
    defer solo.deinit();
    try testing.expect(solo.failed == 0 and solo.released > 0);
    for ([_]bool{ false, true }) |b_spec| {
        var pair = try leaseRun(&ld, dir, &long, &short, b_spec);
        defer pair.deinit();
        errdefer std.debug.print("B {s}, {d} of its tokens inside A's prefill: A alone released {d} pools, read back {d}, filled {d} KB; beside B {d}, {d}, {d} KB\n", .{
            if (b_spec) "in DSpark rounds" else "joined", pair.b_ticks, solo.released, solo.restored, solo.filled >> 10, pair.released, pair.restored, pair.filled >> 10,
        });
        try testing.expect(pair.b_ticks > 0);
        try testing.expectEqual(@as(usize, 0), pair.failed);
        try testing.expectEqual(solo.released, pair.released);
        try testing.expectEqual(solo.restored, pair.restored);
        try testing.expectEqualSlices(u32, solo.a_tokens.items, pair.a_tokens.items);
    }
}

test "cooler contract: a DeepSeek decoder pool left released with no lessee, on Sushi Cooler beside the engine's parked free buffers and on MLX, is held off admission, a co-loaded model's, the other forwards' headroom and a load's preflight, and only the lessee's own reading counts it" {
    try mlx.requireGpu();
    const D = ContractOf("dsv41");
    if (comptime !@hasDecl(D, "prompt")) return error.SkipZigTest;
    const was_enabled = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was_enabled);
    const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    for ([_]bool{ true, false }) |cooler| try lentHeldOff(dir, cooler);
}

fn lentHeldOff(dir: []const u8, cooler: bool) !void {
    const bills = @import("bills.zig");
    const gpu_mem = @import("gpu_mem.zig");
    var lease: LentLease = undefined;
    try lease.open(dir, cooler);
    defer lease.close();
    const engine = lease.engine;
    // On Cooler the request parked the engine's free buffers, which admission credits as its forwards reuse them.
    if (cooler) try testing.expect(gpu_mem.engineFree() > 0);
    const config = &lease.ld.config;
    var qwen = @import("bills_fixtures_test.zig").qwen4ExpOomConfig();
    const Reading = struct { admission: u64, beside: u64, forward: u64, load: u64, own: u64 };
    const read = struct {
        fn at(c: *const model_mod.ModelConfig, q: *const model_mod.ModelConfig) Reading {
            return .{
                .admission = bills.admissionAvailableNow(c),
                .beside = bills.admissionAvailableNow(q),
                .forward = bills.prefillHeadroomNow(c, 0),
                .load = bills.FitReadings.now().preflightAvailable().bytes orelse 0,
                .own = @import("dsv41/dsv41_bill.zig").dsv41Headroom(c).bound(),
            };
        }
    }.at;
    // On a resident pack the first release makes the union workspace, which the read-back holds beside the pools (the
    // read-back cell): made before the reading at rest, so only the pools move these readings.
    if (engine.layers.len - engine.borrow_from < 3) return error.SkipZigTest;
    const last: u16 = @intCast(engine.layers.len - 1);
    _ = try engine.borrowLayer(last - 2);
    defer engine.restoreLayer(last - 2) catch {};
    const rest = read(config, &qwen);
    // A borrower that ended before any decode-width forward (cancelled, failed, its reply done at its prefill) leaves
    // its pools released with no lessee: the next decode-width forward reads them back whole.
    var lent = try engine.borrowLayer(last);
    lent += engine.borrowLayer(last - 1) catch |e| {
        try engine.restoreLayer(last);
        return e;
    };
    const now = read(config, &qwen);
    try engine.restoreLayer(last - 1);
    try engine.restoreLayer(last);
    errdefer std.debug.print("cooler contract on {s}: {d} B lent, union {d} B; at rest {any}, lent {any}\n", .{ if (cooler) "Sushi Cooler" else "MLX", lent, engine.unionBytes(), rest, now });
    try testing.expect(lent > 0);
    // Held off once: a reading that credits no free buffer falls by those the release parked, none by the pools again.
    for ([_][2]u64{ .{ rest.admission, now.admission }, .{ rest.beside, now.beside }, .{ rest.forward, now.forward }, .{ rest.load, now.load } }) |r| {
        try testing.expect(r[1] <= r[0]);
        try testing.expect(r[0] - r[1] < lent / 2);
    }
    // The read-back's room: the lessee's own reading counts what it takes back.
    try testing.expect(now.own > rest.own);
}

/// The DeepSeek STQ tiny on Sushi Cooler or MLX after one served request (a Cooler load's free buffers parked), at a
/// free RAM that never binds: only what the load holds moves the readings.
const LentLease = struct {
    ld: Load,
    sv: Served,
    avail_was: ?u64,
    engine: *@import("expert_stream.zig").Engine,

    fn open(self: *LentLease, dir: []const u8, cooler: bool) !void {
        const D = ContractOf("dsv41");
        const a = testing.allocator;
        const ids = try D.prompt(a, dir);
        defer a.free(ids);
        try self.ld.open(D, dir, cooler);
        errdefer _ = self.ld.close();
        const status = @import("status.zig");
        self.avail_was = status.available_mem_override;
        status.available_mem_override = 1 << 40;
        errdefer status.available_mem_override = self.avail_was;
        try self.sv.init(&self.ld, dir);
        errdefer self.sv.deinit();
        var prompt: [40]u32 = undefined;
        for (&prompt, 0..) |*t, i| t.* = ids[(i * 5 + 2) % ids.len];
        var f: u32 = 0;
        var said = try self.sv.request(&prompt, 8, &f);
        said.deinit(a);
        self.engine = arch_handle.trunk(self.ld.h).expert_stream.?;
    }

    fn close(self: *LentLease) void {
        self.sv.deinit();
        @import("status.zig").available_mem_override = self.avail_was;
        _ = self.ld.close();
    }
};

/// Admission's reading of what a pool's release frees, on either engine: the engine's free buffers, which a Cooler
/// admission credits, less the active bytes, MLX's and the engine's.
fn admissionReading() i64 {
    const gpu_mem = @import("gpu_mem.zig");
    return @as(i64, @intCast(gpu_mem.engineFree())) - @as(i64, @intCast(gpu_mem.activeBytes()));
}

/// One step of a pool's lease on the two last decoder layers, run from the state it starts in, and what every exit of
/// it must leave: the lent total the pools left unmapped, admission no higher than at rest.
const LentSweep = struct {
    engine: *@import("expert_stream.zig").Engine,
    config: *const model_mod.ModelConfig,
    /// The last layer, whose steps are swept, and the one before, released beside it.
    layer: [2]u16,
    /// Each pool as its release moves the reading, measured with the union workspace already made.
    pool: [2]i64 = .{ 0, 0 },
    /// What making the union workspace takes from the reading, measured on the release that makes it.
    union_read: i64 = 0,
    rest: u64 = 0,
    /// Admission read at a stranded layer, before the heal makes its workspace.
    stranded: u64 = 0,
    step: Step = .release,
    /// A host allocation index that strands the last layer: its workspace fails and its take-back too.
    strands_at: ?usize = null,

    const Step = enum { release, release_beside, read_back, heal };

    fn run(self: *LentSweep) anyerror!void {
        const e = self.engine;
        switch (self.step) {
            .release => _ = try e.borrowLayer(self.layer[0]),
            .release_beside => _ = try e.borrowLayer(self.layer[1]),
            .read_back => try e.restoreLayer(self.layer[0]),
            .heal => try e.healReleased(),
        }
    }

    /// The state the step starts from: at rest, the last layer released, or released with no union workspace.
    fn start(self: *LentSweep) !void {
        try self.toRest();
        switch (self.step) {
            .release => {},
            .release_beside, .read_back => _ = try self.engine.borrowLayer(self.layer[0]),
            .heal => {
                try self.strand();
                self.stranded = @import("bills.zig").admissionAvailableNow(self.config);
            },
        }
    }

    fn strand(self: *LentSweep) !void {
        const e = self.engine;
        const gpa = e.allocator;
        defer e.allocator = gpa;
        var k: usize = self.strands_at orelse 0;
        while (true) : (k += 1) {
            if (k > 10_000) return error.NeverStranded;
            try self.toRest();
            var failing = testing.FailingAllocator.init(gpa, .{ .fail_index = k });
            e.allocator = failing.allocator();
            _ = e.borrowLayer(self.layer[0]) catch 0;
            e.allocator = gpa;
            if (e.isBorrowed(self.layer[0]) and e.grouped_union.len == 0) break;
        }
        self.strands_at = k;
    }

    fn toRest(self: *LentSweep) !void {
        const e = self.engine;
        try e.healReleased();
        for (self.layer) |li| try e.restoreLayer(li);
        try testing.expectEqual(@as(u64, 0), @import("gpu_mem.zig").lentBytes());
    }

    /// The lent total is the bytes the released pools left unmapped, measured, and at a stranded layer at least those
    /// and the workspace its heal makes; admission reads no more than at rest, and what it read at a stranded layer
    /// still fits once the heal has made the workspace.
    fn check(self: *LentSweep, refused: []const u8, k: u64) !void {
        const e = self.engine;
        var unmapped: i64 = 0;
        for (self.layer, self.pool) |li, bytes| {
            if (e.isBorrowed(li)) unmapped += bytes;
        }
        const stranded = unmapped > 0 and e.grouped_union.len == 0;
        const lent: i64 = @intCast(@import("gpu_mem.zig").lentBytes());
        const now = @import("bills.zig").admissionAvailableNow(self.config);
        errdefer std.debug.print("{t} with {s} {d} refused: released {any}, union {}; lent {d} B, pools unmapped {d} B, union {d} B; admission at rest {d}, stranded {d}, now {d}\n", .{
            self.step, refused, k, .{ e.isBorrowed(self.layer[0]), e.isBorrowed(self.layer[1]) }, e.grouped_union.len != 0, lent, unmapped, self.union_read, self.rest, self.stranded, now,
        });
        // Never under. Over by at most a page, from one cause: MLX's active bytes are the process's, and MLX frees a
        // finished command buffer's temporaries on its completion thread, which can land inside a release's reading
        // after the synchronize before it (116 B once on MLX); it can only free, so it never reads under.
        if (stranded) {
            try testing.expect(lent >= unmapped + self.union_read);
        } else {
            try testing.expect(lent >= unmapped);
            try testing.expect(lent - unmapped <= 16 << 10);
        }
        try testing.expect(now <= self.rest);
        if (self.step == .heal) try testing.expect(now >= self.stranded);
    }

    /// Each site of `F` (`mlx.fault`, `mlx.engine.fault`) the step reaches refused once, at its first ask.
    fn sweepSites(self: *LentSweep, comptime F: type, name: []const u8) !void {
        const a = testing.allocator;
        try self.start();
        F.sites = .{ .lasts = false };
        defer F.sites = .{};
        const ks = try mlx.sweepPoints(a, &F.sites, self, run);
        defer a.free(ks);
        try self.check("nothing", 0);
        for (ks) |k| {
            try self.start();
            F.arm(k);
            self.run() catch {};
            const fired = if (comptime F == mlx.fault) F.didFire() else !F.armed();
            F.disarm();
            var msg: [300]u8 = undefined;
            _ = mlx.takeError(&msg);
            try testing.expect(fired);
            try self.check(name, k);
        }
    }

    /// Every host allocation from the k-th on refused, k from 0 until the step needs none: a refused step's take-back,
    /// or a heal's restore after its workspace, is refused too.
    fn sweepHost(self: *LentSweep) !void {
        const e = self.engine;
        const gpa = e.allocator;
        defer e.allocator = gpa;
        var k: usize = 0;
        while (true) : (k += 1) {
            try self.start();
            var failing = testing.FailingAllocator.init(gpa, .{ .fail_index = k });
            e.allocator = failing.allocator();
            const r = self.run();
            e.allocator = gpa;
            try self.check("every host allocation from", k);
            if (!failing.has_induced_failure) return r;
        }
    }
};

test "cooler contract: every exit of a DeepSeek pool's release, read-back and heal, one refused step or two included, and of an unload mid-lease, on Sushi Cooler and on MLX, publishes as lent what the pools left free in admission's reading and, at a stranded layer, the workspace its heal makes, and admission never reads more than at rest nor than a heal leaves" {
    try mlx.requireGpu();
    const D = ContractOf("dsv41");
    if (comptime !@hasDecl(D, "prompt")) return error.SkipZigTest;
    const was_enabled = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was_enabled);
    defer mlx.engine.fault.disarm();
    defer mlx.fault.disarm();
    const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    for ([_]bool{ true, false }) |cooler| try lentSweep(dir, cooler);
}

fn lentSweep(dir: []const u8, cooler: bool) !void {
    const gpu_mem = @import("gpu_mem.zig");
    var lease: LentLease = undefined;
    try lease.open(dir, cooler);
    var open = true;
    defer if (open) lease.close();
    const e = lease.engine;
    errdefer std.debug.print("on {s}\n", .{if (cooler) "Sushi Cooler" else "MLX"});
    if (e.layers.len - e.borrow_from < 2) return error.SkipZigTest;
    const last: u16 = @intCast(e.layers.len - 1);
    var sw: LentSweep = .{ .engine = e, .config = &lease.ld.config, .layer = .{ last, last - 1 } };
    // Each pool alone: released beside the other, which made the union workspace. The least of three: a free MLX
    // makes off its thread only adds to a reading (`check`).
    for (0..2) |i| {
        sw.pool[i] = std.math.maxInt(i64);
        for (0..3) |_| {
            _ = try e.borrowLayer(sw.layer[1 - i]);
            const before = admissionReading();
            _ = try e.borrowLayer(sw.layer[i]);
            sw.pool[i] = @min(sw.pool[i], admissionReading() - before);
            try sw.toRest();
        }
        try testing.expect(sw.pool[i] > 0);
    }
    // The first release makes the union workspace out of its pool.
    const before = admissionReading();
    _ = try e.borrowLayer(sw.layer[0]);
    sw.union_read = sw.pool[0] - (admissionReading() - before);
    try sw.toRest();
    try testing.expect(sw.union_read > 0);
    sw.rest = @import("bills.zig").admissionAvailableNow(sw.config);
    for (std.meta.tags(LentSweep.Step)) |step| {
        sw.step = step;
        try sw.sweepSites(mlx.engine.fault, "pool buffer");
        try sw.sweepSites(mlx.fault, "checked call");
        try sw.sweepHost();
    }
    try sw.toRest();
    // An unload mid-lease takes its engine's share out of the process's total.
    _ = try e.borrowLayer(sw.layer[0]);
    _ = try e.borrowLayer(sw.layer[1]);
    try sw.check("nothing", 0);
    open = false;
    lease.close();
    try testing.expectEqual(@as(u64, 0), gpu_mem.lentBytes());
}

test "cooler contract: a DeepSeek read-back holds the union workspace beside the pools it takes back, on Sushi Cooler and on MLX, and admission read during the lease never exceeds what is free through it" {
    try mlx.requireGpu();
    const D = ContractOf("dsv41");
    if (comptime !@hasDecl(D, "prompt")) return error.SkipZigTest;
    const was_enabled = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was_enabled);
    const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    for ([_]bool{ true, false }) |cooler| try readBackHoldsUnion(dir, cooler);
}

fn readBackHoldsUnion(dir: []const u8, cooler: bool) !void {
    const bills = @import("bills.zig");
    var lease: LentLease = undefined;
    try lease.open(dir, cooler);
    defer lease.close();
    const e = lease.engine;
    if (e.layers.len - e.borrow_from < 3) return error.SkipZigTest;
    const config = &lease.ld.config;
    const last: u16 = @intCast(e.layers.len - 1);
    // The first release makes the union workspace; the middle layer stays lent through the read-back of the other two,
    // so the read-back's whole window, the union held beside every pool back, can be read before the union drops.
    const order = [_]u16{ last, last - 1, last - 2 };
    defer for (order) |li| e.restoreLayer(li) catch {};
    for (order) |li| _ = try e.borrowLayer(li);
    const admitted = bills.admissionAvailableNow(config);
    for ([_]u16{ last, last - 2 }) |li| {
        try e.restoreLayer(li);
        const free = bills.admissionAvailableNow(config);
        errdefer std.debug.print("on {s}, read back layer {d}: admission during the lease {d}, through the read-back {d} (union {d} B)\n", .{ if (cooler) "Sushi Cooler" else "MLX", li, admitted, free, e.unionBytes() });
        try testing.expect(e.grouped_union.len != 0);
        try testing.expect(admitted <= free);
    }
}

test "cooler contract: a second DeepSeek model's own lessee reading counts only its own lent pools free, not the first model's, on Sushi Cooler and on MLX" {
    try mlx.requireGpu();
    const D = ContractOf("dsv41");
    if (comptime !@hasDecl(D, "prompt")) return error.SkipZigTest;
    const was_enabled = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was_enabled);
    const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    const dir2 = std.mem.span(std.c.getenv("DSV41_STQ_TINY_PM") orelse return error.SkipZigTest);
    // Each engine is read whatever the other said.
    var failed: ?anyerror = null;
    for ([_]bool{ true, false }) |cooler| twoLessees(dir, dir2, cooler) catch |e| {
        if (e == error.SkipZigTest) return e;
        failed = failed orelse e;
    };
    if (failed) |e| return e;
}

fn twoLessees(dir: []const u8, dir2: []const u8, cooler: bool) !void {
    const own = @import("dsv41/dsv41_bill.zig").dsv41Headroom;
    var lease: LentLease = undefined;
    try lease.open(dir, cooler);
    defer lease.close();
    var other: Load = undefined;
    try other.open(ContractOf("dsv41"), dir2, cooler);
    defer _ = other.close();
    const e = lease.engine;
    if (e.layers.len - e.borrow_from < 3) return error.SkipZigTest;
    const last: u16 = @intCast(e.layers.len - 1);
    // The union workspace made first, so only the pools move the readings.
    _ = try e.borrowLayer(last - 2);
    defer e.restoreLayer(last - 2) catch {};
    const rest = .{ own(&lease.ld.config).bound(), own(&other.config).bound() };
    var lent = try e.borrowLayer(last);
    lent += try e.borrowLayer(last - 1);
    const now = .{ own(&lease.ld.config).bound(), own(&other.config).bound() };
    try e.restoreLayer(last - 1);
    try e.restoreLayer(last);
    errdefer std.debug.print("on {s}: {d} B lent by the first model; its own reading {d} -> {d}, the second's {d} -> {d}\n", .{ if (cooler) "Sushi Cooler" else "MLX", lent, rest[0], now[0], rest[1], now[1] });
    try testing.expect(lent > 0);
    // The lessee's read-back takes its own pools back; another model's borrow or read-back would take them twice.
    try testing.expect(now[0] > rest[0]);
    try testing.expect(now[1] <= rest[1] + lent / 2);
}

/// One request prefilled under `LeaseProbe`, so its wide chunk released decoder pools, and decoding.
const LeasedDecode = struct {
    sv: Served,
    slot: *Served.scheduler.Slot,

    fn begin(self: *LeasedDecode, ld: *Load, dir: []const u8, prompt: []const u32) !void {
        const scheduler = Served.scheduler;
        try self.sv.init(ld, dir);
        errdefer self.sv.deinit();
        const sch = &self.sv.sch;
        self.slot = try sch.submit(.{ .model = self.sv.entry, .prompt_ids = prompt, .sampling = .{ .temperature = 0.0 }, .eos_token_ids = &.{}, .max_tokens = 8 });
        _ = sch.pending.orderedRemove(0);
        errdefer {
            sch.complete(self.slot);
            for (sch.cleanup_queue.items) |x| x.deinit();
            sch.cleanup_queue.clearRetainingCapacity();
        }
        try scheduler.runPrefill(sch, self.slot);
        if (!scheduler.postPrefillTerminal(sch, self.slot)) try sch.decoding.append(testing.allocator, self.slot);
    }

    fn tick(self: *LeasedDecode) void {
        const scheduler = Served.scheduler;
        var one = [_]*scheduler.Slot{self.slot};
        if (self.slot.joinsTick()) scheduler.runDecodeTick(&self.sv.sch, &one) catch |e| self.slot.markError(@errorName(e));
    }

    fn end(self: *LeasedDecode) void {
        const sch = &self.sv.sch;
        sch.complete(self.slot);
        for (sch.cleanup_queue.items) |s| s.deinit();
        sch.cleanup_queue.clearRetainingCapacity();
        self.sv.deinit();
    }
};

test "cooler contract: a DeepSeek decode whose read-back of a released pool is refused decodes through the union workspace and says the same tokens" {
    try mlx.requireGpu();
    const D = ContractOf("dsv41");
    if (comptime !@hasDecl(D, "prompt")) return error.SkipZigTest;
    const was_enabled = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was_enabled);
    defer mlx.engine.fault.disarm();
    const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const ids = try D.prompt(a, dir);
    defer a.free(ids);
    var ld: Load = undefined;
    try ld.open(D, dir, true);
    defer _ = ld.close();
    const borrow = @import("dsv41/dsv41_borrow.zig");
    const xfm = arch_handle.trunk(ld.h);
    const engine = xfm.expert_stream.?;
    const m = @import("dsv41/dsv41_state.zig").modelOf(xfm).?;
    // One 2048-row chunk and the held-back span: the chunk's need less half a pool is the free memory, so it releases a
    // pool, and the first decode step finds room to read it back.
    ld.config.pinned_prefill_chunk = 2048;
    var long: [2048]u32 = undefined;
    for (&long, 0..) |*t, i| t.* = ids[(i * 7 + i / ids.len) % ids.len];
    const wide = long.len - (prefill_plan.SSM_SNAPSHOT_BACKOFF + 1);
    const need = borrow.prefillNeed(&m.cfg, @import("cooler_side.zig").onEngine(&ld.config), wide, wide);
    LeaseProbe.engine = engine;
    defer LeaseProbe.engine = null;
    const at_prefill = need - engine.poolBytes(@intCast(engine.layers.len - 1)) / 2;
    // Once the prefill's working set is free again, the decode has its own cap and the union workspace beside the pools.
    const at_decode = @max(at_prefill, m.cfg.forwardCap(1, long.len + 1) + engine.unionBytes());
    LeaseProbe.base = at_prefill;
    borrow.headroom_fn = &LeaseProbe.headroom;
    defer borrow.headroom_fn = null;
    const Released = struct {
        fn count(e: *const @import("expert_stream.zig").Engine) usize {
            var n: usize = 0;
            for (e.borrow_from..e.layers.len) |li| n += @intFromBool(e.isBorrowed(@intCast(li)));
            return n;
        }
    };
    var want: std.ArrayList(u32) = .empty;
    defer want.deinit(a);
    {
        var r: LeasedDecode = undefined;
        try r.begin(&ld, dir, &long);
        defer r.end();
        try testing.expect(Released.count(engine) >= 2);
        LeaseProbe.base = at_decode;
        LeaseProbe.mark = true;
        LeaseProbe.marked = 0;
        r.tick();
        LeaseProbe.mark = false;
        try testing.expectEqual(@as(usize, 0), Released.count(engine));
        while (r.slot.joinsTick()) r.tick();
        try testing.expect(r.slot.error_code == null);
        try want.appendSlice(a, r.slot.legacy_gen.?.generated_ids.items);
    }
    // The buffers the first read-back asks, between the reads before it and before the second.
    try testing.expectEqual(@as(usize, 2), LeaseProbe.marked);
    const asks = LeaseProbe.marks[1] - LeaseProbe.marks[0];
    try testing.expect(asks > 0);
    // Each is refused in turn: the layer stays released with the union workspace, and the request decodes through it to
    // the same tokens.
    var k: u64 = 1;
    while (k <= asks) : (k += 1) {
        var r: LeasedDecode = undefined;
        LeaseProbe.base = at_prefill;
        try r.begin(&ld, dir, &long);
        defer r.end();
        LeaseProbe.base = at_decode;
        LeaseProbe.arm_at = k;
        defer LeaseProbe.arm_at = 0;
        r.tick();
        try testing.expectEqual(@as(u64, 0), LeaseProbe.arm_at);
        try testing.expect(!mlx.engine.fault.armed());
        mlx.engine.fault.disarm();
        while (r.slot.joinsTick()) r.tick();
        errdefer std.debug.print("ask {d} of the read-back's {d} refused: {?s}\n", .{ k, asks, r.slot.error_code });
        try testing.expect(r.slot.error_code == null);
        try testing.expectEqualSlices(u32, want.items, r.slot.legacy_gen.?.generated_ids.items);
    }
}

/// A box with room for every forward. At a request's first read, after its first forward installed its cap, it records
/// the room that cap leaves, the need it was installed for, and whether a stranded layer was healed by then.
const HealProbe = struct {
    var engine: ?*@import("expert_stream.zig").Engine = null;
    var read: bool = false;
    var room: u64 = 0;
    var need: u64 = 0;
    var healed: bool = false;

    fn headroom(_: *const model_mod.ModelConfig) @import("bills.zig").Headroom {
        if (!read) {
            read = true;
            room = (mlx.engine.poolLimit() orelse std.math.maxInt(u64)) -| (mlx.engine.poolTotal() - mlx.engine.poolFree());
            need = @import("dsv41/dsv41_borrow.zig").forward_need;
            healed = engine.?.grouped_union.len != 0;
        }
        return .{ .live = 1 << 40 };
    }
};

test "cooler contract: a DeepSeek layer left released with no union workspace is healed under the resting cap a request's end leaves, before the next forward installs its cap, which keeps the forward's whole bill" {
    try mlx.requireGpu();
    const D = ContractOf("dsv41");
    if (comptime !@hasDecl(D, "prompt")) return error.SkipZigTest;
    const was_enabled = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was_enabled);
    const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const ids = try D.prompt(a, dir);
    defer a.free(ids);
    var ld: Load = undefined;
    try ld.open(D, dir, true);
    defer _ = ld.close();
    const borrow = @import("dsv41/dsv41_borrow.zig");
    const engine = arch_handle.trunk(ld.h).expert_stream.?;
    var sv: Served = undefined;
    try sv.init(&ld, dir);
    defer sv.deinit();
    var prompt: [80]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = ids[(i * 5 + 2) % ids.len];
    var restored: u32 = 0;
    // A request's end leaves its resting cap in force: the cap the next request's heal runs under.
    var first = try sv.request(&prompt, 4, &restored);
    first.deinit(a);

    // Every host allocation from the swept one on fails, so a release whose workspace fails cannot take its pool back.
    const last: u16 = @intCast(engine.layers.len - 1);
    const gpa = engine.allocator;
    defer engine.allocator = gpa;
    var fail_index: usize = 0;
    while (!engine.isBorrowed(last) or engine.grouped_union.len != 0) : (fail_index += 1) {
        if (engine.isBorrowed(last)) try engine.restoreLayer(last);
        var failing = testing.FailingAllocator.init(gpa, .{ .fail_index = fail_index });
        engine.allocator = failing.allocator();
        _ = engine.borrowLayer(last) catch 0;
        engine.allocator = gpa;
        try testing.expect(fail_index < 10_000);
    }

    HealProbe.engine = engine;
    HealProbe.read = false;
    defer HealProbe.engine = null;
    borrow.headroom_fn = &HealProbe.headroom;
    defer borrow.headroom_fn = null;
    prompt[0] = ids[1];
    var second = try sv.request(&prompt, 4, &restored);
    second.deinit(a);
    try testing.expect(HealProbe.read and HealProbe.healed);
    try testing.expect(!engine.isBorrowed(last));
    // The heal's lasting buffers are in use before the forward's cap goes in, so the cap's room is its need, whole.
    try testing.expectEqual(HealProbe.need, HealProbe.room);
}

/// `LeaseProbe` that also records, at a decode forward's reads (one before each read-back), the pool's bytes in use and
/// the room its cap leaves.
const ReadBackProbe = struct {
    var on: bool = false;
    var reads: usize = 0;
    var in_use: [64]u64 = @splat(0);
    var room: u64 = 0;

    fn headroom(c: *const model_mod.ModelConfig) @import("bills.zig").Headroom {
        if (on and reads < in_use.len) {
            const used = mlx.engine.poolTotal() - mlx.engine.poolFree();
            if (reads == 0) room = (mlx.engine.poolLimit() orelse std.math.maxInt(u64)) -| used;
            in_use[reads] = used;
            reads += 1;
        }
        return LeaseProbe.headroom(c);
    }
};

test "cooler contract: a DeepSeek decode forward reads back every pool its prefill released under its own cap, each read-back keeping only its bank's meta table and resident map in the engine's pool" {
    try mlx.requireGpu();
    const D = ContractOf("dsv41");
    if (comptime !@hasDecl(D, "prompt")) return error.SkipZigTest;
    const was_enabled = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was_enabled);
    const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const ids = try D.prompt(a, dir);
    defer a.free(ids);
    var ld: Load = undefined;
    try ld.open(D, dir, true);
    defer _ = ld.close();
    const borrow = @import("dsv41/dsv41_borrow.zig");
    const pool_class = @import("tensor/pool_class.zig");
    const xfm = arch_handle.trunk(ld.h);
    const engine = xfm.expert_stream.?;
    const m = @import("dsv41/dsv41_state.zig").modelOf(xfm).?;
    ld.config.pinned_prefill_chunk = 2048;
    var long: [2048]u32 = undefined;
    for (&long, 0..) |*t, i| t.* = ids[(i * 7 + i / ids.len) % ids.len];
    const wide = long.len - (prefill_plan.SSM_SNAPSHOT_BACKOFF + 1);
    const need = borrow.prefillNeed(&m.cfg, @import("cooler_side.zig").onEngine(&ld.config), wide, wide);
    LeaseProbe.engine = engine;
    defer LeaseProbe.engine = null;
    LeaseProbe.base = need - engine.poolBytes(@intCast(engine.layers.len - 1)) / 2;
    borrow.headroom_fn = &ReadBackProbe.headroom;
    defer borrow.headroom_fn = null;
    var r: LeasedDecode = undefined;
    try r.begin(&ld, dir, &long);
    defer r.end();
    // Once the prefill's working set is free again, the decode has its own cap and the union workspace beside the pools.
    LeaseProbe.base = @max(LeaseProbe.base, m.cfg.forwardCap(1, long.len + 1) + engine.unionBytes());
    var released: std.ArrayList(u16) = .empty;
    defer released.deinit(a);
    for (engine.borrow_from..engine.layers.len) |li| if (engine.isBorrowed(@intCast(li))) try released.append(a, @intCast(li));
    try testing.expect(released.items.len >= 2);
    ReadBackProbe.on = true;
    ReadBackProbe.reads = 0;
    r.tick();
    ReadBackProbe.on = false;
    try testing.expect(r.slot.error_code == null);
    for (released.items) |li| try testing.expect(!engine.isBorrowed(li));
    try testing.expectEqual(released.items.len, ReadBackProbe.reads);
    // The read-backs ran under the decode forward's cap: a one-row forward ending a token past the prompt.
    try testing.expectEqual(m.cfg.forwardCap(1, long.len + 1), ReadBackProbe.room);
    // Read-back i lies between reads i and i + 1; the last one also drops the union workspace, so it is not counted.
    // Its pool's slabs are host pages the engine wraps, outside the cap: only the meta table and the map stay in use.
    for (released.items[0 .. released.items.len - 1], 0..) |li, i| {
        const layer = &engine.layers[li];
        const meta = layer.grouped.?.cache_ragged.meta;
        const map = layer.resident_map;
        var kept = pool_class.poolBytes(mlx.mlx_array_size(meta) * mlx.mlx_array_itemsize(meta));
        if (map.ctx != null) kept += pool_class.poolBytes(mlx.mlx_array_size(map) * mlx.mlx_array_itemsize(map));
        const grew = ReadBackProbe.in_use[i + 1] -| ReadBackProbe.in_use[i];
        errdefer std.debug.print("layer {d}: the read-back kept {d} B in the engine's pool; its meta table and map take {d} B\n", .{ li, grew, kept });
        try testing.expect(grew <= kept);
    }
}

test "cooler contract: a DeepSeek request admitted while an earlier borrower's pools stay lent has its margin moved by the pools once: held off, and not taken out of the lendable too" {
    try mlx.requireGpu();
    const D = ContractOf("dsv41");
    if (comptime !@hasDecl(D, "prompt")) return error.SkipZigTest;
    const was_enabled = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was_enabled);
    const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    const bills = @import("bills.zig");
    const borrow = @import("dsv41/dsv41_borrow.zig");
    var lease: LentLease = undefined;
    try lease.open(dir, true);
    defer lease.close();
    const engine = lease.engine;
    const config = &lease.ld.config;
    const m = @import("dsv41/dsv41_state.zig").modelOf(arch_handle.trunk(lease.ld.h)).?;
    const ceil_was = bills.static_ceiling_override;
    defer bills.static_ceiling_override = ceil_was;
    bills.static_ceiling_override = 256 << 30;
    if (engine.layers.len - engine.borrow_from < 3) return error.SkipZigTest;
    const last: u16 = @intCast(engine.layers.len - 1);
    // The union workspace made before the reading at rest, as `lentHeldOff` does: only the pools move the readings.
    _ = try engine.borrowLayer(last - 2);
    defer engine.restoreLayer(last - 2) catch {};
    // A prompt whose prefill working set passes what the pools lend at rest, so the prefill term binds the bill.
    const on = @import("cooler_side.zig").onEngine(config);
    var seq: u64 = 2048;
    var chunk: u64 = 2048;
    const mt: u32 = 8;
    const borrowable_rest = bills.borrowableBytes(config);
    while (m.cfg.forwardCapOn(on, bills.requestForwardRows(config, seq, chunk, 0), seq) -| borrowable_rest <= m.cfg.forwardCap(borrow.verify_rows, seq + mt + borrow.verify_rows) + (64 << 20)) {
        if (seq >= 1 << 20) return error.SkipZigTest;
        seq *= 2;
        chunk = @min(seq, 8192);
    }
    const need_rest = bills.prefillNeededAtChunk(config, seq, mt, 8, chunk, .{});
    const avail_rest = bills.admissionAvailableNow(config);
    const own_rest = @import("dsv41/dsv41_bill.zig").dsv41Headroom(config).bound();
    var lent = try engine.borrowLayer(last);
    lent += engine.borrowLayer(last - 1) catch |e| {
        try engine.restoreLayer(last);
        return e;
    };
    const need_lent = bills.prefillNeededAtChunk(config, seq, mt, 8, chunk, .{});
    const avail_lent = bills.admissionAvailableNow(config);
    const own_lent = @import("dsv41/dsv41_bill.zig").dsv41Headroom(config).bound();
    const borrowable_lent = bills.borrowableBytes(config);
    try engine.restoreLayer(last - 1);
    try engine.restoreLayer(last);
    const margin_rest: i64 = @as(i64, @intCast(avail_rest)) - @as(i64, @intCast(need_rest));
    const margin_lent: i64 = @as(i64, @intCast(avail_lent)) - @as(i64, @intCast(need_lent));
    errdefer std.debug.print("double hold: seq {d} chunk {d}: lent {d} B; borrowable {d} -> {d}; need {d} -> {d} (+{d}); available {d} -> {d} (-{d}); margin {d} -> {d} (moved {d}); lessee's own headroom {d} -> {d}\n", .{
        seq, chunk, lent, borrowable_rest, borrowable_lent, need_rest, need_lent, need_lent -| need_rest, avail_rest, avail_lent, avail_rest -| avail_lent, margin_rest, margin_lent, margin_rest - margin_lent, own_rest, own_lent,
    });
    try testing.expect(lent > 0);
    // The verdict on a box whose free RAM binds the live ceiling: the request fits at rest by half the lent bytes; with
    // the pools lent, the RAM they left is free again (the reclaimable pages rise by what the footprint fell).
    {
        const gm = @import("gpu_mem.zig");
        const status = @import("status.zig");
        const avail_was = status.available_mem_override;
        defer status.available_mem_override = avail_was;
        const r0: u64 = 4 << 30;
        status.available_mem_override = r0;
        const a0 = bills.admissionAvailableNow(config);
        const n0 = bills.prefillNeededAtChunk(config, seq, mt, 8, chunk, .{});
        const fp_rest = gm.activeBytes() + gm.cacheBytes();
        const r_rest: u64 = r0 + n0 + lent / 2 - a0;
        status.available_mem_override = r_rest;
        const fits_rest = bills.prefillNeededAtChunk(config, seq, mt, 8, chunk, .{}) <= bills.admissionAvailableNow(config);
        const at_rest = bills.admissionAvailableNow(config);
        _ = try engine.borrowLayer(last);
        _ = try engine.borrowLayer(last - 1);
        const fp_lent = gm.activeBytes() + gm.cacheBytes();
        status.available_mem_override = r_rest + (fp_rest -| fp_lent);
        const need_v = bills.prefillNeededAtChunk(config, seq, mt, 8, chunk, .{});
        const at_lent = bills.admissionAvailableNow(config);
        const fits_lent = need_v <= at_lent;
        const lessee_room = @import("dsv41/dsv41_bill.zig").dsv41Headroom(config).bound();
        try engine.restoreLayer(last - 1);
        try engine.restoreLayer(last);
        errdefer std.debug.print("double hold verdict: footprint {d} -> {d}; at rest available {d}, need {d}, fits {}; with the pools lent available {d}, need {d}, fits {}; the lessee's own headroom then {d} B (its prefill working set {d} B, the lendable left {d} B)\n", .{ fp_rest, fp_lent, at_rest, n0, fits_rest, at_lent, need_v, fits_lent, lessee_room, m.cfg.forwardCapOn(on, bills.requestForwardRows(config, seq, chunk, 0), seq), borrowable_lent });
        try testing.expect(fits_rest);
        try testing.expect(fits_lent);
    }
    // Held off once (`lentHeldOff`: "a reading that credits no free buffer falls by those the release parked, none by
    // the pools again"): the request's prefill runs over the released pools as their lessee (`leaseHolderHeadroomNow`
    // counts them free), so its verdict may not move by the pools a second time through the lendable term.
    try testing.expect(margin_rest - margin_lent < @as(i64, @intCast(lent / 2)));
}

test "cooler contract: a DeepSeek prefill's chunk width holds where it held before its own pools were left lent: the pools move its verdict once" {
    try mlx.requireGpu();
    const D = ContractOf("dsv41");
    if (comptime !@hasDecl(D, "prompt")) return error.SkipZigTest;
    const was_enabled = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was_enabled);
    const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    const bills = @import("bills.zig");
    var lease: LentLease = undefined;
    try lease.open(dir, true);
    defer lease.close();
    const engine = lease.engine;
    const config = &lease.ld.config;
    // The working set is the ceiling the width reads, whatever wired limit the box declares.
    const ceil_was = bills.static_ceiling_override;
    const wired_was = bills.wired_limit_mb_override;
    defer bills.static_ceiling_override = ceil_was;
    defer bills.wired_limit_mb_override = wired_was;
    bills.wired_limit_mb_override = 0;
    if (engine.layers.len - engine.borrow_from < 3) return error.SkipZigTest;
    try testing.expect(bills.adaptivePrefillChunkEnabled(config));
    const last: u16 = @intCast(engine.layers.len - 1);
    _ = try engine.borrowLayer(last - 2);
    defer engine.restoreLayer(last - 2) catch {};
    const chunk: u32 = 2048;
    const Pick = struct {
        fn at(c: *const model_mod.ModelConfig, ceiling: u64) u32 {
            @import("bills.zig").static_ceiling_override = ceiling;
            var st: @import("mtp_plan.zig").AdaptiveWidthState = .{};
            return @import("bills.zig").adaptivePrefillWidthNow(c, 8, 0, chunk, chunk, &st, 0);
        }
    };
    // The ceiling at which the width just holds at rest: the pools still to release are what its chunk takes from.
    var lo: u64 = 0;
    var hi: u64 = 256 << 30;
    try testing.expectEqual(chunk, Pick.at(config, hi));
    try testing.expect(Pick.at(config, lo) < chunk);
    while (hi - lo > 1 << 10) {
        const mid = lo + (hi - lo) / 2;
        if (Pick.at(config, mid) == chunk) hi = mid else lo = mid;
    }
    const edge = hi;
    const borrowable_rest = bills.borrowableBytes(config);
    var lent = try engine.borrowLayer(last);
    lent += engine.borrowLayer(last - 1) catch |e| {
        try engine.restoreLayer(last);
        return e;
    };
    // Lent pools are held off in the reading and netted out of the cost: the ceiling that held the width holds it still.
    const picked = Pick.at(config, edge + lent / 4);
    const borrowable_lent = bills.borrowableBytes(config);
    try engine.restoreLayer(last - 1);
    try engine.restoreLayer(last);
    errdefer std.debug.print("lent width: {d} B lent, borrowable {d} -> {d}; the width held at {d} at rest under a {d} B ceiling, with the pools lent under {d} B: {d}\n", .{ lent, borrowable_rest, borrowable_lent, chunk, edge, edge + lent / 4, picked });
    try testing.expect(lent > 0);
    try testing.expectEqual(chunk, picked);
}

/// A request whose prompt reaches `generate` as one token: a full hot-cache match leaves `full_prompt[hot_matched..]`
/// one token long, and its cache must be bound at the request's end like any other.
fn oneTokenRequestHeld(comptime Q: type, cooler: bool) !void {
    try mlx.requireGpu();
    const was_enabled = mlx.engine.enabled;
    mlx.engine.enabled = cooler;
    defer test_support.restoreEngine(was_enabled);
    const a = testing.allocator;
    const dir = std.mem.span(std.c.getenv("QWEN4_STQ_TINY") orelse return error.SkipZigTest);
    const ids = try Q.prompt(a, dir);
    defer a.free(ids);
    var ld: Load = undefined;
    try ld.open(Q, dir, cooler);
    defer _ = ld.close();
    const bills = @import("bills.zig");
    const c = &ld.config;
    const kv = bills.configuredKvQuantFor(c);
    const kv_bits: u64 = if (kv.scheme == .off) 16 else kv.bits;
    const chunk: u64 = c.prefillStartWidth();
    const n: u32 = 1799;
    const one = [_]u32{ids[0]};
    const turn = arch_handle.turnFor(ld.h, c);
    defer turn.end();
    defer arch_sched.endRequest(c);
    var tok = Tokenizer.initEmpty(a);
    defer tok.deinit();
    try ld.xfm.resetCache();
    var gen = try Generator.initWithOptions(io, a, ld.xfm, &tok, &one, n, .{ .temperature = 0.0 }, &.{}, .{});
    defer gen.deinit(a);
    var steps: usize = 0;
    while (try gen.next(a)) |_| steps += 1;
    var held: usize = 0;
    for (ld.xfm.cache.entries) |entry| if (entry.initialized) {
        held = @max(held, @import("kv_cache.zig").KVCache.bufferCapacity(entry.keys));
    };
    const st = bills.cacheStorage(c, 1, n, kv_bits, chunk, .{});
    if (held > st.rows) std.debug.print("cooler contract: cooler {}: a 1-token prompt, {d} decoded, end_tokens {d}: the cache holds {d} rows, the bill prices {d}\n", .{ cooler, steps, ld.xfm.cache.end_tokens, held, st.rows });
    try testing.expect(ld.xfm.cache.end_tokens >= 1 + n);
    try testing.expect(held <= st.rows);
}

test "cooler contract: a Qwen request reaching generate as one token holds no more cache rows than its bill, on Cooler and on MLX" {
    const Q = ContractOf("qwen4");
    if (comptime !@hasDecl(Q, "prompt")) return error.SkipZigTest;
    try oneTokenRequestHeld(Q, true);
    try oneTokenRequestHeld(Q, false);
}

/// Writes every attention layer of the Flash-Next cache on the engine from its stored rows to `rows`, kv8.
fn fillKvCache(cfg: *const model_mod.ModelConfig, cache: *@import("kv_cache.zig").KVCache, rows: usize) !void {
    const s = mlx.engine.stream();
    var li: u32 = 0;
    while (li < cfg.num_hidden_layers) : (li += 1) {
        if (cfg.isLinearLayer(li)) continue;
        const add: c_int = @intCast(rows - cache.seqLen(li));
        if (add == 0) continue;
        var k = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(k);
        var v = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(v);
        try mlx.check(mlx.mlx_ones(&k, &[_]c_int{ 1, @intCast(cfg.layerKVHeads(li)), add, @intCast(cfg.layerHeadDim(li)) }, 4, .bfloat16, s));
        try mlx.check(mlx.mlx_ones(&v, &[_]c_int{ 1, @intCast(cfg.layerKVHeads(li)), add, @intCast(cfg.layerVHeadDim(li)) }, 4, .bfloat16, s));
        var view = try cache.update(li, k, v, s, 0);
        view.deinit();
    }
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    cache.appendEvalArrays(vec);
    try mlx.check(mlx.mlx_eval(vec));
}

/// A Flash-Next cache on the engine holding `rows` rows of every attention layer at kv8, bound as the engine binds the
/// request of `seq` tokens and `max_tokens` reply (`generate.reserveRequestCapacity`).
fn heldKvCache(cfg: *const model_mod.ModelConfig, rows: usize, seq: u64, max_tokens: u64) !@import("kv_cache.zig").KVCache {
    var cache = try @import("kv_cache.zig").KVCache.initWithConfig(testing.allocator, cfg.num_hidden_layers, @import("kv_quant.zig").KVQuantConfig.affine(8));
    errdefer cache.deinit();
    _ = try @import("generate.zig").reserveRequestCapacity(&cache, cfg, seq, max_tokens, cfg.prefillStartWidth(), 0, mlx.engine.stream());
    try fillKvCache(cfg, &cache, rows);
    return cache;
}

test "cooler contract: the dearest decode regrow of a request holds beside the grown cache no more than the request's bill carries beyond it, at every prompt and reply" {
    const KV = @import("kv_cache.zig").KVCache;
    const bills = @import("bills.zig");
    if (!bills.archBuilt("qwen4")) return error.SkipZigTest;
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was);
    const pool_class = @import("tensor/pool_class.zig");
    const cfg = @import("qwen4/qwen4_bill_test.zig").coolerFlashNext();
    const chunk: u64 = cfg.prefillStartWidth();
    const s = mlx.engine.stream();
    var over: usize = 0;
    // A reply that ends at the first capacity step and one that outlives several, below the long-prompt reservation
    // (`RESERVE_MIN_TOKENS`), which holds the whole reply.
    for ([_]u64{ 4096, 8192 }) |seq| for ([_]u64{ 1, 4096 }) |max_tokens| {
        const terms = bills.prefillRequestTerms(&cfg, seq, max_tokens, 8, chunk, .{});
        var cache = try heldKvCache(&cfg, seq, seq, max_tokens);
        defer cache.deinit();
        // The request's decode: every row past a capacity step regrows the cache, as `growAhead` does ahead of the forward.
        var offset: usize = seq;
        var dearest: u64 = 0;
        var later_most: u64 = 0;
        while (offset < seq + max_tokens) {
            mlx.engine.close();
            mlx.engine.trim();
            const old_rows = KV.bufferCapacity(cache.entries[3].keys);
            mlx.engine.resetPoolPeak(.process);
            try cache.growAhead(1, s);
            mlx.engine.close();
            const peak = mlx.engine.poolPeak(.process);
            mlx.engine.trim();
            // What the grown cache settles at is the request's storage at its last row (`cacheStorage`); the regrow's peak
            // is what it holds past that, which the bill carries as `decode_regrow_bytes`.
            dearest = @max(dearest, peak -| (mlx.engine.poolTotal() - mlx.engine.poolFree()));
            // The peak is under the bill's whole old copy and tail by the layer's four small buffers a rebuild has not yet grown.
            const last = &cache.entries[cfg.num_hidden_layers - 1];
            var later: u64 = 0;
            for ([_]mlx.mlx_array{ last.keys_scales, last.keys_biases, last.values_scales, last.values_biases }) |buf| {
                const row = mlx.mlx_array_size(buf) * mlx.mlx_array_itemsize(buf) / KV.bufferCapacity(buf);
                later += pool_class.poolBytes(KV.bufferCapacity(buf) * row) - pool_class.poolBytes(old_rows * row);
            }
            later_most = @max(later_most, later);
            offset = @min(KV.bufferCapacity(cache.entries[3].keys), seq + max_tokens);
            try fillKvCache(&cfg, &cache, offset);
        }
        const billed = terms.grow_coexist_bytes + terms.decode_regrow_bytes;
        if (dearest > billed or billed > dearest + later_most + billed / 16) {
            over += 1;
            std.debug.print("cooler contract: a {d}-token prompt + {d}: the dearest regrow holds {d} B past the grown buffers, billed {d} B\n", .{ seq, max_tokens, dearest, billed });
        }
    };
    try testing.expectEqual(@as(usize, 0), over);
}

/// The MTP head's one-layer cache on the engine (its own scheme, `qwen4MtpHeadKvConfig`) holding `rows` rows, bound as the
/// engine binds the head for the request of `seq` tokens and `max_tokens` reply (`DraftCache.reserve`).
fn heldHeadCache(cfg: *const model_mod.ModelConfig, rows: usize, seq: u64, max_tokens: u64) !@import("kv_cache.zig").KVCache {
    const KV = @import("kv_cache.zig");
    const bills = @import("bills.zig");
    var cache = try KV.KVCache.initWithConfig(testing.allocator, 1, @import("qwen4/qwen4_forward.zig").qwen4MtpHeadKvConfig(bills.configuredKvQuantFor(cfg)));
    errdefer cache.deinit();
    cache.reserve(@intCast(@import("prefill_plan.zig").reservedPrefillTokens(cfg, seq, max_tokens, cfg.prefillStartWidth())));
    cache.bound(@intCast(KV.KVCache.requestEnd(seq, max_tokens, bills.getEffectiveContextLength(cfg), bills.speculativeRowsPastEnd(cfg, .{ .mtp_on = true }))));
    try fillHeadCache(cfg, &cache, rows);
    return cache;
}

/// Writes the head's layer from its stored rows to `rows`; the write grows the cache inside the graph, as the head's forward does.
fn fillHeadCache(cfg: *const model_mod.ModelConfig, cache: *@import("kv_cache.zig").KVCache, rows: usize) !void {
    const s = mlx.engine.stream();
    const add: c_int = @intCast(rows - cache.seqLen(0));
    if (add == 0) return;
    var k = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(k);
    var v = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(v);
    try mlx.check(mlx.mlx_ones(&k, &[_]c_int{ 1, @intCast(cfg.layerKVHeads(3)), add, @intCast(cfg.layerHeadDim(3)) }, 4, .bfloat16, s));
    try mlx.check(mlx.mlx_ones(&v, &[_]c_int{ 1, @intCast(cfg.layerKVHeads(3)), add, @intCast(cfg.layerVHeadDim(3)) }, 4, .bfloat16, s));
    var view = try cache.update(0, k, v, s, 0);
    view.deinit();
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    cache.appendEvalArrays(vec);
    try mlx.check(mlx.mlx_eval(vec));
}

test "cooler contract: the dearest decode regrow of the MTP head's cache holds beside the grown cache no more than the request's bill carries beyond it, at every prompt and reply" {
    const KV = @import("kv_cache.zig").KVCache;
    const bills = @import("bills.zig");
    if (!bills.archBuilt("qwen4")) return error.SkipZigTest;
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was);
    const cfg = @import("qwen4/qwen4_bill_test.zig").coolerFlashNext();
    // The head's own scheme is dense unless the kv-quant lever is on: its buffers are K and V alone.
    if (@import("qwen4/qwen4_forward.zig").qwen4MtpHeadKvConfig(bills.configuredKvQuantFor(&cfg)).scheme != .off) return error.SkipZigTest;
    const chunk: u64 = cfg.prefillStartWidth();
    var over: usize = 0;
    for ([_]u64{ 4096, 8192 }) |seq| for ([_]u64{ 1, 4096 }) |max_tokens| {
        const terms = bills.prefillRequestTerms(&cfg, seq, max_tokens, 8, chunk, .{ .mtp_on = true });
        var cache = try heldHeadCache(&cfg, seq, seq, max_tokens);
        defer cache.deinit();
        const end = seq + max_tokens + bills.speculativeRowsPastEnd(&cfg, .{ .mtp_on = true });
        // The head's decode: a row past a capacity step regrows its cache inside the forward's graph.
        var dearest: u64 = 0;
        var offset: usize = seq;
        while (offset < end) {
            mlx.engine.close();
            mlx.engine.trim();
            mlx.engine.resetPoolPeak(.process);
            try fillHeadCache(&cfg, &cache, offset + 1);
            mlx.engine.close();
            const peak = mlx.engine.poolPeak(.process);
            mlx.engine.trim();
            dearest = @max(dearest, peak -| (mlx.engine.poolTotal() - mlx.engine.poolFree()));
            offset = @min(KV.bufferCapacity(cache.entries[0].keys), end);
            try fillHeadCache(&cfg, &cache, offset);
        }
        const billed = terms.grow_coexist_bytes + terms.decode_regrow_bytes;
        // The peak is the cache's old copies and zero tails whole, which the bill prices at a sixteenth over their size classes.
        if (dearest > billed or billed > dearest + billed / 16) {
            over += 1;
            std.debug.print("cooler contract: a {d}-token prompt + {d}: the head's dearest regrow holds {d} B past its grown cache, billed {d} B\n", .{ seq, max_tokens, dearest, billed });
        }
    };
    try testing.expectEqual(@as(usize, 0), over);
}

test "cooler contract: the dearest decode regrow of the QSA history holds beside the grown history no more than the request's bill carries beyond it, every layer's regrow in one graph" {
    const bills = @import("bills.zig");
    if (!bills.archBuilt("qwen4")) return error.SkipZigTest;
    try mlx.requireGpu();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was);
    // The unfused score bank is the history's dearest layout: its f32 columns grow beside the pooled keys.
    @import("qwen4/qwen4_qsa_select.zig").qsa_score_fused_override = false;
    defer @import("qwen4/qwen4_qsa_select.zig").qsa_score_fused_override = null;
    const trunk = @import("trunk.zig");
    const cfg = @import("qwen4/qwen4_bill_test.zig").coolerFlashNext();
    const chunk: u64 = cfg.prefillStartWidth();
    const s = mlx.engine.stream();
    const layers: usize = cfg.attnCacheLayerCount();
    const hd: c_int = @intCast(cfg.indexer_head_dim);
    const ratio: usize = cfg.indexer_compress_ratio;
    var over: usize = 0;
    var regrows: usize = 0;
    for ([_]u64{ 4096, 8192 }) |seq| for ([_]u64{ 1, 4096 }) |max_tokens| {
        const terms = bills.prefillRequestTerms(&cfg, seq, max_tokens, 8, chunk, .{});
        const hold: trunk.Hold = .{
            .reserve = @import("prefill_plan.zig").reservedPrefillTokens(&cfg, seq, max_tokens, chunk),
            .end = trunk.KVCache.requestEnd(seq, max_tokens, bills.getEffectiveContextLength(&cfg), 0),
        };
        var pooled: [64]mlx.mlx_array = @splat(.{ .ctx = null });
        var pooled_view: [64]mlx.mlx_array = @splat(.{ .ctx = null });
        var bank: [64]mlx.mlx_array = @splat(.{ .ctx = null });
        var bank_view: [64]mlx.mlx_array = @splat(.{ .ctx = null });
        var pooled_count: [64]c_int = @splat(0);
        var bank_count: [64]c_int = @splat(0);
        defer for (0..layers) |i| {
            for ([_]mlx.mlx_array{ pooled[i], pooled_view[i], bank[i], bank_view[i] }) |a| _ = mlx.mlx_array_free(a);
        };
        // Every layer's history, grown to the prompt's blocks and then a block at a time (a block per `ratio` tokens).
        var blocks: usize = 0;
        var dearest: u64 = 0;
        const end_blocks = (hold.end + ratio - 1) / ratio;
        var target: usize = (seq + ratio - 1) / ratio;
        while (blocks < end_blocks) {
            const add: c_int = @intCast(target - blocks);
            mlx.engine.close();
            mlx.engine.trim();
            mlx.engine.resetPoolPeak(.process);
            const vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(vec);
            var chunks: [2]mlx.mlx_array = .{ mlx.mlx_array_new(), mlx.mlx_array_new() };
            defer for (chunks) |c| {
                _ = mlx.mlx_array_free(c);
            };
            try mlx.check(mlx.mlx_ones(&chunks[0], &[_]c_int{ 1, add, hd }, 3, .bfloat16, s));
            try mlx.check(mlx.mlx_ones(&chunks[1], &[_]c_int{ 1, 1, hd, add }, 4, .float32, s));
            const before = mlx.engine.poolTotal() - mlx.engine.poolFree();
            const caps: usize = if (pooled[0].ctx != null) @intCast(mlx.getShape(pooled[0])[1]) else 0;
            for (0..layers) |i| {
                try trunk.capBufAppend(s, &pooled[i], &pooled_view[i], &pooled_count[i], chunks[0], 1, hold, ratio);
                try trunk.capBufAppend(s, &bank[i], &bank_view[i], &bank_count[i], chunks[1], 3, hold, ratio);
                _ = mlx.mlx_vector_array_append_value(vec, pooled[i]);
                _ = mlx.mlx_vector_array_append_value(vec, bank[i]);
            }
            try mlx.check(mlx.mlx_eval(vec));
            mlx.engine.close();
            const peak = mlx.engine.poolPeak(.process);
            mlx.engine.trim();
            const after = mlx.engine.poolTotal() - mlx.engine.poolFree();
            _ = before;
            // A regrow beyond the first fill: the history held its capacity and the block did not fit.
            if (caps != 0 and @as(usize, @intCast(mlx.getShape(pooled[0])[1])) != caps) {
                regrows += 1;
                dearest = @max(dearest, peak -| after);
            }
            blocks = target;
            target = @min(end_blocks, @as(usize, @intCast(mlx.getShape(pooled[0])[1])) + 1);
            if (target == blocks) break;
        }
        const billed = terms.grow_coexist_bytes + terms.decode_regrow_bytes;
        if (dearest > billed) {
            over += 1;
            std.debug.print("cooler contract: a {d}-token prompt + {d}: the history's dearest regrow holds {d} B past its grown buffers, billed {d} B\n", .{ seq, max_tokens, dearest, billed });
        }
    };
    try testing.expect(regrows > 0);
    try testing.expectEqual(@as(usize, 0), over);
}
