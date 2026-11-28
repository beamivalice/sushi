//! Hy4-Flash on the ledger, on the tiny fixtures (docs/bill_redesign.md part 10, C1): a load that plans commits its
//! model bill at the seal, and its lines sum to the engine's own count of what the load holds, after the load and after
//! each request; admission decides on the lines.
const std = @import("std");
const testing = std.testing;
const mlx = @import("../mlx.zig");
const bills = @import("../bills.zig");
const ledger = @import("../ledger.zig");
const static_bill = @import("../static_bill.zig");
const contract = @import("../cooler_contract_test.zig");
const hy4_contract = @import("hy4_contract.zig");
const hy4_held = @import("hy4_held.zig");
const Load = contract.Load;
const Served = contract.Served;

/// A load on the ledger, as every served load of Hy4 and the harness's run it (`model_load.openLedger`, `sealLedger`):
/// the engine read before its weights, its budget once they are resident, its plans recorded, its chunk chosen on the
/// arena they ask and that arena made, its warm, its seal.
fn openOnLedger(ld: *Load, comptime C: type, dir: []const u8) !static_bill.Before {
    try ld.open(C, dir, true);
    try testing.expect(ld.config.static_ledger != null);
    return ld.before;
}

/// The lines that hold device memory, against the engine's count of what the load holds.
fn expectLinesHeld(l: *const ledger.Ledger, before: static_bill.Before, when: []const u8) !void {
    mlx.engine.close();
    const s = l.snapshot();
    const lines = s.committed - s.of(.margin) - s.of(.host) - s.of(.media) - s.of(.pipelines);
    const held = before.heldSince();
    // Every byte the engine wraps lies in a region the weights or slots line bills.
    const h = @import("../gpu_mem.zig").engineHeld();
    try testing.expect(h.mapped - before.held.mapped <= s.of(.weights) + s.of(.slots));
    if (lines != held) {
        std.debug.print("{s}: the ledger's device lines hold {d} B, the engine {d} B (weights {d}, slots {d}, static {d}, arena {d}, state {d}, parked {d}); engine pool {d} mapped {d} arena {d}\n", .{ when, lines, held, s.of(.weights), s.of(.slots), s.of(.static), s.of(.arena), s.of(.state), s.of(.parked), h.pool - before.held.pool, h.mapped - before.held.mapped, h.arena - before.held.arena });
        return error.TestExpectedEqual;
    }
}

test "hy4 ledger: a request past the room the lines leave is refused before it allocates, and one that fits once the parked state goes releases it (HY4_BF16_TINY)" {
    try mlx.requireGpu();
    const h = try hy4_held.acquire(.bf16_openLight);
    const ids = try hy4_contract.prompt(testing.allocator, h.dir);
    defer testing.allocator.free(ids);
    const ld = h.ld;
    const before = ld.before;
    const l = ld.config.static_ledger.?;
    var sv: Served = undefined;
    try sv.init(ld, h.dir);
    defer sv.deinit();
    sv.quiet_failures = true;
    const prompt = ids[0..@min(ids.len, 40)];
    _ = try sv.requestToCleanup(prompt, 8, false);
    const parked = l.of(.parked);
    try testing.expectEqual(@as(u64, 0), parked);
    const need = l.terms.?.at(prompt.len + 8);
    const budget = l.budget();
    defer l.setBudget(budget);
    // Room for the request only once the parked state is released: admitted, the state released first.
    l.setBudget(l.committed() - parked + need);
    const pool_before = before.heldSince();
    _ = try sv.requestToCleanup(prompt, 8, false);
    try testing.expectEqual(@as(u64, 0), l.of(.parked));
    try testing.expect(before.heldSince() <= pool_before + need);
    try expectLinesHeld(l, before, "after a request that released the parked state");
    // One byte short even with the parked state released: refused, nothing reserved, nothing held more.
    l.setBudget(l.committed() - l.of(.parked) + need - 1);
    const committed = l.committed();
    const held = before.heldSince();
    _ = try sv.requestToCleanup(prompt, 8, false);
    try testing.expectEqual(committed, l.committed());
    try testing.expectEqual(held, before.heldSince());
    try testing.expectEqual(@as(u64, 0), l.of(.state));
    // A slot admitted twice (a retried prefill) holds one reservation, and its end gives back every byte of it.
    l.setBudget(budget);
    const before_slot = l.committed();
    const slot = try pendingSlot(&sv, prompt, 8, null);
    try static_bill.admit(slot);
    try static_bill.admit(slot);
    try testing.expectEqual(need, l.of(.state));
    try testing.expectEqual(before_slot + need, l.committed());
    ended(&sv, slot);
    try testing.expectEqual(@as(u64, 0), l.of(.state));
    try testing.expectEqual(before_slot, l.committed());
}

/// A request submitted to `sv` and taken off its queue, as the scheduler hands it to `runPrefill`; `ended` completes it
/// and runs the loop's cleanup drain, its ledger end among it.
fn pendingSlot(sv: *Served, prompt: []const u32, n: u32, deferred: ?*@import("../vision_queue.zig").DeferredMedia) !*Served.scheduler.Slot {
    const slot = try sv.sch.submit(.{ .model = sv.entry, .prompt_ids = prompt, .sampling = .{ .temperature = 0 }, .eos_token_ids = &.{}, .max_tokens = n, .deferred_media = deferred });
    _ = sv.sch.pending.orderedRemove(0);
    return slot;
}

fn ended(sv: *Served, slot: *Served.scheduler.Slot) void {
    sv.sch.complete(slot);
    Served.scheduler.drainSlots(&sv.sch, sv.sch.cleanup_queue.items);
    sv.sch.cleanup_queue.clearRetainingCapacity();
}

test "hy4 ledger: an image request reserves its Media and its pixels with its state, is refused before it encodes when they do not fit, and its end gives both back (HY4_BF16_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const h = try hy4_held.acquire(.bf16_openLight);
    const ids = try hy4_contract.prompt(a, h.dir);
    defer a.free(ids);
    const ld = h.ld;
    const l = ld.config.static_ledger.?;
    const v = ld.config.hy4flash.?.vision orelse return error.SkipZigTest;
    var sv: Served = undefined;
    try sv.init(ld, h.dir);
    defer sv.deinit();
    const vq = @import("../vision_queue.zig");
    const g: u32 = 2 * v.merge;
    const rows = @import("../arch/media.zig").gridRows(1, g, g, v.merge);
    const pixels = try a.alloc(f32, @as(usize, g) * g * 3 * v.patch * v.patch);
    defer a.free(pixels);
    @memset(pixels, 0);
    var prompt: [40]u32 = undefined;
    for (&prompt, 0..) |*id, i| id.* = if (i >= 3 and i < 3 + rows) ld.config.image_token_id else ids[i % ids.len];
    const text = l.terms.?.at(prompt.len + 8);
    const media = @import("hy4_encode_plan.zig").mediaBytes(v, rows);
    const host: u64 = pixels.len * @sizeOf(f32);
    const committed = l.committed();
    const budget = l.budget();
    defer l.setBudget(budget);
    for ([_]bool{ false, true }) |fits| {
        const d = try a.create(vq.DeferredMedia);
        d.* = .{ .allocator = a, .images = try a.dupe(@import("../chat.zig").ImageData, &.{.{ .pixels = std.mem.sliceAsBytes(pixels), .width = g * v.patch, .height = g * v.patch, .grid_h = g, .grid_w = g }}), .videos = &.{}, .blocks = try a.dupe(vq.MediaBlock, &.{.{ .kind = .image, .rows = rows }}) };
        const slot = try pendingSlot(&sv, &prompt, 8, d);
        defer ended(&sv, slot);
        l.setBudget(committed + text + media + host - @intFromBool(!fits));
        if (!fits) {
            try testing.expectError(error.PrefillDoesNotFit, static_bill.admit(slot));
            try testing.expectEqual(committed, l.committed());
            continue;
        }
        try static_bill.admit(slot);
        try testing.expectEqual(text + media, l.of(.state));
        try testing.expectEqual(host, l.of(.media));
    }
    try testing.expectEqual(@as(u64, 0), l.of(.state));
    try testing.expectEqual(@as(u64, 0), l.of(.media));
}

test "hy4 ledger: a request's end parks at the inference thread's finish, and the ends of a finished slot and of one that never ran leave the lines a running request holds as they were (HY4_BF16_TINY)" {
    try mlx.requireGpu();
    const h = try hy4_held.acquire(.bf16_openLight);
    const ids = try hy4_contract.prompt(testing.allocator, h.dir);
    defer testing.allocator.free(ids);
    const ld = h.ld;
    const l = ld.config.static_ledger.?;
    var sv: Served = undefined;
    try sv.init(ld, h.dir);
    defer sv.deinit();
    const prompt = ids[0..@min(ids.len, 40)];
    const a = try runToFinish(&sv, prompt, 8);
    var a_done = false;
    defer if (!a_done) ended(&sv, a);
    // The request finished: its state is parked before its connection thread completes it.
    try testing.expectEqual(@as(u64, 0), l.of(.state));
    try testing.expectEqual(@as(u64, 0), l.of(.parked));
    // B, the next turn with a longer reply, runs on the model's state.
    const b = try pendingSlot(&sv, prompt, 512, null);
    defer ended(&sv, b);
    try Served.scheduler.runPrefill(&sv.sch, b);
    const parked = l.of(.parked);
    const state = l.of(.state);
    const committed = l.committed();
    try testing.expect(state > 0);
    // A's connection thread completes it while B runs.
    a_done = true;
    ended(&sv, a);
    // C waits behind B and its client leaves: it reserved nothing and took nothing over.
    ended(&sv, try sv.sch.submit(.{ .model = sv.entry, .prompt_ids = prompt, .sampling = .{ .temperature = 0 }, .eos_token_ids = &.{}, .max_tokens = 8 }));
    errdefer std.debug.print("hy4 ledger: the ends moved parked {d} B -> {d} B, state {d} B -> {d} B, committed {d} B -> {d} B while B ran\n", .{ parked, l.of(.parked), state, l.of(.state), committed, l.committed() });
    try testing.expectEqual(parked, l.of(.parked));
    try testing.expectEqual(state, l.of(.state));
    try testing.expectEqual(committed, l.committed());
}

test "hy4 ledger: the scheduler holds an image request the lines would make wait, as it does a text one, and lets it go once its media fits (HY4_BF16_TINY)" {
    try mlx.requireGpu();
    const h = try hy4_held.acquire(.bf16_openLight);
    const ids = try hy4_contract.prompt(testing.allocator, h.dir);
    defer testing.allocator.free(ids);
    const ld = h.ld;
    const l = ld.config.static_ledger.?;
    var sv: Served = undefined;
    try sv.init(ld, h.dir);
    defer sv.deinit();
    const img = try ImagePrompt.make(ld, ids);
    defer testing.allocator.free(img.pixels);
    // A: admitted, running.
    const a_slot = try pendingSlot(&sv, ids[0..@min(ids.len, 40)], 8, null);
    defer ended(&sv, a_slot);
    try static_bill.admit(a_slot);
    sv.sch.live_hold = .{ .requests = 1, .bytes = a_slot.ledger.?.total() };
    defer sv.sch.live_hold = .{};
    // B: an image request, queued.
    const b = try sv.sch.submit(.{ .model = sv.entry, .prompt_ids = &img.prompt, .sampling = .{ .temperature = 0 }, .eos_token_ids = &.{}, .max_tokens = 8, .deferred_media = try img.media(ld) });
    defer ended(&sv, b);
    const need = l.terms.?.at(img.prompt.len + 8) + img.device + img.host;
    const budget = l.budget();
    defer l.setBudget(budget);
    const s0 = l.snapshot();
    // One byte short beside A: B fits once A's lines go. Its text alone would fit beside A.
    l.setBudget(s0.committed - s0.of(.parked) + need - 1);
    try testing.expectEqual(ledger.Verdict.wait, ledger.decide(l.snapshot(), need));
    try testing.expect(l.terms.?.at(img.prompt.len + 8) <= l.snapshot().roomAfter(.parked));
    const slot_admission = @import("../slot_admission.zig");
    try testing.expectEqual(@as(usize, 0), slot_admission.memoryAdmitCount(&sv.sch, &.{0}));
    // Room for it beside A: the scheduler lets it go.
    l.setBudget(s0.committed + need);
    try testing.expectEqual(@as(usize, 1), slot_admission.memoryAdmitCount(&sv.sch, &.{0}));
}

test "hy4 ledger: an image request re-sent at the max_tokens its refusal named is served, alone and beside a running request (HY4_BF16_TINY)" {
    try mlx.requireGpu();
    const h = try hy4_held.acquire(.bf16_openLight);
    const ids = try hy4_contract.prompt(testing.allocator, h.dir);
    defer testing.allocator.free(ids);
    const ld = h.ld;
    const l = ld.config.static_ledger.?;
    var sv: Served = undefined;
    try sv.init(ld, h.dir);
    defer sv.deinit();
    const img = try ImagePrompt.make(ld, ids);
    defer testing.allocator.free(img.pixels);
    const seq = img.prompt.len;
    const asked: u32 = 64;
    const named: u32 = 16;
    // What the request holds at `named` reply tokens, media included.
    const at_named = l.terms.?.at(seq + named) + img.device + img.host;
    const budget = l.budget();
    defer l.setBudget(budget);
    const probe = try img.media(ld);
    defer probe.deinit();
    for ([_]bool{ false, true }) |beside| {
        var runner: ?*Served.scheduler.Slot = null;
        defer if (runner) |r| ended(&sv, r);
        l.setBudget(budget);
        if (beside) {
            runner = try pendingSlot(&sv, ids[0..@min(ids.len, 40)], 8, null);
            try static_bill.admit(runner.?);
            sv.sch.live_hold = .{ .requests = 1, .bytes = runner.?.ledger.?.total() };
        }
        defer sv.sch.live_hold = .{};
        // Room for `named` tokens once the running request (if any) and the parked state are gone.
        const s = l.snapshot();
        l.setBudget(s.committed - s.freeable(.running) + at_named);
        const refused = try refusedOf(static_bill.check(&ld.config, seq, asked, probe));
        errdefer std.debug.print("hy4 ledger: asked {d}: the refusal names max_tokens {d} (maximum context {d}), beside a running request {}\n", .{ asked, refused.fitting_max_tokens, refused.maximum, beside });
        try testing.expectEqual(named, refused.fitting_max_tokens);
        // The re-send is admitted, or held for the running request to end and then admitted.
        try testing.expect(static_bill.check(&ld.config, seq, named, probe) == .proceed);
        const resend = try pendingSlot(&sv, &img.prompt, named, try img.media(ld));
        defer ended(&sv, resend);
        if (beside) {
            try testing.expectEqual(@as(usize, 0), @import("../slot_admission.zig").memoryAdmitCount(&sv.sch, &.{0}));
            ended(&sv, runner.?);
            runner = null;
            sv.sch.live_hold = .{};
        }
        try static_bill.admit(resend);
    }
}

test "hy4 ledger: with a RAM tier, the lines still sum to what the engine holds for the load after each request (HY4_BF16_TINY)" {
    try mlx.requireGpu();
    const h = try hy4_held.acquire(.bf16_openLight);
    const ids = try hy4_contract.prompt(testing.allocator, h.dir);
    defer testing.allocator.free(ids);
    const ld = h.ld;
    const before = ld.before;
    const l = ld.config.static_ledger.?;
    var sv: Served = undefined;
    try sv.initTiers(ld, h.dir, 256 << 20);
    defer sv.deinit();
    const prompt = ids[0..@min(ids.len, 40)];
    for (0..3) |i| {
        _ = try sv.requestToCleanup(prompt[0 .. prompt.len - (i % 2) * 7], 8, false);
        try testing.expectEqual(@as(u64, 0), l.of(.state));
        try expectLinesHeld(l, before, "after a request with a RAM tier");
    }
}

test "hy4 ledger: a request restored whole from its load's cache reports the prompt processed, not zero (HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const h = try hy4_held.acquire(.stq_openLight);
    const ids = try hy4_contract.prompt(a, h.dir);
    defer a.free(ids);
    const ld = h.ld;
    const l = ld.config.static_ledger.?;
    var sv: Served = undefined;
    try sv.initTiers(ld, h.dir, 1 << 30);
    defer sv.deinit();
    const prompt = try a.alloc(u32, 600);
    defer a.free(prompt);
    for (prompt, 0..) |*t, i| t.* = ids[(i * 5 + 1) % ids.len];
    var restored: u32 = 0;
    var first = try sv.request(prompt, 8, &restored);
    first.deinit(a);
    try testing.expectEqual(@as(u64, 0), l.of(.state));
    var second = try sv.sch.submit(.{ .model = sv.entry, .prompt_ids = prompt, .sampling = .{ .temperature = 0 }, .eos_token_ids = &.{}, .max_tokens = 8, .return_progress = true });
    defer {
        sv.sch.complete(second);
        for (sv.sch.cleanup_queue.items) |s| s.deinit();
        sv.sch.cleanup_queue.clearRetainingCapacity();
    }
    _ = sv.sch.pending.orderedRemove(0);
    try Served.scheduler.runPrefill(&sv.sch, second);
    const cache = second.prefill_progress.cache.load(.monotonic);
    const processed = second.prefill_progress.processed.load(.monotonic);
    errdefer std.debug.print("hy4 ledger: a restored prefill of {d} tokens published cache {d}, processed {d}, elapsed {d} ms\n", .{ prompt.len, cache, processed, second.prefill_progress.elapsed_ms.load(.monotonic) });
    try testing.expect(cache > 0);
    try testing.expect(cache <= processed);
    try testing.expectEqual(@as(u64, prompt.len), processed);
    try sv.sch.decoding.append(a, second);
    var one = [_]*Served.scheduler.Slot{second};
    if (second.joinsTick()) try Served.scheduler.runDecodeTick(&sv.sch, &one);
    const after_tick = second.prefill_progress.processed.load(.monotonic);
    errdefer std.debug.print("hy4 ledger: after the first decode of the restored request processed is {d} of {d} (cache {d})\n", .{ after_tick, prompt.len, cache });
    try testing.expectEqual(@as(u64, prompt.len), after_tick);
}

test "hy4 ledger: a load that plans commits its model bill at the seal, its warm runs its prefill in the arena made before it, and its lines sum to what the engine holds for it, after the load and after each request (HY4_BF16_TINY, HY4_STQ_TINY)" {
    hy4_held.releaseHeld();
    try contract.servedOn("hy4flash", &.{ "HY4_BF16_TINY", "HY4_STQ_TINY" }, struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            const cooler_warm = @import("../cooler_warm.zig");
            cooler_warm.longest_for_test = null;
            // Free pool buffers an earlier load or test left, which the load's warm releases.
            {
                const core = @import("../tensor/core.zig");
                const left = try core.fresh(.uint8, &.{16 << 20});
                core.release(left);
            }
            var ld: Load = undefined;
            const before = try openOnLedger(&ld, C, dir);
            defer _ = ld.close();
            defer static_bill.close(&ld.config);
            const l = ld.config.static_ledger.?;
            try testing.expect(l.of(.weights) > 0);
            try testing.expect(l.of(.arena) > 0);
            try testing.expectEqual(bills.BILL_MARGIN_BYTES, l.of(.margin));
            try testing.expect(ld.config.pinned_context > 0);
            // The ledger carries the terms of the chunk the load fixed, at its KV width.
            const kv = bills.defaultKvBits(&ld.config);
            const terms = l.terms.?;
            try testing.expectEqual(bills.ledgerTermsAt(&ld.config, kv, ld.config.static_prefill_chunk).?, terms);
            // The seal computed the context once and kept it on the ledger, where the context readers find it.
            const fits = ledger.contextFor(l.budget() - l.snapshot().modelBill(), terms, bills.ledgerWindow(&ld.config));
            try testing.expectEqual(fits, l.context.load(.acquire));
            try testing.expectEqual(fits, ld.config.pinned_context);
            try testing.expectEqual(fits, bills.memoryContextAtChunk(&ld.config, kv, 0, 0, ld.config.static_prefill_chunk));
            // The arena was made before the warm, which ran its prefill there as a request does: its widest request asked
            // and held of the pool no more than the terms at its reach.
            const w = cooler_warm.longest_for_test orelse return error.TestNoWarmMeasured;
            const reach = bills.warmLongestPrompt(&ld.config, ld.config.static_prefill_chunk) + bills.WARM_REPLY_TOKENS;
            errdefer std.debug.print("{s}: the warm's widest request asked {d} B, held {d} B, against its bill {d} B (the terms at its reach alone {d} B)\n", .{ dir, w.asked, w.held, w.bill, terms.at(reach) });
            try testing.expectEqual(terms.at(reach), w.bill);
            try testing.expect(w.asked <= w.bill);
            try testing.expect(w.held <= w.bill);
            try expectLinesHeld(l, before, "after the load");
            var sv: Served = undefined;
            try sv.init(&ld, dir);
            defer sv.deinit();
            const prompt = ids[0..@min(ids.len, 40)];
            for (0..2) |i| {
                _ = try sv.requestToCleanup(prompt[0 .. prompt.len - i * 7], 8, false);
                try testing.expectEqual(@as(u64, 0), l.of(.state));
                try testing.expectEqual(@as(u64, 0), l.of(.parked));
                try expectLinesHeld(l, before, "after a request");
            }
        }
    }.run);
}

test "hy4 ledger: a reading outside a load on the ledger counts its whole budget as taken, and the next load's budget is what is left beside it and MLX's cache (HY4_BF16_TINY)" {
    hy4_held.releaseHeld();
    try contract.servedOn("hy4flash", &.{"HY4_BF16_TINY"}, struct {
        fn run(comptime C: type, dir: []const u8, _: []const u32) !void {
            const gpu_mem = @import("../gpu_mem.zig");
            const active0 = gpu_mem.activeBytes();
            var ld: Load = undefined;
            _ = try openOnLedger(&ld, C, dir);
            defer _ = ld.close();
            defer static_bill.close(&ld.config);
            const l = ld.config.static_ledger.?;
            mlx.engine.close();
            // What the seal saw of the model bill in the engine's count (`Ledger.holdWhole`) leaves the rest to the reading.
            try testing.expectEqual(l.budget(), bills.promisedActive(gpu_mem.activeBytes(), null) - active0);
            try testing.expectEqual(gpu_mem.activeBytes(), bills.promisedActive(gpu_mem.activeBytes(), &ld.config));
            // A load beside it plans in what the budget leaves, and less the cache MLX holds, which no engine load reuses.
            const next = static_bill.Before.now();
            const planning = bills.ceilingNow().planning();
            try testing.expect(static_bill.budgetFor(next) <= planning -| l.budget());
            defer gpu_mem.cache_bytes_override = null;
            gpu_mem.cache_bytes_override = gpu_mem.engineFree();
            const without = static_bill.budgetFor(next);
            gpu_mem.cache_bytes_override = gpu_mem.engineFree() + (1 << 30);
            try testing.expectEqual(without -| (1 << 30), static_bill.budgetFor(next));
        }
    }.run);
}

fn refusedOf(c: static_bill.Check) !@import("../model_registry.zig").MemoryContextRefusal {
    return switch (c) {
        .refused => |r| r,
        .proceed => error.TestExpectedRefusal,
    };
}

/// A request run to its finish as the inference loop runs it (its prefill, its ticks, the cull), its connection thread
/// not yet back to complete it.
fn runToFinish(sv: *Served, prompt: []const u32, n: u32) !*Served.scheduler.Slot {
    const slot = try pendingSlot(sv, prompt, n, null);
    try Served.scheduler.runPrefill(&sv.sch, slot);
    try sv.sch.decoding.append(testing.allocator, slot);
    var one = [_]*Served.scheduler.Slot{slot};
    while (slot.joinsTick()) try Served.scheduler.runDecodeTick(&sv.sch, &one);
    Served.scheduler.cullDecoding(&sv.sch);
    return slot;
}

test "hy4 ledger: the arena a load's chunk is chosen on is the arena it then holds, in whole pages (HY4_BF16_TINY, HY4_STQ_TINY)" {
    hy4_held.releaseHeld();
    try contract.servedOn("hy4flash", &.{ "HY4_BF16_TINY", "HY4_STQ_TINY" }, struct {
        fn run(comptime C: type, dir: []const u8, _: []const u32) !void {
            const model_load = @import("../model_load.zig");
            const cooler = @import("../arch/cooler.zig");
            const before = static_bill.Before.now();
            var ld: Load = undefined;
            try ld.prepare(C, dir, true);
            try ld.load(dir);
            defer ld.discard();
            const room = try static_bill.open(&ld.config, before);
            var measured: model_load.PlanArena = .{ .xfm = ld.xfm };
            const job = model_load.JobTurn.begin(ld.h, &ld.config);
            defer job.end();
            try cooler.planRecord(ld.xfm, null);
            const chunk = try bills.fixStaticPrefillChunk(&ld.config, .{ .room = room, .arena = measured.arena() });
            const chosen_on = try measured.arena().of(chunk);
            ld.xfm.config.static_prefill_chunk = chunk;
            ld.xfm.config.pinned_prefill_chunk = chunk;
            try cooler.planLoad(ld.xfm, null);
            mlx.engine.close();
            const held = @import("../gpu_mem.zig").engineHeld().arena - before.held.arena;
            errdefer std.debug.print("{s}: the chunk of {d} rows was chosen on an arena of {d} B; the load holds {d} B of it\n", .{ dir, chunk, chosen_on, held });
            try testing.expectEqual(held, chosen_on);
        }
    }.run);
}

/// A one-image prompt of 40 ids on the tiny's vision geometry, and its pixels; caller frees `pixels`.
const ImagePrompt = struct {
    prompt: [40]u32,
    pixels: []f32,
    g: u32,
    rows: usize,
    /// What the request's media adds to its text: the rows its encode writes (`device`) and its pixels (`host`).
    device: u64,
    host: u64,

    fn make(ld: *Load, ids: []const u32) !ImagePrompt {
        const v = ld.config.hy4flash.?.vision orelse return error.SkipZigTest;
        var out: ImagePrompt = undefined;
        out.g = 2 * v.merge;
        out.rows = @import("../arch/media.zig").gridRows(1, out.g, out.g, v.merge);
        out.pixels = try testing.allocator.alloc(f32, @as(usize, out.g) * out.g * 3 * v.patch * v.patch);
        @memset(out.pixels, 0);
        for (&out.prompt, 0..) |*id, i| id.* = if (i >= 3 and i < 3 + out.rows) ld.config.image_token_id else ids[i % ids.len];
        out.device = @import("hy4_encode_plan.zig").mediaBytes(v, out.rows);
        out.host = out.pixels.len * @sizeOf(f32);
        return out;
    }

    fn media(self: *const ImagePrompt, ld: *Load) !*@import("../vision_queue.zig").DeferredMedia {
        const a = testing.allocator;
        const v = ld.config.hy4flash.?.vision.?;
        const vq = @import("../vision_queue.zig");
        const d = try a.create(vq.DeferredMedia);
        d.* = .{ .allocator = a, .images = try a.dupe(@import("../chat.zig").ImageData, &.{.{ .pixels = std.mem.sliceAsBytes(self.pixels), .width = self.g * v.patch, .height = self.g * v.patch, .grid_h = self.g, .grid_w = self.g }}), .videos = &.{}, .blocks = try a.dupe(vq.MediaBlock, &.{.{ .kind = .image, .rows = self.rows }}) };
        return d;
    }
};

/// What a load made in the room of its widest chunk at a context sees at its seal: the chunk it fixed and the seal's result.
const Tight = struct { chunk: u32, tokens: u32, sealed: anyerror!void, arena: bills.ArenaAt };

/// A served load of `ld` (`Load.serve`) whose chunk is chosen in exactly the room that holds its widest chunk beside a
/// context of `ctx` tokens: the static and pipelines its warm adds at the seal were unknown to the choice.
fn sealedTight(ld: *Load, comptime C: type, dir: []const u8, ctx: u32) !Tight {
    const model_load = @import("../model_load.zig");
    const cooler = @import("../arch/cooler.zig");
    try ld.prepare(C, dir, true);
    try ld.load(dir);
    errdefer ld.discard();
    const job = model_load.JobTurn.begin(ld.h, &ld.config);
    defer job.end();
    const kv_cfg = bills.kvCacheForLaunch(&ld.config, null).config;
    try ld.xfm.cache.reinit(ld.config.num_hidden_layers, kv_cfg);
    const mtp_on = ld.mtpOn();
    if (mtp_on) _ = try @import("../arch/arch_state.zig").buildDraftRerank(ld.xfm);
    const policy = @import("../mtp_policy.zig");
    const verify_cap: ?u32 = if (mtp_on) @min(policy.resolveMtpDepthCapForProfile(0, .generic), policy.mtpVerifyDraftsMax(&ld.config)) else null;
    const room = try static_bill.open(&ld.config, ld.before);
    var measured: model_load.PlanArena = .{ .xfm = ld.xfm };
    try cooler.planRecord(ld.xfm, null);
    const widest = @import("../arch/config.zig").staticChunkDefault(&ld.config);
    const tight = (try measured.arena().of(widest)) + bills.ledgerTermsAt(&ld.config, bills.defaultKvBits(&ld.config), widest).?.at(ctx);
    if (tight > room) return error.SkipZigTest;
    const l = ld.config.static_ledger.?;
    l.setBudget(l.budget() - (room - tight));
    ld.config.ctx_override = ctx;
    const chunk = try bills.fixStaticPrefillChunk(&ld.config, .{ .room = tight, .arena = measured.arena() });
    ld.xfm.config.static_prefill_chunk = chunk;
    ld.xfm.config.pinned_prefill_chunk = chunk;
    try cooler.planLoad(ld.xfm, null);
    try model_load.warmLoad(std.Io.Threaded.global_single_threaded.io(), ld.a, &ld.config, ld.xfm, verify_cap, null, kv_cfg);
    const sealed = model_load.sealLedger(&ld.config, ld.xfm, ld.before);
    return .{ .chunk = chunk, .tokens = ctx, .sealed = sealed, .arena = measured.arena() };
}

test "hy4 ledger: a fixed context the seal's lines no longer hold refuses the load by name, and the widest narrower chunk that holds it is measured dry (HY4_BF16_TINY)" {
    hy4_held.releaseHeld();
    try contract.servedOn("hy4flash", &.{"HY4_BF16_TINY"}, struct {
        fn run(comptime C: type, dir: []const u8, _: []const u32) !void {
            var ld: Load = undefined;
            const t = try sealedTight(&ld, C, dir, 8192);
            defer ld.discard();
            const l = ld.config.static_ledger.?;
            try testing.expectEqual(@as(u32, 0), l.context.load(.acquire));
            try testing.expectError(error.ContextDoesNotFit, t.sealed);
            var buf: [256]u8 = undefined;
            const b = switch (bills.prefillChunkBounds(&buf, &ld.config)) {
                .bounds => |b| b,
                else => return error.TestExpectedBounds,
            };
            const bill = static_bill.Before.modelBill(ld.before);
            const held = bill.sum() - bill.arena;
            const kv = bills.defaultKvBits(&ld.config);
            const rung = (try bills.chunkHoldingContext(&ld.config, b, t.chunk, t.tokens, held, l.budget(), t.arena)) orelse return error.TestNoRungHoldsIt;
            errdefer std.debug.print("{s}: a context of {d} tokens fixed at chunk {d}; the rung {d} holds it\n", .{ dir, t.tokens, t.chunk, rung });
            try testing.expect(rung < t.chunk and rung >= b.min);
            try testing.expect(held + try t.arena.of(rung) + bills.ledgerTermsAt(&ld.config, kv, rung).?.at(t.tokens) <= l.budget());
            try testing.expect(held + try t.arena.of(t.chunk) + bills.ledgerTermsAt(&ld.config, kv, t.chunk).?.at(t.tokens) > l.budget());
        }
    }.run);
}

test "hy4 ledger: a cold prompt that spans chunks reports the whole prompt processed before its first token (HY4_STQ_TINY)" {
    hy4_held.releaseHeld();
    try contract.servedOn("hy4flash", &.{"HY4_STQ_TINY"}, struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            const sr = @import("../settings_resolve.zig");
            const saved_launch = sr.launch;
            defer sr.launch = saved_launch;
            sr.launch.prefill_chunk_max = 512;
            sr.launch.prefill_chunk_min = 512;
            var ld: Load = undefined;
            _ = try openOnLedger(&ld, C, dir);
            defer _ = ld.close();
            defer static_bill.close(&ld.config);
            try testing.expectEqual(@as(u32, 512), ld.config.static_prefill_chunk);
            var sv: Served = undefined;
            try sv.initTiers(&ld, dir, 1 << 30);
            defer sv.deinit();
            const prompt = try a.alloc(u32, 600);
            defer a.free(prompt);
            for (prompt, 0..) |*t, i| t.* = ids[(i * 11 + 3) % ids.len];
            var slot = try sv.sch.submit(.{ .model = sv.entry, .prompt_ids = prompt, .sampling = .{ .temperature = 0 }, .eos_token_ids = &.{}, .max_tokens = 8, .return_progress = true });
            defer {
                sv.sch.complete(slot);
                for (sv.sch.cleanup_queue.items) |s| s.deinit();
                sv.sch.cleanup_queue.clearRetainingCapacity();
            }
            _ = sv.sch.pending.orderedRemove(0);
            try Served.scheduler.runPrefill(&sv.sch, slot);
            const cache = slot.prefill_progress.cache.load(.monotonic);
            const processed = slot.prefill_progress.processed.load(.monotonic);
            errdefer std.debug.print("hy4 ledger: a cold prefill of {d} tokens over chunks of {d} published cache {d}, processed {d}, elapsed {d} ms\n", .{ prompt.len, ld.config.static_prefill_chunk, cache, processed, slot.prefill_progress.elapsed_ms.load(.monotonic) });
            try testing.expectEqual(@as(u64, 0), cache);
            try testing.expectEqual(@as(u64, prompt.len), processed);
            try testing.expectEqual(@as(usize, 0), slot.out_buf.items.len);
        }
    }.run);
}
