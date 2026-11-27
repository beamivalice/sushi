const std = @import("std");
const hy4_sched_knobs = @import("hy4_sched_knobs.zig");
const testing = std.testing;
const mlx = @import("../mlx.zig");
const static_bill = @import("../static_bill.zig");
const sched = @import("../arch/sched.zig");
const slot_admission = @import("../slot_admission.zig");
const contract = @import("../cooler_contract_test.zig");
const hy4_model = @import("hy4_model.zig");
const hy4_prefix = @import("hy4_prefix.zig");
const hy4_state = @import("hy4_state.zig");
const hy4_contract = @import("hy4_contract.zig");
const Load = contract.Load;
const Served = contract.Served;
const scheduler = Served.scheduler;

fn appendArrayBytes(a: std.mem.Allocator, arr: mlx.mlx_array, out: *std.ArrayList(u8)) !void {
    try mlx.check(mlx.mlx_array_eval(arr));
    const n = mlx.mlx_array_size(arr) * mlx.mlx_array_itemsize(arr);
    const raw: [*]const u8 = @ptrCast(mlx.mlx_array_data_uint8(arr) orelse return error.HostRead);
    try out.appendSlice(a, raw[0..n]);
}

fn captureRequestBytes(a: std.mem.Allocator, m: *const hy4_model.Model, r: *const hy4_model.Request) !std.ArrayList(u8) {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.appendSlice(a, std.mem.asBytes(&r.pos));
    try out.appendSlice(a, std.mem.sliceAsBytes(r.history.items));
    var arrays: [hy4_model.max_state_arrays]mlx.mlx_array = undefined;
    for (r.state[0..m.config.layers]) |*st| {
        for (hy4_model.stateArrays(st, &arrays)) |arr| try appendArrayBytes(a, arr, &out);
    }
    if (r.last_hidden) |h| try appendArrayBytes(a, h, &out);
    return out;
}

fn endSlot(sv: *Served, slot: *scheduler.Slot) void {
    sv.sch.complete(slot);
    scheduler.drainSlots(&sv.sch, sv.sch.cleanup_queue.items);
    sv.sch.cleanup_queue.clearRetainingCapacity();
}

fn submitPending(sv: *Served, prompt: []const u32, n: u32, mtp: bool) !*scheduler.Slot {
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

const SoloResult = struct { tokens: std.ArrayList(u32), state: std.ArrayList(u8) };

fn soloRun(sv: *Served, prompt: []const u32, n: u32, mtp: bool) !SoloResult {
    const a = testing.allocator;
    const xfm = @import("../arch/handle.zig").trunk(sv.entry.handle.?);
    const m = xfm.arch_state.hy4flash.model.?;
    const slot = try submitPending(sv, prompt, n, mtp);
    defer endSlot(sv, slot);
    try scheduler.runPrefill(&sv.sch, slot);
    if (!scheduler.postPrefillTerminal(&sv.sch, slot)) {
        try sv.sch.decoding.append(a, slot);
        var one = [_]*scheduler.Slot{slot};
        while (slot.joinsTick()) try scheduler.runDecodeTick(&sv.sch, &one);
    }
    if (slot.error_code != null) return error.RequestFailed;
    const req = if (slot.request) |*r| r.as(.hy4flash) else null;
    var state = try captureRequestBytes(a, m, req orelse return error.TestExpectedSlotRequest);
    errdefer state.deinit(a);
    var tokens: std.ArrayList(u32) = .empty;
    errdefer tokens.deinit(a);
    try tokens.appendSlice(a, slot.legacy_gen.?.generated_ids.items);
    scheduler.cullDecoding(&sv.sch);
    return .{ .tokens = tokens, .state = state };
}

test "hy4 request: two requests interleaved tick by tick match solo tokens and end state (HY4_STQ_TINY)" {
    try contract.servedOn("hy4flash", &.{"HY4_STQ_TINY"}, struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            var ld: Load = undefined;
            try ld.open(C, dir, true);
            defer _ = ld.close();
            defer static_bill.close(&ld.config);
            var sv: Served = undefined;
            try sv.init(&ld, dir);
            defer sv.deinit();
            sv.sch.queue_cap = 4;
            const xfm = @import("../arch/handle.zig").trunk(sv.entry.handle.?);
            const m = xfm.arch_state.hy4flash.model.?;
            try testing.expectError(error.Hy4StateNotLent, hy4_prefix.checkLent(m));
            try testing.expect(!slot_admission.modelExclusiveDecode(sv.entry));
            const prompt_a = ids[0..@min(ids.len, 24)];
            const prompt_b = ids[5..@min(ids.len, 35)];
            for ([_]bool{ false, true }) |mtp| {
                if (mtp and xfm.arch_state.hy4flash.drafter == null) continue;
                var solo_a = try soloRun(&sv, prompt_a, 8, mtp);
                defer solo_a.tokens.deinit(a);
                defer solo_a.state.deinit(a);
                var solo_b = try soloRun(&sv, prompt_b, 8, mtp);
                defer solo_b.tokens.deinit(a);
                defer solo_b.state.deinit(a);
                const slot_a = try submitPending(&sv, prompt_a, 8, mtp);
                defer endSlot(&sv, slot_a);
                const slot_b = try submitPending(&sv, prompt_b, 8, mtp);
                defer endSlot(&sv, slot_b);
                try scheduler.runPrefill(&sv.sch, slot_a);
                try sv.sch.decoding.append(a, slot_a);
                try scheduler.runPrefill(&sv.sch, slot_b);
                try sv.sch.decoding.append(a, slot_b);
                var state_a: ?std.ArrayList(u8) = null;
                defer if (state_a) |*s| s.deinit(a);
                var state_b: ?std.ArrayList(u8) = null;
                defer if (state_b) |*s| s.deinit(a);
                while (slot_a.joinsTick() or slot_b.joinsTick()) {
                    var active: [2]*scheduler.Slot = undefined;
                    var k: usize = 0;
                    if (slot_a.joinsTick()) {
                        active[k] = slot_a;
                        k += 1;
                    }
                    if (slot_b.joinsTick()) {
                        active[k] = slot_b;
                        k += 1;
                    }
                    try scheduler.runDecodeTick(&sv.sch, active[0..k]);
                    if (state_a == null and !slot_a.joinsTick()) {
                        state_a = try captureRequestBytes(a, m, slot_a.request.?.as(.hy4flash).?);
                    }
                    if (state_b == null and !slot_b.joinsTick()) {
                        state_b = try captureRequestBytes(a, m, slot_b.request.?.as(.hy4flash).?);
                    }
                    scheduler.cullDecoding(&sv.sch);
                }
                try testing.expect(slot_a.error_code == null and slot_b.error_code == null);
                try testing.expectEqualSlices(u32, solo_a.tokens.items, slot_a.legacy_gen.?.generated_ids.items);
                try testing.expectEqualSlices(u32, solo_b.tokens.items, slot_b.legacy_gen.?.generated_ids.items);
                try testing.expectEqualSlices(u8, solo_a.state.items, state_a.?.items);
                try testing.expectEqualSlices(u8, solo_b.state.items, state_b.?.items);
            }
        }
    }.run);
}

test "hy4 request: join mid-decode and cancel mid-decode leave survivor unchanged and release every cancelled byte (HY4_STQ_TINY)" {
    try contract.servedOn("hy4flash", &.{"HY4_STQ_TINY"}, struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            var ld: Load = undefined;
            try ld.open(C, dir, true);
            defer _ = ld.close();
            defer static_bill.close(&ld.config);
            const l = ld.config.static_ledger.?;
            var sv: Served = undefined;
            try sv.init(&ld, dir);
            defer sv.deinit();
            sv.sch.queue_cap = 4;
            const prompt_a = ids[0..@min(ids.len, 24)];
            const prompt_b = ids[6..@min(ids.len, 34)];
            var solo_a = try soloRun(&sv, prompt_a, 10, false);
            defer solo_a.tokens.deinit(a);
            defer solo_a.state.deinit(a);
            const slot_a = try submitPending(&sv, prompt_a, 10, false);
            var a_ended = false;
            defer if (!a_ended) endSlot(&sv, slot_a);
            try scheduler.runPrefill(&sv.sch, slot_a);
            try sv.sch.decoding.append(a, slot_a);
            var one_a = [_]*scheduler.Slot{slot_a};
            for (0..3) |_| if (slot_a.joinsTick()) try scheduler.runDecodeTick(&sv.sch, &one_a);
            const bill_a = slot_a.ledger.?.state();
            try testing.expectEqual(bill_a, l.of(.state));
            try testing.expect(sched.requestBytes(slot_a) > 0);
            try testing.expectEqual(@as(u64, 0), sched.modelStateBytes(slot_a));
            try testing.expectEqual(@as(u64, 0), sched.parkedStateBytes(sv.entry));
            const slot_b = try submitPending(&sv, prompt_b, 10, false);
            var b_ended = false;
            defer if (!b_ended) endSlot(&sv, slot_b);
            try scheduler.runPrefill(&sv.sch, slot_b);
            try sv.sch.decoding.append(a, slot_b);
            const bill_b = slot_b.ledger.?.state();
            try testing.expectEqual(bill_a + bill_b, l.of(.state));
            try testing.expect(sched.requestBytes(slot_b) > 0);
            try testing.expectEqual(@as(u64, 0), sched.modelStateBytes(slot_b));
            var pair = [_]*scheduler.Slot{ slot_a, slot_b };
            for (0..2) |_| if (slot_a.joinsTick() and slot_b.joinsTick()) try scheduler.runDecodeTick(&sv.sch, &pair);
            slot_b.cancelled.store(true, .release);
            scheduler.cullDecoding(&sv.sch);
            try testing.expectEqual(@as(u64, 0), sched.requestBytes(slot_b));
            try testing.expectEqual(bill_a, l.of(.state));
            try testing.expectEqual(@as(u64, 0), l.of(.parked));
            b_ended = true;
            endSlot(&sv, slot_b);
            try testing.expectEqual(bill_a, l.of(.state));
            while (slot_a.joinsTick()) try scheduler.runDecodeTick(&sv.sch, &one_a);
            scheduler.cullDecoding(&sv.sch);
            try testing.expect(slot_a.error_code == null);
            try testing.expectEqualSlices(u32, solo_a.tokens.items, slot_a.legacy_gen.?.generated_ids.items);
            try testing.expectEqual(@as(u64, 0), sched.requestBytes(slot_a));
            try testing.expectEqual(@as(u64, 0), l.of(.state));
            try testing.expectEqual(@as(u64, 0), l.of(.parked));
            a_ended = true;
            endSlot(&sv, slot_a);
            try testing.expectEqual(@as(u64, 0), l.of(.state));
            try testing.expectEqual(@as(u64, 0), l.of(.parked));
        }
    }.run);
}

test "hy4 request: N running requests bill N states exactly (HY4_STQ_TINY)" {
    try contract.servedOn("hy4flash", &.{"HY4_STQ_TINY"}, struct {
        fn run(comptime C: type, dir: []const u8, ids: []const u32) !void {
            const a = testing.allocator;
            var ld: Load = undefined;
            try ld.open(C, dir, true);
            defer _ = ld.close();
            defer static_bill.close(&ld.config);
            const l = ld.config.static_ledger.?;
            var sv: Served = undefined;
            try sv.init(&ld, dir);
            defer sv.deinit();
            sv.sch.queue_cap = 4;
            const prompts = [_][]const u32{
                ids[0..@min(ids.len, 16)],
                ids[4..@min(ids.len, 28)],
                ids[8..@min(ids.len, 38)],
            };
            const max_toks = [_]u32{ 6, 10, 14 };
            var slots: [3]*scheduler.Slot = undefined;
            var bills_each: [3]u64 = undefined;
            var made: usize = 0;
            var done: [3]bool = @splat(false);
            defer for (0..made) |i| if (!done[i]) endSlot(&sv, slots[i]);
            var expected_state: u64 = 0;
            for (0..3) |i| {
                slots[i] = try submitPending(&sv, prompts[i], max_toks[i], false);
                made = i + 1;
                try scheduler.runPrefill(&sv.sch, slots[i]);
                try sv.sch.decoding.append(a, slots[i]);
                bills_each[i] = slots[i].ledger.?.state();
                expected_state += bills_each[i];
                try testing.expectEqual(expected_state, l.of(.state));
                try testing.expectEqual(@as(u64, 0), l.of(.parked));
                try testing.expectEqual(@as(u64, 0), sched.parkedStateBytes(sv.entry));
                try testing.expectEqual(@as(u64, 0), sched.modelStateBytes(slots[i]));
                try testing.expect(sched.requestBytes(slots[i]) > 0);
                try testing.expectEqual(
                    slots[i].cache.residentBytes() + sched.requestBytes(slots[i]) + slots[i].ring_cps.bytes(),
                    scheduler.slotStateBytes(slots[i]),
                );
            }
            for (0..3) |i| {
                slots[i].cancelled.store(true, .release);
                scheduler.cullDecoding(&sv.sch);
                expected_state -= bills_each[i];
                try testing.expectEqual(expected_state, l.of(.state));
                try testing.expectEqual(@as(u64, 0), sched.requestBytes(slots[i]));
                done[i] = true;
                endSlot(&sv, slots[i]);
            }
            try testing.expectEqual(@as(u64, 0), l.of(.state));
            try testing.expectEqual(@as(u64, 0), l.of(.parked));
        }
    }.run);
}

const ResidentHy4 = struct {
    pub fn prepare(a: std.mem.Allocator, config: *@import("../model.zig").ModelConfig, dir: []const u8) !void {
        try hy4_contract.prepare(a, config, dir);
        config.expert_resident = true;
        config.expert_cache_bytes = std.math.maxInt(u64) / 2;
    }
};

test "hy4 request: two and three resident requests join into batchedForward, match solo tokens and end state, and mid-decode cancel preserves survivors (HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const dir = std.mem.span(std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest);
    const ids = try hy4_contract.prompt(a, dir);
    defer a.free(ids);
    var ld: Load = undefined;
    try ld.open(ResidentHy4, dir, true);
    defer _ = ld.close();
    defer static_bill.close(&ld.config);
    const l = ld.config.static_ledger.?;
    var sv: Served = undefined;
    try sv.init(&ld, dir);
    defer sv.deinit();
    sv.sch.queue_cap = 8;
    const xfm = @import("../arch/handle.zig").trunk(sv.entry.handle.?);
    const m = xfm.arch_state.hy4flash.model.?;
    try testing.expectEqual(true, sched.batches(&ld.config));
    try testing.expectEqual(true, sched.modelBatches(xfm));
    const prompts = [_][]const u32{
        ids[0..@min(ids.len, 16)],
        ids[4..@min(ids.len, 24)],
        ids[8..@min(ids.len, 32)],
    };
    const max_toks: u32 = 8;
    var solos: [3]SoloResult = undefined;
    var solos_made: usize = 0;
    defer for (0..solos_made) |i| {
        solos[i].tokens.deinit(a);
        solos[i].state.deinit(a);
    };
    for (0..3) |i| {
        solos[i] = try soloRun(&sv, prompts[i], max_toks, false);
        solos_made = i + 1;
    }
    for ([_]usize{ 2, 3 }) |count| {
        hy4_state.joined_ticks_for_test = 0;
        var slots: [3]*scheduler.Slot = undefined;
        var made: usize = 0;
        defer for (0..made) |i| endSlot(&sv, slots[i]);
        var states: [3]?std.ArrayList(u8) = @splat(null);
        defer for (&states) |*s| if (s.*) |*v| v.deinit(a);
        for (0..count) |i| {
            slots[i] = try submitPending(&sv, prompts[i], max_toks, false);
            made = i + 1;
            try scheduler.runPrefill(&sv.sch, slots[i]);
            try sv.sch.decoding.append(a, slots[i]);
            if (i + 1 < count) {
                var act: [3]*scheduler.Slot = undefined;
                var k: usize = 0;
                for (0..made) |j| if (slots[j].joinsTick()) {
                    act[k] = slots[j];
                    k += 1;
                };
                if (k > 0) try scheduler.runDecodeTick(&sv.sch, act[0..k]);
                scheduler.cullDecoding(&sv.sch);
            }
        }
        var ticks: usize = 0;
        while (ticks < 64) : (ticks += 1) {
            var act: [3]*scheduler.Slot = undefined;
            var k: usize = 0;
            for (0..count) |i| if (slots[i].joinsTick()) {
                act[k] = slots[i];
                k += 1;
            };
            if (k == 0) break;
            try scheduler.runDecodeTick(&sv.sch, act[0..k]);
            for (0..count) |i| if (states[i] == null and !slots[i].joinsTick()) {
                states[i] = try captureRequestBytes(a, m, slots[i].request.?.as(.hy4flash).?);
            };
            scheduler.cullDecoding(&sv.sch);
        }
        try testing.expect(ticks < 64);
        try testing.expect(hy4_state.joined_ticks_for_test > 0);
        for (0..count) |i| {
            try testing.expectEqual(@as(?[]const u8, null), slots[i].error_code);
            try testing.expectEqualSlices(u32, solos[i].tokens.items, slots[i].legacy_gen.?.generated_ids.items);
            try testing.expectEqualSlices(u8, solos[i].state.items, states[i].?.items);
        }
    }
    hy4_state.joined_ticks_for_test = 0;
    var slots: [3]*scheduler.Slot = undefined;
    var made: usize = 0;
    var ended_flag: [3]bool = @splat(false);
    defer for (0..made) |i| if (!ended_flag[i]) endSlot(&sv, slots[i]);
    for (0..3) |i| {
        slots[i] = try submitPending(&sv, prompts[i], max_toks, false);
        made = i + 1;
        try scheduler.runPrefill(&sv.sch, slots[i]);
        try sv.sch.decoding.append(a, slots[i]);
    }
    const bill0 = slots[0].ledger.?.state();
    const bill1 = slots[1].ledger.?.state();
    const bill2 = slots[2].ledger.?.state();
    try testing.expectEqual(bill0 + bill1 + bill2, l.of(.state));
    var all3 = [_]*scheduler.Slot{ slots[0], slots[1], slots[2] };
    try scheduler.runDecodeTick(&sv.sch, &all3);
    try testing.expect(hy4_state.joined_ticks_for_test > 0);
    slots[1].cancelled.store(true, .release);
    scheduler.cullDecoding(&sv.sch);
    try testing.expectEqual(@as(u64, 0), sched.requestBytes(slots[1]));
    try testing.expectEqual(bill0 + bill2, l.of(.state));
    ended_flag[1] = true;
    endSlot(&sv, slots[1]);
    var ticks: usize = 0;
    while (ticks < 64) : (ticks += 1) {
        var act: [2]*scheduler.Slot = undefined;
        var k: usize = 0;
        for ([_]usize{ 0, 2 }) |i| if (slots[i].joinsTick()) {
            act[k] = slots[i];
            k += 1;
        };
        if (k == 0) break;
        try scheduler.runDecodeTick(&sv.sch, act[0..k]);
        scheduler.cullDecoding(&sv.sch);
    }
    try testing.expect(ticks < 64);
    for ([_]usize{ 0, 2 }) |i| {
        try testing.expect(slots[i].error_code == null);
        try testing.expectEqualSlices(u32, solos[i].tokens.items, slots[i].legacy_gen.?.generated_ids.items);
        try testing.expectEqual(@as(u64, 0), sched.requestBytes(slots[i]));
    }
    try testing.expectEqual(@as(u64, 0), l.of(.state));
    try testing.expectEqual(@as(u64, 0), l.of(.parked));
}

test "hy4 request: a drafting slot rides rows only on a batching load, unconstrained and unpenalised (HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const dir = std.mem.span(std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest);
    const ids = try hy4_contract.prompt(a, dir);
    defer a.free(ids);
    var ld: Load = undefined;
    try ld.open(ResidentHy4, dir, true);
    defer _ = ld.close();
    defer static_bill.close(&ld.config);
    var sv: Served = undefined;
    try sv.init(&ld, dir);
    defer sv.deinit();
    sv.sch.queue_cap = 4;
    const xfm = @import("../arch/handle.zig").trunk(sv.entry.handle.?);
    const m = xfm.arch_state.hy4flash.model.?;
    try testing.expect(sched.modelBatches(xfm));
    const slot = try submitPending(&sv, ids[0..@min(ids.len, 24)], 6, true);
    defer endSlot(&sv, slot);
    try scheduler.runPrefill(&sv.sch, slot);
    const gen = &slot.legacy_gen.?;
    try testing.expect(gen.drafter_on);
    try testing.expectEqual(sched.TickClass.rows_company, sched.tickClass(slot));
    var con: @import("../generate.zig").Constraint = undefined;
    slot.sampling.constraint = &con;
    try testing.expectEqual(sched.TickClass.none, sched.tickClass(slot));
    slot.sampling.constraint = null;
    slot.sampling.repeat_penalty = 1.1;
    try testing.expectEqual(sched.TickClass.none, sched.tickClass(slot));
    slot.sampling.repeat_penalty = 1.0;
    gen.done = true;
    try testing.expectEqual(sched.TickClass.none, sched.tickClass(slot));
    gen.done = false;
    gen.drafter_on = false;
    try testing.expectEqual(sched.TickClass.none, sched.tickClass(slot));
    gen.drafter_on = true;
    const served_was = m.serves_requests;
    m.serves_requests = false;
    defer m.serves_requests = served_was;
    try testing.expectEqual(sched.TickClass.none, sched.tickClass(slot));
}

test "hy4 request: two drafting requests join as plain rows, then the survivor resumes DFly2 with its solo tokens (HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const dir = std.mem.span(std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest);
    const ids = try hy4_contract.prompt(a, dir);
    defer a.free(ids);
    var ld: Load = undefined;
    try ld.open(ResidentHy4, dir, true);
    defer _ = ld.close();
    defer static_bill.close(&ld.config);
    var sv: Served = undefined;
    try sv.init(&ld, dir);
    defer sv.deinit();
    sv.sch.queue_cap = 4;
    const xfm = @import("../arch/handle.zig").trunk(sv.entry.handle.?);
    if (xfm.arch_state.hy4flash.drafter == null) return error.SkipZigTest;
    const prompts = [_][]const u32{ ids[0..@min(ids.len, 16)], ids[4..@min(ids.len, 24)] };
    // The first ends early: the other is alone again and resumes DFly2.
    const max_toks = [2]u32{ 5, 14 };
    var solos: [2]SoloResult = undefined;
    var solos_made: usize = 0;
    defer for (0..solos_made) |i| {
        solos[i].tokens.deinit(a);
        solos[i].state.deinit(a);
    };
    for (0..2) |i| {
        solos[i] = try soloRun(&sv, prompts[i], max_toks[i], true);
        solos_made = i + 1;
    }
    hy4_state.joined_ticks_for_test = 0;
    hy4_sched_knobs.group_ticks_for_test = 0;
    var slots: [2]*scheduler.Slot = undefined;
    var made: usize = 0;
    defer for (0..made) |i| endSlot(&sv, slots[i]);
    for (0..2) |i| {
        slots[i] = try submitPending(&sv, prompts[i], max_toks[i], true);
        made = i + 1;
        try scheduler.runPrefill(&sv.sch, slots[i]);
        try sv.sch.decoding.append(a, slots[i]);
    }
    var both = [_]*scheduler.Slot{ slots[0], slots[1] };
    try scheduler.runDecodeTick(&sv.sch, &both);
    errdefer std.debug.print("joined ticks {d} after the first tick of two drafting requests\n", .{hy4_state.joined_ticks_for_test});
    try testing.expect(hy4_state.joined_ticks_for_test > 0);
    try testing.expectEqual(@as(u64, 0), hy4_sched_knobs.group_ticks_for_test);
    scheduler.cullDecoding(&sv.sch);
    var ticks: usize = 0;
    while (ticks < 96) : (ticks += 1) {
        var act: [2]*scheduler.Slot = undefined;
        var k: usize = 0;
        for (0..2) |i| if (slots[i].joinsTick()) {
            act[k] = slots[i];
            k += 1;
        };
        if (k == 0) break;
        try scheduler.runDecodeTick(&sv.sch, act[0..k]);
        scheduler.cullDecoding(&sv.sch);
    }
    try testing.expect(ticks < 96);
    try testing.expectEqual(@as(u64, 0), hy4_sched_knobs.group_ticks_for_test);
    for (0..2) |i| {
        try testing.expect(slots[i].error_code == null);
        try testing.expectEqualSlices(u32, solos[i].tokens.items, slots[i].legacy_gen.?.generated_ids.items);
    }
    var drafted: u32 = 0;
    for (slots[1].legacy_gen.?.drafter_stats.rows_hist[2..]) |c| drafted += c;
    try testing.expect(drafted > 0);
}

test "hy4 request: three drafting requests join as plain rows, then the last resumes DFly2 with its solo tokens (HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const dir = std.mem.span(std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest);
    const ids = try hy4_contract.prompt(a, dir);
    defer a.free(ids);
    var ld: Load = undefined;
    try ld.open(ResidentHy4, dir, true);
    defer _ = ld.close();
    defer static_bill.close(&ld.config);
    var sv: Served = undefined;
    try sv.init(&ld, dir);
    defer sv.deinit();
    sv.sch.queue_cap = 8;
    const xfm = @import("../arch/handle.zig").trunk(sv.entry.handle.?);
    const m = xfm.arch_state.hy4flash.model.?;
    try testing.expect(m.serves_requests);
    if (xfm.arch_state.hy4flash.drafter == null) return error.SkipZigTest;
    const prompts = [_][]const u32{
        ids[0..@min(ids.len, 16)],
        ids[4..@min(ids.len, 24)],
        ids[8..@min(ids.len, 32)],
    };
    // The first two end early, one after the other: company drops to one and the last resumes DFly2.
    const max_toks = [3]u32{ 5, 9, 14 };
    var solos: [3]SoloResult = undefined;
    var solos_made: usize = 0;
    defer for (0..solos_made) |i| {
        solos[i].tokens.deinit(a);
        solos[i].state.deinit(a);
    };
    for (0..3) |i| {
        solos[i] = try soloRun(&sv, prompts[i], max_toks[i], true);
        solos_made = i + 1;
    }
    hy4_state.joined_ticks_for_test = 0;
    hy4_sched_knobs.group_ticks_for_test = 0;
    var slots: [3]*scheduler.Slot = undefined;
    var made: usize = 0;
    defer for (0..made) |i| endSlot(&sv, slots[i]);
    var bills: [3]u64 = undefined;
    var grown: [3]bool = @splat(false);
    for (0..3) |i| {
        slots[i] = try submitPending(&sv, prompts[i], max_toks[i], true);
        made = i + 1;
        try scheduler.runPrefill(&sv.sch, slots[i]);
        try sv.sch.decoding.append(a, slots[i]);
        bills[i] = slots[i].ledger.?.state();
    }
    var ticks: usize = 0;
    while (ticks < 96) : (ticks += 1) {
        var act: [3]*scheduler.Slot = undefined;
        var k: usize = 0;
        for (0..3) |i| if (slots[i].joinsTick()) {
            act[k] = slots[i];
            k += 1;
        };
        if (k == 0) break;
        try scheduler.runDecodeTick(&sv.sch, act[0..k]);
        for (0..made) |i| if (!grown[i] and !slots[i].joinsTick()) {
            grown[i] = true;
            const bill = slots[i].ledger.?.state();
            if (bill > 0) try testing.expectEqual(bills[i], bill);
        };
        for (0..made) |i| if (slots[i].joinsTick()) try testing.expectEqual(bills[i], slots[i].ledger.?.state());
        scheduler.cullDecoding(&sv.sch);
    }
    try testing.expect(hy4_state.joined_ticks_for_test > 0);
    // Drafting requests in company decode as plain rows: no tick verified a group.
    try testing.expectEqual(@as(u64, 0), hy4_sched_knobs.group_ticks_for_test);
    for (0..3) |i| {
        try testing.expect(grown[i]);
        try testing.expect(slots[i].error_code == null);
        try testing.expectEqualSlices(u32, solos[i].tokens.items, slots[i].legacy_gen.?.generated_ids.items);
        // The last resumed DFly2 once it decoded alone.
        var drafted: u32 = 0;
        for (slots[i].legacy_gen.?.drafter_stats.rows_hist[2..]) |c| drafted += c;
        if (i == 2) try testing.expect(drafted > 0);
    }
}


test "hy4 request: three drafting requests in company verify their drafts as groups and say their solo DFly2 tokens (HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const dir = std.mem.span(std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest);
    const ids = try hy4_contract.prompt(a, dir);
    defer a.free(ids);
    var ld: Load = undefined;
    try ld.open(ResidentHy4, dir, true);
    defer _ = ld.close();
    defer static_bill.close(&ld.config);
    var sv: Served = undefined;
    try sv.init(&ld, dir);
    defer sv.deinit();
    sv.sch.queue_cap = 8;
    const xfm = @import("../arch/handle.zig").trunk(sv.entry.handle.?);
    const m = xfm.arch_state.hy4flash.model.?;
    if (xfm.arch_state.hy4flash.drafter == null) return error.SkipZigTest;
    const prompts = [_][]const u32{
        ids[0..@min(ids.len, 16)],
        ids[4..@min(ids.len, 24)],
        ids[8..@min(ids.len, 32)],
    };
    const max_toks: u32 = 10;
    var solos: [3]SoloResult = undefined;
    var solos_made: usize = 0;
    defer for (0..solos_made) |i| {
        solos[i].tokens.deinit(a);
        solos[i].state.deinit(a);
    };
    for (0..3) |i| {
        solos[i] = try soloRun(&sv, prompts[i], max_toks, true);
        solos_made = i + 1;
    }
    const turn = @import("../test_support.zig").openTurn(ld.h, &ld.config);
    defer turn.end();
    hy4_state.joined_ticks_for_test = 0;
    var slots: [3]*scheduler.Slot = undefined;
    var made: usize = 0;
    defer for (0..made) |i| endSlot(&sv, slots[i]);
    for (0..3) |i| {
        slots[i] = try submitPending(&sv, prompts[i], max_toks, true);
        made = i + 1;
        try scheduler.runPrefill(&sv.sch, slots[i]);
        try sv.sch.decoding.append(a, slots[i]);
    }
    try testing.expect(m.plans != null);
    const hy4_sched = @import("hy4_sched.zig");
    var streams: [3]hy4_sched_knobs.CopyStreamForTest = undefined;
    for (0..3) |i| streams[i] = .{ .request = slots[i].request.?.as(.hy4flash).?, .tokens = solos[i].tokens.items };
    hy4_sched_knobs.copy_streams_for_test = &streams;
    defer hy4_sched_knobs.copy_streams_for_test = null;
    const core = @import("../tensor/core.zig");
    hy4_sched_knobs.groups_on_for_test = true;
    defer hy4_sched_knobs.groups_on_for_test = false;
    hy4_sched_knobs.group_ticks_for_test = 0;
    hy4_sched_knobs.groups_refused_for_test = 0;
    const before = core.stats();
    var rounds: usize = 0;
    while (rounds < 64) : (rounds += 1) {
        var active: [3]*scheduler.Slot = undefined;
        var k: usize = 0;
        for (0..3) |i| if (slots[i].joinsTick()) {
            active[k] = slots[i];
            k += 1;
        };
        if (k == 0) break;
        var live: [3]*scheduler.Slot = undefined;
        const n = try scheduler.batchedTickRows(&sv.sch, active[0..k], &live);
        if (n == 0) continue;
        var batch: []*scheduler.Slot = live[0..n];
        var toks: [3]u32 = undefined;
        var ctxs: [3]*@import("../trunk.zig").ForwardCtx = undefined;
        for (batch, 0..) |slot, i| {
            toks[i] = slot.legacy_gen.?.next_token_id;
            ctxs[i] = &slot.legacy_gen.?.ctx;
        }
        @import("../arch/sched.zig").enterJoinedRows(xfm);
        const out = try hy4_sched.batchedForward(&sv.sch, a, xfm, &batch, toks[0..n], ctxs[0..n], &.{}, .{ .moe_positions = false, .advances_offset = true }, null);
        for (out) |o| _ = mlx.mlx_array_free(o);
        a.free(out);
        scheduler.cullDecoding(&sv.sch);
    }
    try testing.expect(rounds < 64);
    try testing.expect(hy4_state.joined_ticks_for_test > 0);
    const after = core.stats();
    errdefer std.debug.print("group ticks {d}: exec {d} wide {d} miss {any} refused {d}\n", .{ hy4_sched_knobs.group_ticks_for_test, after.executions - before.executions, after.wide_executions - before.wide_executions, core.executionMiss(), hy4_sched_knobs.groups_refused_for_test });
    try testing.expect(hy4_sched_knobs.group_ticks_for_test > 0);
    try testing.expectEqual(@as(u64, 0), hy4_sched_knobs.groups_refused_for_test);
    try testing.expect(after.executions - before.executions > 0);
    try testing.expectEqual(@as(u64, 0), after.wide_executions - before.wide_executions);
    try testing.expectEqual(@as(?@import("../tensor/plan/plan_binder.zig").Miss, null), core.executionMiss());
    for (0..3) |i| {
        const gen = &slots[i].legacy_gen.?;
        errdefer std.debug.print("slot {d}: error {?s}, {d} tokens, {} accepted over {d} rounds, hist {any}\n", .{ i, slots[i].error_code, gen.generated_ids.items.len, gen.drafter_stats.accepted, gen.drafter_stats.rounds, gen.drafter_stats.rows_hist[1..9] });
        try testing.expectEqual(@as(?[]const u8, null), slots[i].error_code);
        try testing.expectEqualSlices(u32, solos[i].tokens.items, gen.generated_ids.items);
        var drafted: u32 = 0;
        for (gen.drafter_stats.rows_hist[2..]) |c| drafted += c;
        try testing.expect(drafted > 0);
        try testing.expectEqual(@as(u32, 0), gen.drafter_stats.plain);
        try testing.expect(gen.drafter_stats.accepted > 0);
        try testing.expect(gen.generated_ids.items.len > gen.drafter_stats.rounds);
    }
}


fn expectBlockShapes(m: *const hy4_model.Model, slots: []const *scheduler.Slot) !void {
    const want = [2]c_int{ m.logits_rows, @intCast(m.config.vocab) };
    for (slots) |slot| {
        const r = slot.request.?.as(.hy4flash).?;
        for (r.logits_fixed) |b| try testing.expectEqualSlices(c_int, &want, mlx.getShape(b.?));
    }
}

test "hy4 request: a request's logits blocks keep one shape across a group tick, a joined tick and a solo forward, none asking the pool (HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const dir = std.mem.span(std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest);
    const ids = try hy4_contract.prompt(a, dir);
    defer a.free(ids);
    var ld: Load = undefined;
    try ld.open(ResidentHy4, dir, true);
    defer _ = ld.close();
    defer static_bill.close(&ld.config);
    var sv: Served = undefined;
    try sv.init(&ld, dir);
    defer sv.deinit();
    sv.sch.queue_cap = 8;
    const xfm = @import("../arch/handle.zig").trunk(sv.entry.handle.?);
    const m = xfm.arch_state.hy4flash.model.?;
    if (xfm.arch_state.hy4flash.drafter == null) return error.SkipZigTest;
    // The joined cap above a verify block's rows is what makes the two shapes differ.
    if (m.logits_rows <= @import("hy4_forward.zig").verify_rows_max) return error.SkipZigTest;
    const prompts = [_][]const u32{ ids[0..@min(ids.len, 16)], ids[4..@min(ids.len, 24)], ids[8..@min(ids.len, 32)] };
    const max_toks: u32 = 12;
    var solos: [3]SoloResult = undefined;
    var solos_made: usize = 0;
    defer for (0..solos_made) |i| {
        solos[i].tokens.deinit(a);
        solos[i].state.deinit(a);
    };
    for (0..3) |i| {
        solos[i] = try soloRun(&sv, prompts[i], max_toks, true);
        solos_made = i + 1;
    }
    const turn = @import("../test_support.zig").openTurn(ld.h, &ld.config);
    defer turn.end();
    var slots: [3]*scheduler.Slot = undefined;
    var made: usize = 0;
    defer for (0..made) |i| endSlot(&sv, slots[i]);
    for (0..3) |i| {
        slots[i] = try submitPending(&sv, prompts[i], max_toks, true);
        made = i + 1;
        try scheduler.runPrefill(&sv.sch, slots[i]);
        try sv.sch.decoding.append(a, slots[i]);
    }
    try testing.expect(m.plans != null);
    const hy4_sched = @import("hy4_sched.zig");
    const core = @import("../tensor/core.zig");
    // A group tick.
    hy4_sched_knobs.groups_on_for_test = true;
    defer hy4_sched_knobs.groups_on_for_test = false;
    hy4_sched_knobs.group_ticks_for_test = 0;
    {
        var batch: []*scheduler.Slot = slots[0..];
        var toks: [3]u32 = undefined;
        var ctxs: [3]*@import("../trunk.zig").ForwardCtx = undefined;
        for (batch, 0..) |slot, i| {
            toks[i] = slot.legacy_gen.?.next_token_id;
            ctxs[i] = &slot.legacy_gen.?.ctx;
        }
        @import("../arch/sched.zig").enterJoinedRows(xfm);
        const out = try hy4_sched.batchedForward(&sv.sch, a, xfm, &batch, &toks, &ctxs, &.{}, .{ .moe_positions = false, .advances_offset = true }, null);
        for (out) |o| _ = mlx.mlx_array_free(o);
        a.free(out);
    }
    try testing.expectEqual(@as(u64, 1), hy4_sched_knobs.group_ticks_for_test);
    try expectBlockShapes(m, &slots);
    // A plain joined tick, then one request's solo planned forward.
    hy4_sched_knobs.groups_on_for_test = false;
    try scheduler.runDecodeTick(&sv.sch, slots[0..3]);
    try expectBlockShapes(m, &slots);
    try testing.expectEqual(@as(?@import("../tensor/plan/plan_binder.zig").Miss, null), core.executionMiss());
    try scheduler.runDecodeTick(&sv.sch, slots[0..1]);
    try expectBlockShapes(m, &slots);
    try testing.expectEqual(@as(?@import("../tensor/plan/plan_binder.zig").Miss, null), core.executionMiss());
    // Group rounds through the scheduler's own tick, inside the sealed turn: no wide phase, no ask past the plan.
    hy4_sched_knobs.groups_on_for_test = true;
    const wide_before = core.stats().wide_executions;
    for (0..2) |_| {
        const was_stats = core.stats();
        try scheduler.runDecodeTick(&sv.sch, slots[0..3]);
        const now = core.stats();
        // The plan's own scratch asks stay far under a logits block: a round that made one asks its bytes more.
        const block_bytes: u64 = @as(u64, @intCast(m.logits_rows)) * m.config.vocab * 4;
        try testing.expect(now.ask_bytes - was_stats.ask_bytes < block_bytes);
        try testing.expect(now.pool_bytes -| was_stats.pool_bytes < block_bytes);
        try testing.expectEqual(was_stats.closed_asks, now.closed_asks);
        try testing.expectEqual(was_stats.wide_asks, now.wide_asks);
        try expectBlockShapes(m, &slots);
        try testing.expectEqual(@as(?@import("../tensor/plan/plan_binder.zig").Miss, null), core.executionMiss());
    }
    try testing.expect(hy4_sched_knobs.group_ticks_for_test > 1);
    try testing.expectEqual(wide_before, core.stats().wide_executions);
    hy4_sched_knobs.groups_on_for_test = false;
    var ticks: usize = 0;
    while (ticks < 96) : (ticks += 1) {
        var act: [3]*scheduler.Slot = undefined;
        var k: usize = 0;
        for (0..3) |i| if (slots[i].joinsTick()) {
            act[k] = slots[i];
            k += 1;
        };
        if (k == 0) break;
        try scheduler.runDecodeTick(&sv.sch, act[0..k]);
        scheduler.cullDecoding(&sv.sch);
    }
    try testing.expectEqual(@as(?@import("../tensor/plan/plan_binder.zig").Miss, null), core.executionMiss());
    for (0..3) |i| {
        try testing.expectEqual(@as(?[]const u8, null), slots[i].error_code);
        try testing.expectEqualSlices(u32, solos[i].tokens.items, slots[i].legacy_gen.?.generated_ids.items);
    }
}

/// Three drafting requests decode in group ticks whose drafts equal their solo streams (every draft accepted), so a round
/// commits as many rows as it verifies; `end_at` makes each request's end-of-turn id its solo stream's token at that index.
fn groupRoundsCut(max_toks: u32, end_at: ?usize) !void {
    try mlx.requireGpu();
    const a = testing.allocator;
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const dir = std.mem.span(std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest);
    const ids = try hy4_contract.prompt(a, dir);
    defer a.free(ids);
    var ld: Load = undefined;
    try ld.open(ResidentHy4, dir, true);
    defer _ = ld.close();
    defer static_bill.close(&ld.config);
    var sv: Served = undefined;
    try sv.init(&ld, dir);
    defer sv.deinit();
    sv.sch.queue_cap = 8;
    const xfm = @import("../arch/handle.zig").trunk(sv.entry.handle.?);
    const m = xfm.arch_state.hy4flash.model.?;
    if (xfm.arch_state.hy4flash.drafter == null) return error.SkipZigTest;
    const prompts = [_][]const u32{ ids[0..@min(ids.len, 16)], ids[4..@min(ids.len, 24)], ids[8..@min(ids.len, 32)] };
    var solos: [3]SoloResult = undefined;
    var solos_made: usize = 0;
    defer for (0..solos_made) |i| {
        solos[i].tokens.deinit(a);
        solos[i].state.deinit(a);
    };
    for (0..3) |i| {
        solos[i] = try soloRun(&sv, prompts[i], 12, true);
        solos_made = i + 1;
    }
    const turn = @import("../test_support.zig").openTurn(ld.h, &ld.config);
    defer turn.end();
    var slots: [3]*scheduler.Slot = undefined;
    var made: usize = 0;
    defer for (0..made) |i| endSlot(&sv, slots[i]);
    for (0..3) |i| {
        slots[i] = try submitPending(&sv, prompts[i], max_toks, true);
        made = i + 1;
        try scheduler.runPrefill(&sv.sch, slots[i]);
        try sv.sch.decoding.append(a, slots[i]);
        if (end_at) |at| {
            slots[i].allocator.free(slots[i].eos_token_ids);
            slots[i].eos_token_ids = try slots[i].allocator.dupe(u32, &.{solos[i].tokens.items[at]});
            slots[i].legacy_gen.?.eos_token_ids = slots[i].eos_token_ids;
        }
    }
    try testing.expect(m.plans != null);
    var streams: [3]hy4_sched_knobs.CopyStreamForTest = undefined;
    for (0..3) |i| streams[i] = .{ .request = slots[i].request.?.as(.hy4flash).?, .tokens = solos[i].tokens.items };
    hy4_sched_knobs.copy_streams_for_test = &streams;
    defer hy4_sched_knobs.copy_streams_for_test = null;
    hy4_sched_knobs.groups_on_for_test = true;
    defer hy4_sched_knobs.groups_on_for_test = false;
    hy4_sched_knobs.group_ticks_for_test = 0;
    var rounds: usize = 0;
    while (rounds < 64) : (rounds += 1) {
        var active: [3]*scheduler.Slot = undefined;
        var k: usize = 0;
        for (0..3) |i| if (slots[i].joinsTick()) {
            active[k] = slots[i];
            k += 1;
        };
        if (k == 0) break;
        try scheduler.runDecodeTick(&sv.sch, active[0..k]);
        scheduler.cullDecoding(&sv.sch);
    }
    try testing.expect(hy4_sched_knobs.group_ticks_for_test > 0);
    for (0..3) |i| {
        const gen = &slots[i].legacy_gen.?;
        try testing.expectEqual(@as(?[]const u8, null), slots[i].error_code);
        // The serial round stops its verify width at the budget and at an end-of-turn draft: the generator's tokens end there.
        const want: usize = if (end_at) |at| std.mem.indexOfScalar(u32, solos[i].tokens.items, solos[i].tokens.items[at]).? + 1 else max_toks;
        try testing.expectEqualSlices(u32, solos[i].tokens.items[0..want], gen.generated_ids.items);
        if (end_at == null) try testing.expectEqual(slots[i].completion_tokens, gen.completion_tokens);
    }
}

test "hy4 request: a group round stops at max_tokens where the serial round does, the generator's count the slot's (HY4_STQ_TINY)" {
    try groupRoundsCut(6, null);
}

test "hy4 request: a group round ends on an end-of-turn draft and verifies nothing past it (HY4_STQ_TINY)" {
    try groupRoundsCut(12, 2);
}

fn submitWith(sv: *Served, prompt: []const u32, n: u32, sampling: @import("../sampler.zig").SamplingParams) !*scheduler.Slot {
    const slot = try sv.sch.submit(.{
        .model = sv.entry,
        .prompt_ids = prompt,
        .sampling = sampling,
        .eos_token_ids = &.{},
        .max_tokens = n,
        .enable_mtp = false,
        .mtp = null,
    });
    _ = sv.sch.pending.orderedRemove(0);
    return slot;
}

fn soloRunWith(sv: *Served, prompt: []const u32, n: u32, sampling: @import("../sampler.zig").SamplingParams) !SoloResult {
    const a = testing.allocator;
    const xfm = @import("../arch/handle.zig").trunk(sv.entry.handle.?);
    const m = xfm.arch_state.hy4flash.model.?;
    const slot = try submitWith(sv, prompt, n, sampling);
    defer endSlot(sv, slot);
    try scheduler.runPrefill(&sv.sch, slot);
    if (!scheduler.postPrefillTerminal(&sv.sch, slot)) {
        try sv.sch.decoding.append(a, slot);
        var one = [_]*scheduler.Slot{slot};
        while (slot.joinsTick()) try scheduler.runDecodeTick(&sv.sch, &one);
    }
    if (slot.error_code != null) return error.RequestFailed;
    const req = if (slot.request) |*r| r.as(.hy4flash) else null;
    var state = try captureRequestBytes(a, m, req orelse return error.TestExpectedSlotRequest);
    errdefer state.deinit(a);
    var tokens: std.ArrayList(u32) = .empty;
    errdefer tokens.deinit(a);
    try tokens.appendSlice(a, slot.legacy_gen.?.generated_ids.items);
    scheduler.cullDecoding(&sv.sch);
    return .{ .tokens = tokens, .state = state };
}

const JoinedResident = struct {
    pub fn prepare(a: std.mem.Allocator, config: *@import("../model.zig").ModelConfig, dir: []const u8) !void {
        try hy4_contract.prepare(a, config, dir);
        config.expert_resident = true;
        config.expert_cache_bytes = std.math.maxInt(u64) / 2;
    }
};

test "hy4 request: a joined tick draws each row through its own planned sample root, greedy and top-p alike (HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const dir = std.mem.span(std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest);
    const ids = try hy4_contract.prompt(a, dir);
    defer a.free(ids);
    var ld: Load = undefined;
    try ld.open(JoinedResident, dir, true);
    defer _ = ld.close();
    defer static_bill.close(&ld.config);
    try testing.expect(sched.batches(&ld.config));
    var sv: Served = undefined;
    try sv.init(&ld, dir);
    defer sv.deinit();
    sv.sch.queue_cap = 4;
    const xfm = @import("../arch/handle.zig").trunk(sv.entry.handle.?);
    const m = xfm.arch_state.hy4flash.model.?;
    try testing.expect(sched.modelBatches(xfm));
    const p = m.plans.?;
    try testing.expect(p.sample.planned);
    const greedy: @import("../sampler.zig").SamplingParams = .{ .temperature = 0 };
    const top_p: @import("../sampler.zig").SamplingParams = .{ .temperature = 0.8, .top_p = 0.9, .top_k = 40, .seed = 7 };
    const prompt_a = ids[0..@min(ids.len, 24)];
    const prompt_b = ids[5..@min(ids.len, 35)];
    var solo_a = try soloRunWith(&sv, prompt_a, 8, greedy);
    defer solo_a.tokens.deinit(a);
    defer solo_a.state.deinit(a);
    var solo_b = try soloRunWith(&sv, prompt_b, 8, top_p);
    defer solo_b.tokens.deinit(a);
    defer solo_b.state.deinit(a);
    const slot_a = try submitWith(&sv, prompt_a, 8, greedy);
    defer endSlot(&sv, slot_a);
    const slot_b = try submitWith(&sv, prompt_b, 8, top_p);
    defer endSlot(&sv, slot_b);
    try scheduler.runPrefill(&sv.sch, slot_a);
    try sv.sch.decoding.append(a, slot_a);
    try scheduler.runPrefill(&sv.sch, slot_b);
    try sv.sch.decoding.append(a, slot_b);
    const draws = p.sample.draws;
    const misses = p.sample.misses;
    const joined = hy4_state.joined_ticks_for_test;
    var total_draws: u64 = 0;
    while (slot_a.joinsTick() or slot_b.joinsTick()) {
        var active: [2]*scheduler.Slot = undefined;
        var k: usize = 0;
        if (slot_a.joinsTick()) {
            active[k] = slot_a;
            k += 1;
        }
        if (slot_b.joinsTick()) {
            active[k] = slot_b;
            k += 1;
        }
        try scheduler.runDecodeTick(&sv.sch, active[0..k]);
        scheduler.cullDecoding(&sv.sch);
        total_draws = p.sample.draws - draws;
    }
    const ticks = hy4_state.joined_ticks_for_test - joined;
    try testing.expect(ticks > 0);
    // Every joined row's draw is an execution of its own mix's root.
    try testing.expectEqual(2 * ticks, total_draws);
    try testing.expectEqual(misses, p.sample.misses);
    try testing.expect(slot_a.error_code == null and slot_b.error_code == null);
    try testing.expectEqualSlices(u32, solo_a.tokens.items, slot_a.legacy_gen.?.generated_ids.items);
    try testing.expectEqualSlices(u32, solo_b.tokens.items, slot_b.legacy_gen.?.generated_ids.items);
}

test "hy4 request: a joined tick whose shorter row ends in the drain leaves the survivor its solo tokens (HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const dir = std.mem.span(std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest);
    const ids = try hy4_contract.prompt(a, dir);
    defer a.free(ids);
    var ld: Load = undefined;
    try ld.open(JoinedResident, dir, true);
    defer _ = ld.close();
    defer static_bill.close(&ld.config);
    var sv: Served = undefined;
    try sv.init(&ld, dir);
    defer sv.deinit();
    sv.sch.queue_cap = 4;
    const xfm = @import("../arch/handle.zig").trunk(sv.entry.handle.?);
    try testing.expect(sched.modelBatches(xfm));
    const greedy: @import("../sampler.zig").SamplingParams = .{ .temperature = 0 };
    const prompt_a = ids[0..@min(ids.len, 24)];
    const prompt_b = ids[5..@min(ids.len, 35)];
    var solo_a = try soloRunWith(&sv, prompt_a, 3, greedy);
    defer solo_a.tokens.deinit(a);
    defer solo_a.state.deinit(a);
    var solo_b = try soloRunWith(&sv, prompt_b, 9, greedy);
    defer solo_b.tokens.deinit(a);
    defer solo_b.state.deinit(a);
    const slot_a = try submitWith(&sv, prompt_a, 3, greedy);
    defer endSlot(&sv, slot_a);
    try scheduler.runPrefill(&sv.sch, slot_a);
    try sv.sch.decoding.append(a, slot_a);
    // Two single ticks leave the short row at its last token, so the join tick's drain ends it.
    var one = [_]*scheduler.Slot{slot_a};
    var early: usize = 0;
    while (early < 2 and slot_a.joinsTick()) : (early += 1) try scheduler.runDecodeTick(&sv.sch, &one);
    const slot_b = try submitWith(&sv, prompt_b, 9, greedy);
    defer endSlot(&sv, slot_b);
    try scheduler.runPrefill(&sv.sch, slot_b);
    try sv.sch.decoding.append(a, slot_b);
    var ticks: usize = 0;
    while ((slot_a.joinsTick() or slot_b.joinsTick()) and ticks < 32) : (ticks += 1) {
        var active: [2]*scheduler.Slot = undefined;
        var k: usize = 0;
        for ([_]*scheduler.Slot{ slot_a, slot_b }) |s| if (s.joinsTick()) {
            active[k] = s;
            k += 1;
        };
        try scheduler.runDecodeTick(&sv.sch, active[0..k]);
        scheduler.cullDecoding(&sv.sch);
    }
    try testing.expect(ticks < 32);
    errdefer std.debug.print("errors A={?s} B={?s}, B ids {any}\n", .{ slot_a.error_code, slot_b.error_code, slot_b.legacy_gen.?.generated_ids.items });
    try testing.expect(slot_a.error_code == null and slot_b.error_code == null);
    try testing.expectEqualSlices(u32, solo_a.tokens.items, slot_a.legacy_gen.?.generated_ids.items);
    try testing.expectEqualSlices(u32, solo_b.tokens.items, slot_b.legacy_gen.?.generated_ids.items);
}

test "hy4 request: two drafting requests whose shorter one ends at the joined tick's entry leave the survivor its solo DFly2 tokens (HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const dir = std.mem.span(std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest);
    const ids = try hy4_contract.prompt(a, dir);
    defer a.free(ids);
    var ld: Load = undefined;
    try ld.open(ResidentHy4, dir, true);
    defer _ = ld.close();
    defer static_bill.close(&ld.config);
    var sv: Served = undefined;
    try sv.init(&ld, dir);
    defer sv.deinit();
    sv.sch.queue_cap = 4;
    const xfm = @import("../arch/handle.zig").trunk(sv.entry.handle.?);
    if (xfm.arch_state.hy4flash.drafter == null) return error.SkipZigTest;
    const prompts = [_][]const u32{ ids[0..@min(ids.len, 16)], ids[4..@min(ids.len, 24)] };
    var solo_long = try soloRun(&sv, prompts[1], 14, true);
    defer solo_long.tokens.deinit(a);
    defer solo_long.state.deinit(a);
    const slots = [2]*scheduler.Slot{ try submitPending(&sv, prompts[0], 14, true), try submitPending(&sv, prompts[1], 14, true) };
    defer for (slots) |s| endSlot(&sv, s);
    for (slots) |s| {
        try scheduler.runPrefill(&sv.sch, s);
        try sv.sch.decoding.append(a, s);
    }
    // A drafting row leaves no pipeline to drain, so the short row's budget is spent as the tick starts: the batch's
    // entry ends it and leaves the other row alone.
    const short = &slots[0].legacy_gen.?;
    short.max_tokens = short.step;
    var ticks: usize = 0;
    while (ticks < 96) : (ticks += 1) {
        var act: [2]*scheduler.Slot = undefined;
        var k: usize = 0;
        for (slots) |s| if (s.joinsTick()) {
            act[k] = s;
            k += 1;
        };
        if (k == 0) break;
        try scheduler.runDecodeTick(&sv.sch, act[0..k]);
        scheduler.cullDecoding(&sv.sch);
    }
    try testing.expect(ticks < 96);
    errdefer std.debug.print("errors {?s} {?s}\n", .{ slots[0].error_code, slots[1].error_code });
    try testing.expect(slots[0].error_code == null and slots[1].error_code == null);
    try testing.expectEqualSlices(u32, solo_long.tokens.items, slots[1].legacy_gen.?.generated_ids.items);
}

test "hy4 request: a drafting row its partner's entry stop leaves alone counts its pending pad once (HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const dir = std.mem.span(std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest);
    const ids = try hy4_contract.prompt(a, dir);
    defer a.free(ids);
    var ld: Load = undefined;
    try ld.open(ResidentHy4, dir, true);
    defer _ = ld.close();
    defer static_bill.close(&ld.config);
    var sv: Served = undefined;
    try sv.init(&ld, dir);
    defer sv.deinit();
    sv.sch.queue_cap = 4;
    const xfm = @import("../arch/handle.zig").trunk(sv.entry.handle.?);
    if (xfm.arch_state.hy4flash.drafter == null) return error.SkipZigTest;
    const prompts = [_][]const u32{ ids[0..@min(ids.len, 16)], ids[4..@min(ids.len, 24)] };
    const slots = [2]*scheduler.Slot{ try submitPending(&sv, prompts[0], 14, true), try submitPending(&sv, prompts[1], 14, true) };
    defer for (slots) |s| endSlot(&sv, s);
    for (slots) |s| {
        try scheduler.runPrefill(&sv.sch, s);
        try sv.sch.decoding.append(a, s);
    }
    const short = &slots[0].legacy_gen.?;
    short.max_tokens = short.step;
    // The survivor's pending token is the second pad of a run: counted twice it would end the request, counted once
    // it leaves a run of two.
    const survivor = &slots[1].legacy_gen.?;
    survivor.next_token_id = 0;
    survivor.consecutive_pad = 1;
    var both = [_]*scheduler.Slot{ slots[0], slots[1] };
    try scheduler.runDecodeTick(&sv.sch, &both);
    errdefer std.debug.print("pad run {d}, finished {}, error {?s}\n", .{ survivor.consecutive_pad, slots[1].finished, slots[1].error_code });
    try testing.expect(slots[0].finished and !slots[1].finished);
    try testing.expectEqual(@as(u32, 2), survivor.consecutive_pad);
    try testing.expect(slots[1].error_code == null);
}

test "hy4 request: after the load, a joined tick at every width the load joins makes no pipeline, drafting rows in company too (HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const dir = std.mem.span(std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest);
    const ids = try hy4_contract.prompt(a, dir);
    defer a.free(ids);
    var ld: Load = undefined;
    try ld.open(ResidentHy4, dir, true);
    defer _ = ld.close();
    defer static_bill.close(&ld.config);
    var sv: Served = undefined;
    try sv.init(&ld, dir);
    defer sv.deinit();
    const rows = @import("hy4_plan.zig").joinedRows(&ld.config);
    if (rows < 2) return error.SkipZigTest;
    const max_rows = @import("sushi_exl3").project_rows.NAX_ROWS_MAX;
    sv.sch.queue_cap = @intCast(rows);
    const mtp = @import("../arch/handle.zig").trunk(sv.entry.handle.?).arch_state.hy4flash.drafter != null;
    // Rows deeper than the window: the ticks take the grouped sliding attention, which the count below proves.
    const hy4_forward = @import("hy4_forward.zig");
    const late = mlx.engine.latePipelines();
    mlx.engine.late_names.clear();
    const grouped = hy4_forward.grouped_chains_for_test;
    var slots: [max_rows]*scheduler.Slot = undefined;
    var made: usize = 0;
    defer for (0..made) |i| endSlot(&sv, slots[i]);
    // Each ends one tick after the one before it, so the tick narrows through every width from `rows` to 2.
    for (0..rows) |i| {
        slots[i] = try submitPending(&sv, ids[i % 4 .. @min(ids.len, 12 + i % 4)], @intCast(2 + i), mtp);
        made = i + 1;
        try scheduler.runPrefill(&sv.sch, slots[i]);
        try sv.sch.decoding.append(a, slots[i]);
    }
    var ticks: usize = 0;
    while (ticks < 4 * max_rows) : (ticks += 1) {
        var act: [max_rows]*scheduler.Slot = undefined;
        var k: usize = 0;
        for (slots[0..made]) |s| if (s.joinsTick()) {
            act[k] = s;
            k += 1;
        };
        if (k == 0) break;
        try scheduler.runDecodeTick(&sv.sch, act[0..k]);
        scheduler.cullDecoding(&sv.sch);
    }
    for (slots[0..made]) |s| try testing.expect(s.error_code == null);
    try testing.expect(hy4_forward.grouped_chains_for_test > grouped);
    errdefer std.debug.print("joined ticks after the load made pipelines:\n{s}\n", .{mlx.engine.late_names.text()});
    try testing.expectEqual(late, mlx.engine.latePipelines());
}

test "hy4 request: after the load, a prompt whose chunk runs one or two rows past a whole span compiles no pipeline (HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const dir = std.mem.span(std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest);
    const ids = try hy4_contract.prompt(a, dir);
    defer a.free(ids);
    var ld: Load = undefined;
    try ld.open(ResidentHy4, dir, true);
    defer _ = ld.close();
    defer static_bill.close(&ld.config);
    var sv: Served = undefined;
    try sv.init(&ld, dir);
    defer sv.deinit();
    const m = @import("../arch/handle.zig").trunk(sv.entry.handle.?).arch_state.hy4flash.model.?;
    var prompt: [300]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = ids[(i * 3 + 1) % ids.len];
    const late = mlx.engine.latePipelines();
    mlx.engine.late_names.clear();
    var restored: u32 = 0;
    // A chunk past a whole span runs its last row or two as a span of their own (`fwd.linearSpans`): at both widths a
    // layer splits a chunk by, before the selection threshold's 512.
    for ([_]usize{ @intCast(@import("hy4_model.zig").gathered_span), @intCast(m.span_rows) }) |span| for ([_]usize{ 1, 2 }) |past| {
        const len = span + past + 1;
        var got = try sv.request(prompt[0..len], 1, &restored);
        got.deinit(a);
        errdefer std.debug.print("a {d}-token prompt ({d} rows past a {d}-row span) made pipelines:\n{s}\n", .{ len, past, span, mlx.engine.late_names.text() });
        try testing.expectEqual(late, mlx.engine.latePipelines());
    };
}

/// A served SSD-only load of the tiny, a prompt of `len` tokens from `ids` at `stride`, asked twice with `n` greedy tokens.
fn repeatOnSsd(dir: []const u8, ids: []const u32, len: usize, stride: usize, n: u32) !struct { first: u32, second: u32, same: bool, points: [8]u32, count: usize } {
    const a = testing.allocator;
    var ld: Load = undefined;
    try ld.open(ResidentHy4, dir, true);
    defer _ = ld.close();
    defer static_bill.close(&ld.config);
    var sv: Served = undefined;
    try sv.init(&ld, dir);
    defer sv.deinit();
    const prompt = try a.alloc(u32, len);
    defer a.free(prompt);
    for (prompt, 0..) |*t, i| t.* = ids[(i * stride + 1) % ids.len];
    var r1: u32 = 0;
    var first = try sv.request(prompt, n, &r1);
    defer first.deinit(a);
    var r2: u32 = 0;
    var second = try sv.request(prompt, n, &r2);
    defer second.deinit(a);
    var out: @TypeOf(repeatOnSsd(dir, ids, len, stride, n) catch unreachable) = .{ .first = r1, .second = r2, .same = std.mem.eql(u32, first.items, second.items), .points = @splat(0), .count = 0 };
    for (sv.entry.prefix_cache.?.disk.?.entries.items) |e| for (e.ssm_positions) |p| if (out.count < out.points.len) {
        out.points[out.count] = p;
        out.count += 1;
    };
    return out;
}

test "hy4 request: a served SSD-first load persists a prompt's end point only at or past the persist floor, and a repeat restores from one it holds (HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const dir = std.mem.span(std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest);
    const ids = try hy4_contract.prompt(a, dir);
    defer a.free(ids);
    const floor = @import("../kv_disk_cache.zig").MIN_PERSIST_TOKENS;
    // A 512-token prompt's end point stands under the floor (the prefill's backoff): it is not persisted, so its repeat
    // re-prefills; only the reply's end, past the repeat's own reach, is written.
    const short = try repeatOnSsd(dir, ids, floor, 3, 2);
    errdefer std.debug.print("512: restored {d} then {d}, points {any}\n", .{ short.first, short.second, short.points[0..short.count] });
    for (short.points[0..short.count]) |p| try testing.expect(p >= floor);
    try testing.expectEqual(@as(u32, 0), short.second);
    try testing.expect(short.same);
    // A 600-token prompt's end point is past the floor: it is persisted and its repeat restores from it.
    const long = try repeatOnSsd(dir, ids, 600, 5, 2);
    errdefer std.debug.print("600: restored {d} then {d}, points {any}\n", .{ long.first, long.second, long.points[0..long.count] });
    try testing.expect(long.second >= floor and long.second < 600);
    try testing.expect(long.same);
}

test "hy4 request: a served reply's end point serves the next turn that adds to the reply; a re-issue of prompt and reply exactly restores at the prompt's end point (HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const dir = std.mem.span(std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest);
    const ids = try hy4_contract.prompt(a, dir);
    defer a.free(ids);
    var ld: Load = undefined;
    try ld.open(ResidentHy4, dir, true);
    defer _ = ld.close();
    defer static_bill.close(&ld.config);
    var sv: Served = undefined;
    try sv.init(&ld, dir);
    defer sv.deinit();
    const prompt = try a.alloc(u32, 600);
    defer a.free(prompt);
    for (prompt, 0..) |*t, i| t.* = ids[(i * 7 + 2) % ids.len];
    var r0: u32 = 0;
    var first = try sv.request(prompt, 4, &r0);
    defer first.deinit(a);
    const end = prompt.len + first.items.len;
    const turn = try a.alloc(u32, end + 1);
    defer a.free(turn);
    @memcpy(turn[0..prompt.len], prompt);
    @memcpy(turn[prompt.len..end], first.items);
    turn[end] = ids[3];
    // The exact re-issue's restore stops one token short of the reply's end point, so it takes the prompt's.
    var exact: u32 = 0;
    var e = try sv.request(turn[0..end], 2, &exact);
    e.deinit(a);
    errdefer std.debug.print("exact re-issue of {d} tokens restored {d}\n", .{ end, exact });
    try testing.expect(exact >= kv_floor() and exact < prompt.len);
    // A turn that adds a token after the reply restores at the reply's end.
    var next: u32 = 0;
    var t = try sv.request(turn, 2, &next);
    t.deinit(a);
    errdefer std.debug.print("the next turn of {d} tokens restored {d}\n", .{ turn.len, next });
    try testing.expect(next >= end);
}

fn kv_floor() u32 {
    return @import("../kv_disk_cache.zig").MIN_PERSIST_TOKENS;
}

test "hy4 request: batched DFly2 draftBatch + grouped verify at N=2, 4, 8 across ragged depths runs on the arena with zero pool asks and zero late pipelines (HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const dir = std.mem.span(std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest);
    const ids = try hy4_contract.prompt(a, dir);
    defer a.free(ids);
    var ld: Load = undefined;
    try ld.open(ResidentHy4, dir, true);
    defer _ = ld.close();
    defer static_bill.close(&ld.config);
    var sv: Served = undefined;
    try sv.init(&ld, dir);
    defer sv.deinit();
    sv.sch.queue_cap = 16;
    const xfm = @import("../arch/handle.zig").trunk(sv.entry.handle.?);
    const m = xfm.arch_state.hy4flash.model.?;
    if (xfm.arch_state.hy4flash.drafter == null) return error.SkipZigTest;
    const d = &xfm.arch_state.hy4flash.drafter.?;
    const core = @import("../tensor/core.zig");

    // 8 slots with ragged depths spanning below and above the tiny fixture's sparse selection threshold.
    const max_n: usize = @min(8, @as(usize, @intCast(@import("hy4_forward.zig").joinedRowsMax())) / 2);
    if (max_n < 2) return error.SkipZigTest;
    const max_toks: u32 = 8;
    var solos: [8]SoloResult = undefined;
    var solos_made: usize = 0;
    defer for (0..solos_made) |i| {
        solos[i].tokens.deinit(a);
        solos[i].state.deinit(a);
    };
    var prompts_buf: [8][48]u32 = undefined;
    var prompts: [8][]const u32 = undefined;
    for (0..max_n) |i| {
        const len: usize = 10 + i * 5;
        for (prompts_buf[i][0..len], 0..) |*t, j| t.* = ids[(j * (i + 3) + i) % ids.len];
        prompts[i] = prompts_buf[i][0..len];
        solos[i] = try soloRun(&sv, prompts[i], max_toks, true);
        solos_made = i + 1;
    }

    const turn = @import("../test_support.zig").openTurn(ld.h, &ld.config);
    defer turn.end();
    var slots: [8]*scheduler.Slot = undefined;
    var made: usize = 0;
    defer for (0..made) |i| endSlot(&sv, slots[i]);
    for (0..max_n) |i| {
        slots[i] = try submitPending(&sv, prompts[i], max_toks, true);
        made = i + 1;
        try scheduler.runPrefill(&sv.sch, slots[i]);
        try sv.sch.decoding.append(a, slots[i]);
    }
    try testing.expect(m.plans != null);

    // Direct `runDraftBatch` at N=2, 4, max_n inside the sealed turn: zero closed/wide asks, zero misses, valid ids/conf.
    const late_before = mlx.engine.latePipelines();
    const st_before = core.stats();
    for ([_]usize{ 2, @min(4, max_n), max_n }) |k| {
        if (k < 2) continue;
        const n_use = @import("hy4_plan.zig").groupDraftWidth(@as(usize, m.config.dfly.draft), @intCast(@import("hy4_forward.zig").joinedRowsMax()), k);
        if (n_use == 0) continue;
        var reqs: [8]*hy4_model.Request = undefined;
        var t1s: [8]u32 = undefined;
        var out: [8]@import("hy4_dfly.zig").Drafter.Draft = undefined;
        for (0..k) |i| {
            reqs[i] = slots[i].request.?.as(.hy4flash).?;
            t1s[i] = slots[i].legacy_gen.?.next_token_id;
        }
        try @import("hy4_plan.zig").runDraftBatch(m, d, reqs[0..k], t1s[0..k], n_use, out[0..k]);
        defer for (out[0..k]) |*dr| dr.deinit();
        for (0..k) |i| {
            try testing.expectEqual(n_use + 1, out[i].ids.len);
            try testing.expectEqual(t1s[i], out[i].ids[0]);
            for (out[i].ids[1..]) |id| try testing.expect(id < m.config.vocab);
            for (out[i].conf) |cf| try testing.expect(std.math.isFinite(cf) and cf >= 0 and cf <= 1);
        }
    }
    const st_after = core.stats();
    try testing.expectEqual(st_before.closed_asks, st_after.closed_asks);
    try testing.expectEqual(st_before.wide_asks, st_after.wide_asks);
    try testing.expectEqual(st_before.wide_executions, st_after.wide_executions);
    try testing.expectEqual(@as(?@import("../tensor/plan/plan_binder.zig").Miss, null), core.executionMiss());

    // Now drive all `max_n` slots to completion under grouped DFly2: every slot matches solo tokens bit-for-bit!
    hy4_sched_knobs.groups_on_for_test = true;
    defer hy4_sched_knobs.groups_on_for_test = false;
    hy4_sched_knobs.group_ticks_for_test = 0;
    var rounds: usize = 0;
    while (rounds < 64) : (rounds += 1) {
        var act: [8]*scheduler.Slot = undefined;
        var k: usize = 0;
        for (0..max_n) |i| if (slots[i].joinsTick()) {
            act[k] = slots[i];
            k += 1;
        };
        if (k == 0) break;
        try scheduler.runDecodeTick(&sv.sch, act[0..k]);
        scheduler.cullDecoding(&sv.sch);
    }
    try testing.expect(hy4_sched_knobs.group_ticks_for_test > 0);
    try testing.expectEqual(late_before, mlx.engine.latePipelines());
    try testing.expectEqual(@as(?@import("../tensor/plan/plan_binder.zig").Miss, null), core.executionMiss());
    for (0..max_n) |i| {
        try testing.expectEqual(@as(?[]const u8, null), slots[i].error_code);
        try testing.expectEqualSlices(u32, solos[i].tokens.items, slots[i].legacy_gen.?.generated_ids.items);
    }
}

test "hy4 request: a slot without handover past the sparse threshold joins grouped DFly2 with zero PlanMiss and matches solo (HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const dir = std.mem.span(std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest);
    const ids = try hy4_contract.prompt(a, dir);
    defer a.free(ids);
    var ld: Load = undefined;
    try ld.open(ResidentHy4, dir, true);
    defer _ = ld.close();
    defer static_bill.close(&ld.config);
    var sv: Served = undefined;
    try sv.init(&ld, dir);
    defer sv.deinit();
    sv.sch.queue_cap = 8;
    const xfm = @import("../arch/handle.zig").trunk(sv.entry.handle.?);
    const m = xfm.arch_state.hy4flash.model.?;
    if (xfm.arch_state.hy4flash.drafter == null) return error.SkipZigTest;
    const core = @import("../tensor/core.zig");

    const thr: usize = @as(usize, m.config.indexer.group_topk) * m.config.indexer.group;
    const prompt_len: usize = @max(thr + 6, 16);
    var p0_buf: [64]u32 = undefined;
    var p1_buf: [64]u32 = undefined;
    if (prompt_len > p0_buf.len) return error.SkipZigTest;
    for (0..prompt_len) |j| {
        p0_buf[j] = ids[(j * 3 + 1) % ids.len];
        p1_buf[j] = ids[(j * 5 + 2) % ids.len];
    }
    const max_toks: u32 = 8;
    var solo0 = try soloRun(&sv, p0_buf[0..prompt_len], max_toks, true);
    defer {
        solo0.tokens.deinit(a);
        solo0.state.deinit(a);
    }
    var solo1 = try soloRun(&sv, p1_buf[0..prompt_len], max_toks, true);
    defer {
        solo1.tokens.deinit(a);
        solo1.state.deinit(a);
    }

    const turn = @import("../test_support.zig").openTurn(ld.h, &ld.config);
    defer turn.end();
    var slot0 = try submitPending(&sv, p0_buf[0..prompt_len], max_toks, true);
    defer endSlot(&sv, slot0);
    var slot1 = try submitPending(&sv, p1_buf[0..prompt_len], max_toks, true);
    defer endSlot(&sv, slot1);
    try scheduler.runPrefill(&sv.sch, slot0);
    try scheduler.runPrefill(&sv.sch, slot1);
    try sv.sch.decoding.append(a, slot0);
    try sv.sch.decoding.append(a, slot1);

    // Simulate slot0 having completed a `keep_hidden = false` plain-pipeline step past `thr` before joining slot1:
    // drop its handover (`last_hidden` and sparse `last_picks`), leaving slot1 with intact handover so `[1, budget]`
    // must permute to `[budget, 1]` and refresh slot0's handover without hitting `PlanMiss`.
    const req0 = slot0.request.?.as(.hy4flash).?;
    const req1 = slot1.request.?.as(.hy4flash).?;
    req0.dropHidden();
    req0.dropPicks();
    try testing.expect(!m.hasHandover(req0));
    try testing.expect(m.hasHandover(req1));

    const misses_before = m.plans.?.misses;
    const late_before = mlx.engine.latePipelines();
    hy4_sched_knobs.groups_on_for_test = true;
    defer hy4_sched_knobs.groups_on_for_test = false;
    hy4_sched_knobs.group_ticks_for_test = 0;

    // First grouped tick: slot0 rides as plain 1 row (refreshing handover), slot1 verifies at full budget.
    var pair = [_]*scheduler.Slot{ slot0, slot1 };
    try scheduler.runDecodeTick(&sv.sch, &pair);
    scheduler.cullDecoding(&sv.sch);
    try testing.expectEqual(misses_before, m.plans.?.misses);
    try testing.expectEqual(@as(?@import("../tensor/plan/plan_binder.zig").Miss, null), core.executionMiss());
    try testing.expect(m.hasHandover(req0));
    try testing.expect(hy4_sched_knobs.group_ticks_for_test >= 1);

    // Drive both slots to completion under grouped DFly2.
    var rounds: usize = 0;
    while (rounds < 32) : (rounds += 1) {
        var act: [2]*scheduler.Slot = undefined;
        var k: usize = 0;
        for ([_]*scheduler.Slot{ slot0, slot1 }) |sl| if (sl.joinsTick()) {
            act[k] = sl;
            k += 1;
        };
        if (k == 0) break;
        try scheduler.runDecodeTick(&sv.sch, act[0..k]);
        scheduler.cullDecoding(&sv.sch);
    }
    try testing.expectEqual(misses_before, m.plans.?.misses);
    try testing.expectEqual(late_before, mlx.engine.latePipelines());
    try testing.expectEqual(@as(?[]const u8, null), slot0.error_code);
    try testing.expectEqual(@as(?[]const u8, null), slot1.error_code);
    try testing.expectEqualSlices(u32, solo0.tokens.items, slot0.legacy_gen.?.generated_ids.items);
    try testing.expectEqualSlices(u32, solo1.tokens.items, slot1.legacy_gen.?.generated_ids.items);
}

const StreamedHy4 = struct {
    pub fn prepare(a: std.mem.Allocator, config: *@import("../model.zig").ModelConfig, dir: []const u8) !void {
        try hy4_contract.prepare(a, config, dir);
        config.expert_resident = false;
        // 6 of 8 slots/layer on HY4_STQ_TINY (not resident): >= 2 * top_k (= 4) unlocks joinedRows == 3.
        config.expert_cache_bytes = 6 * @as(u64, config.hy4flash.?.layers) * 26_112;
        config.expert_cache_slots = 6;
    }
};

test "hy4 request: a streamed Cooler load with >= 2 * top_k slots/layer joins concurrent requests with exact solo parity, zero PlanMiss, zero late pipelines, and exact ledger balance (HY4_STQ_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer @import("../test_support.zig").restoreEngine(was);
    const dir = std.mem.span(std.c.getenv("HY4_STQ_TINY") orelse return error.SkipZigTest);
    const ids = try hy4_contract.prompt(a, dir);
    defer a.free(ids);
    var ld: Load = undefined;
    try ld.open(StreamedHy4, dir, true);
    defer _ = ld.close();
    defer static_bill.close(&ld.config);

    const eng = ld.xfm.expert_stream.?;
    try testing.expect(!eng.plan.resident);
    try testing.expect(eng.residentRoute(1) == null);
    try testing.expect(eng.plan.slots_per_layer >= 4 and eng.plan.slots_per_layer < 8);
    const joined_cap = @import("hy4_plan.zig").joinedRows(&ld.config);
    try testing.expectEqual(@as(usize, 3), joined_cap);
    try testing.expect(@import("hy4_sched.zig").batches(&ld.config));

    var sv: Served = undefined;
    try sv.init(&ld, dir);
    defer sv.deinit();
    sv.sch.queue_cap = 8;
    const xfm = @import("../arch/handle.zig").trunk(sv.entry.handle.?);
    const m = xfm.arch_state.hy4flash.model.?;
    try testing.expect(m.streamed);
    try testing.expect(m.plans != null);
    try testing.expectEqual(@as(usize, 3), m.plans.?.joined);

    const max_toks: u32 = 8;
    var prompts_buf: [3][40]u32 = undefined;
    var prompts: [3][]const u32 = undefined;
    var solos: [3]SoloResult = undefined;
    var solos_made: usize = 0;
    defer for (0..solos_made) |i| {
        solos[i].tokens.deinit(a);
        solos[i].state.deinit(a);
    };
    for (0..3) |i| {
        const len: usize = 10 + i * 6;
        for (prompts_buf[i][0..len], 0..) |*t, j| t.* = ids[(j * (i + 3) + i) % ids.len];
        prompts[i] = prompts_buf[i][0..len];
        solos[i] = try soloRun(&sv, prompts[i], max_toks, false);
        solos_made = i + 1;
    }

    const misses_before = m.plans.?.misses;
    const late_before = mlx.engine.latePipelines();
    const fills_before = eng.fill_experts_total;
    hy4_state.joined_ticks_for_test = 0;
    {
        const turn = @import("../test_support.zig").openTurn(ld.h, &ld.config);
        defer turn.end();
        var slots: [3]*scheduler.Slot = undefined;
        var made: usize = 0;
        defer for (0..made) |i| endSlot(&sv, slots[i]);
        for (0..3) |i| {
            slots[i] = try submitPending(&sv, prompts[i], max_toks, false);
            made = i + 1;
            try scheduler.runPrefill(&sv.sch, slots[i]);
            try sv.sch.decoding.append(a, slots[i]);
        }
        var rounds: usize = 0;
        while (rounds < 32) : (rounds += 1) {
            var act: [3]*scheduler.Slot = undefined;
            var k: usize = 0;
            for (0..3) |i| if (slots[i].joinsTick()) {
                act[k] = slots[i];
                k += 1;
            };
            if (k == 0) break;
            try scheduler.runDecodeTick(&sv.sch, act[0..k]);
            scheduler.cullDecoding(&sv.sch);
        }
        for (0..3) |i| {
            try testing.expectEqual(@as(?[]const u8, null), slots[i].error_code);
            try testing.expectEqualSlices(u32, solos[i].tokens.items, slots[i].legacy_gen.?.generated_ids.items);
        }
    }
    try testing.expect(hy4_state.joined_ticks_for_test > 0);
    try testing.expect(eng.fill_experts_total > fills_before);
    try testing.expectEqual(misses_before, m.plans.?.misses);
    try testing.expectEqual(late_before, mlx.engine.latePipelines());

    // Exact ledger balance after all streamed concurrent requests close.
    mlx.engine.close();
    const l = ld.config.static_ledger.?;
    const snap = l.snapshot();
    const lines = snap.committed - snap.of(.margin) - snap.of(.host) - snap.of(.media) - snap.of(.pipelines);
    try testing.expectEqual(lines, ld.before.heldSince());
}
