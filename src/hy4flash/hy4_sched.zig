//! Hy4-Flash as the scheduler sees it: one request at a time, as a streamed DeepSeek load. Restore points go to the SSD
//! tier as the prefill takes them, and the live record's reach it before its state is lost (`hy4_prefix`).
const std = @import("std");
const model = @import("../model.zig");
const mlx = @import("../mlx.zig");
const hy4_model = @import("hy4_model.zig");
const hy4_prefix = @import("hy4_prefix.zig");
const Slot = @import("../scheduler.zig").Slot;
const Scheduler = @import("../scheduler.zig").Scheduler;
const Trunk = @import("../trunk.zig").Trunk;
const ForwardCtx = @import("../model_handle.zig").ForwardCtx;
const sched = @import("../arch/sched.zig");
const arch_handle = @import("../arch/handle.zig");
const hy4_state = @import("hy4_state.zig");

pub fn initRequest(allocator: std.mem.Allocator, config: *const model.ModelConfig) !?hy4_model.Request {
    _ = config;
    return try hy4_model.Request.init(allocator, 0);
}

fn slotRequest(slot: *@import("../scheduler.zig").Slot) ?*hy4_model.Request {
    return if (slot.request) |*r| r.as(.hy4flash) else null;
}

fn lendSlot(slot: *@import("../scheduler.zig").Slot) ?hy4_model.Model.Lent {
    const xfm = @import("../arch/handle.zig").trunk(slot.model.handle orelse return null);
    const m = xfm.arch_state.hy4flash.model orelse return null;
    if (!m.serves_requests) return null;
    return m.lendFor(slotRequest(slot));
}

/// A stored point's rows are views of the state's, so every one goes to SSD before the next forward, however short the
/// new span (`hy4_prefix.writeThrough`).
pub fn facts(config: *const model.ModelConfig) @import("../arch/sched.zig").Facts {
    _ = config;
    return .{ .always_write_through = true };
}

pub fn writeThrough(
    hc: *@import("../prefix_cache.zig").HotPrefixCache,
    slot: *@import("../scheduler.zig").Slot,
    entry_id: *?u64,
    prompt: []const u32,
    media: @import("../prefix_cache.zig").MediaStamp,
    cps: []@import("../trunk.zig").SSMCheckpoint,
    s: @import("../mlx.zig").mlx_stream,
) bool {
    _ = entry_id;
    const m = @import("../arch/handle.zig").trunk(slot.model.handle orelse return false).arch_state.hy4flash.model orelse return false;
    const lent = lendSlot(slot);
    defer if (lent) |l| l.end();
    hy4_prefix.writeThrough(hc, m, &slot.cache, prompt, slot.has_tools, media, slot.restored_disk, cps, s);
    return true;
}

/// The live restore points of a resident model reach SSD (unload, idle, shutdown).
pub fn flushResident(entry: *@import("../model_registry.zig").LoadedModel, why: []const u8) void {
    const hc = if (entry.prefix_cache) |*p| p else return;
    const xfm = @import("../arch/handle.zig").trunk(entry.handle orelse return);
    const m = xfm.arch_state.hy4flash.model orelse return;
    if (m.serves_requests) return;
    hy4_prefix.flushLive(hc, m, why);
}

pub fn batches(config: *const model.ModelConfig) bool {
    return @import("hy4_plan.zig").joinedRows(config) > 1;
}

/// A joined tick's draws: each row's logits through its own request's planned one-row sample root (D24), the mix
/// keyed by the row's own step, greedy and filtered rows alike. False where the load plans no samples.
pub fn sampleJoinedRows(
    xfm: *Trunk,
    gens: []const *@import("../generate.zig").Generator,
    rows: []const mlx.mlx_array,
    params: []@import("../sampler.zig").SamplingParams,
    out: []i32,
) !bool {
    const m = xfm.arch_state.hy4flash.model orelse return false;
    const p = m.plans orelse return false;
    if (!p.sample.planned) return false;
    std.debug.assert(gens.len == rows.len and rows.len == params.len and out.len == rows.len);
    for (0..rows.len) |i| {
        const gen = gens[i];
        const step = gen.stepFor(rows[i], .{ .decided = &.{gen.next_token_id} });
        const run = try @import("../arch/cooler.zig").beginSample(xfm, step.mix());
        const lazy = @import("../sampler.zig").sampleStep(std.heap.page_allocator, step, xfm.outStream()) catch |e| {
            try run.fail();
            return e;
        };
        const kept = try run.token(lazy);
        defer _ = mlx.mlx_array_free(kept);
        var token: i32 = 0;
        try mlx.check(mlx.mlx_array_item_int32(&token, kept));
        out[i] = token;
        params[i].draw +%= 1;
    }
    return true;
}

/// One request a forward: the composed forward keeps one request's state, and a streamed load's expert cache is one
/// request's working set.
pub fn coolerRows(config: *const model.ModelConfig) @import("../arch/sched.zig").CoolerRows {
    const rows = @import("hy4_plan.zig").joinedRows(config);
    if (rows <= 1) return .{ .rows = 1, .why = "its routed experts stream from SSD (one request's working set), or its decode runs off Sushi Cooler" };
    return .{ .rows = rows, .note = "each row its own request's exact step; a next turn restores from the SSD tier" };
}

pub fn rowCap(config: *const model.ModelConfig) ?usize {
    const rows = @import("hy4_plan.zig").joinedRows(config);
    return if (rows > 1) rows else null;
}

pub fn padsBatch(config: *const model.ModelConfig) bool {
    _ = config;
    return false;
}

pub const rows_company_min: usize = 2;

/// The plans record the joined tick from two rows (`hy4_plan.beginJoined`); one row is the serial forward's.
pub const joined_rows_min: usize = 2;

pub fn tickClass(slot: *const @import("../scheduler.zig").Slot) @import("../arch/sched.zig").TickClass {
    const gen = if (slot.legacy_gen) |*g| g else return .none;
    if (!gen.drafter_on or gen.done) return .none;
    if (slot.sampling.constraint != null or @import("../sampler.zig").penaltyActive(slot.sampling)) return .none;
    return if (modelBatches(arch_handle.trunk(slot.model.handle orelse return .none))) .rows_company else .none;
}

pub fn modelBatches(xfm: *const Trunk) bool {
    const m = xfm.arch_state.hy4flash.model orelse return false;
    return m.serves_requests;
}

pub fn readyToBatch(xfm: *const Trunk, ctx: *ForwardCtx) bool {
    const m = xfm.arch_state.hy4flash.model orelse return true;
    if (!m.serves_requests) return true;
    const r = hy4_state.requestOf(ctx) orelse return false;
    return r.pos > 0;
}

pub fn batchArm(xfm: *Trunk, ctxs: []const *ForwardCtx) ?sched.BatchArm {
    for (ctxs) |ctx| if (!readyToBatch(xfm, ctx)) return null;
    return .{ .moe_positions = false, .advances_offset = true };
}

const max_rows: usize = @intCast(@import("sushi_exl3").project_rows.NAX_ROWS_MAX);

fn groupable(batch: []const *Slot) bool {
    for (batch) |slot| if (slot.legacy_gen) |*g| if (@import("hy4_drafter.zig").groupEligible(g)) return true;
    return false;
}

const knobs = @import("hy4_sched_knobs.zig");

fn noteGroupsRefused(widths: []const u16) void {
    knobs.groups_refused_for_test += 1;
    if (@import("builtin").is_test) return;
    var buf: [256]u8 = undefined;
    const text = std.fmt.bufPrint(&buf, "[hy4-groups] a group tick of widths {any} has no planned root; it decodes as plain rows\n", .{widths}) catch return;
    @import("../log.zig").info("{s}", .{text});
}

fn groupBudget(drafts: usize, cap: usize, slots: usize) usize {
    return @import("hy4_forward.zig").groupRows(drafts, cap, slots);
}

fn groupForward(
    sch: *Scheduler,
    allocator: std.mem.Allocator,
    xfm: *Trunk,
    batch: *[]*Slot,
    next_tokens: []const u32,
    ctxs: []const *ForwardCtx,
    probe: ?@import("hy4_plan.zig").Probe,
) ![]mlx.mlx_array {
    const hy4_drafter = @import("hy4_drafter.zig");
    const hy4_dfly = @import("hy4_dfly.zig");
    const m = xfm.arch_state.hy4flash.model orelse return error.Hy4ModelMissing;
    if (@import("builtin").is_test) {
        hy4_state.joined_ticks_for_test += 1;
        knobs.group_ticks_for_test += 1;
    }
    const n = ctxs.len;
    if (n == 0 or n > max_rows) return error.Hy4RowsWidth;
    const cap: usize = @intCast(@import("hy4_forward.zig").joinedRowsMax());
    const budget: usize = groupBudget(@intCast(m.config.dfly.draft), cap, n);
    var ids: [max_rows][hy4_drafter.max_rows]u32 = undefined;
    var lens: [max_rows]usize = @splat(1);
    var drafted: [max_rows]bool = @splat(false);
    var blocks: [max_rows]?hy4_dfly.Drafter.Draft = @splat(null);
    defer for (0..n) |i| if (blocks[i]) |*b| b.deinit();
    var draft_ns: [max_rows]u64 = @splat(0);
    for (batch.*, 0..) |slot, i| {
        const gen = if (slot.legacy_gen) |*g| g else continue;
        if (!hy4_drafter.groupEligible(gen)) continue;
        var timer = @import("../io_util.zig").Stopwatch.init(sch.io);
        var b = try hy4_drafter.draftBlock(xfm, ctxs[i], gen.next_token_id);
        if (comptime @import("builtin").is_test) if (knobs.copy_streams_for_test) |streams| {
            const req = hy4_state.requestOf(ctxs[i]) orelse return error.Hy4NoRequest;
            for (streams) |cs| if (cs.request == req) {
                const at = gen.generated_ids.items.len;
                for (1..b.ids.len) |j| {
                    if (at + j < cs.tokens.len) b.ids[j] = cs.tokens[at + j];
                }
                break;
            };
        };
        draft_ns[i] = timer.read();
        lens[i] = @min(b.ids.len, budget);
        @memcpy(ids[i][0..lens[i]], b.ids[0..lens[i]]);
        blocks[i] = b;
        drafted[i] = true;
    }
    var groups: [max_rows]@import("hy4_model.zig").Model.Group = undefined;
    var opened: usize = 0;
    errdefer for (ctxs[0..opened]) |ctx| dropVerify(xfm, ctx, false);
    for (batch.*, 0..) |_, i| {
        const req = hy4_state.requestOf(ctxs[i]) orelse return error.Hy4NoRequest;
        if (drafted[i]) {
            try hy4_drafter.beginGroupVerify(xfm, ctxs[i], ids[i][0..lens[i]]);
        } else {
            ids[i][0] = next_tokens[i];
        }
        groups[i] = .{ .request = req, .ids = ids[i][0..lens[i]] };
        opened = i + 1;
    }
    var rows: usize = 0;
    for (lens[0..n]) |t| rows += t;
    var timer = @import("../io_util.zig").Stopwatch.init(sch.io);
    if (probe) |pr| try @import("hy4_plan.zig").beginJoinedGroups(m, pr);
    if (xfm.expert_stream) |engine| engine.beginForward();
    const joined = m.extendGroups(groups[0..n]) catch |e| {
        if (xfm.expert_stream) |engine| engine.finishForward(rows);
        if (probe != null) @import("../tensor/core.zig").abortExecution();
        return e;
    };
    if (xfm.expert_stream) |engine| engine.finishForward(rows);
    const verify_ns = timer.read();
    var views: [max_rows]mlx.mlx_array = undefined;
    for (0..n) |i| views[i] = mlx.mlx_array_new();
    errdefer {
        for (0..n) |i| _ = mlx.mlx_array_free(views[i]);
    }
    var start: usize = 0;
    for (groups[0..n], 0..) |g, i| {
        views[i] = try @import("hy4_model.zig").writeGroupBlock(g.request, joined, @intCast(start), @intCast(start + lens[i]), m.logits_rows, xfm.s);
        start += lens[i];
    }
    _ = mlx.mlx_array_free(joined);
    if (probe) |pr| try @import("hy4_plan.zig").endJoinedGroups(m, pr, groups[0..n]);
    const out = try allocator.alloc(mlx.mlx_array, n);
    var made: usize = 0;
    errdefer {
        for (out[0..made]) |a| _ = mlx.mlx_array_free(a);
        allocator.free(out);
    }
    var kept: usize = 0;
    for (batch.*, 0..) |slot, i| {
        const gen = if (slot.legacy_gen) |*g| g else return error.Hy4NoGenerator;
        const req = groups[i].request;
        if (!drafted[i]) {
            out[made] = views[i];
            made += 1;
            batch.*[kept] = slot;
            kept += 1;
            ctxs[i].cache.step = req.pos;
        } else {
            const b = blocks[i].?;
            // The think-room cut is the serial round's alone: `groupEligible` excludes a think bound and a forced call.
            const cut = hy4_drafter.rounds.roundRows(gen, ids[i][0..lens[i]], lens[i]);
            // The verify ran the planned width; the rows past the cut roll back as rejected drafts do.
            var head = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(head);
            if (cut < lens[i]) {
                const vocab = mlx.getShape(views[i])[2];
                try mlx.check(mlx.mlx_slice(&head, views[i], &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ 1, @intCast(cut), vocab }, 3, &[_]c_int{ 1, 1, 1 }, 3, xfm.s));
            }
            const logits = if (cut < lens[i]) head else views[i];
            const result = hy4_drafter.rounds.settleDrafted(gen, allocator, ids[i][0..cut], logits, b.conf[0 .. cut - 1], draft_ns[i], verify_ns) catch |e| {
                dropVerify(xfm, ctxs[i], true);
                return e;
            };
            @import("../scheduler.zig").publishSpeculativeBlock(sch, slot, gen, result.tokens);
            allocator.free(result.tokens);
            _ = mlx.mlx_array_free(views[i]);
        }
        views[i] = mlx.mlx_array_new();
    }
    batch.* = batch.*[0..kept];
    return try allocator.realloc(out, kept);
}

fn dropVerify(xfm: *Trunk, ctx: *ForwardCtx, reset: bool) void {
    const m = xfm.arch_state.hy4flash.model orelse return;
    const r = hy4_state.requestOf(ctx) orelse return;
    const lent = m.lend(r);
    defer lent.end();
    @import("hy4_state.zig").dropVerifyReq(m, r, reset);
}

pub fn batchedForward(
    sch: *Scheduler,
    allocator: std.mem.Allocator,
    xfm: *Trunk,
    batch: *[]*Slot,
    next_tokens: []const u32,
    ctxs: []const *ForwardCtx,
    rope_offsets: []const u32,
    arm: sched.BatchArm,
    hidden_rows: ?*?[]mlx.mlx_array,
) ![]mlx.mlx_array {
    _ = .{ rope_offsets, arm, hidden_rows };
    const m = xfm.arch_state.hy4flash.model orelse return error.Hy4ModelMissing;
    if (ctxs.len < joined_rows_min) return error.Hy4RowsWidth;
    if (@import("builtin").is_test and knobs.groups_on_for_test and groupable(batch.*)) {
        const hy4_drafter = @import("hy4_drafter.zig");
        const cap: usize = @intCast(@import("hy4_forward.zig").joinedRowsMax());
        const budget = groupBudget(@intCast(m.config.dfly.draft), cap, ctxs.len);
        var widths: [max_rows]u16 = undefined;
        var reqs: [max_rows]*@import("hy4_model.zig").Request = undefined;
        var any = false;
        for (ctxs, 0..) |ctx, i| {
            reqs[i] = hy4_state.requestOf(ctx) orelse return error.Hy4NoRequest;
            const eligible = if (batch.*[i].legacy_gen) |*g| hy4_drafter.groupEligible(g) else false;
            widths[i] = if (eligible) @intCast(budget) else 1;
            any = any or eligible;
        }
        if (any) {
            if (m.plans == null) return groupForward(sch, allocator, xfm, batch, next_tokens, ctxs, null);
            if (@import("hy4_plan.zig").joinedGroupsProbe(m, reqs[0..ctxs.len], widths[0..ctxs.len])) |probe|
                return groupForward(sch, allocator, xfm, batch, next_tokens, ctxs, probe);
            noteGroupsRefused(widths[0..ctxs.len]);
        }
    }
    var segs: [max_rows]@import("hy4_model.zig").Model.Seg = undefined;
    if (ctxs.len > @as(usize, @intCast(@import("hy4_forward.zig").joinedRowsMax()))) return error.Hy4RowsWidth;
    for (ctxs, next_tokens, 0..) |ctx, t, i| {
        segs[i] = .{ .request = hy4_state.requestOf(ctx) orelse return error.Hy4NoRequest, .id = t };
    }
    if (@import("builtin").is_test) hy4_state.joined_ticks_for_test += 1;
    if (m.plans != null) {
        const core = @import("../tensor/core.zig");
        const GemmScope = @import("sushi_exl3").linear.GemmScope;
        const probe = try @import("hy4_plan.zig").beginJoined(m, segs[0..ctxs.len]);
        const scope = GemmScope.begin() catch |e| {
            core.abortExecution();
            return e;
        };
        if (xfm.expert_stream) |engine| engine.beginForward();
        const joined = m.extendRows(segs[0..ctxs.len]) catch |e| {
            scope.end();
            core.abortExecution();
            return e;
        };
        if (xfm.expert_stream) |engine| engine.finishForward(ctxs.len);
        const out = rowsIntoRequests(allocator, joined, segs[0..ctxs.len], m.logits_rows, xfm.s) catch |e| {
            _ = mlx.mlx_array_free(joined);
            scope.end();
            core.abortExecution();
            return e;
        };
        _ = mlx.mlx_array_free(joined);
        scope.end();
        try @import("hy4_plan.zig").endJoined(m, probe, segs[0..ctxs.len]);
        for (ctxs, segs[0..ctxs.len]) |ctx, sg| ctx.cache.step = sg.request.pos;
        return out;
    }
    if (xfm.expert_stream) |engine| engine.beginForward();
    const joined = try m.extendRows(segs[0..ctxs.len]);
    defer _ = mlx.mlx_array_free(joined);
    if (xfm.expert_stream) |engine| engine.finishForward(ctxs.len);
    for (ctxs, segs[0..ctxs.len]) |ctx, sg| ctx.cache.step = sg.request.pos;
    return rowsOf(allocator, joined, ctxs.len, xfm.s);
}

fn rowsIntoRequests(allocator: std.mem.Allocator, joined: mlx.mlx_array, segs: []const hy4_model.Model.Seg, block_rows: c_int, s: mlx.mlx_stream) ![]mlx.mlx_array {
    const out = try allocator.alloc(mlx.mlx_array, segs.len);
    var made: usize = 0;
    errdefer {
        for (out[0..made]) |o| _ = mlx.mlx_array_free(o);
        allocator.free(out);
    }
    const vocab: c_int = mlx.getShape(joined)[1];
    for (segs, 0..) |sg, i| {
        const start = [2]c_int{ @intCast(i), 0 };
        const stop = [2]c_int{ @intCast(i + 1), vocab };
        const step = [2]c_int{ 1, 1 };
        var row = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(row);
        try mlx.check(mlx.mlx_slice(&row, joined, &start, 2, &stop, 2, &step, 2, s));
        const turn = sg.request.logits_turn;
        sg.request.logits_turn +%= 1;
        const block = &sg.request.logits_fixed[turn];
        try hy4_model.writeFixed(block, row, &.{ block_rows, vocab }, s);
        var view = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(view);
        try mlx.check(mlx.mlx_slice(&view, block.*.?, &[_]c_int{ 0, 0 }, 2, &[_]c_int{ 1, vocab }, 2, &[_]c_int{ 1, 1 }, 2, s));
        out[i] = mlx.mlx_array_new();
        made = i + 1;
        const shape = [3]c_int{ 1, 1, vocab };
        try mlx.check(mlx.mlx_reshape(&out[i], view, &shape, 3, s));
    }
    return out;
}

fn rowsOf(allocator: std.mem.Allocator, joined: mlx.mlx_array, n: usize, s: mlx.mlx_stream) ![]mlx.mlx_array {
    const out = try allocator.alloc(mlx.mlx_array, n);
    var made: usize = 0;
    errdefer {
        for (out[0..made]) |o| _ = mlx.mlx_array_free(o);
        allocator.free(out);
    }
    const vocab: c_int = mlx.getShape(joined)[1];
    for (0..n) |i| {
        const start = [2]c_int{ @intCast(i), 0 };
        const stop = [2]c_int{ @intCast(i + 1), vocab };
        const step = [2]c_int{ 1, 1 };
        var row = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(row);
        try mlx.check(mlx.mlx_slice(&row, joined, &start, 2, &stop, 2, &step, 2, s));
        out[i] = mlx.mlx_array_new();
        made = i + 1;
        const shape = [3]c_int{ 1, 1, vocab };
        try mlx.check(mlx.mlx_reshape(&out[i], row, &shape, 3, s));
    }
    return out;
}

pub fn cancelledNeedsCheckpoints() bool {
    return true;
}

/// A lookup restores into the model's own request state (`hy4_prefix`), reserved first at the request's end
/// (`hy4_prefix.reserveRequest`), so a restore into it asks for no buffer.
pub fn restoreTarget(slot: *@import("../scheduler.zig").Slot, xfm: *@import("../trunk.zig").Trunk) ?@import("../arch/sched.zig").RestoreTarget {
    const m = xfm.arch_state.hy4flash.model orelse return null;
    if (m.serves_requests) {
        std.debug.assert(m.bound == null);
        _ = m.lendFor(slotRequest(slot));
    }
    const past_end = @import("../bills.zig").speculativeRowsPastEnd(&xfm.config, .{ .mtp_on = slot.enable_mtp and slot.mtp != null });
    hy4_prefix.reserveRequest(m, slot.prompt_ids.len, slot.max_tokens, past_end) catch |e|
        @import("../log.zig").warn("[hy4-prefix] the request's state was not reserved before its lookup ({t}): its restore reserves it\n", .{e});
    return .{ .rec = @import("../arch/prefix.zig").recurrent("hy4flash", m), .log_tag = "[hy4-prefix]" };
}

pub fn restoreEnd(slot: *@import("../scheduler.zig").Slot, xfm: *@import("../trunk.zig").Trunk) void {
    const m = xfm.arch_state.hy4flash.model orelse return;
    if (!m.serves_requests) return;
    std.debug.assert(m.bound == slotRequest(slot));
    m.unbind(null);
}

/// The request over the model's state finished with `tokens`: the live record's tokens and the tools and media they
/// were computed under, which the next prompt is matched against (`hy4_prefix.rollbackLive`).
pub fn publishLive(sch: *@import("../scheduler.zig").Scheduler, slot: *@import("../scheduler.zig").Slot, hc: *@import("../prefix_cache.zig").HotPrefixCache, tokens: []const u32) void {
    const xfm = @import("../arch/handle.zig").trunk(slot.model.handle orelse return);
    const m = xfm.arch_state.hy4flash.model orelse return;
    const media = @import("../prefix_cache.zig").MediaStamp.of(slot.vision_key, slot.media_start);
    if (m.serves_requests) {
        const lent = lendSlot(slot);
        defer if (lent) |l| l.end();
        hy4_prefix.writeEndPoint(hc, m, &slot.cache, tokens, slot.has_tools, media, slot.restored_disk);
        return;
    }
    hy4_prefix.publishLive(hc, m, tokens, slot.has_tools, media);
    // A point only in RAM reaches SSD on the idle flush, if no other request takes the state first.
    if (hy4_prefix.liveOf(hc).unflushed(if (hc.disk) |*d| d else null)) sch.live_unflushed.store(true, .release);
}

/// The state the model holds outside its hot cache's entries, which a running request's slot bills (the request's own
/// is the model's): the state, its running points and the live record's (`hy4_prefix.heldBytes`).
pub fn modelStateBytes(slot: *const @import("../scheduler.zig").Slot) u64 {
    return parkedStateBytes(slot.model);
}

/// Admission needs the room of the state the model parks: the live record's points reach SSD and go, and the state
/// is released; the next request starts over or restores from SSD.
pub fn releaseParked(entry: *@import("../model_registry.zig").LoadedModel) void {
    const xfm = @import("../arch/handle.zig").trunk(entry.handle orelse return);
    const m = xfm.arch_state.hy4flash.model orelse return;
    if (m.serves_requests) return;
    if (entry.prefix_cache) |*hc| hy4_prefix.releaseLive(hc, m, "admission");
    m.reset();
    @import("../mlx.zig").engine.trim();
}

/// The same with no request on it: what the model keeps for the next one.
pub fn parkedStateBytes(entry: *@import("../model_registry.zig").LoadedModel) u64 {
    const xfm = @import("../arch/handle.zig").trunk(entry.handle orelse return 0);
    const m = xfm.arch_state.hy4flash.model orelse return 0;
    return hy4_prefix.parkedBytes(m, if (entry.prefix_cache) |*hc| hc else null);
}

test "hy4 rows: a group's rows are its drafts plus one, capped by the joined rows the tick shares" {
    try std.testing.expectEqual(@as(usize, 6), groupBudget(5, 16, 2));
    try std.testing.expectEqual(@as(usize, 5), groupBudget(5, 16, 3));
    try std.testing.expectEqual(@as(usize, 4), groupBudget(5, 16, 4));
    try std.testing.expectEqual(@as(usize, 3), groupBudget(5, 16, 5));
    try std.testing.expectEqual(@as(usize, 2), groupBudget(5, 16, 8));
    try std.testing.expectEqual(@as(usize, 1), groupBudget(5, 16, 16));
    try std.testing.expectEqual(@as(usize, 1), groupBudget(0, 16, 2));
    try std.testing.expectEqual(@as(usize, 1), groupBudget(5, 8, 16));
    try std.testing.expectEqual(@as(usize, 8), groupBudget(20, 64, 4));
    try std.testing.expectEqual(@as(usize, 0), groupBudget(5, 16, 0));
}

test "hy4 rows: a resident Cooler load declares the shared ops' joined rows, its cap and batching, a streamed load one row" {
    const testTrunk = @import("../test_support.zig").testTrunk;
    const want: usize = @intCast(@import("hy4_forward.zig").joinedRowsMax());
    try std.testing.expectEqual(if (@import("sushi_exl3").kernels.gemmNaxOn()) @as(usize, 16) else 8, want);
    var config = model.ModelConfig{ .model_type = "hy_v4_flash" };
    config.expert_resident = true;
    config.cooler_for_test = true;
    const resident = coolerRows(&config);
    try std.testing.expectEqual(want, resident.rows);
    try std.testing.expect(resident.note != null);
    try std.testing.expect(batches(&config));
    try std.testing.expectEqual(@as(?usize, want), rowCap(&config));
    try std.testing.expect(!padsBatch(&config));
    config.cooler_for_test = false;
    try std.testing.expectEqual(want, coolerRows(&config).rows);
    config.expert_resident = false;
    try std.testing.expectEqual(@as(usize, 1), coolerRows(&config).rows);
    try std.testing.expect(!batches(&config));
    try std.testing.expectEqual(@as(?usize, null), rowCap(&config));
    var xfm = testTrunk("hy_v4_flash");
    var lm: @import("../model_registry.zig").LoadedModel = undefined;
    var slot = sched.testSlot(&xfm, &lm, .{ .model_type = "hy_v4_flash" });
    try std.testing.expectEqual(sched.TickClass.none, sched.tickClass(&slot));
}

test "hy4 rows: without matrix units the joined rows stay 8, the SIMD arm's width, and with them 16" {
    var config = model.ModelConfig{ .model_type = "hy_v4_flash" };
    config.expert_resident = true;
    config.cooler_for_test = true;
    {
        const scope = @import("sushi_exl3").device.forceMatrix(false);
        defer scope.end();
        try std.testing.expectEqual(@as(usize, 8), @import("hy4_plan.zig").joinedRows(&config));
        try std.testing.expectEqual(@as(?usize, 8), rowCap(&config));
    }
    const scope = @import("sushi_exl3").device.forceMatrix(true);
    defer scope.end();
    try std.testing.expectEqual(@as(usize, 16), @import("hy4_plan.zig").joinedRows(&config));
    try std.testing.expectEqual(@as(?usize, 16), rowCap(&config));
}
