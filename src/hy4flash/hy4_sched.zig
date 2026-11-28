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
    hc.commit_media_chain = slot.media_chain;
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
/// keyed by the row's own step, greedy and filtered rows alike. All rows are encoded before a single GPU sync.
/// False where the load plans no samples.
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
    const chunk_max = @import("hy4_sample_plan.zig").token_slots / 2;
    var base: usize = 0;
    while (base < rows.len) {
        const count = @min(chunk_max, rows.len - base);
        var kept: [chunk_max]mlx.mlx_array = undefined;
        var made: usize = 0;
        defer for (kept[0..made]) |k| {
            _ = mlx.mlx_array_free(k);
        };
        for (0..count) |j| {
            const i = base + j;
            const gen = gens[i];
            const step = gen.stepFor(rows[i]);
            const run = try @import("../arch/cooler.zig").beginSample(xfm, step.mix());
            const lazy = @import("../sampler.zig").sampleStep(std.heap.page_allocator, step, xfm.outStream()) catch |e| {
                run.fail();
                return e;
            };
            kept[j] = try run.token(lazy);
            made = j + 1;
        }
        const vec = mlx.mlx_vector_array_new_data(&kept, made);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_eval(vec));
        for (0..count) |j| {
            const i = base + j;
            var token: i32 = 0;
            try mlx.check(mlx.mlx_array_item_int32(&token, kept[j]));
            out[i] = token;
            params[i].draw +%= 1;
        }
        base += count;
    }
    return true;
}

/// A resident Hy4-Flash Cooler load joins requests up to `joinedRowsMax`; a streamed Cooler load joins up to
/// `min(joinedRowsMax, slots / top_k)` rows when the expert cache holds at least `2 * top_k` slots per layer
/// (fitting one tick in a single route group without intra-tick eviction), else stays exclusive (1 row).
pub fn coolerRows(config: *const model.ModelConfig) @import("../arch/sched.zig").CoolerRows {
    const rows = @import("hy4_plan.zig").joinedRows(config);
    if (rows <= 1) return .{ .rows = 1, .why = "its routed experts stream from SSD with fewer than 2 * top_k cached slots per layer, or its decode runs off Sushi Cooler" };
    if (!config.expert_resident) return .{ .rows = rows, .note = "streamed expert cache holds >= 2 * top_k slots/layer; joined rows amortize dense trunk and coalesce SSD fills" };
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

fn groupable(m: *const hy4_model.Model, batch: []const *Slot, ctxs: []const *ForwardCtx) bool {
    if (!m.keep_hidden) return false;
    for (batch, ctxs) |slot, ctx| {
        const gen = if (slot.legacy_gen) |*g| g else continue;
        if (!@import("hy4_drafter.zig").groupEligible(gen)) continue;
        const req = hy4_state.requestOf(ctx) orelse continue;
        if (m.hasHandover(req)) return true;
    }
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

/// Online linear batch row-cost model for grouped DFly2 verify width allocation.
const GroupCost = struct {
    base_ms: f64 = 10.5,
    row_ms: f64 = 3.2,
    draft_ms: f64 = 4.5,
    row_samples: u32 = 0,
    plain_ms: [max_rows + 1]f64 = @splat(0),

    fn forLoad(resident: bool) GroupCost {
        return if (resident) .{} else .{ .base_ms = 35.0, .row_ms = 14.0, .draft_ms = 5.0 };
    }

    fn observePlain(self: *GroupCost, n: usize, ms: f64) void {
        if (!std.math.isFinite(ms) or ms <= 0 or n == 0 or n > max_rows) return;
        self.plain_ms[n] = if (self.plain_ms[n] > 0) 0.8 * self.plain_ms[n] + 0.2 * ms else ms;
    }

    fn observeVerify(self: *GroupCost, n: usize, total_rows: usize, verify_ms: f64, draft_ms: f64) void {
        if (std.math.isFinite(draft_ms) and draft_ms > 0) self.draft_ms = 0.8 * self.draft_ms + 0.2 * draft_ms;
        if (!std.math.isFinite(verify_ms) or verify_ms <= 0 or total_rows <= n) return;
        const base_n = self.plainAt(n);
        const extra: f64 = @floatFromInt(total_rows - n);
        const sample = std.math.clamp((verify_ms - base_n) / extra, 0.05, 80.0);
        // Converge rapidly on the first few samples (especially streamed loads where SSD cost varies by cache size).
        self.row_samples +|= 1;
        const alpha: f64 = if (self.row_samples <= 4) 1.0 / @as(f64, @floatFromInt(self.row_samples)) else 0.15;
        self.row_ms = (1.0 - alpha) * self.row_ms + alpha * sample;
    }

    fn plainAt(self: *const GroupCost, n: usize) f64 {
        if (n <= max_rows and self.plain_ms[n] > 0) return self.plain_ms[n];
        return self.base_ms + self.row_ms * @as(f64, @floatFromInt(n));
    }
};

/// Exact greedy marginal-gain verify width allocator across `n` slots (`sum(out_rows) <= cap`):
/// starts every slot at `min_rows[i]` (>= 1, clamped so remaining slots always keep at least 1 row inside `cap`),
/// then iteratively adds +1 verify row to the drafting slot with the highest calibrated prefix survival probability
/// `q`, stopping once `q * exp_ms <= exp_tok * row_ms`.
fn allocateGroupWidths(
    n: usize,
    cap: usize,
    base_ms: f64,
    row_ms: f64,
    min_rows: []const usize,
    max_rows_slot: []const usize,
    probs: []const []const f32,
    out_rows: []usize,
) void {
    var total_rows: usize = 0;
    var exp_tok: f64 = 0;
    var surv: [max_rows]f64 = @splat(1.0);
    for (0..n) |i| {
        const rem_slots = n - 1 - i;
        const cap_here = @max(@as(usize, 1), cap -| total_rows -| rem_slots);
        const lo = @max(@as(usize, 1), @min(@min(min_rows[i], max_rows_slot[i]), cap_here));
        out_rows[i] = lo;
        total_rows += lo;
        exp_tok += 1.0;
        for (0..lo - 1) |j| {
            const pj: f64 = if (j < probs[i].len) @floatCast(probs[i][j]) else 0.0;
            surv[i] *= std.math.clamp(pj, 0.0, 1.0);
            exp_tok += surv[i];
        }
    }
    var exp_ms: f64 = base_ms + row_ms * @as(f64, @floatFromInt(total_rows -| n));
    while (total_rows < cap) {
        var best_i: ?usize = null;
        var best_q: f64 = 0;
        for (0..n) |i| {
            const next_j = out_rows[i] - 1;
            if (out_rows[i] >= max_rows_slot[i] or next_j >= probs[i].len) continue;
            const pj: f64 = std.math.clamp(@as(f64, @floatCast(probs[i][next_j])), 0.0, 1.0);
            const q = surv[i] * pj;
            if (q > best_q) {
                best_q = q;
                best_i = i;
            }
        }
        const bi = best_i orelse break;
        if (best_q * exp_ms <= exp_tok * row_ms) break;
        surv[bi] = best_q;
        out_rows[bi] += 1;
        total_rows += 1;
        exp_tok += best_q;
        exp_ms += row_ms;
    }
}

/// Evaluates expected batch tokens/ms for candidate descending width `tpl` mapped onto `perm[0..n]`.
fn expectedTemplateRate(
    n: usize,
    tpl: []const u16,
    perm: []const usize,
    probs: []const []const f32,
    base_ms: f64,
    row_ms: f64,
) f64 {
    var total_rows: usize = 0;
    var exp_tok: f64 = 0;
    for (0..n) |k| {
        const w: usize = tpl[k];
        total_rows += w;
        exp_tok += 1.0;
        const orig = perm[k];
        var surv: f64 = 1.0;
        for (0..w -| 1) |j| {
            const pj: f64 = if (j < probs[orig].len) std.math.clamp(@as(f64, @floatCast(probs[orig][j])), 0.0, 1.0) else 0.0;
            surv *= pj;
            exp_tok += surv;
        }
    }
    const exp_ms = base_ms + row_ms * @as(f64, @floatFromInt(total_rows -| n));
    return exp_tok / @max(exp_ms, 1e-3);
}

/// Picks the planned width template in `templates[0..n_tpl]` maximizing expected tokens/ms for the sorted slot order
/// `perm[0..n]`. Returns null if no eligible template beats plain `n / base_ms` (when `allow_plain`).
fn pickPlannedTemplate(
    n: usize,
    templates: []const [max_rows]u16,
    allowed: []const bool,
    perm: []const usize,
    min_rows: []const usize,
    max_rows_slot: []const usize,
    probs: []const []const f32,
    base_ms: f64,
    row_ms: f64,
    allow_plain: bool,
) ?usize {
    const plain_rate = @as(f64, @floatFromInt(n)) / @max(base_ms, 1e-3);
    const min_rate = if (allow_plain) plain_rate * 1.01 else 0.0;
    var best: ?usize = null;
    var best_rate: f64 = min_rate;
    for ([_]bool{ true, false }) |enforce_min| {
        for (templates, 0..) |tpl, ti| {
            if (!allowed[ti]) continue;
            var fits = true;
            for (0..n) |k| {
                const orig = perm[k];
                const w: usize = tpl[k];
                if (w > max_rows_slot[orig]) {
                    fits = false;
                    break;
                }
                if (enforce_min and w < min_rows[orig]) {
                    fits = false;
                    break;
                }
            }
            if (!fits) continue;
            const rate = expectedTemplateRate(n, tpl[0..n], perm, probs, base_ms, row_ms);
            if (rate > best_rate) {
                best_rate = rate;
                best = ti;
            }
        }
        if (best != null or !enforce_min) break;
    }
    return best;
}

fn batchSignature(ctxs: []const *ForwardCtx) u64 {
    var h: u64 = 0;
    for (ctxs) |ctx| {
        const id: u64 = if (ctx.ssm_member_gen != 0) ctx.ssm_member_gen else @intFromPtr(ctx);
        h ^= id *% 0x9e3779b97f4a7c15;
    }
    return h;
}

fn joinedCap(xfm: *const Trunk, m: *const hy4_model.Model) usize {
    if (m.plans) |p| if (p.joined > 1) return p.joined;
    const cfg_cap = @import("hy4_plan.zig").joinedRows(&xfm.config);
    if (cfg_cap > 1) return cfg_cap;
    return @intCast(@import("hy4_forward.zig").joinedRowsMax());
}

fn groupForward(
    sch: *Scheduler,
    allocator: std.mem.Allocator,
    xfm: *Trunk,
    batch: *[]*Slot,
    next_tokens: []const u32,
    ctxs: []const *ForwardCtx,
) ![]mlx.mlx_array {
    const hy4_drafter = @import("hy4_drafter.zig");
    const hy4_dfly = @import("hy4_dfly.zig");
    const hy4_plan = @import("hy4_plan.zig");
    const m = xfm.arch_state.hy4flash.model orelse return error.Hy4ModelMissing;
    syncModelState(m, xfm.config.expert_resident);
    const n = ctxs.len;
    if (n == 0 or n > max_rows) return error.Hy4RowsWidth;
    const cap: usize = joinedCap(xfm, m);
    const budget: usize = groupBudget(@intCast(m.config.dfly.draft), cap, n);
    if (budget < 2) return plainBatchedForward(sch, allocator, xfm, next_tokens, ctxs, false, 0);

    var reqs: [max_rows]*hy4_model.Request = undefined;
    var draft_ctxs: [max_rows]*ForwardCtx = undefined;
    var draft_t1s: [max_rows]u32 = undefined;
    var draft_slot_idx: [max_rows]usize = undefined;
    var slot_draft_idx: [max_rows]?usize = @splat(null);
    var any_forced = false;
    var nd: usize = 0;
    for (batch.*, ctxs, 0..) |slot, ctx, i| {
        const req = hy4_state.requestOf(ctx) orelse return error.Hy4NoRequest;
        reqs[i] = req;
        const gen = if (slot.legacy_gen) |*g| g else continue;
        if (!hy4_drafter.groupEligible(gen) or !m.hasHandover(req)) continue;
        if (gen.drafter_policy.range.min_rows > 1 or gen.drafter_diag == .rows or gen.drafter_diag == .sweep) any_forced = true;
        slot_draft_idx[i] = nd;
        draft_ctxs[nd] = ctx;
        draft_t1s[nd] = gen.next_token_id;
        draft_slot_idx[nd] = i;
        nd += 1;
    }
    if (nd == 0) return plainBatchedForward(sch, allocator, xfm, next_tokens, ctxs, false, 0);

    // Match the exact `n_use` recorded in `hy4_plan.record` for `nd` drafting slots (`groupDraftWidth`).
    const draft_w: usize = if (nd >= 2) hy4_plan.groupDraftWidth(@as(usize, m.config.dfly.draft), cap, nd) else @as(usize, m.config.dfly.draft);
    if (draft_w == 0) return plainBatchedForward(sch, allocator, xfm, next_tokens, ctxs, false, 0);

    var draft_out: [max_rows]hy4_dfly.Drafter.Draft = undefined;
    var draft_made: usize = 0;
    defer for (draft_out[0..draft_made]) |*b| b.deinit();
    var draft_timer = @import("../io_util.zig").Stopwatch.init(sch.io);
    try hy4_drafter.draftBatch(xfm, draft_ctxs[0..nd], draft_t1s[0..nd], draft_w, draft_out[0..nd]);
    draft_made = nd;
    const batch_draft_ns = draft_timer.read();
    const draft_ms: f64 = @as(f64, @floatFromInt(batch_draft_ns)) / 1e6;

    if (comptime @import("builtin").is_test) if (knobs.copy_streams_for_test) |streams| {
        for (0..nd) |di| {
            const i = draft_slot_idx[di];
            const gen = &batch.*[i].legacy_gen.?;
            const req = reqs[i];
            for (streams) |cs| if (cs.request == req) {
                const at = gen.generated_ids.items.len;
                for (1..draft_out[di].ids.len) |j| {
                    if (at + j < cs.tokens.len) draft_out[di].ids[j] = cs.tokens[at + j];
                }
                break;
            };
        }
    };

    // Allocate desired verify widths per slot: uniform `budget` in tests (preserving tiny-fixture expectations),
    // calibrated marginal-gain confidence knapsack in production.
    var min_rows_slot: [max_rows]usize = @splat(1);
    var max_rows_slot: [max_rows]usize = @splat(1);
    var prob_buf: [max_rows][hy4_drafter.max_rows]f32 = undefined;
    var prob_slices: [max_rows][]const f32 = @splat(&.{});
    var desired_rows: [max_rows]usize = @splat(1);
    for (batch.*, 0..) |slot, i| {
        const di = slot_draft_idx[i] orelse continue;
        const gen = &slot.legacy_gen.?;
        const b = &draft_out[di];
        if (comptime @import("builtin").is_test) {
            const w = @min(b.ids.len, budget);
            min_rows_slot[i] = w;
            max_rows_slot[i] = w;
            desired_rows[i] = w;
        } else {
            const cut = hy4_drafter.rounds.roundRows(gen, b.ids, @min(b.ids.len, cap - n + 1));
            const max_w = switch (gen.drafter_diag) {
                .rows => |r| @min(cut, @max(@as(usize, 1), r)),
                .off, .log => @min(cut, @max(@as(usize, 1), gen.drafter_policy.range.max_rows)),
                .sweep => cut,
            };
            max_rows_slot[i] = @max(@as(usize, 1), max_w);
            min_rows_slot[i] = switch (gen.drafter_diag) {
                .rows => |r| @min(max_rows_slot[i], @max(@as(usize, 1), @min(r, budget))),
                .sweep => @min(max_rows_slot[i], budget),
                .off, .log => @min(max_rows_slot[i], @max(@as(usize, 1), @min(gen.drafter_policy.range.min_rows, budget))),
            };
            const n_conf = @min(max_rows_slot[i] -| 1, b.conf.len);
            for (0..n_conf) |j| {
                // Hy4 DFly2 `b.conf` is already sigmoid probability in (0, 1); calibrate directly without double sigmoid.
                prob_buf[i][j] = group_calib.probability(std.math.clamp(b.conf[j], 0.0, 1.0));
            }
            prob_slices[i] = prob_buf[i][0..n_conf];
        }
    }
    // Price marginal verify rows against true plain tick latency (`plainAt(n)`, excluding sunk `draft_ms`) so we only
    // spend extra verify rows whose marginal tokens/ms beat plain joined decode.
    const base_ms = group_cost.plainAt(n);
    if (comptime !@import("builtin").is_test) {
        allocateGroupWidths(
            n,
            cap,
            base_ms,
            group_cost.row_ms,
            min_rows_slot[0..n],
            max_rows_slot[0..n],
            prob_slices[0..n],
            desired_rows[0..n],
        );
    }

    // Order slots descending by `(desired_rows, first_draft_prob)` so `groups` matches the descending order recorded
    // in `hy4_plan.recordJoinedGroups`.
    var perm: [max_rows]usize = undefined;
    for (0..n) |i| perm[i] = i;
    var i_sort: usize = 1;
    while (i_sort < n) : (i_sort += 1) {
        const key = perm[i_sort];
        const key_w = desired_rows[key];
        const key_p: f32 = if (prob_slices[key].len > 0) prob_slices[key][0] else 0.0;
        var j_sort = i_sort;
        while (j_sort > 0) : (j_sort -= 1) {
            const prev = perm[j_sort - 1];
            const prev_w = desired_rows[prev];
            const prev_p: f32 = if (prob_slices[prev].len > 0) prob_slices[prev][0] else 0.0;
            if (prev_w > key_w or (prev_w == key_w and prev_p >= key_p)) break;
            perm[j_sort] = prev;
        }
        perm[j_sort] = key;
    }

    var sorted_reqs: [max_rows]*hy4_model.Request = undefined;
    var sorted_w: [max_rows]u16 = undefined;
    for (0..n) |k| {
        sorted_reqs[k] = reqs[perm[k]];
        sorted_w[k] = @intCast(desired_rows[perm[k]]);
    }

    // If confidence pruning kept every slot at 1 row, run a plain joined tick (no verify marks or rollbacks).
    if (sorted_w[0] < 2) {
        return plainBatchedForward(sch, allocator, xfm, next_tokens, ctxs, !@import("builtin").is_test and !any_forced, draft_ms);
    }

    var final_w: [max_rows]u16 = sorted_w;
    const use_narrow = m.plans != null;
    if (use_narrow) {
        var total_sorted: usize = 0;
        for (sorted_w[0..n]) |w| total_sorted += w;
        const cap_w = @import("hy4_forward.zig").groupRows(@as(usize, m.config.dfly.draft), cap, n);
        if (total_sorted > cap or sorted_w[0] > cap_w) {
            var templates: [20][max_rows]u16 = undefined;
            const n_tpl = hy4_plan.groupWidthTemplates(@as(usize, m.config.dfly.draft), cap, n, &templates);
            var allowed: [20]bool = @splat(true);
            const allow_plain = !@import("builtin").is_test and !any_forced;
            if (pickPlannedTemplate(
                n,
                templates[0..n_tpl],
                allowed[0..n_tpl],
                perm[0..n],
                min_rows_slot[0..n],
                max_rows_slot[0..n],
                prob_slices[0..n],
                base_ms,
                group_cost.row_ms,
                allow_plain,
            )) |best_ti| {
                @memcpy(final_w[0..n], templates[best_ti][0..n]);
            } else {
                if (any_forced or @import("builtin").is_test) noteGroupsRefused(sorted_w[0..n]);
                return plainBatchedForward(sch, allocator, xfm, next_tokens, ctxs, allow_plain, draft_ms);
            }
        }
    }

    if (@import("builtin").is_test) {
        hy4_state.joined_ticks_for_test += 1;
        knobs.group_ticks_for_test += 1;
    }
    if (!@import("builtin").is_test and !group_dfly_logged) {
        group_dfly_logged = true;
        @import("../log.zig").info("[hy4-groups] grouped DFly2 engaged: {d} slots (widths {any})\n", .{ n, final_w[0..n] });
    }

    var ids_perm: [max_rows][hy4_drafter.max_rows]u32 = undefined;
    var groups: [max_rows]hy4_model.Model.Group = undefined;
    var opened_ctxs: [max_rows]*ForwardCtx = undefined;
    var opened: usize = 0;
    var rollback_reset = false;
    errdefer for (opened_ctxs[0..opened]) |ctx| dropVerify(xfm, ctx, rollback_reset);

    var total_rows: usize = 0;
    for (0..n) |k| {
        const orig = perm[k];
        const w_k: usize = final_w[k];
        if (w_k >= 2) {
            const di = slot_draft_idx[orig].?;
            @memcpy(ids_perm[k][0..w_k], draft_out[di].ids[0..w_k]);
            try hy4_drafter.beginGroupVerify(xfm, ctxs[orig], ids_perm[k][0..w_k]);
            opened_ctxs[opened] = ctxs[orig];
            opened += 1;
        } else {
            ids_perm[k][0] = next_tokens[orig];
        }
        groups[k] = .{ .request = reqs[orig], .ids = ids_perm[k][0..w_k] };
        total_rows += w_k;
    }

    const GemmScope = @import("sushi_exl3").linear.GemmScope;
    var timer = @import("../io_util.zig").Stopwatch.init(sch.io);
    if (use_narrow) try hy4_plan.beginJoinedGroups(m);
    const scope = GemmScope.begin() catch |e| {
        if (use_narrow) @import("../tensor/core.zig").abortNarrow();
        return e;
    };
    if (xfm.expert_stream) |engine| engine.beginForward();
    const joined = m.extendGroups(groups[0..n]) catch |e| {
        if (xfm.expert_stream) |engine| engine.finishForward(total_rows);
        scope.end();
        if (use_narrow) @import("../tensor/core.zig").abortNarrow();
        return e;
    };
    if (xfm.expert_stream) |engine| engine.finishForward(total_rows);
    // From here on the requests' KV rings/states hold the verify rows, so any failure before finish must reset them.
    rollback_reset = true;
    var views_perm: [max_rows]mlx.mlx_array = @splat(mlx.mlx_array_new());
    errdefer {
        for (views_perm[0..n]) |v| _ = mlx.mlx_array_free(v);
    }
    var start: usize = 0;
    for (groups[0..n], 0..) |g, k| {
        const w_k: usize = final_w[k];
        views_perm[k] = hy4_model.writeGroupBlock(g.request, joined, @intCast(start), @intCast(start + w_k), m.logits_rows, xfm.s) catch |e| {
            _ = mlx.mlx_array_free(joined);
            scope.end();
            if (use_narrow) @import("../tensor/core.zig").abortNarrow();
            return e;
        };
        start += w_k;
    }
    _ = mlx.mlx_array_free(joined);
    scope.end();
    if (use_narrow) try hy4_plan.endJoinedGroups(m, groups[0..n]);

    // Map permuted slot outputs back to original slot indices so `batch.*` and `out` preserve slot order.
    var inv_perm: [max_rows]usize = undefined;
    for (0..n) |k| inv_perm[perm[k]] = k;

    // Phase 1: sample ALL verifying slots' verdicts before running any `keepRows` GPU rollback, so slot i+1's verdict
    // never stalls behind slot i's rollback and a mid-group failure drops & resets every open mark before publishing.
    var verify_orig: [max_rows]usize = undefined;
    var verify_k: [max_rows]usize = undefined;
    var verify_cut: [max_rows]usize = undefined;
    var verify_verdicts: [max_rows]hy4_drafter.rounds.Verdict = undefined;
    var n_verifying: usize = 0;
    for (batch.*, 0..) |slot, orig| {
        const k = inv_perm[orig];
        const w_k: usize = final_w[k];
        if (w_k < 2) continue;
        const gen = if (slot.legacy_gen) |*g| g else return error.Hy4NoGenerator;
        // The think-room cut is the serial round's alone: `groupEligible` excludes a think bound and a forced call.
        const cut = hy4_drafter.rounds.roundRows(gen, ids_perm[k][0..w_k], w_k);
        var head = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(head);
        if (cut < w_k) {
            const vocab = mlx.getShape(views_perm[k])[2];
            try mlx.check(mlx.mlx_slice(&head, views_perm[k], &[_]c_int{ 0, 0, 0 }, 3, &[_]c_int{ 1, @intCast(cut), vocab }, 3, &[_]c_int{ 1, 1, 1 }, 3, xfm.s));
        }
        const logits = if (cut < w_k) head else views_perm[k];
        verify_verdicts[n_verifying] = try hy4_drafter.rounds.sampleVerdict(gen, logits, ids_perm[k][0..cut], null);
        verify_orig[n_verifying] = orig;
        verify_k[n_verifying] = k;
        verify_cut[n_verifying] = cut;
        n_verifying += 1;
    }
    const verify_ns: u64 = timer.read();

    // Phase 2: cut back every verifying slot's state back-to-back so GPU pipelines all `keepRows` rollbacks together.
    for (0..n_verifying) |vi| {
        const orig = verify_orig[vi];
        try hy4_drafter.finishGroupVerify(xfm, ctxs[orig], verify_verdicts[vi].accepted);
    }
    const finish_ns: u64 = timer.read();

    // Phase 3: commit verdicts, publish accepted blocks, and gather plain slots' logits.
    const out = try allocator.alloc(mlx.mlx_array, n);
    var made: usize = 0;
    errdefer {
        for (out[0..made]) |a| _ = mlx.mlx_array_free(a);
        allocator.free(out);
    }
    var total_tokens: u32 = 0;
    var kept: usize = 0;
    var vi_cur: usize = 0;
    for (batch.*, 0..) |slot, orig| {
        const k = inv_perm[orig];
        const w_k: usize = final_w[k];
        const gen = if (slot.legacy_gen) |*g| g else return error.Hy4NoGenerator;
        const req = reqs[orig];
        if (w_k < 2) {
            out[made] = views_perm[k];
            views_perm[k] = mlx.mlx_array_new();
            made += 1;
            batch.*[kept] = slot;
            kept += 1;
            ctxs[orig].cache.step = req.pos;
            total_tokens += 1;
        } else {
            const cut = verify_cut[vi_cur];
            const verdict = verify_verdicts[vi_cur];
            vi_cur += 1;
            const di = slot_draft_idx[orig].?;
            const b = &draft_out[di];
            const result = try hy4_drafter.rounds.commitVerdict(gen, allocator, ids_perm[k][0..cut], b.conf[0 .. cut - 1], verdict, batch_draft_ns, verify_ns);
            if (comptime !@import("builtin").is_test) if (cut > 1) {
                group_calib.observeRound(b.conf[0 .. cut - 1], @intCast(cut - 1), @intCast(result.accepted_tokens));
            };
            total_tokens += @intCast(result.tokens.len);
            @import("../scheduler.zig").publishSpeculativeBlock(sch, slot, gen, result.tokens);
            allocator.free(result.tokens);
            _ = mlx.mlx_array_free(views_perm[k]);
            views_perm[k] = mlx.mlx_array_new();
        }
    }
    if (comptime !@import("builtin").is_test) {
        const final_verify_ms: f64 = @as(f64, @floatFromInt(finish_ns)) / 1e6;
        group_cost.observeVerify(n, total_rows, final_verify_ms, draft_ms);
        if (!any_forced and total_tokens > 0) {
            group_gates[n].observeRound(draft_ms + final_verify_ms, total_tokens);
        }
    }
    batch.* = batch.*[0..kept];
    return try allocator.realloc(out, kept);
}

var last_model_ptr: usize = 0;
var group_gates: [max_rows + 1]@import("../spec_cost_gate.zig").Gate = @splat(.{});
var group_gate_sig: [max_rows + 1]u64 = @splat(0);
var group_cost: GroupCost = .{};
var group_calib: @import("../dflash_policy.zig").Calibration = .{ .identity_prior = true, .seen = @splat(10) };
var group_dfly_logged = false;

fn syncModelState(m: *const hy4_model.Model, resident: bool) void {
    const ptr = @intFromPtr(m);
    if (ptr == last_model_ptr) return;
    last_model_ptr = ptr;
    group_gates = @splat(.{});
    group_gate_sig = @splat(0);
    group_cost = GroupCost.forLoad(resident);
    group_calib = .{ .identity_prior = true, .seen = @splat(10) };
    group_dfly_logged = false;
}

fn dropVerify(xfm: *Trunk, ctx: *ForwardCtx, reset: bool) void {
    const m = xfm.arch_state.hy4flash.model orelse return;
    const r = hy4_state.requestOf(ctx) orelse return;
    const lent = m.lend(r);
    defer lent.end();
    @import("hy4_state.zig").dropVerifyReq(m, r, reset);
}

fn plainBatchedForward(
    sch: *Scheduler,
    allocator: std.mem.Allocator,
    xfm: *Trunk,
    next_tokens: []const u32,
    ctxs: []const *ForwardCtx,
    observe_plain_gate: bool,
    prior_draft_ms: f64,
) ![]mlx.mlx_array {
    const m = xfm.arch_state.hy4flash.model orelse return error.Hy4ModelMissing;
    var segs: [max_rows]hy4_model.Model.Seg = undefined;
    if (ctxs.len > @as(usize, @intCast(@import("hy4_forward.zig").joinedRowsMax()))) return error.Hy4RowsWidth;
    for (ctxs, next_tokens, 0..) |ctx, t, i| {
        segs[i] = .{ .request = hy4_state.requestOf(ctx) orelse return error.Hy4NoRequest, .id = t };
    }
    if (@import("builtin").is_test) hy4_state.joined_ticks_for_test += 1;
    var plain_timer = if (observe_plain_gate) @import("../io_util.zig").Stopwatch.init(sch.io) else undefined;
    if (m.plans != null) {
        const core = @import("../tensor/core.zig");
        const GemmScope = @import("sushi_exl3").linear.GemmScope;
        try @import("hy4_plan.zig").beginJoined(m, segs[0..ctxs.len]);
        const scope = GemmScope.begin() catch |e| {
            core.abortNarrow();
            return e;
        };
        if (xfm.expert_stream) |engine| engine.beginForward();
        const joined = m.extendRows(segs[0..ctxs.len]) catch |e| {
            if (xfm.expert_stream) |engine| engine.finishForward(ctxs.len);
            scope.end();
            core.abortNarrow();
            return e;
        };
        if (xfm.expert_stream) |engine| engine.finishForward(ctxs.len);
        const out = rowsIntoRequests(allocator, joined, segs[0..ctxs.len], m.logits_rows, xfm.s) catch |e| {
            _ = mlx.mlx_array_free(joined);
            scope.end();
            core.abortNarrow();
            return e;
        };
        _ = mlx.mlx_array_free(joined);
        scope.end();
        try @import("hy4_plan.zig").endJoined(m, segs[0..ctxs.len]);
        if (observe_plain_gate) {
            try mlx.check(mlx.mlx_array_eval(out[out.len - 1]));
            const ms: f64 = @as(f64, @floatFromInt(plain_timer.read())) / 1e6;
            group_cost.observePlain(ctxs.len, ms);
            if (prior_draft_ms > 0) {
                group_gates[ctxs.len].observeRound(prior_draft_ms + ms, @intCast(ctxs.len));
            } else {
                group_gates[ctxs.len].observePlain(ms / @as(f64, @floatFromInt(ctxs.len)));
            }
        }
        for (ctxs, segs[0..ctxs.len]) |ctx, sg| ctx.cache.step = sg.request.pos;
        return out;
    }
    if (xfm.expert_stream) |engine| engine.beginForward();
    defer if (xfm.expert_stream) |engine| engine.finishForward(ctxs.len);
    const joined = try m.extendRows(segs[0..ctxs.len]);
    defer _ = mlx.mlx_array_free(joined);
    for (ctxs, segs[0..ctxs.len]) |ctx, sg| ctx.cache.step = sg.request.pos;
    return rowsOf(allocator, joined, ctxs.len, xfm.s);
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
    syncModelState(m, xfm.config.expert_resident);
    const n = ctxs.len;
    if (n < joined_rows_min) return error.Hy4RowsWidth;
    if (xfm.arch_state.hy4flash.drafter != null and (!@import("builtin").is_test or knobs.groups_on_for_test) and groupable(m, batch.*, ctxs)) {
        const cap: usize = joinedCap(xfm, m);
        const budget = groupBudget(@intCast(m.config.dfly.draft), cap, n);
        if (budget >= 2) {
            if (comptime !@import("builtin").is_test) {
                const sig = batchSignature(ctxs);
                if (sig != group_gate_sig[n]) {
                    group_gate_sig[n] = sig;
                    if (group_gates[n].mode == .plain) {
                        group_gates[n].mode = .rounds;
                        group_gates[n].window = 0;
                        group_gates[n].ms = 0;
                        group_gates[n].tokens = 0;
                        group_gates[n].strike = false;
                        group_gates[n].trial = false;
                        group_gates[n].plain_tokens = 0;
                    }
                }
            }
            var any_forced = false;
            for (batch.*) |slot| {
                const g = if (slot.legacy_gen) |*gen| gen else continue;
                if (@import("hy4_drafter.zig").groupEligible(g) and (g.drafter_policy.range.min_rows > 1 or g.drafter_diag == .rows or g.drafter_diag == .sweep)) {
                    any_forced = true;
                    break;
                }
            }
            if (!@import("builtin").is_test and !any_forced and group_gates[n].next() == .plain) {
                return plainBatchedForward(sch, allocator, xfm, next_tokens, ctxs, true, 0);
            }
            return groupForward(sch, allocator, xfm, batch, next_tokens, ctxs);
        }
    }
    return plainBatchedForward(sch, allocator, xfm, next_tokens, ctxs, false, 0);
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
        hc.commit_media_chain = slot.media_chain;
        hy4_prefix.writeEndPoint(hc, m, &slot.cache, tokens, slot.has_tools, media, slot.restored_disk);
        return;
    }
    hc.commit_media_chain = slot.media_chain;
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
    // Streamed Cooler load with >= 2 * top_k slots/layer joins min(joinedRowsMax, slots / top_k) rows.
    const hy4_plan = @import("hy4_plan.zig");
    try std.testing.expectEqual(@as(usize, 1), hy4_plan.streamedRowsCap(512, 16, 0, 16));
    try std.testing.expectEqual(@as(usize, 1), hy4_plan.streamedRowsCap(512, 16, 31, 16));
    try std.testing.expectEqual(@as(usize, 2), hy4_plan.streamedRowsCap(512, 16, 32, 16));
    try std.testing.expectEqual(@as(usize, 8), hy4_plan.streamedRowsCap(512, 16, 128, 16));
    try std.testing.expectEqual(@as(usize, 16), hy4_plan.streamedRowsCap(512, 16, 256, 16));
    try std.testing.expectEqual(@as(usize, 16), hy4_plan.streamedRowsCap(512, 16, 390, 16));
    try std.testing.expectEqual(@as(usize, 1), hy4_plan.streamedRowsCap(8, 2, 3, 16));
    try std.testing.expectEqual(@as(usize, 2), hy4_plan.streamedRowsCap(8, 2, 4, 16));
    try std.testing.expectEqual(@as(usize, 3), hy4_plan.streamedRowsCap(8, 2, 6, 16));
    config.hy4flash = try @import("hy4_config.zig").parseText(@import("hy4_config.zig").release_file);
    config.cooler_for_test = true;
    config.expert_cache_slots = 31;
    try std.testing.expectEqual(@as(usize, 1), coolerRows(&config).rows);
    try std.testing.expect(!batches(&config));
    config.expert_cache_slots = 128;
    try std.testing.expectEqual(@min(want, @as(usize, 8)), coolerRows(&config).rows);
    try std.testing.expect(batches(&config));
    try std.testing.expectEqual(@as(?usize, @min(want, @as(usize, 8))), rowCap(&config));
    config.expert_cache_slots = 390;
    try std.testing.expectEqual(want, coolerRows(&config).rows);
    try std.testing.expectEqual(@as(?usize, want), rowCap(&config));
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

test "hy4 rows: confidence knapsack concentrates verify rows on high-confidence code slots and prunes prose to 1" {
    const code_p = [_]f32{ 0.95, 0.93, 0.92 };
    const prose_p = [_]f32{ 0.15, 0.10, 0.10 };
    var out: [4]usize = undefined;
    allocateGroupWidths(
        4,
        16,
        33.5,
        4.65,
        &.{ 1, 1, 1, 1 },
        &.{ 4, 4, 4, 4 },
        &.{ &code_p, &prose_p, &code_p, &prose_p },
        &out,
    );
    try std.testing.expectEqualSlices(usize, &.{ 4, 1, 4, 1 }, &out);

    // All low-confidence prose prunes every slot to 1 plain row.
    allocateGroupWidths(
        4,
        16,
        33.5,
        4.65,
        &.{ 1, 1, 1, 1 },
        &.{ 4, 4, 4, 4 },
        &.{ &prose_p, &prose_p, &prose_p, &prose_p },
        &out,
    );
    try std.testing.expectEqualSlices(usize, &.{ 1, 1, 1, 1 }, &out);
}

test "hy4 rows: planned group templates are sorted descending within cap and pickPlannedTemplate favors mixed code+prose" {
    const hy4_plan = @import("hy4_plan.zig");
    var templates: [20][max_rows]u16 = undefined;
    for ([_]usize{ 2, 3, 4, 5, 6, 7, 8 }) |slots| {
        const cnt = hy4_plan.groupWidthTemplates(5, 16, slots, &templates);
        try std.testing.expect(cnt >= 2);
        for (0..cnt) |ti| {
            var sum: usize = 0;
            for (0..slots) |k| {
                try std.testing.expect(templates[ti][k] >= 1 and templates[ti][k] <= 6);
                if (k > 0) try std.testing.expect(templates[ti][k] <= templates[ti][k - 1]);
                sum += templates[ti][k];
            }
            try std.testing.expect(sum <= 16 and sum > slots);
        }
    }
    const cnt4 = hy4_plan.groupWidthTemplates(5, 16, 4, &templates);
    var allowed: [20]bool = @splat(true);
    const code_p = [_]f32{ 0.96, 0.94, 0.92 };
    const prose_p = [_]f32{ 0.12, 0.10, 0.10 };
    const probs: [4][]const f32 = .{ &prose_p, &code_p, &prose_p, &code_p };
    // Sorted descending order places the two code slots (1, 3) first, prose (0, 2) last.
    const perm = [_]usize{ 1, 3, 0, 2 };
    const picked = pickPlannedTemplate(4, templates[0..cnt4], allowed[0..cnt4], &perm, &.{ 1, 1, 1, 1 }, &.{ 4, 4, 4, 4 }, &probs, 33.5, 4.65, true).?;
    try std.testing.expectEqualSlices(u16, &.{ 4, 4, 1, 1 }, templates[picked][0..4]);

    // When all 4 slots are low-confidence prose, plain wins (null).
    const all_prose: [4][]const f32 = .{ &prose_p, &prose_p, &prose_p, &prose_p };
    try std.testing.expectEqual(@as(?usize, null), pickPlannedTemplate(4, templates[0..cnt4], allowed[0..cnt4], &perm, &.{ 1, 1, 1, 1 }, &.{ 4, 4, 4, 4 }, &all_prose, 33.5, 4.65, true));

    // `groupDraftWidth` keeps `slots * (n_use + 1) <= joined` (16-row NAX decode arm); N=8 mixed (4 code + 4 prose)
    // drafts 1 token (`groupDraftWidth == 1`) and picks `[2,2,2,2,1,1,1,1]` to avoid verifying low-confidence prose.
    try std.testing.expectEqual(@as(usize, 5), hy4_plan.groupDraftWidth(5, 16, 2));
    try std.testing.expectEqual(@as(usize, 4), hy4_plan.groupDraftWidth(5, 16, 3));
    try std.testing.expectEqual(@as(usize, 3), hy4_plan.groupDraftWidth(5, 16, 4));
    try std.testing.expectEqual(@as(usize, 2), hy4_plan.groupDraftWidth(5, 16, 5));
    try std.testing.expectEqual(@as(usize, 1), hy4_plan.groupDraftWidth(5, 16, 8));
    const cnt8 = hy4_plan.groupWidthTemplates(5, 16, 8, &templates);
    const probs8: [8][]const f32 = .{ &code_p, &code_p, &code_p, &code_p, &prose_p, &prose_p, &prose_p, &prose_p };
    const perm8 = [_]usize{ 0, 1, 2, 3, 4, 5, 6, 7 };
    const picked8 = pickPlannedTemplate(8, templates[0..cnt8], allowed[0..cnt8], &perm8, &.{ 1, 1, 1, 1, 1, 1, 1, 1 }, &.{ 2, 2, 2, 2, 2, 2, 2, 2 }, &probs8, 42.0, 3.2, true).?;
    try std.testing.expectEqualSlices(u16, &.{ 2, 2, 2, 2, 1, 1, 1, 1 }, templates[picked8][0..8]);

    // Over-budget min_rows in `allocateGroupWidths` never exceeds cap.
    var out8: [8]usize = undefined;
    allocateGroupWidths(8, 16, 42.0, 3.2, &.{ 4, 4, 4, 4, 4, 4, 4, 4 }, &.{ 4, 4, 4, 4, 4, 4, 4, 4 }, &probs8, &out8);
    var sum8: usize = 0;
    for (out8) |w| {
        try std.testing.expect(w >= 1);
        sum8 += w;
    }
    try std.testing.expect(sum8 <= 16);
}
