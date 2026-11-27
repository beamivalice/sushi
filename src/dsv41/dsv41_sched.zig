//! DeepSeek-V4.1 as the scheduler sees it: a streamed load runs one request at a time over a decode state its slot
//! request holds while it runs and the model parks between requests; a fully resident load joins its slots' requests as
//! rows of one decode forward, each over a state of its own. Restore points always go to SSD, and a thought point is
//! written behind a DSpark round.
const std = @import("std");
const testTrunk = @import("../test_support.zig").testTrunk;
const arch_handle = @import("../arch/handle.zig");
const mlx = @import("../mlx.zig");
const model = @import("../model.zig");
const transformer = @import("../trunk.zig");
const prefix_cache = @import("../prefix_cache.zig");
const prefix_arch = @import("../arch/prefix.zig");
const model_registry = @import("../model_registry.zig");
const scheduler = @import("../scheduler.zig");
const generate = @import("../generate.zig");
const sched = @import("../arch/sched.zig");
const dsv41_prefix = @import("dsv41_prefix.zig");
const dsv41_borrow = @import("dsv41_borrow.zig");
const thought_point = @import("dsv41_thought_point.zig");
const dsv41_config = @import("dsv41_config.zig");
const dsv41_model = @import("dsv41_model.zig");
const dsv41_state = @import("dsv41_state.zig");
const dsv41_serving = @import("dsv41_serving.zig");
const log = @import("../log.zig");

const Slot = scheduler.Slot;
const Scheduler = scheduler.Scheduler;
const Trunk = transformer.Trunk;
const ForwardCtx = transformer.ForwardCtx;
const Generator = generate.Generator;
const ModelConfig = model.ModelConfig;

pub fn facts(config: *const ModelConfig) sched.Facts {
    _ = config;
    return .{ .always_write_through = true };
}

pub fn coolerRows(config: *const ModelConfig) sched.CoolerRows {
    if (!config.expert_resident) {
        const cap = if (config.dsv41) |c| dsv41_config.streamedRowsCap(c.experts, c.top_k, config.expert_cache_slots, joinedRowsMax()) else 1;
        if (cap <= 1) return .{ .rows = 1, .why = "its routed experts stream from SSD (the expert cache is one request's working set)" };
        if (!@import("../cooler_side.zig").onEngine(config)) return .{ .rows = 1, .why = "its decode runs on MLX" };
        if (config.dsv41) |c| if (!c.widthExactLayout()) return .{ .rows = 1, .why = "its FP8 trunk or FP4 experts give a row other bytes beside other rows" };
        return .{ .rows = cap, .note = "streamed expert cache holds >= 2 * top_k slots/layer; joined rows amortize dense trunk and coalesce SSD fills" };
    }
    // Rows are Sushi Cooler's: the user's opt-out to MLX keeps the one-request decode it had.
    if (!@import("../cooler_side.zig").onEngine(config)) return .{ .rows = 1, .why = "its decode runs on MLX" };
    if (config.dsv41) |c| if (!c.widthExactLayout()) return .{ .rows = 1, .why = "its FP8 trunk or FP4 experts give a row other bytes beside other rows" };
    return .{ .rows = joinedRowsMax(), .note = "steps run the exact arm (+~0.6 ms a step on NAX); a next turn restores from the SSD tier" };
}

/// The rows a joined forward carries with each row its one-row step's bytes (`Model.maxJoinedRows`): 16 with matrix
/// units, 8 without.
fn joinedRowsMax() usize {
    return @intCast(@import("sushi_exl3").project_rows.rowsMax(.{ .exact = true, .wide = true }));
}

/// The rows the load joins; 1 for a streamed load.
fn joinedRows(config: *const ModelConfig) usize {
    return coolerRows(config).rows;
}

pub fn batches(config: *const ModelConfig) bool {
    return joinedRows(config) > 1;
}

pub fn rowCap(config: *const ModelConfig) ?usize {
    const n = joinedRows(config);
    return if (n > 1) n else null;
}

pub fn modelBatches(xfm: *const Trunk) bool {
    const m = xfm.arch_state.dsv41.model orelse return false;
    return m.serves_rows;
}

/// Each row's attention reads its own request's state: nothing pads.
pub fn padsBatch(config: *const ModelConfig) bool {
    return joinedRows(config) <= 1;
}

/// Two or more DeepSeek slots decoding together join into one tick: when DSpark is active, `groupForward` runs batched
/// DSpark drafting + joined multi-row verify across the group instead of serializing one slot at a time.
pub const rows_company_min: usize = 2;

/// A DSpark slot on a load that joins rows joins the group tick when in company (running grouped DSpark); alone it
/// runs solo DSpark.
pub fn tickClass(slot: *const Slot) sched.TickClass {
    const gen = if (slot.legacy_gen) |*g| g else return .none;
    if (!gen.drafter_on or gen.done) return .none;
    if (slot.sampling.constraint != null or @import("../sampler.zig").penaltyActive(slot.sampling)) return .none;
    return if (modelBatches(arch_handle.trunk(slot.model.handle orelse return .none))) .rows_company else .none;
}

/// A slot joins once its prefill made its request's state.
pub fn readyToBatch(xfm: *const Trunk, ctx: *ForwardCtx) bool {
    const m = xfm.arch_state.dsv41.model orelse return true;
    if (!m.serves_rows) return true;
    const r = dsv41_state.requestOf(ctx) orelse return false;
    return r.state != null;
}

/// Rows are positioned by each request's state; the tick moves the slot's offset as a solo step would leave it.
pub fn batchArm(xfm: *Trunk, ctxs: []const *ForwardCtx) ?sched.BatchArm {
    const m = xfm.arch_state.dsv41.model orelse return null;
    if (!m.serves_rows) return null;
    for (ctxs) |c| if (!readyToBatch(xfm, c)) return null;
    return .{ .moe_positions = false, .advances_offset = true };
}

/// Per-drafting-slot verify row budget (including `t1`) when `n_drafting` of `n_slots` slots draft inside `cap` total
/// joined rows: plain slots reserve 1 row each, and remaining rows divide evenly across drafting slots up to `block + 1`.
fn groupBudget(block: usize, cap: usize, n_slots: usize, n_drafting: usize) usize {
    if (n_drafting == 0 or n_slots > cap) return 1;
    const avail = cap -| (n_slots - n_drafting);
    return @min(block + 1, @max(1, avail / n_drafting));
}

/// Per-drafting-slot draft width handed to `draftBatch` (`sum(widths) <= cap`): allows high-confidence slots at N=8
/// to draft 2 tokens (8 * 2 = 16 <= cap) so the confidence knapsack can verify 3 rows on code slots and 1 on prose.
fn groupDraftWidth(block: usize, cap: usize, n_drafting: usize) usize {
    if (n_drafting == 0 or cap == 0) return 0;
    return @min(block, @max(1, cap / n_drafting));
}

/// Online linear batch row-cost model for grouped DSpark verify width allocation.
const GroupCost = struct {
    base_ms: f64 = 12.5,
    row_ms: f64 = 3.8,
    draft_ms: f64 = 4.0,
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

/// Exact greedy marginal-gain row allocator across `n` slots (`sum(out_rows) <= cap`):
/// Starts every slot at `min_rows[i]` (at least 1, clamped so remaining slots always keep at least 1 row inside `cap`),
/// then iteratively grants +1 verify row to the drafting slot with the highest calibrated prefix survival probability
/// `q`, stopping as soon as adding that row would reduce expected batch tokens/sec (`q * exp_ms <= exp_tok * row_ms`).
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

fn sigmoidF32(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

fn batchSignature(ctxs: []const *ForwardCtx) u64 {
    var h: u64 = 0;
    for (ctxs) |ctx| {
        const id: u64 = if (ctx.ssm_member_gen != 0) ctx.ssm_member_gen else @intFromPtr(ctx);
        h ^= id *% 0x9e3779b97f4a7c15;
    }
    return h;
}

/// Grouped DSpark speculative tick across `batch`: drafts all eligible slots in ONE batched stage+LM-head pass,
/// allocates verify rows across slots by calibrated confidence marginal gain (`allocateGroupWidths`), verifies
/// all slots (drafting + pruned/plain slots) in ONE joined trunk forward, evaluates ALL verifying verdicts in ONE GPU
/// wait, cuts back & commits kept main rows in ONE joined `finishVerifyJoined`, and publishes each speculating slot's
/// accepted block. Plain and pruned 1-row slots stay in `batch.*` with their `[1, 1, vocab]` logits returned for sampling.
fn groupForward(
    sch: *Scheduler,
    allocator: std.mem.Allocator,
    xfm: *Trunk,
    batch: *[]*Slot,
    next_tokens: []const u32,
    ctxs: []const *ForwardCtx,
) ![]mlx.mlx_array {
    const dspark = @import("dsv41_dspark.zig");
    const sampler = @import("../sampler.zig");
    const io_util = @import("../io_util.zig");
    const m = xfm.arch_state.dsv41.model orelse return error.Dsv41NoModel;
    const n = ctxs.len;
    const cap = m.servedJoinedRows();

    // When a new set of requests enters concurrency `n`, clear any stale `.plain` cooldown inherited from an earlier
    // low-acceptance batch while keeping the calibrated `serial_ms` baseline.
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

    var any_forced = false;
    var wants_draft: [max_rows]bool = @splat(false);
    var n_drafting: usize = 0;
    for (batch.*, 0..) |slot, i| {
        const gen = if (slot.legacy_gen) |*g| g else continue;
        if (!dspark.groupEligible(gen)) continue;
        const diag = gen.drafter_diag;
        const forced = gen.drafter_policy.range.min_rows > 1 or diag == .rows or diag == .sweep;
        const want = switch (diag) {
            .off, .log => true,
            .rows => |r| r > 1,
            .sweep => true,
        };
        if (want) {
            wants_draft[i] = true;
            n_drafting += 1;
            if (forced) any_forced = true;
        }
    }
    const draft_w_cap = groupDraftWidth(m.cfg.dspark.block, cap, n_drafting);
    if (n_drafting == 0 or n >= cap or draft_w_cap == 0) return plainBatchedForward(sch, allocator, xfm, next_tokens, ctxs, false);
    if (!@import("builtin").is_test and !any_forced and group_gates[n].next() == .plain) {
        return plainBatchedForward(sch, allocator, xfm, next_tokens, ctxs, true);
    }

    var draft_slot_idx: [max_rows]usize = undefined;
    var draft_reqs: [max_rows]*dsv41_model.Request = undefined;
    var draft_t1s: [max_rows]u32 = undefined;
    var draft_widths: [max_rows]u32 = undefined;
    var draft_blocks: [max_rows]dsv41_model.Draft = undefined;
    var drafted_made: usize = 0;
    defer for (draft_blocks[0..drafted_made]) |*b| b.deinit();

    var di: usize = 0;
    for (batch.*, ctxs, 0..) |slot, ctx, i| {
        if (!wants_draft[i]) continue;
        const gen = &slot.legacy_gen.?;
        const req = dsv41_state.requestOf(ctx) orelse return error.Dsv41NoRequest;
        const rem = gen.max_tokens - gen.completion_tokens;
        var max_d = @min(draft_w_cap, @min(cap - n, rem -| 1));
        switch (gen.drafter_diag) {
            .rows => |r| max_d = @min(max_d, r -| 1),
            .off, .log => {
                max_d = @min(max_d, gen.drafter_policy.range.max_rows -| 1);
                if (gen.drafter_policy.draftLimit()) |lim| max_d = @min(max_d, lim);
            },
            .sweep => {},
        }
        if (max_d == 0) {
            wants_draft[i] = false;
            continue;
        }
        draft_slot_idx[di] = i;
        draft_reqs[di] = req;
        draft_t1s[di] = gen.next_token_id;
        draft_widths[di] = @intCast(max_d);
        di += 1;
    }
    n_drafting = di;
    if (n_drafting == 0) return plainBatchedForward(sch, allocator, xfm, next_tokens, ctxs, false);

    var draft_timer = io_util.Stopwatch.init(sch.io);
    try dsv41_serving.dsv41DraftBatch(xfm, draft_reqs[0..n_drafting], draft_t1s[0..n_drafting], draft_widths[0..n_drafting], draft_blocks[0..n_drafting]);
    drafted_made = n_drafting;
    const draft_ns = draft_timer.read();
    const draft_ms: f64 = @as(f64, @floatFromInt(draft_ns)) / 1e6;

    // Allocate verify rows across all slots by calibrated confidence marginal gain (exact greedy knapsack).
    var slot_draft_idx: [max_rows]?usize = @splat(null);
    for (0..n_drafting) |d| slot_draft_idx[draft_slot_idx[d]] = d;
    var min_rows_slot: [max_rows]usize = @splat(1);
    var max_rows_slot: [max_rows]usize = @splat(1);
    var prob_buf: [max_rows][16]f32 = undefined;
    var prob_slices: [max_rows][]const f32 = @splat(&.{});
    var chosen_rows: [max_rows]usize = @splat(1);
    const fair_budget = groupBudget(m.cfg.dspark.block, cap, n, n_drafting);
    for (batch.*, 0..) |slot, i| {
        const d = slot_draft_idx[i] orelse continue;
        const gen = &slot.legacy_gen.?;
        const full_len = 1 + draft_widths[d];
        const cut = dspark.roundRows(gen, draft_blocks[d].ids[0..full_len], full_len);
        max_rows_slot[i] = cut;
        min_rows_slot[i] = switch (gen.drafter_diag) {
            .rows => |r| @min(cut, @max(@as(usize, 1), @min(r, fair_budget))),
            .sweep => @min(cut, fair_budget),
            .off, .log => @min(cut, @max(@as(usize, 1), @min(gen.drafter_policy.range.min_rows, fair_budget))),
        };
        const n_conf = @min(cut -| 1, draft_blocks[d].conf.len);
        for (0..n_conf) |j| {
            prob_buf[i][j] = gen.drafter_policy.calib.probability(sigmoidF32(draft_blocks[d].conf[j]));
        }
        prob_slices[i] = prob_buf[i][0..n_conf];
    }
    // Price marginal verify rows against true plain tick latency (`plainAt(n)`, excluding sunk `draft_ms`) so we only
    // spend extra verify rows whose marginal tokens/ms beat plain joined decode.
    allocateGroupWidths(
        n,
        cap,
        group_cost.plainAt(n),
        group_cost.row_ms,
        min_rows_slot[0..n],
        max_rows_slot[0..n],
        prob_slices[0..n],
        chosen_rows[0..n],
    );

    var segs: [max_rows]dsv41_model.Seg = undefined;
    var plain_ids: [max_rows][1]u32 = undefined;
    var verify_slot_idx: [max_rows]usize = undefined;
    var verify_draft_idx: [max_rows]usize = undefined;
    var verify_reqs: [max_rows]*dsv41_model.Request = undefined;
    var slot_verify_idx: [max_rows]?usize = @splat(null);
    var n_verifying: usize = 0;
    var total_verify_rows: usize = 0;

    var opened: usize = 0;
    errdefer for (ctxs[0..opened]) |ctx| dsv41_serving.dsv41DropGroupVerify(ctx);
    for (ctxs, 0..) |ctx, i| {
        const req = dsv41_state.requestOf(ctx) orelse return error.Dsv41NoRequest;
        if (slot_draft_idx[i]) |d| if (chosen_rows[i] >= 2) {
            const cut = chosen_rows[i];
            try dsv41_serving.dsv41BeginGroupVerify(xfm, ctx, cut);
            segs[i] = .{ .request = req, .ids = draft_blocks[d].ids[0..cut] };
            slot_verify_idx[i] = n_verifying;
            verify_slot_idx[n_verifying] = i;
            verify_draft_idx[n_verifying] = d;
            verify_reqs[n_verifying] = req;
            n_verifying += 1;
            total_verify_rows += cut;
            opened = i + 1;
            continue;
        };
        plain_ids[i] = .{next_tokens[i]};
        segs[i] = .{ .request = req, .ids = &plain_ids[i] };
        total_verify_rows += 1;
        opened = i + 1;
    }

    if (!group_dspark_logged and n_verifying > 0) {
        group_dspark_logged = true;
        log.info("[dsv41-dspark] grouped DSpark engaged: {d} slots ({d} verifying, {d} total rows)\n", .{ n, n_verifying, total_verify_rows });
    }

    var verify_timer = io_util.Stopwatch.init(sch.io);
    const joined = try dsv41_serving.forwardRowsDeferred(xfm, segs[0..n]);
    defer _ = mlx.mlx_array_free(joined);

    const vs = dsv41_serving.verdictStream(xfm, xfm.s);
    const vocab: c_int = mlx.getShape(joined)[1];
    var views: [max_rows]mlx.mlx_array = @splat(mlx.mlx_array_new());
    defer for (views[0..n]) |v| if (v.ctx != null) {
        _ = mlx.mlx_array_free(v);
    };
    var row_off: usize = 0;
    for (segs[0..n], 0..) |sg, i| {
        const rows_i = sg.ids.len;
        const start = [2]c_int{ @intCast(row_off), 0 };
        const stop = [2]c_int{ @intCast(row_off + rows_i), vocab };
        const step = [2]c_int{ 1, 1 };
        var slice2 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(slice2);
        try mlx.check(mlx.mlx_slice(&slice2, joined, &start, 2, &stop, 2, &step, 2, vs));
        const shape3 = [3]c_int{ 1, @intCast(rows_i), vocab };
        try mlx.check(mlx.mlx_reshape(&views[i], slice2, &shape3, 3, vs));
        row_off += rows_i;
    }

    // If confidence pruning kept every slot at 1 row, evaluate once (for gate accounting) and return plain views.
    if (n_verifying == 0) {
        try mlx.check(mlx.mlx_array_eval(joined));
        const verify_ns = verify_timer.read();
        if (comptime !@import("builtin").is_test) {
            const verify_ms: f64 = @as(f64, @floatFromInt(verify_ns)) / 1e6;
            group_cost.observePlain(n, verify_ms);
            if (!any_forced) group_gates[n].observeRound(draft_ms + verify_ms, @intCast(n));
        }
        const out = try allocator.alloc(mlx.mlx_array, n);
        for (ctxs, segs[0..n], 0..) |ctx, sg, i| {
            ctx.cache.step = sg.request.state.?.n;
            out[i] = views[i];
            views[i] = mlx.mlx_array_new();
        }
        return out;
    }

    // Encode ALL verifying slots' verdicts lazily, then wait ONCE across the whole group.
    var lvs: [max_rows]dspark.LazyVerdict = @splat(.{});
    var lv_made: usize = 0;
    defer for (lvs[0..lv_made]) |*lv| lv.deinit();
    var lazy_arrs: [max_rows]mlx.mlx_array = undefined;
    for (0..n_verifying) |vi| {
        const si = verify_slot_idx[vi];
        const gen = &batch.*[si].legacy_gen.?;
        lvs[vi] = try dspark.verdictBegin(dspark.verdictInFor(gen, views[si], segs[si].ids, null));
        lv_made = vi + 1;
        lazy_arrs[vi] = lvs[vi].lazy();
    }
    const eval_vec = mlx.mlx_vector_array_new_data(&lazy_arrs, n_verifying);
    defer _ = mlx.mlx_vector_array_free(eval_vec);
    try mlx.check(mlx.mlx_eval(eval_vec));
    const verify_ns = verify_timer.read();

    var verdicts: [max_rows]dspark.Verdict = undefined;
    var accepteds: [max_rows]usize = undefined;
    for (0..n_verifying) |vi| {
        const si = verify_slot_idx[vi];
        const gen = &batch.*[si].legacy_gen.?;
        verdicts[vi] = try lvs[vi].read(segs[si].ids[1..]);
        if (!sampler.isGreedyTemperature(gen.sampling.temperature)) gen.sampling.draw +%= verdicts[vi].accepted + 1;
        accepteds[vi] = verdicts[vi].accepted;
    }

    // Cut back all verifying requests and commit kept main rows in ONE joined `ds.main_proj` pass.
    try dsv41_serving.dsv41FinishGroupVerify(xfm, verify_reqs[0..n_verifying], accepteds[0..n_verifying]);
    if (comptime !@import("builtin").is_test) {
        const finish_ms: f64 = @as(f64, @floatFromInt(verify_timer.read())) / 1e6;
        group_cost.observeVerify(n, total_verify_rows, finish_ms, draft_ms);
        if (!any_forced) {
            var total_tokens: u32 = @intCast(n - n_verifying);
            for (accepteds[0..n_verifying]) |acc| total_tokens += @intCast(1 + acc);
            group_gates[n].observeRound(draft_ms + finish_ms, total_tokens);
        }
    }

    const out = try allocator.alloc(mlx.mlx_array, n);
    var kept: usize = 0;
    errdefer {
        for (out[0..kept]) |a| _ = mlx.mlx_array_free(a);
        allocator.free(out);
    }
    for (batch.*, ctxs, 0..) |slot, ctx, i| {
        const req = segs[i].request;
        ctx.cache.step = req.state.?.n;
        if (slot_verify_idx[i]) |vi| {
            const gen = &slot.legacy_gen.?;
            const d = verify_draft_idx[vi];
            slot.moe_seq_offset = req.state.?.n;
            const cut = segs[i].ids.len;
            const result = try dspark.commitVerdict(gen, allocator, segs[i].ids, draft_blocks[d].conf[0 .. cut - 1], verdicts[vi], draft_ns, verify_ns);
            defer allocator.free(result.tokens);
            @import("../scheduler.zig").publishSpeculativeBlock(sch, slot, gen, result.tokens);
            afterSpecBlock(slot, gen, result.tokens);
        } else {
            out[kept] = views[i];
            views[i] = mlx.mlx_array_new();
            batch.*[kept] = slot;
            kept += 1;
        }
    }
    batch.* = batch.*[0..kept];
    return try allocator.realloc(out, kept);
}

fn plainBatchedForward(
    sch: *Scheduler,
    allocator: std.mem.Allocator,
    xfm: *Trunk,
    next_tokens: []const u32,
    ctxs: []const *ForwardCtx,
    observe_gate: bool,
) ![]mlx.mlx_array {
    const io_util = @import("../io_util.zig");
    var segs: [max_rows]dsv41_model.Seg = undefined;
    var ids: [max_rows][1]u32 = undefined;
    if (ctxs.len > max_rows) return error.Dsv41RowsWidth;
    for (ctxs, next_tokens, 0..) |ctx, t, i| {
        ids[i] = .{t};
        segs[i] = .{ .request = dsv41_state.requestOf(ctx) orelse return error.Dsv41NoRequest, .ids = &ids[i] };
    }
    if (!rows_logged) {
        rows_logged = true;
        log.info("[dsv41-rows] joined decode tick engaged: {d} rows\n", .{ctxs.len});
    }
    var timer = if (observe_gate) io_util.Stopwatch.init(sch.io) else undefined;
    const joined = try dsv41_serving.forwardRowsDeferred(xfm, segs[0..ctxs.len]);
    defer _ = mlx.mlx_array_free(joined);
    if (observe_gate) {
        try mlx.check(mlx.mlx_array_eval(joined));
        if (comptime !@import("builtin").is_test) {
            const ms: f64 = @as(f64, @floatFromInt(timer.read())) / 1e6;
            group_cost.observePlain(ctxs.len, ms);
            group_gates[ctxs.len].observePlain(ms / @as(f64, @floatFromInt(ctxs.len)));
        }
    }
    for (ctxs, segs[0..ctxs.len]) |ctx, sg| ctx.cache.step = sg.request.state.?.n;
    return rowsOf(allocator, joined, ctxs.len, dsv41_serving.verdictStream(xfm, xfm.s));
}

/// One decode forward over the group's requests: runs grouped DSpark (`groupForward`) whenever any slot in the batch
/// is DSpark-eligible and fits in the joined row budget, else plain joined rows (`plainBatchedForward`).
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
    if (ctxs.len > max_rows) return error.Dsv41RowsWidth;
    const m = xfm.arch_state.dsv41.model orelse return error.Dsv41NoModel;
    syncModelState(m, xfm.config.expert_resident);
    if (m.dspark != null and ctxs.len < m.servedJoinedRows()) {
        const dspark = @import("dsv41_dspark.zig");
        for (batch.*) |slot| {
            if (slot.legacy_gen) |*g| if (dspark.groupEligible(g)) {
                return groupForward(sch, allocator, xfm, batch, next_tokens, ctxs);
            };
        }
    }
    return plainBatchedForward(sch, allocator, xfm, next_tokens, ctxs, false);
}

const max_rows = 16;
var last_model_ptr: usize = 0;
var rows_logged = false;
var group_dspark_logged = false;
var group_gates: [max_rows + 1]@import("../spec_cost_gate.zig").Gate = @splat(.{});
var group_gate_sig: [max_rows + 1]u64 = @splat(0);
var group_cost: GroupCost = .{};

fn syncModelState(m: *const dsv41_model.Model, resident: bool) void {
    const ptr = @intFromPtr(m);
    if (ptr == last_model_ptr) return;
    last_model_ptr = ptr;
    rows_logged = false;
    group_dspark_logged = false;
    group_gates = @splat(.{});
    group_gate_sig = @splat(0);
    group_cost = GroupCost.forLoad(resident);
}

/// Row `i` of `[n, vocab]` logits as the `[1, 1, vocab]` a one-row step returns, for each row.
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

/// A slot's request holds the decode state while it runs; the model parks it between requests.
pub fn initRequest(allocator: std.mem.Allocator, config: *const ModelConfig) !?dsv41_model.Request {
    _ = config;
    return try dsv41_model.Request.init(allocator, 0);
}

pub fn modelStateBytes(slot: *const Slot) u64 {
    return thought_point.dsv41StateBytes(slot);
}

pub fn parkedStateBytes(entry: *model_registry.LoadedModel) u64 {
    return thought_point.dsv41ModelStateBytes(entry);
}

/// The next exclusive request is admitted as soon as the slot leaves `decoding` and takes the record over, so a
/// commit at the cleanup drain would publish onto its state.
pub fn commitsAtCull(slot: *const Slot) bool {
    const xfm = arch_handle.trunk(slot.model.handle orelse return false);
    return xfm.arch_state.dsv41.model != null;
}

pub const prepare_commit_label = "DeepSeek row marker";
pub const cancelled_entry_label = "a DeepSeek entry";

/// The cache holds a one-byte-a-token row marker for the model's rows.
pub fn prepareCommit(slot: *Slot) !void {
    const xfm = arch_handle.trunk(slot.model.handle orelse return);
    const dsv = xfm.arch_state.dsv41.model orelse return;
    try dsv41_prefix.markRows(&slot.cache, @import("dsv41_model.zig").stateStream(dsv));
}

pub fn publishLive(sch: *Scheduler, slot: *Slot, hc: *prefix_cache.HotPrefixCache, tokens: []const u32) void {
    thought_point.publishLive(sch, slot, hc, tokens);
}

/// A restore point is the whole state, so a cancelled prefill with none restores as a miss.
pub fn cancelledNeedsCheckpoints() bool {
    return true;
}

/// With joined rows the restore installs into the slot's own request, lent until `restoreEnd`.
pub fn restoreTarget(slot: *Slot, xfm: *Trunk) ?sched.RestoreTarget {
    const dsv = xfm.arch_state.dsv41.model orelse return null;
    if (dsv.serves_rows) {
        // Nothing is lent at a lookup, so `restoreEnd` ends this lend by clearing it.
        std.debug.assert(dsv.bound == null);
        _ = dsv41_model.lendFor(dsv, slotRequest(slot));
    }
    return .{ .rec = prefix_arch.recurrent("dsv41", dsv), .log_tag = "[dsv41-prefix]" };
}

/// The lookup runs off any forward, where nothing is lent: the lend `restoreTarget` opened is the only one to end.
pub fn restoreEnd(slot: *Slot, xfm: *Trunk) void {
    const dsv = xfm.arch_state.dsv41.model orelse return;
    if (!dsv.serves_rows) return;
    std.debug.assert(dsv.bound == slotRequest(slot));
    dsv41_model.unbindRows(dsv, null);
}

/// The slot's DeepSeek request; null when it holds none.
fn slotRequest(slot: *Slot) ?*dsv41_model.Request {
    return if (slot.request) |*r| r.as(.dsv41) else null;
}

/// With joined rows a hook reads the slot's own state, lent for its span (`dsv41_model.lendFor`).
fn lendSlot(slot: *Slot) ?dsv41_model.Lent {
    const xfm = arch_handle.trunk(slot.model.handle orelse return null);
    return dsv41_model.lendFor(xfm.arch_state.dsv41.model orelse return null, slotRequest(slot));
}

pub fn writeThrough(
    hc: *prefix_cache.HotPrefixCache,
    slot: *Slot,
    entry_id: *?u64,
    prompt: []const u32,
    media: prefix_cache.MediaStamp,
    cps: []transformer.SSMCheckpoint,
    _: mlx.mlx_stream,
) bool {
    _ = entry_id;
    const dsv = arch_handle.trunk(slot.model.handle orelse return false).arch_state.dsv41.model orelse return false;
    const lent = lendSlot(slot);
    defer if (lent) |l| l.end();
    hc.commit_restored = slot.restored_disk;
    hc.commit_media_chain = slot.media_chain;
    dsv41_prefix.writeThroughDsv41(hc, dsv, &slot.cache, prompt, slot.has_tools, media, cps, "prefill");
    return true;
}

pub fn armThoughtPoint(slot: *Slot, gen: *Generator, tier_takes_points: bool) void {
    thought_point.armDsv41ThoughtPoint(slot, gen, tier_takes_points);
}

pub fn afterSpecBlock(slot: *Slot, gen: *Generator, block: []const u32) void {
    const lent = lendSlot(slot);
    defer if (lent) |l| l.end();
    thought_point.writeDsv41ThoughtPoint(slot, gen, block);
}

pub fn endRequest(config: *const ModelConfig) void {
    dsv41_borrow.endRequestMlx(config);
}

pub fn flushResident(entry: *model_registry.LoadedModel, why: []const u8) void {
    dsv41_prefix.flushResident(entry, why);
}

test "an arch whose request state lives in the model restores into it under its own log tag and needs checkpoints from a cancelled prefill" {
    var xfm = testTrunk("deepseek_v41");
    var lm: model_registry.LoadedModel = undefined;
    var slot = sched.testSlot(&xfm, &lm, .{ .model_type = "deepseek_v41" });
    // A model the hooks only name, and read only as a one-row load.
    const m = try std.testing.allocator.create(@import("dsv41_model.zig").Model);
    defer std.testing.allocator.destroy(m);
    m.serves_rows = false;
    xfm.arch_state.dsv41.model = m;
    const target = sched.restoreTarget(&slot, &xfm).?;
    try std.testing.expectEqualStrings("[dsv41-prefix]", target.log_tag);
    try std.testing.expect(!target.takes_dflash and !target.takes_mtp);
    try std.testing.expect(sched.cancelledNeedsCheckpoints(&slot));
    try std.testing.expect(sched.commitsAtCull(&slot));
    xfm.arch_state.dsv41.model = null;
    try std.testing.expect(sched.restoreTarget(&slot, &xfm) == null);
    try std.testing.expect(!sched.commitsAtCull(&slot));
}

test "a DeepSeek slot with no DSpark round ticks alone or joins as plain, and two drafting slots of the same model join for grouped DSpark" {
    const sched_policy = @import("../sched_policy.zig");
    var xfm = testTrunk("deepseek_v41");
    var lm: model_registry.LoadedModel = undefined;
    var other: model_registry.LoadedModel = undefined;
    var slots = [_]Slot{ sched.testSlot(&xfm, &lm, .{ .model_type = "deepseek_v41" }), sched.testSlot(&xfm, &lm, .{ .model_type = "deepseek_v41" }), sched.testSlot(&xfm, &lm, .{ .model_type = "deepseek_v41" }), sched.testSlot(&xfm, &other, .{ .model_type = "deepseek_v41" }) };
    // A slot with no generator drafting: no rows class.
    try std.testing.expectEqual(sched.TickClass.none, sched.tickClass(&slots[0]));
    try std.testing.expectEqual(sched_policy.TickBucket.serial, sched_policy.bucketOf(.spec_active, .none, .{}));
    try std.testing.expectEqual(sched_policy.TickBucket.batch, sched_policy.bucketOf(.ok, .none, .{}));
    // Two of the model's slots are company for grouped DSpark; a slot of another model never counts.
    try std.testing.expect(!sched.rowsInCompany(&slots[0], &.{&slots[0]}, &.{}));
    try std.testing.expect(!sched.rowsInCompany(&slots[0], &.{&slots[0]}, &.{&slots[3]}));
    try std.testing.expect(sched.rowsInCompany(&slots[0], &.{&slots[0]}, &.{&slots[1]}));
    try std.testing.expectEqual(@as(usize, 6), groupBudget(5, 16, 2, 2));
    try std.testing.expectEqual(@as(usize, 4), groupBudget(5, 16, 4, 4));
    try std.testing.expectEqual(@as(usize, 2), groupBudget(5, 16, 8, 8));
    try std.testing.expectEqual(@as(usize, 5), groupBudget(5, 16, 4, 3));
}

test "streamed DeepSeek coolerRows joins rows when the expert cache holds at least 2 * top_k slots per layer" {
    try std.testing.expectEqual(@as(usize, 1), dsv41_config.streamedRowsCap(384, 8, 0, 16));
    try std.testing.expectEqual(@as(usize, 1), dsv41_config.streamedRowsCap(384, 8, 15, 16));
    try std.testing.expectEqual(@as(usize, 2), dsv41_config.streamedRowsCap(384, 8, 16, 16));
    try std.testing.expectEqual(@as(usize, 8), dsv41_config.streamedRowsCap(384, 8, 64, 16));
    try std.testing.expectEqual(@as(usize, 16), dsv41_config.streamedRowsCap(384, 8, 128, 16));
    try std.testing.expectEqual(@as(usize, 16), dsv41_config.streamedRowsCap(384, 8, 262, 16));
    try std.testing.expectEqual(@as(usize, 2), dsv41_config.streamedRowsCap(8, 2, 4, 16));
    try std.testing.expectEqual(@as(usize, 3), dsv41_config.streamedRowsCap(8, 2, 6, 16));
    try std.testing.expectEqual(@as(usize, 1), dsv41_config.streamedRowsCap(8, 2, 3, 16));
    var cfg = ModelConfig{ .model_type = "deepseek_v41", .expert_streaming = true, .cooler_for_test = true };
    cfg.dsv41 = try dsv41_config.releaseStqForTest();
    // Unset or tight streamed cache (< 2 * top_k; release top_k == 6) stays exclusive (1 row).
    try std.testing.expectEqual(@as(usize, 1), coolerRows(&cfg).rows);
    cfg.expert_cache_slots = 11;
    try std.testing.expectEqual(@as(usize, 1), coolerRows(&cfg).rows);
    // >= 2 * top_k slots unlocks joined rows capped at min(joinedRowsMax, slots / top_k) on Cooler, still refused on MLX.
    cfg.expert_cache_slots = 48;
    try std.testing.expectEqual(@min(joinedRowsMax(), @as(usize, 8)), coolerRows(&cfg).rows);
    cfg.expert_cache_slots = 96;
    try std.testing.expectEqual(joinedRowsMax(), coolerRows(&cfg).rows);
    try std.testing.expect(batches(&cfg));
    cfg.cooler_for_test = false;
    try std.testing.expectEqual(@as(usize, 1), coolerRows(&cfg).rows);
}

test "allocateGroupWidths concentrates verify rows on high-confidence slots and prunes low-confidence slots to 1" {
    try std.testing.expectEqual(@as(usize, 5), groupDraftWidth(5, 16, 2));
    try std.testing.expectEqual(@as(usize, 4), groupDraftWidth(5, 16, 4));
    try std.testing.expectEqual(@as(usize, 2), groupDraftWidth(5, 16, 8));

    // N=8 mixed (4 code @ 0.92,0.88 + 4 prose @ 0.40,0.30), cap=16, base_ms=48, row_ms=3.8:
    // Code slots get 3 rows (2 drafts), prose slots prune to 1 row (0 wasted verify rows), total = 16!
    const code = [_]f32{ 0.92, 0.88 };
    const prose = [_]f32{ 0.40, 0.30 };
    const mins: [8]usize = @splat(1);
    const maxs: [8]usize = @splat(3);
    const probs = [_][]const f32{ &code, &code, &code, &code, &prose, &prose, &prose, &prose };
    var out: [8]usize = undefined;
    allocateGroupWidths(8, 16, 48.0, 3.8, &mins, &maxs, &probs, &out);
    try std.testing.expectEqual([8]usize{ 3, 3, 3, 3, 1, 1, 1, 1 }, out);

    // All low confidence at N=8 -> every slot prunes to 1 row.
    const low_probs: [8][]const f32 = @splat(&prose);
    allocateGroupWidths(8, 16, 48.0, 3.8, &mins, &maxs, &low_probs, &out);
    try std.testing.expectEqual([8]usize{ 1, 1, 1, 1, 1, 1, 1, 1 }, out);

    // N=2 (1 code + 1 prose), cap=16, base_ms=23, row_ms=3.8: code takes 5 rows, prose takes 2 rows.
    const c5 = [_]f32{ 0.95, 0.90, 0.85, 0.70, 0.45 };
    const p5 = [_]f32{ 0.65, 0.50, 0.35, 0.25, 0.20 };
    var out2: [2]usize = undefined;
    allocateGroupWidths(2, 16, 23.0, 3.8, &.{ 1, 1 }, &.{ 6, 6 }, &.{ &c5, &p5 }, &out2);
    try std.testing.expect(out2[0] >= 4 and out2[0] <= 5);
    try std.testing.expectEqual(@as(usize, 2), out2[1]);

    // Over-budget min_rows never exceeds cap and leaves every slot at least 1 row.
    const big_mins: [8]usize = @splat(4);
    const big_maxs: [8]usize = @splat(4);
    allocateGroupWidths(8, 16, 48.0, 3.8, &big_mins, &big_maxs, &low_probs, &out);
    var sum_out: usize = 0;
    for (out) |w| {
        try std.testing.expect(w >= 1);
        sum_out += w;
    }
    try std.testing.expect(sum_out <= 16);
}
