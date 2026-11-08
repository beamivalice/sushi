const std = @import("std");
const sched_policy = @import("sched_policy.zig");
const mlx = @import("mlx.zig");
const trunk = @import("trunk.zig");
const moe_gqmv = @import("moe_gqmv.zig");
const generate_mod = @import("generate.zig");
const depth_bounds = @import("mtp_depth_bounds.zig");
const model_registry_mod = @import("model_registry.zig");
const log = @import("log.zig");
const io_util = @import("io_util.zig");
const Trunk = trunk.Trunk;
const arch_state = @import("arch/arch_state.zig");
const qwen4_state = @import("qwen4/qwen4_state.zig");
const Generator = generate_mod.Generator;
const MtpHeadRef = generate_mod.MtpHeadRef;
const round_cost_mod = @import("round_cost.zig");
const group_cost_mod = @import("mtp_group_cost.zig");
const LoadedModel = model_registry_mod.LoadedModel;
const slot_admission = @import("slot_admission.zig");
const MAX_BATCH_GROUP = sched_policy.MAX_BATCH_GROUP;
const batchedKvKeepCount = sched_policy.batchedKvKeepCount;
const sched = @import("arch/sched.zig");
const slotMtpGroupable = sched.mtpGroupable;
const Planner = @import("mtp_group_planner.zig");
const degenerate = @import("degenerate.zig");
const mtp_plan = @import("mtp_plan.zig");
const sampler = @import("sampler.zig");
const testing = std.testing;
const scheduler_mod = @import("scheduler.zig");
const slot_outcome = @import("slot_outcome.zig");
const mtp_round = @import("mtp_round.zig");
const mtp_policy = @import("mtp_policy.zig");
const LoopAction = scheduler_mod.LoopAction;
const Scheduler = scheduler_mod.Scheduler;
const Slot = scheduler_mod.Slot;
const SpecTickMode = sched_policy.SpecTickMode;
const batchedTickRows = scheduler_mod.batchedTickRows;
const capMtpRoundToThinkBound = scheduler_mod.capMtpRoundToThinkBound;
const finishSlot = slot_outcome.finishSlot;
const loopGuardTick = scheduler_mod.loopGuardTick;
const loopRecoveryAction = scheduler_mod.loopRecoveryAction;
const loopStopReason = slot_outcome.loopStopReason;
const publishSpeculativeBlock = scheduler_mod.publishSpeculativeBlock;
const runBatchedDecodeTick = scheduler_mod.runBatchedDecodeTick;
const runDecodeTick = scheduler_mod.runDecodeTick;
const runSingleDecodeTick = scheduler_mod.runSingleDecodeTick;
const specTickMode = sched_policy.specTickMode;
const thinkBoundDraftLimit = scheduler_mod.thinkBoundDraftLimit;
const thinkBoundForcesPlain = scheduler_mod.thinkBoundForcesPlain;

pub fn plannerShape(slots: []const *Slot, widths: []const u8) group_cost_mod.GroupShape {
    const xfm = slots[0].model.transformer.?;
    var shape = group_cost_mod.GroupShape{ .n = @intCast(slots.len), .kv_format = cacheCostFormat(slots[0].cache.config) };
    for (slots, widths, 0..) |slot, width, i| {
        const gen = &slot.legacy_gen.?;
        const kv: u8 = @intCast(xfm.round_cost.bucketOf(@intCast(slot.moe_seq_offset)));
        if (width == 0) {
            shape.rows[i] = group_cost_mod.GroupShape.row(0, 0, 0, kv, 0, 0, group_cost_mod.GroupShape.samplingMode(false, false, false, !sampler.isGreedyTemperature(gen.sampling.temperature)));
        } else {
            const off = Generator.mtpRoundOff0(gen.mtp_hist_stash, gen.mtp_cache.?.step());
            const history = 1 + if (gen.mtp_hist_stash) |stash| stash.n else @as(usize, 0);
            shape.head_format = cacheCostFormat(qwen4_state.mtpOf(xfm).?.cache.config);
            shape.rows[i] = group_cost_mod.GroupShape.row(width, width, width, kv, @intCast(xfm.round_cost.bucketOf(@intCast(off))), @intCast(@min(history, round_cost_mod.MAX_WIDTH + 2)), gen.mtpPlannerMode());
        }
    }
    return shape;
}

fn plannerInputSame(a: group_cost_mod.GroupShape, b: group_cost_mod.GroupShape) bool {
    var aa = a;
    var bb = b;
    aa.attention = 0;
    aa.moe = 0;
    aa.head_calls = 0;
    bb.attention = 0;
    bb.moe = 0;
    bb.head_calls = 0;
    return std.meta.eql(aa, bb);
}

fn plannerPriceTransition(previous: u8, width: u8, pending: bool) bool {
    return width > 0 and (pending or previous != width);
}

fn plannerRecurringInputSame(a: group_cost_mod.GroupShape, b: group_cost_mod.GroupShape) bool {
    if (a.n != b.n or a.width() == 0 or b.width() == 0) return false;
    inline for (.{ @as(u6, 24), @as(u6, 32) }) |shift| {
        var a_max: u8 = 0;
        var b_max: u8 = 0;
        for (a.rows[0..a.n], b.rows[0..b.n]) |arow, brow| {
            a_max = @max(a_max, group_cost_mod.GroupShape.part(arow, shift));
            b_max = @max(b_max, group_cost_mod.GroupShape.part(brow, shift));
        }
        if (a_max != b_max) return false;
    }
    var aa = a;
    var bb = b;
    const recurring_mask = ~((@as(u64, 0xff) << 24) | (@as(u64, 0xff) << 32) | (@as(u64, 0xff) << 40));
    for (aa.rows[0..aa.n], bb.rows[0..bb.n]) |*arow, *brow| {
        if (group_cost_mod.GroupShape.part(arow.*, 40) < 2 or group_cost_mod.GroupShape.part(brow.*, 40) < 2) return false;
        arow.* &= recurring_mask;
        brow.* &= recurring_mask;
    }
    return plannerInputSame(aa.canonical(), bb.canonical());
}

const PlannerPrices = struct {
    slots: []const *Slot,
    geometry: group_cost_mod.GroupShape,

    fn init(slots: []const *Slot) PlannerPrices {
        var widths: [Planner.MAX_ROWS]u8 = @splat(0);
        for (slots, 0..) |slot, i| widths[i] = @intFromBool(slot.legacy_gen.?.mtp != null and slot.legacy_gen.?.mtp_cache != null);
        return .{ .slots = slots, .geometry = plannerShape(slots, widths[0..slots.len]) };
    }

    fn keyFor(self: *const PlannerPrices, slots: []const *Slot, widths: []const u8) group_cost_mod.GroupShape {
        var key = self.geometry;
        key.n = @intCast(slots.len);
        key.rows = @splat(0);
        key.head_format = 0;
        for (slots, widths, 0..) |slot, width, i| {
            for (self.slots, 0..) |original, j| {
                if (original != slot) continue;
                const code = self.geometry.rows[j];
                key.rows[i] = if (width == 0) group_cost_mod.GroupShape.row(0, 0, 0, group_cost_mod.GroupShape.part(code, 24), 0, 0, group_cost_mod.GroupShape.part(code, 48) & 8) else (code & ~@as(u64, 0xffffff)) | @as(u64, width) * 0x10101;
                if (width > 0) key.head_format = self.geometry.head_format;
                break;
            }
        }
        return key.canonical();
    }

    fn sample(self: *const PlannerPrices, slots: []const *Slot, widths: []const u8, trusted: bool) ?Planner.Price {
        if (slots.len == 0) return .{ .ms = 0, .samples = std.math.maxInt(u32) };
        const key = self.keyFor(slots, widths);
        const table = &slots[0].model.transformer.?.mtp_group_cost;
        var selected: ?Planner.Price = null;
        var score: f32 = 0;
        var max_gap: f32 = 0;
        var mean_upper: f32 = 0;
        var mean_lower: f32 = std.math.inf(f32);
        const plain = key.width() == 0;
        for (table.shapes) |entry| {
            if (entry.cell.n == 0 or !plannerInputSame(key, entry.shape)) continue;
            if (trusted and entry.cell.n < Planner.MIN_SAMPLES) continue;
            const deviation = 2 * @sqrt(entry.variance);
            const mean_deviation = 2 * @sqrt(entry.variance * entry.mean_weight_sq);
            mean_upper = @max(mean_upper, entry.cell.ms + mean_deviation);
            mean_lower = @min(mean_lower, entry.cell.ms - mean_deviation);
            const value = if (plain) entry.cell.ms - deviation else entry.cell.ms + deviation;
            for (entry.gap_ms[0..key.n]) |gap| max_gap = @max(max_gap, gap);
            if (selected == null or (if (plain) value < score else value > score)) {
                score = value;
                selected = .{ .ms = entry.cell.ms, .variance = entry.variance, .samples = entry.cell.n };
            }
        }
        if (selected) |*value| {
            value.max_gap_ms = max_gap;
            if (!plain) value.mean_upper_ms = mean_upper;
            if (plain) value.mean_lower_ms = mean_lower;
            return selected;
        }
        if (plain) return null;

        var pooled_n: u32 = 0;
        var sum: f64 = 0;
        var sum_sq: f64 = 0;
        var mean_sum_sq: f64 = 0;
        max_gap = 0;
        for (table.shapes) |entry| {
            if (entry.cell.n == 0 or !plannerRecurringInputSame(key, entry.shape)) continue;
            const n: f64 = @floatFromInt(entry.cell.n);
            const mean: f64 = entry.cell.ms;
            pooled_n +|= entry.cell.n;
            sum += n * mean;
            sum_sq += n * (@as(f64, entry.variance) + mean * mean);
            // History-to-history differences remain uncertainty in a pooled prediction.
            mean_sum_sq += n * (@as(f64, entry.variance) * entry.mean_weight_sq + mean * mean);
            for (entry.gap_ms[0..key.n]) |gap| max_gap = @max(max_gap, gap);
        }
        const required: u32 = if (trusted) Planner.MIN_SAMPLES else 1;
        if (pooled_n < required) return null;
        const n: f64 = @floatFromInt(pooled_n);
        const mean = sum / n;
        const variance = @max(0, sum_sq / n - mean * mean);
        const mean_variance = @max(0, mean_sum_sq / n - mean * mean);
        return .{ .ms = @floatCast(mean), .variance = @floatCast(variance), .samples = pooled_n, .max_gap_ms = max_gap, .mean_upper_ms = @floatCast(mean + 2 * @sqrt(mean_variance)) };
    }

    pub fn price(self: *PlannerPrices, widths: []const u8) ?Planner.Price {
        var positive: [Planner.MAX_ROWS]*Slot = undefined;
        var drafts: [Planner.MAX_ROWS]u8 = undefined;
        var zeros: [Planner.MAX_ROWS]*Slot = undefined;
        const zero_widths: [Planner.MAX_ROWS]u8 = @splat(0);
        var pn: usize = 0;
        var zn: usize = 0;
        for (self.slots, widths) |slot, width| {
            if (width == 0) {
                zeros[zn] = slot;
                zn += 1;
            } else {
                positive[pn] = slot;
                drafts[pn] = width;
                pn += 1;
            }
        }
        const plain = self.sample(zeros[0..zn], zero_widths[0..zn], true) orelse return null;
        const spec = self.sample(positive[0..pn], drafts[0..pn], true) orelse return null;
        const deviation = @sqrt(plain.variance) + @sqrt(spec.variance);
        return .{ .ms = plain.ms + spec.ms, .variance = deviation * deviation, .samples = @min(plain.samples, spec.samples), .max_gap_ms = @max(plain.max_gap_ms, spec.max_gap_ms), .mean_upper_ms = if (spec.mean_upper_ms) |bound| plain.ms + 2 * @sqrt(plain.variance) + bound else null, .mean_lower_ms = if (pn == 0) plain.mean_lower_ms else null };
    }

    fn probeEstimate(self: *PlannerPrices, widths: []const u8, plain: Planner.Price) f32 {
        if (self.sample(self.slots, widths, false)) |value| return value.ms + 2 * @sqrt(value.variance);
        var widest: u8 = 0;
        for (widths) |width| widest = @max(widest, width);
        var lower: [Planner.MAX_ROWS]u8 = undefined;
        for ([_]u8{ 2, 1 }) |cap| {
            if (cap >= widest) continue;
            for (widths, 0..) |width, i| lower[i] = @min(width, cap);
            if (self.sample(self.slots, lower[0..widths.len], false)) |value| return scaledProbeEstimate(value, widest, cap);
        }
        if (self.slots.len >= 4 and self.slots.len % 2 == 0) {
            const half = self.slots.len / 2;
            if (self.sample(self.slots[0..half], widths[0..half], false)) |a| {
                if (self.sample(self.slots[half..], widths[half..], false)) |b| return a.ms + b.ms + 2 * (@sqrt(a.variance) + @sqrt(b.variance));
            }
        }
        return self.probeFallbackEstimate(widths, plain);
    }

    fn probeFallbackEstimate(_: *PlannerPrices, widths: []const u8, plain: Planner.Price) f32 {
        var widest: u8 = 0;
        for (widths) |width| widest = @max(widest, width);
        const upper = plain.ms + 2 * @sqrt(plain.variance);
        if (widest <= 2) return 2 * upper;
        return 1.5 * upper * @as(f32, @floatFromInt(widest + 1));
    }

    fn scaledProbeEstimate(value: Planner.Price, widest: u8, measured: u8) f32 {
        const ratio = @as(f32, @floatFromInt(widest + 1)) / @as(f32, @floatFromInt(measured + 1));
        return value.ms * ratio;
    }
};

var planner_engaged_logged = false;
var planner_calibration_block_logged = false;
var planner_depth_two_reject_logged = false;

const ProbeSource = enum { calibrate, recovery };

/// Widths for a calibration or recovery probe round, priced against the plain
/// tick. False leaves `decision` untouched: no candidate, no price, or over budget.
fn planProbe(
    prices: *PlannerPrices,
    active: []*Slot,
    rows: []const Planner.Row,
    decision: *Planner.Decision,
    comptime source: ProbeSource,
) bool {
    const zero_widths: [Planner.MAX_ROWS]u8 = @splat(0);
    var candidate: [Planner.MAX_ROWS]u8 = @splat(0);
    for (active, rows, 0..) |slot, row, i| {
        const gen = &slot.legacy_gen.?;
        candidate[i] = switch (source) {
            .calibrate => Planner.probeWidth(gen.mtp_planner_probes, gen.max_tokens -| gen.completion_tokens, row.cap),
            .recovery => gen.mtp_planner_recovery.width(gen.completion_tokens, gen.max_tokens -| gen.completion_tokens, row.cap, gen.mtp_planner_probes),
        } orelse return false;
    }
    const plain = prices.price(zero_widths[0..active.len]) orelse return false;
    const estimate = prices.probeEstimate(candidate[0..active.len], plain);
    var latency_limit: f32 = std.math.floatMax(f32);
    var available = true;
    for (rows) |row| {
        latency_limit = @min(latency_limit, row.latency_ms);
        if (estimate > row.latency_ms) available = false;
    }
    if (!available) {
        if (source == .calibrate and !planner_calibration_block_logged) {
            planner_calibration_block_logged = true;
            log.info("[mtp-planner] calibration blocked widths={any} predicted_ms={d:.2} latency_ms={d:.2}\n", .{ candidate[0..active.len], estimate, latency_limit });
        }
        return false;
    }
    decision.widths = candidate;
    decision.upper_ms = estimate;
    return true;
}

/// Drafts the planner may give an MTP row: the depth range, the tokens `max_tokens` leaves and the
/// thought room (0 = a plain row). A row with a think bound is `observe`d first.
fn plannerRowCap(gen: *const Generator) u32 {
    const depth = if (gen.mtp_depth > 0) gen.mtp_depth else @max(1, gen.mtp_depth_current);
    const cap = @min(@min(depth, Planner.MAX_DEPTH), gen.max_tokens -| gen.completion_tokens -| 1);
    return if (thinkBoundDraftLimit(gen)) |limit| @min(cap, limit) else cap;
}

pub fn tryPlannerTick(sch: *Scheduler, active: []*Slot) anyerror!bool {
    if (!Planner.enabled() or active.len == 0 or active.len > Planner.MAX_ROWS) return false;
    const xfm = active[0].model.transformer orelse return false;
    if (!qwen4_state.isLoaded(xfm) or qwen4_state.mtpOfConst(xfm) == null) return false;
    if (active.len == 1) {
        const gen = if (active[0].legacy_gen) |*g| g else return false;
        if (!gen.mtp_planner_owned) return false;
    }
    var rows: [Planner.MAX_ROWS]Planner.Row = undefined;
    var lenses: [Planner.MAX_ROWS]u32 = undefined;
    var has_mtp = false;
    var pending_pipeline = false;
    var pending_draft = false;
    for (active, 0..) |slot, i| {
        if (slot.model != active[0].model or !slot.allow_batch_mtp) return false;
        if (sampler.penaltyActive(slot.sampling)) return false;
        const gen = if (slot.legacy_gen) |*g| g else return false;
        if (arch_state.mropeOf(&gen.ctx.arch_ctx) != null) return false;
        pending_pipeline = pending_pipeline or gen.has_pending_logits or gen.has_pending_token;
        pending_draft = pending_draft or gen.mtp_pre_draft != null;
        const why = sch.batchVerdict(slot);
        const mtp_row = why == .spec_active and slotMtpGroupable(slot);
        if (why != .ok and !mtp_row) return false;
        has_mtp = has_mtp or mtp_row;
        if (gen.sampling.think_bound) |tb| tb.observe(gen.generated_ids.items);
        const cap = if (mtp_row) plannerRowCap(gen) else 0;
        rows[i] = .{ .history = &gen.mtp_planner_history, .cap = @intCast(cap), .latency_ms = Planner.LATENCY_MS };
        lenses[i] = @intCast(slot.moe_seq_offset);
    }
    if (!has_mtp) return false;
    std.sort.insertion(u32, lenses[0..active.len], {}, std.sort.asc(u32));
    if (active.len > 1 and batchedKvKeepCount(lenses[0..active.len]) != active.len) return false;
    const entry = Planner.entry(pending_pipeline, pending_draft);
    if (entry == .wait_predraft) {
        // Consume the pending rounds without perpetually replacing their successors.
        for (active) |slot| slot.legacy_gen.?.mtp_planner_pending = true;
        return false;
    }
    if (entry == .drain) {
        for (active, rows[0..active.len]) |slot, row| {
            const gen = &slot.legacy_gen.?;
            if (gen.mtp != null) try gen.mtpDetachHead(slot.allocator, true);
            if (row.cap > 0) gen.mtp_planner_owned = true;
            slot.planner_plain_transition = true;
            slot.mtp_plain_tick = row.cap > 0;
        }
        log.info("[mtp-planner] rows={d} action=drain\n", .{active.len});
        try runBatchedDecodeTick(sch, active);
        return true;
    }

    var live: [Planner.MAX_ROWS]*Slot = undefined;
    var count: usize = 0;
    for (active) |slot| {
        const gen = &slot.legacy_gen.?;
        if (try loopGuardTick(sch, slot, gen)) continue;
        if (try gen.checkDispatchStop()) {
            slot.stop_path = "batch";
            finishSlot(sch, slot, gen.finish_reason);
            continue;
        }
        live[count] = slot;
        count += 1;
    }
    if (count != active.len) {
        if (count > 0) try runDecodeTick(sch, live[0..count]);
        return true;
    }
    for (active, rows[0..active.len]) |slot, row| if (row.cap > 0) {
        slot.legacy_gen.?.mtp_planner_owned = true;
    };
    var prices = PlannerPrices.init(active);
    var decision = Planner.choose(rows[0..active.len], &prices);
    var probe = false;
    var recovering = false;
    var probe_rounds: u8 = std.math.maxInt(u8);
    for (active, rows[0..active.len]) |slot, row| if (row.cap > 0) {
        probe_rounds = @min(probe_rounds, slot.legacy_gen.?.mtp_planner_probes);
    };
    if (!planner_depth_two_reject_logged and probe_rounds >= 12 and Planner.shouldProbe(decision, probe_rounds)) {
        var depth_two: [Planner.MAX_ROWS]u8 = @splat(0);
        var expected: f32 = 0;
        for (rows[0..active.len], 0..) |row, i| {
            depth_two[i] = @min(2, row.cap);
            expected += row.history.expected(depth_two[i]);
        }
        planner_depth_two_reject_logged = true;
        log.info("[mtp-planner] depth-two rejected price={any} expected_tokens={d:.2} plain_rate={d:.1}\n", .{ prices.price(depth_two[0..active.len]), expected, decision.plain_rate });
    }
    if (Planner.shouldProbe(decision, probe_rounds)) {
        probe = planProbe(&prices, active, rows[0..active.len], &decision, .calibrate);
    }
    if (!probe and Planner.shouldProbe(decision, probe_rounds) and planProbe(&prices, active, rows[0..active.len], &decision, .recovery)) {
        probe = true;
        recovering = true;
    }
    // A width the range moves no longer matches what a probe was meant to price.
    for (rows[0..active.len], 0..) |row, i| {
        const width: u8 = @intCast(depth_bounds.plannerWidth(decision.widths[i], row.cap, depth_bounds.active));
        if (width != decision.widths[i] or depth_bounds.active.pinned() != null) {
            decision.widths[i] = width;
            probe = false;
            recovering = false;
        }
    }
    var stale: [Planner.MAX_ROWS]bool = undefined;
    for (active, 0..) |slot, row| stale[row] = slot.legacy_gen.?.mtp_hidden_stale;
    const execution = Planner.execution(decision.widths[0..active.len], stale[0..active.len]);
    const prime = execution.prime;
    var positives: [Planner.MAX_ROWS]*Slot = undefined;
    var plains: [Planner.MAX_ROWS]*Slot = undefined;
    const pn = execution.speculative_n;
    const zn = execution.plain_n;
    var changed = false;
    for (active, decision.widths[0..active.len]) |slot, width| {
        const row_changed = slot.planner_last_width != width;
        changed = changed or row_changed;
        slot.planner_price_transition = plannerPriceTransition(slot.planner_last_width, width, slot.planner_price_transition);
        slot.planner_last_width = width;
    }
    for (execution.plain[0..zn], 0..) |row, i| {
        const slot = active[row];
        const gen = &slot.legacy_gen.?;
        slot.planner_plain_transition = gen.mtp_hist_stash != null or gen.mtp_pre_draft != null or prime;
        if (gen.mtp != null) try gen.mtpDetachHead(slot.allocator, true);
        slot.mtp_plain_tick = execution.capture[row];
        plains[i] = slot;
    }
    for (execution.speculative[0..pn], 0..) |row, i| {
        const slot = active[row];
        const gen = &slot.legacy_gen.?;
        gen.mtp_planner_width = decision.widths[row];
        gen.mtp_planner_probe = probe;
        gen.mtp_planner_recovering = recovering;
        positives[i] = slot;
    }
    if (!planner_engaged_logged) {
        planner_engaged_logged = true;
        log.info("[mtp-planner] engaged: margin=5% noise=2sigma latency_ms={d:.1} max_rows=8\n", .{Planner.LATENCY_MS});
    }
    if (changed or prime or probe) log.info("[mtp-planner] rows={d} widths={any} action={s} predicted_ms={d:.2} rate={d:.1} plain={d:.1}\n", .{ active.len, decision.widths[0..active.len], if (prime) "prime" else if (recovering) "recover" else if (probe) "calibrate" else "choose", decision.upper_ms, decision.rate, decision.plain_rate });
    if (pn > 0) {
        try runBatchedMtpHeadTick(sch, positives[0..pn]);
    }
    if (zn > 0) try runBatchedDecodeTick(sch, plains[0..zn]);
    return true;
}

/// A plain batched tick samples no repeat or presence penalty, so a penalised slot keeps its MTP round.
pub fn anyPenalised(slots: []const *Slot) bool {
    for (slots) |slot| {
        if (sampler.penaltyActive(slot.sampling)) return true;
    }
    return false;
}

/// MTP slots on one model at or past this count decode on the plain batched tick instead. Every slot that reaches
/// `runMtpGroups` runs its rounds solo (see there), so the threshold is the same for all of them.
pub const mtp_crowd_threshold: usize = 3;

pub fn mtpCrowdThresholdFor(_: *const Slot) usize {
    return mtp_crowd_threshold;
}

fn runBatchedMtpHeadTick(sch: *Scheduler, group: []*Slot) !void {
    const cost_xfm = group[0].model.transformer.?;
    cost_xfm.cost_trace_active = qwen4_state.isLoaded(cost_xfm);
    cost_xfm.cost_attention = 0;
    cost_xfm.cost_moe = 0;
    const head_calls_before = moe_gqmv.mtp_head_row_dispatches;
    defer cost_xfm.cost_trace_active = false;
    defer for (group) |slot| {
        if (slot.legacy_gen) |*gen| gen.mtp_batch_head = false;
    };
    var tick_sw = io_util.Stopwatch.init(group[0].io);
    var gens: [MAX_BATCH_GROUP]*generate_mod.Generator = undefined;
    var chains: [MAX_BATCH_GROUP]mtp_round.MtpPreDraft = undefined;
    var opens: [MAX_BATCH_GROUP]mtp_round.MtpRoundOpen = undefined;
    var live: [MAX_BATCH_GROUP]*Slot = undefined;
    var n: usize = 0;
    var n_chain: usize = 0;
    defer for (opens[0..n]) |*o| {
        if (o.chain.drafts.len != 0) o.chain.deinit(live[0].allocator);
    };
    defer for (chains[0..n_chain]) |*c| c.deinit(live[0].allocator);

    for (group) |slot| {
        const gen = &slot.legacy_gen.?;
        if (try loopGuardTick(sch, slot, gen)) continue;
        gen.mtp_group_cap = 0;
        capMtpRoundToThinkBound(gen);
        gen.mtp_batch_head = true;
        const begun = gen.mtpRoundBegin(slot.allocator) catch |e| {
            gen.mtp_batch_head = false;
            slot.markError(@errorName(e));
            continue;
        };
        switch (begun) {
            .done => |r| publishMtpResult(sch, slot, gen, r),
            .verify => |st_in| {
                var state = st_in;
                defer state.deinit(slot.allocator);
                gen.mtpRoundVerify(&state) catch |e| {
                    slot.markError(@errorName(e));
                    continue;
                };
                mtpFinishPublish(sch, slot, gen, &state);
            },
            .open => |o| {
                opens[n] = o;
                gens[n] = gen;
                live[n] = slot;
                n += 1;
            },
        }
    }
    if (n == 0) return;

    var depth: u32 = 1;
    for (opens[0..n], 0..) |*o, i| {
        depth = @max(depth, o.chain.m);
        chains[i] = o.chain;
        o.chain = .{
            .plan = chains[i].plan,
            .off0 = 0,
            .t1 = 0,
            .t1_arr = .{ .ctx = null },
            .drafts = &.{},
            .draft_arrs = undefined,
            .n_drafted = 0,
            .conf_arrs = null,
            .n_conf = 0,
            .q_probs = null,
            .n_qp = 0,
            .h_chain = null,
            .m = 0,
        };
        n_chain = i + 1;
    }

    const chain_lap = mtp_policy.SubLap.start(generate_mod.Generator.mtpTraceOn(), live[0].io);
    generate_mod.Generator.mtpChainBuildBatched(gens[0..n], chains[0..n], 0, depth) catch |e| {
        for (live[0..n]) |slot| slot.markError(@errorName(e));
        return;
    };
    // Dispatched before the verify build so the chain's GPU time covers the build.
    generate_mod.Generator.mtpChainDispatchBatched(chains[0..n]) catch |e| {
        for (live[0..n]) |slot| slot.markError(@errorName(e));
        return;
    };
    if (chain_lap.read()) |chain_ns| {
        for (gens[0..n]) |gen| gen.mtpTraceSub(.chain, chain_ns);
    }

    const taken = n;
    n = 0;
    n_chain = 0;
    var states: [MAX_BATCH_GROUP]mtp_round.MtpRoundState = undefined;
    var vgens: [MAX_BATCH_GROUP]*generate_mod.Generator = undefined;
    var vlive: [MAX_BATCH_GROUP]*Slot = undefined;
    var vn: usize = 0;
    defer for (states[0..vn], vlive[0..vn]) |*st, slot| {
        st.deinit(slot.allocator);
    };
    for (0..taken) |i| {
        opens[i].chain = chains[i];
        states[vn] = gens[i].mtpRoundContinue(live[i].allocator, opens[i]) catch |e| {
            live[i].markError(@errorName(e));
            continue;
        };
        vgens[vn] = gens[i];
        vlive[vn] = live[i];
        vn += 1;
    }
    if (vn == 0) return;

    var sts: [MAX_BATCH_GROUP]*mtp_round.MtpRoundState = undefined;
    for (states[0..vn], 0..) |*st, i| sts[i] = st;
    if (generate_mod.Generator.mtpGroupVerify(vgens[0..vn], sts[0..vn])) |_| {
        generate_mod.Generator.mtpGroupAccept(vgens[0..vn], sts[0..vn]) catch |e| {
            for (vlive[0..vn]) |slot| slot.markError(@errorName(e));
            return;
        };
    } else |e| {
        for (vlive[0..vn]) |slot| slot.markError(@errorName(e));
        return;
    }
    for (states[0..vn], vgens[0..vn], vlive[0..vn]) |*st, gen, slot| {
        if (slot.state == .errored) continue;
        mtpFinishPublish(sch, slot, gen, st);
    }
    observeGroupRound(vlive[0..vn], states[0..vn], group.len, &tick_sw, head_calls_before);
}

/// Every MTP slot that reaches here is Flash-Next (`slotMtpGroupable` needs its GDN trunk) or MiMo
/// (`slotMimoMtpCrowdable`), and both keep each round solo: a MiMo verify holds each row's decode arithmetic only
/// in its own forward, and Flash-Next verify rows are expert bytes, so a batched verify measured no better.
pub fn runMtpGroups(sch: *Scheduler, slots: []*Slot) !void {
    for (slots) |slot| try runSingleDecodeTick(sch, slot);
}

fn groupCostSampleComplete(expected: usize, retained: []const u32, published: []const bool) bool {
    if (expected < 1 or expected > group_cost_mod.MAX_GROUP_ROWS or retained.len != expected or published.len != expected) return false;
    for (retained, published) |count, complete| if (count == 0 or !complete) return false;
    return true;
}

fn cacheCostFormat(config: trunk.KVQuantConfig) u64 {
    return (@as(u64, config.group_size) << 16) | (@as(u64, config.bits) << 8) | @backingInt(config.scheme);
}

/// One wall for the group grid: from the tick's first round-begin to its last publish.
/// Both group sites hand their tick stopwatch here rather than starting one of their own.
fn observeGroupRound(live: []*Slot, states: []const mtp_round.MtpRoundState, expected: usize, tick_sw: *io_util.Stopwatch, head_calls_before: u64) void {
    const xfm = live[0].model.transformer orelse return;
    if (live.len > group_cost_mod.MAX_GROUP_ROWS) return;
    var price_transition = false;
    for (live) |slot| price_transition = price_transition or slot.planner_price_transition;
    defer {
        for (live) |slot| slot.planner_price_transition = false;
    }
    var emitted: [group_cost_mod.MAX_GROUP_ROWS]u32 = undefined;
    var published: [group_cost_mod.MAX_GROUP_ROWS]bool = undefined;
    var gaps: [group_cost_mod.MAX_GROUP_ROWS]f32 = undefined;
    for (live, states, 0..) |slot, st, i| {
        emitted[i] = st.retained;
        published[i] = slot.state == .decoding and !slot.cancelled.load(.acquire);
        gaps[i] = slot.mtp_publish_gap_ms;
    }
    if (!groupCostSampleComplete(expected, emitted[0..live.len], published[0..live.len])) return;
    if (price_transition) return;
    var key = group_cost_mod.GroupShape{
        .n = @intCast(live.len),
        .kv_format = cacheCostFormat(live[0].cache.config),
        .head_format = if (qwen4_state.mtpOfConst(xfm)) |head| cacheCostFormat(head.cache.config) else 0,
        .attention = xfm.cost_attention,
        .moe = xfm.cost_moe,
        .head_calls = @intCast(@min(std.math.maxInt(u16), moe_gqmv.mtp_head_row_dispatches -% head_calls_before)),
    };
    for (live, states, 0..) |slot, st, i| {
        const chain = st.chain;
        if (chain.m > round_cost_mod.MAX_WIDTH or chain.n_drafted > round_cost_mod.MAX_WIDTH or chain.head_input_rows > round_cost_mod.MAX_WIDTH + 2) return;
        const mode = group_cost_mod.GroupShape.samplingMode(slot.legacy_gen.?.mtp.?.canRerankDrafts(), chain.conf_arrs != null, chain.q_probs != null, !sampler.isGreedyTemperature(slot.legacy_gen.?.sampling.temperature));
        key.rows[i] = group_cost_mod.GroupShape.row(@intCast(chain.m), @intCast(chain.n_drafted), @intCast(@min(chain.plan.m_lo, chain.n_drafted)), @intCast(xfm.round_cost.bucketOf(@intCast(st.moe_seq_offset_snap))), @intCast(xfm.round_cost.bucketOf(@intCast(chain.off0))), @intCast(chain.head_input_rows), mode);
        key.padding += @intCast(st.verify_len - (1 + chain.m));
    }
    const ms = @as(f32, @floatFromInt(tick_sw.read())) / @as(f32, std.time.ns_per_ms);
    _ = xfm.mtp_group_cost.observeShape(key, ms, emitted[0..live.len], gaps[0..live.len], .round);
}

/// One verify forward for a whole group: ragged rows ride the row-axis form, equal-lane
/// trunks the merged padded forward, one row a solo round.
/// Finish one row's round and publish it; a failed finish errors the slot instead.
fn mtpFinishPublish(sch: *Scheduler, slot: *Slot, gen: *Generator, st: *mtp_round.MtpRoundState) void {
    const r = gen.mtpRoundFinish(slot.allocator, st) catch |e| {
        slot.markError(@errorName(e));
        return;
    };
    publishMtpResult(sch, slot, gen, r);
}

fn publishMtpResult(sch: *Scheduler, slot: *Slot, gen: *Generator, result: ?Generator.DrafterStepResult) void {
    const r = result orelse {
        finishSlot(sch, slot, gen.finish_reason);
        return;
    };
    defer slot.allocator.free(r.tokens);
    publishSpeculativeBlock(sch, slot, gen, r.tokens);
}

test "loopStopReason: a VARIED-phrasing restatement loop is cut at the near-repeat tier" {
    // The agent-traffic shape (2026-08-04): the same intent restated forever in
    // slightly different words. No exact cycle exists, so tiers 1 and 2 are
    // blind and this ran to max_tokens.
    var ids = std.ArrayList(u32).empty;
    defer ids.deinit(testing.allocator);
    const phrasings = [_][]const u32{
        &[_]u32{ 40, 41, 42, 43, 44, 45, 46 },
        &[_]u32{ 40, 41, 42, 43, 44, 46 },
        &[_]u32{ 40, 41, 42, 43, 44, 45, 47, 48, 46 },
        &[_]u32{ 49, 40, 41, 42, 43, 44, 45, 46 },
        &[_]u32{ 40, 41, 42, 43, 44, 45, 50, 46 },
    };
    var i: usize = 0;
    while (ids.items.len < degenerate.near_repeat_min_span + 32) : (i += 1) {
        try ids.appendSlice(testing.allocator, phrasings[i % phrasings.len]);
    }
    const reason = loopStopReason(ids.items) orelse return error.TestExpectedLoopCut;
    try testing.expectEqualStrings("stop", reason);

    // A long answer that keeps introducing new material is untouched, however
    // repetitive its scaffolding.
    var healthy = std.ArrayList(u32).empty;
    defer healthy.deinit(testing.allocator);
    var line: u32 = 0;
    while (healthy.items.len < degenerate.near_repeat_window + 32) : (line += 1) {
        try healthy.appendSlice(testing.allocator, &[_]u32{ 10, 11, 12 });
        try healthy.append(testing.allocator, 1000 + line);
        try healthy.appendSlice(testing.allocator, &[_]u32{ 13, 14 });
    }
    try testing.expect(loopStopReason(healthy.items) == null);
}

test "only the models with per-slot head state are admitted to the MTP slot groups, whose rounds run solo" {
    var t: trunk.Trunk = undefined;
    t.arch_state = .{};
    t.moe_layers = &.{};
    t.expert_stream = null;
    for ([_][]const u8{ "qwen4_exp", "mimo_v2", "glm5_next", "deepseek_v41", "qwen3_moe", "" }) |model_type| {
        t.config = .{ .model_type = model_type, .full_attention_interval = 4 };
        const admitted = sched.modelBatches(&t);
        try testing.expectEqual(t.config.isQwen4() or t.config.isMimo(), admitted);
    }
}

test "a fired thinking bound keeps an MTP slot speculating and the answer in MTP ticks" {
    if (comptime !@import("arch/drafter.zig").Drafter.has("qwen4")) return error.SkipZigTest;
    var t: trunk.Trunk = undefined;
    var forced = [_]u32{7};
    var tb = mtp_plan.ThinkBound{ .budget = 4, .opener_id = null, .closer_id = 7, .forced = &forced, .in_think = true, .fired = true };
    var gen: Generator = undefined;
    gen.sampling = .{ .think_bound = &tb };
    gen.think_closed_by_loop = false;
    gen.mtp = MtpHeadRef.of("qwen4", &t);
    // What commitThoughtClose leaves behind after a head resumed.
    gen.spec_disabled_runtime = false;
    gen.spec_disable_reason = .none;
    try testing.expect(!thinkBoundForcesPlain(&gen));
    try testing.expectEqual(SpecTickMode.mtp, specTickMode(true, gen.mtp != null, false, false));
    // A head that may not resume leaves `.think_bound`: plain, as before.
    gen.spec_disable_reason = .think_bound;
    try testing.expect(thinkBoundForcesPlain(&gen));
    // DFlash2 carries no MTP head: a fired bound still decodes it plain.
    gen.mtp = null;
    gen.dspark = false;
    gen.spec_disable_reason = .none;
    try testing.expect(thinkBoundForcesPlain(&gen));
    gen.think_closed_by_loop = true;
    tb.fired = false;
    try testing.expect(thinkBoundForcesPlain(&gen));
}

test "an MTP round never accepts past the position serial decode closes the thought at" {
    // Serial decode fires when the thought holds `budget` tokens; a round emits t1 plus its accepted drafts.
    for ([_]u32{ 1, 2, 3, 9, 48 }) |budget| {
        for ([_]u32{ 0, 1, 2, 3, 5, 8 }) |accept_every| {
            var tb = mtp_plan.ThinkBound{ .budget = budget, .opener_id = null, .closer_id = 99, .forced = &.{99}, .in_think = true };
            var emitted: u32 = 0;
            var rounds: u32 = 0;
            while (!tb.due()) : (rounds += 1) {
                const room = tb.room().?;
                const drafts: u32 = mtp_plan.mtpDraftCapForRoom(room) orelse 0;
                const accepted = @min(drafts, accept_every);
                emitted += 1 + accepted;
                tb.count = emitted;
                try testing.expect(emitted <= budget);
            }
            try testing.expectEqual(budget, emitted);
        }
    }
    var tb = mtp_plan.ThinkBound{ .budget = 5, .opener_id = null, .closer_id = 9, .forced = &.{9}, .in_think = true, .count = 4 };
    try testing.expectEqual(@as(?u32, 1), tb.room());
    try testing.expectEqual(@as(?u32, null), mtp_plan.mtpDraftCapForRoom(1));
    try testing.expectEqual(@as(?u32, 3), mtp_plan.mtpDraftCapForRoom(4));
    tb.fired = true;
    try testing.expectEqual(@as(?u32, null), tb.room());
    tb.fired = false;
    tb.in_think = false;
    try testing.expectEqual(@as(?u32, null), tb.room());
}

test "the last thought token before the bound runs serial, wider rooms cap the drafts" {
    var tb = mtp_plan.ThinkBound{ .budget = 5, .opener_id = null, .closer_id = 9, .forced = &.{9}, .in_think = true, .count = 4 };
    var gen: Generator = undefined;
    gen.sampling = .{ .think_bound = &tb };
    gen.mtp_group_cap = 0;
    gen.mtp_serial_left = 0;
    capMtpRoundToThinkBound(&gen);
    try testing.expectEqual(@as(u32, 1), gen.mtp_serial_left);
    try testing.expectEqual(@as(u32, 0), gen.mtp_group_cap);
    gen.mtp_serial_left = 0;
    tb.count = 1;
    capMtpRoundToThinkBound(&gen);
    try testing.expectEqual(@as(u32, 3), gen.mtp_group_cap);
    try testing.expectEqual(@as(u32, 0), gen.mtp_serial_left);
    gen.sampling = .{};
    gen.mtp_group_cap = 0;
    capMtpRoundToThinkBound(&gen);
    try testing.expectEqual(@as(u32, 0), gen.mtp_group_cap);
}

test "a group's row cap and the thought room both bound a round's drafts" {
    var tb = mtp_plan.ThinkBound{ .budget = 12, .opener_id = null, .closer_id = 9, .forced = &.{9}, .in_think = true, .count = 1 };
    var gen: Generator = undefined;
    gen.sampling = .{ .think_bound = &tb };
    // Room 11 leaves the group's row cap of 3 in force.
    gen.mtp_group_cap = 3;
    gen.mtp_serial_left = 0;
    capMtpRoundToThinkBound(&gen);
    try testing.expectEqual(@as(u32, 3), gen.mtp_group_cap);
    // Room 4 caps below the row cap of 6.
    gen.mtp_group_cap = 6;
    tb.count = 8;
    capMtpRoundToThinkBound(&gen);
    try testing.expectEqual(@as(u32, 3), gen.mtp_group_cap);
    try testing.expectEqual(@as(u32, 0), gen.mtp_serial_left);
}

test "a planner row never gets more drafts than its thought room" {
    var tb = mtp_plan.ThinkBound{ .budget = 20, .opener_id = null, .closer_id = 9, .forced = &.{9}, .in_think = true, .count = 0 };
    var gen: Generator = undefined;
    gen.mtp_depth = 4;
    gen.mtp_depth_current = 4;
    gen.max_tokens = 1000;
    gen.completion_tokens = 10;
    gen.sampling = .{};
    try testing.expectEqual(@as(u32, 4), plannerRowCap(&gen));
    gen.sampling = .{ .think_bound = &tb };
    try testing.expectEqual(@as(u32, 4), plannerRowCap(&gen));
    tb.count = 17;
    try testing.expectEqual(@as(u32, 2), plannerRowCap(&gen));
    // Room 1: the slot takes a plain row.
    tb.count = 19;
    try testing.expectEqual(@as(u32, 0), plannerRowCap(&gen));
    // A bound that already fired no longer limits the answer.
    tb.fired = true;
    try testing.expectEqual(@as(u32, 4), plannerRowCap(&gen));
}

test "a plain batched tick checks the thinking budget before its rows forward" {
    // A thought past its budget with no room left to close it: the bound fires, the slot decodes on.
    const forced = [_]u32{ 9, 7 };
    var tb = mtp_plan.ThinkBound{ .budget = 2, .opener_id = 5, .closer_id = 7, .forced = &forced, .in_think = false };
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(testing.allocator);
    try ids.appendSlice(testing.allocator, &.{ 5, 11, 12, 13 });
    var gen: Generator = undefined;
    gen.generated_ids = ids;
    gen.loop_guard_start = 0;
    gen.sampling = .{ .think_bound = &tb };
    gen.completion_tokens = 4;
    gen.max_tokens = 4;
    gen.has_pending_logits = false;
    gen.has_pending_token = false;
    gen.step = 0;
    gen.next_token_id = 9;
    gen.eos_token_ids = &.{2};
    gen.consecutive_pad = 0;
    gen.timeout_ns = 0;
    gen.stall = .{};
    gen.timer = io_util.Stopwatch.init(testing.io);
    var slot: Slot = undefined;
    slot.legacy_gen = gen;
    var sch: Scheduler = undefined;
    var active = [_]*Slot{&slot};
    var live: [1]*Slot = undefined;
    try testing.expectEqual(@as(usize, 1), try batchedTickRows(&sch, &active, &live));
    try testing.expect(tb.fired);
}

test "group cost geometry rejects partial rounds and keeps complete cache formats" {
    try testing.expect(groupCostSampleComplete(2, &.{ 1, 3 }, &.{ true, true }));
    try testing.expect(!groupCostSampleComplete(3, &.{ 1, 3 }, &.{ true, true }));
    try testing.expect(!groupCostSampleComplete(2, &.{ 1, 0 }, &.{ true, true }));
    try testing.expect(!groupCostSampleComplete(2, &.{ 1, 3 }, &.{ true, false }));
    const a = trunk.KVQuantConfig.affine(8);
    var b = a;
    b.group_size = 128;
    try testing.expect(cacheCostFormat(a) != cacheCostFormat(b));
    try testing.expect(cacheCostFormat(a) != cacheCostFormat(trunk.KVQuantConfig.affine(4)));
    try testing.expect(cacheCostFormat(a) != cacheCostFormat(trunk.KVQuantConfig.affine(6)));
    try testing.expect(cacheCostFormat(trunk.KVQuantConfig.affine(4)) != cacheCostFormat(trunk.KVQuantConfig.affine(6)));
}

test "scheduler prices each shared execution once and preserves row sampling geometry" {
    var xfm: Trunk = undefined;
    xfm.round_cost = .{ .layout = .long };
    xfm.mtp_group_cost = .{};
    var model: LoadedModel = undefined;
    model.transformer = &xfm;
    var a: Slot = undefined;
    var b: Slot = undefined;
    a.model = &model;
    b.model = &model;
    const slots = [_]*Slot{ &a, &b };
    var geometry = group_cost_mod.GroupShape{ .n = 2, .kv_format = 8, .head_format = 8 };
    geometry.rows[0] = group_cost_mod.GroupShape.row(1, 1, 1, 2, 1, 3, 1);
    geometry.rows[1] = group_cost_mod.GroupShape.row(1, 1, 1, 5, 2, 2, 9);
    var prices = PlannerPrices{ .slots = &slots, .geometry = geometry };
    const speculative = prices.keyFor(&slots, &.{ 2, 4 });
    const swapped = prices.keyFor(&.{ &b, &a }, &.{ 4, 2 });
    try testing.expectEqual(speculative, swapped);
    for (0..3) |_| _ = xfm.mtp_group_cost.observeShape(speculative, 30, &.{ 2, 4 }, &.{ 30, 30 }, .round);
    try testing.expectEqual(@as(f32, 30), prices.price(&.{ 2, 4 }).?.ms);
    const plain_a = prices.keyFor(&.{&a}, &.{0});
    const spec_b = prices.keyFor(&.{&b}, &.{4});
    for (0..3) |_| {
        _ = xfm.mtp_group_cost.observeShape(plain_a, 10, &.{1}, &.{10}, .round);
        _ = xfm.mtp_group_cost.observeShape(spec_b, 25, &.{4}, &.{25}, .round);
    }
    try testing.expectEqual(@as(f32, 35), prices.price(&.{ 0, 4 }).?.ms);
    try testing.expect(prices.price(&.{ 4, 0 }) == null);
    const plain_b = prices.keyFor(&.{&b}, &.{0});
    try testing.expectEqual(@as(u8, 8), group_cost_mod.GroupShape.part(plain_b.rows[0], 48));
    try testing.expectEqual(@as(u8, 5), group_cost_mod.GroupShape.part(plain_b.rows[0], 24));
    try testing.expectEqual(@as(u8, 0), group_cost_mod.GroupShape.part(plain_b.rows[0], 40));
    try testing.expectEqual(@as(u64, 0), plain_b.head_format);
}

test "planner pools recurring head histories without mixing the cold first history" {
    var xfm: Trunk = undefined;
    xfm.round_cost = .{ .layout = .long };
    xfm.mtp_group_cost = .{};
    var model: LoadedModel = undefined;
    model.transformer = &xfm;
    var slots_storage: [4]Slot = undefined;
    var slots: [4]*Slot = undefined;
    for (&slots_storage, 0..) |*slot, i| {
        slot.model = &model;
        slots[i] = slot;
    }
    var geometry = group_cost_mod.GroupShape{ .n = 4, .kv_format = 8, .head_format = 8 };
    @memset(geometry.rows[0..4], group_cost_mod.GroupShape.row(2, 2, 2, 2, 1, 4, 1));
    var prices = PlannerPrices{ .slots = &slots, .geometry = geometry };

    var cold = geometry;
    cold.rows[0] = group_cost_mod.GroupShape.row(2, 2, 2, 2, 1, 1, 1);
    _ = xfm.mtp_group_cost.observeShape(cold, 180, &.{ 2, 2, 2, 2 }, &.{ 180, 180, 180, 180 }, .round);
    var h2 = geometry;
    h2.rows[0] = group_cost_mod.GroupShape.row(2, 2, 2, 2, 1, 2, 1);
    _ = xfm.mtp_group_cost.observeShape(h2, 74, &.{ 2, 3, 3, 2 }, &.{ 74, 74, 74, 74 }, .round);
    const probe_seed = prices.sample(&slots, &.{ 2, 2, 2, 2 }, false).?;
    try testing.expectEqual(@as(u32, 1), probe_seed.samples);
    try testing.expectEqual(@as(f32, 74), probe_seed.ms);
    var h3 = geometry;
    h3.rows[1] = group_cost_mod.GroupShape.row(2, 2, 2, 1, 1, 3, 1);
    _ = xfm.mtp_group_cost.observeShape(h3, 76, &.{ 3, 2, 3, 2 }, &.{ 76, 76, 76, 76 }, .round);
    var other_head_kv = geometry;
    other_head_kv.rows[1] = group_cost_mod.GroupShape.row(2, 2, 2, 2, 2, 3, 1);
    _ = xfm.mtp_group_cost.observeShape(other_head_kv, 20, &.{ 3, 3, 3, 3 }, &.{ 20, 20, 20, 20 }, .round);
    var other_mode = geometry;
    other_mode.rows[1] = group_cost_mod.GroupShape.row(2, 2, 2, 2, 1, 3, 9);
    _ = xfm.mtp_group_cost.observeShape(other_mode, 20, &.{ 3, 3, 3, 3 }, &.{ 20, 20, 20, 20 }, .round);
    try testing.expect(prices.price(&.{ 2, 2, 2, 2 }) == null);

    var h4 = geometry;
    h4.rows[2] = group_cost_mod.GroupShape.row(2, 2, 2, 2, 0, 2, 1);
    _ = xfm.mtp_group_cost.observeShape(h4, 78, &.{ 3, 3, 2, 2 }, &.{ 78, 78, 78, 78 }, .round);
    const pooled = prices.price(&.{ 2, 2, 2, 2 }).?;
    try testing.expectEqual(@as(u32, 3), pooled.samples);
    try testing.expect(pooled.ms >= 74 and pooled.ms <= 78);
    try testing.expect(pooled.variance > 0);
    try testing.expect(pooled.max_gap_ms >= 78);

    const plain = prices.keyFor(&slots, &.{ 0, 0, 0, 0 });
    for (0..3) |_| _ = xfm.mtp_group_cost.observeShape(plain, 37, &.{ 1, 1, 1, 1 }, &.{ 37, 37, 37, 37 }, .round);
    var histories: [4]Planner.History = @splat(.{});
    var rows: [4]Planner.Row = undefined;
    for (&histories, 0..) |*history, i| {
        for (0..3) |_| history.observe(2, 2);
        rows[i] = .{ .history = history };
    }
    const decision = Planner.choose(&rows, &prices);
    try testing.expectEqualSlices(u8, &.{ 2, 2, 2, 2 }, decision.widths[0..4]);
}

test "mean cost pooling retains differences between histories and both worst-case bounds" {
    var xfm: Trunk = undefined;
    xfm.round_cost = .{ .layout = .long };
    xfm.mtp_group_cost = .{};
    var model: LoadedModel = undefined;
    model.transformer = &xfm;
    var storage: [2]Slot = undefined;
    var slots: [2]*Slot = undefined;
    for (&storage, 0..) |*slot, i| {
        slot.model = &model;
        slots[i] = slot;
    }
    var geometry = group_cost_mod.GroupShape{ .n = 2 };
    @memset(geometry.rows[0..2], group_cost_mod.GroupShape.row(2, 2, 2, 4, 4, 4, 1));
    var prices = PlannerPrices{ .slots = &slots, .geometry = geometry };
    var a = geometry;
    var b = geometry;
    a.rows[0] = group_cost_mod.GroupShape.row(2, 2, 2, 4, 4, 2, 1);
    b.rows[0] = group_cost_mod.GroupShape.row(2, 2, 2, 4, 4, 3, 1);
    for (0..20) |_| {
        _ = xfm.mtp_group_cost.observeShape(a, 64, &.{ 3, 3 }, &.{ 64, 64 }, .round);
        _ = xfm.mtp_group_cost.observeShape(b, 80, &.{ 3, 3 }, &.{ 80, 80 }, .round);
    }
    const pooled = prices.price(&.{ 2, 2 }).?;
    try testing.expectApproxEqAbs(@as(f32, 88), pooled.mean_upper_ms.?, 0.001);
    xfm.round_cost = .{ .layout = .long };
    xfm.mtp_group_cost = .{};
    a = geometry;
    b = geometry;
    b.head_calls = 2;
    xfm.mtp_group_cost.shapes[0] = .{ .shape = a, .cell = .{ .ms = 60, .n = 100 }, .variance = 225, .mean_weight_sq = 1.0 / 19.0 };
    xfm.mtp_group_cost.shapes[1] = .{ .shape = b, .cell = .{ .ms = 75, .n = 3 }, .variance = 9, .mean_weight_sq = 1.0 / 3.0 };
    const exact = prices.price(&.{ 2, 2 }).?;
    try testing.expectApproxEqAbs(@as(f32, 90), exact.ms + 2 * @sqrt(exact.variance), 0.001);
    try testing.expectApproxEqAbs(@as(f32, 75) + 2 * @sqrt(@as(f32, 3)), exact.mean_upper_ms.?, 0.001);
    a = prices.keyFor(&slots, &.{ 0, 0 });
    b = a;
    b.head_calls = 2;
    xfm.mtp_group_cost.shapes[0] = .{ .shape = a, .cell = .{ .ms = 40, .n = 100 }, .variance = 25, .mean_weight_sq = 1.0 / 19.0 };
    xfm.mtp_group_cost.shapes[1] = .{ .shape = b, .cell = .{ .ms = 35, .n = 3 }, .variance = 1, .mean_weight_sq = 1.0 / 3.0 };
    const plain = prices.price(&.{ 0, 0 }).?;
    try testing.expectApproxEqAbs(@as(f32, 35) - 2 * @sqrt(@as(f32, 1.0 / 3.0)), plain.mean_lower_ms.?, 0.001);
}

test "cold grouped depths one and two use a bounded two-tick bootstrap" {
    var prices: PlannerPrices = undefined;
    prices.slots = &.{};
    const plain = Planner.Price{ .ms = 40, .variance = 4, .samples = 10 };
    try testing.expectEqual(@as(f32, 88), prices.probeFallbackEstimate(&.{ 1, 1, 1, 1 }, plain));
    try testing.expectEqual(@as(f32, 88), prices.probeFallbackEstimate(&.{ 2, 2, 2, 2 }, plain));
    try testing.expectEqual(@as(f32, 330), prices.probeFallbackEstimate(&.{ 4, 4, 4, 4 }, plain));
    const lower = Planner.Price{ .ms = 60, .variance = 100, .samples = 2, .max_gap_ms = 170 };
    try testing.expectEqual(@as(f32, 90), PlannerPrices.scaledProbeEstimate(lower, 2, 1));
}

test "transition pricing skips the first realized round after prime or width change" {
    try testing.expect(plannerPriceTransition(255, 1, false));
    try testing.expect(plannerPriceTransition(1, 1, true));
    try testing.expect(!plannerPriceTransition(1, 1, false));
    try testing.expect(plannerPriceTransition(1, 2, false));
    try testing.expect(!plannerPriceTransition(2, 0, true));
}

test "loop recovery: a loop inside an open think block closes the thought once" {
    const OPEN: u32 = 10;
    const CLOSE: u32 = 11;
    const forced = [_]u32{ 30, 31, CLOSE, 32 };
    var tb = mtp_plan.ThinkBound{ .budget = std.math.maxInt(u32), .opener_id = OPEN, .closer_id = CLOSE, .forced = &forced, .in_think = true };
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(testing.allocator);
    try ids.appendSlice(testing.allocator, &.{ 5, 6 });
    for (0..degenerate.degenerate_loop_min_span / 3 + 1) |_| try ids.appendSlice(testing.allocator, &.{ 101, 102, 103 });

    try testing.expectEqual(LoopAction.close_thought, loopRecoveryAction(&tb, ids.items, 400, 4096, false));
    // No room to close the thought: the ordinary cut.
    try testing.expectEqual(LoopAction.cut, loopRecoveryAction(&tb, ids.items, 4095, 4096, false));
    // Once recovered, a second loop is cut.
    try testing.expectEqual(LoopAction.cut, loopRecoveryAction(&tb, ids.items, 400, 4096, true));
}

test "loop recovery: an answer-side loop is still cut, a thought the prompt opened is recoverable" {
    const OPEN: u32 = 10;
    const CLOSE: u32 = 11;
    const forced = [_]u32{ 30, CLOSE, 32 };
    var tb = mtp_plan.ThinkBound{ .budget = std.math.maxInt(u32), .opener_id = OPEN, .closer_id = CLOSE, .forced = &forced, .in_think = true };
    const closed = [_]u32{ 5, 6, CLOSE, 7, 8 };
    try testing.expectEqual(LoopAction.cut, loopRecoveryAction(&tb, &closed, 5, 4096, false));

    var reopened = mtp_plan.ThinkBound{ .budget = std.math.maxInt(u32), .opener_id = OPEN, .closer_id = CLOSE, .forced = &forced, .in_think = false };
    const inside = [_]u32{ 7, OPEN, 5, 6 };
    try testing.expectEqual(LoopAction.close_thought, loopRecoveryAction(&reopened, &inside, 4, 4096, false));
}
