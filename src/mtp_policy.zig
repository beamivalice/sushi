const std = @import("std");
const chip_mod = @import("chip.zig");
const depth_bounds = @import("mtp_depth_bounds.zig");
const generate_mod = @import("generate.zig");
const group_cost = @import("mtp_group_cost.zig");
const group_planner = @import("mtp_group_planner.zig");
const io_util = @import("io_util.zig");
const log = @import("log.zig");
const mtp_mod = @import("mtp.zig");
const round_cost = @import("round_cost.zig");
const sampler = @import("sampler.zig");
const transformer_mod = @import("transformer.zig");
const Generator = generate_mod.Generator;
const MTP_ADAPTIVE_CONFIRM = Generator.MTP_ADAPTIVE_CONFIRM;
const MTP_ADAPTIVE_DEFAULT_CAP = Generator.MTP_ADAPTIVE_DEFAULT_CAP;
const MTP_ADAPTIVE_MARGIN = Generator.MTP_ADAPTIVE_MARGIN;
const MTP_ADAPTIVE_MIN_KV = Generator.MTP_ADAPTIVE_MIN_KV;
const MTP_ADAPTIVE_MIN_KV_MIMO = Generator.MTP_ADAPTIVE_MIN_KV_MIMO;
const MTP_ADAPTIVE_PROBE_TOKENS = Generator.MTP_ADAPTIVE_PROBE_TOKENS;
const MTP_ADAPTIVE_PROBE_WARM = Generator.MTP_ADAPTIVE_PROBE_WARM;
const MTP_DEMOTE_BELOW = Generator.MTP_DEMOTE_BELOW;
const MTP_DEPTH_SWITCH_WARMUP = Generator.MTP_DEPTH_SWITCH_WARMUP;
const MTP_DEPTH_WINDOW = Generator.MTP_DEPTH_WINDOW;
const MTP_DISABLE_BELOW = Generator.MTP_DISABLE_BELOW;
const MTP_EV_COST_BETA = Generator.MTP_EV_COST_BETA;
const MTP_EV_DEFAULT_COSTS = Generator.MTP_EV_DEFAULT_COSTS;
const MTP_EV_EMA_BETA = Generator.MTP_EV_EMA_BETA;
const MTP_EV_EXPLORE_MIN_R = Generator.MTP_EV_EXPLORE_MIN_R;
const MTP_EV_G17_NAX_QWEN4_MIXED_4_8_GS64_COSTS = Generator.MTP_EV_G17_NAX_QWEN4_MIXED_4_8_GS64_COSTS;
const MTP_EV_G17_NAX_QWEN4_Q4_GS64_COSTS = Generator.MTP_EV_G17_NAX_QWEN4_Q4_GS64_COSTS;
const MTP_EV_MIMO_EXL3_COSTS = Generator.MTP_EV_MIMO_EXL3_COSTS;
const MTP_EV_TAU_MAX = Generator.MTP_EV_TAU_MAX;
const MTP_EV_TAU_MIN = Generator.MTP_EV_TAU_MIN;
const MTP_EV_WARMUP_ROUNDS = Generator.MTP_EV_WARMUP_ROUNDS;
const MTP_EXT_DRY_COOLDOWN = Generator.MTP_EXT_DRY_COOLDOWN;
const MTP_EXT_DRY_MIN = Generator.MTP_EXT_DRY_MIN;
const MTP_EXT_DRY_ROUNDS = Generator.MTP_EXT_DRY_ROUNDS;
const MTP_EXT_SYNC_BUDGET = Generator.MTP_EXT_SYNC_BUDGET;
const MTP_PROMOTE_ABOVE = Generator.MTP_PROMOTE_ABOVE;
const MTP_PROMOTE_COOLDOWN = Generator.MTP_PROMOTE_COOLDOWN;
const MTP_REGIME_EXPLORE_BLOCK = Generator.MTP_REGIME_EXPLORE_BLOCK;
const MTP_REGIME_EXPLORE_DRAG = Generator.MTP_REGIME_EXPLORE_DRAG;
const MTP_REGIME_EXPLORE_PERIOD = Generator.MTP_REGIME_EXPLORE_PERIOD;
const MTP_REGIME_EXPLORE_PERIOD_MAX = Generator.MTP_REGIME_EXPLORE_PERIOD_MAX;
const MTP_REGIME_MARGIN = Generator.MTP_REGIME_MARGIN;
const MTP_REGIME_MIN_SAMPLES = Generator.MTP_REGIME_MIN_SAMPLES;
const MtpAdaptive = Generator.MtpAdaptive;
const MtpAdaptiveAction = Generator.MtpAdaptiveAction;
const MtpAdaptiveArm = Generator.MtpAdaptiveArm;
const MtpAdaptiveVote = Generator.MtpAdaptiveVote;
const MtpCostSource = Generator.MtpCostSource;
const MtpEvCosts = Generator.MtpEvCosts;
const MtpHeadRef = generate_mod.MtpHeadRef;
const MtpRegime = Generator.MtpRegime;
const MtpRoundPlan = Generator.MtpRoundPlan;
const MtpTrace = Generator.MtpTrace;
const MtpWidthTrial = Generator.MtpWidthTrial;
const mtpRoundOff0 = Generator.mtpRoundOff0;
const readEnvBool = generate_mod.readEnvBool;
const testing = std.testing;

/// Model-level twin of `mtpAdaptiveArchEligible`: only the in-checkpoint heads were
/// calibrated. `model_has_mtp` alone let every sidecar pack fold a cell nothing reads.
pub fn mtpAdaptiveModelEligible(model_has_mtp: bool, checkpoint_head: bool) bool {
    return model_has_mtp and checkpoint_head;
}

pub fn mtpAdaptiveModelOk(self: *const Generator) bool {
    return mtpAdaptiveModelEligible(self.model_has_mtp, self.xfm.qwen4_mtp != null or self.xfm.config.isMimo());
}

/// Will anyone read a serial cell for this model? Gated on the model, not the request: a
/// per-request `enable_mtp:false` on a calibrated checkpoint is the cleanest serial sample.
pub fn serialCellWanted(self: *const Generator) bool {
    if (!mtpAdaptiveSerialEnabled() or !mtpCostTableEnabled()) return false;
    if (!self.mtpAdaptiveModelOk()) return false;
    // Not `xfm.qwen4_mtp != null`: the weights load with the trunk even under `--no-mtp`.
    if (!self.model_has_mtp) return false;
    return mtpAdaptiveKvEligible(self.mtpKvLen(), mtpAdaptiveMinKvFor(self.xfm.config.isMimo()));
}

/// Drop the serial cell's pending interval: something ran between two decode ticks that
/// was not decode (a prefill chunk under `interleaveDecodeTick`).
pub fn invalidateSerialClock(self: *Generator) void {
    self.mtp_serial_clock = null;
    self.mtp_serial_warm = 0;
}

/// Same for the spec round's clock: the interval since the previous round end held a
/// prefill chunk. The next round times itself and is dropped as a transition.
pub fn invalidateRoundClock(self: *Generator) void {
    self.mtp_regime_clock = null;
    self.spec_round_prev_width = null;
}

/// Fold one plain serial decode token into the model's serial cell. The first
/// `MTP_ADAPTIVE_PROBE_WARM` ticks of a block are transitions; contention restarts the clock.
pub fn observeSerialTick(self: *Generator) void {
    if (!self.serialCellWanted()) {
        self.mtp_serial_clock = null;
        self.mtp_serial_warm = 0;
        return;
    }
    if (!self.spec_cost_solo) {
        self.mtp_serial_clock = null;
        self.mtp_serial_warm = 0;
        return;
    }
    const ms = self.mtpSerialWallMs() orelse return;
    const warm = self.mtp_serial_warm < MTP_ADAPTIVE_PROBE_WARM;
    if (warm) self.mtp_serial_warm += 1;
    _ = self.xfm.round_cost.observeSerial(self.mtpKvLen(), ms, true, warm);
}

/// Pure depth-policy step. `rate` is the windowed per-draft acceptance
/// probability. Returns the new depth; 0 means "disable speculation".
pub fn mtpNextDepth(current: u32, configured: u32, rate: f32) u32 {
    if (current > 1 and rate < MTP_DEMOTE_BELOW) return current - 1;
    if (current <= 1 and rate < MTP_DISABLE_BELOW) return 0;
    if (current < configured and rate > MTP_PROMOTE_ABOVE) return current + 1;
    return current;
}

/// Confidence-gated depth decision. Demoting is cheap (still
/// speculating) so it reacts on a small sample; DISABLING is sticky and
/// PROMOTING raises verify cost, so both require more evidence. A
/// 16-round window at a true 73% per-draft rate essentially never dips
/// below the 0.50 disable floor; a 5-round window does (observed live:
/// an early-story cold streak disabled a request that would have run
/// 1.11x at depth 1).
pub fn mtpDepthDecision(current: u32, configured: u32, rate: f32, window_rounds: u32, promote_blocked: bool) u32 {
    const next_depth = mtpNextDepth(current, configured, rate);
    if (next_depth == 0 and window_rounds < MTP_DEPTH_WINDOW) return current;
    if (next_depth > current and (window_rounds < 8 or promote_blocked)) return current;
    return next_depth;
}

/// Windowed adaptive-depth update, called once per nextMtp round with
/// that round's (drafted, accepted) counts.
pub fn updateMtpDepth(self: *Generator, drafted: u32, accepted: u32) void {
    const idx = self.mtp_window_idx % MTP_DEPTH_WINDOW;
    self.mtp_window_drafted[idx] = @intCast(drafted);
    self.mtp_window_accepted[idx] = @intCast(accepted);
    self.mtp_window_idx += 1;
    if (self.mtp_promote_cooldown > 0) self.mtp_promote_cooldown -= 1;
    if (self.mtp_rounds_since_switch < MTP_DEPTH_SWITCH_WARMUP) {
        self.mtp_rounds_since_switch += 1;
        return;
    }
    const n = @min(self.mtp_window_idx, MTP_DEPTH_WINDOW);
    var drafted_sum: u32 = 0;
    var accepted_sum: u32 = 0;
    var i: u32 = 0;
    while (i < n) : (i += 1) {
        drafted_sum += self.mtp_window_drafted[i];
        accepted_sum += self.mtp_window_accepted[i];
    }
    if (drafted_sum == 0) return;
    const rate = @as(f32, @floatFromInt(accepted_sum)) / @as(f32, @floatFromInt(drafted_sum));
    const decided = mtpDepthDecision(self.mtp_depth_current, self.mtp_depth, rate, n, self.mtp_promote_cooldown > 0);
    const next_depth = if (decided == 0) 0 else depth_bounds.clampWidth(decided, depth_bounds.floorFor(depth_bounds.active, self.mtp_depth), self.mtp_depth);
    if (next_depth == self.mtp_depth_current) return;
    if (next_depth == 0) {
        log.info(
            "  mtp=disabled (windowed per-draft rate {d:.2} < {d:.2} at depth 1)\n",
            .{ rate, MTP_DISABLE_BELOW },
        );
        self.spec_disabled_runtime = true;
        self.spec_disable_reason = .acceptance;
        return;
    }
    log.debug("  [mtp-depth] {d} -> {d} (windowed per-draft rate {d:.2})\n", .{ self.mtp_depth_current, next_depth, rate });
    if (next_depth < self.mtp_depth_current) self.mtp_promote_cooldown = MTP_PROMOTE_COOLDOWN;
    self.mtp_depth_current = next_depth;
    self.mtp_rounds_since_switch = 0;
    // Reset both windows so the new depth is judged on its own rounds.
    self.mtp_window_idx = 0;
    self.mtp_price.reset();
}

/// One-shot guard for the per-silicon cap log (the row is a property of
/// the machine, so it is worth saying once per process, not per request).
pub var mtp_depth_cap_logged: bool = false;

/// Minimum kv separation before two anchors can identify B. Below it the
/// difference is dominated by round-to-round noise, not by the KV read.
/// Round-end bookkeeping shared by the full- and partial-accept paths:
/// the kv term, the regime gate and the round-cost table all read ONE
/// inter-round wall clock (tok/s is measured between round ends, so
/// per-round work outside the round stopwatch belongs to the width).
pub fn mtpRoundEndObserve(self: *Generator, m: u32, tokens: u32, two_chunk: bool, m_lo: u32, width_trial: bool, round_ms: f32) void {
    if (group_planner.enabled() and self.mtp_planner_width != null) return;
    // An interval spanning a speculative round is neither arm's number.
    self.mtp_serial_clock = null;
    self.mtp_serial_warm = 0;
    const post_warmup = self.mtp_ev_rounds >= MTP_EV_WARMUP_ROUNDS;
    // The table wants rounds the EV controller PLANNED: the round that
    // ends warmup was still the legacy controller's (a w2 sample there
    // anchored the table at 2 on the M4 base 9B).
    const ev_planned = self.mtp_ev_rounds > MTP_EV_WARMUP_ROUNDS;
    const wall = if (post_warmup) self.mtpRegimeWallMs(round_ms) else round_ms;
    self.mtp_round_log.add(m, wall);
    const tok: f32 = @floatFromInt(tokens);
    if (post_warmup and self.spec_cost_solo and !width_trial) mtpRegimeObserve(&self.mtp_regime, two_chunk, m_lo, wall, tok);
    // A trial round was a single-chunk shape at another depth: not a
    // regime sample, but the shape its successor transitions from.
    if (width_trial) self.mtp_regime.last_two = false;
    // The table is the cost of the SINGLE-CHUNK shape at each width — the
    // quantity the m_lo loop compares. A two-chunk round chose its width
    // by its own confidence gate (tokens biased high) and paid a sync
    // (ms biased high); observed as width m_hi it read single-chunk 6
    // as better than the 5 -> 6 two-chunk it was measured FROM (M4 Max
    // 27B @16k, -4.5%). The shape question stays the regime gate's.
    const shape_changed = self.spec_round_prev_two_chunk != two_chunk or self.spec_round_prev_two_chunk2 != two_chunk;
    self.spec_round_prev_two_chunk2 = self.spec_round_prev_two_chunk;
    self.spec_round_prev_two_chunk = two_chunk;
    // The realized price of speculation for this request (every non-trial round, extension
    // included). Gated with its one consumer, `mtpAdaptiveVoteFor`.
    if (post_warmup and self.spec_cost_solo and self.mtpAdaptiveModelOk()) self.mtp_price.observe(wall, tok, width_trial);
    self.specObserveRound(m, wall, tok, ev_planned and !two_chunk, shape_changed);
}

/// Feed the model's round-cost table (`Transformer.round_cost`, shared
/// by every request and both block decoders). Width = drafts this round
/// (0 = serial). The previous width is tracked on EVERY round so the
/// first observed one is not a transition by accident; a rejected
/// sample that is not an expected drop (contended, transition) logs its
/// numbers — a silent reject is the probe's anti-pattern.
pub fn specObserveRound(self: *Generator, width: u32, wall_ms: f32, tokens: f32, observe: bool, shape_changed: bool) void {
    const transition = shape_changed or
        (if (self.spec_round_prev_width) |p| p != width else true) or
        (if (self.spec_round_prev_width2) |p| p != width else true);
    self.spec_round_prev_width2 = self.spec_round_prev_width;
    self.spec_round_prev_width = width;
    if (!observe) return;
    const kv = self.mtpKvLen();
    const v = self.xfm.round_cost.observe(width, kv, wall_ms, tokens, self.spec_cost_solo, transition);
    if (v == .bad_sample or v == .out_of_range) {
        log.warn("[spec-cost] table rejected sample ({s}): width={d} kv={d} ms={d:.2} tokens={d:.1}\n", .{ @tagName(v), width, kv, wall_ms, tokens });
    }
}

/// Write the model's round-cost table to disk when this request folded
/// new samples (request end; inference thread, so no lock).
pub fn persistRoundCost(self: *Generator) void {
    const t = &self.xfm.round_cost;
    const folded = round_cost.totalFolded(t);
    if (folded == t.stored_at) return;
    round_cost.storeCached(self.timer.io, self.xfm.round_cost_key_buf[0..self.xfm.round_cost_key_len], t);
    t.stored_at = folded;
}

/// Round-cost table kill switch — SUSHI_MTP_COST_TABLE=0 keeps the
/// table OBSERVING (its `[spec-stats]` fields stay comparable across an
/// A/B) but the plan reads only the fitted prior and no width trial runs.
pub var mtp_cost_table_cache: ?bool = null;

pub fn mtpCostTableEnabledFromEnv(raw: ?[]const u8) bool {
    const value = raw orelse return true;
    return value.len == 0 or value[0] != '0';
}

pub fn mtpCostTableEnabled() bool {
    if (mtp_cost_table_cache) |v| return v;
    const raw: ?[]const u8 = if (std.c.getenv("SUSHI_MTP_COST_TABLE")) |p| std.mem.span(p) else null;
    const on = mtpCostTableEnabledFromEnv(raw);
    mtp_cost_table_cache = on;
    return on;
}

/// Marginal round cost of draft position k (1-based), kv-blind. This is
/// the fitted SHAPE; `mtpEvMarginalCostAt` applies the kv term.
pub fn mtpEvMarginalCost(costs: MtpEvCosts, k: u32) f32 {
    const verify_cost = if (costs.nax_from != 0 and k >= costs.nax_from)
        costs.per_pos_nax
    else if (k <= costs.flat_max)
        costs.per_pos_lo
    else
        costs.per_pos_hi;
    return costs.draft + verify_cost;
}

/// Resolve the configured depth cap (`--mtp-max-depth`). 0 = auto (flag not passed):
/// MTP_ADAPTIVE_DEFAULT_CAP under the EV controller and DEFAULT_DEPTH in fixed mode.
/// Explicit values always win.
pub fn mtpDepthCapForProfile(configured: u32, adaptive: bool, profile: mtp_mod.MtpCostProfile) u32 {
    const chip = chip_mod.chipBrand();
    const cap = mtpDepthCapResolved(configured, adaptive, profile, chip);
    // Name the row ONCE when a per-silicon measurement is what fenced the
    // depth. Without it `[spec-stats] depth=4` on an M1 Pro reads the same
    // as the EV controller having chosen 4, or as `--mtp-max-depth 4` — and
    // the fence is exactly what someone debugging MTP there needs to see.
    if (configured == 0 and adaptive and !mtp_depth_cap_logged) {
        const row = mtp_mod.adaptiveDepthCapForMachine(chip, MTP_ADAPTIVE_DEFAULT_CAP);
        if (cap == row.cap and row.cap < MTP_ADAPTIVE_DEFAULT_CAP) {
            mtp_depth_cap_logged = true;
            log.info("[mtp] adaptive depth cap {d} ({s} row, default {d})\n", .{
                row.cap,
                row.label,
                MTP_ADAPTIVE_DEFAULT_CAP,
            });
        }
    }
    return depth_bounds.liftCap(cap, depth_bounds.active, mtp_mod.MAX_DEPTH);
}

/// The cap with no per-silicon row applied: explicit wins, else the
/// adaptive default (fixed mode keeps DEFAULT_DEPTH), never below `--mtp-min-depth`.
pub fn mtpDepthCapFree(configured: u32) u32 {
    if (configured != 0) return @min(mtp_mod.MAX_DEPTH, @max(1, configured));
    const free = if (mtpAdaptiveEnabled()) MTP_ADAPTIVE_DEFAULT_CAP else mtp_mod.DEFAULT_DEPTH;
    return depth_bounds.liftCap(free, depth_bounds.active, mtp_mod.MAX_DEPTH);
}

/// Same, with the chip string injected (tests, and the one live caller).
pub fn mtpDepthCapForProfileChip(configured: u32, adaptive: bool, profile: mtp_mod.MtpCostProfile, chip: []const u8) u32 {
    return mtpDepthCapResolved(configured, adaptive, profile, chip);
}

/// Same, with the machine's MEASURED verify ladder when one exists.
///
/// The per-silicon rows exist because the EV controller cannot see time:
/// it scores accepted-tokens-per-round, which on an M1 Pro IMPROVED
/// (3.65 -> 4.00) while realized tok/s fell 21%. A measured curve sees
/// exactly the cliff those rows encode, on whatever chip and whatever
/// quant geometry is actually resident — which the chip key cannot
/// express (the split-K lane is 4-bit/g64 only, so a 6-bit pack on the
/// same box is a different answer). The calibrated NAX profiles keep
/// their own cap until a probe on that silicon is validated against it.
pub fn mtpDepthCapResolved(
    configured: u32,
    adaptive: bool,
    profile: mtp_mod.MtpCostProfile,
    chip: []const u8,
) u32 {
    if (configured != 0) return @min(mtp_mod.MAX_DEPTH, @max(1, configured));
    if (!adaptive) return mtp_mod.DEFAULT_DEPTH;
    return switch (profile) {
        // The per-silicon row is the COLD-START cap: the measured
        // round-cost table may plan above it on trusted widths.
        .generic => mtp_mod.adaptiveDepthCapForMachine(chip, MTP_ADAPTIVE_DEFAULT_CAP).cap,
        // qwen4's measured surface has no NAX region to reach: depths 7-8
        // price at .345/pos against sub-60% tail acceptance even on
        // saturated echo, so the calibrated cap keeps the default.
        .g17_nax_qwen4_q4_gs64, .g17_nax_qwen4_mixed_4_8_gs64 => MTP_ADAPTIVE_DEFAULT_CAP,
        // Three trained heads; the depth cap clamps to them at load.
        .mimo_exl3 => MTP_ADAPTIVE_DEFAULT_CAP,
    };
}

pub fn resolveMtpDepthCapForProfile(configured: u32, profile: mtp_mod.MtpCostProfile) u32 {
    return mtpDepthCapForProfile(configured, mtpAdaptiveEnabled(), profile);
}

/// Expected committed tokens for an m-deep round: the always-committed t1
/// plus the acceptance chain sum (draft k lands iff drafts 0..k all land).
pub fn mtpEvExpectedTokens(a: []const f32, m: u32) f32 {
    var chain: f32 = 1.0;
    var tok: f32 = 1.0;
    var k: u32 = 0;
    while (k < m and k < a.len) : (k += 1) {
        chain *= a[k];
        tok += chain;
    }
    return tok;
}

pub fn mtpEvMarginalCostAt(costs: MtpEvCosts, k: u32, kv_len: u32) f32 {
    _ = kv_len;
    return mtpEvMarginalCost(costs, k);
}

/// Round cost in verify-base units (piecewise per-position marginals).
pub fn mtpEvRoundCost(costs: MtpEvCosts, m: u32, with_sync: bool) f32 {
    return mtpEvRoundCostAt(costs, m, with_sync, 0);
}

pub fn mtpEvRoundCostAt(costs: MtpEvCosts, m: u32, with_sync: bool, kv_len: u32) f32 {
    var c: f32 = 1.0 + (if (with_sync) costs.sync else 0.0);
    var k: u32 = 1;
    while (k <= m) : (k += 1) c += mtpEvMarginalCostAt(costs, k, kv_len);
    return c;
}

/// Pure EV plan: pick (m_lo, m_hi, tau) maximizing expected tok/round-cost.
/// `m_lo_max` damps the base-depth climb (hysteresis — the caller passes
/// last round's m_lo + 1); demotions are never damped.
///  1. m_lo = argmax over single-chunk depths of E(m)/T(m).
///  2. m_hi = deepest position whose marginal chain still pays under FULL
///     confidence in chunk A (the best case the gate can certify).
///  3. tau: extend when the confidence-implied chain beats the stop rate
///     on the margin — c*S/dt > r  =>  tau = r*dt/S.
/// There is deliberately NO separate "is the sync worth it" gate: tau
/// already keeps low-confidence rounds single-chunk, the horizon check
/// collapses m_hi on cold EMAs (killing the sync entirely), and a
/// prior-weighted expected-gain gate measurably starves exploration —
/// deep indices are only observed when extension fires, so a gate fed by
/// their priors blocks the first trial forever (live: ext_rounds=0 on
/// pure echo).
pub fn mtpEvPlanFor(a: []const f32, cap_in: u32, costs: MtpEvCosts, m_lo_max: u32) MtpRoundPlan {
    return mtpEvPlanForAt(a, cap_in, costs, m_lo_max, 0);
}

pub fn mtpEvPlanForAt(a: []const f32, cap_in: u32, costs: MtpEvCosts, m_lo_max: u32, kv_len: u32) MtpRoundPlan {
    return mtpEvPlanSrc(a, cap_in, MtpCostSource.init(costs, kv_len, null), m_lo_max);
}

pub fn mtpEvPlanSrc(a: []const f32, cap_in: u32, src: MtpCostSource, m_lo_max: u32) MtpRoundPlan {
    const cap: u32 = @intCast(@min(@as(usize, @max(1, cap_in)), a.len));
    const lo_cap: u32 = @min(cap, @max(1, m_lo_max));
    var m_lo: u32 = 1;
    var best_r: f32 = 0.0;
    // With measured costs the standing base (m_lo_max - 1, what the
    // caller planned last round) keeps its place unless a challenger
    // beats it by the switch margin: cells are EMAs of a few samples
    // and a 5% reversal between neighbours is noise, while every
    // change of base is a transition round the table will not count
    // and the model pays for.
    const standing: u32 = if (src.fromTable()) @min(lo_cap, m_lo_max -| 1) else 0;
    var standing_r: f32 = 0.0;
    var m: u32 = 1;
    while (m <= lo_cap) : (m += 1) {
        const tok = src.measuredTokens(m) orelse mtpEvExpectedTokens(a, m);
        const r = tok / src.roundCost(m, false);
        if (m == standing) standing_r = r;
        if (r > best_r) {
            best_r = r;
            m_lo = m;
        }
    }
    if (standing >= 1 and m_lo != standing and best_r <= standing_r * (1.0 + round_cost.SWITCH_MARGIN)) {
        m_lo = standing;
        best_r = standing_r;
    }
    if (m_lo >= cap) return .{ .m_lo = m_lo, .m_hi = m_lo, .tau_ln = 0.0 };
    var m_hi: u32 = m_lo;
    var cond: f32 = 1.0;
    var s_sum: f32 = 0.0; // expected extension tokens, conditional on chunk A
    var t_sum: f32 = 0.0; // extension marginal cost (piecewise)
    while (m_hi < cap) {
        cond *= a[m_hi];
        const mc = src.marginal(m_hi + 1);
        if (cond <= best_r * mc) break;
        s_sum += cond;
        t_sum += mc;
        m_hi += 1;
    }
    if (m_hi == m_lo) {
        // Exploration valve for the horizon itself (2026-07-22): the
        // index one past m_lo is observable ONLY when extension fires —
        // at m_lo-deep rounds a[m_lo] never updates — so an EMA dragged
        // cold under an earlier workload (or an easier tau regime)
        // would otherwise close the horizon FOREVER: m stays at m_lo,
        // a[m_lo] never re-proves itself, and pure echo runs at
        // ext_rounds=0 (the exact class the TAU_MAX valve exists for,
        // resurfaced by refit #3's honest marginals). Whenever the base
        // itself pays, keep ONE extension position reachable at the
        // clamped tau; the dry-spell gate bounds the consideration tax
        // on workloads where confidence never clears it.
        if (best_r <= MTP_EV_EXPLORE_MIN_R) return .{ .m_lo = m_lo, .m_hi = m_lo, .tau_ln = 0.0 };
        const mc = src.marginal(m_lo + 1);
        // A MEASURED marginal the position cannot repay even at full
        // confidence (1 <= best_r * mc) closes the valve: the valve
        // exists to observe a[m_lo], and with the table that is the
        // width trial's job. The fitted prior keeps the valve exactly
        // as Phase 1 measured it.
        if (src.fromTable() and best_r * mc >= 1.0) return .{ .m_lo = m_lo, .m_hi = m_lo, .tau_ln = 0.0 };
        const s = @max(a[m_lo], 1e-6);
        const tau_x = std.math.clamp(best_r * mc / s, MTP_EV_TAU_MIN, MTP_EV_TAU_MAX);
        return .{ .m_lo = m_lo, .m_hi = m_lo + 1, .tau_ln = @log(tau_x) };
    }
    // The TAU_MAX clamp doubles as the exploration valve: on razor-thin
    // horizons the honest tau approaches 1 ("never extend"), and 0.95
    // lets near-perfect-confidence rounds through so the deep EMAs can
    // observe reality at all.
    const tau = std.math.clamp(best_r * t_sum / s_sum, MTP_EV_TAU_MIN, MTP_EV_TAU_MAX);
    return .{ .m_lo = m_lo, .m_hi = m_hi, .tau_ln = @log(tau) };
}

/// Update the conditional acceptance EMAs from one realized round.
/// Acceptance is prefix-structured: indices < accepted saw a success, the
/// index AT `accepted` saw the reject (when one happened), and deeper
/// indices were never conditionally reached — no observation.
pub fn mtpEvObserve(a: []f32, drafted: u32, accepted: u32, beta: f32) void {
    var i: usize = 0;
    while (i < accepted and i < a.len) : (i += 1) a[i] += beta * (1.0 - a[i]);
    if (accepted < drafted and accepted < a.len) a[accepted] += beta * (0.0 - a[accepted]);
}

/// Chain log-confidence of a drafted chunk: sum of per-draft log p_head,
/// clamped to <= 0 per term (a log-prob is never positive; bf16 noise
/// can be). NaN poisons to -inf so a broken confidence can never extend.
pub fn mtpChainLogConf(confs: []const f32) f32 {
    var sum: f32 = 0.0;
    for (confs) |c| {
        if (std.math.isNan(c)) return -std.math.inf(f32);
        sum += @min(0.0, c);
    }
    return sum;
}

/// Adaptive (EV) controller gate — DEFAULT ON. SUSHI_MTP_ADAPTIVE=0
/// reverts to the fixed-depth windowed controller for same-boot A/Bs.
pub var mtp_adaptive_cache: ?bool = null;

pub fn mtpAdaptiveEnabled() bool {
    if (mtp_adaptive_cache) |v| return v;
    var on = true;
    if (std.c.getenv("SUSHI_MTP_ADAPTIVE")) |p| {
        const val = std.mem.span(p);
        if (val.len > 0 and val[0] == '0') on = false;
    }
    mtp_adaptive_cache = on;
    return on;
}

pub fn mtpTraceOn() bool {
    return mtpTraceEnabled();
}

pub fn mtpTraceSub(self: *Generator, sub: MtpTrace.Sub, dur_ns: u64) void {
    if (!mtpTraceEnabled()) return;
    self.mtp_trace.addSub(sub, dur_ns);
}

pub var mtp_trace_cache: ?bool = null;

pub fn mtpTraceEnabled() bool {
    if (mtp_trace_cache) |v| return v;
    const on = readEnvBool("SUSHI_MTP_TRACE");
    mtp_trace_cache = on;
    return on;
}

pub fn mtpEvCostsForProfile(profile: mtp_mod.MtpCostProfile) MtpEvCosts {
    return switch (profile) {
        .generic => MTP_EV_DEFAULT_COSTS,
        .g17_nax_qwen4_q4_gs64 => MTP_EV_G17_NAX_QWEN4_Q4_GS64_COSTS,
        .g17_nax_qwen4_mixed_4_8_gs64 => MTP_EV_G17_NAX_QWEN4_MIXED_4_8_GS64_COSTS,
        .mimo_exl3 => MTP_EV_MIMO_EXL3_COSTS,
    };
}

/// Cost-aware dry-explore threshold (consecutive dry considered rounds
/// tolerated before a cooldown), derived from the LIVE-measured sync
/// fraction `sync_ms / round_ms`. Bounded to
/// [MTP_EXT_DRY_MIN, MTP_EXT_DRY_ROUNDS] so a fresh trial always fits
/// inside an echo stretch. Unmeasured (either EMA still 0) keeps the fixed
/// budget — no behavior change until the live cost is observed.
pub fn mtpExtDryThresholdFor(sync_ms: f32, round_ms: f32) u32 {
    if (sync_ms <= 0.0 or round_ms <= 0.0) return MTP_EXT_DRY_ROUNDS;
    const frac = sync_ms / round_ms; // measured exploration cost fraction
    if (frac <= 0.0) return MTP_EXT_DRY_ROUNDS;
    const max_f: f32 = @floatFromInt(MTP_EXT_DRY_ROUNDS);
    // Tolerate ~SYNC_BUDGET round-times of accumulated sync per dry burst:
    // a costlier sync (larger frac) yields fewer tolerated dry rounds.
    const raw = @min(MTP_EXT_SYNC_BUDGET / frac, max_f);
    const rounded: u32 = @intFromFloat(@round(@max(0.0, raw)));
    return @min(MTP_EXT_DRY_ROUNDS, @max(MTP_EXT_DRY_MIN, rounded));
}

/// Pure dry-spell policy step, called once per post-warmup EV round
/// whose plan considers extension. Returns whether the two-chunk shape
/// is allowed this round; mutates the streak/cooldown state. The caller
/// bumps `streak` on a considered-but-not-extended round and zeroes it
/// when extension fires. `threshold` is the cost-aware dry limit
/// (mtpExtDryThresholdFor) — the live-cost lever that shortens dry
/// exploration bursts when the measured sync is expensive.
pub fn mtpExtDryAllows(dry_streak: *u32, cooldown: *u32, threshold: u32) bool {
    if (cooldown.* > 0) {
        cooldown.* -= 1;
        return false;
    }
    if (dry_streak.* >= threshold) {
        cooldown.* = MTP_EXT_DRY_COOLDOWN - 1;
        dry_streak.* = 0;
        return false;
    }
    return true;
}

/// Fold a nanosecond wall-time sample into a millisecond EMA. Seeds on the
/// first sample (prev <= 0). Pure — the live-cost measurement plumbing is
/// just Stopwatch reads feeding this.
pub fn mtpEmaMs(prev_ms: f32, sample_ns: u64) f32 {
    const sample_ms = @as(f32, @floatFromInt(sample_ns)) / 1.0e6;
    if (prev_ms <= 0.0) return sample_ms;
    return prev_ms + MTP_EV_COST_BETA * (sample_ms - prev_ms);
}

/// The running mean over a shape's first 1/beta samples, the cost EMA after, so no single
/// round (an all-rejected one emits 1 token) outweighs the rest.
pub fn regimeFold(prev: f32, sample: f32, n: u32) f32 {
    const w = @max(MTP_EV_COST_BETA, 1.0 / @as(f32, @floatFromInt(n)));
    return prev + w * (sample - prev);
}

/// Wall time since the previous round ended (the first round reads its
/// own stopwatch), so inter-round work is charged to the shape.
pub fn mtpRegimeWallMs(self: *Generator, round_ms: f32) f32 {
    if (self.mtp_regime_clock) |*c| {
        const ns = c.read();
        c.reset();
        return @as(f32, @floatFromInt(ns)) / @as(f32, std.time.ns_per_ms);
    }
    self.mtp_regime_clock = io_util.Stopwatch.init(self.timer.io);
    return round_ms;
}

/// Both shapes are compared at the SAME base depth: a shape observed at
/// a new m_lo reseeds (the warmup climb's depth-1..3 rounds read 19
/// ms/tok against 11 for a two-chunk round at m_lo 4 — not a comparison).
pub fn mtpRegimeObserve(r: *MtpRegime, two_chunk: bool, m_lo: u32, round_ms: f32, tokens: f32) void {
    if (round_ms <= 0.0 or tokens <= 0.0) return;
    // The first observed round has no predecessor to transition from
    // (dropping it cost one extra pre-verdict extension round, the whole
    // of the M1 Pro 27B's -2.6% on v5.2).
    const transition = r.last_two != null and r.last_two.? != two_chunk;
    r.last_two = two_chunk;
    if (transition) return;
    if (two_chunk) {
        if (r.two_m != m_lo) {
            r.two_n = 0;
            r.two_m = m_lo;
        }
        r.two_n += 1;
        r.two_ms = regimeFold(r.two_ms, round_ms, r.two_n);
        r.two_tok = regimeFold(r.two_tok, tokens, r.two_n);
    } else {
        if (r.one_m != m_lo) {
            r.one_n = 0;
            r.one_m = m_lo;
        }
        r.one_n += 1;
        r.one_ms = regimeFold(r.one_ms, round_ms, r.one_n);
        r.one_tok = regimeFold(r.one_tok, tokens, r.one_n);
    }
}

pub fn mtpRegimeMeasured(n: u32) bool {
    return n >= MTP_REGIME_MIN_SAMPLES;
}

/// Null until BOTH shapes have been measured at the same base depth.
pub fn mtpRegimeTwoChunkWorse(r: MtpRegime) ?bool {
    return mtpRegimeVerdict(r, null);
}

/// Verdict with HYSTERESIS: a standing "worse" only flips to "better"
/// once two-chunk is at or below single, not merely inside the margin.
/// The majority shape's first observed round after a trial block is
/// still elevated (M1 Pro 9B: single 23.8 steady, 24.6-24.9 after a
/// block), which pulled the ratio inside the margin and flipped the
/// verdict 5-7 times per boot on v5.2, each flip a run of two-chunk
/// rounds on a box where that shape loses 10%.
pub fn mtpRegimeVerdict(r: MtpRegime, prev: ?bool) ?bool {
    if (!mtpRegimeMeasured(r.two_n) or !mtpRegimeMeasured(r.one_n) or r.two_m != r.one_m) return null;
    const ratio = (r.two_ms / r.two_tok) / (r.one_ms / r.one_tok);
    if (prev == true) return ratio > 1.0;
    return ratio > 1.0 + MTP_REGIME_MARGIN;
}

/// Rounds between trials of the worse shape, from the measured gap.
pub fn mtpRegimeExplorePeriod(r: MtpRegime) u32 {
    if (r.two_tok <= 0.0 or r.one_tok <= 0.0) return MTP_REGIME_EXPLORE_PERIOD;
    const two = r.two_ms / r.two_tok;
    const one = r.one_ms / r.one_tok;
    const gap = @abs(two - one) / @min(two, one);
    const block: f32 = @floatFromInt(MTP_REGIME_EXPLORE_BLOCK);
    const p: u32 = @intFromFloat(@ceil(block * gap / MTP_REGIME_EXPLORE_DRAG));
    return @min(MTP_REGIME_EXPLORE_PERIOD_MAX, @max(MTP_REGIME_EXPLORE_PERIOD, p));
}

/// Which shape this round runs: null = as the plan wrote it (two-chunk),
/// false = single-chunk. The plan's own default is two-chunk, so that
/// side measures itself; an unmeasured SINGLE is tried at once (v2
/// compared one_m to a two_m only a two-chunk round can set, and pinned
/// single forever). Once both are measured the worse shape runs as a
/// trial BLOCK every period so its EMA keeps tracking the workload; a
/// single-chunk trial is what gets it measured at all when the horizon
/// opens on every round (pure echo never plans one by itself).
pub fn mtpRegimeForce(r: *MtpRegime, round_idx: u32) ?bool {
    if (r.last_idx == round_idx) return r.last_force;
    const force = mtpRegimeForceAt(r, round_idx);
    r.last_idx = round_idx;
    r.last_force = force;
    return force;
}

pub fn mtpRegimeForceAt(r: *MtpRegime, round_idx: u32) ?bool {
    if (!mtpRegimeMeasured(r.two_n)) return null;
    if (!mtpRegimeMeasured(r.one_n) or r.one_m != r.two_m) return false;
    const worse = mtpRegimeVerdict(r.*, r.sched_verdict) orelse return null;
    if (r.sched_verdict != worse) {
        if (r.sched_verdict == null) r.verdict_round = round_idx;
        r.sched_verdict = worse;
        r.trial_end = 0;
        r.next_trial = round_idx + mtpRegimeExplorePeriod(r.*);
    }
    const minority: ?bool = if (worse) null else false;
    const majority: ?bool = if (worse) false else null;
    if (round_idx < r.trial_end) return minority;
    if (round_idx >= r.next_trial) {
        r.trials += 1;
        r.trial_end = round_idx + MTP_REGIME_EXPLORE_BLOCK;
        r.next_trial = r.trial_end + mtpRegimeExplorePeriod(r.*);
        return minority;
    }
    return majority;
}

/// Per-round draft plan. Fixed mode (and EV warmup): today's adaptive
/// depth, no extension — byte-identical round shape to the legacy path.
/// Post-warmup EV mode: the pure plan over the acceptance EMAs, with the
/// base-depth climb damped to one step per round and two-chunk plans
/// gated by the extension dry-spell policy.
/// `--mtp-min-depth n --mtp-max-depth n`: every round drafts exactly n, the
/// EV/windowed controllers never demote or disable — the per-index
/// acceptance meter (`acc_idx=` on the trace line) and the byte bar's mode.
pub fn mtpPinnedDepth() ?u32 {
    return depth_bounds.active.pinned();
}

/// A plan held inside the depth range: the EV argmax may sit outside it.
pub fn mtpPlanInBounds(plan: MtpRoundPlan, lo: u32, hi: u32) MtpRoundPlan {
    var p = plan;
    p.m_lo = depth_bounds.clampWidth(p.m_lo, lo, hi);
    p.m_hi = depth_bounds.clampWidth(p.m_hi, p.m_lo, hi);
    if (p.m_hi == p.m_lo and plan.m_hi != plan.m_lo) p.tau_ln = 0.0;
    return p;
}

/// Drafts per round whose verify rows stay byte-identical to serial decode.
/// MiMo's verify is decode-shaped row by row and the FP8 trunk GEMV keeps a
/// decode row's arithmetic only to `MIMO_VERIFY_ROWS_MAX` rows, so a MiMo
/// round never plans past the budget; another arch's rows are row-identical
/// at any width up to `MAX_DEPTH`.
pub fn mtpVerifyDraftsMax(is_mimo: bool) u32 {
    return if (is_mimo) @intCast(transformer_mod.MIMO_VERIFY_ROWS_MAX - 1) else mtp_mod.MAX_DEPTH;
}

/// The width a pinned round drafts: never past the verify row budget.
pub fn mtpForcedWidth(forced: u32, verify_max: u32) u32 {
    return @min(forced, verify_max);
}

/// The switch's floor for this arch.
pub fn mtpAdaptiveMinKvFor(is_mimo: bool) u32 {
    return if (is_mimo) MTP_ADAPTIVE_MIN_KV_MIMO else MTP_ADAPTIVE_MIN_KV;
}

/// May the adaptive switch run at this context at all? Read before the vote and the probe.
pub fn mtpAdaptiveKvEligible(kv_len: u32, min_kv: u32) bool {
    return kv_len >= min_kv;
}

/// The bucket the adaptive switch speaks in. Every site resolves it the same way (the
/// decision and the re-entry once disagreed and oscillated); the fallback is the table's
/// own grid, never the free `bucketFor`.
pub fn mtpAdaptiveBucketOf(read: ?usize, kv_len: u32, layout: round_cost.Layout) usize {
    return read orelse round_cost.bucketForLayout(kv_len, layout);
}

/// One serial tick of the adaptive arm at `kv_len`, buckets resolved as the decision resolved them.
pub fn mtpAdaptiveSerialTickFor(a: *MtpAdaptive, t: *const round_cost.Table, kv_len: u32) MtpAdaptiveAction {
    return a.serialTickAt(mtpAdaptiveBucketOf(t.bucketToRead(kv_len), kv_len, t.layout), t.bucketOf(kv_len));
}

pub fn mtpAdaptiveBucket(self: *const Generator, kv_len: u32) usize {
    const t = &self.xfm.round_cost;
    return mtpAdaptiveBucketOf(t.bucketToRead(kv_len), kv_len, t.layout);
}

/// Must the realized price window be dropped? It prices one arm in one bucket.
pub fn mtpAdaptiveRegimeMoved(
    prev_bucket: ?usize,
    prev_arm: MtpAdaptiveArm,
    bucket: ?usize,
    arm: MtpAdaptiveArm,
) bool {
    if (prev_arm != arm) return true;
    if (prev_bucket == null and bucket == null) return false;
    if (prev_bucket == null or bucket == null) return true;
    return prev_bucket.? != bucket.?;
}

/// Apply the rule above around a `MtpAdaptive` transition (both call sites).
pub fn mtpAdaptiveSyncWindow(self: *Generator, prev_bucket: ?usize, prev_arm: MtpAdaptiveArm) void {
    if (mtpAdaptiveRegimeMoved(prev_bucket, prev_arm, self.mtp_adaptive.bucket, self.mtp_adaptive.arm)) {
        self.mtp_price.reset();
    }
}

/// One round's vote. Both prices must be measured: the table's cell (cross-request, but
/// its `tok` column is a workload mixture) and this request's window (per-request, empty
/// until full). A modeled denominator under-predicted committed tokens by 12-31% and
/// switched where MTP was ~10% faster. `serial` needs BOTH; anything missing = `.undecided`.
pub fn mtpAdaptiveVoteFor(
    table_ms_tok: ?f32,
    window_ms_tok: ?f32,
    serial_ms_tok: ?f32,
    margin: f32,
) MtpAdaptiveVote {
    const tbl = table_ms_tok orelse return .undecided;
    const win = window_ms_tok orelse return .undecided;
    const ser = serial_ms_tok orelse return .undecided;
    if (!std.math.isFinite(tbl) or !(tbl > 0)) return .undecided;
    if (!std.math.isFinite(win) or !(win > 0)) return .undecided;
    if (!std.math.isFinite(ser) or !(ser > 0)) return .undecided;
    if (!std.math.isFinite(margin) or margin < 0) return .undecided;
    const bar = ser * (1.0 + margin);
    return if (tbl > bar and win > bar) .serial else .mtp;
}

/// Arm a bounded serial probe, returning the bucket it will measure. One bucket parameter
/// (the flag and the "already measured?" check once keyed on different buckets). The
/// sample lands in the bucket the request is IN.
/// A probe buys the LAST missing input: a request that cannot fill its price window can
/// never use what the probe teaches it and just pays 8 serial tokens.
pub fn mtpSerialProbeUseful(window_ms_tok: ?f32) bool {
    return window_ms_tok != null;
}

pub fn mtpSerialProbeArm(
    t: *round_cost.Table,
    bucket: usize,
    solo: bool,
    idle: bool,
    may_resume: bool,
    useful: bool,
) ?usize {
    if (!solo or !idle or !may_resume or !useful) return null;
    if (bucket >= round_cost.N_BUCKETS) return null;
    if (t.serialMsPerTok(bucket) != null) return null;
    // Bounded retries, not a one-shot flag: an interrupted probe must not burn the bucket's only chance.
    if (t.serial_probes[bucket] >= round_cost.MAX_SERIAL_PROBES) return null;
    t.serial_probes[bucket] += 1;
    return bucket;
}

/// Whole-mechanism kill switch (`SUSHI_MTP_ADAPTIVE_SERIAL=0`): no vote, no probe, no
/// serial fold. Independent of `SUSHI_MTP_ADAPTIVE` (the depth controller's lever).
pub var mtp_adaptive_serial_cache: ?bool = null;

/// Default on; only an exact "0" turns it off.
pub fn mtpAdaptiveSerialEnabledFromEnv(raw: ?[]const u8) bool {
    const value = raw orelse return true;
    return !std.mem.eql(u8, value, "0");
}

pub fn mtpAdaptiveSerialEnabled() bool {
    if (mtp_adaptive_serial_cache) |v| return v;
    const raw: ?[]const u8 = if (std.c.getenv("SUSHI_MTP_ADAPTIVE_SERIAL")) |p| std.mem.span(p) else null;
    const on = mtpAdaptiveSerialEnabledFromEnv(raw);
    mtp_adaptive_serial_cache = on;
    return on;
}

/// Drift between the head's position bookkeeping and what its next forward demands
/// (`pos_offset == pos_base + seq_offset`; `seq_offset == 0` re-seeds). 0 = in sync.
pub fn mtpHeadPositionDrift(pos_base: i64, seq_offset: usize, next_pos_offset: i64) i64 {
    if (seq_offset == 0) return 0;
    return next_pos_offset - (pos_base + @as(i64, @intCast(seq_offset)));
}

/// The drift the next round would hit. A serial block advances the trunk while the head's
/// `seq_offset` stays put. Null = no head state.
pub fn mtpHeadDriftNow(self: *const Generator) ?i64 {
    if (self.mtp_cache == null) return null;
    const mc = &self.mtp_cache.?;
    return switch (mc.*) {
        .mimo => 0,
        .qwen4 => |r| blk: {
            r.t.qwen4MtpActivate(r.st);
            const m = &(r.t.qwen4_mtp orelse break :blk 0);
            const off0 = mtpRoundOff0(self.mtp_hist_stash, mc.step());
            break :blk mtpHeadPositionDrift(@intCast(m.pos_base), m.seq_offset, @intCast(off0 + 1));
        },
    };
}

/// May this request resume MTP after a serial block? An M-RoPE turn ropes the head at an
/// absolute position where the content gap is a wrong answer, so vision turns never come back.
pub fn mtpAdaptiveHeadMayResume(self: *const Generator) bool {
    // The one predicate all three round-start doors ask (re-entry, `apply_stash`, the probe).
    if (self.mtp_adaptive.sticky_serial) return false;
    return self.ctx.mrope_pos == null;
}

/// Does a `to_serial` on this head make the arm sticky? Only a module-owned head.
pub fn stickyOnSerialSwitch(head: ?MtpHeadRef) bool {
    const h = head orelse return false;
    return h.moduleOwned();
}

/// Does this tick newly release the module head? True exactly once per request.
pub fn stickyReleaseNow(sticky: bool, already_released: bool) bool {
    return sticky and !already_released;
}

/// Has this request released the module-owned MTP head?
pub fn mtpModuleHeadReleased(self: *const Generator) bool {
    return self.mtp_head_released;
}

/// Armed but not landed. The release lives in `nextMtp`'s serial branch and the batched
/// tick never calls `nextMtp`, so `scheduler.batchable` keeps the slot out of a group for the
/// one tick the release needs.
pub fn mtpReleasePending(self: *const Generator) bool {
    return self.mtp_adaptive.sticky_serial and !self.mtp_head_released;
}

/// Release the module-owned head at a serial block boundary, after `mtpDetachHead`; never
/// where the switch is decided (that round still runs).
pub fn mtpMaybeReleaseModuleHead(self: *Generator) void {
    if (!stickyReleaseNow(self.mtp_adaptive.sticky_serial, self.mtp_head_released)) return;
    self.mtp_head_released = true;
    log.info(
        "  [mtp] sticky-serial: module head released for the rest of the request (kv {d}) — other slots may claim it\n",
        .{self.mtpKvLen()},
    );
}

/// The heads the adaptive serial switch was calibrated for: qwen4_exp's and MiMo's; DeepSeek's
/// DSpark has a different verify surface. Not `moduleOwned()`: different question.
pub fn mtpAdaptiveHeadEligible(tag: std.meta.Tag(MtpHeadRef)) bool {
    return tag == .qwen4 or tag == .mimo;
}

pub fn mtpAdaptiveArchEligible(self: *const Generator) bool {
    const head = self.mtp orelse return false;
    return mtpAdaptiveHeadEligible(std.meta.activeTag(head));
}

pub fn mtpAdaptiveSerialStep(self: *Generator, m_lo: u32, kv_len: u32) bool {
    // Read after the plan (m_lo is the width it prices) and before the width trial.
    if (mtpAdaptiveSerialEnabled() and mtpCostTableEnabled() and
        self.mtpAdaptiveArchEligible() and
        mtpAdaptiveKvEligible(kv_len, mtpAdaptiveMinKvFor(self.xfm.config.isMimo())))
    {
        const t = &self.xfm.round_cost;
        const b = self.mtpAdaptiveBucket(kv_len);
        const table_ms_tok = t.msPerTok(m_lo, b);
        const window_ms_tok = self.mtp_price.msPerTok();
        const serial_ms = t.serialMsPerTok(b);
        const vote = mtpAdaptiveVoteFor(table_ms_tok, window_ms_tok, serial_ms, MTP_ADAPTIVE_MARGIN);
        const prev_bucket = self.mtp_adaptive.bucket;
        const prev_arm = self.mtp_adaptive.arm;
        const action = self.mtp_adaptive.round(self.mtp_ev_rounds, b, vote, MTP_ADAPTIVE_CONFIRM);
        self.mtpAdaptiveSyncWindow(prev_bucket, prev_arm);
        if (action == .to_serial) {
            // The round under construction still runs; the switch takes effect next tick.
            log.info(
                "  [mtp] adaptive: bucket {s} mtp table {d:.2} / window {d:.2} ms/tok (w{d}) vs serial {d:.2} ms/tok -> serial (from the next round)\n",
                .{ round_cost.bucketName(self.xfm.round_cost.layout, b), table_ms_tok.?, window_ms_tok.?, m_lo, serial_ms.? },
            );
            self.spec_disabled_runtime = true;
            self.spec_disable_reason = .adaptive;
            // On a module-owned head the switch is one-way; the latch is consumed at the block boundary.
            if (stickyOnSerialSwitch(self.mtp)) self.mtp_adaptive.stickSerial();
            return true;
        }
        // Nothing to decide with: teach the bucket a serial token, once.
        const idle = self.mtp_serial_left == 0 and self.mtp_serial_exit == .none;
        if (mtpSerialProbeArm(t, b, self.spec_cost_solo, idle, self.mtpAdaptiveHeadMayResume(), mtpSerialProbeUseful(window_ms_tok))) |own| {
            self.mtp_serial_left = MTP_ADAPTIVE_PROBE_TOKENS;
            log.info(
                "  [mtp] adaptive: bucket {s} has no serial cell -> probing {d} serial tokens\n",
                .{ round_cost.bucketName(self.xfm.round_cost.layout, own), MTP_ADAPTIVE_PROBE_TOKENS },
            );
            return true;
        }
    }
    return false;
}

/// One debug line per planned round with every input the planner read.
pub fn mtpPlannerMode(self: *const Generator) u8 {
    const head = self.mtp orelse return 0;
    const sharp = !sampler.isGreedyTemperature(self.mtpDraftSampling(0).temperature);
    return group_cost.GroupShape.samplingMode(head.canRerankDrafts(), false, sharp, !sampler.isGreedyTemperature(self.sampling.temperature));
}

pub fn mtpPlannerObserve(self: *Generator, observed: u32, accepted: u32) void {
    if (!group_planner.enabled() or !self.mtp_planner_owned) return;
    self.mtp_planner_history.observe(observed, @min(observed, accepted));
    self.mtp_planner_spec_rounds += 1;
    self.mtp_hidden_stale = false;
    self.mtp_planner_recovery.observe(self.completion_tokens, self.mtp_planner_recovering);
    if (self.mtp_planner_probe and !self.mtp_planner_recovering) self.mtp_planner_probes +|= 1;
}

pub fn mtpRoundPlan(self: *Generator) MtpRoundPlan {
    if (group_planner.enabled()) if (self.mtp_planner_width) |width| {
        return .{ .m_lo = width, .m_hi = width, .tau_ln = 0 };
    };
    var plan = self.mtpRoundPlanTraced();
    if (self.mtp_group_cap > 0) {
        plan.m_lo = @min(plan.m_lo, self.mtp_group_cap);
        plan.m_hi = @min(plan.m_hi, self.mtp_group_cap);
    }
    return plan;
}

pub fn mtpRoundPlanTraced(self: *Generator) MtpRoundPlan {
    if (!log.isDebug()) return self.mtpRoundPlanInner();
    const disabled_before = self.spec_disabled_runtime;
    const serial_left_before = self.mtp_serial_left;
    const plan = self.mtpRoundPlanInner();
    const kv_len = self.mtpKvLen();
    const t = &self.xfm.round_cost;
    const b = self.mtpAdaptiveBucket(kv_len);
    const src = MtpCostSource.init(self.mtp_ev_costs, kv_len, if (mtpCostTableEnabled()) t else null);
    const serial_step: []const u8 = if (!disabled_before and self.spec_disabled_runtime)
        "switch"
    else if (self.mtp_serial_left > serial_left_before)
        "probe"
    else
        "none";
    var cells: [256]u8 = undefined;
    const a = self.mtp_ev_accept;
    const cap_row: u32 = @min(@max(@as(u32, 1), self.mtp_depth), mtp_mod.MAX_DEPTH);
    const cap_free: u32 = @min(@max(cap_row, self.mtp_depth_free), mtp_mod.MAX_DEPTH);
    const target = mtpWidthTrialTarget(t, kv_len, plan, cap_free, self.mtp_m_lo_streak >= 2);
    log.debug(
        "  [mtp-plan] r={d} kv={d} bucket={s} m_lo={d} m_hi={d} tau={d:.3} a=[{d:.2},{d:.2},{d:.2},{d:.2},{d:.2},{d:.2}] src={s} trial={s}{d} serial_step={s} streak={d} serial_left={d} drops=t{d}/c{d}/b{d} cap={d}/{d} solo={} rte={d} wt={d}/{d} tgt={?d} cells={s}\n",
        .{
            self.mtp_ev_rounds,                        kv_len,
            round_cost.bucketName(t.layout, b),        plan.m_lo,
            plan.m_hi,                                 plan.tau_ln,
            a[0],                                      a[1],
            a[2],                                      a[3],
            a[4],                                      a[5],
            if (src.fromTable()) "table" else "prior", if (plan.width_trial) "w" else "-",
            if (plan.width_trial) plan.m_lo else 0,    serial_step,
            self.mtp_m_lo_streak,                      self.mtp_serial_left,
            t.dropped_transition,                      t.dropped_contended,
            t.dropped_bad,                             cap_row,
            cap_free,                                  self.spec_cost_solo,
            self.mtp_regime.trial_end,                 self.mtp_width_trial.next_trial,
            self.mtp_width_trial.trial_end,            target,
            t.formatBucket(b, &cells),
        },
    );
    return plan;
}

pub fn mtpRoundPlanInner(self: *Generator) MtpRoundPlan {
    const cap_row: u32 = @min(@max(@as(u32, 1), self.mtp_depth), mtp_mod.MAX_DEPTH);
    if (mtpPinnedDepth()) |d| {
        const forced = mtpForcedWidth(@min(d, cap_row), mtpVerifyDraftsMax(self.xfm.config.isMimo()));
        self.mtp_ev_m_lo_prev = forced;
        return .{ .m_lo = forced, .m_hi = forced, .tau_ln = 0.0 };
    }
    const cap_free: u32 = @min(@max(cap_row, self.mtp_depth_free), mtp_mod.MAX_DEPTH);
    var cap: u32 = cap_row;
    const kv_len = self.mtpKvLen();
    if (!mtpAdaptiveEnabled() or self.mtp_ev_rounds < MTP_EV_WARMUP_ROUNDS) {
        const d = depth_bounds.clampWidth(self.mtp_depth_current, depth_bounds.floorFor(depth_bounds.active, cap), cap);
        self.mtp_ev_m_lo_prev = d;
        // Independent of the depth controller's kill switch and warmup.
        _ = self.mtpAdaptiveSerialStep(d, kv_len);
        return .{ .m_lo = d, .m_hi = d, .tau_ln = 0.0 };
    }
    const src = MtpCostSource.init(self.mtp_ev_costs, kv_len, if (mtpCostTableEnabled()) &self.xfm.round_cost else null);
    if (src.fromTable() and !self.xfm.round_cost.first_use_logged) {
        self.xfm.round_cost.first_use_logged = true;
        var buf: [256]u8 = undefined;
        log.info("[mtp] cost table: bucket {s} measured {s} (ms/tok) replaces the fitted surface (scale {d:.4} at w{d})\n", .{
            round_cost.bucketName(self.xfm.round_cost.layout, src.bucket),
            self.xfm.round_cost.formatBucket(src.bucket, &buf),
            src.scale,
            self.xfm.round_cost.narrowestMeasured(src.bucket) orelse 0,
        });
    }
    // The per-silicon row is the COLD-START cap: the plan may exceed it
    // only up to the widest TRUSTED width (never onto the prior's guess
    // above the row — that reopens the regime gate on ties the row had
    // closed), and the width trial may reach one past that to measure.
    if (src.fromTable()) cap = @min(cap_free, @max(cap_row, self.xfm.round_cost.widestMeasured(src.bucket) orelse cap_row));
    var plan = mtpPlanInBounds(
        mtpEvPlanSrc(self.mtp_ev_accept[0..cap], cap, src, self.mtp_ev_m_lo_prev + 1),
        depth_bounds.floorFor(depth_bounds.active, cap),
        cap,
    );
    if (plan.m_lo == self.mtp_ev_m_lo_prev) self.mtp_m_lo_streak +|= 1 else self.mtp_m_lo_streak = 0;
    self.mtp_ev_m_lo_prev = plan.m_lo;
    // Live-cost lever: shorten dry exploration bursts when the MEASURED
    // chunk-A sync is an expensive fraction of the round (mtpExtDryThresholdFor).
    const dry_threshold = mtpExtDryThresholdFor(self.mtp_ev_sync_ms, self.mtp_ev_round_ms);
    if (plan.m_hi > plan.m_lo and
        !mtpExtDryAllows(&self.mtp_ext_dry_streak, &self.mtp_ext_cooldown, dry_threshold))
    {
        plan.m_hi = plan.m_lo;
        plan.tau_ln = 0.0;
    }
    if (plan.m_hi > plan.m_lo) {
        const force = mtpRegimeForce(&self.mtp_regime, self.mtp_ev_rounds);
        const worse = self.mtp_regime.sched_verdict;
        if (worse != null and worse != self.mtp_regime_verdict) {
            self.mtp_regime_verdict = worse;
            const r = self.mtp_regime;
            log.info("[mtp] regime gate: two-chunk {d:.2} ms/tok vs single {d:.2} ms/tok -> two-chunk {s} (period {d})\n", .{ r.two_ms / r.two_tok, r.one_ms / r.one_tok, if (worse.?) "throttled" else "every round", mtpRegimeExplorePeriod(r) });
        }
        if (force == false) {
            plan.m_hi = plan.m_lo;
            plan.tau_ln = 0.0;
        }
    }
    if (self.mtpAdaptiveSerialStep(plan.m_lo, kv_len)) return plan;
    // Width trial: a single-chunk round at the width the table needs
    // next (`mtpWidthTrialTarget`), one 2-round block per period. Never
    // inside a regime trial block (that block is the regime's own
    // measurement), and only while solo.
    if (mtpCostTableEnabled() and self.spec_cost_solo and self.mtp_ev_rounds >= self.mtp_regime.trial_end) {
        const base_settled = self.mtp_m_lo_streak >= 2;
        if (mtpWidthTrialTarget(&self.xfm.round_cost, kv_len, plan, cap_free, base_settled)) |target| {
            const period = mtpWidthTrialPeriod(&self.xfm.round_cost, kv_len, plan.m_lo);
            const reread = round_cost.schedulePeriodReread(self.xfm.round_cost.layout);
            if (mtpWidthTrialForce(&self.mtp_width_trial, self.mtp_ev_rounds, period, reread)) {
                plan = mtpWidthTrialPlan(target);
            }
        }
    }
    return plan;
}

/// The single-chunk plan a width trial runs. Built from a scalar on
/// purpose: `plan = .{ .m_lo = plan.m_lo + 1, .m_hi = plan.m_lo + 1 }`
/// reads the already-written m_lo for m_hi (result-location aliasing)
/// and planned a two-chunk round — the simulated loop caught it.
pub fn mtpWidthTrialPlan(width: u32) MtpRoundPlan {
    return .{ .m_lo = width, .m_hi = width, .tau_ln = 0.0, .width_trial = true };
}

/// Which width a trial measures: the plan's own base when the bucket
/// the plan reads has not measured it, else an unmeasured m_lo+1 under
/// any shape, else a periodic m_lo+1 on single-chunk plans. Null =
/// nothing to try.
pub fn mtpWidthTrialTarget(t: *const round_cost.Table, kv_len: u32, plan: MtpRoundPlan, cap: u32, base_settled: bool) ?u32 {
    const b = t.bucketToRead(kv_len) orelse t.bucketOf(kv_len);
    // A single-chunk plan feeds its own base every round; only a
    // two-chunk plan (extensions and syncs never feed) owes a trial of
    // it — and only once the base has stopped climbing, or the first EV
    // rounds trial widths the plan is merely passing through (M1 Pro
    // 27B warm boot: one block at w3 in rep 1).
    if (t.measuredMs(plan.m_lo, b) == null) return if (plan.m_hi > plan.m_lo and base_settled) plan.m_lo else null;
    // m_lo-1 is deliberately NOT trialled: every trial block is a 3-4%
    // hit on the request that carries it (peer cells, 2026-08-22), the
    // prior below the anchor already prices narrower widths, and the
    // one cliff this was for (M1 Pro 27B, 4 -> 5) is found from 4 by
    // the m_lo+1 trial below.
    if (plan.m_lo >= cap) return null;
    // An unmeasured m_lo+1 is measured under ANY shape: a two-chunk
    // plan extends there every round on echo, blind to its cost (the
    // M1 Pro 27B's 4 -> 5 at +150 ms); the regime gate would catch it,
    // the table should know it — unless its first sample already
    // settled it as clearly worse. A settled one is re-tried only where
    // a single-chunk plan cannot otherwise reach it, at the long period.
    const up = plan.m_lo + 1;
    if (t.measuredMs(up, b) == null and !t.clearlyWorse(up, plan.m_lo, b)) return up;
    if (plan.m_hi == plan.m_lo) return up;
    return null;
}

/// `reread` is the layout's answer (`round_cost.schedulePeriodReread`): every layout re-reads
/// the period so a cold bucket cannot inherit a neighbour's date.
pub fn mtpWidthTrialForce(t: *MtpWidthTrial, round_idx: u32, period: u32, reread: bool) bool {
    return t.force(round_idx, period, reread);
}

/// Rounds between width trials: the regime's drag rule on the measured
/// ms/tok gap between m_lo+1 and m_lo (a width G worse, tried once in
/// G/DRAG rounds, costs ~DRAG of throughput); the default period while
/// either is unmeasured, so an unknown width is learned soon.
pub fn mtpWidthTrialPeriod(t: *const round_cost.Table, kv_len: u32, m_lo: u32) u32 {
    const b = t.bucketToRead(kv_len) orelse return round_cost.EXPLORE_PERIOD_COLD;
    // An untrusted target keeps the cold period until it is either
    // trusted or settled as clearly worse (then its raw sample sizes
    // the gap): a raw 10%-better w5 read as a 60-round period and no
    // 22-round request ever trialled it again (M4 Max 27B, w5 stuck
    // at one sample).
    if (t.msPerTok(m_lo + 1, b) == null and !t.clearlyWorse(m_lo + 1, m_lo, b)) return round_cost.EXPLORE_PERIOD_COLD;
    return round_cost.trialPeriod(t.msPerTok(m_lo, b), t.rawMsPerTok(m_lo + 1, b));
}

/// Track the only evidence that can justify sticky-disable: whether the
/// first draft landed while the EV base depth was exactly one. Wider base
/// rounds reset the probation window, and later extension misses do not
/// count against the depth-one floor. Returns a rate only after a full
/// fresh window has been observed.
pub fn mtpFloorDisableObserve(
    drafted_window: *[MTP_DEPTH_WINDOW]u8,
    accepted_window: *[MTP_DEPTH_WINDOW]u8,
    window_idx: *u32,
    m_lo: u32,
    drafted: u32,
    accepted: u32,
) ?f32 {
    std.debug.assert(drafted >= 1);
    if (m_lo != 1) {
        window_idx.* = 0;
        return null;
    }

    const idx = window_idx.* % MTP_DEPTH_WINDOW;
    drafted_window[idx] = 1;
    accepted_window[idx] = @intFromBool(accepted > 0);
    window_idx.* += 1;
    const n = @min(window_idx.*, MTP_DEPTH_WINDOW);
    if (n < MTP_DEPTH_WINDOW) return null;

    var accepted_sum: u32 = 0;
    var i: u32 = 0;
    while (i < n) : (i += 1) accepted_sum += accepted_window[i];
    return @as(f32, @floatFromInt(accepted_sum)) / @as(f32, @floatFromInt(n));
}

/// EV-mode per-round update: EMAs always; during warmup the legacy
/// windowed controller keeps running (today's behavior while EMAs fill);
/// post-warmup only the sticky disable floor is checked — EV owns depth.
pub fn updateMtpEvRound(self: *Generator, drafted: u32, accepted: u32) void {
    mtpEvObserve(&self.mtp_ev_accept, drafted, accepted, MTP_EV_EMA_BETA);
    self.mtp_ev_rounds += 1;
    if (mtpPinnedDepth() != null) return;
    if (self.mtp_ev_rounds <= MTP_EV_WARMUP_ROUNDS) {
        self.updateMtpDepth(drafted, accepted);
        // Warmup may evaluate several depths. None of that mixed evidence
        // belongs in the post-warmup depth-one sticky-disable window.
        if (self.mtp_ev_rounds == MTP_EV_WARMUP_ROUNDS) self.mtp_window_idx = 0;
        return;
    }
    // EV owns promotion/demotion. Sticky-disable is judged only from a
    // full, homogeneous window of first-draft outcomes at base depth one.
    const rate = mtpFloorDisableObserve(
        &self.mtp_window_drafted,
        &self.mtp_window_accepted,
        &self.mtp_window_idx,
        self.mtp_ev_m_lo_prev,
        drafted,
        accepted,
    ) orelse return;
    if (rate < MTP_DISABLE_BELOW) {
        log.info(
            "  mtp=disabled (EV: depth-1 first-draft rate {d:.2} < {d:.2})\n",
            .{ rate, MTP_DISABLE_BELOW },
        );
        self.spec_disabled_runtime = true;
        self.spec_disable_reason = .acceptance;
    }
}

/// Close one traced round; emits + resets the summary at the cadence.
pub fn mtpTraceRoundEnd(self: *Generator, m: u32, accepted: u32, m_lo: u32) void {
    if (!mtpTraceEnabled()) return;
    if (!self.mtp_trace.endRound(m, accepted, m > m_lo)) return;
    const t = &self.mtp_trace;
    var acc_buf: [64]u8 = undefined;
    log.info(
        MtpTrace.LINE_FMT,
        .{
            t.rounds,
            t.avgMs(.draft),
            t.avgMs(.sync),
            t.avgMs(.ext),
            t.avgMs(.verify),
            t.avgMs(.corr),
            t.avgMs(.eval),
            t.avgMs(.hist),
            t.avgMs(.commit),
            t.avgMs(.predraft),
            t.avgMs(.gap),
            t.totalAvgMs(),
            @as(f64, @floatFromInt(t.drafted)) / @as(f64, @floatFromInt(t.rounds)),
            @as(f64, @floatFromInt(t.accepted)) / @as(f64, @floatFromInt(t.rounds)),
            @as(f64, @floatFromInt(t.extended)) / @as(f64, @floatFromInt(t.rounds)),
            t.accIdxStr(&acc_buf),
            t.subAvgMs(.build),
            t.subAvgMs(.ple),
            t.subAvgMs(.ple_sync),
            t.subAvgMs(.dispatch),
            t.subAvgMs(.wait),
            t.subAvgMs(.chain),
        },
    );
    t.reset();
}

test "mtpNextDepth: adaptive depth policy transitions" {
    const configured: u32 = 3;
    // Hot at configured depth: stay.
    try testing.expectEqual(@as(u32, 3), Generator.mtpNextDepth(3, configured, 0.9));
    // Sagging at depth > 1: step down (one level at a time). The demote
    // floor is 0.40 under capture-based rollback (a rejected draft costs
    // only its own head pass, not a trunk re-forward).
    try testing.expectEqual(@as(u32, 2), Generator.mtpNextDepth(3, configured, 0.35));
    try testing.expectEqual(@as(u32, 1), Generator.mtpNextDepth(2, configured, 0.30));
    // Mid-band (0.40..0.60): hold.
    try testing.expectEqual(@as(u32, 3), Generator.mtpNextDepth(3, configured, 0.48));
    try testing.expectEqual(@as(u32, 2), Generator.mtpNextDepth(2, configured, 0.55));
    try testing.expectEqual(@as(u32, 1), Generator.mtpNextDepth(1, configured, 0.55));
    // Hot below configured depth: promote (band top is 0.60).
    try testing.expectEqual(@as(u32, 2), Generator.mtpNextDepth(1, configured, 0.73));
    try testing.expectEqual(@as(u32, 3), Generator.mtpNextDepth(2, configured, 0.70));
    // Never exceeds configured.
    try testing.expectEqual(@as(u32, 3), Generator.mtpNextDepth(3, configured, 0.99));
    // Depth 1 at 0.40: speculation still pays (~+27% over AR at current
    // round costs — the disable floor sits at the measured breakeven, not
    // at "acceptance feels low"). The old 0.50 floor DISABLED here and
    // cratered the oQ4e 16K/32K ladder cells to bare AR (24-26 tok/s).
    try testing.expectEqual(@as(u32, 1), Generator.mtpNextDepth(1, configured, 0.40));
    // Depth 1 below the true breakeven (+margin): disable (0).
    try testing.expectEqual(@as(u32, 0), Generator.mtpNextDepth(1, configured, 0.15));
    // Demote-before-disable: a terrible rate at depth 2 still goes through 1.
    try testing.expectEqual(@as(u32, 1), Generator.mtpNextDepth(2, configured, 0.10));
}

test "mtpDepthDecision: confidence gates on disable, promote, cooldown" {
    const W = Generator.MTP_DEPTH_WINDOW;
    // Disable needs a FULL window of evidence; small samples hold at depth 1.
    try testing.expectEqual(@as(u32, 1), Generator.mtpDepthDecision(1, 3, 0.15, 5, false));
    try testing.expectEqual(@as(u32, 1), Generator.mtpDepthDecision(1, 3, 0.15, W - 1, false));
    try testing.expectEqual(@as(u32, 0), Generator.mtpDepthDecision(1, 3, 0.15, W, false));
    // Rates the old 0.50 floor killed keep speculating at any window size.
    try testing.expectEqual(@as(u32, 1), Generator.mtpDepthDecision(1, 3, 0.40, W, false));
    // Promote needs >= 8 rounds AND no active cooldown.
    try testing.expectEqual(@as(u32, 1), Generator.mtpDepthDecision(1, 3, 0.95, 7, false));
    try testing.expectEqual(@as(u32, 2), Generator.mtpDepthDecision(1, 3, 0.95, 8, false));
    try testing.expectEqual(@as(u32, 1), Generator.mtpDepthDecision(1, 3, 0.95, 8, true));
    // Demote reacts on a small sample, even during cooldown.
    try testing.expectEqual(@as(u32, 1), Generator.mtpDepthDecision(2, 3, 0.30, 5, true));
}

test "mtpDepthCapFor: auto cap follows the selected cost profile; explicit always wins" {
    // 0 = auto (--mtp-max-depth not passed).
    // .generic's auto cap is per-silicon, so the chip must be injected here
    // or the assertion is a property of whichever Mac runs the suite.
    try testing.expectEqual(Generator.MTP_ADAPTIVE_DEFAULT_CAP, Generator.mtpDepthCapForProfileChip(0, true, .generic, "Apple M4 Max"));
    try testing.expectEqual(@as(u32, 6), Generator.mtpDepthCapForProfileChip(0, true, .generic, ""));
    try testing.expectEqual(@as(u32, 4), Generator.mtpDepthCapForProfileChip(0, true, .generic, "Apple M1 Pro"));
    // EXL3 packs included: the M5 Max row is the default.
    try testing.expectEqual(@as(u32, 6), Generator.mtpDepthCapForProfileChip(0, true, .generic, "Apple M5 Max"));
    // An explicit depth still outranks the table on a measured chip.
    try testing.expectEqual(@as(u32, 8), Generator.mtpDepthCapForProfileChip(8, true, .generic, "Apple M1 Pro"));
    try testing.expectEqual(mtp_mod.DEFAULT_DEPTH, Generator.mtpDepthCapForProfileChip(0, false, .generic, "Apple M1 Pro"));
    // qwen4's calibrated surface keeps the DEFAULT cap: no NAX region exists
    // to justify opening depths 7-8 on that arch.
    try testing.expectEqual(Generator.MTP_ADAPTIVE_DEFAULT_CAP, Generator.mtpDepthCapForProfile(0, true, .g17_nax_qwen4_q4_gs64));
    try testing.expectEqual(@as(u32, 6), Generator.mtpDepthCapForProfile(0, true, .g17_nax_qwen4_q4_gs64));
    for ([_]mtp_mod.MtpCostProfile{ .generic, .g17_nax_qwen4_q4_gs64, .g17_nax_qwen4_mixed_4_8_gs64 }) |profile| {
        try testing.expectEqual(mtp_mod.DEFAULT_DEPTH, Generator.mtpDepthCapForProfile(0, false, profile));
    }
    // Explicit values ignore both controller mode and profile, and remain
    // clamped to [1, MAX_DEPTH].
    try testing.expectEqual(@as(u32, 5), Generator.mtpDepthCapForProfile(5, true, .generic));
    try testing.expectEqual(@as(u32, 5), Generator.mtpDepthCapForProfile(5, false, .g17_nax_qwen4_q4_gs64));
    try testing.expectEqual(@as(u32, 7), Generator.mtpDepthCapForProfile(7, true, .generic));
    try testing.expectEqual(@as(u32, 8), Generator.mtpDepthCapForProfile(8, true, .generic));
    try testing.expectEqual(@as(u32, 2), Generator.mtpDepthCapForProfile(2, true, .g17_nax_qwen4_q4_gs64));
    try testing.expectEqual(mtp_mod.MAX_DEPTH, Generator.mtpDepthCapForProfile(12, true, .generic));
}

test "mtpFloorDisableObserve: extension misses do not poison depth one" {
    const W = Generator.MTP_DEPTH_WINDOW;
    var drafted: [W]u8 = @splat(0);
    var accepted: [W]u8 = @splat(0);
    var idx: u32 = 0;

    var rate: ?f32 = null;
    for (0..W) |_| {
        // An eight-wide extension accepted its first draft but no later one.
        rate = Generator.mtpFloorDisableObserve(&drafted, &accepted, &idx, 1, 8, 1);
    }
    try testing.expectApproxEqAbs(@as(f32, 1.0), rate.?, 1e-5);
    for (drafted) |sample| try testing.expectEqual(@as(u8, 1), sample);

    drafted = @splat(0);
    accepted = @splat(0);
    idx = 0;
    for (0..W) |i| {
        rate = Generator.mtpFloorDisableObserve(
            &drafted,
            &accepted,
            &idx,
            1,
            8,
            @intFromBool(i % 2 == 0),
        );
    }
    try testing.expectApproxEqAbs(@as(f32, 0.5), rate.?, 1e-5);
    // A 50% depth-one window is comfortably above the breakeven floor —
    // this rate must never disable (the old 0.50 floor sat exactly here).
    try testing.expect(!(rate.? < Generator.MTP_DISABLE_BELOW));
}

test "mtpFloorDisableObserve: disable needs 16 fresh failures at base depth one" {
    const W = Generator.MTP_DEPTH_WINDOW;
    var drafted: [W]u8 = @splat(0);
    var accepted: [W]u8 = @splat(0);
    var idx: u32 = 0;

    // Fifteen depth-one failures are insufficient.
    for (0..W - 1) |_| {
        try testing.expectEqual(
            @as(?f32, null),
            Generator.mtpFloorDisableObserve(&drafted, &accepted, &idx, 1, 1, 0),
        );
    }
    // A wider base round invalidates that probation window.
    try testing.expectEqual(
        @as(?f32, null),
        Generator.mtpFloorDisableObserve(&drafted, &accepted, &idx, 2, 4, 0),
    );
    try testing.expectEqual(@as(u32, 0), idx);

    // It takes another complete run of depth-one failures to produce a rate.
    for (0..W - 1) |_| {
        try testing.expectEqual(
            @as(?f32, null),
            Generator.mtpFloorDisableObserve(&drafted, &accepted, &idx, 1, 1, 0),
        );
    }
    const rate = Generator.mtpFloorDisableObserve(&drafted, &accepted, &idx, 1, 1, 0);
    try testing.expectApproxEqAbs(@as(f32, 0.0), rate.?, 1e-5);
}

test "mtpEvExpectedTokens: 1 + sum of acceptance chain products" {
    const a = [_]f32{ 0.5, 0.5, 0.5 };
    try testing.expectApproxEqAbs(@as(f32, 1.5), Generator.mtpEvExpectedTokens(&a, 1), 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1.75), Generator.mtpEvExpectedTokens(&a, 2), 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1.875), Generator.mtpEvExpectedTokens(&a, 3), 1e-5);
    // Zero acceptance: every round still commits exactly the t1 bonus token.
    const z = [_]f32{ 0.0, 0.0 };
    try testing.expectApproxEqAbs(@as(f32, 1.0), Generator.mtpEvExpectedTokens(&z, 2), 1e-5);
}

test "mtpEvRoundCost: piecewise marginals (flat verify region, then the GDN width ramp)" {
    const costs = Generator.MtpEvCosts{ .draft = 0.10, .per_pos_lo = 0.09, .per_pos_hi = 0.22, .flat_max = 3, .sync = 0.02 };
    try testing.expectApproxEqAbs(@as(f32, 1.19), Generator.mtpEvRoundCost(costs, 1, false), 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1.57), Generator.mtpEvRoundCost(costs, 3, false), 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1.59), Generator.mtpEvRoundCost(costs, 3, true), 1e-5);
    // Positions 4+ pay the ramp: +0.32 each instead of +0.19.
    try testing.expectApproxEqAbs(@as(f32, 2.21), Generator.mtpEvRoundCost(costs, 5, false), 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.19), Generator.mtpEvMarginalCost(costs, 3), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.32), Generator.mtpEvMarginalCost(costs, 4), 1e-6);
}

test "mtpEvCostsForProfile: G17 profiles are explicit" {
    const generic = Generator.mtpEvCostsForProfile(.generic);
    try testing.expectEqual(Generator.MTP_EV_DEFAULT_COSTS, generic);

    const qwen4 = Generator.mtpEvCostsForProfile(.g17_nax_qwen4_q4_gs64);
    try testing.expectEqual(Generator.MTP_EV_G17_NAX_QWEN4_Q4_GS64_COSTS, qwen4);
    try testing.expect(!std.meta.eql(generic, qwen4));
}

test "mtpEvPlanFor: mid-decay acceptance picks a shallow base and a confidence-gated extension" {
    const costs = Generator.MtpEvCosts{ .draft = 0.10, .per_pos_lo = 0.09, .per_pos_hi = 0.22, .flat_max = 3, .sync = 0.02 };
    // Conditional acceptance decays: unconditional EV peaks at m=2, but the
    // marginal chain CONDITIONAL on chunk A landing stays profitable through
    // the flat verify region — the "draft deeper on easy stretches" shape.
    const a = [_]f32{ 0.7, 0.6, 0.55, 0.5, 0.45, 0.42, 0.4, 0.38 };
    const plan = Generator.mtpEvPlanFor(&a, 8, costs, 8);
    try testing.expectEqual(@as(u32, 2), plan.m_lo);
    try testing.expectEqual(@as(u32, 3), plan.m_hi);
    // tau = r(m_lo)*t_ext/S = 1.5362*0.19/0.55 = 0.5307 -> ln = -0.6335.
    try testing.expectApproxEqAbs(@as(f32, -0.6335), plan.tau_ln, 5e-3);
}

test "mtpEvPlanFor: hot flat acceptance rides the flat region and extends into the ramp on confidence" {
    const costs = Generator.MtpEvCosts{ .draft = 0.10, .per_pos_lo = 0.09, .per_pos_hi = 0.22, .flat_max = 3, .sync = 0.02 };
    const a = [_]f32{ 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9 };
    const plan = Generator.mtpEvPlanFor(&a, 8, costs, 8);
    // Static optimum m=3 (r=2.1904); ramp positions 4..6 pay only under full
    // confidence (0.9^k chain vs r*0.32 = 0.70 threshold).
    try testing.expectEqual(@as(u32, 3), plan.m_lo);
    try testing.expectEqual(@as(u32, 6), plan.m_hi);
    // tau = 2.1904*0.96/2.439 = 0.8621 -> ln = -0.1484.
    try testing.expectApproxEqAbs(@as(f32, -0.1484), plan.tau_ln, 5e-3);
}

test "mtpEvPlanFor: a stale-cold EMA one past m_lo cannot close the horizon permanently" {
    // a[1] is observable ONLY when extension fires (at m=1 rounds it never
    // updates), so a value dragged down under an old workload must not be
    // able to lock the horizon shut — the live regression was ext_rounds=0
    // on the equivalence echo after refit #3's honest marginals. When the
    // base pays, one extension position stays reachable at the clamped
    // exploration tau; realized extensions then re-observe a[1] honestly.
    const costs = Generator.MTP_EV_DEFAULT_COSTS;
    const a = [_]f32{ 0.75, 0.28, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85 };
    const plan = Generator.mtpEvPlanFor(&a, 8, costs, 8);
    try testing.expectEqual(@as(u32, 1), plan.m_lo);
    try testing.expectEqual(@as(u32, 2), plan.m_hi);
    // tau clamps to TAU_MAX: only near-perfect-confidence rounds extend.
    try testing.expectApproxEqAbs(@log(Generator.MTP_EV_TAU_MAX), plan.tau_ln, 1e-6);
}

test "mtpEvPlanFor: cold acceptance collapses to depth 1, single chunk" {
    const costs = Generator.MtpEvCosts{ .draft = 0.10, .per_pos_lo = 0.09, .per_pos_hi = 0.22, .flat_max = 3, .sync = 0.02 };
    const a = [_]f32{ 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 };
    const plan = Generator.mtpEvPlanFor(&a, 8, costs, 8);
    try testing.expectEqual(@as(u32, 1), plan.m_lo);
    try testing.expectEqual(@as(u32, 1), plan.m_hi);
}

test "mtpEvPlanFor: m_lo_max damps the climb without killing the extension horizon" {
    const costs = Generator.MtpEvCosts{ .draft = 0.10, .per_pos_lo = 0.09, .per_pos_hi = 0.22, .flat_max = 3, .sync = 0.02 };
    const a = [_]f32{ 0.7, 0.6, 0.55, 0.5, 0.45, 0.42, 0.4, 0.38 };
    // Same EMAs as the mid-decay case, but the controller may only raise the
    // base one step (hysteresis): m_lo caps at 1 while m_hi stays deeper.
    const plan = Generator.mtpEvPlanFor(&a, 8, costs, 1);
    try testing.expectEqual(@as(u32, 1), plan.m_lo);
    try testing.expectEqual(@as(u32, 3), plan.m_hi);
    try testing.expect(plan.tau_ln < 0.0);
}

test "mtpEvPlanFor: unobserved deep indices at the prior still open the extension horizon (exploration)" {
    const costs = Generator.MtpEvCosts{ .draft = 0.10, .per_pos_lo = 0.09, .per_pos_hi = 0.22, .flat_max = 3, .sync = 0.02 };
    // The echo shape after warmup: shallow indices observed hot, deep indices
    // never reached (still at MTP_EV_PRIOR). The horizon must open past m_lo
    // so extension can get its first trial — this is the live ext_rounds=0
    // regression (a prior at the measured average sat razor-under the ramp
    // break-even of ~0.78 and extension never fired on pure echo).
    const p = Generator.MTP_EV_PRIOR;
    const a = [_]f32{ 0.97, 0.97, p, p, p, p, p };
    const plan = Generator.mtpEvPlanFor(&a, 7, costs, 8);
    try testing.expectEqual(@as(u32, 3), plan.m_lo);
    try testing.expect(plan.m_hi > plan.m_lo);
    // tau = r(3)*0.32/0.85 = 0.8898 -> ln = -0.1168 (under the 0.95 clamp).
    try testing.expectApproxEqAbs(@as(f32, -0.1168), plan.tau_ln, 5e-3);
}

test "mtpRegime: the worse shape runs as a scheduled trial block, unmeasured runs as planned" {
    const measure = struct {
        fn shape(r: *Generator.MtpRegime, two_chunk: bool, ms: f32, tok: f32) void {
            for (0..Generator.MTP_REGIME_MIN_SAMPLES) |_| Generator.mtpRegimeObserve(r, two_chunk, 4, ms, tok);
        }
    };
    var r = Generator.MtpRegime{};
    // Unseeded: the plan stands (two-chunk measures itself); once two-chunk
    // is measured, an unmeasured single is forced AT ONCE. The first round
    // of a shape is its transition and is not observed.
    try testing.expect(Generator.mtpRegimeTwoChunkWorse(r) == null);
    try testing.expect(Generator.mtpRegimeForce(&r, 1) == null);
    measure.shape(&r, true, 129.0, 5.65); // first round counts
    try testing.expect(Generator.mtpRegimeForce(&r, 2) == false);
    // M4 base echo (measured): two-chunk 22.83 vs single 18.11 ms/tok = 26%
    // worse, so a 2-round trial block recurs every 53 rounds (1% drag). The
    // single's first round is a transition (dropped), the next ones measure.
    Generator.mtpRegimeObserve(&r, false, 4, 90.5, 5.0);
    try testing.expect(Generator.mtpRegimeForce(&r, 3) == false);
    measure.shape(&r, false, 90.5, 5.0);
    try testing.expect(Generator.mtpRegimeTwoChunkWorse(r).?);
    try testing.expectEqual(@as(u32, 53), Generator.mtpRegimeExplorePeriod(r));
    try testing.expect(Generator.mtpRegimeForce(&r, 14) == false); // verdict forms: next trial at 67
    try testing.expect(Generator.mtpRegimeForce(&r, 66) == false);
    try testing.expect(Generator.mtpRegimeForce(&r, 67) == null); // block
    try testing.expect(Generator.mtpRegimeForce(&r, 68) == null);
    try testing.expect(Generator.mtpRegimeForce(&r, 69) == false); // and not a third
    try testing.expectEqual(@as(u32, 69 + 53), r.next_trial);
    try testing.expectEqual(@as(u32, 14), r.verdict_round);
    try testing.expectEqual(@as(u32, 1), r.trials);
    // Asked twice for the same round (pre-draft + entry): same answer, no
    // schedule drift.
    try testing.expect(Generator.mtpRegimeForce(&r, 122) == null);
    try testing.expect(Generator.mtpRegimeForce(&r, 122) == null);
    try testing.expectEqual(@as(u32, 2), r.trials);
    try testing.expectEqual(@as(u32, 124 + 53), r.next_trial);
    // Two-chunk 64 ms for 6 (10.7) against single 56 for 4.96 (11.3):
    // two-chunk runs every round and a single-chunk block is scheduled once
    // per period to keep the other regime's EMA alive.
    var m = Generator.MtpRegime{};
    measure.shape(&m, true, 64.0, 6.0);
    Generator.mtpRegimeObserve(&m, false, 4, 56.0, 4.96);
    measure.shape(&m, false, 56.0, 4.96);
    try testing.expect(!Generator.mtpRegimeTwoChunkWorse(m).?);
    try testing.expectEqual(@as(u32, 12), Generator.mtpRegimeExplorePeriod(m)); // 5.8% gap
    try testing.expect(Generator.mtpRegimeForce(&m, 5) == null); // verdict forms: next trial at 17
    try testing.expect(Generator.mtpRegimeForce(&m, 16) == null);
    try testing.expect(Generator.mtpRegimeForce(&m, 17) == false);
    try testing.expect(Generator.mtpRegimeForce(&m, 18) == false);
    try testing.expect(Generator.mtpRegimeForce(&m, 19) == null);
    // M4 Max 27B @16k: 13.4 vs 12.95 ms/tok is inside the margin — the plan
    // stands (two-chunk measured +3.7% at arm level).
    var n = Generator.MtpRegime{};
    measure.shape(&n, true, 80.4, 6.0);
    Generator.mtpRegimeObserve(&n, false, 4, 64.75, 5.0);
    measure.shape(&n, false, 64.75, 5.0);
    try testing.expect(!Generator.mtpRegimeTwoChunkWorse(n).?);
    // Hysteresis (M1 Pro 9B v5.2): a standing "worse" at 26.4 vs 23.8 does
    // not flip when the post-block single reads 25.3 (ratio 1.04, inside the
    // margin); it flips only once two-chunk is at or below single.
    var h = Generator.MtpRegime{ .two_ms = 26.4, .two_tok = 1.0, .two_m = 4, .two_n = 4, .one_ms = 25.3, .one_tok = 1.0, .one_m = 4, .one_n = 4 };
    try testing.expect(Generator.mtpRegimeVerdict(h, true).?);
    try testing.expect(!Generator.mtpRegimeVerdict(h, null).?);
    h.one_ms = 26.5;
    try testing.expect(!Generator.mtpRegimeVerdict(h, true).?);
}

test "mtpRegime: a simulated round loop reaches BOTH shapes, and the worse one keeps being re-tried" {
    // Live 2026-08-22 (M4 base): v2's "try the unmeasured single shape at
    // once" rule compared one_m against a two_m that only a two-chunk round
    // can set, so it forced single-chunk forever — caps 5/6 reported cap-4
    // numbers with ext_rounds=0 and no verdict line, which reads as a PASS
    // on an echo workload. v4's `idx % period` gate then ran trial chains
    // because the block's own observation moved the period (14 of 72 rounds
    // as trials against 7). The gate is driven here exactly as mtpRoundPlan
    // drives it: ask, run that shape, feed the observation.
    const Sim = struct {
        fn run(two_ms: f32, one_ms: f32, rounds: u32) struct { two: u32, one: u32 } {
            var r = Generator.MtpRegime{};
            var two: u32 = 0;
            var one: u32 = 0;
            var i: u32 = 0;
            while (i < rounds) : (i += 1) {
                // The plan always offers two-chunk (pure echo); the gate decides.
                const two_chunk = Generator.mtpRegimeForce(&r, i) orelse true;
                if (two_chunk) two += 1 else one += 1;
                Generator.mtpRegimeObserve(&r, two_chunk, 4, if (two_chunk) two_ms else one_ms, if (two_chunk) 6.0 else 5.0);
            }
            return .{ .two = two, .one = one };
        }
    };
    // M4 base: two-chunk 26% worse. Pre-verdict 4 rounds, then 2-round
    // blocks every 53: 4 + 2*3 = 10 two-chunk rounds in 200, not 20.
    const worse = Sim.run(129.0, 90.5, 200);
    try testing.expect(worse.one > 185);
    try testing.expect(worse.two >= 6 and worse.two <= 12);
    // M4 Max: two-chunk better (5.8% gap): 2-round single blocks every 12.
    const better = Sim.run(64.0, 56.0, 200);
    try testing.expect(better.two > 160);
    try testing.expect(better.one >= 25 and better.one <= 40);
}

test "mtpRegime: one all-rejected round cannot set the verdict" {
    // A request's first two-chunk round (Flash-Next @204k, depth 6) emitted 1 token in 61.19 ms and
    // its first single round 3 in 33.09 ms; alone they read 61.2 vs 11.0 ms/tok. Pooled over the
    // rounds that follow, two-chunk is the cheaper shape (14.3 vs 14.9 ms/tok).
    var r = Generator.MtpRegime{};
    Generator.mtpRegimeObserve(&r, true, 1, 61.19, 1.0);
    Generator.mtpRegimeObserve(&r, false, 1, 33.0, 3.0); // transition
    Generator.mtpRegimeObserve(&r, false, 1, 33.09, 3.0);
    try testing.expect(Generator.mtpRegimeTwoChunkWorse(r) == null);
    for ([_]f32{ 34.0, 33.0, 34.0 }) |ms| Generator.mtpRegimeObserve(&r, false, 1, ms, 2.0);
    Generator.mtpRegimeObserve(&r, true, 1, 40.0, 4.0); // transition
    for ([_]f32{ 42.0, 40.0, 43.0 }) |ms| Generator.mtpRegimeObserve(&r, true, 1, ms, 4.0);
    try testing.expect(!Generator.mtpRegimeTwoChunkWorse(r).?);
}

test "mtpRegime: a shape is measured over MTP_REGIME_MIN_SAMPLES rounds before it is judged" {
    const n = Generator.MTP_REGIME_MIN_SAMPLES;
    var r = Generator.MtpRegime{};
    var i: u32 = 0;
    // The planned shape (two-chunk) runs until it has its samples, then single-chunk does:
    // its first round is the transition, the next n are its samples.
    while (i < n) : (i += 1) {
        try testing.expect(Generator.mtpRegimeForce(&r, i) == null);
        Generator.mtpRegimeObserve(&r, true, 4, 60.0, 3.0);
    }
    while (i < 2 * n + 1) : (i += 1) {
        try testing.expect(Generator.mtpRegimeForce(&r, i) == false);
        Generator.mtpRegimeObserve(&r, false, 4, 45.0, 2.0);
    }
    try testing.expect(!Generator.mtpRegimeTwoChunkWorse(r).?);
    try testing.expect(Generator.mtpRegimeForce(&r, i) == null);
    // A new base depth starts both shapes over.
    Generator.mtpRegimeObserve(&r, true, 5, 70.0, 4.0); // transition
    Generator.mtpRegimeObserve(&r, true, 5, 70.0, 4.0);
    try testing.expect(Generator.mtpRegimeTwoChunkWorse(r) == null);
    try testing.expect(Generator.mtpRegimeForce(&r, i + 1) == null);
}

test "mtpRegime: rounds with random acceptance run the cheaper shape most of the time" {
    // Per-round tokens are the draft chain's accepted prefix + 1 (per-draft acceptance 0.6): a
    // single-chunk round drafts 1, a two-chunk round 2, and two-chunk costs 15% less per token.
    const p: f32 = 0.6;
    const one_ms: f32 = 30.0;
    const two_ms: f32 = 0.85 * one_ms * (1.0 + p + p * p) / (1.0 + p);
    var two_rounds: u32 = 0;
    var rounds: u32 = 0;
    var seed: u64 = 0;
    while (seed < 64) : (seed += 1) {
        var prng = std.Random.DefaultPrng.init(seed);
        const rand = prng.random();
        var r = Generator.MtpRegime{};
        var i: u32 = 0;
        while (i < 300) : (i += 1) {
            const two_chunk = Generator.mtpRegimeForce(&r, i) orelse true;
            const drafts: u32 = if (two_chunk) 2 else 1;
            var tokens: f32 = 1.0;
            var d: u32 = 0;
            while (d < drafts and rand.float(f32) < p) : (d += 1) tokens += 1.0;
            if (two_chunk) two_rounds += 1;
            rounds += 1;
            Generator.mtpRegimeObserve(&r, two_chunk, 1, if (two_chunk) two_ms else one_ms, tokens);
        }
    }
    try testing.expect(@as(f32, @floatFromInt(two_rounds)) > 0.72 * @as(f32, @floatFromInt(rounds)));
}

test "MtpCostSource: a measured cliff stops the plan where the fitted surface would extend" {
    // M1 Pro 27B: depth 4 -> 5 is a +43% round. The fitted surface prices
    // position 5 at 0.26 floor units and extends; the table has measured
    // it. Same acceptance, same cap, two answers.
    const a = [_]f32{ 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95 };
    const prior = Generator.mtpEvPlanForAt(&a, 8, Generator.MTP_EV_DEFAULT_COSTS, 8, 10000);
    try testing.expect(prior.m_hi >= 5);
    var t = round_cost.Table{};
    for (0..round_cost.MIN_SAMPLES) |_| {
        _ = t.observe(3, 10000, 60.0, 4.0, true, false);
        _ = t.observe(4, 10000, 70.0, 5.0, true, false);
        _ = t.observe(5, 10000, 102.0, 6.0, true, false);
    }
    const src = Generator.MtpCostSource.init(Generator.MTP_EV_DEFAULT_COSTS, 10000, &t);
    try testing.expect(src.fromTable());
    // Anchored at the narrowest measured width: the two sources agree
    // there, and below it the prior's shape fills in.
    try testing.expectApproxEqAbs(Generator.mtpEvRoundCostAt(Generator.MTP_EV_DEFAULT_COSTS, 3, false, 10000), src.roundCost(3, false), 1e-4);
    try testing.expectApproxEqAbs(Generator.mtpEvRoundCostAt(Generator.MTP_EV_DEFAULT_COSTS, 2, false, 10000), src.roundCost(2, false), 1e-4);
    try testing.expect(src.marginal(5) > 2.5 * src.marginal(4));
    // Past the widest measured width the cliff's slope continues (it is
    // steeper than the prior's marginal here).
    try testing.expectApproxEqAbs(src.roundCost(5, false) + (src.roundCost(5, false) - src.roundCost(4, false)), src.roundCost(6, false), 1e-3);
    const measured = Generator.mtpEvPlanSrc(&a, 8, src, 8);
    try testing.expectEqual(@as(u32, 4), measured.m_hi);
    try testing.expect(measured.m_lo <= 4);
    // ONE sample at the cliff (untrusted) already floors the horizon.
    var t1 = round_cost.Table{};
    for (0..round_cost.MIN_SAMPLES) |_| {
        _ = t1.observe(3, 10000, 60.0, 4.0, true, false);
        _ = t1.observe(4, 10000, 70.0, 5.0, true, false);
    }
    _ = t1.observe(5, 10000, 102.0, 6.0, true, false);
    const src1 = Generator.MtpCostSource.init(Generator.MTP_EV_DEFAULT_COSTS, 10000, &t1);
    try testing.expectEqual(@as(u32, 4), Generator.mtpEvPlanSrc(&a, 8, src1, 8).m_hi);
    // ...and the trial stops asking for it (two-chunk plan, nothing owed).
    try testing.expect(Generator.mtpWidthTrialTarget(&t1, 10000, .{ .m_lo = 4, .m_hi = 5, .tau_ln = 0 }, 8, true) == null);
    // An unmeasured base is trialled only under a two-chunk plan with a settled base.
    const empty_t = round_cost.Table{};
    try testing.expect(Generator.mtpWidthTrialTarget(&empty_t, 1000, .{ .m_lo = 3, .m_hi = 3, .tau_ln = 0 }, 8, true) == null);
    try testing.expect(Generator.mtpWidthTrialTarget(&empty_t, 1000, .{ .m_lo = 3, .m_hi = 5, .tau_ln = 0 }, 8, false) == null);
    try testing.expectEqual(@as(u32, 3), Generator.mtpWidthTrialTarget(&empty_t, 1000, .{ .m_lo = 3, .m_hi = 5, .tau_ln = 0 }, 8, true).?);
    try testing.expect(Generator.mtpWidthTrialPeriod(&t1, 10000, 4) >= 100);
    // An inactive bucket with no active neighbour is the prior, verbatim.
    const empty = round_cost.Table{};
    const cold = Generator.MtpCostSource.init(Generator.MTP_EV_DEFAULT_COSTS, 10000, &empty);
    try testing.expect(!cold.fromTable());
    const cold_plan = Generator.mtpEvPlanSrc(&a, 8, cold, 8);
    try testing.expectEqual(prior.m_hi, cold_plan.m_hi);
    try testing.expectEqual(prior.m_lo, cold_plan.m_lo);
}

test "mtpWidthTrial: blocks per period, idempotent per round, period grows with the measured gap" {
    var wt = Generator.MtpWidthTrial{};
    var forced: u32 = 0;
    var i: u32 = 10;
    while (i < 210) : (i += 1) {
        const f = Generator.mtpWidthTrialForce(&wt, i, 8, true);
        try testing.expectEqual(f, Generator.mtpWidthTrialForce(&wt, i, 8, true)); // asked twice, same answer
        if (f) forced += 1;
    }
    // Identity checkable from the log: forced rounds == block * trials.
    try testing.expectEqual(wt.trials * round_cost.EXPLORE_BLOCK, forced);
    try testing.expect(wt.trials >= 17 and wt.trials <= 19);
    // Unmeasured next width: default period. 10% worse: 60. 21% worse: 128.
    var t = round_cost.Table{};
    try testing.expectEqual(round_cost.EXPLORE_PERIOD_COLD, Generator.mtpWidthTrialPeriod(&t, 1000, 4));
    for (0..round_cost.MIN_SAMPLES) |_| {
        _ = t.observe(4, 1000, 40.0, 4.0, true, false);
        _ = t.observe(5, 1000, 55.0, 5.0, true, false);
    }
    try testing.expectEqual(@as(u32, 60), Generator.mtpWidthTrialPeriod(&t, 1000, 4));
    // One better-looking sample is not a period: cold until trusted.
    var u = round_cost.Table{};
    for (0..round_cost.MIN_SAMPLES) |_| _ = u.observe(4, 1000, 40.0, 4.0, true, false);
    _ = u.observe(5, 1000, 45.0, 5.0, true, false);
    try testing.expectEqual(round_cost.EXPLORE_PERIOD_COLD, Generator.mtpWidthTrialPeriod(&u, 1000, 4));
    for (0..round_cost.MIN_SAMPLES) |_| _ = t.observe(6, 1000, 80.0, 6.0, true, false);
    try testing.expectEqual(@as(u32, 128), Generator.mtpWidthTrialPeriod(&t, 1000, 5));
}

test "round_cost: a simulated round loop measures every width the chooser picks and settles under the cliff" {
    // Driven as mtpRoundPlan drives it: ask the plan (prior until the
    // bucket has two widths), run that width, feed the table with a
    // synthetic machine whose round cost is flat-ish to depth 4 and cliffs
    // at 5 (M1 Pro 27B shape). Acceptance is high, so the prior alone
    // would sit at depth 6+.
    const Machine = struct {
        fn roundMs(m: u32) f32 {
            return switch (m) {
                0 => 50.0,
                1 => 56.0,
                2 => 62.0,
                3 => 68.0,
                4 => 74.0,
                else => 74.0 + 35.0 * @as(f32, @floatFromInt(m - 4)),
            };
        }
    };
    const a = [_]f32{ 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95 };
    var t = round_cost.Table{};
    var wt = Generator.MtpWidthTrial{};
    var prev: ?u32 = null;
    var prev2: ?u32 = null;
    var prev_two = false;
    var prev_two2 = false;
    var late_wide: u32 = 0;
    var streak: u32 = 0;
    var picked: [round_cost.MAX_WIDTH + 1]u32 = @splat(0);
    var m_lo_prev: u32 = 1;
    var last_m: u32 = 0;
    var i: u32 = 10;
    while (i < 600) : (i += 1) {
        const src = Generator.MtpCostSource.init(Generator.MTP_EV_DEFAULT_COSTS, 1000, &t);
        var plan = Generator.mtpEvPlanSrc(&a, 8, src, m_lo_prev + 1);
        if (plan.m_lo == m_lo_prev) streak += 1 else streak = 0;
        if (Generator.mtpWidthTrialTarget(&t, 1000, plan, 8, streak >= 2)) |target| {
            // `t` is a default (`.legacy`) table, so this loop runs the arm-once schedule it was fitted on.
            if (Generator.mtpWidthTrialForce(&wt, i, Generator.mtpWidthTrialPeriod(&t, 1000, plan.m_lo), round_cost.schedulePeriodReread(t.layout))) plan = Generator.mtpWidthTrialPlan(target);
        }
        // Run it: a two-chunk plan extends (confidence is high), so the
        // realized width is m_hi — and, as in mtpRoundEndObserve, only a
        // single-chunk round feeds the table.
        const m = plan.m_hi;
        const two_chunk = plan.m_hi > plan.m_lo;
        picked[m] += 1;
        if (i >= 200 and m >= 5) late_wide += 1;
        last_m = m;
        const tokens: f32 = Generator.mtpEvExpectedTokens(&a, m);
        const transition = prev_two or prev_two2 or (if (prev) |p| p != m else true) or (if (prev2) |p| p != m else true);
        if (!two_chunk) _ = t.observe(m, 1000, Machine.roundMs(m), tokens, true, transition);
        prev2 = prev;
        prev = m;
        prev_two2 = prev_two;
        prev_two = two_chunk;
        m_lo_prev = plan.m_lo;
    }
    // The cliff was found from ONE sample (clearly worse: never trusted,
    // never re-trialled at the cold period) and the loop settled under it.
    try testing.expect(t.rawMs(5, 0) != null);
    try testing.expect(t.clearlyWorse(5, 4, 0));
    try testing.expect(t.measuredMs(4, 0) != null);
    try testing.expect(last_m == 4);
    try testing.expect(picked[4] > 450);
    // Before the table activates the prior extends 4 -> 5 blind every round
    // (live, the regime gate throttles that shape; the sim has no gate), so
    // the bar is the settled tail: from round 200 on, width 5 appears only
    // as scheduled re-trial blocks (three in 400 rounds at this cliff).
    try testing.expect(late_wide <= 3 * round_cost.EXPLORE_BLOCK);
    try testing.expect(picked[6] + picked[7] + picked[8] <= 4); // the prior's first rounds only
    // Identity: forced rounds == 2 * trials (each block is 2 rounds).
    try testing.expect(wt.trials >= 2);
}

test "mtpEvPlanFor: DEFAULT costs carry the post-sdpa-split surface (2026-08-15 refit #4)" {
    // Pins MTP_EV_DEFAULT_COSTS to the surface measured AFTER the hd-256
    // causal sdpa split (same-session saturated forced-depth echo sweep,
    // Jundot oQ4e 27B @8K cold reps, M4 Max: T(1)=44.6 .. T(8)=142.3 →
    // floor ≈ 38.2 ms, marginals k<=4 ≈ 0.20, k5-6 ≈ 0.36, k7-8 ≈ 0.62).
    // flat_max 4: the old hi over-priced k4 at 0.34 (measured 0.24), so
    // moderate content under-drafted it. The k>=7 third region carries the
    // plain-SIMD verify-qmm register cliff — only reachable when
    // --mtp-max-depth forces past the generic cap of 6.
    const costs = Generator.MTP_EV_DEFAULT_COSTS;
    // Hot uniform 90%: base now rides the WIDER flat region (m_lo 4, was 3
    // under refit #3), extension one step into the ramp.
    const hot = [_]f32{ 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9 };
    const hot_plan = Generator.mtpEvPlanFor(&hot, 8, costs, 8);
    try testing.expectEqual(@as(u32, 4), hot_plan.m_lo);
    try testing.expectEqual(@as(u32, 5), hot_plan.m_hi);
    try testing.expectApproxEqAbs(@as(f32, -0.0943), hot_plan.tau_ln, 5e-3);
    // Marginal 75%: base stays 3, one cheap flat extension position.
    const mid = [_]f32{ 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75 };
    const mid_plan = Generator.mtpEvPlanFor(&mid, 8, costs, 8);
    try testing.expectEqual(@as(u32, 3), mid_plan.m_lo);
    try testing.expectEqual(@as(u32, 4), mid_plan.m_hi);
    try testing.expectApproxEqAbs(@as(f32, -0.786), mid_plan.tau_ln, 5e-3);
    // The oQ4e head's measured decayed chain still plans a 2-deep base.
    const oq4e = [_]f32{ 0.78, 0.50, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45 };
    const oq4e_plan = Generator.mtpEvPlanFor(&oq4e, 8, costs, 8);
    try testing.expectEqual(@as(u32, 2), oq4e_plan.m_lo);
}

test "mtpExtDryAllows: a dry streak collapses to single-chunk, cooldown expires into a fresh trial" {
    var streak: u32 = 0;
    var cooldown: u32 = 0;
    const thr = Generator.MTP_EXT_DRY_ROUNDS;
    // Considering rounds below the streak threshold stay allowed.
    var i: u32 = 0;
    while (i < thr - 1) : (i += 1) {
        try testing.expect(Generator.mtpExtDryAllows(&streak, &cooldown, thr));
        streak += 1; // caller records "considered, did not extend"
    }
    // The full dry streak trips the cooldown: consideration collapses.
    streak = thr;
    try testing.expect(!Generator.mtpExtDryAllows(&streak, &cooldown, thr));
    try testing.expectEqual(@as(u32, 0), streak);
    // Cooldown rounds stay collapsed and count down.
    var blocked: u32 = 0;
    while (!Generator.mtpExtDryAllows(&streak, &cooldown, thr)) blocked += 1;
    try testing.expectEqual(Generator.MTP_EXT_DRY_COOLDOWN - 1, blocked);
    // After the cooldown, consideration re-opens (fresh trial, streak 0).
    try testing.expectEqual(@as(u32, 0), streak);
    try testing.expect(Generator.mtpExtDryAllows(&streak, &cooldown, thr));
    // An extension firing resets the streak (caller does streak = 0) — the
    // gate never trips on workloads where extension pays.
}

test "mtpExtDryAllows: a lower cost-aware threshold trips the cooldown sooner" {
    var streak: u32 = 0;
    var cooldown: u32 = 0;
    const thr: u32 = 4; // e.g. an expensive sync throttled hard
    var i: u32 = 0;
    while (i < thr - 1) : (i += 1) {
        try testing.expect(Generator.mtpExtDryAllows(&streak, &cooldown, thr));
        streak += 1;
    }
    streak = thr;
    try testing.expect(!Generator.mtpExtDryAllows(&streak, &cooldown, thr));
    try testing.expectEqual(@as(u32, 0), streak);
    // The fresh trial still fits inside an echo stretch even at the floor.
    try testing.expect(Generator.MTP_EXT_DRY_MIN + Generator.MTP_EXT_DRY_COOLDOWN < 70);
}

test "mtpExtDryThresholdFor: cost-aware dry threshold scales inversely with the measured sync fraction" {
    const D = Generator;
    // Unmeasured (either EMA still 0) keeps the fixed dry budget — no behavior
    // change until the live cost is known.
    try testing.expectEqual(D.MTP_EXT_DRY_ROUNDS, D.mtpExtDryThresholdFor(0.0, 45.0));
    try testing.expectEqual(D.MTP_EXT_DRY_ROUNDS, D.mtpExtDryThresholdFor(2.4, 0.0));
    // A near-free sync (tiny fraction) tolerates the full dry budget.
    try testing.expectEqual(D.MTP_EXT_DRY_ROUNDS, D.mtpExtDryThresholdFor(0.2, 45.0));
    // An expensive sync (large fraction) collapses toward the floor.
    try testing.expectEqual(D.MTP_EXT_DRY_MIN, D.mtpExtDryThresholdFor(9.0, 45.0));
    // Monotonic: a costlier sync never tolerates MORE dry rounds.
    try testing.expect(D.mtpExtDryThresholdFor(4.0, 45.0) <= D.mtpExtDryThresholdFor(1.0, 45.0));
    // Always bounded — never below the floor (a zero threshold would trip
    // every round and close the horizon; the fresh-trial guarantee survives
    // any cost).
    try testing.expect(D.mtpExtDryThresholdFor(1000.0, 45.0) >= D.MTP_EXT_DRY_MIN);
    // The measured 8K operating point (sync ~2.4 ms of a ~45 ms round) lands
    // in the throttled band, well below the fixed 16.
    const at_8k = D.mtpExtDryThresholdFor(2.4, 45.0);
    try testing.expect(at_8k >= D.MTP_EXT_DRY_MIN and at_8k < D.MTP_EXT_DRY_ROUNDS);
}

test "mtpEvPlanFor: cap 1 is a plain depth-1 round" {
    const costs = Generator.MTP_EV_DEFAULT_COSTS;
    const a = [_]f32{ 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9 };
    const plan = Generator.mtpEvPlanFor(&a, 1, costs, 8);
    try testing.expectEqual(@as(u32, 1), plan.m_lo);
    try testing.expectEqual(@as(u32, 1), plan.m_hi);
}

test "mtpEvObserve: conditional EMA updates hit accepted indices, the reject index, and nothing past it" {
    var a = [_]f32{ 0.5, 0.5, 0.5, 0.5 };
    // 3 drafted, 1 accepted: index 0 saw a success, index 1 saw the reject,
    // index 2 was never conditionally reached (no observation).
    Generator.mtpEvObserve(&a, 3, 1, 0.15);
    try testing.expectApproxEqAbs(@as(f32, 0.575), a[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.425), a[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.5), a[2], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.5), a[3], 1e-5);
    // Full accept: every drafted index saw a success, none saw a reject.
    var b = [_]f32{ 0.5, 0.5, 0.5, 0.5 };
    Generator.mtpEvObserve(&b, 2, 2, 0.15);
    try testing.expectApproxEqAbs(@as(f32, 0.575), b[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.575), b[1], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.5), b[2], 1e-5);
}

test "mtpChainLogConf: sums clamped log-confidences; NaN can never pass a gate" {
    try testing.expectApproxEqAbs(@as(f32, -0.3), Generator.mtpChainLogConf(&[_]f32{ -0.1, -0.2 }), 1e-5);
    // Positive numeric noise clamps to 0 (a log-prob is never > 0).
    try testing.expectApproxEqAbs(@as(f32, -0.1), Generator.mtpChainLogConf(&[_]f32{ 0.05, -0.1 }), 1e-5);
    // NaN -> -inf: `chain >= tau_ln` is false for every finite tau.
    const nan_chain = Generator.mtpChainLogConf(&[_]f32{ -0.1, std.math.nan(f32) });
    try testing.expect(nan_chain == -std.math.inf(f32));
    try testing.expect(!(nan_chain >= @log(@as(f32, 0.05))));
}

test "mtpAdaptiveVoteFor: a switch needs BOTH measured prices past the margin" {
    const G = Generator;
    const margin = G.MTP_ADAPTIVE_MARGIN;
    try testing.expectEqual(G.MtpAdaptiveVote.serial, G.mtpAdaptiveVoteFor(12.0, 12.0, 10.0, margin));
    // Either price inside the bar keeps speculation.
    try testing.expectEqual(G.MtpAdaptiveVote.mtp, G.mtpAdaptiveVoteFor(12.0, 9.0, 10.0, margin));
    try testing.expectEqual(G.MtpAdaptiveVote.mtp, G.mtpAdaptiveVoteFor(9.0, 12.0, 10.0, margin));
    try testing.expectEqual(G.MtpAdaptiveVote.mtp, G.mtpAdaptiveVoteFor(9.0, 9.0, 10.0, margin));
    try testing.expectEqual(G.MtpAdaptiveVote.mtp, G.mtpAdaptiveVoteFor(12.0, 12.0, 11.5, margin));
    try testing.expectEqual(G.MtpAdaptiveVote.serial, G.mtpAdaptiveVoteFor(12.0, 12.0, 11.4, margin));
    try testing.expectEqual(G.MtpAdaptiveVote.mtp, G.mtpAdaptiveVoteFor(12.0, 12.0, 12.0, 0.0));

    const nan = std.math.nan(f32);
    try testing.expectEqual(G.MtpAdaptiveVote.undecided, G.mtpAdaptiveVoteFor(null, 12.0, 10.0, margin));
    try testing.expectEqual(G.MtpAdaptiveVote.undecided, G.mtpAdaptiveVoteFor(12.0, null, 10.0, margin));
    try testing.expectEqual(G.MtpAdaptiveVote.undecided, G.mtpAdaptiveVoteFor(12.0, 12.0, null, margin));
    try testing.expectEqual(G.MtpAdaptiveVote.undecided, G.mtpAdaptiveVoteFor(0.0, 12.0, 10.0, margin));
    try testing.expectEqual(G.MtpAdaptiveVote.undecided, G.mtpAdaptiveVoteFor(12.0, 0.0, 10.0, margin));
    try testing.expectEqual(G.MtpAdaptiveVote.undecided, G.mtpAdaptiveVoteFor(12.0, 12.0, 0.0, margin));
    try testing.expectEqual(G.MtpAdaptiveVote.undecided, G.mtpAdaptiveVoteFor(nan, 12.0, 10.0, margin));
    try testing.expectEqual(G.MtpAdaptiveVote.undecided, G.mtpAdaptiveVoteFor(12.0, nan, 10.0, margin));
    try testing.expectEqual(G.MtpAdaptiveVote.undecided, G.mtpAdaptiveVoteFor(12.0, 12.0, nan, margin));
    try testing.expectEqual(G.MtpAdaptiveVote.undecided, G.mtpAdaptiveVoteFor(12.0, 12.0, 10.0, nan));
}

test "mtpAdaptiveBucketOf: the decision and the re-entry resolve ONE bucket (H6 oscillation)" {
    const G = Generator;
    const kv: u32 = 40_000;
    const own = round_cost.bucketFor(kv);

    try testing.expectEqual(own, G.mtpAdaptiveBucketOf(null, kv, .long));
    try testing.expectEqual(@as(usize, own + 1), G.mtpAdaptiveBucketOf(own + 1, kv, .long));
    try testing.expect(G.mtpAdaptiveBucketOf(own + 1, kv, .long) != round_cost.bucketFor(kv));

    // Two resolvers oscillated: switch, crossing, re-enter, switch again.
    var osc = G.MtpAdaptive{};
    _ = osc.round(0, G.mtpAdaptiveBucketOf(own + 1, kv, .long), .serial, 1);
    try testing.expectEqual(G.MtpAdaptiveArm.serial, osc.arm);
    try testing.expectEqual(G.MtpAdaptiveAction.to_mtp, osc.serialTick(round_cost.bucketFor(kv)));

    var ok = G.MtpAdaptive{};
    const b = G.mtpAdaptiveBucketOf(own + 1, kv, .long);
    _ = ok.round(0, b, .serial, 1);
    var i: u32 = 0;
    while (i < 50) : (i += 1) {
        try testing.expectEqual(G.MtpAdaptiveAction.none, ok.serialTick(G.mtpAdaptiveBucketOf(own + 1, kv, .long)));
    }
    try testing.expectEqual(G.MtpAdaptiveArm.serial, ok.arm);
}

test "adaptive serial re-enters when the request crosses into a bucket nothing has measured" {
    // The pagoda request: w2 trusted in 32-64k only, the switch at 33k, then 58k serial tokens
    // to 91.8k that never re-entered: the read bucket kept resolving 64-128k back onto 32-64k.
    const G = Generator;
    var t = round_cost.Table{ .layout = .long };
    for (0..round_cost.MIN_SAMPLES) |_| _ = t.observe(2, 33_000, 36.8, 1.65, true, false);
    const kv_switch: u32 = 33_000;
    const read = G.mtpAdaptiveBucketOf(t.bucketToRead(kv_switch), kv_switch, t.layout);
    var a = G.MtpAdaptive{};
    for (0..G.MTP_ADAPTIVE_CONFIRM) |i| _ = a.round(@intCast(i), read, .serial, G.MTP_ADAPTIVE_CONFIRM);
    try testing.expectEqual(G.MtpAdaptiveArm.serial, a.arm);

    var kv: u32 = kv_switch + 1;
    while (kv < 65_536) : (kv += 61) {
        try testing.expectEqual(G.MtpAdaptiveAction.none, G.mtpAdaptiveSerialTickFor(&a, &t, kv));
    }
    try testing.expectEqual(G.MtpAdaptiveAction.to_mtp, G.mtpAdaptiveSerialTickFor(&a, &t, 65_536));
    try testing.expectEqual(G.MtpAdaptiveArm.undecided, a.arm);

    // A switch that read a neighbour (own bucket unmeasured) must not re-enter on its next tick (H6).
    var lo = round_cost.Table{ .layout = .long };
    for (0..round_cost.MIN_SAMPLES) |_| _ = lo.observe(2, 20_000, 36.8, 1.65, true, false);
    var n = G.MtpAdaptive{};
    _ = n.round(0, G.mtpAdaptiveBucketOf(lo.bucketToRead(40_000), 40_000, lo.layout), .serial, 1);
    kv = 40_001;
    while (kv < 65_536) : (kv += 61) {
        try testing.expectEqual(G.MtpAdaptiveAction.none, G.mtpAdaptiveSerialTickFor(&n, &lo, kv));
    }
    try testing.expectEqual(G.MtpAdaptiveAction.to_mtp, G.mtpAdaptiveSerialTickFor(&n, &lo, 65_536));
}

test "mtpAdaptiveRegimeMoved: a switch either way, or a crossing, drops the price window" {
    const G = Generator;
    const A = G.MtpAdaptiveArm;
    try testing.expect(!G.mtpAdaptiveRegimeMoved(null, .undecided, null, .undecided));
    try testing.expect(!G.mtpAdaptiveRegimeMoved(3, .mtp, 3, .mtp));
    try testing.expect(G.mtpAdaptiveRegimeMoved(3, A.mtp, 3, A.serial));
    try testing.expect(G.mtpAdaptiveRegimeMoved(3, A.serial, 3, A.undecided));
    try testing.expect(G.mtpAdaptiveRegimeMoved(3, A.undecided, 3, A.mtp));
    try testing.expect(G.mtpAdaptiveRegimeMoved(3, A.mtp, 4, A.mtp));
    try testing.expect(G.mtpAdaptiveRegimeMoved(null, A.mtp, 4, A.mtp));
    try testing.expect(G.mtpAdaptiveRegimeMoved(3, A.mtp, null, A.mtp));

    var a = G.MtpAdaptive{};
    var pb = a.bucket;
    var pa = a.arm;
    _ = a.round(0, 5, .serial, 1); // first vote: bucket null -> 5, arm -> serial
    try testing.expect(G.mtpAdaptiveRegimeMoved(pb, pa, a.bucket, a.arm));
    pb = a.bucket;
    pa = a.arm;
    _ = a.serialTick(6); // crossing: re-opens the decision
    try testing.expect(G.mtpAdaptiveRegimeMoved(pb, pa, a.bucket, a.arm));
    // A tick that changes nothing must not drop the window, or it can never fill.
    pb = a.bucket;
    pa = a.arm;
    _ = a.serialTick(6);
    try testing.expect(!G.mtpAdaptiveRegimeMoved(pb, pa, a.bucket, a.arm));
}

test "mtpAdaptiveKvEligible: short context is below the floor, and the floor is the default" {
    const G = Generator;
    const floor = G.MTP_ADAPTIVE_MIN_KV;
    try testing.expectEqual(@as(u32, 32768), floor);

    var reached: u32 = 0;
    var i: u32 = 0;
    while (i < 30) : (i += 1) {
        const kv: u32 = 60 + i * 7 + 200;
        if (G.mtpAdaptiveKvEligible(kv, floor)) reached += 1;
    }
    try testing.expectEqual(@as(u32, 0), reached);

    try testing.expect(!G.mtpAdaptiveKvEligible(0, floor));
    try testing.expect(!G.mtpAdaptiveKvEligible(floor - 1, floor));
    try testing.expect(!G.mtpAdaptiveKvEligible(8_200, floor));
    try testing.expect(!G.mtpAdaptiveKvEligible(16_400, floor));
    try testing.expect(G.mtpAdaptiveKvEligible(floor, floor));
    try testing.expect(G.mtpAdaptiveKvEligible(62_755, floor)); // the A/B prose prompt
    try testing.expect(G.mtpAdaptiveKvEligible(374_000, floor)); // the ladder's top rung

    try testing.expect(G.mtpAdaptiveKvEligible(1, 0));
    try testing.expect(!G.mtpAdaptiveKvEligible(62_755, 131_072));
}

test "mtpHeadPositionDrift: re-entry may only resume a head that is provably in sync" {
    const G = Generator;
    try testing.expectEqual(@as(i64, 0), G.mtpHeadPositionDrift(100, 20, 120));
    try testing.expectEqual(@as(i64, 7), G.mtpHeadPositionDrift(100, 20, 127));
    try testing.expectEqual(@as(i64, -3), G.mtpHeadPositionDrift(100, 20, 117));
    try testing.expectEqual(@as(i64, 0), G.mtpHeadPositionDrift(100, 0, 999));
    try testing.expectEqual(@as(i64, 0), G.mtpHeadPositionDrift(0, 0, 0));

    var a = G.MtpAdaptive{};
    _ = a.round(0, 3, .serial, 1);
    a.declineReentry();
    try testing.expect(a.reentry_declined);
    try testing.expectEqual(G.MtpAdaptiveArm.serial, a.arm);
    try testing.expectEqual(G.MtpAdaptiveAction.none, a.serialTick(4)); // crossing
    var i: u32 = 0;
    while (i < 2000) : (i += 1) try testing.expectEqual(G.MtpAdaptiveAction.none, a.serialTick(3));
}

test "mtpSerialProbeArm: bounded RETRIES per bucket, and none once the cell is trusted" {
    const G = Generator;
    var t = round_cost.Table{};
    const b = round_cost.bucketFor(20_000);

    var fired: u32 = 0;
    var i: u32 = 0;
    while (i < 50) : (i += 1) {
        if (G.mtpSerialProbeArm(&t, b, true, true, true, true) != null) fired += 1;
    }
    try testing.expectEqual(@as(u32, round_cost.MAX_SERIAL_PROBES), fired);
    try testing.expect(round_cost.MAX_SERIAL_PROBES > 1); // else it is the old flag

    var u = round_cost.Table{};
    const kv: u32 = 20_000;
    try testing.expect(G.mtpSerialProbeArm(&u, b, true, true, true, true) != null);
    var k: u32 = 0;
    while (k < round_cost.MIN_SAMPLES) : (k += 1) _ = u.observeSerial(kv, 16.0, true, false);
    try testing.expect(u.serialMsPerTok(b) != null);
    try testing.expect(G.mtpSerialProbeArm(&u, b, true, true, true, true) == null);

    var v = round_cost.Table{};
    try testing.expect(G.mtpSerialProbeArm(&v, round_cost.bucketFor(1000), true, true, true, true) != null);
    try testing.expect(G.mtpSerialProbeArm(&v, b, true, true, true, true) != null);

    var w = round_cost.Table{};
    try testing.expect(G.mtpSerialProbeArm(&w, b, false, true, true, true) == null);
    try testing.expect(G.mtpSerialProbeArm(&w, b, true, false, true, true) == null);
    try testing.expect(G.mtpSerialProbeArm(&w, b, true, true, false, true) == null);
    try testing.expect(G.mtpSerialProbeArm(&w, round_cost.N_BUCKETS, true, true, true, true) == null);
    try testing.expectEqual(@as(u8, 0), w.serial_probes[b]);
}

test "S21: a to_serial on a per-request head is not sticky; the release stays a one-shot" {
    const G = Generator;
    // The qwen4 head keeps per-request state now (`Qwen4MtpState`), so a serial switch
    // no longer has to give the module back.
    try testing.expect(!G.stickyOnSerialSwitch(MtpHeadRef{ .qwen4 = undefined }));

    var a = G.MtpAdaptive{};
    try testing.expectEqual(G.MtpAdaptiveAction.to_serial, a.round(0, 3, .serial, 1));
    a.stickSerial();
    try testing.expect(a.sticky_serial);
    try testing.expectEqual(G.MtpAdaptiveArm.serial, a.arm);

    try testing.expectEqual(G.MtpAdaptiveAction.none, a.serialTick(4));
    try testing.expectEqual(G.MtpAdaptiveAction.none, a.serialTick(5));
    var i: u32 = 0;
    while (i < 1024) : (i += 1) {
        try testing.expectEqual(G.MtpAdaptiveAction.none, a.serialTick(3));
    }
    try testing.expectEqual(G.MtpAdaptiveAction.none, a.round(1, 3, .mtp, 1));
    try testing.expectEqual(G.MtpAdaptiveAction.none, a.round(2, 4, .serial, 1));
    try testing.expectEqual(G.MtpAdaptiveArm.serial, a.arm);

    try testing.expect(G.stickyReleaseNow(true, false));
    try testing.expect(!G.stickyReleaseNow(true, true));
    try testing.expect(!G.stickyReleaseNow(false, false));
    try testing.expect(!G.stickyReleaseNow(false, true));
}

test "S21: a head that is not module-owned keeps its re-arm, and nothing about it changes" {
    const G = Generator;
    try testing.expect(!G.stickyOnSerialSwitch(MtpHeadRef{ .qwen4 = undefined }));
    try testing.expect(!G.stickyOnSerialSwitch(null));

    var b = G.MtpAdaptive{};
    try testing.expectEqual(G.MtpAdaptiveAction.to_serial, b.round(0, 3, .serial, 1));
    try testing.expect(!b.sticky_serial);
    try testing.expectEqual(G.MtpAdaptiveAction.to_mtp, b.serialTick(4));
    try testing.expectEqual(G.MtpAdaptiveArm.undecided, b.arm);

    try testing.expect(!Generator.mtpAdaptiveSerialEnabledFromEnv("0"));
}

test "mtpAdaptiveSerialEnabledFromEnv: the mechanism is on unless the lever says 0" {
    try testing.expect(Generator.mtpAdaptiveSerialEnabledFromEnv(null));
    try testing.expect(Generator.mtpAdaptiveSerialEnabledFromEnv(""));
    try testing.expect(Generator.mtpAdaptiveSerialEnabledFromEnv("1"));
    try testing.expect(!Generator.mtpAdaptiveSerialEnabledFromEnv("0"));
}

test "characterization: a sidecar (legacy-layout) boot plans exactly as 26.9.1 did" {
    // A sidecar (legacy-layout) boot must plan exactly as before: a store-version bump made
    // every boot cold and a cold table runs every round as a two-chunk round (-25% decode on
    // the 27B). Every expectation below was recorded by running the 26.9.1 planner.
    const G = Generator;
    const a = [_]f32{ 0.85, 0.78, 0.70, 0.60, 0.50, 0.42, 0.35, 0.30 };
    const kv: u32 = 8192; // one 27B rung; ~3.0-3.5 tok/round at m_lo 2-3
    const costs = G.MTP_EV_DEFAULT_COSTS;
    const cap: u32 = 6;

    const legacy_empty = round_cost.Table{ .layout = .legacy };
    try testing.expectEqual(@as(usize, 3), legacy_empty.bucketOf(kv));
    try testing.expectEqual(@as(usize, 3), round_cost.bucketFor(kv));

    // The warm table, restored from the file 26.9.1 wrote.
    const rc1_body = "rc1\n2 3 44.0000 2.7000 3\n3 3 64.0000 3.0500 3\n4 3 92.0000 3.2000 3\n";
    const warm = round_cost.parse(rc1_body, .legacy) orelse return error.TestUnexpectedResult;
    const warm_src = G.MtpCostSource.init(costs, kv, &warm);
    try testing.expect(warm_src.fromTable());

    // Warm: a measured marginal the position cannot repay closes the valve; every round is single-chunk.
    var prev: u32 = 1;
    for (0..8) |_| {
        const p = G.mtpEvPlanSrc(&a, cap, warm_src, prev + 1);
        try testing.expectEqual(@as(u32, 2), p.m_lo);
        try testing.expectEqual(@as(u32, 2), p.m_hi); // no extension
        try testing.expectEqual(@as(f32, 0.0), p.tau_ln);
        prev = p.m_lo;
    }

    // The same request on a cold table: every round extends.
    const cold_src = G.MtpCostSource.init(costs, kv, null);
    try testing.expect(!cold_src.fromTable());
    const cold0 = G.mtpEvPlanSrc(&a, cap, cold_src, 2);
    try testing.expectEqual(@as(u32, 2), cold0.m_lo);
    try testing.expectEqual(@as(u32, 4), cold0.m_hi);
    try testing.expectApproxEqAbs(@as(f32, -0.444614), cold0.tau_ln, 1e-5);
    prev = cold0.m_lo;
    for (0..7) |_| {
        const p = G.mtpEvPlanSrc(&a, cap, cold_src, prev + 1);
        try testing.expectEqual(@as(u32, 3), p.m_lo);
        try testing.expectEqual(@as(u32, 4), p.m_hi); // two-chunk, every round
        try testing.expectApproxEqAbs(@as(f32, -0.477666), p.tau_ln, 1e-5);
        prev = p.m_lo;
    }

    // A warm table whose cells make one extension position pay: extend by exactly one.
    var shallow = round_cost.Table{ .layout = .legacy };
    for (0..round_cost.MIN_SAMPLES) |_| {
        _ = shallow.observe(2, kv, 44.0, 2.70, true, false);
        _ = shallow.observe(3, kv, 51.0, 3.20, true, false);
        _ = shallow.observe(4, kv, 62.0, 3.50, true, false);
    }
    const shallow_src = G.MtpCostSource.init(costs, kv, &shallow);
    prev = 1;
    for (0..8) |_| {
        const p = G.mtpEvPlanSrc(&a, cap, shallow_src, prev + 1);
        try testing.expectEqual(@as(u32, 2), p.m_lo);
        try testing.expectEqual(@as(u32, 3), p.m_hi);
        try testing.expectApproxEqAbs(@as(f32, -0.488353), p.tau_ln, 1e-5);
        prev = p.m_lo;
    }

    // The width is a table-state property too: warm holds the base at 2, cold climbs to 3.
    var wide = round_cost.Table{ .layout = .legacy };
    for (0..round_cost.MIN_SAMPLES) |_| {
        _ = wide.observe(2, kv, 44.0, 2.70, true, false);
        _ = wide.observe(3, kv, 51.0, 3.20, true, false);
        _ = wide.observe(4, kv, 55.0, 4.00, true, false);
        _ = wide.observe(5, kv, 75.0, 4.20, true, false);
    }
    const wide_src = G.MtpCostSource.init(costs, kv, &wide);
    prev = 1;
    for (0..8) |_| {
        const p = G.mtpEvPlanSrc(&a, cap, wide_src, prev + 1);
        try testing.expectEqual(@as(u32, 2), p.m_lo);
        try testing.expectEqual(@as(u32, 4), p.m_hi);
        try testing.expectApproxEqAbs(@as(f32, -0.506371), p.tau_ln, 1e-5);
        prev = p.m_lo;
    }
    prev = 1;
    var cold_final: u32 = 0;
    for (0..8) |_| {
        const p = G.mtpEvPlanSrc(&a, cap, cold_src, prev + 1);
        cold_final = p.m_lo;
        prev = p.m_lo;
    }
    try testing.expectEqual(@as(u32, 3), cold_final);
    try testing.expect(cold_final != G.mtpEvPlanSrc(&a, cap, wide_src, 8).m_lo);

    try testing.expectEqual(
        @as(?u32, 3),
        G.mtpWidthTrialTarget(&warm, kv, .{ .m_lo = 2, .m_hi = 2, .tau_ln = 0 }, cap, true),
    );
}

test "characterization: a sidecar boot's width-trial SCHEDULE re-reads its period like qwen4" {
    // Which ROUND carries a trial: the previous arm-once schedule kept a
    // neighbour bucket's date; the re-read is every layout's.
    const G = Generator;
    const legacy = round_cost.Table{ .layout = .legacy };
    const long = round_cost.Table{ .layout = .long };
    try testing.expect(round_cost.schedulePeriodReread(legacy.layout));
    try testing.expect(round_cost.schedulePeriodReread(long.layout));

    const run = struct {
        fn f(reread: bool) ?u32 {
            var wt = G.MtpWidthTrial{};
            var i: u32 = 12;
            while (i < 200) : (i += 1) {
                const period: u32 = if (i < 18) 124 else round_cost.EXPLORE_PERIOD_COLD;
                if (G.mtpWidthTrialForce(&wt, i, period, reread)) return i;
            }
            return null;
        }
    }.f;
    try testing.expectEqual(@as(?u32, 136), run(false)); // the arm-once schedule
    try testing.expectEqual(@as(?u32, 20), run(true)); // shipped: every layout
}

test "mtpAdaptiveHeadEligible: the qwen4 head and MiMo's heads switch to serial, DSpark does not" {
    const G = Generator;
    try testing.expect(G.mtpAdaptiveHeadEligible(.qwen4));
    try testing.expect(G.mtpAdaptiveHeadEligible(.mimo));
    try testing.expect(!G.mtpAdaptiveHeadEligible(.dsv41));
}

test "mtpAdaptiveMinKvFor: MiMo's switch starts at 64k, qwen4's at 32k" {
    const G = Generator;
    try testing.expectEqual(@as(u32, 65536), G.mtpAdaptiveMinKvFor(true));
    try testing.expectEqual(G.MTP_ADAPTIVE_MIN_KV, G.mtpAdaptiveMinKvFor(false));
}

test "mtpAdaptiveModelEligible: the serial row and its price window are the module head's, not every MTP model's" {
    const G = Generator;
    try testing.expect(!G.mtpAdaptiveModelEligible(true, false));
    try testing.expect(G.mtpAdaptiveModelEligible(true, true));
    // `--no-mtp` on the calibrated arch still declines: the head's weights load with the trunk.
    try testing.expect(!G.mtpAdaptiveModelEligible(false, true));
    try testing.expect(!G.mtpAdaptiveModelEligible(false, false));
}

test "mtpSerialProbeUseful: a probe buys the LAST missing input, never the first" {
    const G = Generator;
    try testing.expect(!G.mtpSerialProbeUseful(null));
    try testing.expect(G.mtpSerialProbeUseful(12.5));

    var t = round_cost.Table{ .layout = .long };
    const b = round_cost.bucketFor(20_000);
    try testing.expect(G.mtpSerialProbeArm(&t, b, true, true, true, G.mtpSerialProbeUseful(null)) == null);
    try testing.expectEqual(@as(u8, 0), t.serial_probes[b]);
    try testing.expectEqual(@as(?usize, b), G.mtpSerialProbeArm(&t, b, true, true, true, G.mtpSerialProbeUseful(12.5)));
    try testing.expectEqual(@as(u8, 1), t.serial_probes[b]);
}

test "selected depth bypasses legacy planning and keeps acceptance private" {
    const previous = group_planner.enabled_override;
    group_planner.enabled_override = true;
    defer group_planner.enabled_override = previous;
    var gen: Generator = undefined;
    gen.mtp_planner_width = 2;
    gen.mtp_planner_owned = true;
    gen.mtp_planner_probe = true;
    gen.mtp_planner_probes = 0;
    gen.mtp_planner_recovery = .{};
    gen.mtp_planner_recovering = false;
    gen.completion_tokens = 10;
    gen.mtp_planner_spec_rounds = 0;
    gen.mtp_planner_history = .{};
    gen.mtp_hidden_stale = true;
    const plan = gen.mtpRoundPlan();
    try testing.expectEqual(@as(u32, 2), plan.m_lo);
    try testing.expectEqual(plan.m_lo, plan.m_hi);
    try testing.expect(!plan.width_trial);
    gen.mtpPlannerObserve(2, 1);
    try testing.expectEqual(@as(u16, 1), gen.mtp_planner_history.accepted[0]);
    try testing.expectEqual(@as(u16, 0), gen.mtp_planner_history.accepted[1]);
    try testing.expectEqual(@as(u8, 1), gen.mtp_planner_probes);
    try testing.expect(!gen.mtp_hidden_stale);
    gen.mtp_planner_recovering = true;
    gen.completion_tokens = 12;
    gen.mtpPlannerObserve(2, 2);
    try testing.expectEqual(@as(u8, 1), gen.mtp_planner_probes);
    try testing.expectEqual(@as(u8, 1), gen.mtp_planner_recovery.rounds);
    try testing.expectEqual(@as(u32, 12), gen.mtp_planner_recovery.last_spec_tokens);
}

test "MTP depth bounds: an EV plan outside min..max is clamped, one inside is untouched" {
    const wide = Generator.MtpRoundPlan{ .m_lo = 3, .m_hi = 7, .tau_ln = -1.5 };
    const low = Generator.mtpPlanInBounds(wide, 5, 8);
    try testing.expectEqual(@as(u32, 5), low.m_lo);
    try testing.expectEqual(@as(u32, 7), low.m_hi);
    try testing.expectEqual(@as(f32, -1.5), low.tau_ln);

    const high = Generator.mtpPlanInBounds(wide, 1, 2);
    try testing.expectEqual(@as(u32, 2), high.m_lo);
    try testing.expectEqual(@as(u32, 2), high.m_hi);
    // The extension collapsed: no confidence read, as in any single-chunk plan.
    try testing.expectEqual(@as(f32, 0.0), high.tau_ln);

    const pinned = Generator.mtpPlanInBounds(wide, 4, 4);
    try testing.expectEqual(@as(u32, 4), pinned.m_lo);
    try testing.expectEqual(@as(u32, 4), pinned.m_hi);

    const inside = Generator.mtpPlanInBounds(wide, 1, 8);
    try testing.expectEqual(wide, inside);
}

test "MTP depth bounds: a min above the default cap lifts both caps, an explicit max leaves them" {
    const before = depth_bounds.active;
    defer depth_bounds.active = before;
    depth_bounds.active = .{ .min = 7 };
    try testing.expectEqual(@as(u32, 7), Generator.mtpDepthCapForProfile(0, true, .generic));
    try testing.expectEqual(@as(u32, 7), Generator.mtpDepthCapFree(0));
    depth_bounds.active = .{ .min = 2, .max = 4 };
    try testing.expectEqual(@as(u32, 4), Generator.mtpDepthCapForProfile(4, true, .generic));
    try testing.expectEqual(@as(u32, 4), Generator.mtpDepthCapFree(4));
}
