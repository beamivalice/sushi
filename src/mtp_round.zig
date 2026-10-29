const std = @import("std");
const depth_bounds = @import("mtp_depth_bounds.zig");
const expectError = @import("test_expect.zig").expectError;
const generate_mod = @import("generate.zig");
const mtp_policy = @import("mtp_policy.zig");
const group_planner = @import("mtp_group_planner.zig");
const io_util = @import("io_util.zig");
const log = @import("log.zig");
const mimo_mtp = @import("mimo2/mimo_mtp.zig");
const mlx = @import("mlx.zig");
const model_mod = @import("model.zig");
const mtp_acceptance = @import("mtp_acceptance.zig");
const mtp_lookup = @import("mtp_lookup.zig");
const mtp_mod = @import("mtp.zig");
const mtp_plan = @import("mtp_plan.zig");
const round_cost = @import("round_cost.zig");
const sampler = @import("sampler.zig");
const transformer_mod = @import("transformer.zig");
const CommittedArgmax = generate_mod.CommittedArgmax;
const DrafterStepResult = Generator.DrafterStepResult;
const ForwardCtx = transformer_mod.ForwardCtx;
const Generator = generate_mod.Generator;
const MTP_EV_WARMUP_ROUNDS = mtp_policy.MTP_EV_WARMUP_ROUNDS;
const MtpCostSource = mtp_policy.MtpCostSource;
const MtpRoundPlan = mtp_policy.MtpRoundPlan;
const SubLap = mtp_policy.SubLap;
const Transformer = transformer_mod.Transformer;
const isEosId = generate_mod.isEosId;
const mtpAdaptiveEnabled = Generator.mtpAdaptiveEnabled;
const mtpAdaptiveSerialEnabled = Generator.mtpAdaptiveSerialEnabled;
const mtpAdaptiveSerialTickFor = Generator.mtpAdaptiveSerialTickFor;
const mtpCostTableEnabled = Generator.mtpCostTableEnabled;
const mtpEmaMs = Generator.mtpEmaMs;
const mtpPinnedDepth = Generator.mtpPinnedDepth;
const mtpTraceEnabled = Generator.mtpTraceEnabled;
const mtpVerifyDraftsMax = Generator.mtpVerifyDraftsMax;
const testing = std.testing;
const verifyArgmax = generate_mod.verifyArgmax;
const verifyRows2d = generate_mod.verifyRows2d;
const tokenizer_mod = @import("tokenizer.zig");
const Tokenizer = tokenizer_mod.Tokenizer;

var mtp_trace_sync_env: ?bool = null;

/// Lazy log-confidence of one MTP draft: `logits[draft] − logsumexp(logits)`
/// = log p_head(draft). Two vocab reductions on the head's own (draft-head)
/// logits — must be built BEFORE the caller frees the logits handle (lazy
/// graphs hold their inputs internally). Returns a `[1]`-shaped lazy array.
fn draftConfidenceGraph(logits: mlx.mlx_array, draft_id: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var lse = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(lse);
    try mlx.check(mlx.mlx_logsumexp_axis(&lse, logits, -1, false, s));
    var taken = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(taken);
    try mlx.check(mlx.mlx_take_axis(&taken, logits, draft_id, -1, s));
    const flat = [_]c_int{1};
    var t_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(t_flat);
    try mlx.check(mlx.mlx_reshape(&t_flat, taken, &flat, 1, s));
    var l_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(l_flat);
    try mlx.check(mlx.mlx_reshape(&l_flat, lse, &flat, 1, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_subtract(&out, t_flat, l_flat, s));
    return out;
}

/// The chunk-A boundary sync: ONE bounded GPU round-trip that realizes the
/// chunk's draft ids (needed on the CPU later anyway) plus their
/// confidences, and returns the chain log-confidence
/// `Σ min(0, ln p_head(draft_i))` for the extension gate.
fn readChainConfidence(draft_arrs: []const mlx.mlx_array, conf_arrs: []const mlx.mlx_array, s: mlx.mlx_stream) !f32 {
    var conf_vec = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(conf_vec);
    {
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        for (conf_arrs) |arr| _ = mlx.mlx_vector_array_append_value(vec, arr);
        var cat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cat);
        try mlx.check(mlx.mlx_concatenate_axis(&cat, vec, 0, s));
        try mlx.check(mlx.mlx_astype(&conf_vec, cat, .float32, s));
    }
    {
        const eval_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(eval_vec);
        for (draft_arrs) |arr| _ = mlx.mlx_vector_array_append_value(eval_vec, arr);
        _ = mlx.mlx_vector_array_append_value(eval_vec, conf_vec);
        try mlx.check(mlx.mlx_async_eval(eval_vec));
    }
    try mlx.check(mlx.mlx_array_eval(conf_vec));
    const data = mlx.mlx_array_data_float32(conf_vec) orelse return error.MlxArrayDataNull;
    return Generator.mtpChainLogConf(data[0..conf_arrs.len]);
}

/// Round origin of the MTP head cache: with a pending stash the cache
/// still holds the PREVIOUS round's draft tail (its step is stale), so
/// the committed length is the stash origin plus the entries the stash
/// itself will append; without one, the cache is fully committed.
pub fn mtpRoundOff0(stash: ?MtpHistStash, cache_step: usize) usize {
    if (stash) |st| return st.off0 + st.n;
    return cache_step;
}

/// Committed-history length actually IN the head cache at rest — what a
/// prefix-cache commit may snapshot. Everything past it is speculative:
/// a pending stash's entries are NOT in the cache yet (the cache past
/// `stash.off0` is the producing round's stale draft tail), and a built
/// cross-round pre-draft has already appended NEXT-round draft entries
/// past ITS `off0`. The min over both boundaries is always safe.
pub fn mtpCommittedLen(cache_step: usize, pre_draft_off0: ?usize, stash_off0: ?usize) usize {
    var committed = cache_step;
    if (pre_draft_off0) |o| committed = @min(committed, o);
    if (stash_off0) |o| committed = @min(committed, o);
    return committed;
}

/// `mtpCommittedLen` over this Generator's live state; 0 when MTP is
/// not active.
pub fn mtpCommittedHistoryLen(self: *const Generator) usize {
    if (self.mtp_cache == null) return 0;
    return mtpCommittedLen(
        self.mtp_cache.?.step(),
        if (self.mtp_pre_draft) |*pd| pd.off0 else null,
        if (self.mtp_hist_stash) |*st| st.off0 else null,
    );
}

pub fn mtpMropeContext(self: *const Generator) ?mtp_mod.MropeContext {
    const positions = self.ctx.mrope_pos orelse return null;
    return .{
        .pos = positions,
        .total = self.ctx.mrope_total,
        .delta = self.ctx.mrope_delta,
        .base = self.mtp_position_base,
    };
}

/// Allocate one request-local categorical draw index while the lazy MTP
/// graph is built. The same counter covers draft and correction samples,
/// including cross-round pre-drafts, so no two keyed draws reuse a key.
/// An argmax depth draws nothing.
pub fn mtpSamplingDraw(self: *Generator, params: sampler.SamplingParams) sampler.SamplingParams {
    if (sampler.isGreedyTemperature(params.temperature)) return params;
    var draw = params;
    draw.draw = self.sampling.draw;
    self.sampling.draw +%= 1;
    return draw;
}

/// Allocate an empty draft chain for `plan` (nothing built yet).
pub fn mtpChainInit(self: *Generator, allocator: std.mem.Allocator, plan: MtpRoundPlan, t1: u32) !MtpPreDraft {
    const consider_ext = plan.m_hi > plan.m_lo;
    const sharp_drafts = !sampler.isGreedyTemperature(self.mtpDraftSampling(0).temperature);
    const drafts = try allocator.alloc(u32, plan.m_hi);
    errdefer allocator.free(drafts);
    const draft_arrs = try allocator.alloc(mlx.mlx_array, plan.m_hi);
    errdefer allocator.free(draft_arrs);
    const conf_arrs: ?[]mlx.mlx_array = if (consider_ext) try allocator.alloc(mlx.mlx_array, plan.m_lo) else null;
    errdefer if (conf_arrs) |slots| allocator.free(slots);
    const q_probs: ?[]mlx.mlx_array = if (sharp_drafts) try allocator.alloc(mlx.mlx_array, plan.m_hi) else null;
    const t1_i32: i32 = @intCast(t1);
    const t1_shape = [_]c_int{1};
    return .{
        .plan = plan,
        .off0 = mtpRoundOff0(self.mtp_hist_stash, self.mtp_cache.?.step()),
        .head_input_rows = @intCast(1 + if (self.mtp_hist_stash) |stash| stash.n else @as(usize, 0)),
        .t1 = t1,
        .t1_arr = mlx.mlx_array_new_data(&t1_i32, &t1_shape, 1, .int32),
        .drafts = drafts,
        .draft_arrs = draft_arrs,
        .n_drafted = 0,
        .conf_arrs = conf_arrs,
        .n_conf = 0,
        .q_probs = q_probs,
        .n_qp = 0,
        .h_chain = null,
        .m = plan.m_lo,
    };
}

/// A lookup draft is a point mass copied from the context, so it takes exact acceptance
/// (keep it with probability p) whatever the MTP mode: a lossy rule that accepts any
/// plausible token outright makes copying the context certain, and the output loops.
pub fn acceptGraphFor(self: *const Generator, lookup: bool) MtpGraphFn {
    return if (lookup) mtpBatchedExactGraph else self.mtp_accept_graph;
}

pub fn acceptPrefixFor(self: *const Generator, lookup: bool) mtp_acceptance.PrefixFn {
    return if (lookup) mtp_acceptance.exactPrefix else self.mtp_accept_prefix;
}

/// May a lookup stand in for this request's next MTP chain? Not for a batched head or a
/// planner-owned round, which builds every chain as a group.
pub fn mtpLookupAllowed(self: *const Generator) bool {
    if (generate_mod.mtp_lookup_disabled) return false;
    if (self.mtp == null or self.mtp_batch_head) return false;
    return !(group_planner.enabled() and self.mtp_planner_owned);
}

/// The width the lookup gate prices the MTP chain at: the one its rounds have been drafting,
/// inside this plan. A two-chunk plan's rounds extend past `m_lo`, and their acceptance with them.
pub fn mtpLookupPriceWidth(plan: MtpRoundPlan, drafted_ema: f32) u32 {
    const drafted: u32 = @intFromFloat(@round(std.math.clamp(drafted_ema, 0, @as(f32, @floatFromInt(mtp_mod.MAX_DEPTH)))));
    return std.math.clamp(drafted, plan.m_lo, @max(plan.m_lo, plan.m_hi));
}

/// A lookup's drafts are verify rows too, so MiMo's stop at its decode-row budget.
pub fn mtpLookupDraftCap(is_mimo: bool) u32 {
    return @min(mtp_lookup.MAX_DRAFT_STRONG, mtpVerifyDraftsMax(is_mimo));
}

/// End a round that stays out of the cost table. The regime clock runs from
/// the last observed round, so an unobserved round must end the interval too.
pub fn mtpRoundUntimed(self: *Generator) void {
    if (self.mtp_regime_clock) |*c| c.reset();
    self.mtp_serial_clock = null;
    self.mtp_serial_warm = 0;
}

/// Price a lookup round into the table's lookup row: the wall since the previous round
/// ended, the quantity the width grid stores. No sample before the regime clock runs.
pub fn mtpLookupObserve(self: *Generator, drafts: u32, tokens: u32) void {
    const c = if (self.mtp_regime_clock) |*c| c else return;
    const ms = @as(f32, @floatFromInt(c.read())) / @as(f32, std.time.ns_per_ms);
    _ = self.xfm.round_cost.observeLookup(drafts, self.mtpKvLen(), ms, @floatFromInt(tokens), self.spec_cost_solo);
}

/// Both rounds in the planner's EV cost units: the MTP chain at `width` as the planner
/// prices it, a lookup of k drafts from the measured lookup row, else as the MTP round
/// at k without its k head steps (the same k + 1 verify rows).
pub fn mtpLookupCostsFor(src: MtpCostSource, width: u32) mtp_lookup.Costs {
    var c = mtp_lookup.Costs{ .mtp = src.roundCost(width, false), .lookup = undefined };
    for (&c.lookup, 0..) |*x, ki| {
        const k: u32 = @intCast(ki);
        const measured: ?f32 = if (src.fromTable()) src.table.?.lookupMs(k, src.bucket) else null;
        x.* = if (measured) |ms| ms * src.scale else src.roundCost(k, false) - @as(f32, @floatFromInt(k)) * src.costs.draft;
    }
    return c;
}

/// Count a round's accepted drafts toward its kind and that kind's EMA.
pub fn mtpRoundAcceptObserve(self: *Generator, lookup: bool, drafted: u32, accepted: u32) void {
    if (lookup) {
        self.mtp_lookup_streak = accepted == drafted;
        self.mtp_lookup_accepted += accepted;
        self.mtp_lookup_ema = mtp_lookup.emaStep(self.mtp_lookup_ema, accepted);
    } else {
        self.mtp_accepted_tokens += accepted;
        self.mtp_round_ema = mtp_lookup.emaStep(self.mtp_round_ema, accepted);
        self.mtp_round_drafted_ema = mtp_lookup.emaStep(self.mtp_round_drafted_ema, drafted);
        self.mtp_lookup_ema = mtp_lookup.driftStep(self.mtp_lookup_ema);
    }
}

/// The committed-token index, synced to `generated_ids`.
pub fn mtpLookupIndex(self: *Generator, allocator: std.mem.Allocator) !*mtp_lookup.Index {
    const lines = LookupLines{ .tok = self.tok };
    if (self.mtp_lookup_idx == null) {
        var idx = mtp_lookup.Index.init(allocator);
        errdefer idx.deinit();
        try idx.extend(self.prompt_ids_owned, lines);
        self.mtp_lookup_idx = idx;
        self.mtp_lookup_gen = 0;
    }
    const gen = self.generated_ids.items;
    try self.mtp_lookup_idx.?.extend(gen[self.mtp_lookup_gen..], lines);
    self.mtp_lookup_gen = gen.len;
    return &self.mtp_lookup_idx.?;
}

/// A chain of the context's continuation after `t1`, or null when the gate
/// prefers the MTP head.
pub fn mtpLookupChain(self: *Generator, allocator: std.mem.Allocator, plan: MtpRoundPlan, t1: u32) !?MtpPreDraft {
    if (!self.mtpLookupAllowed()) return null;
    const idx = try self.mtpLookupIndex(allocator);
    const remaining: u32 = @intCast(self.max_tokens -| self.completion_tokens -| 1);
    const lookup_cap = depth_bounds.lookupCap(mtpLookupDraftCap(self.xfm.config.isMimo()), depth_bounds.active);
    const got = idx.match(t1, lookup_cap);
    const kv = self.mtpKvLen();
    const src = MtpCostSource.init(self.mtp_ev_costs, kv, if (mtpCostTableEnabled()) &self.xfm.round_cost else null);
    const k = mtp_lookup.gate(got, remaining, self.mtp_lookup_ema, self.mtp_round_ema, self.mtp_lookup_streak, mtpLookupCostsFor(src, mtpLookupPriceWidth(plan, self.mtp_round_drafted_ema)));
    if (k == 0 or k < depth_bounds.floorFor(depth_bounds.active, lookup_cap)) return null;
    const Once = struct {
        var logged = false;
    };
    if (!Once.logged) {
        Once.logged = true;
        log.info("[mtp] prompt-lookup drafts engaged: k={d} suffix={d}\n", .{ k, got.?.suffix });
    }
    // No MTP chain consumes the stash this round; left pending, it would grow
    // across lookup rounds into one oversized head forward.
    try self.mtpApplyStash();
    const drafts = try allocator.alloc(u32, k);
    errdefer allocator.free(drafts);
    @memcpy(drafts, got.?.draft[0..k]);
    const draft_arrs = try allocator.alloc(mlx.mlx_array, k);
    errdefer allocator.free(draft_arrs);
    const one = [_]c_int{1};
    for (drafts, draft_arrs) |d, *arr| {
        const v: i32 = @intCast(d);
        arr.* = mlx.mlx_array_new_data(&v, &one, 1, .int32);
    }
    const t1_i32: i32 = @intCast(t1);
    var lplan = plan;
    lplan.m_lo = k;
    lplan.m_hi = k;
    lplan.width_trial = false;
    return .{
        .plan = lplan,
        .off0 = mtpRoundOff0(self.mtp_hist_stash, self.mtp_cache.?.step()),
        .head_input_rows = 0,
        .t1 = t1,
        .t1_arr = mlx.mlx_array_new_data(&t1_i32, &one, 1, .int32),
        .drafts = drafts,
        .draft_arrs = draft_arrs,
        .n_drafted = k,
        .conf_arrs = null,
        .n_conf = 0,
        .q_probs = null,
        .n_qp = 0,
        .h_chain = null,
        .m = k,
        .lookup = true,
    };
}

/// Build draft steps [from..to) of `chain` — graph construction only, no
/// sync; the caller dispatches. Each step's sampled token ([1] lazy
/// array) feeds the next step's embedding lookup; the MTP post-norm
/// hidden chains as the next h_prev. Step 0 merges the deferred history
/// append (consumes `self.mtp_hist_stash`); a chunk-B call resumes from
/// `chain.h_chain`.
pub fn mtpChainBuild(self: *Generator, chain: *MtpPreDraft, from: u32, to: u32) !void {
    const xfm = self.xfm;
    const s = xfm.s;
    const head = self.mtp.?;
    const mc = &self.mtp_cache.?;
    const mtp_mrope_ctx = self.mtpMropeContext();
    const rerank_drafts = head.canRerankDrafts();
    std.debug.assert(chain.n_drafted == from);
    var i: u32 = from;
    while (i < to) : (i += 1) {
        const draft_sampling = self.mtpDraftSampling(i);
        const h_prev_arg: mlx.mlx_array = if (chain.h_chain) |h| h else self.last_hidden;
        const prev_tok_arr: mlx.mlx_array = if (i == 0) chain.t1_arr else chain.draft_arrs[i - 1];
        // Rerank drafts skip the head's own logits projection entirely:
        // the token comes off the coarse readout's re-scored shortlist,
        // greedily (argmax) or sampled from a q over those 32 rows.
        const path = mtpDraftStepPath(
            rerank_drafts,
            !sampler.isGreedyTemperature(draft_sampling.temperature),
            chain.conf_arrs != null and i < chain.plan.m_lo,
        );
        const use_rerank = path != .full_logits;
        // `want_logits: bool` could not tell a rerank draft from a history
        // append: both skip the vocab projection, but the draft still needs
        // the vector that projection would have consumed.
        const want: mtp_mod.StepWant = if (use_rerank) .mixed else .logits;
        var step_out = if (i == 0 and self.mtp_hist_stash != null) blk: {
            // Deferred history append (stashed at the END of the
            // previous round, Phase 5a) merged into this chain's first
            // draft: ONE (n+1)-row head forward appends the
            // committed-history entries AND the first draft entry,
            // replacing the old per-round appendHistory forward. RoPE
            // offsets and cache-append order are byte-identical to the
            // appendHistory-then-stepArr sequence (pinned by the merged-
            // forward equivalence test in mtp.zig).
            var st = self.mtp_hist_stash.?;
            self.mtp_hist_stash = null;
            defer st.deinit();
            // Drop the previous round's stale draft tail — the old
            // Phase 5a truncate, moved to consume time (offset-only).
            try mc.truncate(st.off0, s);
            var merged_ids = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(merged_ids);
            var merged_hidden = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(merged_hidden);
            {
                const idv = mlx.mlx_vector_array_new();
                defer _ = mlx.mlx_vector_array_free(idv);
                _ = mlx.mlx_vector_array_append_value(idv, st.ids);
                _ = mlx.mlx_vector_array_append_value(idv, prev_tok_arr);
                try mlx.check(mlx.mlx_concatenate_axis(&merged_ids, idv, 0, s));
                const hv = mlx.mlx_vector_array_new();
                defer _ = mlx.mlx_vector_array_free(hv);
                _ = mlx.mlx_vector_array_append_value(hv, st.hidden);
                _ = mlx.mlx_vector_array_append_value(hv, h_prev_arg);
                try mlx.check(mlx.mlx_concatenate_axis(&merged_hidden, hv, 1, s));
            }
            var merged_host: [mtp_mod.MAX_DEPTH + 2]u32 = undefined;
            @memcpy(merged_host[0..st.n], st.host_ids[0..st.n]);
            merged_host[st.n] = chain.t1;
            break :blk try head.forward(xfm, mc, merged_ids, merged_hidden, @intCast(st.off0), want, mtp_mrope_ctx, merged_host[0 .. st.n + 1]);
        } else try head.forward(xfm, mc, prev_tok_arr, h_prev_arg, @intCast(chain.off0 + i), want, mtp_mrope_ctx, if (i == 0) &[_]u32{chain.t1} else null);
        errdefer {
            if (step_out.logits.ctx != null) _ = mlx.mlx_array_free(step_out.logits);
            _ = mlx.mlx_array_free(step_out.hidden_next);
        }
        // A `.mixed` step that publishes `rerank_x` means `hidden_next` is
        // NOT what the lm_head consumes — on qwen4_exp it is the pre-mixer
        // `[B,S,hc*H]` stream and the mixer output is the rerank input.
        // Feeding the wrong one is a shape error, not a wrong answer.
        defer if (step_out.rerank_x.ctx != null) {
            _ = mlx.mlx_array_free(step_out.rerank_x);
        };
        if (use_rerank and chain.q_probs != null) {
            const rerank_x = if (step_out.rerank_x.ctx != null) step_out.rerank_x else step_out.hidden_next;
            const slots = chain.q_probs.?;
            if (try head.draftShortlist(xfm, rerank_x, draft_sampling.suppress_mask)) |shortlist| {
                var sl = shortlist;
                defer sl.deinit();
                const prop = try sampler.shortlistStepProposal(sl, self.mtpSamplingDraw(draft_sampling), s);
                chain.draft_arrs[i] = prop.id;
                slots[i] = prop.q;
            } else {
                // The coarse head just retired: propose greedily and hand
                // the verify the one-hot q that proposal really has.
                const id = try head.draftSelect(xfm, rerank_x, draft_sampling.suppress_mask);
                const prop = try sampler.argmaxStepProposal(id, @intCast(mlx.getShape(xfm.lm_head_w)[0]), s);
                chain.draft_arrs[i] = prop.id;
                slots[i] = prop.q;
            }
            chain.n_qp = i + 1;
        } else if (use_rerank) {
            const rerank_x = if (step_out.rerank_x.ctx != null) step_out.rerank_x else step_out.hidden_next;
            chain.draft_arrs[i] = try head.draftSelect(xfm, rerank_x, draft_sampling.suppress_mask);
        } else if (chain.q_probs) |slots| {
            const prop = try sampler.logitsStepProposal(step_out.logits, self.mtpSamplingDraw(draft_sampling), s);
            slots[i] = prop.q;
            chain.n_qp = i + 1;
            chain.draft_arrs[i] = prop.id;
        } else {
            chain.draft_arrs[i] = sampler.sampleTokenLazy(step_out.logits, draft_sampling, s);
        }
        chain.n_drafted = i + 1;
        if (chain.conf_arrs != null and i < chain.plan.m_lo) {
            // Chunk-A confidence: log p_head(draft) — built from the
            // step's own logits BEFORE they're freed (lazy graphs
            // hold their inputs internally).
            chain.conf_arrs.?[i] = try draftConfidenceGraph(step_out.logits, chain.draft_arrs[i], s);
            chain.n_conf = i + 1;
        }
        if (step_out.logits.ctx != null) _ = mlx.mlx_array_free(step_out.logits);
        step_out.logits = .{ .ctx = null };
        if (chain.h_chain) |h_old| _ = mlx.mlx_array_free(h_old);
        chain.h_chain = step_out.hidden_next;
        step_out.hidden_next = .{ .ctx = null };
    }
}

/// Fire the chain's built graphs [from..to) at the GPU (async, no
/// sync) so the head chain computes while the CPU builds the round's
/// remaining graphs. Chunk-A dispatches (from == 0) on two-chunk plans
/// also carry the confidence graphs and the chain hidden, so a
/// consume-time extension decision reads materialized arrays (near-free
/// sync) and chunk B resumes from a realized buffer. Dispatch timing
/// only — lazy sampling ops bind their PRNG key at graph BUILD time, so
/// values are identical.
pub fn mtpChainDispatch(chain: *const MtpPreDraft, from: u32, to: u32) !void {
    const ev = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ev);
    for (chain.draft_arrs[from..to]) |arr| _ = mlx.mlx_vector_array_append_value(ev, arr);
    if (chain.q_probs) |slots| {
        for (slots[from..to]) |arr| _ = mlx.mlx_vector_array_append_value(ev, arr);
    }
    if (from == 0) {
        if (chain.conf_arrs) |slots| {
            for (slots[0..chain.n_conf]) |arr| _ = mlx.mlx_vector_array_append_value(ev, arr);
            if (chain.h_chain) |h| _ = mlx.mlx_vector_array_append_value(ev, h);
        }
    }
    try mlx.check(mlx.mlx_async_eval(ev));
}

// One async eval over every row's built draft graphs so the chain runs while the CPU builds
// the verify; a lazy sampling op binds its PRNG key at graph build, so the drafts do not move.
pub fn mtpChainDispatchBatched(chains: []const MtpPreDraft) !void {
    const ev = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(ev);
    var any = false;
    for (chains) |chain| {
        const to = chain.n_drafted;
        if (to == 0) continue;
        any = true;
        for (chain.draft_arrs[0..to]) |arr| _ = mlx.mlx_vector_array_append_value(ev, arr);
        if (chain.q_probs) |slots| {
            for (slots[0..@min(to, chain.n_qp)]) |arr| _ = mlx.mlx_vector_array_append_value(ev, arr);
        }
        if (chain.conf_arrs) |slots| {
            for (slots[0..chain.n_conf]) |arr| _ = mlx.mlx_vector_array_append_value(ev, arr);
            if (chain.h_chain) |h| _ = mlx.mlx_vector_array_append_value(ev, h);
        }
    }
    if (!any) return;
    try mlx.check(mlx.mlx_async_eval(ev));
}

pub fn mtpDraftActiveRows(chains: []const MtpPreDraft, step: u32, indices: []usize) usize {
    var n: usize = 0;
    for (chains, 0..) |chain, row| {
        if (step >= chain.m) continue;
        indices[n] = row;
        n += 1;
    }
    return n;
}

pub fn mtpDraftStepPath(can_rerank: bool, sampled: bool, conf_needed: bool) DraftStepPath {
    // The chunk-A extension gate wants ln p_head(draft) over the whole
    // head, which the shortlist cannot answer.
    if (conf_needed or !can_rerank) return .full_logits;
    return if (sampled) .rerank_sampled else .rerank_greedy;
}

/// Whether this row drafts through the rerank shortlist at `step`. Greedy and
/// sampled rerank rows share the group: same coarse readout, same
/// `.mixed_last_row` head forward, only the tail over the 32 candidates
/// differs — so a mixed sampled/greedy workload does not split the step into
/// two head forwards.
pub fn mtpRowRerankMode(g: *Generator, c: *const MtpPreDraft, step: u32) bool {
    const conf_needed = c.conf_arrs != null and step < c.plan.m_lo;
    return mtpDraftStepPath(g.mtp.?.canRerankDrafts(), c.q_probs != null, conf_needed) != .full_logits;
}

pub fn mtpChainBuildBatched(gens: []const *Generator, chains: []MtpPreDraft, from: u32, to: u32) !void {
    if (gens.len != chains.len) return error.MtpBatchLengthMismatch;
    if (gens.len > 32) return error.MtpBatchTooWide;
    var step = from;
    while (step < to) : (step += 1) {
        var indices: [32]usize = undefined;
        const n = mtpDraftActiveRows(chains, step, &indices);
        if (n == 0) break;
        var modes: [32]bool = undefined;
        for (indices[0..n], 0..) |row, i| {
            modes[i] = mtpRowRerankMode(gens[row], &chains[row], step);
        }
        for ([_]bool{ true, false }) |mode| {
            var active_gens: [32]*Generator = undefined;
            var active_chains: [32]MtpPreDraft = undefined;
            var row_map: [32]usize = undefined;
            var count: usize = 0;
            for (indices[0..n], 0..) |row, i| {
                if (modes[i] != mode) continue;
                active_gens[count] = gens[row];
                active_chains[count] = chains[row];
                row_map[count] = row;
                count += 1;
            }
            if (count == 0) continue;
            defer for (row_map[0..count], 0..) |row, i| {
                chains[row] = active_chains[i];
            };
            try mtpChainBuildBatchedActive(active_gens[0..count], active_chains[0..count], step, step + 1);
        }
    }
}

/// One draft step for a group whose rerank rows include sampled chains:
/// per-row shortlists off ONE coarse readout, then one batched proposal
/// over the rows that sample this depth when they share a sampler and none
/// is seeded (the `sampleRowsLazy` gate), else one proposal per row. Every
/// other row takes its argmax.
pub fn mtpGroupSampledDrafts(
    gens: []const *Generator,
    chains: []MtpPreDraft,
    stacked_x: mlx.mlx_array,
    step: u32,
) !void {
    const xfm = gens[0].xfm;
    const s = xfm.s;
    const n = gens.len;
    var shortlists: [32]mtp_mod.Shortlist = undefined;
    const have = try xfm.qwen4DraftShortlistsBatched(stacked_x, gens[0].sampling.suppress_mask, shortlists[0..n]);
    if (!have) {
        // No coarse head: every row proposes greedily and hands the verify
        // the one-hot q that proposal really has.
        const vocab: c_int = @intCast(mlx.getShape(xfm.lm_head_w)[0]);
        for (gens, chains, 0..) |g, *c, k| {
            const k_c: c_int = @intCast(k);
            var xk = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(xk);
            const xsh = mlx.getShape(stacked_x);
            try mlx.check(mlx.mlx_slice(&xk, stacked_x, &[_]c_int{ k_c, 0, 0 }, 3, &[_]c_int{ k_c + 1, 1, xsh[xsh.len - 1] }, 3, &[_]c_int{ 1, 1, 1 }, 3, s));
            const id = try g.mtp.?.draftSelect(xfm, xk, g.sampling.suppress_mask);
            if (c.q_probs) |slots| {
                const prop = try sampler.argmaxStepProposal(id, vocab, s);
                c.draft_arrs[step] = prop.id;
                slots[step] = prop.q;
                c.n_qp = step + 1;
            } else c.draft_arrs[step] = id;
            c.n_drafted = step + 1;
        }
        return;
    }
    defer for (shortlists[0..n]) |*sl| sl.deinit();

    // Which rows sample, and whether one batched draw can serve them all.
    var sampled: [32]usize = undefined;
    var params: [32]sampler.SamplingParams = undefined;
    var samples: [32]bool = @splat(false);
    var k_sampled: usize = 0;
    for (gens, chains, 0..) |g, *c, row| {
        const row_params = g.mtpDraftSampling(step);
        if (c.q_probs == null or sampler.isGreedyTemperature(row_params.temperature)) continue;
        samples[row] = true;
        sampled[k_sampled] = row;
        params[k_sampled] = row_params;
        k_sampled += 1;
    }
    const batched = k_sampled > 1 and
        sampler.sampleRowsHomogeneous(params[0..k_sampled]) and
        !sampler.sampleRowsAnySeed(params[0..k_sampled]);

    for (gens, chains, 0..) |g, *c, row| {
        if (samples[row]) continue;
        if (c.q_probs) |slots| {
            const prop = try sampler.shortlistStepProposal(shortlists[row], g.mtpDraftSampling(step), s);
            c.draft_arrs[step] = prop.id;
            slots[step] = prop.q;
            c.n_qp = step + 1;
        } else c.draft_arrs[step] = try mtp_mod.shortlistArgmax(s, shortlists[row]);
        c.n_drafted = step + 1;
    }
    if (!batched) {
        for (sampled[0..k_sampled], params[0..k_sampled]) |row, row_params| {
            const prop = try sampler.shortlistProposal(shortlists[row], gens[row].mtpSamplingDraw(row_params), s);
            chains[row].draft_arrs[step] = prop.id;
            chains[row].n_drafted = step + 1;
            chains[row].q_probs.?[step] = prop.q;
            chains[row].n_qp = step + 1;
        }
        return;
    }
    const group = try sampler.shortlistProposalRows(shortlists[0..n], sampled[0..k_sampled], params[0], s);
    for (sampled[0..k_sampled], 0..) |row, j| {
        chains[row].draft_arrs[step] = group.ids[j];
        chains[row].n_drafted = step + 1;
        chains[row].q_probs.?[step] = group.qs[j];
        chains[row].n_qp = step + 1;
    }
}

pub fn mtpBatchHeadPosition(pos_base: c_int, seq_offset: usize) c_int {
    return if (seq_offset == 0 and pos_base == -1) 1 else pos_base + @as(c_int, @intCast(seq_offset));
}

pub fn mtpChainBuildBatchedActive(gens: []const *Generator, chains: []MtpPreDraft, from: u32, to: u32) !void {
    if (gens.len == 0) return;
    std.debug.assert(gens.len == chains.len);
    const xfm = gens[0].xfm;
    var any_mrope = false;
    var all_qwen4 = true;
    for (gens) |g| {
        if (g.mtpMropeContext() != null) any_mrope = true;
        switch (g.mtp.?) {
            .qwen4 => {},
            .mimo, .dsv41 => all_qwen4 = false,
        }
    }
    if (!all_qwen4 or any_mrope or gens.len == 1) {
        for (gens, chains) |g, *c| try g.mtpChainBuild(c, from, to);
        return;
    }
    const n = gens.len;
    var i: u32 = from;
    while (i < to) : (i += 1) {
        var rows: [32]transformer_mod.Transformer.Qwen4MtpBatchRow = undefined;
        var merged_ids: [32]mlx.mlx_array = undefined;
        var merged_hidden: [32]mlx.mlx_array = undefined;
        var own_ids: [32]bool = @splat(false);
        var own_h: [32]bool = @splat(false);
        for (merged_ids[0..n], 0..) |*p, k| {
            p.* = .{ .ctx = null };
            merged_hidden[k] = .{ .ctx = null };
        }
        defer for (0..n) |k| {
            if (own_ids[k] and merged_ids[k].ctx != null) _ = mlx.mlx_array_free(merged_ids[k]);
            if (own_h[k] and merged_hidden[k].ctx != null) _ = mlx.mlx_array_free(merged_hidden[k]);
        };
        var modes: [32]bool = undefined;
        for (gens, chains, 0..) |g, *c, row| modes[row] = mtpRowRerankMode(g, c, i);
        const use_rerank = modes[0];
        const project: transformer_mod.Transformer.Qwen4MtpProject = if (use_rerank) .mixed_last_row else .last_row;
        for (gens, chains, 0..) |g, *c, k| {
            const qref = g.mtp_cache.?.qwen4;
            const st = g.mtp_cache.?.qwen4StateForBatch();
            const h_prev_arg: mlx.mlx_array = if (c.h_chain) |h| h else g.last_hidden;
            const prev_tok_arr: mlx.mlx_array = if (i == 0) c.t1_arr else c.draft_arrs[i - 1];
            if (i == 0 and g.mtp_hist_stash != null) {
                var stash = g.mtp_hist_stash.?;
                g.mtp_hist_stash = null;
                defer stash.deinit();
                try qref.t.qwen4MtpTruncateOn(st, stash.off0);
                var mids = mlx.mlx_array_new();
                var mh = mlx.mlx_array_new();
                {
                    const idv = mlx.mlx_vector_array_new();
                    defer _ = mlx.mlx_vector_array_free(idv);
                    _ = mlx.mlx_vector_array_append_value(idv, stash.ids);
                    _ = mlx.mlx_vector_array_append_value(idv, prev_tok_arr);
                    try mlx.check(mlx.mlx_concatenate_axis(&mids, idv, 0, xfm.s));
                    const hv = mlx.mlx_vector_array_new();
                    defer _ = mlx.mlx_vector_array_free(hv);
                    _ = mlx.mlx_vector_array_append_value(hv, stash.hidden);
                    _ = mlx.mlx_vector_array_append_value(hv, h_prev_arg);
                    try mlx.check(mlx.mlx_concatenate_axis(&mh, hv, 1, xfm.s));
                }
                merged_ids[k] = mids;
                merged_hidden[k] = mh;
                own_ids[k] = true;
                own_h[k] = true;
            } else {
                merged_ids[k] = prev_tok_arr;
                merged_hidden[k] = h_prev_arg;
            }
            rows[k] = .{
                .st = st,
                .stream = merged_hidden[k],
                .token_ids = merged_ids[k],
                .pos_offset = mtpBatchHeadPosition(st.pos_base, st.seq_offset),
            };
        }
        const outs = try xfm.qwen4MtpForwardBatched(rows[0..n], project, to);
        defer {
            for (outs) |out| {
                if (out.logits.ctx != null) _ = mlx.mlx_array_free(out.logits);
                if (out.stream.ctx != null) _ = mlx.mlx_array_free(out.stream);
                if (out.mixed.ctx != null) _ = mlx.mlx_array_free(out.mixed);
            }
            xfm.allocator.free(outs);
        }
        var mixed_parts: [32]mlx.mlx_array = undefined;
        if (use_rerank) {
            for (outs, 0..) |o, k| mixed_parts[k] = o.mixed;
            const mvec = mlx.mlx_vector_array_new_data(mixed_parts[0..n].ptr, n);
            defer _ = mlx.mlx_vector_array_free(mvec);
            var stacked_x = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(stacked_x);
            try mlx.check(mlx.mlx_concatenate_axis(&stacked_x, mvec, 0, xfm.s));
            var any_sampled = false;
            for (chains) |*c| {
                if (c.q_probs != null) any_sampled = true;
            }
            if (any_sampled) {
                try mtpGroupSampledDrafts(gens, chains, stacked_x, i);
            } else {
                const ids = try xfm.qwen4DraftSelectBatched(stacked_x, gens[0].sampling.suppress_mask);
                defer _ = mlx.mlx_array_free(ids);
                for (chains, 0..) |*c, k| {
                    const k_c: c_int = @intCast(k);
                    var one = mlx.mlx_array_new();
                    try mlx.check(mlx.mlx_slice(&one, ids, &[_]c_int{ k_c, 0 }, 2, &[_]c_int{ k_c + 1, 1 }, 2, &[_]c_int{ 1, 1 }, 2, xfm.s));
                    c.draft_arrs[i] = one;
                }
            }
            for (chains, 0..) |*c, k| {
                c.n_drafted = i + 1;
                if (c.h_chain) |h_old| _ = mlx.mlx_array_free(h_old);
                c.h_chain = outs[k].stream;
                outs[k].stream = .{ .ctx = null };
            }
        } else {
            for (gens, chains, 0..) |g, *c, k| {
                const draft_sampling = g.mtpDraftSampling(i);
                if (c.q_probs) |slots| {
                    const prop = try sampler.logitsStepProposal(outs[k].logits, g.mtpSamplingDraw(draft_sampling), xfm.s);
                    slots[i] = prop.q;
                    c.n_qp = i + 1;
                    c.draft_arrs[i] = prop.id;
                } else {
                    c.draft_arrs[i] = sampler.sampleTokenLazy(outs[k].logits, draft_sampling, xfm.s);
                }
                c.n_drafted = i + 1;
                if (c.conf_arrs != null and i < c.plan.m_lo) {
                    c.conf_arrs.?[i] = try draftConfidenceGraph(outs[k].logits, c.draft_arrs[i], xfm.s);
                    c.n_conf = i + 1;
                }
                if (c.h_chain) |h_old| _ = mlx.mlx_array_free(h_old);
                c.h_chain = outs[k].stream;
                outs[k].stream = .{ .ctx = null };
            }
        }
    }
}

pub fn mtpBatchedExactGraph(
    probs_all: mlx.mlx_array,
    draft_arrs: []const mlx.mlx_array,
    q_probs: ?[]const mlx.mlx_array,
    m: u32,
    _: f32,
    sampling: sampler.SamplingParams,
    s: mlx.mlx_stream,
) !MtpBatchedGraph {
    return mtpBatchedAcceptGraph(probs_all, draft_arrs, q_probs, m, sampling, s);
}

pub fn mtpBatchedAcceptGraph(
    probs_all: mlx.mlx_array,
    draft_arrs: []const mlx.mlx_array,
    q_probs: ?[]const mlx.mlx_array,
    m: u32,
    sampling: sampler.SamplingParams,
    s: mlx.mlx_stream,
) !MtpBatchedGraph {
    const shape = mlx.getShape(probs_all); // [1, 1+m, V]
    const vocab = shape[2];
    const mi: c_int = @intCast(m);
    const strides2 = [_]c_int{ 1, 1 };

    const p2d = try verifyRows2d(probs_all, m, s);
    defer _ = mlx.mlx_array_free(p2d);

    var ids_2d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids_2d);
    {
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        for (draft_arrs[0..m]) |arr| _ = mlx.mlx_vector_array_append_value(vec, arr);
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_concatenate_axis(&flat, vec, 0, s));
        const id2_shape = [_]c_int{ mi, 1 };
        try mlx.check(mlx.mlx_reshape(&ids_2d, flat, &id2_shape, 2, s));
    }

    // Draft-position rows [m, V] and the bonus row [1, V] (slice views).
    var p_rows = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(p_rows);
    var bonus = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bonus);
    {
        const r_start = [_]c_int{ 0, 0 };
        const r_stop = [_]c_int{ mi, vocab };
        try mlx.check(mlx.mlx_slice(&p_rows, p2d, &r_start, 2, &r_stop, 2, &strides2, 2, s));
        const b_start = [_]c_int{ mi, 0 };
        const b_stop = [_]c_int{ mi + 1, vocab };
        try mlx.check(mlx.mlx_slice(&bonus, p2d, &b_start, 2, &b_stop, 2, &strides2, 2, s));
    }

    // Proposal stack [m, V]: the full sharpened q rows under sharp
    // drafts (exact Leviathan residual), one-hot of the lazy draft ids
    // under greedy proposals.
    var proposal = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(proposal);
    if (q_probs) |qs| {
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        for (qs[0..m]) |arr| _ = mlx.mlx_vector_array_append_value(vec, arr);
        try mlx.check(mlx.mlx_concatenate_axis(&proposal, vec, 0, s));
    } else {
        var indices = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(indices);
        try mlx.check(mlx.mlx_arange(&indices, 0, @as(f64, @floatFromInt(vocab)), 1, .int32, s));
        var onehot_b = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(onehot_b);
        try mlx.check(mlx.mlx_equal(&onehot_b, indices, ids_2d, s));
        try mlx.check(mlx.mlx_astype(&proposal, onehot_b, mlx.mlx_array_dtype(p2d), s));
    }

    // residual = max(p − proposal, 0) rows, then [residual; bonus] and
    // ONE categorical over the whole stack.
    var corr_samples = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(corr_samples);
    {
        var diff = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(diff);
        try mlx.check(mlx.mlx_subtract(&diff, p_rows, proposal, s));
        const zero = mlx.mlx_array_new_float(0.0);
        defer _ = mlx.mlx_array_free(zero);
        var residual = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(residual);
        try mlx.check(mlx.mlx_maximum(&residual, diff, zero, s));
        var stack = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(stack);
        {
            const vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(vec);
            _ = mlx.mlx_vector_array_append_value(vec, residual);
            _ = mlx.mlx_vector_array_append_value(vec, bonus);
            try mlx.check(mlx.mlx_concatenate_axis(&stack, vec, 0, s));
        }
        var log_stack = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(log_stack);
        try mlx.check(mlx.mlx_log(&log_stack, stack, s));
        const key = sampler.seedKey(sampling);
        defer _ = mlx.mlx_array_free(key);
        var sampled = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sampled);
        try mlx.check(mlx.mlx_random_categorical(&sampled, log_stack, -1, key, s));
        try mlx.check(mlx.mlx_astype(&corr_samples, sampled, .int32, s));
    }

    // accept_p[k] = p_rows[k][draft_k]; accept_q[k] = proposal density.
    var accept_p = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(accept_p);
    var accept_q: mlx.mlx_array = .{ .ctx = null };
    errdefer if (accept_q.ctx != null) {
        _ = mlx.mlx_array_free(accept_q);
    };
    {
        var taken = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(taken);
        try mlx.check(mlx.mlx_take_along_axis(&taken, p_rows, ids_2d, -1, s));
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        const m_shape = [_]c_int{mi};
        try mlx.check(mlx.mlx_reshape(&flat, taken, &m_shape, 1, s));
        try mlx.check(mlx.mlx_astype(&accept_p, flat, .float32, s));
    }
    if (q_probs != null) {
        var taken = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(taken);
        try mlx.check(mlx.mlx_take_along_axis(&taken, proposal, ids_2d, -1, s));
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        const m_shape = [_]c_int{mi};
        try mlx.check(mlx.mlx_reshape(&flat, taken, &m_shape, 1, s));
        accept_q = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&accept_q, flat, .float32, s));
    }

    return .{ .corr_samples = corr_samples, .accept_p = accept_p, .accept_q = accept_q };
}

pub fn mtpBatchedTypicalGraph(
    probs_all: mlx.mlx_array,
    draft_arrs: []const mlx.mlx_array,
    q_probs: ?[]const mlx.mlx_array,
    m: u32,
    delta: f32,
    sampling: sampler.SamplingParams,
    s: mlx.mlx_stream,
) !MtpBatchedGraph {
    return mtpBatchedLossyGraph(.typical, probs_all, draft_arrs, q_probs, m, delta, sampling, s);
}

pub fn mtpBatchedTokenV3Graph(
    probs_all: mlx.mlx_array,
    draft_arrs: []const mlx.mlx_array,
    q_probs: ?[]const mlx.mlx_array,
    m: u32,
    alpha: f32,
    sampling: sampler.SamplingParams,
    s: mlx.mlx_stream,
) !MtpBatchedGraph {
    return mtpBatchedLossyGraph(.tokenv3, probs_all, draft_arrs, q_probs, m, alpha, sampling, s);
}

/// Specialized at compile time; the installed route adds no mode switch
/// to a measured MTP round. Both lanes keep corrections in one lazy graph.
pub fn mtpBatchedLossyGraph(
    comptime mode: enum { typical, tokenv3 },
    probs_all: mlx.mlx_array,
    draft_arrs: []const mlx.mlx_array,
    q_probs: ?[]const mlx.mlx_array,
    m: u32,
    param: f32,
    sampling: sampler.SamplingParams,
    s: mlx.mlx_stream,
) !MtpBatchedGraph {
    const shape = mlx.getShape(probs_all); // [1, 1+m, V]
    const vocab = shape[2];
    const mi: c_int = @intCast(m);
    const strides2 = [_]c_int{ 1, 1 };

    const p2d = try verifyRows2d(probs_all, m, s);
    defer _ = mlx.mlx_array_free(p2d);

    var ids_2d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids_2d);
    {
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        for (draft_arrs[0..m]) |arr| _ = mlx.mlx_vector_array_append_value(vec, arr);
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        try mlx.check(mlx.mlx_concatenate_axis(&flat, vec, 0, s));
        const id2_shape = [_]c_int{ mi, 1 };
        try mlx.check(mlx.mlx_reshape(&ids_2d, flat, &id2_shape, 2, s));
    }

    var p_rows = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(p_rows);
    var bonus = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bonus);
    {
        const r_start = [_]c_int{ 0, 0 };
        const r_stop = [_]c_int{ mi, vocab };
        try mlx.check(mlx.mlx_slice(&p_rows, p2d, &r_start, 2, &r_stop, 2, &strides2, 2, s));
        const b_start = [_]c_int{ mi, 0 };
        const b_stop = [_]c_int{ mi + 1, vocab };
        try mlx.check(mlx.mlx_slice(&bonus, p2d, &b_start, 2, &b_stop, 2, &strides2, 2, s));
    }

    // TokenV3 needs the FULL q row. Typical acceptance uses only p and
    // does not add a proposal concat or one-hot operation to the graph.
    var proposal: mlx.mlx_array = .{ .ctx = null };
    defer if (proposal.ctx != null) {
        _ = mlx.mlx_array_free(proposal);
    };
    if (comptime mode == .tokenv3) {
        proposal = mlx.mlx_array_new();
        if (q_probs) |qs| {
            const vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(vec);
            for (qs[0..m]) |arr| _ = mlx.mlx_vector_array_append_value(vec, arr);
            try mlx.check(mlx.mlx_concatenate_axis(&proposal, vec, 0, s));
        } else {
            var indices = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(indices);
            try mlx.check(mlx.mlx_arange(&indices, 0, @as(f64, @floatFromInt(vocab)), 1, .int32, s));
            var onehot_b = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(onehot_b);
            try mlx.check(mlx.mlx_equal(&onehot_b, indices, ids_2d, s));
            try mlx.check(mlx.mlx_astype(&proposal, onehot_b, mlx.mlx_array_dtype(p2d), s));
        }
    }

    // TokenV3 Eq. 11: pi = q*1[Top] + p*sum(q outside Top).
    // The p*eta term applies to EVERY token, including those inside Top.
    var pi: mlx.mlx_array = .{ .ctx = null };
    defer if (pi.ctx != null) {
        _ = mlx.mlx_array_free(pi);
    };
    var defer_mask: mlx.mlx_array = .{ .ctx = null };
    defer if (defer_mask.ctx != null) {
        _ = mlx.mlx_array_free(defer_mask);
    };
    if (comptime mode == .tokenv3) {
        var peak = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(peak);
        try mlx.check(mlx.mlx_max_axis(&peak, p_rows, 1, true, s));
        const complement = mlx.mlx_array_new_float(1.0 - param);
        defer _ = mlx.mlx_array_free(complement);
        var cutoff = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cutoff);
        try mlx.check(mlx.mlx_multiply(&cutoff, peak, complement, s));
        defer_mask = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_less(&defer_mask, p_rows, cutoff, s));
        const zero = mlx.mlx_array_new_float(0.0);
        defer _ = mlx.mlx_array_free(zero);
        var deferred_q = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(deferred_q);
        try mlx.check(mlx.mlx_where(&deferred_q, defer_mask, proposal, zero, s));
        var eta = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(eta);
        try mlx.check(mlx.mlx_sum_axis(&eta, deferred_q, 1, true, s));
        var kept_q = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(kept_q);
        try mlx.check(mlx.mlx_where(&kept_q, defer_mask, zero, proposal, s));
        var recycled = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(recycled);
        try mlx.check(mlx.mlx_multiply(&recycled, p_rows, eta, s));
        pi = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_add(&pi, kept_q, recycled, s));
    }
    const decision_rows = if (comptime mode == .tokenv3) pi else p_rows;

    var corr_samples = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(corr_samples);
    {
        var corr_rows: mlx.mlx_array = .{ .ctx = null };
        defer if (corr_rows.ctx != null) {
            _ = mlx.mlx_array_free(corr_rows);
        };
        if (comptime mode == .typical) {
            corr_rows = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_array_set(&corr_rows, p_rows));
        } else {
            var diff = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(diff);
            try mlx.check(mlx.mlx_subtract(&diff, pi, proposal, s));
            const zero = mlx.mlx_array_new_float(0.0);
            defer _ = mlx.mlx_array_free(zero);
            corr_rows = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_maximum(&corr_rows, diff, zero, s));
        }
        var stack = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(stack);
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        _ = mlx.mlx_vector_array_append_value(vec, corr_rows);
        _ = mlx.mlx_vector_array_append_value(vec, bonus);
        try mlx.check(mlx.mlx_concatenate_axis(&stack, vec, 0, s));
        var log_stack = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(log_stack);
        try mlx.check(mlx.mlx_log(&log_stack, stack, s));
        const key = sampler.seedKey(sampling);
        defer _ = mlx.mlx_array_free(key);
        var sampled = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sampled);
        try mlx.check(mlx.mlx_random_categorical(&sampled, log_stack, -1, key, s));
        try mlx.check(mlx.mlx_astype(&corr_samples, sampled, .int32, s));
    }

    var accept_p = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(accept_p);
    {
        var taken = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(taken);
        try mlx.check(mlx.mlx_take_along_axis(&taken, decision_rows, ids_2d, -1, s));
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        const m_shape = [_]c_int{mi};
        try mlx.check(mlx.mlx_reshape(&flat, taken, &m_shape, 1, s));
        try mlx.check(mlx.mlx_astype(&accept_p, flat, .float32, s));
    }

    var accept_q: mlx.mlx_array = .{ .ctx = null };
    errdefer if (accept_q.ctx != null) {
        _ = mlx.mlx_array_free(accept_q);
    };
    if (comptime mode == .typical) {
        // H(p) = -sum p*log(p), with log(0) safely masked by clamping
        // only the log operand. The p factor remains exactly zero.
        const tiny = mlx.mlx_array_new_float(1e-30);
        defer _ = mlx.mlx_array_free(tiny);
        var safe_p = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(safe_p);
        try mlx.check(mlx.mlx_maximum(&safe_p, p_rows, tiny, s));
        var log_p = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(log_p);
        try mlx.check(mlx.mlx_log(&log_p, safe_p, s));
        var p_log_p = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(p_log_p);
        try mlx.check(mlx.mlx_multiply(&p_log_p, p_rows, log_p, s));
        var sum = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sum);
        try mlx.check(mlx.mlx_sum_axis(&sum, p_log_p, 1, false, s));
        var scaled = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(scaled);
        try mlx.check(mlx.mlx_exp(&scaled, sum, s)); // exp(-H)
        const delta = mlx.mlx_array_new_float(param);
        defer _ = mlx.mlx_array_free(delta);
        var raw_floor = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(raw_floor);
        try mlx.check(mlx.mlx_multiply(&raw_floor, scaled, delta, s));
        const eps = mlx.mlx_array_new_float(1.0);
        defer _ = mlx.mlx_array_free(eps);
        accept_q = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_minimum(&accept_q, raw_floor, eps, s));
    } else if (q_probs != null) {
        var taken = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(taken);
        try mlx.check(mlx.mlx_take_along_axis(&taken, proposal, ids_2d, -1, s));
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        const m_shape = [_]c_int{mi};
        try mlx.check(mlx.mlx_reshape(&flat, taken, &m_shape, 1, s));
        accept_q = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&accept_q, flat, .float32, s));
    }

    var accept_defer: mlx.mlx_array = .{ .ctx = null };
    errdefer if (accept_defer.ctx != null) {
        _ = mlx.mlx_array_free(accept_defer);
    };
    if (comptime mode == .tokenv3) {
        var taken = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(taken);
        try mlx.check(mlx.mlx_take_along_axis(&taken, defer_mask, ids_2d, -1, s));
        const m_shape = [_]c_int{mi};
        accept_defer = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&accept_defer, taken, &m_shape, 1, s));
    }
    return .{ .corr_samples = corr_samples, .accept_p = accept_p, .accept_q = accept_q, .accept_defer = accept_defer };
}

/// A pre-draft chain is only worth its cost when a later round reads it: a request whose
/// budget is spent, or whose pending token ends it, keeps no reader.
pub fn mtpPreDraftOpenAllows(completion_tokens: u32, max_tokens: u32, pending_is_eos: bool) bool {
    return completion_tokens < max_tokens and !pending_is_eos;
}

/// Test seam: `false` leaves a completed round's stash unconsumed by a successor pre-draft.
pub var mtp_predraft_enabled: bool = true;

/// Cross-round pre-draft (round pipelining): at the round's tail — the
/// accept decision made, trunk committed/rolled back, EV updated,
/// last_hidden/next_token_id already pointing at the next round — build
/// the NEXT round's chunk-A draft chain (consuming the history stash
/// Phase 5a just created, exactly as the next round's Phase 1 would)
/// and fire it at the GPU. The CPU then returns to the scheduler for
/// emit/SSE work while the head chain runs; the next nextMtp call finds
/// its drafts already materialized. The plan is computed AFTER this
/// round's EV update, so it is byte-identical to the one the next
/// round's entry would compute.
pub fn mtpMaybePreDraft(self: *Generator, allocator: std.mem.Allocator) !void {
    if (!mtpPreDraftOpenAllows(self.completion_tokens, self.max_tokens, isEosId(self.next_token_id, self.eos_token_ids))) return;
    if (!mtp_predraft_enabled or self.spec_disabled_runtime or self.mtp_batch_head or self.mtp_planner_pending) return;
    std.debug.assert(self.mtp_pre_draft == null);
    const plan = self.mtpRoundPlan();
    // The plan itself can end speculation or park the request on a serial probe.
    if (self.spec_disabled_runtime or self.mtp_serial_left > 0) return;
    if (try self.mtpLookupChain(allocator, plan, self.next_token_id)) |lc| {
        self.mtp_pre_draft = lc;
        return;
    }
    var chain = try self.mtpChainInit(allocator, plan, self.next_token_id);
    errdefer chain.deinit(allocator);
    try self.mtpChainBuild(&chain, 0, plan.m_lo);
    try mtpChainDispatch(&chain, 0, plan.m_lo);
    self.mtp_pre_draft = chain;
}

/// Will this serial block be left behind? Only then is applying the deferred history stash worth a head forward.
pub fn mtpSerialMayResume(self: *const Generator) bool {
    if (self.mtp_serial_left > 0 or self.mtp_serial_exit != .none) return true;
    return self.spec_disable_reason == .adaptive and self.mtpAdaptiveHeadMayResume();
}

/// Leave the head cleanly for a serial block; `apply_stash` applies the deferred history
/// stash so the head history is complete up to the block. Idempotent.
pub fn mtpDetachHead(self: *Generator, allocator: std.mem.Allocator, apply_stash: bool) !void {
    if (self.mtp_pre_draft) |*pd| {
        if (self.mtp_planner_owned) try self.mtp_cache.?.truncate(self.mtpCommittedHistoryLen(), self.xfm.s);
        pd.deinit(allocator);
        self.mtp_pre_draft = null;
    }
    if (apply_stash) try self.mtpApplyStash();
}

/// Append the deferred history stash to the head cache now, as a
/// history-only head forward (lazy; the next head forward realizes it).
pub fn mtpApplyStash(self: *Generator) !void {
    var st = self.mtp_hist_stash orelse return;
    self.mtp_hist_stash = null;
    defer st.deinit();
    const mc = &self.mtp_cache.?;
    try mc.truncate(st.off0, self.xfm.s);
    // A committed-history append: the mixer and the vocab projection are dead work.
    const out = try self.mtp.?.forward(
        self.xfm,
        mc,
        st.ids,
        st.hidden,
        @intCast(st.off0),
        .none,
        self.mtpMropeContext(),
        st.host_ids[0..st.n],
    );
    if (out.logits.ctx != null) _ = mlx.mlx_array_free(out.logits);
    if (out.hidden_next.ctx != null) _ = mlx.mlx_array_free(out.hidden_next);
    if (out.rerank_x.ctx != null) _ = mlx.mlx_array_free(out.rerank_x);
}

/// Where an MTP slot re-enters after a forced commit: a planner-owned slot rejoins the planner, which drains
/// the commit's pending logits and primes a fresh hidden; a serial ramp would keep its whole group plain.
pub fn mtpSerialExitAfterForced(planner_owned: bool) MtpSerialExit {
    return if (planner_owned) .none else .drain;
}

/// A forced commit (`commitForcedTokens`) ran plain through the trunk, so the head's history has a
/// hole. Re-seed a qwen4 head (its positions refuse a gap) and take the serial exit ramp back to
/// MTP. False = not an MTP slot that may resume: the caller disables speculation.
pub fn mtpResumeAfterForced(self: *Generator, allocator: std.mem.Allocator) !bool {
    if (self.mtp == null or self.mtp_cache == null or !self.mtpAdaptiveHeadMayResume()) return false;
    try self.mtpDetachHead(allocator, false);
    switch (self.mtp_cache.?) {
        .qwen4 => |r| {
            if (self.mtp_hist_stash) |*st| st.deinit();
            self.mtp_hist_stash = null;
            r.t.qwen4MtpActivate(r.st);
            try r.t.qwen4MtpReset();
        },
        .mimo => {},
    }
    self.spec_disabled_runtime = false;
    self.spec_disable_reason = .none;
    self.mtp_serial_exit = mtpSerialExitAfterForced(group_planner.enabled() and self.mtp_planner_owned);
    return true;
}

pub fn mtpSerialOneToken(allocator: std.mem.Allocator, tok: u32) !DrafterStepResult {
    const tokens = try allocator.alloc(u32, 1);
    tokens[0] = tok;
    return DrafterStepResult{ .tokens = tokens, .accepted_tokens = 0 };
}

/// Inter-tick wall clock for the table's serial cell (twin of `mtpRegimeWallMs`). Null on the first read of a block.
pub fn mtpSerialWallMs(self: *Generator) ?f32 {
    if (self.mtp_serial_clock) |*c| {
        const ns = c.read();
        c.reset();
        return @as(f32, @floatFromInt(ns)) / @as(f32, std.time.ns_per_ms);
    }
    self.mtp_serial_clock = io_util.Stopwatch.init(self.timer.io);
    return null;
}

/// May `mtpSerialCaptureTick` run? It forwards the one pending token with capture, so
/// pipelined logits or a pipelined token would publish a hidden for an uncommitted row.
/// A runtime check: `std.debug.assert` compiles to nothing in ReleaseFast.
pub fn mtpSerialCaptureReady(has_pending_logits: bool, has_pending_token: bool) bool {
    return !has_pending_logits and !has_pending_token;
}

/// Abandon the ramp back to MTP and finish this request on the serial arm. Logged once per request.
pub fn mtpSerialGiveUp(self: *Generator, allocator: std.mem.Allocator, why: []const u8) !?DrafterStepResult {
    if (!self.mtp_serial_giveup_logged) {
        self.mtp_serial_giveup_logged = true;
        log.info("  [mtp] adaptive: {s} at the serial exit -> serial for the rest of the request\n", .{why});
    }
    self.mtp_serial_exit = .none;
    self.mtp_serial_left = 0;
    self.spec_disabled_runtime = true;
    self.spec_disable_reason = .adaptive;
    const tok = try self.next(allocator) orelse return null;
    return try mtpSerialOneToken(allocator, tok);
}

/// One tick of a serial block. `.none` is the real serial arm; `.drain` and `.capture` are the exit ramp.
pub fn mtpSerialTick(self: *Generator, allocator: std.mem.Allocator) !?DrafterStepResult {
    if (self.mtp_serial_exit == .capture) return try self.mtpSerialCaptureTick(allocator);
    if (self.mtp_serial_exit == .drain) {
        switch (try self.drainPipelineForSpec(allocator)) {
            .stopped => return null,
            .stay_disabled => return try self.mtpSerialGiveUp(allocator, "unexpected pipeline state"),
            .already_clean => {
                self.mtp_serial_exit = .capture;
                return try self.mtpSerialCaptureTick(allocator);
            },
            .drained => |tok| {
                self.mtp_serial_exit = .capture;
                return try mtpSerialOneToken(allocator, tok);
            },
        }
    }
    const tok = try self.next(allocator) orelse return null;
    self.observeSerialTick();
    if (self.mtp_serial_left > 0) {
        self.mtp_serial_left -= 1;
        if (self.mtp_serial_left == 0) self.mtp_serial_exit = .drain;
    }
    return try mtpSerialOneToken(allocator, tok);
}

/// The one synchronous step that ends a serial block: forward the pending token with
/// capture and sample its successor without forwarding it (`nextMtp`'s entry invariant).
pub fn mtpSerialCaptureTick(self: *Generator, allocator: std.mem.Allocator) !?DrafterStepResult {
    if (!mtpSerialCaptureReady(self.has_pending_logits, self.has_pending_token)) {
        return try self.mtpSerialGiveUp(allocator, "pipeline not drained");
    }
    if (try self.checkStop()) return null;
    const token = self.next_token_id;
    const tok_i32: i32 = @intCast(token);
    const tok_shape = [_]c_int{ 1, 1 };
    const tok_input = mlx.mlx_array_new_data(&tok_i32, &tok_shape, 2, .int32);
    defer _ = mlx.mlx_array_free(tok_input);

    var hidden = mlx.mlx_array_new();
    const logits = self.xfm.forwardWithCapture(&self.ctx, tok_input, &hidden) catch |e| {
        _ = mlx.mlx_array_free(hidden);
        return e;
    };
    defer _ = mlx.mlx_array_free(logits);
    if (self.has_last_hidden) _ = mlx.mlx_array_free(self.last_hidden);
    self.last_hidden = hidden;
    self.has_last_hidden = true;
    self.advanceStep(1);
    try self.generated_ids.append(allocator, token);

    // Both checks below return on a Metal abort instead of ending the process, so the
    // handle needs an owner on the error path; a scoped `defer`, not errdefer + manual free.
    const val: i32 = blk: {
        const lazy = try self.sampleLazy(logits, .committed);
        defer _ = mlx.mlx_array_free(lazy);
        try mlx.check(mlx.mlx_array_eval(lazy));
        var v: i32 = 0;
        try mlx.check(mlx.mlx_array_item_int32(&v, lazy));
        break :blk v;
    };
    self.next_token_id = @intCast(val);

    self.mtp_serial_exit = .none;
    self.mtp_serial_clock = null;
    self.mtp_serial_warm = 0;
    self.mtp_regime_clock = null;
    return try mtpSerialOneToken(allocator, token);
}

/// One row's greedy verdict. Takes the committed-argmax wrapper, so no raw
/// argmax can reach a commit; `mtpGreedyVerdict` is the pure core.
pub fn mtpAcceptRowGreedy(am: CommittedArgmax, len: u32, drafts: []const u32, completion: u32, max_tokens: u32) !MtpRowVerdict {
    return mtpGreedyVerdict(try am.ids(len), drafts, completion, max_tokens);
}

pub fn mtpGreedyVerdict(am: []const i32, drafts: []const u32, completion: u32, max_tokens: u32) MtpRowVerdict {
    var accepted: u32 = 0;
    while (accepted < drafts.len and @as(u32, @intCast(am[accepted])) == drafts[accepted]) accepted += 1;
    accepted = mtp_plan.capAcceptedForTokenBudget(accepted, completion, max_tokens);
    return .{ .accepted = accepted, .next_pending = @intCast(am[accepted]) };
}

/// The committed increment of a verified row: its pending token plus the drafts it
/// accepted. The group's `[N]` retained-position vector is these values.
pub fn mtpRetained(accepted: u32) u32 {
    return 1 + accepted;
}

/// Full accept keeps the state the verify forward left, so it needs every draft
/// accepted AND the row's own width; a padded row always takes the rollback arm.
pub fn mtpFullAccept(accepted: u32, m: u32, verify_len: u32) bool {
    return accepted == m and verify_len == 1 + m;
}

pub fn nextMtp(self: *Generator, allocator: std.mem.Allocator) !?DrafterStepResult {
    var st = switch (try self.mtpRoundBegin(allocator)) {
        .done => |r| return r,
        .verify => |v| v,
        .open => |o| try self.mtpRoundContinue(allocator, o),
    };
    defer st.deinit(allocator);
    try self.mtpRoundVerify(&st);
    return self.mtpRoundFinish(allocator, &st);
}

/// Phases 0–2: serial-block dispatch, the draft chain, rollback anchors, the verify input.
pub fn mtpRoundBegin(self: *Generator, allocator: std.mem.Allocator) !MtpRoundBegin {
    if (self.done) return .{ .done = null };
    std.debug.assert(self.mtp != null);
    std.debug.assert(self.mtp_cache != null);
    std.debug.assert(self.has_last_hidden);
    if (group_planner.enabled() and self.mtp_planner_owned and self.mtp_hidden_stale) return error.MtpHiddenStale;
    // Release-enforced guard (issue #97): the MTP path cannot honor a
    // grammar constraint or logprobs (compiled-out asserts before).
    if (mtp_plan.mtpUnsupported(self.sampling, self.logprobs_n)) return error.SpecDecodeUnsupported;

    // `--max-mtp-ctx`, checked at the top of the round against prompt + generated. Sticky
    // (context only grows), routed through the same `spec_disabled_runtime` fallback.
    if (!self.spec_disabled_runtime and
        !mtp_mod.mtpCtxWithinLimit(generate_mod.max_mtp_ctx, self.mtpKvLen()))
    {
        log.info(
            "  mtp=off (ctx {d} > --max-mtp-ctx {d})\n",
            .{ self.mtpKvLen(), generate_mod.max_mtp_ctx },
        );
        self.spec_disabled_runtime = true;
        self.spec_disable_reason = .max_ctx;
    }

    // Adaptive serial re-entry: sticky, except when the request crossed into a different KV
    // bucket. `--max-mtp-ctx` (`.max_ctx`) never re-enters.
    if (self.spec_disabled_runtime and self.spec_disable_reason == .adaptive and
        mtpAdaptiveSerialEnabled() and self.mtpAdaptiveHeadMayResume())
    {
        const prev_bucket = self.mtp_adaptive.bucket;
        const prev_arm = self.mtp_adaptive.arm;
        const action = mtpAdaptiveSerialTickFor(&self.mtp_adaptive, &self.xfm.round_cost, self.mtpKvLen());
        self.mtpAdaptiveSyncWindow(prev_bucket, prev_arm);
        // Re-entry only when the head can prove it is in sync (else `qwen4MtpForward` returns
        // `error.MtpPositionGap`). A decline latches: the drift cannot shrink on its own.
        var may_reenter = action == .to_mtp;
        if (may_reenter) {
            if (self.mtpHeadDriftNow()) |drift| {
                if (drift != 0) {
                    log.info(
                        "  [mtp] adaptive: re-entry declined (head out of sync by {d}) -> serial for the rest of the request\n",
                        .{drift},
                    );
                    self.mtp_adaptive.declineReentry();
                    may_reenter = false;
                }
            }
        }
        if (may_reenter) {
            log.info(
                "  [mtp] adaptive: kv {d} crossed into bucket {s} -> mtp\n",
                .{ self.mtpKvLen(), round_cost.bucketName(self.xfm.round_cost.layout, self.xfm.round_cost.bucketOf(self.mtpKvLen())) },
            );
            self.spec_disabled_runtime = false;
            self.spec_disable_reason = .none;
            self.mtp_serial_exit = .drain;
        }
    }

    // Serial block: the runtime-disabled fallback, the bounded probe, or the ramp back.
    if (self.spec_disabled_runtime or self.mtp_serial_left > 0 or self.mtp_serial_exit != .none) {
        try self.mtpDetachHead(allocator, self.mtpSerialMayResume());
        self.mtpMaybeReleaseModuleHead();
        return .{ .done = try self.mtpSerialTick(allocator) };
    }

    if (try self.checkStop()) return .{ .done = null }; // t1 is this block's first emit: stop before drafting

    const head = self.mtp.?;
    // Cross-request EV seed: inherit the head's last healthy acceptance
    // surface so the controller plans from round 1 instead of re-warming
    // (~10 legacy rounds + a +1/round base climb — a third of a short
    // generation). Demotion stays instant (EMA decay + sticky disable are
    // per-request), so a workload change costs a few rounds, not the win.
    // A pinned depth (`--mtp-min-depth` == `--mtp-max-depth`) is a measurement
    // mode: every round drafts exactly n and the controller never plans, so a
    // seed must not be applied (nor, at deinit, published).
    if (self.mtp_ev_rounds == 0 and self.mtp_attempted == 0 and
        mtpAdaptiveEnabled() and mtpPinnedDepth() == null and
        (!group_planner.enabled() or !self.mtp_planner_owned))
    {
        if (head.evSeed()) |seed| {
            self.mtp_ev_accept = seed.accept;
            self.mtp_ev_m_lo_prev = @min(@max(seed.m_lo, 1), mtp_mod.MAX_DEPTH);
            self.mtp_ev_rounds = MTP_EV_WARMUP_ROUNDS;
        }
    }
    const tracing = mtpTraceEnabled();
    var ph: io_util.Stopwatch = undefined;
    if (tracing) {
        ph = io_util.Stopwatch.init(self.timer.io);
        // Close the scheduler gap opened at the previous round's return.
        if (self.mtp_gap_watch) |*gw| self.mtp_trace.add(.gap, gw.read());
        self.mtp_gap_watch = null;
    }
    // Always-on round wall-clock for the live-cost round EMA (the
    // denominator of the sync fraction). Read at both commit exits.
    const round_watch = io_util.Stopwatch.init(self.timer.io);

    // ── Phase 0/1: acquire this round's draft chain ──
    // Round plan: fixed mode (and EV warmup) is a single chunk at the
    // windowed adaptive depth; post-warmup EV mode plans a base chunk
    // m_lo plus a confidence-gated extension to m_hi (see the EV
    // controller section below). `chain.m` is the tokens actually
    // drafted this round — it grows from m_lo to m_hi iff the gate
    // clears at the chunk boundary below.
    //
    // Either the PREVIOUS round built + dispatched chunk A at its tail
    // (cross-round pipelining, `mtpMaybePreDraft` — the drafts are
    // already materializing on the GPU) or we build it here (round 1,
    // or a round with no successor state).
    //
    // No head-cache snapshot in either case: a snapshot refcount-shares
    // the head's KV buffer, which forces every draft append's
    // slice_update to copy-on-write the WHOLE history buffer (~268
    // MB/append at 64k). Rollback is truncate — offset-only — since
    // draft entries only ever append past chain.off0 (the committed
    // origin, resolved from the pending history stash when one exists).
    const chain: MtpPreDraft = if (self.mtp_pre_draft) |pd| blk: {
        self.mtp_pre_draft = null;
        // The pre-draft consumed the history stash when it was built.
        std.debug.assert(self.mtp_hist_stash == null);
        std.debug.assert(pd.t1 == self.next_token_id);
        break :blk pd;
    } else blk: {
        const plan_now = self.mtpRoundPlan();
        if (try self.mtpLookupChain(allocator, plan_now, self.next_token_id)) |lc| break :blk lc;
        var c = try self.mtpChainInit(allocator, plan_now, self.next_token_id);
        errdefer c.deinit(allocator);
        if (self.mtp_batch_head) {
            return .{ .open = .{
                .chain = c,
                .tracing = tracing,
                .ph = ph,
                .round_watch = round_watch,
            } };
        }
        const chain_lap = SubLap.start(tracing, self.timer.io);
        try self.mtpChainBuild(&c, 0, plan_now.m_lo);
        try mtpChainDispatch(&c, 0, plan_now.m_lo);
        chain_lap.stop(&self.mtp_trace, .chain);
        break :blk c;
    };
    return .{ .verify = try self.mtpRoundAfterChain(allocator, chain, tracing, ph, round_watch) };
}

pub fn mtpRoundContinue(self: *Generator, allocator: std.mem.Allocator, open: MtpRoundOpen) !MtpRoundState {
    var chain = open.chain;
    if (chain.n_drafted < chain.m) {
        const chain_lap = SubLap.start(open.tracing, self.timer.io);
        self.mtpChainBuild(&chain, chain.n_drafted, chain.m) catch |err| {
            chain.deinit(allocator);
            return err;
        };
        chain_lap.stop(&self.mtp_trace, .chain);
    }
    return self.mtpRoundAfterChain(allocator, chain, open.tracing, open.ph, open.round_watch);
}

pub fn mtpRoundAfterChain(
    self: *Generator,
    allocator: std.mem.Allocator,
    chain_in: MtpPreDraft,
    tracing: bool,
    ph_in: io_util.Stopwatch,
    round_watch: io_util.Stopwatch,
) !MtpRoundState {
    var chain = chain_in;
    errdefer chain.deinit(allocator);
    var ph = ph_in;
    const s = self.xfm.s;
    const plan = chain.plan;
    // A chain pre-drafted before this slot joined a group may be wider than the
    // cap: use its first `cap` drafts (the head's tail past them is truncated at
    // the next consume, like any rejected draft).
    // A lookup chain was sized by its own gate.
    const cap: u32 = if (self.mtp_group_cap > 0) self.mtp_group_cap else if (chain.lookup) chain.m else mtp_mod.MAX_DEPTH;
    const m_lo: u32 = @min(plan.m_lo, cap);
    const m_max: u32 = @min(plan.m_hi, cap);
    chain.m = @min(chain.m, cap);
    if (tracing) {
        self.mtp_trace.add(.draft, ph.read());
        ph.reset();
    }

    // ── chunk-A boundary: extension decision (two-chunk EV plans) ──
    // The one bounded sync of the round; near-free when the chain was
    // pre-drafted (ids + confidences already materialized).
    if (m_max > m_lo) {
        var sync_watch = io_util.Stopwatch.init(self.timer.io);
        const chain_ln = try readChainConfidence(chain.draft_arrs[0..m_lo], chain.conf_arrs.?[0..m_lo], s);
        const sync_ns = sync_watch.read();
        // Live sync-cost EMA: measured only on considered rounds, so it
        // holds the true per-sync cost (not the suppressed-round average).
        self.mtp_ev_sync_ms = mtpEmaMs(self.mtp_ev_sync_ms, sync_ns);
        if (tracing) {
            self.mtp_trace.add(.sync, sync_ns);
            ph.reset();
        }
        if (chain_ln >= plan.tau_ln) {
            chain.m = m_max;
            self.mtp_ext_rounds += 1;
            self.mtp_ext_dry_streak = 0;
            try self.mtpChainBuild(&chain, m_lo, m_max);
            try mtpChainDispatch(&chain, m_lo, m_max);
        } else {
            // Considered but the gate didn't clear — feeds the
            // dry-spell policy (mtpExtDryAllows).
            self.mtp_ext_dry_streak +|= 1;
        }
        if (tracing) {
            self.mtp_trace.add(.ext, ph.read());
            ph.reset();
        }
    }
    const m: u32 = chain.m;
    // Chain complete — release the chain hidden (only chunk B needed it).
    if (chain.h_chain) |h| {
        _ = mlx.mlx_array_free(h);
        chain.h_chain = null;
    }
    const draft_arrs = chain.draft_arrs;

    // ── Phase 2: record rollback anchors (NO snapshot on the GDN path) ──
    // A KVCache.snapshot() refcount-shares the KV buffers, which forces
    // verify's slice_update writes to COPY-on-write every full-attention
    // layer's WHOLE buffer — ~4.3 GB per round at 64k context, the
    // dominant round cost at long context. On a GDN trunk (every real MTP
    // target is a GDN hybrid) rollback needs only the pre-verify
    // LENGTH (KV truncate is offset-only; the stale tail past it is
    // unreachable) plus the verify pass's per-position SSM capture, so no
    // snapshot is taken at all. A hypothetical pure-attention target
    // (ssm_entries == null) keeps the proven snapshot + re-forward path.
    const kv_step_snap = self.ctx.cache.step;
    // MiMo is attention-only: a truncate IS its rollback, rings included.
    var kv_snap: ?transformer_mod.KVCacheSnapshot = if (self.ctx.ssm_entries != null or self.xfm.config.isMimo()) null else try self.ctx.cache.snapshot();
    errdefer if (kv_snap) |*snap| snap.deinit();
    const moe_seq_offset_snap = self.ctx.moe_seq_offset.*;

    // ── Phase 3: verify input [t1, drafts...] as one [1, 1+m] tensor ──
    const reshape_2d = [_]c_int{ 1, 1 };
    var t1_2d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(t1_2d);
    try mlx.check(mlx.mlx_reshape(&t1_2d, chain.t1_arr, &reshape_2d, 2, s));

    var verify_input = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(verify_input);
    {
        const drafts_2d = try allocator.alloc(mlx.mlx_array, m);
        var drafts_2d_n: usize = 0;
        defer {
            for (drafts_2d[0..drafts_2d_n]) |arr| _ = mlx.mlx_array_free(arr);
            allocator.free(drafts_2d);
        }
        for (draft_arrs[0..m], drafts_2d) |dlazy, *out| {
            out.* = mlx.mlx_array_new();
            drafts_2d_n += 1;
            try mlx.check(mlx.mlx_reshape(out, dlazy, &reshape_2d, 2, s));
        }
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        _ = mlx.mlx_vector_array_append_value(vec, t1_2d);
        for (drafts_2d) |arr| _ = mlx.mlx_vector_array_append_value(vec, arr);
        try mlx.check(mlx.mlx_concatenate_axis(&verify_input, vec, 1, s));
    }

    return .{
        .chain = chain,
        .tracing = tracing,
        .ph = ph,
        .round_watch = round_watch,
        .kv_step_snap = kv_step_snap,
        .kv_snap = kv_snap,
        .moe_seq_offset_snap = moe_seq_offset_snap,
        .verify_input = verify_input,
        .verify_len = 1 + m,
    };
}

/// Phase 3 of a solo round: the trunk verify forward over `[t1, drafts…]`.
/// A batched group replaces this with one forward for every slot (`mtpGroupVerify`).
pub fn mtpRoundVerify(self: *Generator, st: *MtpRoundState) !void {
    const xfm = self.xfm;
    const tracing = st.tracing;
    var new_hidden = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(new_hidden);
    var verify_hidden_all = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(verify_hidden_all);
    // Enable per-position SSM capture for the verify pass on a GDN trunk
    // so partial accept can roll back without re-forwarding the accepted
    // prefix (mirrors the serial path — the re-forward re-runs the 48-layer
    // sequential recurrence AND a full trunk weight read, and at depth > 1
    // MOST rounds are partial, so it dominated the round cost).
    self.ctx.capture_ssm_seq = self.ctx.ssm_entries != null;
    // DIAGNOSTIC (SUSHI_MTP_TRACE_SYNC=1): drain the GPU before the
    // verify build so the `sync` lap shows the pending lazy work (draft
    // chain, rollback, commit) and `verify` the forward alone.
    if (tracing and transformer_mod.diagEnvOnCached(&mtp_trace_sync_env, "SUSHI_MTP_TRACE_SYNC")) {
        var sync_watch = io_util.Stopwatch.init(self.timer.io);
        try mlx.check(mlx.mlx_array_eval(st.verify_input));
        self.mtp_trace.add(.sync, sync_watch.read());
        st.ph.reset();
    }
    // Captures the post-final-norm hidden at the LAST position (next
    // round's h_prev) AND all 1+m positions (history re-append).
    //
    // qwen4_exp: the n-gram PLE at trunk layer 1 is a HOST gather keyed on
    // these ids, and every id past t1 is still a lazy draft — gathering
    // eagerly parks the build of layers 2..N behind the whole draft chain
    // finishing on the GPU. Defer the leaf so the graph builds while the
    // chain runs, then sync ONCE (`flushDeferredPle`, below) before Phase
    // 4 evaluates anything. Other arches never set `ple_pending`.
    self.ctx.ple_defer = true;
    self.ctx.verify_rows = xfm.config.isMimo();
    defer self.ctx.verify_rows = false;
    var verify_logits = xfm.forwardWithCaptureAll(&self.ctx, st.verify_input, &new_hidden, &verify_hidden_all) catch |e| {
        self.ctx.ple_defer = false;
        self.ctx.capture_ssm_seq = false;
        xfm.discardDeferredPle(&self.ctx);
        return e;
    };
    errdefer _ = mlx.mlx_array_free(verify_logits);
    self.ctx.ple_defer = false;
    self.ctx.capture_ssm_seq = false;
    xfm.verify_laps.ple_sync_ns = 0;
    if (tracing) {
        const build_ns = st.ph.read();
        self.mtp_trace.add(.verify, build_ns);
        self.mtp_trace.addSub(.build, build_ns);
        st.ph.reset();
    }
    const ple_lap = SubLap.start(tracing, self.timer.io);
    defer if (tracing) {
        ple_lap.stop(&self.mtp_trace, .ple);
        self.mtp_trace.addSub(.ple_sync, xfm.verify_laps.ple_sync_ns);
    };
    // Fill the deferred PLE leaf: the one host read of the verify ids, and
    // the point at which the entry's n-gram history + `spec_ple_tokens`
    // advance. Must precede BOTH the first eval of anything downstream of
    // the leaf (Phase 4) and `ssmRollbackFromCapture` (Phase 5).
    errdefer xfm.discardDeferredPle(&self.ctx);
    try xfm.flushDeferredPle(&self.ctx);
    _ = try self.thinkShiftRows(&verify_logits, st.verify_input);
    _ = try self.penaltyShiftRows(&verify_logits, st.verify_input);
    st.verify_logits = verify_logits;
    st.new_hidden = new_hidden;
    st.verify_hidden_all = verify_hidden_all;
}

pub fn mtpGroupVerify(gens: []const *Generator, states: []const *MtpRoundState) !void {
    const n = gens.len;
    std.debug.assert(n == states.len);
    std.debug.assert(n >= 1 and n <= MTP_GROUP_ROWS_MAX);
    const xfm = gens[0].xfm;
    for (gens[1..n]) |gen| {
        if (gen.xfm != xfm) return error.MtpGroupModelMismatch;
    }
    var rows: [MTP_GROUP_ROWS_MAX]mlx.mlx_array = undefined;
    var ctxs: [MTP_GROUP_ROWS_MAX]*ForwardCtx = undefined;
    var out_logits: [MTP_GROUP_ROWS_MAX]mlx.mlx_array = undefined;
    var out_last: [MTP_GROUP_ROWS_MAX]mlx.mlx_array = undefined;
    var out_all: [MTP_GROUP_ROWS_MAX]mlx.mlx_array = undefined;
    for (gens[0..n], states[0..n], 0..) |gen, st, i| {
        rows[i] = st.verify_input;
        ctxs[i] = &gen.ctx;
    }
    var rollback: [MTP_GROUP_ROWS_MAX]Transformer.VerifyRollback = undefined;
    const retain_rollback = n > 1 and xfm.qwen4 != null;
    for (gens, states) |gen, st| st.verify_ctx = &gen.ctx;
    var row_ns: [MTP_GROUP_ROWS_MAX]u64 = undefined;
    var tracing_any = false;
    for (states[0..n]) |st| {
        if (st.tracing) tracing_any = true;
    }
    try xfm.forwardRowAxisVerify(rows[0..n], ctxs[0..n], out_logits[0..n], out_last[0..n], out_all[0..n], if (tracing_any) row_ns[0..n] else null, if (retain_rollback) rollback[0..n] else null);
    var shifted_any = false;
    for (gens[0..n], states[0..n], 0..) |gen, st, i| {
        if (retain_rollback) st.verify_rollback = rollback[i];
        st.verify_logits = out_logits[i];
        st.new_hidden = out_last[i];
        st.verify_hidden_all = out_all[i];
        // Row 0 owns the joined block for the round when the lm_head ran once.
        if (i == 0) st.group_logits = xfm.takeJoinedVerifyLogits();
        if (try gen.thinkShiftRows(&st.verify_logits, st.verify_input)) shifted_any = true;
        if (try gen.penaltyShiftRows(&st.verify_logits, st.verify_input)) shifted_any = true;
        if (st.tracing) {
            gen.mtp_trace.add(.verify, row_ns[i]);
            const laps = xfm.verify_laps;
            gen.mtp_trace.addSub(.build, laps.build_ns);
            gen.mtp_trace.addSub(.ple, laps.ple_ns);
            gen.mtp_trace.addSub(.ple_sync, laps.ple_sync_ns);
            st.ph.reset();
        }
    }
    // The joined block holds the lm_head's unshifted rows.
    if (shifted_any and states[0].group_logits.ctx != null) {
        _ = mlx.mlx_array_free(states[0].group_logits);
        states[0].group_logits = .{ .ctx = null };
    }
}

pub fn mtpGroupAccept(gens: []const *Generator, states: []const *MtpRoundState) !void {
    const n = gens.len;
    std.debug.assert(n == states.len);
    if (n == 0) return;
    var any_tracing = false;
    for (states[0..n]) |st| {
        if (st.tracing) any_tracing = true;
    }
    const dispatch_lap = SubLap.start(any_tracing, gens[0].timer.io);
    defer if (dispatch_lap.read()) |taken| {
        for (gens[0..n], states[0..n]) |gen, st| {
            if (st.tracing) gen.mtp_trace.addSub(.dispatch, taken);
        }
    };
    for (gens[1..n]) |gen| {
        if (gen.xfm != gens[0].xfm) return error.MtpGroupModelMismatch;
    }
    const s = gens[0].xfm.s;
    const eval_vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(eval_vec);
    errdefer for (states[0..n]) |st| {
        st.verify_argmax.deinit();
        st.releasePreparedAccept();
    };
    for (gens[0..n], states[0..n]) |gen, st| {
        for (st.chain.draft_arrs[0..st.chain.m]) |arr| _ = mlx.mlx_vector_array_append_value(eval_vec, arr);
        _ = mlx.mlx_vector_array_append_value(eval_vec, st.new_hidden);
        _ = mlx.mlx_vector_array_append_value(eval_vec, st.verify_hidden_all);
        if (!sampler.isGreedyTemperature(gen.sampling.temperature)) continue;
        st.verify_argmax = try verifyArgmax(st.verify_logits, gen.sampling.suppress_mask, s);
        _ = mlx.mlx_vector_array_append_value(eval_vec, st.verify_argmax.lazy());
    }
    try mtpGroupSampledAccept(gens[0..n], states[0..n], eval_vec, s);
    try mlx.check(mlx.mlx_async_eval(eval_vec));
}

/// Whether ONE filtered block can serve every sampled row of a group: the
/// filter is per REQUEST, and one categorical cannot carry a per-row key.
pub fn mtpGroupAcceptBatchable(params: []const sampler.SamplingParams) bool {
    if (params.len == 0) return false;
    return sampler.sampleRowsHomogeneous(params) and !sampler.sampleRowsAnySeed(params);
}

/// Phase 4 for the sampled rows of a group, built together: ONE
/// `probsAllPositions` over their joined verify logits — the block the
/// lm_head already projected in one go — and each row's accept graph on the
/// group's single async eval, instead of a build-wait-build staircase per
/// row. Declines (every row then builds its own in `mtpRoundFinish`) when
/// the rows do not share one sampler, any row is seeded, or a row was
/// padded past `1 + m`.
pub fn mtpGroupSampledAccept(
    gens: []const *Generator,
    states: []const *MtpRoundState,
    eval_vec: mlx.mlx_vector_array,
    s: mlx.mlx_stream,
) !void {
    var rows: [MTP_GROUP_ROWS_MAX]usize = undefined;
    var params: [MTP_GROUP_ROWS_MAX]sampler.SamplingParams = undefined;
    var k: usize = 0;
    for (gens, states, 0..) |gen, st, i| {
        if (sampler.isGreedyTemperature(gen.sampling.temperature)) continue;
        if (st.chain.m == 0 or st.verify_len != 1 + st.chain.m) return;
        rows[k] = i;
        params[k] = gen.sampling;
        k += 1;
    }
    if (k == 0 or !mtpGroupAcceptBatchable(params[0..k])) return;

    // Offsets of each sampled row in the block, ascending: the projection's
    // own `offset += row.seq` walk, which is why the joined array can stand
    // in for the concatenation whenever every row of it is sampled.
    var offs: [MTP_GROUP_ROWS_MAX]c_int = undefined;
    var total: c_int = 0;
    for (rows[0..k], 0..) |i, j| {
        offs[j] = total;
        total += @intCast(states[i].verify_len);
    }

    const joined = states[0].group_logits;
    const use_joined = k == gens.len and joined.ctx != null and
        mlx.getShape(joined).len == 3 and mlx.getShape(joined)[1] == total;
    var block = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(block);
    if (use_joined) {
        try mlx.check(mlx.mlx_array_set(&block, joined));
    } else {
        var parts: [MTP_GROUP_ROWS_MAX]mlx.mlx_array = undefined;
        for (rows[0..k], 0..) |i, j| parts[j] = states[i].verify_logits;
        const vec = mlx.mlx_vector_array_new_data(&parts, k);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&block, vec, 1, s));
    }

    const probs = try sampler.groupProbsBlock(block, params[0], s);
    defer _ = mlx.mlx_array_free(probs);
    for (rows[0..k], 0..) |i, j| {
        const st = states[i];
        const row_probs = try sampler.groupProbsRow(probs, offs[j], st.verify_len, s);
        defer _ = mlx.mlx_array_free(row_probs);
        // The row's OWN acceptance mode, threshold and draw index: only the
        // filter is shared, so a seeded row keeps its own key.
        const gen = gens[i];
        const bg = try gen.acceptGraphFor(st.chain.lookup)(
            row_probs,
            st.chain.draft_arrs[0..st.chain.m],
            if (st.chain.q_probs) |qs| qs[0..st.chain.m] else null,
            st.chain.m,
            gen.mtp_accept_param,
            gen.mtpSamplingDraw(gen.sampling),
            s,
        );
        st.accept_p = bg.accept_p;
        st.accept_q = bg.accept_q;
        st.accept_defer = bg.accept_defer;
        st.corr = bg.corr_samples;
        _ = mlx.mlx_vector_array_append_value(eval_vec, st.accept_p);
        if (st.accept_q.ctx != null) _ = mlx.mlx_vector_array_append_value(eval_vec, st.accept_q);
        if (st.accept_defer.ctx != null) _ = mlx.mlx_vector_array_append_value(eval_vec, st.accept_defer);
        _ = mlx.mlx_vector_array_append_value(eval_vec, st.corr);
    }
}

/// Phases 4–5 of a round: accept, stash the history, commit or roll back.
/// `st.verify_len` may exceed `1 + m` when a batched group padded the row;
/// then even a full accept takes the rollback arm (KV + SSM clamp to `1 + m`).
pub fn mtpStopPrefix(drafts: []const u32, accepted: u32, eos: []const u32) struct { accepted: u32, stop: ?u32 } {
    for (drafts[0..accepted], 0..) |token, i| {
        if (isEosId(token, eos)) return .{ .accepted = @intCast(i), .stop = token };
    }
    return .{ .accepted = accepted, .stop = null };
}

// Cap accepted drafts with capAcceptedForTokenBudget before committing.
pub fn mtpRoundFinish(self: *Generator, allocator: std.mem.Allocator, st: *MtpRoundState) !?DrafterStepResult {
    var failed = false;
    defer if (!failed) {
        if (st.verify_rollback) |*saved| saved.deinit();
        st.verify_rollback = null;
        st.verify_ctx = null;
    };
    errdefer failed = true;
    const xfm = self.xfm;
    const s = xfm.s;
    const tracing = st.tracing;
    var ph = st.ph;
    var round_watch = st.round_watch;
    const chain = &st.chain;
    const plan = chain.plan;
    const m_lo: u32 = plan.m_lo;
    const m_max: u32 = plan.m_hi;
    const m: u32 = chain.m;
    const observed_limit = @min(m, self.max_tokens -| self.completion_tokens -| 1);
    const t1: u32 = chain.t1;
    const mtp_off0: usize = chain.off0;
    const drafts = chain.drafts;
    const draft_arrs = chain.draft_arrs;
    const q_probs = chain.q_probs;
    const kv_step_snap = st.kv_step_snap;
    const moe_seq_offset_snap = st.moe_seq_offset_snap;
    var verify_logits = st.verify_logits;
    st.verify_logits = .{ .ctx = null };
    errdefer _ = mlx.mlx_array_free(verify_logits);
    var new_hidden = st.new_hidden;
    st.new_hidden = .{ .ctx = null };
    errdefer _ = mlx.mlx_array_free(new_hidden);
    const verify_hidden_all = st.verify_hidden_all;
    // Always free the transient capture buffers before returning, however
    // we exit this round (full accept, partial accept, or error).
    defer if (self.ctx.ssm_entries) |entries| {
        for (entries) |*entry| transformer_mod.ssmFreeSpecCapture(entry);
    };
    // A lookup round is no MTP width sample, nor is the MTP round after one:
    // its first head step consumes a wider history stash than MTP makes.
    const times_round = !chain.lookup and !self.mtp_after_lookup;
    self.mtp_after_lookup = chain.lookup;
    if (chain.lookup) {
        self.mtp_lookup_rounds += 1;
        self.mtp_lookup_drafted += m;
    } else {
        self.mtp_attempted += 1;
        self.mtp_drafted_tokens += m;
    }

    // ── Phase 4: decide longest accepted prefix ──
    // Stochastic path is fully BATCHED: accept probabilities for every
    // draft AND a candidate correction token for every possible reject
    // position are built lazily (draft ids stay lazy arrays — never read
    // on the CPU inside a graph-building loop), then ONE async eval
    // realizes the whole round. The old per-draft probAt()/sampleResidual()
    // calls cost one GPU round-trip sync EACH — 3-5 syncs per round that
    // stalled the pipeline for milliseconds while the GPU sat idle.
    const stochastic = !sampler.isGreedyTemperature(self.sampling.temperature);
    const vl_shape = mlx.getShape(verify_logits);

    var accept_p_vec = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(accept_p_vec);
    var accept_q_vec: mlx.mlx_array = .{ .ctx = null };
    defer if (accept_q_vec.ctx != null) {
        _ = mlx.mlx_array_free(accept_q_vec);
    };
    var accept_defer_vec: mlx.mlx_array = .{ .ctx = null };
    defer if (accept_defer_vec.ctx != null) {
        _ = mlx.mlx_array_free(accept_defer_vec);
    };
    // ONE [1+m] pre-sampled correction array, indexed at the realized accept.
    var corr_batch: mlx.mlx_array = .{ .ctx = null };
    defer if (corr_batch.ctx != null) {
        _ = mlx.mlx_array_free(corr_batch);
    };

    // The group already built and dispatched this row's Phase 4 with the rest
    // of the round (`mtpGroupSampledAccept`); nothing here but the handover.
    const prepared = stochastic and st.accept_p.ctx != null;
    if (prepared) {
        _ = mlx.mlx_array_free(accept_p_vec);
        accept_p_vec = st.accept_p;
        accept_q_vec = st.accept_q;
        accept_defer_vec = st.accept_defer;
        corr_batch = st.corr;
        st.accept_p = .{ .ctx = null };
        st.accept_q = .{ .ctx = null };
        st.accept_defer = .{ .ctx = null };
        st.corr = .{ .ctx = null };
    } else if (stochastic) {
        // A group-padded row filters only the 1+m positions it reads.
        var live_logits = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(live_logits);
        try mlx.check(mlx.mlx_slice(&live_logits, verify_logits, &.{ 0, 0, 0 }, 3, &.{ vl_shape[0], @intCast(1 + m), vl_shape[2] }, 3, &.{ 1, 1, 1 }, 3, s));
        const probs_all = try sampler.probsAllPositions(live_logits, self.sampling, s);
        defer _ = mlx.mlx_array_free(probs_all);
        const bg = try self.acceptGraphFor(chain.lookup)(
            probs_all,
            draft_arrs[0..m],
            if (q_probs) |qs| qs[0..m] else null,
            m,
            self.mtp_accept_param,
            self.mtpSamplingDraw(self.sampling),
            s,
        );
        corr_batch = bg.corr_samples;
        _ = mlx.mlx_array_free(accept_p_vec);
        accept_p_vec = bg.accept_p;
        accept_q_vec = bg.accept_q;
        accept_defer_vec = bg.accept_defer;
    }

    const group_argmax = st.verify_argmax.present();
    var verify_argmax = st.verify_argmax;
    st.verify_argmax = .{};
    defer verify_argmax.deinit();
    if (!group_argmax and !stochastic) {
        verify_argmax = try verifyArgmax(verify_logits, self.sampling.suppress_mask, s);
    }
    _ = mlx.mlx_array_free(verify_logits);
    verify_logits = .{ .ctx = null };
    if (tracing) {
        self.mtp_trace.add(.corr, ph.read());
        ph.reset();
    }

    // ── Phase 4b: one batched async eval for the whole round ──
    // A prepared row rode the group's eval; everything here is already on it.
    const dispatch_lap = SubLap.start(tracing, self.timer.io);
    if (!prepared) {
        const eval_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(eval_vec);
        for (draft_arrs[0..m]) |arr| _ = mlx.mlx_vector_array_append_value(eval_vec, arr);
        if (stochastic) {
            _ = mlx.mlx_vector_array_append_value(eval_vec, accept_p_vec);
            if (accept_q_vec.ctx != null) _ = mlx.mlx_vector_array_append_value(eval_vec, accept_q_vec);
            if (accept_defer_vec.ctx != null) _ = mlx.mlx_vector_array_append_value(eval_vec, accept_defer_vec);
            _ = mlx.mlx_vector_array_append_value(eval_vec, corr_batch);
        } else {
            _ = mlx.mlx_vector_array_append_value(eval_vec, verify_argmax.lazy());
        }
        _ = mlx.mlx_vector_array_append_value(eval_vec, new_hidden);
        _ = mlx.mlx_vector_array_append_value(eval_vec, verify_hidden_all);
        try mlx.check(mlx.mlx_async_eval(eval_vec));
    }
    dispatch_lap.stop(&self.mtp_trace, .dispatch);
    const wait_lap = SubLap.start(tracing, self.timer.io);
    for (draft_arrs[0..m], 0..) |arr, idx| {
        try mlx.check(mlx.mlx_array_eval(arr));
        var v: i32 = 0;
        try mlx.check(mlx.mlx_array_item_int32(&v, arr));
        drafts[idx] = @intCast(v);
    }
    if (!stochastic) {
        // Separate graph branch from the draft chain — force it before
        // bulk-reading (see the v26.5.6 0%-acceptance note in nextDrafter).
        try mlx.check(mlx.mlx_array_eval(verify_argmax.lazy()));
    }
    wait_lap.stop(&self.mtp_trace, .wait);

    var accepted: u32 = 0;
    var greedy_next_pending: u32 = 0;
    if (stochastic) {
        // Sharp drafts: full Leviathan ratio min(1, p/q) against the
        // proposal's own density. Greedy-forced drafts keep the exact
        // one-hot rule min(1, target_p).
        try mlx.check(mlx.mlx_array_eval(accept_p_vec));
        const p_data = mlx.mlx_array_data_float32(accept_p_vec) orelse {
            return error.MlxArrayDataNull;
        };
        var q_data: ?[]const f32 = null;
        if (accept_q_vec.ctx != null) {
            try mlx.check(mlx.mlx_array_eval(accept_q_vec));
            const q_ptr = mlx.mlx_array_data_float32(accept_q_vec) orelse {
                return error.MlxArrayDataNull;
            };
            q_data = q_ptr[0..m];
        }
        var defer_data: ?[]const bool = null;
        if (accept_defer_vec.ctx != null) {
            try mlx.check(mlx.mlx_array_eval(accept_defer_vec));
            const defer_ptr = mlx.mlx_array_data_bool(accept_defer_vec) orelse return error.MlxArrayDataNull;
            defer_data = defer_ptr[0..m];
        }
        accepted = self.acceptPrefixFor(chain.lookup)(p_data[0..m], q_data, defer_data, &self.prng);
        accepted = mtp_plan.capAcceptedForTokenBudget(
            accepted,
            self.completion_tokens,
            self.max_tokens,
        );
    } else {
        const verdict = try mtpAcceptRowGreedy(verify_argmax, st.verify_len, drafts[0..m], self.completion_tokens, self.max_tokens);
        accepted = verdict.accepted;
        greedy_next_pending = verdict.next_pending;
    }

    const stop_prefix = mtpStopPrefix(drafts, accepted, if (self.mtp_planner_owned) self.eos_token_ids else &.{});
    if (stop_prefix.stop) |token| log.debug("[mtp-stop] accepted EOS token={d} accepted={d}->{d}\n", .{ token, accepted, stop_prefix.accepted });
    accepted = stop_prefix.accepted;
    var next_pending: u32 = blk: {
        if (stop_prefix.stop) |stop| break :blk stop;
        if (stochastic) {
            try mlx.check(mlx.mlx_array_eval(corr_batch));
            const d = mlx.mlx_array_data_int32(corr_batch) orelse {
                return error.MlxArrayDataNull;
            };
            break :blk @intCast(d[accepted]);
        } else {
            break :blk greedy_next_pending;
        }
    };
    // A round that carries the thought's closer ends on it, so the state stands where the restore point sits.
    if (self.thought_point != null and accepted < 64) {
        var path: [64]u32 = undefined;
        path[0] = t1;
        @memcpy(path[1 .. 1 + accepted], drafts[0..accepted]);
        if (self.thoughtRoundCut(path[0 .. 1 + accepted], moe_seq_offset_snap)) |keep| {
            const cut = mtp_plan.cutMtpRound(drafts, accepted, next_pending, keep);
            accepted = cut.accepted;
            next_pending = cut.pending;
        }
    }

    if (tracing) {
        self.mtp_trace.add(.eval, ph.read());
        ph.reset();
    }
    log.debug("  [mtp-round] off0={d} t1={d} m={d}/{d} drafts={any} accepted={d}\n", .{ mtp_off0, t1, m, m_max, drafts[0..m], accepted });

    // ── Phase 5a: stash the committed history for a DEFERRED append ──
    // The old shape paid a second head forward (appendHistory) here every
    // round to rebuild history from true verify hiddens — then the next
    // round's first draft re-entered the head anyway. Instead, stash the
    // (tokens, hiddens) pair — tokens as [t1, drafts[0..accepted]], the
    // hiddens the SAME concat of h_prev + ORIGINAL verify hiddens the old
    // appendHistory received — and fold the append into that first draft
    // step (the i==0 merged branch above). The head cache keeps this
    // round's draft tail past mtp_off0 until the consume-time truncate;
    // nothing reads it in between. Rounds with no successor (EOS/length/
    // runtime disable) never pay for the append; deinit frees the stash.
    {
        std.debug.assert(self.mtp_hist_stash == null);
        const n_commit: usize = accepted;
        const ids_i32 = try allocator.alloc(i32, 1 + n_commit);
        defer allocator.free(ids_i32);
        ids_i32[0] = @intCast(t1);
        for (drafts[0..n_commit], 0..) |d, idx| ids_i32[1 + idx] = @intCast(d);
        const id_shape = [_]c_int{@intCast(1 + n_commit)};
        const stash_ids = mlx.mlx_array_new_data(ids_i32.ptr, &id_shape, 1, .int32);
        errdefer _ = mlx.mlx_array_free(stash_ids);

        var hist_hidden = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(hist_hidden);
        if (n_commit == 0) {
            try mlx.check(mlx.mlx_array_set(&hist_hidden, self.last_hidden));
        } else {
            const vh_shape = mlx.getShape(verify_hidden_all);
            var vh_slice = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(vh_slice);
            const start = [_]c_int{ 0, 0, 0 };
            const stop = [_]c_int{ 1, @intCast(n_commit), vh_shape[2] };
            const strides = [_]c_int{ 1, 1, 1 };
            try mlx.check(mlx.mlx_slice(&vh_slice, verify_hidden_all, &start, 3, &stop, 3, &strides, 3, s));
            const vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(vec);
            _ = mlx.mlx_vector_array_append_value(vec, self.last_hidden);
            _ = mlx.mlx_vector_array_append_value(vec, vh_slice);
            try mlx.check(mlx.mlx_concatenate_axis(&hist_hidden, vec, 1, s));
        }
        self.mtp_hist_stash = .{
            .ids = stash_ids,
            .hidden = hist_hidden,
            .n = 1 + n_commit,
            .off0 = mtp_off0,
        };
        for (ids_i32, 0..) |id, idx| self.mtp_hist_stash.?.host_ids[idx] = @intCast(id);
    }
    if (tracing) {
        self.mtp_trace.add(.hist, ph.read());
        ph.reset();
    }

    // ── Phase 5b: commit / rollback the trunk ──
    st.retained = mtpRetained(accepted);
    if (mtpFullAccept(accepted, m, st.verify_len)) {
        const tokens = try allocator.alloc(u32, 1 + m);
        tokens[0] = t1;
        for (drafts[0..m], 0..) |d, idx| tokens[1 + idx] = d;

        try self.generated_ids.append(allocator, t1);
        for (drafts[0..m]) |d| try self.generated_ids.append(allocator, d);

        if (self.has_last_hidden) _ = mlx.mlx_array_free(self.last_hidden);
        self.last_hidden = new_hidden;
        self.has_last_hidden = true;
        new_hidden = .{ .ctx = null };

        self.mtpRoundAcceptObserve(chain.lookup, m, m);
        self.next_token_id = next_pending;
        self.advanceStep(1 + m);

        if (!chain.lookup) {
            self.mtpPlannerObserve(observed_limit, m);
            if (!group_planner.enabled() or self.mtp_planner_width == null) {
                if (mtpAdaptiveEnabled()) self.updateMtpEvRound(m, m) else self.updateMtpDepth(m, m);
            }
        }
        if (tracing) {
            self.mtp_trace.add(.commit, ph.read());
            ph.reset();
        }
        try self.mtpMaybePreDraft(allocator);
        if (tracing) {
            self.mtp_trace.add(.predraft, ph.read());
            self.mtp_gap_watch = io_util.Stopwatch.init(self.timer.io);
        }
        self.mtpTraceRoundEnd(m, m, m_lo);
        if (times_round) {
            self.mtpRoundEndObserve(m, m + 1, m_max > m_lo, m_lo, plan.width_trial, @as(f32, @floatFromInt(round_watch.read())) / @as(f32, std.time.ns_per_ms));
            self.mtp_ev_round_ms = mtpEmaMs(self.mtp_ev_round_ms, round_watch.read());
        } else {
            if (chain.lookup) self.mtpLookupObserve(m, m + 1);
            self.mtpRoundUntimed();
        }
        self.thoughtPointTake();
        return DrafterStepResult{
            .tokens = tokens,
            .accepted_tokens = m,
        };
    }

    // Partial accept: roll back the trunk. On a GDN trunk the verify pass
    // captured per-position SSM/conv state, so roll back by truncating the
    // KV cache to the accepted length and slicing the capture — NO
    // re-forward of the accepted prefix (mirrors the serial fast path; the
    // re-forward is a full trunk weight read, and at depth > 1 most rounds
    // are partial, so it dominated the round cost). The next round's
    // h_prev is the TRUE verify hidden at the last committed position
    // (input index `accepted`), which forwardWithCaptureAll captured.
    // Non-GDN archs keep the proven restore + re-forward fallback.
    _ = mlx.mlx_array_free(new_hidden);
    new_hidden = .{ .ctx = null };

    const gdn_captured = if (self.ctx.ssm_entries) |entries|
        entries.len > 0 and transformer_mod.ssmHasSpecCapture(&entries[0])
    else
        false;

    var re_new_hidden = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(re_new_hidden);
    if (self.ctx.ssm_entries == null and xfm.config.isMimo()) {
        const accepted_len: usize = 1 + @as(usize, accepted);
        try self.ctx.cache.truncate(moe_seq_offset_snap + accepted_len, s);
        self.ctx.moe_seq_offset.* = moe_seq_offset_snap + accepted_len;
        const vh_shape = mlx.getShape(verify_hidden_all);
        try mlx.check(mlx.mlx_slice(&re_new_hidden, verify_hidden_all, &.{ 0, @intCast(accepted), 0 }, 3, &.{ 1, @as(c_int, @intCast(accepted)) + 1, vh_shape[2] }, 3, &.{ 1, 1, 1 }, 3, s));
    } else if (gdn_captured) {
        const accepted_len: usize = 1 + @as(usize, accepted);
        // `truncate` overwrites cache.step with its length arg; on this
        // family cache.step is a stale counter the model never reads
        // (positioning is moe_seq_offset), so preserve the pre-verify
        // value — keeps prefix-cache kv_step bookkeeping identical to
        // the restore-based fallback (same rule as the serial path).
        try self.ctx.cache.truncate(moe_seq_offset_snap + accepted_len, s);
        self.ctx.cache.step = kv_step_snap;
        try self.rollbackSsmFromCapture(self.ctx.ssm_entries.?, accepted, st.verify_len, s);
        self.ctx.moe_seq_offset.* = moe_seq_offset_snap + accepted_len;

        const vh_shape = mlx.getShape(verify_hidden_all);
        const start = [_]c_int{ 0, @intCast(accepted), 0 };
        const stop = [_]c_int{ 1, @as(c_int, @intCast(accepted)) + 1, vh_shape[2] };
        const strides = [_]c_int{ 1, 1, 1 };
        try mlx.check(mlx.mlx_slice(&re_new_hidden, verify_hidden_all, &start, 3, &stop, 3, &strides, 3, s));
    } else if (st.kv_snap) |*snap| {
        try self.ctx.cache.restore(snap);
        self.ctx.moe_seq_offset.* = moe_seq_offset_snap;

        const re_seq_len: c_int = @intCast(1 + accepted);
        const re_input_buf = try allocator.alloc(i32, 1 + accepted);
        defer allocator.free(re_input_buf);
        re_input_buf[0] = @intCast(t1);
        for (drafts[0..accepted], 0..) |d, idx| re_input_buf[1 + idx] = @intCast(d);
        const re_shape = [_]c_int{ 1, re_seq_len };
        const re_input = mlx.mlx_array_new_data(re_input_buf.ptr, &re_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(re_input);

        const re_logits = try xfm.forwardWithCapture(&self.ctx, re_input, &re_new_hidden);
        _ = mlx.mlx_array_free(re_logits);
    } else {
        // GDN trunk whose verify pass produced no capture — cannot roll
        // back safely. Unreachable on real targets (every
        // GDN layer populates the capture when capture_ssm_seq is set);
        // pinned by tests/test_mtp_equivalence.sh.
        _ = mlx.mlx_array_free(re_new_hidden);
        re_new_hidden = .{ .ctx = null };
        return error.MtpRollbackUnavailable;
    }

    const tokens = try allocator.alloc(u32, 1 + accepted);
    tokens[0] = t1;
    for (drafts[0..accepted], 0..) |d, idx| tokens[1 + idx] = d;

    try self.generated_ids.append(allocator, t1);
    for (drafts[0..accepted]) |d| try self.generated_ids.append(allocator, d);

    if (self.has_last_hidden) _ = mlx.mlx_array_free(self.last_hidden);
    self.last_hidden = re_new_hidden;
    self.has_last_hidden = true;
    re_new_hidden = .{ .ctx = null };

    self.mtpRoundAcceptObserve(chain.lookup, m, accepted);
    self.next_token_id = next_pending;
    self.advanceStep(1 + accepted);

    if (!chain.lookup) {
        self.mtpPlannerObserve(if (stop_prefix.stop != null) accepted else observed_limit, accepted);
        if (!group_planner.enabled() or self.mtp_planner_width == null) {
            if (mtpAdaptiveEnabled()) self.updateMtpEvRound(m, accepted) else self.updateMtpDepth(m, accepted);
        }
    }
    if (tracing) {
        self.mtp_trace.add(.commit, ph.read());
        ph.reset();
    }
    try self.mtpMaybePreDraft(allocator);
    if (tracing) {
        self.mtp_trace.add(.predraft, ph.read());
        self.mtp_gap_watch = io_util.Stopwatch.init(self.timer.io);
    }
    self.mtpTraceRoundEnd(m, accepted, m_lo);
    if (times_round) {
        self.mtpRoundEndObserve(m, accepted + 1, m_max > m_lo, m_lo, plan.width_trial, @as(f32, @floatFromInt(round_watch.read())) / @as(f32, std.time.ns_per_ms));
        self.mtp_ev_round_ms = mtpEmaMs(self.mtp_ev_round_ms, round_watch.read());
    } else {
        if (chain.lookup) self.mtpLookupObserve(m, accepted + 1);
        self.mtpRoundUntimed();
    }
    self.thoughtPointTake();
    return DrafterStepResult{
        .tokens = tokens,
        .accepted_tokens = accepted,
    };
}

/// Greedy drafts for THIS request. `top_k == 1` is a deterministic target
/// whose filtered p is one-hot, so its argmax proposal is already optimal.
/// A greedy target keeps the argmax proposal (the temp-0 identity contract); a sampled
/// target proposes from the draft sampler, where `min(1, p[argmax])` collapses on a flat row.
pub fn mtpDraftGreedyFor(target: sampler.SamplingParams) bool {
    return target.temperature <= MTP_SAMPLED_PROPOSAL_MIN_TEMP or target.top_k == 1;
}

pub fn mtpDraftTempFor(config: *const model_mod.ModelConfig) f32 {
    return if (std.mem.eql(u8, config.model_type, "qwen4_exp")) MTP_DRAFT_TEMP_QWEN4 else MTP_DRAFT_TEMP;
}

/// Draft-proposal sampler for a round: greedy targets keep greedy drafts
/// (temp-0 identity contract); stochastic targets draft from the fixed
/// sharpened distribution unless greedy is forced. The constants are not
/// tunable: swept 2026-08-04 on the oQ4e head (prose@0.6, forced depth 3,
/// sampled drafts) at draft temp 0.6/0.7/0.85/1.0 and per-draft
/// acceptance was flat at 49.7-56.9% — no arm beat 0.6.
pub fn mtpDraftSamplingFor(target: sampler.SamplingParams, force_greedy: bool, draft_temp: f32) sampler.SamplingParams {
    var d = target;
    if (force_greedy or sampler.isGreedyTemperature(target.temperature)) {
        d.temperature = 0.0;
        return d;
    }
    d.temperature = draft_temp;
    d.top_p = MTP_DRAFT_TOP_P;
    d.top_k = MTP_DRAFT_TOP_K;
    return d;
}

/// Greedy drafts at chain depth `step` (0 = the round's first draft). The
/// greedy tail samples depth 0 of a sampled request and takes the argmax after it.
pub fn mtpDraftGreedyAt(target: sampler.SamplingParams, greedy_tail: bool, step: u32) bool {
    return mtpDraftGreedyFor(target) or (greedy_tail and step > 0);
}

/// This request's draft sampler at chain depth `step`: the per-request
/// proposal mode, the greedy tail and the loaded family's draft temperature,
/// resolved in ONE place so no call site can read a different proposal than
/// the one the verify's q came from.
pub fn mtpDraftSampling(self: *const Generator, step: u32) sampler.SamplingParams {
    return mtpDraftSamplingFor(
        self.sampling,
        mtpDraftGreedyAt(self.sampling, self.mtp_greedy_tail, step),
        mtpDraftTempFor(&self.xfm.config),
    );
}

/// Full Leviathan acceptance ratio; q clamped so a sampled draft (q > 0
/// by construction) can never divide by an underflowed zero.
pub fn specAcceptProb(p: f32, q_draft: f32) f32 {
    return @min(1.0, p / @max(q_draft, 1e-12));
}

/// KV length the next verify forward reads.
pub fn mtpKvLen(self: *const Generator) u32 {
    return @intCast(self.prompt_ids_owned.len + self.generated_ids.items.len);
}

pub const MtpGraphFn = *const fn (mlx.mlx_array, []const mlx.mlx_array, ?[]const mlx.mlx_array, u32, f32, sampler.SamplingParams, mlx.mlx_stream) anyerror!MtpBatchedGraph;

/// Qwen native-MTP speculative round. Structure mirrors `nextDrafter`
/// (same verify invariant: cache.step = prompt_len + emitted, t1 NOT in
/// cache on entry, verify input `[t1, draft[0..m-1]]`, bonus from row m,
/// partial-accept snapshot/restore + re-forward) with one addition: the
/// MTP head's committed-history KV cache. Draft steps append m temporary
/// entries built from MTP-PREDICTED hiddens; after the verify decision we
/// restore the round-boundary snapshot and re-append the committed pairs
/// from TRUE trunk hiddens, so the history never accumulates drift.
pub const MtpHistStash = struct {
    /// `[n]` int32 committed token ids: `[t1, drafts[0..accepted]]`.
    ids: mlx.mlx_array,
    /// `[1, n, H]` trunk hiddens paired 1:1 with `ids` (a lazy concat of
    /// last_hidden + a verify-capture slice — the handle pins the ~90 KB
    /// parent until consumed, deliberately NOT a deep copy).
    hidden: mlx.mlx_array,
    n: usize,
    /// Head-cache position of ids[0]'s entry (the producing round's
    /// mtp_off0). The consume-time truncate drops the producing round's
    /// stale draft tail past it.
    off0: usize,
    /// `ids` on the host (the MiMo head records committed tokens).
    host_ids: [mtp_mod.MAX_DEPTH + 1]u32 = undefined,

    pub fn deinit(self: *MtpHistStash) void {
        _ = mlx.mlx_array_free(self.ids);
        _ = mlx.mlx_array_free(self.hidden);
    }
};

/// One round's lazily-built MTP draft chain — the Phase 0/1 state that
/// cross-round pre-drafting (`mtpMaybePreDraft`) moves into the PREVIOUS
/// round's tail. Owns every handle it holds; `deinit` frees whatever was
/// built so far, so it is safe on partial builds and on a pre-draft left
/// unconsumed at request end.
pub const MtpPreDraft = struct {
    plan: MtpRoundPlan,
    /// Head-cache position of this round's first draft entry.
    off0: usize,
    t1: u32,
    /// `[1]` int32 t1 — the verify input needs it again in Phase 3.
    t1_arr: mlx.mlx_array,
    /// Host draft ids, filled at the Phase 4b sync. len = plan.m_hi.
    drafts: []u32,
    /// Lazy `[1]` int32 draft ids. len = plan.m_hi; [0..n_drafted) valid.
    draft_arrs: []mlx.mlx_array,
    n_drafted: u32,
    /// Rows the head's first step reads (the stash plus t1); 0 for a lookup chain.
    head_input_rows: u32 = 1,
    /// Chunk-A log-confidence graphs (two-chunk plans only). len m_lo.
    conf_arrs: ?[]mlx.mlx_array,
    n_conf: u32,
    /// Proposal densities q, present when depth 0 samples; an argmax
    /// depth's row is its one-hot. len m_hi; [0..n_qp) valid.
    q_probs: ?[]mlx.mlx_array,
    n_qp: u32,
    /// Chain hidden after the last built step (owned) — a chunk-B
    /// extension resumes from it; freed once the chain is complete.
    h_chain: ?mlx.mlx_array,
    /// Tokens actually drafted this round; starts at m_lo, grows to
    /// m_hi iff the confidence gate clears at the chunk boundary.
    m: u32,
    /// Drafts are a context continuation (`mtpLookupChain`): nothing to
    /// build, and no MTP statistics are fed.
    lookup: bool = false,

    pub fn deinit(self: *MtpPreDraft, allocator: std.mem.Allocator) void {
        _ = mlx.mlx_array_free(self.t1_arr);
        for (self.draft_arrs[0..self.n_drafted]) |arr| _ = mlx.mlx_array_free(arr);
        allocator.free(self.draft_arrs);
        if (self.conf_arrs) |slots| {
            for (slots[0..self.n_conf]) |arr| _ = mlx.mlx_array_free(arr);
            allocator.free(slots);
        }
        if (self.q_probs) |slots| {
            for (slots[0..self.n_qp]) |arr| _ = mlx.mlx_array_free(arr);
            allocator.free(slots);
        }
        if (self.h_chain) |h| _ = mlx.mlx_array_free(h);
        allocator.free(self.drafts);
    }
};

pub const LookupLines = struct {
    tok: *const Tokenizer,
    pub fn hasNewline(self: LookupLines, id: u32) bool {
        return self.tok.tokenHasNewline(id);
    }
};

/// What one draft step reads. Both rerank paths skip the head's own vocab
/// projection (the token comes off the coarse readout's exactly re-scored
/// top-32); `full_logits` pays the whole lm_head because something
/// downstream needs a density over the vocabulary. A SAMPLED proposal does
/// not: q over the shortlist is the proposal's true density, zero off it,
/// and the verify's residual covers the rest.
pub const DraftStepPath = enum { rerank_greedy, rerank_sampled, full_logits };

/// Batched accept/correction graph for one stochastic round. Replaces
/// the per-position construction — m single-element gathers + concat
/// for accept_p (again for accept_q under sharp drafts) and 1+m
/// SEPARATE full-vocab log+categorical correction samplers — with
/// single batched kernels: one [m,1] take_along_axis per accept vector
/// and ONE [1+m, V] log+categorical covering every possible reject
/// position plus the full-accept bonus. Identical math and identical
/// output distributions; only the dispatch count (and the PRNG key
/// split pattern — one batched categorical draw instead of 1+m
/// sequential ones) changes.
pub const MtpBatchedGraph = struct {
    /// [1+m] int32 pre-sampled corrections (index a = residual sample
    /// for a reject at position a; index m = full-accept bonus).
    corr_samples: mlx.mlx_array,
    /// [m] f32 filtered target probability of each draft.
    accept_p: mlx.mlx_array,
    /// [m] f32 proposal density at each draft (sharp drafts only;
    /// null-ctx under greedy proposals). Typical mode stores its entropy
    /// floor here instead.
    accept_q: mlx.mlx_array,
    /// [m] bool TokenV3 deferral decisions; null for other modes.
    accept_defer: mlx.mlx_array = .{ .ctx = null },

    pub fn deinit(self: *MtpBatchedGraph) void {
        _ = mlx.mlx_array_free(self.corr_samples);
        _ = mlx.mlx_array_free(self.accept_p);
        if (self.accept_q.ctx != null) _ = mlx.mlx_array_free(self.accept_q);
        if (self.accept_defer.ctx != null) _ = mlx.mlx_array_free(self.accept_defer);
    }
};

/// A speculative round between its draft chain and its verdict. `mtpRoundBegin`
/// builds it, `mtpRoundVerify` (solo) or the scheduler's group verify fills the
/// three verify outputs, `mtpRoundFinish` consumes them.
pub const MtpRoundState = struct {
    chain: MtpPreDraft,
    tracing: bool,
    ph: io_util.Stopwatch,
    round_watch: io_util.Stopwatch,
    kv_step_snap: usize,
    kv_snap: ?transformer_mod.KVCacheSnapshot,
    moe_seq_offset_snap: usize,
    /// `[1, 1+m]` int32.
    verify_input: mlx.mlx_array,
    /// Positions the verify forward ran; `1 + m` solo, the group's widest row batched.
    verify_len: u32,
    /// `c_n`: the positions this row committed, stamped by the finish.
    retained: u32 = 0,
    verify_rollback: ?Transformer.VerifyRollback = null,
    verify_ctx: ?*ForwardCtx = null,
    verify_argmax: CommittedArgmax = .{},
    verify_logits: mlx.mlx_array = .{ .ctx = null },
    new_hidden: mlx.mlx_array = .{ .ctx = null },
    verify_hidden_all: mlx.mlx_array = .{ .ctx = null },
    /// The group's joined `[1, Σ(1+m_i), V]` verify block, held by the FIRST
    /// row's state; the accept side filters it once for every sampled row.
    group_logits: mlx.mlx_array = .{ .ctx = null },
    /// Phase-4 terms the group prepared for this row (sampled rows only):
    /// the accept graph's own outputs, whichever acceptance mode built them.
    /// `accept_p` null-ctx = this row builds its own in the finish.
    accept_p: mlx.mlx_array = .{ .ctx = null },
    accept_q: mlx.mlx_array = .{ .ctx = null },
    accept_defer: mlx.mlx_array = .{ .ctx = null },
    corr: mlx.mlx_array = .{ .ctx = null },

    pub fn deinit(self: *MtpRoundState, allocator: std.mem.Allocator) void {
        self.chain.deinit(allocator);
        if (self.kv_snap) |*snap| snap.deinit();
        _ = mlx.mlx_array_free(self.verify_input);
        self.verify_argmax.deinit();
        if (self.verify_logits.ctx != null) _ = mlx.mlx_array_free(self.verify_logits);
        if (self.group_logits.ctx != null) _ = mlx.mlx_array_free(self.group_logits);
        self.releasePreparedAccept();
        if (self.new_hidden.ctx != null) _ = mlx.mlx_array_free(self.new_hidden);
        if (self.verify_hidden_all.ctx != null) _ = mlx.mlx_array_free(self.verify_hidden_all);
        if (self.verify_ctx) |ctx| {
            if (ctx.ssm_entries) |entries| for (entries) |*entry| transformer_mod.ssmFreeSpecCapture(entry);
        }
        if (self.verify_rollback) |*saved| {
            saved.restore();
            saved.deinit();
        }
    }

    /// Drop the group-prepared Phase-4 terms, so the row's finish builds its own.
    pub fn releasePreparedAccept(self: *MtpRoundState) void {
        inline for (.{ &self.accept_p, &self.accept_q, &self.accept_defer, &self.corr }) |slot| {
            if (slot.ctx != null) {
                _ = mlx.mlx_array_free(slot.*);
                slot.* = .{ .ctx = null };
            }
        }
    }
};

pub const MtpRowVerdict = struct { accepted: u32, next_pending: u32 };

pub const MTP_GROUP_ROWS_MAX = 32;

pub const MtpRoundOpen = struct {
    chain: MtpPreDraft,
    tracing: bool,
    ph: io_util.Stopwatch,
    round_watch: io_util.Stopwatch,
};

pub const MtpRoundBegin = union(enum) { done: ?DrafterStepResult, verify: MtpRoundState, open: MtpRoundOpen };

/// Target temperature at or below which the GREEDY proposal still wins:
/// the argmax accepts with `p_target[argmax]`, high only while the target
/// row is sharp. Measured crossover between 0.3 and 0.7 on Flash Next.
pub const MTP_SAMPLED_PROPOSAL_MIN_TEMP: f32 = 0.5;

// ── Sharpened stochastic draft proposals (Lightning-class acceptance) ──
// Drafts for a stochastic target are SAMPLED from a fixed sharper
// distribution (constants mirror oMLX's _DRAFT_SAMPLER_*: their comment —
// matched-temp drafting "collapses to ~10-20% on high-entropy content"),
// acceptance is the full Leviathan/Chen ratio min(1, p/q) with q = the
// draft sampler's own filtered distribution, and rejection re-samples
// from normalize(max(p-q, 0)). Output distribution provably equals the
// target's filtered p for ANY proposal q (pinned by the toy-vocab
// exactness test); q only moves the ACCEPTANCE RATE, which is why the
// draft head's quantization never affects correctness.
pub const MTP_DRAFT_TEMP: f32 = 0.6;

pub const MTP_DRAFT_TOP_P: f32 = 0.95;

pub const MTP_DRAFT_TOP_K: u32 = 20;

/// Draft temperature is per FAMILY: the sidecar head's 0.6 does not carry
/// over, and Flash Next drafts best at the target's own temperature.
pub const MTP_DRAFT_TEMP_QWEN4: f32 = 1.0;

/// Where a serial block sits in its ramp back to `nextMtp`'s entry invariant.
pub const MtpSerialExit = enum { none, drain, capture };

test "a planner-owned slot rejoins the planner after a forced commit, any other takes the serial ramp" {
    try testing.expectEqual(MtpSerialExit.none, Generator.mtpSerialExitAfterForced(true));
    try testing.expectEqual(MtpSerialExit.drain, Generator.mtpSerialExitAfterForced(false));
}

test "specAcceptProb: full Leviathan ratio, q-clamped" {
    // p <= q: accept with p/q.
    try testing.expect(@abs(Generator.specAcceptProb(0.2, 0.4) - 0.5) < 1e-6);
    // p > q: always accept.
    try testing.expectEqual(@as(f32, 1.0), Generator.specAcceptProb(0.4, 0.2));
    try testing.expectEqual(@as(f32, 1.0), Generator.specAcceptProb(0.4, 0.4));
    // Degenerate q underflow never divides by zero.
    try testing.expectEqual(@as(f32, 1.0), Generator.specAcceptProb(0.5, 0.0));
}

test "spec sampling exactness: draft-from-q + ratio-accept + residual reproduces target p (toy vocab)" {
    // Host-level simulation of the exact per-position algorithm the MTP
    // stochastic round runs: draft ~ q, accept with min(1, p/q), on reject
    // sample from normalize(max(p - q, 0)). The output distribution must
    // equal p (Leviathan/Chen) — this is the invariant the one-hot rule
    // broke for sampled proposals.
    const p = [_]f64{ 0.1, 0.2, 0.3, 0.4 };
    const q = [_]f64{ 0.4, 0.3, 0.2, 0.1 };
    var residual: [4]f64 = undefined;
    var res_sum: f64 = 0;
    for (0..4) |i| {
        residual[i] = @max(p[i] - q[i], 0);
        res_sum += residual[i];
    }
    for (&residual) |*r| r.* /= res_sum;

    var prng = std.Random.DefaultPrng.init(0x5A3E);
    const rnd = prng.random();
    var counts = [_]u64{ 0, 0, 0, 0 };
    const N: usize = 400_000;
    for (0..N) |_| {
        // draft ~ q
        var u = rnd.float(f64);
        var draft: usize = 0;
        var acc: f64 = 0;
        for (q, 0..) |qi, i| {
            acc += qi;
            if (u < acc) {
                draft = i;
                break;
            }
        }
        const a = Generator.specAcceptProb(@floatCast(p[draft]), @floatCast(q[draft]));
        if (rnd.float(f32) < a) {
            counts[draft] += 1;
        } else {
            u = rnd.float(f64);
            acc = 0;
            var res_tok: usize = 3;
            for (residual, 0..) |ri, i| {
                acc += ri;
                if (u < acc) {
                    res_tok = i;
                    break;
                }
            }
            counts[res_tok] += 1;
        }
    }
    for (0..4) |i| {
        const freq = @as(f64, @floatFromInt(counts[i])) / @as(f64, @floatFromInt(N));
        try testing.expect(@abs(freq - p[i]) < 0.01);
    }
}

test "spec sampling exactness: a SHORTLIST proposal (q zero off the top-k) still reproduces target p" {
    // Bar: with q supported on a subset, the marginal is still p, and the
    // off-shortlist tokens arrive only through the residual.
    const p = [_]f64{ 0.30, 0.25, 0.20, 0.15, 0.10 };
    const q = [_]f64{ 0.50, 0.30, 0.20, 0.00, 0.00 }; // shortlist = {0, 1, 2}
    const off_shortlist = [_]usize{ 3, 4 };

    var residual: [5]f64 = undefined;
    var res_sum: f64 = 0;
    for (0..5) |i| {
        residual[i] = @max(p[i] - q[i], 0);
        res_sum += residual[i];
    }
    for (&residual) |*r| r.* /= res_sum;

    var prng = std.Random.DefaultPrng.init(0x51157);
    const rnd = prng.random();
    var counts = [_]u64{ 0, 0, 0, 0, 0 };
    const N: usize = 400_000;
    for (0..N) |_| {
        var u = rnd.float(f64);
        var draft: usize = 0;
        var acc: f64 = 0;
        for (q, 0..) |qi, i| {
            acc += qi;
            if (u < acc) {
                draft = i;
                break;
            }
        }
        // A draft is never drawn off the shortlist, so q[draft] > 0 always.
        try testing.expect(q[draft] > 0);
        if (rnd.float(f32) < Generator.specAcceptProb(@floatCast(p[draft]), @floatCast(q[draft]))) {
            counts[draft] += 1;
        } else {
            u = rnd.float(f64);
            acc = 0;
            var res_tok: usize = 4;
            for (residual, 0..) |ri, i| {
                acc += ri;
                if (u < acc) {
                    res_tok = i;
                    break;
                }
            }
            counts[res_tok] += 1;
        }
    }
    for (0..5) |i| {
        const freq = @as(f64, @floatFromInt(counts[i])) / @as(f64, @floatFromInt(N));
        try testing.expect(@abs(freq - p[i]) < 0.01);
    }
    // The residual is the ONLY route to a token the shortlist missed.
    for (off_shortlist) |i| try testing.expect(counts[i] > 0);
}

test "batched corrections read only 1+m rows of a group-padded verify block" {
    const s = mlx.gpuStream();
    // The block above plus one pad row, m=2.
    const p_data = [_]f32{ 0.5, 0.5, 0, 0, 0, 0.25, 0.75, 0, 0, 0, 0, 1.0, 1.0, 0, 0, 0 };
    const p_shape = [_]c_int{ 1, 4, 4 };
    const probs_all = mlx.mlx_array_new_data(&p_data, &p_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(probs_all);
    const id_shape = [_]c_int{1};
    const d0_data: i32 = 0;
    const d1_data: i32 = 2;
    const d0 = mlx.mlx_array_new_data(&d0_data, &id_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(d0);
    const d1 = mlx.mlx_array_new_data(&d1_data, &id_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(d1);
    const drafts = [_]mlx.mlx_array{ d0, d1 };

    var g = try Generator.mtpBatchedAcceptGraph(probs_all, &drafts, null, 2, .{}, s);
    defer g.deinit();
    try mlx.check(mlx.mlx_array_eval(g.corr_samples));
    const corr = mlx.mlx_array_data_int32(g.corr_samples) orelse return error.InvalidDtype;
    try testing.expectEqual(@as(i32, 1), corr[0]);
    try testing.expectEqual(@as(i32, 1), corr[1]);
    try testing.expectEqual(@as(i32, 3), corr[2]);
}

test "MTP typical batched graph uses entropy floor and target-row correction" {
    const s = mlx.gpuStream();
    const p_data = [_]f32{ 0.5, 0.5, 0, 0, 0, 0, 0, 1 };
    const p_shape = [_]c_int{ 1, 2, 4 };
    const probs = mlx.mlx_array_new_data(&p_data, &p_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(probs);
    const id: i32 = 0;
    const id_shape = [_]c_int{1};
    const draft = mlx.mlx_array_new_data(&id, &id_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(draft);
    const drafts = [_]mlx.mlx_array{draft};
    var g = try Generator.mtpBatchedTypicalGraph(probs, &drafts, null, 1, 0.2, .{}, s);
    defer g.deinit();
    try mlx.check(mlx.mlx_array_eval(g.accept_p));
    try mlx.check(mlx.mlx_array_eval(g.accept_q));
    const p = mlx.mlx_array_data_float32(g.accept_p) orelse return error.InvalidDtype;
    const floor = mlx.mlx_array_data_float32(g.accept_q) orelse return error.InvalidDtype;
    try testing.expectApproxEqAbs(@as(f32, 0.5), p[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.1), floor[0], 1e-6);
    try testing.expect(g.accept_defer.ctx == null);
    try mlx.check(mlx.mlx_array_eval(g.corr_samples));
    const corr = mlx.mlx_array_data_int32(g.corr_samples) orelse return error.InvalidDtype;
    try testing.expect(corr[0] == 0 or corr[0] == 1);
    try testing.expectEqual(@as(i32, 3), corr[1]);
}

test "a prompt-lookup draft is kept only as often as sampling would keep it under typical acceptance" {
    // A copied token at p = 0.05 in a high-entropy row clears typical's floor (0.2 * e^-H), so
    // the typical route keeps it every time; a lookup round must keep it about 5% of the time.
    const s = mlx.gpuStream();
    const V = 32;
    var p_data: [2 * V]f32 = undefined;
    p_data[0] = 0.05;
    for (p_data[1..V]) |*v| v.* = 0.95 / @as(f32, V - 1);
    for (p_data[V..]) |*v| v.* = 1.0 / @as(f32, V);
    const p_shape = [_]c_int{ 1, 2, V };
    const probs = mlx.mlx_array_new_data(&p_data, &p_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(probs);
    const id: i32 = 0;
    const id_shape = [_]c_int{1};
    const draft = mlx.mlx_array_new_data(&id, &id_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(draft);
    const drafts = [_]mlx.mlx_array{draft};

    var g: Generator = undefined;
    g.mtp_accept_graph = Generator.mtpBatchedTypicalGraph;
    g.mtp_accept_prefix = mtp_acceptance.typicalPrefix;
    g.mtp_accept_param = 0.2;
    var prng = std.Random.DefaultPrng.init(0x100C);
    var kept: [2]u32 = .{ 0, 0 };
    for ([_]bool{ false, true }, 0..) |lookup, arm| {
        var bg = try g.acceptGraphFor(lookup)(probs, &drafts, null, 1, g.mtp_accept_param, .{}, s);
        defer bg.deinit();
        try mlx.check(mlx.mlx_array_eval(bg.accept_p));
        const p = mlx.mlx_array_data_float32(bg.accept_p) orelse return error.InvalidDtype;
        var q: ?[]const f32 = null;
        if (bg.accept_q.ctx != null) {
            try mlx.check(mlx.mlx_array_eval(bg.accept_q));
            q = (mlx.mlx_array_data_float32(bg.accept_q) orelse return error.InvalidDtype)[0..1];
        }
        for (0..400) |_| kept[arm] += g.acceptPrefixFor(lookup)(p[0..1], q, null, &prng);
    }
    try testing.expectEqual(@as(u32, 400), kept[0]);
    try testing.expect(kept[1] > 0 and kept[1] < 60);
}

test "mtpCommittedLen: speculative tails (pre-draft, stash) bound the committable history" {
    try testing.expectEqual(@as(usize, 42), Generator.mtpCommittedLen(42, null, null));
    // Pending stash: the cache past off0 is the producing round's stale draft tail.
    try testing.expectEqual(@as(usize, 40), Generator.mtpCommittedLen(46, null, 40));
    // Built pre-draft: next-round draft entries sit past ITS off0.
    try testing.expectEqual(@as(usize, 43), Generator.mtpCommittedLen(47, 43, null));
    // Both live: the min wins.
    try testing.expectEqual(@as(usize, 40), Generator.mtpCommittedLen(47, 43, 40));
}

test "mtpRoundOff0: a pending history stash overrides the stale cache length" {
    // No stash: the head cache is fully committed — its step IS the origin.
    try testing.expectEqual(@as(usize, 42), Generator.mtpRoundOff0(null, 42));
    // Pending stash: the cache still carries the previous round's draft tail
    // (step is stale/uncommitted), so the origin is where the stash's entries
    // will END once the consume-time truncate + merged forward run. A round
    // that read mc.step here would draft at the WRONG RoPE offsets.
    const st = MtpHistStash{
        .ids = .{ .ctx = null },
        .hidden = .{ .ctx = null },
        .n = 3, // t1 + 2 accepted drafts
        .off0 = 40,
    };
    // cache_step (46 = 40 committed + 6 stale draft entries) must be ignored.
    try testing.expectEqual(@as(usize, 43), Generator.mtpRoundOff0(st, 46));
}

test "mtpSerialCaptureReady: the capture step's entry invariant is a RUNTIME check, not an assert" {
    const G = Generator;
    try testing.expect(G.mtpSerialCaptureReady(false, false));
    try testing.expect(!G.mtpSerialCaptureReady(true, false));
    try testing.expect(!G.mtpSerialCaptureReady(false, true));
    try testing.expect(!G.mtpSerialCaptureReady(true, true));
}

test "group acceptance: every row's verdict comes from its OWN am, at any accept position" {
    const accept = Generator.mtpGreedyVerdict;
    {
        const am = [_]i32{ 11, 22, 33 };
        const drafts = [_]u32{ 11, 22 };
        const v = accept(&am, &drafts, 0, 400);
        try testing.expectEqual(@as(u32, 2), v.accepted);
        try testing.expectEqual(@as(u32, 33), v.next_pending);
    }
    {
        const am = [_]i32{ 11, 99, 33 };
        const drafts = [_]u32{ 11, 22 };
        const v = accept(&am, &drafts, 0, 400);
        try testing.expectEqual(@as(u32, 1), v.accepted);
        try testing.expectEqual(@as(u32, 99), v.next_pending);
    }
    {
        const am = [_]i32{ 7, 22, 33 };
        const drafts = [_]u32{ 11, 22 };
        const v = accept(&am, &drafts, 0, 400);
        try testing.expectEqual(@as(u32, 0), v.accepted);
        try testing.expectEqual(@as(u32, 7), v.next_pending);
    }
    {
        const am = [_]i32{ 11, 22, 33, 44, 55 };
        const drafts = [_]u32{ 11, 22 };
        const v = accept(&am, &drafts, 0, 400);
        try testing.expectEqual(@as(u32, 2), v.accepted);
        try testing.expectEqual(@as(u32, 33), v.next_pending);
    }
    {
        const am = [_]i32{ 11, 22, 33, 44 };
        const drafts = [_]u32{ 11, 22, 33 };
        const full = accept(&am, &drafts, 0, 400);
        try testing.expectEqual(@as(u32, 3), full.accepted);
        try testing.expectEqual(@as(u32, 44), full.next_pending);
        const clipped = accept(&am, &drafts, 397, 400);
        try testing.expectEqual(@as(u32, 2), clipped.accepted);
        try testing.expectEqual(@as(u32, 33), clipped.next_pending);
        const none = accept(&am, &drafts, 399, 400);
        try testing.expectEqual(@as(u32, 0), none.accepted);
        try testing.expectEqual(@as(u32, 11), none.next_pending);
    }
    {
        const am = [_]i32{42};
        const v = accept(&am, &.{}, 0, 400);
        try testing.expectEqual(@as(u32, 0), v.accepted);
        try testing.expectEqual(@as(u32, 42), v.next_pending);
    }
}

test "batched MTP draft rows stop at their own depth and allocation" {
    var chains: [3]MtpPreDraft = undefined;
    var wide: [5]mlx.mlx_array = undefined;
    var short: [1]mlx.mlx_array = undefined;
    var middle: [3]mlx.mlx_array = undefined;
    chains[0].m = 4;
    chains[0].draft_arrs = &wide;
    chains[1].m = 1;
    chains[1].draft_arrs = &short;
    chains[2].m = 3;
    chains[2].draft_arrs = &middle;
    var indices: [3]usize = undefined;
    const expected = [_][]const usize{ &.{ 0, 1, 2 }, &.{ 0, 2 }, &.{ 0, 2 }, &.{0}, &.{}, &.{} };
    for (expected, 0..) |want, step| {
        const n = Generator.mtpDraftActiveRows(&chains, @intCast(step), &indices);
        try testing.expectEqualSlices(usize, want, indices[0..n]);
        for (indices[0..n]) |row| try testing.expect(step < chains[row].draft_arrs.len);
    }
}

test "batched MTP draft rows stop without touching inactive generators" {
    var gens: [2]Generator = undefined;
    var ptrs = [_]*Generator{ &gens[0], &gens[1] };
    var chains: [2]MtpPreDraft = undefined;
    chains[0].m = 4;
    chains[0].n_drafted = 4;
    chains[1].m = 1;
    chains[1].n_drafted = 1;
    try Generator.mtpChainBuildBatched(&ptrs, &chains, 4, 9);
    try testing.expectEqual(@as(u32, 4), chains[0].n_drafted);
    try testing.expectEqual(@as(u32, 1), chains[1].n_drafted);
}

test "MTP continuation releases its owned chain when drafting fails" {
    const allocator = testing.allocator;
    const s = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(s);
    var xfm: Transformer = undefined;
    xfm.s = s;
    xfm.qwen4_mtp = null;
    var state: Transformer.Qwen4MtpState = undefined;
    var gen: Generator = undefined;
    gen.xfm = &xfm;
    gen.mtp = .{ .qwen4 = &xfm };
    gen.mtp_cache = .{ .qwen4 = .{ .t = &xfm, .st = &state, .allocator = allocator } };
    gen.sampling = .{ .temperature = 0 };
    gen.ctx.mrope_pos = null;
    gen.mtp_hist_stash = null;
    gen.last_hidden = .{ .ctx = null };
    const chain: MtpPreDraft = .{
        .plan = .{ .m_lo = 1, .m_hi = 2, .tau_ln = 0 },
        .off0 = 0,
        .t1 = 1,
        .t1_arr = mlx.mlx_array_new(),
        .drafts = try allocator.alloc(u32, 2),
        .draft_arrs = try allocator.alloc(mlx.mlx_array, 2),
        .n_drafted = 0,
        .conf_arrs = try allocator.alloc(mlx.mlx_array, 1),
        .n_conf = 0,
        .q_probs = try allocator.alloc(mlx.mlx_array, 2),
        .n_qp = 0,
        .h_chain = null,
        .m = 1,
    };
    const open: MtpRoundOpen = .{
        .chain = chain,
        .tracing = false,
        .ph = undefined,
        .round_watch = undefined,
    };
    try expectError(error.NoMtpHead, gen.mtpRoundContinue(allocator, open));
}

test "MTP commit stops before an accepted EOS and keeps EOS pending" {
    const drafts = [_]u32{ 11, 2, 12, 3 };
    const eos = [_]u32{ 2, 3 };
    const stop = Generator.mtpStopPrefix(&drafts, 4, &eos);
    try testing.expectEqual(@as(u32, 1), stop.accepted);
    try testing.expectEqual(@as(?u32, 2), stop.stop);
    try testing.expectEqual(@as(u32, 1), Generator.mtpStopPrefix(&drafts, 2, &eos).accepted);
    const rejected_eos = Generator.mtpStopPrefix(&drafts, 1, &eos);
    try testing.expectEqual(@as(u32, 1), rejected_eos.accepted);
    try testing.expect(rejected_eos.stop == null);
    try testing.expectEqual(@as(u32, 0), Generator.mtpStopPrefix(&drafts, 0, &eos).accepted);
    try testing.expectEqual(@as(u32, 4), Generator.mtpStopPrefix(&drafts, 4, &.{}).accepted);
}

test "mtpPreDraftOpenAllows: a pre-draft needs a later round to read it" {
    const G = Generator;
    try testing.expect(G.mtpPreDraftOpenAllows(9, 10, false));
    try testing.expect(!G.mtpPreDraftOpenAllows(10, 10, false));
    try testing.expect(!G.mtpPreDraftOpenAllows(11, 10, false));
    try testing.expect(!G.mtpPreDraftOpenAllows(5, 10, true));
}

test "MTP detaching a pre-draft preserves the committed head boundary" {
    const allocator = testing.allocator;
    const stream = mlx.mlx_default_cpu_stream_new();
    defer _ = mlx.mlx_stream_free(stream);
    for ([_]bool{ false, true }) |owned| {
        for ([_]bool{ false, true }) |apply_stash| {
            var xfm: Transformer = undefined;
            xfm.s = stream;
            var gen: Generator = undefined;
            gen.xfm = &xfm;
            gen.mtp_planner_owned = owned;
            const head_state = try allocator.create(mimo_mtp.State);
            head_state.* = .{ .allocator = allocator, .hid_base = 9 };
            gen.mtp_cache = .{ .mimo = .{ .st = head_state } };
            defer gen.mtp_cache.?.deinit();
            gen.mtp_hist_stash = null;
            const arrays = try allocator.alloc(mlx.mlx_array, 3);
            for (arrays) |*array| array.* = mlx.mlx_array_new();
            gen.mtp_pre_draft = .{
                .plan = .{ .m_lo = 3, .m_hi = 3, .tau_ln = 0 },
                .off0 = 6,
                .t1 = 1,
                .t1_arr = mlx.mlx_array_new(),
                .drafts = try allocator.alloc(u32, 3),
                .draft_arrs = arrays,
                .n_drafted = 3,
                .conf_arrs = null,
                .n_conf = 0,
                .q_probs = null,
                .n_qp = 0,
                .h_chain = null,
                .m = 3,
            };
            defer if (gen.mtp_pre_draft) |*draft| draft.deinit(allocator);
            const committed = gen.mtpCommittedHistoryLen();
            try gen.mtpDetachHead(allocator, apply_stash);
            try testing.expect(gen.mtp_pre_draft == null);
            try testing.expectEqual(@as(usize, 6), committed);
            try testing.expectEqual(if (owned) committed else @as(usize, 9), gen.mtpCommittedHistoryLen());
        }
    }
}

test "cold grouped MTP head seeds the solo position and preserves valid origins" {
    try testing.expectEqual(@as(c_int, 1), Generator.mtpBatchHeadPosition(-1, 0));
    try testing.expectEqual(@as(c_int, 0), Generator.mtpBatchHeadPosition(0, 0));
    try testing.expectEqual(@as(c_int, 8), Generator.mtpBatchHeadPosition(8, 0));
    try testing.expectEqual(@as(c_int, 12), Generator.mtpBatchHeadPosition(8, 4));
    try testing.expectEqual(@as(c_int, 10), Generator.mtpBatchHeadPosition(1, 9));
}

test "mtpLookupAllowed: the qwen4 and MiMo heads, also at a pinned depth, never a batched head or a planner-owned round" {
    const off = generate_mod.mtp_lookup_disabled;
    const bounds_before = depth_bounds.active;
    const planner = group_planner.enabled_override;
    defer {
        generate_mod.mtp_lookup_disabled = off;
        depth_bounds.active = bounds_before;
        group_planner.enabled_override = planner;
    }
    generate_mod.mtp_lookup_disabled = false;
    depth_bounds.active = .{};
    group_planner.enabled_override = true;
    var g: Generator = undefined;
    g.mtp = .{ .qwen4 = undefined };
    g.mtp_batch_head = false;
    g.mtp_planner_owned = false;
    try testing.expect(g.mtpLookupAllowed());

    g.mtp = .{ .mimo = undefined };
    try testing.expect(g.mtpLookupAllowed());
    g.mtp = null;
    try testing.expect(!g.mtpLookupAllowed());
    g.mtp = .{ .qwen4 = undefined };

    depth_bounds.active = .{ .min = 3, .max = 3 };
    try testing.expect(g.mtpLookupAllowed());
    depth_bounds.active = .{};

    g.mtp_batch_head = true;
    try testing.expect(!g.mtpLookupAllowed());
    g.mtp_batch_head = false;

    g.mtp_planner_owned = true;
    try testing.expect(!g.mtpLookupAllowed());
    group_planner.enabled_override = false;
    try testing.expect(g.mtpLookupAllowed());

    generate_mod.mtp_lookup_disabled = true;
    try testing.expect(!g.mtpLookupAllowed());
}

test "mtpLookupCostsFor: a lookup is the MTP round at its draft count minus the head steps, until the lookup row is measured" {
    const costs = mtp_policy.MTP_EV_DEFAULT_COSTS;
    const prior = mtp_policy.MtpCostSource.init(costs, 1000, null);
    const c = Generator.mtpLookupCostsFor(prior, 3);
    try testing.expectApproxEqAbs(Generator.mtpEvRoundCostAt(costs, 3, false, 1000), c.mtp, 1e-6);
    for (c.lookup, 0..) |x, ki| {
        const k: u32 = @intCast(ki);
        const want = Generator.mtpEvRoundCostAt(costs, k, false, 1000) - @as(f32, @floatFromInt(k)) * costs.draft;
        try testing.expectApproxEqAbs(want, x, 1e-6);
    }

    var t = round_cost.Table{};
    for (0..round_cost.MIN_SAMPLES + 1) |_| _ = t.observe(2, 1000, 30.0, 2.5, true, false);
    _ = t.observeLookup(5, 1000, 500.0, 6.0, true);
    for (0..round_cost.MIN_SAMPLES) |_| _ = t.observeLookup(5, 1000, 40.0, 6.0, true);
    const live = mtp_policy.MtpCostSource.init(costs, 1000, &t);
    try testing.expect(live.fromTable());
    const m = Generator.mtpLookupCostsFor(live, 2);
    try testing.expectApproxEqAbs(30.0 * live.scale, m.mtp, 1e-4);
    try testing.expectApproxEqAbs(40.0 * live.scale, m.lookup[5], 1e-4);
    // An unmeasured draft count keeps the table-scaled MTP round minus its head steps.
    try testing.expectApproxEqAbs(live.roundCost(4, false) - 4 * costs.draft, m.lookup[4], 1e-4);
}

test "mtpRoundAcceptObserve: a lookup round feeds only the lookup counters, an MTP round only the head's" {
    var g: Generator = undefined;
    g.mtp_accepted_tokens = 0;
    g.mtp_lookup_accepted = 0;
    g.mtp_lookup_ema = @floatFromInt(mtp_lookup.MAX_DRAFT);
    g.mtp_round_ema = 0;
    g.mtp_round_drafted_ema = 0;
    g.mtp_lookup_streak = false;

    g.mtpRoundAcceptObserve(true, 8, 8);
    try testing.expectEqual(@as(u64, 0), g.mtp_accepted_tokens);
    try testing.expectEqual(@as(u64, 8), g.mtp_lookup_accepted);
    try testing.expect(g.mtp_lookup_streak);
    try testing.expectApproxEqAbs(mtp_lookup.emaStep(@floatFromInt(mtp_lookup.MAX_DRAFT), 8), g.mtp_lookup_ema, 1e-6);
    try testing.expectEqual(@as(f32, 0), g.mtp_round_ema);

    g.mtpRoundAcceptObserve(true, 8, 3);
    try testing.expect(!g.mtp_lookup_streak);

    const lookup_before = g.mtp_lookup_ema;
    g.mtpRoundAcceptObserve(false, 4, 2);
    try testing.expectEqual(@as(u64, 2), g.mtp_accepted_tokens);
    try testing.expectEqual(@as(u64, 11), g.mtp_lookup_accepted);
    try testing.expectApproxEqAbs(mtp_lookup.emaStep(0, 2), g.mtp_round_ema, 1e-6);
    try testing.expectApproxEqAbs(mtp_lookup.emaStep(0, 4), g.mtp_round_drafted_ema, 1e-6);
    try testing.expectApproxEqAbs(mtp_lookup.driftStep(lookup_before), g.mtp_lookup_ema, 1e-6);
}

test "the lookup gate prices the MTP chain at the width its rounds draft, not the plan's base width" {
    // A two-chunk plan (m_lo 1, extending to 3) whose rounds accept ~2.7: priced at its base width, the
    // MTP side reads 3.7 tokens for a one-draft round and a copy's seven-draft lookup loses.
    const two_chunk = mtp_policy.MtpRoundPlan{ .m_lo = 1, .m_hi = 3, .tau_ln = 0 };
    try testing.expectEqual(@as(u32, 1), Generator.mtpLookupPriceWidth(two_chunk, 0));
    try testing.expectEqual(@as(u32, 3), Generator.mtpLookupPriceWidth(two_chunk, 2.9));
    try testing.expectEqual(@as(u32, 3), Generator.mtpLookupPriceWidth(two_chunk, 6));
    try testing.expectEqual(@as(u32, 3), Generator.mtpLookupPriceWidth(.{ .m_lo = 3, .m_hi = 3, .tau_ln = 0 }, 1));

    var t = round_cost.Table{};
    for (0..round_cost.MIN_SAMPLES + 1) |_| {
        _ = t.observe(1, 1000, 28.0, 1.9, true, false);
        _ = t.observe(3, 1000, 45.0, 3.0, true, false);
    }
    _ = t.observeLookup(7, 1000, 500.0, 8.0, true);
    for (0..round_cost.MIN_SAMPLES) |_| _ = t.observeLookup(7, 1000, 71.0, 7.5, true);
    const src = mtp_policy.MtpCostSource.init(mtp_policy.MTP_EV_MIMO_EXL3_COSTS, 1000, &t);
    try testing.expect(src.fromTable());
    var d: [20]u32 = undefined;
    for (&d, 0..) |*x, i| x.* = @intCast(i);
    const copy = mtp_lookup.Match{ .draft = d[0..7], .suffix = mtp_lookup.STRONG_SUFFIX };
    try testing.expectEqual(@as(u32, 0), mtp_lookup.gate(copy, 1000, 6.5, 2.7, false, Generator.mtpLookupCostsFor(src, two_chunk.m_lo)));
    const width = Generator.mtpLookupPriceWidth(two_chunk, 2.9);
    try testing.expectEqual(@as(u32, 7), mtp_lookup.gate(copy, 1000, 6.5, 2.7, false, Generator.mtpLookupCostsFor(src, width)));
}

test "MTP depth bounds: a lookup draft is never wider than an explicit max" {
    var idx = mtp_lookup.Index.init(testing.allocator);
    defer idx.deinit();
    const Lines = struct {
        pub fn hasNewline(_: @This(), id: u32) bool {
            return id == 0;
        }
    };
    var text: [80]u32 = undefined;
    for (&text, 0..) |*t, i| t.* = @intCast(1 + i % 40);
    try idx.extend(&text, Lines{});
    const cap = depth_bounds.lookupCap(Generator.mtpLookupDraftCap(false), .{ .max = 3 });
    const got = idx.match(1, cap) orelse return error.NoMatch;
    try testing.expectEqual(@as(usize, 3), got.draft.len);
}
