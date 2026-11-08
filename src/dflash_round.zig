//! The generic DFlash2 round, named by its format and not by an architecture: any target whose arch lists
//! `.dflash2` in its drafter `kinds` and has no native round (GLM has its own) runs it.
const std = @import("std");
const mlx = @import("mlx.zig");
const model_handle = @import("model_handle.zig");
const trunk = @import("trunk.zig");
const spec_types = @import("spec_types.zig");
const generate = @import("generate.zig");
const dflash_mod = @import("dflash.zig");
const dflash_policy = @import("dflash_policy.zig");
const drafter = @import("arch/drafter.zig");
const io_util = @import("io_util.zig");
const log = @import("log.zig");
const mtp_plan = @import("mtp_plan.zig");
const sampler = @import("sampler.zig");
const Generator = generate.Generator;
const DrafterStepResult = Generator.DrafterStepResult;
const verifyArgmax = spec_types.verifyArgmax;
const CommittedArgmax = spec_types.CommittedArgmax;

/// One DFlash2 round of a target without a native round of its own: draft a block, verify it in one forward,
/// accept, roll recurrent state back from the capture. Cap accepted drafts with capAcceptedForTokenBudget before
/// committing.
pub fn nextDflash(self: *Generator, allocator: std.mem.Allocator) !?DrafterStepResult {
    // The kv term is the same physics for either block decoder (one
    // forward, one KV read, shared across the block's rows), so a DFlash
    // round is an observation for it too — and on a DFlash-only server
    // it is the ONLY source.
    const dflash_kv_watch = io_util.Stopwatch.init(self.timer.io);
    // The round-cost table sees the same round: drafts = block - 1, or
    // 0 when the runtime gate fell back to serial.
    const dflash_gen_before = self.generated_ids.items.len;
    self.dflash_round_width = 0;
    const dflash_rounds_before: u64 = self.dflash_attempted;
    defer {
        const ms = @as(f32, @floatFromInt(dflash_kv_watch.read())) / @as(f32, std.time.ns_per_ms);
        const emitted = self.generated_ids.items.len - dflash_gen_before;
        const post_warmup = dflash_rounds_before >= dflash_policy.DFLASH_GATE_WARMUP;
        const wall = if (post_warmup) self.mtpRegimeWallMs(ms) else ms;
        self.specObserveRound(self.dflash_round_width, wall, @floatFromInt(emitted), post_warmup and emitted > 0, false);
        if (self.dflash_round_width > 0) self.dflash_wall_ms += wall;
    }
    if (self.done) return null;
    std.debug.assert(self.dflash != null);
    std.debug.assert(self.dflash_ctx != null or self.spec_disabled_runtime);
    if (mtp_plan.specDecodeUnsupported(self.sampling, self.logprobs_n)) return error.SpecDecodeUnsupported;

    if (self.completion_tokens >= self.max_tokens) {
        self.done = true;
        self.finish_reason = "length";
        return null;
    }

    if (self.spec_disabled_runtime) {
        const tok_opt = try self.next(allocator);
        if (tok_opt == null) return null;
        const tokens = try allocator.alloc(u32, 1);
        tokens[0] = tok_opt.?;
        return DrafterStepResult{ .tokens = tokens, .accepted_tokens = 0 };
    }

    const round_width: u32 = @max(self.dflash_block_size, 2) - 1;
    self.dflash_round_width = round_width;

    if (try self.checkStop()) return null; // t1 is this block's first emit: stop before drafting

    const xfm = self.xfm;
    const s = xfm.s;
    const model = self.dflash.?;
    const dctx = &self.dflash_ctx.?;
    const bs: u32 = round_width + 1;
    const m_full: u32 = bs - 1;
    // Drafts this round verifies: all of the block's, or the policy's rows - 1 once the selector has spoken.
    var m: u32 = m_full;
    const t1: u32 = self.next_token_id;
    // Positions come from moe_seq_offset (cache.step is a bookkeeping
    // counter the model never reads — same rule as nextMtp).
    const anchor_pos: usize = self.ctx.moe_seq_offset.*;
    const kv_step_snap = self.ctx.cache.step;
    std.debug.assert(dctx.absLen() == anchor_pos);

    const tracing = traceEnabled();
    var ph: io_util.Stopwatch = undefined;
    if (tracing) {
        ph = io_util.Stopwatch.init(self.timer.io);
        if (self.dflash_gap_watch) |*gw| self.dflash_trace.add(.gap, gw.read());
        self.dflash_gap_watch = null;
    }

    // ── Phase 1: one assistant forward drafts all m tokens ──
    // DFlash reads mask rows 1..bs-1 (anchor row dropped).
    const noise_rows: u32 = bs;
    const noise_ids = try allocator.alloc(i32, noise_rows);
    defer allocator.free(noise_ids);
    noise_ids[0] = @intCast(t1);
    for (noise_ids[1..]) |*v| v.* = @intCast(model.config.mask_token_id);
    const noise_shape = [_]c_int{ 1, @intCast(noise_rows) };
    const noise_input = mlx.mlx_array_new_data(noise_ids.ptr, &noise_shape, 2, .int32);
    defer _ = mlx.mlx_array_free(noise_input);
    // RAW table rows — no embed norm, no scale (the DFlash contract).
    const noise_embeds = try xfm.rawEmbedding(noise_input);
    defer _ = mlx.mlx_array_free(noise_embeds);

    const blk_hidden = try dflash_mod.forwardBlock(model, dctx, noise_embeds, anchor_pos);
    defer _ = mlx.mlx_array_free(blk_hidden);
    if (tracing) {
        try mlx.check(mlx.mlx_array_eval(blk_hidden));
        self.dflash_trace.add(.assist, ph.read());
        ph.reset();
    }
    const draft_logits_all = try model.draftLogits(xfm, blk_hidden);
    defer _ = mlx.mlx_array_free(draft_logits_all);

    // Slice the m draft rows: DFlash drops the anchor row (1..bs-1).
    const dl_shape = mlx.getShape(draft_logits_all);
    var draft_logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(draft_logits);
    {
        const row0: c_int = 1;
        const start = [_]c_int{ 0, row0, 0 };
        const stop = [_]c_int{ 1, row0 + @as(c_int, @intCast(m_full)), dl_shape[2] };
        const strides = [_]c_int{ 1, 1, 1 };
        try mlx.check(mlx.mlx_slice(&draft_logits, draft_logits_all, &start, 3, &stop, 3, &strides, 3, s));
    }
    // Sampled requests accept through the full Leviathan ratio
    // min(1, p/q), so a GREEDY draft is a one-hot q — and at temperature
    // the target row is flat, which is exactly where min(1, p(argmax))
    // collapses. Drafting FROM the request's own distribution makes q
    // track p and keeps acceptance flat across temperature. Greedy
    // requests keep the argmax path untouched, so the byte-equality
    // guard is unaffected.
    const stochastic = !sampler.isGreedyTemperature(self.sampling.temperature);
    // DFlash2 path selector: when the sidecar ships one, drafts come from
    // the pairwise-scored path trace instead of per-position argmax /
    // block sampling. Greedy requests keep the byte-equality bar (a
    // selector draft only survives verify if it IS the trunk argmax);
    // stochastic requests sample the selector's own candidate softmax and
    // accept through min(1, p/q) with q read off the traced path —
    // exact by construction.
    const use_selector = model.selector != null;
    var sel_path: ?dflash_mod.SelectedPath = null;
    defer if (sel_path) |*sp| sp.deinit(allocator);
    var draft_ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(draft_ids);
    if (use_selector) {
        const sel_temp: f32 = if (stochastic) self.sampling.temperature else 0.0;
        sel_path = try dflash_mod.selectPath(
            allocator,
            &model.selector.?,
            model.config.selector_top_k,
            blk_hidden,
            draft_logits,
            t1,
            sel_temp,
            self.prng.random(),
            s,
        );
        if (adaptiveRows(self)) {
            m = dflash_policy.pickRows(&self.dflash_calib, sel_path.?.conf, dflash_mod.minRows(), self.dflash_block_size) - 1;
            self.dflash_round_width = m;
        }
        const ids_i32 = try allocator.alloc(i32, m);
        defer allocator.free(ids_i32);
        for (sel_path.?.ids[0..m], ids_i32) |v, *d| d.* = @intCast(v);
        const row_shape = [_]c_int{ 1, @intCast(m) };
        const host_arr = mlx.mlx_array_new_data(ids_i32.ptr, &row_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(host_arr);
        try mlx.check(mlx.mlx_array_set(&draft_ids, host_arr));
    } else {
        var draft_amax = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(draft_amax);
        try mlx.check(mlx.mlx_argmax_axis(&draft_amax, draft_logits, 2, false, s));
        try mlx.check(mlx.mlx_astype(&draft_ids, draft_amax, .int32, s));
    }
    if (tracing) {
        try mlx.check(mlx.mlx_array_eval(draft_ids));
        self.dflash_trace.add(.head, ph.read());
        ph.reset();
    }

    // ── Phase 2: verify input [t1, drafts...] — [1, bs] int32, lazy ──
    const t1_i32: i32 = @intCast(t1);
    const t1_shape = [_]c_int{ 1, 1 };
    const t1_arr = mlx.mlx_array_new_data(&t1_i32, &t1_shape, 2, .int32);
    defer _ = mlx.mlx_array_free(t1_arr);
    var verify_input = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(verify_input);
    {
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        _ = mlx.mlx_vector_array_append_value(vec, t1_arr);
        _ = mlx.mlx_vector_array_append_value(vec, draft_ids);
        try mlx.check(mlx.mlx_concatenate_axis(&verify_input, vec, 1, s));
    }

    // Verify forward with layer captures — this round's verify IS the
    // next round's context producer.
    const cap_out = try allocator.alloc(mlx.mlx_array, model.config.target_layer_ids.len);
    defer {
        for (cap_out) |a| _ = mlx.mlx_array_free(a);
        allocator.free(cap_out);
    }
    for (cap_out) |*a| a.* = mlx.mlx_array_new();
    var cl = model_handle.CaptureLayers{ .ids = model.config.target_layer_ids, .out = cap_out, .hc_mean = xfm.config.hc_count };
    self.ctx.capture_layers = &cl;
    defer self.ctx.capture_layers = null;
    // Per-position SSM capture on a GDN trunk so partial accept can roll
    // back the recurrent state without re-forwarding (mirrors nextMtp).
    self.ctx.capture_ssm_seq = self.ctx.ssm_entries != null;
    var verify_logits = try xfm.forwardWith(&self.ctx, verify_input);
    errdefer _ = mlx.mlx_array_free(verify_logits);
    _ = try self.thinkShiftRows(&verify_logits, verify_input);
    self.ctx.capture_ssm_seq = false;
    defer if (self.ctx.ssm_entries) |entries| {
        for (entries) |*entry| trunk.ssmFreeSpecCapture(entry);
    };
    self.dflash_attempted += 1;
    if (tracing) {
        // Captures ride the same forward — eval them here or their cost
        // is billed to `append`.
        const vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vec);
        _ = mlx.mlx_vector_array_append_value(vec, verify_logits);
        for (cap_out) |a| _ = mlx.mlx_vector_array_append_value(vec, a);
        try mlx.check(mlx.mlx_eval(vec));
        self.dflash_trace.add(.verify, ph.read());
        ph.reset();
    }

    // ── Phase 3: decide the longest accepted prefix ──
    const vl_shape = mlx.getShape(verify_logits);
    var per_pos_logits: ?[]mlx.mlx_array = null;
    var per_pos_logits_n: usize = 0;
    defer if (per_pos_logits) |slots| {
        for (slots[0..per_pos_logits_n]) |arr| _ = mlx.mlx_array_free(arr);
        allocator.free(slots);
    };
    if (stochastic) {
        const slots = try allocator.alloc(mlx.mlx_array, m + 1);
        per_pos_logits = slots;
        const slice_strides = [_]c_int{ 1, 1, 1 };
        for (slots, 0..) |*slot, idx| {
            slot.* = mlx.mlx_array_new();
            per_pos_logits_n = idx + 1;
            const start = [_]c_int{ 0, @intCast(idx), 0 };
            const stop = [_]c_int{ vl_shape[0], @as(c_int, @intCast(idx)) + 1, vl_shape[2] };
            try mlx.check(mlx.mlx_slice(slot, verify_logits, &start, 3, &stop, 3, &slice_strides, 3, s));
        }
    }
    // Committed entries: masked like every other block decoder's.
    var verify_argmax: CommittedArgmax = .{};
    defer verify_argmax.deinit();
    if (!stochastic) {
        verify_argmax = try verifyArgmax(verify_logits, self.sampling.suppress_mask, s);
    }
    _ = mlx.mlx_array_free(verify_logits);
    verify_logits = .{ .ctx = null };

    // One batched dispatch, then read (drafts branch + verify branch are
    // separate graphs — eval BOTH before reading either; the v26.5.6
    // 0%-acceptance class).
    {
        const eval_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(eval_vec);
        _ = mlx.mlx_vector_array_append_value(eval_vec, draft_ids);
        if (!stochastic) _ = mlx.mlx_vector_array_append_value(eval_vec, verify_argmax.lazy());
        try mlx.check(mlx.mlx_async_eval(eval_vec));
    }
    var drafts = try allocator.alloc(u32, m);
    errdefer allocator.free(drafts);
    if (sel_path) |*sp| {
        @memcpy(drafts, sp.ids[0..m]);
    } else {
        try mlx.check(mlx.mlx_array_eval(draft_ids));
        const draft_data = mlx.mlx_array_data_int32(draft_ids) orelse return error.MlxArrayDataNull;
        for (drafts, 0..) |*d, idx| d.* = @intCast(draft_data[idx]);
    }
    if (!stochastic) try mlx.check(mlx.mlx_array_eval(verify_argmax.lazy()));

    var accepted: u32 = 0;
    if (stochastic) {
        var k: u32 = 0;
        while (k < m) : (k += 1) {
            const target_p = try sampler.probsAtLastPos(per_pos_logits.?[k], self.sampling, s);
            defer _ = mlx.mlx_array_free(target_p);
            const p_draft = try sampler.probAt(target_p, drafts[k], s);
            const accept_prob: f32 = if (sel_path) |*sp| blk: {
                // Selector-sampled draft: q is the traced step's own
                // candidate softmax — exact, no GPU read.
                const kk = sp.cand_ids.len / @as(usize, m_full);
                break :blk Generator.specAcceptProb(p_draft, sp.q.?[@as(usize, k) * kk + sp.chosen_idx[k]]);
            } else @min(1.0, p_draft);
            const u: f32 = self.prng.random().float(f32);
            if (u >= accept_prob) break;
            accepted += 1;
        }
    } else {
        const argmax_data = try verify_argmax.ids(1 + m);
        var k: u32 = 0;
        while (k < m) : (k += 1) {
            if (@as(u32, @intCast(argmax_data[k])) != drafts[k]) break;
            accepted += 1;
        }
    }

    self.dflash_drafts_verified += m;
    if (sel_path) |*sp| self.dflash_calib.observeRound(sp.conf, m, accepted);

    // A speculative verify may accept the whole block even when only a
    // few output positions remain. Cap the accepted prefix BEFORE cache
    // and assistant-context commit so every representation of the turn —
    // returned tokens, generated_ids, usage, KV and DFlash context — ends
    // at exactly max_tokens.
    accepted = mtp_plan.capAcceptedForTokenBudget(
        accepted,
        self.completion_tokens,
        self.max_tokens,
    );

    const next_pending: u32 = blk: {
        if (stochastic) {
            const correction_logits = per_pos_logits.?[accepted];
            const probs = try sampler.probsAtLastPos(correction_logits, self.sampling, s);
            defer _ = mlx.mlx_array_free(probs);
            if (accepted < m) {
                // norm(max(0, p − q)) against the q the draft was drawn
                // from — a one-hot here is the WRONG residual for a
                // sampled draft, and wrong silently.
                const q_row = if (sel_path) |*sp|
                    try sampler.selectorQRow(sp, accepted, m_full, vl_shape[2], s)
                else
                    try sampler.pldOneHotRow(drafts[accepted], vl_shape[2], s);
                defer _ = mlx.mlx_array_free(q_row);
                break :blk try sampler.sampleResidual(probs, q_row, s);
            } else {
                break :blk try sampler.sampleFromProbs(probs, s);
            }
        } else {
            const argmax_data = try verify_argmax.ids(1 + m);
            break :blk @intCast(argmax_data[accepted]);
        }
    };

    if (tracing) {
        self.dflash_trace.add(.accept, ph.read());
        ph.reset();
    }

    // ── Phase 4: commit — truncate on partial accept, then grow the
    // assistant context by exactly the committed positions ──
    const n_commit: usize = 1 + @as(usize, accepted);
    if (accepted < m) {
        try self.ctx.cache.truncate(anchor_pos + n_commit, s);
        // Same bookkeeping as nextMtp's GDN arm: preserve the
        // pre-verify cache.step (prefix-cache kv_step contract),
        // roll every linear layer's recurrent state back to the
        // accepted position from the verify pass's capture, and
        // re-point moe_seq_offset at the committed length.
        self.ctx.cache.step = kv_step_snap;
        if (self.ctx.ssm_entries) |entries| {
            const gdn_captured = entries.len > 0 and trunk.ssmHasSpecCapture(&entries[0]);
            if (!gdn_captured) return error.SpecRollbackUnavailable;
            try self.rollbackSsmFromCapture(entries, accepted, 1 + m, s);
        }
        self.ctx.moe_seq_offset.* = anchor_pos + n_commit;
    }
    if (accepted == m) {
        try dflash_mod.appendContext(model, dctx, cap_out, anchor_pos);
    } else {
        const sliced = try allocator.alloc(mlx.mlx_array, cap_out.len);
        var sliced_n: usize = 0;
        defer {
            for (sliced[0..sliced_n]) |a| _ = mlx.mlx_array_free(a);
            allocator.free(sliced);
        }
        for (cap_out, sliced) |full, *out| {
            out.* = mlx.mlx_array_new();
            sliced_n += 1;
            const fsh = mlx.getShape(full);
            const start = [_]c_int{ 0, 0, 0 };
            const stop = [_]c_int{ 1, @intCast(n_commit), fsh[2] };
            const strides = [_]c_int{ 1, 1, 1 };
            try mlx.check(mlx.mlx_slice(out, full, &start, 3, &stop, 3, &strides, 3, s));
        }
        try dflash_mod.appendContext(model, dctx, sliced, anchor_pos);
    }
    // Materialize the appended context off the critical path so next
    // round's assistant forward doesn't drag this round's verify graph.
    {
        const eval_vec = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(eval_vec);
        dctx.appendEvalArrays(eval_vec);
        if (tracing) {
            try mlx.check(mlx.mlx_eval(eval_vec));
            self.dflash_trace.add(.append, ph.read());
            ph.reset();
        } else {
            try mlx.check(mlx.mlx_async_eval(eval_vec));
        }
    }

    const tokens = try allocator.alloc(u32, n_commit);
    tokens[0] = t1;
    for (drafts[0..accepted], 0..) |d, idx| tokens[1 + idx] = d;
    try self.generated_ids.append(allocator, t1);
    for (drafts[0..accepted]) |d| try self.generated_ids.append(allocator, d);
    allocator.free(drafts);

    self.dflash_accepted_tokens += accepted;
    self.next_token_id = next_pending;
    self.advanceStep(@intCast(n_commit));
    checkRuntimeGate(self);
    if (self.completion_tokens >= self.max_tokens) {
        self.done = true;
        self.finish_reason = "length";
    }

    if (tracing) {
        traceRoundEnd(self, accepted);
        self.dflash_gap_watch = io_util.Stopwatch.init(self.timer.io);
    }

    return DrafterStepResult{
        .tokens = tokens,
        .accepted_tokens = accepted,
    };
}

/// Rows are chosen per round where a verify row costs bytes; every other target verifies the whole block.
fn adaptiveRows(self: *const Generator) bool {
    return drafter.dflashAdaptiveRows(self.xfm) and self.dflash_block_size > dflash_mod.minRows();
}

/// DFlash runtime economics gate. Sticky within the request: once a
/// `DFLASH_GATE_WARMUP`-round sample proves the block-parallel path yields
/// less than its width-normalized request-class threshold, subsequent
/// ticks use the regular pipelined decoder through `nextDflash`'s entry
/// fallback.
fn checkRuntimeGate(self: *Generator) void {
    if (self.spec_disabled_runtime) return;
    const serial = dflash_policy.dflashSerialPrice(&self.xfm.round_cost, self.xfm.round_cost.bucketOf(self.mtpKvLen()));
    if (!self.dflash_gate_window.shouldDisable(self.dflash_attempted, self.dflash_accepted_tokens, self.dflash_wall_ms, self.dflashMeanRows(), serial)) return;
    const w = self.dflash_gate_window;
    const rounds = self.dflash_attempted - w.attempted;
    const accepted = self.dflash_accepted_tokens - w.accepted;
    log.info(
        "  dflash=disabled (runtime yield {d:.2} accepted/round, {d:.2} ms/token against {d:.2} serial, over {d} rounds after the warmup)\n",
        .{ @as(f32, @floatFromInt(accepted)) / @as(f32, @floatFromInt(rounds)), (self.dflash_wall_ms - w.wall_ms) / @as(f64, @floatFromInt(rounds + accepted)), serial orelse 0.0, rounds },
    );
    self.stopDflash();
}

var trace_cache: ?bool = null;
fn traceEnabled() bool {
    if (trace_cache) |v| return v;
    const on = spec_types.readEnvBool("SUSHI_DFLASH_TRACE");
    trace_cache = on;
    return on;
}

/// Close one traced dflash round; emits + resets at the cadence.
fn traceRoundEnd(self: *Generator, accepted: u32) void {
    if (!self.dflash_trace.endRound(accepted)) return;
    const t = &self.dflash_trace;
    log.info(
        "  [dflash-trace] rounds={d} avg_ms assist={d:.2} head={d:.2} verify={d:.2} accept={d:.2} append={d:.2} gap={d:.2} total={d:.2} | acc_avg={d:.2}\n",
        .{
            t.rounds,
            t.avgMs(.assist),
            t.avgMs(.head),
            t.avgMs(.verify),
            t.avgMs(.accept),
            t.avgMs(.append),
            t.avgMs(.gap),
            t.totalAvgMs(),
            @as(f64, @floatFromInt(t.accepted)) / @as(f64, @floatFromInt(t.rounds)),
        },
    );
    t.reset();
}
