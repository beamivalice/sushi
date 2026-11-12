const std = @import("std");
const dsv41_state = @import("dsv41_state.zig");
const dspark_rows = @import("../dspark_rows.zig");
const dsv41_serving = @import("dsv41_serving.zig");
const generate_mod = @import("../generate.zig");
const io_util = @import("../io_util.zig");
const log = @import("../log.zig");
const mlx = @import("../mlx.zig");
const mtp_lookup = @import("../mtp_lookup.zig");
const mtp_mod = @import("../mtp.zig");
const mtp_plan = @import("../mtp_plan.zig");
const row_grammar = @import("row_grammar.zig");
const rp_mod = @import("../reasoning_protocol.zig");
const sampler = @import("../sampler.zig");
const Trunk = @import("../trunk.zig").Trunk;
const arch_dspark = @import("../arch/dspark.zig");
const Constraint = generate_mod.Constraint;
const DrafterStepResult = Generator.DrafterStepResult;
const Generator = generate_mod.Generator;
const isEosId = generate_mod.isEosId;
const verifyArgmax = generate_mod.verifyArgmax;

/// Rows a DSpark round may verify so it emits at most `room` tokens and the bound fires at the
/// token serial decode fires at. A round always verifies its t1 row.
pub fn dsparkRowsForRoom(rows: usize, room: ?u32) usize {
    const r = room orelse return rows;
    return @max(1, @min(rows, r));
}

/// Verify width of a seeded sampled DSpark request: the one most rounds chose on the shipped pack, the median of its
/// adaptive widths (2 to 3 rows at ~70% per-draft acceptance), where the round-cost model also puts the best tokens per ms.
const dspark_seeded_rows: u32 = 3;

/// The fixed width a request's DSpark rounds verify at, or null for the adaptive policy. A seeded sampled request
/// must not take its width from measured ms, so a repeat or a stream gets the same rounds.
pub fn dsparkSeededRows(sampling: sampler.SamplingParams, range: dspark_rows.Range) ?usize {
    if (sampling.seed == null or sampler.isGreedyTemperature(sampling.temperature)) return null;
    return std.math.clamp(dspark_seeded_rows, range.min_rows, range.max_rows);
}

pub const dsv41_max_drafts = dspark_rows.max_drafts;

/// The widest verify `dsv41VerifyImpl` accepts, a ring slack's rows.
pub const dsv41_max_verify_rows: u8 = @intCast(@import("dsv41_config.zig").ring_slack);

/// The widest copy round. A verify's rows are their one-row step's bits at every width on a machine with matrix units
/// (`Model.Forward.verify`); without them the trunk linears leave the multi-row arm past
/// `project_rows.PROJECT_ROWS_MAX` for the sorted GEMM, which sums in another order, so a copy round stays inside it
/// on every machine. Adaptive rounds stay inside the block (6).
pub const dsv41_exact_verify_rows: u8 = 8;

/// `SUSHI_DSV41_DSPARK_ROWS` (diagnostic): `sweep` cycles the verify width over `dspark_sweep_rows`, a number forces
/// it (rows past the block repeat its last draft, so only their cost means anything), `log` keeps the policy; each
/// round logs a `[dspark-round]` line.
pub const DsparkDiag = dspark_rows.Diag;

// Odd length: the forwards alternate between a fast and a slow state, and an even cycle would pair one with each width.
const dspark_sweep_rows = [_]u8{ 1, 2, 3, 4, 5, 6, 8 };

var dspark_diag_cache: ?DsparkDiag = null;

pub fn parseDsparkDiag(v: []const u8) DsparkDiag {
    if (std.mem.eql(u8, v, "sweep")) return .sweep;
    if (std.mem.eql(u8, v, "log")) return .log;
    return if (std.fmt.parseInt(u8, v, 10)) |n| .{ .rows = std.math.clamp(n, 1, dsv41_max_verify_rows) } else |_| .off;
}

pub fn dsparkDiag() DsparkDiag {
    if (dspark_diag_cache) |d| return d;
    const d: DsparkDiag = if (std.c.getenv("SUSHI_DSV41_DSPARK_ROWS")) |r| parseDsparkDiag(std.mem.span(r)) else .off;
    dspark_diag_cache = d;
    return d;
}

/// Drafts one stage block holds, from the loaded model; the policy's default where DeepSeek's model is not loaded.
pub fn stageBlock(xfm: *Trunk) u32 {
    return if (dsv41_state.modelOf(xfm)) |m| m.cfg.dspark.block else arch_dspark.default_block;
}

/// The copy arm: the context's continuation after `t1` into `drafts`, as many as a copy round verifies, or 0 for a stage
/// round. Adaptive rounds only; a forced or seeded width keeps its rows.
fn dsparkCopyDrafts(self: *Generator, allocator: std.mem.Allocator, t1: u32, drafts: []u32) !usize {
    const remaining = self.max_tokens - self.completion_tokens;
    const cap = @min(drafts.len, remaining -| 1, (self.dspark_copy_rows orelse dsv41_exact_verify_rows) - 1);
    const block: usize = if (dsv41_state.modelOf(self.xfm)) |m| m.cfg.dspark.block else return 0;
    if (cap <= block) return 0;
    const idx = try self.mtpLookupIndex(allocator);
    const found = idx.matchBest(t1, @intCast(self.dspark_copy_backoff.width(cap)));
    const n = mtp_lookup.copyLen(found, cap, block, mtp_lookup.COPY_SUFFIX);
    if (n == 0 or !self.dspark_copy_backoff.take()) return 0;
    @memcpy(drafts[0..n], found.?.draft[0..n]);
    return n;
}

/// DSpark drafts argmax tokens, a point mass the verify rows accept exactly at any temperature. Penalties shift the
/// rows as in an MTP round, and a thinking bound fires between rounds (`thinkBoundTick`), after which the request
/// decodes serially.
pub const dsparkEligible = dspark_rows.eligible;

/// One DSpark round (deepseek_v41, greedy): the stages draft a block from the pending token, a verify chunk
/// `[t1, drafts]` reads it through the trunk, and the longest prefix of drafts the trunk's own argmax agrees with
/// stays, with the trunk's token after it pending. The model cuts its state back to the kept rows.
pub fn nextDspark(self: *Generator, allocator: std.mem.Allocator) !?DrafterStepResult {
    if (self.done) return null;
    if (!self.dspark) return error.SpecDecodeUnsupported;
    if (self.completion_tokens >= self.max_tokens) {
        self.done = true;
        self.finish_reason = "length";
        return null;
    }
    const constraint = self.sampling.constraint;
    if (constraint) |c| {
        switch (try dsparkConstrainedEntry(self, allocator, c)) {
            .proceed => {},
            .stop => return null,
            .served => |r| return r,
        }
    } else if (try self.checkStop()) return null;
    const xfm = self.xfm;
    const s = xfm.outStream();
    const t1 = self.next_token_id;
    const diag = self.dspark_diag;
    var timer = io_util.Stopwatch.init(self.timer.io);

    const policy = &self.dspark_policy;
    // `--max-mtp-ctx` against prompt + generated, as `mtpRoundBegin` checks it; context only grows, so it is sticky.
    if (!self.spec_disabled_runtime and !mtp_mod.mtpCtxWithinLimit(self.opts.max_mtp_ctx, self.mtpKvLen())) {
        log.info("  dspark=off (ctx {d} > --max-mtp-ctx {d})\n", .{ self.mtpKvLen(), self.opts.max_mtp_ctx });
        self.spec_disabled_runtime = true;
        self.spec_disable_reason = .max_ctx;
    }
    const think_room: ?u32 = if (self.sampling.think_bound) |tb| room: {
        tb.observe(self.generated_ids.items);
        break :room tb.room();
    } else null;
    const adaptive = (diag == .off or diag == .log) and dsparkSeededRows(self.sampling, policy.range) == null;
    var ids_buf: [dsv41_max_drafts + 1]u32 = undefined;
    const copied: usize = if (self.dspark_copy and adaptive and !self.spec_disabled_runtime and (think_room orelse 2) > 1)
        try dsparkCopyDrafts(self, allocator, t1, ids_buf[1..])
    else
        0;
    const policy_rows: ?usize = if (copied > 0) copied + 1 else if (self.spec_disabled_runtime and self.spec_disable_reason == .max_ctx) 1 else switch (diag) {
        .off, .log => dsparkSeededRows(self.sampling, policy.range) orelse if (policy.next() == .plain) 1 else null,
        .rows => |n| n,
        .sweep => dspark_sweep_rows[self.dspark_stats.rounds % dspark_sweep_rows.len],
    };
    const forced: ?usize = if (think_room != null and think_room.? <= 1) 1 else policy_rows;
    ids_buf[0] = t1;
    var conf_buf: [dsv41_max_drafts]f32 = @splat(0);
    var block: ?@import("dsv41_model.zig").Draft = null;
    defer if (block) |*b| b.deinit();
    var block_len: usize = 1 + copied;
    if (copied == 0 and (forced == null or forced.? > 1)) {
        block = try dsv41_serving.dsv41DraftImpl(xfm, t1);
        block_len = block.?.ids.len;
        @memcpy(ids_buf[0..block_len], block.?.ids);
        @memcpy(conf_buf[0 .. block_len - 1], block.?.conf);
    }
    const draft_ns = timer.read();
    timer.reset();
    if (forced) |f| if (f > block_len) for (block_len..f) |i| {
        ids_buf[i] = ids_buf[block_len - 1];
    };
    // The budget pre-commit invariant: t1 and the kept drafts together fit the tokens still allowed.
    const wanted: usize = forced orelse policy.rows(conf_buf[0 .. block_len - 1]);
    var rows: usize = dsparkRowsForRoom(mtp_plan.rowsThroughId(ids_buf[0..@min(wanted, self.max_tokens - self.completion_tokens)], self.dspark_round_ends_at), think_room);
    const vocab: usize = if (dsv41_state.modelOf(xfm)) |m| m.cfg.vocab else 0;
    var rows_masked = false;
    if (constraint) |c| {
        var mask_timer = io_util.Stopwatch.init(self.timer.io);
        try self.dspark_masks.resize(allocator, rows * vocab);
        const planned = try row_grammar.plan(rowView(self, c), t1, ids_buf[1..rows], self.dspark_masks.items, vocab);
        rows = planned.rows;
        rows_masked = planned.masked;
        self.dspark_stats.mask_ns += mask_timer.read();
    }
    const ctx_before = self.ctx.cache.step;
    var logits = try dsv41_serving.dsv41VerifyImpl(xfm, &self.ctx, ids_buf[0..rows]);
    defer _ = mlx.mlx_array_free(logits);
    if (self.sampling.think_penalty.active() or sampler.penaltyActive(self.sampling)) {
        var row_ids: [dsv41_max_drafts + 1]i32 = undefined;
        if (rows > row_ids.len) return error.SpecDecodeUnsupported;
        for (ids_buf[0..rows], row_ids[0..rows]) |id, *o| o.* = @intCast(id);
        const row_ids_arr = mlx.mlx_array_new_data(&row_ids, &[_]c_int{ 1, @intCast(rows) }, 2, .int32);
        defer _ = mlx.mlx_array_free(row_ids_arr);
        _ = try self.thinkShiftRows(&logits, row_ids_arr);
        _ = try self.penaltyShiftRows(&logits, row_ids_arr);
    }
    var masked_logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(masked_logits);
    if (rows_masked) try sampler.applyRowMasks(&masked_logits, logits, self.dspark_masks.items[0 .. rows * vocab], rows, vocab, s);
    const verdict_logits = if (rows_masked) masked_logits else logits;
    var accepted: u32 = 0;
    var pending: u32 = 0;
    if (sampler.isGreedyTemperature(self.sampling.temperature)) {
        var verified = try verifyArgmax(verdict_logits, self.sampling.suppress_mask, dsv41_serving.verdictStream(xfm, s));
        defer verified.deinit();
        try mlx.check(mlx.mlx_array_eval(verified.lazy()));
        const best = try verified.ids(rows);
        while (accepted + 1 < rows and @as(u32, @intCast(best[accepted])) == ids_buf[1 + accepted]) accepted += 1;
        pending = @intCast(best[accepted]);
    } else {
        const verdict = try dsparkSampledVerdict(self, verdict_logits, ids_buf[1..rows]);
        accepted = verdict.accepted;
        pending = verdict.pending;
    }
    try dsv41_serving.dsv41FinishImpl(xfm, &self.ctx, accepted);
    const verify_ns = timer.read();

    const n_commit: usize = 1 + accepted;
    const tokens = try allocator.dupe(u32, ids_buf[0..n_commit]);
    errdefer allocator.free(tokens);
    if (constraint) |c| for (tokens, self.generated_ids.items.len..) |t, index| try dsparkObserve(self, c, index, t);
    try self.generated_ids.appendSlice(allocator, tokens);
    self.advanceStep(@intCast(n_commit));
    self.next_token_id = pending;
    self.dspark_pending = constraint != null;
    const st = &self.dspark_stats;
    st.rounds += 1;
    st.plain += @intFromBool(rows == 1);
    st.rows_hist[rows] += 1;
    st.drafted += rows - 1;
    st.accepted += accepted;
    st.copy_rounds += @intFromBool(copied > 0);
    st.copy_drafted += if (copied > 0) rows - 1 else 0;
    st.copy_accepted += if (copied > 0) accepted else 0;
    st.draft_ns += draft_ns;
    st.verify_ns += verify_ns;
    if (diag == .off or diag == .log) {
        const ms = struct {
            fn of(ns: u64) f32 {
                return @floatCast(@as(f64, @floatFromInt(ns)) / 1e6);
            }
        }.of;
        if (copied > 0) {
            // The stage policy learns from stage blocks only: a wide copy verify would move its fitted row price.
            self.dspark_copy_backoff.observeCopy(rows - 1, accepted, ms(draft_ns + verify_ns));
        } else {
            self.dspark_copy_backoff.observeStage(n_commit, ms(draft_ns + verify_ns));
            if (block == null) policy.observePlain(ms(verify_ns)) else policy.observeRound(conf_buf[0 .. block_len - 1], @intCast(rows), accepted, ms(draft_ns), ms(verify_ns));
        }
    }
    if (dsparkDiag() != .off) log.info("  [dspark-round] ctx={d} rows={d} accepted={d} draft_ms={d:.1} verify_ms={d:.1} gate={s} conf={any}\n", .{
        ctx_before,
        rows,
        accepted,
        @as(f64, @floatFromInt(draft_ns)) / 1e6,
        @as(f64, @floatFromInt(verify_ns)) / 1e6,
        @tagName(policy.gate.mode),
        conf_buf[0 .. block_len - 1],
    });
    if (self.completion_tokens >= self.max_tokens) {
        self.done = true;
        self.finish_reason = "length";
    }
    return .{ .tokens = tokens, .accepted_tokens = accepted };
}

pub const ConstrainedEntry = union(enum) { proceed, stop, served: DrafterStepResult };

pub const Pending = union(enum) { commit, stop, forced: u32 };

pub fn rowView(self: *const Generator, c: *Constraint) row_grammar.View {
    return .{ .grammar = c.grammar, .token_bytes = c.token_bytes, .proto = c.proto, .pstate = &c.pstate, .eos_ids = self.eos_token_ids };
}

pub fn dsparkNoteSerial(self: *Generator, why: []const u8) void {
    if (self.dspark_serial_logged) return;
    self.dspark_serial_logged = true;
    log.info("  [spec] dspark: this request decodes serially while {s}\n", .{why});
}

/// The start of a constrained round: serial ticks where the constraint cannot be verified in rows, else the
/// pending token (sampled here from the serial state's logits on the first round) judged as serial decode would.
pub fn dsparkConstrainedEntry(self: *Generator, allocator: std.mem.Allocator, c: *Constraint) !ConstrainedEntry {
    if (self.checkBudgetStop()) return .stop;
    const phase = c.pstate.phase;
    const vocab: usize = if (dsv41_state.modelOf(self.xfm)) |m| m.cfg.vocab else 0;
    const why: ?[]const u8 = if (c.pstate.recovering)
        "a forced boundary drains"
    else if (phase != .reasoning and phase != .json_body)
        "the reasoning protocol is at a channel header"
    else if (c.token_bytes.bytes.len > vocab)
        "the token table is wider than the vocabulary"
    else
        null;
    if (why) |w| {
        dsparkNoteSerial(self, w);
        self.dspark_stats.serial += 1;
        const token = (try self.next(allocator)) orelse return .stop;
        return .{ .served = .{ .tokens = try allocator.dupe(u32, &.{token}), .accepted_tokens = 0 } };
    }
    if (self.has_pending_logits) try dsparkSamplePending(self, allocator, c);
    return switch (try dsparkJudgePending(self, allocator, c)) {
        .commit => .proceed,
        .stop => .stop,
        .forced => |t| .{ .served = .{ .tokens = try allocator.dupe(u32, &.{t}), .accepted_tokens = 0 } },
    };
}

/// The serial state's pending logits become a pending token, sampled under the mask of the committed state.
pub fn dsparkSamplePending(self: *Generator, allocator: std.mem.Allocator, c: *Constraint) !void {
    const logits = self.pending_logits;
    self.has_pending_logits = false;
    defer _ = mlx.mlx_array_free(logits);
    var masked = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(masked);
    const kind = try row_grammar.maskFor(rowView(self, c), c.mask_buf);
    if (kind == .masked) try sampler.applyGrammarMask(allocator, &masked, logits, c.mask_buf, self.xfm.s);
    const lazy = try self.sampleLazy(if (kind == .masked) masked else logits, .committed);
    defer _ = mlx.mlx_array_free(lazy);
    try mlx.check(mlx.mlx_array_eval(lazy));
    var val: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&val, lazy));
    self.next_token_id = @intCast(val);
    self.dspark_pending = true;
}

/// What serial decode does with the token it just sampled before committing it: an EOS stops (inside the
/// reasoning it forces the boundary instead), and so do three pad tokens in a row.
pub fn dsparkJudgePending(self: *Generator, allocator: std.mem.Allocator, c: *Constraint) !Pending {
    const t = self.next_token_id;
    const eos = isEosId(t, self.eos_token_ids);
    const pad_run = t == 0 and self.consecutive_pad + 1 >= 3;
    if (c.pstate.phase == .reasoning) {
        const proto = c.proto.?;
        const closes = eos and proto.closer_atomic != null and t == proto.closer_atomic.?;
        if ((eos and !closes) or pad_run) {
            self.dspark_pending = false;
            switch (try self.forceConstraintTransition(allocator)) {
                .committed => |forced| {
                    log.info("[grammar] reasoning boundary forced after {s}\n", .{if (eos) "premature EOS" else "terminal padding"});
                    return .{ .forced = forced };
                },
                .activated => unreachable,
                .refused => {
                    self.done = true;
                    self.finish_reason = "stop";
                    self.stop_token = t;
                    return .stop;
                },
            }
        }
    } else if (eos or pad_run) {
        self.dspark_pending = false;
        self.done = true;
        self.finish_reason = "stop";
        self.stop_token = t;
        return .stop;
    }
    self.consecutive_pad = if (t == 0) self.consecutive_pad + 1 else 0;
    return .commit;
}

/// Commits `token`, the generated token at `index`, to the constraint as serial decode does after sampling it.
pub fn dsparkObserve(self: *Generator, c: *Constraint, index: usize, token: u32) !void {
    const adv = try row_grammar.advance(rowView(self, c), token);
    if (!adv.crossed) return;
    const bytes = if (token < c.token_bytes.bytes.len) c.token_bytes.bytes[token] else null;
    c.pending_span = if (bytes == null)
        .{ .token_index = @intCast(index + 1), .byte_offset = 0 }
    else
        .{ .token_index = @intCast(index), .byte_offset = rp_mod.payloadByteOffset(bytes, adv.suffix) };
    self.loop_guard_start = index;
    log.info("[grammar] reasoning boundary reached; enforcing final-answer schema\n", .{});
}

/// Serial decode taking over from a DSpark round: emit the pending token, then forward it for the next sample.
pub fn dsparkEmitPending(self: *Generator, allocator: std.mem.Allocator) !?u32 {
    const c = self.sampling.constraint.?;
    if (self.checkBudgetStop()) return null;
    switch (try dsparkJudgePending(self, allocator, c)) {
        .stop => return null,
        .forced => |t| return t,
        .commit => {},
    }
    self.dspark_pending = false;
    const token = self.next_token_id;
    try dsparkObserve(self, c, self.generated_ids.items.len, token);
    self.advanceStep(1);
    try self.generated_ids.append(allocator, token);
    if (self.step >= self.max_tokens) {
        self.done = true;
        self.finish_reason = "length";
        return token;
    }
    const tok_i32: i32 = @intCast(token);
    const tok_input = mlx.mlx_array_new_data(&tok_i32, &[_]c_int{ 1, 1 }, 2, .int32);
    defer _ = mlx.mlx_array_free(tok_input);
    self.pending_logits = try self.xfm.forwardWith(&self.ctx, tok_input);
    self.has_pending_logits = true;
    return token;
}

/// The longest prefix of point-mass `drafts` that the verify rows' own draws reproduce, and the draw after it. Row
/// j draws with the key of the generated position it predicts, so the tokens are the same whatever the round mix. A
/// point-mass draft survives a draw of the row's density with probability p(draft): the exact rule.
pub fn dsparkSampledVerdict(self: *Generator, logits: mlx.mlx_array, drafts: []const u32) !struct { accepted: u32, pending: u32 } {
    const s = dsv41_serving.verdictStream(self.xfm, self.xfm.s);
    const flat = try dsv41_serving.sampledDraws(self.sampling, logits, drafts.len + 1, s);
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_array_eval(flat));
    const drawn = mlx.mlx_array_data_int32(flat) orelse return error.MlxArrayDataNull;
    var accepted: u32 = 0;
    while (accepted < drafts.len and @as(u32, @intCast(drawn[accepted])) == drafts[accepted]) accepted += 1;
    self.sampling.draw +%= accepted + 1;
    return .{ .accepted = accepted, .pending = @intCast(drawn[accepted]) };
}

pub const DsparkResume = dspark_rows.Resume;

/// A forced commit left the shim-seeded state; drain it to DSpark's round entry (the closer published, its
/// successor sampled and not in the cache). The stage windows already hold the forced rows, since every
/// forward commits its main rows. The thought is closed, so its restore-point cut is disarmed.
pub fn dsparkResumeAfterForced(self: *Generator, allocator: std.mem.Allocator) !DsparkResume {
    if (!self.dspark or self.spec_disabled_runtime or self.sampling.constraint != null) return .refused;
    switch (try self.drainPipelineForSpec(allocator)) {
        .drained => |closer| {
            self.dspark_round_ends_at = null;
            log.info("  [dspark] resumed after the forced thought close at {d} generated tokens\n", .{self.generated_ids.items.len});
            return .{ .resumed = closer };
        },
        .stopped => return .stopped,
        .already_clean, .stay_disabled => return .refused,
    }
}
