//! The drafter rounds, shared by every arch whose drafter drafts a block (DeepSeek's DSpark, Hy4-Flash's DFly2): a
//! block drafted from the pending token, a verify chunk `[t1, drafts]` read through the trunk, the longest prefix of
//! drafts the trunk agrees with kept, and the model cut back to the kept rows. The arch supplies the model's side as
//! hooks (`Rounds(A)`):
//! - `Draft` (`ids` with the pending token first, `conf` a draft, `deinit`), `draft(xfm, ctx, t1)`,
//!   `verify(xfm, ctx, ids)` (the rows' logits), `finish(xfm, ctx, accepted)` (keep the first `accepted + 1` rows);
//! - `verdictStream(xfm, s)`, `sampledDraws(sampling, logits, rows, s)`, `vocab(xfm)`, `stageBlock(xfm)`,
//!   `copyArm(xfm)` (null: no copy rounds), `copy_narrow`, `max_verify_rows`, `diag_env`, `row_grammar`.
//! - optional: `draftMeasured(xfm, ctx, t1, limit, meter)`, a draft cut to the policy's `limit` drafts and marked on the
//!   round's meter.
const std = @import("std");
const drafter_rows = @import("drafter_rows.zig");
const generate_mod = @import("generate.zig");
const io_util = @import("io_util.zig");
const log = @import("log.zig");
const mlx = @import("mlx.zig");
const mtp_lookup = @import("mtp_lookup.zig");
const mtp_mod = @import("mtp.zig");
const mtp_plan = @import("mtp_plan.zig");
const rp_mod = @import("reasoning_protocol.zig");
const sampler = @import("sampler.zig");
const Trunk = @import("trunk.zig").Trunk;
const Constraint = generate_mod.Constraint;
const DrafterStepResult = Generator.DrafterStepResult;
const Generator = generate_mod.Generator;
const isEosId = generate_mod.isEosId;
const verifyArgmax = generate_mod.verifyArgmax;
const arch_cooler = @import("arch/cooler.zig");
const think_penalty = @import("think_penalty.zig");
const seen_penalty = @import("seen_penalty.zig");
pub const RoundMeter = @import("round_meter.zig").RoundMeter;

/// What a copy round may verify: the widest verify whose rows are the serial step's, and the stage block.
pub const CopyArm = struct { exact_rows: usize, block: usize };

pub fn Rounds(comptime A: type) type {
    return struct {
        /// The rows of `ids` a round of `wanted` publishes: no more than the tokens still allowed, and nothing past an
        /// end-of-turn draft, where the reply's restore point stands.
        pub fn roundRows(self: *const Generator, ids: []const u32, wanted: usize) usize {
            return mtp_plan.rowsThroughAny(ids[0..@min(wanted, self.max_tokens - self.completion_tokens)], self.eos_token_ids);
        }

        /// Rows a drafter round may verify so it emits at most `room` tokens and the bound fires at the
        /// token serial decode fires at. A round always verifies its t1 row.
        pub fn drafterRowsForRoom(rows: usize, room: ?u32) usize {
            const r = room orelse return rows;
            return @max(1, @min(rows, r));
        }

        /// The width every drafter round verifies at where a row's bytes move with the width: the one most rounds chose
        /// on the shipped pack, the median of its adaptive widths (2 to 3 rows at ~70% per-draft acceptance), where the
        /// round-cost model also puts the best tokens per ms.
        const drafter_fixed_rows: u32 = 3;

        /// The fixed width a request's drafter rounds verify at, or null for the adaptive policy. Where a row's bytes
        /// move with the verify width, a request whose text repeats must not take its width from measured ms, or its
        /// text would follow the timings: a greedy one on a layout that is not width-exact (`layout_exact`, the arch's
        /// `widthExact`: the route keeps greedy rows exact on an STQ layout), and a seeded sampled one where its draws
        /// are not (`exact_rows`: on the engine and that layout). An unseeded sampled request repeats nothing and keeps
        /// the policy.
        pub fn drafterFixedRows(sampling: sampler.SamplingParams, range: drafter_rows.Range, exact_rows: bool, layout_exact: bool) ?usize {
            const exact = if (sampler.isGreedyTemperature(sampling.temperature)) layout_exact else if (sampling.seed != null) exact_rows else true;
            if (exact) return null;
            return std.math.clamp(drafter_fixed_rows, range.min_rows, range.max_rows);
        }

        pub const max_drafts = drafter_rows.max_drafts;

        /// The widest verify the arch's `verify` accepts.
        pub const max_verify_rows: u8 = A.max_verify_rows;

        /// `SUSHI_DSV41_DSPARK_ROWS` (diagnostic): `sweep` cycles the verify width over `drafter_sweep_rows`, a number
        /// forces it (rows past the block repeat its last draft, so only their cost means anything), `log` keeps the
        /// policy; each round logs a `[drafter-round]` line.
        pub const DrafterDiag = drafter_rows.Diag;

        // Odd length: the forwards alternate between a fast and a slow state, and an even cycle would pair one with each width.
        const drafter_sweep_rows = [_]u8{ 1, 2, 3, 4, 5, 6, 8 };

        var drafter_diag_cache: ?DrafterDiag = null;

        pub fn parseDrafterDiag(v: []const u8) DrafterDiag {
            if (std.mem.eql(u8, v, "sweep")) return .sweep;
            if (std.mem.eql(u8, v, "log")) return .log;
            return if (std.fmt.parseInt(u8, v, 10)) |n| .{ .rows = std.math.clamp(n, 1, max_verify_rows) } else |_| .off;
        }

        pub fn drafterDiag() DrafterDiag {
            if (drafter_diag_cache) |d| return d;
            const d: DrafterDiag = if (std.c.getenv(A.diag_env)) |r| parseDrafterDiag(std.mem.span(r)) else .off;
            drafter_diag_cache = d;
            return d;
        }

        /// Drafts one stage block holds, from the loaded model (the arch's own default where its drafter is not loaded).
        pub fn stageBlock(xfm: *Trunk) u32 {
            return A.stageBlock(xfm);
        }

        /// Drafts of the widest pass that still queues its routed chains: a copy round past them pays a wait a layer.
        const copy_narrow: usize = A.copy_narrow;

        /// The copy arm: the context's continuation after `t1` into `drafts`, as many as a copy round verifies, or 0 for a stage
        /// round. Adaptive rounds only; a forced or seeded width keeps its rows.
        fn drafterCopyDrafts(self: *Generator, allocator: std.mem.Allocator, t1: u32, drafts: []u32) !usize {
            const arm = A.copyArm(self.xfm) orelse return 0;
            const remaining = self.max_tokens - self.completion_tokens;
            // A copy round is as wide as a verify whose rows are the serial step's; adaptive rounds stay inside the block.
            const rows = self.drafter_copy_rows orelse @min(max_verify_rows, arm.exact_rows);
            const cap = @min(drafts.len, remaining -| 1, rows - 1);
            const block: usize = arm.block;
            if (cap <= block) return 0;
            const idx = try self.mtpLookupIndex(allocator);
            const match = idx.matchBest(t1, @intCast(cap)) orelse return 0;
            const available = idx.copyAvailable(t1, match, block);
            const width_cap = if (match.suffix >= mtp_lookup.STRONG_SUFFIX) cap else @min(cap, copy_narrow);
            const n = @min(available, self.drafter_copy_backoff.width(width_cap, copy_narrow));
            if (n == 0 or !self.drafter_copy_backoff.take()) return 0;
            @memcpy(drafts[0..n], match.draft[0..n]);
            return n;
        }

        /// A full stage block whose rows the context already holds whole is extended by the context's continuation, as
        /// many rows again at most, inside the widest serial-exact verify. Adaptive stage rounds of an arch with copy
        /// rounds only.
        fn extendStage(self: *Generator, allocator: std.mem.Allocator, ids: []u32, rows: usize) !usize {
            if (!self.drafter_copy or self.spec_disabled_runtime or @import("mtp_depth_bounds.zig").active.explicit()) return rows;
            const arm = A.copyArm(self.xfm) orelse return rows;
            if (rows != arm.block + 1 or rows <= mtp_lookup.NGRAM) return rows;
            const cap = @min(ids.len, self.drafter_copy_rows orelse max_verify_rows, arm.exact_rows, self.max_tokens - self.completion_tokens);
            const extra = @min(cap -| rows, rows - 1);
            if (extra == 0) return rows;
            const idx = try self.mtpLookupIndex(allocator);
            const match = idx.matchTail(ids[0..rows], @intCast(extra)) orelse return rows;
            if (match.suffix < rows - 1) return rows;
            @memcpy(ids[rows..][0..match.draft.len], match.draft);
            return rows + match.draft.len;
        }

        /// A block drafter drafts argmax tokens, a point mass the verify rows accept exactly at any temperature.
        /// Penalties shift the rows as in an MTP round, and a thinking bound fires between rounds (`thinkBoundTick`),
        /// after which the request decodes serially.
        pub const drafterEligible = drafter_rows.eligible;

        fn plainAfter(self: *const Generator, fixed_rows: ?usize, steps: u32) bool {
            if (self.spec_disabled_runtime and self.spec_disable_reason == .max_ctx) return true;
            return switch (self.drafter_diag) {
                .off, .log => fixed_rows == null and self.drafter_policy.nextIsPlainAfter(steps),
                .rows => |n| n == 1,
                .sweep => false,
            };
        }

        fn drafterPlainPipelineRun(
            self: *Generator,
            allocator: std.mem.Allocator,
            timer: *io_util.Stopwatch,
            ctx_before: usize,
            drain_only: bool,
            keep_after_next: bool,
        ) !?DrafterStepResult {
            const xfm = self.xfm;
            const tok: u32 = if (drain_only) switch (try self.drainPipelineForSpec(allocator)) {
                .stopped => {
                    self.dropPending();
                    return null;
                },
                .drained => |t| t,
                .already_clean => unreachable,
            } else blk: {
                const was_keep = A.setPlainKeepHidden(xfm, keep_after_next);
                defer _ = A.setPlainKeepHidden(xfm, was_keep);
                break :blk (try self.next(allocator)) orelse return null;
            };
            const verify_ns = timer.read();
            const tokens = try allocator.dupe(u32, &.{tok});
            errdefer allocator.free(tokens);
            self.drafter_pending = false;
            const st = &self.drafter_stats;
            st.rounds += 1;
            st.plain += 1;
            st.rows_hist[1] += 1;
            st.verify_ns += verify_ns;
            if (self.drafter_diag == .off or self.drafter_diag == .log) {
                const ms: f32 = @floatCast(@as(f64, @floatFromInt(verify_ns)) / 1e6);
                self.drafter_copy_backoff.observeStage(1, ms);
                self.drafter_policy.observePipelinedPlain(ms);
            }
            if (drafterDiag() != .off) log.info("  [drafter-round] ctx={d} rows=1 accepted=0 draft_ms=0.0 verify_ms={d:.1} gate={s} conf={{}}\n", .{
                ctx_before,
                @as(f64, @floatFromInt(verify_ns)) / 1e6,
                @tagName(self.drafter_policy.gate.mode),
            });
            if (self.completion_tokens >= self.max_tokens) {
                self.done = true;
                self.finish_reason = "length";
            }
            return .{ .tokens = tokens, .accepted_tokens = 0 };
        }

        fn drafterPlainPipelineTick(self: *Generator, allocator: std.mem.Allocator) !?DrafterStepResult {
            const xfm = self.xfm;
            const s = xfm.outStream();
            var timer = io_util.Stopwatch.init(self.timer.io);
            if (!self.spec_disabled_runtime and !mtp_mod.mtpCtxWithinLimit(self.opts.max_mtp_ctx, self.mtpKvLen())) {
                log.info("  drafter=off (ctx {d} > --max-mtp-ctx {d})\n", .{ self.mtpKvLen(), self.opts.max_mtp_ctx });
                self.spec_disabled_runtime = true;
                self.spec_disable_reason = .max_ctx;
            }
            const think_room: ?u32 = if (self.sampling.think_bound) |tb| room: {
                tb.observe(self.generated_ids.items);
                break :room tb.room();
            } else null;
            const layout_exact = if (comptime @hasDecl(A, "widthExact")) A.widthExact(xfm) else false;
            const fixed_rows = drafterFixedRows(self.sampling, self.drafter_policy.range, mlx.engine.isStream(s) and layout_exact, layout_exact);
            const policy_plain = if (self.spec_disabled_runtime and self.spec_disable_reason == .max_ctx)
                true
            else switch (self.drafter_diag) {
                .off, .log => fixed_rows == null and self.drafter_policy.next() == .plain,
                .rows => |n| n == 1,
                .sweep => false,
            };
            const ctx_before = self.ctx.cache.step;
            const can_continue = policy_plain and !self.samplesSync() and self.sampling.call_force == null and
                self.drafter_round_ends_at == null and
                (think_room == null or think_room.? > 1) and
                self.step + 1 < self.max_tokens and
                plainAfter(self, fixed_rows, 1);
            const keep_after_next = !plainAfter(self, fixed_rows, 2) or self.step + 2 >= self.max_tokens or (think_room != null and think_room.? <= 2);
            return drafterPlainPipelineRun(self, allocator, &timer, ctx_before, !can_continue, keep_after_next);
        }

        /// One drafter round (greedy): the stages draft a block from the pending token, a verify chunk `[t1, drafts]`
        /// reads it through the trunk, and the longest prefix of drafts the trunk's own argmax agrees with stays, with
        /// the trunk's token after it pending. The model cuts its state back to the kept rows.
        pub fn nextDrafterRound(self: *Generator, allocator: std.mem.Allocator) !?DrafterStepResult {
            if (drafterDiag() == .off) return measuredRound(self, allocator, null);
            var meter = RoundMeter.init(self.timer.io);
            return measuredRound(self, allocator, &meter);
        }

        /// `nextDrafterRound` with each phase of the round marked on `meter`.
        pub fn measuredRound(self: *Generator, allocator: std.mem.Allocator, meter: ?*RoundMeter) !?DrafterStepResult {
            if (self.done) return null;
            if (!self.drafter_on) return error.SpecDecodeUnsupported;
            if (self.completion_tokens >= self.max_tokens) {
                self.dropPending();
                self.done = true;
                self.finish_reason = "length";
                return null;
            }
            const constraint = self.sampling.constraint;
            if (constraint) |c| {
                switch (try drafterConstrainedEntry(self, allocator, c)) {
                    .proceed => {},
                    .stop => return null,
                    .served => |r| return r,
                }
            } else if (comptime @hasDecl(A, "setPlainKeepHidden")) {
                if (self.pending != null) return try drafterPlainPipelineTick(self, allocator);
                if (try self.checkStop()) return null;
            } else if (try self.checkStop()) return null;
            const xfm = self.xfm;
            const s = xfm.outStream();
            const t1 = self.next_token_id;
            const diag = self.drafter_diag;
            var timer = io_util.Stopwatch.init(self.timer.io);

            const policy = &self.drafter_policy;
            // `--max-mtp-ctx` against prompt + generated, as `mtpRoundBegin` checks it; context only grows, so it is sticky.
            if (!self.spec_disabled_runtime and !mtp_mod.mtpCtxWithinLimit(self.opts.max_mtp_ctx, self.mtpKvLen())) {
                log.info("  drafter=off (ctx {d} > --max-mtp-ctx {d})\n", .{ self.mtpKvLen(), self.opts.max_mtp_ctx });
                self.spec_disabled_runtime = true;
                self.spec_disable_reason = .max_ctx;
            }
            const think_room: ?u32 = if (self.sampling.think_bound) |tb| room: {
                tb.observe(self.generated_ids.items);
                break :room tb.room();
            } else null;
            const layout_exact = if (comptime @hasDecl(A, "widthExact")) A.widthExact(xfm) else false;
            const fixed_rows = drafterFixedRows(self.sampling, policy.range, mlx.engine.isStream(s) and layout_exact, layout_exact);
            const adaptive = (diag == .off or diag == .log) and fixed_rows == null;
            var ids_buf: [max_drafts + 1]u32 = undefined;
            const copied: usize = if (self.drafter_copy and adaptive and !self.spec_disabled_runtime and (think_room orelse 2) > 1)
                try drafterCopyDrafts(self, allocator, t1, ids_buf[1..])
            else
                0;
            const policy_rows: ?usize = if (copied > 0) copied + 1 else if (self.spec_disabled_runtime and self.spec_disable_reason == .max_ctx) 1 else switch (diag) {
                .off, .log => fixed_rows orelse if (policy.next() == .plain) 1 else null,
                .rows => |n| n,
                .sweep => drafter_sweep_rows[self.drafter_stats.rounds % drafter_sweep_rows.len],
            };
            const forced: ?usize = if (think_room != null and think_room.? <= 1) 1 else policy_rows;
            ids_buf[0] = t1;
            var conf_buf: [max_drafts]f32 = @splat(0);
            var block: ?A.Draft = null;
            defer if (block) |*b| b.deinit();
            var block_len: usize = 1 + copied;
            if (meter) |m| m.mark(.entry);
            if (copied == 0 and (forced == null or forced.? > 1)) {
                // An arch that drafts to a limit takes the policy's shorter block after misses.
                block = if (comptime @hasDecl(A, "draftMeasured"))
                    try A.draftMeasured(xfm, &self.ctx, t1, if (adaptive and forced == null) policy.draftLimit() else null, meter)
                else
                    try A.draft(xfm, &self.ctx, t1);
                block_len = block.?.ids.len;
                @memcpy(ids_buf[0..block_len], block.?.ids);
                @memcpy(conf_buf[0 .. block_len - 1], block.?.conf);
                if (meter) |m| m.mark(.draft_read);
            }
            const draft_ns = timer.read();
            timer.reset();
            if (forced) |f| if (f > block_len) for (block_len..f) |i| {
                ids_buf[i] = ids_buf[block_len - 1];
            };
            // The budget pre-commit invariant: t1 and the kept drafts together fit the tokens still allowed.
            const wanted = forced orelse try extendStage(self, allocator, &ids_buf, policy.rows(conf_buf[0 .. block_len - 1]));
            const turn = ids_buf[0..roundRows(self, &ids_buf, wanted)];
            var rows: usize = drafterRowsForRoom(mtp_plan.rowsThroughId(turn, self.drafter_round_ends_at), think_room);
            const vocab: usize = A.vocab(xfm);
            var rows_masked = false;
            if (constraint) |c| {
                var mask_timer = io_util.Stopwatch.init(self.timer.io);
                try self.drafter_masks.resize(allocator, rows * vocab);
                const planned = try A.row_grammar.plan(rowView(self, c), t1, ids_buf[1..rows], self.drafter_masks.items, vocab);
                rows = planned.rows;
                rows_masked = planned.masked;
                self.drafter_stats.mask_ns += mask_timer.read();
            }
            if (meter) |m| m.mark(.plan);
            const ctx_before = self.ctx.cache.step;
            const next_plain = rows == 1 and block == null and copied == 0 and plainAfter(self, fixed_rows, 1);
            if (comptime @hasDecl(A, "setPlainKeepHidden")) {
                if (next_plain and constraint == null and !self.samplesSync() and self.sampling.call_force == null and
                    self.drafter_round_ends_at == null and (think_room == null or think_room.? > 1) and
                    self.step + 1 < self.max_tokens)
                {
                    const keep_after_next = !plainAfter(self, fixed_rows, 2) or self.step + 2 >= self.max_tokens or (think_room != null and think_room.? <= 2);
                    return drafterPlainPipelineRun(self, allocator, &timer, ctx_before, false, keep_after_next);
                }
            }
            const logits = if (comptime @hasDecl(A, "verifyPlain"))
                if (next_plain) try A.verifyPlain(xfm, &self.ctx, ids_buf[0]) else try A.verify(xfm, &self.ctx, ids_buf[0..rows])
            else
                try A.verify(xfm, &self.ctx, ids_buf[0..rows]);
            defer _ = mlx.mlx_array_free(logits);
            if (meter) |m| m.mark(.verify);
            // An arch whose open verify holds state until `finish` gives it up if the round fails before then.
            var finished = false;
            errdefer if (!finished) if (comptime @hasDecl(A, "abort")) A.abort(xfm, &self.ctx);
            const verdict = try sampleVerdict(self, logits, ids_buf[0..rows], if (rows_masked) self.drafter_masks.items[0 .. rows * vocab] else null);
            const accepted = verdict.accepted;
            const pending = verdict.pending;
            if (meter) |m| m.mark(.verdict);
            finished = true;
            try A.finish(xfm, &self.ctx, accepted);
            const verify_ns = timer.read();
            if (meter) |m| m.mark(.finish);

            const n_commit: usize = 1 + accepted;
            const tokens = try allocator.dupe(u32, ids_buf[0..n_commit]);
            errdefer allocator.free(tokens);
            if (constraint) |c| for (tokens, self.generated_ids.items.len..) |t, index| try drafterObserve(self, c, index, t);
            try self.generated_ids.appendSlice(allocator, tokens);
            self.advanceStep(@intCast(n_commit));
            self.next_token_id = pending;
            self.drafter_pending = constraint != null;
            const st = &self.drafter_stats;
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
                const draft_ms: f32 = if (comptime @import("builtin").is_test)
                    if (self.drafter_timing) |c| (if (block != null) c.draft_ms else 0) else ms(draft_ns)
                else
                    ms(draft_ns);
                const verify_ms: f32 = if (comptime @import("builtin").is_test)
                    if (self.drafter_timing) |c| c.verifyMs(@intCast(rows)) else ms(verify_ns)
                else
                    ms(verify_ns);
                if (copied > 0) {
                    // The stage policy learns from stage blocks only: a wide copy verify would move its fitted row price.
                    self.drafter_copy_backoff.observeCopy(rows - 1, accepted, draft_ms + verify_ms, copy_narrow);
                } else {
                    self.drafter_copy_backoff.observeStage(n_commit, draft_ms + verify_ms);
                    if (block == null) policy.observePlain(verify_ms) else policy.observeRound(conf_buf[0 .. block_len - 1], @intCast(rows), accepted, draft_ms, verify_ms);
                }
            }
            if (meter) |m| {
                m.mark(.publish);
                if (drafterDiag() != .off) m.report(ctx_before, rows, accepted, if (copied > 0) "copy" else if (block != null) "stage" else "plain");
            }
            if (drafterDiag() != .off) log.info("  [drafter-round] ctx={d} rows={d} accepted={d} draft_ms={d:.1} verify_ms={d:.1} gate={s} conf={any}\n", .{
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

        pub fn settleDrafted(self: *Generator, allocator: std.mem.Allocator, ids: []const u32, logits: mlx.mlx_array, conf: []const f32, draft_ns: u64, verify_ns: u64) !DrafterStepResult {
            const xfm = self.xfm;
            const diag = self.drafter_diag;
            const rows = ids.len;
            const t1 = ids[0];
            const verdict = try sampleVerdict(self, logits, ids, null);
            const accepted = verdict.accepted;
            try A.finish(xfm, &self.ctx, accepted);
            const n_commit: usize = 1 + accepted;
            const tokens = try allocator.dupe(u32, ids[0..n_commit]);
            errdefer allocator.free(tokens);
            try self.generated_ids.appendSlice(allocator, tokens);
            self.advanceStep(@intCast(n_commit));
            self.next_token_id = verdict.pending;
            self.drafter_pending = false;
            const st = &self.drafter_stats;
            st.rounds += 1;
            st.plain += @intFromBool(rows == 1);
            st.rows_hist[rows] += 1;
            st.drafted += rows - 1;
            st.accepted += accepted;
            st.draft_ns += draft_ns;
            st.verify_ns += verify_ns;
            if (diag == .off or diag == .log) {
                const ms = struct {
                    fn of(ns: u64) f32 {
                        return @floatCast(@as(f64, @floatFromInt(ns)) / 1e6);
                    }
                }.of;
                const draft_ms = ms(draft_ns);
                const verify_ms = ms(verify_ns);
                self.drafter_copy_backoff.observeStage(n_commit, draft_ms + verify_ms);
                self.drafter_policy.observeRound(conf, @intCast(rows), accepted, draft_ms, verify_ms);
            }
            if (diag != .off) log.info("  [drafter-round] grouped rows={d} accepted={d} draft_ms={d:.1} verify_ms={d:.1} t1={d} conf={any}\n", .{
                rows,
                accepted,
                @as(f64, @floatFromInt(draft_ns)) / 1e6,
                @as(f64, @floatFromInt(verify_ns)) / 1e6,
                t1,
                conf,
            });
            if (self.completion_tokens >= self.max_tokens) {
                self.done = true;
                self.finish_reason = "length";
            }
            return .{ .tokens = tokens, .accepted_tokens = accepted };
        }

        pub const ConstrainedEntry = union(enum) { proceed, stop, served: DrafterStepResult };        pub const Pending = union(enum) { commit, stop, forced: u32 };

        pub fn rowView(self: *const Generator, c: *Constraint) A.row_grammar.View {
            return .{ .grammar = c.grammar, .token_bytes = c.token_bytes, .proto = c.proto, .pstate = &c.pstate, .eos_ids = self.eos_token_ids };
        }

        pub fn drafterNoteSerial(self: *Generator, why: []const u8) void {
            if (self.drafter_serial_logged) return;
            self.drafter_serial_logged = true;
            log.info("  [spec] drafter: this request decodes serially while {s}\n", .{why});
        }

        /// The start of a constrained round: serial ticks where the constraint cannot be verified in rows, else the
        /// pending token (sampled here from the serial state's logits on the first round) judged as serial decode would.
        pub fn drafterConstrainedEntry(self: *Generator, allocator: std.mem.Allocator, c: *Constraint) !ConstrainedEntry {
            if (self.checkBudgetStop()) return .stop;
            const phase = c.pstate.phase;
            const vocab: usize = A.vocab(self.xfm);
            const why: ?[]const u8 = if (c.pstate.recovering)
                "a forced boundary drains"
            else if (phase != .reasoning and phase != .json_body)
                "the reasoning protocol is at a channel header"
            else if (c.token_bytes.bytes.len > vocab)
                "the token table is wider than the vocabulary"
            else
                null;
            if (why) |w| {
                drafterNoteSerial(self, w);
                self.drafter_stats.serial += 1;
                const token = (try self.next(allocator)) orelse return .stop;
                return .{ .served = .{ .tokens = try allocator.dupe(u32, &.{token}), .accepted_tokens = 0 } };
            }
            if (self.pending != null) try drafterSamplePending(self, c);
            return switch (try drafterJudgePending(self, allocator, c)) {
                .commit => .proceed,
                .stop => .stop,
                .forced => |t| .{ .served = .{ .tokens = try allocator.dupe(u32, &.{t}), .accepted_tokens = 0 } },
            };
        }

        /// The serial state's pending logits become a pending token, sampled under the mask of the committed state.
        pub fn drafterSamplePending(self: *Generator, c: *Constraint) !void {
            const logits = try self.takeLogits();
            defer _ = mlx.mlx_array_free(logits);
            const kind = try A.row_grammar.maskFor(rowView(self, c), c.mask_buf);
            const lazy = try self.sampleMasked(logits, if (kind == .masked) c.mask_buf else null, .committed);
            defer _ = mlx.mlx_array_free(lazy);
            try mlx.check(mlx.mlx_array_eval(lazy));
            var val: i32 = 0;
            try mlx.check(mlx.mlx_array_item_int32(&val, lazy));
            self.next_token_id = @intCast(val);
            self.drafter_pending = true;
        }

        /// What serial decode does with the token it just sampled before committing it: an EOS stops (inside the
        /// reasoning it forces the boundary instead), and so do three pad tokens in a row.
        pub fn drafterJudgePending(self: *Generator, allocator: std.mem.Allocator, c: *Constraint) !Pending {
            const t = self.next_token_id;
            const eos = isEosId(t, self.eos_token_ids);
            const pad_run = t == 0 and self.consecutive_pad + 1 >= 3;
            if (c.pstate.phase == .reasoning) {
                const proto = c.proto.?;
                const closes = eos and proto.closer_atomic != null and t == proto.closer_atomic.?;
                if ((eos and !closes) or pad_run) {
                    self.drafter_pending = false;
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
                self.drafter_pending = false;
                self.done = true;
                self.finish_reason = "stop";
                self.stop_token = t;
                return .stop;
            }
            self.consecutive_pad = if (t == 0) self.consecutive_pad + 1 else 0;
            return .commit;
        }

        /// Commits `token`, the generated token at `index`, to the constraint as serial decode does after sampling it.
        pub fn drafterObserve(self: *Generator, c: *Constraint, index: usize, token: u32) !void {
            const adv = try A.row_grammar.advance(rowView(self, c), token);
            if (!adv.crossed) return;
            const bytes = if (token < c.token_bytes.bytes.len) c.token_bytes.bytes[token] else null;
            c.pending_span = if (bytes == null)
                .{ .token_index = @intCast(index + 1), .byte_offset = 0 }
            else
                .{ .token_index = @intCast(index), .byte_offset = rp_mod.payloadByteOffset(bytes, adv.suffix) };
            self.loop_guard_start = index;
            log.info("[grammar] reasoning boundary reached; enforcing final-answer schema\n", .{});
        }

        /// Serial decode taking over from a drafter round: emit the pending token, then forward it for the next sample.
        pub fn drafterEmitPending(self: *Generator, allocator: std.mem.Allocator) !?u32 {
            const c = self.sampling.constraint.?;
            if (self.checkBudgetStop()) return null;
            switch (try drafterJudgePending(self, allocator, c)) {
                .stop => return null,
                .forced => |t| return t,
                .commit => {},
            }
            self.drafter_pending = false;
            const token = self.next_token_id;
            try drafterObserve(self, c, self.generated_ids.items.len, token);
            self.advanceStep(1);
            try self.generated_ids.append(allocator, token);
            if (self.step >= self.max_tokens) {
                self.done = true;
                self.finish_reason = "length";
                return token;
            }
            const tok_i32: i32 = @intCast(token);
            const tok_input = try mlx.newData(&tok_i32, &[_]c_int{ 1, 1 }, .int32);
            defer _ = mlx.mlx_array_free(tok_input);
            self.pending = .{ .logits = try self.xfm.forwardWith(&self.ctx, tok_input) };
            return token;
        }

        pub const Verdict = struct { accepted: u32, pending: u32 };

        pub const Streams = struct { ops: mlx.mlx_stream, greedy: mlx.mlx_stream, draws: mlx.mlx_stream };

        /// What a verify block's verdict reads: its rows' logits `[1, rows, vocab]` and ids (`[t1, drafts...]`), the
        /// request's sampling, the think penalty's shift and the penalties as the generator decided them for the rows,
        /// and each row's grammar mask (`rows * vocab` bools), when given.
        pub const VerdictIn = struct {
            logits: mlx.mlx_array,
            ids: []const u32,
            sampling: sampler.SamplingParams,
            think: ?sampler.Shift = null,
            /// The committed ids the penalties count, when they apply.
            committed: ?[]const u32 = null,
            masks: ?[]const bool = null,
            /// Where the shifts run, the argmax's stream and the draws' (`A.verdictStream` of the trunk's output stream
            /// and of its own).
            streams: Streams,

            pub fn mix(in: VerdictIn) sampler.Mix {
                var m: sampler.Mix = .{ .entry = .verdict, .rows = @intCast(in.ids.len), .suppress = in.sampling.suppress_mask != null, .greedy = sampler.isGreedyTemperature(in.sampling.temperature), .grammar = in.masks != null };
                if (in.think) |sh| m.think = sampler.thinkOf(sh.mask, sh.state, true);
                if (in.committed != null) m.penalties(sampler.penaltiesOf(in.sampling));
                return m;
            }
        };

        /// The verdict on `ids` read from `logits`: on a load that plans its samples, one execution of the root its mix
        /// names (D24). A sampled verdict advances the request's draws past the rows it drew.
        fn sampleVerdict(self: *Generator, logits: mlx.mlx_array, ids: []const u32, masks: ?[]const bool) !Verdict {
            const xfm = self.xfm;
            const streams: Streams = .{ .ops = xfm.outStream(), .greedy = A.verdictStream(xfm, xfm.outStream()), .draws = A.verdictStream(xfm, xfm.s) };
            const in: VerdictIn = .{ .logits = logits, .ids = ids, .sampling = self.sampling, .think = self.rowsShift(), .committed = if (sampler.penaltyActive(self.sampling)) self.generated_ids.items else null, .masks = masks, .streams = streams };
            const run = try arch_cooler.beginSample(self.xfm, in.mix());
            const v = verdictOf(in) catch |e| {
                try run.fail();
                return e;
            };
            try run.end();
            if (!sampler.isGreedyTemperature(self.sampling.temperature)) self.sampling.draw +%= v.accepted + 1;
            return v;
        }

        /// The longest prefix of `in.ids`' drafts the rows agree with, and the token after it: the rows shifted for the
        /// think penalty and the penalties as the serial ticks would sample them, masked row by row, then each row's
        /// argmax (greedy) or its own draw. Row j draws with the key of the generated position it predicts, so the tokens
        /// are the same whatever the round mix; a point-mass draft survives a draw of the row's density with probability
        /// p(draft): the exact rule.
        pub fn verdictOf(in: VerdictIn) !Verdict {
            const s = in.streams.ops;
            const rows = in.ids.len;
            const drafts = in.ids[1..];
            var cur = in.logits;
            var held: [3]mlx.mlx_array = undefined;
            var n_held: usize = 0;
            defer for (held[0..n_held]) |h| {
                _ = mlx.mlx_array_free(h);
            };
            if (in.think != null or in.committed != null) {
                var row_ids: [max_drafts + 1]i32 = undefined;
                if (rows > row_ids.len) return error.SpecDecodeUnsupported;
                for (in.ids, row_ids[0..rows]) |id, *o| o.* = @intCast(id);
                const row_ids_arr = try mlx.newData(&row_ids, &[_]c_int{ 1, @intCast(rows) }, .int32);
                defer _ = mlx.mlx_array_free(row_ids_arr);
                if (in.think) |sh| {
                    held[n_held] = mlx.mlx_array_new();
                    n_held += 1;
                    try think_penalty.shiftScoped(&held[n_held - 1], cur, sh.mask, sh.state, row_ids_arr, s);
                    cur = held[n_held - 1];
                }
                if (in.committed) |committed| {
                    const shape = mlx.getShape(cur);
                    const counts = try seen_penalty.buildCounts(std.heap.page_allocator, committed, @intCast(shape[shape.len - 1]));
                    defer _ = mlx.mlx_array_free(counts);
                    held[n_held] = mlx.mlx_array_new();
                    n_held += 1;
                    try seen_penalty.shiftRows(&held[n_held - 1], cur, row_ids_arr, counts, sampler.penaltiesOf(in.sampling), s);
                    cur = held[n_held - 1];
                }
            }
            if (in.masks) |masks| {
                const vocab: usize = @intCast(mlx.getShape(cur)[2]);
                held[n_held] = mlx.mlx_array_new();
                n_held += 1;
                try sampler.applyRowMasks(&held[n_held - 1], cur, masks, rows, vocab, s);
                cur = held[n_held - 1];
            }
            const vocab: u32 = @intCast(mlx.getShape(cur)[2]);
            var accepted: u32 = 0;
            if (sampler.isGreedyTemperature(in.sampling.temperature)) {
                var verified = try verifyArgmax(cur, in.sampling.suppress_mask, in.streams.greedy);
                defer verified.deinit();
                try mlx.check(mlx.mlx_array_eval(verified.lazy()));
                const best = try verified.result(rows, vocab);
                while (accepted + 1 < rows and @as(u32, @intCast(best[accepted])) == drafts[accepted]) accepted += 1;
                return .{ .accepted = accepted, .pending = @intCast(best[accepted]) };
            }
            const flat = try A.sampledDraws(in.sampling, cur, rows, in.streams.draws);
            defer _ = mlx.mlx_array_free(flat);
            try mlx.check(mlx.mlx_array_eval(flat));
            const drawn = try mlx.hostResult(i32, flat, vocab);
            while (accepted < drafts.len and @as(u32, @intCast(drawn[accepted])) == drafts[accepted]) accepted += 1;
            return .{ .accepted = accepted, .pending = @intCast(drawn[accepted]) };
        }

        pub const DrafterResume = drafter_rows.Resume;

        /// A forced commit left the shim-seeded state; drain it to the drafter round's entry (the closer published, its
        /// successor sampled and not in the cache). The stage windows already hold the forced rows, since every
        /// forward commits its main rows. The thought is closed, so its restore-point cut is disarmed.
        pub fn drafterResumeAfterForced(self: *Generator, allocator: std.mem.Allocator) !DrafterResume {
            if (!self.drafter_on or self.spec_disabled_runtime or self.sampling.constraint != null) return .refused;
            switch (try self.drainPipelineForSpec(allocator)) {
                .drained => |closer| {
                    self.drafter_round_ends_at = null;
                    log.info("  [drafter] resumed after the forced thought close at {d} generated tokens\n", .{self.generated_ids.items.len});
                    return .{ .resumed = closer };
                },
                .stopped => return .stopped,
                .already_clean => return .refused,
            }
        }
    };
}
