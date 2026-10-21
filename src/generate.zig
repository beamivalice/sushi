const std = @import("std");
const mlx = @import("mlx.zig");
const transformer_mod = @import("transformer.zig");
const qwen4_forward = @import("qwen4_forward.zig");
const tokenizer_mod = @import("tokenizer.zig");
const model_mod = @import("model.zig");
const log = @import("log.zig");
const json_grammar = @import("json_grammar.zig");
const json_schema = @import("json_schema.zig");
const token_mask = @import("token_mask.zig");
const rp_mod = @import("reasoning_protocol.zig");
const io_util = @import("io_util.zig");
const mtp_lookup = @import("mtp_lookup.zig");
const mtp_mod = @import("mtp.zig");
const mimo_mtp = @import("mimo_mtp.zig");
const mtp_acceptance = @import("mtp_acceptance.zig");
const model_settings = @import("model_settings.zig");
const round_cost = @import("round_cost.zig");
const group_cost = @import("mtp_group_cost.zig");
const group_planner = @import("mtp_group_planner.zig");
const depth_bounds = @import("mtp_depth_bounds.zig");
const chip_mod = @import("chip.zig");
const scheduler_mod = @import("scheduler.zig");
const think_penalty = @import("think_penalty.zig");
const seen_penalty = @import("seen_penalty.zig");
const logit_bias = @import("logit_bias.zig");

const Transformer = transformer_mod.Transformer;
const Tokenizer = tokenizer_mod.Tokenizer;
const ForwardCtx = transformer_mod.ForwardCtx;
const SSMCacheEntrySnapshot = transformer_mod.SSMCacheEntrySnapshot;
const ssmSnapshot = transformer_mod.ssmSnapshot;
const ssmSnapshotDeinit = transformer_mod.ssmSnapshotDeinit;
const ssmRestore = transformer_mod.ssmRestore;
const SSMCheckpoint = transformer_mod.SSMCheckpoint;
const captureSsmCheckpoint = transformer_mod.captureSsmCheckpoint;
const glm5_forward = @import("glm5_forward.zig");
const glm5_prefix = @import("glm5_prefix.zig");
const dflash_mod = @import("dflash.zig");
const dflash_policy = @import("dflash_policy.zig");
const DflashModel = dflash_mod.DflashModel;
const KVCache = transformer_mod.KVCache;
pub const ThinkPenalty = think_penalty.ThinkPenalty;

/// Module-level overrides for prefill behavior. Defaults match the original
/// hardcoded values; main.zig may overwrite these from CLI flags before
/// `serve()` runs. Per-request reads happen on the same thread that did the
/// CLI parse, so no atomicity needed.
pub var prefill_chunk_override: usize = 8192;

/// Set by `--prefill-chunk` in main.zig. An operator-chosen width outranks the
/// per-model chunk the memory sizer pins (`ModelConfig.pinned_prefill_chunk`);
/// without this flag the default 8192 is indistinguishable from a request.
/// Same set-once-at-CLI-parse contract as `prefill_chunk_override`.
pub var prefill_chunk_explicit: bool = false;

/// Set once by the serve CLI before request construction. Direct callers may
/// override it through InitOptions; exact remains the library default.
pub var mtp_acceptance_default: mtp_acceptance.Mode = .exact;
/// `--mtp-typical` / `--mtp-tokenv3` chose the default above.
pub var mtp_acceptance_explicit: bool = false;

/// THIS model's acceptance mode: the launch flag > its `mtp_acceptance` > exact.
pub fn mtpAcceptanceFor(setting: ?mtp_acceptance.Mode) model_settings.Pick(mtp_acceptance.Mode) {
    return model_settings.pick(mtp_acceptance.Mode, model_settings.launchFlag(mtp_acceptance.Mode, mtp_acceptance_default, mtp_acceptance_explicit), setting, mtp_acceptance_default);
}

/// `--mtp-greedy-tail` was given.
pub var mtp_greedy_tail_explicit: bool = false;

/// `--no-mtp-lookup`: no prompt-lookup drafts in MTP rounds or GLM DFlash2 chains .
pub var mtp_lookup_disabled: bool = false;

/// THIS model's greedy tail: the launch flag > its `mtp_greedy_tail` > off.
pub fn mtpGreedyTailFor(setting: ?bool) model_settings.Pick(bool) {
    return model_settings.pick(bool, model_settings.launchFlag(bool, true, mtp_greedy_tail_explicit), setting, false);
}

pub fn tickUbenchN(raw: ?[*:0]const u8) u32 {
    const v = raw orelse return 0;
    if (v[0] == '0') return 0;
    return std.fmt.parseInt(u32, std.mem.sliceTo(v, 0), 10) catch 0;
}

var tick_ubench_left: ?u32 = null;

pub fn tickUbenchArmed() bool {
    if (tick_ubench_left == null) {
        tick_ubench_left = tickUbenchN(std.c.getenv("SUSHI_DECODE_TICK_UBENCH"));
    }
    return tick_ubench_left.? > 0;
}

fn tickUbenchConsume() void {
    if (tick_ubench_left) |*n| {
        if (n.* > 0) n.* -= 1;
    }
}

const TickKind = enum(u2) { fwd, sample, resolve, eval };

const TickProf = struct {
    on: bool,
    wall: io_util.Stopwatch,
    ns: [4]u64 = @splat(0),

    fn begin() TickProf {
        if (!tickUbenchArmed()) return .{ .on = false, .wall = undefined };
        return .{ .on = true, .wall = io_util.Stopwatch.init(std.Io.Threaded.global_single_threaded.io()) };
    }

    fn mark(self: *const TickProf) u64 {
        if (!self.on) return 0;
        return self.wall.read();
    }

    fn add(self: *TickProf, kind: TickKind, t0: u64) void {
        if (!self.on) return;
        self.ns[@backingInt(kind)] += self.wall.read() - t0;
    }

    fn finish(self: *TickProf) void {
        if (!self.on) return;
        const wall = self.wall.read();
        const fwd = self.ns[@backingInt(TickKind.fwd)];
        const sample = self.ns[@backingInt(TickKind.sample)];
        const resolve = self.ns[@backingInt(TickKind.resolve)];
        const eval = self.ns[@backingInt(TickKind.eval)];
        const accounted = fwd + sample + resolve + eval;
        const rest = if (wall > accounted) wall - accounted else 0;
        const ms = struct {
            fn f(n: u64) f64 {
                return @as(f64, @floatFromInt(n)) / 1e6;
            }
        }.f;
        log.info("[tick-ubench] wall={d:.3} ms fwd={d:.3} sample={d:.3} resolve={d:.3} eval={d:.3} rest={d:.3}\n", .{
            ms(wall),
            ms(fwd),
            ms(sample),
            ms(resolve),
            ms(eval),
            ms(rest),
        });
        tickUbenchConsume();
    }
};

/// Per-prefill state for the adaptive width; lives on `runPrefill`'s stack. The policy is
/// `server.adaptivePrefillWidth`.
/// The one place a chosen width becomes the running width.
pub fn commitAdaptiveWidth(cur: *usize, st: *AdaptiveWidthState, w: u32) void {
    cur.* = w;
    st.transitions +|= 1;
    st.width_min = @min(st.width_min, w);
    st.width_max = @max(st.width_max, w);
}

pub const AdaptiveWidthState = struct {
    supporting: u8 = 0,
    /// One-way ratchet: a prefill that has stepped down never widens again.
    ratcheted: bool = false,
    transitions: u32 = 0,
    width_min: u32 = 0,
    width_max: u32 = 0,
};

/// `--max-mtp-ctx N`: MTP stays off past N context tokens (0 = unlimited). Read by the HTTP
/// admission gate and by `nextMtp` per round (`mtp.mtpCtxWithinLimit`).
pub var max_mtp_ctx: u32 = 0;

/// Why a request left speculative decoding for the rest of its generation.
pub const SpecDisableReason = enum {
    none,
    acceptance,
    max_ctx,
    /// A thinking budget closed the thought through a plain forward; the spec
    /// state is not resynced, the answer decodes regular.
    think_bound,
    /// The measured round cost more per token than a measured serial token (`MtpAdaptive`).
    adaptive,
};

/// One layer's materialized-score budget for unfused-SDPA prefill (see
/// boundedPrefillChunk). 4 GiB keeps the full 8K chunk for every context up
/// to ~16K on 16-head models (further on fewer heads) and degrades gradually
/// to the 512 floor as heads × context grows toward 262K.
pub const PREFILL_SCORES_BUDGET_BYTES: u64 = 4 << 30;
/// Lower bound for the auto-capped prefill chunk; also its rounding grain
/// (repeating chunk sizes let the MLX allocator cache reuse score buffers).
pub const PREFILL_CHUNK_FLOOR: usize = 512;

/// MLX's fused SDPA kernels cover head_dim <= 128. Every Gemma-4 and
/// Qwen3.5/3.6 checkpoint ships head_dim=256, which falls back to the
/// composed path that MATERIALIZES a [heads, chunk, total_kv] bf16 score
/// tensor per layer — at an 8K chunk × 255K ctx × 16 heads that is ~67 GB
/// and an uncatchable Metal command-buffer OOM. Cap the chunk so ONE layer's
/// score tensor stays within PREFILL_SCORES_BUDGET_BYTES at this prompt's
/// FINAL KV length (the last chunk attends to everything). Fused head dims
/// and short contexts return `base_chunk` untouched, so typical traffic
/// keeps full prefill throughput; the cap only bites when heads × total_ctx
/// actually outgrows the budget. Never raises a caller-lowered base.
///
/// DELIBERATELY ignores the sushi_attn_p256 fused kernel (unlike
/// prefillEvalCadence / prefillMemoryNeeded, which drop their score term via
/// transformer.prefillHeadDimFused): the fused kernel removes the SCORE
/// transient, but a big chunk still scales the OTHER per-chunk transients
/// (MoE gather buffers, per-chunk KV concat) — measured LIVE on
/// gemma-4-26B-A4B at a 99K prompt: fused @ chunk 8192 = 736 tok/s / 61.2 GB
/// peak vs fused @ chunk 1024 = 712 tok/s / 39.5 GB. +3% speed is not worth
/// +22 GB peak (a 64 GB Mac dies), so the cap stays keyed on raw head_dim.
///
/// `sliding_band_arch` (config.has_sliding_window) picks the policy family:
/// archs WITHOUT sliding-band layers (qwen3_5/3_6: GDN + full attention)
/// additionally cap the auto chunk at 2048 — composed-causal prefill
/// measured strictly faster and ~9 GB lighter there (see the inline
/// comment). Gemma keeps the formula-only policy for its fused band layers.
/// `score_head_dim` is the width the SCORE tensor is contracted at
/// (`ModelConfig.prefillScoreHeadDim`) — not necessarily `head_dim`. An MLA arch
/// scores at 192 while storing 128-wide values, and reading `head_dim` there
/// silently exempted the one arch in the tree whose composed path materializes
/// the biggest score tensors. The two hd-256 policy branches below stay keyed
/// on 256 exactly: both were measured on hd-256 checkpoints and neither
/// generalizes (the fused kernel is hd-256-only; the 2048 composed cap was
/// tuned against a 27B's own prefill ladder).
pub fn boundedPrefillChunk(base_chunk: usize, score_head_dim: u32, n_heads: u32, total_ctx: usize, sliding_band_arch: bool, is_moe: bool, long_ctx_gated: bool) usize {
    const head_dim = score_head_dim;
    if (head_dim <= 128 or n_heads == 0 or total_ctx == 0) return base_chunk;
    // qk 192 (mimo_v2 global layers, MLA): `sushi_attn_pd` builds no score
    // tensor, so the budget formula below — which would pin the floor 512 at
    // any context past ~256k — measures nothing. What still scales with the
    // chunk is the MoE gather and the ringed-SWA staging, and those are what
    // `server.resolvePrefillChunk` prices against this machine; the 4096 cap
    // is the measured gemma-26B MoE lesson carried over as a ceiling only.
    // 2048 is the default (a 64k prompt prefills within noise of 4096); an
    // explicit `--prefill-chunk` may raise it to that ceiling.
    if (head_dim == 192 and transformer_mod.prefillHeadDimFused(head_dim)) {
        return @min(base_chunk, @as(usize, if (prefill_chunk_explicit) 4096 else 2048));
    }
    // Non-sliding hd-256 archs under FUSED causal (the default since the
    // budgeted-dispatch flip): no score tensor exists, so the scores-budget
    // formula below is moot — and its old shrink (1024 at 64K on 24 heads)
    // starved the dequant+GEMM qmm route, which needs M >= 2048 to engage
    // (the 64K rung was the ladder's weakest for exactly this reason).
    // MoE keeps the 4096 cap: expert-gather transients scale with the chunk
    // (the gemma-26B@99K lesson: +3% speed for +22 GB peak is a bad trade).
    // DENSE hybrids have no gather transients and a full-size chunk halves
    // the per-chunk dequant sweeps: chunk 8192 measured +1.4% over 4096 at
    // the 8K rung on Qwen3.6-27B dense (M4 Max, 2026-07-30), flat at 32K.
    // Never raises a caller-lowered base.
    if (head_dim == 256 and !sliding_band_arch and transformer_mod.fused256CausalMode() == .all) {
        // A long-context-gated MoE prices the 8192 rung per request in `chooseRequestPrefillChunk`
        // (+8% prefill for +3 GB peak here); every other MoE keeps the measured 4096 cap.
        return @min(base_chunk, if (is_moe and !long_ctx_gated) @as(usize, 4096) else @as(usize, 8192));
    }
    // Composed-causal fallback (the parity oracle, `fused256_override = false`): SMALL chunks
    // measured strictly faster AND lighter on the 27B (2026-07-12 ladder,
    // M4 Max): 8K 225 -> 235.8 tok/s and peak 28.9 -> 19.8 GB at chunk
    // 2048; 32K 205.4 -> 209.3. Chunk boundaries ARE block-level causal
    // skipping for composed attention. Sliding-band archs (gemma) keep big
    // chunks — their local layers run the fused band kernel, which
    // block-skips in-kernel and wants the fewest KV re-walks (26B@99K:
    // 712 tok/s at the formula chunk).
    const causal_cap: usize = if (sliding_band_arch or head_dim != 256) base_chunk else @min(base_chunk, 2048);
    const per_row: u64 = @as(u64, n_heads) * @as(u64, total_ctx) * 2;
    const max_chunk: u64 = PREFILL_SCORES_BUDGET_BYTES / per_row;
    if (max_chunk >= causal_cap) return causal_cap;
    const floored = @max(
        @as(u64, PREFILL_CHUNK_FLOOR),
        max_chunk - (max_chunk % PREFILL_CHUNK_FLOOR),
    );
    return @intCast(@min(floored, @as(u64, causal_cap)));
}

/// The prefill chunk `initWithOptions` will actually use for a request:
/// --prefill-chunk / default, capped by boundedPrefillChunk. Exported so server.zig's admission guard
/// (checkAttentionMemory) models the SAME chunk the prefill will run with —
/// the guard and the real prefill must not drift.
pub fn effectivePrefillChunk(head_dim: u32, n_heads: u32, total_ctx: usize, sliding_band_arch: bool, is_moe: bool, long_ctx_gated: bool, pinned_chunk: usize) usize {
    // `pinned_chunk` is the machine-sized cap frozen at load
    // (`ModelConfig.pinned_prefill_chunk`, from `server.resolvePrefillChunk`):
    // the widest chunk whose one-off transient reserve still leaves this box a
    // real KV budget. It NARROWS, never raises, and an explicit
    // `--prefill-chunk` outranks it — the operator asked for a width.
    const base: usize = if (prefill_chunk_explicit or pinned_chunk == 0)
        prefill_chunk_override
    else
        @min(prefill_chunk_override, pinned_chunk);
    return boundedPrefillChunk(base, head_dim, n_heads, total_ctx, sliding_band_arch, is_moe, long_ctx_gated);
}

/// The width a request forwards at, and the width its admission bills: `effectivePrefillChunk`,
/// never wider than the width handed in (0 = none). An explicit width outranks only the load-time pin.
pub fn requestPrefillChunk(head_dim: u32, n_heads: u32, total_ctx: usize, sliding_band_arch: bool, is_moe: bool, long_ctx_gated: bool, pinned_prefill_chunk: usize) usize {
    const width = effectivePrefillChunk(head_dim, n_heads, total_ctx, sliding_band_arch, is_moe, long_ctx_gated, pinned_prefill_chunk);
    return if (pinned_prefill_chunk == 0) width else @min(width, pinned_prefill_chunk);
}

/// Read an unsigned integer from an environment variable, falling back to
/// `default` when unset, empty, or unparseable. Uses libc getenv to stay
/// allocator-free at call sites.
fn readEnvUsize(name: [:0]const u8, default: usize) usize {
    const raw = std.c.getenv(name.ptr);
    if (raw == null) return default;
    const slice = std.mem.sliceTo(raw.?, 0);
    if (slice.len == 0) return default;
    return std.fmt.parseInt(usize, slice, 10) catch default;
}

/// Truthy if the env var is exactly "1". Anything else (unset, "0", "true",
/// "yes") is false — keep matching surface tight to avoid surprises.
fn readEnvBool(name: [:0]const u8) bool {
    const raw = std.c.getenv(name.ptr);
    if (raw == null) return false;
    const slice = std.mem.sliceTo(raw.?, 0);
    return std.mem.eql(u8, slice, "1");
}

fn freeArray(a: mlx.mlx_array) void {
    _ = mlx.mlx_array_free(a);
}

/// Grammar-constrained sampling state. The caller owns `grammar`, `token_bytes`,
/// and `mask_buf`; the generator only reads them. `mask_buf.len` must equal
/// `token_bytes.bytes.len` (the tokenizer's vocab size).
/// Which MTP head a generator drives.
///
/// The controller around it is head-agnostic — the EV planner, the acceptance
/// gate, `mtpBatchedAcceptGraph`, the pre-draft and the horizon valve all work
/// in tokens and probabilities — so the split is exactly these five operations
/// and nothing else.
pub const MtpHeadRef = union(enum) {
    qwen: *mtp_mod.MtpModel,
    /// qwen4_exp: the head and its history live on the Transformer
    /// (`qwen4_mtp`, module-owned ⇒ single-flight); row r of the history is
    /// (pre-mixer stream at position r, token r+1), query position r+1.
    qwen4: *Transformer,
    /// mimo_v2: the checkpoint's trained heads; per-request rows live in `mimo_mtp.State`.
    mimo: *mimo_mtp.Head,

    /// Is this head's decode state module-owned (one per model, `Qwen4Mtp.cache`) rather
    /// than per-request? Both the scheduler and the sticky-serial arm ask this.
    pub fn moduleOwned(self: MtpHeadRef) bool {
        return switch (self) {
            .qwen => false,
            // Per-request state swaps onto the module (`Qwen4MtpState`); nothing is shared.
            .qwen4, .mimo => false,
        };
    }

    pub fn makeCache(self: MtpHeadRef, allocator: std.mem.Allocator) !MtpCacheRef {
        return switch (self) {
            .qwen => |h| .{ .qwen = try h.makeCache(allocator) },
            .qwen4 => |t| blk: {
                const st = try allocator.create(transformer_mod.Transformer.Qwen4MtpState);
                errdefer allocator.destroy(st);
                st.* = try t.qwen4MtpStateNew();
                t.qwen4MtpActivate(st);
                break :blk .{ .qwen4 = .{ .t = t, .st = st, .allocator = allocator } };
            },
            .mimo => |h| .{ .mimo = .{ .st = try h.newState(allocator) } },
        };
    }

    /// The head projects only what the caller consumes: this step wants the
    /// LAST row (`want_logits`) or no logits at all, never the S-row block —
    /// the merged history step after a partial accept is `1 + accepted` rows
    /// wide and a full-block mixer + 248320-wide lm_head over it is thrown
    /// away but for one row.
    fn qwen4Step(t: *Transformer, id_arr: mlx.mlx_array, hidden: mlx.mlx_array, rope_offset: c_int, want: mtp_mod.StepWant, mrope_ctx: ?mtp_mod.MropeContext) !mtp_mod.StepOut {
        // `.mixed` is NOT `.none`: a rerank draft still needs the mixer output,
        // which is the vector the lm_head consumes. `hidden_next` is the
        // PRE-mixer stream on this arm and would be a silent shape error there.
        const project: Transformer.Qwen4MtpProject = switch (want) {
            .logits => .last_row,
            .mixed => .mixed_last_row,
            .none => .none,
        };
        const out = try t.qwen4MtpForward(hidden, id_arr, rope_offset + 1, mrope_ctx, project);
        return .{ .logits = out.logits, .hidden_next = out.stream, .rerank_x = out.mixed };
    }

    /// One head forward over L positions. `.logits` returns the LAST row only,
    /// on both arms. `.mixed` asks for the lm_head's INPUT vector instead: on
    /// the sidecar that is `hidden_next` and nothing extra is produced, on
    /// qwen4_exp it is the mixer output, returned in `rerank_x`. `host_ids` are
    /// `id_arr`'s values when the caller holds them (a round's committed rows);
    /// the MiMo arm records them.
    pub fn forward(
        self: MtpHeadRef,
        target: *Transformer,
        cache: *MtpCacheRef,
        id_arr: mlx.mlx_array,
        hidden: mlx.mlx_array,
        rope_offset: c_int,
        want: mtp_mod.StepWant,
        mrope_ctx: ?mtp_mod.MropeContext,
        host_ids: ?[]const u32,
    ) !mtp_mod.StepOut {
        return switch (self) {
            .mimo => |h| mimoStep(h, target, &cache.mimo, id_arr, hidden, @intCast(rope_offset), want, host_ids),
            .qwen => |h| mtp_mod.forwardWithMrope(h, target, &cache.qwen, id_arr, hidden, rope_offset, want == .logits, mrope_ctx),
            .qwen4 => |t| blk: {
                cache.activate();
                break :blk qwen4Step(t, id_arr, hidden, rope_offset, want, mrope_ctx);
            },
        };
    }

    /// MiMo draft steps chain no hidden: step 0 carries the round's committed rows
    /// (`hidden` the target's), each later step only its draft. The step index
    /// travels in `hidden_next`, a scalar the next call recognises by its rank.
    fn mimoStep(h: *mimo_mtp.Head, target: *Transformer, ref: *MtpCacheRef.MimoRef, id_arr: mlx.mlx_array, hidden: mlx.mlx_array, rope_offset: usize, want: mtp_mod.StepWant, host_ids: ?[]const u32) !mtp_mod.StepOut {
        const chained = mlx.mlx_array_ndim(hidden) != 3;
        if (want == .none) {
            try h.appendHistory(target, ref.st, host_ids orelse return error.MimoMtpNeedsHostIds, hidden, rope_offset);
            return .{ .logits = .{ .ctx = null }, .hidden_next = mlx.mlx_array_new_int(0) };
        }
        const step_i: usize = if (chained) ref.next_step else 0;
        const last = if (chained)
            try h.draftStep(target, ref.st, step_i, id_arr, &.{}, null, rope_offset)
        else
            try h.draftStep(target, ref.st, 0, null, host_ids orelse return error.MimoMtpNeedsHostIds, hidden, rope_offset);
        ref.next_step = step_i + 1;
        return mimoStepOut(target, freeArray, last, step_i + 1, want);
    }

    /// Hands a draft step's last row over: through the trunk's lm_head (then freed)
    /// under `.logits`, else as `rerank_x`. Takes `last`; on error nothing leaks.
    fn mimoStepOut(target: anytype, comptime free: fn (mlx.mlx_array) void, last: mlx.mlx_array, step_next: usize, want: mtp_mod.StepWant) !mtp_mod.StepOut {
        errdefer free(last);
        const hidden_next = mlx.mlx_array_new_int(@intCast(step_next));
        errdefer free(hidden_next);
        if (want != .logits) return .{ .logits = .{ .ctx = null }, .hidden_next = hidden_next, .rerank_x = last };
        const logits = try target.lmHeadLogits(last);
        free(last);
        return .{ .logits = logits, .hidden_next = hidden_next };
    }

    /// Append committed history without projecting logits.
    pub fn appendHistory(
        self: MtpHeadRef,
        target: *Transformer,
        cache: *MtpCacheRef,
        token_ids: []const u32,
        hidden: mlx.mlx_array,
        rope_offset: c_int,
        mrope_ctx: ?mtp_mod.MropeContext,
        allocator: std.mem.Allocator,
    ) !void {
        switch (self) {
            .qwen => |h| try mtp_mod.appendHistoryWithMrope(h, target, &cache.qwen, token_ids, hidden, rope_offset, mrope_ctx),
            .mimo => |h| try h.appendHistory(target, cache.mimo.st, token_ids, hidden, @intCast(rope_offset)),
            .qwen4 => |t| {
                const ids_i32 = try allocator.alloc(i32, token_ids.len);
                defer allocator.free(ids_i32);
                for (token_ids, 0..) |tok, i| ids_i32[i] = @intCast(tok);
                const shape = [_]c_int{@intCast(token_ids.len)};
                const id_arr = mlx.mlx_array_new_data(ids_i32.ptr, &shape, 1, .int32);
                defer _ = mlx.mlx_array_free(id_arr);
                cache.activate();
                const out = try qwen4Step(t, id_arr, hidden, rope_offset, .none, mrope_ctx);
                _ = mlx.mlx_array_free(out.hidden_next);
            },
        }
    }

    /// Draft-rerank scheme (see mtp.zig): greedy drafts pick via a coarse
    /// 2-bit readout + trunk-head top-32 re-score instead of a full
    /// draft-head projection + argmax.
    pub fn canRerankDrafts(self: MtpHeadRef) bool {
        return switch (self) {
            .qwen => |h| h.canRerankDrafts(),
            // Built lazily on the first ask: `loadQwen4Mtp` runs while the
            // Transformer is still under construction and its lm_head — the
            // very weight the coarse head is a copy of — is not assigned yet.
            .qwen4 => |t| t.qwen4DraftRerankReady(),
            .mimo => |h| h.canRerankDrafts(),
        };
    }

    /// `x` is whatever the TARGET's lm_head consumes for this arm — the
    /// sidecar's post-norm hidden, or qwen4_exp's MIXER output. Never
    /// `hidden_next` on qwen4_exp: that is the pre-mixer `[B,S,hc*H]` stream.
    pub fn draftSelect(
        self: MtpHeadRef,
        target: *Transformer,
        x: mlx.mlx_array,
        suppress_mask: ?mlx.mlx_array,
    ) !mlx.mlx_array {
        return switch (self) {
            .qwen => |h| h.draftSelect(target, x, suppress_mask),
            .qwen4 => |t| t.qwen4DraftSelect(x, suppress_mask),
            .mimo => |h| h.draftSelect(target, x, suppress_mask),
        };
    }

    /// The exact re-scored top-32 shortlist a SAMPLED draft proposes from, or
    /// null when the coarse rerank head is unavailable. `x` is the same vector
    /// `draftSelect` takes.
    pub fn draftShortlist(
        self: MtpHeadRef,
        target: *Transformer,
        x: mlx.mlx_array,
        suppress_mask: ?mlx.mlx_array,
    ) !?mtp_mod.Shortlist {
        return switch (self) {
            .qwen => |h| h.draftShortlist(target, x, suppress_mask),
            .qwen4 => |t| t.qwen4DraftShortlist(x, suppress_mask),
            .mimo => |h| h.draftShortlist(target, x, suppress_mask),
        };
    }

    /// Every G17/NAX cost surface is calibrated against one exact runtime
    /// geometry — the dense Qwen3.6/3.8-27B sidecars on the `.qwen` arm, the
    /// qwen4_exp in-checkpoint head on its own arm — so every other head
    /// plans under `generic`: an off-profile head served by a calibrated
    /// surface would plan depths the measurement never covered.
    pub fn costProfile(self: MtpHeadRef, target: *const Transformer, kv: transformer_mod.KVQuantConfig) mtp_mod.MtpCostProfile {
        return switch (self) {
            .qwen => |h| h.m5NaxCostProfile(target),
            .qwen4 => |t| mtp_mod.qwen4G17CostProfileForKv(t, kv),
            .mimo => if (target.config.expert_layout == .exl3_k4) .mimo_exl3 else .generic,
        };
    }

    /// The head's last healthy acceptance surface, or null when it has never
    /// published one. BOTH arms store it on the head object itself — the
    /// sidecar `MtpModel` and the in-checkpoint `Qwen4Mtp` — so a seed dies
    /// with the head and can never cross models.
    pub fn evSeed(self: MtpHeadRef) ?struct { accept: [mtp_mod.MAX_DEPTH]f32, m_lo: u32 } {
        return switch (self) {
            .qwen => |h| if (h.ev_seed_accept) |a| .{ .accept = a, .m_lo = h.ev_seed_m_lo } else null,
            .mimo => |h| if (h.ev_seed_accept) |a| .{ .accept = a, .m_lo = h.ev_seed_m_lo } else null,
            .qwen4 => |t| blk: {
                if (t.qwen4_mtp) |*m| {
                    if (m.ev_seed_accept) |a| break :blk .{ .accept = a, .m_lo = m.ev_seed_m_lo };
                }
                break :blk null;
            },
        };
    }

    pub fn setEvSeed(self: MtpHeadRef, accept: [mtp_mod.MAX_DEPTH]f32, m_lo: u32) void {
        switch (self) {
            .qwen => |h| {
                h.ev_seed_accept = accept;
                h.ev_seed_m_lo = m_lo;
            },
            .mimo => |h| {
                h.ev_seed_accept = accept;
                h.ev_seed_m_lo = m_lo;
            },
            .qwen4 => |t| {
                // A model served with `--no-mtp` has no head to seed.
                if (t.qwen4_mtp) |*m| {
                    m.ev_seed_accept = accept;
                    m.ev_seed_m_lo = m_lo;
                }
            },
        }
    }
};

var mtp_trace_sync_env: ?bool = null;

/// `SUSHI_MTP_HEAD_PERSIST=0`: the qwen4_exp head's committed history is neither
/// snapshotted into a prefix-cache entry nor restored from one.
var mtp_head_persist_env: ?bool = null;
/// Only a literal "0" turns persistence off.
pub fn mtpHeadPersistFromEnv(raw: ?[]const u8) bool {
    const v = raw orelse return true;
    return !std.mem.eql(u8, v, "0");
}
pub fn mtpHeadPersistEnabled() bool {
    if (mtp_head_persist_env) |v| return v;
    const raw = std.c.getenv("SUSHI_MTP_HEAD_PERSIST");
    const on = mtpHeadPersistFromEnv(if (raw) |r| std.mem.sliceTo(r, 0) else null);
    mtp_head_persist_env = on;
    return on;
}

/// The head's committed-history cache.
pub const MtpCacheRef = union(enum) {
    qwen: KVCache,
    qwen4: Qwen4Ref,
    mimo: MimoRef,

    /// A MiMo request's head rows, and the draft step its next forward runs.
    pub const MimoRef = struct { st: *mimo_mtp.State, next_step: usize = 0 };

    /// The in-checkpoint head plus this request's own half of it.
    pub const Qwen4Ref = struct { t: *Transformer, st: *transformer_mod.Transformer.Qwen4MtpState, allocator: std.mem.Allocator };

    /// Install this request's state on the qwen4 head; no-op on the sidecar arm.
    pub fn activate(self: *const MtpCacheRef) void {
        switch (self.*) {
            .qwen, .mimo => {},
            .qwen4 => |r| r.t.qwen4MtpActivate(r.st),
        }
    }

    fn qwen4StateForBatch(self: *const MtpCacheRef) *Transformer.Qwen4MtpState {
        const ref = self.qwen4;
        ref.t.qwen4MtpRelease(ref.st);
        return ref.st;
    }

    pub fn step(self: *const MtpCacheRef) usize {
        return switch (self.*) {
            .qwen => |*c| c.step,
            .qwen4 => |r| blk: {
                r.t.qwen4MtpActivate(r.st);
                break :blk r.t.qwen4_mtp.?.seq_offset;
            },
            .mimo => |r| r.st.step(),
        };
    }

    /// The underlying KVCache — what the prefix cache's spec-snap machinery
    /// snapshots and restores. The qwen4 head's state is not KV-only, so a caller taking this
    /// pointer must also carry `head()`. Null when head persistence is off.
    pub fn kv(self: *MtpCacheRef) ?*KVCache {
        return switch (self.*) {
            .qwen => |*c| c,
            .qwen4 => |r| blk: {
                r.t.qwen4MtpActivate(r.st);
                break :blk if (mtpHeadPersistEnabled()) &r.t.qwen4_mtp.?.cache else null;
            },
            // Head rows are rebuilt from the prompt's last window; a snapshot would carry nothing more.
            .mimo => null,
        };
    }

    /// The Transformer owning the in-checkpoint head; null on the sidecar arm and whenever `kv()` is null.
    pub fn head(self: *MtpCacheRef) ?*Transformer {
        return switch (self.*) {
            .qwen, .mimo => null,
            .qwen4 => |r| blk: {
                r.t.qwen4MtpActivate(r.st);
                break :blk if (mtpHeadPersistEnabled()) r.t else null;
            },
        };
    }

    pub fn truncate(self: *MtpCacheRef, len: usize, s: mlx.mlx_stream) !void {
        switch (self.*) {
            .qwen => |*c| try c.truncate(len, s),
            .qwen4 => |r| {
                r.t.qwen4MtpActivate(r.st);
                try r.t.qwen4MtpTruncate(len);
            },
            .mimo => |*r| {
                try r.st.truncate(s, len);
                r.next_step = 0;
            },
        }
    }

    pub fn deinit(self: *MtpCacheRef) void {
        switch (self.*) {
            .qwen => |*c| c.deinit(),
            .qwen4 => |r| {
                r.t.qwen4MtpRelease(r.st);
                r.st.deinit();
                r.allocator.destroy(r.st);
            },
            .mimo => |r| {
                const allocator = r.st.allocator;
                r.st.deinit();
                allocator.destroy(r.st);
            },
        }
    }

    /// Append this cache's storage to a prefill eval batch, so the chunk's
    /// activation graph can be freed with the chunk instead of accumulating
    /// across the whole prompt.
    pub fn appendEvalArrays(self: *const MtpCacheRef, vec: mlx.mlx_vector_array) void {
        switch (self.*) {
            .qwen => |*c| c.appendEvalArrays(vec),
            .qwen4 => |r| {
                r.t.qwen4MtpActivate(r.st);
                r.t.qwen4_mtp.?.cache.appendEvalArrays(vec);
            },
            .mimo => |r| r.st.appendEvalArrays(vec),
        }
    }
};

/// An MTP committed-history cache restored from the prefix cache, plus the
/// absolute target position its index 0 represents.
pub const MtpRestored = struct { cache: MtpCacheRef, base: usize };

pub const Constraint = struct {
    grammar: *json_grammar.Grammar,
    token_bytes: *const token_mask.TokenBytes,
    mask_buf: []bool,
    /// Final-answer grammars normally apply at token 0 (thinking-off and all
    /// prompt-committed content channels). A reasoning protocol instead makes
    /// generation start in the model's channel machinery: either inside an
    /// already-open reasoning block (prompt-opened) or at the unresolved
    /// opener choice. The protocol state drives the phase switches below.
    proto: ?*const rp_mod.Protocol = null,
    pstate: rp_mod.State = .{},
    /// Where the constrained payload begins in the generated stream — the
    /// authoritative reasoning→answer boundary the response emitters route
    /// on. Set exactly once, at the transition; `null` until then.
    pending_span: ?rp_mod.ConstraintSpan = null,
};

/// A forced recovery must leave room for the whole remaining transition AND
/// at least one constrained answer token. At the ordinary completion cap the
/// caller keeps the existing length-stop behavior — no reserve is created.
pub fn forcedBoundaryCanContinue(completion_tokens: u32, max_tokens: u32, transition_tokens: usize) bool {
    return completion_tokens +| @as(u32, @intCast(transition_tokens)) + 1 <= max_tokens;
}

/// RAII bundle for grammar-constrained sampling. Owns the parsed schema,
/// grammar state machine, and per-step mask buffer. The embedded `Constraint`
/// holds pointers/slices into the surrounding struct, so this struct must NOT
/// be moved after `initFromValue`. Construct via `var sc: SchemaConstraint =
/// undefined; try sc.initFromValue(...);` and pass `&sc.constraint` to
/// `SamplingParams`.
pub const SchemaConstraint = struct {
    schema: json_schema.Schema,
    grammar: json_grammar.Grammar,
    mask_buf: []bool,
    constraint: Constraint,
    allocator: std.mem.Allocator,

    /// Initialize in-place from a JSON schema value. On failure, any partial
    /// allocations made during this call are freed and the struct is left
    /// undefined (do not call `deinit`).
    pub fn initFromValue(
        self: *SchemaConstraint,
        allocator: std.mem.Allocator,
        schema_value: std.json.Value,
        token_bytes: *const token_mask.TokenBytes,
    ) !void {
        self.allocator = allocator;
        self.schema = try json_schema.parse(allocator, schema_value);
        errdefer self.schema.deinit();

        self.grammar = try json_grammar.Grammar.init(allocator, &self.schema);
        errdefer self.grammar.deinit();

        self.mask_buf = try allocator.alloc(bool, token_bytes.bytes.len);
        errdefer allocator.free(self.mask_buf);

        self.constraint = .{
            .grammar = &self.grammar,
            .token_bytes = token_bytes,
            .mask_buf = self.mask_buf,
        };
    }

    pub fn deinit(self: *SchemaConstraint) void {
        self.allocator.free(self.mask_buf);
        self.grammar.deinit();
        self.schema.deinit();
    }

    /// Route this request through a resolved reasoning protocol. Must be
    /// called before generation starts; the protocol is a per-request value
    /// the caller keeps alive (the Constraint borrows it).
    pub fn deferWithProtocol(self: *SchemaConstraint, proto: *const rp_mod.Protocol) void {
        self.constraint.proto = proto;
        self.constraint.pstate = proto.startState();
        if (self.constraint.pstate.phase == .json_body) self.constraint.pending_span = .{ .token_index = 0, .byte_offset = 0 };
    }
};

test "decode tick ubench count is off unless a positive integer" {
    try std.testing.expectEqual(@as(u32, 0), tickUbenchN(null));
    try std.testing.expectEqual(@as(u32, 0), tickUbenchN("0"));
    try std.testing.expectEqual(@as(u32, 1), tickUbenchN("1"));
    try std.testing.expectEqual(@as(u32, 8), tickUbenchN("8"));
}

test "forced recovery leaves room for the whole transition and one answer token" {
    try std.testing.expect(forcedBoundaryCanContinue(0, 2, 1));
    try std.testing.expect(forcedBoundaryCanContinue(8, 10, 1));
    try std.testing.expect(!forcedBoundaryCanContinue(0, 1, 1));
    try std.testing.expect(!forcedBoundaryCanContinue(9, 10, 1));
    // A multi-token transition plans against the WHOLE sequence.
    try std.testing.expect(forcedBoundaryCanContinue(0, 4, 3));
    try std.testing.expect(!forcedBoundaryCanContinue(0, 3, 3));
    // An in-place activation needs no transition token.
    try std.testing.expect(forcedBoundaryCanContinue(0, 1, 0));
}

test "forced recovery is committed and counted exactly once" {
    const src = @embedFile("generate.zig");
    const start = std.mem.indexOf(u8, src, "pub fn forceConstraint" ++ "Transition(") orelse return error.CallSiteMoved;
    const tail = src[start..];
    const end = std.mem.indexOf(u8, tail, "\n    pub fn deinit(") orelse return error.CallSiteMoved;
    const body = tail[0..end];
    // One committed token per call, through the live forward path.
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "generated_ids.append(allocator, tok)"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "advanceStep(1)"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, body, "xfm.forwardWith(&self.ctx, tok_input)"));
}

/// Per-token logprob info (OpenAI format).
pub const TokenLogprob = struct {
    token_id: u32,
    logprob: f32,
};

/// Logprob result for a single generated token.
pub const LogprobResult = struct {
    token_logprob: f32, // logprob of the chosen token
    top_logprobs: []TokenLogprob, // top N alternatives (caller must free)
};

/// A thinking budget enforced while decoding: once `budget` tokens have been
/// generated inside the think block, `forced` (an early-stop sentence, the
/// pack's atomic closer, a separator) is committed through the model and the
/// answer follows. Markers are single token ids, so only packs whose think
/// tags are atomic tokens can be bound; `observe` walks the generated ids
/// incrementally and `due` asks whether the tail is an open block at budget.
pub const ThinkBound = struct {
    budget: u32,
    opener_id: ?u32,
    closer_id: u32,
    forced: []const u32,
    in_think: bool,
    count: u32 = 0,
    cursor: usize = 0,
    fired: bool = false,

    pub fn observe(self: *ThinkBound, ids: []const u32) void {
        while (self.cursor < ids.len) : (self.cursor += 1) {
            const id = ids[self.cursor];
            if (id == self.closer_id) {
                self.in_think = false;
            } else if (self.opener_id != null and id == self.opener_id.?) {
                self.in_think = true;
                self.count = 0;
            } else if (self.in_think) {
                self.count += 1;
            }
        }
    }

    pub fn due(self: *const ThinkBound) bool {
        return !self.fired and self.in_think and self.count >= self.budget;
    }
};

/// A forced tool call (`tool_choice` required or named) enforced while
/// decoding: the thought runs free, and the position after its closer (or
/// token 0 when the prompt already closed it) is `forced` instead of a sample.
/// For `required` the name follows as one of `names`, the model choosing among
/// the ids that continue one. Handler-owned like `ThinkBound`; every successor
/// decision consults `decide`, which reads the published tokens incrementally.
pub const CallForce = struct {
    /// Separator ids, then the call's; `call_at` indexes the call, which alone
    /// is committed when whitespace already follows the closer.
    forced: []const u32,
    call_at: usize,
    closer_id: ?u32,
    /// Committed at token 0 in `.open`, where the template leaves the think
    /// opener to the model.
    opener: [1]u32 = .{0},
    phase: Phase,
    /// The token before the next position: the last prompt id until one is published.
    last: u32,
    /// Each declared function's name and closing delimiter as ids; empty when
    /// `forced` already names the function.
    names: []const []const u32 = &.{},
    /// The storage `names` slice into, when allocated.
    names_flat: []const u32 = &.{},
    /// Scratch for a branch's choices, at least `names.len` long.
    choice_buf: []u32 = &.{},
    name_from: ?usize = null,
    cursor: usize = 0,
    queue: []const u32 = &.{},

    pub const Phase = enum { open, thinking, answer, name, done };
    pub const Decision = union(enum) { sample, force: u32, choose: []const u32 };

    pub fn pending(self: *const CallForce) bool {
        return self.phase != .done or self.queue.len > 0;
    }

    /// What the next position must be.
    pub fn decide(self: *CallForce, published: []const u32) Decision {
        while (self.cursor < published.len) : (self.cursor += 1) {
            const id = published[self.cursor];
            if (self.phase == .thinking and self.closer_id != null and id == self.closer_id.?) self.phase = .answer;
            self.last = id;
        }
        if (self.queue.len == 0) switch (self.phase) {
            .open => {
                self.queue = &self.opener;
                self.phase = .thinking;
            },
            .answer => {
                const after_closer = self.closer_id != null and self.last == self.closer_id.?;
                self.queue = if (after_closer) self.forced else self.forced[self.call_at..];
                self.phase = if (self.names.len > 0) .name else .done;
            },
            .name => return self.nextNameToken(published),
            .thinking, .done => return .sample,
        };
        if (self.queue.len == 0) return .sample;
        const id = self.queue[0];
        self.queue = self.queue[1..];
        return .{ .force = id };
    }

    fn nextNameToken(self: *CallForce, published: []const u32) Decision {
        const from = self.name_from orelse published.len;
        self.name_from = from;
        const typed = published[from..];
        var n: usize = 0;
        var complete = false;
        for (self.names) |name| {
            if (name.len < typed.len or !std.mem.eql(u32, name[0..typed.len], typed)) continue;
            if (name.len == typed.len) {
                complete = true;
                break;
            }
            if (std.mem.indexOfScalar(u32, self.choice_buf[0..n], name[typed.len]) == null) {
                self.choice_buf[n] = name[typed.len];
                n += 1;
            }
        }
        if (!complete and n == 1) return .{ .force = self.choice_buf[0] };
        if (!complete and n > 1) return .{ .choose = self.choice_buf[0..n] };
        self.phase = .done;
        return .sample;
    }

    /// Frees what `server.armCallForce` allocated.
    pub fn deinit(self: *CallForce, allocator: std.mem.Allocator) void {
        allocator.free(self.forced);
        allocator.free(self.names_flat);
        allocator.free(self.names);
        allocator.free(self.choice_buf);
    }
};

pub fn isGreedyTemperature(temperature: f32) bool {
    return temperature < 0.01;
}

/// Sampling parameters for token generation.
pub const SamplingParams = struct {
    temperature: f32 = 1.0,
    top_p: f32 = 1.0,
    top_k: u32 = 0, // 0 = disabled
    repeat_penalty: f32 = 1.0,
    presence_penalty: f32 = 0.0, // 0.0 = disabled
    seed: ?u64 = null,
    /// Draw index under `seed`: every sample takes a fresh key.
    draw: u64 = 0,
    /// When non-null, generation is constrained to outputs that satisfy the
    /// grammar at byte level. Forces a synchronous sampling path (no lazy
    /// pipeline) since grammar advancement requires the realized token id.
    constraint: ?*Constraint = null,
    /// In-stream thinking budget (`ThinkBound`), owned by the request handler
    /// like `constraint`; null = no bound.
    think_bound: ?*ThinkBound = null,
    /// Forced tool call (`CallForce`), owned by the request handler like
    /// `think_bound`; null = the model chooses whether to call.
    call_force: ?*CallForce = null,
    /// Reserved-token suppression mask: `[vocab]` bool, true = the sampler
    /// must never draw this id (reserved specials like `<|fim_hole|>`, which
    /// a degenerate distribution can rank top-5 at a collapsed position — a
    /// reserved marker in chat output is always a bug). Model-lifetime,
    /// OWNED by the Transformer (`suppress_mask`), non-owning here; wired by
    /// `Generator.initWithOptions` so every sampling path inherits it.
    /// Applied by both samplers and both stochastic-verify filters, fully
    /// lazy (`mlx_where` + -inf, no host sync); logprobs deliberately keep
    /// reading the RAW logits — the field reports the model, the mask is
    /// sampling policy. Null = no suppression (kill switch, no-template
    /// models, every non-suppressing arch).
    suppress_mask: ?mlx.mlx_array = null,
    /// Overthinking-marker penalty; the Generator's copy tracks the span over its committed ids.
    think_penalty: ThinkPenalty = .{},
};

/// Whether a request consumes only each position's argmax: greedy (or top-1), no logit-moving
/// penalty, no logprobs, no grammar or forced call choosing among ids. Only then may a
/// forward serve pruned or shortlisted logits.
pub fn argmaxOnlyRequest(sampling: SamplingParams, logprobs_n: u32) bool {
    return (isGreedyTemperature(sampling.temperature) or sampling.top_k == 1) and
        !penaltyActive(sampling) and
        !sampling.think_penalty.active() and
        sampling.constraint == null and
        sampling.call_force == null and
        logprobs_n == 0;
}

/// Build the `[vocab]` bool suppression mask (true = never sample) on the
/// host, once per model load. Caller owns the returned array.
///
/// Two disjoint sources of "never sample this row":
///   - `ids`: the reserved specials (`tokenizer.reservedOutputIds`).
///   - `[defined_vocab, vocab)`: the checkpoint's PADDING rows. A config's
///     `vocab_size` is a padded matrix dimension, not a vocabulary — the
///     tokenizer's highest defined id + 1 is. The rows in between decode to
///     nothing, so a sample from one emits nothing while consuming a step and
///     poisoning the KV; they were the one class of unsampleable row this
///     mask never covered.
/// `defined_vocab == 0` or `>= vocab` means "no padding known" and adds
/// nothing — a mask that suppressed the whole vocab is `-inf` everywhere,
/// which argmaxes to id 0 (the all-false-mask class, from the other side).
pub fn buildSuppressMask(ids: []const u32, defined_vocab: usize, vocab: usize, s: mlx.mlx_stream) !mlx.mlx_array {
    _ = s;
    const alloc = std.heap.page_allocator;
    const buf = try alloc.alloc(bool, vocab);
    defer alloc.free(buf);
    fillSuppressMask(buf, ids, defined_vocab);
    const shape = [_]c_int{@intCast(vocab)};
    return mlx.mlx_array_new_data(buf.ptr, &shape, 1, .bool_);
}

/// The host half of `buildSuppressMask`, pure so the row set is testable
/// without a GPU. `buf.len` is the mask's vocab.
pub fn fillSuppressMask(buf: []bool, ids: []const u32, defined_vocab: usize) void {
    @memset(buf, false);
    for (ids) |id| {
        if (id < buf.len) buf[id] = true;
    }
    if (defined_vocab > 0 and defined_vocab < buf.len) {
        @memset(buf[defined_vocab..], true);
    }
}

/// Derive + install the reserved-token suppression mask on a freshly-built
/// Transformer (both the serve load path and the CLI run path call this).
/// The id set is `tokenizer.reservedOutputIds`: `special: true` added tokens
/// minus EOS/stop ids minus template-emitted markers. Never fails a load — any error logs and
/// leaves the mask null (suppression off), and engagement is one-shot-logged
/// so a silent no-op is visible (the silent-fallback class).
pub fn installSuppressMask(xfm: *Transformer, tok: *const Tokenizer, chat_template: []const u8, eos_ids: []const u32) void {
    const ids = tokenizer_mod.reservedOutputIds(xfm.allocator, tok.flagged_specials, chat_template, eos_ids) catch |err| {
        log.warn("[suppress] derivation failed ({s}); reserved-token suppression off\n", .{@errorName(err)});
        return;
    };
    defer xfm.allocator.free(ids);
    const logits_dim: usize = xfm.config.vocab_size;
    // Padding rows past the tokenizer's last defined id.
    const defined_vocab: usize = tok.definedVocabSize();
    const pad_rows: usize = if (defined_vocab > 0 and defined_vocab < logits_dim)
        logits_dim - defined_vocab
    else
        0;
    if (ids.len == 0 and pad_rows == 0) return;
    xfm.suppress_mask = buildSuppressMask(ids, defined_vocab, logits_dim, xfm.s) catch |err| {
        log.warn("[suppress] mask build failed ({s}); reserved-token suppression off\n", .{@errorName(err)});
        return;
    };
    log.info(
        "[suppress] {d} of {d} flagged specials masked from sampling (template + eos exempt); {d} padding rows past id {d} of {d}\n",
        .{ ids.len, tok.flagged_specials.len, pad_rows, defined_vocab, logits_dim },
    );
}

/// Map the overthinking markers onto this tokenizer once per load (`think_penalty.markerIds`).
/// Never fails a load: an error leaves the mask null and the penalty inert.
pub fn installThinkMarkers(xfm: *Transformer, tok: *const Tokenizer) void {
    const ids = think_penalty.markerIds(xfm.allocator, &think_penalty.WORDS, tok) catch |err| {
        log.warn("[think-penalty] marker mapping failed ({s}); penalty off\n", .{@errorName(err)});
        return;
    };
    defer xfm.allocator.free(ids);
    const logits_dim: usize = xfm.config.vocab_size;
    xfm.think_marker_mask = think_penalty.buildMask(xfm.allocator, ids, logits_dim) catch |err| {
        log.warn("[think-penalty] mask build failed ({s}); penalty off\n", .{@errorName(err)});
        return;
    };
    log.debug("[think-penalty] {d} marker tokens from {d} words (multi-token spellings skipped)\n", .{ ids.len, think_penalty.WORDS.len });
}

/// `out = where(mask, -inf, logits)` — the masked lanes get EXACTLY -inf
/// (never an additive -inf, whose 0×-inf/NaN edge the parity rules exist
/// for). `[V]` broadcasts over both `[1, V]` and `[1, 1, V]` logits.
pub fn applySuppressMask(out: *mlx.mlx_array, logits: mlx.mlx_array, mask: mlx.mlx_array, s: mlx.mlx_stream) !void {
    const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer _ = mlx.mlx_array_free(neg_inf);
    try mlx.check(mlx.mlx_where(out, mask, neg_inf, logits, s));
}

/// A scalar in the array's own dtype, so a masked bf16 block is not widened to
/// f32 by its own `-inf`.
fn scalarLike(v: f32, like: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const raw = mlx.mlx_array_new_float(v);
    defer _ = mlx.mlx_array_free(raw);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_astype(&out, raw, mlx.mlx_array_dtype(like), s));
    return out;
}

/// A greedy spec-verify argmax whose entries are COMMITTED as next tokens.
/// `verifyArgmax` is its only constructor and every commit site takes this type,
/// so a raw `mlx_argmax_axis` cannot reach a commit without the reserved-id mask.
pub const CommittedArgmax = struct {
    arr: mlx.mlx_array = .{ .ctx = null },

    pub fn deinit(self: *CommittedArgmax) void {
        if (self.arr.ctx != null) _ = mlx.mlx_array_free(self.arr);
        self.arr = .{ .ctx = null };
    }

    pub fn present(self: CommittedArgmax) bool {
        return self.arr.ctx != null;
    }

    /// The lazy handle, for an eval vector or a forced eval - never for a commit.
    pub fn lazy(self: CommittedArgmax) mlx.mlx_array {
        return self.arr;
    }

    /// The realized ids; the caller must have evaluated `lazy()` first.
    pub fn ids(self: CommittedArgmax, len: usize) ![]const i32 {
        const data = mlx.mlx_array_data_int32(self.arr) orelse return error.MlxArrayDataNull;
        return data[0..len];
    }
};

/// The greedy spec-verify argmax, over the reserved-id mask when the model has
/// one. The sampled verify filters through `applySuppressMask` and the rerank
/// draft select takes the mask, so without this the greedy verify was the one
/// path that could commit a flagged special or a padding row past the defined
/// vocabulary — ids the serial sampler can never draw.
fn verifyArgmax(logits: mlx.mlx_array, mask: ?mlx.mlx_array, s: mlx.mlx_stream) !CommittedArgmax {
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    if (mask) |m| {
        const neg_inf = try scalarLike(-std.math.inf(f32), logits, s);
        defer _ = mlx.mlx_array_free(neg_inf);
        var masked = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(masked);
        try mlx.check(mlx.mlx_where(&masked, m, neg_inf, logits, s));
        try mlx.check(mlx.mlx_argmax_axis(&out, masked, 2, false, s));
    } else {
        try mlx.check(mlx.mlx_argmax_axis(&out, logits, 2, false, s));
    }
    return .{ .arr = out };
}

/// A speculative decode step (DFlash / MTP) cannot honor a grammar
/// constraint — the drafts bypass the per-token grammar mask — nor per-token
/// logprobs, which the draft path never captures. The scheduler and server
/// already select a spec mode only when both are absent, but `nextDflash` /
/// `nextMtp` only *documented* that with `std.debug.assert`,
/// which compiles out in the ReleaseFast build users actually run (issue #97).
/// A future dispatch bug that let a constrained request reach a spec step would
/// then read the constrained branch's placeholder token as a real commit and
/// stream silently off-schema output as a normal 200. So the spec steps gate on
/// this real, release-enforced check and return `error.SpecDecodeUnsupported`
/// instead of asserting.
fn specDecodeUnsupported(sampling: SamplingParams, logprobs_n: u32) bool {
    return draftsRefused(sampling) or logprobs_n != 0;
}

/// MTP rounds shift every verify row for a repeat or presence penalty (`penaltyShiftRows`), so
/// only what a draft run cannot honour keeps them off.
pub fn mtpUnsupported(sampling: SamplingParams, logprobs_n: u32) bool {
    return mtpRefusal(sampling, logprobs_n) != null;
}

/// Why a request runs without MTP, for the log; null when MTP may run.
pub fn mtpRefusal(sampling: SamplingParams, logprobs_n: u32) ?[]const u8 {
    if (logprobs_n != 0) return "logprobs requested";
    if (sampling.constraint != null) return "grammar-constrained output";
    if (sampling.call_force != null) return "forced tool call";
    return null;
}

/// A grammar mask or a forced tool call decides tokens a draft run would skip past, and a
/// repeat or presence penalty reads ids no verify row has committed yet; only MTP shifts the rows.
pub fn draftsRefused(sampling: SamplingParams) bool {
    return sampling.constraint != null or sampling.call_force != null or penaltyActive(sampling);
}

/// Logit biases and the think penalty ride the verifier (`Generator.glmPenalty`).
pub fn glmDflashEligible(sampling: SamplingParams, logprobs_n: u32) bool {
    return !specDecodeUnsupported(sampling, logprobs_n) and sampling.think_bound == null;
}

/// A verbatim-copy round: `pending` and the context's next three tokens as a chain, when the copy
/// agrees back `mtp_lookup.STRONG_SUFFIX` tokens and the budget holds the whole chain.
pub fn glmLookupProposal(got: ?mtp_lookup.Match, pending: u32, budget: u32) ?@import("glm5_dflash.zig").Proposal {
    const drafts = Generator.glm_lookup_drafts;
    const m = got orelse return null;
    if (m.suffix < mtp_lookup.STRONG_SUFFIX or m.draft.len < drafts or budget <= drafts) return null;
    var chain = @import("glm5_dflash.zig").Proposal{ .count = drafts + 1 };
    chain.tokens[0] = pending;
    chain.parents[0] = -1;
    for (m.draft[0..drafts], 1..) |token, row| {
        chain.tokens[row] = token;
        chain.parents[row] = @intCast(row - 1);
    }
    return chain;
}

/// Takes the verifier's MLA capacity right after prefill, inside the request's admission bill: no
/// round grows a buffer, and a later admission already sees the capacity as live memory.
pub const GlmSpecWindow = struct { snapshot: transformer_mod.KVCacheSnapshot, base_pos: usize };

fn reserveGlmDflash(config: *const model_mod.ModelConfig, ctx: ForwardCtx, assistant: *const dflash_mod.DflashCtx, max_tokens: u32, s: mlx.mlx_stream) !void {
    const request = ctx.glm5_request orelse return error.GlmRequestMissing;
    const kv = ctx.cache.config;
    const kv_bits: u64 = if (kv.scheme == .off) 16 else kv.bits;
    const held = request.residentBytes() +| assistant.cache.residentBytes();
    const budget = @import("server.zig").glmDflashReserveBudget(config, request.offset, max_tokens, kv_bits, held);
    _ = try @import("glm5_dflash_reserve.zig").reserve(request, request.offset + max_tokens + 3, budget, s);
}

/// Only the synchronous serial sampler (`sampleToken`) applies these penalties. A repeat
/// penalty of 0 or below is off: it would divide by zero or flip signs.
pub fn penaltyActive(sampling: SamplingParams) bool {
    return repeatPenaltyOn(sampling.repeat_penalty) or sampling.presence_penalty != 0.0;
}

fn repeatPenaltyOn(repeat_penalty: f32) bool {
    return seen_penalty.repeatOn(repeat_penalty);
}

/// Generation result (for non-streaming use).
pub const GenerationResult = struct {
    text: []u8,
    token_ids: []u32,
    prompt_tokens: u32,
    completion_tokens: u32,
    finish_reason: []const u8,
    prefill_tps: f64,
    decode_tps: f64,
    /// Wall-clock nanoseconds spent on prefill (prompt processing).
    prefill_ns: u64 = 0,
    /// Wall-clock nanoseconds spent on decode (token generation).
    decode_ns: u64 = 0,
    /// Prompt tokens served from a KV-cache prefix (the hot prefix cache).
    /// `prompt_tokens - cached_tokens`
    /// is what was actually run through the model this turn, so `prefill_tps`
    /// reflects real compute rather than an inflated full-prompt rate.
    cached_tokens: u32 = 0,
    logprobs: ?[]LogprobResult = null, // per-token logprobs (caller must free)
    /// Non-null only when the degenerate-tail guard cut this generation:
    /// the `finish_details.type` value emitted beside `finish_reason`
    /// ("stop" for loop cuts — see `scheduler.loopStopReason`).
    /// Static string; nothing to free.
    finish_details: ?[]const u8 = null,
    /// Absolute byte offset into `text` where the constrained JSON payload
    /// begins, when a reasoning protocol was active and the payload began.
    /// 0 = direct answer. Null when no protocol ran, the payload never began
    /// (completion cap mid-reasoning), or the boundary fell outside the
    /// emitted (loop-trimmed) tokens.
    constraint_payload_byte: ?usize = null,
};

/// Throughput in tokens/sec. Returns 0 when no time elapsed so unmeasured paths
/// report 0 rather than inf / NaN.
pub fn tokensPerSec(tokens: u64, elapsed_ns: u64) f64 {
    if (elapsed_ns == 0) return 0.0;
    const tok_f: f64 = @floatFromInt(tokens);
    const ns_f: f64 = @floatFromInt(elapsed_ns);
    return tok_f * @as(f64, @floatFromInt(std.time.ns_per_s)) / ns_f;
}

/// True prefill compute throughput: divides by the tokens actually pushed through
/// the model (prompt minus the prefix served from KV cache). A near-full cache
/// hit therefore reports the small suffix's real rate, not an inflated
/// full-prompt number. With `cached_tokens == 0` this is just the full-prompt
/// rate, matching the pre-instrumentation behavior.
pub fn prefillTokensPerSec(prompt_tokens: u32, cached_tokens: u32, prefill_ns: u64) f64 {
    const uncached: u32 = if (prompt_tokens > cached_tokens) prompt_tokens - cached_tokens else 0;
    return tokensPerSec(uncached, prefill_ns);
}

/// Pick the end position of the next prefill chunk starting at `pos`.
///
/// Base behavior: advance by `default_chunk` (the memory-bound `PREFILL_CHUNK`),
/// clamped to `prefix_len`. When SSM checkpointing is active, shrink the chunk so
/// it ends exactly on the next `ssm_cp_stride`-aligned ABSOLUTE position — that
/// lays down a stride-aligned SSM snapshot without changing what the model sees
/// (attention is causal; SSM/conv update chunk-locally, so the forward result is
/// identical to an unchunked run).
///
/// Pulled out of the prefill loop so the chunk-count behavior is unit-testable:
/// the stride directly controls how many chunks a cold prefill costs, and on
/// large MoE/hybrid models each extra chunk re-streams the (huge) expert weights
/// from HBM — the dominant cold-prefill cost. A too-small stride therefore
/// silently tanks MoE prefill throughput (~25% on 35B-class models for an
/// 850-token prompt at stride 256). Keeping typical prompts single-chunk is what
/// Which checkpoint a capped list drops. An SSD record carries its media identity and persists
/// whole, so a checkpoint past the first media row restores like any other and the generic
/// span-preserving thinning decides; protecting the boundary's checkpoint instead dropped every
/// one above it, which is exactly the state an image-bearing turn restores from next time.
fn checkpointDropIndex(items: []const SSMCheckpoint, policy: transformer_mod.ThinPolicy) usize {
    return transformer_mod.ssmCheckpointDropIndex(items, policy, null);
}

test "a capped list thins span-preserving whether or not the prompt carries media" {
    // The SSD record no longer ends at the first media row, so there is no boundary to protect:
    // a checkpoint above the media is a restore point like any other.
    var list = [_]SSMCheckpoint{
        .{ .pos = 512, .layers = &.{} },
        .{ .pos = 768, .layers = &.{} },
        .{ .pos = 1536, .layers = &.{} },
    };
    try testing.expectEqual(transformer_mod.ssmCheckpointDropIndex(&list, .min_span_recency, null), checkpointDropIndex(&list, .min_span_recency));
    list = .{ .{ .pos = 768, .layers = &.{} }, .{ .pos = 1536, .layers = &.{} }, .{ .pos = 2048, .layers = &.{} } };
    try testing.expectEqual(transformer_mod.ssmCheckpointDropIndex(&list, .min_span_recency, null), checkpointDropIndex(&list, .min_span_recency));
}

/// `ssm_checkpoint_stride`'s default guards.
/// A trailing remainder smaller than this merges into the preceding chunk
/// instead of becoming its own chunk. Chat-templated prompts routinely land a
/// token or two past a chunk multiple (an "8192-token" prompt tokenizes to
/// 8193); a 1-token final chunk pays a full graph build + eval barrier +
/// cache clear for one token. The merged chunk's attention-score transient
/// grows by at most TAIL_MERGE_MAX/default_chunk (~6% at 8192) — within the
/// score-budget slack `boundedPrefillChunk` already carries.
pub const TAIL_MERGE_MAX: usize = 512;

/// The tail a chunk of `default_chunk` tokens may absorb where the width is chosen per chunk:
/// an eighth of the width keeps the flat 512's ~6% bound at every rung (at the ladder floor
/// 512 is +100% of the transient a step-down just bought). Reach it through `tailMergeMaxFor`.
pub fn tailMergeMax(default_chunk: usize) usize {
    return @min(TAIL_MERGE_MAX, @max(default_chunk / 8, 1));
}

/// The tail bound the chunk loop uses: scaled only under the per-chunk adaptive width, the
/// flat constant everywhere else (chunk boundaries are not byte-stable).
pub fn tailMergeMaxFor(default_chunk: usize, adaptive_width: bool) usize {
    return if (adaptive_width) tailMergeMax(default_chunk) else TAIL_MERGE_MAX;
}

pub fn nextChunkEnd(
    pos: usize,
    prefix_len: usize,
    default_chunk: usize,
    want_ssm_cp: bool,
    ssm_cp_stride: usize,
    ssm_cp_offset: usize,
    // Is the per-chunk adaptive width live for this prefill? Only then does the merge bound scale.
    adaptive_width: bool,
) usize {
    var end = @min(pos + default_chunk, prefix_len);
    if (want_ssm_cp and ssm_cp_stride > 0) {
        const abs_pos = pos + ssm_cp_offset;
        const abs_end = end + ssm_cp_offset;
        const next_boundary_abs = ((abs_pos / ssm_cp_stride) + 1) * ssm_cp_stride;
        if (next_boundary_abs > abs_pos and next_boundary_abs < abs_end) {
            // A stride boundary lands inside this chunk — end exactly on it
            // (never tail-merge past it; the boundary IS the snapshot point).
            return next_boundary_abs - ssm_cp_offset;
        }
    }
    if (end < prefix_len and prefix_len - end < tailMergeMaxFor(default_chunk, adaptive_width)) {
        // Absorb a tiny tail instead of paying a full graph build + eval
        // barrier for a few tokens. With checkpointing active this can only
        // extend within a boundary-free span (the boundary case returned
        // above), so at most a snapshot < TAIL_MERGE_MAX tokens before the
        // end is skipped — and the always-on end-of-prompt snapshot lands
        // right there anyway.
        end = prefix_len;
    }
    return end;
}

/// `nextChunkEnd` for a GLM prefill, whose checkpoint grid must be hit exactly: the tail merge never
/// absorbs a grid point.
pub fn glmChunkEnd(pos: usize, prefix_len: usize, default_chunk: usize, grid: usize, offset: usize, adaptive_width: bool) usize {
    // Grid 0 = no checkpoints for this request (prefix cache off, or admission shed them all).
    if (grid == 0) return nextChunkEnd(pos, prefix_len, default_chunk, false, 0, offset, adaptive_width);
    const next_grid = ((pos + offset) / grid + 1) * grid - offset;
    return @min(nextChunkEnd(pos, prefix_len, default_chunk, true, grid, offset, adaptive_width), next_grid);
}

/// Tokens held back from the chunked-prefill loop and forwarded together with
/// the final (logits) forward when SSM checkpointing is active, so the
/// always-on snapshot lands SSM_SNAPSHOT_BACKOFF tokens BEFORE the prompt end.
///
/// A snapshot exactly at the prompt end is unreachable for the next turn's
/// prefix match: the template's generation-prompt suffix
/// ("<|im_start|>assistant\n" + think opener) renders differently once the
/// turn enters history, so the match always falls a few tokens short of the
/// full prompt — "[hot-cache] hybrid miss (no checkpoint ≤ 870 of 897)" was
/// llmprobe's prompt-cache-prefix cell failing (the 2026-06-10 class; fine
/// strides used to mask it by laying boundaries underneath). 30 covers every
/// template suffix in the fleet with margin (ChatML+think ≈ 7 tokens, laguna
/// pre-opened think ≈ 10) while keeping the tail forward UNDER the
/// prefill-eval-cadence threshold (seq >= 32): a 65-token tail was treated as
/// a prefill and paid ~450ms of mid-loop eval bubbles on the 27B, where the
/// 31-token tail rides the verify-shaped fast path. Cold cost is then ~zero
/// (the final forward pays its full weight sweep for ONE token anyway).
/// Warm restores re-forward ≤ backoff+1 tokens.
pub const SSM_SNAPSHOT_BACKOFF: usize = 30;

/// How many trailing prompt tokens the final (logits) forward covers: the
/// held-back snapshot window plus the last token itself. Pure so the
/// backoff/loop-bound interaction is unit-testable.
///
/// A restored tail that fits the window (`restored`: the prefill resumes
/// from a checkpoint) IS the cold run's final span, so it forwards as one
/// span of the same rows: a 30 + 1 split reads different kernel tilings and
/// greedy flips at a bf16 near-tie. No new snapshot: the restored one is it.
pub fn ssmSnapshotBackoff(want_ssm_cp: bool, prefix_len: usize, restored: bool) usize {
    if (!want_ssm_cp) return 0;
    if (prefix_len <= SSM_SNAPSHOT_BACKOFF) return if (restored) prefix_len else 0;
    return SSM_SNAPSHOT_BACKOFF;
}

/// GLM's checkpoint grid: absolute multiples of its widest prefill chunk. A narrower chunk divides
/// it and `nextChunkEnd` ends a chunk on every grid point, so a grid point is a chunk boundary in
/// every request whatever widths it runs. A test seam; serving never changes it.
pub var glm_checkpoint_stride: usize = glm5_forward.prefill_chunk;

/// GLM restores only on an IndexPool boundary, so its prompt-end snapshot backs off up to 3
/// more tokens to land on an absolute multiple of 4.
pub fn glmSnapshotBackoff(backoff: usize, offset: usize, prefix_len: usize) usize {
    if (backoff >= prefix_len) return backoff;
    return @min(backoff + (offset + prefix_len - backoff) % glm5_prefix.pool_size, prefix_len);
}

/// What a GLM prefill captures: the grid points inside `(offset, loop end]`, then one checkpoint at the
/// loop's end. Admission's bill and the prefill loop both read it.
pub const GlmCaptureSchedule = struct {
    /// Tail tokens held back from the chunk loop for the final span.
    backoff: usize,
    /// Checkpoints taken over the request.
    events: u32,
    /// Most held at once: one past the cap before each thin.
    peak: u32,
    /// Absolute position of the newest checkpoint, 0 = none.
    newest: usize,
};

/// The GLM prefill's capture schedule for a tail of `tail_len` tokens starting at absolute `offset`;
/// `cap` 0 thins nothing.
pub fn glmCaptureSchedule(grid: usize, cap: u32, offset: usize, tail_len: usize) GlmCaptureSchedule {
    var out: GlmCaptureSchedule = .{ .backoff = 0, .events = 0, .peak = 0, .newest = 0 };
    if (tail_len <= 1) return out;
    const prefix_len = tail_len - 1;
    out.backoff = glmSnapshotBackoff(ssmSnapshotBackoff(true, prefix_len, offset > 0), offset, prefix_len);
    const loop_end = prefix_len - out.backoff;
    if (loop_end == 0) return out;
    const end = offset + loop_end;
    var events: usize = end / grid - offset / grid;
    out.newest = end / grid * grid;
    // A grid point at the loop's end is already captured by the chunk loop.
    if (end % grid != 0) {
        events += 1;
        out.newest = end;
    }
    out.events = @intCast(@min(events, std.math.maxInt(u32)));
    out.peak = if (cap > 0) @min(out.events, cap + 1) else out.events;
    return out;
}

/// GLM restores only on an IndexPool boundary; every other arch at any checkpoint position.
fn restorableCheckpoint(glm: ?*glm5_forward.Request, pos: usize) bool {
    return glm == null or pos % glm5_prefix.pool_size == 0;
}

/// A prefill checkpoint at absolute `pos`: the slot's SSM entries, or the GLM request's KDA state.
fn capturePrefillCheckpoint(allocator: std.mem.Allocator, xfm: *Transformer, ctx: *const ForwardCtx, glm: ?*glm5_forward.Request, pos: usize) !SSMCheckpoint {
    if (glm) |request| {
        if (request.offset != pos) return error.GlmCachePositionMismatch;
        return glm5_prefix.captureKda(allocator, request, xfm.s);
    }
    return captureSsmCheckpoint(allocator, ctx.ssm_entries.?, pos, xfm.s);
}

/// Effective SSM-checkpoint stride for a model, given the base (configured)
/// stride: checkpointing never sub-divides the prefill chunk, on ANY arch.
///
/// The old policy kept the fine base stride on dense (non-MoE) hybrids on the
/// theory that their prefill is compute-bound so extra chunks are ~free. That
/// was true before `prefillDqGemm`: since the dq route only engages at
/// M >= 2048, a fine stride pushes EVERY projection of EVERY chunk onto the
/// slow small-M qmm path, and the per-chunk fixed costs (graph build, eval
/// barrier, MTP history capture) multiply on top. Measured on Qwen3.6-27B
/// dense GDN (M4 Max, 2026-07-30): stride 256 chunked an 8K prompt into 33
/// pieces at 211 tok/s vs 254 at coarse chunks — a 17-20% cold-prefill tax at
/// every context length, which is how llm_context_benchmarks read us 19%
/// behind oMLX. MoE pays even more (expert re-streaming, ~25%).
///
/// Warm mid-prompt reuse granularity drops to the chunk size, but the
/// always-on end-of-prompt snapshot still covers the dominant append-growth
/// multi-turn case (llmprobe's cache-hit tests restore from it at any
/// stride). `base == 0` (checkpointing disabled) is preserved, and a larger
/// explicit stride is never shrunk.
pub fn effectiveSsmCheckpointStride(base: usize, prefill_chunk: usize) usize {
    if (base == 0) return 0;
    return @max(base, prefill_chunk);
}

/// Tokens of KV capacity this prefill reserves up front (#353): removes the grow transient a
/// long prefill pays, at the price of allocating generation headroom early. Gated on
/// `longCtxGated`; every other arch keeps proportional growth (`reserve_tokens` stays 0).
/// `server.prefillRequestTerms` bills it.
pub fn reservedPrefillTokens(
    config: *const model_mod.ModelConfig,
    seq: u64,
    max_tokens: u64,
    chunk: u64,
) u64 {
    if (!config.reservesKvCapacity()) return 0;
    return transformer_mod.KVCache.reservedTokens(
        seq,
        max_tokens,
        chunk,
        config.max_position_embeddings,
    );
}

/// Reserve this request's cache capacity, and grow the restored layers to it, before the
/// prefill's first chunk writes (`KVCache.growToReservation`). Returns the reserved tokens.
pub fn reserveRequestCapacity(cache: *KVCache, config: *const model_mod.ModelConfig, total_ctx: u64, max_tokens: u64, chunk: u64, s: mlx.mlx_stream) !u64 {
    const reserved = reservedPrefillTokens(config, total_ctx, max_tokens, chunk);
    cache.reserve(@intCast(reserved));
    try cache.growToReservation(s);
    return reserved;
}

/// SSM checkpoints exist to feed prefix-cache reuse, and image-bearing
/// prompts are excluded from prefix reuse (equal placeholder IDs do not imply
/// equal images) — so vision prefills skip checkpointing even now that they
/// chunk (the splice scatter itself is chunk-safe via vision_splice_offset).
/// A vision prompt checkpoints like text once it chunks like text (#197):
/// without a checkpoint a hybrid's image turn is a guaranteed hot-cache miss.
pub fn shouldCheckpointSsmPrefill(stride: u32, has_ssm: bool, has_vision: bool) bool {
    return stride > 0 and has_ssm and (!has_vision or visionChunkedPrefillEnabled());
}

/// Chunked vision prefill (issue #197). Default ON: the splice resumes its
/// row index across chunk boundaries via `ForwardCtx.vision_splice_offset`,
/// so an image-bearing prompt prefills under the same chunk-bounded memory
/// envelope as text. SUSHI_VISION_CHUNKED=0 restores the whole-prompt
/// single forward (and the memory guard's full-width bill with it).
var vision_chunked_cached: ?bool = null;
pub fn visionChunkedPrefillEnabled() bool {
    if (vision_chunked_cached) |v| return v;
    const raw = std.c.getenv("SUSHI_VISION_CHUNKED");
    const on = raw == null or !std.mem.eql(u8, std.mem.sliceTo(raw.?, 0), "0");
    vision_chunked_cached = on;
    return on;
}

/// What the memory guard should bill a prefill at: full width only for a
/// vision prompt with chunking killed. The guard and this loop must agree or
/// admission either over-refuses (bills seq for a chunked prefill) or
/// under-bills (uncatchable Metal OOM).
pub fn visionPrefillUnchunked(has_vision: bool) bool {
    return has_vision and !visionChunkedPrefillEnabled();
}

/// Placeholder tokens (image + video soft tokens) in `ids` — the number of
/// source rows a chunk's splice consumes. Host-side count, no GPU sync.
pub fn countSpliceRows(ids: []const i32, image_token_id: u32, video_token_id: u32) usize {
    var n: usize = 0;
    for (ids) |id| {
        if (id == @as(i32, @intCast(image_token_id)) or
            (video_token_id > 0 and id == @as(i32, @intCast(video_token_id)))) n += 1;
    }
    return n;
}

/// Number of chunks a cold prefill of `prefix_len` tokens splits into for the
/// given chunk size / SSM-checkpoint stride. Mirrors the loop in `init` exactly
/// (drives the same `nextChunkEnd`), so a test on this is a faithful proxy for
/// the real prefill chunk count. Each chunk on a memory-bound MoE prefill
/// re-streams the expert weights, so this is effectively the cold-prefill
/// weight-traffic multiplier.
pub fn prefillChunkCount(
    prefix_len: usize,
    default_chunk: usize,
    want_ssm_cp: bool,
    ssm_cp_stride: usize,
    ssm_cp_offset: usize,
    adaptive_width: bool,
) usize {
    var pos: usize = 0;
    var n: usize = 0;
    while (pos < prefix_len) {
        const end = nextChunkEnd(pos, prefix_len, default_chunk, want_ssm_cp, ssm_cp_stride, ssm_cp_offset, adaptive_width);
        pos = end;
        n += 1;
    }
    return n;
}

/// Generated tokens between two returns of the decode loop's transients to
/// MLX's allocator. 256 is what the non-speculative paths have always used.
pub const CACHE_CLEAR_INTERVAL: u32 = 256;

/// PURE: has `interval` tokens passed since the last clear?
///
/// Interval arithmetic, not `step % interval == 0`: a spec-decode round emits
/// `1 + accepted` tokens, so a modulo test can step clean over every multiple
/// and never fire (issue #110 — at stride 5 it fires zero times in the first
/// 1024 steps). For the stride-1 paths this is byte-identical to the modulo
/// form it replaces.
pub fn shouldClearAllocatorCache(step: u32, last_clear: u32, interval: u32) bool {
    if (interval == 0) return false;
    return step -| last_clear >= interval;
}

/// PURE: does this request return MLX's allocator pool once, between its last
/// prefill chunk and its first decode tick?
///
/// The per-chunk clear runs before the last chunks' transient is freed, so that
/// transient parks in the pool up to the cap and the first decode tick allocates
/// on top of it (8.1 GB parked at the first tick of a 393k prefill). Gated on an
/// arch that reserves its KV up front: every other arch keeps its call pattern exactly.
pub fn clearsPoolAtPrefillEnd(config: *const model_mod.ModelConfig) bool {
    return config.reservesKvCapacity();
}

/// Number of accepted draft tokens that may accompany the always-committed
/// anchor token without crossing the request's output-token ceiling.
pub fn capAcceptedForTokenBudget(accepted: u32, completion: u32, max_tokens: u32) u32 {
    const remaining = max_tokens -| completion;
    if (remaining <= 1) return 0;
    return @min(accepted, remaining - 1);
}

test "capAcceptedForTokenBudget keeps speculative commits inside max_tokens" {
    try std.testing.expectEqual(@as(u32, 15), capAcceptedForTokenBudget(15, 0, 400));
    try std.testing.expectEqual(@as(u32, 2), capAcceptedForTokenBudget(15, 397, 400));
    try std.testing.expectEqual(@as(u32, 0), capAcceptedForTokenBudget(15, 399, 400));
    try std.testing.expectEqual(@as(u32, 0), capAcceptedForTokenBudget(15, 400, 400));
    try std.testing.expectEqual(@as(u32, 0), capAcceptedForTokenBudget(15, 401, 400));
}

test "every speculative decoder caps accepted drafts before commit" {
    // Class guard: every speculative path commits an always-emitted anchor
    // plus zero or more accepted drafts. A new/forgotten arm must cap that
    // accepted prefix before it mutates generated ids, KV/module state, or
    // usage. Output-only clipping in the scheduler is already too late.
    const source = @embedFile("generate.zig");
    const names = [_][]const u8{
        "nextDflash",
        "mtpRoundFinish",
    };
    for (names) |name| {
        const signature = try std.fmt.allocPrint(testing.allocator, "    pub fn {s}", .{name});
        defer testing.allocator.free(signature);
        const start = std.mem.indexOf(u8, source, signature) orelse return error.MissingSpecDecoder;
        const end = std.mem.indexOfPos(u8, source, start + signature.len, "\n    pub ") orelse source.len;
        if (std.mem.indexOf(u8, source[start..end], "capAcceptedForTokenBudget(") == null) {
            std.debug.print("{s} does not cap accepted drafts before commit\n", .{name});
            return error.UncappedSpecDecoder;
        }
    }
}

/// Step-based generator. Call `init` to prefill, then `next` per token.
/// Uses a fully-lazy async pipeline matching mlx-lm: sample + next forward are
/// built as a single lazy computation graph, async_eval'd together. The GPU
/// never idles between token generation steps.
pub const Generator = struct {
    const MtpGraphFn = *const fn (mlx.mlx_array, []const mlx.mlx_array, ?[]const mlx.mlx_array, u32, f32, SamplingParams, mlx.mlx_stream) anyerror!MtpBatchedGraph;
    xfm: *Transformer,
    /// Forward-pass context. Stores per-request KVCache pointer, moe_seq_offset
    /// pointer, ssm_entries slice, vision_embeddings handle, and capture_hidden
    /// override. The legacy single-slot path uses `xfm.defaultCtx()` (pointing at
    /// the Transformer's own fields). Phase 2 concurrent batching constructs a
    /// per-slot ForwardCtx pointing at the slot's own KVCache, etc., so multiple
    /// generators can share one Transformer's weights without colliding on
    /// per-request state. Stored by value; `&self.ctx` is what we pass to
    /// `xfm.forwardWith` / `lazyForward` / drafter step.
    ctx: ForwardCtx,
    tok: *const Tokenizer,
    next_token_id: u32,
    step: u32,
    /// `step` at the last `mlx_clear_cache()`. Advanced only by `advanceStep`,
    /// which is the ONE place `step` may move.
    last_cache_clear_step: u32 = 0,
    max_tokens: u32,
    sampling: SamplingParams,
    prompt_tokens: u32,
    completion_tokens: u32,
    finish_reason: []const u8,
    done: bool,
    eos_token_ids: []const u32,
    generated_ids: std.ArrayList(u32),
    /// Degenerate-tail detection must not reconsider a reasoning loop after
    /// its boundary has activated the final-answer grammar. The scheduler
    /// scans only this suffix; a later loop in the answer is still handled by
    /// the ordinary repetition-stop policy.
    loop_guard_start: usize = 0,
    consecutive_pad: u32 = 0, // count of consecutive token-0 (pad) generations
    timeout_ns: u64, // 0 = no timeout; measures SILENCE, not total time (see StallClock)
    stall: StallClock = .{},
    timer: io_util.Stopwatch,
    logprobs_n: u32 = 0, // 0 = disabled, >0 = number of top_logprobs to return
    last_logprob: ?LogprobResult = null, // logprob result for the most recently returned token
    /// Logprobs for the token the NEXT `next()` call will return.
    ///
    /// The decode loop returns `next_token_id` and, in the same call, forwards
    /// it to sample its successor — so the result `sampleToken` hands back
    /// describes the token AFTER the one being returned. Publishing it as
    /// `last_logprob` there shifted the whole array by one: the caller zips
    /// token_ids with logprobs by index, so at temp 0 every entry reported the
    /// distribution of the token that FOLLOWED it (a one-token "OK" reply came
    /// back with `<|role_end|>` at rank 1). Seeded at init with t1's own
    /// distribution — t1 is sampled from the prefill's final forward, which
    /// this loop never sees.
    pending_logprob: ?LogprobResult = null,
    // Async pipeline state: pre-computed forward pass logits for next decode step
    pending_logits: mlx.mlx_array = .{},
    has_pending_logits: bool = false,
    // Deferred token: lazy array from async pipeline, eval'd at start of next iteration
    pending_token: mlx.mlx_array = .{},
    has_pending_token: bool = false,

    // ── Spec-decode shared state (DFlash) ──
    // Post-final-norm hidden state at the last produced token's position.
    // Owned by the Generator (freed in `deinit`). Captured by
    // `forwardCaptureHidden` during prefill final-token forward and every
    // verify forward; the drafter's h_prev seed.
    last_hidden: mlx.mlx_array = .{},
    has_last_hidden: bool = false,
    /// PRNG for the DFlash stochastic-verify accept test (probability-
    /// ratio requires a uniform draw per draft step). Seeded from
    /// `sampling.seed` when set, otherwise from system time at init.
    prng: std.Random.DefaultPrng = std.Random.DefaultPrng.init(0),

    // Owned copy of the lookup prompt ids: the MTP prompt-lookup index extends from
    // `prompt_ids_owned ++ generated_ids`, and the caller-supplied `prompt_ids` slice is freed after
    // `init` returns. `generated_ids` (above) tracks post-prefill tokens; `prompt_ids_owned` is the
    // immutable prefix.
    prompt_ids_owned: []u32 = &.{},
    /// Allocator that owns `prompt_ids_owned`. Stored so `deinit` can free it
    /// without requiring callers to thread the allocator a second time. (Other
    /// owned slices are freed via the `allocator` argument to `deinit` for
    /// historical reasons; this one is set during `initWithOptions`.)
    prompt_ids_alloc: ?std.mem.Allocator = null,
    /// Tokens proposed per round (the DFlash-resolved block size).
    drafter_block_size: u32 = 4,

    // ── DFlash block-drafter state ──
    // External block-parallel assistant (src/dflash.zig). When
    // `dflash != null`, callers use `nextDflash` instead of `next`. Weights
    // are owned by the server (per-model); the per-request context cache
    // (`dflash_ctx`) is OWNED by the Generator — built during prefill from
    // the trunk's capture_layers hiddens, freed in `deinit`.
    dflash: ?*DflashModel = null,
    dflash_ctx: ?dflash_mod.DflashCtx = null,
    glm_dflash_native: bool = false,
    /// Accepted share of a DFlash2 round's drafts, smoothed; the grouped tick prices draft rows with it.
    glm_draft_rate: f32 = 0.8,
    /// Verbatim-copy rounds (`glmLookupChain`) and the drafts they landed, for `[spec-stats]`.
    glm_lookup: struct { rounds: u64 = 0, landed: u64 = 0 } = .{},
    /// The GLM assistant window as prefill left it, committed with the prefix: every GLM restore
    /// point lies at or below the prompt end, which a window cropped at the reply's end can miss.
    glm_prefill_window: ?GlmSpecWindow = null,
    /// Cumulative GLM DFlash2 drafts proposed and round phases, for `[spec-stats]`.
    glm_round: struct { drafted: u64 = 0, draft_ns: u64 = 0, verify_ns: u64 = 0, replay_ns: u64 = 0, commit_ns: u64 = 0 } = .{},
    /// Effective block size (assistant config, capped by the machine's verify lanes).
    /// Drafts per round = dflash_block_size - 1.
    dflash_block_size: u32 = 0,
    /// Wall ms of this request's DFlash rounds, for the cost gate.
    dflash_wall_ms: f64 = 0,
    /// Drafts the verifies of this request carried, and the selector-confidence calibration that picks rows.
    dflash_drafts_verified: u64 = 0,
    dflash_calib: dflash_policy.Calibration = .{},
    /// Drafts this round ran (0 = serial), read by the table feed.
    dflash_round_width: u32 = 0,
    /// Stats: count of nextDflash calls that ran a verify forward.
    dflash_attempted: u64 = 0,
    /// Stats: cumulative draft tokens accepted (excluding always-accepted t1).
    dflash_accepted_tokens: u64 = 0,
    /// Per-phase wall-time trace (SUSHI_DFLASH_TRACE=1; else untouched).
    /// Unlike the MTP trace this one INSERTS eval barriers to attribute a
    /// fully-lazy round — a traced round is slower than a real one by
    /// whatever overlap the barriers destroy. Diagnostic only.
    dflash_trace: DflashTrace = .{},
    /// Trace-only stopwatch across the scheduler gap (round return → next
    /// round entry); bills the `.gap` phase. Null when not tracing.
    dflash_gap_watch: ?io_util.Stopwatch = null,

    // ── Qwen native MTP head state ──
    // The model's own one-layer multi-token-prediction head (src/mtp.zig).
    // When `mtp != null`, callers use `nextMtp` instead of `next`. The head
    // is owned by the server (loaded with the model); the Generator only
    // holds a non-owning pointer. `mtp_cache` is the head's committed-history
    // KV cache — OWNED by the Generator (built during prefill, freed in
    // `deinit`).
    mtp: ?MtpHeadRef = null,
    /// Installed verifier route; selected once at init, not checked per token.
    mtp_accept_graph: MtpGraphFn = mtpBatchedExactGraph,
    mtp_accept_prefix: mtp_acceptance.PrefixFn = mtp_acceptance.exactPrefix,
    mtp_accept_param: f32 = 0,
    /// A sampled request drafts only depth 0 from the draft sampler, every later depth by argmax.
    mtp_greedy_tail: bool = false,
    /// Does the model have a usable MTP head? `--no-mtp` clears it; a per-request `enable_mtp:false` does not.
    model_has_mtp: bool = false,
    mtp_cache: ?MtpCacheRef = null,
    /// Absolute target position represented by MTP-cache position 0. Usually
    /// zero; nonzero when the head keeps only a suffix of a restored/long
    /// prompt. Used to map the sidecar's relative offsets into Qwen M-RoPE.
    mtp_position_base: usize = 0,
    /// CONFIGURED max tokens drafted per round (verify length = depth + 1); the
    /// floor is `depth_bounds.active.min`.
    mtp_depth: u32 = mtp_mod.DEFAULT_DEPTH,
    /// The cap WITHOUT the per-silicon row: an explicit --mtp-max-depth, else
    /// the adaptive default. The row is the cold-start cap; once the table
    /// has trusted widths it may plan up to this instead (the M4 base row
    /// of 4 measured -6% against what the table found at 6).
    mtp_depth_free: u32 = mtp_mod.DEFAULT_DEPTH,
    /// CURRENT adaptive depth (see updateMtpDepth). Starts at the floor,
    /// demoted/promoted per windowed acceptance, always inside floor..`mtp_depth`.
    mtp_depth_current: u32 = mtp_mod.DEFAULT_DEPTH,
    /// Stats: count of nextMtp calls that ran a verify forward.
    mtp_attempted: u64 = 0,
    /// Stats: cumulative draft tokens accepted (excluding always-accepted t1).
    mtp_accepted_tokens: u64 = 0,
    /// Adaptive-depth moving window: per-round drafted/accepted counts.
    mtp_window_drafted: [MTP_DEPTH_WINDOW]u8 = @splat(0),
    mtp_window_accepted: [MTP_DEPTH_WINDOW]u8 = @splat(0),
    mtp_window_idx: u32 = 0,
    mtp_rounds_since_switch: u32 = 0,
    /// Rounds remaining during which promotion is blocked (set after a
    /// demotion so a failed depth excursion isn't immediately retried).
    mtp_promote_cooldown: u32 = 0,
    /// Cumulative drafted tokens across rounds. The EV controller varies m
    /// per round, so `attempts x depth` no longer measures proposals — this
    /// is the honest per_draft_pct denominator.
    mtp_drafted_tokens: u64 = 0,
    /// Prompt-lookup rounds (`mtp_lookup`), kept apart from the MTP counters so the head's
    /// acceptance stats stay the head's.
    mtp_lookup_rounds: u64 = 0,
    mtp_lookup_drafted: u64 = 0,
    mtp_lookup_accepted: u64 = 0,
    /// Prompt + committed output, indexed up to `mtp_lookup_gen` generated tokens.
    mtp_lookup_idx: ?mtp_lookup.Index = null,
    mtp_lookup_gen: usize = 0,
    /// Landed drafts per round, smoothed, for lookup and MTP rounds (`mtp_lookup.gate`).
    mtp_lookup_ema: f32 = @floatFromInt(mtp_lookup.MAX_DRAFT),
    mtp_round_ema: f32 = 0,
    /// Drafts per MTP round, smoothed: the width `mtp_round_ema`'s rounds actually ran.
    mtp_round_drafted_ema: f32 = 0,
    /// The previous round was a lookup round.
    mtp_after_lookup: bool = false,
    /// The last lookup round landed every draft (`mtp_lookup.gate` streak).
    mtp_lookup_streak: bool = false,
    /// Rounds where the confidence gate extended into chunk B.
    mtp_ext_rounds: u64 = 0,
    /// This request's round walls, for the `stalls=` field.
    mtp_round_log: round_cost.RoundLog = .{},
    /// Speculative rounds that rolled recurrent state back on a partial accept.
    partial_rounds: u64 = 0,
    /// Extension dry-spell gate: consecutive extension-CONSIDERED rounds
    /// whose confidence gate did not clear, and the single-chunk cooldown
    /// that a full dry streak triggers (see mtpExtDryAllows).
    mtp_ext_dry_streak: u32 = 0,
    mtp_ext_cooldown: u32 = 0,
    /// EV controller: conditional acceptance EMA per draft index,
    /// a[i] = P(draft i accepted | drafts 0..i-1 accepted). Optimistic prior;
    /// warmup rounds pull the low indices to reality before it can matter.
    mtp_ev_accept: [mtp_mod.MAX_DEPTH]f32 = @splat(MTP_EV_PRIOR),
    /// Rounds seen by the EV controller (drives the legacy-behavior warmup).
    mtp_ev_rounds: u32 = 0,
    /// Last round's planned m_lo (base-depth climb damping: +1/round max).
    mtp_ev_m_lo_prev: u32 = 1,
    /// Round-cost surface selected once for this target+MTP head. The M5/G17
    /// NAX surface requires the measured trunk, native sidecar, and 3-bit
    /// draft-only head; every other combination keeps the M1-M4 surface.
    mtp_ev_costs: MtpEvCosts = MTP_EV_DEFAULT_COSTS,
    /// Live-cost EMAs (ms), 0 = unseeded. `mtp_ev_sync_ms` is the measured
    /// chunk-A confidence-read sync — updated ONLY on extension-considered
    /// rounds, so it holds the true cost of a sync when one happens (not the
    /// suppressed-round average). `mtp_ev_round_ms` is the realized round
    /// wall-clock. Their ratio drives the cost-aware exploration throttle
    /// (mtpExtDryThresholdFor) so a dry sync tax self-limits to a small
    /// fraction of the round. Always-on; the trace is not required.
    mtp_ev_sync_ms: f32 = 0,
    mtp_ev_round_ms: f32 = 0,
    mtp_regime: MtpRegime = .{},
    mtp_regime_verdict: ?bool = null,
    /// Width-trial schedule for the round-cost table (one 2-round block per
    /// period at m_lo+1, so the table learns the next cliff once).
    mtp_width_trial: MtpWidthTrial = .{},
    /// Consecutive plans with the same m_lo (the base has stopped climbing).
    mtp_m_lo_streak: u32 = 0,
    /// Drafts of the previous TWO rounds (0 = serial), every round: a round
    /// is a transition unless both predecessors ran its width (the first
    /// round after a width change is the transition, the second is still
    /// elevated — live, every w4 sample taken beside a trial read 17% high
    /// on the M4 base and fed the churn that produced it).
    spec_round_prev_width: ?u32 = null,
    spec_round_prev_width2: ?u32 = null,
    /// Shape of the previous two MTP rounds: a shape change is a transition too.
    spec_round_prev_two_chunk: bool = false,
    spec_round_prev_two_chunk2: bool = false,
    /// Wall clock between round ends: the regime compares the quantity
    /// tok/s reports, so per-round work OUTSIDE the round stopwatch (token
    /// publish, stop checks, scheduler) must be in the denominator — fewer
    /// rounds is part of a wider shape's win.
    mtp_regime_clock: ?io_util.Stopwatch = null,
    /// True while this generator is the ONLY decoding stream. Contention is
    /// strictly one-sided — it only ADDS time — so a busy server simply
    /// stops feeding the kv-term learner rather than teaching it a lie.
    /// Set per tick by the scheduler.
    spec_cost_solo: bool = true,
    /// Draft-depth cap while this slot verifies inside a batched group (0 = none):
    /// the group's rows must stay on the split-K verify lane (`scheduler.mtpGroupRowCap`).
    mtp_group_cap: u32 = 0,
    mtp_planner_owned: bool = false,
    mtp_planner_pending: bool = false,
    mtp_planner_width: ?u8 = null,
    mtp_planner_probe: bool = false,
    mtp_planner_probes: u8 = 0,
    mtp_planner_recovery: group_planner.Recovery = .{},
    mtp_planner_recovering: bool = false,
    mtp_planner_history: group_planner.History = .{},
    mtp_hidden_stale: bool = false,
    mtp_planner_plain_ticks: u32 = 0,
    mtp_planner_prime_ticks: u32 = 0,
    mtp_planner_spec_rounds: u32 = 0,
    mtp_planner_max_gap_ms: f32 = 0,
    /// This slot's next `mtpRoundBegin` inits the chain and returns `.open` so the
    /// scheduler can run the head step for the group in one graph.
    mtp_batch_head: bool = false,
    /// Per-phase wall-time trace (SUSHI_MTP_TRACE=1; else untouched).
    mtp_trace: MtpTrace = .{},
    /// Trace-only: stopwatch running across the scheduler gap (round return
    /// → next round entry); bills the `.gap` phase. Null when not tracing.
    mtp_gap_watch: ?io_util.Stopwatch = null,
    /// Deferred committed-history append: the round's (tokens, true verify
    /// hiddens) pair, folded into the NEXT round's first draft step as one
    /// multi-row head forward instead of a separate appendHistory forward.
    /// Rounds with no successor (EOS/length/runtime disable) never pay for
    /// the append; the stash is freed unconsumed in `deinit`.
    mtp_hist_stash: ?MtpHistStash = null,
    /// Cross-round pre-draft (round pipelining): the NEXT round's chunk-A
    /// draft chain, built and async-dispatched at the CURRENT round's tail
    /// so the head chain runs on the GPU while the CPU emits tokens. The
    /// build consumes `mtp_hist_stash`, so the two are mutually exclusive
    /// (asserted at consume). Freed unconsumed in `deinit`.
    mtp_pre_draft: ?MtpPreDraft = null,

    // ── Phase 1: SSM checkpoints captured during prefill ──
    /// Owned SSM-state snapshots taken at stride-aligned positions during
    /// chunked prefill. Drained by the scheduler in `commitSlotIfApplicable`
    /// via `takeSsmCheckpoints()`. Empty on non-hybrid models or when
    /// `ssm_checkpoint_stride == 0`. Allocator: the Generator's `allocator`
    /// (passed to `initWithOptions`); the same allocator must be passed to
    /// `deinit` for any checkpoint that wasn't taken.
    ssm_checkpoints: std.ArrayList(SSMCheckpoint) = std.ArrayList(SSMCheckpoint).empty,
    /// Allocator used for `ssm_checkpoints` storage and each checkpoint's
    /// per-layer slice. Set during `initWithOptions`. We track it separately
    /// from the `allocator` argument to `deinit` because `takeSsmCheckpoints`
    /// transfers ownership: the consumer (HotPrefixCache) must use the SAME
    /// allocator to free, since the layer-slice backing memory was allocated
    /// here.
    ssm_checkpoint_alloc: ?std.mem.Allocator = null,
    /// Most checkpoints the prefill held at once, counting the copy taken before each thin.
    ssm_cp_peak: u32 = 0,

    // ── Runtime acceptance gate ──
    // Set to true mid-request when the per-request acceptance rate
    // (`*_accepted_tokens / *_attempted`) falls below
    // the DFlash gate threshold after its warmup attempts. When set,
    // `nextDflash` and `nextMtp` short-circuit to `next()` for the
    // remainder of the request — the prompt-time gate could not foresee that
    // the workload's *runtime* draft acceptance rate wasn't paying for the
    // per-step verify overhead. The flag is sticky for the rest of the
    // generation; we never re-enable speculation within a single request.
    spec_disabled_runtime: bool = false,
    /// Why `spec_disabled_runtime` was set, for `[spec-stats]`.
    spec_disable_reason: SpecDisableReason = .none,
    /// A loop convicted inside the thought already closed it: one recovery per request.
    think_closed_by_loop: bool = false,
    /// Adaptive serial controller: which arm this request is on and the KV bucket it was decided in.
    mtp_adaptive: MtpAdaptive = .{},
    /// This request has released the module-owned MTP head (sticky serial arm reached a serial
    /// block boundary). Per request: one slow request must not turn MTP off process-wide.
    mtp_head_released: bool = false,
    /// This request's realized MTP price (ms per emitted token) over a trailing window.
    mtp_price: MtpPriceWindow = .{},
    /// Ticks left in a bounded serial block (the serial probe); the slot stays an MTP slot.
    mtp_serial_left: u32 = 0,
    /// Where a serial block sits in its ramp back to `nextMtp`'s entry invariant.
    mtp_serial_exit: MtpSerialExit = .none,
    /// Ticks of the current serial block spent warming (the previous round's tail).
    mtp_serial_warm: u32 = 0,
    mtp_serial_giveup_logged: bool = false,
    /// Inter-tick wall clock feeding the table's serial cell. Null = the next read seeds.
    mtp_serial_clock: ?io_util.Stopwatch = null,
    // DFlash has different economics from sequential-token drafters: one
    // assistant call proposes the whole block and the M5 NAX verify lane makes
    // wide blocks cheap. Per-draft percentage therefore penalizes block 16
    // even when it is decisively profitable. The stable cross-width signal is
    // accepted drafts per verify round. M5 Muse sweeps put code/tool traffic at
    // 4.4-15.0 accepted/round and the regressing prose/vision class at
    // 1.0-1.5; two is the measured break-even boundary. M4 block-5 model-card
    // workloads accepting 62-86% remain above it as well.
    // Warmup 8, not 3 (2026-09-03): three rounds tripped on ONE dry opening
    // (llmprobe's 16k rung: 0/3, drafter off for the whole request at 25
    // tok/s; the same prompt restored from cache decoded at 73).
    pub const DFLASH_GATE_WARMUP: u64 = 8;
    /// A DFlash round that costs more per emitted token than this multiple of a serial step is dropped.
    pub const DFLASH_GATE_MARGIN: f32 = 0.05;

    /// Accepted drafts per round at which a round of `rows` verifier rows costs one serial step per
    /// token, used until a serial sample exists. A round costs about 0.25 + 0.28 serial steps per
    /// row (Sushi-2.6bpw, M5 Max: 24.1, 30.6 and 46.7 ms at 2, 4 and 8 rows against a 13.3 ms token).
    pub fn dflashBreakEvenAccepted(rows: f32) f32 {
        return 0.25 + 0.28 * rows;
    }

    /// Does the request's DFlash yield lose to serial decoding? With a measured serial price (the
    /// round-cost table's cell) the test is ms per emitted token; without one it is the ladder's
    /// break-even yield. Never decided before the warmup rounds.
    pub fn dflashGateShouldDisable(attempted: u64, accepted: u64, wall_ms: f64, rows: f32, serial_ms_tok: ?f32) bool {
        if (attempted < DFLASH_GATE_WARMUP) return false;
        if (serial_ms_tok) |serial| {
            if (std.math.isFinite(serial) and serial > 0) {
                const emitted: f64 = @floatFromInt(attempted + accepted);
                return wall_ms / emitted > @as(f64, serial) * (1.0 + DFLASH_GATE_MARGIN);
            }
        }
        const avg = @as(f32, @floatFromInt(accepted)) / @as(f32, @floatFromInt(attempted));
        return avg < dflashBreakEvenAccepted(rows);
    }

    /// Verifier rows a round of this request carried on average: the block's until a round has run.
    fn dflashMeanRows(self: *const Generator) f32 {
        if (self.dflash_attempted == 0 or self.dflash_drafts_verified == 0) return @floatFromInt(self.dflash_block_size);
        return 1 + @as(f32, @floatFromInt(self.dflash_drafts_verified)) / @as(f32, @floatFromInt(self.dflash_attempted));
    }

    /// The serial price a DFlash request is held to: the plain-tick cell at width 0, which this
    /// path's own fallback ticks feed, else the serial row MTP's adaptive serial feeds.
    pub fn dflashSerialPrice(t: *const round_cost.Table, bucket: usize) ?f32 {
        return t.msPerTok(0, bucket) orelse t.serialMsPerTok(bucket);
    }

    /// Emit a stable, easy-to-grep one-line summary of spec-decode acceptance
    /// for this request. External tooling parses the `[spec-stats]` prefix;
    /// keep the format stable.
    ///
    /// No-op when this Generator never ran a speculative path.
    ///
    /// Field semantics:
    /// - `attempts` = number of speculative rounds (one verify forward each).
    /// - `accepts` = total drafted tokens accepted across all rounds (excludes
    ///   the always-committed t1 token at the start of each round).
    /// - `avg_per_round` = accepts/attempts. Bounded by `(block_size - 1)` for
    ///   the drafter.
    /// - `per_draft_pct` (drafter only) = accepts / (attempts × (block_size-1)),
    ///   the per-draft acceptance probability comparable to vLLM's reported
    ///   "62% acceptance rate" metric.
    /// Which QSA arm served this request's attention (qwen4_exp only; no-op
    /// elsewhere). The engaged/declined meters are one-shots for the whole
    /// PROCESS, so they cannot say what ran on THIS request at THIS context
    /// length — a gather that quietly hands every call back to the dense mask
    /// (a full kv8 dequant per layer per forward) otherwise reads like a
    /// healthy run. Emitted next to `[spec-stats]`; resets the tally.
    pub fn logQsaArms(self: *Generator) void {
        // The tally rides THIS request's ForwardCtx — the one every forward
        // of this slot was handed. A Transformer-level counter printed the
        // union of every slot that decoded or prefilled since the last
        // finisher reset it.
        const c = self.ctx.qsa_arms.take() orelse return;
        const arm = c.majority() orelse return;
        log.info(
            "  [qsa-arms] qsa={s} calls={d} mask={d} decode={d} verify={d} prefill={d}\n",
            .{ @tagName(arm), c.total(), c.mask, c.decode, c.verify, c.prefill },
        );
    }

    pub fn logSpecStats(self: *const Generator) void {
        var table_buf: [256]u8 = undefined;
        var lookup_buf: [256]u8 = undefined;
        const table_bucket = self.xfm.round_cost.bucketOf(self.mtpKvLen());
        if (self.mtp != null and (self.mtp_attempted > 0 or self.mtp_lookup_rounds > 0)) {
            const avg_per_round: f64 = @as(f64, @floatFromInt(self.mtp_accepted_tokens)) /
                @as(f64, @floatFromInt(@max(self.mtp_attempted, 1)));
            // Depth varies per round under the EV controller — the honest
            // denominator is the DRAFTED count, not attempts x cap.
            const drafts_proposed: u64 = if (self.mtp_drafted_tokens > 0)
                self.mtp_drafted_tokens
            else
                self.mtp_attempted * @as(u64, self.mtp_depth);
            const per_draft_pct: f64 = if (drafts_proposed > 0)
                100.0 * @as(f64, @floatFromInt(self.mtp_accepted_tokens)) /
                    @as(f64, @floatFromInt(drafts_proposed))
            else
                0.0;
            const stalls = self.mtp_round_log.stalls();
            // The line is at the formatter's 32-argument limit.
            var stalls_buf: [32]u8 = undefined;
            const stalls_txt = std.fmt.bufPrint(&stalls_buf, "{d}/{d:.0}", .{ stalls.count, stalls.max_ms }) catch "?";
            log.info(
                "  [spec-stats] mode=mtp attempts={d} accepts={d} avg_per_round={d:.2} per_draft_pct={d:.1}% depth={d} drafted={d} ext_rounds={d} partial_rounds={d} runtime_disabled={s} reason={s} adaptive={s} serial_cell={d:.2} sync_ms={d:.2} round_ms={d:.2} two_ms_tok={d:.2} one_ms_tok={d:.2} verdict_round={d} trials={d} width_trials={d} table={s}:{s} table_drops=t{d}/c{d}/b{d}/i{d} serial_drops=t{d}/c{d}/b{d} lookup={d}/{d}/{d} stalls={s}\n",
                .{
                    self.mtp_attempted,
                    self.mtp_accepted_tokens,
                    avg_per_round,
                    per_draft_pct,
                    self.mtp_depth,
                    self.mtp_drafted_tokens,
                    self.mtp_ext_rounds,
                    self.partial_rounds,
                    if (self.spec_disabled_runtime) "true" else "false",
                    @tagName(self.spec_disable_reason),
                    @tagName(self.mtp_adaptive.arm),
                    self.xfm.round_cost.serialMsPerTok(table_bucket) orelse 0.0,
                    self.mtp_ev_sync_ms,
                    self.mtp_ev_round_ms,
                    if (self.mtp_regime.two_tok > 0) self.mtp_regime.two_ms / self.mtp_regime.two_tok else 0.0,
                    if (self.mtp_regime.one_tok > 0) self.mtp_regime.one_ms / self.mtp_regime.one_tok else 0.0,
                    self.mtp_regime.verdict_round,
                    self.mtp_regime.trials,
                    self.mtp_width_trial.trials,
                    round_cost.bucketName(self.xfm.round_cost.layout, table_bucket),
                    self.xfm.round_cost.formatBucket(table_bucket, &table_buf),
                    self.xfm.round_cost.dropped_transition,
                    self.xfm.round_cost.dropped_contended,
                    self.xfm.round_cost.dropped_bad,
                    self.xfm.round_cost.dropped_implausible,
                    self.xfm.round_cost.serial_dropped_transition,
                    self.xfm.round_cost.serial_dropped_contended,
                    self.xfm.round_cost.serial_dropped_bad,
                    self.mtp_lookup_rounds,
                    self.mtp_lookup_drafted,
                    self.mtp_lookup_accepted,
                    stalls_txt,
                },
            );
            if (self.mtp_lookup_rounds > 0) log.info("  [spec-stats] lookup_table={s}:{s}\n", .{
                round_cost.bucketName(self.xfm.round_cost.layout, table_bucket),
                self.xfm.round_cost.formatLookupBucket(table_bucket, &lookup_buf),
            });
            return;
        }
        if (self.dflash != null and self.dflash_attempted > 0) {
            const avg_per_round: f64 = @as(f64, @floatFromInt(self.dflash_accepted_tokens)) /
                @as(f64, @floatFromInt(self.dflash_attempted));
            const drafts_per_round: u32 = if (self.dflash_block_size >= 1) self.dflash_block_size - 1 else 0;
            const drafts_proposed: u64 = if (self.glm_dflash_native) self.glm_round.drafted else (if (self.dflash_drafts_verified > 0) self.dflash_drafts_verified else self.dflash_attempted * @as(u64, drafts_per_round));
            const per_draft_pct: f64 = if (drafts_proposed > 0)
                100.0 * @as(f64, @floatFromInt(self.dflash_accepted_tokens)) /
                    @as(f64, @floatFromInt(drafts_proposed))
            else
                0.0;
            // GLM's native path keeps its historical line: it has no yield gate and prints the old generic bar.
            var gate_buf: [16]u8 = undefined;
            var serial_buf: [32]u8 = undefined;
            const gate_txt = std.fmt.bufPrint(&gate_buf, "{d:.2}", .{if (self.glm_dflash_native) 1.80 else dflashBreakEvenAccepted(self.dflashMeanRows())}) catch "";
            const serial_txt = if (self.glm_dflash_native) "" else std.fmt.bufPrint(&serial_buf, " serial_ms={d:.2}", .{dflashSerialPrice(&self.xfm.round_cost, table_bucket) orelse 0.0}) catch "";
            log.info(
                "  [spec-stats] mode=dflash attempts={d} accepts={d} avg_per_round={d:.2} gate_min={s}{s} per_draft_pct={d:.1}% block_size={d} partial_rounds={d} runtime_disabled={s} table={s}:{s} table_drops=t{d}/c{d}/b{d}/i{d}\n",
                .{
                    self.dflash_attempted,
                    self.dflash_accepted_tokens,
                    avg_per_round,
                    gate_txt,
                    serial_txt,
                    per_draft_pct,
                    self.dflash_block_size,
                    self.partial_rounds,
                    if (self.spec_disabled_runtime) "true" else "false",
                    round_cost.bucketName(self.xfm.round_cost.layout, table_bucket),
                    self.xfm.round_cost.formatBucket(table_bucket, &table_buf),
                    self.xfm.round_cost.dropped_transition,
                    self.xfm.round_cost.dropped_contended,
                    self.xfm.round_cost.dropped_bad,
                    self.xfm.round_cost.dropped_implausible,
                },
            );
            if (self.glm_dflash_native) {
                const rounds: f64 = @floatFromInt(self.dflash_attempted);
                const r = self.glm_round;
                log.info("  [spec-stats] glm_round_ms draft={d:.2} verify={d:.2} replay={d:.2} commit={d:.2} lookup={d}/{d}\n", .{
                    @as(f64, @floatFromInt(r.draft_ns)) / rounds / 1e6,
                    @as(f64, @floatFromInt(r.verify_ns)) / rounds / 1e6,
                    @as(f64, @floatFromInt(r.replay_ns)) / rounds / 1e6,
                    @as(f64, @floatFromInt(r.commit_ns)) / rounds / 1e6,
                    self.glm_lookup.rounds,
                    self.glm_lookup.landed,
                });
            }
            return;
        }
    }

    /// Prefill the prompt and prepare for token-by-token generation.
    /// Backwards-compatible — prefer `initWithOptions` for new callers.
    pub fn init(
        io: std.Io,
        allocator: std.mem.Allocator,
        xfm: *Transformer,
        tok: *const Tokenizer,
        prompt_ids: []const u32,
        max_tokens: u32,
        sampling: SamplingParams,
        eos_token_ids: []const u32,
    ) !Generator {
        return initWithOptions(io, allocator, xfm, tok, prompt_ids, max_tokens, sampling, eos_token_ids, .{});
    }

    /// Receiver for SSM checkpoints salvaged out of a cancelled prefill.
    /// `initWithOptions` captures stride checkpoints into a local list that
    /// its errdefer frees — on `error.Cancelled` it instead moves them here
    /// so the scheduler can commit a partial-prefix cache entry. Ownership
    /// transfers either to the hot-cache entry (`commitWithSsm`) or back to
    /// the sink's `deinit`.
    pub const CancelledCheckpointSink = struct {
        /// Absolute prompt-token count forwarded into the KV when the chunk
        /// loop aborted (`ssm_checkpoint_pos_offset + pos`). This — NOT
        /// `cache.step`, which only advances when init completes — is the
        /// authoritative length for a cancelled-prefill commit.
        forwarded: usize = 0,
        checkpoints: []SSMCheckpoint = &.{},
        /// The allocator the checkpoints (and the slice itself) were
        /// allocated with — the same `sch.allocator` family the hot-cache
        /// entry later frees them with.
        alloc: ?std.mem.Allocator = null,

        pub fn deinit(self: *CancelledCheckpointSink) void {
            const a = self.alloc orelse return;
            for (self.checkpoints) |*cp| cp.deinit(a);
            if (self.checkpoints.len > 0) a.free(self.checkpoints);
            self.checkpoints = &.{};
            self.alloc = null;
        }
    };

    pub const InitOptions = struct {
        /// The machine-sized prefill chunk frozen at load
        /// (`ModelConfig.pinned_prefill_chunk`, resolved by
        /// `server.resolvePrefillChunk`). 0 = unpinned, keep the launch width.
        ///
        /// It arrives per REQUEST rather than off `xfm.config` because the
        /// Transformer holds a COPY of the config taken when it was built, and
        /// the pin is written to the registry's config after load — the copy
        /// never sees it. The scheduler reads `slot.model.config`, which IS the
        /// object the admission guard bills against, so bill and forward cannot
        /// disagree. (Live 2026-08-14: pinned 4096, prefilled at 8192.)
        pinned_prefill_chunk: usize = 0,
        /// A decode-share ceiling applied after explicit and environment chunk settings.
        decode_share_width_cap: usize = 0,
        /// Number of tokens per draft round. Default 4 (3 drafter steps +
        /// 1 t1 prepend → length-4 verify forward).
        drafter_block_size: u32 = 4,
        /// Enable the DFlash block-drafter. When set, `dflash` must be
        /// non-null and `bind()`-ed to `xfm`. Prefill captures the trunk's
        /// target_layer_ids outputs chunk-by-chunk into the per-request
        /// assistant context cache (`Generator.dflash_ctx`). Same
        /// lazy-pre-forward skip semantics as drafter/MTP.
        dflash_enabled: bool = false,
        /// Non-owning pointer to the loaded DFlash assistant.
        dflash: ?*DflashModel = null,
        /// Effective DFlash block size (loader-resolved: assistant config
        /// capped by the machine's verify lanes). 0 → the assistant config's value.
        dflash_block_size: u32 = 0,
        /// Enable the Qwen native MTP head. When set, `mtp` must be non-null
        /// and `bind()`-ed to `xfm`. Prefill builds the head's committed-
        /// history KV cache chunk-by-chunk (full-hidden capture) and the
        /// final-token forward captures `last_hidden`, exactly like the
        /// drafter path. Same lazy-pre-forward skip semantics as drafter.
        mtp_enabled: bool = false,
        /// Explicit verifier selected by the caller at construction.
        mtp_acceptance: mtp_acceptance.Mode = .exact,
        mtp_greedy_tail: bool = false,
        /// Non-owning pointer to the loaded MTP head.
        mtp: ?MtpHeadRef = null,
        /// The model's head before this request's opt-out (the qwen4 weights load with the trunk regardless).
        model_has_mtp: bool = false,
        /// Max tokens drafted per nextMtp round. 0 = auto (`--mtp-max-depth` not
        /// passed): resolved by `resolveMtpDepthCap` — MTP_ADAPTIVE_NAX_CAP
        /// for the measured M5 target+sidecar profile, otherwise
        /// MTP_ADAPTIVE_DEFAULT_CAP under the EV controller; DEFAULT_DEPTH in
        /// fixed mode. Explicit depths remain unchanged.
        mtp_depth: u32 = 0,
        /// When set, this slice (rather than `prompt_ids`) becomes the
        /// `prompt_ids_owned` source for the MTP prompt-lookup index. Used by the
        /// server's KV-cache-reuse path to forward only the trailing tokens
        /// while still giving the lookup the full prompt for matching.
        lookup_prompt: ?[]const u32 = null,
        /// Per-slot forward context (Phase 2 concurrent batching). When null,
        /// `initWithOptions` builds one from `xfm.defaultCtx()` so the legacy
        /// single-slot path is unchanged. Phase 2 callers pass a ForwardCtx
        /// whose `cache` / `moe_seq_offset` / `ssm_entries` / `vision_embeddings`
        /// point at the slot's own state. Stored by value on the Generator.
        ctx: ?ForwardCtx = null,
        /// Skip the lazy first-token pre-forward (regular path only). When set,
        /// init samples t1 synchronously and leaves `pending_logits` /
        /// `pending_token` empty — cache.step lands at exactly prompt_len with
        /// t1 NOT in cache. The first `next()` call's transition shim will
        /// sync-forward `[t1]` to seed pending_logits before the lazy chain.
        /// Used by the Phase 2 scheduler so a slot's cache state matches
        /// `forwardBatchedDecode`'s expectation (cache.step == prompt_len at
        /// the start of every decode tick). PLD / drafter paths already skip
        /// the lazy pre-forward unconditionally; this flag generalizes that
        /// behavior to the regular sampling path. Has no effect when
        /// `drafter_enabled` is true.
        skip_lazy_preforward: bool = false,
        /// Phase 1 (performance-plan): during prefill, capture an SSM
        /// checkpoint every `ssm_checkpoint_stride` tokens. 0 = disabled.
        /// Snapshots land in `Generator.ssm_checkpoints` for the caller to
        /// drain into the hot prefix cache via `takeSsmCheckpoints()`. Only
        /// effective when the model has hybrid layers (otherwise the
        /// `ssm_entries` slice is empty and snapshots become no-op stubs).
        /// Chunked prefill aligns chunk ends to stride positions so each
        /// snapshot reflects a coherent state.
        ssm_checkpoint_stride: u32 = 0,
        /// Cap on retained checkpoints. Past it the list is thinned span-preservingly
        /// (`transformer.ssmCheckpointDropIndex`): lowest and newest always survive. 0 = unlimited.
        ssm_checkpoint_max: u32 = 16,
        /// Phase 1: absolute position of the FIRST token in `prompt_ids`.
        /// On a cold prefill this is 0. On the warm path (where the
        /// scheduler restored some prefix and now forwards only the tail),
        /// callers pass `hot_matched` so the captured checkpoints stamp
        /// absolute positions usable by future warm-path lookups against
        /// the full prompt.
        ssm_checkpoint_pos_offset: usize = 0,
        /// Placeholder rows inside a restored prefix (prefix-cache hit on an
        /// image prompt): the vision splice starts here, not at row 0.
        vision_rows_before: usize = 0,
        /// A DFlash assistant context restored from the prefix cache, whose
        /// `absLen()` already equals `ssm_checkpoint_pos_offset`. Ownership
        /// transfers to the Generator. Null = start the assistant blind at
        /// the offset (a restore forwards no trunk layers, so a cold context
        /// is what a reused prefix would otherwise get).
        dflash_ctx_restored: ?dflash_mod.DflashCtx = null,
        /// An MTP committed-history cache restored from the prefix cache,
        /// whose `base + cache.step()` already equals
        /// `ssm_checkpoint_pos_offset`. Ownership transfers to the
        /// Generator. Null = start the history blind at the offset (same
        /// contract as `dflash_ctx_restored`).
        mtp_cache_restored: ?MtpRestored = null,
        /// Cooperative abort for abandoned requests: checked between prefill
        /// chunks. The scheduler passes `&slot.cancelled`, set by the conn
        /// thread when the client disconnects mid-prefill. When it flips,
        /// `initWithOptions` returns `error.Cancelled` instead of grinding
        /// out the rest of a multi-minute ghost prefill.
        cancel_flag: ?*const std.atomic.Value(bool) = null,
        /// When the chunk loop aborts on `cancel_flag`, the SSM checkpoints
        /// captured so far move into this sink instead of dying with the
        /// failed construction — the scheduler salvages a partial-prefix
        /// cache commit from them. Null keeps the old drop-everything
        /// behaviour.
        cancelled_checkpoint_sink: ?*CancelledCheckpointSink = null,
        /// Per-token logprobs count for this request (0 = disabled). Callers
        /// that set `Generator.logprobs_n` after init must ALSO pass it here:
        /// init's split-prefill final-token forward is a single-row dispatch,
        /// and the certified lm_head prune may only engage when the request
        /// consumes nothing but the argmax — a logprobs request reads the
        /// full logit row, which the pruned head does not produce.
        logprobs_n: u32 = 0,
        /// LIVE prefill progress, in tokens actually forwarded so far by THIS
        /// prefill. Bumped once per chunk (not per token), read off-thread by
        /// the metrics gauge sampler.
        ///
        /// Without it the panel is blind during prefill: `prompt_tokens_total`
        /// and `prefill_time_seconds` only advance at request COMPLETION, and
        /// generated tokens only accrue during decode — so a multi-minute
        /// prefill saturates the GPU while both tiles read 0 / "—". The
        /// scheduler resets it to 0 when the prefill ends.
        prefill_progress: ?*std.atomic.Value(u64) = null,
        /// Total tokens this prefill will forward (post-cache tail, same scale
        /// as `prefill_progress`), published once; null on the bypass engines.
        prefill_expected: ?*std.atomic.Value(u64) = null,

        /// Called once per completed prefill chunk boundary (chunk state
        /// evaluated, allocator cache cleared), except the final one. The
        /// scheduler runs decode ticks for the streams already decoding at
        /// this seam, so a long cold prefill stalls them for at most one
        /// chunk-forward instead of its whole duration. Null = the prefill
        /// runs atomically (pre-interleave behavior, and the
        /// SUSHI_PREFILL_INTERLEAVE=0 kill switch).
        interleave_hook: ?InterleaveHook = null,
        /// SSD-first write-through (qwen4_exp and GLM; see `WriteThroughHook`).
        write_through_hook: ?WriteThroughHook = null,
        /// The per-chunk prefill width, asked at every chunk boundary after that chunk's
        /// `mlx_clear_cache` and before the interleave tick. Null keeps the admitted width.
        /// Presence is NOT the arch gate (`serve` installs it process-wide); read `adaptive_chunk_width`.
        chunk_width_hook: ?ChunkWidthHook = null,
        /// Is the per-chunk adaptive width live for this request (`server.adaptivePrefillChunkEnabled`)?
        /// The only thing that may widen `tailMergeMaxFor`'s bound.
        adaptive_chunk_width: bool = false,
    };

    pub const InterleaveHook = struct {
        ctx: *anyopaque,
        call: *const fn (ctx: *anyopaque) void,
    };

    /// SSD-first write-through: called at every completed prefill chunk with the absolute KV
    /// position and the SSM checkpoints so far, so a cancelled prefill leaves a restorable prefix.
    /// The hook may `release` a checkpoint whose state it has put on SSD; its position stays listed.
    pub const WriteThroughHook = struct {
        ctx: *anyopaque,
        call: *const fn (ctx: *anyopaque, abs_kv_pos: usize, cps: []SSMCheckpoint) void,
    };

    /// Returns the width the next chunk should run at (`cur` unchanged = no change).
    pub const ChunkWidthHook = struct {
        ctx: *anyopaque,
        call: *const fn (ctx: *anyopaque, pos: usize, cur: u32, cap: u32, st: *AdaptiveWidthState) u32,
        /// Re-price a widen after the interleave tick has allocated for co-tenant slots. A
        /// step-down needs no confirmation: it is the safe direction.
        confirm: *const fn (ctx: *anyopaque, pos: usize, want: u32) bool,
    };

    /// Selects the source slice that `initWithOptions` will dupe into
    /// `prompt_ids_owned`. When `lookup_prompt` is non-null it wins (server
    /// cache-reuse path: full original prompt for the MTP lookup); otherwise the
    /// caller's `prompt_ids` is used (back-compat path).
    pub fn pickLookupPromptSource(prompt_ids: []const u32, lookup_prompt: ?[]const u32) []const u32 {
        return lookup_prompt orelse prompt_ids;
    }

    pub fn initWithOptions(
        io: std.Io,
        allocator: std.mem.Allocator,
        xfm: *Transformer,
        tok: *const Tokenizer,
        prompt_ids: []const u32,
        max_tokens: u32,
        sampling_in: SamplingParams,
        eos_token_ids: []const u32,
        options_in: InitOptions,
    ) !Generator {
        // The logits forward starts at the prompt's last token; there is none to index.
        if (prompt_ids.len == 0) return error.EmptyPrompt;
        // Reserved-token suppression rides the sampling params from HERE —
        // the one chokepoint every init site funnels through — so every
        // sampling path (serial, DFlash/MTP corrections, draft heads,
        // stochastic-verify filters, batched decode via `gen.sampling`)
        // inherits the model's mask without per-site wiring.
        var sampling = sampling_in;
        sampling.suppress_mask = xfm.suppress_mask;
        sampling.think_penalty.file_biases = xfm.logit_bias;
        const bias_vocab: usize = xfm.config.vocab_size;
        try sampling.think_penalty.prepare(allocator, bias_vocab, xfm.think_marker_mask, xfm.s);
        errdefer sampling.think_penalty.deinitPrepared();
        var options = options_in;
        const glm_dflash_native = xfm.glm5 != null and options.dflash_enabled and options.dflash != null and glmDflashEligible(sampling, options.logprobs_n);
        if (xfm.glm5 != null) {
            options.mtp_enabled = false;
            options.dflash_enabled = glm_dflash_native;
        }
        const s = xfm.s;
        // Per-slot ForwardCtx (Phase 2). Stored by value on the Generator;
        // callers either supply one (scheduler) or fall through to
        // `xfm.defaultCtx()` for the legacy single-slot path. We pass
        // `&ctx` to every forward call below; the cache/moe/ssm fields
        // mutate in-place through their pointers.
        var ctx: ForwardCtx = options.ctx orelse xfm.defaultCtx();

        // lm_head gate: the certified prune and MiMo's coarse shortlist serve
        // the ARGMAX, not the tail distribution, so they engage only when this
        // request consumes nothing else (`argmaxOnlyRequest`).
        ctx.argmax_only = argmaxOnlyRequest(sampling, options.logprobs_n);

        const ids_i32 = try allocator.alloc(i32, prompt_ids.len);
        defer allocator.free(ids_i32);
        for (prompt_ids, 0..) |id, i| {
            ids_i32[i] = @intCast(id);
        }

        // Clone the lookup prompt for the lifetime of the Generator. The MTP
        // lookup index needs `prompt + generated`, and the caller-owned
        // slice can be freed after init. When `options.lookup_prompt`
        // is set (server cache-reuse path), it carries the full original prompt
        // so the match coverage isn't gutted when only a trailing tail was
        // forwarded into the KV cache. Defaults to `prompt_ids` otherwise.
        // Allocated up front so init's errdefer paths don't have to track
        // partial state.
        const owned_src = pickLookupPromptSource(prompt_ids, options.lookup_prompt);
        const prompt_owned = try allocator.dupe(u32, owned_src);
        errdefer allocator.free(prompt_owned);

        // Split prefill: process first N-1 tokens (cache-only, skip lm_head eval),
        // then the last token (produces logits for sampling). This mirrors mlx-lm's
        // generate_step which avoids the expensive lm_head projection over the full
        // sequence length. For vocab_size=262144, skipping lm_head on N-1 tokens
        // avoids a [N-1, hidden] @ [hidden, 262144] matmul.
        //
        // Chunked prefill: large prompts are processed in PREFILL_CHUNK-sized pieces
        // to bound peak activation memory. Each chunk fills KV cache entries for its
        // positions, gets eval'd, and intermediates are freed before the next chunk.
        // Without chunking, Gemma-4 MoE's 2 MLPs × 4 stacked layers can spike to
        // ~20 GB of activations alone on a 50k-token prompt, causing Metal OOM.
        // Vision requests skip chunking since image token positions must be visible
        // in a single forward pass for spliceVisionEmbeddings to work correctly.
        // PREFILL_CHUNK comes from the module-level `prefill_chunk_override`
        // (set by --prefill-chunk in main.zig).
        //
        // Safety cap: on unfused head dims (>128 — every Gemma-4/Qwen3.5/3.6)
        // the composed SDPA materializes [heads, chunk, total_kv] scores per
        // layer, so the chunk shrinks with the prompt's FINAL KV length to keep
        // that one tensor bounded (boundedPrefillChunk). Warm-path restores
        // start at ssm_checkpoint_pos_offset, so the final KV length is that
        // offset plus everything we're about to forward.
        const total_ctx_for_chunk = options.ssm_checkpoint_pos_offset + prompt_ids.len;
        const PREFILL_CHUNK: usize = decodeShareCapped(requestPrefillChunk(
            xfm.config.prefillScoreHeadDim(),
            xfm.config.num_attention_heads,
            total_ctx_for_chunk,
            xfm.config.has_sliding_window,
            xfm.config.isMoe(),
            xfm.config.longCtxGated(),
            options.pinned_prefill_chunk,
        ), options.decode_share_width_cap);
        // Where cold prefill time goes (chunked-forward vs eval vs last-token-forward), at debug level.
        const trace_enabled = log.isDebug();
        var prefill_sw = io_util.Stopwatch.init(io);
        var chunked_ns: u64 = 0;
        var eval_ns: u64 = 0;
        var n_chunks: usize = 0;

        // Phase 1: SSM checkpointing during prefill. When enabled, the chunked
        // prefill loop forces a chunk boundary at every multiple of
        // `ssm_checkpoint_stride`, then snapshots `ctx.ssm_entries` after that
        // chunk evaluates. Snapshots accumulate in `Generator.ssm_checkpoints`
        // for the scheduler to drain in `commitSlotIfApplicable`. Plain-attn
        // models have an empty `ssm_entries` slice, so this becomes a no-op
        // even at stride > 0 — but we still bail early so we never allocate
        // empty checkpoints.
        var ssm_checkpoints: std.ArrayList(SSMCheckpoint) = std.ArrayList(SSMCheckpoint).empty;
        var ssm_cp_peak: u32 = 0;
        errdefer {
            for (ssm_checkpoints.items) |*cp| cp.deinit(allocator);
            ssm_checkpoints.deinit(allocator);
        }
        const has_vision = ctx.vision_embeddings != null;
        const glm_request = ctx.glm5_request;
        const want_ssm_cp = shouldCheckpointSsmPrefill(
            options.ssm_checkpoint_stride,
            (ctx.ssm_entries != null and ctx.ssm_entries.?.len > 0) or glm_request != null,
            has_vision,
        );
        // The snapshot backoff: a checkpoint AT the prompt end is
        // unreachable next turn.
        const want_state_cp = want_ssm_cp;
        // Coarsen the checkpoint stride so checkpointing never sub-divides the
        // prefill chunk, on ANY arch (see effectiveSsmCheckpointStride: fine
        // strides push every projection under prefillDqGemm's M>=2048 floor and
        // multiply per-chunk fixed costs — 17-25% cold-prefill tax measured on
        // dense AND MoE hybrids). Warm reuse keeps chunk-granularity snapshots
        // plus the always-on end-of-prompt snapshot (the append-growth case
        // llmprobe's cache-hit tests pin — verified green at coarse strides).
        // Coarsen against the UNCAPPED base chunk: the head_dim safety cap
        // above must not densify checkpoint spacing (16× more captures at
        // 255K ctx otherwise). nextChunkEnd already shortens a chunk to land
        // on stride boundaries, so a capped chunk stays compatible.
        const ssm_cp_stride: usize = if (!want_ssm_cp)
            0
        else if (glm_request != null)
            glm_checkpoint_stride
        else
            effectiveSsmCheckpointStride(@intCast(options.ssm_checkpoint_stride), @max(PREFILL_CHUNK, prefill_chunk_override));
        // Absolute KV position of `prompt_ids[0]`. Warm-path callers (the
        // scheduler after restoring a checkpoint) pass the matched prefix
        // length so the snapshots stamp positions valid in the full original
        // sequence, not relative offsets inside the tail-only prefill.
        const ssm_cp_offset: usize = options.ssm_checkpoint_pos_offset;

        // Qwen native MTP: build the head's committed-history KV cache during
        // prefill. Entry j pairs (trunk hidden at prompt position j, token at
        // j+1); the (hidden[last], t1) pair is appended by the first nextMtp
        // round. On KV-prefix reuse the history covers only the freshly
        // forwarded tail — RoPE offsets are cache-relative, so a late-starting
        // history is self-consistent (sliding-window history semantics).
        const mtp_active = options.mtp_enabled and options.mtp != null;
        const AcceptRoute = struct {
            graph: MtpGraphFn,
            prefix: mtp_acceptance.PrefixFn,
            param: f32,
        };
        const accept_route: AcceptRoute = switch (options.mtp_acceptance) {
            .exact => .{ .graph = mtpBatchedExactGraph, .prefix = mtp_acceptance.exactPrefix, .param = 0 },
            .typical => |cfg| .{ .graph = mtpBatchedTypicalGraph, .prefix = mtp_acceptance.typicalPrefix, .param = cfg.delta },
            .tokenv3 => |alpha| .{ .graph = mtpBatchedTokenV3Graph, .prefix = mtp_acceptance.tokenV3Prefix, .param = alpha },
        };
        if (mtp_active and std.meta.activeTag(options.mtp_acceptance) != .exact) {
            switch (options.mtp_acceptance) {
                .typical => |cfg| log.info("  [mtp-accept] typical delta={d:.3} eps={d:.3} NOT distribution-exact\n", .{ cfg.delta, cfg.eps }),
                .tokenv3 => |alpha| log.info("  [mtp-accept] TokenV3 alpha={d:.3} NOT distribution-exact\n", .{alpha}),
                .exact => unreachable,
            }
        }
        var mtp_cache: ?MtpCacheRef = null;
        var mtp_position_base: usize = ssm_cp_offset;
        var mtp_history_started = false;
        if (mtp_active) {
            if (options.mtp_cache_restored) |restored| {
                // A restored history already covers [base, ssm_cp_offset);
                // the tail prefill APPENDS to it (RoPE offsets are
                // cache-relative, so continuation is self-consistent).
                std.debug.assert(restored.base + restored.cache.step() == ssm_cp_offset);
                mtp_cache = restored.cache;
                mtp_position_base = restored.base;
                mtp_history_started = true;
            } else {
                mtp_cache = try options.mtp.?.makeCache(allocator);
            }
        }
        errdefer if (mtp_cache) |*mc| mc.deinit();

        // DFlash: build the assistant's context cache during prefill — every
        // chunk's target_layer_ids captures project through the encoder and
        // append immediately (the chunk's activation graph then frees with
        // the chunk eval). On KV-prefix reuse the context covers only the
        // freshly forwarded tail (`base_pos = ssm_cp_offset`) — sliding-
        // window semantics, same rule as the MTP history.
        const dflash_active = options.dflash_enabled and options.dflash != null;
        var dflash_ctx: ?dflash_mod.DflashCtx = if (!dflash_active)
            null
        else if (options.dflash_ctx_restored) |restored| blk: {
            std.debug.assert(restored.absLen() == ssm_cp_offset);
            break :blk restored;
        } else try dflash_mod.DflashCtx.init(allocator, options.dflash.?, ssm_cp_offset);
        errdefer if (dflash_ctx) |*dc| dc.deinit();
        var dfl_out_buf: []mlx.mlx_array = &.{};
        defer if (dfl_out_buf.len > 0) allocator.free(dfl_out_buf);
        var dfl_cl: transformer_mod.CaptureLayers = undefined;
        if (dflash_active) {
            dfl_out_buf = try allocator.alloc(mlx.mlx_array, options.dflash.?.config.target_layer_ids.len);
            dfl_cl = .{ .ids = options.dflash.?.config.target_layer_ids, .out = dfl_out_buf, .hc_mean = xfm.config.hc_count };
        }

        // Start of the final (logits) forward's token span. Without SSM
        // checkpointing this is the last prompt token (the classic 1-token
        // logits forward); with checkpointing the chunk loop stops
        // SSM_SNAPSHOT_BACKOFF tokens early so the always-on snapshot lands
        // where the next turn's prefix match can actually reach it (see
        // ssmSnapshotBackoff), and the final forward covers the held-back
        // tail + the last token in the same weight sweep.
        var final_start: usize = prompt_ids.len - 1;
        // Chunked vision prefill (issue #197): placeholder rows consumed by
        // completed chunks, fed to ctx.vision_splice_offset per forward
        // (chunk loop AND final-span forward). Stays 0 on the kill-switch arm
        // so that path is byte-identical to the old whole-prompt behavior.
        const vision_chunked = has_vision and visionChunkedPrefillEnabled();
        var vision_rows_consumed: usize = options.vision_rows_before;
        var adapt_state: AdaptiveWidthState = .{
            .width_min = @intCast(PREFILL_CHUNK),
            .width_max = @intCast(PREFILL_CHUNK),
        };

        if (prompt_ids.len > 1) {
            const prefix_len = prompt_ids.len - 1;
            const snapshot_backoff = if (glm_request != null and want_state_cp)
                glmCaptureSchedule(glm_checkpoint_stride, options.ssm_checkpoint_max, ssm_cp_offset, prompt_ids.len).backoff
            else
                ssmSnapshotBackoff(want_state_cp, prefix_len, ssm_cp_offset > 0);
            const loop_end = prefix_len - snapshot_backoff;
            final_start = loop_end;
            // Vision prompts chunk like text (issue #197) — the splice offset
            // below keeps the row scatter chunk-exact. Kill switch restores
            // the whole-prompt forward.
            const default_chunk = if (has_vision and !vision_chunked) loop_end else PREFILL_CHUNK;
            // Per-chunk adaptive width: the first chunk runs the arch's widest width, and every
            // boundary re-asks the same estimator. `cap_adapt` is the widest this arch forwards
            // for this prompt, never wider than `ssm_cp_stride`.
            const adapt_chunked = !(has_vision and !vision_chunked);
            // The scaled tail-merge bound reads the arch predicate, never `chunk_width_hook != null`
            // (installed process-wide).
            const width_is_adaptive = adapt_chunked and options.adaptive_chunk_width;
            const cap_adapt: u32 = if (!adapt_chunked) 0 else @intCast(@min(effectivePrefillChunk(
                xfm.config.prefillScoreHeadDim(),
                xfm.config.num_attention_heads,
                total_ctx_for_chunk,
                xfm.config.has_sliding_window,
                xfm.config.isMoe(),
                xfm.config.longCtxGated(),
                0,
            ), @as(usize, xfm.config.prefillStartWidth())));
            adapt_state.width_min = @intCast(default_chunk);
            adapt_state.width_max = @intCast(default_chunk);
            var cur_chunk: usize = default_chunk;

            // Reserve the whole request's cache capacity before the first chunk writes: a grow
            // is not in place, so a long prefill's peak carried a second copy of everything so
            // far (~7.75 GB at 458k on qwen4_exp, #353). Same number the admission guard billed
            // (`KVCache.reservedTokens`), in ABSOLUTE positions: `prompt_ids` is the tail after
            // a prefix-cache hit. `cp_thin` names this site's previous retention policy.
            const cp_thin: transformer_mod.ThinPolicy =
                if (xfm.config.longCtxGated() or glm_request != null) .min_span_recency else .oldest;

            const reserved_tokens = try reserveRequestCapacity(ctx.cache, &xfm.config, total_ctx_for_chunk, max_tokens, default_chunk, s);
            // The arch's own per-request buffers reserve at the same length.
            if (mtp_cache) |*mc| mc.activate();
            transformer_mod.reserveQsaHistoryWithHead(
                ctx.ssm_entries,
                if (xfm.qwen4_mtp) |*m| &m.entry else null,
                @intCast(reserved_tokens),
            );
            // Admission billed the request's top at `default_chunk`; the chunks before it start at
            // the widest width and step down only where one no longer fits beside the reserved KV.
            const start_chunk = if (width_is_adaptive) decodeShareCapped(cap_adapt, options.decode_share_width_cap) else 0;
            if (start_chunk > cur_chunk) {
                cur_chunk = start_chunk;
                adapt_state.width_min = @intCast(start_chunk);
                adapt_state.width_max = @intCast(start_chunk);
                if (options.chunk_width_hook) |hk| {
                    const w = hk.call(hk.ctx, ssm_cp_offset, @intCast(cur_chunk), cap_adapt, &adapt_state);
                    if (w != 0 and w < cur_chunk) commitAdaptiveWidth(&cur_chunk, &adapt_state, w);
                }
            }

            var pos: usize = 0;
            if (options.prefill_expected) |e| e.store(@intCast(loop_end), .monotonic);
            while (pos < loop_end) {
                // Abandoned-request abort: the client disconnected and the
                // conn thread flagged the slot. Bail before the next chunk —
                // the KV built so far is freed with the slot.
                if (options.cancel_flag) |cf| {
                    if (cf.load(.acquire)) {
                        // Salvage the chunk loop's progress: `forwarded` is
                        // the authoritative length (cache.step only advances
                        // when init completes), the stride checkpoints ride
                        // for hybrid restore (they die with the failed
                        // construction otherwise). toOwnedSlice empties the
                        // local list so the errdefer frees nothing (on OOM
                        // the items stay and the errdefer cleans up — the
                        // sink keeps the length, the commit declines the
                        // checkpoints only).
                        if (options.cancelled_checkpoint_sink) |sink| {
                            sink.forwarded = ssm_cp_offset + pos;
                            if (ssm_checkpoints.items.len > 0) {
                                if (ctx.ssm_entries) |ents| {
                                    transformer_mod.handoffQsaHistoryToLatest(ssm_checkpoints.items, ents, xfm.s) catch {};
                                }
                                if (ssm_checkpoints.toOwnedSlice(allocator)) |owned| {
                                    sink.checkpoints = owned;
                                    sink.alloc = allocator;
                                } else |_| {}
                            }
                        }
                        return error.Cancelled;
                    }
                }
                // An MLX failure in a previous chunk latched instead of killing the process (#353);
                // abandon this request, keep the server. No checkpoint salvage: the state failed.
                try mlx.checkError();
                // Pick this chunk's end. Normal path: hit the configured chunk
                // size. Phase 1 path: if a checkpoint stride boundary lands
                // inside the would-be chunk, shrink the chunk so it ends
                // exactly on that boundary. That gives us an snapshot-point
                // every `stride` tokens without changing the model's seen
                // input — the forward result is identical to the unchunked
                // version because attention is causal and SSM/conv update
                // chunk-locally. Boundary alignment is in ABSOLUTE position
                // (pos + offset), so the saved snapshot list is correct for
                // the full prompt, not the truncated tail.
                const end = if (glm_request != null)
                    glmChunkEnd(pos, loop_end, cur_chunk, ssm_cp_stride, ssm_cp_offset, width_is_adaptive)
                else
                    nextChunkEnd(pos, loop_end, cur_chunk, want_ssm_cp, ssm_cp_stride, ssm_cp_offset, width_is_adaptive);
                if (has_vision) ctx.vision_splice_offset = vision_rows_consumed;
                const chunk_len: c_int = @intCast(end - pos);
                const chunk_shape = [_]c_int{ 1, chunk_len };
                const chunk_input = mlx.mlx_array_new_data(@ptrCast(&ids_i32[pos]), &chunk_shape, 2, .int32);
                defer _ = mlx.mlx_array_free(chunk_input);

                const chunk_start_ns = if (trace_enabled) prefill_sw.read() else 0;
                var chunk_hidden_all = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(chunk_hidden_all);
                if (dflash_active) {
                    for (dfl_out_buf) |*a| a.* = mlx.mlx_array_new();
                    ctx.capture_layers = &dfl_cl;
                }
                errdefer if (dflash_active) {
                    for (dfl_out_buf) |a| _ = mlx.mlx_array_free(a);
                };
                const chunk_logits = if (mtp_active) blk: {
                    var last_unused = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(last_unused);
                    break :blk try xfm.forwardWithCaptureAll(&ctx, chunk_input, &last_unused, &chunk_hidden_all);
                } else try xfm.forwardWith(&ctx, chunk_input);
                _ = mlx.mlx_array_free(chunk_logits);
                if (dflash_active) {
                    ctx.capture_layers = null;
                    try dflash_mod.appendContext(options.dflash.?, &dflash_ctx.?, dfl_out_buf, ssm_cp_offset + pos);
                    for (dfl_out_buf) |*a| {
                        _ = mlx.mlx_array_free(a.*);
                        a.* = .{ .ctx = null };
                    }
                }
                if (trace_enabled) {
                    const dt = prefill_sw.read() - chunk_start_ns;
                    chunked_ns += dt;
                    log.debug(
                        "  [prefill-trace] chunk pos={d} end={d} width={d} ms={d}\n",
                        .{ pos, end, end - pos, dt / std.time.ns_per_ms },
                    );
                }

                // MTP history for this chunk: hiddens [pos, end) pair with
                // tokens [pos+1, end+1) — prompt_ids[end] always exists since
                // the chunk loop spans [0, prefix_len) and prompt_ids has
                // prefix_len + 1 entries.
                if (mtp_active) {
                    if (!mtp_history_started) {
                        std.debug.assert(mtp_cache.?.step() == 0);
                        mtp_position_base = ssm_cp_offset + pos;
                        mtp_history_started = true;
                    }
                    const mtp_mrope_ctx: ?mtp_mod.MropeContext = if (ctx.mrope_pos) |positions| .{
                        .pos = positions,
                        .total = ctx.mrope_total,
                        .delta = ctx.mrope_delta,
                        .base = mtp_position_base,
                    } else null;
                    try options.mtp.?.appendHistory(
                        xfm,
                        &mtp_cache.?,
                        prompt_ids[pos + 1 .. end + 1],
                        chunk_hidden_all,
                        @intCast(mtp_cache.?.step()),
                        mtp_mrope_ctx,
                        allocator,
                    );
                }

                // Eval KV cache — materializes this chunk's K/V, frees activation graph
                const eval_start_ns = if (trace_enabled) prefill_sw.read() else 0;
                {
                    const eval_vec = mlx.mlx_vector_array_new();
                    defer _ = mlx.mlx_vector_array_free(eval_vec);
                    for (ctx.cache.entries) |*entry| {
                        if (!entry.initialized) continue;
                        _ = mlx.mlx_vector_array_append_value(eval_vec, entry.keys);
                        _ = mlx.mlx_vector_array_append_value(eval_vec, entry.values);
                    }
                    if (chunk_hidden_all.ctx != null) {
                        _ = mlx.mlx_vector_array_append_value(eval_vec, chunk_hidden_all);
                    }
                    // Materialize this chunk's MTP history entries alongside
                    // the trunk KV so the chunk's activation graph (incl. the
                    // full-hidden capture) can be freed before the next chunk.
                    // MiMo's last chunk leaves its heads' catch-up to the first draft chain,
                    // which runs after the scheduler streams t1.
                    if (mtp_cache) |*mc| if (mc.* != .mimo or end < loop_end) mc.appendEvalArrays(eval_vec);
                    // Same discipline for the DFlash context appended above.
                    if (dflash_ctx) |*dc| dc.appendEvalArrays(eval_vec);
                    // Phase 1: also force SSM state to materialize so any
                    // snapshot below holds a concrete tensor, not a lazy node
                    // that would re-execute the prefill graph if anyone reads
                    // from it later. Unconditional on EVERY chunk (not just
                    // stride-aligned ones): a tail-merged final chunk ends
                    // off-boundary, and the always-on end-of-prompt snapshot
                    // then hits exactly that re-execution — capturing an
                    // un-evaluated state re-ran the whole last chunk's GDN
                    // scan (measured −4% on an 8K Qwen3.6-27B prefill). The
                    // states are ~KB-to-MB scale; evaluating them alongside
                    // the KV costs nothing measurable.
                    if (want_ssm_cp) if (ctx.ssm_entries) |entries| {
                        for (entries) |*ssm| {
                            if (!ssm.initialized) continue;
                            if (ssm.conv_state.ctx != null) {
                                _ = mlx.mlx_vector_array_append_value(eval_vec, ssm.conv_state);
                            }
                            if (ssm.ssm_state.ctx != null) {
                                _ = mlx.mlx_vector_array_append_value(eval_vec, ssm.ssm_state);
                            }
                        }
                    };
                    _ = mlx.mlx_eval(eval_vec);
                }
                // Every chunk: handing the pool to the next chunk was slower and larger (glm5-perf).
                _ = mlx.mlx_clear_cache();
                if (trace_enabled) {
                    const eval_dt = prefill_sw.read() - eval_start_ns;
                    eval_ns += eval_dt;
                    var mem_buf: [256]u8 = undefined;
                    const line = formatPrefillMemLine(&mem_buf, pos, end, eval_dt / std.time.ns_per_ms, mlxPrefillMem());
                    log.debug("{s}", .{line});
                }

                // This chunk's latch, read before anything persists it, snapshots it or yields
                // the thread: Metal returns zeros before it aborts, so a later read let the
                // write-through index garbage and a co-tenant decode eat the latch.
                try mlx.checkError();

                // Phase 1: snapshot SSM state at stride-aligned boundaries.
                // We snapshot AFTER the eval above so the underlying buffers
                // are realized; the snapshot is just a refcount-share of the
                // already-resident state.
                const abs_end_for_cp2 = end + ssm_cp_offset;
                if (want_ssm_cp and ssm_cp_stride > 0 and abs_end_for_cp2 % ssm_cp_stride == 0 and restorableCheckpoint(glm_request, abs_end_for_cp2)) {
                    const cp = try capturePrefillCheckpoint(allocator, xfm, &ctx, glm_request, abs_end_for_cp2);
                    ssm_checkpoints.append(allocator, cp) catch |e| {
                        var doomed = cp;
                        doomed.deinit(allocator);
                        return e;
                    };
                    ssm_cp_peak = @max(ssm_cp_peak, @as(u32, @intCast(ssm_checkpoints.items.len)));
                    // The head's own QSA leftover at this position: its ring is 32 rows and
                    // the clamp back to this checkpoint comes a whole generated tail later.
                    if (mtp_active) {
                        mtp_cache.?.activate();
                    }
                    if (mtp_active) xfm.qwen4MtpMarkQsaLeftover(abs_end_for_cp2 -| mtp_position_base) catch |e| {
                        log.debug("[qwen4] MTP head leftover mark failed: {s}\n", .{@errorName(e)});
                    };
                    // Thin the interior, never the oldest (#330): drop-oldest survivors covered only
                    // the last `max * stride` tokens and left no affordable trim point.
                    if (options.ssm_checkpoint_max > 0 and
                        ssm_checkpoints.items.len > options.ssm_checkpoint_max)
                    {
                        var dropped = ssm_checkpoints.orderedRemove(
                            checkpointDropIndex(ssm_checkpoints.items, cp_thin),
                        );
                        dropped.deinit(allocator);
                    }
                }

                if (vision_chunked) {
                    vision_rows_consumed += countSpliceRows(
                        ids_i32[pos..end],
                        xfm.config.image_token_id,
                        xfm.config.video_token_id,
                    );
                }
                pos = end;
                n_chunks += 1;
                // Publish progress once per chunk — same cadence discipline as
                // `inflight_generated_tokens` (once per decode tick), never per token.
                if (options.prefill_progress) |p| p.store(@intCast(pos), .monotonic);
                // Persist this chunk now: the KV bytes for [0, pos) are final.
                if (options.write_through_hook) |wt| {
                    wt.call(wt.ctx, pos + ssm_cp_offset, ssm_checkpoints.items);
                }
                // The next chunk's width, re-priced at this boundary: after the `mlx_clear_cache`
                // (steady state) and before the interleave tick (a co-tenant's decode is not this
                // prefill's pressure). A step-down commits here; a widen is re-priced after the tick.
                var pending_widen: u32 = 0;
                var widen_confirm: ?*const fn (*anyopaque, usize, u32) bool = null;
                var widen_ctx: ?*anyopaque = null;
                if (adapt_chunked and pos < loop_end) {
                    if (options.chunk_width_hook) |hk| {
                        const next_w = hk.call(hk.ctx, ssm_cp_offset + pos, @intCast(cur_chunk), cap_adapt, &adapt_state);
                        if (next_w != 0 and next_w != cur_chunk) {
                            if (next_w < cur_chunk) {
                                commitAdaptiveWidth(&cur_chunk, &adapt_state, next_w);
                            } else {
                                pending_widen = next_w;
                                widen_confirm = hk.confirm;
                                widen_ctx = hk.ctx;
                            }
                        }
                    }
                }
                // Yield to the scheduler between chunks — never after the
                // last (the post-prefill decode tick covers that boundary).
                if (pos < loop_end) {
                    if (options.interleave_hook) |hk| hk.call(hk.ctx);
                }
                // The tick allocated for co-tenant slots; re-price the widen against what is left.
                if (pending_widen != 0) {
                    if (widen_confirm) |cf| {
                        if (cf(widen_ctx.?, ssm_cp_offset + pos, pending_widen)) {
                            commitAdaptiveWidth(&cur_chunk, &adapt_state, pending_widen);
                        } else {
                            adapt_state.supporting = 0;
                        }
                    }
                }
            }
            try mlx.checkError();
            // One line per request when the width moved.
            if (adapt_state.transitions > 0) {
                log.info("[prefill] adaptive: {d} chunks, width {d}..{d}, {d} change(s)\n", .{
                    n_chunks,
                    adapt_state.width_min,
                    adapt_state.width_max,
                    adapt_state.transitions,
                });
            }

            // Phase 1: always-on snapshot at the post-prefill position
            // (= prefix_len, i.e., prompt_ids.len - 1). The stride loop
            // captures snapshots at [stride, 2*stride, ...]; this final
            // capture covers the most common warm-path case where the next
            // turn's prompt fully matches turn-1's prompt and matched lands
            // at prompt_ids.len. Without this, a stride=256 setup with a
            // 750-token prompt could only restore at position 512 (losing
            // ~234 tokens of potential reuse to the next stride boundary).
            // With it, the cache restores to position 749 (~99% of the
            // prompt) and only the last token + new tail re-forwards.
            // Skipped on `prompt_ids.len == 1` (no prefill chunks ran).
            if (want_ssm_cp and loop_end > 0) {
                // The snapshot sits at the chunk loop's end — snapshot_backoff
                // tokens BEFORE the prompt end, where the next turn's prefix
                // match can actually reach it (the template's generation
                // suffix renders differently in history, so a match always
                // falls a few tokens short of the full prompt).
                const final_abs = loop_end + ssm_cp_offset;
                // Skip if we already captured at this exact position (the
                // chunked loop would have done so when loop_end happens
                // to be a stride multiple).
                const already_have = ssm_checkpoints.items.len > 0 and
                    ssm_checkpoints.items[ssm_checkpoints.items.len - 1].pos == final_abs;
                if (!already_have and restorableCheckpoint(glm_request, final_abs)) {
                    // SSM state is already materialized — the chunked loop
                    // evaluated it at every chunk boundary. The final chunk
                    // may have been a stride-aligned one (already evaluated)
                    // or a partial tail (also evaluated). The snapshot is a
                    // cheap refcount-share.
                    const cp = try capturePrefillCheckpoint(allocator, xfm, &ctx, glm_request, final_abs);
                    ssm_checkpoints.append(allocator, cp) catch |e| {
                        var doomed = cp;
                        doomed.deinit(allocator);
                        return e;
                    };
                    ssm_cp_peak = @max(ssm_cp_peak, @as(u32, @intCast(ssm_checkpoints.items.len)));
                    // The head's own QSA leftover at this position: its ring is 32 rows and
                    // the clamp back to this checkpoint comes a whole generated tail later.
                    if (mtp_active) {
                        mtp_cache.?.activate();
                    }
                    if (mtp_active) xfm.qwen4MtpMarkQsaLeftover(final_abs -| mtp_position_base) catch |e| {
                        log.debug("[qwen4] MTP head leftover mark failed: {s}\n", .{@errorName(e)});
                    };
                    if (options.ssm_checkpoint_max > 0 and
                        ssm_checkpoints.items.len > options.ssm_checkpoint_max)
                    {
                        var dropped = ssm_checkpoints.orderedRemove(
                            checkpointDropIndex(ssm_checkpoints.items, cp_thin),
                        );
                        dropped.deinit(allocator);
                    }
                }
            }
        }

        // Chunked vision: the final-span forward continues the row scatter
        // where the chunk loop left it (an image ending in the final span
        // otherwise re-splices from row 0). One-shot engagement line — the
        // silent-no-op class; tests grep for it per arm.
        if (has_vision) {
            ctx.vision_splice_offset = vision_rows_consumed;
            if (vision_chunked and n_chunks > 1) log.debug("[vision] chunked prefill: {d} chunks, {d} placeholder rows consumed\n", .{ n_chunks, vision_rows_consumed });
        }

        // Process the final span — the last token, plus (under SSM
        // checkpointing) the snapshot-backoff tail held back from the chunk
        // loop. One forward, one weight sweep, logits sliced to the last
        // position so every consumer sees the classic [1, 1, V] shape.
        const tail_len: usize = prompt_ids.len - final_start;
        const last_shape = [_]c_int{ 1, @as(c_int, @intCast(tail_len)) };
        const last_input = mlx.mlx_array_new_data(@ptrCast(&ids_i32[final_start]), &last_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(last_input);

        // The MTP head needs the post-final-norm hidden as its first-step h_prev.
        // `forwardWithCapture` captures the LAST position regardless of span
        // length. When the span holds backed-off tokens the head's history must
        // also cover them (a hole right before the generation point is
        // acceptance-critical), so capture ALL positions and append.
        const need_capture = mtp_active;
        var captured_hidden: mlx.mlx_array = mlx.mlx_array_new();
        var has_captured_hidden = false;
        const last_start_ns = if (trace_enabled) prefill_sw.read() else 0;
        const tail_mtp_capture = mtp_active and tail_len > 1;
        var tail_hidden_all = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(tail_hidden_all);
        // DFlash context for the tail span (rides ctx into all three forward
        // arms below; the forwardWithCapture* wrappers only override
        // capture_hidden/_all).
        if (dflash_active) {
            for (dfl_out_buf) |*a| a.* = mlx.mlx_array_new();
            ctx.capture_layers = &dfl_cl;
        }
        errdefer if (dflash_active) {
            for (dfl_out_buf) |a| _ = mlx.mlx_array_free(a);
        };
        const raw_logits = if (tail_mtp_capture) blk: {
            has_captured_hidden = true;
            break :blk try xfm.forwardWithCaptureAll(&ctx, last_input, &captured_hidden, &tail_hidden_all);
        } else if (need_capture) blk: {
            has_captured_hidden = true;
            break :blk try xfm.forwardWithCapture(&ctx, last_input, &captured_hidden);
        } else try xfm.forwardWith(&ctx, last_input);
        // History entries for the held-back span: (hidden[j], token[j+1]) for
        // j in [final_start, prefix_len) — same pairing as the chunk loop.
        // The last row's pair (hidden[last], t1) is appended by the first
        // nextMtp round, exactly as before.
        if (tail_mtp_capture) {
            if (!mtp_history_started) {
                std.debug.assert(mtp_cache.?.step() == 0);
                mtp_position_base = ssm_cp_offset + final_start;
                mtp_history_started = true;
            }
            var tail_hist_hidden = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(tail_hist_hidden);
            const all_shape = mlx.getShape(tail_hidden_all);
            const start = [_]c_int{ 0, 0, 0 };
            const stop = [_]c_int{ all_shape[0], @intCast(tail_len - 1), all_shape[2] };
            const strides = [_]c_int{ 1, 1, 1 };
            try mlx.check(mlx.mlx_slice(&tail_hist_hidden, tail_hidden_all, &start, 3, &stop, 3, &strides, 3, xfm.s));
            const tail_mrope_ctx: ?mtp_mod.MropeContext = if (ctx.mrope_pos) |positions| .{
                .pos = positions,
                .total = ctx.mrope_total,
                .delta = ctx.mrope_delta,
                .base = mtp_position_base,
            } else null;
            try options.mtp.?.appendHistory(
                xfm,
                &mtp_cache.?,
                prompt_ids[final_start + 1 .. prompt_ids.len],
                tail_hist_hidden,
                @intCast(mtp_cache.?.step()),
                tail_mrope_ctx,
                allocator,
            );
        }
        if (dflash_active) {
            ctx.capture_layers = null;
            try dflash_mod.appendContext(options.dflash.?, &dflash_ctx.?, dfl_out_buf, ssm_cp_offset + final_start);
            for (dfl_out_buf) |*a| {
                _ = mlx.mlx_array_free(a.*);
                a.* = .{ .ctx = null };
            }
        }
        // Slice to the last position when the span is longer than one token,
        // so downstream sampling/grammar paths see the classic shape.
        const logits = if (tail_len == 1) raw_logits else blk: {
            defer _ = mlx.mlx_array_free(raw_logits);
            const lshape = mlx.getShape(raw_logits);
            var sliced = mlx.mlx_array_new();
            const start = [_]c_int{ 0, lshape[1] - 1, 0 };
            const stop = [_]c_int{ lshape[0], lshape[1], lshape[2] };
            const strides = [_]c_int{ 1, 1, 1 };
            try mlx.check(mlx.mlx_slice(&sliced, raw_logits, &start, 3, &stop, 3, &strides, 3, xfm.s));
            break :blk sliced;
        };
        if (trace_enabled) {
            const last_ns = prefill_sw.read() - last_start_ns;
            const total_ns = prefill_sw.read();
            const ms = std.time.ns_per_ms;
            log.debug(
                "  [prefill-trace] tokens={d} chunks={d} chunk_size={d} chunk_widths={d}..{d} chunked={d}ms eval={d}ms last_token={d}ms total={d}ms{s}\n",
                .{
                    prompt_ids.len,
                    n_chunks,
                    PREFILL_CHUNK,
                    adapt_state.width_min,
                    adapt_state.width_max,
                    chunked_ns / ms,
                    eval_ns / ms,
                    last_ns / ms,
                    total_ns / ms,
                    if (need_capture) " [capture-hidden]" else "",
                },
            );
        }
        errdefer if (has_captured_hidden) {
            _ = mlx.mlx_array_free(captured_hidden);
        };

        // Attach the SSM-checkpoint buffer to whichever Generator variant
        // we're about to return. Clears the local list so the errdefer above
        // doesn't double-free. All four init paths below call this once
        // before returning their Generator.
        const attachCp = struct {
            fn f(g: *Generator, list: *std.ArrayList(SSMCheckpoint), a: std.mem.Allocator, peak: u32) void {
                g.ssm_checkpoints = list.*;
                g.ssm_cp_peak = peak;
                g.ssm_checkpoint_alloc = a;
                list.* = std.ArrayList(SSMCheckpoint).empty;
            }
        }.f;

        // Constrained generation skips the lazy first-sample fast path: we cannot
        // sample the first token until we have applied the grammar mask, and we
        // cannot pipeline because grammar advancement depends on the realized id.
        if (sampling.constraint != null) {
            // Grammar-constrained requests never speculate; release the MTP
            // history cache / DFlash context if dispatch enabled them anyway.
            if (mtp_cache) |*mc| {
                mc.deinit();
                mtp_cache = null;
            }
            if (dflash_ctx) |*dc| {
                dc.deinit();
                dflash_ctx = null;
            }
            var gen = Generator{
                .xfm = xfm,
                .model_has_mtp = options.model_has_mtp,
                .ctx = ctx,
                .tok = tok,
                .next_token_id = 0,
                .step = 0,
                .max_tokens = max_tokens,
                .sampling = sampling,
                .prompt_tokens = @intCast(prompt_ids.len),
                .completion_tokens = 0,
                .finish_reason = "length",
                .done = false,
                .eos_token_ids = eos_token_ids,
                .generated_ids = std.ArrayList(u32).empty,
                .timeout_ns = 0,
                .timer = io_util.Stopwatch.init(io),
                .last_hidden = if (has_captured_hidden) captured_hidden else mlx.mlx_array_new(),
                .has_last_hidden = has_captured_hidden,
                .prompt_ids_owned = prompt_owned,
                .prompt_ids_alloc = allocator,
            };
            gen.pending_logits = logits;
            gen.has_pending_logits = true;
            attachCp(&gen, &ssm_checkpoints, allocator, ssm_cp_peak);
            return gen;
        }

        // A forced tool call decides each successor after its predecessor is
        // known, which no draft path does: drafts never start for it.
        if (sampling.call_force != null) {
            if (mtp_cache) |*mc| {
                mc.deinit();
                mtp_cache = null;
            }
            if (dflash_ctx) |*dc| {
                dc.deinit();
                dflash_ctx = null;
            }
        }

        // DFlash / MTP path: sample synchronously and DO NOT
        // pre-forward the sampled token. The first nextDflash /
        // nextMtp call needs the cache at exactly prompt_len (last prompt
        // token forwarded; first sampled token deferred). The lazy
        // pre-forward path below would over-advance the cache and corrupt
        // every verify forward.
        if (sampling.call_force == null and (mtp_active or dflash_active)) {
            const sample_lazy = try firstTokenSample(xfm, logits, sampling, s);
            _ = mlx.mlx_array_free(logits);
            try mlx.check(mlx.mlx_array_eval(sample_lazy));
            var first_val: i32 = 0;
            try mlx.check(mlx.mlx_array_item_int32(&first_val, sample_lazy));
            _ = mlx.mlx_array_free(sample_lazy);

            if (glm_dflash_native) try reserveGlmDflash(&xfm.config, ctx, &dflash_ctx.?, max_tokens, s);
            var glm_window: ?GlmSpecWindow = if (glm_dflash_native and want_ssm_cp) .{ .snapshot = try dflash_ctx.?.cache.snapshot(), .base_pos = dflash_ctx.?.base_pos } else null;
            errdefer if (glm_window) |*w| w.snapshot.deinit();
            const mtp_cost_profile: mtp_mod.MtpCostProfile = if (mtp_active)
                options.mtp.?.costProfile(xfm, ctx.cache.config)
            else
                .generic;
            const dflash_bs: u32 = if (dflash_active)
                (if (options.dflash_block_size > 0) options.dflash_block_size else options.dflash.?.config.block_size)
            else if (options.dflash_block_size > 0)
                options.dflash_block_size
            else
                0;
            const mtp_cap = resolveMtpDepthCapForProfile(options.mtp_depth, mtp_cost_profile);
            var gen = Generator{
                .xfm = xfm,
                .model_has_mtp = options.model_has_mtp,
                .ctx = ctx,
                .tok = tok,
                .next_token_id = @intCast(first_val),
                .step = 0,
                .max_tokens = max_tokens,
                .sampling = blk: {
                    var sp = sampling;
                    sp.draw = 1; // t1 above was draw 0
                    break :blk sp;
                },
                .prompt_tokens = @intCast(prompt_ids.len),
                .completion_tokens = 0,
                .finish_reason = "length",
                .done = false,
                .eos_token_ids = eos_token_ids,
                .generated_ids = std.ArrayList(u32).empty,
                .timeout_ns = 0,
                .timer = io_util.Stopwatch.init(io),
                .last_hidden = if (need_capture) captured_hidden else mlx.mlx_array_new(),
                .has_last_hidden = need_capture,
                .prng = std.Random.DefaultPrng.init(sampling.seed orelse @intCast(std.Io.Timestamp.now(io, .real).toMilliseconds())),
                .prompt_ids_owned = prompt_owned,
                .prompt_ids_alloc = allocator,
                .drafter_block_size = options.drafter_block_size,
                .dflash = if (dflash_active) options.dflash else null,
                .dflash_ctx = dflash_ctx,
                .glm_dflash_native = glm_dflash_native,
                .glm_prefill_window = glm_window,
                .dflash_block_size = dflash_bs,
                .mtp = if (mtp_active) options.mtp else null,
                .mtp_accept_graph = accept_route.graph,
                .mtp_accept_prefix = accept_route.prefix,
                .mtp_accept_param = accept_route.param,
                .mtp_greedy_tail = options.mtp_greedy_tail,
                .mtp_cache = mtp_cache,
                .mtp_position_base = mtp_position_base,
                .mtp_depth = mtp_cap,
                .mtp_depth_free = if (xfm.mtp_depth_free != 0) xfm.mtp_depth_free else mtpDepthCapFree(options.mtp_depth),
                .mtp_ev_costs = mtpEvCostsForProfile(mtp_cost_profile),
                // Start at the floor (depth 1 by default) and climb with evidence:
                // the cheap depth is the safe default (1.11x on cold/creative
                // content), and hot workloads promote within ~8 rounds.
                .mtp_depth_current = depth_bounds.floorFor(depth_bounds.active, mtp_cap),
            };
            mtp_cache = null; // ownership transferred to the Generator
            dflash_ctx = null; // ownership transferred to the Generator
            glm_window = null;
            // pending_logits/pending_token left empty — the lazy pipeline is
            // skipped under drafter / MTP. The speculative `next*` paths
            // drive every subsequent step with predictable cache offset.
            attachCp(&gen, &ssm_checkpoints, allocator, ssm_cp_peak);
            return gen;
        }

        // Phase 2: scheduler-managed slots ask init to sample t1 synchronously
        // and skip the lazy pre-forward. Cache lands at prompt_len with t1 NOT
        // in cache — matches `forwardBatchedDecode`'s expectation and
        // the drafter init path's invariant. Generator.next's transition
        // shim handles the bootstrap on the first decode tick.
        if (options.skip_lazy_preforward) {
            const sample_lazy = try firstTokenSample(xfm, logits, sampling, s);
            try mlx.check(mlx.mlx_array_eval(sample_lazy));
            var first_val: i32 = 0;
            try mlx.check(mlx.mlx_array_item_int32(&first_val, sample_lazy));
            _ = mlx.mlx_array_free(sample_lazy);
            // t1's own distribution comes from the prefill's final forward and
            // is invisible to the decode loop — read it here or the first
            // returned token has no logprobs and the array shifts by one.
            const first_lp = try firstTokenLogprobs(allocator, logits, @intCast(first_val), options.logprobs_n, s);
            _ = mlx.mlx_array_free(logits);

            var gen = Generator{
                .pending_logprob = first_lp,
                .xfm = xfm,
                .model_has_mtp = options.model_has_mtp,
                .ctx = ctx,
                .tok = tok,
                .next_token_id = @intCast(first_val),
                .step = 0,
                .max_tokens = max_tokens,
                .sampling = blk: {
                    var sp = sampling;
                    sp.draw = 1; // t1 above was draw 0
                    break :blk sp;
                },
                .prompt_tokens = @intCast(prompt_ids.len),
                .completion_tokens = 0,
                .finish_reason = "length",
                .done = false,
                .eos_token_ids = eos_token_ids,
                .generated_ids = std.ArrayList(u32).empty,
                .timeout_ns = 0,
                .timer = io_util.Stopwatch.init(io),
                .last_hidden = if (has_captured_hidden) captured_hidden else mlx.mlx_array_new(),
                .has_last_hidden = has_captured_hidden,
                .prng = std.Random.DefaultPrng.init(sampling.seed orelse @intCast(std.Io.Timestamp.now(io, .real).toMilliseconds())),
                .prompt_ids_owned = prompt_owned,
                .prompt_ids_alloc = allocator,
            };
            attachCp(&gen, &ssm_checkpoints, allocator, ssm_cp_peak);
            return gen;
        }

        // Regular path: sample first token lazily, then build the next forward pass
        const lazy_token = try firstTokenSample(xfm, logits, sampling, s);

        const next_logits = try lazyForward(xfm, &ctx, lazy_token);

        // Async-eval the decode pipeline (single-token graphs, much smaller)
        {
            const eval_vec = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(eval_vec);
            _ = mlx.mlx_vector_array_append_value(eval_vec, lazy_token);
            _ = mlx.mlx_vector_array_append_value(eval_vec, next_logits);
            _ = mlx.mlx_async_eval(eval_vec);
        }

        // Sync to get the first token value
        try mlx.check(mlx.mlx_array_eval(lazy_token));
        var val: i32 = 0;
        try mlx.check(mlx.mlx_array_item_int32(&val, lazy_token));
        _ = mlx.mlx_array_free(lazy_token);
        // See the skip_lazy_preforward branch: t1's distribution lives only in
        // the prefill's own output, so it is read here, against the id that was
        // actually drawn.
        const first_lp = try firstTokenLogprobs(allocator, logits, @intCast(val), options.logprobs_n, s);
        _ = mlx.mlx_array_free(logits);

        var gen = Generator{
            .pending_logprob = first_lp,
            .xfm = xfm,
            .model_has_mtp = options.model_has_mtp,
            .ctx = ctx,
            .tok = tok,
            .next_token_id = @intCast(val),
            .step = 0,
            .max_tokens = max_tokens,
            .sampling = blk: {
                var sp = sampling;
                sp.draw = 1; // t1 above was draw 0
                break :blk sp;
            },
            .prompt_tokens = @intCast(prompt_ids.len),
            .completion_tokens = 0,
            .finish_reason = "length",
            .done = false,
            .eos_token_ids = eos_token_ids,
            .generated_ids = std.ArrayList(u32).empty,
            .timeout_ns = 0,
            .timer = io_util.Stopwatch.init(io),
            .last_hidden = if (has_captured_hidden) captured_hidden else mlx.mlx_array_new(),
            .has_last_hidden = has_captured_hidden,
            .prng = std.Random.DefaultPrng.init(sampling.seed orelse @intCast(std.Io.Timestamp.now(io, .real).toMilliseconds())),
            .prompt_ids_owned = prompt_owned,
            .prompt_ids_alloc = allocator,
        };

        gen.pending_logits = next_logits;
        gen.has_pending_logits = true;

        attachCp(&gen, &ssm_checkpoints, allocator, ssm_cp_peak);
        return gen;
    }

    /// Return the prefill's parked transients once, at the handover to decode.
    /// Shares `advanceStep`'s clock: a clear here is a clear, so the decode
    /// cadence measures its next interval from this point.
    pub fn clearPoolBeforeDecode(self: *Generator) void {
        if (!clearsPoolAtPrefillEnd(&self.xfm.config)) return;
        _ = mlx.mlx_clear_cache();
        self.last_cache_clear_step = self.step;
    }

    /// The ONE place `step` and `completion_tokens` advance — every decode path
    /// (plain, constrained, drafter, MTP, batched) routes through here.
    ///
    /// Advancing them by hand is how three paths — `nextDrafter`, `nextMtp` and
    /// `scheduler.runBatchedDecodeTick` — ended up never calling
    /// `mlx_clear_cache()` at all (issue #110). MLX parks freed buffers in a
    /// size-keyed pool instead of returning them to the OS, so a decode path
    /// with no clear ratchets the process footprint while `active_bytes` stays
    /// flat: the reporter's process reached 81 GB with the panel reading 19.6.
    /// A `-mtp` checkpoint on a dense trunk defaults straight onto one of them.
    /// A source-scan test pins that no new path can reintroduce the hole.
    pub fn advanceStep(self: *Generator, n: u32) void {
        self.completion_tokens += n;
        self.step += n;
        if (shouldClearAllocatorCache(self.step, self.last_cache_clear_step, CACHE_CLEAR_INTERVAL)) {
            _ = mlx.mlx_clear_cache();
            self.last_cache_clear_step = self.step;
        }
    }

    pub fn hasPrePayloadConstraint(self: *const Generator) bool {
        const constraint = self.sampling.constraint orelse return false;
        return constraint.proto != null and constraint.pstate.phase != .json_body;
    }

    /// The pending reasoning→payload boundary, consumed by the scheduler when
    /// it publishes the token that carries it (same delivery path as the
    /// token itself, so a streaming reader cannot see one without the other).
    pub fn takeConstraintSpan(self: *Generator) ?rp_mod.ConstraintSpan {
        const span = self.sampling.constraint orelse return null;
        const s = span.pending_span orelse return null;
        span.pending_span = null;
        return s;
    }

    pub const ConstraintRecovery = union(enum) {
        /// One forced token was committed through the live forward path; the
        /// caller publishes it like a sampled token.
        committed: u32,
        /// The choice state recovers by answering directly: nothing to
        /// commit, the next sample is JSON-masked.
        activated,
        /// The remaining transition does not fit the completion allowance.
        refused,
    };

    /// Force the reasoning→constrained transition when the protocol has not
    /// reached the JSON body: a reasoning loop, premature EOS, or terminal
    /// padding recovers into a valid final answer when the remaining
    /// completion allowance fits the WHOLE remaining transition plus one
    /// answer token. Commits at most one forced token per call (the next
    /// tick drains the rest), so cancellation, stop sequences, inference
    /// errors, and stalled forwards keep their usual precedence.
    pub fn forceConstraintTransition(self: *Generator, allocator: std.mem.Allocator) !ConstraintRecovery {
        const constraint = self.sampling.constraint orelse return .refused;
        const proto = constraint.proto orelse return .refused;
        std.debug.assert(!self.has_pending_token);

        switch (constraint.pstate.phase) {
            .json_body => return .refused,
            .choice => {
                if (constraint.pstate.open_cursor == 0) {
                    // The model never committed a channel and emitted no
                    // opener bytes: a direct constrained answer needs no
                    // transition token at all. The pending logits stay —
                    // they are the distribution the next JSON-masked sample
                    // reads.
                    constraint.pstate.phase = .json_body;
                    constraint.pstate.recovering = false;
                    constraint.pending_span = .{ .token_index = @intCast(self.generated_ids.items.len), .byte_offset = 0 };
                    self.loop_guard_start = self.generated_ids.items.len;
                    log.info("[grammar] reasoning loop at the channel choice; enforcing JSON schema immediately\n", .{});
                    return .activated;
                }
                // Opener bytes are already in the output: starting the JSON
                // here would strand them (the response split reads them as an
                // unclosed opener and files the answer as reasoning).
                // Complete the opener, close it, then constrain. Fall through
                // to the shared planner below.
                log.info("[grammar] completing the generated reasoning opener before the boundary\n", .{});
                if (self.has_pending_logits) {
                    _ = mlx.mlx_array_free(self.pending_logits);
                    self.has_pending_logits = false;
                }
                if (!constraint.pstate.recovering) {
                    var planned: [rp_mod.MAX_TRANSITION_TOKENS]u32 = undefined;
                    const planned_len = proto.planRecovery(&constraint.pstate, &planned) orelse return .refused;
                    if (!forcedBoundaryCanContinue(self.completion_tokens, self.max_tokens, planned_len)) return .refused;
                    constraint.pstate.recovering = true;
                    constraint.pstate.forced_cursor = 0;
                    constraint.pstate.pending_len = @intCast(planned_len);
                    @memcpy(constraint.pstate.pending[0..planned_len], planned[0..planned_len]);
                }
                return .{ .committed = try self.commitForcedConstraintToken(allocator, constraint) };
            },
            .reasoning, .header => {
                if (self.has_pending_logits) {
                    _ = mlx.mlx_array_free(self.pending_logits);
                    self.has_pending_logits = false;
                }
                if (!constraint.pstate.recovering) {
                    var planned: [rp_mod.MAX_TRANSITION_TOKENS]u32 = undefined;
                    const planned_len = proto.planRecovery(&constraint.pstate, &planned) orelse return .refused;
                    if (!forcedBoundaryCanContinue(self.completion_tokens, self.max_tokens, planned_len)) return .refused;
                    constraint.pstate.recovering = true;
                    constraint.pstate.forced_cursor = 0;
                    constraint.pstate.pending_len = @intCast(planned_len);
                    @memcpy(constraint.pstate.pending[0..planned_len], planned[0..planned_len]);
                }
                return .{ .committed = try self.commitForcedConstraintToken(allocator, constraint) };
            },
        }
    }

    /// Commit ONE canonical transition token through the live model state,
    /// then leave its logits pending for the next sample. Called once per
    /// scheduler tick while `pstate.recovering` drains.
    fn commitForcedConstraintToken(self: *Generator, allocator: std.mem.Allocator, constraint: *Constraint) !u32 {
        if (self.has_pending_logits) {
            _ = mlx.mlx_array_free(self.pending_logits);
            self.has_pending_logits = false;
        }
        const seq = constraint.pstate.pending[0..constraint.pstate.pending_len];
        const tok = seq[constraint.pstate.forced_cursor];
        constraint.pstate.forced_cursor += 1;

        try self.generated_ids.append(allocator, tok);
        self.loop_guard_start = self.generated_ids.items.len;
        self.next_token_id = tok;
        self.advanceStep(1);
        self.consecutive_pad = 0;

        const tok_i32: i32 = @intCast(tok);
        const tok_shape = [_]c_int{ 1, 1 };
        const tok_input = mlx.mlx_array_new_data(&tok_i32, &tok_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(tok_input);
        const next_logits = try self.xfm.forwardWith(&self.ctx, tok_input);
        const arr = [_]mlx.mlx_array{next_logits};
        const vec = mlx.mlx_vector_array_new_data(&arr, 1);
        _ = mlx.mlx_async_eval(vec);
        _ = mlx.mlx_vector_array_free(vec);
        self.pending_logits = next_logits;
        self.has_pending_logits = true;

        if (constraint.pstate.forced_cursor >= seq.len) {
            // The transition drained: the next sample is the first constrained
            // payload token. Reset the repetition window at the final-body
            // transition so the drained sequence cannot re-trigger recovery.
            constraint.pstate.recovering = false;
            constraint.pstate.phase = .json_body;
            constraint.pending_span = .{ .token_index = @intCast(self.generated_ids.items.len), .byte_offset = 0 };
            self.loop_guard_start = self.generated_ids.items.len;
        }
        return tok;
    }

    pub fn loopGuardStart(self: *const Generator) usize {
        return @min(self.loop_guard_start, self.generated_ids.items.len);
    }

    pub fn deinit(self: *Generator, allocator: std.mem.Allocator) void {
        self.sampling.think_penalty.deinitPrepared();
        if (group_planner.enabled() and self.mtp_planner_owned) log.info("[mtp-planner-stats] tokens={d} plain={d} prime={d} spec={d} probes={d} max_gap_ms={d:.2} recovery={d}\n", .{ self.completion_tokens, self.mtp_planner_plain_ticks, self.mtp_planner_prime_ticks, self.mtp_planner_spec_rounds, self.mtp_planner_probes, self.mtp_planner_max_gap_ms, self.mtp_planner_recovery.rounds });
        if (self.last_logprob) |*lp| {
            allocator.free(lp.top_logprobs);
        }
        if (self.pending_logprob) |*lp| {
            allocator.free(lp.top_logprobs);
        }
        if (self.has_pending_logits) {
            _ = mlx.mlx_array_free(self.pending_logits);
            self.has_pending_logits = false;
        }
        if (self.has_pending_token) {
            _ = mlx.mlx_array_free(self.pending_token);
            self.has_pending_token = false;
        }
        if (self.has_last_hidden) {
            _ = mlx.mlx_array_free(self.last_hidden);
            self.has_last_hidden = false;
        }
        if (self.prompt_ids_alloc) |a| {
            a.free(self.prompt_ids_owned);
            self.prompt_ids_owned = &.{};
            self.prompt_ids_alloc = null;
        }
        self.xfm.markQsaPooledRopeStale();
        if (self.mtp_cache) |*mc| {
            mc.deinit();
            self.mtp_cache = null;
        }
        if (self.dflash_ctx) |*dc| {
            dc.deinit();
            self.dflash_ctx = null;
        }
        if (self.glm_prefill_window) |*w| {
            w.snapshot.deinit();
            self.glm_prefill_window = null;
        }
        if (self.mtp_hist_stash) |*st| {
            st.deinit();
            self.mtp_hist_stash = null;
        }
        if (self.mtp_pre_draft) |*pd| {
            pd.deinit(allocator);
            self.mtp_pre_draft = null;
        }
        if (self.mtp_lookup_idx) |*idx| {
            idx.deinit();
            self.mtp_lookup_idx = null;
        }
        // Publish the EV surface for the next request's seed.
        // Only healthy runs qualify — a runtime-disabled or barely-sampled
        // run would poison the next request's plans (inference thread only,
        // same discipline as every other head-state write).
        if (self.mtp) |head| {
            // A pinned-depth run never consulted the EV plan, so its surface
            // is not one the controller chose — publishing it would hand a
            // later ordinary request a diagnostic's numbers.
            if (mtpAdaptiveEnabled() and mtpPinnedDepth() == null and
                !self.spec_disabled_runtime and self.mtp_attempted >= 8 and
                (!group_planner.enabled() or !self.mtp_planner_owned))
            {
                head.setEvSeed(self.mtp_ev_accept, self.mtp_ev_m_lo_prev);
            }
        }
        // Free any SSM checkpoints the caller didn't claim. Each layer-slice
        // was allocated by `ssm_checkpoint_alloc` (= the allocator passed to
        // `initWithOptions`), so we use that one. The ArrayList itself was
        // also created with that allocator.
        if (self.ssm_checkpoint_alloc) |a| {
            for (self.ssm_checkpoints.items) |*cp| cp.deinit(a);
            self.ssm_checkpoints.deinit(a);
            self.ssm_checkpoints = std.ArrayList(SSMCheckpoint).empty;
            self.ssm_checkpoint_alloc = null;
        } else {
            // Defensive: if init never set it, the list is empty — but the
            // backing ArrayList state may still need a deinit call. Use the
            // passed allocator as a fallback.
            self.ssm_checkpoints.deinit(allocator);
            self.ssm_checkpoints = std.ArrayList(SSMCheckpoint).empty;
        }
        self.generated_ids.deinit(allocator);
    }

    /// Transfer ownership of accumulated SSM checkpoints to the caller.
    /// Returns an owned slice; caller must free each `SSMCheckpoint` via
    /// `cp.deinit(allocator)` and the slice itself via `allocator.free`,
    /// where `allocator` is the same one passed to `initWithOptions`.
    /// After return, `ssm_checkpoints` is empty and the Generator owns
    /// nothing related to checkpoints.
    pub fn takeSsmCheckpoints(self: *Generator) []SSMCheckpoint {
        const a = self.ssm_checkpoint_alloc orelse return &[_]SSMCheckpoint{};
        const out = self.ssm_checkpoints.toOwnedSlice(a) catch return &[_]SSMCheckpoint{};
        return out;
    }

    /// Legacy→batched transition (scheduler.runBatchedDecodeTick): consume
    /// the lazy pipeline state so the slot can join a batched tick. The
    /// legacy pipelined decode keeps a lookahead token ALREADY FORWARDED
    /// into the KV cache (`pending_token` / `next_token_id`) plus
    /// `pending_logits` for the position after it. Dropping that state and
    /// re-forwarding `next_token_id` would append a duplicate position to
    /// the cache and re-emit an already-emitted token — corrupting every
    /// stream whose slot enters a batch mid-generation
    /// (tests/test_batched_transition.sh).
    ///
    /// Returns the token to emit this step (the pipelined lookahead), or
    /// null when generation stopped (`checkStop`: EOS / pad-run /
    /// max_tokens / timeout — `finish_reason` is set). On return:
    /// `next_token_id` is sampled but NOT in the cache and pending state is
    /// empty — exactly the batched-tick entry invariant.
    pub fn drainPipelineForBatch(self: *Generator, allocator: std.mem.Allocator) !?u32 {
        try self.resolvePendingToken();
        if (try self.checkStop()) {
            if (self.has_pending_logits) {
                _ = mlx.mlx_array_free(self.pending_logits);
                self.has_pending_logits = false;
            }
            return null;
        }
        // Both pipeline shapes (fresh-from-prefill and post-`next()` fast
        // path) carry pending_logits alongside the in-cache lookahead; a
        // lookahead without logits would force a re-forward of an in-cache
        // token, which is the corruption this method exists to prevent.
        if (!self.has_pending_logits) return error.MissingPendingLogits;

        const token = self.next_token_id;
        self.advanceStep(1);
        try self.generated_ids.append(allocator, token);

        const step_logits = self.pending_logits;
        self.has_pending_logits = false;
        const lazy = blk: {
            defer _ = mlx.mlx_array_free(step_logits);
            break :blk try self.sampleLazy(step_logits, .committed);
        };
        try mlx.check(mlx.mlx_array_eval(lazy));
        var val: i32 = 0;
        try mlx.check(mlx.mlx_array_item_int32(&val, lazy));
        _ = mlx.mlx_array_free(lazy);
        self.next_token_id = @intCast(val);
        return token;
    }

    /// Resolve the deferred pending token: eval the lazy array and extract the u32 value.
    /// This is called at the START of each iteration, giving the GPU maximum time
    /// to compute since the async_eval at the END of the previous iteration.
    /// The ONE lazy sampler for a slot's own draws: advances the seed draw index. `after` says
    /// what precedes the sampled position, which decides the think penalty there.
    pub fn sampleLazy(self: *Generator, logits: mlx.mlx_array, after: ThinkAfter) !mlx.mlx_array {
        const shifted = try self.thinkShifted(logits, after);
        defer if (shifted.ctx != logits.ctx) {
            _ = mlx.mlx_array_free(shifted);
        };
        defer self.sampling.draw +%= 1;
        // A penalised slot reaches here only from the constrained steps and a forced call's
        // name choice, whose logits follow `generated_ids` (`samplesSync`, `draftsRefused`).
        return penalizedSampleLazy(std.heap.page_allocator, shifted, self.sampling, self.generated_ids.items, self.xfm.s);
    }

    /// This slot samples on the synchronous serial path (`sampleToken`), never the lazy pipeline.
    pub fn samplesSync(self: *const Generator) bool {
        return self.logprobs_n > 0 or penaltyActive(self.sampling);
    }

    /// What precedes a sampled position beyond the committed `generated_ids`.
    pub const ThinkAfter = union(enum) {
        committed,
        /// Decided ids not committed yet (t1, a block decoder's accepted prefix).
        decided: []const u32,
        /// The pipelined token still in flight, `[1]` or `[1, 1]`.
        pending: mlx.mlx_array,
        /// No penalty here: a draft's proposal, or rows a verify already shifted.
        none,
    };

    /// `logits` under the think penalty where the span is open; the input itself when nothing
    /// shifts, else a new handle the caller frees.
    pub fn thinkShifted(self: *Generator, logits: mlx.mlx_array, after: ThinkAfter) !mlx.mlx_array {
        const tp = &self.sampling.think_penalty;
        if (!tp.active() or after == .none) return logits;
        tp.observe(self.generated_ids.items);
        var state = tp.*;
        switch (after) {
            .decided => |ids| state.phase = tp.phaseAfter(ids),
            else => {},
        }
        if (!tp.hasBias() and (self.xfm.think_marker_mask == null or state.phase == .after or (after != .pending and state.phase != .inside))) return logits;
        const s = self.xfm.s;
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        if (after == .pending) {
            var ids = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ids);
            try mlx.check(mlx.mlx_reshape(&ids, after.pending, &[_]c_int{ 1, 1 }, 2, s));
            try think_penalty.shiftScoped(&out, logits, self.xfm.think_marker_mask, state, ids, s);
        } else {
            try think_penalty.shiftScoped(&out, logits, self.xfm.think_marker_mask, state, null, s);
        }
        return out;
    }

    /// Shift a verify block in place: row j follows the committed ids, then `ids[0..j]`
    /// (`[1, L]`, the rows the forward ran). True = the block changed.
    pub fn thinkShiftRows(self: *Generator, logits: *mlx.mlx_array, ids: mlx.mlx_array) !bool {
        const tp = &self.sampling.think_penalty;
        if (!tp.active()) return false;
        tp.observe(self.generated_ids.items);
        if (!tp.hasBias() and (self.xfm.think_marker_mask == null or tp.phase == .after)) return false;
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        try think_penalty.shiftScoped(&out, logits.*, self.xfm.think_marker_mask, tp.*, ids, self.xfm.s);
        _ = mlx.mlx_array_free(logits.*);
        logits.* = out;
        return true;
    }

    /// Row j of a verify block under the request's repeat and presence penalties, over the committed
    /// ids and `ids[0..j]` (inclusive): the logits its serial tick would sample. Drafts stay raw.
    pub fn penaltyShiftRows(self: *Generator, logits: *mlx.mlx_array, ids: mlx.mlx_array) !bool {
        if (!penaltyActive(self.sampling)) return false;
        const shape = mlx.getShape(logits.*);
        const committed = try think_penalty.buildMask(std.heap.page_allocator, self.generated_ids.items, @intCast(shape[shape.len - 1]));
        defer _ = mlx.mlx_array_free(committed);
        var out = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(out);
        try seen_penalty.shiftRows(&out, logits.*, ids, committed, self.sampling.repeat_penalty, self.sampling.presence_penalty, self.xfm.s);
        _ = mlx.mlx_array_free(logits.*);
        logits.* = out;
        return true;
    }

    /// The host-decided penalty for a sample after the committed ids, for `sampleToken`.
    fn markerShiftCommitted(self: *Generator) ?MarkerShift {
        const tp = &self.sampling.think_penalty;
        if (!tp.active()) return null;
        tp.observe(self.generated_ids.items);
        if (!tp.hasBias() and (self.xfm.think_marker_mask == null or tp.phase != .inside)) return null;
        return .{ .mask = self.xfm.think_marker_mask, .state = tp.* };
    }

    /// What a forced tool call makes of the next position.
    fn forcedNext(self: *Generator) CallForce.Decision {
        const cf = self.sampling.call_force orelse return .sample;
        return cf.decide(self.generated_ids.items);
    }

    /// A sample restricted to `ids`, the branch of a forced call's name.
    fn sampleAmong(self: *Generator, allocator: std.mem.Allocator, logits: mlx.mlx_array, ids: []const u32) !mlx.mlx_array {
        const shape = mlx.getShape(logits);
        const mask = try allocator.alloc(bool, @intCast(shape[shape.len - 1]));
        defer allocator.free(mask);
        @memset(mask, false);
        for (ids) |id| {
            if (id < mask.len) mask[id] = true;
        }
        var masked = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(masked);
        try applyGrammarMask(allocator, &masked, logits, mask, self.xfm.s);
        return self.sampleLazy(masked, .committed);
    }

    /// While a forced call is armed every successor is decided AFTER its
    /// predecessor is known, so the pipelined fast path stands aside.
    fn callForcePending(self: *const Generator) bool {
        const cf = self.sampling.call_force orelse return false;
        return cf.pending();
    }

    fn resolvePendingToken(self: *Generator) !void {
        if (!self.has_pending_token) return;
        try mlx.check(mlx.mlx_array_eval(self.pending_token));
        var val: i32 = 0;
        try mlx.check(mlx.mlx_array_item_int32(&val, self.pending_token));
        _ = mlx.mlx_array_free(self.pending_token);
        self.has_pending_token = false;
        self.next_token_id = @intCast(val);
    }

    const DrainResult = union(enum) {
        /// No pending pipeline state — the spec entry invariant already holds.
        already_clean,
        /// One token was emitted while draining; caller returns it this step.
        drained: u32,
        /// The drained token hit a stop condition — generation is over.
        stopped,
        /// Unexpected half-state; do not re-enable speculation.
        stay_disabled,
    };

    /// Transition from the pipelined `next()` state back to the spec-decode
    /// entry invariant (next_token_id known but NOT in cache, no pending
    /// state). The pipeline holds `pending_token` (lazy, its forward already
    /// in the cache) and `pending_logits` (logits for the position after it):
    /// resolving the token, emitting it, and sampling its successor from
    /// `pending_logits` WITHOUT forwarding lands exactly on the invariant.
    /// One sync. Also handles the shim-seeded state (`pending_logits` only).
    fn drainPipelineForSpec(self: *Generator, allocator: std.mem.Allocator) !DrainResult {
        if (!self.has_pending_logits) {
            if (!self.has_pending_token) return .already_clean;
            // pending_token without pending_logits never occurs in the
            // pipelined state machine; bail rather than risk the invariant.
            return .stay_disabled;
        }
        try self.resolvePendingToken();
        if (try self.checkStop()) return .stopped;
        const token = self.next_token_id;
        self.advanceStep(1);
        try self.generated_ids.append(allocator, token);

        const step_logits = self.pending_logits;
        self.has_pending_logits = false;
        // With the latch these checks return instead of ending the process, so the handle needs an owner.
        const val: i32 = blk: {
            const lazy = sampled: {
                defer _ = mlx.mlx_array_free(step_logits);
                break :sampled try self.sampleLazy(step_logits, .committed);
            };
            defer _ = mlx.mlx_array_free(lazy);
            try mlx.check(mlx.mlx_array_eval(lazy));
            var v: i32 = 0;
            try mlx.check(mlx.mlx_array_item_int32(&v, lazy));
            break :blk v;
        };
        self.next_token_id = @intCast(val);
        return .{ .drained = token };
    }

    pub const ForcedCommit = struct {
        /// Tokens emitted by this call: any pending token plus every forced
        /// token but the last, which stays as `next_token_id` for the next
        /// regular tick to publish.
        emitted: []const u32,
        stopped: bool,
    };

    /// Commit `forced` through the model as ONE multi-token forward, from any
    /// of the three inter-tick states (pipelined, spec exit, shim-seeded), and
    /// leave the generator in the shim-seeded state a regular `next()` reads.
    /// The caller routes every later tick through the regular path.
    pub fn commitForcedTokens(self: *Generator, allocator: std.mem.Allocator, forced: []const u32) !ForcedCommit {
        std.debug.assert(forced.len > 0);
        var emitted = std.ArrayList(u32).empty;
        errdefer emitted.deinit(allocator);
        var input = std.ArrayList(u32).empty;
        defer input.deinit(allocator);

        if (self.has_pending_token or self.has_pending_logits) {
            // Pipelined or shim-seeded: `next_token_id` is in the cache, unpublished.
            try self.resolvePendingToken();
            if (try self.checkStop()) return .{ .emitted = try emitted.toOwnedSlice(allocator), .stopped = true };
            try self.generated_ids.append(allocator, self.next_token_id);
            try emitted.append(allocator, self.next_token_id);
            self.advanceStep(1);
            if (self.has_pending_logits) {
                _ = mlx.mlx_array_free(self.pending_logits);
                self.has_pending_logits = false;
            }
        } else {
            // Spec exit: `next_token_id` is decided but not in the cache.
            try input.append(allocator, self.next_token_id);
            try self.generated_ids.append(allocator, self.next_token_id);
            try emitted.append(allocator, self.next_token_id);
            self.advanceStep(1);
        }
        try input.appendSlice(allocator, forced);
        for (forced[0 .. forced.len - 1]) |f| {
            try self.generated_ids.append(allocator, f);
            try emitted.append(allocator, f);
        }
        self.advanceStep(@intCast(forced.len - 1));
        self.consecutive_pad = 0;
        self.loop_guard_start = self.generated_ids.items.len;

        const ids_i32 = try allocator.alloc(i32, input.items.len);
        defer allocator.free(ids_i32);
        for (input.items, 0..) |t, i| ids_i32[i] = @intCast(t);
        const shape = [_]c_int{ 1, @intCast(input.items.len) };
        const tok_input = mlx.mlx_array_new_data(ids_i32.ptr, &shape, 2, .int32);
        defer _ = mlx.mlx_array_free(tok_input);
        const all_logits = try self.xfm.forwardWith(&self.ctx, tok_input);
        defer _ = mlx.mlx_array_free(all_logits);
        const shp = mlx.getShape(all_logits);
        var logits = mlx.mlx_array_new();
        const start = [_]c_int{ 0, shp[1] - 1, 0 };
        const stop = [_]c_int{ shp[0], shp[1], shp[2] };
        const strides = [_]c_int{ 1, 1, 1 };
        try mlx.check(mlx.mlx_slice(&logits, all_logits, &start, 3, &stop, 3, &strides, 3, self.xfm.s));
        const arr = [_]mlx.mlx_array{logits};
        const vec = mlx.mlx_vector_array_new_data(&arr, 1);
        _ = mlx.mlx_async_eval(vec);
        _ = mlx.mlx_vector_array_free(vec);
        self.pending_logits = logits;
        self.has_pending_logits = true;
        self.next_token_id = forced[forced.len - 1];
        self.mtp_hidden_stale = true;
        return .{ .emitted = try emitted.toOwnedSlice(allocator), .stopped = false };
    }

    /// The ONE place a partial accept rolls recurrent state back from the verify capture:
    /// the loop and the `partial_rounds` count travel together, so a caller cannot take the
    /// rollback without being counted.
    fn rollbackSsmFromCapture(self: *Generator, entries: []transformer_mod.SSMCacheEntry, accepted: u32, verify_len: u32, s: mlx.mlx_stream) !void {
        self.partial_rounds += 1;
        for (entries) |*entry| {
            try transformer_mod.ssmRollbackFromCapture(entry, accepted, verify_len, s);
        }
    }

    /// Result of one speculative step: the tokens to emit and how many drafts landed.
    pub const DrafterStepResult = struct {
        /// Tokens to emit this step. On a full accept this is
        /// `[t1, ...all_drafts]` (length `block_size`); on partial accept j
        /// it is `[t1, draft[0..j]]` (length `1+j`). The corrected fallback
        /// becomes `next_token_id` for the next call.
        tokens: []const u32,
        /// Number of *drafted* tokens accepted (excludes always-accepted t1).
        accepted_tokens: u32,
    };

    /// DFlash block-drafter step. ONE assistant forward proposes
    /// `block_size - 1` drafts, conditioned on the trunk's cached
    /// target_layer_ids hiddens (the per-request `dflash_ctx`); the trunk
    /// verify over `[t1, drafts...]` follows the standard spec invariant
    /// (`cache.step = prompt_len + emitted`, t1 NOT in cache on entry,
    /// correction from ORIGINAL `verify_logits[accepted]`).
    ///
    /// Rollback is an offset-only `cache.truncate` — legal because
    /// `DflashModel.bind` restricts targets to the pure-KVCache standard
    /// path (no SSM, no module-owned state), where a position's K/V depend
    /// only on earlier positions (causal) and are identical whether computed
    /// in a width-16 verify or any re-forward. No re-forward runs at all:
    /// the next round's context comes from slicing THIS round's verify
    /// captures to the committed prefix — exactly the reference's
    /// `hidden_states[i+1][:, :n_accepted]`.
    ///
    /// Drafts are greedy (argmax over the trunk lm_head on assistant
    /// hiddens, anchor row DROPPED — reference `[:, 1:]`); sampled requests
    /// use the same one-hot Leviathan acceptance.
    const GlmSampleContext = struct { generator: *Generator, logits: mlx.mlx_array };

    fn sampleGlmNode(raw: *anyopaque, row: usize) !u32 {
        const context: *GlmSampleContext = @ptrCast(@alignCast(raw));
        const gen = context.generator;
        var ops = @import("glm5_model.zig").Ops{ .s = gen.xfm.s };
        defer ops.deinit();
        const logits = try ops.slice(context.logits, 1, @intCast(row), @intCast(row + 1));
        const sampled = sampleTokenLazy(logits, gen.sampling, gen.xfm.s);
        defer _ = mlx.mlx_array_free(sampled);
        try mlx.check(mlx.mlx_array_eval(sampled));
        var token: i32 = 0;
        try mlx.check(mlx.mlx_array_item_int32(&token, sampled));
        gen.sampling.draw +%= 1;
        return @intCast(token);
    }

    fn sampledGlmDecisions(raw: *anyopaque, verified: *@import("glm5_dflash_model.zig").Verified, budget: usize, eos: []const u32) !void {
        const gen: *Generator = @ptrCast(@alignCast(raw));
        var context = GlmSampleContext{ .generator = gen, .logits = verified.logits };
        _ = try @import("glm5_dflash_tree.zig").sampledTargets(verified.tokens[0..verified.count], verified.parents[0..verified.count], verified.targets[0..verified.count], budget, eos, &context, sampleGlmNode);
    }

    /// Commits this request's row of a grouped GLM tick (`verifyGroups`, one pending token) and
    /// returns the row's logits. A DFlash2 request also appends the row's captures to its
    /// assistant context, so its rounds resume when it decodes alone again.
    pub fn glmRowCommit(self: *Generator, verified: *const @import("glm5_dflash_model.zig").Verified) !mlx.mlx_array {
        const request = self.ctx.glm5_request orelse return error.GlmRequestMissing;
        const s = self.xfm.s;
        if (self.glm_dflash_native) {
            var verification = try verified.prepareCommitConsuming(request, 1, self.eos_token_ids, s);
            defer verification.deinit();
            _ = @import("glm5_dflash.zig").commitVerified(self.dflash.?, &self.dflash_ctx.?, request, &verification, 1, self.eos_token_ids) catch |err| {
                request.failed = true;
                return err;
            };
        } else try @import("glm5_dflash_model.zig").commitPlainRow(verified, request, s);
        self.ctx.cache.step = request.offset;
        var logits = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(logits);
        try mlx.check(mlx.mlx_array_set(&logits, verified.logits));
        return logits;
    }

    fn nextGlmDflash(self: *Generator, allocator: std.mem.Allocator) !?DrafterStepResult {
        if (!try self.glmRoundReady()) return null;
        const schedule = @import("glm5_dflash_model.zig").bindDefaultSchedule();
        defer schedule.restore();
        var timer = io_util.Stopwatch.init(self.timer.io);
        const lookup = try self.glmLookupChain(allocator);
        const proposal = lookup orelse try self.glmPropose(dflash_mod.glmMaxNodes(self.dflash_block_size, dflash_mod.drafterDepthExplicit()));
        const draft_ns = timer.read();
        timer.reset();
        const request = self.ctx.glm5_request.?;
        // Batched rows keep every serial row's bits, so sampled decisions read the same logits.
        const group = @import("glm5_dflash_model.zig").Group{ .request = request, .tokens = proposal.tokens[0..proposal.count], .parents = proposal.parents[0..proposal.count], .penalty = self.glmPenalty() };
        var verified: [1]@import("glm5_dflash_model.zig").Verified = undefined;
        try @import("glm5_dflash_model.zig").verifyGroups(self.xfm.glm5.?, &.{group}, self.dflash.?.config.target_layer_ids, .affine_rows_ffn, &verified);
        const layerwise = &verified[0];
        defer layerwise.deinit();
        const verify_ns = timer.read();
        const result = try self.glmRoundEnd(allocator, &proposal, layerwise, draft_ns, verify_ns, lookup != null);
        if (lookup != null) {
            self.glm_lookup.rounds += 1;
            self.glm_lookup.landed += result.accepted_tokens;
        }
        return result;
    }

    /// Drafts a verbatim-copy round takes from the context instead of the assistant: four rows,
    /// the widest the native verify overlay reads.
    pub const glm_lookup_drafts = 3;

    /// The context's continuation after the pending token when the output is copying a long
    /// span (`mtp_lookup.STRONG_SUFFIX` tokens of agreement), as a chain in place of the
    /// assistant's tree.
    fn glmLookupChain(self: *Generator, allocator: std.mem.Allocator) !?@import("glm5_dflash.zig").Proposal {
        if (mtp_lookup_disabled or isEosId(self.next_token_id, self.eos_token_ids)) return null;
        const got = (try self.mtpLookupIndex(allocator)).match(self.next_token_id, glm_lookup_drafts);
        const chain = glmLookupProposal(got, self.next_token_id, self.max_tokens -| self.completion_tokens) orelse return null;
        if (self.glm_lookup.rounds == 0) log.info("[glm-dflash] lookup chain engaged: {d} drafts, suffix {d}\n", .{ glm_lookup_drafts, got.?.suffix });
        return chain;
    }

    /// The logit policy this request's DFlash2 tree rows take in the verifier; null = none.
    pub fn glmPenalty(self: *Generator) ?@import("glm5_dflash_model.zig").Penalty {
        const tp = &self.sampling.think_penalty;
        if (!tp.active()) return null;
        tp.observe(self.generated_ids.items);
        if (!tp.hasBias() and (self.xfm.think_marker_mask == null or tp.phase == .after)) return null;
        return .{ .state = tp.*, .markers = self.xfm.think_marker_mask };
    }

    /// A DFlash2 GLM round may run: not done, no stop due.
    pub fn glmRoundReady(self: *Generator) !bool {
        if (self.done or try self.checkStop()) return false;
        if (!glmDflashEligible(self.sampling, self.logprobs_n)) return error.SpecDecodeUnsupported;
        if (self.ctx.glm5_request == null) return error.GlmRequestMissing;
        const target = self.xfm.glm5.?;
        target.s = self.xfm.s;
        target.suppress_mask = self.xfm.suppress_mask;
        self.dflash.?.s = self.xfm.s;
        return true;
    }

    /// This request's round draft of at most `max_nodes` nodes.
    pub fn glmPropose(self: *Generator, max_nodes: usize) !@import("glm5_dflash.zig").Proposal {
        return @import("glm5_dflash.zig").proposeRound(self.dflash.?, &self.dflash_ctx.?, self.xfm.glm5.?, self.ctx.glm5_request.?, self.next_token_id, max_nodes, self.max_tokens - self.completion_tokens, self.eos_token_ids, 4);
    }

    /// Decides (sampled requests draw along the visited path), commits and books a verified
    /// round, alone or as one group of a grouped tick.
    pub fn glmRoundEnd(self: *Generator, allocator: std.mem.Allocator, proposal: *const @import("glm5_dflash.zig").Proposal, layerwise: *@import("glm5_dflash_model.zig").Verified, draft_ns: u64, verify_ns: u64, lookup: bool) !DrafterStepResult {
        const budget = self.max_tokens - self.completion_tokens;
        const sampled = !isGreedyTemperature(self.sampling.temperature) and self.sampling.top_k != 1;
        if (sampled) try sampledGlmDecisions(self, layerwise, budget, self.eos_token_ids);
        const request = self.ctx.glm5_request.?;
        var round = try @import("glm5_dflash.zig").finishRound(self.timer.io, self.dflash.?, &self.dflash_ctx.?, self.xfm.glm5.?, request, proposal, layerwise, budget, self.eos_token_ids);
        round.draft_ns = draft_ns;
        round.verify_ns = verify_ns;
        // Accepted share of the tree's depth: a fork can land one draft, a chain two.
        var depth: [16]u32 = @splat(0);
        var deepest: u32 = 0;
        for (proposal.parents[1..proposal.count], 1..) |parent, row| {
            depth[row] = depth[@intCast(parent)] + 1;
            deepest = @max(deepest, depth[row]);
        }
        if (deepest > 0 and !lookup) {
            const rate: f32 = @as(f32, @floatFromInt(round.accepted_drafts)) / @as(f32, @floatFromInt(deepest));
            self.glm_draft_rate = 0.7 * self.glm_draft_rate + 0.3 * rate;
        }
        var emitted = round.count;
        for (round.tokens[1..round.count], 1..) |token, i| {
            if (tokenStops(token, self.eos_token_ids, &self.consecutive_pad)) {
                if (!isEosId(token, self.eos_token_ids)) emitted = i;
                self.done = true;
                self.finish_reason = "stop";
                break;
            }
        }
        const tokens = try allocator.dupe(u32, round.tokens[0..emitted]);
        errdefer allocator.free(tokens);
        try self.generated_ids.appendSlice(allocator, tokens);
        self.advanceStep(@intCast(tokens.len));
        self.ctx.cache.step = request.offset;
        self.next_token_id = round.pending orelse tokens[tokens.len - 1];
        self.dflash_attempted += 1;
        self.dflash_accepted_tokens += round.accepted_drafts;
        self.dflash_round_width = @intCast(round.verified_rows - 1);
        self.glm_round.drafted += round.verified_rows - 1;
        self.glm_round.draft_ns += round.draft_ns;
        self.glm_round.verify_ns += round.verify_ns;
        self.glm_round.replay_ns += round.replay_ns;
        self.glm_round.commit_ns += round.commit_ns;
        if (round.stopped) {
            self.done = true;
            self.finish_reason = "stop";
        }
        return .{ .tokens = tokens, .accepted_tokens = @intCast(round.accepted_drafts) };
    }

    pub fn nextDflash(self: *Generator, allocator: std.mem.Allocator) !?DrafterStepResult {
        if (self.glm_dflash_native) return self.nextGlmDflash(allocator);
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
            const post_warmup = dflash_rounds_before >= DFLASH_GATE_WARMUP;
            const wall = if (post_warmup) self.mtpRegimeWallMs(ms) else ms;
            self.specObserveRound(self.dflash_round_width, wall, @floatFromInt(emitted), post_warmup and emitted > 0, false);
            if (self.dflash_round_width > 0) self.dflash_wall_ms += wall;
        }
        if (self.done) return null;
        std.debug.assert(self.dflash != null);
        std.debug.assert(self.dflash_ctx != null);
        if (specDecodeUnsupported(self.sampling, self.logprobs_n)) return error.SpecDecodeUnsupported;

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

        const tracing = dflashTraceEnabled();
        var ph: io_util.Stopwatch = undefined;
        if (tracing) {
            ph = io_util.Stopwatch.init(self.timer.io);
            if (self.dflash_gap_watch) |*gw| self.dflash_trace.add(.gap, gw.read());
            self.dflash_gap_watch = null;
        }

        // ── Phase 1: one assistant forward drafts all m tokens ──
        // Row mapping is the export's convention (`anchor_row_drafts`):
        // DFlash reads mask rows 1..bs-1 (anchor row dropped), DSpark reads
        // ALL rows starting at the anchor — so DSpark needs only m noise rows
        // for the same m drafts and the same verify width.
        const noise_rows: u32 = if (model.config.anchor_row_drafts) m_full else bs;
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

        // Slice the m draft rows: DFlash drops the anchor row (1..bs-1),
        // DSpark reads every row (0..m) — the anchor row IS draft 0.
        const dl_shape = mlx.getShape(draft_logits_all);
        var draft_logits = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(draft_logits);
        {
            const row0: c_int = if (model.config.anchor_row_drafts) 0 else 1;
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
        const stochastic = !isGreedyTemperature(self.sampling.temperature);
        // DFlash2 path selector: when the sidecar ships one, drafts come from
        // the pairwise-scored path trace instead of per-position argmax /
        // block sampling. Greedy requests keep the byte-equality bar (a
        // selector draft only survives verify if it IS the trunk argmax);
        // stochastic requests sample the selector's own candidate softmax and
        // accept through min(1, p/q) with q read off the traced path —
        // exact by construction.
        const use_selector = model.selector != null;
        // DSpark: the block's base logits are position-parallel, but each
        // position's draft is picked from logits CORRECTED by the token
        // drafted at the previous one (the Markov bigram bias). Chaining it
        // is what the head is for — dropping it drafts every position from
        // an uncorrected distribution the sidecar was never trained to emit.
        const use_markov = model.markov != null;
        var sel_path: ?dflash_mod.SelectedPath = null;
        defer if (sel_path) |*sp| sp.deinit(allocator);
        var draft_q: mlx.mlx_array = .{ .ctx = null }; // [m, V] proposal density
        defer if (draft_q.ctx != null) {
            _ = mlx.mlx_array_free(draft_q);
        };
        var draft_ids = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(draft_ids);
        if (use_markov) {
            const mh = &model.markov.?;
            const ids_i32 = try allocator.alloc(i32, m);
            defer allocator.free(ids_i32);
            var q_rows: ?[]mlx.mlx_array = null;
            defer if (q_rows) |rows| {
                for (rows) |r| _ = mlx.mlx_array_free(r);
                allocator.free(rows);
            };
            if (stochastic) {
                const rows = try allocator.alloc(mlx.mlx_array, m);
                for (rows) |*r| r.* = .{ .ctx = null };
                q_rows = rows;
            }
            var prev: u32 = t1;
            var step: u32 = 0;
            while (step < m) : (step += 1) {
                var base_row = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(base_row);
                {
                    const start = [_]c_int{ 0, @intCast(step), 0 };
                    const stop = [_]c_int{ 1, @as(c_int, @intCast(step)) + 1, dl_shape[2] };
                    const strides = [_]c_int{ 1, 1, 1 };
                    try mlx.check(mlx.mlx_slice(&base_row, draft_logits, &start, 3, &stop, 3, &strides, 3, s));
                }
                const corrected = try mh.stepLogits(base_row, prev, s);
                defer _ = mlx.mlx_array_free(corrected);
                if (stochastic) {
                    // Sample the step from the request's own filtered
                    // distribution and keep the row as q — the accept ratio
                    // and the reject residual both need the density the
                    // draft was actually drawn from.
                    const qrow = try filteredProbsBlock(corrected, self.sampling, s);
                    var logq = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(logq);
                    try mlx.check(mlx.mlx_log(&logq, qrow, s));
                    const null_key = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(null_key);
                    var sampled = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(sampled);
                    try mlx.check(mlx.mlx_random_categorical(&sampled, logq, -1, null_key, s));
                    try mlx.check(mlx.mlx_array_eval(sampled));
                    const sd = mlx.mlx_array_data_int32(sampled) orelse return error.MlxArrayDataNull;
                    prev = @intCast(sd[0]);
                    q_rows.?[step] = qrow;
                } else {
                    var amax = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(amax);
                    try mlx.check(mlx.mlx_argmax_axis(&amax, corrected, 2, false, s));
                    var as_i32 = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(as_i32);
                    try mlx.check(mlx.mlx_astype(&as_i32, amax, .int32, s));
                    try mlx.check(mlx.mlx_array_eval(as_i32));
                    const ad = mlx.mlx_array_data_int32(as_i32) orelse return error.MlxArrayDataNull;
                    prev = @intCast(ad[0]);
                }
                ids_i32[step] = @intCast(prev);
            }
            if (q_rows) |rows| {
                const vec = mlx.mlx_vector_array_new();
                defer _ = mlx.mlx_vector_array_free(vec);
                for (rows) |r| _ = mlx.mlx_vector_array_append_value(vec, r);
                draft_q = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_concatenate_axis(&draft_q, vec, 0, s));
            }
            const row_shape = [_]c_int{ 1, @as(c_int, @intCast(m)) };
            const host_arr = mlx.mlx_array_new_data(ids_i32.ptr, &row_shape, 2, .int32);
            defer _ = mlx.mlx_array_free(host_arr);
            try mlx.check(mlx.mlx_array_set(&draft_ids, host_arr));
        } else if (use_selector) {
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
            if (self.dflashAdaptiveRows()) {
                m = dflash_policy.pickRows(&self.dflash_calib, sel_path.?.conf, dflash_mod.qwenMinRows(), self.dflash_block_size) - 1;
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
        var cl = transformer_mod.CaptureLayers{ .ids = model.config.target_layer_ids, .out = cap_out, .hc_mean = xfm.config.hc_count };
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
            for (entries) |*entry| transformer_mod.ssmFreeSpecCapture(entry);
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
                const target_p = try probsAtLastPos(per_pos_logits.?[k], self.sampling, s);
                defer _ = mlx.mlx_array_free(target_p);
                const p_draft = try probAt(target_p, drafts[k], s);
                const accept_prob: f32 = if (draft_q.ctx != null) blk: {
                    const q_row = try sliceProbRow(draft_q, k, s);
                    defer _ = mlx.mlx_array_free(q_row);
                    break :blk specAcceptProb(p_draft, try probAt(q_row, drafts[k], s));
                } else if (sel_path) |*sp| blk: {
                    // Selector-sampled draft: q is the traced step's own
                    // candidate softmax — exact, no GPU read.
                    const kk = sp.cand_ids.len / @as(usize, m_full);
                    break :blk specAcceptProb(p_draft, sp.q.?[@as(usize, k) * kk + sp.chosen_idx[k]]);
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
        accepted = capAcceptedForTokenBudget(
            accepted,
            self.completion_tokens,
            self.max_tokens,
        );

        const next_pending: u32 = blk: {
            if (stochastic) {
                const correction_logits = per_pos_logits.?[accepted];
                const probs = try probsAtLastPos(correction_logits, self.sampling, s);
                defer _ = mlx.mlx_array_free(probs);
                if (accepted < m) {
                    // norm(max(0, p − q)) against the q the draft was drawn
                    // from — a one-hot here is the WRONG residual for a
                    // sampled draft, and wrong silently.
                    const q_row = if (draft_q.ctx != null)
                        try sliceProbRow(draft_q, accepted, s)
                    else if (sel_path) |*sp|
                        try selectorQRow(sp, accepted, m_full, vl_shape[2], s)
                    else
                        try pldOneHotRow(drafts[accepted], vl_shape[2], s);
                    defer _ = mlx.mlx_array_free(q_row);
                    break :blk try sampleResidual(probs, q_row, s);
                } else {
                    break :blk try sampleFromProbs(probs, s);
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
                const gdn_captured = entries.len > 0 and entries[0].spec_state_seq.ctx != null;
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
        self.checkDflashRuntimeGate();
        if (self.completion_tokens >= self.max_tokens) {
            self.done = true;
            self.finish_reason = "length";
        }

        if (tracing) {
            self.dflashTraceRoundEnd(accepted);
            self.dflash_gap_watch = io_util.Stopwatch.init(self.timer.io);
        }

        return DrafterStepResult{
            .tokens = tokens,
            .accepted_tokens = accepted,
        };
    }

    /// Rows are chosen per round on Flash-Next, whose verify rows are bytes; every other target
    /// verifies the whole block.
    fn dflashAdaptiveRows(self: *const Generator) bool {
        return self.xfm.qwen4 != null and self.dflash_block_size > dflash_mod.qwenMinRows();
    }

    /// DFlash runtime economics gate. Sticky within the request: once a
    /// `DFLASH_GATE_WARMUP`-round sample proves the block-parallel path yields
    /// less than its width-normalized request-class threshold, subsequent
    /// ticks use the regular pipelined decoder through `nextDflash`'s entry
    /// fallback.
    fn checkDflashRuntimeGate(self: *Generator) void {
        if (self.spec_disabled_runtime) return;
        const serial = dflashSerialPrice(&self.xfm.round_cost, self.xfm.round_cost.bucketOf(self.mtpKvLen()));
        if (!dflashGateShouldDisable(self.dflash_attempted, self.dflash_accepted_tokens, self.dflash_wall_ms, self.dflashMeanRows(), serial)) return;
        const avg = @as(f32, @floatFromInt(self.dflash_accepted_tokens)) /
            @as(f32, @floatFromInt(self.dflash_attempted));
        log.info(
            "  dflash=disabled (runtime yield {d:.2} accepted/round, {d:.2} ms/token against {d:.2} serial, after {d} attempts)\n",
            .{ avg, self.dflash_wall_ms / @as(f64, @floatFromInt(self.dflash_attempted + self.dflash_accepted_tokens)), serial orelse 0.0, self.dflash_attempted },
        );
        self.spec_disabled_runtime = true;
    }

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

    fn mtpMropeContext(self: *const Generator) ?mtp_mod.MropeContext {
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
    fn mtpSamplingDraw(self: *Generator, params: SamplingParams) SamplingParams {
        if (isGreedyTemperature(params.temperature)) return params;
        var draw = params;
        draw.draw = self.sampling.draw;
        self.sampling.draw +%= 1;
        return draw;
    }

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

    /// Allocate an empty draft chain for `plan` (nothing built yet).
    fn mtpChainInit(self: *Generator, allocator: std.mem.Allocator, plan: MtpRoundPlan, t1: u32) !MtpPreDraft {
        const consider_ext = plan.m_hi > plan.m_lo;
        const sharp_drafts = !isGreedyTemperature(self.mtpDraftSampling(0).temperature);
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
    fn acceptGraphFor(self: *const Generator, lookup: bool) MtpGraphFn {
        return if (lookup) mtpBatchedExactGraph else self.mtp_accept_graph;
    }

    fn acceptPrefixFor(self: *const Generator, lookup: bool) mtp_acceptance.PrefixFn {
        return if (lookup) mtp_acceptance.exactPrefix else self.mtp_accept_prefix;
    }

    // Prompt lookup inside the MTP round, ported from mlx-serve #523/#533 (Samuel Reed).
    comptime {
        // A lookup round commits up to MAX_DRAFT_STRONG drafts into `MtpHistStash`.
        std.debug.assert(mtp_lookup.MAX_DRAFT_STRONG <= mtp_mod.MAX_DEPTH);
    }

    /// May a lookup stand in for this request's next MTP chain? Not for a batched head or a
    /// planner-owned round, which builds every chain as a group.
    pub fn mtpLookupAllowed(self: *const Generator) bool {
        if (mtp_lookup_disabled) return false;
        const head = self.mtp orelse return false;
        if (head == .qwen or self.mtp_batch_head) return false;
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
    fn mtpRoundUntimed(self: *Generator) void {
        if (self.mtp_regime_clock) |*c| c.reset();
        self.mtp_serial_clock = null;
        self.mtp_serial_warm = 0;
    }

    /// Price a lookup round into the table's lookup row: the wall since the previous round
    /// ended, the quantity the width grid stores. No sample before the regime clock runs.
    fn mtpLookupObserve(self: *Generator, drafts: u32, tokens: u32) void {
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
    fn mtpRoundAcceptObserve(self: *Generator, lookup: bool, drafted: u32, accepted: u32) void {
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

    const LookupLines = struct {
        tok: *const Tokenizer,
        pub fn hasNewline(self: LookupLines, id: u32) bool {
            return self.tok.tokenHasNewline(id);
        }
    };

    /// The committed-token index, synced to `generated_ids`.
    fn mtpLookupIndex(self: *Generator, allocator: std.mem.Allocator) !*mtp_lookup.Index {
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
    fn mtpLookupChain(self: *Generator, allocator: std.mem.Allocator, plan: MtpRoundPlan, t1: u32) !?MtpPreDraft {
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
    fn mtpChainBuild(self: *Generator, chain: *MtpPreDraft, from: u32, to: u32) !void {
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
                !isGreedyTemperature(draft_sampling.temperature),
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
                    const prop = try shortlistStepProposal(sl, self.mtpSamplingDraw(draft_sampling), s);
                    chain.draft_arrs[i] = prop.id;
                    slots[i] = prop.q;
                } else {
                    // The coarse head just retired: propose greedily and hand
                    // the verify the one-hot q that proposal really has.
                    const id = try head.draftSelect(xfm, rerank_x, draft_sampling.suppress_mask);
                    const prop = try argmaxStepProposal(id, @intCast(mlx.getShape(xfm.lm_head_w)[0]), s);
                    chain.draft_arrs[i] = prop.id;
                    slots[i] = prop.q;
                }
                chain.n_qp = i + 1;
            } else if (use_rerank) {
                const rerank_x = if (step_out.rerank_x.ctx != null) step_out.rerank_x else step_out.hidden_next;
                chain.draft_arrs[i] = try head.draftSelect(xfm, rerank_x, draft_sampling.suppress_mask);
            } else if (chain.q_probs) |slots| {
                const prop = try logitsStepProposal(step_out.logits, self.mtpSamplingDraw(draft_sampling), s);
                slots[i] = prop.q;
                chain.n_qp = i + 1;
                chain.draft_arrs[i] = prop.id;
            } else {
                chain.draft_arrs[i] = sampleTokenLazy(step_out.logits, draft_sampling, s);
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
    fn mtpChainDispatch(chain: *const MtpPreDraft, from: u32, to: u32) !void {
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

    fn mtpDraftActiveRows(chains: []const MtpPreDraft, step: u32, indices: []usize) usize {
        var n: usize = 0;
        for (chains, 0..) |chain, row| {
            if (step >= chain.m) continue;
            indices[n] = row;
            n += 1;
        }
        return n;
    }

    /// What one draft step reads. Both rerank paths skip the head's own vocab
    /// projection (the token comes off the coarse readout's exactly re-scored
    /// top-32); `full_logits` pays the whole lm_head because something
    /// downstream needs a density over the vocabulary. A SAMPLED proposal does
    /// not: q over the shortlist is the proposal's true density, zero off it,
    /// and the verify's residual covers the rest.
    pub const DraftStepPath = enum { rerank_greedy, rerank_sampled, full_logits };

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
    fn mtpRowRerankMode(g: *Generator, c: *const MtpPreDraft, step: u32) bool {
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
    fn mtpGroupSampledDrafts(
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
                    const prop = try argmaxStepProposal(id, vocab, s);
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
        var params: [32]SamplingParams = undefined;
        var samples: [32]bool = @splat(false);
        var k_sampled: usize = 0;
        for (gens, chains, 0..) |g, *c, row| {
            const row_params = g.mtpDraftSampling(step);
            if (c.q_probs == null or isGreedyTemperature(row_params.temperature)) continue;
            samples[row] = true;
            sampled[k_sampled] = row;
            params[k_sampled] = row_params;
            k_sampled += 1;
        }
        const batched = k_sampled > 1 and
            sampleRowsHomogeneous(params[0..k_sampled]) and
            !sampleRowsAnySeed(params[0..k_sampled]);

        for (gens, chains, 0..) |g, *c, row| {
            if (samples[row]) continue;
            if (c.q_probs) |slots| {
                const prop = try shortlistStepProposal(shortlists[row], g.mtpDraftSampling(step), s);
                c.draft_arrs[step] = prop.id;
                slots[step] = prop.q;
                c.n_qp = step + 1;
            } else c.draft_arrs[step] = try mtp_mod.shortlistArgmax(s, shortlists[row]);
            c.n_drafted = step + 1;
        }
        if (!batched) {
            for (sampled[0..k_sampled], params[0..k_sampled]) |row, row_params| {
                const prop = try shortlistProposal(shortlists[row], gens[row].mtpSamplingDraw(row_params), s);
                chains[row].draft_arrs[step] = prop.id;
                chains[row].n_drafted = step + 1;
                chains[row].q_probs.?[step] = prop.q;
                chains[row].n_qp = step + 1;
            }
            return;
        }
        const group = try shortlistProposalRows(shortlists[0..n], sampled[0..k_sampled], params[0], s);
        for (sampled[0..k_sampled], 0..) |row, j| {
            chains[row].draft_arrs[step] = group.ids[j];
            chains[row].n_drafted = step + 1;
            chains[row].q_probs.?[step] = group.qs[j];
            chains[row].n_qp = step + 1;
        }
    }

    fn mtpBatchHeadPosition(pos_base: c_int, seq_offset: usize) c_int {
        return if (seq_offset == 0 and pos_base == -1) 1 else pos_base + @as(c_int, @intCast(seq_offset));
    }

    fn mtpChainBuildBatchedActive(gens: []const *Generator, chains: []MtpPreDraft, from: u32, to: u32) !void {
        if (gens.len == 0) return;
        std.debug.assert(gens.len == chains.len);
        const xfm = gens[0].xfm;
        var any_mrope = false;
        var all_qwen4 = true;
        for (gens) |g| {
            if (g.mtpMropeContext() != null) any_mrope = true;
            switch (g.mtp.?) {
                .qwen4 => {},
                .qwen, .mimo => all_qwen4 = false,
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
                        const prop = try logitsStepProposal(outs[k].logits, g.mtpSamplingDraw(draft_sampling), xfm.s);
                        slots[i] = prop.q;
                        c.n_qp = i + 1;
                        c.draft_arrs[i] = prop.id;
                    } else {
                        c.draft_arrs[i] = sampleTokenLazy(outs[k].logits, draft_sampling, xfm.s);
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

    fn mtpBatchedExactGraph(
        probs_all: mlx.mlx_array,
        draft_arrs: []const mlx.mlx_array,
        q_probs: ?[]const mlx.mlx_array,
        m: u32,
        _: f32,
        sampling: SamplingParams,
        s: mlx.mlx_stream,
    ) !MtpBatchedGraph {
        return mtpBatchedAcceptGraph(probs_all, draft_arrs, q_probs, m, sampling, s);
    }

    pub fn mtpBatchedAcceptGraph(
        probs_all: mlx.mlx_array,
        draft_arrs: []const mlx.mlx_array,
        q_probs: ?[]const mlx.mlx_array,
        m: u32,
        sampling: SamplingParams,
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
            const key = seedKey(sampling);
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
        sampling: SamplingParams,
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
        sampling: SamplingParams,
        s: mlx.mlx_stream,
    ) !MtpBatchedGraph {
        return mtpBatchedLossyGraph(.tokenv3, probs_all, draft_arrs, q_probs, m, alpha, sampling, s);
    }

    /// Specialized at compile time; the installed route adds no mode switch
    /// to a measured MTP round. Both lanes keep corrections in one lazy graph.
    fn mtpBatchedLossyGraph(
        comptime mode: enum { typical, tokenv3 },
        probs_all: mlx.mlx_array,
        draft_arrs: []const mlx.mlx_array,
        q_probs: ?[]const mlx.mlx_array,
        m: u32,
        param: f32,
        sampling: SamplingParams,
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
            const key = seedKey(sampling);
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
    fn mtpPreDraftOpenAllows(completion_tokens: u32, max_tokens: u32, pending_is_eos: bool) bool {
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
    fn mtpMaybePreDraft(self: *Generator, allocator: std.mem.Allocator) !void {
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

    // Serial blocks inside nextMtp. Leaving MTP is cheap; coming back needs `t1 NOT in cache`
    // and an `h_prev` for that position: `drainPipelineForSpec` lands the first, the capture
    // tick the second. The head's committed history does not grow across a serial block
    // (a content gap that costs acceptance for a while); M-RoPE turns never come back.

    /// Will this serial block be left behind? Only then is applying the deferred history stash worth a head forward.
    fn mtpSerialMayResume(self: *const Generator) bool {
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
    fn mtpApplyStash(self: *Generator) !void {
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

    fn mtpSerialOneToken(allocator: std.mem.Allocator, tok: u32) !DrafterStepResult {
        const tokens = try allocator.alloc(u32, 1);
        tokens[0] = tok;
        return DrafterStepResult{ .tokens = tokens, .accepted_tokens = 0 };
    }

    /// Inter-tick wall clock for the table's serial cell (twin of `mtpRegimeWallMs`). Null on the first read of a block.
    fn mtpSerialWallMs(self: *Generator) ?f32 {
        if (self.mtp_serial_clock) |*c| {
            const ns = c.read();
            c.reset();
            return @as(f32, @floatFromInt(ns)) / @as(f32, std.time.ns_per_ms);
        }
        self.mtp_serial_clock = io_util.Stopwatch.init(self.timer.io);
        return null;
    }

    /// Model-level twin of `mtpAdaptiveArchEligible`: only the in-checkpoint heads were
    /// calibrated. `model_has_mtp` alone let every sidecar pack fold a cell nothing reads.
    pub fn mtpAdaptiveModelEligible(model_has_mtp: bool, checkpoint_head: bool) bool {
        return model_has_mtp and checkpoint_head;
    }

    fn mtpAdaptiveModelOk(self: *const Generator) bool {
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

    /// May `mtpSerialCaptureTick` run? It forwards the one pending token with capture, so
    /// pipelined logits or a pipelined token would publish a hidden for an uncommitted row.
    /// A runtime check: `std.debug.assert` compiles to nothing in ReleaseFast.
    pub fn mtpSerialCaptureReady(has_pending_logits: bool, has_pending_token: bool) bool {
        return !has_pending_logits and !has_pending_token;
    }

    /// Abandon the ramp back to MTP and finish this request on the serial arm. Logged once per request.
    fn mtpSerialGiveUp(self: *Generator, allocator: std.mem.Allocator, why: []const u8) !?DrafterStepResult {
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
    fn mtpSerialTick(self: *Generator, allocator: std.mem.Allocator) !?DrafterStepResult {
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
    fn mtpSerialCaptureTick(self: *Generator, allocator: std.mem.Allocator) !?DrafterStepResult {
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

    /// One row's greedy verdict. Takes the committed-argmax wrapper, so no raw
    /// argmax can reach a commit; `mtpGreedyVerdict` is the pure core.
    pub fn mtpAcceptRowGreedy(am: CommittedArgmax, len: u32, drafts: []const u32, completion: u32, max_tokens: u32) !MtpRowVerdict {
        return mtpGreedyVerdict(try am.ids(len), drafts, completion, max_tokens);
    }

    pub fn mtpGreedyVerdict(am: []const i32, drafts: []const u32, completion: u32, max_tokens: u32) MtpRowVerdict {
        var accepted: u32 = 0;
        while (accepted < drafts.len and @as(u32, @intCast(am[accepted])) == drafts[accepted]) accepted += 1;
        accepted = capAcceptedForTokenBudget(accepted, completion, max_tokens);
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

    pub const MTP_GROUP_ROWS_MAX = 32;

    pub const MtpRoundOpen = struct {
        chain: MtpPreDraft,
        tracing: bool,
        ph: io_util.Stopwatch,
        round_watch: io_util.Stopwatch,
    };

    pub const MtpRoundBegin = union(enum) { done: ?DrafterStepResult, verify: MtpRoundState, open: MtpRoundOpen };

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
        if (mtpUnsupported(self.sampling, self.logprobs_n)) return error.SpecDecodeUnsupported;

        // `--max-mtp-ctx`, checked at the top of the round against prompt + generated. Sticky
        // (context only grows), routed through the same `spec_disabled_runtime` fallback.
        if (!self.spec_disabled_runtime and
            !mtp_mod.mtpCtxWithinLimit(max_mtp_ctx, self.mtpKvLen()))
        {
            log.info(
                "  mtp=off (ctx {d} > --max-mtp-ctx {d})\n",
                .{ self.mtpKvLen(), max_mtp_ctx },
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

    fn mtpRoundAfterChain(
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
        // target is qwen3_5-family hybrid) rollback needs only the pre-verify
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
            if (!isGreedyTemperature(gen.sampling.temperature)) continue;
            st.verify_argmax = try verifyArgmax(st.verify_logits, gen.sampling.suppress_mask, s);
            _ = mlx.mlx_vector_array_append_value(eval_vec, st.verify_argmax.lazy());
        }
        try mtpGroupSampledAccept(gens[0..n], states[0..n], eval_vec, s);
        try mlx.check(mlx.mlx_async_eval(eval_vec));
    }

    /// Whether ONE filtered block can serve every sampled row of a group: the
    /// filter is per REQUEST, and one categorical cannot carry a per-row key.
    fn mtpGroupAcceptBatchable(params: []const SamplingParams) bool {
        if (params.len == 0) return false;
        return sampleRowsHomogeneous(params) and !sampleRowsAnySeed(params);
    }

    /// Phase 4 for the sampled rows of a group, built together: ONE
    /// `probsAllPositions` over their joined verify logits — the block the
    /// lm_head already projected in one go — and each row's accept graph on the
    /// group's single async eval, instead of a build-wait-build staircase per
    /// row. Declines (every row then builds its own in `mtpRoundFinish`) when
    /// the rows do not share one sampler, any row is seeded, or a row was
    /// padded past `1 + m`.
    fn mtpGroupSampledAccept(
        gens: []const *Generator,
        states: []const *MtpRoundState,
        eval_vec: mlx.mlx_vector_array,
        s: mlx.mlx_stream,
    ) !void {
        var rows: [MTP_GROUP_ROWS_MAX]usize = undefined;
        var params: [MTP_GROUP_ROWS_MAX]SamplingParams = undefined;
        var k: usize = 0;
        for (gens, states, 0..) |gen, st, i| {
            if (isGreedyTemperature(gen.sampling.temperature)) continue;
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

        const probs = try groupProbsBlock(block, params[0], s);
        defer _ = mlx.mlx_array_free(probs);
        for (rows[0..k], 0..) |i, j| {
            const st = states[i];
            const row_probs = try groupProbsRow(probs, offs[j], st.verify_len, s);
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
    fn mtpStopPrefix(drafts: []const u32, accepted: u32, eos: []const u32) struct { accepted: u32, stop: ?u32 } {
        for (drafts[0..accepted], 0..) |token, i| {
            if (isEosId(token, eos)) return .{ .accepted = @intCast(i), .stop = token };
        }
        return .{ .accepted = accepted, .stop = null };
    }

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
        const stochastic = !isGreedyTemperature(self.sampling.temperature);
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
            const probs_all = try probsAllPositions(live_logits, self.sampling, s);
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
            accepted = capAcceptedForTokenBudget(
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
        const next_pending: u32 = blk: {
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
            entries.len > 0 and entries[0].spec_state_seq.ctx != null
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
            // back safely. Unreachable on real targets (every qwen3_5-family
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
        return DrafterStepResult{
            .tokens = tokens,
            .accepted_tokens = accepted,
        };
    }

    // ── MTP adaptive depth ──
    // Unlike the drafter's binary gate, the MTP head has a useful fallback
    // BETWEEN "full depth" and "off": depth 1. Measured on Qwen3.6-27B
    // (M4 Max, greedy): creative content runs 48% per-draft at depth 2
    // (a regression vs AR) but 73% at depth 1 (1.11× AR); code runs 89% at
    // depth 2 (1.45× AR). A windowed controller demotes/promotes between
    // depths and only disables outright when even depth 1 can't pay for its
    // verify overhead.
    pub const MTP_DEPTH_WINDOW: u32 = 16; // rounds in the moving window
    pub const MTP_DEPTH_SWITCH_WARMUP: u32 = 5; // rounds before re-evaluating after a switch
    // Thresholds assume the capture-based rollback (no re-forward on partial
    // accept): a rejected draft costs ONLY its own MTP-layer + draft-head
    // pass (~2 ms), not a second trunk forward (~30-50 ms). Extra depth pays
    // whenever the marginal accept probability clears draft-cost/trunk-cost
    // ≈ 0.05-0.10, so the demote floor sits far lower than the old
    // re-forward-era 0.60 — hysteresis band keeps switch churn down.
    pub const MTP_DEMOTE_BELOW: f32 = 0.40; // per-draft rate at depth > 1 → step down
    pub const MTP_PROMOTE_ABOVE: f32 = 0.60; // per-draft rate below configured depth → step up
    // Disable floor = the MEASURED depth-1 breakeven plus margin, not a
    // quality judgment: a d1 round costs ~AR+6 ms (44 vs 38.4 ms at 8K on
    // the 27B, mtp-trace 2026-07-22) and yields (1+p) tokens, so speculation
    // pays down to p ≈ 0.15 — at p=0.45 it is +27% over AR. The old 0.50
    // floor sticky-disabled mid-request on the oQ4e head at long context
    // (window rate dips 0.45-0.55) and cratered 16K/32K ladder decode to
    // bare AR (24-26 tok/s vs oMLX's 41-47, which never fully disables).
    pub const MTP_DISABLE_BELOW: f32 = 0.20; // per-draft rate at depth 1 → disable (sticky)
    pub const MTP_PROMOTE_COOLDOWN: u32 = 32; // rounds promotion stays blocked after a demotion

    /// Target temperature at or below which the GREEDY proposal still wins:
    /// the argmax accepts with `p_target[argmax]`, high only while the target
    /// row is sharp. Measured crossover between 0.3 and 0.7 on Flash Next.
    pub const MTP_SAMPLED_PROPOSAL_MIN_TEMP: f32 = 0.5;

    /// Greedy drafts for THIS request. `top_k == 1` is a deterministic target
    /// whose filtered p is one-hot, so its argmax proposal is already optimal.
    /// A greedy target keeps the argmax proposal (the temp-0 identity contract); a sampled
    /// target proposes from the draft sampler, where `min(1, p[argmax])` collapses on a flat row.
    pub fn mtpDraftGreedyFor(target: SamplingParams) bool {
        return target.temperature <= MTP_SAMPLED_PROPOSAL_MIN_TEMP or target.top_k == 1;
    }

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

    pub fn mtpDraftTempFor(config: *const model_mod.ModelConfig) f32 {
        return if (std.mem.eql(u8, config.model_type, "qwen4_exp")) MTP_DRAFT_TEMP_QWEN4 else MTP_DRAFT_TEMP;
    }

    /// Draft-proposal sampler for a round: greedy targets keep greedy drafts
    /// (temp-0 identity contract); stochastic targets draft from the fixed
    /// sharpened distribution unless greedy is forced. The constants are not
    /// tunable: swept 2026-08-04 on the oQ4e head (prose@0.6, forced depth 3,
    /// sampled drafts) at draft temp 0.6/0.7/0.85/1.0 and per-draft
    /// acceptance was flat at 49.7-56.9% — no arm beat 0.6.
    pub fn mtpDraftSamplingFor(target: SamplingParams, force_greedy: bool, draft_temp: f32) SamplingParams {
        var d = target;
        if (force_greedy or isGreedyTemperature(target.temperature)) {
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
    pub fn mtpDraftGreedyAt(target: SamplingParams, greedy_tail: bool, step: u32) bool {
        return mtpDraftGreedyFor(target) or (greedy_tail and step > 0);
    }

    /// This request's draft sampler at chain depth `step`: the per-request
    /// proposal mode, the greedy tail and the loaded family's draft temperature,
    /// resolved in ONE place so no call site can read a different proposal than
    /// the one the verify's q came from.
    pub fn mtpDraftSampling(self: *const Generator, step: u32) SamplingParams {
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
    fn updateMtpDepth(self: *Generator, drafted: u32, accepted: u32) void {
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

    // ── MTP EV (expected-value) adaptive controller ──
    // Fixed-depth drafting is the warm-decode ceiling: at ~77% per-draft the
    // marginal chain decays with index, so one global depth wastes verify
    // width on hard stretches and leaves easy stretches (code boilerplate
    // where 8/8 accept) under-drafted. The EV controller tracks CONDITIONAL
    // per-index acceptance EMAs a[i] = P(draft i accepted | i-1 accepted) and
    // plans each round as two chunks: a base chunk `m_lo` (the static EV
    // optimum), then — when the head's own confidence on chunk A clears a
    // cost-derived threshold tau — an extension to `m_hi`. Only rounds that
    // CONSIDER extension pay the one bounded chunk-A sync; when the plan
    // collapses to m_lo == m_hi the round is byte-identical in shape to the
    // fixed-depth path (no confidence graph, no sync).
    // Disable via SUSHI_MTP_ADAPTIVE=0 (reverts to the windowed
    // fixed-depth controller above).

    /// Default depth cap when `--mtp-max-depth` is not passed (0 = auto) and the
    /// EV controller is active. 6 keeps the verify forward at seq 1+6 = 7,
    /// the split-K verify-qmm kernel's ceiling on M1-M4. Eligible M5/G17
    /// targets use MTP_ADAPTIVE_NAX_CAP instead: their measured NAX round-cost
    /// surface makes depths 7/8 profitable. Explicit depths always win.
    /// This is the DEFAULT ROW only: `mtp.adaptiveDepthCapForMachine` lowers
    /// it on chips whose verify-width cliff was measured (M1 Pro: 4).
    pub const MTP_ADAPTIVE_DEFAULT_CAP: u32 = 6;
    pub const MTP_ADAPTIVE_NAX_CAP: u32 = 8;
    /// Fraction by which a width's cost-per-position may exceed the running
    /// best before the measured ladder calls it past the cliff. 5% is well
    /// inside the gap the tile cliff opens (the M4 ladder turns up 12% at
    /// width 8) and well outside probe rep-to-rep noise, which is bounded
    /// below by keeping the MIN of the reps.
    pub const MTP_CLIFF_TOLERANCE: f32 = 0.05;
    /// One-shot guard for the per-silicon cap log (the row is a property of
    /// the machine, so it is worth saying once per process, not per request).
    var mtp_depth_cap_logged: bool = false;
    /// Rounds of legacy (fixed-depth windowed) behavior while the EMAs fill.
    /// Warmup, but converges in ROUNDS, not 43 s of offline calibration.
    pub const MTP_EV_WARMUP_ROUNDS: u32 = 10;
    /// EMA step for the per-index acceptance estimates. 0.15 demotes fast on
    /// cold streaks (5 consecutive rejects: 0.72 -> 0.32) without letting a
    /// single unlucky round move the plan.
    pub const MTP_EV_EMA_BETA: f32 = 0.15;
    /// Optimistic prior for unobserved indices. Deliberately ABOVE the
    /// measured average per-draft rate (~77%): a deep index is only ever
    /// observed when extension fires, and on this cost surface the
    /// break-even conditional acceptance for a ramp position is ~0.78 — a
    /// realistic prior would sit razor-under it and extension would never
    /// get its first trial (measured live: ext_rounds=0 on a pure-echo
    /// workload). The tau gate (only near-perfect-confidence rounds extend)
    /// plus demote-fast EMAs bound the cost of an optimistic trial to a few
    /// rounds.
    pub const MTP_EV_PRIOR: f32 = 0.85;
    /// Clamp band for the extension confidence threshold.
    pub const MTP_EV_TAU_MIN: f32 = 0.05;
    pub const MTP_EV_TAU_MAX: f32 = 0.95;
    /// Minimum base-round EV (tokens per round-cost) for the exploration
    /// horizon: below this the base itself barely beats AR, so keeping an
    /// extension position reachable would only add sync tax to a dying
    /// speculation (the sticky-disable floor is likely next anyway). 1.10
    /// keeps fully-cold EMAs collapsed on every cost profile (cold best_r:
    /// generic 1.00, G17-NAX 1.07/1.08).
    pub const MTP_EV_EXPLORE_MIN_R: f32 = 1.10;

    /// Round-cost model in units of the fixed round cost (verify-forward
    /// floor + round eval/read + commit ≈ 1.0 ≈ 32 ms on the 27B since the
    /// deferred history append). Ratios are machine-stable where absolute ms
    /// are not. Refit via SUSHI_MTP_TRACE on Qwen3.6-27B GDN (M4 Max,
    /// 2026-07-13, saturated fixed depths, same-session sweep AFTER the
    /// deferred-append round shape landed): T(1)=42.0, T(3)=62.6,
    /// T(6)=111.6, T(7)=114.9 ms — the surface is PIECEWISE: ~10.3 ms
    /// marginal per position in the flat verify region (seq <= 4), ~16.3
    /// ms/pos for positions 4-6, and position 7 nearly free (+3.3 ms —
    /// verify seq 8 rides the same row tile as 5-7), averaged into
    /// per_pos_hi. ATTRIBUTION (the GDN-vs-qmm µbench in transformer.zig,
    /// SUSHI_GDN_UBENCH=1): the ladder is ~90% qmm ROW-COUNT cost — the
    /// GDN recurrence kernel is nearly flat over verify widths (0.13→0.26 ms
    /// per dispatch, T 2→64) and contributes <1 ms/round; the earlier
    /// "GDN sequential width ramp" reading was a mis-attribution.
    /// The old linear ~1.5 ms/pos model came from a depth-6 run whose
    /// windowed controller was silently demoting underneath — never fit
    /// costs from a run whose realized m_avg you didn't check.
    pub const MtpEvCosts = struct {
        draft: f32, // one sequential MTP-head step (fwd + draft lm_head)
        per_pos_lo: f32, // marginal verify+capture per position, flat region
        per_pos_hi: f32, // ... beyond flat_max (qmm row-tile ramp)
        flat_max: u32, // last draft index in the flat verify region
        sync: f32, // the chunk-A confidence read-back
        /// First draft depth whose verify forward lands on the M5 NAX tile.
        /// Zero disables the third region. Depth k verifies k+1 rows, so the
        /// default NAX M=8 takeover starts at draft position 7.
        nax_from: u32 = 0,
        per_pos_nax: f32 = 0.0,
        /// Serial-floor cost in ms at kv ~= 0 (the probe's width-1 rung).
        /// Zero — every hand-typed table — makes the kv term below a no-op,
        /// so the tables keep behaving exactly as they did.
        floor_ms: f32 = 0,
        /// Context length the marginals above were FITTED AT. The kv term
        /// scales relative to THIS, not to zero: refit #4 was measured at 8K,
        /// so its 0.20 already contains 8K of KV read and re-scaling it from
        /// a kv~=0 floor would discount it twice. Zero = unknown, which
        /// disables the kv term for that surface rather than guessing.
        kv_ref_tokens: u32 = 0,
        /// Learned per-KV-token round cost B (ms/token), the term that makes
        /// wide speculation approach FREE at long context.
        ///
        /// A verify forward of width k reads the weights once and the KV
        /// once, both SHARED across all k query rows; only arithmetic scales
        /// with k. So T(k, L) ~= W + B*L + C(k) and T(k,L)/T(1,L) -> 1 as L
        /// grows: the optimal width RISES with context, and every constant
        /// this controller ever shipped was fitted at ONE context length.
        /// Learned online rather than probed — a boot probe measures at
        /// kv ~= 0, where B is invisible, and probing it would mean
        /// allocating a 32k KV cache purely to time it.
        kv_ms_per_token: f32 = 0,
    };
    /// 2026-08-15 refit #4, AFTER the hd-256 causal sdpa split
    /// (`splitCausalSdpa` — dense verify q 6..9 now rides the vector path,
    /// invalidating the ramp the old hi partially priced): same-session
    /// saturated ECHO sweep, Jundot oQ4e 27B @8K cold reps
    /// (--prefix-cache-entries 0), M4 Max, SUSHI_MTP_ADAPTIVE=0 forced
    /// depths, saturated (m_avg==N) trace windows only — T(1)=44.6,
    /// T(2)=51.0, T(3)=59.2, T(4)=68.2, T(6)=95.4, T(8)=142.3 ms → floor
    /// ≈ 38.2 ms. Composite marginals in floor units: k<=4 ≈ 0.20 (6.4-9.0
    /// ms/pos — flat_max moves 3 → 4, the old hi over-priced k4 at 0.34 and
    /// under-drafted it on moderate content), k5-6 ≈ 0.36 (13.6 ms/pos),
    /// k7-8 ≈ 0.62 (23.5 ms/pos — the plain-SIMD verify-qmm register cliff
    /// at M 8/9, expressed through the generic third region; only reachable
    /// when --mtp-max-depth forces past the generic cap of 6). The G17 NAX
    /// tables below predate the sdpa split — refit them on an M5 when one
    /// is available. draft/verify split not separately identifiable; only
    /// the sums enter the controller.
    /// (Refit #3, 2026-07-22, post round-pipelining, for the record:
    /// T(1)=45.4, T(2)=53.3 → .10/.10/.24@3/.01 on a ~37.9 ms floor.
    /// Refit #2, 2026-07-13, post-verify-qmm: T(1)=44.6, T(3)=54.4,
    /// T(6)=89.9 → .06/.06/.24/.02 on a ~40 ms floor.)
    /// MiMo MCG K2.5 w12 on M5 Max (2026-09-24, kv8, 4k): a serial forward 24.2 ms, verify
    /// forwards 34.9 / 44.7 / 59.8 ms at 2 / 3 / 4 rows, three pre-drafted head steps ~2.5-3 ms.
    /// Each position's own routed experts make a verify row ~.45 of a forward, flat to depth 3.
    pub const MTP_EV_MIMO_EXL3_COSTS: MtpEvCosts = .{ .draft = 0.04, .per_pos_lo = 0.44, .per_pos_hi = 0.44, .flat_max = 3, .sync = 0.02 };
    pub const MTP_EV_DEFAULT_COSTS: MtpEvCosts = .{ .draft = 0.10, .per_pos_lo = 0.10, .per_pos_hi = 0.26, .flat_max = 4, .sync = 0.01, .nax_from = 7, .per_pos_nax = 0.52 };
    /// M5 Max/G17 refit (2026-07-17), same-session saturated fixed-depth
    /// sweep after the NAX m16 verify lane landed: T(1..4) ~= 41.35 ms,
    /// T(6)=62.15 ms, T(8)=68.39 ms. In floor units this identifies
    /// draft+hi ~= .21 and draft+nax ~= .10; T(8)/T(6) is reproduced by
    /// 2.19/1.99 = 1.1005 (measured 1.1004). The profile is selected only
    /// for the calibrated dense Qwen3.6-27B homogeneous affine-4/gs-64
    /// checkpoint with its native affine-8/gs-32 sidecar and successfully
    /// built affine-3/gs-64 draft head, when the trunk lm_head routes both
    /// M=8 and M=9 through NAX; every other combination retains DEFAULT.
    pub const MTP_EV_G17_NAX_COSTS: MtpEvCosts = .{
        .draft = 0.06,
        .per_pos_lo = 0.06,
        .per_pos_hi = 0.15,
        .flat_max = 3,
        .sync = 0.02,
        .nax_from = 7,
        .per_pos_nax = 0.04,
    };
    /// M5 Max/G17 affine-4/gs-32 sidecar refit (2026-07-18). A saturated
    /// fixed-depth sweep gave T(1)=36.04, T(3)=43.01, T(6)=62.56, and
    /// T(8)=66.06 ms. The fitted composite marginals (`draft + verify`) are
    /// .107/.200/.054; depths 4 and 5 independently validate the rounded
    /// .11/.20/.05 surface after matching-baseline correction. The split
    /// between draft and verify is not separately identifiable.
    pub const MTP_EV_G17_NAX_Q4_GS32_COSTS: MtpEvCosts = .{
        .draft = 0.03,
        .per_pos_lo = 0.08,
        .per_pos_hi = 0.17,
        .flat_max = 3,
        .sync = 0.02,
        .nax_from = 7,
        .per_pos_nax = 0.02,
    };
    /// M5 Max/G17 uniform affine-4/gs-64 Qwen3.8-27B refit (2026-08-16).
    /// Temperature-gated, reversed two-pass saturated echo gave T(1)=
    /// 33.61-33.91, T(3)=38.455-38.465, T(6)=56.555-57.000, and NAX T(8)=
    /// 61.890-61.925 ms/round. NAX-off T(8)=88.535-88.610 while depth-6
    /// on/off stayed within 0.8%, isolating the M>=8 lane. A 31.4 ms fitted
    /// floor gives rounded composite marginals .075/.195/.08. The profile
    /// stays distinct because this trunk is uniform q4, not mixed q4/q5/q6.
    pub const MTP_EV_G17_NAX_Q4_GS64_COSTS: MtpEvCosts = .{
        .draft = 0.02,
        .per_pos_lo = 0.055,
        .per_pos_hi = 0.175,
        .flat_max = 3,
        .sync = 0.01,
        .nax_from = 7,
        .per_pos_nax = 0.06,
    };
    /// M5 Max/G17 uniform affine-6/gs-64 Qwen3.8-27B refit (2026-08-17).
    /// Temperature-gated, six-pass counterbalanced calibration gave median
    /// traced T(1)=44.98, T(3)=49.895, T(6)=64.495, and NAX T(8)=76.79
    /// ms/round. A 42.52 ms fitted floor yields rounded composite marginals
    /// .06/.115/.145 and reproduces T(8)/T(6): 1.815/1.525=1.1902 versus
    /// 1.1907 measured. Depth 8 was 108.86 tok/s versus 102.93 at depth 7.
    pub const MTP_EV_G17_NAX_Q6_GS64_COSTS: MtpEvCosts = .{
        .draft = 0.02,
        .per_pos_lo = 0.04,
        .per_pos_hi = 0.095,
        .flat_max = 3,
        .sync = 0.01,
        .nax_from = 7,
        .per_pos_nax = 0.125,
    };
    /// M5 Max/G17 uniform affine-8/gs-64 Qwen3.8-27B refit (2026-08-17).
    /// Five settled counterbalanced pairs found the q8-specific NAX takeover
    /// at verify M=7: depth 6 rose from 73.42 to 77.89 tok/s (+6.09%), while
    /// M=6 was -1.14%. Median traced NAX T(6)=85.94 ms and T(8)=90.68 ms over
    /// the 53.0575 ms floor give normalized 1.62/1.71. The additive controller
    /// cannot encode the one-time M7 kernel transition separately, so k=5..6
    /// are a positive .19 bridge and the subsequent k=7..8 NAX marginal is
    /// .045. This reproduces both measured endpoints without negative costs.
    /// The profile is revoked when q8 NAX is off.
    pub const MTP_EV_G17_NAX_Q8_GS64_COSTS: MtpEvCosts = .{
        .draft = 0.005,
        .per_pos_lo = 0.055,
        .per_pos_hi = 0.185,
        .flat_max = 4,
        .sync = 0.01,
        .nax_from = 7,
        .per_pos_nax = 0.04,
    };
    /// M5 Max/G17 oQ4e mixed affine q4/q5/q6, gs64 refit (2026-07-23).
    /// Saturated deterministic echo, same-session fixed widths:
    /// T(1)=34.37, T(3)=40.31, T(6)=61.20, T(8)=62.79 ms/round.
    /// Normalizing the inferred 31.40 ms floor gives composite marginals
    /// .095/.220/.025. The split below is arbitrary but positive; only
    /// `draft + per_pos_*` enters the controller. The profile is selected
    /// solely for the exact resident oQ4e layer-role fingerprint.
    pub const MTP_EV_G17_NAX_OQ4E_Q4_GS64_COSTS: MtpEvCosts = .{
        .draft = 0.02,
        .per_pos_lo = 0.075,
        .per_pos_hi = 0.20,
        .flat_max = 3,
        .sync = 0.01,
        .nax_from = 7,
        .per_pos_nax = 0.005,
    };
    /// qwen4_exp (Qwen3.8-Flash-Next 125B-A6B) on G17, M5 Max 40-core refit
    /// (2026-08-27/28). Fitted at ~8.5k context like every prior surface
    /// (a short-context fit planned too shallow at 8.5k, measured -4% —
    /// every constant this controller ever shipped was fitted at ONE
    /// context length, and 8K is the house one). Forced-depth saturated
    /// echo (the test_mtp_equivalence echo appended to ~8.3k of neutral
    /// filler, temp 0, `--prefix-cache-entries 0`, round-cost persist off),
    /// depths {1,2,3,4,6} x two reversed passes x 3 reps, medians of
    /// per-request round_ms: T(1)=25.41, T(2)=31.18, T(3)=35.70,
    /// T(4)=41.33, T(6)=53.58 ms -> 20.34 ms floor, composite
    /// (draft + per-position) marginals .257/.303 (SSE 0.28), shipped
    /// house-style below as `.draft = 0.02` plus per-position .237/.283 —
    /// composite minus draft, the same split every calibrated surface
    /// carries. Short-context fit for the record: T(1..4,6,8)=
    /// 23.61/28.12/33.91/38.87/50.16/62.70 -> floor 18.24, composite
    /// .283/.310/.344 (the .344 ships as per-position `.325`).
    /// A verify row on this arch is BYTES — a second row reads its own
    /// experts — so the marginals sit 3-5x above the dense-sidecar surfaces
    /// and there is NO NAX flattening: routed expert banks never ride the
    /// vqmm NAX lane (that is the M5 plan's grouped-expert item, and this
    /// table is its before-number). `per_pos_nax` carries the short-context
    /// k>=7 STEEPENING (+11%/pos, a measurement not a discount) and is
    /// inert under this profile's default cap of 6. sync stays nominal per
    /// the dry-spell-gate doctrine (measured chunk-boundary sync ~1.7 ms is
    /// governed by the realized-rate gate, not the prior). draft/verify
    /// split arbitrary but positive; only the sums enter the controller.
    pub const MTP_EV_G17_NAX_QWEN4_Q4_GS64_COSTS: MtpEvCosts = .{
        .draft = 0.02,
        .per_pos_lo = 0.237,
        .per_pos_hi = 0.283,
        .flat_max = 4,
        .sync = 0.02,
        .nax_from = 7,
        .per_pos_nax = 0.325,
    };

    /// 2026-09-13 M5 Max mixed4/8 refit after the single-stream verifier
    /// fast paths. Affine8 target/head KV, 3-bit draft shortlist, ~8.6k
    /// context; forced depths {1,2,3,4,5,6,8}, three echo repetitions in
    /// each of two reversed boot orders. Median full-round EMAs (ms):
    /// 24.145/28.970/34.035/38.470/44.310/51.970/61.600.
    /// Fit floor=19.037 and position slopes=4.983/8.018/4.815 ms;
    /// max relative residual 1.30%. Depth6 verifies seven positions and
    /// leaves the optimized width2..6 envelope, hence flat_max=5.
    /// Only composite marginals are identifiable; retain draft=.02 and
    /// nominal sync=.02. Runtime fingerprint and KV guards live in mtp.zig.
    /// Evidence and reproduction: notes/MTP-SINGLE-VERIFY.md.
    pub const MTP_EV_G17_NAX_QWEN4_MIXED_4_8_GS64_COSTS: MtpEvCosts = .{
        .draft = 0.02,
        .per_pos_lo = 0.241753,
        .per_pos_hi = 0.401180,
        .flat_max = 5,
        .sync = 0.02,
        .nax_from = 7,
        .per_pos_nax = 0.232929,
    };

    /// Minimum kv separation before two anchors can identify B. Below it the
    /// difference is dominated by round-to-round noise, not by the KV read.
    /// Round-end bookkeeping shared by the full- and partial-accept paths:
    /// the kv term, the regime gate and the round-cost table all read ONE
    /// inter-round wall clock (tok/s is measured between round ends, so
    /// per-round work outside the round stopwatch belongs to the width).
    fn mtpRoundEndObserve(self: *Generator, m: u32, tokens: u32, two_chunk: bool, m_lo: u32, width_trial: bool, round_ms: f32) void {
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
    fn specObserveRound(self: *Generator, width: u32, wall_ms: f32, tokens: f32, observe: bool, shape_changed: bool) void {
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
    var mtp_cost_table_cache: ?bool = null;
    pub fn mtpCostTableEnabledFromEnv(raw: ?[]const u8) bool {
        const value = raw orelse return true;
        return value.len == 0 or value[0] != '0';
    }

    fn mtpCostTableEnabled() bool {
        if (mtp_cost_table_cache) |v| return v;
        const raw: ?[]const u8 = if (std.c.getenv("SUSHI_MTP_COST_TABLE")) |p| std.mem.span(p) else null;
        const on = mtpCostTableEnabledFromEnv(raw);
        mtp_cost_table_cache = on;
        return on;
    }

    /// KV length the next verify forward reads.
    fn mtpKvLen(self: *const Generator) u32 {
        return @intCast(self.prompt_ids_owned.len + self.generated_ids.items.len);
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

    /// One round's draft plan. `m_hi > m_lo` means "pay the chunk-A sync and
    /// extend to m_hi when the chain log-confidence clears tau_ln".
    pub const MtpRoundPlan = struct {
        m_lo: u32,
        m_hi: u32,
        tau_ln: f32,
        /// A width-trial round (single-chunk at m_lo+1, measuring the round
        /// cost table): skipped by the regime gate, which compares shapes
        /// at ONE base depth.
        width_trial: bool = false,
    };

    /// Resolve the configured depth cap (`--mtp-max-depth`). 0 = auto (flag not passed):
    /// MTP_ADAPTIVE_NAX_CAP only when the EV controller and a calibrated G17
    /// cost profile are both active, MTP_ADAPTIVE_DEFAULT_CAP otherwise, and
    /// DEFAULT_DEPTH in fixed mode. Explicit values always win.
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
            .g17_nax_q8_gs32, .g17_nax_q4_gs32, .g17_nax_q4_gs64, .g17_nax_q6_gs64, .g17_nax_q8_gs64, .g17_nax_oq4e_q4_gs64 => MTP_ADAPTIVE_NAX_CAP,
            // qwen4's measured surface has no NAX region to reach: depths 7-8
            // price at .345/pos against sub-60% tail acceptance even on
            // saturated echo, so the calibrated cap keeps the default — the
            // NAX cap exists for surfaces that flatten past position 6.
            .g17_nax_qwen4_q4_gs64, .g17_nax_qwen4_mixed_4_8_gs64 => MTP_ADAPTIVE_DEFAULT_CAP,
            // Three trained heads; the depth cap clamps to them at load.
            .mimo_exl3 => MTP_ADAPTIVE_DEFAULT_CAP,
        };
    }

    pub fn resolveMtpDepthCapForProfile(configured: u32, profile: mtp_mod.MtpCostProfile) u32 {
        return mtpDepthCapForProfile(configured, mtpAdaptiveEnabled(), profile);
    }

    /// Legacy q8 boolean selector retained for source compatibility.
    pub fn mtpDepthCapFor(configured: u32, adaptive: bool, nax_profile: bool) u32 {
        const profile: mtp_mod.MtpCostProfile = if (nax_profile) .g17_nax_q8_gs32 else .generic;
        return mtpDepthCapForProfile(configured, adaptive, profile);
    }

    /// Legacy q8 boolean selector retained for source compatibility.
    pub fn resolveMtpDepthCap(configured: u32, nax_profile: bool) u32 {
        const profile: mtp_mod.MtpCostProfile = if (nax_profile) .g17_nax_q8_gs32 else .generic;
        return resolveMtpDepthCapForProfile(configured, profile);
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

    /// Smallest marginal the table may report, in floor units: a measured
    /// pair that reads a wider round CHEAPER is noise, and a non-positive
    /// marginal would make every extension free.
    pub const MTP_EV_TABLE_MIN_MARGINAL: f32 = 0.02;

    /// Where the EV plan's round costs come from: the measured round-cost
    /// table when the kv bucket (or its nearest active neighbour) has
    /// MIN_WIDTHS measured widths, else the fitted surface as the cold-start
    /// prior. The table is in ms; the plan compares ratios but has one
    /// absolute threshold (`MTP_EV_EXPLORE_MIN_R`), so the table is scaled
    /// into FLOOR UNITS at its narrowest measured width — the two sources
    /// agree there by construction and the table's slopes take over.
    pub const MtpCostSource = struct {
        costs: MtpEvCosts,
        kv_len: u32 = 0,
        table: ?*const round_cost.Table = null,
        bucket: usize = 0,
        scale: f32 = 0,

        pub fn init(costs: MtpEvCosts, kv_len: u32, table: ?*const round_cost.Table) MtpCostSource {
            var src = MtpCostSource{ .costs = costs, .kv_len = kv_len, .table = table };
            const t = table orelse return src;
            const b = t.bucketToRead(kv_len) orelse return src;
            const ref = t.narrowestMeasured(b) orelse return src;
            const ref_ms = t.measuredMs(ref, b) orelse return src;
            if (!(ref_ms > 0) or ref > mtp_mod.MAX_DEPTH) return src;
            src.bucket = b;
            src.scale = mtpEvRoundCostAt(costs, ref, false, kv_len) / ref_ms;
            return src;
        }

        pub fn fromTable(self: MtpCostSource) bool {
            return self.scale > 0;
        }

        /// Realized tokens per single-chunk round at a MEASURED width, else
        /// null (the acceptance-EMA model fills in). The model's E(6) on the
        /// M4 Max @16k said 6.85 while the measured single-6 round emitted
        /// 6.0 — the 6th draft's rejections cost a rollback the model cannot
        /// see, and the table can.
        pub fn measuredTokens(self: MtpCostSource, m: u32) ?f32 {
            if (self.scale <= 0) return null;
            return self.table.?.measuredTok(m, self.bucket);
        }

        pub fn roundCost(self: MtpCostSource, m: u32, with_sync: bool) f32 {
            if (self.scale > 0) {
                const sync: f32 = if (with_sync) self.costs.sync else 0.0;
                const t = self.table.?;
                if (t.roundMs(m, self.bucket)) |ms| return ms * self.scale + sync;
                // Past the widest measured width every extra position costs
                // max(last measured slope, prior marginal): continuous with
                // the table, never more optimistic than the prior (a shallow
                // measured slope run upward let the plan race to 8 unmeasured).
                if (t.widestMeasured(self.bucket)) |w| {
                    if (m > w) {
                        const slope = (t.lastSlope(self.bucket) orelse 0.0) * self.scale;
                        var c = t.measuredMs(w, self.bucket).? * self.scale;
                        var k: u32 = w + 1;
                        while (k <= m) : (k += 1) {
                            c += @max(slope, mtpEvMarginalCostAt(self.costs, k, self.kv_len));
                            // One sample is evidence for WORSE, never for
                            // cheaper: an untrusted cell floors the cost.
                            if (t.rawMs(k, self.bucket)) |raw| c = @max(c, raw * self.scale);
                        }
                        return c + sync;
                    }
                }
            }
            return mtpEvRoundCostAt(self.costs, m, with_sync, self.kv_len);
        }

        pub fn marginal(self: MtpCostSource, k: u32) f32 {
            if (self.scale > 0 and k >= 1) {
                return @max(self.roundCost(k, false) - self.roundCost(k - 1, false), MTP_EV_TABLE_MIN_MARGINAL);
            }
            return mtpEvMarginalCostAt(self.costs, k, self.kv_len);
        }
    };

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

    /// Per-phase wall-time accumulator behind SUSHI_MTP_TRACE=1. Pure
    /// bookkeeping; `nextMtp` stamps phases with a Stopwatch and emits one
    /// summary line every LOG_EVERY rounds. Zero cost when the env is absent
    /// (every stamp is guarded on the cached env check).
    pub const MtpTrace = struct {
        pub const LOG_EVERY: u32 = 32;
        pub const Phase = enum(u4) { draft, sync, ext, verify, corr, eval, hist, commit, predraft, gap };
        pub const N_PHASES = @typeInfo(Phase).@"enum".field_names.len;
        // Sub-laps attribute time inside a phase and are not a partition of it: a group's build lap is
        // stamped into every row while each row's `verify` carries only its own share.
        pub const Sub = enum(u3) { build, ple, ple_sync, dispatch, wait, chain };
        pub const N_SUBS = @typeInfo(Sub).@"enum".field_names.len;

        rounds: u32 = 0,
        ns: [N_PHASES]u64 = @splat(0),
        sub_ns: [N_SUBS]u64 = @splat(0),
        drafted: u64 = 0,
        accepted: u64 = 0,
        extended: u32 = 0,
        /// Per draft index: rounds that drafted index i / accepted it.
        drafted_idx: [mtp_mod.MAX_DEPTH]u32 = @splat(0),
        accepted_idx: [mtp_mod.MAX_DEPTH]u32 = @splat(0),

        pub fn add(self: *MtpTrace, phase: Phase, dur_ns: u64) void {
            self.ns[@backingInt(phase)] += dur_ns;
        }

        pub fn addSub(self: *MtpTrace, sub: Sub, dur_ns: u64) void {
            self.sub_ns[@backingInt(sub)] += dur_ns;
        }

        /// Close one round; true when a summary line is due (caller logs,
        /// then calls reset()).
        pub fn endRound(self: *MtpTrace, drafted_n: u32, accepted_n: u32, was_extended: bool) bool {
            self.rounds += 1;
            self.drafted += drafted_n;
            self.accepted += accepted_n;
            if (was_extended) self.extended += 1;
            var i: usize = 0;
            while (i < drafted_n and i < mtp_mod.MAX_DEPTH) : (i += 1) {
                self.drafted_idx[i] += 1;
                if (i < accepted_n) self.accepted_idx[i] += 1;
            }
            return self.rounds >= LOG_EVERY;
        }

        /// `a0/a1/...` acceptance per drafted index (only indices ever drafted).
        pub fn accIdxStr(self: *const MtpTrace, buf: []u8) []const u8 {
            var w: std.Io.Writer = .fixed(buf);
            var i: usize = 0;
            while (i < mtp_mod.MAX_DEPTH and self.drafted_idx[i] > 0) : (i += 1) {
                w.print("{s}{d:.2}", .{ if (i == 0) "" else "/", @as(f64, @floatFromInt(self.accepted_idx[i])) / @as(f64, @floatFromInt(self.drafted_idx[i])) }) catch break;
            }
            return w.buffered();
        }

        pub fn avgMs(self: *const MtpTrace, phase: Phase) f64 {
            if (self.rounds == 0) return 0.0;
            return @as(f64, @floatFromInt(self.ns[@backingInt(phase)])) /
                (@as(f64, @floatFromInt(self.rounds)) * 1e6);
        }

        pub fn subAvgMs(self: *const MtpTrace, sub: Sub) f64 {
            if (self.rounds == 0) return 0.0;
            return @as(f64, @floatFromInt(self.sub_ns[@backingInt(sub)])) /
                (@as(f64, @floatFromInt(self.rounds)) * 1e6);
        }

        pub fn totalAvgMs(self: *const MtpTrace) f64 {
            if (self.rounds == 0) return 0.0;
            var total: u64 = 0;
            for (self.ns) |v| total += v;
            return @as(f64, @floatFromInt(total)) / (@as(f64, @floatFromInt(self.rounds)) * 1e6);
        }

        // Profilers parse the phase summary as one run from `rounds=` to `acc_idx=`; new fields go behind it.
        pub const LINE_FMT = "  [mtp-trace] rounds={d} avg_ms draft={d:.2} sync={d:.2} ext={d:.2} verify={d:.2} corr={d:.2} eval={d:.2} hist={d:.2} commit={d:.2} predraft={d:.2} gap={d:.2} total={d:.2} | m_avg={d:.2} acc_avg={d:.2} ext_rate={d:.2} acc_idx={s} | sub_ms build={d:.2} ple={d:.2} ple_sync={d:.2} dispatch={d:.2} wait={d:.2} chain={d:.2}\n";

        pub fn reset(self: *MtpTrace) void {
            self.* = .{};
        }
    };

    pub const SubLap = struct {
        watch: ?io_util.Stopwatch,

        pub fn start(on: bool, io: std.Io) SubLap {
            return .{ .watch = if (on) io_util.Stopwatch.init(io) else null };
        }

        pub fn read(self: SubLap) ?u64 {
            return if (self.watch) |w| w.read() else null;
        }

        pub fn stop(self: SubLap, trace: *MtpTrace, sub: MtpTrace.Sub) void {
            if (self.read()) |ns| trace.addSub(sub, ns);
        }
    };

    /// Per-phase wall-time accumulator behind SUSHI_DFLASH_TRACE=1.
    /// `nextDflash` stamps phases with a Stopwatch and emits one summary line
    /// every LOG_EVERY rounds. Zero cost when the env is absent.
    pub const DflashTrace = struct {
        pub const LOG_EVERY: u32 = 16;
        pub const Phase = enum(u4) { assist, head, verify, accept, append, gap };
        pub const N_PHASES = @typeInfo(Phase).@"enum".field_names.len;

        rounds: u32 = 0,
        ns: [N_PHASES]u64 = @splat(0),
        accepted: u64 = 0,

        pub fn add(self: *DflashTrace, phase: Phase, dur_ns: u64) void {
            self.ns[@backingInt(phase)] += dur_ns;
        }

        /// Close one round; true when a summary line is due (caller logs,
        /// then calls reset()).
        pub fn endRound(self: *DflashTrace, accepted_n: u32) bool {
            self.rounds += 1;
            self.accepted += accepted_n;
            return self.rounds >= LOG_EVERY;
        }

        pub fn avgMs(self: *const DflashTrace, phase: Phase) f64 {
            if (self.rounds == 0) return 0.0;
            return @as(f64, @floatFromInt(self.ns[@backingInt(phase)])) /
                (@as(f64, @floatFromInt(self.rounds)) * 1e6);
        }

        pub fn totalAvgMs(self: *const DflashTrace) f64 {
            if (self.rounds == 0) return 0.0;
            var total: u64 = 0;
            for (self.ns) |v| total += v;
            return @as(f64, @floatFromInt(total)) / (@as(f64, @floatFromInt(self.rounds)) * 1e6);
        }

        pub fn reset(self: *DflashTrace) void {
            self.* = .{};
        }
    };

    var dflash_trace_cache: ?bool = null;
    fn dflashTraceEnabled() bool {
        if (dflash_trace_cache) |v| return v;
        const on = readEnvBool("SUSHI_DFLASH_TRACE");
        dflash_trace_cache = on;
        return on;
    }

    /// Close one traced dflash round; emits + resets at the cadence.
    fn dflashTraceRoundEnd(self: *Generator, accepted: u32) void {
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

    /// Adaptive (EV) controller gate — DEFAULT ON. SUSHI_MTP_ADAPTIVE=0
    /// reverts to the fixed-depth windowed controller for same-boot A/Bs.
    var mtp_adaptive_cache: ?bool = null;

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

    var mtp_trace_cache: ?bool = null;
    fn mtpTraceEnabled() bool {
        if (mtp_trace_cache) |v| return v;
        const on = readEnvBool("SUSHI_MTP_TRACE");
        mtp_trace_cache = on;
        return on;
    }

    pub fn mtpEvCostsForProfile(profile: mtp_mod.MtpCostProfile) MtpEvCosts {
        return switch (profile) {
            .generic => MTP_EV_DEFAULT_COSTS,
            .g17_nax_q8_gs32 => MTP_EV_G17_NAX_COSTS,
            .g17_nax_q4_gs32 => MTP_EV_G17_NAX_Q4_GS32_COSTS,
            .g17_nax_q4_gs64 => MTP_EV_G17_NAX_Q4_GS64_COSTS,
            .g17_nax_q6_gs64 => MTP_EV_G17_NAX_Q6_GS64_COSTS,
            .g17_nax_q8_gs64 => MTP_EV_G17_NAX_Q8_GS64_COSTS,
            .g17_nax_oq4e_q4_gs64 => MTP_EV_G17_NAX_OQ4E_Q4_GS64_COSTS,
            .g17_nax_qwen4_q4_gs64 => MTP_EV_G17_NAX_QWEN4_Q4_GS64_COSTS,
            .g17_nax_qwen4_mixed_4_8_gs64 => MTP_EV_G17_NAX_QWEN4_MIXED_4_8_GS64_COSTS,
            .mimo_exl3 => MTP_EV_MIMO_EXL3_COSTS,
        };
    }

    /// Extension dry-spell gate constants: after MTP_EXT_DRY_ROUNDS
    /// consecutive extension-CONSIDERED rounds whose confidence gate never
    /// cleared, consideration collapses to single-chunk for
    /// MTP_EXT_DRY_COOLDOWN rounds, then re-opens for a fresh trial.
    /// Rationale (2026-07-22, post-round-pipelining sweep): a two-chunk
    /// round pays a REAL mid-pipeline cost the EV surface can't see — the
    /// chunk-A boundary sync fires ~0.3 ms after the pre-draft dispatch, so
    /// it blocks on the still-running head chain + confidence graphs
    /// (measured sync 0.5-3.7 ms/round at ext_rate 0.00; EV default lost
    /// 3-6 tok/s at 0.5K to fixed depths from this tax alone). The gate is
    /// fed by the REALIZED extension rate, never by priors — the war-story
    /// failure mode (docs/gotchas/engine-mlx.md, ext_rounds=0 on echo) was
    /// a prior-fed expected-gain gate blocking the FIRST trial; this one
    /// guarantees a fresh 16-round trial every 48 rounds, and a single
    /// extension firing resets the streak entirely. Worst-case waste on a
    /// permanently-dry workload: a third of rounds pay the sync. The
    /// cooldown is deliberately SHORT relative to a typical request (a
    /// 160-token echo ≈ 70 rounds): a 64-round blackout swallowed the whole
    /// echoing stretch after a dry preamble and re-created the very
    /// ext_rounds=0 regression this design avoids (caught by
    /// tests/test_mtp_equivalence.sh).
    pub const MTP_EXT_DRY_ROUNDS: u32 = 16;
    pub const MTP_EXT_DRY_COOLDOWN: u32 = 32;
    /// Floor for the cost-aware dry threshold. Even the most expensive sync
    /// tolerates this many dry considered rounds before a cooldown, so the
    /// worst-case fresh-trial cadence (MTP_EXT_DRY_MIN + MTP_EXT_DRY_COOLDOWN)
    /// still fits inside a typical echo stretch (~70 rounds) — the horizon is
    /// never closed regardless of measured cost.
    pub const MTP_EXT_DRY_MIN: u32 = 3;
    /// Tolerated cumulative chunk-A sync per dry exploration burst, expressed
    /// as a fraction of one round. The cost-aware threshold is
    /// SYNC_BUDGET / (sync_ms/round_ms): a costlier sync backs off sooner, a
    /// near-free one keeps the full MTP_EXT_DRY_ROUNDS budget. Calibrated so
    /// the measured 8K operating point (sync ~2.4 ms of a ~45 ms round,
    /// frac ~0.053) lands near a ~15% exploration duty (threshold ~6, cooldown
    /// 32) — matching oMLX's duty-bounded probing.
    pub const MTP_EXT_SYNC_BUDGET: f32 = 0.30;

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

    /// EMA weight for the live round/sync cost EMAs. Slower than the
    /// acceptance EMA (cost drifts with context/thermal, not content) and
    /// paired with the seed-on-first-sample rule so warmup fills them cleanly.
    pub const MTP_EV_COST_BETA: f32 = 0.10;

    /// Fold a nanosecond wall-time sample into a millisecond EMA. Seeds on the
    /// first sample (prev <= 0). Pure — the live-cost measurement plumbing is
    /// just Stopwatch reads feeding this.
    pub fn mtpEmaMs(prev_ms: f32, sample_ns: u64) f32 {
        const sample_ms = @as(f32, @floatFromInt(sample_ns)) / 1.0e6;
        if (prev_ms <= 0.0) return sample_ms;
        return prev_ms + MTP_EV_COST_BETA * (sample_ms - prev_ms);
    }

    /// Measured ms-per-emitted-token of the two round SHAPES the EV plan can
    /// take: a two-chunk plan (extension considered, sync paid whether or not
    /// it fires) against a single-chunk plan at m_lo. Throughput is tokens
    /// over round cost, and both halves are observable per round — no cost
    /// surface, no units. The hand-fitted marginals can only say whether the
    /// extension's VERIFY pays; they cannot see that on an M4 base the whole
    /// two-chunk round runs 17.4 ms/token against 14.4 single (cap 5+ lost
    /// 17% to cap 4 on echo) while on an M4 Max the same shape is a wash.
    pub const MtpRegime = struct {
        two_ms: f32 = 0,
        two_tok: f32 = 0,
        two_m: u32 = 0,
        two_n: u32 = 0,
        one_ms: f32 = 0,
        one_tok: f32 = 0,
        one_m: u32 = 0,
        one_n: u32 = 0,
        /// Shape of the previous round: a round whose shape differs from its
        /// predecessor is a TRANSITION and is not observed. The minority shape
        /// was only ever measured on transition rounds and read 5-7% slow
        /// (M4 Max 27B @16k interleaved 13.65 vs 12.9 ms/tok; homogeneous arms
        /// 13.0 vs 13.2) — the verify width change is a one-off cost.
        last_two: ?bool = null,
        /// Trial schedule: the minority shape runs for rounds [trial_start,
        /// trial_end) and the next block starts at next_trial. Explicit, not
        /// `idx % period` — the period moves with the EMAs, and a block's own
        /// observation moved it by one per round so `idx % period` stayed
        /// inside the block: M4 base v4 ran 14 of 72 rounds as trials
        /// against 7 on v3 and lost 2-4%.
        sched_verdict: ?bool = null,
        trial_end: u32 = 0,
        next_trial: u32 = 0,
        /// Diagnostics for `[spec-stats]`: the round the first verdict formed
        /// at, and trial blocks started — splits ext_rounds into pre-verdict
        /// (seeding) and scheduled exploration.
        verdict_round: u32 = 0,
        trials: u32 = 0,
        /// Idempotency: `mtpRegimeForce` mutates the schedule, and
        /// `mtpRoundPlan` has two call sites (pre-draft at the previous
        /// round's tail, or the round's own entry). Asking twice for the
        /// same round answers the same and advances nothing.
        last_idx: ?u32 = null,
        last_force: ?bool = null,
    };

    /// The worse regime still runs one round in this many (at least) so its
    /// EMAs keep tracking the live workload (context grows, thermals move).
    pub const MTP_REGIME_EXPLORE_PERIOD: u32 = 8;
    /// Exploration drag the period is sized to: a shape G worse than the
    /// other, run once in G/DRAG rounds, costs ~DRAG of throughput. M4 base
    /// measured two-chunk 26% worse; at 1-in-8 that was a 2% structural
    /// drag that kept cap 5 from cap-4 parity (52.3 vs 54.6).
    pub const MTP_REGIME_EXPLORE_DRAG: f32 = 0.01;
    pub const MTP_REGIME_EXPLORE_PERIOD_MAX: u32 = 128;
    /// A trial is a BLOCK of consecutive rounds: the first is the transition
    /// (unobserved), the second is the steady-state measurement.
    pub const MTP_REGIME_EXPLORE_BLOCK: u32 = 2;

    /// The running mean over a shape's first 1/beta samples, the cost EMA after, so no single
    /// round (an all-rejected one emits 1 token) outweighs the rest.
    fn regimeFold(prev: f32, sample: f32, n: u32) f32 {
        const w = @max(MTP_EV_COST_BETA, 1.0 / @as(f32, @floatFromInt(n)));
        return prev + w * (sample - prev);
    }

    /// Wall time since the previous round ended (the first round reads its
    /// own stopwatch), so inter-round work is charged to the shape.
    fn mtpRegimeWallMs(self: *Generator, round_ms: f32) f32 {
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

    /// The minority shape is measured from interleaved rounds and reads a
    /// few percent slow (M4 Max 27B @16k: two-chunk 13.4 ms/tok interleaved
    /// vs 13.0 homogeneous), so a throttle needs a margin the noise cannot
    /// cross; the M4 base loss it exists for is 21%.
    pub const MTP_REGIME_MARGIN: f32 = 0.05;
    /// Rounds each shape is measured over before a verdict: a round's tokens run 1..m+1, so a
    /// shape judged on one round is judged on its acceptance luck.
    pub const MTP_REGIME_MIN_SAMPLES: u32 = 4;

    fn mtpRegimeMeasured(n: u32) bool {
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

    fn mtpRegimeForceAt(r: *MtpRegime, round_idx: u32) ?bool {
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

    // Adaptive serial: the EV controller picks the best depth but never compares a round with
    // the serial step it replaces, and the acceptance floor is a model fitted where a verify
    // row costs ~0.1 of a forward. On qwen4_exp a verify row is bytes (62.7k prose: 47-58 tok/s
    // speculative vs 55 serial). So the two measured prices are compared per KV bucket:
    //     planned ms/token = roundMs(m_lo) / E[tokens per round at m_lo]
    //     serial  ms/token = the bucket's measured plain-decode token
    // Any missing input is `.undecided`. The only cost a request may pay is a bounded serial
    // probe for a bucket that has never seen a serial token.

    /// A round worse than serial by less than this is a tie.
    pub const MTP_ADAPTIVE_MARGIN: f32 = 0.05;
    /// Consecutive rounds that must vote serial.
    pub const MTP_ADAPTIVE_CONFIRM: u32 = 3;
    /// Serial ticks one probe runs; `MTP_ADAPTIVE_PROBE_WARM` of them are discarded.
    pub const MTP_ADAPTIVE_PROBE_TOKENS: u32 = 8;
    pub const MTP_ADAPTIVE_PROBE_WARM: u32 = 2;
    /// KV below which the switch does not exist: no vote, no probe. Rounds lose to serial from
    /// the 32-64k bucket up; below that the probes cost more than a switch could buy.
    pub const MTP_ADAPTIVE_MIN_KV: u32 = 32768;
    /// MiMo's verify rows stay cheap until its global layers read many keys, so below this its
    /// rounds beat serial even on prose and a probe could only cost.
    pub const MTP_ADAPTIVE_MIN_KV_MIMO: u32 = 65536;

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

    fn mtpAdaptiveBucket(self: *const Generator, kv_len: u32) usize {
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
    fn mtpAdaptiveSyncWindow(self: *Generator, prev_bucket: ?usize, prev_arm: MtpAdaptiveArm) void {
        if (mtpAdaptiveRegimeMoved(prev_bucket, prev_arm, self.mtp_adaptive.bucket, self.mtp_adaptive.arm)) {
            self.mtp_price.reset();
        }
    }

    /// Rounds in the realized price window. Full-window-only: a restore's warmup rounds must not vote.
    pub const MTP_PRICE_WINDOW: u32 = 16;

    /// This request's realized ms per emitted token over a trailing window of speculative
    /// rounds. Width trials are skipped, extension rounds kept. A new request starts empty.
    pub const MtpPriceWindow = struct {
        ms: [MTP_PRICE_WINDOW]f32 = @splat(0),
        tok: [MTP_PRICE_WINDOW]f32 = @splat(0),
        idx: u32 = 0,
        filled: u32 = 0,

        pub fn reset(self: *MtpPriceWindow) void {
            self.idx = 0;
            self.filled = 0;
        }

        pub fn observe(self: *MtpPriceWindow, round_ms: f32, emitted: f32, width_trial: bool) void {
            if (width_trial) return;
            if (!std.math.isFinite(round_ms) or !(round_ms > 0)) return;
            if (!std.math.isFinite(emitted) or !(emitted > 0)) return;
            const i = self.idx % MTP_PRICE_WINDOW;
            self.ms[i] = round_ms;
            self.tok[i] = emitted;
            self.idx +%= 1;
            if (self.filled < MTP_PRICE_WINDOW) self.filled += 1;
        }

        /// Realized ms per emitted token, or null until the window is full. Summed, not averaged per round.
        pub fn msPerTok(self: *const MtpPriceWindow) ?f32 {
            if (self.filled < MTP_PRICE_WINDOW) return null;
            var ms: f32 = 0;
            var tok: f32 = 0;
            for (self.ms, self.tok) |m, t| {
                ms += m;
                tok += t;
            }
            if (!(ms > 0) or !(tok > 0)) return null;
            const v = ms / tok;
            return if (std.math.isFinite(v)) v else null;
        }
    };

    pub const MtpAdaptiveVote = enum { undecided, mtp, serial };
    pub const MtpAdaptiveArm = enum { undecided, mtp, serial };
    pub const MtpAdaptiveAction = enum { none, to_serial, to_mtp };
    /// Where a serial block sits in its ramp back to `nextMtp`'s entry invariant.
    pub const MtpSerialExit = enum { none, drain, capture };

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

    /// The standing choice and the streak behind it. Pure.
    pub const MtpAdaptive = struct {
        arm: MtpAdaptiveArm = .undecided,
        confirm: u32 = 0,
        bucket: ?usize = null,
        /// A re-entry was refused because the head's position bookkeeping is out of sync; latched.
        reentry_declined: bool = false,
        /// The switch left MTP on a module-owned head, so the arm is sticky: no re-entry, probe
        /// or round for the rest of the request. That is what buys the head's release.
        sticky_serial: bool = false,
        /// Round index of the last `round()`: `mtpRoundPlan` has two call sites per round.
        last_round: ?u32 = null,
        switches: u32 = 0,
        /// The request's own KV bucket at its first serial tick. The crossing watches it: the
        /// read bucket resolves a never-measured bucket back onto the one the switch read.
        serial_own: ?usize = null,

        pub fn round(self: *MtpAdaptive, round_idx: u32, bucket: usize, vote: MtpAdaptiveVote, need: u32) MtpAdaptiveAction {
            if (self.sticky_serial) return .none;
            if (self.last_round) |r| {
                if (r == round_idx) return .none;
            }
            self.last_round = round_idx;
            // A crossing invalidates the streak.
            if (self.bucket == null or self.bucket.? != bucket) {
                self.bucket = bucket;
                self.confirm = 0;
            }
            switch (vote) {
                .undecided => {
                    self.confirm = 0;
                    return .none;
                },
                .mtp => {
                    self.confirm = 0;
                    self.arm = .mtp;
                    return .none;
                },
                .serial => {
                    self.confirm += 1;
                    if (self.confirm < @max(@as(u32, 1), need)) return .none;
                    self.confirm = 0;
                    self.arm = .serial;
                    self.serial_own = null;
                    self.switches += 1;
                    return .to_serial;
                },
            }
        }

        /// Latch a refused re-entry: back to the serial arm, never ask again for this request.
        pub fn declineReentry(self: *MtpAdaptive) void {
            self.arm = .serial;
            self.reentry_declined = true;
        }

        /// Make the serial arm permanent for this request (a `to_serial` on a module-owned head).
        pub fn stickSerial(self: *MtpAdaptive) void {
            self.arm = .serial;
            self.sticky_serial = true;
        }

        /// `serialTick` plus a crossing of the request's own bucket (`own`).
        pub fn serialTickAt(self: *MtpAdaptive, bucket: usize, own: usize) MtpAdaptiveAction {
            if (self.arm == .serial and !self.sticky_serial and !self.reentry_declined) {
                const at = self.serial_own orelse own;
                self.serial_own = at;
                if (own != at) return self.reopen(bucket);
            }
            return self.serialTick(bucket);
        }

        pub fn serialTick(self: *MtpAdaptive, bucket: usize) MtpAdaptiveAction {
            // Sticky outranks both re-entry triggers, the bucket crossing included.
            if (self.sticky_serial) return .none;
            if (self.reentry_declined) return .none;
            if (self.arm != .serial) return .none;
            if (self.bucket) |b| {
                if (b == bucket) return .none; // only a bucket crossing re-opens the decision
            }
            return self.reopen(bucket);
        }

        fn reopen(self: *MtpAdaptive, bucket: usize) MtpAdaptiveAction {
            self.bucket = bucket;
            self.confirm = 0;
            self.serial_own = null;
            self.arm = .undecided;
            self.switches += 1;
            return .to_mtp;
        }
    };

    /// Whole-mechanism kill switch (`SUSHI_MTP_ADAPTIVE_SERIAL=0`): no vote, no probe, no
    /// serial fold. Independent of `SUSHI_MTP_ADAPTIVE` (the depth controller's lever).
    var mtp_adaptive_serial_cache: ?bool = null;
    /// Default on; only an exact "0" turns it off.
    pub fn mtpAdaptiveSerialEnabledFromEnv(raw: ?[]const u8) bool {
        const value = raw orelse return true;
        return !std.mem.eql(u8, value, "0");
    }

    fn mtpAdaptiveSerialEnabled() bool {
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
    fn mtpHeadDriftNow(self: *const Generator) ?i64 {
        if (self.mtp_cache == null) return null;
        const mc = &self.mtp_cache.?;
        return switch (mc.*) {
            .qwen, .mimo => 0,
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
    fn mtpAdaptiveHeadMayResume(self: *const Generator) bool {
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
    fn mtpMaybeReleaseModuleHead(self: *Generator) void {
        if (!stickyReleaseNow(self.mtp_adaptive.sticky_serial, self.mtp_head_released)) return;
        self.mtp_head_released = true;
        log.info(
            "  [mtp] sticky-serial: module head released for the rest of the request (kv {d}) — other slots may claim it\n",
            .{self.mtpKvLen()},
        );
    }

    /// The heads the adaptive serial switch was calibrated for: qwen4_exp's and MiMo's; a
    /// sidecar pack has a different verify surface. Not `moduleOwned()`: different question.
    pub fn mtpAdaptiveHeadEligible(tag: std.meta.Tag(MtpHeadRef)) bool {
        return tag == .qwen4 or tag == .mimo;
    }

    fn mtpAdaptiveArchEligible(self: *const Generator) bool {
        const head = self.mtp orelse return false;
        return mtpAdaptiveHeadEligible(std.meta.activeTag(head));
    }

    fn mtpAdaptiveSerialStep(self: *Generator, m_lo: u32, kv_len: u32) bool {
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
        const sharp = !isGreedyTemperature(self.mtpDraftSampling(0).temperature);
        return group_cost.GroupShape.samplingMode(head.canRerankDrafts(), false, sharp, !isGreedyTemperature(self.sampling.temperature));
    }

    fn mtpPlannerObserve(self: *Generator, observed: u32, accepted: u32) void {
        if (!group_planner.enabled() or !self.mtp_planner_owned) return;
        self.mtp_planner_history.observe(observed, @min(observed, accepted));
        self.mtp_planner_spec_rounds += 1;
        self.mtp_hidden_stale = false;
        self.mtp_planner_recovery.observe(self.completion_tokens, self.mtp_planner_recovering);
        if (self.mtp_planner_probe and !self.mtp_planner_recovering) self.mtp_planner_probes +|= 1;
    }

    fn mtpRoundPlan(self: *Generator) MtpRoundPlan {
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

    fn mtpRoundPlanTraced(self: *Generator) MtpRoundPlan {
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

    fn mtpRoundPlanInner(self: *Generator) MtpRoundPlan {
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

    /// Width-trial schedule: same shape as the regime's (explicit
    /// trial_end/next_trial, a 2-round block — transition then measurement),
    /// idempotent per round because `mtpRoundPlan` has two call sites.
    pub const MtpWidthTrial = round_cost.TrialSchedule;

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
    fn mtpFloorDisableObserve(
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
    fn updateMtpEvRound(self: *Generator, drafted: u32, accepted: u32) void {
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
    fn mtpTraceRoundEnd(self: *Generator, m: u32, accepted: u32, m_lo: u32) void {
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

    /// Returns the next token ID, or null when generation is finished.
    ///
    /// Pipeline architecture (matches mlx-lm's generator pattern):
    ///
    ///   The KEY to effective pipelining is the ORDER of operations:
    ///   1. Build next step's lazy graph (depends on pending lazy token)
    ///   2. async_eval the next graph — GPU computes pending token as a DEPENDENCY,
    ///      then continues with the forward pass
    ///   3. eval(pending_token) — returns INSTANTLY since GPU already computed it
    ///   4. Return the token while GPU continues computing the next forward pass
    ///
    ///   This mirrors mlx-lm's: _step(y) → async_eval(next_y) → yield y.item()
    ///   where y.item() is instant because async_eval forced y's computation.
    pub fn next(self: *Generator, allocator: std.mem.Allocator) !?u32 {
        var tick_prof = TickProf.begin();
        defer tick_prof.finish();
        if (self.done) return null;
        if (self.sampling.constraint != null) return self.nextConstrained(allocator);

        // Transition shim: speculative-decode paths may exit with
        // `next_token_id` set but `pending_logits` unset (drafter's exit
        // invariant is "t1 NOT in cache" — its hand-off to `next()` would
        // otherwise crash on the slow path which assumes pending_logits is
        // always lazily seeded). When we observe that state, synchronously
        // forward `[next_token_id]` to seed `pending_logits` so the fast
        // path picks up cleanly. the speculative exit state already matches `next()`'s
        // invariant, so this only fires for drafter→next runtime-gate
        // fallbacks (and any future spec methods that share drafter's shape).
        if (!self.has_pending_logits and !self.has_pending_token and
            self.step < self.max_tokens and !self.samplesSync())
        {
            const tok_i32: i32 = @intCast(self.next_token_id);
            const tok_shape = [_]c_int{ 1, 1 };
            const tok_input = mlx.mlx_array_new_data(&tok_i32, &tok_shape, 2, .int32);
            defer _ = mlx.mlx_array_free(tok_input);
            const t_fwd = tick_prof.mark();
            self.pending_logits = try self.xfm.forwardWith(&self.ctx, tok_input);
            tick_prof.add(.fwd, t_fwd);
            self.has_pending_logits = true;
        }

        // ── Phase 1: Build and submit the NEXT step FIRST ──
        // This forces the GPU to compute the pending token as a dependency,
        // so when we eval it in Phase 2, it's already ready.
        if (self.has_pending_logits and !self.samplesSync() and self.step + 1 < self.max_tokens and !self.callForcePending()) {
            const step_logits = self.pending_logits;
            self.has_pending_logits = false;

            const t_sample = tick_prof.mark();
            const lazy_token = blk: {
                defer _ = mlx.mlx_array_free(step_logits);
                break :blk try self.sampleLazy(step_logits, if (self.has_pending_token)
                    .{ .pending = self.pending_token }
                else
                    .{ .decided = &.{self.next_token_id} });
            };
            tick_prof.add(.sample, t_sample);

            var adopted = false;
            defer if (!adopted) {
                _ = mlx.mlx_array_free(lazy_token);
            };
            const t_fwd = tick_prof.mark();
            if (lazyForward(self.xfm, &self.ctx, lazy_token)) |next_logits| {
                tick_prof.add(.fwd, t_fwd);
                defer if (!adopted) {
                    _ = mlx.mlx_array_free(next_logits);
                };
                const arr = [_]mlx.mlx_array{ lazy_token, next_logits };
                const vec = mlx.mlx_vector_array_new_data(&arr, 2);
                const t_eval = tick_prof.mark();
                _ = mlx.mlx_async_eval(vec);
                tick_prof.add(.eval, t_eval);
                _ = mlx.mlx_vector_array_free(vec);

                // NOW resolve the pending token — GPU already computed it as a
                // dependency of the graph we just submitted. Should be instant.
                const t_res = tick_prof.mark();
                try self.resolvePendingToken();
                tick_prof.add(.resolve, t_res);

                // Check stop conditions on the resolved token
                if (try self.checkStop()) return null;

                const token = self.next_token_id;
                self.advanceStep(1);
                try self.generated_ids.append(allocator, token);

                // Store new pending state
                self.pending_token = lazy_token;
                self.has_pending_token = true;
                self.pending_logits = next_logits;
                self.has_pending_logits = true;
                adopted = true;

                return token;
            } else |_| {
                tick_prof.add(.fwd, t_fwd);
                // lazyForward failed — fall through to slow path
                try mlx.check(mlx.mlx_array_eval(lazy_token));
                var val: i32 = 0;
                try mlx.check(mlx.mlx_array_item_int32(&val, lazy_token));
                _ = mlx.mlx_array_free(lazy_token);
                adopted = true;
                self.next_token_id = @intCast(val);
                self.has_pending_token = false;
            }
        }

        // ── Phase 2: Slow path (first token, last token, logprobs, or pipeline miss) ──
        const t_res = tick_prof.mark();
        try self.resolvePendingToken();
        tick_prof.add(.resolve, t_res);

        if (try self.checkStop()) return null;

        const token = self.next_token_id;
        self.advanceStep(1);
        try self.generated_ids.append(allocator, token);

        const step_logits = if (self.has_pending_logits) blk: {
            const logits = self.pending_logits;
            self.has_pending_logits = false;
            break :blk logits;
        } else blk: {
            const tok_i32: i32 = @intCast(token);
            const tok_shape = [_]c_int{ 1, 1 };
            const tok_input = mlx.mlx_array_new_data(&tok_i32, &tok_shape, 2, .int32);
            defer _ = mlx.mlx_array_free(tok_input);
            const t_fwd = tick_prof.mark();
            const out = try self.xfm.forwardWith(&self.ctx, tok_input);
            tick_prof.add(.fwd, t_fwd);
            break :blk out;
        };

        // Logprobs and penalties: fully synchronous
        if (self.samplesSync()) {
            defer _ = mlx.mlx_array_free(step_logits);
            const t_sample = tick_prof.mark();
            const result: SampleResult = switch (self.forcedNext()) {
                .sample => try sampleToken(allocator, step_logits, self.sampling, self.generated_ids.items, self.logprobs_n, self.markerShiftCommitted(), self.xfm.s),
                .force => |id| .{ .token_id = id, .logprob_result = try firstTokenLogprobs(allocator, step_logits, id, self.logprobs_n, self.xfm.s) },
                .choose => |ids| blk: {
                    const lazy = try self.sampleAmong(allocator, step_logits, ids);
                    defer _ = mlx.mlx_array_free(lazy);
                    try mlx.check(mlx.mlx_array_eval(lazy));
                    var v: i32 = 0;
                    try mlx.check(mlx.mlx_array_item_int32(&v, lazy));
                    const id: u32 = @intCast(v);
                    break :blk .{ .token_id = id, .logprob_result = try firstTokenLogprobs(allocator, step_logits, id, self.logprobs_n, self.xfm.s) };
                },
            };
            tick_prof.add(.sample, t_sample);
            self.sampling.draw +%= 1;
            self.next_token_id = result.token_id;
            // `result` belongs to the token we just SAMPLED, which the next
            // call returns; `token` is carrying the previous round's result.
            if (self.last_logprob) |*lp| allocator.free(lp.top_logprobs);
            self.last_logprob = self.pending_logprob;
            self.pending_logprob = result.logprob_result;
            if (self.step < self.max_tokens) self.startAsyncForward(result.token_id);
            return token;
        }

        // Last token or pipeline bootstrap
        const t_sample = tick_prof.mark();
        const lazy_token = blk: {
            defer _ = mlx.mlx_array_free(step_logits);
            break :blk switch (self.forcedNext()) {
                .sample => try self.sampleLazy(step_logits, .committed),
                .force => |id| tokenArray(id),
                .choose => |ids| try self.sampleAmong(allocator, step_logits, ids),
            };
        };
        tick_prof.add(.sample, t_sample);

        if (self.step < self.max_tokens) {
            const t_fwd = tick_prof.mark();
            const next_logits = lazyForward(self.xfm, &self.ctx, lazy_token) catch {
                tick_prof.add(.fwd, t_fwd);
                try mlx.check(mlx.mlx_array_eval(lazy_token));
                var val: i32 = 0;
                try mlx.check(mlx.mlx_array_item_int32(&val, lazy_token));
                _ = mlx.mlx_array_free(lazy_token);
                self.next_token_id = @intCast(val);
                return token;
            };
            tick_prof.add(.fwd, t_fwd);

            const arr = [_]mlx.mlx_array{ lazy_token, next_logits };
            const vec = mlx.mlx_vector_array_new_data(&arr, 2);
            const t_eval = tick_prof.mark();
            _ = mlx.mlx_async_eval(vec);
            tick_prof.add(.eval, t_eval);
            _ = mlx.mlx_vector_array_free(vec);

            self.pending_token = lazy_token;
            self.has_pending_token = true;
            self.pending_logits = next_logits;
            self.has_pending_logits = true;
        } else {
            try mlx.check(mlx.mlx_array_eval(lazy_token));
            var val: i32 = 0;
            try mlx.check(mlx.mlx_array_item_int32(&val, lazy_token));
            _ = mlx.mlx_array_free(lazy_token);
            self.next_token_id = @intCast(val);
        }

        return token;
    }

    /// Sample one token while the reasoning protocol has not reached the JSON
    /// body: drain a forced recovery sequence, work the unresolved opener
    /// choice under the union mask, or sample unconstrained reasoning while
    /// the close-delimiter matcher watches. The active JSON phase reuses the
    /// existing grammar mask builder.
    fn nextReasoningStep(self: *Generator, allocator: std.mem.Allocator, constraint: *Constraint, proto: *const rp_mod.Protocol) !?u32 {
        // Recovery drain: at most one canonical transition token per tick,
        // through the same forward/accounting path as a sampled token.
        if (constraint.pstate.recovering) {
            return try self.commitForcedConstraintToken(allocator, constraint);
        }
        switch (constraint.pstate.phase) {
            .choice, .header => return self.nextChoiceStep(allocator, constraint, proto),
            .reasoning => {},
            .json_body => unreachable,
        }

        const step_logits = self.pending_logits;
        self.has_pending_logits = false;
        var owns_step_logits = true;
        defer {
            if (owns_step_logits) _ = mlx.mlx_array_free(step_logits);
        }

        // Boundary-crossing candidates (tokens carrying the close delimiter
        // plus a payload suffix) are validated BEFORE they can be sampled: a
        // suffix the schema rejects is masked out here, never sampled and
        // degraded afterwards.
        const invalid_crossers = try rp_mod.applyReasoningMask(proto, &constraint.pstate, constraint.grammar, constraint.token_bytes, constraint.mask_buf);
        const lazy = blk: {
            if (invalid_crossers == 0) break :blk try self.sampleLazy(step_logits, .committed);
            var masked_logits = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(masked_logits);
            try applyGrammarMask(allocator, &masked_logits, step_logits, constraint.mask_buf, self.xfm.s);
            break :blk try self.sampleLazy(masked_logits, .committed);
        };
        defer _ = mlx.mlx_array_free(lazy);
        try mlx.check(mlx.mlx_array_eval(lazy));
        var val: i32 = 0;
        try mlx.check(mlx.mlx_array_item_int32(&val, lazy));
        const token: u32 = @intCast(val);
        self.next_token_id = token;

        for (self.eos_token_ids) |eos_id| {
            if (token == eos_id and (proto.closer_atomic == null or token != proto.closer_atomic.?)) {
                _ = mlx.mlx_array_free(step_logits);
                owns_step_logits = false;
                switch (try self.forceConstraintTransition(allocator)) {
                    .committed => |forced| {
                        log.info("[grammar] reasoning boundary forced after premature EOS\n", .{});
                        return forced;
                    },
                    .activated => unreachable, // the reasoning phase always needs its close delimiter
                    .refused => {
                        self.done = true;
                        self.finish_reason = "stop";
                        return null;
                    },
                }
            }
        }
        if (token == 0) {
            self.consecutive_pad += 1;
            if (self.consecutive_pad >= 3) {
                _ = mlx.mlx_array_free(step_logits);
                owns_step_logits = false;
                switch (try self.forceConstraintTransition(allocator)) {
                    .committed => |forced| {
                        log.info("[grammar] reasoning boundary forced after terminal padding\n", .{});
                        return forced;
                    },
                    .activated => unreachable,
                    .refused => {
                        self.done = true;
                        self.finish_reason = "stop";
                        return null;
                    },
                }
            }
        } else {
            self.consecutive_pad = 0;
        }

        const token_bytes = if (token < constraint.token_bytes.bytes.len) constraint.token_bytes.bytes[token] else null;
        const payload_suffix = rp_mod.observeProtocolToken(proto, &constraint.pstate, token, token_bytes);
        if (constraint.pstate.invalid) return error.InvalidProtocolTransition;
        if (payload_suffix) |suffix| {
            // The close delimiter completed inside this token; any bytes after
            // it are the first payload bytes. The pre-sample mask validated
            // every crossing candidate's suffix. A mismatch is an internal
            // error and must never disable the constraint.
            for (suffix) |b| {
                const ok = try constraint.grammar.acceptByte(b);
                if (!ok) {
                    return error.InvalidProtocolPayload;
                }
            }
            constraint.pstate.phase = .json_body;
            // A byte-spelled close shares this token: the payload begins at
            // byte_offset inside it. An atomic close has no bytes at all —
            // the payload begins with the NEXT token (never unwrap the
            // null token bytes).
            constraint.pending_span = if (token_bytes == null)
                .{ .token_index = @intCast(self.generated_ids.items.len + 1), .byte_offset = 0 }
            else
                .{ .token_index = @intCast(self.generated_ids.items.len), .byte_offset = rp_mod.payloadByteOffset(token_bytes, suffix) };
            self.loop_guard_start = self.generated_ids.items.len;
            log.info("[grammar] reasoning boundary reached; enforcing final-answer schema\n", .{});
        }

        self.advanceStep(1);
        try self.generated_ids.append(allocator, token);

        if (self.step < self.max_tokens) {
            const tok_i32: i32 = @intCast(token);
            const tok_shape = [_]c_int{ 1, 1 };
            const tok_input = mlx.mlx_array_new_data(&tok_i32, &tok_shape, 2, .int32);
            defer _ = mlx.mlx_array_free(tok_input);
            const next_logits = try self.xfm.forwardWith(&self.ctx, tok_input);
            const arr = [_]mlx.mlx_array{next_logits};
            const vec = mlx.mlx_vector_array_new_data(&arr, 1);
            _ = mlx.mlx_async_eval(vec);
            _ = mlx.mlx_vector_array_free(vec);
            self.pending_logits = next_logits;
            self.has_pending_logits = true;
        } else {
            self.done = true;
            self.finish_reason = "length";
        }
        return token;
    }

    /// One constrained sample at the unresolved opener choice. The mask is
    /// the union of schema-legal tokens (the direct answer is constrained
    /// from its first byte) and opener-prefix candidates; once opener bytes
    /// are in flight only opener continuations remain. The JSON grammar never
    /// consumes protocol bytes.
    fn nextChoiceStep(self: *Generator, allocator: std.mem.Allocator, constraint: *Constraint, proto: *const rp_mod.Protocol) !?u32 {
        const s = self.xfm.s;

        const allowed = if (constraint.pstate.phase == .header)
            try rp_mod.applyHeaderMask(proto, &constraint.pstate, constraint.grammar, constraint.token_bytes, constraint.mask_buf)
        else
            try rp_mod.applyChoiceMask(proto, &constraint.pstate, constraint.grammar, constraint.token_bytes, constraint.mask_buf);
        if (allowed == 0) {
            return error.NoValidProtocolToken;
        }

        const step_logits = self.pending_logits;
        self.has_pending_logits = false;
        defer _ = mlx.mlx_array_free(step_logits);

        var masked_logits = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(masked_logits);
        try applyGrammarMask(allocator, &masked_logits, step_logits, constraint.mask_buf, s);

        const lazy = try self.sampleLazy(masked_logits, .committed);
        defer _ = mlx.mlx_array_free(lazy);
        try mlx.check(mlx.mlx_array_eval(lazy));
        var val: i32 = 0;
        try mlx.check(mlx.mlx_array_item_int32(&val, lazy));
        const token: u32 = @intCast(val);
        self.next_token_id = token;

        const token_bytes = if (token < constraint.token_bytes.bytes.len) constraint.token_bytes.bytes[token] else null;

        // The choice mask never admits EOS (the grammar is incomplete) and
        // pad tokens carry no admissible bytes; stop the way the constrained
        // path does if one is sampled anyway.
        for (self.eos_token_ids) |eos_id| {
            if (token == eos_id and constraint.pstate.phase != .header) {
                self.done = true;
                self.finish_reason = "stop";
                return null;
            }
        }
        if (token == 0) {
            self.consecutive_pad += 1;
            if (self.consecutive_pad >= 3) {
                self.done = true;
                self.finish_reason = "stop";
                return null;
            }
        } else {
            self.consecutive_pad = 0;
        }

        const outcome: rp_mod.ChoiceOutcome = if (constraint.pstate.phase == .header) blk: {
            const suffix = rp_mod.observeProtocolToken(proto, &constraint.pstate, token, token_bytes);
            if (constraint.pstate.invalid) return error.InvalidProtocolTransition;
            if (suffix) |bytes| break :blk .{ .payload = bytes };
            break :blk if (constraint.pstate.phase == .reasoning) .opened else .progress;
        } else rp_mod.observeChoice(proto, &constraint.pstate, token, token_bytes);
        switch (outcome) {
            .progress => {},
            .opened => {
                self.loop_guard_start = self.generated_ids.items.len;
                log.info("[grammar] reasoning opener generated; reasoning unconstrained\n", .{});
            },
            .json, .payload => {
                const payload = if (outcome == .payload) outcome.payload else token_bytes;
                if (payload) |bytes| {
                    for (bytes) |b| {
                        const ok = try constraint.grammar.acceptByte(b);
                        if (!ok) {
                            return error.InvalidProtocolPayload;
                        }
                    }
                }
                constraint.pstate.phase = .json_body;
                constraint.pending_span = .{
                    .token_index = @intCast(self.generated_ids.items.len + @as(usize, if (token_bytes == null) 1 else 0)),
                    .byte_offset = if (outcome == .payload) rp_mod.payloadByteOffset(token_bytes, outcome.payload) else 0,
                };
                self.loop_guard_start = self.generated_ids.items.len;
            },
        }

        self.advanceStep(1);
        try self.generated_ids.append(allocator, token);

        if (self.step < self.max_tokens) {
            const tok_i32: i32 = @intCast(token);
            const tok_shape = [_]c_int{ 1, 1 };
            const tok_input = mlx.mlx_array_new_data(&tok_i32, &tok_shape, 2, .int32);
            defer _ = mlx.mlx_array_free(tok_input);
            const next_logits = try self.xfm.forwardWith(&self.ctx, tok_input);
            const arr = [_]mlx.mlx_array{next_logits};
            const vec = mlx.mlx_vector_array_new_data(&arr, 1);
            _ = mlx.mlx_async_eval(vec);
            _ = mlx.mlx_vector_array_free(vec);
            self.pending_logits = next_logits;
            self.has_pending_logits = true;
        } else {
            self.done = true;
            self.finish_reason = "length";
        }
        return token;
    }

    /// Synchronous, grammar-constrained sampling step. Used whenever
    /// `sampling.constraint` is non-null. Builds a token mask from the grammar's
    /// current state, applies it to the pending logits, samples, advances the
    /// grammar by the sampled token's bytes, and pre-launches the next forward
    /// pass to overlap with the next mask build.
    fn nextConstrained(self: *Generator, allocator: std.mem.Allocator) !?u32 {
        if (self.logprobs_n == 0 or !self.has_pending_logits) return self.nextConstrainedToken(allocator);
        // A constrained token belongs to these raw logits, before the grammar mask.
        var logits = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(logits);
        try mlx.check(mlx.mlx_array_set(&logits, self.pending_logits));
        const token = (try self.nextConstrainedToken(allocator)) orelse return null;
        const lp = try computeLogprobs(allocator, logits, token, self.logprobs_n, self.xfm.s);
        if (self.last_logprob) |old| allocator.free(old.top_logprobs);
        self.pending_logprob = lp;
        self.last_logprob = self.pending_logprob;
        self.pending_logprob = null;
        return token;
    }

    fn nextConstrainedToken(self: *Generator, allocator: std.mem.Allocator) !?u32 {
        if (!self.has_pending_logits) {
            self.done = true;
            return null;
        }
        if (self.stall.expired(self.timer.read(), self.generated_ids.items.len, self.timeout_ns)) {
            self.done = true;
            self.finish_reason = "length";
            return null;
        }
        if (self.step >= self.max_tokens) {
            self.done = true;
            self.finish_reason = "length";
            return null;
        }
        const constraint = self.sampling.constraint.?;
        if (constraint.proto) |proto| {
            if (constraint.pstate.phase != .json_body) return self.nextReasoningStep(allocator, constraint, proto);
        }
        const s = self.xfm.s;

        var allowed = (try token_mask.buildMask(constraint.grammar, constraint.token_bytes, constraint.mask_buf)).allowed;
        // The tokenizer's EOS is not necessarily the model's only stop id.
        if (constraint.grammar.isComplete()) {
            for (self.eos_token_ids) |eos_id| {
                if (eos_id < constraint.mask_buf.len and !constraint.mask_buf[eos_id]) {
                    constraint.mask_buf[eos_id] = true;
                    allowed += 1;
                }
            }
        }
        if (allowed == 0) {
            // 0 legal bytes is a grammar dead end; otherwise the vocabulary lacks a token for them.
            const legal_bytes = blk: {
                const m = constraint.grammar.allowedBytes() catch break :blk 0;
                var n: u32 = 0;
                for (m.bits) |w| n += @popCount(w);
                break :blk n;
            };
            log.warn("[grammar] no token satisfies the schema at this position ({d} legal bytes) — disabling further mask enforcement\n", .{legal_bytes});
            constraint.grammar.dead = true;
            @memset(constraint.mask_buf, true);
        }

        const step_logits = self.pending_logits;
        self.has_pending_logits = false;
        defer _ = mlx.mlx_array_free(step_logits);

        var masked_logits = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(masked_logits);
        try applyGrammarMask(allocator, &masked_logits, step_logits, constraint.mask_buf, s);

        // The next forward is built on the still-lazy sample so the CPU graph
        // build overlaps this step's GPU work; the token is realized after
        // dispatch and only then advances the grammar (the mask for the next
        // logits is built on the next call, off the realized state).
        const lazy = try self.sampleLazy(masked_logits, .committed);
        defer _ = mlx.mlx_array_free(lazy);
        var next_logits: ?mlx.mlx_array = null;
        if (self.step + 1 < self.max_tokens) {
            if (lazyForward(self.xfm, &self.ctx, lazy)) |nl| {
                const arr = [_]mlx.mlx_array{ lazy, nl };
                const vec = mlx.mlx_vector_array_new_data(&arr, 2);
                _ = mlx.mlx_async_eval(vec);
                _ = mlx.mlx_vector_array_free(vec);
                next_logits = nl;
            } else |_| {}
        }
        errdefer if (next_logits) |nl| {
            _ = mlx.mlx_array_free(nl);
        };
        try mlx.check(mlx.mlx_array_eval(lazy));
        var val: i32 = 0;
        try mlx.check(mlx.mlx_array_item_int32(&val, lazy));
        const token: u32 = @intCast(val);
        self.next_token_id = token;

        // Stop on EOS — do not advance grammar or include in output. A
        // dispatched forward on the stop token is dropped, as on the lazy path.
        for (self.eos_token_ids) |eos_id| {
            if (token == eos_id) {
                if (next_logits) |nl| _ = mlx.mlx_array_free(nl);
                self.done = true;
                self.finish_reason = "stop";
                return null;
            }
        }
        if (token == 0) {
            self.consecutive_pad += 1;
            if (self.consecutive_pad >= 3) {
                if (next_logits) |nl| _ = mlx.mlx_array_free(nl);
                self.done = true;
                self.finish_reason = "stop";
                return null;
            }
        } else {
            self.consecutive_pad = 0;
        }

        // Advance the grammar by the sampled token's byte sequence. The mask
        // guarantees every byte is accepted (or the token has no byte form, e.g. a
        // special tag) — so a rejection here means a bug we want to surface.
        if (token < constraint.token_bytes.bytes.len) {
            if (constraint.token_bytes.bytes[token]) |bytes| {
                for (bytes) |b| {
                    const ok = try constraint.grammar.acceptByte(b);
                    if (!ok) {
                        log.warn("[grammar] sampled token {d} produced byte 0x{x} that was rejected — disabling further mask enforcement\n", .{ token, b });
                        constraint.grammar.dead = true;
                        break;
                    }
                }
            }
        }

        self.advanceStep(1);
        try self.generated_ids.append(allocator, token);

        if (next_logits) |nl| {
            self.pending_logits = nl;
            self.has_pending_logits = true;
        } else if (self.step < self.max_tokens) {
            // Pipeline miss: forward the realized token.
            const tok_i32: i32 = @intCast(token);
            const tok_shape = [_]c_int{ 1, 1 };
            const tok_input = mlx.mlx_array_new_data(&tok_i32, &tok_shape, 2, .int32);
            defer _ = mlx.mlx_array_free(tok_input);
            const nl = try self.xfm.forwardWith(&self.ctx, tok_input);
            const arr = [_]mlx.mlx_array{nl};
            const vec = mlx.mlx_vector_array_new_data(&arr, 1);
            _ = mlx.mlx_async_eval(vec);
            _ = mlx.mlx_vector_array_free(vec);
            self.pending_logits = nl;
            self.has_pending_logits = true;
        } else {
            self.done = true;
            self.finish_reason = "length";
        }

        return token;
    }

    /// Check all stop conditions. Returns true if generation should stop.
    pub fn checkStop(self: *Generator) !bool {
        if (self.step >= self.max_tokens) {
            self.done = true;
            self.finish_reason = "length";
            return true;
        }
        if (self.stall.expired(self.timer.read(), self.generated_ids.items.len, self.timeout_ns)) {
            self.done = true;
            self.finish_reason = "length";
            return true;
        }
        if (tokenStops(self.next_token_id, self.eos_token_ids, &self.consecutive_pad)) {
            self.done = true;
            self.finish_reason = "stop";
            return true;
        }
        return false;
    }

    /// Legacy sync forward for logprobs path.
    fn startAsyncForward(self: *Generator, token_id: u32) void {
        const tok_i32: i32 = @intCast(token_id);
        const tok_shape = [_]c_int{ 1, 1 };
        const tok_input = mlx.mlx_array_new_data(&tok_i32, &tok_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(tok_input);

        const logits = self.xfm.forwardWith(&self.ctx, tok_input) catch return;
        const arr = [_]mlx.mlx_array{logits};
        const vec = mlx.mlx_vector_array_new_data(&arr, 1);
        _ = mlx.mlx_async_eval(vec);
        _ = mlx.mlx_vector_array_free(vec);

        self.pending_logits = logits;
        self.has_pending_logits = true;
    }
};

/// Build forward pass from a lazy sampled token array.
/// Reshapes [1] -> [1, 1] and calls transformer forward. All lazy (no eval).
fn lazyForward(xfm: *Transformer, ctx: *ForwardCtx, lazy_token: mlx.mlx_array) !mlx.mlx_array {
    const tok_shape = [_]c_int{ 1, 1 };
    var reshaped = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(reshaped);
    try mlx.check(mlx.mlx_reshape(&reshaped, lazy_token, &tok_shape, 2, xfm.s));
    // The graph is built while the previous step still runs on the GPU; an
    // arch with a host-side token lookup (qwen4 n-gram PLE) fills its leaf
    // afterwards, so the one sync on the token comes AFTER the build.
    ctx.ple_defer = true;
    defer ctx.ple_defer = false;
    const logits = xfm.forwardWith(ctx, reshaped) catch |e| {
        xfm.discardDeferredPle(ctx);
        return e;
    };
    xfm.flushDeferredPle(ctx) catch |e| {
        xfm.discardDeferredPle(ctx);
        _ = mlx.mlx_array_free(logits);
        return e;
    };
    return logits;
}

/// Sample a token lazily from logits — returns a lazy MLX array (no eval).
/// Handles temperature scaling, top-k, and top-p, but defers materialization.
/// The returned array has shape [1] with the sampled token ID.
/// Caller must free the returned array.
/// Compute the probability distribution over the vocabulary at the LAST
/// position of `logits_3d` (shape `[B, S, V]`), with the SAME temperature +
/// top-k + top-p masking the sampler would apply. Both `target_p` and `draft_q`
/// in the stochastic-verify accept test must be computed via this function so
/// the ratio `p[draft] / q[draft]` is well-defined over the kept support.
/// Caller owns the returned array; shape `[B, V]`.
/// Batched sibling of `probsAtLastPos`: temperature → top-k → top-p →
/// softmax over EVERY position of `[1, L, V]` logits in one set of
/// row-parallel kernels. A per-position loop pays L separate ~vocab-sized
/// sort/topk kernel launches per spec-decode round; batched it's one each.
/// All filter helpers operate on the last axis, so leading dims pass through.
fn probsAllPositions(logits_3d: mlx.mlx_array, sampling: SamplingParams, s: mlx.mlx_stream) !mlx.mlx_array {
    var current = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(current);
    try mlx.check(mlx.mlx_array_set(&current, logits_3d));

    // Reserved-token suppression (batched MTP stochastic verify) — same
    // rationale as `probsAtLastPos`.
    if (sampling.suppress_mask) |m| {
        var masked = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(masked);
        try applySuppressMask(&masked, current, m, s);
        _ = mlx.mlx_array_free(current);
        current = masked;
    }

    if (sampling.temperature != 1.0) {
        const t = mlx.mlx_array_new_float(sampling.temperature);
        defer _ = mlx.mlx_array_free(t);
        var scaled = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(scaled);
        try mlx.check(mlx.mlx_divide(&scaled, current, t, s));
        _ = mlx.mlx_array_free(current);
        current = scaled;
    }
    if (sampling.top_k > 0 or sampling.top_p < 1.0) {
        var masked = mlx.mlx_array_new();
        filterTopKTopP(&masked, current, sampling.top_p, sampling.top_k, s) catch {};
        _ = mlx.mlx_array_free(current);
        current = masked;
    }

    var probs = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(probs);
    try mlx.check(mlx.mlx_softmax_axis(&probs, current, -1, true, s));
    _ = mlx.mlx_array_free(current);
    current = .{ .ctx = null };
    return probs;
}

/// The target densities of a whole GROUP of verify rows in one filtered block:
/// the rows' `[1, Σ(1+m_i), V]` logits filtered as one. The filter is row-wise
/// on the last axis, so a row's slice of this block is byte-for-byte the
/// `probsAllPositions` of that row alone.
fn groupProbsBlock(block_logits: mlx.mlx_array, sampling: SamplingParams, s: mlx.mlx_stream) !mlx.mlx_array {
    return probsAllPositions(block_logits, sampling, s);
}

/// The `[1+m, V]` rows a round reads from its verify block: a batched group
/// right-pads every row to its widest draft, so the block may be longer.
fn verifyRows2d(probs_all: mlx.mlx_array, m: u32, s: mlx.mlx_stream) !mlx.mlx_array {
    const shape = mlx.getShape(probs_all);
    const rows: c_int = @intCast(1 + m);
    if (shape.len != 3 or shape[0] != 1 or shape[1] < rows) return error.MtpVerifyBlockShape;
    var head = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(head);
    try mlx.check(mlx.mlx_slice(&head, probs_all, &.{ 0, 0, 0 }, 3, &.{ 1, rows, shape[2] }, 3, &.{ 1, 1, 1 }, 3, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, head, &.{ rows, shape[2] }, 2, s));
    return out;
}

/// One row's `[1, 1+m, V]` densities as a VIEW into the group's block.
fn groupProbsRow(block_probs: mlx.mlx_array, offset: c_int, len: u32, s: mlx.mlx_stream) !mlx.mlx_array {
    const shape = mlx.getShape(block_probs);
    const start = [_]c_int{ 0, offset, 0 };
    const stop = [_]c_int{ shape[0], offset + @as(c_int, @intCast(len)), shape[2] };
    const strides = [_]c_int{ 1, 1, 1 };
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_slice(&out, block_probs, &start, 3, &stop, 3, &strides, 3, s));
    return out;
}

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

fn probsAtLastPos(logits_3d: mlx.mlx_array, sampling: SamplingParams, s: mlx.mlx_stream) !mlx.mlx_array {
    const shape = mlx.getShape(logits_3d);
    const seq_len = shape[1];
    var current = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(current);
    if (seq_len == 1) {
        const sq_shape = [_]c_int{ shape[0], shape[2] };
        try mlx.check(mlx.mlx_reshape(&current, logits_3d, &sq_shape, 2, s));
    } else {
        const start = [_]c_int{ 0, seq_len - 1, 0 };
        const stop = [_]c_int{ shape[0], seq_len, shape[2] };
        const strides = [_]c_int{ 1, 1, 1 };
        var sliced = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sliced);
        try mlx.check(mlx.mlx_slice(&sliced, logits_3d, &start, 3, &stop, 3, &strides, 3, s));
        const sq_shape = [_]c_int{ shape[0], shape[2] };
        try mlx.check(mlx.mlx_reshape(&current, sliced, &sq_shape, 2, s));
    }

    // Reserved-token suppression: the filtered probs feed spec-verify
    // acceptance AND the residual corrections, so a suppressed draft's
    // acceptance probability is exactly 0 and the residual can't re-draw it.
    if (sampling.suppress_mask) |m| {
        var masked = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(masked);
        try applySuppressMask(&masked, current, m, s);
        _ = mlx.mlx_array_free(current);
        current = masked;
    }

    const owned = current;
    current = .{ .ctx = null };
    return filteredProbsRows(owned, sampling, s);
}

/// Temperature → top-k → top-p → softmax over the LAST axis of an owned
/// `[.., V]` array (consumed). The ONE place a proposal or target density is
/// built: a draft sampled from a distribution that is not byte-for-byte the
/// `q` handed to `specAcceptProb` breaks exactness silently, so both sides
/// come through here. Row-independent, so it serves a single `[1, V]` row and
/// a whole `[m, V]` block identically.
fn filteredProbsRows(owned_rows: mlx.mlx_array, sampling: SamplingParams, s: mlx.mlx_stream) !mlx.mlx_array {
    var current = owned_rows;
    errdefer _ = mlx.mlx_array_free(current);
    // Apply temperature → top-k → top-p (same order as `sampleTokenLazy`).
    if (sampling.temperature != 1.0) {
        const t = mlx.mlx_array_new_float(sampling.temperature);
        defer _ = mlx.mlx_array_free(t);
        var scaled = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(scaled);
        try mlx.check(mlx.mlx_divide(&scaled, current, t, s));
        _ = mlx.mlx_array_free(current);
        current = scaled;
    }
    if (sampling.top_k > 0 or sampling.top_p < 1.0) {
        var masked = mlx.mlx_array_new();
        filterTopKTopP(&masked, current, sampling.top_p, sampling.top_k, s) catch {};
        _ = mlx.mlx_array_free(current);
        current = masked;
    }

    // Softmax: tokens at -inf become 0, kept tokens renormalize to sum=1.
    var probs = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(probs);
    try mlx.check(mlx.mlx_softmax_axis(&probs, current, -1, true, s));
    _ = mlx.mlx_array_free(current);
    current = .{ .ctx = null };
    return probs;
}

/// Proposal densities for EVERY row of a `[1, m, V]` draft-logit block, as
/// `[m, V]`. One filter pass and (at the call site) one categorical over m
/// rows, rather than m of each — the draft block is on the critical path
/// ahead of the verify forward.
fn filteredProbsBlock(logits_3d: mlx.mlx_array, sampling: SamplingParams, s: mlx.mlx_stream) !mlx.mlx_array {
    const shape = mlx.getShape(logits_3d);
    var rows = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(rows);
    const rows_shape = [_]c_int{ shape[0] * shape[1], shape[2] };
    try mlx.check(mlx.mlx_reshape(&rows, logits_3d, &rows_shape, 2, s));
    if (sampling.suppress_mask) |m| {
        var masked = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(masked);
        try applySuppressMask(&masked, rows, m, s);
        _ = mlx.mlx_array_free(rows);
        rows = masked;
    }
    const owned = rows;
    rows = .{ .ctx = null };
    return filteredProbsRows(owned, sampling, s);
}

/// Read `probs[0, token_id]` as f32. Forces realization with a single eval.
fn probAt(probs: mlx.mlx_array, token_id: u32, s: mlx.mlx_stream) !f32 {
    const idx_val: i32 = @intCast(token_id);
    const idx_shape = [_]c_int{1};
    const idx_arr = mlx.mlx_array_new_data(&idx_val, &idx_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(idx_arr);

    var taken = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(taken);
    try mlx.check(mlx.mlx_take_axis(&taken, probs, idx_arr, -1, s));

    // Cast to f32 so item_float32 is exact regardless of source dtype (bf16 etc.).
    var as_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(as_f32);
    try mlx.check(mlx.mlx_astype(&as_f32, taken, .float32, s));
    try mlx.check(mlx.mlx_array_eval(as_f32));
    var v: f32 = 0;
    try mlx.check(mlx.mlx_array_item_float32(&v, as_f32));
    return v;
}

/// Sample one token from probability distribution `probs` (shape `[B, V]`).
/// Returns a u32 token id (caller can append directly).
fn sampleFromProbs(probs: mlx.mlx_array, s: mlx.mlx_stream) !u32 {
    // mlx_random_categorical takes logits and applies softmax. Feed log(probs)
    // so the categorical's softmax recovers the original distribution.
    var log_probs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(log_probs);
    try mlx.check(mlx.mlx_log(&log_probs, probs, s));

    const null_key = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(null_key);
    var sampled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sampled);
    try mlx.check(mlx.mlx_random_categorical(&sampled, log_probs, -1, null_key, s));
    try mlx.check(mlx.mlx_array_eval(sampled));
    var v: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&v, sampled));
    return @intCast(v);
}

/// Build a one-hot float32 row vector of shape `[1, vocab]` with 1.0 at
/// `index` and 0.0 elsewhere. Used by the DFlash stochastic-verify reject path,
/// which models the draft (an n-gram lookup, not a probabilistic model) as a
/// degenerate one-hot distribution. Caller owns the returned array.
fn pldOneHotRow(index: u32, vocab: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    var indices = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(indices);
    try mlx.check(mlx.mlx_arange(&indices, 0, @as(f64, @floatFromInt(vocab)), 1, .int32, s));

    const target_val: i32 = @intCast(index);
    const tgt_shape = [_]c_int{1};
    const target_idx = mlx.mlx_array_new_data(&target_val, &tgt_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(target_idx);

    var mask_bool = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask_bool);
    try mlx.check(mlx.mlx_equal(&mask_bool, indices, target_idx, s));

    var mask_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask_f32);
    try mlx.check(mlx.mlx_astype(&mask_f32, mask_bool, .float32, s));

    const out_shape = [_]c_int{ 1, vocab };
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&out, mask_f32, &out_shape, 2, s));
    return out;
}

/// One row of a `[m, V]` density block as its own `[1, V]` array, so it can
/// feed `probAt` and `sampleResidual` (both row-shaped). Caller owns it.
fn sliceProbRow(rows_2d: mlx.mlx_array, row: u32, s: mlx.mlx_stream) !mlx.mlx_array {
    const shape = mlx.getShape(rows_2d);
    const start = [_]c_int{ @intCast(row), 0 };
    const stop = [_]c_int{ @as(c_int, @intCast(row)) + 1, shape[1] };
    const strides = [_]c_int{ 1, 1 };
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_slice(&out, rows_2d, &start, 2, &stop, 2, &strides, 2, s));
    return out;
}

/// The DFlash2 selector's proposal density for one step as a `[1, V]` row:
/// its per-candidate softmax scattered into zeros — q is zero off the
/// candidate set by construction, so this is the exact density the draft was
/// drawn from (the residual-correction contract).
fn selectorQRow(sp: *const dflash_mod.SelectedPath, step: usize, m: u32, vocab: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const kk = sp.cand_ids.len / @as(usize, m);
    var zeros = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(zeros);
    const z_shape = [_]c_int{ 1, vocab };
    try mlx.check(mlx.mlx_zeros(&zeros, &z_shape, 2, .float32, s));
    const row_shape = [_]c_int{ 1, @intCast(kk) };
    const ids_arr = mlx.mlx_array_new_data(sp.cand_ids.ptr + step * kk, &row_shape, 2, .int32);
    defer _ = mlx.mlx_array_free(ids_arr);
    const vals_arr = mlx.mlx_array_new_data(sp.q.?.ptr + step * kk, &row_shape, 2, .float32);
    defer _ = mlx.mlx_array_free(vals_arr);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_put_along_axis(&out, zeros, ids_arr, vals_arr, 1, s));
    return out;
}

/// One draft depth's token and the density it was drawn from: the `[1, V]`
/// row the verify's accept ratio and residual read for that depth.
const StepProposal = struct { id: mlx.mlx_array, q: mlx.mlx_array };

/// Sampled draft from the exact re-scored top-32: q is the draft sampler's
/// filtered softmax over the shortlist, the draft is drawn from that q, and q
/// goes to the verify as a dense `[1, V]` row that is zero off the shortlist.
/// Leviathan is exact for any q, so the residual reaches the tokens the
/// shortlist missed; filtering inside 32 rows costs acceptance at most. The
/// contract: the row the verify sees is the array the draft was drawn from.
fn shortlistProposal(sl: mtp_mod.Shortlist, sampling: SamplingParams, s: mlx.mlx_stream) !StepProposal {
    const shape = mlx.getShape(sl.exact);
    const n = shape[shape.len - 1];

    // f32: the density `min(1, p/q)` divides by must be the one the
    // categorical drew from, and a bf16 softmax does not sum to 1.
    var rows = mlx.mlx_array_new();
    {
        var flat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(flat);
        const row_shape = [_]c_int{ 1, n };
        try mlx.check(mlx.mlx_reshape(&flat, sl.exact, &row_shape, 2, s));
        errdefer _ = mlx.mlx_array_free(rows);
        try mlx.check(mlx.mlx_astype(&rows, flat, .float32, s));
    }
    // The suppress mask already applied to the coarse readout; the shortlist
    // holds no reserved id.
    var draft = sampling;
    draft.suppress_mask = null;
    const q32 = try filteredProbsRows(rows, draft, s);
    errdefer _ = mlx.mlx_array_free(q32);

    var id = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(id);
    {
        const pick = try sampleFromProbsLazy(q32, sampling, s);
        defer _ = mlx.mlx_array_free(pick);
        var picked = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(picked);
        try mlx.check(mlx.mlx_take_axis(&picked, sl.cands, pick, 0, s));
        try mlx.check(mlx.mlx_astype(&id, picked, .int32, s));
    }

    const q = try scatterShortlistQ(sl, q32, s);
    _ = mlx.mlx_array_free(q32);
    return .{ .id = id, .q = q };
}

/// An argmax depth of a chain that carries densities (consumes `amax`): the
/// id reshaped to the `[1]` a sampled depth drafts, since the verify
/// concatenates a chain's ids, and its one-hot q.
fn argmaxStepProposal(amax: mlx.mlx_array, vocab: c_int, s: mlx.mlx_stream) !StepProposal {
    defer _ = mlx.mlx_array_free(amax);
    var id = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(id);
    try mlx.check(mlx.mlx_reshape(&id, amax, &[_]c_int{1}, 1, s));
    return .{ .id = id, .q = try lazyOneHotRow(id, vocab, s) };
}

/// A depth of a chain that carries densities, off the rerank shortlist: a
/// sampling depth draws from it (`shortlistProposal`), an argmax depth (the
/// greedy tail) takes the token a greedy chain drafts.
fn shortlistStepProposal(sl: mtp_mod.Shortlist, sampling: SamplingParams, s: mlx.mlx_stream) !StepProposal {
    if (!isGreedyTemperature(sampling.temperature)) return shortlistProposal(sl, sampling, s);
    return argmaxStepProposal(try mtp_mod.shortlistArgmax(s, sl), sl.rows, s);
}

/// `shortlistStepProposal` off the draft head's full `[1, 1, V]` logits.
fn logitsStepProposal(logits: mlx.mlx_array, sampling: SamplingParams, s: mlx.mlx_stream) !StepProposal {
    if (isGreedyTemperature(sampling.temperature)) {
        const shape = mlx.getShape(logits);
        return argmaxStepProposal(sampleTokenLazy(logits, sampling, s), shape[shape.len - 1], s);
    }
    const q = try probsAtLastPos(logits, sampling, s);
    errdefer _ = mlx.mlx_array_free(q);
    return .{ .id = try sampleFromProbsLazy(q, sampling, s), .q = q };
}

/// `shortlistProposal` for K rows at once: ONE filtered block over the
/// stacked `[K, 32]` logits and ONE categorical (the kernels are launch-bound,
/// so rows are nearly free and a per-row loop is not). Requires one shared
/// sampler (`sampleRowsHomogeneous`) and no seeded row: one categorical
/// cannot carry a per-row key.
fn shortlistProposalRows(
    shortlists: []const mtp_mod.Shortlist,
    rows: []const usize,
    sampling: SamplingParams,
    s: mlx.mlx_stream,
) !struct {
    ids: [32]mlx.mlx_array,
    qs: [32]mlx.mlx_array,
} {
    const K = rows.len;
    std.debug.assert(K > 0 and K <= 32);
    const n = mlx.getShape(shortlists[rows[0]].exact)[mlx.getShape(shortlists[rows[0]].exact).len - 1];

    // [K, 32] f32, for the same reason as the single-row proposal.
    var block = mlx.mlx_array_new();
    {
        var parts: [32]mlx.mlx_array = @splat(.{ .ctx = null });
        defer for (parts[0..K]) |a| {
            if (a.ctx != null) _ = mlx.mlx_array_free(a);
        };
        for (rows, 0..) |row, j| {
            parts[j] = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_reshape(&parts[j], shortlists[row].exact, &[_]c_int{ 1, n }, 2, s));
        }
        const vec = mlx.mlx_vector_array_new_data(parts[0..K].ptr, K);
        defer _ = mlx.mlx_vector_array_free(vec);
        var cat = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(cat);
        try mlx.check(mlx.mlx_concatenate_axis(&cat, vec, 0, s));
        errdefer _ = mlx.mlx_array_free(block);
        try mlx.check(mlx.mlx_astype(&block, cat, .float32, s));
    }
    // The suppress mask already applied to the coarse readout.
    var draft = sampling;
    draft.suppress_mask = null;
    const q_block = try filteredProbsRows(block, draft, s);
    defer _ = mlx.mlx_array_free(q_block);

    // ONE categorical over the block -> [K] shortlist indices.
    var picks = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(picks);
    {
        var logq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(logq);
        try mlx.check(mlx.mlx_log(&logq, q_block, s));
        const null_key = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(null_key);
        try mlx.check(mlx.mlx_random_categorical(&picks, logq, -1, null_key, s));
    }

    var ids: [32]mlx.mlx_array = @splat(.{ .ctx = null });
    var qs: [32]mlx.mlx_array = @splat(.{ .ctx = null });
    var done: usize = 0;
    errdefer for (0..done) |j| {
        if (ids[j].ctx != null) _ = mlx.mlx_array_free(ids[j]);
        if (qs[j].ctx != null) _ = mlx.mlx_array_free(qs[j]);
    };
    for (rows, 0..) |row, j| {
        const j_c: c_int = @intCast(j);
        var pick = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pick);
        try mlx.check(mlx.mlx_slice(&pick, picks, &[_]c_int{j_c}, 1, &[_]c_int{j_c + 1}, 1, &[_]c_int{1}, 1, s));
        var picked = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(picked);
        try mlx.check(mlx.mlx_take_axis(&picked, shortlists[row].cands, pick, 0, s));
        ids[j] = mlx.mlx_array_new();
        done = j + 1;
        try mlx.check(mlx.mlx_astype(&ids[j], picked, .int32, s));

        var q_row = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(q_row);
        try mlx.check(mlx.mlx_slice(&q_row, q_block, &[_]c_int{ j_c, 0 }, 2, &[_]c_int{ j_c + 1, n }, 2, &[_]c_int{ 1, 1 }, 2, s));
        qs[j] = try scatterShortlistQ(shortlists[row], q_row, s);
    }
    return .{ .ids = ids, .qs = qs };
}

/// One row's `[1, 32]` shortlist density as the dense `[1, V]` q the verify
/// takes, zero off the shortlist.
fn scatterShortlistQ(sl: mtp_mod.Shortlist, q32: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    const n = mlx.getShape(q32)[mlx.getShape(q32).len - 1];
    var zeros = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(zeros);
    try mlx.check(mlx.mlx_zeros(&zeros, &[_]c_int{ 1, sl.rows }, 2, .float32, s));
    var cand_row = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cand_row);
    try mlx.check(mlx.mlx_reshape(&cand_row, sl.cands, &[_]c_int{ 1, n }, 2, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_put_along_axis(&out, zeros, cand_row, q32, 1, s));
    return out;
}

/// A lazy one-hot `[1, V]` f32 proposal row: the q of an argmax proposal, so
/// `min(1, p/q) == p`. The sampled path's fallback when the coarse head
/// retires mid-chain.
fn lazyOneHotRow(id: mlx.mlx_array, vocab: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    var indices = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(indices);
    try mlx.check(mlx.mlx_arange(&indices, 0, @as(f64, @floatFromInt(vocab)), 1, .int32, s));
    var flat_id = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat_id);
    const one = [_]c_int{1};
    try mlx.check(mlx.mlx_reshape(&flat_id, id, &one, 1, s));
    var hit = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hit);
    try mlx.check(mlx.mlx_equal(&hit, indices, flat_id, s));
    var as_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(as_f32);
    try mlx.check(mlx.mlx_astype(&as_f32, hit, .float32, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    const row_shape = [_]c_int{ 1, vocab };
    try mlx.check(mlx.mlx_reshape(&out, as_f32, &row_shape, 2, s));
    return out;
}

/// Sample from the residual distribution `residual = max(target - draft, 0)`,
/// renormalized. Used on stochastic-verify reject so the corrected token
/// preserves the target distribution (per Leviathan et al. speculative
/// decoding paper).
fn sampleResidual(target_probs: mlx.mlx_array, draft_probs: mlx.mlx_array, s: mlx.mlx_stream) !u32 {
    var diff = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(diff);
    try mlx.check(mlx.mlx_subtract(&diff, target_probs, draft_probs, s));

    const zero = mlx.mlx_array_new_float(0.0);
    defer _ = mlx.mlx_array_free(zero);
    var residual = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(residual);
    try mlx.check(mlx.mlx_maximum(&residual, diff, zero, s));

    return sampleFromProbs(residual, s);
}

/// Lazy categorical sample from an already-filtered probability row
/// ([1, vocab]): log(probs) puts masked tokens at -inf, categorical draws
/// within the kept set — the same distribution as sampling the filtered
/// logits directly, but the caller keeps `probs` as the proposal density q.
fn sampleFromProbsLazy(probs: mlx.mlx_array, sampling: SamplingParams, s: mlx.mlx_stream) !mlx.mlx_array {
    var logp = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(logp);
    try mlx.check(mlx.mlx_log(&logp, probs, s));
    var sampled = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(sampled);
    const key = seedKey(sampling);
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_categorical(&sampled, logp, -1, key, s));
    return sampled;
}

/// `sampleTokenLazy` under the request's repeat/presence penalty over `generated_ids`.
fn penalizedSampleLazy(allocator: std.mem.Allocator, logits: mlx.mlx_array, sampling: SamplingParams, generated_ids: []const u32, s: mlx.mlx_stream) !mlx.mlx_array {
    if (penaltyActive(sampling) and generated_ids.len > 0) {
        var penalized = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(penalized);
        try applyRepeatPenaltyWithAllocator(allocator, &penalized, logits, generated_ids, sampling.repeat_penalty, sampling.presence_penalty, s);
        return sampleTokenLazy(penalized, sampling, s);
    }
    return sampleTokenLazy(logits, sampling, s);
}

/// A committed token as the `[1]` int32 array a sample would be.
fn tokenArray(id: u32) mlx.mlx_array {
    const v: i32 = @intCast(id);
    const shape = [_]c_int{1};
    return mlx.mlx_array_new_data(&v, &shape, 1, .int32);
}

/// Prefill's first token under the think penalty when the prompt leaves the span open.
fn firstTokenSample(xfm: *Transformer, logits: mlx.mlx_array, sampling: SamplingParams, s: mlx.mlx_stream) !mlx.mlx_array {
    const tp = sampling.think_penalty;
    if (!tp.active()) return firstTokenLazy(logits, sampling, s);
    if (!tp.hasBias() and (xfm.think_marker_mask == null or tp.phase != .inside)) return firstTokenLazy(logits, sampling, s);
    var shifted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(shifted);
    try think_penalty.shiftScoped(&shifted, logits, xfm.think_marker_mask, tp, null, s);
    return firstTokenLazy(shifted, sampling, s);
}

/// Prefill's first token: a forced call's when it owns token 0, else a sample.
fn firstTokenLazy(logits: mlx.mlx_array, sampling: SamplingParams, s: mlx.mlx_stream) mlx.mlx_array {
    if (sampling.call_force) |cf| switch (cf.decide(&.{})) {
        .force => |id| return tokenArray(id),
        .sample, .choose => {},
    };
    return sampleTokenLazy(logits, sampling, s);
}

pub fn sampleTokenLazy(logits_in: mlx.mlx_array, sampling: SamplingParams, s: mlx.mlx_stream) mlx.mlx_array {
    // Reserved-token suppression first, so every path below (greedy fast
    // path included — the collapse drew `<|fim_hole|>` at temp 0) sees the
    // masked row. Lazy like everything else here. The handle starts null-ctx
    // and is only materialized when a mask exists — the fast path below
    // exists to save ONE FFI call per decode step, so the no-mask case (every
    // non-suppressing model) must not spend two on an empty array.
    var suppressed: mlx.mlx_array = .{ .ctx = null };
    defer if (suppressed.ctx != null) {
        _ = mlx.mlx_array_free(suppressed);
    };
    const logits = if (sampling.suppress_mask) |m| blk: {
        suppressed = mlx.mlx_array_new();
        applySuppressMask(&suppressed, logits_in, m, s) catch break :blk logits_in;
        break :blk suppressed;
    } else logits_in;

    const shape = mlx.getShape(logits);
    const seq_len = shape[1];

    // Greedy + seq_len==1 (the decode hot path): one mlx op total. argmax_axis
    // over the vocab dim of a `[1, 1, V]` tensor yields a `[1, 1]` int array,
    // which downstream (resolvePendingToken / lazyForward / async_eval vector)
    // treats identically to `[1]`. Skipping the otherwise-needed reshape +
    // argmax-on-2D combo cuts ~one FFI call per decode step.
    if (seq_len == 1 and isGreedyTemperature(sampling.temperature)) {
        var result = mlx.mlx_array_new();
        _ = mlx.mlx_argmax_axis(&result, logits, -1, false, s);
        return result;
    }

    // Extract last position: [1, seq_len, vocab] -> [1, vocab]
    // `current` is the single owned intermediate — freed before each reassignment.
    var current = mlx.mlx_array_new();

    if (seq_len == 1) {
        const sq_shape = [_]c_int{ 1, shape[2] };
        _ = mlx.mlx_reshape(&current, logits, &sq_shape, 2, s);
    } else {
        const start = [_]c_int{ 0, seq_len - 1, 0 };
        const stop = [_]c_int{ 1, seq_len, shape[2] };
        const strides = [_]c_int{ 1, 1, 1 };
        var sliced = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sliced);
        _ = mlx.mlx_slice(&sliced, logits, &start, 3, &stop, 3, &strides, 3, s);

        const sq_shape = [_]c_int{ 1, shape[2] };
        _ = mlx.mlx_reshape(&current, sliced, &sq_shape, 2, s);
    }

    // Greedy: argmax (no temperature)
    if (isGreedyTemperature(sampling.temperature)) {
        var result = mlx.mlx_array_new();
        _ = mlx.mlx_argmax_axis(&result, current, -1, false, s);
        _ = mlx.mlx_array_free(current);
        return result;
    }

    // Scale by 1/temperature
    if (sampling.temperature != 1.0) {
        const temp_arr = mlx.mlx_array_new_float(sampling.temperature);
        defer _ = mlx.mlx_array_free(temp_arr);
        var next = mlx.mlx_array_new();
        _ = mlx.mlx_divide(&next, current, temp_arr, s);
        _ = mlx.mlx_array_free(current);
        current = next;
    }

    // Apply top-k and top-p in one pass (lazy)
    if (sampling.top_k > 0 or sampling.top_p < 1.0) {
        var next = mlx.mlx_array_new();
        filterTopKTopP(&next, current, sampling.top_p, sampling.top_k, s) catch {};
        _ = mlx.mlx_array_free(current);
        current = next;
    }

    // Sample from categorical distribution (lazy — no eval!)
    var sampled = mlx.mlx_array_new();
    const key = seedKey(sampling);
    defer _ = mlx.mlx_array_free(key);
    _ = mlx.mlx_random_categorical(&sampled, current, -1, key, s);
    _ = mlx.mlx_array_free(current);

    return sampled; // Shape [1], lazy
}

pub var sample_rows_evals: u64 = 0;

/// One sampling eval for every row of a batched tick. Homogeneous unseeded rows
/// draw over one stacked `[N, V]` graph (one argmax or one categorical draw);
/// a seed, mixed filters or a lone row take the per-row arm, whose parts are
/// reshaped to `[1]` before the concat because a greedy draw is `[1, 1]` and a
/// categorical draw is `[1]`. Every row advances its `draw` before the eval, so
/// a failed eval never replays a key.
pub fn sampleRows(out: []i32, row_logits: []const mlx.mlx_array, params: []SamplingParams, s: mlx.mlx_stream) !void {
    std.debug.assert(out.len == row_logits.len and out.len == params.len);
    const n = row_logits.len;
    if (n == 0) return;
    const lazy = try sampleRowsLazy(row_logits, params, s);
    defer _ = mlx.mlx_array_free(lazy);
    for (params) |*p| p.draw +%= 1;
    sample_rows_evals += 1;
    try mlx.check(mlx.mlx_array_eval(lazy));
    if (n == 1) {
        try mlx.check(mlx.mlx_array_item_int32(&out[0], lazy));
        return;
    }
    const data = mlx.mlx_array_data_uint32(lazy) orelse return error.MlxError;
    for (out, data[0..n]) |*dst, v| dst.* = @intCast(v);
}

/// Rows may share one stacked graph only when every filter matches; a seed
/// always takes the per-row arm so its key stays per row.
fn sampleRowsHomogeneous(params: []const SamplingParams) bool {
    const p0 = params[0];
    for (params[1..]) |p| {
        if (p.temperature != p0.temperature or p.top_k != p0.top_k or p.top_p != p0.top_p) return false;
        const m0 = p0.suppress_mask;
        const m1 = p.suppress_mask;
        if (m0 == null and m1 == null) continue;
        if (m0 == null or m1 == null) return false;
        if (m0.?.ctx != m1.?.ctx) return false;
    }
    return true;
}

fn sampleRowsAllGreedy(params: []const SamplingParams) bool {
    for (params) |p| {
        if (!isGreedyTemperature(p.temperature)) return false;
    }
    return true;
}

fn concatLogitRows(rows: []const mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    if (rows.len == 1) {
        var out = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_array_set(&out, rows[0]));
        return out;
    }
    const vec = mlx.mlx_vector_array_new_data(rows.ptr, rows.len);
    defer _ = mlx.mlx_vector_array_free(vec);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_concatenate_axis(&out, vec, 0, s));
    return out;
}

fn logitsAsNV(logits: mlx.mlx_array, n: c_int, s: mlx.mlx_stream) !mlx.mlx_array {
    const sh = mlx.getShape(logits);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    if (sh.len == 2) {
        try mlx.check(mlx.mlx_array_set(&out, logits));
        return out;
    }
    if (sh.len != 3 or sh[0] != n) return error.BadShape;
    if (sh[1] == 1) {
        const sq = [_]c_int{ n, sh[2] };
        try mlx.check(mlx.mlx_reshape(&out, logits, &sq, 2, s));
        return out;
    }
    const start = [_]c_int{ 0, sh[1] - 1, 0 };
    const stop = [_]c_int{ n, sh[1], sh[2] };
    const strides = [_]c_int{ 1, 1, 1 };
    var sliced = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sliced);
    try mlx.check(mlx.mlx_slice(&sliced, logits, &start, 3, &stop, 3, &strides, 3, s));
    const sq = [_]c_int{ n, sh[2] };
    try mlx.check(mlx.mlx_reshape(&out, sliced, &sq, 2, s));
    return out;
}

fn sampleRowsLazyConcat(row_logits: []const mlx.mlx_array, params: []const SamplingParams, s: mlx.mlx_stream) !mlx.mlx_array {
    var parts: [scheduler_mod.MAX_BATCH_GROUP]mlx.mlx_array = undefined;
    const n = row_logits.len;
    if (n > parts.len) return error.BatchTooWide;
    var n_parts: usize = 0;
    // One release on every exit: the concat output holds its own references.
    defer {
        for (parts[0..n_parts]) |a| _ = mlx.mlx_array_free(a);
    }
    const one = [_]c_int{1};
    for (row_logits, params) |logits, p| {
        const raw = sampleTokenLazy(logits, p, s);
        defer _ = mlx.mlx_array_free(raw);
        parts[n_parts] = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&parts[n_parts], raw, &one, 1, s));
        n_parts += 1;
    }
    return concatLogitRows(parts[0..n_parts], s);
}

fn sampleRowsAnySeed(params: []const SamplingParams) bool {
    for (params) |p| {
        if (p.seed != null) return true;
    }
    return false;
}

fn sampleRowsLazy(row_logits: []const mlx.mlx_array, params: []const SamplingParams, s: mlx.mlx_stream) !mlx.mlx_array {
    const n: c_int = @intCast(row_logits.len);
    if (row_logits.len == 1 or !sampleRowsHomogeneous(params) or sampleRowsAnySeed(params)) {
        return sampleRowsLazyConcat(row_logits, params, s);
    }
    const stacked = try concatLogitRows(row_logits, s);
    defer _ = mlx.mlx_array_free(stacked);
    var current = try logitsAsNV(stacked, n, s);
    var owned = true;
    errdefer if (owned) {
        _ = mlx.mlx_array_free(current);
    };
    const p0 = params[0];

    var suppressed: mlx.mlx_array = .{ .ctx = null };
    defer if (suppressed.ctx != null) {
        _ = mlx.mlx_array_free(suppressed);
    };
    if (p0.suppress_mask) |m| {
        suppressed = mlx.mlx_array_new();
        applySuppressMask(&suppressed, current, m, s) catch {
            _ = mlx.mlx_array_free(suppressed);
            suppressed = .{ .ctx = null };
        };
        if (suppressed.ctx != null) {
            _ = mlx.mlx_array_free(current);
            current = suppressed;
            suppressed = .{ .ctx = null };
        }
    }

    if (sampleRowsAllGreedy(params)) {
        var result = mlx.mlx_array_new();
        errdefer _ = mlx.mlx_array_free(result);
        try mlx.check(mlx.mlx_argmax_axis(&result, current, -1, false, s));
        _ = mlx.mlx_array_free(current);
        owned = false;
        return result;
    }

    if (p0.temperature != 1.0) {
        const temp_arr = mlx.mlx_array_new_float(p0.temperature);
        defer _ = mlx.mlx_array_free(temp_arr);
        var next = mlx.mlx_array_new();
        mlx.check(mlx.mlx_divide(&next, current, temp_arr, s)) catch |err| {
            _ = mlx.mlx_array_free(next);
            return err;
        };
        _ = mlx.mlx_array_free(current);
        current = next;
    }
    if (p0.top_k > 0 or p0.top_p < 1.0) {
        var next = mlx.mlx_array_new();
        filterTopKTopP(&next, current, p0.top_p, p0.top_k, s) catch {
            _ = mlx.mlx_array_free(next);
            next = current;
            current = mlx.mlx_array_new();
        };
        if (current.ctx != null) _ = mlx.mlx_array_free(current);
        current = next;
    }

    const key = seedKey(p0);
    defer _ = mlx.mlx_array_free(key);
    var sampled = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(sampled);
    try mlx.check(mlx.mlx_random_categorical(&sampled, current, -1, key, s));
    _ = mlx.mlx_array_free(current);
    owned = false;
    return sampled;
}

/// PRNG key for one draw: `seed` mixed with the draw index, so a seeded
/// request replays byte-for-byte and consecutive draws never share a key.
/// Null-ctx (MLX global RNG) when the request set no seed.
fn seedKey(sampling: SamplingParams) mlx.mlx_array {
    const seed = sampling.seed orelse return mlx.mlx_array_new();
    var key = mlx.mlx_array_new();
    _ = mlx.mlx_random_key(&key, seed +% sampling.draw *% 0x9E3779B97F4A7C15);
    return key;
}

/// Convenience: generate all tokens at once (non-streaming).
pub fn generate(
    io: std.Io,
    allocator: std.mem.Allocator,
    xfm: *Transformer,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
    max_tokens: u32,
    sampling: SamplingParams,
    eos_token_ids: []const u32,
    timeout_ns: u64,
    logprobs_n: u32,
) !GenerationResult {
    var timer = io_util.Stopwatch.init(io);
    // logprobs_n rides InitOptions so init's argmax-only gate sees it (a
    // post-init field write would let the split-prefill final-token forward
    // engage the pruned lm_head on a logprobs request).
    var gen = try Generator.initWithOptions(io, allocator, xfm, tok, prompt_ids, max_tokens, sampling, eos_token_ids, .{ .logprobs_n = logprobs_n });
    gen.timeout_ns = timeout_ns;
    gen.logprobs_n = logprobs_n;
    defer gen.deinit(allocator);

    const prefill_ns = timer.read();
    const prefill_tps: f64 = if (prefill_ns > 0)
        @as(f64, @floatFromInt(prompt_ids.len)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(prefill_ns))
    else
        0.0;
    log.debug("Prefill: {d}ms ({d} tokens, {d:.3} tok/s)\n", .{
        prefill_ns / std.time.ns_per_ms,
        prompt_ids.len,
        prefill_tps,
    });

    var output_ids = std.ArrayList(u32).empty;
    defer output_ids.deinit(allocator);

    var logprob_results = std.ArrayList(LogprobResult).empty;
    defer {
        if (logprobs_n == 0) {
            for (logprob_results.items) |*lp| allocator.free(lp.top_logprobs);
            logprob_results.deinit(allocator);
        }
    }

    timer.reset();
    while (try gen.next(allocator)) |token_id| {
        try output_ids.append(allocator, token_id);
        if (logprobs_n > 0) {
            if (gen.last_logprob) |lp| {
                try logprob_results.append(allocator, lp);
                gen.last_logprob = null; // Transfer ownership
            }
        }
    }

    const decode_ns = timer.read();
    const num_decoded = output_ids.items.len;
    const decode_tps: f64 = if (decode_ns > 0)
        @as(f64, @floatFromInt(num_decoded)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(decode_ns))
    else
        0.0;
    log.debug("Decode: {d}ms ({d} tokens, {d:.3} tok/s)\n", .{
        decode_ns / std.time.ns_per_ms,
        num_decoded,
        decode_tps,
    });

    const strip_leading = tok.tok_type == .sentencepiece_bpe;
    const text = try tok.decode(allocator, output_ids.items, strip_leading);
    const token_ids = try output_ids.toOwnedSlice(allocator);

    return .{
        .text = text,
        .token_ids = token_ids,
        .prompt_tokens = gen.prompt_tokens,
        .completion_tokens = gen.completion_tokens,
        .finish_reason = gen.finish_reason,
        .prefill_tps = prefill_tps,
        .decode_tps = decode_tps,
        .logprobs = if (logprobs_n > 0) try logprob_results.toOwnedSlice(allocator) else null,
    };
}

/// MTP-enabled non-streaming variant of `generate`. Drives `nextMtp` (the model's own multi-token-prediction head).
/// `head` must already be `bind()`-ed to `xfm`.
pub fn generateMtp(
    io: std.Io,
    allocator: std.mem.Allocator,
    xfm: *Transformer,
    head: *mtp_mod.MtpModel,
    tok: *const Tokenizer,
    prompt_ids: []const u32,
    max_tokens: u32,
    sampling: SamplingParams,
    eos_token_ids: []const u32,
    timeout_ns: u64,
    depth: u32,
    lookup_prompt: ?[]const u32,
) !GenerationResult {
    var timer = io_util.Stopwatch.init(io);
    var gen = try Generator.initWithOptions(io, allocator, xfm, tok, prompt_ids, max_tokens, sampling, eos_token_ids, .{
        .mtp_enabled = true,
        .mtp_acceptance = mtp_acceptance_default,
        .mtp = MtpHeadRef{ .qwen = head },
        .mtp_depth = depth,
        .lookup_prompt = lookup_prompt,
    });
    gen.timeout_ns = timeout_ns;
    defer gen.deinit(allocator);

    const prefill_ns = timer.read();
    const prefill_tps: f64 = if (prefill_ns > 0)
        @as(f64, @floatFromInt(prompt_ids.len)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(prefill_ns))
    else
        0.0;
    log.debug("Prefill (mtp): {d}ms ({d} tokens, {d:.3} tok/s)\n", .{
        prefill_ns / std.time.ns_per_ms,
        prompt_ids.len,
        prefill_tps,
    });

    var output_ids = std.ArrayList(u32).empty;
    defer output_ids.deinit(allocator);

    timer.reset();

    decode: while (!gen.done and gen.completion_tokens < max_tokens) {
        const result = (try gen.nextMtp(allocator)) orelse break;
        defer allocator.free(result.tokens);
        for (result.tokens) |tok_id| {
            if (isEosId(tok_id, eos_token_ids)) {
                gen.done = true;
                gen.finish_reason = "stop";
                break :decode;
            }
            try output_ids.append(allocator, tok_id);
            if (output_ids.items.len >= max_tokens) {
                gen.done = true;
                gen.finish_reason = "length";
                break :decode;
            }
        }
        if (timeout_ns > 0 and timer.read() >= timeout_ns) {
            gen.done = true;
            gen.finish_reason = "length";
            break;
        }
    }

    const decode_ns = timer.read();
    const num_decoded = output_ids.items.len;
    const decode_tps: f64 = if (decode_ns > 0)
        @as(f64, @floatFromInt(num_decoded)) * @as(f64, @floatFromInt(std.time.ns_per_s)) / @as(f64, @floatFromInt(decode_ns))
    else
        0.0;
    if (gen.mtp_attempted > 0) {
        const avg_acc: f64 = @as(f64, @floatFromInt(gen.mtp_accepted_tokens)) / @as(f64, @floatFromInt(gen.mtp_attempted));
        log.info("Decode (mtp): {d}ms ({d} tokens, {d:.3} tok/s; mtp accept={d}/{d} attempts, avg {d:.2} tokens/attempt)\n", .{
            decode_ns / std.time.ns_per_ms,
            num_decoded,
            decode_tps,
            gen.mtp_accepted_tokens,
            gen.mtp_attempted,
            avg_acc,
        });
    } else {
        log.debug("Decode (mtp): {d}ms ({d} tokens, {d:.3} tok/s; no draft attempts)\n", .{
            decode_ns / std.time.ns_per_ms,
            num_decoded,
            decode_tps,
        });
    }
    gen.logSpecStats();
    gen.logQsaArms();
    const strip_leading = tok.tok_type == .sentencepiece_bpe;
    const text = try tok.decode(allocator, output_ids.items, strip_leading);
    const token_ids = try output_ids.toOwnedSlice(allocator);
    return .{
        .text = text,
        .token_ids = token_ids,
        .prompt_tokens = gen.prompt_tokens,
        .completion_tokens = gen.completion_tokens,
        .finish_reason = gen.finish_reason,
        .prefill_tps = prefill_tps,
        .decode_tps = decode_tps,
        .logprobs = null,
    };
}

pub fn isEosId(id: u32, eos: []const u32) bool {
    for (eos) |e| if (id == e) return true;
    return false;
}

/// Token-only stop conditions of `checkStop`; `consecutive_pad` is carried state reset by any real token.
pub fn tokenStops(next_token_id: u32, eos_token_ids: []const u32, consecutive_pad: *u32) bool {
    if (isEosId(next_token_id, eos_token_ids)) return true;
    if (next_token_id != 0) {
        consecutive_pad.* = 0;
        return false;
    }
    consecutive_pad.* += 1;
    return consecutive_pad.* >= 3;
}

/// Max cycle length (in tokens) scanned by `isDegenerateTailLoop`, and how many
/// identical repetitions of that cycle count as "stuck". A real answer — prose,
/// code, a markdown table — essentially never repeats an identical ≤8-token
/// cycle 16 times in a row, so these won't fire on legitimate output, while a
/// model that has collapsed into spamming one short phrase is caught within a
/// few dozen tokens instead of running all the way to `max_tokens`.
pub const degenerate_loop_max_period: usize = 8;
pub const degenerate_loop_reps: usize = 16;
/// A short cycle must also cover this many tokens before it convicts: at
/// period 1 or 2, 16 reps is a digit row or a zeroed array, not a loop.
pub const degenerate_loop_min_span: usize = 128;
// Tier 2 (2026-08-02 shooter wrap-up class): a two-sentence cycle of ~58
// tokens repeated 26 times evaded the 8-token tier. Long periods demand
// fewer reps — 10 verbatim repetitions of a 9..64-token cycle is
// degeneration with overwhelming probability (identical long lines in real
// code repeat a handful of times, not ten).
pub const degenerate_loop_long_max_period: usize = 64;
pub const degenerate_loop_long_reps: usize = 10;
/// ...except when the "line" is a tile-map row (2026-09-15: a 27B wrote ten
/// identical `"100000000000000000000001",` rows, a 28-token cycle, in an
/// honest map file). The long tier also demands this much cycling: 37 such
/// rows, or a 58-token sentence 18 times. A real loop still dies in ~13 s.
pub const degenerate_loop_long_min_span: usize = 1024;

// Tier 3 (2026-08-04 agent-traffic class): a restatement loop that VARIES its
// phrasing has no exact cycle at any period, so both tiers above are blind by
// construction. What it does have is a long stretch of output that recycles a
// tiny vocabulary and introduces almost no new n-grams.
//
// Everything about this tier is deliberately reluctant, because unlike the
// exact tiers it is a fuzzy judgement and a false cut truncates a real answer:
// the window is LONG (a legitimate repetitive passage — a table, a block of
// near-identical code, a list scaffold — is finite and ends well inside it),
// the two ratios must BOTH be low (either one alone convicts honest output —
// a numeric table has few distinct tokens, a repeated code scaffold has few
// distinct n-grams), and the window is longer than the exact tiers' reach so
// this tier only ever speaks about spans they have already declined.
pub const near_repeat_window: usize = 1024;
pub const near_repeat_ngram: usize = 4;
pub const near_repeat_max_ngram_ratio: f32 = 0.35;
pub const near_repeat_max_token_ratio: f32 = 0.12;
/// Third ratio (2026-08-05): how much of the window's SECOND half is n-grams
/// its first half never had. Measured across both shapes at the shipped
/// window: restatement loops 0.019-0.022, healthy repetitive output
/// 0.298-0.827 (dense procedural scene code, a markdown table, the voxel
/// artifact that was wrongly cut). 0.10 sits ~4.5x above the loops and ~3x
/// below the closest healthy case.
pub const near_repeat_max_novelty: f32 = 0.10;

/// Open-addressed distinct-counter sized for one window. Stack-resident, so
/// the whole tier is allocation-free and runs in O(window) per decode tick.
fn DistinctSet(comptime cap: usize) type {
    return struct {
        const Self = @This();
        const empty_key: u64 = std.math.maxInt(u64);
        keys: [cap]u64 = @splat(empty_key),
        n: usize = 0,

        /// True when `key` is already present. Read-only; used to ask whether
        /// the window's second half is introducing anything its first half
        /// did not have.
        fn contains(self: *const Self, key: u64) bool {
            const k = if (key == empty_key) 0 else key;
            var i: usize = @intCast(std.hash.Wyhash.hash(0, std.mem.asBytes(&k)) % cap);
            while (true) {
                if (self.keys[i] == empty_key) return false;
                if (self.keys[i] == k) return true;
                i = (i + 1) % cap;
            }
        }

        /// True when `key` was not already present.
        fn insert(self: *Self, key: u64) bool {
            // maxInt is the empty sentinel; fold the one colliding key onto 0.
            const k = if (key == empty_key) 0 else key;
            var i: usize = @intCast(std.hash.Wyhash.hash(0, std.mem.asBytes(&k)) % cap);
            while (true) {
                if (self.keys[i] == empty_key) {
                    self.keys[i] = k;
                    self.n += 1;
                    return true;
                }
                if (self.keys[i] == k) return false;
                i = (i + 1) % cap;
            }
        }
    };
}

/// The three-ratio judgement over ONE window-sized span. Split out so the trim
/// search can slide the same window backwards without re-deriving the rule.
fn nearRepeatWindowIsDegenerate(window: []const u32) bool {
    // Load factor 0.5 keeps the linear probe short even when every entry is
    // distinct (the healthy case, which is also the hot one).
    var toks = DistinctSet(near_repeat_window * 2){};
    for (window) |t| _ = toks.insert(t);
    const token_ratio = @as(f32, @floatFromInt(toks.n)) / @as(f32, @floatFromInt(window.len));
    if (token_ratio > near_repeat_max_token_ratio) return false;

    var grams = DistinctSet(near_repeat_window * 2){};
    var i: usize = 0;
    while (i + near_repeat_ngram <= window.len) : (i += 1) {
        _ = grams.insert(gramHash(window[i .. i + near_repeat_ngram]));
    }
    const n_gram_positions = window.len - near_repeat_ngram + 1;
    const gram_ratio = @as(f32, @floatFromInt(grams.n)) / @as(f32, @floatFromInt(n_gram_positions));
    if (gram_ratio > near_repeat_max_ngram_ratio) return false;

    // Third ratio: is the window still PROGRESSING? Both ratios above are
    // properties of a vocabulary, and procedurally generated code has the same
    // vocabulary profile as a loop — a fixed call template plus a small colour
    // palette (live 2026-08-05: a voxel scene cut at 16241 tokens, the user got
    // no file at all). What a loop does NOT do is keep introducing material:
    // measured, a restatement loop's second half brings 1.9-2.2% n-grams its
    // first half never had, while healthy repetitive output brings 29.8-82.7%.
    // Requiring all THREE keeps the tier's reluctance in the direction that
    // matters — a missed loop still ends at max_tokens, a false cut destroys
    // work that was going fine.
    return halfNovelty(window, near_repeat_ngram) <= near_repeat_max_novelty;
}

/// Share of the window's second-half distinct n-grams absent from its first
/// half. 1.0 when the second half has no n-grams at all (nothing new because
/// there is nothing) is the loop reading, so that case returns 0.
fn halfNovelty(window: []const u32, n: usize) f32 {
    var first_half = DistinctSet(near_repeat_window * 2){};
    const mid = window.len / 2;
    var fi: usize = 0;
    while (fi + n <= mid) : (fi += 1) {
        _ = first_half.insert(gramHash(window[fi .. fi + n]));
    }
    var second_half = DistinctSet(near_repeat_window * 2){};
    var novel: usize = 0;
    var distinct_second: usize = 0;
    var si: usize = mid;
    while (si + n <= window.len) : (si += 1) {
        const h = gramHash(window[si .. si + n]);
        if (!second_half.insert(h)) continue; // count each distinct gram once
        distinct_second += 1;
        if (!first_half.contains(h)) novel += 1;
    }
    if (distinct_second == 0) return 0;
    return @as(f32, @floatFromInt(novel)) / @as(f32, @floatFromInt(distinct_second));
}

/// One hash for both the ratio pass and the novelty pass — two spellings of
/// this would silently compare different things.
fn gramHash(gram: []const u32) u64 {
    var h: u64 = 0;
    for (gram) |t| h = h *% 0x100000001b3 ^ t;
    return h;
}

/// Detect a NEAR-repeat tail loop: the last `near_repeat_window` tokens keep
/// restating the same thing in slightly different words. Pure; reads only the
/// tail, so cost is independent of total generated length.
pub fn isNearRepeatTailLoop(tokens: []const u32) bool {
    if (tokens.len < near_repeat_window) return false;
    return nearRepeatWindowIsDegenerate(tokens[tokens.len - near_repeat_window ..]);
}

/// How far back the near-repeat trim search steps, and how far it may reach.
/// The window is judged whole at each stop, so the set stays window-sized
/// however long the loop ran — sizing a set to the FULL span instead would
/// put tens of KB on the inference thread's stack.
pub const near_repeat_step: usize = 128;
/// Live 2026-09-15: a 27B wrote five 24x19 maps of identical rows in 2.8k tokens.
pub const near_repeat_min_span: usize = 4096;
pub const near_repeat_max_lookback: usize = 8192;

/// A convicted degenerate tail: which tier saw it, and where the degenerate
/// span BEGINS in the generated ids.
///
/// The start is what makes a loop cut recoverable. An agent client re-sends
/// the cut turn's content as history, the model reads its own loop back and
/// resumes it — five loop-stops in a row, each firing sooner than the last
/// (live 2026-08-05, under pi). Emitting only the prefix means
/// the loop cannot round-trip into the next prompt.
pub const DegenerateTail = struct {
    tier: Tier,
    /// First index of the degenerate span; `tokens[0..start]` is what a
    /// client should be shown. For the exact tiers ONE copy of the cycle is
    /// deliberately kept — the truncated answer should still show what the
    /// model was doing when it got stuck, and one copy cannot sustain a loop.
    start: usize,

    pub const Tier = enum { exact_cycle, long_cycle, near_repeat };
};

/// The smallest period in `[min_period, max_period]` whose cycle repeats
/// `reps` times at the tail, or null. `isDegenerateTailLoopRange` is this
/// predicate — one implementation, so the detector and the trim can never
/// disagree about what was convicted.
fn exactCyclePeriod(tokens: []const u32, min_period: usize, max_period: usize, reps: usize, min_span: usize) ?usize {
    if (max_period == 0 or reps < 2) return null;
    var p: usize = @max(min_period, 1);
    while (p <= max_period) : (p += 1) {
        const span = @max(p * reps, min_span);
        if (tokens.len < span) continue;
        const tail = tokens[tokens.len - span ..];
        var periodic = true;
        var i: usize = p;
        while (i < tail.len) : (i += 1) {
            if (tail[i] != tail[i - p]) {
                periodic = false;
                break;
            }
        }
        if (periodic) return p;
    }
    return null;
}

/// Walk the period-`p` cycle backwards past the `reps` that convicted it: a
/// loop that ran 200 times must be trimmed at 200, not at the threshold.
fn trailingCycleStart(tokens: []const u32, p: usize) usize {
    var start = tokens.len - p;
    while (start >= p) {
        if (!std.mem.eql(u32, tokens[start - p .. start], tokens[start .. start + p])) break;
        start -= p;
    }
    return start; // first index of the FIRST copy of the cycle
}

/// Convict a degenerate tail and say where it starts. Tier order matches
/// `scheduler.loopStopReason`: the exact tiers speak first, and the fuzzy
/// near-repeat tier only ever judges spans they have already declined.
pub fn degenerateTail(tokens: []const u32) ?DegenerateTail {
    if (exactCyclePeriod(tokens, 1, degenerate_loop_max_period, degenerate_loop_reps, degenerate_loop_min_span)) |p| {
        return .{ .tier = .exact_cycle, .start = trailingCycleStart(tokens, p) + p };
    }
    if (exactCyclePeriod(
        tokens,
        degenerate_loop_max_period + 1,
        degenerate_loop_long_max_period,
        degenerate_loop_long_reps,
        degenerate_loop_long_min_span,
    )) |p| {
        return .{ .tier = .long_cycle, .start = trailingCycleStart(tokens, p) + p };
    }
    if (tokens.len < near_repeat_window) return null;
    if (!nearRepeatWindowIsDegenerate(tokens[tokens.len - near_repeat_window ..])) return null;

    // Slide the window back while it keeps convicting. A restatement loop
    // that has run for 3000 tokens is degenerate for all 3000 — trimming only
    // the last window would hand the client the rest of the loop back.
    var start = tokens.len - near_repeat_window;
    const floor = if (tokens.len > near_repeat_window + near_repeat_max_lookback)
        tokens.len - near_repeat_window - near_repeat_max_lookback
    else
        0;
    while (start >= floor + near_repeat_step) {
        const cand = start - near_repeat_step;
        if (!nearRepeatWindowIsDegenerate(tokens[cand .. cand + near_repeat_window])) break;
        start = cand;
    }
    // A file of near-identical rows (a lazy tile map, a zero-heavy table)
    // reads as a loop under every content measure; what it does that a loop
    // never does is END. Convict only once the degenerate span has outrun any
    // such file — a real restatement loop is still cut at ~4k tokens instead
    // of max_tokens, and a false cut destroys work.
    if (tokens.len - start < near_repeat_min_span) return null;
    return .{ .tier = .near_repeat, .start = start };
}

/// Stall clock for the request timeout: the deadline measures time since the
/// last PRODUCED token, not since the request started. A wall-clock request
/// timeout kills legitimate long generations — live capture 2026-07-03:
/// Qwen3.6-27B writing a 33KB file in one tool call decodes for >300s at
/// ~30 tok/s and was guillotined mid-call by the 300s default, which then
/// surfaced as a "butchered" path-only tool call. Progress is detected from
/// the generated-token COUNT at each check, so every decode path (regular,
/// drafter, MTP — which don't all share an emit site) resets the clock
/// without instrumentation; a request that stops producing (hung forward,
/// deadlock) still times out after `timeout_ns` of silence.
pub const StallClock = struct {
    last_progress_ns: u64 = 0,
    last_progress_count: usize = 0,

    pub fn expired(self: *StallClock, now_ns: u64, generated_count: usize, timeout_ns: u64) bool {
        if (generated_count != self.last_progress_count) {
            self.last_progress_count = generated_count;
            self.last_progress_ns = now_ns;
        }
        if (timeout_ns == 0) return false;
        return now_ns -| self.last_progress_ns >= timeout_ns;
    }
};

/// Detect a degenerate tail loop: the model is stuck emitting the same short
/// token cycle over and over. Returns true when the last `reps` repetitions of
/// some period-`p` cycle (1 ≤ p ≤ `max_period`) are byte-identical.
///
/// Motivation: Gemma 4 12B sometimes collapses after a large/confusing tool
/// result and spams the thinking opener `<|channel>thought` forever; with no
/// repeat penalty (the default) and a now-generous `max_tokens`, nothing else
/// stops it. The decode loop calls this each tick and cuts the slot short.
///
/// Pure and cheap: only the trailing `max_period * reps` ids are inspected, so
/// cost is independent of total generated length.
pub fn isDegenerateTailLoop(tokens: []const u32, max_period: usize, reps: usize) bool {
    return exactCyclePeriod(tokens, 1, max_period, reps, degenerate_loop_min_span) != null;
}

/// Range variant so a long-period tier can scan 9..64 without also lowering
/// the rep threshold for short cycles (a few "ha ha ha" reps stay legal).
pub fn isDegenerateTailLoopRange(tokens: []const u32, min_period: usize, max_period: usize, reps: usize) bool {
    return exactCyclePeriod(tokens, min_period, max_period, reps, 0) != null;
}

/// Compute a pooled (per the model's `pooling_mode` — mean by default),
/// L2-normalized embedding from token IDs.
/// Returns a float32 array of shape [hidden_size]. Caller must free the returned slice.
pub fn computeEmbedding(
    allocator: std.mem.Allocator,
    xfm: *Transformer,
    token_ids: []const u32,
) ![]f32 {
    const seqs = [_][]const u32{token_ids};
    const rows = try computeEmbeddingsBatch(allocator, xfm, &seqs);
    defer allocator.free(rows);
    return rows[0];
}

/// GPU batch-size cap for encoder embedding forwards: bounds padded-batch
/// memory while keeping the GPU saturated.
pub const EMBED_MAX_BATCH: usize = 64;

/// Padded-token budget per embedding sub-batch (issue #117): the item cap
/// alone lets ONE long input inflate the whole padded allocation (64 rows
/// padded to a 32K outlier = 2M positions of hidden state). The budget bounds
/// rows × padded length at the historical worst case (64 rows × 512 — every
/// legacy BERT batch is unchanged); long-context inputs simply ride smaller
/// sub-batches. A single over-budget input still runs, alone.
pub const EMBED_TOKEN_BUDGET: usize = EMBED_MAX_BATCH * 512;

/// End index (exclusive) of the embedding sub-batch starting at `start`:
/// grows while under BOTH the item cap and the padded-footprint budget
/// (rows × running max length). Always takes at least one item, so an input
/// longer than the whole budget is processed rather than looping forever.
/// Input order is preserved — sub-batches are contiguous slices.
pub fn embedSubBatchEnd(seqs: []const []const u32, start: usize, max_items: usize, budget: usize) usize {
    var end = start;
    var max_len: usize = 0;
    while (end < seqs.len and end - start < max_items) {
        const grown_max = @max(max_len, seqs[end].len);
        if (end > start and (end - start + 1) * grown_max > budget) break;
        max_len = grown_max;
        end += 1;
    }
    return end;
}

/// One padded batch of token sequences ready for an encoder forward.
pub const PaddedBatch = struct {
    ids: []i32, // [B * max_len] row-major, pad id 0
    lengths: []usize, // [B]
    max_len: usize,

    pub fn deinit(self: *PaddedBatch, allocator: std.mem.Allocator) void {
        allocator.free(self.ids);
        allocator.free(self.lengths);
    }
};

/// Pad `seqs` into one [B, max_len] i32 buffer (pad id 0). Padded positions
/// are excluded from attention (`buildKeyPadMask`) and pooling
/// (`maskedMeanPoolNormalize`), so the pad id value never leaks into results.
pub fn buildPaddedBatch(allocator: std.mem.Allocator, seqs: []const []const u32) !PaddedBatch {
    var max_len: usize = 0;
    for (seqs) |seq| max_len = @max(max_len, seq.len);
    if (max_len == 0) return error.EmptyInput;

    const ids = try allocator.alloc(i32, seqs.len * max_len);
    errdefer allocator.free(ids);
    const lengths = try allocator.alloc(usize, seqs.len);
    errdefer allocator.free(lengths);
    @memset(ids, 0);
    for (seqs, 0..) |seq, b| {
        lengths[b] = seq.len;
        for (seq, 0..) |id, t| ids[b * max_len + t] = @intCast(id);
    }
    return .{ .ids = ids, .lengths = lengths, .max_len = max_len };
}

/// Additive key-padding mask [B, 1, 1, max_len] (bf16): 0 over real keys,
/// -inf over padding. Broadcasts across heads and query positions; padded
/// QUERIES still produce garbage rows, but pooling drops them.
pub fn buildKeyPadMask(allocator: std.mem.Allocator, lengths: []const usize, max_len: usize, s: mlx.mlx_stream) !mlx.mlx_array {
    const buf = try allocator.alloc(f32, lengths.len * max_len);
    defer allocator.free(buf);
    for (lengths, 0..) |len, b| {
        for (0..max_len) |t| {
            buf[b * max_len + t] = if (t < len) 0 else -std.math.inf(f32);
        }
    }
    const shape = [_]c_int{ @intCast(lengths.len), 1, 1, @intCast(max_len) };
    const f32_mask = mlx.mlx_array_new_data(buf.ptr, &shape, 4, .float32);
    defer _ = mlx.mlx_array_free(f32_mask);
    var mask = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&mask, f32_mask, .bfloat16, s));
    return mask;
}

/// Mean-pool `hidden` [B, T, H] over each row's first `lengths[b]` positions.
/// Returns the pooled [B, H] mlx array (f32-promoted); caller frees.
pub fn maskedMeanPool(allocator: std.mem.Allocator, hidden: mlx.mlx_array, lengths: []const usize, s: mlx.mlx_stream) !mlx.mlx_array {
    const shape = mlx.getShape(hidden);
    const batch: usize = @intCast(shape[0]);
    const seq_len: usize = @intCast(shape[1]);

    // Pool weights [B, T, 1]: 1/len over real positions, 0 over padding — a
    // weighted sum along T is then exactly the masked mean. f32 weights also
    // promote a bf16 hidden so the final data extraction is float32-safe.
    const wbuf = try allocator.alloc(f32, batch * seq_len);
    defer allocator.free(wbuf);
    for (lengths, 0..) |len, b| {
        const denom: f32 = @floatFromInt(@max(len, 1));
        for (0..seq_len) |t| {
            wbuf[b * seq_len + t] = if (t < len) 1.0 / denom else 0.0;
        }
    }
    const wshape = [_]c_int{ shape[0], shape[1], 1 };
    const weights = mlx.mlx_array_new_data(wbuf.ptr, &wshape, 3, .float32);
    defer _ = mlx.mlx_array_free(weights);

    var weighted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(weighted);
    try mlx.check(mlx.mlx_multiply(&weighted, hidden, weights, s));

    var pooled = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(pooled);
    try mlx.check(mlx.mlx_sum_axis(&pooled, weighted, 1, false, s)); // [B, H]
    return pooled;
}

/// Mean-pool `hidden` [B, T, H] over each row's first `lengths[b]` positions
/// and L2-normalize. Returns B owned rows of H f32 each (plus the outer
/// slice); caller frees all.
pub fn maskedMeanPoolNormalize(allocator: std.mem.Allocator, hidden: mlx.mlx_array, lengths: []const usize, s: mlx.mlx_stream) ![][]f32 {
    const pooled = try maskedMeanPool(allocator, hidden, lengths, s);
    defer _ = mlx.mlx_array_free(pooled);
    return l2NormalizeRows(allocator, pooled, s);
}

/// Single-position pooling (issue #116): select ONE hidden row per batch
/// element — the CLS token (position 0; bge/mxbai) or the last real,
/// non-padding token (`lengths[b]-1`; Qwen3-Embedding). Same one-hot
/// weighted-sum shape as `maskedMeanPool`, so padded garbage can never leak
/// and a bf16 hidden is promoted to f32 by the weights. Returns the pooled
/// [B, H] mlx array; caller frees.
pub fn gatherTokenPool(allocator: std.mem.Allocator, hidden: mlx.mlx_array, lengths: []const usize, mode: model_mod.PoolingMode, s: mlx.mlx_stream) !mlx.mlx_array {
    const shape = mlx.getShape(hidden);
    const batch: usize = @intCast(shape[0]);
    const seq_len: usize = @intCast(shape[1]);

    const wbuf = try allocator.alloc(f32, batch * seq_len);
    defer allocator.free(wbuf);
    @memset(wbuf, 0);
    for (lengths, 0..) |len, b| {
        const pos: usize = switch (mode) {
            .cls => 0,
            .last_token => @max(len, 1) - 1,
            .mean => return error.InvalidPoolingMode,
        };
        wbuf[b * seq_len + @min(pos, seq_len - 1)] = 1.0;
    }
    const wshape = [_]c_int{ shape[0], shape[1], 1 };
    const weights = mlx.mlx_array_new_data(wbuf.ptr, &wshape, 3, .float32);
    defer _ = mlx.mlx_array_free(weights);

    var weighted = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(weighted);
    try mlx.check(mlx.mlx_multiply(&weighted, hidden, weights, s));

    var pooled = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(pooled);
    try mlx.check(mlx.mlx_sum_axis(&pooled, weighted, 1, false, s)); // [B, H]
    return pooled;
}

/// L2-normalize each row of `pooled` [B, H] and read out as owned f32 rows.
pub fn l2NormalizeRows(allocator: std.mem.Allocator, pooled: mlx.mlx_array, s: mlx.mlx_stream) ![][]f32 {
    const pshape = mlx.getShape(pooled);
    const batch: usize = @intCast(pshape[0]);
    const dim: usize = @intCast(pshape[1]);

    // L2 normalize rows: pooled / max(sqrt(sum(pooled^2)), eps).
    var squared = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(squared);
    try mlx.check(mlx.mlx_multiply(&squared, pooled, pooled, s));

    var sum_sq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sum_sq);
    try mlx.check(mlx.mlx_sum_axis(&sum_sq, squared, -1, true, s));

    var norm = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(norm);
    try mlx.check(mlx.mlx_sqrt(&norm, sum_sq, s));

    const eps = mlx.mlx_array_new_float(1e-12);
    defer _ = mlx.mlx_array_free(eps);
    var norm_safe = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(norm_safe);
    try mlx.check(mlx.mlx_maximum(&norm_safe, norm, eps, s));

    var normalized = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(normalized);
    try mlx.check(mlx.mlx_divide(&normalized, pooled, norm_safe, s));

    try mlx.check(mlx.mlx_array_eval(normalized));
    const data_ptr = mlx.mlx_array_data_float32(normalized) orelse return error.MlxError;

    const rows = try allocator.alloc([]f32, batch);
    var done: usize = 0;
    errdefer {
        for (rows[0..done]) |r| allocator.free(r);
        allocator.free(rows);
    }
    for (rows, 0..) |*row, b| {
        row.* = try allocator.alloc(f32, dim);
        @memcpy(row.*, data_ptr[b * dim .. (b + 1) * dim]);
        done += 1;
    }
    return rows;
}

/// Compute embeddings for many token sequences in GPU batches: each chunk of
/// up to `EMBED_MAX_BATCH` sequences is padded to its own max length,
/// forwarded ONCE through the encoder with a key-padding mask,
/// masked-mean-pooled, and L2-normalized. Input order preserved. Caller
/// frees every returned row and the outer slice.
pub fn computeEmbeddingsBatch(
    allocator: std.mem.Allocator,
    xfm: *Transformer,
    seqs: []const []const u32,
) ![][]f32 {
    const results = try allocator.alloc([]f32, seqs.len);
    var filled: usize = 0;
    errdefer {
        for (results[0..filled]) |r| allocator.free(r);
        allocator.free(results);
    }
    var start: usize = 0;
    while (start < seqs.len) {
        // A decoder-arch embedder (Qwen3-Embedding) forwards through the KV
        // cache: without a reset, a sub-batch attends to earlier sub-batches' rows.
        try xfm.resetCache();
        const sub = seqs[start..embedSubBatchEnd(seqs, start, EMBED_MAX_BATCH, EMBED_TOKEN_BUDGET)];
        var pb = try buildPaddedBatch(allocator, sub);
        defer pb.deinit(allocator);

        const shape = [_]c_int{ @intCast(sub.len), @intCast(pb.max_len) };
        const input = mlx.mlx_array_new_data(pb.ids.ptr, &shape, 2, .int32);
        defer _ = mlx.mlx_array_free(input);

        // A single sequence has no padding, so it needs no mask.
        var mask: ?mlx.mlx_array = null;
        defer if (mask) |m| {
            _ = mlx.mlx_array_free(m);
        };
        if (sub.len > 1) mask = try buildKeyPadMask(allocator, pb.lengths, pb.max_len, xfm.s);

        const hidden = try xfm.forwardEmbeddingMasked(input, mask);
        defer _ = mlx.mlx_array_free(hidden);

        // Pool (per the checkpoint's declared mode — mean by default) → normalize.
        const pooled = switch (xfm.config.effectivePooling()) {
            .mean => try maskedMeanPool(allocator, hidden, pb.lengths, xfm.s),
            .cls, .last_token => |m| try gatherTokenPool(allocator, hidden, pb.lengths, m, xfm.s),
        };
        defer _ = mlx.mlx_array_free(pooled);
        const rows = try l2NormalizeRows(allocator, pooled, xfm.s);
        defer allocator.free(rows);
        for (rows, 0..) |r, i| {
            results[start + i] = r;
            filled += 1;
        }
        start += sub.len;
    }
    return results;
}

const SampleResult = struct {
    token_id: u32,
    logprob_result: ?LogprobResult = null,
};

/// A host-decided think penalty for one sample: lower the `mask` ids by `lambda`.
const MarkerShift = struct { mask: ?mlx.mlx_array, state: ThinkPenalty };

/// Sample a token from the last position's logits.
/// temperature < 0.01 (`isGreedyTemperature`): greedy argmax. Otherwise: scale logits, apply top_p, and sample.
/// If logprobs_n > 0, also computes logprobs for the sampled token and top N alternatives.
fn sampleToken(allocator: std.mem.Allocator, logits: mlx.mlx_array, sampling: SamplingParams, generated_ids: ?[]const u32, logprobs_n: u32, marker_shift: ?MarkerShift, s: mlx.mlx_stream) !SampleResult {
    const shape = mlx.getShape(logits);
    const seq_len = shape[1];

    // Extract last position: [1, seq_len, vocab] -> [1, vocab]
    var last_logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(last_logits);

    if (seq_len == 1) {
        const sq_shape = [_]c_int{ 1, shape[2] };
        try mlx.check(mlx.mlx_reshape(&last_logits, logits, &sq_shape, 2, s));
    } else {
        const start = [_]c_int{ 0, seq_len - 1, 0 };
        const stop = [_]c_int{ 1, seq_len, shape[2] };
        const strides = [_]c_int{ 1, 1, 1 };
        var sliced = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sliced);
        try mlx.check(mlx.mlx_slice(&sliced, logits, &start, 3, &stop, 3, &strides, 3, s));

        const sq_shape = [_]c_int{ 1, shape[2] };
        try mlx.check(mlx.mlx_reshape(&last_logits, sliced, &sq_shape, 2, s));
    }

    // Track current working logits (avoid copies when no transform needed)
    var current = last_logits;

    // Reserved-token suppression — before any other transform, so the greedy
    // arm below can't draw a suppressed id. `logprobs_logits` stays
    // `last_logits`: the reported distribution is the MODEL's, so under
    // suppression rank 1 may legitimately differ from the chosen token.
    var suppressed = mlx.mlx_array_new();
    var suppressed_owned = false;
    defer if (suppressed_owned) {
        _ = mlx.mlx_array_free(suppressed);
    };
    if (sampling.suppress_mask) |m| {
        try applySuppressMask(&suppressed, current, m, s);
        current = suppressed;
        suppressed_owned = true;
    }

    var marked = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(marked);
    if (marker_shift) |m| {
        try think_penalty.shiftScoped(&marked, current, m.mask, m.state, null, s);
        current = marked;
    }

    // Apply repeat penalty to already-generated tokens
    var penalized = mlx.mlx_array_new();
    var penalized_owned = false;
    defer if (penalized_owned) {
        _ = mlx.mlx_array_free(penalized);
    };

    if (penaltyActive(sampling)) {
        if (generated_ids) |ids| {
            if (ids.len > 0) {
                try applyRepeatPenalty(&penalized, current, ids, sampling.repeat_penalty, sampling.presence_penalty, s);
                current = penalized;
                penalized_owned = true;
            }
        }
    }

    // Greedy if temperature is ~0
    if (isGreedyTemperature(sampling.temperature)) {
        const token_id = try argmax(current, s);
        var logprob_result: ?LogprobResult = null;
        if (logprobs_n > 0) {
            logprob_result = try computeLogprobs(allocator, last_logits, token_id, logprobs_n, s);
        }
        return .{ .token_id = token_id, .logprob_result = logprob_result };
    }

    // Scale logits by 1/temperature
    var scaled = mlx.mlx_array_new();
    var scaled_owned = false;
    defer if (scaled_owned) {
        _ = mlx.mlx_array_free(scaled);
    };

    if (sampling.temperature != 1.0) {
        const temp_arr = mlx.mlx_array_new_float(sampling.temperature);
        defer _ = mlx.mlx_array_free(temp_arr);
        try mlx.check(mlx.mlx_divide(&scaled, current, temp_arr, s));
        current = scaled;
        scaled_owned = true;
    }

    // Logprobs report the MODEL's distribution, so they read the position's raw
    // logits — not `current`, which by here carries the client's temperature
    // (and any penalty) and would make the same token's logprob move with a
    // sampling knob.
    const logprobs_logits = last_logits;

    // Apply top-k and top-p in one pass
    var filtered = mlx.mlx_array_new();
    var filtered_owned = false;
    defer if (filtered_owned) {
        _ = mlx.mlx_array_free(filtered);
    };

    if (sampling.top_k > 0 or sampling.top_p < 1.0) {
        try filterTopKTopP(&filtered, current, sampling.top_p, sampling.top_k, s);
        current = filtered;
        filtered_owned = true;
    }

    // Sample from categorical distribution
    var sampled = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sampled);

    const key = seedKey(sampling);
    defer _ = mlx.mlx_array_free(key);
    try mlx.check(mlx.mlx_random_categorical(&sampled, current, -1, key, s));

    // Eval and extract
    try mlx.check(mlx.mlx_array_eval(sampled));
    var val: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&val, sampled));

    const token_id: u32 = @intCast(val);

    // Compute logprobs after sampling (we now know the token_id)
    var logprob_result: ?LogprobResult = null;
    if (logprobs_n > 0) {
        logprob_result = try computeLogprobs(allocator, logprobs_logits, token_id, logprobs_n, s);
    }

    return .{ .token_id = token_id, .logprob_result = logprob_result };
}

/// Logprobs for the FIRST generated token, taken from the prefill's
/// final-position logits — the one distribution the decode loop never sees.
/// `chosen` is the id the lazy sampler actually drew, so the reported
/// `token_logprob` belongs to the token that was really emitted (re-sampling
/// here would disagree with it at any temperature above 0).
fn firstTokenLogprobs(allocator: std.mem.Allocator, logits: mlx.mlx_array, chosen: u32, logprobs_n: u32, s: mlx.mlx_stream) !?LogprobResult {
    if (logprobs_n == 0) return null;
    const shape = mlx.getShape(logits);
    if (shape.len != 3) return null;
    var last = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(last);
    const sq_shape = [_]c_int{ 1, shape[2] };
    if (shape[1] == 1) {
        try mlx.check(mlx.mlx_reshape(&last, logits, &sq_shape, 2, s));
    } else {
        const start = [_]c_int{ 0, shape[1] - 1, 0 };
        const stop = [_]c_int{ 1, shape[1], shape[2] };
        const strides = [_]c_int{ 1, 1, 1 };
        var sliced = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sliced);
        try mlx.check(mlx.mlx_slice(&sliced, logits, &start, 3, &stop, 3, &strides, 3, s));
        try mlx.check(mlx.mlx_reshape(&last, sliced, &sq_shape, 2, s));
    }
    return try computeLogprobs(allocator, last, chosen, logprobs_n, s);
}

/// Compute log-probabilities from logits. Returns the logprob of the chosen token
/// and the top N alternatives with their token IDs and logprobs.
///
/// `logits` are the MODEL's logits for the position — before temperature,
/// penalties and top-k/top-p — so the reported distribution is the model's, as
/// OpenAI's is. Reading them after temperature made every value move with a
/// knob the client set (and at temp 0 the distribution SATURATES, so most
/// entries report exactly 0.0).
///
/// Ids travel WITH their values through `mlx_argpartition_axis` + a gather.
/// Recovering them afterwards by scanning the vocab for float equality — what
/// this did — is ambiguous the moment two logits tie, which under the
/// saturation above is everywhere: rank 1 was measured to be the chosen token
/// in 0 of 5 positions on a trivial greedy prompt.
fn computeLogprobs(allocator: std.mem.Allocator, logits: mlx.mlx_array, chosen_token: u32, top_n: u32, s: mlx.mlx_stream) !LogprobResult {
    // log_softmax in f32: `log(softmax(x))` in bf16 lands on bf16's grid (0.125 apart
    // between -16 and -32), and f16 underflows to -inf.
    var logits32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(logits32);
    try mlx.check(mlx.mlx_astype(&logits32, logits, .float32, s));
    var lse = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(lse);
    try mlx.check(mlx.mlx_logsumexp_axis(&lse, logits32, -1, true, s));
    var log_probs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(log_probs);
    try mlx.check(mlx.mlx_subtract(&log_probs, logits32, lse, s));

    const lp_shape = mlx.getShape(log_probs);
    const rank = lp_shape.len;
    const vocab_size: usize = @intCast(lp_shape[rank - 1]);
    const k: usize = @min(@as(usize, @min(top_n, 20)), vocab_size);

    // Top-k INDICES, carried alongside their values. Negating turns "k largest"
    // into the "k smallest" that argpartition puts in the leading slots.
    var neg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(neg);
    try mlx.check(mlx.mlx_negative(&neg, log_probs, s));

    var part_idx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(part_idx);
    try mlx.check(mlx.mlx_argpartition_axis(&part_idx, neg, @intCast(if (k == 0) 0 else k - 1), -1, s));

    var start_buf: [8]c_int = @splat(0);
    var stop_buf: [8]c_int = @splat(1);
    var stride_buf: [8]c_int = @splat(1);
    for (0..rank) |i| stop_buf[i] = lp_shape[i];
    stop_buf[rank - 1] = @intCast(k);

    var idx_k = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(idx_k);
    try mlx.check(mlx.mlx_slice(&idx_k, part_idx, &start_buf, rank, &stop_buf, rank, &stride_buf, rank, s));

    var vals_raw = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vals_raw);
    try mlx.check(mlx.mlx_take_axis(&vals_raw, log_probs, idx_k, @intCast(rank - 1), s));

    var vals_k = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vals_k);
    try mlx.check(mlx.mlx_astype(&vals_k, vals_raw, .float32, s));

    var ids_k = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids_k);
    try mlx.check(mlx.mlx_astype(&ids_k, idx_k, .int32, s));

    try mlx.check(mlx.mlx_array_eval(log_probs));
    try mlx.check(mlx.mlx_array_eval(vals_k));
    try mlx.check(mlx.mlx_array_eval(ids_k));

    // Read the logprob of the chosen token from the full array
    const lp_data = mlx.mlx_array_data_float32(log_probs);
    const chosen_logprob: f32 = if (lp_data) |ptr|
        (if (chosen_token < vocab_size) ptr[chosen_token] else -100.0)
    else
        -100.0;

    const val_ptr = mlx.mlx_array_data_float32(vals_k);
    const id_ptr = mlx.mlx_array_data_int32(ids_k);

    var top_logprobs = try allocator.alloc(TokenLogprob, k);
    errdefer allocator.free(top_logprobs);
    var filled: usize = 0;
    if (val_ptr) |vp| {
        if (id_ptr) |ip| {
            for (0..k) |i| {
                const tid = ip[i];
                if (tid < 0 or @as(usize, @intCast(tid)) >= vocab_size) continue;
                top_logprobs[filled] = .{ .token_id = @intCast(tid), .logprob = vp[i] };
                filled += 1;
            }
        }
    }
    // argpartition leaves the winners UNORDERED; ties break on the lower id so
    // the ranking is deterministic run to run.
    std.mem.sort(TokenLogprob, top_logprobs[0..filled], {}, struct {
        fn lt(_: void, a: TokenLogprob, b: TokenLogprob) bool {
            if (a.logprob != b.logprob) return a.logprob > b.logprob;
            return a.token_id < b.token_id;
        }
    }.lt);

    if (filled < top_logprobs.len) {
        top_logprobs = allocator.realloc(top_logprobs, filled) catch top_logprobs;
    }

    return .{
        .token_logprob = chosen_logprob,
        .top_logprobs = top_logprobs,
    };
}

/// Apply a grammar token mask to logits. `mask[i]==true` keeps `logits[i]`,
/// `false` replaces it with `-inf`. The mask is broadcast over leading dims so
/// `logits` can be either `[1, vocab]` or `[1, 1, vocab]`.
fn applyGrammarMask(allocator: std.mem.Allocator, res: *mlx.mlx_array, logits: mlx.mlx_array, mask: []const bool, s: mlx.mlx_stream) !void {
    const shape = mlx.getShape(logits);
    const vocab_size: usize = @intCast(shape[shape.len - 1]);
    const logit_mask = try maskForLogitVocab(allocator, mask, vocab_size);
    defer logit_mask.deinit(allocator);

    // Zig's `bool` is one byte and matches MLX's `.bool_` storage exactly.
    const arr_shape = [_]c_int{@intCast(vocab_size)};
    const mask_arr = mlx.mlx_array_new_data(@ptrCast(logit_mask.slice.ptr), &arr_shape, 1, .bool_);
    defer _ = mlx.mlx_array_free(mask_arr);

    const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer _ = mlx.mlx_array_free(neg_inf);

    try mlx.check(mlx.mlx_where(res, mask_arr, logits, neg_inf, s));
}

const LogitMaskView = struct {
    slice: []const bool,
    owned: ?[]bool = null,

    fn deinit(self: LogitMaskView, allocator: std.mem.Allocator) void {
        if (self.owned) |buf| allocator.free(buf);
    }
};

fn maskForLogitVocab(allocator: std.mem.Allocator, mask: []const bool, vocab_size: usize) !LogitMaskView {
    if (mask.len == vocab_size) return .{ .slice = mask };

    var adjusted = try allocator.alloc(bool, vocab_size);
    @memset(adjusted, false);
    const copy_len = @min(mask.len, vocab_size);
    @memcpy(adjusted[0..copy_len], mask[0..copy_len]);
    return .{ .slice = adjusted, .owned = adjusted };
}

/// Rank cap for the filter helpers. The widest shape a caller passes is
/// `[B, S, V]`; anything past this is a named error, never a wrong axis.
const SAMPLER_MAX_NDIM = 4;

/// `a[..., start..stop]`, every leading axis passed through whole.
fn sliceLastAxis(res: *mlx.mlx_array, a: mlx.mlx_array, start: usize, stop: usize, s: mlx.mlx_stream) !void {
    const shape = mlx.getShape(a);
    if (shape.len > SAMPLER_MAX_NDIM) return error.UnsupportedSamplerRank;
    var lo: [SAMPLER_MAX_NDIM]c_int = undefined;
    var hi: [SAMPLER_MAX_NDIM]c_int = undefined;
    var st: [SAMPLER_MAX_NDIM]c_int = undefined;
    for (shape, 0..) |dim, i| {
        const last = i == shape.len - 1;
        lo[i] = if (last) @intCast(start) else 0;
        hi[i] = if (last) @intCast(stop) else dim;
        st[i] = 1;
    }
    const n = shape.len;
    try mlx.check(mlx.mlx_slice(res, a, &lo, n, &hi, n, &st, n, s));
}

/// Column ids of `a` ranked by value DESCENDING, ties by LOWEST ORIGINAL INDEX.
///
/// mlx's merge sort is stable — `ThreadSort` swaps only on a strict less-than and
/// the merge takes from B only when `b < a` — so an ascending argsort of the
/// negated row is exactly that order, and it is the `sushi_qsa_select` tie rule.
/// Both filters and both of their routes rank through here, which is what makes a
/// shortlist cutoff agree with the whole row's on a tie-heavy row. Pinned by
/// `sampler ranks break ties by the lowest original index`.
fn ranksDescending(res: *mlx.mlx_array, a: mlx.mlx_array, s: mlx.mlx_stream) !void {
    var neg = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(neg);
    try mlx.check(mlx.mlx_negative(&neg, a, s));
    try mlx.check(mlx.mlx_argsort_axis(res, neg, -1, s));
}

/// Candidate-chunk width for `topRanksDescending`: cost is one sort of
/// `V / chunk` chunk maxima plus one of `m * chunk` candidates, balanced at
/// sqrt(V / m).
fn shortlistChunkWidth(v: usize, m: usize) usize {
    const bal = @sqrt(@as(f64, @floatFromInt(v)) / @as(f64, @floatFromInt(@max(m, 1))));
    const c: usize = @intFromFloat(@max(16.0, @min(2048.0, bal)));
    return @min(c, v);
}

/// The top `m` of every row — values and their column ids, both in
/// `ranksDescending` order — without ranking the whole row.
///
/// Every element of a chunk is at most that chunk's max, so a chunk whose max is
/// not among the `m` largest chunk maxima holds no top-`m` element: at most `m`
/// chunks can, and ranking their `m * chunk` candidates is exact. On Metal a
/// full-row `mlx_argsort_axis` is a multi-block merge sort over the whole
/// vocabulary — and so are `mlx_topk` and `mlx_argpartition`, which mlx routes to
/// that same sort, so neither of those buys a shortlist.
fn topRanksDescending(
    vals: *mlx.mlx_array,
    ids: *mlx.mlx_array,
    logits: mlx.mlx_array,
    m: u32,
    s: mlx.mlx_stream,
) !void {
    const shape = mlx.getShape(logits);
    if (shape.len == 0 or shape.len > SAMPLER_MAX_NDIM) return error.UnsupportedSamplerRank;
    const v: usize = @intCast(shape[shape.len - 1]);
    const want: usize = @min(@as(usize, m), v);
    if (want == 0) return error.EmptyShortlist;
    const chunk = shortlistChunkWidth(v, want);
    const groups = (v + chunk - 1) / chunk;

    // Nothing left to win once the candidate set is the row itself.
    if (want >= v or want * chunk >= v) {
        var order = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(order);
        try ranksDescending(&order, logits, s);
        try sliceLastAxis(ids, order, 0, want, s);
        return mlx.check(mlx.mlx_take_along_axis(vals, logits, ids.*, -1, s));
    }

    // Pad the last chunk with -inf so the reshape is exact.
    var padded = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(padded);
    if (groups * chunk != v) {
        const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
        defer _ = mlx.mlx_array_free(neg_inf);
        var pad_val = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(pad_val);
        try mlx.check(mlx.mlx_astype(&pad_val, neg_inf, mlx.mlx_array_dtype(logits), s));
        const axes = [_]c_int{@intCast(shape.len - 1)};
        const low = [_]c_int{0};
        const high = [_]c_int{@intCast(groups * chunk - v)};
        try mlx.check(mlx.mlx_pad(&padded, logits, &axes, 1, &low, 1, &high, 1, pad_val, "constant", s));
    } else {
        try mlx.check(mlx.mlx_array_set(&padded, logits));
    }

    var dims: [SAMPLER_MAX_NDIM + 1]c_int = undefined;
    @memcpy(dims[0 .. shape.len - 1], shape[0 .. shape.len - 1]);
    const lead = shape.len - 1;
    dims[lead] = @intCast(groups);
    dims[lead + 1] = @intCast(chunk);
    var blk = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(blk);
    try mlx.check(mlx.mlx_reshape(&blk, padded, &dims, lead + 2, s));

    var chunk_max = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(chunk_max);
    try mlx.check(mlx.mlx_max_axis(&chunk_max, blk, -1, false, s));

    // Chunks ranked by max, ties by lowest chunk id. A tie group's contributing
    // chunks are a PREFIX of its ascending-id list — a lower id holds a lower
    // column id, which outranks — so taking the lowest ids is what keeps the
    // shortlist equal to the whole row's ranking.
    var chunk_ranks = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(chunk_ranks);
    try ranksDescending(&chunk_ranks, chunk_max, s);
    var picked = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(picked);
    try sliceLastAxis(&picked, chunk_ranks, 0, want, s);
    // Ascending chunk ids lay the candidates out in ascending COLUMN id, so the
    // stable rank below breaks candidate ties the way the whole row would.
    var win = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(win);
    try mlx.check(mlx.mlx_sort_axis(&win, picked, -1, s));
    var win_i32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(win_i32);
    try mlx.check(mlx.mlx_astype(&win_i32, win, .int32, s));

    var win_col = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(win_col);
    try mlx.check(mlx.mlx_expand_dims(&win_col, win_i32, -1, s));
    dims[lead] = @intCast(want);
    var gather_idx = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(gather_idx);
    try mlx.check(mlx.mlx_broadcast_to(&gather_idx, win_col, &dims, lead + 2, s));
    var cand_2d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cand_2d);
    try mlx.check(mlx.mlx_take_along_axis(&cand_2d, blk, gather_idx, @intCast(lead), s));

    // Column id of every candidate: chunk id * chunk + offset within the chunk.
    const chunk_arr = mlx.mlx_array_new_int(@intCast(chunk));
    defer _ = mlx.mlx_array_free(chunk_arr);
    var base = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(base);
    try mlx.check(mlx.mlx_multiply(&base, win_col, chunk_arr, s));
    var offsets = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(offsets);
    try mlx.check(mlx.mlx_arange(&offsets, 0, @floatFromInt(chunk), 1, .int32, s));
    var cand_ids_2d = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cand_ids_2d);
    try mlx.check(mlx.mlx_add(&cand_ids_2d, base, offsets, s));

    dims[lead] = @intCast(want * chunk);
    var cand_vals = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cand_vals);
    try mlx.check(mlx.mlx_reshape(&cand_vals, cand_2d, &dims, lead + 1, s));
    var cand_ids_flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cand_ids_flat);
    try mlx.check(mlx.mlx_reshape(&cand_ids_flat, cand_ids_2d, &dims, lead + 1, s));
    // A padded tail column is -inf, so it only ranks when the row holds fewer
    // than `want` finite values; clamp so the scatter stays in range regardless.
    const last_col = mlx.mlx_array_new_int(@intCast(v - 1));
    defer _ = mlx.mlx_array_free(last_col);
    var cand_ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cand_ids);
    try mlx.check(mlx.mlx_minimum(&cand_ids, cand_ids_flat, last_col, s));

    var cand_ranks = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(cand_ranks);
    try ranksDescending(&cand_ranks, cand_vals, s);
    var sel = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(sel);
    try sliceLastAxis(&sel, cand_ranks, 0, want, s);
    try mlx.check(mlx.mlx_take_along_axis(vals, cand_vals, sel, -1, s));
    try mlx.check(mlx.mlx_take_along_axis(ids, cand_ids, sel, -1, s));
}

/// Apply top-k filtering: keep only the top k logits, set the rest to -inf.
fn applyTopK(res: *mlx.mlx_array, logits: mlx.mlx_array, k: u32, s: mlx.mlx_stream) !void {
    // Per-ROW top-k by RANK: the k highest-ranked column ids, scattered back as a
    // keep mask. A value cutoff (`logits >= kth largest`) keeps every token tied
    // with the k-th, and bf16 logits tie at the top constantly on a 250k vocab —
    // top_k 1 stopped being greedy.
    const shape = mlx.getShape(logits);
    const v: usize = @intCast(shape[shape.len - 1]);
    if (k == 0 or k >= v) return mlx.check(mlx.mlx_array_set(res, logits));
    var vals = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vals);
    var ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids);
    try topRanksDescending(&vals, &ids, logits, k, s);
    try keepByIndex(res, logits, ids, s);
}

/// Mask `logits` to -inf everywhere except the per-row `keep_idx` columns.
fn keepByIndex(res: *mlx.mlx_array, logits: mlx.mlx_array, keep_idx: mlx.mlx_array, s: mlx.mlx_stream) !void {
    const shape = mlx.getShape(logits);
    var zeros = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(zeros);
    try mlx.check(mlx.mlx_zeros(&zeros, shape.ptr, shape.len, .bool_, s));
    const t = mlx.mlx_array_new_bool(true);
    defer _ = mlx.mlx_array_free(t);
    var mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask);
    try mlx.check(mlx.mlx_put_along_axis(&mask, zeros, keep_idx, t, -1, s));
    const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer _ = mlx.mlx_array_free(neg_inf);
    try mlx.check(mlx.mlx_where(res, mask, logits, neg_inf, s));
}

/// Top-k and top-p in ONE pass. A bounded `applyTopP` ranks the row's top
/// `top_k` columns and scatters its keep flags at exactly those column ids, so a
/// preceding `applyTopK` selects the same shortlist and masks a SUPERSET of what
/// the nucleus keeps — one full-vocab selection and one full-row `where` of pure
/// tax on every filtered row. Byte-identical to the pair either way round.
fn filterTopKTopP(res: *mlx.mlx_array, logits: mlx.mlx_array, top_p: f32, top_k: u32, s: mlx.mlx_stream) !void {
    if (top_p >= 1.0) return applyTopK(res, logits, top_k, s);
    return applyTopP(res, logits, top_p, top_k, s);
}

/// Apply top-p (nucleus) sampling: mask logits outside the top-p probability
/// mass. The nucleus is decided over ranked values and scattered back by column
/// id, so a token tied with the cutoff value but outside the mass is masked
/// (top_p -> 0 is greedy even on tied bf16 logits).
///
/// `nucleus_bound` is the caller's top-k. With top-k applied first the row holds
/// exactly `k` finite columns, so the nucleus cannot reach past the k-shortlist
/// and the cutoff is exact without ranking the row; 0 means the row is unfiltered
/// and the nucleus is unbounded.
fn applyTopP(res: *mlx.mlx_array, logits: mlx.mlx_array, top_p: f32, nucleus_bound: u32, s: mlx.mlx_stream) !void {
    const shape = mlx.getShape(logits);
    const v: usize = @intCast(shape[shape.len - 1]);
    const bounded = nucleus_bound > 0 and nucleus_bound < v;

    var vals = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vals);
    var ids = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ids);
    if (bounded) {
        try topRanksDescending(&vals, &ids, logits, nucleus_bound, s);
    } else {
        try ranksDescending(&ids, logits, s);
        try mlx.check(mlx.mlx_take_along_axis(&vals, logits, ids, -1, s));
    }

    // f32 throughout: mlx instantiates cumsum at the INPUT dtype and a bf16
    // accumulator loses nucleus tail mass. The ranked values ARE the row's
    // finite entries (top-k left exactly k), so their softmax is the row's.
    var vals_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(vals_f32);
    try mlx.check(mlx.mlx_astype(&vals_f32, vals, .float32, s));
    var probs = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(probs);
    try mlx.check(mlx.mlx_softmax_axis(&probs, vals_f32, -1, true, s));

    // Mass STRICTLY above each rank — an exclusive scan down the ranking, so the
    // term is a function of the shortlist alone (everything outranking a top-k
    // column is itself top-k). Rank 0 sees 0, so the argmax is always kept —
    // the threshold floors above 0 so a literal top_p 0 is greedy, not empty.
    var above = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(above);
    try mlx.check(mlx.mlx_cumsum(&above, probs, -1, false, false, s));
    const threshold = mlx.mlx_array_new_float(@max(top_p, std.math.floatMin(f32)));
    defer _ = mlx.mlx_array_free(threshold);
    var in_nucleus = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(in_nucleus);
    try mlx.check(mlx.mlx_less(&in_nucleus, above, threshold, s));

    var zeros = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(zeros);
    try mlx.check(mlx.mlx_zeros(&zeros, shape.ptr, shape.len, .bool_, s));
    var mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask);
    try mlx.check(mlx.mlx_put_along_axis(&mask, zeros, ids, in_nucleus, -1, s));
    const neg_inf = mlx.mlx_array_new_float(-std.math.inf(f32));
    defer _ = mlx.mlx_array_free(neg_inf);
    try mlx.check(mlx.mlx_where(res, mask, logits, neg_inf, s));
}

/// Apply repeat penalty to already-generated tokens.
/// Uses pure MLX GPU ops — no CPU readback, preserves lazy evaluation graph.
fn applyRepeatPenalty(res: *mlx.mlx_array, logits: mlx.mlx_array, generated_ids: []const u32, repeat_penalty: f32, presence_penalty: f32, s: mlx.mlx_stream) !void {
    return applyRepeatPenaltyWithAllocator(std.heap.page_allocator, res, logits, generated_ids, repeat_penalty, presence_penalty, s);
}

fn applyRepeatPenaltyWithAllocator(allocator: std.mem.Allocator, res: *mlx.mlx_array, logits: mlx.mlx_array, generated_ids: []const u32, repeat_penalty: f32, presence_penalty: f32, s: mlx.mlx_stream) !void {
    const shape = mlx.getShape(logits);
    const vocab_size: usize = @intCast(shape[shape.len - 1]);

    // Collect unique token ids
    var seen_set = std.AutoHashMap(u32, void).init(allocator);
    defer seen_set.deinit();
    for (generated_ids) |id| {
        if (id < vocab_size) {
            try seen_set.put(id, {});
        }
    }

    if (seen_set.count() == 0) return;

    // Build boolean mask: true at positions of seen tokens
    const mask_data = try allocator.alloc(u8, vocab_size);
    defer allocator.free(mask_data);
    @memset(mask_data, 0);

    var it = seen_set.keyIterator();
    while (it.next()) |id_ptr| {
        mask_data[id_ptr.*] = 1;
    }

    const arr_shape = [_]c_int{ 1, @intCast(vocab_size) };
    const mask_arr = mlx.mlx_array_new_data(mask_data.ptr, &arr_shape, 2, .bool_);
    defer _ = mlx.mlx_array_free(mask_arr);

    try seen_penalty.apply(res, logits, mask_arr, repeat_penalty, presence_penalty, s);
}

/// Greedy argmax over the last axis.
fn argmax(last_logits: mlx.mlx_array, s: mlx.mlx_stream) !u32 {
    var argmax_arr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(argmax_arr);
    try mlx.check(mlx.mlx_argmax_axis(&argmax_arr, last_logits, -1, false, s));

    try mlx.check(mlx.mlx_array_eval(argmax_arr));
    var val: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&val, argmax_arr));

    return @intCast(val);
}

// ── Tests ──

const testing = std.testing;
const expectError = @import("test_expect.zig").expectError;

test "SamplingParams defaults" {
    const params = SamplingParams{};
    try testing.expectApproxEqAbs(@as(f32, 1.0), params.temperature, 0.001);
    try testing.expectApproxEqAbs(@as(f32, 1.0), params.top_p, 0.001);
    try testing.expectEqual(@as(u32, 0), params.top_k);
    try testing.expectApproxEqAbs(@as(f32, 1.0), params.repeat_penalty, 0.001);
    try testing.expectApproxEqAbs(@as(f32, 0.0), params.presence_penalty, 0.001);
    try testing.expect(params.seed == null);
}

test "SamplingParams custom values" {
    const params = SamplingParams{
        .temperature = 0.7,
        .top_p = 0.9,
        .top_k = 40,
        .repeat_penalty = 1.1,
        .presence_penalty = 0.5,
        .seed = 42,
    };
    try testing.expectApproxEqAbs(@as(f32, 0.7), params.temperature, 0.001);
    try testing.expectApproxEqAbs(@as(f32, 0.9), params.top_p, 0.001);
    try testing.expectEqual(@as(u32, 40), params.top_k);
    try testing.expectEqual(@as(u64, 42), params.seed.?);
}

test "specDecodeUnsupported: release-enforced guard for spec + constraint/logprobs (issue #97)" {
    // Speculative decode (drafter/MTP) cannot honor a grammar constraint or
    // per-token logprobs. nextDflash/nextMtp only asserted this, which
    // compiles out in ReleaseFast (issue #97) — this is the real check they now
    // use to fail loud instead of streaming silently off-schema output.
    try testing.expect(!specDecodeUnsupported(.{}, 0)); // plain sampling → spec OK
    try testing.expect(specDecodeUnsupported(.{}, 1)); // any logprobs → not OK
    try testing.expect(specDecodeUnsupported(.{}, 5));
    // A grammar constraint present → not OK. The guard only null-checks the
    // pointer (never dereferences), so a dummy address is sufficient here.
    var dummy: Constraint = undefined;
    try testing.expect(specDecodeUnsupported(.{ .constraint = &dummy }, 0));
    try testing.expect(specDecodeUnsupported(.{ .constraint = &dummy }, 3));
}

test "specDecodeUnsupported: a forced tool call keeps every draft path off" {
    // Drafts are accepted in runs, so no draft path can hand the position after
    // the think closer to the forced call.
    var cf = CallForce{ .forced = &.{1}, .call_at = 0, .closer_id = null, .phase = .answer, .last = 0 };
    try testing.expect(specDecodeUnsupported(.{ .call_force = &cf }, 0));
}

test "a repeat or presence penalty keeps DFlash and the drafter off and samples on the synchronous path" {
    try testing.expect(specDecodeUnsupported(.{ .presence_penalty = 0.5 }, 0));
    try testing.expect(specDecodeUnsupported(.{ .repeat_penalty = 1.1 }, 0));
    try testing.expect(draftsRefused(.{ .presence_penalty = 0.5 }));
    try testing.expect(!draftsRefused(.{}));
    var g: Generator = undefined;
    g.logprobs_n = 0;
    g.sampling = .{ .presence_penalty = 0.5 };
    try testing.expect(g.samplesSync());
    g.sampling = .{ .repeat_penalty = 1.2 };
    try testing.expect(g.samplesSync());
    g.sampling = .{};
    try testing.expect(!g.samplesSync());
    g.logprobs_n = 2;
    try testing.expect(g.samplesSync());
}

test "penalty rows: every verify row equals its serial tick over the committed ids and the round's ids through it" {
    const a = testing.allocator;
    const s = mlx.gpuStream();
    const v: usize = 40;
    const Case = struct { committed: []const u32, ids: []const i32 };
    const cases = [_]Case{
        .{ .committed = &.{ 3, 7, 7, 12 }, .ids = &.{ 7, 5, 5, 31, 3 } },
        .{ .committed = &.{}, .ids = &.{ 9, 9, 2 } },
        .{ .committed = &.{ 0, 39 }, .ids = &.{ 20 } },
    };
    const Pen = struct { repeat: f32, presence: f32 };
    const pens = [_]Pen{ .{ .repeat = 1.3, .presence = 0 }, .{ .repeat = 1, .presence = 1.5 }, .{ .repeat = 1.2, .presence = 0.7 }, .{ .repeat = 0.8, .presence = 0.25 } };
    for ([_]mlx.mlx_dtype{ .float32, .bfloat16 }) |dtype| {
        for (cases) |c| {
            const rows = c.ids.len;
            var host: [8 * v]f32 = undefined;
            for (host[0 .. rows * v], 0..) |*x, i| x.* = @as(f32, @floatFromInt((i * 37) % 23)) * 0.375 - 4.0;
            const shape = [_]c_int{ 1, @intCast(rows), @intCast(v) };
            const raw = mlx.mlx_array_new_data(&host, &shape, 3, .float32);
            defer _ = mlx.mlx_array_free(raw);
            var block = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(block);
            try mlx.check(mlx.mlx_astype(&block, raw, dtype, s));
            const id_shape = [_]c_int{ 1, @intCast(rows) };
            const ids = mlx.mlx_array_new_data(c.ids.ptr, &id_shape, 2, .int32);
            defer _ = mlx.mlx_array_free(ids);
            const committed = try think_penalty.buildMask(a, c.committed, v);
            defer _ = mlx.mlx_array_free(committed);
            for (pens) |p| {
                var shifted = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(shifted);
                try seen_penalty.shiftRows(&shifted, block, ids, committed, p.repeat, p.presence, s);
                const got = try samplerTestReadFlat(a, shifted, rows * v, s);
                defer a.free(got);
                for (0..rows) |j| {
                    var history: [16]u32 = undefined;
                    @memcpy(history[0..c.committed.len], c.committed);
                    for (c.ids[0 .. j + 1], 0..) |id, k| history[c.committed.len + k] = @intCast(id);
                    var row = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(row);
                    try mlx.check(mlx.mlx_slice(&row, block, &.{ 0, @intCast(j), 0 }, 3, &.{ 1, @as(c_int, @intCast(j)) + 1, @intCast(v) }, 3, &.{ 1, 1, 1 }, 3, s));
                    var serial = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(serial);
                    try applyRepeatPenalty(&serial, row, history[0 .. c.committed.len + j + 1], p.repeat, p.presence, s);
                    const want = try samplerTestReadFlat(a, serial, v, s);
                    defer a.free(want);
                    try testing.expectEqualSlices(f32, want, got[j * v ..][0..v]);
                }
            }
        }
    }
}

test "a penalty keeps DFlash and the drafter off but not MTP" {
    try testing.expect(!mtpUnsupported(.{ .presence_penalty = 1.5 }, 0));
    try testing.expect(!mtpUnsupported(.{ .repeat_penalty = 1.1, .presence_penalty = 0.5 }, 0));
    try testing.expect(mtpUnsupported(.{ .presence_penalty = 1.5 }, 2));
    var cf = CallForce{ .forced = &.{1}, .call_at = 0, .closer_id = null, .phase = .answer, .last = 0 };
    try testing.expect(mtpUnsupported(.{ .call_force = &cf, .presence_penalty = 1.5 }, 0));
    try testing.expect(specDecodeUnsupported(.{ .presence_penalty = 1.5 }, 0));
    try testing.expect(mtpRefusal(.{ .presence_penalty = 1.5 }, 0) == null);
    try testing.expectEqualStrings("logprobs requested", mtpRefusal(.{}, 1).?);
    try testing.expectEqualStrings("forced tool call", mtpRefusal(.{ .call_force = &cf }, 0).?);
}

test "a repeat penalty of 0 or below is off" {
    try testing.expect(!penaltyActive(.{ .repeat_penalty = 0 }));
    try testing.expect(!penaltyActive(.{ .repeat_penalty = -1.5 }));
    try testing.expect(penaltyActive(.{ .repeat_penalty = 0, .presence_penalty = 0.3 }));
    try testing.expect(!argmaxOnlyRequest(.{ .temperature = 0, .repeat_penalty = 1.1 }, 0));
    try testing.expect(argmaxOnlyRequest(.{ .temperature = 0, .repeat_penalty = 0 }, 0));
}

test "a constrained penalised step picks what sampleToken picks on the same masked logits, penalised once" {
    const a = testing.allocator;
    const s = mlx.gpuStream();
    const v = 16;
    // Once: id 0 falls below id 1 in case A and stays above it in case B; twice would flip B too.
    const Case = struct { runner_up: f32, want: u32 };
    for ([_]Case{ .{ .runner_up = 4.8, .want = 1 }, .{ .runner_up = 4.2, .want = 0 } }) |c| {
        var host: [v]f32 = @splat(0);
        host[0] = 5.0;
        host[1] = c.runner_up;
        host[9] = 9.0;
        const shape = [_]c_int{ 1, 1, v };
        const logits = mlx.mlx_array_new_data(&host, &shape, 3, .float32);
        defer _ = mlx.mlx_array_free(logits);
        var allowed: [v]bool = @splat(true);
        allowed[9] = false;
        var masked = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(masked);
        try applyGrammarMask(a, &masked, logits, &allowed, s);
        const sampling = SamplingParams{ .temperature = 0, .presence_penalty = 0.5 };
        const generated = [_]u32{ 0, 3 };
        const sync = try sampleToken(a, masked, sampling, &generated, 0, null, s);
        const lazy = try penalizedSampleLazy(std.heap.page_allocator, masked, sampling, &generated, s);
        defer _ = mlx.mlx_array_free(lazy);
        const got = try samplerTestReadFlat(a, lazy, 1, s);
        defer a.free(got);
        try testing.expectEqual(c.want, sync.token_id);
        try testing.expectEqual(@as(f32, @floatFromInt(c.want)), got[0]);
    }
}

test "GenerationResult fields" {
    // Just verifying the struct shape compiles correctly with all fields
    const result = GenerationResult{
        .text = @constCast("hello"),
        .token_ids = @constCast(&[_]u32{ 1, 2, 3 }),
        .prompt_tokens = 10,
        .completion_tokens = 3,
        .finish_reason = "stop",
        .prefill_tps = 100.0,
        .decode_tps = 35.0,
        .logprobs = null,
    };
    try testing.expectEqual(@as(u32, 10), result.prompt_tokens);
    try testing.expectEqual(@as(u32, 3), result.completion_tokens);
    try testing.expectEqualStrings("stop", result.finish_reason);
    try testing.expect(result.logprobs == null);
}

test "tokensPerSec basic and zero-time" {
    // 100 tokens in 1 second = 100 tok/s.
    try testing.expectApproxEqAbs(@as(f64, 100.0), tokensPerSec(100, std.time.ns_per_s), 1e-6);
    // 50 tokens in 0.5s = 100 tok/s.
    try testing.expectApproxEqAbs(@as(f64, 100.0), tokensPerSec(50, std.time.ns_per_s / 2), 1e-6);
    // Zero elapsed → 0, never inf/NaN.
    try testing.expectEqual(@as(f64, 0.0), tokensPerSec(100, 0));
}

const PrefillMem = struct { active: usize, cache: usize, peak: usize };

fn mlxPrefillMem() PrefillMem {
    var active: usize = 0;
    var cache: usize = 0;
    var peak: usize = 0;
    _ = mlx.mlx_get_active_memory(&active);
    _ = mlx.mlx_get_cache_memory(&cache);
    _ = mlx.mlx_get_peak_memory(&peak);
    return .{ .active = active, .cache = cache, .peak = peak };
}

fn formatPrefillMemLine(buf: []u8, pos: usize, end: usize, eval_ms: u64, mem: PrefillMem) []const u8 {
    return std.fmt.bufPrint(buf, "  [prefill-trace] mem pos={d} end={d} eval_ms={d} active_bytes={d} cache_bytes={d} peak_bytes={d}\n", .{
        pos,
        end,
        eval_ms,
        mem.active,
        mem.cache,
        mem.peak,
    }) catch "";
}

test "prefill-trace mem line names active_bytes and cache_bytes" {
    var buf: [256]u8 = undefined;
    const line = formatPrefillMemLine(&buf, 8192, 16384, 12, .{ .active = 11, .cache = 22, .peak = 33 });
    try std.testing.expect(std.mem.indexOf(u8, line, "active_bytes=11") != null);
    try std.testing.expect(std.mem.indexOf(u8, line, "cache_bytes=22") != null);
    try std.testing.expect(std.mem.indexOf(u8, line, "peak_bytes=33") != null);
    try std.testing.expect(std.mem.indexOf(u8, line, "eval_ms=12") != null);
}

test "prefillTokensPerSec divides by uncached tokens" {
    // Cold prefill: 754 tokens, none cached, 2s → 377 tok/s.
    try testing.expectApproxEqAbs(
        @as(f64, 377.0),
        prefillTokensPerSec(754, 0, 2 * std.time.ns_per_s),
        1e-6,
    );
    // Warm prefill: 754-token prompt, 700 cached, only 54 ran. A fast 54-token
    // suffix in 0.1s is 540 tok/s — NOT 7540 (the inflated full-prompt rate).
    try testing.expectApproxEqAbs(
        @as(f64, 540.0),
        prefillTokensPerSec(754, 700, std.time.ns_per_s / 10),
        1e-6,
    );
    // Full cache hit: 0 uncached → 0 tok/s (no compute happened).
    try testing.expectEqual(@as(f64, 0.0), prefillTokensPerSec(754, 754, std.time.ns_per_s));
    // Defensive: cached > prompt (shouldn't happen) clamps to 0, no underflow.
    try testing.expectEqual(@as(f64, 0.0), prefillTokensPerSec(10, 20, std.time.ns_per_s));
}

test "maskForLogitVocab pads and truncates to logits size" {
    const short_mask = [_]bool{ true, false, true };
    const padded = try maskForLogitVocab(testing.allocator, &short_mask, 5);
    defer padded.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 5), padded.slice.len);
    try testing.expect(padded.slice[0]);
    try testing.expect(!padded.slice[1]);
    try testing.expect(padded.slice[2]);
    try testing.expect(!padded.slice[3]);
    try testing.expect(!padded.slice[4]);

    const long_mask = [_]bool{ false, true, true, true };
    const truncated = try maskForLogitVocab(testing.allocator, &long_mask, 2);
    defer truncated.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 2), truncated.slice.len);
    try testing.expect(!truncated.slice[0]);
    try testing.expect(truncated.slice[1]);
}

test "argmax selects highest value" {
    // Create a simple logits array [1, 5] with values [0.1, 0.5, 0.9, 0.2, 0.3]
    const data = [_]f32{ 0.1, 0.5, 0.9, 0.2, 0.3 };
    const shape = [_]c_int{ 1, 5 };
    const s = mlx.gpuStream();
    const arr = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(arr);

    const result = try argmax(arr, s);
    try testing.expectEqual(@as(u32, 2), result); // index 2 has value 0.9
}

test "argmax with negative values" {
    const data = [_]f32{ -5.0, -1.0, -3.0, -0.5, -2.0 };
    const shape = [_]c_int{ 1, 5 };
    const s = mlx.gpuStream();
    const arr = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(arr);

    const result = try argmax(arr, s);
    try testing.expectEqual(@as(u32, 3), result); // index 3 has value -0.5 (highest)
}

test "applyRepeatPenalty reduces seen token logits" {
    const s = mlx.gpuStream();
    // logits: [1.0, 2.0, 3.0, 4.0, 5.0]
    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const shape = [_]c_int{ 1, 5 };
    const logits = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);

    // Penalize tokens at indices 1 and 3
    const generated = [_]u32{ 1, 3 };
    var res = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(res);

    try applyRepeatPenalty(&res, logits, &generated, 2.0, 0.0, s);
    try mlx.check(mlx.mlx_array_eval(res));

    const res_data = mlx.mlx_array_data_float32(res).?;
    // Index 0: untouched → 1.0
    try testing.expectApproxEqAbs(@as(f32, 1.0), res_data[0], 0.01);
    // Index 1: positive, divided by 2.0 → 1.0
    try testing.expectApproxEqAbs(@as(f32, 1.0), res_data[1], 0.01);
    // Index 2: untouched → 3.0
    try testing.expectApproxEqAbs(@as(f32, 3.0), res_data[2], 0.01);
    // Index 3: positive, divided by 2.0 → 2.0
    try testing.expectApproxEqAbs(@as(f32, 2.0), res_data[3], 0.01);
    // Index 4: untouched → 5.0
    try testing.expectApproxEqAbs(@as(f32, 5.0), res_data[4], 0.01);
}

test "applyRepeatPenalty with negative logits" {
    const s = mlx.gpuStream();
    // Mix of positive and negative logits
    const data = [_]f32{ -2.0, 3.0, -1.0, 4.0 };
    const shape = [_]c_int{ 1, 4 };
    const logits = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);

    // Penalize all tokens
    const generated = [_]u32{ 0, 1, 2, 3 };
    var res = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(res);

    try applyRepeatPenalty(&res, logits, &generated, 2.0, 0.0, s);
    try mlx.check(mlx.mlx_array_eval(res));

    const res_data = mlx.mlx_array_data_float32(res).?;
    // Index 0: negative, multiplied by 2.0 → -4.0
    try testing.expectApproxEqAbs(@as(f32, -4.0), res_data[0], 0.01);
    // Index 1: positive, divided by 2.0 → 1.5
    try testing.expectApproxEqAbs(@as(f32, 1.5), res_data[1], 0.01);
    // Index 2: negative, multiplied by 2.0 → -2.0
    try testing.expectApproxEqAbs(@as(f32, -2.0), res_data[2], 0.01);
    // Index 3: positive, divided by 2.0 → 2.0
    try testing.expectApproxEqAbs(@as(f32, 2.0), res_data[3], 0.01);
}

test "applyRepeatPenalty presence penalty" {
    const s = mlx.gpuStream();
    const data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const shape = [_]c_int{ 1, 4 };
    const logits = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);

    const generated = [_]u32{ 0, 2 };
    var res = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(res);

    try applyRepeatPenalty(&res, logits, &generated, 1.0, 0.5, s);
    try mlx.check(mlx.mlx_array_eval(res));

    const res_data = mlx.mlx_array_data_float32(res).?;
    // Index 0: seen, presence penalty subtracted → 1.0 - 0.5 = 0.5
    try testing.expectApproxEqAbs(@as(f32, 0.5), res_data[0], 0.01);
    // Index 1: unseen → 2.0
    try testing.expectApproxEqAbs(@as(f32, 2.0), res_data[1], 0.01);
    // Index 2: seen → 3.0 - 0.5 = 2.5
    try testing.expectApproxEqAbs(@as(f32, 2.5), res_data[2], 0.01);
    // Index 3: unseen → 4.0
    try testing.expectApproxEqAbs(@as(f32, 4.0), res_data[3], 0.01);
}

test "applyRepeatPenalty combined penalties" {
    const s = mlx.gpuStream();
    const data = [_]f32{ 2.0, -1.0, 3.0 };
    const shape = [_]c_int{ 1, 3 };
    const logits = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);

    const generated = [_]u32{ 0, 1 };
    var res = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(res);

    try applyRepeatPenalty(&res, logits, &generated, 2.0, 1.0, s);
    try mlx.check(mlx.mlx_array_eval(res));

    const res_data = mlx.mlx_array_data_float32(res).?;
    // Index 0: positive, divide by 2.0 = 1.0, then - 1.0 = 0.0
    try testing.expectApproxEqAbs(@as(f32, 0.0), res_data[0], 0.01);
    // Index 1: negative, multiply by 2.0 = -2.0, then - 1.0 = -3.0
    try testing.expectApproxEqAbs(@as(f32, -3.0), res_data[1], 0.01);
    // Index 2: unseen → 3.0
    try testing.expectApproxEqAbs(@as(f32, 3.0), res_data[2], 0.01);
}

test "sampleToken greedy selects argmax" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();

    // Create logits [1, 1, 5] — 5 vocab entries, token at index 3 has highest
    const data = [_]f32{ 1.0, 0.5, 0.1, 5.0, 0.2 };
    const logits_shape = [_]c_int{ 1, 1, 5 };
    const logits = mlx.mlx_array_new_data(&data, &logits_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(logits);

    const params = SamplingParams{ .temperature = 0.0 };
    const result = try sampleToken(allocator, logits, params, null, 0, null, s);
    try testing.expectEqual(@as(u32, 3), result.token_id);
}

test "sampleToken with temperature produces valid token" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();

    const data = [_]f32{ 1.0, 2.0, 3.0 };
    const logits_shape = [_]c_int{ 1, 1, 3 };
    const logits = mlx.mlx_array_new_data(&data, &logits_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(logits);

    const params = SamplingParams{ .temperature = 0.5 };
    const result = try sampleToken(allocator, logits, params, null, 0, null, s);
    // Token should be in valid range
    try testing.expect(result.token_id < 3);
}

test "seeded lazy sampling replays the same draws and advances per draw" {
    const s = mlx.gpuStream();
    const data = [_]f32{ 0.0, 0.0 };
    const shape = [_]c_int{ 1, 1, 2 };
    const logits = mlx.mlx_array_new_data(&data, &shape, 3, .float32);
    defer _ = mlx.mlx_array_free(logits);

    var runs: [2][24]i32 = undefined;
    for (&runs) |*run| {
        var params = SamplingParams{ .temperature = 1.0, .seed = 42 };
        for (run) |*out| {
            const lazy = sampleTokenLazy(logits, params, s);
            defer _ = mlx.mlx_array_free(lazy);
            try mlx.check(mlx.mlx_array_eval(lazy));
            try mlx.check(mlx.mlx_array_item_int32(out, lazy));
            params.draw +%= 1;
        }
    }
    try testing.expectEqualSlices(i32, &runs[0], &runs[1]);
    // The same key every draw replays ONE coin flip 24 times (p = 2^-23).
    var all_same = true;
    for (runs[0][1..]) |v| all_same = all_same and v == runs[0][0];
    try testing.expect(!all_same);
}

test "batched greedy ids equal two solo ids and one sampling eval" {
    const s = mlx.gpuStream();
    const data = [_]f32{
        0.1, 5.0, 0.2, 0.3,
        0.4, 0.5, 0.6, 9.0,
    };
    const row_shape = [_]c_int{ 1, 1, 4 };
    const r0 = mlx.mlx_array_new_data(&data[0], &row_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(r0);
    const r1 = mlx.mlx_array_new_data(&data[4], &row_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(r1);

    const greedy = SamplingParams{ .temperature = 0.0 };
    const solo0 = try evalLazyToken(sampleTokenLazy(r0, greedy, s));
    const solo1 = try evalLazyToken(sampleTokenLazy(r1, greedy, s));

    sample_rows_evals = 0;
    var ids: [2]i32 = undefined;
    var params = [_]SamplingParams{ greedy, greedy };
    const rows = [_]mlx.mlx_array{ r0, r1 };
    try sampleRows(&ids, &rows, &params, s);
    try testing.expectEqual(@as(u64, 1), sample_rows_evals);
    try testing.expectEqual(@as(i32, @intCast(solo0)), ids[0]);
    try testing.expectEqual(@as(i32, @intCast(solo1)), ids[1]);
}

test "batched mixed greedy and sampled rows produce two ids" {
    const s = mlx.gpuStream();
    const data = [_]f32{
        0.1, 5.0, 0.2, 0.3,
        0.4, 0.5, 0.6, 9.0,
    };
    const row_shape = [_]c_int{ 1, 1, 4 };
    const r0 = mlx.mlx_array_new_data(&data[0], &row_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(r0);
    const r1 = mlx.mlx_array_new_data(&data[4], &row_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(r1);

    const greedy = SamplingParams{ .temperature = 0.0 };
    const sampled = SamplingParams{ .temperature = 0.7 };
    const solo_g = try evalLazyToken(sampleTokenLazy(r0, greedy, s));

    sample_rows_evals = 0;
    var ids: [2]i32 = undefined;
    var params = [_]SamplingParams{ greedy, sampled };
    const rows = [_]mlx.mlx_array{ r0, r1 };
    try sampleRows(&ids, &rows, &params, s);
    try testing.expectEqual(@as(u64, 1), sample_rows_evals);
    try testing.expectEqual(@as(i32, @intCast(solo_g)), ids[0]);
    try testing.expect(ids[1] >= 0 and ids[1] < 4);
}

test "batched stacked seedless top_k top_p rows equal solo draws" {
    const s = mlx.gpuStream();
    const data = [_]f32{
        0.1, 5.0, 0.2, 0.3,
        0.4, 0.5, 0.6, 9.0,
    };
    const row_shape = [_]c_int{ 1, 1, 4 };
    const r0 = mlx.mlx_array_new_data(&data[0], &row_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(r0);
    const r1 = mlx.mlx_array_new_data(&data[4], &row_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(r1);

    const p = SamplingParams{ .temperature = 0.7, .top_k = 1, .top_p = 0.9 };
    const solo0 = try evalLazyToken(sampleTokenLazy(r0, p, s));
    const solo1 = try evalLazyToken(sampleTokenLazy(r1, p, s));

    sample_rows_evals = 0;
    var ids: [2]i32 = undefined;
    var params = [_]SamplingParams{ p, p };
    const rows = [_]mlx.mlx_array{ r0, r1 };
    try sampleRows(&ids, &rows, &params, s);
    try testing.expectEqual(@as(u64, 1), sample_rows_evals);
    try testing.expectEqual(@as(i32, @intCast(solo0)), ids[0]);
    try testing.expectEqual(@as(i32, @intCast(solo1)), ids[1]);
}

test "top-k and top-p break exact ties by rank, not by value" {
    // bf16 logits tie at the top constantly on a 250k vocab; a value cutoff
    // keeps every tied token and top_k 1 / top_p -> 0 stop being greedy.
    const s = mlx.gpuStream();
    const data = [_]f32{
        2.0, 5.0, 5.0, 5.0, 1.0,
        0.1, 0.2, 9.0, 0.3, 0.4,
    };
    const shape = [_]c_int{ 2, 5 };
    const logits = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);
    for ([_]bool{ true, false }) |use_k| {
        var masked = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(masked);
        if (use_k) try applyTopK(&masked, logits, 1, s) else try applyTopP(&masked, logits, 1e-6, 0, s);
        try mlx.check(mlx.mlx_array_eval(masked));
        const out = mlx.mlx_array_data_float32(masked).?[0..10];
        var kept0: usize = 0;
        for (out[0..5]) |v| kept0 += @intFromBool(v != -std.math.inf(f32));
        try testing.expectEqual(@as(usize, 1), kept0);
        for (out[5..10], 0..) |v, i| try testing.expect((v != -std.math.inf(f32)) == (i == 2));
    }
    // A 3-of-5 top-k on the tied row keeps exactly the three tied tokens.
    var masked3 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(masked3);
    try applyTopK(&masked3, logits, 3, s);
    try mlx.check(mlx.mlx_array_eval(masked3));
    const out3 = mlx.mlx_array_data_float32(masked3).?[0..5];
    for (out3, 0..) |v, i| try testing.expect((v != -std.math.inf(f32)) == (i >= 1 and i <= 3));
}

test "batched stacked seedless rows honour a suppress mask" {
    const s = mlx.gpuStream();
    const mask = try buildSuppressMask(&[_]u32{1}, 0, 4, s);
    defer _ = mlx.mlx_array_free(mask);
    const data = [_]f32{
        0.1, 5.0, 0.2, 0.3,
        0.4, 9.0, 0.5, 0.6,
    };
    const row_shape = [_]c_int{ 1, 1, 4 };
    const r0 = mlx.mlx_array_new_data(&data[0], &row_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(r0);
    const r1 = mlx.mlx_array_new_data(&data[4], &row_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(r1);

    const p = SamplingParams{ .temperature = 0.7, .top_k = 1, .top_p = 0.9, .suppress_mask = mask };
    const solo0 = try evalLazyToken(sampleTokenLazy(r0, p, s));
    const solo1 = try evalLazyToken(sampleTokenLazy(r1, p, s));

    sample_rows_evals = 0;
    var ids: [2]i32 = undefined;
    var params = [_]SamplingParams{ p, p };
    const rows = [_]mlx.mlx_array{ r0, r1 };
    try sampleRows(&ids, &rows, &params, s);
    try testing.expectEqual(@as(u64, 1), sample_rows_evals);
    try testing.expectEqual(@as(i32, @intCast(solo0)), ids[0]);
    try testing.expectEqual(@as(i32, @intCast(solo1)), ids[1]);
    try testing.expect(ids[0] != 1);
    try testing.expect(ids[1] != 1);
}

test "batched seeded draws keep per-row keys" {
    const s = mlx.gpuStream();
    const data = [_]f32{
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    };
    const row_shape = [_]c_int{ 1, 1, 4 };
    const r0 = mlx.mlx_array_new_data(&data[0], &row_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(r0);
    const r1 = mlx.mlx_array_new_data(&data[4], &row_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(r1);

    const p0 = SamplingParams{ .temperature = 1.0, .seed = 7, .draw = 2 };
    const p1 = SamplingParams{ .temperature = 1.0, .seed = 99, .draw = 5 };
    const solo0 = try evalLazyToken(sampleTokenLazy(r0, p0, s));
    const solo1 = try evalLazyToken(sampleTokenLazy(r1, p1, s));

    sample_rows_evals = 0;
    var ids: [2]i32 = undefined;
    var params = [_]SamplingParams{ p0, p1 };
    const rows = [_]mlx.mlx_array{ r0, r1 };
    try sampleRows(&ids, &rows, &params, s);
    try testing.expectEqual(@as(u64, 1), sample_rows_evals);
    try testing.expectEqual(@as(i32, @intCast(solo0)), ids[0]);
    try testing.expectEqual(@as(i32, @intCast(solo1)), ids[1]);
    try testing.expectEqual(p0.draw +% 1, params[0].draw);
    try testing.expectEqual(p1.draw +% 1, params[1].draw);
}

test "sampleToken from prefill logits (seq_len > 1)" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();

    // [1, 3, 4] — 3 positions, 4 vocab, should take last position
    const data = [_]f32{
        0.1, 0.2, 0.3, 0.4, // pos 0
        0.5, 0.6, 0.7, 0.8, // pos 1
        9.0, 0.1, 0.1, 0.1, // pos 2 — token 0 is clearly highest
    };
    const logits_shape = [_]c_int{ 1, 3, 4 };
    const logits = mlx.mlx_array_new_data(&data, &logits_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(logits);

    const params = SamplingParams{ .temperature = 0.0 };
    const result = try sampleToken(allocator, logits, params, null, 0, null, s);
    try testing.expectEqual(@as(u32, 0), result.token_id); // pos 2, index 0 = 9.0
}

test "Generator.dflashGateShouldDisable: a serial price decides by ms per token, the ladder break-even decides without one" {
    const G = Generator;
    const w = G.DFLASH_GATE_WARMUP;
    const round_ms: f64 = 30.6; // a 4-row round, Sushi-2.6bpw M5 Max
    // Never decided before the warmup rounds.
    try testing.expect(!G.dflashGateShouldDisable(w - 1, 0, round_ms * @as(f64, @floatFromInt(w - 1)), 4.0, 13.3));
    // Against a 13.3 ms serial token: one accepted draft per round costs 15.3 ms per token and loses,
    // one and a half costs 12.2 and wins.
    try testing.expect(G.dflashGateShouldDisable(w, w, round_ms * @as(f64, @floatFromInt(w)), 4.0, 13.3));
    try testing.expect(!G.dflashGateShouldDisable(w, w + w / 2, round_ms * @as(f64, @floatFromInt(w)), 4.0, 13.3));
    // A slower serial token (a larger pack) keeps rounds the 13.3 ms one would drop.
    try testing.expect(!G.dflashGateShouldDisable(w, w, round_ms * @as(f64, @floatFromInt(w)), 4.0, 20.0));
    // No serial sample: the ladder's break-even yield, 1.37 accepted drafts at 4 rows.
    try testing.expectApproxEqAbs(@as(f32, 1.37), G.dflashBreakEvenAccepted(4.0), 0.001);
    try testing.expect(G.dflashGateShouldDisable(w, w + w / 5, 0, 4.0, null));
    try testing.expect(!G.dflashGateShouldDisable(w, w + w / 2, 0, 4.0, null));
    // A round never offers fewer accepted drafts than its break-even asks for, up to eight rows.
    for ([_]f32{ 3, 4, 5, 6, 8 }) |rows| try testing.expect(G.dflashBreakEvenAccepted(rows) < rows - 1);
    // Rows the policy left out of a round are not a cost: 2.5 accepted per round at five rows on average
    // clears the bar the eight-row block would set at 2.49.
    const rounds = 3 * w;
    const acc = rounds * 5 / 2;
    try testing.expect(!G.dflashGateShouldDisable(rounds, acc, 0, 5.0, null));
    try testing.expect(G.dflashGateShouldDisable(rounds, acc - rounds / 20, 0, 8.0, null));
}

test "Generator.dflashSerialPrice reads the width-0 cell the DFlash fallback feeds, then the serial row" {
    var t = round_cost.Table{ .layout = .long };
    try testing.expect(Generator.dflashSerialPrice(&t, 0) == null);
    for (0..round_cost.MIN_SAMPLES) |_| _ = t.observe(0, 1000, 13.4, 1, true, false);
    try testing.expectApproxEqAbs(@as(f32, 13.4), Generator.dflashSerialPrice(&t, 0).?, 1e-3);
    var row = round_cost.Table{ .layout = .long };
    for (0..round_cost.MIN_SAMPLES) |_| _ = row.observeSerial(1000, 15.0, true, false);
    try testing.expectApproxEqAbs(@as(f32, 15.0), Generator.dflashSerialPrice(&row, 0).?, 1e-3);
}

test "InitOptions.lookup_prompt overrides prompt_ids_owned source" {
    // When the server's cache-reuse path forwards only a trailing-tail
    // prompt slice but supplies the full original prompt via
    // `InitOptions.lookup_prompt`, the lookup buffer must be cloned from
    // the full slice — not the truncated tail.
    const tail = [_]u32{99};
    const full = [_]u32{ 10, 20, 30, 99 };
    const src = Generator.pickLookupPromptSource(&tail, &full);
    try testing.expectEqual(@as(usize, 4), src.len);
    try testing.expectEqualSlices(u32, &full, src);
}

test "InitOptions.lookup_prompt = null preserves existing behavior" {
    // Back-compat path: when callers don't set `lookup_prompt`, the source
    // is the unmodified `prompt_ids` slice — same buffer the Generator
    // received pre-fix.
    const prompt = [_]u32{ 1, 2, 3, 4, 5 };
    const src = Generator.pickLookupPromptSource(&prompt, null);
    try testing.expectEqual(prompt.len, src.len);
    try testing.expectEqualSlices(u32, &prompt, src);
    try testing.expectEqual(@as([*]const u32, prompt[0..].ptr), src.ptr);
}

test "StallClock: progress resets the deadline, silence expires it, 0 disables" {
    var clock = StallClock{};
    const s = std.time.ns_per_s;
    // Producing tokens keeps resetting the deadline — a healthy generation
    // can run arbitrarily long (the live bug: a 33KB tool call at 30 tok/s
    // takes >300s and was guillotined mid-call by the wall-clock timeout).
    try std.testing.expect(!clock.expired(0 * s, 0, 300 * s));
    try std.testing.expect(!clock.expired(299 * s, 1000, 300 * s)); // progress at 299s
    try std.testing.expect(!clock.expired(598 * s, 2000, 300 * s)); // progress again
    // No new tokens for the full window -> stalled.
    try std.testing.expect(!clock.expired(700 * s, 2000, 300 * s));
    try std.testing.expect(clock.expired(898 * s, 2000, 300 * s));
    // 0 = disabled, even after silence.
    var off = StallClock{};
    try std.testing.expect(!off.expired(0, 0, 0));
    try std.testing.expect(!off.expired(10_000 * s, 0, 0));
}

test "isDegenerateTailLoop catches a repeated channel-opener cycle" {
    const P = degenerate_loop_max_period;
    const R = degenerate_loop_reps;

    // Gemma 4 12B failure mode: the model spams the thinking opener
    // `<|channel>thought\n` — model that as a 3-token cycle. After enough
    // identical repetitions the tail is a pure period-3 loop → fire.
    {
        var ids = std.ArrayList(u32).empty;
        defer ids.deinit(testing.allocator);
        try ids.appendSlice(testing.allocator, &[_]u32{ 7, 8, 9 }); // some real prefix
        var k: usize = 0;
        while (k < degenerate_loop_min_span / 3 + 1) : (k += 1) {
            try ids.appendSlice(testing.allocator, &[_]u32{ 101, 102, 103 }); // <|channel>,thought,\n
        }
        try testing.expect(isDegenerateTailLoop(ids.items, P, R));
    }

    // A single token stuck on repeat (period 1) counts once it fills the span.
    {
        var ids = std.ArrayList(u32).empty;
        defer ids.deinit(testing.allocator);
        var k: usize = 0;
        while (k < degenerate_loop_min_span + 1) : (k += 1) try ids.append(testing.allocator, 42);
        try testing.expect(isDegenerateTailLoop(ids.items, P, R));
    }
}

/// Ids a CallForce commits after `published`, each published in turn, until it samples.
fn drainCallForce(cf: *CallForce, published: []const u32) ![]u32 {
    var gen = std.ArrayList(u32).empty;
    defer gen.deinit(std.testing.allocator);
    try gen.appendSlice(std.testing.allocator, published);
    while (true) switch (cf.decide(gen.items)) {
        .force => |id| try gen.append(std.testing.allocator, id),
        .sample => break,
        .choose => return error.UnexpectedChoice,
    };
    return std.testing.allocator.dupe(u32, gen.items[published.len..]);
}

test "CallForce: the call follows the closed thought, every other position samples" {
    const OPEN: u32 = 900;
    const CLOSE: u32 = 901;
    const NL2: u32 = 271;
    const forced = [_]u32{ NL2, 10, 11, 12 }; // separator, then the call
    const call = forced[1..];
    const t = std.testing;

    // A prompt-opened thought runs free; the closer brings the separator and the call.
    {
        var cf = CallForce{ .forced = &forced, .call_at = 1, .closer_id = CLOSE, .phase = .thinking, .last = 5 };
        try t.expect(cf.pending());
        try t.expect(cf.decide(&.{ 1, 2 }) == .sample);
        const got = try drainCallForce(&cf, &.{ 1, 2, CLOSE });
        defer t.allocator.free(got);
        try t.expectEqualSlices(u32, &forced, got);
        try t.expect(!cf.pending());
    }
    // A thought the prompt closed on the closer itself: from token 0, separator first.
    {
        var cf = CallForce{ .forced = &forced, .call_at = 1, .closer_id = CLOSE, .phase = .answer, .last = CLOSE };
        const got = try drainCallForce(&cf, &.{});
        defer t.allocator.free(got);
        try t.expectEqualSlices(u32, &forced, got);
    }
    // The prompt already wrote the separator: only the call.
    {
        var cf = CallForce{ .forced = &forced, .call_at = 1, .closer_id = CLOSE, .phase = .answer, .last = NL2 };
        const got = try drainCallForce(&cf, &.{});
        defer t.allocator.free(got);
        try t.expectEqualSlices(u32, call, got);
    }
    // A reasoning-budget commit already wrote the separator after the closer.
    {
        var cf = CallForce{ .forced = &forced, .call_at = 1, .closer_id = CLOSE, .phase = .thinking, .last = 5 };
        const got = try drainCallForce(&cf, &.{ 1, CLOSE, NL2 });
        defer t.allocator.free(got);
        try t.expectEqualSlices(u32, call, got);
    }
    // The template leaves the opener to the model: it is committed at token 0,
    // the thought then runs free until its closer.
    {
        var cf = CallForce{ .forced = &forced, .call_at = 1, .closer_id = CLOSE, .opener = .{OPEN}, .phase = .open, .last = 5 };
        const first = try drainCallForce(&cf, &.{});
        defer t.allocator.free(first);
        try t.expectEqualSlices(u32, &.{OPEN}, first);
        try t.expect(cf.decide(&.{ OPEN, 1, 2 }) == .sample);
        const got = try drainCallForce(&cf, &.{ OPEN, 1, 2, CLOSE });
        defer t.allocator.free(got);
        try t.expectEqualSlices(u32, &forced, got);
    }
}

test "CallForce: a required call names one of the declared functions, the model picking among them" {
    const CLOSE: u32 = 901;
    const t = std.testing;
    const head = [_]u32{ 10, 11 }; // the call up to the name
    // Each declared name as ids, its closing delimiter included.
    const get_time = [_]u32{ 20, 21, 30 };
    const get_weather = [_]u32{ 20, 22, 30 };
    const list_files = [_]u32{ 23, 30 };
    const names = [_][]const u32{ &get_time, &get_weather, &list_files };
    var choices: [names.len]u32 = undefined;
    var cf = CallForce{ .forced = &head, .call_at = 0, .closer_id = CLOSE, .phase = .answer, .last = CLOSE, .names = &names, .choice_buf = &choices };

    var gen = std.ArrayList(u32).empty;
    defer gen.deinit(t.allocator);
    for (head) |want| {
        try t.expectEqual(want, cf.decide(gen.items).force);
        try gen.append(t.allocator, want);
    }
    try t.expectEqualSlices(u32, &.{ 20, 23 }, cf.decide(gen.items).choose);
    try gen.append(t.allocator, 20); // the model's pick
    try t.expectEqualSlices(u32, &.{ 21, 22 }, cf.decide(gen.items).choose);
    try gen.append(t.allocator, 22);
    try t.expect(cf.pending());
    try t.expectEqual(@as(u32, 30), cf.decide(gen.items).force);
    try gen.append(t.allocator, 30);
    try t.expect(cf.decide(gen.items) == .sample);
    try t.expect(!cf.pending());
}

test "ThinkBound: counts only tokens inside the think block and fires at the budget" {
    const OPEN: u32 = 10;
    const CLOSE: u32 = 11;
    const forced = [_]u32{ 30, 31, CLOSE, 32 };
    var tb = ThinkBound{ .budget = 3, .opener_id = OPEN, .closer_id = CLOSE, .forced = &forced, .in_think = false };

    // Content before any opener never counts.
    tb.observe(&[_]u32{ 1, 2, 3, 4, 5 });
    try testing.expect(!tb.due());

    // Opened by the model: the budget counts from the opener.
    tb.observe(&[_]u32{ 1, 2, 3, 4, 5, OPEN, 6, 7 });
    try testing.expect(!tb.due());
    tb.observe(&[_]u32{ 1, 2, 3, 4, 5, OPEN, 6, 7, 8 });
    try testing.expect(tb.due());

    // A model that closed on its own is never forced.
    var closed = ThinkBound{ .budget = 3, .opener_id = OPEN, .closer_id = CLOSE, .forced = &forced, .in_think = true };
    closed.observe(&[_]u32{ 6, 7, CLOSE, 8, 9, 10, 11 });
    try testing.expect(!closed.due());

    // Prompt-opened: the count starts at token 0 and the forced closer ends it.
    var po = ThinkBound{ .budget = 2, .opener_id = null, .closer_id = CLOSE, .forced = &forced, .in_think = true };
    po.observe(&[_]u32{ 6, 7 });
    try testing.expect(po.due());
    po.fired = true;
    po.observe(&[_]u32{ 6, 7, 30, 31, CLOSE, 32, 40 });
    try testing.expect(!po.in_think);
    try testing.expect(!po.due());
}

test "degenerateTail: a short exact cycle convicts only past the minimum span" {
    const al = testing.allocator;
    // Live 2026-09-15 (pi, Qwen3.8 per-digit tokenizer): a 24-wide map wall
    // row "111111111111111111111111" is 24 identical tokens and was cut as a
    // period-1 loop mid-thought. Short cycles are common in honest code (digit
    // rows, zeroed arrays), so the bar is a SPAN of identical cycling, not a
    // rep count that period 1 reaches in a few dozen bytes.
    var ids = std.ArrayList(u32).empty;
    defer ids.deinit(al);
    try ids.appendSlice(al, &[_]u32{ 7, 8, 9, 1 });
    for (0..24) |_| try ids.append(al, 16);
    try testing.expect(degenerateTail(ids.items) == null);

    // A 32-element zeroed row: `0, 0, 0, ...` is a period-2 cycle of 64 tokens.
    var zeros = std.ArrayList(u32).empty;
    defer zeros.deinit(al);
    try zeros.appendSlice(al, &[_]u32{ 7, 8, 9, 1 });
    for (0..32) |_| try zeros.appendSlice(al, &[_]u32{ 15, 11 });
    try testing.expect(degenerateTail(zeros.items) == null);

    // Stuck for real: one token past the span is a loop and trims to one copy.
    for (24..degenerate_loop_min_span) |_| try ids.append(al, 16);
    const d = degenerateTail(ids.items) orelse return error.TestExpectedLoop;
    try testing.expectEqual(DegenerateTail.Tier.exact_cycle, d.tier);
    try testing.expectEqual(@as(usize, 5), d.start);
}

/// Build `n` tokens by cycling through `phrasings`, which share a vocabulary.
fn tVaried(al: std.mem.Allocator, phrasings: []const []const u32, n: usize) !std.ArrayList(u32) {
    var ids = std.ArrayList(u32).empty;
    var i: usize = 0;
    while (ids.items.len < n) : (i += 1) {
        try ids.appendSlice(al, phrasings[i % phrasings.len]);
    }
    return ids;
}

test "isNearRepeatTailLoop catches a VARIED-phrasing restatement loop" {
    // Live 2026-08-04, under pi: the model restated the same
    // intent forever while varying the wording — "I need to break this down."
    // / "I need to break this." / "I need to break this down into pieces." —
    // so no exact cycle exists at ANY period and both exact tiers are blind by
    // construction. What IS invariant is that a long stretch of output recycles
    // a tiny vocabulary and introduces almost no new n-grams.
    const al = testing.allocator;
    const phrasings = [_][]const u32{
        &[_]u32{ 40, 41, 42, 43, 44, 45, 46 }, // I need to break this down .
        &[_]u32{ 40, 41, 42, 43, 44, 46 }, // I need to break this .
        &[_]u32{ 40, 41, 42, 43, 44, 45, 47, 48, 46 }, // ... down into pieces .
        &[_]u32{ 49, 40, 41, 42, 43, 44, 45, 46 }, // So I need to break this down .
        &[_]u32{ 40, 41, 42, 43, 44, 45, 50, 46 }, // ... down first .
        &[_]u32{ 51, 42, 43, 44, 45, 46 }, // Let me break this down .
    };
    var ids = try tVaried(al, &phrasings, near_repeat_window + 64);
    defer ids.deinit(al);
    try testing.expect(isNearRepeatTailLoop(ids.items));

    // Below the window the tier says nothing at all: it is a last-resort net
    // for output that has already run a long way, and a false cut truncates a
    // real answer. Short exact cycles remain tier 1/2's job.
    try testing.expect(!isNearRepeatTailLoop(ids.items[0 .. near_repeat_window - 1]));
}

test "isNearRepeatTailLoop leaves legitimately repetitive output alone" {
    const al = testing.allocator;

    // Healthy prose/code: a repeated scaffold, but every line introduces a new
    // identifier. Recycled STRUCTURE is normal; recycled VOCABULARY is not.
    {
        var ids = std.ArrayList(u32).empty;
        defer ids.deinit(al);
        var line: u32 = 0;
        while (ids.items.len < near_repeat_window + 64) : (line += 1) {
            try ids.appendSlice(al, &[_]u32{ 10, 11, 12 }); // `const x =`
            try ids.append(al, 1000 + line); // a fresh identifier
            try ids.appendSlice(al, &[_]u32{ 13, 14 }); // `;\n`
        }
        try testing.expect(!isNearRepeatTailLoop(ids.items));
    }

    // A numeric table: FEW distinct tokens (digits + separators, and this
    // family pre-tokenizes digits singly) but the 4-grams keep changing. The
    // two ratios have to be read together — either one alone convicts this.
    {
        var ids = std.ArrayList(u32).empty;
        defer ids.deinit(al);
        var seed: u32 = 12345;
        while (ids.items.len < near_repeat_window + 64) {
            try ids.append(al, 200); // '|'
            for (0..4) |_| {
                seed = seed *% 1664525 +% 1013904223;
                try ids.append(al, 100 + (seed >> 16) % 10); // a digit
            }
            try ids.appendSlice(al, &[_]u32{ 200, 201 }); // '|', '\n'
        }
        try testing.expect(!isNearRepeatTailLoop(ids.items));
    }

    // Fully novel output.
    {
        var ids = std.ArrayList(u32).empty;
        defer ids.deinit(al);
        for (0..near_repeat_window + 64) |i| try ids.append(al, @intCast(i));
        try testing.expect(!isNearRepeatTailLoop(ids.items));
    }
}

test "degenerateTail acquits a low-entropy STRUCTURED file that ends inside the span bar" {
    // Live 2026-09-15 under pi: a tool call rewriting five 24x19 tile maps of
    // '0'/'1' on a per-digit tokenizer was cut as a near-repeat loop at window
    // fill and the agent got an empty turn. Six distinct tokens and sixteen
    // possible 4-grams satisfy every content ratio by construction, and the
    // real 27B output (rows nearly all `100000000000000000000001`) is
    // indistinguishable from a loop by content. What such a file does, and a
    // loop never does, is END: the bar is the degenerate SPAN.
    const al = testing.allocator;
    const Shape = enum { bit_grid, lazy_map, hex_dump };
    for ([_]Shape{ .bit_grid, .lazy_map, .hex_dump }) |shape| {
        var ids = std.ArrayList(u32).empty;
        defer ids.deinit(al);
        var seed: u32 = 777;
        var row: usize = 0;
        while (ids.items.len < 2800) : (row += 1) {
            try ids.append(al, 200); // '"'
            for (0..24) |col| {
                seed = seed *% 1664525 +% 1013904223;
                const r = seed >> 16;
                const tok: u32 = switch (shape) {
                    .bit_grid => if (col == 0 or col == 23 or row % 19 == 0) 101 else @as(u32, if (r % 10 < 3) 101 else 100),
                    .lazy_map => if (col == 0 or col == 23 or row % 19 == 0 or (row % 5 == 0 and col != 10)) 101 else 100,
                    .hex_dump => 100 + r % 16,
                };
                try ids.append(al, tok);
            }
            try ids.appendSlice(al, &[_]u32{ 200, 201, 202 }); // '"', ',', '\n'
        }
        try testing.expect(degenerateTail(ids.items) == null);
    }
}

test "isDegenerateTailLoop does not fire on healthy or briefly-repeating output" {
    const P = degenerate_loop_max_period;
    const R = degenerate_loop_reps;

    // Strictly increasing ids — no cycle at all.
    {
        var ids: [200]u32 = undefined;
        for (&ids, 0..) |*v, i| v.* = @intCast(i);
        try testing.expect(!isDegenerateTailLoop(&ids, P, R));
    }
    // A short burst of repetition (well under R reps) must be left alone — a
    // model legitimately writing "ha ha ha" or a few identical list bullets.
    {
        var ids = std.ArrayList(u32).empty;
        defer ids.deinit(testing.allocator);
        try ids.appendSlice(testing.allocator, &[_]u32{ 1, 2, 3, 4, 5 });
        var k: usize = 0;
        while (k < R - 1) : (k += 1) try ids.appendSlice(testing.allocator, &[_]u32{ 50, 51 });
        try testing.expect(!isDegenerateTailLoop(ids.items, P, R));
    }
    // Periodic tail but with a longer period than we scan for → ignored.
    {
        var ids = std.ArrayList(u32).empty;
        defer ids.deinit(testing.allocator);
        var k: usize = 0;
        var base: u32 = 0;
        while (k < R) : (k += 1) {
            // period = P + 3 (> max_period); never a pure short cycle.
            var j: u32 = 0;
            while (j < P + 3) : (j += 1) try ids.append(testing.allocator, base + j);
            base = 0; // same long block repeats, but its period exceeds the scan window
        }
        try testing.expect(!isDegenerateTailLoop(ids.items, P, R));
    }
    // Too few tokens to judge.
    try testing.expect(!isDegenerateTailLoop(&[_]u32{ 1, 1 }, P, R));
}

test "isNearRepeatTailLoop leaves PROCEDURAL code alone — it recycles a vocabulary while still progressing" {
    // Live 2026-08-05: a pi session was asked for an elaborate voxel scene and
    // the tier cut it at 16241 generated tokens, so the user got NO file at
    // all. The output was healthy — dense `fillBox(x,y,z, x,y,z, C.name);`
    // lines — but it is exactly the shape the first two ratios were built to
    // tolerate and cannot: a fixed template plus a small colour palette gives
    // a tiny distinct-token ratio, and the templated call shape keeps the
    // 4-gram ratio low too. Measured on the real artifact: 0.068 / 0.351
    // against bars of 0.12 / 0.35 — it cleared conviction by 0.001, and the
    // generation's own tail did not.
    //
    // What separates it from a loop is PROGRESS: every line carries new
    // coordinates, so the window's second half keeps introducing n-grams the
    // first half never had (0.298-0.632 measured, against 0.019-0.022 for the
    // restatement loops this tier exists for).
    // Shape taken from the measured artifact, not invented: a CONTIGUOUS run
    // of template tokens (`\n  fillBox(`, `, C.`, `);`) followed by the
    // varying coordinates. The contiguity is what makes it convict — most
    // 4-gram windows sit entirely inside the fixed run and repeat every line.
    // This fixture scores 0.033 / 0.316 against bars of 0.12 / 0.35.
    const al = testing.allocator;
    var ids = std.ArrayList(u32).empty;
    defer ids.deinit(al);
    var rng: u32 = 99;
    while (ids.items.len < near_repeat_window * 2) {
        var f: u32 = 0;
        while (f < 14) : (f += 1) try ids.append(al, 500 + f);
        var v: usize = 0;
        while (v < 4) : (v += 1) {
            rng = rng *% 1664525 +% 1013904223;
            try ids.append(al, 600 + (rng >> 16) % 19); // a fresh coordinate
            try ids.append(al, 499); // separator
        }
    }
    try testing.expect(!isNearRepeatTailLoop(ids.items));
    try testing.expect(degenerateTail(ids.items) == null);
}

test "degenerateTail: the exact tier reports its tier and keeps ONE cycle" {
    const al = testing.allocator;
    // Identical 3-token cycles past the span bar after a real prefix. The cut
    // is a truncation, so what is emitted should still SHOW what the model got
    // stuck on — one copy of the cycle survives, the rest do not.
    var ids = std.ArrayList(u32).empty;
    defer ids.deinit(al);
    try ids.appendSlice(al, &[_]u32{ 7, 8, 9, 10 });
    var k: usize = 0;
    while (k < degenerate_loop_min_span / 3 + 1) : (k += 1) try ids.appendSlice(al, &[_]u32{ 101, 102, 103 });

    const d = degenerateTail(ids.items) orelse return error.TestExpectedLoop;
    try testing.expectEqual(DegenerateTail.Tier.exact_cycle, d.tier);
    // 4 prefix + 1 kept cycle = 7 tokens survive.
    try testing.expectEqual(@as(usize, 7), d.start);
    // What survives is the honest prefix plus exactly one cycle.
    try testing.expectEqualSlices(u32, &[_]u32{ 7, 8, 9, 10, 101, 102, 103 }, ids.items[0..d.start]);
}

test "degenerateTail: the trim start walks back PAST the near-repeat window" {
    const al = testing.allocator;
    // The near-repeat tier judges the last 1024 tokens, but a restatement
    // loop that has been running for 3000 tokens is degenerate for all 3000.
    // Trimming only the window would hand the client the other ~2000 back,
    // which is the whole failure this exists to stop.
    const phrasings = [_][]const u32{
        &[_]u32{ 1, 2, 3, 4, 5 },
        &[_]u32{ 1, 2, 3, 5, 4 },
        &[_]u32{ 1, 2, 4, 3, 5 },
    };
    var honest = std.ArrayList(u32).empty;
    defer honest.deinit(al);
    var i: u32 = 0;
    while (i < 900) : (i += 1) try honest.append(al, 1000 + i); // all distinct = healthy

    // Pick phrasings pseudo-randomly: a deterministic rotation would be an
    // exact cycle and the long-period tier would convict it first, which is
    // not the tier under test.
    var loop = std.ArrayList(u32).empty;
    defer loop.deinit(al);
    var rng: u32 = 12345;
    while (loop.items.len < near_repeat_min_span + 1000) {
        rng = rng *% 1664525 +% 1013904223;
        try loop.appendSlice(al, phrasings[(rng >> 16) % phrasings.len]);
    }

    var ids = std.ArrayList(u32).empty;
    defer ids.deinit(al);
    try ids.appendSlice(al, honest.items);
    try ids.appendSlice(al, loop.items);

    const d = degenerateTail(ids.items) orelse return error.TestExpectedLoop;
    try testing.expectEqual(DegenerateTail.Tier.near_repeat, d.tier);
    // Well past the single window, and never into the healthy prefix.
    try testing.expect(d.start < ids.items.len - near_repeat_window);
    try testing.expect(d.start >= honest.items.len - near_repeat_step);
}

test "degenerateTail: healthy output is never convicted, so nothing is trimmed" {
    var ids: [4000]u32 = undefined;
    for (&ids, 0..) |*v, i| v.* = @intCast(i);
    try testing.expect(degenerateTail(&ids) == null);
}

test "degenerateTail: the long-period tier keeps one copy of its sentence cycle" {
    const al = testing.allocator;
    var ids = std.ArrayList(u32).empty;
    defer ids.deinit(al);
    try ids.appendSlice(al, &[_]u32{ 1, 2 });
    var cycle: [40]u32 = undefined;
    for (&cycle, 0..) |*v, i| v.* = @intCast(500 + i);
    var k: usize = 0;
    while (k < degenerate_loop_long_min_span / cycle.len + 1) : (k += 1) try ids.appendSlice(al, &cycle);

    const d = degenerateTail(ids.items) orelse return error.TestExpectedLoop;
    try testing.expectEqual(DegenerateTail.Tier.long_cycle, d.tier);
    try testing.expectEqual(@as(usize, 2 + cycle.len), d.start);
}

/// The checkpoints a GLM prefill takes, replayed from the loop's own chunk ends (test oracle for `glmCaptureSchedule`).
pub fn glmCaptureOracle(grid: usize, cap: u32, offset: usize, tail_len: usize, width: usize) GlmCaptureSchedule {
    var r: GlmCaptureSchedule = .{ .backoff = 0, .events = 0, .peak = 0, .newest = 0 };
    if (tail_len <= 1) return r;
    const prefix_len = tail_len - 1;
    r.backoff = glmSnapshotBackoff(ssmSnapshotBackoff(true, prefix_len, offset > 0), offset, prefix_len);
    const loop_end = prefix_len - r.backoff;
    var held: u32 = 0;
    var pos: usize = 0;
    while (pos < loop_end) {
        pos = glmChunkEnd(pos, loop_end, width, grid, offset, true);
        if ((pos + offset) % grid != 0) continue;
        r.events += 1;
        r.newest = pos + offset;
        held += 1;
        r.peak = @max(r.peak, held);
        if (cap > 0 and held > cap) held -= 1;
    }
    if (loop_end > 0 and r.newest != loop_end + offset) {
        r.events += 1;
        r.newest = loop_end + offset;
        held += 1;
        r.peak = @max(r.peak, held);
    }
    return r;
}

/// Walk a chunked prefill as `runPrefill` does, taking the width from `widths` at each boundary (the last entry repeats).
fn walkChunkEnds(
    out: []usize,
    prefix_len: usize,
    widths: []const usize,
    want_ssm_cp: bool,
    stride: usize,
    offset: usize,
) []usize {
    var pos: usize = 0;
    var n: usize = 0;
    while (pos < prefix_len and n < out.len) {
        const w = widths[@min(n, widths.len - 1)];
        const end = nextChunkEnd(pos, prefix_len, w, want_ssm_cp, stride, offset, true);
        out[n] = end;
        n += 1;
        pos = end;
    }
    return out[0..n];
}

test "a width that changes at every boundary never moves a checkpoint" {
    // `ssm_cp_stride` derives from the launch width and every ladder rung divides it, so a
    // mixed-width prefill hits the same checkpoint positions as a fixed-width one.
    const t = testing;
    const mixed = [_]usize{ 4096, 4096, 2048, 1024, 512, 512, 1024, 2048, 4096, 512, 2048, 1024, 4096, 512 };
    const fixed = [_]usize{4096};

    for ([_]usize{ 8192, 4096 }) |stride| {
        for ([_]usize{ 0, 1000 }) |offset| {
            for ([_]usize{ 40960, 40000 }) |prefix_len| {
                var fb: [512]usize = undefined;
                var mb: [512]usize = undefined;
                const f = walkChunkEnds(&fb, prefix_len, &fixed, true, stride, offset);
                const m = walkChunkEnds(&mb, prefix_len, &mixed, true, stride, offset);

                for ([_][]usize{ f, m }) |ends| {
                    var pos: usize = 0;
                    for (ends) |end| {
                        try t.expect(end > pos);
                        try t.expect(end <= prefix_len);
                        // No chunk straddles a stride boundary.
                        const abs_pos = pos + offset;
                        const abs_end = end + offset;
                        const next_b = ((abs_pos / stride) + 1) * stride;
                        if (end != prefix_len) try t.expect(!(next_b > abs_pos and next_b < abs_end));
                        pos = end;
                    }
                    try t.expectEqual(prefix_len, pos);
                }

                var fi: usize = 0;
                var mi: usize = 0;
                while (true) {
                    while (fi < f.len and (f[fi] + offset) % stride != 0) fi += 1;
                    while (mi < m.len and (m[mi] + offset) % stride != 0) mi += 1;
                    if (fi >= f.len or mi >= m.len) break;
                    try t.expectEqual(f[fi], m[mi]);
                    fi += 1;
                    mi += 1;
                }
                while (fi < f.len and (f[fi] + offset) % stride != 0) fi += 1;
                while (mi < m.len and (m[mi] + offset) % stride != 0) mi += 1;
                try t.expectEqual(fi >= f.len, mi >= m.len);
            }
        }
    }
}

test "every ladder rung the adaptive width can take divides the checkpoint stride" {
    const t = testing;
    const stride = effectiveSsmCheckpointStride(256, prefill_chunk_override);
    try t.expect(stride > 0);
    for ([_]usize{ 8192, 4096, 2048, 1024, PREFILL_CHUNK_FLOOR }) |rung| {
        try t.expect(rung <= stride);
        try t.expectEqual(@as(usize, 0), stride % rung);
    }
}

test "commitAdaptiveWidth: the summary follows the running width from both commit sites" {
    const t = testing;
    var cur: usize = 2048;
    var st: AdaptiveWidthState = .{ .width_min = 2048, .width_max = 2048 };
    commitAdaptiveWidth(&cur, &st, 1024);
    try t.expectEqual(@as(usize, 1024), cur);
    try t.expectEqual(@as(u32, 1), st.transitions);
    try t.expectEqual(@as(u32, 1024), st.width_min);
    try t.expectEqual(@as(u32, 2048), st.width_max);
    commitAdaptiveWidth(&cur, &st, 4096);
    try t.expectEqual(@as(usize, 4096), cur);
    try t.expectEqual(@as(u32, 2), st.transitions);
    try t.expectEqual(@as(u32, 1024), st.width_min);
    try t.expectEqual(@as(u32, 4096), st.width_max);
}

test "tailMergeMax: the tail a chunk may absorb scales with the width" {
    const t = testing;
    try t.expectEqual(TAIL_MERGE_MAX, tailMergeMax(8192)); // no-op where it was justified
    try t.expectEqual(TAIL_MERGE_MAX, tailMergeMax(4096));
    try t.expectEqual(@as(usize, 256), tailMergeMax(2048));
    try t.expectEqual(@as(usize, 128), tailMergeMax(1024));
    try t.expectEqual(@as(usize, 64), tailMergeMax(PREFILL_CHUNK_FLOOR));
    try t.expectEqual(@as(usize, 1), tailMergeMax(1));
    for ([_]usize{ 8192, 4096, 2048, 1024, PREFILL_CHUNK_FLOOR }) |w| {
        try t.expect(tailMergeMax(w) * 8 <= w);
    }

    try t.expectEqual(@as(usize, 8492), nextChunkEnd(0, 8492, 8192, false, 0, 0, true));
    try t.expectEqual(@as(usize, 512), nextChunkEnd(0, 812, PREFILL_CHUNK_FLOOR, false, 0, 0, true));
}

test "the scaled tail-merge bound is gated on the per-chunk adaptive width" {
    // The scaled bound is gated: every arch whose resolved chunk is under 4096 would otherwise
    // merge a smaller tail and run an extra chunk.
    const t = testing;

    try t.expectEqual(TAIL_MERGE_MAX, tailMergeMaxFor(PREFILL_CHUNK_FLOOR, false));
    try t.expectEqual(TAIL_MERGE_MAX, tailMergeMaxFor(1024, false));
    try t.expectEqual(TAIL_MERGE_MAX, tailMergeMaxFor(2048, false));
    try t.expectEqual(@as(usize, 64), tailMergeMaxFor(PREFILL_CHUNK_FLOOR, true));
    try t.expectEqual(@as(usize, 256), tailMergeMaxFor(2048, true));
    try t.expectEqual(tailMergeMaxFor(4096, false), tailMergeMaxFor(4096, true));
    try t.expectEqual(tailMergeMaxFor(8192, false), tailMergeMaxFor(8192, true));

    try t.expectEqual(@as(usize, 812), nextChunkEnd(0, 812, PREFILL_CHUNK_FLOOR, false, 0, 0, false));
    try t.expectEqual(@as(usize, 512), nextChunkEnd(0, 812, PREFILL_CHUNK_FLOOR, false, 0, 0, true));
    try t.expectEqual(@as(usize, 1324), nextChunkEnd(0, 1324, 1024, false, 0, 0, false));
    try t.expectEqual(@as(usize, 1024), nextChunkEnd(0, 1324, 1024, false, 0, 0, true));
    try t.expectEqual(@as(usize, 8492), nextChunkEnd(0, 8492, 8192, false, 0, 0, false));
    try t.expectEqual(@as(usize, 8492), nextChunkEnd(0, 8492, 8192, false, 0, 0, true));

    try t.expectEqual(@as(usize, 1), prefillChunkCount(812, PREFILL_CHUNK_FLOOR, false, 0, 0, false));
    try t.expectEqual(@as(usize, 2), prefillChunkCount(812, PREFILL_CHUNK_FLOOR, false, 0, 0, true));
}

test "nextChunkEnd: a tiny trailing remainder merges into the last chunk" {
    // A chat-templated prompt often lands a token or two past the chunk size
    // (8192-target prompts tokenize to 8193). A 1-token trailing chunk pays a
    // FULL graph + eval-barrier + cache-clear for one token — pure overhead.
    // Without checkpoint alignment, remainders under the merge floor extend
    // the current chunk instead.
    try testing.expectEqual(@as(usize, 8193), nextChunkEnd(0, 8193, 8192, false, 0, 0, false));
    // A substantial remainder stays its own chunk.
    try testing.expectEqual(@as(usize, 8192), nextChunkEnd(0, 8192 + 600, 8192, false, 0, 0, false));
    // Mid-prompt chunks are untouched.
    try testing.expectEqual(@as(usize, 8192), nextChunkEnd(0, 16385, 8192, false, 0, 0, false));
    try testing.expectEqual(@as(usize, 16385), nextChunkEnd(8192, 16385, 8192, false, 0, 0, false));
    // With SSM-checkpoint alignment active, a tiny tail STILL merges: the old
    // 1-token trailing chunk existed only to lay a snapshot one token before
    // the always-on end-of-prompt snapshot — pure overhead. A boundary strictly
    // INSIDE the chunk still wins over merging (next case).
    try testing.expectEqual(@as(usize, 8193), nextChunkEnd(0, 8193, 8192, true, 8192, 0, false));
    try testing.expectEqual(@as(usize, 4096), nextChunkEnd(0, 8193, 8192, true, 4096, 0, false));
}

test "prefillChunkCount: SSM-checkpoint stride controls cold-prefill chunking" {
    const PREFILL_CHUNK: usize = 8192;
    // Non-hybrid (or checkpointing off): a sub-PREFILL_CHUNK prompt is ONE chunk.
    try testing.expectEqual(@as(usize, 1), prefillChunkCount(851, PREFILL_CHUNK, false, 0, 0, false));
    try testing.expectEqual(@as(usize, 1), prefillChunkCount(8000, PREFILL_CHUNK, false, 0, 0, false));
    // Tail merge: one token past a chunk boundary is still ONE chunk.
    try testing.expectEqual(@as(usize, 1), prefillChunkCount(8193, PREFILL_CHUNK, false, 0, 0, false));
    try testing.expectEqual(@as(usize, 2), prefillChunkCount(16385, PREFILL_CHUNK, false, 0, 0, false));
    // Mechanically, a raw fine stride still splits an 851-token prefill into 4
    // chunks (851 spans boundaries 256/512/768) — which is why
    // effectiveSsmCheckpointStride coarsens every stride to the prefill chunk:
    // per-chunk costs (expert re-streaming on MoE, sub-dq-gemm-floor GEMMs +
    // fixed overhead everywhere) taxed cold prefill 17-25%.
    try testing.expectEqual(@as(usize, 4), prefillChunkCount(851, PREFILL_CHUNK, true, 256, 0, false));
    // Boundary alignment is ABSOLUTE (warm path passes an offset): a tail-only
    // prefill starting mid-sequence still snaps to global strides. offset=2000,
    // prefix tail of 200 (abs 2000..2200), stride 256 -> boundary 2048/2304? only
    // 2048 falls inside (2000..2200) -> 2 chunks.
    try testing.expectEqual(@as(usize, 2), prefillChunkCount(200, PREFILL_CHUNK, true, 256, 2000, false));
}

test "boundedPrefillChunk: fused head dims and short contexts keep the base chunk" {
    // head_dim <= 128 rides MLX's fused SDPA — no materialized scores, no cap,
    // at ANY context length.
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 128, 16, 1_000_000, true, false, false));
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 64, 32, 1_000_000, true, false, false));
    // hd 256 but short context: 16 heads x 8192 ctx x 8192 chunk x 2B
    // = 2 GiB scores, inside the 4 GiB budget -> full chunk kept. This is the
    // fleet-protection property: every Gemma-4 / Qwen3.5/3.6 checkpoint ships
    // head_dim 256, so typical prompts must keep full prefill throughput.
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 256, 16, 8192, true, false, false));
    // Degenerate inputs never cap.
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 256, 0, 100_000, true, false, false));
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 256, 16, 0, true, false, false));
}

test "boundedPrefillChunk: caps hd-256 long context even with the fused kernel active" {
    // The sushi_attn_p256 kernel removes the SCORE transient, but a big chunk
    // still scales the MoE-gather / KV-concat transients — measured +22 GB
    // peak for +3% speed at a 99K prompt. The cap deliberately ignores
    // prefillHeadDimFused (see the fn doc); pin that with the override ON.
    transformer_mod.fused256_override = true;
    defer transformer_mod.fused256_override = null;
    try testing.expectEqual(@as(usize, 1024), boundedPrefillChunk(8192, 256, 16, 100_000, true, false, false));
}

test "boundedPrefillChunk: long context shrinks to the scores budget, floored and rounded" {
    // gemma-4-26B geometry (16 heads): budget/(16*ctx*2) …
    // ctx 32768 -> exactly 4096.
    try testing.expectEqual(@as(usize, 4096), boundedPrefillChunk(8192, 256, 16, 32768, true, false, false));
    // ctx 100000 -> raw 1342, rounded down to the 512 grain -> 1024.
    try testing.expectEqual(@as(usize, 1024), boundedPrefillChunk(8192, 256, 16, 100_000, true, false, false));
    // ctx 262144 (the PR-#69 255K case) -> 512.
    try testing.expectEqual(@as(usize, 512), boundedPrefillChunk(8192, 256, 16, 262_144, true, false, false));
    // Qwen3.6-27B geometry (24 heads) at 262144: raw 341 -> floor 512.
    try testing.expectEqual(@as(usize, 512), boundedPrefillChunk(8192, 256, 24, 262_144, true, false, false));
    // e4b geometry (8 heads) at 131072: exactly 2048.
    try testing.expectEqual(@as(usize, 2048), boundedPrefillChunk(8192, 256, 8, 131_072, true, false, false));
}

test "boundedPrefillChunk: qk 192 is fused; the kill switch restores its score budget" {
    // A hybrid MLA arch can declare head_dim 128 (its value width) so it fell
    // through the "fused SDPA covers it" early-out — while it scores at qk
    // width 192, the same width mimo_v2's global layers score at. 32 heads.
    //
    // `sushi_attn_pd` serves that width now: no score tensor, so the budget
    // formula measures nothing and the chunk keeps its default at every
    // context. Without this the formula pins the 512 floor past ~256k.
    transformer_mod.fused256_override = true;
    const saved_explicit = prefill_chunk_explicit;
    defer prefill_chunk_explicit = saved_explicit;
    prefill_chunk_explicit = false;
    try testing.expectEqual(@as(usize, 2048), boundedPrefillChunk(8192, 192, 32, 8192, false, true, false));
    try testing.expectEqual(@as(usize, 2048), boundedPrefillChunk(8192, 192, 32, 262_144, false, true, false));
    try testing.expectEqual(@as(usize, 2048), boundedPrefillChunk(8192, 192, 32, 1_048_576, false, true, false));
    // An explicit `--prefill-chunk` raises it as far as the MoE ceiling.
    prefill_chunk_explicit = true;
    try testing.expectEqual(@as(usize, 4096), boundedPrefillChunk(8192, 192, 32, 262_144, false, true, false));
    try testing.expectEqual(@as(usize, 3072), boundedPrefillChunk(3072, 192, 32, 262_144, false, true, false));
    prefill_chunk_explicit = false;
    // Never raises a caller-lowered base.
    try testing.expectEqual(@as(usize, 512), boundedPrefillChunk(512, 192, 32, 1_048_576, false, true, false));
    // A real hd-256 MoE keeps its own measured 4096 branch.
    try testing.expectEqual(@as(usize, 4096), boundedPrefillChunk(8192, 256, 32, 8192, false, true, false));

    // Composed (`fused256_override = false`): scores are back and so is the
    // budget, unchanged — guards and dispatch move on ONE switch.
    transformer_mod.fused256_override = false;
    defer transformer_mod.fused256_override = null;
    // Short prompts keep the full chunk: 32 * 8192 * 8192 * 2B = 4 GiB exactly.
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 192, 32, 8192, false, true, false));
    // Then it halves with the prompt, holding one score tensor at the budget.
    try testing.expectEqual(@as(usize, 4096), boundedPrefillChunk(8192, 192, 32, 16384, false, true, false));
    try testing.expectEqual(@as(usize, 2048), boundedPrefillChunk(8192, 192, 32, 32768, false, true, false));
    // The 38201-token prompt that measured a +28.8 GB peak at chunk 8192:
    // 4 GiB / (32 * 38201 * 2) = 1757 -> floored to the 512 grain.
    try testing.expectEqual(@as(usize, 1536), boundedPrefillChunk(8192, 192, 32, 38_201, false, true, false));
    try testing.expectEqual(@as(usize, 512), boundedPrefillChunk(8192, 192, 32, 262_144, false, true, false));
}

test "boundedPrefillChunk: never raises a caller-lowered base chunk" {
    // --prefill-chunk 1024 with headroom for 4096: the explicit lower value wins.
    try testing.expectEqual(@as(usize, 1024), boundedPrefillChunk(1024, 256, 16, 32768, true, false, false));
    // Even the floor never raises a tiny explicit base.
    try testing.expectEqual(@as(usize, 256), boundedPrefillChunk(256, 256, 16, 262_144, true, false, false));
}

test "boundedPrefillChunk: fused-causal (default) non-sliding hd-256 — MoE caps at 4096, dense keeps the full chunk" {
    // With the causal arm FUSED (default since the budgeted-dispatch flip) no
    // score tensor exists, so the scores-budget formula is moot for the
    // qwen3_5/3_6 class — and its old shrink to 1024 at 64K starved the
    // dequant+GEMM qmm route (engages at M >= 2048): the 64K rung was the
    // ladder's weakest for exactly this reason.
    std.debug.assert(transformer_mod.fused256_override == null);
    // DENSE hybrids (Qwen3.6-27B class): no expert-gather transients, and a
    // full-size chunk halves per-chunk dequant sweeps — chunk 8192 measured
    // +1.4% over 4096 at the 8K rung (M4 Max, 2026-07-30), flat at 32K.
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 256, 24, 8192, false, false, false));
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 256, 24, 65536, false, false, false));
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 256, 24, 140_000, false, false, false));
    // MoE keeps the 4096 cap: expert-gather transients scale with the chunk
    // (gemma-26B@99K: +3% speed for +22 GB peak is a bad trade).
    try testing.expectEqual(@as(usize, 4096), boundedPrefillChunk(8192, 256, 24, 8192, false, true, false));
    try testing.expectEqual(@as(usize, 4096), boundedPrefillChunk(8192, 256, 24, 140_000, false, true, false));
    // Never raises a caller-lowered base.
    try testing.expectEqual(@as(usize, 1024), boundedPrefillChunk(1024, 256, 24, 8192, false, false, false));
    // Sliding-band archs (gemma: fused band kernel wants big chunks) keep
    // the formula-only policy.
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 256, 16, 8192, true, false, false));
    // Fused head dims never cap regardless of arch.
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 128, 24, 1_000_000, false, false, false));
}

test "boundedPrefillChunk: a long-context-gated MoE hd-256 arch offers the 8192 rung, an ungated one caps at 4096" {
    std.debug.assert(transformer_mod.fused256_override == null);
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 256, 24, 8192, false, true, true));
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 256, 24, 140_000, false, true, true));
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 256, 24, 400_000, false, true, true));
    try testing.expectEqual(@as(usize, 4096), boundedPrefillChunk(8192, 256, 24, 8192, false, true, false));
    try testing.expectEqual(@as(usize, 4096), boundedPrefillChunk(8192, 256, 24, 140_000, false, true, false));
    try testing.expectEqual(@as(usize, 2048), boundedPrefillChunk(2048, 256, 24, 8192, false, true, true));
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 256, 24, 140_000, false, false, true));
}

test "boundedPrefillChunk: composed-causal keeps the 2048 cap + score formula" {
    // Composed causal (`fused256_override = false`): SMALLER
    // chunks measured faster on the 27B ladder (2026-07-12, M4 Max): 8K
    // prompt 225 -> 235.8 tok/s at chunk 2048 (peak 28.9 -> 19.8 GB).
    // Chunking IS block-level causal skipping for composed attention, and
    // the score transient shrinks with it.
    transformer_mod.fused256_override = false;
    defer transformer_mod.fused256_override = null;
    try testing.expectEqual(@as(usize, 2048), boundedPrefillChunk(8192, 256, 24, 8192, false, false, false));
    try testing.expectEqual(@as(usize, 2048), boundedPrefillChunk(8192, 256, 24, 32768, false, false, false));
    // The scores-budget formula still wins BELOW the cap: 64K on 24 heads
    // yields 1024 (measured better than 2048 there: 186 vs 182.3 tok/s).
    try testing.expectEqual(@as(usize, 1024), boundedPrefillChunk(8192, 256, 24, 65536, false, false, false));
}

test "ssmSnapshotBackoff: engages only under checkpointing and past the backoff length" {
    // No checkpointing (pure-attention archs, stride 0, vision): zero — the
    // final forward stays the classic 1-token logits pass.
    try testing.expectEqual(@as(usize, 0), ssmSnapshotBackoff(false, 8192, false));
    // Short cold prompts: nothing to back off (loop must keep >= 1 token).
    try testing.expectEqual(@as(usize, 0), ssmSnapshotBackoff(true, SSM_SNAPSHOT_BACKOFF, false));
    try testing.expectEqual(@as(usize, 0), ssmSnapshotBackoff(true, 1, false));
    // A restored tail inside the window is the cold path's final span: one forward of
    // the same rows, no new snapshot (greedy bytes must match the cold run).
    try testing.expectEqual(SSM_SNAPSHOT_BACKOFF, ssmSnapshotBackoff(true, SSM_SNAPSHOT_BACKOFF, true));
    try testing.expectEqual(@as(usize, 1), ssmSnapshotBackoff(true, 1, true));
    try testing.expectEqual(@as(usize, 0), ssmSnapshotBackoff(true, 0, true));
    // Checkpointing + long prompt: the always-on snapshot lands backoff
    // tokens before the prompt end, where the next turn's prefix match can
    // reach it (template generation-suffix divergence class).
    try testing.expectEqual(SSM_SNAPSHOT_BACKOFF, ssmSnapshotBackoff(true, 8192, false));
    try testing.expectEqual(SSM_SNAPSHOT_BACKOFF, ssmSnapshotBackoff(true, SSM_SNAPSHOT_BACKOFF + 1, true));
    // The tail forward must stay UNDER the prefill-eval-cadence threshold
    // (seq >= 32 turns it into a "prefill" costing ~450ms of eval bubbles):
    // tail = backoff + 1 <= 31.
    try testing.expect(SSM_SNAPSHOT_BACKOFF + 1 < 32);
}

test "effectiveSsmCheckpointStride: checkpointing never sub-divides the prefill chunk (dense AND MoE)" {
    const PREFILL_CHUNK: usize = 8192;
    // Disabled stays disabled.
    try testing.expectEqual(@as(usize, 0), effectiveSsmCheckpointStride(0, PREFILL_CHUNK));
    // Sub-chunk strides coarsen on ALL archs: they push every projection under
    // prefillDqGemm's M>=2048 floor (slow small-M qmm) and multiply per-chunk
    // fixed costs. Measured on Qwen3.6-27B dense (M4 Max, 2026-07-30): stride
    // 256 = 33 chunks at 8K = 211 tok/s vs 254 coarse.
    try testing.expectEqual(@as(usize, 8192), effectiveSsmCheckpointStride(256, PREFILL_CHUNK));
    // A larger explicit stride is respected (never shrunk).
    try testing.expectEqual(@as(usize, 16384), effectiveSsmCheckpointStride(16384, PREFILL_CHUNK));
    // End-to-end: an 851-tok prefill is 1 chunk (was 4 at the raw 256 stride
    // on dense hybrids — the llm_context_benchmarks small-prompt regression).
    try testing.expectEqual(@as(usize, 1), prefillChunkCount(851, PREFILL_CHUNK, true, effectiveSsmCheckpointStride(256, PREFILL_CHUNK), 0, false));
    // An 8K prefill splits only at the (memory-bound) chunk size, never
    // finer: 2 chunks at chunk 4096, not 33.
    try testing.expectEqual(@as(usize, 2), prefillChunkCount(8238, 4096, true, effectiveSsmCheckpointStride(256, PREFILL_CHUNK), 0, false));
}

test "vision prefill checkpoints SSM state only when it chunks like text" {
    const prefix_len: usize = 1587;
    const checkpoint_stride: u32 = 256;

    // Chunked vision (the default) checkpoints like text — a hybrid's image
    // turn is otherwise a guaranteed hot-cache miss (no checkpoint <= match).
    vision_chunked_cached = true;
    defer vision_chunked_cached = null;
    try testing.expect(shouldCheckpointSsmPrefill(checkpoint_stride, true, true));
    // The whole-prompt kill-switch arm has no chunk boundaries to snapshot at.
    vision_chunked_cached = false;
    const vision_checkpoints = shouldCheckpointSsmPrefill(checkpoint_stride, true, true);
    try testing.expect(!vision_checkpoints);
    try testing.expectEqual(
        prefix_len,
        nextChunkEnd(0, prefix_len, prefix_len, vision_checkpoints, @intCast(checkpoint_stride), 0, false),
    );

    const text_checkpoints = shouldCheckpointSsmPrefill(checkpoint_stride, true, false);
    try testing.expect(text_checkpoints);
    try testing.expectEqual(
        @as(usize, checkpoint_stride),
        nextChunkEnd(0, prefix_len, prefix_len, text_checkpoints, @intCast(checkpoint_stride), 0, false),
    );
}

fn mtpEvTestGenerator() Generator {
    var g: Generator = undefined;
    g.mtp_depth = Generator.MTP_ADAPTIVE_DEFAULT_CAP;
    g.mtp_depth_current = 1;
    g.mtp_window_drafted = @splat(0);
    g.mtp_window_accepted = @splat(0);
    g.mtp_window_idx = 0;
    g.mtp_rounds_since_switch = 0;
    g.mtp_promote_cooldown = 0;
    g.mtp_ev_accept = @splat(Generator.MTP_EV_PRIOR);
    g.mtp_ev_rounds = Generator.MTP_EV_WARMUP_ROUNDS;
    g.mtp_ev_m_lo_prev = 1;
    g.mtp_ev_costs = Generator.MTP_EV_DEFAULT_COSTS;
    g.spec_disabled_runtime = false;
    return g;
}

/// One flat f32 host copy of a lazy array, for the proposal-row assertions.
fn proposalTestRead(allocator: std.mem.Allocator, arr: mlx.mlx_array, n: usize, s: mlx.mlx_stream) ![]f32 {
    var as_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(as_f32);
    try mlx.check(mlx.mlx_astype(&as_f32, arr, .float32, s));
    var dense = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dense);
    try mlx.check(mlx.mlx_contiguous(&dense, as_f32, false, s));
    const flat_shape = [_]c_int{@intCast(n)};
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, dense, &flat_shape, 1, s));
    try mlx.check(mlx.mlx_array_eval(flat));
    const out = try allocator.alloc(f32, n);
    @memcpy(out, mlx.mlx_array_data_float32(flat).?[0..n]);
    return out;
}

test "shortlistProposal: q is zero off the shortlist, sums to 1, and peaks on the best candidate" {
    // Bar: q is a normalized density over the 32 candidates only, and the
    // draft is one of them.
    if (mlx.noGpuBackend()) return;
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    const V: c_int = 4096;
    const N: usize = 32;

    var ids: [N]u32 = undefined;
    var logits: [N]f32 = undefined;
    for (0..N) |i| {
        ids[i] = @intCast(37 + i * 101); // spread over the row, none at 0
        logits[i] = @as(f32, @floatFromInt(i)) * 0.25;
    }
    const best_id = ids[N - 1]; // highest logit
    const id_shape = [_]c_int{@intCast(N)};
    const cands = mlx.mlx_array_new_data(&ids, &id_shape, 1, .uint32);
    const exact_shape = [_]c_int{ 1, 1, @intCast(N) };
    const exact = mlx.mlx_array_new_data(&logits, &exact_shape, 3, .float32);
    var sl = mtp_mod.Shortlist{ .cands = cands, .exact = exact, .rows = V };
    defer sl.deinit();

    const sampling = Generator.mtpDraftSamplingFor(.{ .temperature = 1.0 }, false, Generator.MTP_DRAFT_TEMP);
    const prop = try shortlistProposal(sl, sampling, s);
    defer _ = mlx.mlx_array_free(prop.id);
    defer _ = mlx.mlx_array_free(prop.q);

    const q = try proposalTestRead(allocator, prop.q, @intCast(V), s);
    defer allocator.free(q);
    var sum: f64 = 0;
    var nonzero: usize = 0;
    var peak: usize = 0;
    for (q, 0..) |v, i| {
        sum += v;
        if (v != 0) {
            nonzero += 1;
            try testing.expect(std.mem.indexOfScalar(u32, &ids, @intCast(i)) != null);
        }
        if (v > q[peak]) peak = i;
    }
    try testing.expect(nonzero > 0 and nonzero <= N);
    try testing.expect(@abs(sum - 1.0) < 1e-4);
    try testing.expectEqual(best_id, @as(u32, @intCast(peak)));

    // Every draw lands on a candidate — never on a token q calls impossible.
    var draw: usize = 0;
    while (draw < 8) : (draw += 1) {
        const p2 = try shortlistProposal(sl, sampling, s);
        defer _ = mlx.mlx_array_free(p2.id);
        defer _ = mlx.mlx_array_free(p2.q);
        const drawn = try proposalTestRead(allocator, p2.id, 1, s);
        defer allocator.free(drawn);
        const tok: u32 = @intFromFloat(drawn[0]);
        try testing.expect(std.mem.indexOfScalar(u32, &ids, tok) != null);
    }
}

test "shortlistProposalRows: every row's q belongs to that row's own shortlist" {
    // The axis-bug class: one batched filter + one categorical over [K, 32] must
    // stay row-independent. Three rows with DISJOINT candidate ids and different
    // logits, so a cross-row leak or a flattened reduction is visible.
    if (mlx.noGpuBackend()) return;
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    const V: c_int = 4096;
    const N: usize = 3;
    const M: usize = 32;

    var ids: [N][M]u32 = undefined;
    var logits: [N][M]f32 = undefined;
    var sl: [N]mtp_mod.Shortlist = undefined;
    for (0..N) |r| {
        for (0..M) |i| {
            ids[r][i] = @intCast(11 + r * 41 + i * 127); // disjoint per row
            // Row r peaks at a different position, so an argmax read off the
            // wrong row lands on the wrong id.
            logits[r][i] = if (i == (r * 7 + 3) % M) 6.0 else @as(f32, @floatFromInt(i)) * 0.05;
        }
        const id_shape = [_]c_int{@intCast(M)};
        const ex_shape = [_]c_int{ 1, 1, @intCast(M) };
        sl[r] = .{
            .cands = mlx.mlx_array_new_data(&ids[r], &id_shape, 1, .uint32),
            .exact = mlx.mlx_array_new_data(&logits[r], &ex_shape, 3, .float32),
            .rows = V,
        };
    }
    defer for (&sl) |*x| x.deinit();

    const sampling = Generator.mtpDraftSamplingFor(.{ .temperature = 1.0 }, false, Generator.MTP_DRAFT_TEMP);
    const rows = [_]usize{ 0, 1, 2 };
    const out = try shortlistProposalRows(&sl, &rows, sampling, s);
    defer for (0..N) |j| {
        _ = mlx.mlx_array_free(out.ids[j]);
        _ = mlx.mlx_array_free(out.qs[j]);
    };

    for (0..N) |r| {
        const q = try proposalTestRead(allocator, out.qs[r], @intCast(V), s);
        defer allocator.free(q);
        var sum: f64 = 0;
        var peak: usize = 0;
        for (q, 0..) |v, i| {
            sum += v;
            if (v != 0) try testing.expect(std.mem.indexOfScalar(u32, &ids[r], @intCast(i)) != null);
            if (v > q[peak]) peak = i;
        }
        try testing.expect(@abs(sum - 1.0) < 1e-4);
        try testing.expectEqual(ids[r][(r * 7 + 3) % M], @as(u32, @intCast(peak)));
        const drawn = try proposalTestRead(allocator, out.ids[r], 1, s);
        defer allocator.free(drawn);
        try testing.expect(std.mem.indexOfScalar(u32, &ids[r], @intFromFloat(drawn[0])) != null);
    }
}

test "mtpDraftStepPath: a sampled proposal keeps the rerank shortlist" {
    // Bar: a sampled proposal reads the rerank shortlist; only the chunk-A
    // confidence gate needs the full lm_head.
    const P = Generator.DraftStepPath;
    try testing.expectEqual(P.rerank_sampled, Generator.mtpDraftStepPath(true, true, false));
    try testing.expectEqual(P.rerank_greedy, Generator.mtpDraftStepPath(true, false, false));
    // The extension gate wants ln p_head over the whole vocabulary.
    try testing.expectEqual(P.full_logits, Generator.mtpDraftStepPath(true, true, true));
    try testing.expectEqual(P.full_logits, Generator.mtpDraftStepPath(true, false, true));
    // No coarse head: nothing to shortlist.
    try testing.expectEqual(P.full_logits, Generator.mtpDraftStepPath(false, true, false));
    try testing.expectEqual(P.full_logits, Generator.mtpDraftStepPath(false, false, false));
}

test "mtpDraftSamplingFor: sharpened fixed proposal for stochastic targets, greedy stays greedy" {
    // Stochastic target: drafts sample from the FIXED sharpened distribution
    // (temp 0.6 / top_p 0.95 / top_k 20 — oMLX Lightning's _DRAFT_SAMPLER_*
    // constants; matched-temp drafting collapses on high-entropy content).
    const target = SamplingParams{ .temperature = 1.0, .top_p = 1.0, .top_k = 0, .repeat_penalty = 1.1 };
    const d = Generator.mtpDraftSamplingFor(target, false, Generator.MTP_DRAFT_TEMP);
    try testing.expectEqual(@as(f32, 0.6), d.temperature);
    try testing.expectEqual(@as(f32, 0.95), d.top_p);
    try testing.expectEqual(@as(u32, 20), d.top_k);
    // Non-sampler fields ride through untouched.
    try testing.expectEqual(@as(f32, 1.1), d.repeat_penalty);

    // Greedy target keeps greedy drafts (the temp-0 identity contract).
    const greedy = SamplingParams{ .temperature = 0.0 };
    try testing.expectEqual(@as(f32, 0.0), Generator.mtpDraftSamplingFor(greedy, false, Generator.MTP_DRAFT_TEMP).temperature);
    const near_greedy = SamplingParams{ .temperature = 0.005 };
    try testing.expectEqual(@as(f32, 0.0), Generator.mtpDraftSamplingFor(near_greedy, false, Generator.MTP_DRAFT_TEMP).temperature);

    // An explicit greedy request wins.
    try testing.expectEqual(@as(f32, 0.0), Generator.mtpDraftSamplingFor(target, true, Generator.MTP_DRAFT_TEMP).temperature);

    // The draft temperature is the FAMILY's, passed in, never a global.
    try testing.expectEqual(@as(f32, 1.0), Generator.mtpDraftSamplingFor(target, false, 1.0).temperature);
}

test "mtpDraftGreedyFor: the proposal mode is a property of the REQUEST" {
    const sampled = SamplingParams{ .temperature = 1.0 };
    const warm = SamplingParams{ .temperature = 0.7 };
    const cold = SamplingParams{ .temperature = 0.0 };
    const near_cold = SamplingParams{ .temperature = 0.005 };
    // top_k 1 is a deterministic target: its filtered p is one-hot, so the
    // argmax proposal already accepts with probability 1.
    const one_way = SamplingParams{ .temperature = 1.0, .top_k = 1 };

    // Sampled targets propose from the shortlist, greedy ones do not.
    try testing.expect(!Generator.mtpDraftGreedyFor(sampled));
    try testing.expect(!Generator.mtpDraftGreedyFor(warm));
    try testing.expect(Generator.mtpDraftGreedyFor(cold));
    // Bar: a sharp target keeps the argmax, a flat one samples.
    try testing.expect(Generator.mtpDraftGreedyFor(.{ .temperature = 0.3 }));
    try testing.expect(Generator.mtpDraftGreedyFor(near_cold));
    try testing.expect(Generator.mtpDraftGreedyFor(one_way));
}

test "mtpDraftGreedyAt: the greedy tail samples depth 0 only; a greedy request drafts the same bytes either way" {
    const sampled = SamplingParams{ .temperature = 1.0, .top_p = 0.95, .top_k = 20 };
    for (0..4) |d| {
        const step: u32 = @intCast(d);
        try testing.expect(!Generator.mtpDraftGreedyAt(sampled, false, step));
        try testing.expectEqual(step > 0, Generator.mtpDraftGreedyAt(sampled, true, step));
    }
    // A request whose chain is greedy already stays greedy at every depth.
    try testing.expect(Generator.mtpDraftGreedyAt(.{ .temperature = 0.3 }, true, 0));

    // The byte-identity contract: a greedy target's draft sampler is the same
    // value at every depth with the tail on as off, so its chain allocates no q
    // and drafts the argmax exactly as before.
    for ([_]f32{ 0.0, 0.005 }) |t| {
        const target = SamplingParams{ .temperature = t, .top_p = 0.95, .top_k = 20, .seed = 7 };
        for (0..mtp_mod.MAX_DEPTH) |d| {
            const step: u32 = @intCast(d);
            const on = Generator.mtpDraftSamplingFor(target, Generator.mtpDraftGreedyAt(target, true, step), Generator.MTP_DRAFT_TEMP_QWEN4);
            const off = Generator.mtpDraftSamplingFor(target, Generator.mtpDraftGreedyAt(target, false, step), Generator.MTP_DRAFT_TEMP_QWEN4);
            try testing.expect(std.meta.eql(on, off));
            try testing.expect(isGreedyTemperature(on.temperature));
        }
    }
}

test "mtpGreedyTailFor: the greedy tail flag outranks the per-model mtp_greedy_tail, which outranks off" {
    const saved = mtp_greedy_tail_explicit;
    defer mtp_greedy_tail_explicit = saved;
    mtp_greedy_tail_explicit = false;
    try testing.expectEqual(model_settings.Pick(bool){ .value = false, .source = .default }, mtpGreedyTailFor(null));
    try testing.expectEqual(model_settings.Pick(bool){ .value = true, .source = .model_settings }, mtpGreedyTailFor(true));
    try testing.expectEqual(model_settings.Pick(bool){ .value = false, .source = .model_settings }, mtpGreedyTailFor(false));
    mtp_greedy_tail_explicit = true;
    try testing.expectEqual(model_settings.Pick(bool){ .value = true, .source = .flag }, mtpGreedyTailFor(false));
    try testing.expectEqual(model_settings.Pick(bool){ .value = true, .source = .flag }, mtpGreedyTailFor(null));
}

/// Toy-vocab helpers for the host-level exactness simulations.
const ToyDist = struct {
    fn draw(dist: []const f64, rnd: std.Random) usize {
        const u = rnd.float(f64);
        var acc: f64 = 0;
        for (dist, 0..) |mass, i| {
            acc += mass;
            if (u < acc) return i;
        }
        return dist.len - 1;
    }

    /// normalize(max(p - q, 0)): the reject residual.
    fn residual(comptime V: usize, p: [V]f64, q: [V]f64) [V]f64 {
        var r: [V]f64 = undefined;
        var sum: f64 = 0;
        for (&r, p, q) |*x, pi, qi| {
            x.* = @max(pi - qi, 0);
            sum += x.*;
        }
        for (&r) |*x| x.* /= sum;
        return r;
    }

    fn oneHot(comptime V: usize, id: usize) [V]f64 {
        var d: [V]f64 = @splat(0);
        d[id] = 1;
        return d;
    }
};

test "spec sampling exactness: a greedy tail (depth 0 sampled, argmax after it, one-hot q) reproduces the target chain (toy vocab)" {
    // Two draft depths over V = 4, run through the engine's exact acceptance
    // (`exactPrefix`): depth 0 is drawn from q0 and handed to the verify as q0,
    // depth 1 is the draft head's argmax after the depth-0 token with a one-hot q.
    // Rejecting at a depth resamples normalize(max(p - q, 0)); a reject at depth 0
    // leaves depth 1 to the next round, i.e. to the target itself. Every emitted
    // pair must follow the target's joint p0(x0) p1(x1 | x0).
    const target = SamplingParams{ .temperature = 1.0 };
    try testing.expect(!Generator.mtpDraftGreedyAt(target, true, 0));
    try testing.expect(Generator.mtpDraftGreedyAt(target, true, 1));
    const V = 4;
    const p0 = [V]f64{ 0.1, 0.2, 0.3, 0.4 };
    const p1 = [V][V]f64{
        .{ 0.7, 0.1, 0.1, 0.1 },
        .{ 0.1, 0.2, 0.3, 0.4 },
        .{ 0.25, 0.25, 0.25, 0.25 },
        .{ 0.05, 0.05, 0.1, 0.8 },
    };
    const q0 = [V]f64{ 0.4, 0.3, 0.2, 0.1 };
    // The head's depth-1 argmax per depth-0 token: the target's mode on two rows, not on the others.
    const head_argmax = [V]usize{ 0, 1, 3, 2 };

    var prng = std.Random.DefaultPrng.init(0x6A11);
    var counts = std.mem.zeroes([V][V]u64);
    var tail_accepts: u64 = 0;
    const N: usize = 400_000;
    for (0..N) |_| {
        const d0 = ToyDist.draw(&q0, prng.random());
        const d1 = head_argmax[d0];
        const p_at = [_]f32{ @floatCast(p0[d0]), @floatCast(p1[d0][d1]) };
        const q_at = [_]f32{ @floatCast(q0[d0]), 1.0 };
        const accepted = mtp_acceptance.exactPrefix(&p_at, &q_at, null, &prng);
        var x0 = d0;
        var x1 = d1;
        switch (accepted) {
            0 => {
                x0 = ToyDist.draw(&ToyDist.residual(V, p0, q0), prng.random());
                x1 = ToyDist.draw(&p1[x0], prng.random());
            },
            1 => x1 = ToyDist.draw(&ToyDist.residual(V, p1[d0], ToyDist.oneHot(V, d1)), prng.random()),
            else => tail_accepts += 1,
        }
        counts[x0][x1] += 1;
    }
    for (0..V) |a| {
        for (0..V) |b| {
            const freq = @as(f64, @floatFromInt(counts[a][b])) / @as(f64, @floatFromInt(N));
            try testing.expect(@abs(freq - p0[a] * p1[a][b]) < 0.004);
        }
    }
    // The argmax tail lands (0.17 of rounds here), so the joint is not the target by rejection alone.
    try testing.expect(tail_accepts > N / 10);
}

test "filteredProbsBlock: every draft row is the SAME density the sample is drawn from" {
    // Exactness rests on q being the true proposal: the draft must be drawn
    // from byte-for-byte the row handed to `specAcceptProb`. The way that
    // breaks silently is filtering the two differently — so pin that a row
    // is a normalized distribution supported ONLY on the kept set, and that
    // `mlx_random_categorical` over log(q) can never land outside it.
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    const V: c_int = 8;
    const M: c_int = 3;
    // Three rows with deliberately different argmaxes, so a row-blind
    // implementation (broadcasting one row's mask) fails.
    const raw = [_]f32{
        5.0, 4.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 1.0, 4.0, 5.0, 0.0,
        0.0, 9.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0,
    };
    const shape = [_]c_int{ 1, M, V };
    const logits = mlx.mlx_array_new_data(&raw, &shape, 3, .float32);
    defer _ = mlx.mlx_array_free(logits);

    var sampling = SamplingParams{};
    sampling.temperature = 0.8;
    sampling.top_k = 2;
    const q = try filteredProbsBlock(logits, sampling, s);
    defer _ = mlx.mlx_array_free(q);
    try mlx.check(mlx.mlx_array_eval(q));
    const qsh = mlx.getShape(q);
    try testing.expectEqual(M, qsh[0]);
    try testing.expectEqual(V, qsh[1]);

    const data = mlx.mlx_array_data_float32(q) orelse return error.MlxArrayDataNull;
    // Per row: sums to 1, and exactly top_k entries carry mass.
    const want_support = [_][2]usize{ .{ 0, 1 }, .{ 5, 6 }, .{ 1, 3 } };
    for (0..@intCast(M)) |r| {
        var sum: f32 = 0;
        var nonzero: usize = 0;
        for (0..@intCast(V)) |c| {
            const v = data[r * @as(usize, @intCast(V)) + c];
            sum += v;
            if (v > 0) nonzero += 1;
        }
        try testing.expect(@abs(sum - 1.0) < 1e-5);
        try testing.expectEqual(@as(usize, 2), nonzero);
        for (want_support[r]) |c| {
            try testing.expect(data[r * @as(usize, @intCast(V)) + c] > 0);
        }
    }

    // Draw the way nextDflash does and assert every sample is in q's support.
    var logq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(logq);
    try mlx.check(mlx.mlx_log(&logq, q, s));
    var draws: usize = 0;
    while (draws < 24) : (draws += 1) {
        const null_key = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(null_key);
        var sampled = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sampled);
        try mlx.check(mlx.mlx_random_categorical(&sampled, logq, -1, null_key, s));
        var ids = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ids);
        try mlx.check(mlx.mlx_astype(&ids, sampled, .int32, s));
        try mlx.check(mlx.mlx_array_eval(ids));
        const idd = mlx.mlx_array_data_int32(ids) orelse return error.MlxArrayDataNull;
        for (0..@intCast(M)) |r| {
            const tok: usize = @intCast(idd[r]);
            // q(draft) > 0 is what keeps the accept ratio finite.
            try testing.expect(data[r * @as(usize, @intCast(V)) + tok] > 0);
            // And `sliceProbRow` must hand back THAT row, not another.
            const row = try sliceProbRow(q, @intCast(r), s);
            defer _ = mlx.mlx_array_free(row);
            const got = try probAt(row, @intCast(tok), s);
            try testing.expectApproxEqAbs(data[r * @as(usize, @intCast(V)) + tok], got, 1e-6);
        }
    }
    _ = allocator;
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

test "MTP EV seed round-trips on the qwen4 head; a fresh or absent head reads null" {
    // The in-checkpoint head stores the seed on `Qwen4Mtp` — per loaded model,
    // exactly like the sidecar's `MtpModel.ev_seed_*`. Before this it was a
    // null/no-op stub, so every qwen4 request re-warmed the controller from
    // scratch. Only the two seed fields are touched here; the rest of the head
    // (and of the Transformer) is never read on this path.
    var t: Transformer = undefined;
    t.qwen4_mtp = null;
    const ref = MtpHeadRef{ .qwen4 = &t };

    // `--no-mtp`: no head to seed, and setting one must not fault.
    try testing.expect(ref.evSeed() == null);
    var zeros: [mtp_mod.MAX_DEPTH]f32 = @splat(0.5);
    ref.setEvSeed(zeros, 4);
    try testing.expect(ref.evSeed() == null);

    var head: transformer_mod.Qwen4Mtp = undefined;
    head.ev_seed_accept = null;
    head.ev_seed_m_lo = 1;
    t.qwen4_mtp = head;
    // A head that has never published reads null — not a zeroed surface.
    try testing.expect(ref.evSeed() == null);

    var accept: [mtp_mod.MAX_DEPTH]f32 = @splat(0.0);
    accept[0] = 0.91;
    accept[1] = 0.80;
    accept[2] = 0.68;
    ref.setEvSeed(accept, 5);
    const got = ref.evSeed() orelse return error.SeedMissing;
    try testing.expectEqual(@as(u32, 5), got.m_lo);
    try testing.expectEqualSlices(f32, &accept, &got.accept);

    // Overwrite, not accumulate: the LAST healthy request wins.
    zeros[0] = 0.10;
    ref.setEvSeed(zeros, 2);
    const again = ref.evSeed() orelse return error.SeedMissing;
    try testing.expectEqual(@as(u32, 2), again.m_lo);
    try testing.expectEqualSlices(f32, &zeros, &again.accept);
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
    for ([_]mtp_mod.MtpCostProfile{ .g17_nax_q8_gs32, .g17_nax_q4_gs32, .g17_nax_q4_gs64, .g17_nax_q6_gs64, .g17_nax_q8_gs64, .g17_nax_oq4e_q4_gs64 }) |profile| {
        try testing.expectEqual(Generator.MTP_ADAPTIVE_NAX_CAP, Generator.mtpDepthCapForProfile(0, true, profile));
        try testing.expectEqual(@as(u32, 8), Generator.mtpDepthCapForProfile(0, true, profile));
    }
    // qwen4's calibrated surface keeps the DEFAULT cap: no NAX region exists
    // to justify opening depths 7-8 on that arch.
    try testing.expectEqual(Generator.MTP_ADAPTIVE_DEFAULT_CAP, Generator.mtpDepthCapForProfile(0, true, .g17_nax_qwen4_q4_gs64));
    try testing.expectEqual(@as(u32, 6), Generator.mtpDepthCapForProfile(0, true, .g17_nax_qwen4_q4_gs64));
    for ([_]mtp_mod.MtpCostProfile{ .generic, .g17_nax_q8_gs32, .g17_nax_q4_gs32, .g17_nax_q4_gs64, .g17_nax_q6_gs64, .g17_nax_q8_gs64, .g17_nax_oq4e_q4_gs64, .g17_nax_qwen4_q4_gs64, .g17_nax_qwen4_mixed_4_8_gs64 }) |profile| {
        try testing.expectEqual(mtp_mod.DEFAULT_DEPTH, Generator.mtpDepthCapForProfile(0, false, profile));
    }
    // Explicit values ignore both controller mode and profile, and remain
    // clamped to [1, MAX_DEPTH].
    try testing.expectEqual(@as(u32, 5), Generator.mtpDepthCapForProfile(5, true, .generic));
    try testing.expectEqual(@as(u32, 5), Generator.mtpDepthCapForProfile(5, false, .g17_nax_q8_gs32));
    try testing.expectEqual(@as(u32, 7), Generator.mtpDepthCapForProfile(7, true, .generic));
    try testing.expectEqual(@as(u32, 8), Generator.mtpDepthCapForProfile(8, true, .generic));
    try testing.expectEqual(@as(u32, 2), Generator.mtpDepthCapForProfile(2, true, .g17_nax_q4_gs32));
    try testing.expectEqual(mtp_mod.MAX_DEPTH, Generator.mtpDepthCapForProfile(12, true, .generic));

    // The original boolean helpers remain source-compatible and map true to
    // the pre-existing q8 profile.
    // The live resolver must NAME the row it applied, once. A silicon fence
    // nobody can see in the log is indistinguishable from the EV controller's
    // own choice at the same depth (dflash's ready line does the same).
    {
        const src = @embedFile("generate.zig");
        const start = std.mem.indexOf(u8, src, "pub fn mtpDepthCapForProfile(configured") orelse return error.MissingResolver;
        const end = std.mem.indexOfPos(u8, src, start + 1, "\n    }\n") orelse return error.MissingResolverEnd;
        const body = src[start..end];
        try testing.expect(std.mem.indexOf(u8, body, "row.label") != null);
        try testing.expect(std.mem.indexOf(u8, body, "mtp_depth_cap_logged") != null);
    }

    const live_generic = mtp_mod.adaptiveDepthCapForMachine(chip_mod.chipBrand(), Generator.MTP_ADAPTIVE_DEFAULT_CAP).cap;
    try testing.expectEqual(live_generic, Generator.mtpDepthCapFor(0, true, false));
    try testing.expectEqual(@as(u32, 8), Generator.mtpDepthCapFor(0, true, true));
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

test "updateMtpEvRound: sticky disable uses the first draft at base depth one" {
    var good = mtpEvTestGenerator();
    for (0..Generator.MTP_DEPTH_WINDOW * 2) |_| good.updateMtpEvRound(8, 1);
    try testing.expect(!good.spec_disabled_runtime);
    for (good.mtp_window_drafted) |sample| try testing.expectEqual(@as(u8, 1), sample);
    for (good.mtp_window_accepted) |sample| try testing.expectEqual(@as(u8, 1), sample);

    var bad = mtpEvTestGenerator();
    for (0..Generator.MTP_DEPTH_WINDOW - 1) |_| {
        bad.updateMtpEvRound(8, 0);
        try testing.expect(!bad.spec_disabled_runtime);
    }
    bad.updateMtpEvRound(8, 0);
    try testing.expect(bad.spec_disabled_runtime);
}

test "updateMtpEvRound: warmup and wider base rounds reset floor evidence" {
    var g = mtpEvTestGenerator();
    g.mtp_ev_rounds = Generator.MTP_EV_WARMUP_ROUNDS - 1;
    g.mtp_window_idx = 7;
    g.updateMtpEvRound(4, 0);
    try testing.expectEqual(Generator.MTP_EV_WARMUP_ROUNDS, g.mtp_ev_rounds);
    try testing.expectEqual(@as(u32, 0), g.mtp_window_idx);

    for (0..Generator.MTP_DEPTH_WINDOW - 1) |_| g.updateMtpEvRound(1, 0);
    try testing.expect(!g.spec_disabled_runtime);
    try testing.expectEqual(Generator.MTP_DEPTH_WINDOW - 1, g.mtp_window_idx);

    g.mtp_ev_m_lo_prev = 2;
    g.updateMtpEvRound(4, 0);
    try testing.expectEqual(@as(u32, 0), g.mtp_window_idx);
    try testing.expect(!g.spec_disabled_runtime);
}

test "the qwen4 rerank draft feeds the MIXER output, never the pre-mixer stream" {
    // The one way this change can be wrong without failing to compile. On the
    // sidecar arm `hidden_next` IS the lm_head's input, so the obvious wiring
    // (`draftSelect(xfm, step_out.hidden_next, ...)`) is correct there and
    // silently wrong on qwen4_exp, whose `hidden_next` is the pre-mixer
    // `[B,S,hc*H]` stream — 4x the width the head consumes. The chain must
    // therefore prefer `rerank_x` when a step publishes one.
    const source = @embedFile("generate.zig");
    const build_at = std.mem.indexOf(u8, source, "fn mtpChainBuild(") orelse return error.MissingChainBuild;
    const end = std.mem.indexOfPos(u8, source, build_at, "\n    /// Fire the chain's built graphs") orelse source.len;
    const body = source[build_at..end];

    const sel = std.mem.indexOf(u8, body, "head.draftSelect(") orelse return error.MissingDraftSelect;
    const line_start = std.mem.lastIndexOfScalar(u8, body[0..sel], '\n') orelse 0;
    const line_end = std.mem.indexOfScalarPos(u8, body, sel, '\n') orelse body.len;
    const call = body[line_start..line_end];
    // The rerank input is the resolved vector, never `hidden_next` directly.
    try testing.expect(std.mem.indexOf(u8, call, "rerank_x") != null);
    try testing.expect(std.mem.indexOf(u8, call, "hidden_next") == null);
    try testing.expect(std.mem.indexOf(u8, body, "step_out.rerank_x.ctx != null") != null);

    // ...and a `.mixed` step is what asks for it: a rerank draft must not
    // collapse onto `.none` (the history-append mode), which produces no
    // mixer output at all.
    try testing.expect(std.mem.indexOf(u8, body, "if (use_rerank) .mixed else .logits") != null);

    // The qwen4 arm maps `.mixed` onto the projection that SKIPS the lm_head
    // but still runs the mixer. If it ever mapped to `.none`, `rerank_x` would
    // be null and the chain would silently hand over the pre-mixer stream.
    const step_at = std.mem.indexOf(u8, source, "fn qwen4Step(") orelse return error.MissingQwen4Step;
    const step_end = std.mem.indexOfPos(u8, source, step_at, "\n    /// One head forward") orelse source.len;
    const step_body = source[step_at..step_end];
    try testing.expect(std.mem.indexOf(u8, step_body, ".mixed => .mixed_last_row") != null);
    try testing.expect(std.mem.indexOf(u8, step_body, ".rerank_x = out.mixed") != null);

    // And the owned mixer vector is freed by the chain, not leaked per step.
    try testing.expect(std.mem.indexOf(u8, body, "mlx_array_free(step_out.rerank_x)") != null);
}

test "mtpForcedWidth: a forced MiMo round never plans past the verify row budget" {
    // A 5..8 forced depth would hand the forward 6..9 rows, which the decode-shaped
    // verify cannot serve: the forward would silently take the prefill arm.
    for ([_]u32{ 1, 3, 5, 6, 8 }) |forced| {
        const mimo_max = Generator.mtpVerifyDraftsMax(true);
        try testing.expectEqual(@min(forced, mimo_max), Generator.mtpForcedWidth(forced, mimo_max));
    }
    // Another arch's rows are row-identical at any width the lever can name.
    for ([_]u32{ 1, 4, 5, 8 }) |forced| {
        const max = Generator.mtpVerifyDraftsMax(false);
        try testing.expectEqual(forced, Generator.mtpForcedWidth(forced, max));
    }
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

test "Qwen4 mixed MTP cost prior prices the measured width-seven fallback" {
    const costs = Generator.mtpEvCostsForProfile(.g17_nax_qwen4_mixed_4_8_gs64);
    // The 8k forced-depth echo sweep measures about 5 ms per position
    // through depth 5, then an approximately 8 ms step at depth 6.
    // Width 7 is outside the single-stream verifier's optimized envelope.
    const optimized = Generator.mtpEvMarginalCost(costs, 5);
    const fallback = Generator.mtpEvMarginalCost(costs, 6);
    try testing.expect(fallback > 1.4 * optimized);
    try testing.expect(Generator.mtpEvMarginalCost(costs, 7) < fallback);

    // At this confidence the old smooth ramp picks depth 6. The measured
    // fallback cost makes depth 5 the better cold-start expected rate.
    const acceptance: [6]f32 = @splat(0.97);
    const plan = Generator.mtpEvPlanFor(&acceptance, 6, costs, 6);
    try testing.expectEqual(@as(u32, 5), plan.m_lo);
}

test "mtpEvCostsForProfile: G17 profiles are explicit" {
    const generic = Generator.mtpEvCostsForProfile(.generic);
    try testing.expectEqual(Generator.MTP_EV_DEFAULT_COSTS, generic);

    const q8 = Generator.mtpEvCostsForProfile(.g17_nax_q8_gs32);
    try testing.expectEqual(Generator.MTP_EV_G17_NAX_COSTS, q8);
    try testing.expectEqual(@as(u32, 7), q8.nax_from);

    const q4 = Generator.mtpEvCostsForProfile(.g17_nax_q4_gs32);
    try testing.expectEqual(Generator.MTP_EV_G17_NAX_Q4_GS32_COSTS, q4);
    try testing.expectEqual(@as(u32, 7), q4.nax_from);

    const q4_gs64 = Generator.mtpEvCostsForProfile(.g17_nax_q4_gs64);
    try testing.expectEqual(Generator.MTP_EV_G17_NAX_Q4_GS64_COSTS, q4_gs64);
    try testing.expectEqual(@as(u32, 7), q4_gs64.nax_from);

    const oq4e = Generator.mtpEvCostsForProfile(.g17_nax_oq4e_q4_gs64);
    try testing.expectEqual(Generator.MTP_EV_G17_NAX_OQ4E_Q4_GS64_COSTS, oq4e);
    try testing.expectEqual(@as(u32, 7), oq4e.nax_from);

    const qwen4 = Generator.mtpEvCostsForProfile(.g17_nax_qwen4_q4_gs64);
    try testing.expectEqual(Generator.MTP_EV_G17_NAX_QWEN4_Q4_GS64_COSTS, qwen4);
    try testing.expect(!std.meta.eql(generic, qwen4));
}

test "MTP_EV_G17_NAX_COSTS reproduces the measured M5 depth-6/depth-8 ratio" {
    const costs = Generator.MTP_EV_G17_NAX_COSTS;
    const t6 = Generator.mtpEvRoundCost(costs, 6, false);
    const t8 = Generator.mtpEvRoundCost(costs, 8, false);
    try testing.expectApproxEqAbs(@as(f32, 1.99), t6, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 2.19), t8, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 68.39 / 62.15), t8 / t6, 2e-3);
    try testing.expectApproxEqAbs(@as(f32, 0.21), Generator.mtpEvMarginalCost(costs, 6), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.10), Generator.mtpEvMarginalCost(costs, 7), 1e-6);
}

test "MTP_EV_G17_NAX_Q4_GS32_COSTS encodes calibrated composite marginals" {
    const costs = Generator.MTP_EV_G17_NAX_Q4_GS32_COSTS;
    try testing.expect(costs.draft > 0.0);
    try testing.expect(costs.per_pos_lo > 0.0);
    try testing.expect(costs.per_pos_hi > 0.0);
    try testing.expect(costs.per_pos_nax > 0.0);
    try testing.expectApproxEqAbs(@as(f32, 0.11), Generator.mtpEvMarginalCost(costs, 3), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.20), Generator.mtpEvMarginalCost(costs, 4), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.05), Generator.mtpEvMarginalCost(costs, 7), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.93), Generator.mtpEvRoundCost(costs, 6, false), 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 2.03), Generator.mtpEvRoundCost(costs, 8, false), 1e-5);
}

test "MTP_EV_G17_NAX_Q4_GS64_COSTS reproduces the calibrated M5 surface" {
    const costs = Generator.MTP_EV_G17_NAX_Q4_GS64_COSTS;
    try testing.expectApproxEqAbs(@as(f32, 0.075), Generator.mtpEvMarginalCost(costs, 3), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.195), Generator.mtpEvMarginalCost(costs, 4), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.08), Generator.mtpEvMarginalCost(costs, 7), 1e-6);
    const t1 = Generator.mtpEvRoundCost(costs, 1, false);
    const t3 = Generator.mtpEvRoundCost(costs, 3, false);
    const t6 = Generator.mtpEvRoundCost(costs, 6, false);
    const t8 = Generator.mtpEvRoundCost(costs, 8, false);
    try testing.expectApproxEqAbs(@as(f32, 1.075), t1, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1.225), t3, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1.81), t6, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1.97), t8, 1e-5);
}

test "MTP_EV_G17_NAX_Q6_GS64_COSTS reproduces the calibrated M5 surface" {
    const costs = Generator.MTP_EV_G17_NAX_Q6_GS64_COSTS;
    try testing.expectApproxEqAbs(@as(f32, 0.06), Generator.mtpEvMarginalCost(costs, 3), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.115), Generator.mtpEvMarginalCost(costs, 4), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.145), Generator.mtpEvMarginalCost(costs, 7), 1e-6);
    const t6 = Generator.mtpEvRoundCost(costs, 6, false);
    const t8 = Generator.mtpEvRoundCost(costs, 8, false);
    try testing.expectApproxEqAbs(@as(f32, 1.525), t6, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1.815), t8, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 76.79 / 64.495), t8 / t6, 1e-3);
}

test "MTP_EV_G17_NAX_Q8_GS64_COSTS bridges the discontinuous NAX takeover" {
    const costs = Generator.MTP_EV_G17_NAX_Q8_GS64_COSTS;
    try testing.expectApproxEqAbs(@as(f32, 0.06), Generator.mtpEvMarginalCost(costs, 4), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.19), Generator.mtpEvMarginalCost(costs, 5), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.045), Generator.mtpEvMarginalCost(costs, 7), 1e-6);
    const t6 = Generator.mtpEvRoundCost(costs, 6, false);
    const t8 = Generator.mtpEvRoundCost(costs, 8, false);
    try testing.expectApproxEqAbs(@as(f32, 1.62), t6, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1.71), t8, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 85.94 / 53.0575), t6, 2e-3);
    try testing.expectApproxEqAbs(@as(f32, 90.68 / 53.0575), t8, 2e-3);
}

test "MTP_EV_G17_NAX_OQ4E_Q4_GS64_COSTS reproduces the measured M5 surface" {
    const costs = Generator.MTP_EV_G17_NAX_OQ4E_Q4_GS64_COSTS;
    try testing.expectApproxEqAbs(@as(f32, 0.095), Generator.mtpEvMarginalCost(costs, 3), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.22), Generator.mtpEvMarginalCost(costs, 4), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.025), Generator.mtpEvMarginalCost(costs, 7), 1e-6);
    const t6 = Generator.mtpEvRoundCost(costs, 6, false);
    const t8 = Generator.mtpEvRoundCost(costs, 8, false);
    try testing.expectApproxEqAbs(@as(f32, 1.945), t6, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 1.995), t8, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 62.79 / 61.20), t8 / t6, 5e-4);
}

test "mtpEvPlanFor: MiMo's surface drafts one on prose and three on code, where generic costs over-draft prose" {
    // Conditional per-index acceptance measured with the three heads (forced depth 3).
    const prose = [_]f32{ 0.59, 0.53, 0.42, 0, 0, 0, 0, 0 };
    const code = [_]f32{ 1.0, 0.91, 0.89, 0, 0, 0, 0, 0 };
    const mimo = Generator.mtpEvCostsForProfile(.mimo_exl3);
    try testing.expectEqual(@as(u32, 1), Generator.mtpEvPlanFor(&prose, 3, mimo, 3).m_lo);
    try testing.expectEqual(@as(u32, 3), Generator.mtpEvPlanFor(&code, 3, mimo, 3).m_lo);
    try testing.expect(Generator.mtpEvPlanFor(&prose, 3, Generator.MTP_EV_DEFAULT_COSTS, 3).m_lo > 1);
}

test "the MiMo EXL3 round-cost surface serves every KV width, kv6 included, and other layouts stay generic" {
    var xfm: Transformer = undefined;
    const head = MtpHeadRef{ .mimo = undefined };
    const kv = transformer_mod.KVQuantConfig;
    xfm.config = .{ .expert_layout = .exl3_k4 };
    for ([_]kv{ kv.dense, kv.affine(4), kv.affine(6), kv.affine(8) }) |width| {
        try testing.expectEqual(mtp_mod.MtpCostProfile.mimo_exl3, head.costProfile(&xfm, width));
    }
    xfm.config = .{ .expert_layout = .quantized_split };
    try testing.expectEqual(mtp_mod.MtpCostProfile.generic, head.costProfile(&xfm, kv.affine(6)));
}

test "mtpEvPlanFor: M5 NAX surfaces open depth 8 from realistic warmup EMAs" {
    const p = Generator.MTP_EV_PRIOR;
    const a = [_]f32{ 0.97, 0.89, p, p, p, p, p, p };

    // The M1-M4 surface stops at the first expensive ramp position.
    const generic = Generator.mtpEvPlanFor(&a, 8, Generator.MTP_EV_DEFAULT_COSTS, 3);
    try testing.expectEqual(@as(u32, 3), generic.m_lo);
    try testing.expect(generic.m_hi < 8);

    // The M5 fit captures both its cheaper intermediate widths and the NAX
    // takeover at draft position 7, so the same evidence reaches depth 8.
    for ([_]Generator.MtpEvCosts{ Generator.MTP_EV_G17_NAX_COSTS, Generator.MTP_EV_G17_NAX_Q4_GS32_COSTS, Generator.MTP_EV_G17_NAX_Q4_GS64_COSTS, Generator.MTP_EV_G17_NAX_Q6_GS64_COSTS, Generator.MTP_EV_G17_NAX_Q8_GS64_COSTS, Generator.MTP_EV_G17_NAX_OQ4E_Q4_GS64_COSTS }) |costs| {
        const nax = Generator.mtpEvPlanFor(&a, 8, costs, 3);
        try testing.expectEqual(@as(u32, 3), nax.m_lo);
        try testing.expectEqual(@as(u32, 8), nax.m_hi);
        try testing.expect(nax.tau_ln < 0.0);
    }

    // qwen4_exp's measured surface has no NAX takeover — a verify row is
    // bytes at every width — so the same base depth stands but the same
    // evidence must NOT extend to depth 8.
    const qwen4_plan = Generator.mtpEvPlanFor(&a, 8, Generator.MTP_EV_G17_NAX_QWEN4_Q4_GS64_COSTS, 3);
    try testing.expectEqual(@as(u32, 3), qwen4_plan.m_lo);
    try testing.expect(qwen4_plan.m_hi < 8);

    const cold = [_]f32{ 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 };
    for ([_]Generator.MtpEvCosts{ Generator.MTP_EV_G17_NAX_COSTS, Generator.MTP_EV_G17_NAX_Q4_GS32_COSTS, Generator.MTP_EV_G17_NAX_OQ4E_Q4_GS64_COSTS, Generator.MTP_EV_G17_NAX_QWEN4_Q4_GS64_COSTS }) |costs| {
        const cold_plan = Generator.mtpEvPlanFor(&cold, 8, costs, 8);
        try testing.expectEqual(@as(u32, 1), cold_plan.m_lo);
        try testing.expectEqual(@as(u32, 1), cold_plan.m_hi);
    }
    // Qwen3.8's measured depth-1 marginal is cheap enough that even this
    // synthetic 20% first-draft case clears the exploration floor narrowly.
    // The base stays at one and only one confidence-gated position is exposed.
    const q38_cold = Generator.mtpEvPlanFor(&cold, 8, Generator.MTP_EV_G17_NAX_Q4_GS64_COSTS, 8);
    try testing.expectEqual(@as(u32, 1), q38_cold.m_lo);
    try testing.expectEqual(@as(u32, 2), q38_cold.m_hi);
    try testing.expect(q38_cold.tau_ln < 0.0);
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

test "mtpRegimeObserve: averages the first 1/beta samples, then moves by the cost beta, reseeds on a new base depth" {
    var r = Generator.MtpRegime{};
    Generator.mtpRegimeObserve(&r, true, 3, 45.0, 4.0); // first round counts
    try testing.expect(r.two_tok > 0.0);
    Generator.mtpRegimeObserve(&r, false, 3, 45.0, 4.0); // transition: dropped
    try testing.expect(r.one_tok == 0.0);
    Generator.mtpRegimeObserve(&r, false, 3, 50.0, 4.0);
    try testing.expectApproxEqAbs(@as(f32, 50.0), r.one_ms, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 4.0), r.one_tok, 1e-6);
    Generator.mtpRegimeObserve(&r, false, 3, 60.0, 5.0);
    try testing.expectApproxEqAbs(@as(f32, 55.0), r.one_ms, 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 4.5), r.one_tok, 1e-4);
    for (0..8) |_| Generator.mtpRegimeObserve(&r, false, 3, 60.0, 5.0);
    try testing.expectApproxEqAbs(@as(f32, 59.0), r.one_ms, 1e-3);
    Generator.mtpRegimeObserve(&r, false, 3, 70.0, 5.0);
    try testing.expectApproxEqAbs(59.0 + Generator.MTP_EV_COST_BETA * 11.0, r.one_ms, 1e-3);
    // The climb moved m_lo: the depth-3 rounds are not the depth-4 regime.
    Generator.mtpRegimeObserve(&r, false, 4, 70.0, 5.0);
    try testing.expectApproxEqAbs(@as(f32, 70.0), r.one_ms, 1e-6);
    // A two-chunk reading at a different base depth yields no verdict.
    Generator.mtpRegimeObserve(&r, true, 3, 80.0, 6.0);
    Generator.mtpRegimeObserve(&r, true, 3, 80.0, 6.0);
    try testing.expect(Generator.mtpRegimeTwoChunkWorse(r) == null);
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

test "mtpEmaMs: seeds on the first sample then folds nanoseconds into a ms EMA" {
    const D = Generator;
    // First sample seeds (prev <= 0): 45 ms == 45_000_000 ns, exactly.
    try testing.expectApproxEqAbs(@as(f32, 45.0), D.mtpEmaMs(0.0, 45_000_000), 1e-3);
    // A subsequent sample folds at MTP_EV_COST_BETA toward the new value.
    const folded = D.mtpEmaMs(45.0, 40_000_000); // 40 ms sample
    try testing.expectApproxEqAbs(45.0 + D.MTP_EV_COST_BETA * (40.0 - 45.0), folded, 1e-3);
    // Env parse: default on, "0" off, "" on.
    try testing.expect(D.mtpCostTableEnabledFromEnv(null));
    try testing.expect(!D.mtpCostTableEnabledFromEnv("0"));
    try testing.expect(D.mtpCostTableEnabledFromEnv(""));
}

test "batched corrections: point-mass residuals sample deterministically, accept vec gathers draft probs" {
    const s = mlx.gpuStream();
    // V=4, m=2. Row residuals are POINT MASSES so the batched categorical
    // is deterministic end-to-end (no seed dependence):
    //   row0: p=[.5,.5,0,0], draft=0 → residual [0,.5,0,0] → corr 1
    //   row1: p=[0,.25,.75,0], draft=2 → residual [0,.25,0,0] → corr 1
    //   bonus: p=[0,0,0,1] → corr 3
    const p_data = [_]f32{ 0.5, 0.5, 0, 0, 0, 0.25, 0.75, 0, 0, 0, 0, 1.0 };
    const p_shape = [_]c_int{ 1, 3, 4 };
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

    // Greedy proposals (q == null): one-hot residuals.
    var g = try Generator.mtpBatchedAcceptGraph(probs_all, &drafts, null, 2, .{}, s);
    defer g.deinit();
    try mlx.check(mlx.mlx_array_eval(g.corr_samples));
    const corr = mlx.mlx_array_data_int32(g.corr_samples) orelse return error.InvalidDtype;
    try testing.expectEqual(@as(i32, 1), corr[0]);
    try testing.expectEqual(@as(i32, 1), corr[1]);
    try testing.expectEqual(@as(i32, 3), corr[2]);
    try mlx.check(mlx.mlx_array_eval(g.accept_p));
    const ap = mlx.mlx_array_data_float32(g.accept_p) orelse return error.InvalidDtype;
    try testing.expectApproxEqAbs(@as(f32, 0.5), ap[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.75), ap[1], 1e-6);
    try testing.expect(g.accept_q.ctx == null);

    // Sharp proposals: q rows equal to the one-hots reproduce the same
    // residuals, and accept_q gathers the proposal's own density.
    const q_data0 = [_]f32{ 1, 0, 0, 0 };
    const q_data1 = [_]f32{ 0, 0, 1, 0 };
    const q_shape = [_]c_int{ 1, 4 };
    const q0 = mlx.mlx_array_new_data(&q_data0, &q_shape, 2, .float32);
    defer _ = mlx.mlx_array_free(q0);
    const q1 = mlx.mlx_array_new_data(&q_data1, &q_shape, 2, .float32);
    defer _ = mlx.mlx_array_free(q1);
    const qs = [_]mlx.mlx_array{ q0, q1 };
    var gs = try Generator.mtpBatchedAcceptGraph(probs_all, &drafts, &qs, 2, .{}, s);
    defer gs.deinit();
    try mlx.check(mlx.mlx_array_eval(gs.corr_samples));
    const corr2 = mlx.mlx_array_data_int32(gs.corr_samples) orelse return error.InvalidDtype;
    try testing.expectEqual(@as(i32, 1), corr2[0]);
    try testing.expectEqual(@as(i32, 1), corr2[1]);
    try testing.expectEqual(@as(i32, 3), corr2[2]);
    try mlx.check(mlx.mlx_array_eval(gs.accept_q));
    const aq = mlx.mlx_array_data_float32(gs.accept_q) orelse return error.InvalidDtype;
    try testing.expectApproxEqAbs(@as(f32, 1.0), aq[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.0), aq[1], 1e-6);
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

test "seeded MTP draft and correction draws replay despite unrelated MLX random draws" {
    const s = mlx.gpuStream();
    const q_data = [_]f32{ 0.5, 0.5, 0.0, 0.0 };
    const q_shape = [_]c_int{ 1, 4 };
    const q = mlx.mlx_array_new_data(&q_data, &q_shape, 2, .float32);
    defer _ = mlx.mlx_array_free(q);
    const p_data = [_]f32{ 0.5, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0 };
    const p_shape = [_]c_int{ 1, 2, 4 };
    const p = mlx.mlx_array_new_data(&p_data, &p_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(p);
    const d_data: i32 = 3;
    const d_shape = [_]c_int{1};
    const d = mlx.mlx_array_new_data(&d_data, &d_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(d);
    const drafts = [_]mlx.mlx_array{d};

    var runs: [2][16][3]i32 = undefined;
    for (&runs) |*run| {
        for (run, 0..) |*out, index| {
            const draft_params = SamplingParams{ .seed = 0xC0FFEE, .draw = @as(u64, @intCast(index)) * 2 };
            const lazy_draft = try sampleFromProbsLazy(q, draft_params, s);
            defer _ = mlx.mlx_array_free(lazy_draft);
            try mlx.check(mlx.mlx_array_eval(lazy_draft));
            try mlx.check(mlx.mlx_array_item_int32(&out[0], lazy_draft));

            const correction_params = SamplingParams{ .seed = 0xC0FFEE, .draw = draft_params.draw + 1 };
            var graph = try Generator.mtpBatchedAcceptGraph(p, &drafts, null, 1, correction_params, s);
            defer graph.deinit();
            try mlx.check(mlx.mlx_array_eval(graph.corr_samples));
            const corrected = mlx.mlx_array_data_int32(graph.corr_samples) orelse return error.MlxArrayDataNull;
            out[1] = corrected[0];
            out[2] = corrected[1];
        }

        // A seeded replay must not depend on unrelated requests consuming the
        // process-global MLX RNG between runs.
        const unrelated = try sampleFromProbsLazy(q, SamplingParams{}, s);
        defer _ = mlx.mlx_array_free(unrelated);
        try mlx.check(mlx.mlx_array_eval(unrelated));
    }
    try testing.expectEqualSlices([3]i32, &runs[0], &runs[1]);
    var all_same = true;
    for (runs[0][1..]) |draw| all_same = all_same and std.mem.eql(i32, &draw, &runs[0][0]);
    try testing.expect(!all_same);
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

test "typical + greedy tail: an argmax tail is accepted when it clears the entropy floor" {
    // Depth 0 is the sampled token 1 (p 0.3, floor 0.2 * exp(-H) = 0.071); depth 1
    // is the head's argmax, judged against row 1's floor of 0.130. The q rows the
    // chain carries (a density, then a one-hot) do not move a typical decision.
    const s = mlx.gpuStream();
    const p_data = [_]f32{ 0.5, 0.3, 0.2, 0.0, 0.02, 0.9, 0.04, 0.04, 0.0, 0.0, 0.0, 1.0 };
    const probs = mlx.mlx_array_new_data(&p_data, &[_]c_int{ 1, 3, 4 }, 3, .float32);
    defer _ = mlx.mlx_array_free(probs);
    const q0_data = [_]f32{ 0.3, 0.5, 0.2, 0.0 };
    const q0 = mlx.mlx_array_new_data(&q0_data, &[_]c_int{ 1, 4 }, 2, .float32);
    defer _ = mlx.mlx_array_free(q0);

    const Case = struct { tail: i32, accepted: u32 };
    // The target's own mode clears the floor; a head argmax the target puts 0.04 on does not.
    for ([_]Case{ .{ .tail = 1, .accepted = 2 }, .{ .tail = 2, .accepted = 1 } }) |case| {
        const d0_id: i32 = 1;
        const d0 = mlx.mlx_array_new_data(&d0_id, &[_]c_int{1}, 1, .int32);
        defer _ = mlx.mlx_array_free(d0);
        const d1 = mlx.mlx_array_new_data(&case.tail, &[_]c_int{1}, 1, .int32);
        defer _ = mlx.mlx_array_free(d1);
        const q1 = try lazyOneHotRow(d1, 4, s);
        defer _ = mlx.mlx_array_free(q1);
        var g = try Generator.mtpBatchedTypicalGraph(probs, &.{ d0, d1 }, &.{ q0, q1 }, 2, 0.2, .{}, s);
        defer g.deinit();
        try mlx.check(mlx.mlx_array_eval(g.accept_p));
        try mlx.check(mlx.mlx_array_eval(g.accept_q));
        const p = mlx.mlx_array_data_float32(g.accept_p) orelse return error.InvalidDtype;
        const floor = mlx.mlx_array_data_float32(g.accept_q) orelse return error.InvalidDtype;
        try testing.expectApproxEqAbs(@as(f32, 0.0714), floor[0], 1e-3);
        try testing.expectApproxEqAbs(@as(f32, 0.1300), floor[1], 1e-3);
        var rng = std.Random.DefaultPrng.init(1);
        try testing.expectEqual(case.accepted, mtp_acceptance.typicalPrefix(p[0..2], floor[0..2], null, &rng));
    }
}

test "greedy tail: each depth's verify q is the density its draft came from" {
    // Depth 0 of a sampled request draws from the shortlist; depth 1 takes its
    // argmax. The rows the chain hands the verify must be exactly those
    // proposals: a shortlist density with the draft in its support, then a
    // one-hot at the argmax, so accept_q = [q0(d0), 1] and a reject at depth 1
    // can never re-draw d1.
    if (mlx.noGpuBackend()) return;
    const s = mlx.gpuStream();
    const allocator = testing.allocator;
    const V: c_int = 4096;
    const N: usize = 32;
    var ids: [N]u32 = undefined;
    var logits: [N]f32 = undefined;
    for (0..N) |i| {
        ids[i] = @intCast(37 + i * 101);
        logits[i] = @as(f32, @floatFromInt(i)) * 0.25;
    }
    var sl = mtp_mod.Shortlist{
        .cands = mlx.mlx_array_new_data(&ids, &[_]c_int{@intCast(N)}, 1, .uint32),
        .exact = mlx.mlx_array_new_data(&logits, &[_]c_int{ 1, 1, @intCast(N) }, 3, .float32),
        .rows = V,
    };
    defer sl.deinit();

    const target = SamplingParams{ .temperature = 1.0 };
    var props: [2]StepProposal = undefined;
    var built: usize = 0;
    defer for (props[0..built]) |p| {
        _ = mlx.mlx_array_free(p.id);
        _ = mlx.mlx_array_free(p.q);
    };
    for (&props, 0..) |*p, d| {
        const greedy = Generator.mtpDraftGreedyAt(target, true, @intCast(d));
        p.* = try shortlistStepProposal(sl, Generator.mtpDraftSamplingFor(target, greedy, Generator.MTP_DRAFT_TEMP), s);
        built += 1;
        // Every depth drafts the same `[1]` id shape: the verify concatenates them.
        try testing.expectEqualSlices(c_int, &.{1}, mlx.getShape(p.id));
    }

    const q0 = try proposalTestRead(allocator, props[0].q, @intCast(V), s);
    defer allocator.free(q0);
    const q1 = try proposalTestRead(allocator, props[1].q, @intCast(V), s);
    defer allocator.free(q1);
    const id_reads = [_][]f32{
        try proposalTestRead(allocator, props[0].id, 1, s),
        try proposalTestRead(allocator, props[1].id, 1, s),
    };
    defer for (id_reads) |r| allocator.free(r);
    const id0: usize = @intFromFloat(id_reads[0][0]);
    const id1: usize = @intFromFloat(id_reads[1][0]);
    var sum0: f64 = 0;
    var support0: usize = 0;
    for (q0) |v| {
        sum0 += v;
        if (v > 0) support0 += 1;
    }
    try testing.expect(@abs(sum0 - 1.0) < 1e-4 and support0 > 1 and q0[id0] > 0);
    try testing.expectEqual(@as(usize, ids[N - 1]), id1);
    for (q1, 0..) |v, i| try testing.expectEqual(@as(f32, if (i == id1) 1 else 0), v);

    // Target rows: row 1 splits its mass between the argmax draft and one other id.
    const Vu: usize = @intCast(V);
    const other: usize = ids[0];
    const p_data = try allocator.alloc(f32, 3 * Vu);
    defer allocator.free(p_data);
    @memset(p_data, 0);
    @memcpy(p_data[0..Vu], q0);
    p_data[Vu + id1] = 0.5;
    p_data[Vu + other] = 0.5;
    p_data[2 * Vu + other] = 1.0;
    const probs = mlx.mlx_array_new_data(p_data.ptr, &[_]c_int{ 1, 3, V }, 3, .float32);
    defer _ = mlx.mlx_array_free(probs);
    var g = try Generator.mtpBatchedAcceptGraph(probs, &.{ props[0].id, props[1].id }, &.{ props[0].q, props[1].q }, 2, .{}, s);
    defer g.deinit();
    try mlx.check(mlx.mlx_array_eval(g.accept_q));
    try mlx.check(mlx.mlx_array_eval(g.accept_p));
    try mlx.check(mlx.mlx_array_eval(g.corr_samples));
    const aq = mlx.mlx_array_data_float32(g.accept_q) orelse return error.InvalidDtype;
    const ap = mlx.mlx_array_data_float32(g.accept_p) orelse return error.InvalidDtype;
    const corr = mlx.mlx_array_data_int32(g.corr_samples) orelse return error.InvalidDtype;
    try testing.expectApproxEqAbs(q0[id0], aq[0], 1e-6);
    try testing.expectEqual(@as(f32, 1.0), aq[1]);
    try testing.expectEqual(@as(f32, 0.5), ap[1]);
    try testing.expectEqual(@as(i32, @intCast(other)), corr[1]);
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

test "MTP TokenV3 batched graph uses full pi for sampled and one-hot drafts" {
    const s = mlx.gpuStream();
    const p_data = [_]f32{ 0.8, 0.19, 0.01, 0, 0, 0, 0, 1 };
    const p_shape = [_]c_int{ 1, 2, 4 };
    const probs = mlx.mlx_array_new_data(&p_data, &p_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(probs);
    const id: i32 = 2;
    const id_shape = [_]c_int{1};
    const draft = mlx.mlx_array_new_data(&id, &id_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(draft);
    const drafts = [_]mlx.mlx_array{draft};
    const q_data = [_]f32{ 0.1, 0.2, 0.7, 0 };
    const q_shape = [_]c_int{ 1, 4 };
    const q = mlx.mlx_array_new_data(&q_data, &q_shape, 2, .float32);
    defer _ = mlx.mlx_array_free(q);
    const qs = [_]mlx.mlx_array{q};

    var sampled = try Generator.mtpBatchedTokenV3Graph(probs, &drafts, &qs, 1, 0.95, .{}, s);
    defer sampled.deinit();
    try mlx.check(mlx.mlx_array_eval(sampled.accept_p));
    try mlx.check(mlx.mlx_array_eval(sampled.accept_q));
    try mlx.check(mlx.mlx_array_eval(sampled.accept_defer));
    const pi_draft = mlx.mlx_array_data_float32(sampled.accept_p) orelse return error.InvalidDtype;
    const q_draft = mlx.mlx_array_data_float32(sampled.accept_q) orelse return error.InvalidDtype;
    const deferred = mlx.mlx_array_data_bool(sampled.accept_defer) orelse return error.InvalidDtype;
    try testing.expectApproxEqAbs(@as(f32, 0.007), pi_draft[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.7), q_draft[0], 1e-6);
    try testing.expect(deferred[0]);
    try mlx.check(mlx.mlx_array_eval(sampled.corr_samples));
    const corr = mlx.mlx_array_data_int32(sampled.corr_samples) orelse return error.InvalidDtype;
    try testing.expect(corr[0] == 0 or corr[0] == 1);
    try testing.expectEqual(@as(i32, 3), corr[1]);

    var onehot = try Generator.mtpBatchedTokenV3Graph(probs, &drafts, null, 1, 0.95, .{}, s);
    defer onehot.deinit();
    try mlx.check(mlx.mlx_array_eval(onehot.accept_p));
    const onehot_pi = mlx.mlx_array_data_float32(onehot.accept_p) orelse return error.InvalidDtype;
    try testing.expectApproxEqAbs(@as(f32, 0.01), onehot_pi[0], 1e-6);
    try testing.expect(onehot.accept_q.ctx == null);
}

test "MTP TokenV3 full Qwen3.8 vocabulary graph stays finite at depth three" {
    const s = mlx.gpuStream();
    const vocab: usize = 248_320;
    const p_data = try testing.allocator.alloc(f32, 4 * vocab);
    defer testing.allocator.free(p_data);
    @memset(p_data, 0);
    for (0..3) |row| {
        p_data[row * vocab] = 0.8;
        p_data[row * vocab + 1] = 0.19;
        p_data[row * vocab + 2] = 0.01;
    }
    p_data[3 * vocab + 3] = 1.0;
    const p_shape = [_]c_int{ 1, 4, @intCast(vocab) };
    const probs = mlx.mlx_array_new_data(p_data.ptr, &p_shape, 3, .float32);
    defer _ = mlx.mlx_array_free(probs);
    const q_data = try testing.allocator.alloc(f32, vocab);
    defer testing.allocator.free(q_data);
    @memset(q_data, 0);
    q_data[0] = 0.1;
    q_data[1] = 0.2;
    q_data[2] = 0.7;
    const q_shape = [_]c_int{ 1, @intCast(vocab) };
    const q = mlx.mlx_array_new_data(q_data.ptr, &q_shape, 2, .float32);
    defer _ = mlx.mlx_array_free(q);
    const draft_id: i32 = 2;
    const id_shape = [_]c_int{1};
    const draft = mlx.mlx_array_new_data(&draft_id, &id_shape, 1, .int32);
    defer _ = mlx.mlx_array_free(draft);
    const drafts = [_]mlx.mlx_array{ draft, draft, draft };
    const qs = [_]mlx.mlx_array{ q, q, q };
    var g = try Generator.mtpBatchedTokenV3Graph(probs, &drafts, &qs, 3, 0.95, .{}, s);
    defer g.deinit();
    try mlx.check(mlx.mlx_array_eval(g.accept_p));
    try mlx.check(mlx.mlx_array_eval(g.accept_defer));
    try mlx.check(mlx.mlx_array_eval(g.corr_samples));
    const pi_draft = mlx.mlx_array_data_float32(g.accept_p) orelse return error.InvalidDtype;
    const deferred = mlx.mlx_array_data_bool(g.accept_defer) orelse return error.InvalidDtype;
    const corr = mlx.mlx_array_data_int32(g.corr_samples) orelse return error.InvalidDtype;
    for (0..3) |k| {
        try testing.expectApproxEqAbs(@as(f32, 0.007), pi_draft[k], 1e-5);
        try testing.expect(deferred[k]);
        try testing.expect(corr[k] == 0 or corr[k] == 1);
    }
    try testing.expectEqual(@as(i32, 3), corr[3]);
}

test "mtpEvPlanFor: cap 1 is a plain depth-1 round" {
    const costs = Generator.MTP_EV_DEFAULT_COSTS;
    const a = [_]f32{ 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9 };
    const plan = Generator.mtpEvPlanFor(&a, 1, costs, 8);
    try testing.expectEqual(@as(u32, 1), plan.m_lo);
    try testing.expectEqual(@as(u32, 1), plan.m_hi);
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
    const st = Generator.MtpHistStash{
        .ids = .{ .ctx = null },
        .hidden = .{ .ctx = null },
        .n = 3, // t1 + 2 accepted drafts
        .off0 = 40,
    };
    // cache_step (46 = 40 committed + 6 stale draft entries) must be ignored.
    try testing.expectEqual(@as(usize, 43), Generator.mtpRoundOff0(st, 46));
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

test "MtpTrace: per-phase accumulation, round averaging, log cadence, reset" {
    var t = Generator.MtpTrace{};
    // 2 rounds: draft 2ms+4ms, eval 10ms+30ms.
    t.add(.draft, 2_000_000);
    t.add(.eval, 10_000_000);
    try testing.expect(!t.endRound(3, 2, false));
    t.add(.draft, 4_000_000);
    t.add(.eval, 30_000_000);
    try testing.expect(!t.endRound(7, 6, true));
    try testing.expectApproxEqAbs(@as(f64, 3.0), t.avgMs(.draft), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 20.0), t.avgMs(.eval), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 0.0), t.avgMs(.sync), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 23.0), t.totalAvgMs(), 1e-9);
    try testing.expectEqual(@as(u64, 10), t.drafted);
    try testing.expectEqual(@as(u64, 8), t.accepted);
    try testing.expectEqual(@as(u32, 1), t.extended);
    // Log line falls due exactly at LOG_EVERY rounds.
    var i: u32 = 2;
    while (i < Generator.MtpTrace.LOG_EVERY - 1) : (i += 1) {
        try testing.expect(!t.endRound(1, 1, false));
    }
    try testing.expect(t.endRound(1, 1, false));
    t.reset();
    try testing.expectEqual(@as(u32, 0), t.rounds);
    try testing.expectApproxEqAbs(@as(f64, 0.0), t.avgMs(.draft), 1e-9);
}

test "the [mtp-trace] phase summary stays contiguous for its log consumers" {
    const line = try std.fmt.allocPrint(testing.allocator, Generator.MtpTrace.LINE_FMT, .{
        @as(u32, 32), 0.45, 0.0, 0.0,         4.09, 56.86, 0.0,  0.0,   0.03, 0.0,  0.07, 61.57,
        2.0,          1.4,  0.0, "0.91/0.80", 4.09, 7.13,  6.40, 52.28, 0.94, 0.44,
    });
    defer testing.allocator.free(line);
    try testing.expect(std.mem.indexOf(u8, line, "total=61.57 | m_avg=") != null);
    const acc_at = std.mem.indexOf(u8, line, "acc_idx=").?;
    const sub_at = std.mem.indexOf(u8, line, "| sub_ms ").?;
    try testing.expect(acc_at < sub_at);
}

test "MtpTrace sub-laps attribute inside a phase without changing the round total" {
    var t = Generator.MtpTrace{};
    t.add(.verify, 20_000_000);
    t.addSub(.build, 15_000_000);
    t.addSub(.ple, 3_000_000);
    t.addSub(.ple_sync, 2_000_000);
    t.add(.corr, 6_000_000);
    t.addSub(.dispatch, 2_000_000);
    t.add(.eval, 60_000_000);
    t.addSub(.wait, 58_000_000);
    t.add(.gap, 12_000_000);
    t.addSub(.chain, 2_000_000);
    try testing.expect(!t.endRound(2, 2, false));
    try testing.expectApproxEqAbs(@as(f64, 98.0), t.totalAvgMs(), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 20.0), t.avgMs(.verify), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 20.0), t.subAvgMs(.build) + t.subAvgMs(.ple) + t.subAvgMs(.ple_sync), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 58.0), t.subAvgMs(.wait), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 2.0), t.subAvgMs(.chain), 1e-9);
    t.reset();
    try testing.expectApproxEqAbs(@as(f64, 0.0), t.subAvgMs(.build), 1e-9);
    try testing.expectApproxEqAbs(@as(f64, 0.0), t.totalAvgMs(), 1e-9);
}

test "buildPaddedBatch pads to max length with zeros and records lengths" {
    const seqs = [_][]const u32{
        &[_]u32{ 101, 7592, 102 },
        &[_]u32{ 101, 102 },
    };
    var pb = try buildPaddedBatch(testing.allocator, &seqs);
    defer pb.deinit(testing.allocator);
    try testing.expectEqual(@as(usize, 3), pb.max_len);
    try testing.expectEqual(@as(usize, 2), pb.lengths.len);
    try testing.expectEqual(@as(usize, 3), pb.lengths[0]);
    try testing.expectEqual(@as(usize, 2), pb.lengths[1]);
    const expected = [_]i32{ 101, 7592, 102, 101, 102, 0 };
    try testing.expectEqualSlices(i32, &expected, pb.ids);
}

test "buildKeyPadMask is additive zero on real keys, -inf on padding" {
    const s = mlx.gpuStream();
    const lengths = [_]usize{ 3, 1 };
    const mask = try buildKeyPadMask(testing.allocator, &lengths, 3, s);
    defer _ = mlx.mlx_array_free(mask);
    try testing.expectEqualSlices(c_int, &[_]c_int{ 2, 1, 1, 3 }, mlx.getShape(mask));
    var f32_mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(f32_mask);
    try mlx.check(mlx.mlx_astype(&f32_mask, mask, .float32, s));
    try mlx.check(mlx.mlx_array_eval(f32_mask));
    const data = mlx.mlx_array_data_float32(f32_mask).?;
    // Batch row 0: all three keys real.
    try testing.expectEqual(@as(f32, 0), data[0]);
    try testing.expectEqual(@as(f32, 0), data[1]);
    try testing.expectEqual(@as(f32, 0), data[2]);
    // Batch row 1: one real key, two padded.
    try testing.expectEqual(@as(f32, 0), data[3]);
    try testing.expect(std.math.isInf(data[4]) and data[4] < 0);
    try testing.expect(std.math.isInf(data[5]) and data[5] < 0);
}

test "maskedMeanPoolNormalize excludes padded positions and unit-normalizes" {
    const s = mlx.gpuStream();
    // hidden [2, 3, 2]; row 0 has 2 real positions (pad slot holds garbage
    // that must not leak into the pool), row 1 has 3.
    const data = [_]f32{
        1, 0, 3, 4, 100, 100,
        0, 2, 0, 4, 0,   6,
    };
    const shape = [_]c_int{ 2, 3, 2 };
    const hidden = mlx.mlx_array_new_data(&data, &shape, 3, .float32);
    defer _ = mlx.mlx_array_free(hidden);
    const lengths = [_]usize{ 2, 3 };
    const rows = try maskedMeanPoolNormalize(testing.allocator, hidden, &lengths, s);
    defer {
        for (rows) |r| testing.allocator.free(r);
        testing.allocator.free(rows);
    }
    // Row 0: mean of (1,0),(3,4) = (2,2) → L2-normalized (1/√2, 1/√2).
    try testing.expectApproxEqAbs(@as(f32, 0.70710678), rows[0][0], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 0.70710678), rows[0][1], 1e-4);
    // Row 1: mean of (0,2),(0,4),(0,6) = (0,4) → normalized (0, 1).
    try testing.expectApproxEqAbs(@as(f32, 0.0), rows[1][0], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 1.0), rows[1][1], 1e-4);
}

test "embedSubBatchEnd: item cap, padded-footprint budget, oversize singleton, order" {
    const short = [_]u32{ 1, 2 };
    const long: [600]u32 = @splat(0);
    // Item cap alone (all short): full batch.
    const all_short = [_][]const u32{ &short, &short, &short };
    try testing.expectEqual(@as(usize, 3), embedSubBatchEnd(&all_short, 0, 64, 64 * 512));
    try testing.expectEqual(@as(usize, 2), embedSubBatchEnd(&all_short, 0, 2, 64 * 512));

    // One long input caps how many rows pad to its length: 600-token rows fit
    // budget/600 = 54 per sub-batch at the default budget, not 64.
    const many_long: [60][]const u32 = @splat(&long);
    const end = embedSubBatchEnd(&many_long, 0, 64, 64 * 512);
    try testing.expectEqual(@as(usize, (64 * 512) / 600), end);

    // A mixed batch stops BEFORE a long input would inflate the padded
    // footprint of everything before it.
    const mixed = [_][]const u32{ &short, &short, &long };
    const mixed_end = embedSubBatchEnd(&mixed, 0, 64, 4);
    try testing.expectEqual(@as(usize, 2), mixed_end);

    // An input larger than the whole budget still runs — alone.
    const huge: [40000]u32 = @splat(0);
    const with_huge = [_][]const u32{ &huge, &short };
    try testing.expectEqual(@as(usize, 1), embedSubBatchEnd(&with_huge, 0, 64, 64 * 512));
    // ...and the next sub-batch starts right after it (order preserved).
    try testing.expectEqual(@as(usize, 2), embedSubBatchEnd(&with_huge, 1, 64, 64 * 512));
}

test "gatherTokenPool: cls takes position 0, last_token takes the last real position" {
    const s = mlx.gpuStream();
    // hidden [2, 3, 2]; row 0 has 2 real positions (position 2 is pad garbage
    // that must never be selected), row 1 has 3.
    const data = [_]f32{
        1, 0, 3, 4, 100, 100,
        0, 2, 0, 4, 0,   6,
    };
    const shape = [_]c_int{ 2, 3, 2 };
    const hidden = mlx.mlx_array_new_data(&data, &shape, 3, .float32);
    defer _ = mlx.mlx_array_free(hidden);
    const lengths = [_]usize{ 2, 3 };

    const cls = try gatherTokenPool(testing.allocator, hidden, &lengths, .cls, s);
    defer _ = mlx.mlx_array_free(cls);
    try testing.expectEqualSlices(c_int, &[_]c_int{ 2, 2 }, mlx.getShape(cls));
    try mlx.check(mlx.mlx_array_eval(cls));
    const cls_data = mlx.mlx_array_data_float32(cls).?;
    try testing.expectEqual(@as(f32, 1), cls_data[0]); // hidden[0,0,:]
    try testing.expectEqual(@as(f32, 0), cls_data[1]);
    try testing.expectEqual(@as(f32, 0), cls_data[2]); // hidden[1,0,:]
    try testing.expectEqual(@as(f32, 2), cls_data[3]);

    const last = try gatherTokenPool(testing.allocator, hidden, &lengths, .last_token, s);
    defer _ = mlx.mlx_array_free(last);
    try mlx.check(mlx.mlx_array_eval(last));
    const last_data = mlx.mlx_array_data_float32(last).?;
    try testing.expectEqual(@as(f32, 3), last_data[0]); // hidden[0,1,:] — NOT the pad slot
    try testing.expectEqual(@as(f32, 4), last_data[1]);
    try testing.expectEqual(@as(f32, 0), last_data[2]); // hidden[1,2,:]
    try testing.expectEqual(@as(f32, 6), last_data[3]);
}

test "gatherTokenPool: mean mode is not a gather — callers must dispatch it to maskedMeanPool" {
    const s = mlx.gpuStream();
    const data = [_]f32{ 1, 2 };
    const shape = [_]c_int{ 1, 1, 2 };
    const hidden = mlx.mlx_array_new_data(&data, &shape, 3, .float32);
    defer _ = mlx.mlx_array_free(hidden);
    const lengths = [_]usize{1};
    try testing.expectError(error.InvalidPoolingMode, gatherTokenPool(testing.allocator, hidden, &lengths, .mean, s));
}

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

// ── Allocator-cache clear cadence (issue #110) ───────────────────────────────

test "clear cadence survives variable spec strides" {
    // A spec-decode round emits `1 + accepted` tokens, so `step` advances by a
    // VARIABLE amount and `step % 256 == 0` can walk clean over every multiple
    // — silently, because a decode path that never clears is output-identical
    // to one that does. Interval arithmetic against the last clear cannot be
    // stepped over.
    for ([_]u32{ 1, 3, 5, 9 }) |stride| {
        var step: u32 = 0;
        var last_clear: u32 = 0;
        var clears: usize = 0;
        var max_gap: u32 = 0;
        while (step < 4096) {
            step += stride;
            if (shouldClearAllocatorCache(step, last_clear, CACHE_CLEAR_INTERVAL)) {
                const gap = step - last_clear;
                if (gap > max_gap) max_gap = gap;
                last_clear = step;
                clears += 1;
            }
        }
        try testing.expect(clears >= 15);
        // A stride can overshoot the interval by at most stride-1 tokens.
        try testing.expect(max_gap <= CACHE_CLEAR_INTERVAL + stride);
    }
}

test "the allocator pool is returned at the prefill/decode handover, on archs that reserve their KV" {
    // The per-chunk clear runs BEFORE the last chunks' transient is freed, so it
    // parks in MLX's pool up to the cap and the first decode tick allocates on top
    // of it (measured 8.1 GB parked at the first tick of a 393k prefill; a cold
    // 524k died 0.5 GB over the plan ceiling on exactly that tick). One clear at the
    // handover returns it. Every other arch keeps its previous call pattern.
    const t = testing;
    var qwen4 = model_mod.ModelConfig{ .model_type = "qwen4_exp" };
    try t.expect(clearsPoolAtPrefillEnd(&qwen4));
    // MiMo reserves its KV up front too: a 256k request parked 3.2 GiB there, and the next admission read it as spent.
    var mimo = model_mod.ModelConfig{ .model_type = "mimo_v2", .has_sliding_window = true, .sliding_window = 128, .head_dim = 192 };
    try t.expect(clearsPoolAtPrefillEnd(&mimo));
    for ([_][]const u8{
        "qwen3_5",
        "qwen3_5_moe",
        "qwen3_next",
        "lfm2",
        "nemotron_h",
        "bailing_hybrid",
        "llama",
        "gemma4",
        "deepseek_v4",
        "muse_glimmer",
    }) |mt| {
        var cfg = model_mod.ModelConfig{ .model_type = mt };
        try t.expect(!clearsPoolAtPrefillEnd(&cfg));
    }
}

test "no decode path advances `step` outside advanceStep" {
    // Class guard for #110. `Generator.step` is the clear cadence's clock, so a
    // path that bumps it by hand is a path that strands its round's transients
    // in MLX's buffer pool forever — which is exactly what `nextDrafter`,
    // `nextMtp` and `scheduler.runBatchedDecodeTick` did, and the `-mtp`
    // checkpoints default onto one of those. A new decode path added later must
    // route through `advanceStep` or fail here.
    //
    // The needles are concatenated so this test's OWN source doesn't match them.
    const needle = ".step" ++ " +=";
    const allowed = "self.step" ++ " += n";
    const sources = [_]struct { name: []const u8, src: []const u8 }{
        .{ .name = "generate.zig", .src = @embedFile("generate.zig") },
        .{ .name = "scheduler.zig", .src = @embedFile("scheduler.zig") },
    };
    var total: usize = 0;
    for (sources) |file| {
        var it = std.mem.splitScalar(u8, file.src, '\n');
        var lineno: usize = 0;
        while (it.next()) |raw| {
            lineno += 1;
            // Strip trailing line comments: several of them spell out
            // `cache.step += 1` while describing the KV cache's own counter.
            const line = if (std.mem.indexOf(u8, raw, "//")) |c| raw[0..c] else raw;
            if (std.mem.indexOf(u8, line, needle) == null) continue;
            total += 1;
            if (std.mem.indexOf(u8, line, allowed) == null) {
                std.debug.print("{s}:{d}: raw step advance outside advanceStep: {s}\n", .{
                    file.name, lineno, std.mem.trim(u8, line, " "),
                });
                return error.RawStepAdvance;
            }
        }
    }
    // Exactly one: the assignment inside `advanceStep`. Zero means the field was
    // renamed and this guard went vacuous — update it with the rename.
    try testing.expectEqual(@as(usize, 1), total);

    // Same discipline for the clock itself: every clear the Generator makes records
    // it, so the decode cadence measures from the last ACTUAL clear. Two writers —
    // `advanceStep` and the prefill/decode handover — and a third that forgot the
    // write would leave the cadence firing against a stale clock.
    const clock = "last_cache_clear_step" ++ " =";
    var writers: usize = 0;
    var clock_it = std.mem.splitScalar(u8, sources[0].src, '\n');
    while (clock_it.next()) |raw| {
        const line = if (std.mem.indexOf(u8, raw, "//")) |c| raw[0..c] else raw;
        if (std.mem.indexOf(u8, line, clock) != null) writers += 1;
    }
    try testing.expectEqual(@as(usize, 2), writers);
}

/// The last global layer's key buffer: its data address and its row capacity.
fn globalKeysBuffer(xfm: *Transformer) !struct { ptr: usize, cap: c_int } {
    const e = &xfm.cache.entries[try lastGlobalKvLayer(xfm)];
    try mlx.check(mlx.mlx_array_eval(e.keys));
    const ptr: usize = if (mlx.mlx_array_dtype(e.keys) == .uint32)
        @intFromPtr(mlx.mlx_array_data_uint32(e.keys).?)
    else
        @intFromPtr(mlx.mlx_array_data_bfloat16(e.keys).?);
    return .{ .ptr = ptr, .cap = mlx.getShape(e.keys)[2] };
}

fn lastGlobalKvLayer(xfm: *Transformer) !usize {
    var layer = xfm.cache.entries.len;
    while (layer > 0) {
        layer -= 1;
        if (xfm.config.isGlobalLayer(@intCast(layer)) and xfm.cache.entries[layer].initialized) return layer;
    }
    return error.NoGlobalLayer;
}

/// Do two caches hold the same first `rows` rows in every stored buffer of `layer`?
fn kvRowsEqual(x: *Transformer, y: *Transformer, layer: usize, rows: usize) !bool {
    const ex = &x.cache.entries[layer];
    const ey = &y.cache.entries[layer];
    const pairs = [_][2]mlx.mlx_array{
        .{ ex.keys, ey.keys },                   .{ ex.values, ey.values },
        .{ ex.keys_scales, ey.keys_scales },     .{ ex.keys_biases, ey.keys_biases },
        .{ ex.values_scales, ey.values_scales }, .{ ex.values_biases, ey.values_biases },
    };
    const s = x.s;
    for (pairs) |p| {
        if (p[0].ctx == null or mlx.mlx_array_ndim(p[0]) != 4) continue;
        var parts: [2]mlx.mlx_array = .{ .{}, .{} };
        defer for (parts) |a| {
            if (a.ctx != null) _ = mlx.mlx_array_free(a);
        };
        for (p, &parts) |buf, *part| {
            const sh = mlx.getShape(buf);
            part.* = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(part, buf, &[_]c_int{ 0, 0, 0, 0 }, 4, &[_]c_int{ sh[0], sh[1], @intCast(rows), sh[3] }, 4, &[_]c_int{ 1, 1, 1, 1 }, 4, s));
        }
        var eq = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(eq);
        try mlx.check(mlx.mlx_array_equal(&eq, parts[0], parts[1], false, s));
        try mlx.check(mlx.mlx_array_eval(eq));
        var same = false;
        try mlx.check(mlx.mlx_array_item_bool(&same, eq));
        if (!same) return false;
    }
    return true;
}

fn pldTestTransformer(io: std.Io, config: model_mod.ModelConfig, weights: anytype, kv: transformer_mod.KVQuantConfig) !Transformer {
    var xfm = try Transformer.init(io, testing.allocator, config, weights);
    errdefer xfm.deinit();
    try xfm.cache.reinit(config.num_hidden_layers, kv);
    xfm.cache.setSwaRing(config.sliding_window);
    return xfm;
}

fn arraysBitEqual(x: mlx.mlx_array, y: mlx.mlx_array, s: mlx.mlx_stream) !bool {
    var eq = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(eq);
    try mlx.check(mlx.mlx_array_equal(&eq, x, y, false, s));
    try mlx.check(mlx.mlx_array_eval(eq));
    var same = false;
    try mlx.check(mlx.mlx_array_item_bool(&same, eq));
    return same;
}

test "mimo ring marks leave the prefill's logits, cache and greedy tokens unchanged (MIMO_V2_MODEL)" {
    const model_dir = std.c.getenv("MIMO_V2_MODEL") orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var config = try model_mod.parseConfig(io, a, std.mem.span(model_dir));
    defer if (config.ngram_table_path) |p| a.free(p);
    var weights = try model_mod.loadWeightsForConfig(io, a, std.mem.span(model_dir), &config, false);
    defer weights.deinit();
    try transformer_mod.stackMimoFixtureExperts(&weights, config, mlx.gpuStream());
    model_mod.resolveWeightPrefix(&config, &weights);
    var tok_dummy: Tokenizer = undefined;
    const greedy = SamplingParams{ .temperature = 0.0 };
    // 512-row chunks: the first mark falls in a chunk that does not compact the ring, the others in ones that do.
    var prompt: [1500]u32 = undefined;
    for (&prompt, 0..) |*v, i| v.* = @intCast((i * 7) % config.vocab_size);
    const at = [_]usize{ 300, 700, 1100 };
    for ([_]transformer_mod.KVQuantConfig{ transformer_mod.KVQuantConfig.dense, transformer_mod.KVQuantConfig.affine(8) }) |kv| {
        var plain = try pldTestTransformer(io, config, &weights, kv);
        defer plain.deinit();
        var marked = try pldTestTransformer(io, config, &weights, kv);
        defer marked.deinit();
        var marks: [at.len]transformer_mod.KVCacheSnapshot = undefined;
        var n: usize = 0;
        defer for (marks[0..n]) |*m| m.deinit();
        for (at) |p| {
            marks[n] = try marked.cache.ringMark(p);
            n += 1;
        }
        var plain_gen = try Generator.initWithOptions(io, a, &plain, &tok_dummy, &prompt, 24, greedy, &.{}, .{ .pinned_prefill_chunk = 512 });
        defer plain_gen.deinit(a);
        marked.cache.ring_marks = &marks;
        var marked_gen = try Generator.initWithOptions(io, a, &marked, &tok_dummy, &prompt, 24, greedy, &.{}, .{ .pinned_prefill_chunk = 512 });
        defer marked_gen.deinit(a);
        marked.cache.ring_marks = &.{};

        for (&marks) |*m| try testing.expect(marked.cache.ringMarkComplete(m));
        try testing.expect(try arraysBitEqual(plain_gen.pending_logits, marked_gen.pending_logits, plain.s));
        for (plain.cache.entries, 0..) |e, li| {
            try testing.expectEqual(e.offset, marked.cache.entries[li].offset);
            try testing.expect(try kvRowsEqual(&plain, &marked, li, e.offset));
        }
        for (0..16) |_| {
            const want = (try plain_gen.next(a)) orelse return error.ShortSerial;
            try testing.expectEqual(want, (try marked_gen.next(a)) orelse return error.ShortSerial);
        }
    }
}

test "mimo MTP prompt-lookup rounds decode like serial ticks, a lookup first round included (MIMO_V2_MODEL)" {
    const model_dir = std.c.getenv("MIMO_V2_MODEL") orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const dir = std.mem.span(model_dir);
    var config = try model_mod.parseConfig(io, a, dir);
    defer if (config.ngram_table_path) |p| a.free(p);
    var weights = try model_mod.loadWeightsForConfig(io, a, dir, &config, false);
    defer weights.deinit();
    try transformer_mod.stackMimoFixtureExperts(&weights, config, mlx.gpuStream());
    model_mod.resolveWeightPrefix(&config, &weights);
    var mtp_weights = try @import("mimo_source.zig").loadMtpWeights(io, a, dir);
    defer mtp_weights.deinit();
    var head = (try mimo_mtp.Head.load(a, mlx.gpuStream(), &config, &mtp_weights)) orelse return error.SkipZigTest;
    defer head.deinit();
    var tok = Tokenizer.initEmptyForTests(a, .byte_level_bpe);
    defer tok.deinit();
    const want: usize = 64;
    const kv = transformer_mod.KVQuantConfig.affine(8);
    var prompt: [168]u32 = undefined;
    for (&prompt, 0..) |*v, i| v.* = @intCast((i * 7) % config.vocab_size);

    const entries = [_]logit_bias.Bias{
        .{ .id = 1, .delta = 2, .scope = .reasoning },
        .{ .id = 2, .delta = -1, .scope = .answer },
        .{ .id = 3, .delta = -2, .scope = .all },
    };
    const penalties = [_]ThinkPenalty{
        .{},
        .{ .biases = &entries, .phase = .inside, .closer_id = std.math.maxInt(u32) },
        .{ .biases = &entries, .phase = .after },
    };
    for (penalties) |penalty| {
    const greedy = SamplingParams{ .temperature = 0, .think_penalty = penalty };
    var serial_xfm = try pldTestTransformer(io, config, &weights, kv);
    defer serial_xfm.deinit();
    var serial_gen = try Generator.initWithOptions(io, a, &serial_xfm, &tok, &prompt, want + 16, greedy, &.{}, .{ .skip_lazy_preforward = true });
    defer serial_gen.deinit(a);
    var serial: [want]u32 = undefined;
    for (&serial) |*t| t.* = (try serial_gen.next(a)) orelse return error.ShortSerial;

    var xfm = try pldTestTransformer(io, config, &weights, kv);
    defer xfm.deinit();
    head.target = &xfm;
    xfm.mtp_depth_free = Generator.mtpVerifyDraftsMax(true);
    var gen = try Generator.initWithOptions(io, a, &xfm, &tok, &prompt, want + 16, greedy, &.{}, .{
        .mtp_enabled = true,
        .mtp = .{ .mimo = &head },
        .model_has_mtp = true,
        .mtp_depth = Generator.mtpVerifyDraftsMax(true),
    });
    defer gen.deinit(a);
    // The index ends on the prompt, so the round opens on its tail: an earlier copy of that
    // tail runs into the serial continuation with one wrong token, so a lookup serves round 1,
    // lands and misses drafts, and MTP rounds take over where the copy stops agreeing.
    var idx = mtp_lookup.Index.init(a);
    const Lines = struct {
        pub fn hasNewline(_: @This(), _: u32) bool {
            return false;
        }
    };
    try idx.extend(prompt[prompt.len - 40 ..], Lines{});
    var copy = serial;
    copy[20] = (copy[20] + 1) % @as(u32, @intCast(config.vocab_size));
    try idx.extend(&copy, Lines{});
    try idx.extend(&prompt, Lines{});
    gen.mtp_lookup_idx = idx;
    gen.mtp_lookup_gen = 0;

    var got: [want]u32 = undefined;
    var n: usize = 0;
    var first_round_lookup = false;
    while (n < want) {
        const r = (try gen.nextMtp(a)) orelse break;
        defer a.free(r.tokens);
        if (n == 0) first_round_lookup = gen.mtp_lookup_rounds == 1;
        for (r.tokens) |t| {
            if (n == want) break;
            got[n] = t;
            n += 1;
        }
    }
    try testing.expectEqual(want, n);
    try testing.expect(first_round_lookup);
    try testing.expect(gen.mtp_attempted > 0);
    try testing.expectEqualSlices(u32, &serial, &got);
    const eval = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(eval);
    gen.mtp_cache.?.appendEvalArrays(eval);
    try mlx.check(mlx.mlx_eval(eval));
    }
}

fn evalLazyToken(lazy: mlx.mlx_array) !u32 {
    defer _ = mlx.mlx_array_free(lazy);
    try mlx.check(mlx.mlx_array_eval(lazy));
    var val: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&val, lazy));
    return @intCast(val);
}

test "fillSuppressMask: padding rows past the tokenizer's last defined id are suppressed" {
    // A config's `vocab_size` is a padded matrix dimension, not a vocabulary.
    // qwen4_exp declares 248320 while the tokenizer defines 248044 + 33 =
    // 248077 ids: 243 rows that decode to NOTHING, carry whatever the
    // initializer left, and were sampleable — a drawn one emits no text while
    // consuming a step and entering the KV. Same mask, second row class.
    var buf: [16]bool = undefined;
    fillSuppressMask(&buf, &[_]u32{3}, 12);
    for (buf, 0..) |v, i| {
        const want = (i == 3) or (i >= 12);
        try testing.expectEqual(want, v);
    }

    // No padding known (0) and a fully-defined vocab (== len) both add
    // nothing: a mask over the WHOLE vocab is -inf everywhere, and argmax
    // over that row returns id 0 — the all-false-mask class from the other
    // side. The reserved id must still be the only true.
    for ([_]usize{ 0, 16, 99 }) |defined| {
        fillSuppressMask(&buf, &[_]u32{3}, defined);
        for (buf, 0..) |v, i| try testing.expectEqual(i == 3, v);
    }
}

test "Tokenizer.definedVocabSize: highest defined id + 1, 0 when nothing is defined" {
    const allocator = testing.allocator;
    var tok = Tokenizer.initEmptyForTests(allocator, .byte_level_bpe);
    defer tok.deinit();
    try testing.expectEqual(@as(usize, 0), tok.definedVocabSize());
    // Added specials sit ABOVE the base vocab and are not contiguous with it,
    // so the answer is the maximum key, never the entry count.
    try tok.id_to_token.put(0, "a");
    try tok.id_to_token.put(1, "b");
    try tok.id_to_token.put(11, "<|special|>");
    try testing.expectEqual(@as(usize, 12), tok.definedVocabSize());
}

test "suppress_mask: a suppressed id is unreachable from both samplers, everything else stays" {
    // A collapsed distribution can rank `<|fim_hole|>` (a reserved FIM marker) in the
    // top-5 at every degenerate position, and greedy DREW it live. A reserved
    // marker in chat output is always a bug, so the sampler carries a
    // model-lifetime suppression mask. Contract pinned here: with the mask,
    // neither sampler can return a suppressed id on any of its paths (greedy
    // fast path, greedy general, categorical, sync sampleToken); with a null
    // mask the same logits DO pick it (red-on-revert); and logprobs keep
    // reporting the MODEL's raw distribution — the suppressed id still ranks
    // where the model put it, because the field reports the model while the
    // mask is sampling policy.
    if (mlx.noGpuBackend()) return;
    const s = mlx.gpuStream();
    const allocator = testing.allocator;

    const V: usize = 8;
    const mask = try buildSuppressMask(&[_]u32{3}, 0, V, s);
    defer _ = mlx.mlx_array_free(mask);

    // id 3 (suppressed) dominates, id 5 is the runner-up, the rest are so far
    // down that after masking the categorical underflows to exactly id 5.
    var data = [_]f32{ -1000, -1000, -1000, 100, -1000, 50, -1000, -1000 };

    // Greedy fast path: [1, 1, V].
    const shape3 = [_]c_int{ 1, 1, V };
    const logits3 = mlx.mlx_array_new_data(&data, &shape3, 3, .float32);
    defer _ = mlx.mlx_array_free(logits3);
    const masked_sp = SamplingParams{ .temperature = 0.0, .suppress_mask = mask };
    const open_sp = SamplingParams{ .temperature = 0.0 };
    try testing.expectEqual(@as(u32, 5), try evalLazyToken(sampleTokenLazy(logits3, masked_sp, s)));
    try testing.expectEqual(@as(u32, 3), try evalLazyToken(sampleTokenLazy(logits3, open_sp, s)));

    // Categorical path at temp 1.0 (exercises the seq_len==1 reshape arm and
    // the sampler proper; the masked lane's probability is exactly 0 and the
    // -1000 lanes underflow, so the draw is deterministic).
    const masked_t1 = SamplingParams{ .temperature = 1.0, .suppress_mask = mask };
    try testing.expectEqual(@as(u32, 5), try evalLazyToken(sampleTokenLazy(logits3, masked_t1, s)));

    // Sync sampler (logprobs/penalty path): same policy, and the reported
    // distribution stays the model's own — rank 1 is the SUPPRESSED id.
    const r = try sampleToken(allocator, logits3, masked_sp, null, 2, null, s);
    const lp = r.logprob_result orelse return error.NoLogprobs;
    defer allocator.free(lp.top_logprobs);
    try testing.expectEqual(@as(u32, 5), r.token_id);
    try testing.expectEqual(@as(u32, 3), lp.top_logprobs[0].token_id);

    // Stochastic-verify filters: a suppressed draft's acceptance probability
    // must be exactly 0 (always rejected), and the residual it corrects from
    // can never re-draw it.
    const probs = try probsAtLastPos(logits3, masked_t1, s);
    defer _ = mlx.mlx_array_free(probs);
    try testing.expectEqual(@as(f32, 0.0), try probAt(probs, 3, s));
    try testing.expect(try probAt(probs, 5, s) > 0.99);
}

test "computeLogprobs: bf16 logits report the exact log-probability" {
    if (mlx.noGpuBackend()) return;
    const s = mlx.gpuStream();
    const raw = [_]f32{ 5, 5, 5, -20 };
    const a32 = mlx.mlx_array_new_data(&raw, &[_]c_int{ 1, 4 }, 2, .float32);
    defer _ = mlx.mlx_array_free(a32);
    var a16 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(a16);
    try mlx.check(mlx.mlx_astype(&a16, a32, .bfloat16, s));
    const r = try computeLogprobs(testing.allocator, a16, 3, 4, s);
    defer testing.allocator.free(r.top_logprobs);
    // -26.0986; the bf16 grid reads -26.125 there.
    try testing.expectApproxEqAbs(-20.0 - (5.0 + @log(@as(f32, 3.0))), r.token_logprob, 1e-3);
}

test "computeLogprobs: rank 1 is the argmax and ranks descend, under a tie-saturated distribution" {
    // The producer used to recover token ids by scanning the whole vocab for
    // FLOAT EQUALITY with each `mlx_topk` value. Ties make that ambiguous, and
    // at temp 0 the post-temperature distribution saturates so ties are
    // everywhere — measured 0/5 positions where rank 1 was the chosen token on
    // a trivial prompt. The observable contract: with greedy sampling the
    // chosen token IS the argmax, so rank 1 must equal it.
    if (mlx.noGpuBackend()) return;
    const s = mlx.gpuStream();
    const allocator = testing.allocator;

    // Six exactly-tied entries, then a strict runner-up and a strict max.
    const data = [_]f32{ 1, 1, 1, 1, 1, 1, 2, 3 };
    const shape = [_]c_int{ 1, 8 };
    const logits = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);

    const chosen: u32 = 7; // argmax
    const r = try computeLogprobs(allocator, logits, chosen, 3, s);
    defer allocator.free(r.top_logprobs);

    try testing.expectEqual(@as(usize, 3), r.top_logprobs.len);
    try testing.expectEqual(chosen, r.top_logprobs[0].token_id);
    try testing.expectEqual(@as(u32, 6), r.top_logprobs[1].token_id);
    try testing.expect(r.top_logprobs[2].token_id < 6);
    // Strictly descending, and ids distinct.
    try testing.expect(r.top_logprobs[0].logprob > r.top_logprobs[1].logprob);
    try testing.expect(r.top_logprobs[1].logprob > r.top_logprobs[2].logprob);
    try testing.expect(r.top_logprobs[0].token_id != r.top_logprobs[1].token_id);
    try testing.expect(r.top_logprobs[1].token_id != r.top_logprobs[2].token_id);
    // The chosen token's own logprob must agree with its rank-1 entry.
    try testing.expectApproxEqAbs(r.top_logprobs[0].logprob, r.token_logprob, 1e-6);
}

test "sampleToken: reported logprobs are the model's, not the client's temperature" {
    // Same prompt, same chosen token, three temperatures used to report three
    // different logprobs (-0.2129 / -0.0607 / -2.1566 at 0 / 0.6 / 2.0). The
    // value belongs to the model, so it must not move with a sampling knob.
    if (mlx.noGpuBackend()) return;
    const s = mlx.gpuStream();
    const allocator = testing.allocator;

    const data = [_]f32{ 0, 1, 2, 3 };
    const shape = [_]c_int{ 1, 1, 4 };
    const logits = mlx.mlx_array_new_data(&data, &shape, 3, .float32);
    defer _ = mlx.mlx_array_free(logits);

    // log_softmax([0,1,2,3])[3], computed independently.
    var denom: f64 = 0;
    for (data) |v| denom += @exp(@as(f64, v));
    const expect_top: f32 = @floatCast(3.0 - @log(denom));

    const temps = [_]f32{ 0.0, 0.6, 2.0 };
    for (temps) |t| {
        const sp = SamplingParams{ .temperature = t, .seed = 7 };
        const r = try sampleToken(allocator, logits, sp, null, 2, null, s);
        const lp = r.logprob_result orelse return error.NoLogprobs;
        defer allocator.free(lp.top_logprobs);
        try testing.expectEqual(@as(usize, 2), lp.top_logprobs.len);
        try testing.expectEqual(@as(u32, 3), lp.top_logprobs[0].token_id);
        try testing.expectEqual(@as(u32, 2), lp.top_logprobs[1].token_id);
        try testing.expectApproxEqAbs(expect_top, lp.top_logprobs[0].logprob, 1e-5);
    }
}

test "logprobs publish through the one-token delay, never straight from sampleToken" {
    // The decode loop returns `next_token_id` and, in the SAME call, forwards
    // it to sample its successor — so a path that publishes `sampleToken`'s
    // result as `last_logprob` shifts the whole array by one. Live symptom: a
    // one-token "OK" reply came back with `<|role_end|>` at rank 1, which reads
    // as a broken ranking rather than a broken pairing, and sent a real
    // quantization hunt down the wrong path for a day.
    //
    // The needles are concatenated so this test's OWN source doesn't match.
    const needle = "last_logprob" ++ " = ";
    const allowed = "last_logprob" ++ " = self.pending_logprob";
    const src = @embedFile("generate.zig");
    var it = std.mem.splitScalar(u8, src, '\n');
    var lineno: usize = 0;
    var seen: usize = 0;
    while (it.next()) |raw| {
        lineno += 1;
        const line = if (std.mem.indexOf(u8, raw, "//")) |c| raw[0..c] else raw;
        if (std.mem.indexOf(u8, line, needle) == null) continue;
        // `= null` is the ownership hand-off to the scheduler, not a publish.
        if (std.mem.indexOf(u8, line, "= null") != null) continue;
        seen += 1;
        if (std.mem.indexOf(u8, line, allowed) == null) {
            std.debug.print("generate.zig:{d}: logprobs published without the one-token delay: {s}\n", .{
                lineno, std.mem.trim(u8, line, " "),
            });
            return error.LogprobPairingBypass;
        }
    }
    try testing.expect(seen >= 1);
}

test "prefill chunk loop yields to the interleave hook between chunks, never after the last" {
    // The scheduler runs decode ticks for active streams at this seam so a
    // long prefill cannot stall them for its whole duration. The final
    // boundary is excluded: the post-prefill decode tick covers it.
    const src = @embedFile("generate.zig");
    const progress = "if (options.prefill_progress) |p| p.store(@intCast(pos), .monotonic);";
    const at = std.mem.indexOf(u8, src, progress) orelse return error.ProgressPublishMissing;
    const guard = "if (pos < loop_end) {";
    const guard_at = std.mem.indexOfPos(u8, src, at, guard) orelse return error.InterleaveGuardMissing;
    const call = "if (options.interleave" ++ "_hook) |hk| hk.call(hk.ctx);";
    const call_at = std.mem.indexOfPos(u8, src, guard_at, call) orelse return error.InterleaveCallMissing;
    // The call sits INSIDE the guard (same statement block, before the loop
    // re-tests `pos`), not somewhere later in the file.
    try std.testing.expect(call_at - guard_at < 200);
}

test "countSpliceRows counts image and video placeholders, video only when declared" {
    const ids = [_]i32{ 7, 99, 99, 8, 9, 77 };
    try testing.expectEqual(@as(usize, 2), countSpliceRows(&ids, 99, 0));
    try testing.expectEqual(@as(usize, 3), countSpliceRows(&ids, 99, 77));
    try testing.expectEqual(@as(usize, 0), countSpliceRows(ids[0..1], 99, 77));
}

test "vision prefill chunks by default and the splice offset feeds every prefill forward" {
    // Source scan: the chunk-size pick keys on the kill switch (not bare
    // has_vision), the chunk loop and the final-span forward both set
    // ctx.vision_splice_offset, and the loop advances the consumed-row count.
    const src = @embedFile("generate.zig");
    const marker = "const default_chunk = if (has_vision and !vision" ++ "_chunked) loop_end else PREFILL_CHUNK;";
    try testing.expect(std.mem.indexOf(u8, src, marker) != null);
    const set_off = "ctx.vision_splice" ++ "_offset = vision_rows_consumed;";
    const first = std.mem.indexOf(u8, src, set_off) orelse return error.ChunkOffsetMissing;
    try testing.expect(std.mem.indexOfPos(u8, src, first + 1, set_off) != null); // final-span site too
    try testing.expect(std.mem.indexOf(u8, src, "vision_rows_consumed += countSplice" ++ "Rows(") != null);
}

test "MtpTrace per-index acceptance: index i counts only rounds that drafted it" {
    var t: Generator.MtpTrace = .{};
    _ = t.endRound(3, 3, false); // all three accepted
    _ = t.endRound(3, 1, false); // a0 accepted, a1 rejected, a2 never judged
    _ = t.endRound(1, 0, false); // depth-1 round: only index 0 drafted
    var buf: [64]u8 = undefined;
    try std.testing.expectEqualStrings("0.67/0.50/0.50", t.accIdxStr(&buf));
    t.reset();
    try std.testing.expectEqualStrings("", t.accIdxStr(&buf));
}

test "qwen4: the coarse rerank head is READY before any draft has run" {
    // The head is a full `requantizeRows` of the trunk lm_head plus a
    // synchronous eval of its ~240 MB. Built lazily it landed inside the
    // FIRST request's draft chain — on the inference thread, mid-round,
    // draining the stream, and paid as first-token latency by whoever
    // happened to be first after a load. `qwen4BuildDraftRerank` is that
    // build as a standalone LOAD-time step: no Generator, no chain, no
    // request. Afterwards `canRerankDrafts()` is a pure read.
    if (mlx.noGpuBackend()) return;
    const s = mlx.gpuStream();

    var fx = try mtp_mod.RerankFixture.init(s, mtp_mod.TOP32_MIN_ROWS + 96, 256, 8, 64, 0xC0DE);
    defer fx.deinit();
    // Only the rerank fields are read by the build; the rest of the head is
    // the trunk's business.
    var head: transformer_mod.Qwen4Mtp = undefined;
    head.rerank = null;
    head.rerank_logged = false;
    head.rerank_tried = false;
    fx.xfm.qwen4_mtp = head;
    defer if (fx.xfm.qwen4_mtp) |*m| {
        if (m.rerank) |*rc| rc.deinit();
    };

    // Nothing is built by construction...
    try testing.expect(fx.xfm.qwen4_mtp.?.rerank == null);
    // ...the eager build stands on its own...
    try testing.expect(fx.xfm.qwen4BuildDraftRerank());
    const built = fx.xfm.qwen4_mtp.?.rerank orelse return error.NoCoarseHead;
    try testing.expectEqual(fx.vocab, built.rows);
    try testing.expect(built.bits > 0 and built.bits < 8);

    // ...and the draft path now only READS it. Same mlx handle => the ask
    // did not rebuild anything (which is what the lazy path did).
    const head_ctx = built.q.w.ctx;
    const ref = MtpHeadRef{ .qwen4 = &fx.xfm };
    try testing.expect(ref.canRerankDrafts());
    try testing.expectEqual(head_ctx, fx.xfm.qwen4_mtp.?.rerank.?.q.w.ctx);

    // One-shot: a second eager build (both load paths firing, a re-probe)
    // keeps the head it already has.
    try testing.expect(fx.xfm.qwen4BuildDraftRerank());
    try testing.expectEqual(head_ctx, fx.xfm.qwen4_mtp.?.rerank.?.q.w.ctx);
}

test "every request-finalize seam that logs [spec-stats] also logs [qsa-arms]" {
    // Class guard for a wiring bug this test was written after: `logQsaArms`
    // was added beside the three `gen.logSpecStats()` calls in THIS file, which
    // are the legacy/CLI path only. The SERVER finalizes requests in
    // `scheduler.finishSlot` ("scheduler-driven slots finalize here instead"),
    // so `[qsa-arms]` was dead on every served request while looking wired.
    // A per-request meter is worth nothing if it misses the path that actually
    // serves requests, so the SEAM — not the symbol — is what gets pinned:
    // every call site of the request summary must carry both. Needles are
    // assembled at comptime so this test's own source cannot satisfy the scan.
    const spec_call = "logSpec" ++ "Stats();";
    const qsa_call = "logQsa" ++ "Arms();";
    inline for (.{ @embedFile("generate.zig"), @embedFile("scheduler.zig") }) |src| {
        var i: usize = 0;
        var seams: usize = 0;
        while (std.mem.indexOfPos(u8, src, i, spec_call)) |at| {
            seams += 1;
            // The paired call must sit in the same short block, not merely
            // somewhere else in the file.
            const win_end = @min(src.len, at + 400);
            if (std.mem.indexOfPos(u8, src, at, qsa_call)) |q| {
                if (q >= win_end) return error.SpecStatsSeamMissesQsaArms;
            } else return error.SpecStatsSeamMissesQsaArms;
            i = at + spec_call.len;
        }
        try std.testing.expect(seams > 0);
    }
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

test "mtpAdaptiveVoteFor: four real switch events of the qwen4 A/B" {
    // Four switches the v1 controller made on a 62.7k prose prompt where MTP was the faster
    // arm (56.8 vs 51.6 tok/s): the modeled token count ran 12-31% under the cell's own `tok`.
    const G = Generator;
    const margin = G.MTP_ADAPTIVE_MARGIN;
    const Event = struct {
        name: []const u8,
        table_ms_tok: f32, // Table.msPerTok(m_lo, bucket), from `table=` in [spec-stats]
        serial_ms_tok: f32, // serial_cell= at the moment of the switch
        window_ms_tok: f32, // round_ms EMA / (avg_per_round + 1)
        window_full: bool, // rounds run before the vote >= MTP_PRICE_WINDOW
    };
    const events = [_]Event{
        .{ .name = "b1 long1", .table_ms_tok = 16.12, .serial_ms_tok = 19.20, .window_ms_tok = 15.40, .window_full = true },
        // Voted at round ~3 on prefix-cache-restore warmup rounds; the full-window gate saves it.
        .{ .name = "b1 long2", .table_ms_tok = 15.85, .serial_ms_tok = 18.16, .window_ms_tok = 24.17, .window_full = false },
        .{ .name = "b4 long1", .table_ms_tok = 17.28, .serial_ms_tok = 19.53, .window_ms_tok = 16.63, .window_full = true },
        .{ .name = "b4 long2", .table_ms_tok = 17.91, .serial_ms_tok = 17.08, .window_ms_tok = 15.97, .window_full = true },
    };
    for (events) |e| {
        try testing.expectEqual(
            G.MtpAdaptiveVote.mtp,
            G.mtpAdaptiveVoteFor(e.table_ms_tok, e.table_ms_tok, e.serial_ms_tok, margin),
        );
        // Must not vote serial on any row; an unfilled window is `.undecided`.
        const win: ?f32 = if (e.window_full) e.window_ms_tok else null;
        const got = G.mtpAdaptiveVoteFor(e.table_ms_tok, win, e.serial_ms_tok, margin);
        try testing.expect(got != .serial);
        try testing.expectEqual(
            if (e.window_full) G.MtpAdaptiveVote.mtp else G.MtpAdaptiveVote.undecided,
            got,
        );
    }
    // Rule B ungated reproduces the b1-long2 mistake.
    const b1l2 = events[1];
    try testing.expectEqual(
        G.MtpAdaptiveVote.serial,
        G.mtpAdaptiveVoteFor(30.0, b1l2.window_ms_tok, b1l2.serial_ms_tok, margin),
    );
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

test "MtpPriceWindow: reads nothing until FULL, skips width trials, prices per TOKEN" {
    const G = Generator;
    const N = G.MTP_PRICE_WINDOW;
    var w = G.MtpPriceWindow{};
    try testing.expect(w.msPerTok() == null);

    var i: u32 = 0;
    while (i < N - 1) : (i += 1) {
        w.observe(20.0, 2.0, false);
        try testing.expect(w.msPerTok() == null); // still not full
    }
    w.observe(20.0, 2.0, false);
    try testing.expectApproxEqAbs(@as(f32, 10.0), w.msPerTok().?, 1e-4);

    var t = G.MtpPriceWindow{};
    i = 0;
    while (i < N * 2) : (i += 1) t.observe(999.0, 1.0, true);
    try testing.expect(t.msPerTok() == null);
    try testing.expectEqual(@as(u32, 0), t.filled);

    var d = G.MtpPriceWindow{};
    d.observe(std.math.nan(f32), 2.0, false);
    d.observe(20.0, 0.0, false);
    d.observe(-1.0, 2.0, false);
    try testing.expectEqual(@as(u32, 0), d.filled);

    // Summed over the window: 8 rounds at 1 tok/30 ms + 8 at 4 tok/60 ms = 18.0 ms/tok, not 22.5.
    var m = G.MtpPriceWindow{};
    i = 0;
    while (i < N / 2) : (i += 1) m.observe(30.0, 1.0, false);
    while (i < N) : (i += 1) m.observe(60.0, 4.0, false);
    try testing.expectApproxEqAbs(@as(f32, 18.0), m.msPerTok().?, 1e-4);

    i = 0;
    while (i < N) : (i += 1) m.observe(10.0, 2.0, false);
    try testing.expectApproxEqAbs(@as(f32, 5.0), m.msPerTok().?, 1e-4);
    m.reset();
    try testing.expect(m.msPerTok() == null);
}

test "MtpAdaptive: the switch needs CONFIRM consecutive rounds, counted once per round" {
    const G = Generator;
    const need = G.MTP_ADAPTIVE_CONFIRM;
    var a = G.MtpAdaptive{};
    var r: u32 = 0;
    while (r < need - 1) : (r += 1) {
        try testing.expectEqual(G.MtpAdaptiveAction.none, a.round(r, 3, .serial, need));
    }
    try testing.expectEqual(G.MtpAdaptiveAction.to_serial, a.round(need - 1, 3, .serial, need));
    try testing.expectEqual(G.MtpAdaptiveArm.serial, a.arm);
    try testing.expectEqual(@as(u32, 1), a.switches);

    // `mtpRoundPlan` has two call sites per round: the second call at the same index must not advance the streak.
    var b = G.MtpAdaptive{};
    var i: u32 = 0;
    while (i < need - 1) : (i += 1) {
        try testing.expectEqual(G.MtpAdaptiveAction.none, b.round(i, 3, .serial, need));
        try testing.expectEqual(G.MtpAdaptiveAction.none, b.round(i, 3, .serial, need));
    }
    try testing.expectEqual(G.MtpAdaptiveArm.undecided, b.arm);
    try testing.expectEqual(G.MtpAdaptiveAction.to_serial, b.round(need - 1, 3, .serial, need));

    var c = G.MtpAdaptive{};
    _ = c.round(0, 3, .serial, need);
    _ = c.round(1, 3, .serial, need);
    try testing.expectEqual(G.MtpAdaptiveAction.none, c.round(2, 3, .mtp, need));
    try testing.expectEqual(G.MtpAdaptiveArm.mtp, c.arm);
    try testing.expectEqual(G.MtpAdaptiveAction.none, c.round(3, 3, .serial, need));
    try testing.expectEqual(G.MtpAdaptiveAction.none, c.round(4, 3, .serial, need));
    try testing.expectEqual(G.MtpAdaptiveAction.to_serial, c.round(5, 3, .serial, need));

    var d = G.MtpAdaptive{};
    _ = d.round(0, 3, .serial, need);
    _ = d.round(1, 3, .serial, need);
    try testing.expectEqual(G.MtpAdaptiveAction.none, d.round(2, 3, .undecided, need));
    try testing.expectEqual(G.MtpAdaptiveAction.none, d.round(3, 3, .serial, need));
    try testing.expectEqual(G.MtpAdaptiveArm.undecided, d.arm);

    var e = G.MtpAdaptive{};
    _ = e.round(0, 3, .serial, need);
    _ = e.round(1, 3, .serial, need);
    try testing.expectEqual(G.MtpAdaptiveAction.none, e.round(2, 4, .serial, need));
    try testing.expectEqual(@as(u32, 1), e.confirm);

    var f = G.MtpAdaptive{};
    try testing.expectEqual(G.MtpAdaptiveAction.to_serial, f.round(0, 3, .serial, 0));
}

test "MtpAdaptive: only a bucket crossing re-opens the serial decision" {
    const G = Generator;
    var off = G.MtpAdaptive{};
    _ = off.round(0, 3, .serial, 1);
    var i: u32 = 0;
    while (i < 1024) : (i += 1) {
        try testing.expectEqual(G.MtpAdaptiveAction.none, off.serialTick(3));
    }
    try testing.expectEqual(G.MtpAdaptiveArm.serial, off.arm);
    try testing.expectEqual(G.MtpAdaptiveAction.to_mtp, off.serialTick(4));
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

test "S21: a KV-only sidecar head keeps its re-arm, and nothing about it changes" {
    const G = Generator;
    try testing.expect(!G.stickyOnSerialSwitch(MtpHeadRef{ .qwen = undefined }));
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

test "mtpSerialCaptureReady: the capture step's entry invariant is a RUNTIME check, not an assert" {
    const G = Generator;
    try testing.expect(G.mtpSerialCaptureReady(false, false));
    try testing.expect(!G.mtpSerialCaptureReady(true, false));
    try testing.expect(!G.mtpSerialCaptureReady(false, true));
    try testing.expect(!G.mtpSerialCaptureReady(true, true));
}

test "MTP head persistence kill switch: only a literal 0 turns it off" {
    try testing.expect(mtpHeadPersistFromEnv(null));
    try testing.expect(mtpHeadPersistFromEnv(""));
    try testing.expect(mtpHeadPersistFromEnv("1"));
    try testing.expect(!mtpHeadPersistFromEnv("0"));
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

test "mtpAdaptiveHeadEligible: the qwen4 head and MiMo's heads switch to serial, a sidecar head does not" {
    const G = Generator;
    try testing.expect(G.mtpAdaptiveHeadEligible(.qwen4));
    try testing.expect(G.mtpAdaptiveHeadEligible(.mimo));
    try testing.expect(!G.mtpAdaptiveHeadEligible(.qwen));
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

test "reservedPrefillTokens: the KV capacity reservation is qwen4_exp-only; every other arch keeps proportional growth" {
    // The reservation pre-buys prompt + generation headroom + slack past 32k; a pure cost on an
    // arch nobody measured, so it is gated.
    const t = std.testing;
    const chunk: u64 = 4096;
    const seq: u64 = 200_000;
    const max_tokens: u64 = 2048;

    var qwen4 = model_mod.ModelConfig{ .model_type = "qwen4_exp", .max_position_embeddings = 1_048_576 };
    const KVC = transformer_mod.KVCache;
    try t.expectEqual(
        KVC.reservedTokens(seq, max_tokens, chunk, 1_048_576),
        reservedPrefillTokens(&qwen4, seq, max_tokens, chunk),
    );
    try t.expect(reservedPrefillTokens(&qwen4, seq, max_tokens, chunk) > seq);

    // Every other arch: zero, which leaves `nextCapacityReserved` == `nextCapacity`.
    for ([_][]const u8{ "qwen3_5", "qwen3_5_moe", "lfm2", "nemotron_h", "bailing_hybrid", "llama" }) |mt| {
        var cfg = model_mod.ModelConfig{ .model_type = mt, .max_position_embeddings = 262_144 };
        try t.expectEqual(@as(u64, 0), reservedPrefillTokens(&cfg, seq, max_tokens, chunk));
    }

    var cache = try KVC.init(t.allocator, 4);
    defer cache.deinit();
    cache.reserve(0);
    try t.expectEqual(@as(usize, 0), cache.reserve_tokens);
}

test "a warm prefill grows its restored layers to the reservation before the first chunk writes" {
    // seq <= C < R: left to the decode, the first step past C grows every layer inside one forward.
    if (mlx.noGpuBackend()) return;
    const t = std.testing;
    const s = mlx.gpuStream();
    const KVC = transformer_mod.KVCache;
    const seq: u64 = 40_000;
    const matched: c_int = 36_000;
    const capacity: usize = 40_960;
    const qwen4 = model_mod.ModelConfig{ .model_type = "qwen4_exp", .max_position_embeddings = 1_048_576 };
    const mimo = model_mod.ModelConfig{ .model_type = "mimo_v2", .has_sliding_window = true, .sliding_window = 128, .head_dim = 192, .max_position_embeddings = 1_048_576 };
    var rows = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rows);
    try mlx.check(mlx.mlx_zeros(&rows, &[_]c_int{ 1, 2, matched, 64 }, 4, .bfloat16, s));
    var tail = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(tail);
    try mlx.check(mlx.mlx_zeros(&tail, &[_]c_int{ 1, 2, @intCast(seq - @as(u64, @intCast(matched))), 64 }, 4, .bfloat16, s));
    var one = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(one);
    try mlx.check(mlx.mlx_zeros(&one, &[_]c_int{ 1, 2, 1, 64 }, 4, .bfloat16, s));
    for ([_]*const model_mod.ModelConfig{ &qwen4, &mimo }) |cfg| {
        const ring: u32 = if (cfg.swaRingTokens() > 0) cfg.sliding_window else 0;
        for ([_]transformer_mod.KVQuantConfig{ .dense, .affine(8), .affine(6) }) |kvq| {
            for ([_]bool{ true, false }) |donate| {
                // The previous turn's cache: it reserved C and holds the matched rows. Layer 1 rings on MiMo.
                var donor = try KVC.initWithConfig(t.allocator, 2, kvq);
                defer donor.deinit();
                donor.setSwaRing(ring);
                donor.reserve(capacity);
                for (0..2) |li| {
                    var v = try donor.update(@intCast(li), rows, rows, s, if (li == 1) ring else 0);
                    v.deinit();
                }
                var snap = try donor.snapshotRetained(s);
                defer snap.deinit();
                var warm = try KVC.initWithConfig(t.allocator, 2, kvq);
                defer warm.deinit();
                warm.setSwaRing(ring);
                try warm.restore(&snap);
                if (donate) {
                    snap.releaseHandles();
                    warm.adoptRestored();
                }
                const ring_cap = mlx.getShape(warm.entries[1].keys)[2];

                const reserved = try reserveRequestCapacity(&warm, cfg, seq, 32_000, 2048, s);
                try t.expect(seq <= capacity and capacity < reserved);
                try t.expectEqual(@as(c_int, capacity), mlx.getShape(donor.entries[0].keys)[2]);
                try t.expect(mlx.getShape(warm.entries[0].keys)[2] >= reserved);
                try t.expect(mlx.getShape(warm.entries[0].values)[2] >= reserved);
                try t.expectEqual(@as(usize, @intCast(matched)), warm.entries[0].offset);
                if (ring > 0) try t.expectEqual(ring_cap, mlx.getShape(warm.entries[1].keys)[2]);

                // The prefill's rows and the first decode step write in place: nothing grows.
                const grows = KVC.kv_cap_buf_grows;
                var pv = try warm.update(0, tail, tail, s, 0);
                pv.deinit();
                var dv = try warm.update(0, one, one, s, 0);
                dv.deinit();
                try t.expectEqual(grows, KVC.kv_cap_buf_grows);
            }
        }
    }
}

test "a block decoder's entry token stops the round before it drafts" {
    // Bar: an EOS or third pad as the round's entry token ends the request before any draft.
    const eos = [_]u32{ 151645, 151643 };
    var pad: u32 = 0;
    try testing.expect(tokenStops(151645, &eos, &pad));
    try testing.expect(tokenStops(151643, &eos, &pad));
    try testing.expect(!tokenStops(17, &eos, &pad));

    // A pad run stops on the third; any real token resets it.
    pad = 0;
    try testing.expect(!tokenStops(0, &eos, &pad));
    try testing.expect(!tokenStops(0, &eos, &pad));
    try testing.expect(tokenStops(0, &eos, &pad));
    pad = 0;
    try testing.expect(!tokenStops(0, &eos, &pad));
    try testing.expect(!tokenStops(9, &eos, &pad));
    try testing.expectEqual(@as(u32, 0), pad);
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
    var chains: [3]Generator.MtpPreDraft = undefined;
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
    var chains: [2]Generator.MtpPreDraft = undefined;
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
    const chain: Generator.MtpPreDraft = .{
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
    const open: Generator.MtpRoundOpen = .{
        .chain = chain,
        .tracing = false,
        .ph = undefined,
        .round_watch = undefined,
    };
    try expectError(error.NoMtpHead, gen.mtpRoundContinue(allocator, open));
}

test "batched MTP state includes the last row activated by round begin" {
    const allocator = testing.allocator;
    var xfm: Transformer = undefined;
    xfm.qwen4_mtp_owner = null;
    var head: transformer_mod.Qwen4Mtp = undefined;
    head.cache = try KVCache.init(allocator, 1);
    head.entry = .{ .conv_state = .{ .ctx = null }, .ssm_state = .{ .ctx = null }, .initialized = true };
    head.qsa_marks = .{};
    head.seq_offset = 0;
    head.pos_base = 0;
    xfm.qwen4_mtp = head;
    defer xfm.qwen4_mtp.?.cache.deinit();
    var a: Transformer.Qwen4MtpState = .{ .cache = try KVCache.init(allocator, 1), .entry = head.entry };
    defer a.deinit();
    var b: Transformer.Qwen4MtpState = .{ .cache = try KVCache.init(allocator, 1), .entry = head.entry };
    defer b.deinit();
    xfm.qwen4MtpActivate(&a);
    xfm.qwen4_mtp.?.seq_offset = 7;
    xfm.qwen4_mtp.?.cache.step = 7;
    xfm.qwen4_mtp.?.pos_base = 3;
    xfm.qwen4MtpActivate(&b);
    xfm.qwen4_mtp.?.seq_offset = 2;
    xfm.qwen4_mtp.?.cache.step = 2;
    xfm.qwen4_mtp.?.pos_base = 11;
    const caches = [_]MtpCacheRef{
        .{ .qwen4 = .{ .t = &xfm, .st = &a, .allocator = allocator } },
        .{ .qwen4 = .{ .t = &xfm, .st = &b, .allocator = allocator } },
    };
    const want = [_]Transformer.Qwen4MtpState.Meta{
        .{ .stash_len = 7, .origin = 3, .cache_step = 7 },
        .{ .stash_len = 2, .origin = 11, .cache_step = 2 },
    };
    for (caches, want) |cache, meta| {
        const state = cache.qwen4StateForBatch();
        try testing.expectEqual(meta, state.meta());
    }
    try testing.expect(xfm.qwen4_mtp_owner == null);
}

const MtpChainTestSlot = struct {
    cache: KVCache,
    off: usize = 0,
    entries: []transformer_mod.SSMCacheEntry,
    gen: ?Generator = null,

    fn init(a: std.mem.Allocator, io: std.Io, xfm: *Transformer, tok: *Tokenizer, prompt: []const u32, depth: u32) !*@This() {
        const slot = try a.create(@This());
        errdefer a.destroy(slot);
        var cache = try KVCache.init(a, xfm.config.num_hidden_layers);
        errdefer cache.deinit();
        const entries = try a.alloc(transformer_mod.SSMCacheEntry, xfm.config.num_hidden_layers);
        slot.* = .{ .cache = cache, .entries = entries };
        for (entries) |*e| e.* = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = false };
        errdefer {
            for (entries) |*e| {
                _ = mlx.mlx_array_free(e.conv_state);
                _ = mlx.mlx_array_free(e.ssm_state);
                transformer_mod.ssmFreeQsaState(e);
            }
            a.free(entries);
        }
        const ctx: ForwardCtx = .{ .cache = &slot.cache, .moe_seq_offset = &slot.off, .ssm_entries = entries, .capture_hidden = null, .vision_embeddings = null };
        slot.gen = try Generator.initWithOptions(io, a, xfm, tok, prompt, 256, .{ .temperature = 0 }, &.{}, .{
            .ctx = ctx,
            .mtp = .{ .qwen4 = xfm },
            .mtp_enabled = true,
            .model_has_mtp = true,
            .mtp_depth = depth,
            .skip_lazy_preforward = true,
        });
        return slot;
    }

    fn deinit(slot: *@This(), a: std.mem.Allocator) void {
        if (slot.gen) |*g| g.deinit(a);
        for (slot.entries) |*e| {
            _ = mlx.mlx_array_free(e.conv_state);
            _ = mlx.mlx_array_free(e.ssm_state);
            transformer_mod.ssmFreeQsaState(e);
        }
        a.free(slot.entries);
        slot.cache.deinit();
        a.destroy(slot);
    }

    fn begin(g: *Generator, a: std.mem.Allocator, depth: u32, confidence: bool) !Generator.MtpRoundOpen {
        g.mtp_depth_current = depth;
        g.spec_cost_solo = false;
        g.mtp_batch_head = true;
        defer g.mtp_batch_head = false;
        const begun = try g.mtpRoundBegin(a);
        var open = switch (begun) {
            .open => |o| o,
            .verify => |v| {
                var st = v;
                st.deinit(a);
                return error.ExpectedOpenRound;
            },
            .done => |r| {
                if (r) |result| a.free(result.tokens);
                return error.MtpRoundDeclined;
            },
        };
        errdefer open.chain.deinit(a);
        try testing.expectEqual(depth, open.chain.m);
        if (confidence) {
            if (open.chain.conf_arrs != null) std.debug.print("[mtp fixture] existing confidence plan={any}\n", .{open.chain.plan});
            try testing.expect(open.chain.conf_arrs == null);
            open.chain.drafts = try a.realloc(open.chain.drafts, depth + 1);
            open.chain.draft_arrs = try a.realloc(open.chain.draft_arrs, depth + 1);
            open.chain.conf_arrs = try a.alloc(mlx.mlx_array, depth);
            open.chain.plan.m_hi = depth + 1;
            open.chain.plan.tau_ln = 1;
        }
        return open;
    }

    fn equal(a: mlx.mlx_array, b: mlx.mlx_array, s: mlx.mlx_stream, name: []const u8) !void {
        if (a.ctx == null or b.ctx == null) {
            try testing.expect(a.ctx == null and b.ctx == null);
            return;
        }
        var same = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(same);
        try mlx.check(mlx.mlx_array_equal(&same, a, b, false, s));
        try mlx.check(mlx.mlx_array_eval(same));
        var yes = false;
        try mlx.check(mlx.mlx_array_item_bool(&yes, same));
        if (!yes) std.debug.print("[mtp round mismatch] {s}: {any} vs {any}\n", .{ name, mlx.getShape(a), mlx.getShape(b) });
        try testing.expect(yes);
    }
};

test "Generator rounds preserve acceptance and next-round stashes at N=2/4" {
    const model_dir = std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const predraft = Generator.mtp_predraft_enabled;
    const bounds_before = depth_bounds.active;
    const planner = group_planner.enabled_override;
    defer {
        Generator.mtp_predraft_enabled = predraft;
        depth_bounds.active = bounds_before;
        group_planner.enabled_override = planner;
    }
    // Observe the completed-round stash before any successor consumes it.
    Generator.mtp_predraft_enabled = false;
    group_planner.enabled_override = true;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const coarse_before = transformer_mod.mtp_coarse_pair_dispatches;
    const hc_prepared_before = qwen4_forward.mtp_verify_hc_prepared_calls;
    const verify_kernel_before = transformer_mod.mtp_verify_kernel_calls;
    var config = try model_mod.parseConfig(io, a, std.mem.span(model_dir));
    defer if (config.ngram_table_path) |path| a.free(path);
    var weights = try model_mod.loadWeights(io, a, std.mem.span(model_dir));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Transformer.init(io, a, config, &weights);
    defer xfm.deinit();
    xfm.compileQwen4Hc();
    xfm.compileGdnGate();
    xfm.compileMoeRouting();
    var tok = Tokenizer.initEmptyForTests(a, .byte_level_bpe);
    defer tok.deinit();
    var prompt: [96]u32 = undefined;
    for (&prompt, 0..) |*id, i| id.* = @intCast(1 + (i * 7919) % 40000);
    for ([_]usize{ 2, 4 }) |n| {
        for ([_]u32{ 1, 2, 4 }) |depth| {
            depth_bounds.active = .{ .min = depth, .max = depth };
            var solo: [4]?*MtpChainTestSlot = @splat(null);
            var batch: [4]?*MtpChainTestSlot = @splat(null);
            defer for (solo, batch) |ss, bs| {
                if (ss) |x| x.deinit(a);
                if (bs) |x| x.deinit(a);
            };
            for (0..n) |i| {
                const ids = prompt[0..if (i % 2 == 0) @as(usize, 96) else 61];
                solo[i] = try MtpChainTestSlot.init(a, io, &xfm, &tok, ids, depth);
                batch[i] = try MtpChainTestSlot.init(a, io, &xfm, &tok, ids, depth);
            }
            for (0..3) |round| {
                var ss: [4]?Generator.MtpRoundState = @splat(null);
                var results: [4]?Generator.DrafterStepResult = @splat(null);
                defer for (&ss, results) |*state, result| {
                    if (state.*) |*v| v.deinit(a);
                    if (result) |r| a.free(r.tokens);
                };
                for (0..n) |i| {
                    const g = &solo[i].?.gen.?;
                    const open = try MtpChainTestSlot.begin(g, a, depth, i % 2 == 1);
                    ss[i] = try g.mtpRoundContinue(a, open);
                    try g.mtpRoundVerify(&ss[i].?);
                }
                var opens: [4]Generator.MtpRoundOpen = undefined;
                var chains: [4]Generator.MtpPreDraft = undefined;
                var owns: [4]bool = @splat(false);
                var gens: [4]*Generator = undefined;
                defer for (&chains, owns) |*chain, owned| {
                    if (owned) chain.deinit(a);
                };
                for (0..n) |i| {
                    gens[i] = &batch[i].?.gen.?;
                    opens[i] = try MtpChainTestSlot.begin(gens[i], a, depth, i % 2 == 1);
                    chains[i] = opens[i].chain;
                    owns[i] = true;
                }
                try Generator.mtpChainBuildBatched(gens[0..n], chains[0..n], 0, depth);
                var batch_states: [4]?Generator.MtpRoundState = @splat(null);
                defer for (&batch_states) |*state| {
                    if (state.*) |*p| p.deinit(a);
                };
                var batch_ptrs: [4]*Generator.MtpRoundState = undefined;
                for (0..n) |i| {
                    opens[i].chain = chains[i];
                    owns[i] = false;
                    batch_states[i] = try gens[i].mtpRoundContinue(a, opens[i]);
                    batch_ptrs[i] = &batch_states[i].?;
                }
                const grouped_verify = @import("mtp_group_planner.zig").enabled();
                if (grouped_verify) {
                    try Generator.mtpGroupVerify(gens[0..n], batch_ptrs[0..n]);
                    try Generator.mtpGroupAccept(gens[0..n], batch_ptrs[0..n]);
                } else {
                    for (gens[0..n], batch_ptrs[0..n]) |g, state| try g.mtpRoundVerify(state);
                }
                for (0..n) |i| {
                    const bs = batch_ptrs[i];
                    std.debug.print("[mtp compare] N={d} depth={d} round={d} row={d}\n", .{ n, depth, round, i });
                    try testing.expectEqual(depth, bs.chain.m);
                    for (ss[i].?.chain.draft_arrs[0..depth], bs.chain.draft_arrs[0..depth]) |x, y| try MtpChainTestSlot.equal(x, y, xfm.s, "x");
                    try MtpChainTestSlot.equal(ss[i].?.verify_logits, bs.verify_logits, xfm.s, "ss[i].?.verify_logits");
                    const sg = &solo[i].?.gen.?;
                    results[i] = try sg.mtpRoundFinish(a, &ss[i].?);
                    try testing.expect(results[i] != null);
                    const result = (try gens[i].mtpRoundFinish(a, bs)) orelse return error.MtpRoundDeclined;
                    defer a.free(result.tokens);
                    try testing.expectEqualSlices(u32, results[i].?.tokens, result.tokens);
                    try testing.expectEqual(ss[i].?.retained, bs.retained);
                    try testing.expectEqual(sg.next_token_id, gens[i].next_token_id);
                    try testing.expectEqual(sg.mtp_accepted_tokens, gens[i].mtp_accepted_tokens);
                    try testing.expectEqual(solo[i].?.off, batch[i].?.off);
                    try MtpChainTestSlot.equal(sg.last_hidden, gens[i].last_hidden, xfm.s, "sg.last_hidden");
                    const sh = sg.mtp_hist_stash orelse return error.MissingSoloStash;
                    const bh = gens[i].mtp_hist_stash orelse return error.MissingBatchStash;
                    try testing.expectEqual(sh.off0, bh.off0);
                    try MtpChainTestSlot.equal(sh.ids, bh.ids, xfm.s, "sh.ids");
                    try MtpChainTestSlot.equal(sh.hidden, bh.hidden, xfm.s, "sh.hidden");
                    const hs = sg.mtp_cache.?.qwen4StateForBatch();
                    const hb = gens[i].mtp_cache.?.qwen4StateForBatch();
                    try testing.expectEqual(hs.meta(), hb.meta());
                    try MtpChainTestSlot.equal(hs.entry.aux_state, hb.entry.aux_state, xfm.s, "hs.entry.aux_state");
                    try MtpChainTestSlot.equal(hs.entry.qsa_pooled, hb.entry.qsa_pooled, xfm.s, "hs.entry.qsa_pooled");
                    for (solo[i].?.entries, batch[i].?.entries) |se, be| {
                        try MtpChainTestSlot.equal(se.conv_state, be.conv_state, xfm.s, "se.conv_state");
                        try MtpChainTestSlot.equal(se.ssm_state, be.ssm_state, xfm.s, "se.ssm_state");
                        try MtpChainTestSlot.equal(se.aux_state, be.aux_state, xfm.s, "se.aux_state");
                        try MtpChainTestSlot.equal(se.qsa_pooled, be.qsa_pooled, xfm.s, "se.qsa_pooled");
                        try testing.expectEqual(se.ple_prev_valid, be.ple_prev_valid);
                        if (se.ple_prev_valid) try testing.expectEqualSlices(u32, &se.ple_prev, &be.ple_prev);
                    }
                    std.debug.print("[mtp round] N={d} depth={d} round={d} row={d} accepted={d} stash={d}\n", .{ n, depth, round, i, bs.retained - 1, mlx.mlx_array_size(bh.ids) });
                }
            }
        }
    }
    if (transformer_mod.verifySharedHardware()) {
        try testing.expect(transformer_mod.mtp_coarse_pair_dispatches > coarse_before);
        {
            try testing.expect(qwen4_forward.mtp_verify_hc_prepared_calls > hc_prepared_before);
            std.debug.print("[prepared HC] calls={d}\n", .{qwen4_forward.mtp_verify_hc_prepared_calls - hc_prepared_before});
        }
        inline for (.{ .dense_tiles, .hc_rows, .route_pack }, 0..) |lever, i| {
            {
                try testing.expect(transformer_mod.mtp_verify_kernel_calls[i] > verify_kernel_before[i]);
                std.debug.print("[verify kernel] {s} calls={d}\n", .{ @tagName(lever), transformer_mod.mtp_verify_kernel_calls[i] - verify_kernel_before[i] });
            }
        }
    }
}

test "greedy tail on real qwen4 chains: depth 0 hands the verify its density, every later depth a one-hot at its draft (QWEN4_TEST_MODEL)" {
    const model_dir = std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const predraft = Generator.mtp_predraft_enabled;
    const bounds_before = depth_bounds.active;
    const planner = group_planner.enabled_override;
    defer {
        Generator.mtp_predraft_enabled = predraft;
        depth_bounds.active = bounds_before;
        group_planner.enabled_override = planner;
    }
    Generator.mtp_predraft_enabled = false;
    group_planner.enabled_override = false;
    const depth: u32 = 3;
    depth_bounds.active = .{ .min = depth, .max = depth };
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var config = try model_mod.parseConfig(io, a, std.mem.span(model_dir));
    defer if (config.ngram_table_path) |path| a.free(path);
    var weights = try model_mod.loadWeights(io, a, std.mem.span(model_dir));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Transformer.init(io, a, config, &weights);
    defer xfm.deinit();
    xfm.compileQwen4Hc();
    xfm.compileGdnGate();
    xfm.compileMoeRouting();
    var tok = Tokenizer.initEmptyForTests(a, .byte_level_bpe);
    defer tok.deinit();
    var prompt: [96]u32 = undefined;
    for (&prompt, 0..) |*id, i| id.* = @intCast(1 + (i * 7919) % 40000);

    const Check = struct {
        fn chain(c: *const Generator.MtpPreDraft, s: mlx.mlx_stream) !void {
            const qs = c.q_probs orelse return error.ExpectedProposalRows;
            try testing.expectEqual(c.m, c.n_qp);
            for (0..c.m) |i| {
                const shape = mlx.getShape(qs[i]);
                const q = try proposalTestRead(testing.allocator, qs[i], @intCast(shape[shape.len - 1]), s);
                defer testing.allocator.free(q);
                const id_read = try proposalTestRead(testing.allocator, c.draft_arrs[i], 1, s);
                defer testing.allocator.free(id_read);
                const id: usize = @intFromFloat(id_read[0]);
                var sum: f64 = 0;
                var support: usize = 0;
                for (q) |v| {
                    sum += v;
                    if (v > 0) support += 1;
                }
                try testing.expect(@abs(sum - 1.0) < 1e-3 and q[id] > 0);
                if (i > 0) try testing.expect(support == 1 and q[id] == 1.0);
            }
        }
    };

    // Solo chains (`mtpChainBuild`), then two rows through the grouped head
    // (`mtpChainBuildBatched`); every round then verifies and commits.
    var slots: [3]?*MtpChainTestSlot = @splat(null);
    defer for (slots) |slot| {
        if (slot) |x| x.deinit(a);
    };
    for (&slots, 0..) |*slot, i| {
        slot.* = try MtpChainTestSlot.init(a, io, &xfm, &tok, prompt[0..if (i == 1) @as(usize, 61) else 96], depth);
        const g = &slot.*.?.gen.?;
        g.sampling = .{ .temperature = 1.0, .top_p = 0.95, .top_k = 20, .seed = 11 + i, .suppress_mask = g.sampling.suppress_mask };
        g.mtp_greedy_tail = true;
    }
    for (0..3) |_| {
        const g = &slots[0].?.gen.?;
        var st = try g.mtpRoundContinue(a, try MtpChainTestSlot.begin(g, a, depth, false));
        defer st.deinit(a);
        try Check.chain(&st.chain, xfm.s);
        try g.mtpRoundVerify(&st);
        const r = (try g.mtpRoundFinish(a, &st)) orelse return error.MtpRoundDeclined;
        a.free(r.tokens);
    }
    for (0..3) |_| {
        var gens: [2]*Generator = undefined;
        var opens: [2]Generator.MtpRoundOpen = undefined;
        var chains: [2]Generator.MtpPreDraft = undefined;
        var owns: [2]bool = @splat(false);
        defer for (&chains, owns) |*c, owned| {
            if (owned) c.deinit(a);
        };
        for (0..2) |i| {
            gens[i] = &slots[1 + i].?.gen.?;
            opens[i] = try MtpChainTestSlot.begin(gens[i], a, depth, false);
            chains[i] = opens[i].chain;
            owns[i] = true;
        }
        try Generator.mtpChainBuildBatched(&gens, &chains, 0, depth);
        var states: [2]?Generator.MtpRoundState = @splat(null);
        defer for (&states) |*st| {
            if (st.*) |*v| v.deinit(a);
        };
        for (0..2) |i| try Check.chain(&chains[i], xfm.s);
        for (0..2) |i| {
            opens[i].chain = chains[i];
            owns[i] = false;
            states[i] = try gens[i].mtpRoundContinue(a, opens[i]);
        }
        for (0..2) |i| {
            try gens[i].mtpRoundVerify(&states[i].?);
            const r = (try gens[i].mtpRoundFinish(a, &states[i].?)) orelse return error.MtpRoundDeclined;
            a.free(r.tokens);
        }
    }
}

test "acceptance failure restores every unfinished row after verify returns" {
    const model_dir = std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var config = try model_mod.parseConfig(io, a, std.mem.span(model_dir));
    defer if (config.ngram_table_path) |path| a.free(path);
    var weights = try model_mod.loadWeights(io, a, std.mem.span(model_dir));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Transformer.init(io, a, config, &weights);
    defer xfm.deinit();
    xfm.compileQwen4Hc();
    xfm.compileGdnGate();
    xfm.compileMoeRouting();
    var tok = Tokenizer.initEmptyForTests(a, .byte_level_bpe);
    defer tok.deinit();
    var prompt: [96]u32 = undefined;
    for (&prompt, 0..) |*id, i| id.* = @intCast(1 + (i * 7919) % 40000);
    var slots: [2]?*MtpChainTestSlot = @splat(null);
    defer for (slots) |slot| {
        if (slot) |p| p.deinit(a);
    };
    var states: [2]?Generator.MtpRoundState = @splat(null);
    defer for (&states) |*state| {
        if (state.*) |*p| p.deinit(a);
    };
    var gens: [2]*Generator = undefined;
    var ptrs: [2]*Generator.MtpRoundState = undefined;
    var before: [2]usize = undefined;
    var old_keys: [2]mlx.mlx_array = @splat(.{ .ctx = null });
    var layers: [2]usize = undefined;
    var saved_ssm: [2][]transformer_mod.SSMCacheEntrySnapshot = undefined;
    var saved_rows: usize = 0;
    defer for (saved_ssm[0..saved_rows]) |snapshots| {
        for (snapshots) |*snapshot| transformer_mod.ssmSnapshotDeinit(snapshot);
        a.free(snapshots);
    };
    const Prefix = struct {
        fn equal(left: mlx.mlx_array, right: mlx.mlx_array, len: usize, s: mlx.mlx_stream) !void {
            const lsh = mlx.getShape(left);
            const rsh = mlx.getShape(right);
            var left_slice = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(left_slice);
            var right_slice = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(right_slice);
            try mlx.check(mlx.mlx_slice(&left_slice, left, &.{ 0, 0, 0, 0 }, 4, &.{ lsh[0], lsh[1], @intCast(len), lsh[3] }, 4, &.{ 1, 1, 1, 1 }, 4, s));
            try mlx.check(mlx.mlx_slice(&right_slice, right, &.{ 0, 0, 0, 0 }, 4, &.{ rsh[0], rsh[1], @intCast(len), rsh[3] }, 4, &.{ 1, 1, 1, 1 }, 4, s));
            try MtpChainTestSlot.equal(left_slice, right_slice, s, "KV valid prefix");
        }
        fn state(saved: []const transformer_mod.SSMCacheEntrySnapshot, current: []const transformer_mod.SSMCacheEntry, s: mlx.mlx_stream) !void {
            try testing.expectEqual(saved.len, current.len);
            for (saved, current) |snapshot, entry| {
                try testing.expectEqual(snapshot.initialized, entry.initialized);
                try MtpChainTestSlot.equal(snapshot.conv_state, entry.conv_state, s, "restored conv state");
                try MtpChainTestSlot.equal(snapshot.ssm_state, entry.ssm_state, s, "restored SSM state");
                try MtpChainTestSlot.equal(snapshot.aux_state, entry.aux_state, s, "restored QSA/PLE state");
                try MtpChainTestSlot.equal(snapshot.qsa_pooled, entry.qsa_pooled, s, "restored QSA pool");
                try testing.expectEqual(snapshot.qsa_ratio, entry.qsa_ratio);
                try testing.expectEqual(snapshot.ple_prev_valid, entry.ple_prev_valid);
                try testing.expectEqualSlices(u32, &snapshot.ple_prev, &entry.ple_prev);
            }
        }
    };
    defer for (old_keys) |keys| {
        if (keys.ctx != null) _ = mlx.mlx_array_free(keys);
    };
    for (0..2) |i| {
        slots[i] = try MtpChainTestSlot.init(a, io, &xfm, &tok, prompt[0..if (i == 0) @as(usize, 96) else 61], 1);
        const slot = slots[i].?;
        gens[i] = &slot.gen.?;
        states[i] = try gens[i].mtpRoundContinue(a, try MtpChainTestSlot.begin(gens[i], a, 1, false));
        ptrs[i] = &states[i].?;
        before[i] = slot.off;
        for (slot.cache.entries, 0..) |entry, layer| {
            if (!entry.initialized) continue;
            layers[i] = layer;
            old_keys[i] = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_array_set(&old_keys[i], entry.keys));
            break;
        }
        try testing.expect(old_keys[i].ctx != null);
        saved_ssm[i] = try a.alloc(transformer_mod.SSMCacheEntrySnapshot, slot.entries.len);
        for (slot.entries, saved_ssm[i]) |*entry, *snapshot| snapshot.* = transformer_mod.ssmSnapshot(entry);
        saved_rows += 1;
    }
    const syncs = transformer_mod.deferred_ple_group_syncs;
    try Generator.mtpGroupVerify(&gens, &ptrs);
    try testing.expectEqual(@as(u64, 1), transformer_mod.deferred_ple_group_syncs - syncs);
    for (states) |state| try testing.expect(state.?.verify_rollback != null);
    mlx.fault.arm(1);
    const result = Generator.mtpGroupAccept(&gens, &ptrs);
    mlx.fault.disarm();
    try testing.expect(mlx.fault.didFire());
    try testing.expectError(error.MlxError, result);
    for (&states, 0..) |*state, i| {
        state.*.?.deinit(a);
        state.* = null;
        const slot = slots[i].?;
        try testing.expectEqual(before[i], slot.off);
        try testing.expectEqual(before[i], slot.cache.entries[layers[i]].offset);
        try Prefix.equal(old_keys[i], slot.cache.entries[layers[i]].keys, before[i], xfm.s);
        try Prefix.state(saved_ssm[i], slot.entries, xfm.s);
        for (slot.entries) |entry| {
            try testing.expectEqual(@as(u32, 0), entry.spec_ple_len);
            try testing.expect(entry.spec_state_seq.ctx == null);
        }
    }
    for (0..2) |i| {
        states[i] = try gens[i].mtpRoundContinue(a, try MtpChainTestSlot.begin(gens[i], a, 1, false));
        ptrs[i] = &states[i].?;
    }
    try Generator.mtpGroupVerify(&gens, &ptrs);
    try Generator.mtpGroupAccept(&gens, &ptrs);
    try mlx.check(mlx.mlx_synchronize(xfm.s));
    for (&states, 0..) |*state, i| {
        state.*.?.deinit(a);
        state.* = null;
        const slot = slots[i].?;
        try testing.expectEqual(before[i], slot.off);
        try testing.expectEqual(before[i], slot.cache.entries[layers[i]].offset);
        try Prefix.equal(old_keys[i], slot.cache.entries[layers[i]].keys, before[i], xfm.s);
        try Prefix.state(saved_ssm[i], slot.entries, xfm.s);
        for (slot.entries) |entry| {
            try testing.expectEqual(@as(u32, 0), entry.spec_ple_len);
            try testing.expect(entry.spec_state_seq.ctx == null);
        }
    }
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
            gen.mtp_cache = .{ .qwen = try KVCache.init(allocator, 0) };
            defer gen.mtp_cache.?.deinit();
            gen.mtp_cache.?.qwen.step = 9;
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

const L11PlainReplay = struct {
    fn tick(xfm: *Transformer, slots: []const *MtpChainTestSlot, capture: bool) !void {
        const a = testing.allocator;
        var tokens: [4]u32 = undefined;
        var offsets: [4]u32 = undefined;
        var contexts: [4]*ForwardCtx = undefined;
        var params: [4]SamplingParams = undefined;
        for (slots, 0..) |slot, i| {
            const gen = &slot.gen.?;
            tokens[i] = gen.next_token_id;
            offsets[i] = @intCast(slot.off);
            contexts[i] = &gen.ctx;
            params[i] = gen.sampling;
        }
        var hidden: ?[]mlx.mlx_array = null;
        defer if (hidden) |rows| {
            for (rows) |row| _ = mlx.mlx_array_free(row);
            a.free(rows);
        };
        const logits = try xfm.forwardMoeBatchedDecode(tokens[0..slots.len], contexts[0..slots.len], offsets[0..slots.len], if (capture) &hidden else null);
        defer {
            for (logits) |row| _ = mlx.mlx_array_free(row);
            a.free(logits);
        }
        var sampled: [4]i32 = undefined;
        try sampleRows(sampled[0..slots.len], logits, params[0..slots.len], xfm.s);
        for (slots, 0..) |slot, i| {
            const gen = &slot.gen.?;
            try gen.generated_ids.append(a, gen.next_token_id);
            gen.advanceStep(1);
            gen.next_token_id = @intCast(sampled[i]);
            gen.sampling.draw = params[i].draw;
            slot.off += 1;
            gen.mtp_hidden_stale = !capture;
            if (hidden) |rows| {
                if (gen.has_last_hidden) _ = mlx.mlx_array_free(gen.last_hidden);
                gen.last_hidden = rows[i];
                rows[i] = .{ .ctx = null };
                gen.has_last_hidden = true;
            }
        }
    }
};

test "plain to prime resumes the same MTP history and next draft as legacy hidden capture" {
    const path = std.c.getenv("MTP_TEST_MODEL") orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var config = try model_mod.parseConfig(io, a, std.mem.span(path));
    defer if (config.ngram_table_path) |name| a.free(name);
    var weights = try model_mod.loadWeights(io, a, std.mem.span(path));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Transformer.init(io, a, config, &weights);
    defer xfm.deinit();
    xfm.compileQwen4Hc();
    xfm.compileGdnGate();
    xfm.compileMoeRouting();
    var tok = Tokenizer.initEmptyForTests(a, .byte_level_bpe);
    defer tok.deinit();
    const previous = group_planner.enabled_override;
    group_planner.enabled_override = true;
    defer group_planner.enabled_override = previous;
    var prompt: [96]u32 = undefined;
    for (&prompt, 0..) |*id, i| id.* = @intCast(1 + (i * 7919) % 40000);
    var reference: [2]*MtpChainTestSlot = undefined;
    var planned: [2]*MtpChainTestSlot = undefined;
    var built: usize = 0;
    defer for (reference[0..built], planned[0..built]) |ref, candidate| {
        ref.deinit(a);
        candidate.deinit(a);
    };
    for (0..2) |row| {
        const ids = prompt[0..if (row == 0) @as(usize, 96) else 61];
        reference[row] = try MtpChainTestSlot.init(a, io, &xfm, &tok, ids, 2);
        planned[row] = MtpChainTestSlot.init(a, io, &xfm, &tok, ids, 2) catch |err| {
            reference[row].deinit(a);
            return err;
        };
        built += 1;
        for ([_]*MtpChainTestSlot{ reference[row], planned[row] }) |slot| {
            const gen = &slot.gen.?;
            gen.mtp_planner_owned = true;
            gen.mtp_planner_width = 2;
            gen.mtp_batch_head = true;
            const result = (try gen.nextMtp(a)) orelse return error.MissingInitialMtpRound;
            a.free(result.tokens);
            try testing.expect(gen.mtp_hist_stash != null);
            try gen.mtpDetachHead(a, true);
        }
    }
    for (0..48) |_| {
        try L11PlainReplay.tick(&xfm, &reference, true);
        try L11PlainReplay.tick(&xfm, &planned, false);
    }
    for (planned) |slot| try expectError(error.MtpHiddenStale, slot.gen.?.mtpRoundBegin(a));
    try L11PlainReplay.tick(&xfm, &reference, true);
    try L11PlainReplay.tick(&xfm, &planned, true);
    for (reference, planned) |ref, candidate| {
        const rg = &ref.gen.?;
        const cg = &candidate.gen.?;
        try testing.expectEqual(rg.next_token_id, cg.next_token_id);
        try testing.expectEqual(ref.off, candidate.off);
        try MtpChainTestSlot.equal(rg.last_hidden, cg.last_hidden, xfm.s, "resumed hidden");
        const rh = rg.mtp_cache.?.qwen4StateForBatch();
        const ch = cg.mtp_cache.?.qwen4StateForBatch();
        try testing.expectEqual(rh.meta(), ch.meta());
        try MtpChainTestSlot.equal(rh.entry.aux_state, ch.entry.aux_state, xfm.s, "resumed head keys");
        try MtpChainTestSlot.equal(rh.entry.qsa_pooled, ch.entry.qsa_pooled, xfm.s, "resumed head pool");
        var rs = try rg.mtpRoundContinue(a, try MtpChainTestSlot.begin(rg, a, 2, false));
        defer rs.deinit(a);
        var cs = try cg.mtpRoundContinue(a, try MtpChainTestSlot.begin(cg, a, 2, false));
        defer cs.deinit(a);
        for (rs.chain.draft_arrs[0..2], cs.chain.draft_arrs[0..2]) |left, right| {
            try MtpChainTestSlot.equal(left, right, xfm.s, "resumed draft ids");
        }
        try rg.mtpRoundVerify(&rs);
        try cg.mtpRoundVerify(&cs);
        try MtpChainTestSlot.equal(rs.verify_logits, cs.verify_logits, xfm.s, "resumed verify logits");
        rg.mtp_batch_head = true;
        cg.mtp_batch_head = true;
        const rr = (try rg.mtpRoundFinish(a, &rs)) orelse return error.MissingReferenceRound;
        defer a.free(rr.tokens);
        const cr = (try cg.mtpRoundFinish(a, &cs)) orelse return error.MissingPlannedRound;
        defer a.free(cr.tokens);
        try testing.expectEqualSlices(u32, rr.tokens, cr.tokens);
        try testing.expectEqual(rr.accepted_tokens, cr.accepted_tokens);
        try testing.expectEqual(rg.next_token_id, cg.next_token_id);
        try MtpChainTestSlot.equal(rg.mtp_hist_stash.?.ids, cg.mtp_hist_stash.?.ids, xfm.s, "next draft stash ids");
        try MtpChainTestSlot.equal(rg.mtp_hist_stash.?.hidden, cg.mtp_hist_stash.?.hidden, xfm.s, "next draft stash hidden");
    }
}

test "cold grouped MTP head seeds the solo position and preserves valid origins" {
    try testing.expectEqual(@as(c_int, 1), Generator.mtpBatchHeadPosition(-1, 0));
    try testing.expectEqual(@as(c_int, 0), Generator.mtpBatchHeadPosition(0, 0));
    try testing.expectEqual(@as(c_int, 8), Generator.mtpBatchHeadPosition(8, 0));
    try testing.expectEqual(@as(c_int, 12), Generator.mtpBatchHeadPosition(8, 4));
    try testing.expectEqual(@as(c_int, 10), Generator.mtpBatchHeadPosition(1, 9));
}

test "cold grouped MTP rounds detach and resume exactly like solo heads" {
    const path = std.c.getenv("MTP_TEST_MODEL") orelse return error.SkipZigTest;
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var config = try model_mod.parseConfig(io, a, std.mem.span(path));
    defer if (config.ngram_table_path) |name| a.free(name);
    var weights = try model_mod.loadWeights(io, a, std.mem.span(path));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Transformer.init(io, a, config, &weights);
    defer xfm.deinit();
    xfm.compileQwen4Hc();
    xfm.compileGdnGate();
    xfm.compileMoeRouting();
    var tok = Tokenizer.initEmptyForTests(a, .byte_level_bpe);
    defer tok.deinit();
    const prior = group_planner.enabled_override;
    group_planner.enabled_override = true;
    defer group_planner.enabled_override = prior;
    var prompt: [96]u32 = undefined;
    for (&prompt, 0..) |*id, i| id.* = @intCast(1 + (i * 7919) % 40000);
    var solo: [2]*MtpChainTestSlot = undefined;
    var grouped: [2]*MtpChainTestSlot = undefined;
    var built: usize = 0;
    defer for (solo[0..built], grouped[0..built]) |ref, row| {
        ref.deinit(a);
        row.deinit(a);
    };
    for (0..2) |row| {
        const ids = prompt[0..if (row == 0) @as(usize, 96) else 61];
        solo[row] = try MtpChainTestSlot.init(a, io, &xfm, &tok, ids, 1);
        grouped[row] = MtpChainTestSlot.init(a, io, &xfm, &tok, ids, 1) catch |err| {
            solo[row].deinit(a);
            return err;
        };
        built += 1;
        for ([_]*MtpChainTestSlot{ solo[row], grouped[row] }) |slot| {
            const gen = &slot.gen.?;
            try testing.expect(gen.mtp_hist_stash == null and gen.mtp_pre_draft == null);
            gen.mtp_cache.?.activate();
            try xfm.qwen4MtpReset();
            gen.mtp_planner_owned = true;
            gen.mtp_planner_width = 1;
            gen.mtp_batch_head = true;
        }
    }
    for (0..2) |_| {
        for (solo) |slot| {
            const result = (try slot.gen.?.nextMtp(a)) orelse return error.MissingSoloRound;
            a.free(result.tokens);
        }
        var gens: [2]*Generator = undefined;
        var opens: [2]Generator.MtpRoundOpen = undefined;
        var chains: [2]Generator.MtpPreDraft = undefined;
        var owns: [2]bool = @splat(false);
        defer for (&chains, owns) |*chain, owned| {
            if (owned) chain.deinit(a);
        };
        for (grouped, 0..) |slot, row| {
            gens[row] = &slot.gen.?;
            opens[row] = try MtpChainTestSlot.begin(gens[row], a, 1, false);
            gens[row].mtp_batch_head = true;
            chains[row] = opens[row].chain;
            owns[row] = true;
        }
        try Generator.mtpChainBuildBatched(&gens, &chains, 0, 1);
        var states: [2]?Generator.MtpRoundState = @splat(null);
        defer for (&states) |*state| {
            if (state.*) |*value| value.deinit(a);
        };
        var pointers: [2]*Generator.MtpRoundState = undefined;
        for (0..2) |row| {
            opens[row].chain = chains[row];
            owns[row] = false;
            states[row] = try gens[row].mtpRoundContinue(a, opens[row]);
            pointers[row] = &states[row].?;
        }
        try Generator.mtpGroupVerify(&gens, &pointers);
        try Generator.mtpGroupAccept(&gens, &pointers);
        for (gens, pointers) |gen, state| {
            const result = (try gen.mtpRoundFinish(a, state)) orelse return error.MissingGroupedRound;
            a.free(result.tokens);
        }
    }
    for (solo, grouped) |ref, row| {
        const rg = &ref.gen.?;
        const bg = &row.gen.?;
        try testing.expect(bg.mtp_hist_stash.?.off0 > 0);
        try bg.mtpDetachHead(a, true);
        try rg.mtpDetachHead(a, true);
        const rh = rg.mtp_cache.?.qwen4StateForBatch();
        const bh = bg.mtp_cache.?.qwen4StateForBatch();
        try testing.expectEqual(@as(c_int, 1), bh.pos_base);
        try testing.expectEqual(rh.meta(), bh.meta());
        try testing.expectEqualSlices(u32, rg.generated_ids.items, bg.generated_ids.items);
        try MtpChainTestSlot.equal(rg.last_hidden, bg.last_hidden, xfm.s, "cold head committed hidden");
        try MtpChainTestSlot.equal(rh.entry.aux_state, bh.entry.aux_state, xfm.s, "cold head QSA keys");
        try MtpChainTestSlot.equal(rh.entry.qsa_pooled, bh.entry.qsa_pooled, xfm.s, "cold head QSA pool");
        const rr = (try rg.nextMtp(a)) orelse return error.MissingSoloResume;
        defer a.free(rr.tokens);
        const br = (try bg.nextMtp(a)) orelse return error.MissingGroupedResume;
        defer a.free(br.tokens);
        try testing.expectEqualSlices(u32, rr.tokens, br.tokens);
        try testing.expectEqual(rr.accepted_tokens, br.accepted_tokens);
        try MtpChainTestSlot.equal(rg.mtp_hist_stash.?.ids, bg.mtp_hist_stash.?.ids, xfm.s, "cold resumed stash ids");
        try MtpChainTestSlot.equal(rg.mtp_hist_stash.?.hidden, bg.mtp_hist_stash.?.hidden, xfm.s, "cold resumed stash hidden");
    }
}

/// Deterministic Gaussian logits (Box-Muller over an xorshift stream): diffuse
/// enough that the nucleus for top_p 0.95 spans thousands of tokens, which is
/// where a narrow cumulative accumulator loses the tail.
fn samplerTestRow(host: []f32, seed: u64, sigma: f32) void {
    var rng: u64 = seed | 1;
    var i: usize = 0;
    while (i < host.len) : (i += 2) {
        var u: [2]f64 = undefined;
        for (&u) |*d| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            d.* = (@as(f64, @floatFromInt(rng >> 11)) + 1.0) / 9007199254740994.0;
        }
        const r = @sqrt(-2.0 * @log(u[0]));
        const a = 6.283185307179586 * u[1];
        host[i] = @floatCast(r * @cos(a) * @as(f64, sigma));
        if (i + 1 < host.len) host[i + 1] = @floatCast(r * @sin(a) * @as(f64, sigma));
    }
}

/// Nucleus the sampler is meant to keep, in f64: ranked descending, a column is
/// in when less than top_p of the mass outranks it.
fn samplerTestExactKeep(allocator: std.mem.Allocator, row: []const f32, top_p: f32) !usize {
    const sorted = try allocator.alloc(f64, row.len);
    defer allocator.free(sorted);
    for (row, sorted) |x, *d| d.* = @floatCast(x);
    std.mem.sort(f64, sorted, {}, comptime std.sort.desc(f64));
    const mx = sorted[0];
    var z: f64 = 0;
    for (sorted) |d| z += @exp(d - mx);
    var above: f64 = 0;
    var keep: usize = 0;
    for (sorted) |d| {
        if (above >= @as(f64, @floatCast(top_p))) break;
        above += @exp(d - mx) / z;
        keep += 1;
    }
    return keep;
}

/// Read an mlx array back as f32 host values. A slice is a strided VIEW, so the
/// read goes through a contiguous copy and a flat reshape, never the base
/// pointer.
fn samplerTestReadFlat(allocator: std.mem.Allocator, arr: mlx.mlx_array, n: usize, s: mlx.mlx_stream) ![]f32 {
    var as_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(as_f32);
    try mlx.check(mlx.mlx_astype(&as_f32, arr, .float32, s));
    var dense = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(dense);
    try mlx.check(mlx.mlx_contiguous(&dense, as_f32, false, s));
    const flat_shape = [_]c_int{@intCast(n)};
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, dense, &flat_shape, 1, s));
    try mlx.check(mlx.mlx_array_eval(flat));
    const out = try allocator.alloc(f32, n);
    @memcpy(out, mlx.mlx_array_data_float32(flat).?[0..n]);
    return out;
}

/// Bit equality of two mlx arrays read back as f32: both routes end in one
/// `where` over the same logits, so agreeing on the kept columns is agreeing on
/// every byte.
fn samplerTestExpectSameBits(a: mlx.mlx_array, b: mlx.mlx_array, n: usize, s: mlx.mlx_stream) !void {
    const ad = try samplerTestReadFlat(testing.allocator, a, n, s);
    defer testing.allocator.free(ad);
    const bd = try samplerTestReadFlat(testing.allocator, b, n, s);
    defer testing.allocator.free(bd);
    for (ad, bd, 0..) |x, y, i| {
        if (@as(u32, @bitCast(x)) != @as(u32, @bitCast(y))) {
            std.debug.print("mismatch at {d}: {d} vs {d}\n", .{ i, x, y });
            return error.FilteredLogitsDiffer;
        }
    }
}

fn samplerTestKeptCount(row: []const f32) usize {
    var n: usize = 0;
    for (row) |x| {
        if (x != -std.math.inf(f32)) n += 1;
    }
    return n;
}

test "sampler ranks break ties by the lowest original index" {
    // The contract both filters and both of their routes share. mlx's stable
    // merge sort is what delivers it; if a bump breaks stability this fails.
    const s = mlx.gpuStream();
    const data = [_]f32{ 2.0, 5.0, 5.0, 5.0, 1.0, 5.0 };
    const shape = [_]c_int{ 1, 6 };
    const logits = mlx.mlx_array_new_data(&data, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(logits);
    var ranks = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ranks);
    try ranksDescending(&ranks, logits, s);
    var as_f32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(as_f32);
    try mlx.check(mlx.mlx_astype(&as_f32, ranks, .float32, s));
    const got = try samplerTestReadFlat(testing.allocator, as_f32, 6, s);
    defer testing.allocator.free(got);
    // The four tied 5.0s come first in ascending column order, then 2.0, then 1.0.
    try testing.expectEqualSlices(f32, &[_]f32{ 1, 2, 3, 5, 0, 4 }, got);
}

test "applyTopP keeps the exact nucleus on a bf16 vocab row" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const v: usize = 248320;
    const host = try allocator.alloc(f32, v);
    defer allocator.free(host);
    const rounded = try allocator.alloc(f32, v);
    defer allocator.free(rounded);

    for ([_]u64{ 1, 2, 3, 4, 5, 6 }) |seed| {
        samplerTestRow(host, seed *% 0x9E3779B97F4A7C15, 4.0);
        const shape = [_]c_int{ 1, @intCast(v) };
        const f32_row = mlx.mlx_array_new_data(host.ptr, &shape, 2, .float32);
        defer _ = mlx.mlx_array_free(f32_row);
        var bf16_row = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(bf16_row);
        try mlx.check(mlx.mlx_astype(&bf16_row, f32_row, .bfloat16, s));

        // The reference reads the row the kernel sees: bf16-rounded values.
        var rt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(rt);
        try mlx.check(mlx.mlx_astype(&rt, bf16_row, .float32, s));
        const rt_host = try samplerTestReadFlat(allocator, rt, v, s);
        defer allocator.free(rt_host);
        @memcpy(rounded, rt_host);
        const want = try samplerTestExactKeep(allocator, rounded, 0.95);

        var res = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(res);
        try applyTopP(&res, bf16_row, 0.95, 0, s);
        const res_host = try samplerTestReadFlat(allocator, res, v, s);
        defer allocator.free(res_host);
        // The boundary column's own probability is ~1e-5 here while the f32 scan
        // carries ~1e-7 of error, so the exclusive mass can cross top_p one
        // column early or late against the f64 reference. The defect this pins
        // was 148 columns wide.
        const got = samplerTestKeptCount(res_host);
        const diff = @as(i64, @intCast(want)) - @as(i64, @intCast(got));
        try testing.expect(diff >= -1 and diff <= 1);
    }
}

test "applyTopP at top_p 0 keeps exactly the argmax" {
    // The nucleus is "mass strictly above the rank < top_p"; rank 0 sees 0, so a
    // literal 0 kept nothing and the sampler drew uniformly from the vocabulary.
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const v: usize = 8192;
    const host = try allocator.alloc(f32, v);
    defer allocator.free(host);
    samplerTestRow(host, 0x9E3779B97F4A7C15, 4.0);
    const shape = [_]c_int{ 1, @intCast(v) };
    const row = mlx.mlx_array_new_data(host.ptr, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(row);
    for ([_]u32{ 0, 40 }) |bound| {
        var res = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(res);
        try applyTopP(&res, row, 0.0, bound, s);
        const res_host = try samplerTestReadFlat(allocator, res, v, s);
        defer allocator.free(res_host);
        try testing.expectEqual(@as(usize, 1), samplerTestKeptCount(res_host));
        var best: usize = 0;
        for (host, 0..) |x, i| if (x > host[best]) {
            best = i;
        };
        try testing.expect(std.math.isFinite(res_host[best]));
    }
}

test "topRanksDescending returns the exact top m values and column ids" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    for ([_]usize{ 8192, 248320 }) |v| {
        const rows: usize = 3;
        const host = try allocator.alloc(f32, rows * v);
        defer allocator.free(host);
        for (0..rows) |r| samplerTestRow(host[r * v ..][0..v], (r + 1) *% 0xD1B54A32D192ED03, 4.0);
        // Row 2 quantized to 0.25 steps: thousands of exact ties.
        for (host[2 * v ..][0..v]) |*x| x.* = @round(x.* * 4.0) / 4.0;
        const shape = [_]c_int{ @intCast(rows), @intCast(v) };
        const block = mlx.mlx_array_new_data(host.ptr, &shape, 2, .float32);
        defer _ = mlx.mlx_array_free(block);

        for ([_]u32{ 1, 20, 257 }) |m| {
            var vals = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(vals);
            var ids = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ids);
            try topRanksDescending(&vals, &ids, block, m, s);
            // The whole row's ranking, restricted to its first m: the bar the
            // shortlist has to reproduce, ties included.
            var order = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(order);
            try ranksDescending(&order, block, s);
            var ref_ids = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ref_ids);
            try sliceLastAxis(&ref_ids, order, 0, m, s);
            var ref_vals = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ref_vals);
            try mlx.check(mlx.mlx_take_along_axis(&ref_vals, block, ref_ids, -1, s));
            var ids_f = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ids_f);
            try mlx.check(mlx.mlx_astype(&ids_f, ids, .float32, s));
            var ref_ids_f = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(ref_ids_f);
            try mlx.check(mlx.mlx_astype(&ref_ids_f, ref_ids, .float32, s));
            try samplerTestExpectSameBits(vals, ref_vals, rows * m, s);
            try samplerTestExpectSameBits(ids_f, ref_ids_f, rows * m, s);
        }
    }
}

test "applyTopK keeps exactly k columns, the highest ranked ones" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const v: usize = 248320;
    const rows: usize = 3;
    const host = try allocator.alloc(f32, rows * v);
    defer allocator.free(host);
    for (0..rows) |r| samplerTestRow(host[r * v ..][0..v], (r + 7) *% 0xD1B54A32D192ED03, 4.0);
    // Row 1 quantized to 1.0 steps: a value cutoff would keep thousands here.
    for (host[v..][0..v]) |*x| x.* = @round(x.*);
    const shape = [_]c_int{ @intCast(rows), @intCast(v) };
    const block = mlx.mlx_array_new_data(host.ptr, &shape, 2, .float32);
    defer _ = mlx.mlx_array_free(block);

    for ([_]u32{ 1, 20, 64 }) |k| {
        var res = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(res);
        try applyTopK(&res, block, k, s);
        const rd = try samplerTestReadFlat(allocator, res, rows * v, s);
        defer allocator.free(rd);
        for (0..rows) |r| {
            try testing.expectEqual(@as(usize, k), samplerTestKeptCount(rd[r * v ..][0..v]));
        }
        // Same columns as ranking the whole row.
        var order = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(order);
        try ranksDescending(&order, block, s);
        var ref_ids = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ref_ids);
        try sliceLastAxis(&ref_ids, order, 0, k, s);
        var ref = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ref);
        try keepByIndex(&ref, block, ref_ids, s);
        try samplerTestExpectSameBits(res, ref, rows * v, s);
    }
}

test "shortlist top-p equals the whole row's ranking, ties and 248320-wide rows" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    for ([_]usize{ 8192, 248320 }) |v| {
        const rows: usize = 4;
        const host = try allocator.alloc(f32, rows * v);
        defer allocator.free(host);
        for (0..rows) |r| samplerTestRow(host[r * v ..][0..v], (r + 11) *% 0xD1B54A32D192ED03, 4.0);
        // Row 2: 0.25-step ties. Row 3: every column identical — the extreme the
        // rank contract exists for.
        for (host[2 * v ..][0..v]) |*x| x.* = @round(x.* * 4.0) / 4.0;
        for (host[3 * v ..][0..v]) |*x| x.* = 1.5;

        const shape = [_]c_int{ @intCast(rows), @intCast(v) };
        const block = mlx.mlx_array_new_data(host.ptr, &shape, 2, .float32);
        defer _ = mlx.mlx_array_free(block);

        for ([_]struct { p: f32, k: u32 }{
            .{ .p = 0.95, .k = 20 },
            .{ .p = 0.8, .k = 50 },
            .{ .p = 0.99, .k = 5 },
            .{ .p = 0.5, .k = 1 },
            .{ .p = 0.95, .k = 257 },
            .{ .p = 1e-6, .k = 20 },
        }) |cfg| {
            var masked = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(masked);
            try applyTopK(&masked, block, cfg.k, s);

            var short = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(short);
            try applyTopP(&short, masked, cfg.p, cfg.k, s);
            var full = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(full);
            try applyTopP(&full, masked, cfg.p, 0, s);
            try samplerTestExpectSameBits(short, full, rows * v, s);

            // Same filtered logits and the same key must draw the same token.
            for (0..8) |d| {
                const key = seedKey(.{ .seed = 12345, .draw = @intCast(d) });
                defer _ = mlx.mlx_array_free(key);
                var a = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(a);
                var b = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(b);
                try mlx.check(mlx.mlx_random_categorical(&a, short, -1, key, s));
                try mlx.check(mlx.mlx_random_categorical(&b, full, -1, key, s));
                try samplerTestExpectSameBits(a, b, rows, s);
            }
        }
    }
}

test "shortlist top-p holds on rank-3 [B, L, V] blocks" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const v: usize = 8192;
    const rows: usize = 3;
    const host = try allocator.alloc(f32, rows * v);
    defer allocator.free(host);
    for (0..rows) |r| samplerTestRow(host[r * v ..][0..v], (r + 23) *% 0xD1B54A32D192ED03, 4.0);
    const shape = [_]c_int{ 1, @intCast(rows), @intCast(v) };
    const block = mlx.mlx_array_new_data(host.ptr, &shape, 3, .float32);
    defer _ = mlx.mlx_array_free(block);

    var masked = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(masked);
    try applyTopK(&masked, block, 20, s);
    var short = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(short);
    try applyTopP(&short, masked, 0.95, 20, s);
    var full = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(full);
    try applyTopP(&full, masked, 0.95, 0, s);
    try samplerTestExpectSameBits(short, full, rows * v, s);

    // Per-ROW cutoffs: a block-wide filter would mask whole rows to -inf.
    const sd = try samplerTestReadFlat(allocator, short, rows * v, s);
    defer allocator.free(sd);
    for (0..rows) |r| {
        const kept = samplerTestKeptCount(sd[r * v ..][0..v]);
        try testing.expect(kept > 0 and kept <= 20);
    }
}

test "a group's joined probability block slices row-for-row into the per-row filter" {
    // Ragged widths, three rows: the axis-bug class ("one row cannot reveal an
    // axis bug"), and the offsets the projection's own walk produces.
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const v: usize = 8192;
    const widths = [_]u32{ 3, 2, 4 };
    const sampling = SamplingParams{ .temperature = 1.0, .top_p = 0.95, .top_k = 20 };

    var rows: [widths.len]mlx.mlx_array = undefined;
    var built: usize = 0;
    defer for (rows[0..built]) |a| {
        _ = mlx.mlx_array_free(a);
    };
    for (widths, 0..) |w, r| {
        const host = try allocator.alloc(f32, @as(usize, w) * v);
        defer allocator.free(host);
        for (0..w) |p| samplerTestRow(host[p * v ..][0..v], (r * 8 + p + 1) *% 0xD1B54A32D192ED03, 4.0);
        const shape = [_]c_int{ 1, @intCast(w), @intCast(v) };
        const f32_rows = mlx.mlx_array_new_data(host.ptr, &shape, 3, .float32);
        defer _ = mlx.mlx_array_free(f32_rows);
        rows[r] = mlx.mlx_array_new();
        built = r + 1;
        // bf16 like the verify block the lm_head produces.
        try mlx.check(mlx.mlx_astype(&rows[r], f32_rows, .bfloat16, s));
    }

    var joined = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(joined);
    {
        const vec = mlx.mlx_vector_array_new_data(&rows, widths.len);
        defer _ = mlx.mlx_vector_array_free(vec);
        try mlx.check(mlx.mlx_concatenate_axis(&joined, vec, 1, s));
    }
    const block = try groupProbsBlock(joined, sampling, s);
    defer _ = mlx.mlx_array_free(block);

    var offset: c_int = 0;
    for (widths, 0..) |w, r| {
        const got = try groupProbsRow(block, offset, w, s);
        defer _ = mlx.mlx_array_free(got);
        const want = try probsAllPositions(rows[r], sampling, s);
        defer _ = mlx.mlx_array_free(want);
        const g_shape = mlx.getShape(got);
        try testing.expectEqual(@as(c_int, 1), g_shape[0]);
        try testing.expectEqual(@as(c_int, @intCast(w)), g_shape[1]);
        try samplerTestExpectSameBits(got, want, @as(usize, w) * v, s);
        offset += @intCast(w);
    }
}

test "the group accept block declines a mixed sampler and any seeded row" {
    const a = SamplingParams{ .temperature = 1.0, .top_p = 0.95, .top_k = 20 };
    var b = a;
    b.top_k = 40;
    var seeded = a;
    seeded.seed = 7;
    try testing.expect(Generator.mtpGroupAcceptBatchable(&.{ a, a }));
    try testing.expect(!Generator.mtpGroupAcceptBatchable(&.{ a, b }));
    try testing.expect(!Generator.mtpGroupAcceptBatchable(&.{ a, seeded }));
    try testing.expect(!Generator.mtpGroupAcceptBatchable(&.{}));
}

/// The composition every sampling site ran before `filterTopKTopP`: top-k, then
/// top-p bounded by that same k. The reference the one-pass filter is pinned to.
fn samplerTestComposed(res: *mlx.mlx_array, logits: mlx.mlx_array, top_p: f32, top_k: u32, s: mlx.mlx_stream) !void {
    var after_k = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(after_k);
    if (top_k > 0) {
        try applyTopK(&after_k, logits, top_k, s);
    } else {
        try mlx.check(mlx.mlx_array_set(&after_k, logits));
    }
    if (top_p < 1.0) return applyTopP(res, after_k, top_p, top_k, s);
    return mlx.check(mlx.mlx_array_set(res, after_k));
}

test "filterTopKTopP is byte-identical to the two-pass composition" {
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const Setting = struct { p: f32, k: u32 };
    const settings = [_]Setting{
        .{ .p = 0.95, .k = 20 }, .{ .p = 0.9, .k = 1 },  .{ .p = 1.0, .k = 20 },
        .{ .p = 0.95, .k = 0 },  .{ .p = 0.5, .k = 50 }, .{ .p = 0.95, .k = 999999 },
    };
    for ([_]usize{ 8192, 248320 }) |v| {
        const rows: usize = 3;
        const host = try allocator.alloc(f32, rows * v);
        defer allocator.free(host);
        samplerTestRow(host[0..v], 0x9E3779B97F4A7C15, 4.0);
        // Row 1 quantized to 0.25 steps: thousands of exact ties at the cutoff.
        samplerTestRow(host[v..][0..v], 0xD1B54A32D192ED03, 4.0);
        for (host[v..][0..v]) |*x| x.* = @round(x.* * 4.0) / 4.0;
        // Row 2 all equal: every column ties, so the ranking alone decides.
        @memset(host[2 * v ..][0..v], 1.25);

        // `[m, V]` and `[1, L, V]`: the mask must stay per ROW on both ranks.
        for ([_]usize{ 2, 3 }) |rank| {
            const shape2 = [_]c_int{ @intCast(rows), @intCast(v) };
            const shape3 = [_]c_int{ 1, @intCast(rows), @intCast(v) };
            const shape: []const c_int = if (rank == 2) &shape2 else &shape3;
            const f32_block = mlx.mlx_array_new_data(host.ptr, shape.ptr, @intCast(shape.len), .float32);
            defer _ = mlx.mlx_array_free(f32_block);
            var block = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(block);
            try mlx.check(mlx.mlx_astype(&block, f32_block, .bfloat16, s));
            for (settings) |cfg| {
                var want = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(want);
                try samplerTestComposed(&want, block, cfg.p, cfg.k, s);
                var got = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(got);
                try filterTopKTopP(&got, block, cfg.p, cfg.k, s);
                samplerTestExpectSameBits(got, want, rows * v, s) catch |e| {
                    std.debug.print("v={d} rank={d} top_p={d} top_k={d}\n", .{ v, rank, cfg.p, cfg.k });
                    return e;
                };
            }
        }
    }
}

test "no block decoder commits an argmax over a suppressed id" {
    // A row whose raw argmax is a padding id past the defined vocabulary, plus a
    // flagged special: both must lose to the best legal column. Run at the block
    // widths the committing decoders verify at (MTP and the gemma drafter `1 + m`,
    // DFlash a whole block), so a per-position axis bug cannot hide. WHICH decoders
    // are covered is a compile-time property: every commit site takes a
    // `CommittedArgmax`, and only `verifyArgmax` returns one.
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const v: usize = 512;
    const defined: usize = 500;
    const flagged = [_]u32{7};
    for ([_]usize{ 2, 3, 5 }) |rows| try suppressedArgmaxCase(allocator, s, v, defined, &flagged, rows);
}

fn suppressedArgmaxCase(allocator: std.mem.Allocator, s: mlx.mlx_stream, v: usize, defined: usize, flagged: []const u32, rows: usize) !void {
    const host = try allocator.alloc(f32, rows * v);
    defer allocator.free(host);
    @memset(host, 0.0);
    for (0..rows) |r| {
        const row = host[r * v ..][0..v];
        row[505] = 9.0; // padding row: highest of all
        row[7] = 8.0; // flagged special: second
        row[100 + r] = 7.0; // the best LEGAL id, different per position
    }
    const shape = [_]c_int{ 1, @intCast(rows), @intCast(v) };
    const f32_block = mlx.mlx_array_new_data(host.ptr, &shape, 3, .float32);
    defer _ = mlx.mlx_array_free(f32_block);
    var block = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(block);
    try mlx.check(mlx.mlx_astype(&block, f32_block, .bfloat16, s));
    const mask = try buildSuppressMask(flagged, defined, v, s);
    defer _ = mlx.mlx_array_free(mask);

    var raw = try verifyArgmax(block, null, s);
    defer raw.deinit();
    const raw_ids = try samplerTestReadFlat(allocator, raw.lazy(), rows, s);
    defer allocator.free(raw_ids);
    for (raw_ids) |id| try testing.expectEqual(@as(f32, 505), id);

    var masked = try verifyArgmax(block, mask, s);
    defer masked.deinit();
    const ids = try samplerTestReadFlat(allocator, masked.lazy(), rows, s);
    defer allocator.free(ids);
    for (ids, 0..) |id, r| try testing.expectEqual(@as(f32, @floatFromInt(100 + r)), id);
}

test "serial and MTP agree at the greedy temperature cutoff" {
    const cutoff: f32 = 0.01;
    const bits: u32 = @bitCast(cutoff);
    for ([_]f32{ 0.0, 0.005, @bitCast(bits - 1), cutoff, @bitCast(bits + 1), 0.6 }) |temperature| {
        const sampling = SamplingParams{ .temperature = temperature };
        const serial_greedy = sampleRowsAllGreedy(&.{sampling});
        const mtp_greedy = Generator.mtpDraftSamplingFor(sampling, false, 0.6).temperature == 0.0;
        try testing.expectEqual(temperature < cutoff, serial_greedy);
        try testing.expectEqual(serial_greedy, mtp_greedy);
    }
}

test "mtpLookupAllowed: the qwen4 and MiMo heads, also at a pinned depth, never a batched head or a planner-owned round" {
    const off = mtp_lookup_disabled;
    const bounds_before = depth_bounds.active;
    const planner = group_planner.enabled_override;
    defer {
        mtp_lookup_disabled = off;
        depth_bounds.active = bounds_before;
        group_planner.enabled_override = planner;
    }
    mtp_lookup_disabled = false;
    depth_bounds.active = .{};
    group_planner.enabled_override = true;
    var g: Generator = undefined;
    g.mtp = .{ .qwen4 = undefined };
    g.mtp_batch_head = false;
    g.mtp_planner_owned = false;
    try testing.expect(g.mtpLookupAllowed());

    g.mtp = .{ .mimo = undefined };
    try testing.expect(g.mtpLookupAllowed());
    g.mtp = .{ .qwen = undefined };
    try testing.expect(!g.mtpLookupAllowed());
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

    mtp_lookup_disabled = true;
    try testing.expect(!g.mtpLookupAllowed());
}

test "--no-mtp-lookup: a GLM DFlash2 lookup chain is declined too" {
    const off = mtp_lookup_disabled;
    defer mtp_lookup_disabled = off;
    mtp_lookup_disabled = true;
    var g: Generator = undefined;
    try testing.expect((try g.glmLookupChain(testing.allocator)) == null);
}

test "mtpLookupDraftCap: a MiMo lookup drafts no wider than its verify keeps decode rows" {
    try testing.expectEqual(Generator.mtpVerifyDraftsMax(true), Generator.mtpLookupDraftCap(true));
    try testing.expectEqual(mtp_lookup.MAX_DRAFT_STRONG, Generator.mtpLookupDraftCap(false));
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
    const got = idx.match(1, Generator.mtpLookupDraftCap(true)) orelse return error.NoMatch;
    var free = Generator.mtpLookupCostsFor(Generator.MtpCostSource.init(Generator.MTP_EV_DEFAULT_COSTS, 1000, null), 3);
    for (&free.lookup) |*c| c.* = 0.01;
    try testing.expectEqual(Generator.mtpVerifyDraftsMax(true), mtp_lookup.gate(got, 1000, 8, 0, true, free));
}

test "mtpLookupCostsFor: a lookup is the MTP round at its draft count minus the head steps, until the lookup row is measured" {
    const costs = Generator.MTP_EV_DEFAULT_COSTS;
    const prior = Generator.MtpCostSource.init(costs, 1000, null);
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
    const live = Generator.MtpCostSource.init(costs, 1000, &t);
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
    const two_chunk = Generator.MtpRoundPlan{ .m_lo = 1, .m_hi = 3, .tau_ln = 0 };
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
    const src = Generator.MtpCostSource.init(Generator.MTP_EV_MIMO_EXL3_COSTS, 1000, &t);
    try testing.expect(src.fromTable());
    var d: [20]u32 = undefined;
    for (&d, 0..) |*x, i| x.* = @intCast(i);
    const copy = mtp_lookup.Match{ .draft = d[0..7], .suffix = mtp_lookup.STRONG_SUFFIX };
    try testing.expectEqual(@as(u32, 0), mtp_lookup.gate(copy, 1000, 6.5, 2.7, false, Generator.mtpLookupCostsFor(src, two_chunk.m_lo)));
    const width = Generator.mtpLookupPriceWidth(two_chunk, 2.9);
    try testing.expectEqual(@as(u32, 7), mtp_lookup.gate(copy, 1000, 6.5, 2.7, false, Generator.mtpLookupCostsFor(src, width)));
}

test "upstream bugfix: constrained logprobs describe the returned token before masking" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    const s = mlx.gpuStream();
    // Each call is the final allowed token, so no model forward is needed.
    var xfm: Transformer = undefined;
    xfm.s = s;
    var bytes = try token_mask.TokenBytes.init(std.heap.ArenaAllocator.init(a), &.{ null, "true", "false", "invalid" }, null);
    defer bytes.deinit();
    const parsed = try std.json.parseFromSlice(std.json.Value, a, "{\"type\":\"boolean\"}", .{});
    defer parsed.deinit();
    var gen = Generator{
        .xfm = &xfm,
        .ctx = undefined,
        .tok = undefined,
        .next_token_id = 0,
        .step = 0,
        .max_tokens = 1,
        .sampling = .{ .temperature = 0 },
        .prompt_tokens = 0,
        .completion_tokens = 0,
        .finish_reason = "length",
        .done = false,
        .eos_token_ids = &.{},
        .generated_ids = .empty,
        .timeout_ns = 0,
        .timer = io_util.Stopwatch.init(testing.io),
        .last_hidden = .{ .ctx = null },
        .has_last_hidden = false,
        .logprobs_n = 4,
    };
    defer gen.generated_ids.deinit(a);
    defer if (gen.last_logprob) |lp| a.free(lp.top_logprobs);
    defer if (gen.has_pending_logits) {
        _ = mlx.mlx_array_free(gen.pending_logits);
    };
    for ([_][4]f32{ .{ 0, 2, 1, 4 }, .{ 0, 1, 3, 5 } }, [_]u32{ 1, 2 }) |values, want| {
        var sc: SchemaConstraint = undefined;
        try sc.initFromValue(a, parsed.value, &bytes);
        defer sc.deinit();
        gen.sampling.constraint = &sc.constraint;
        gen.max_tokens = gen.step + 1;
        gen.done = false;
        gen.pending_logits = mlx.mlx_array_new_data(&values, &.{ 1, 1, 4 }, 3, .float32);
        gen.has_pending_logits = true;
        const expected = try computeLogprobs(a, gen.pending_logits, want, 4, s);
        defer a.free(expected.top_logprobs);
        try testing.expectEqual(@as(?u32, want), try gen.nextConstrained(a));
        const got = gen.last_logprob orelse return error.NoLogprobs;
        try testing.expectApproxEqAbs(expected.token_logprob, got.token_logprob, 1e-6);
        try testing.expectEqual(@as(u32, 3), got.top_logprobs[0].token_id);
        try testing.expectEqual(@as(?u32, null), try gen.nextConstrained(a));
    }
    try testing.expectEqualSlices(u32, &.{ 1, 2 }, gen.generated_ids.items);
}

pub fn decodeShareCapped(width: usize, cap: usize) usize {
    return if (cap == 0) width else @min(width, cap);
}

test "decodeShareCapped: the share cap narrows an explicit and an env width" {
    const saved_override = prefill_chunk_override;
    const saved_explicit = prefill_chunk_explicit;
    defer {
        prefill_chunk_override = saved_override;
        prefill_chunk_explicit = saved_explicit;
    }
    prefill_chunk_override = 8192;
    prefill_chunk_explicit = true;
    const explicit = effectivePrefillChunk(128, 8, 1024, false, false, false, 2048);
    try std.testing.expectEqual(@as(usize, 8192), explicit);
    try std.testing.expectEqual(@as(usize, 1024), decodeShareCapped(explicit, 1024));
    try std.testing.expectEqual(@as(usize, 8192), decodeShareCapped(explicit, 0));
}

/// Records what `mimoStepOut` frees instead of freeing it, so a double free is a count, not a crash.
const StepOutFrees = struct {
    var seen: [8]?*anyopaque = @splat(null);
    var n: usize = 0;
    fn record(a: mlx.mlx_array) void {
        seen[n] = a.ctx;
        n += 1;
    }
    fn count(ctx: ?*anyopaque) usize {
        var c: usize = 0;
        for (seen[0..n]) |x| c += @intFromBool(x == ctx);
        return c;
    }
    fn releaseAll() void {
        for (seen[0..n], 0..) |x, i| {
            if (std.mem.indexOfScalar(?*anyopaque, seen[0..i], x) == null) _ = mlx.mlx_array_free(.{ .ctx = x });
        }
        n = 0;
    }
};

test "mimoStepOut: a failed lm_head projection frees the step's arrays once each" {
    const Failing = struct {
        fn lmHeadLogits(_: @This(), _: mlx.mlx_array) !mlx.mlx_array {
            return error.InjectedFault;
        }
    };
    StepOutFrees.n = 0;
    defer StepOutFrees.releaseAll();
    const last = mlx.mlx_array_new_int(7);
    try testing.expectError(error.InjectedFault, MtpHeadRef.mimoStepOut(Failing{}, StepOutFrees.record, last, 1, .logits));
    try testing.expectEqual(@as(usize, 1), StepOutFrees.count(last.ctx));
    try testing.expectEqual(@as(usize, 2), StepOutFrees.n);
}

test "mimoStepOut: logits consume the row once; a rerank step hands it over" {
    const Ok = struct {
        fn lmHeadLogits(_: @This(), _: mlx.mlx_array) !mlx.mlx_array {
            return mlx.mlx_array_new_int(3);
        }
    };
    StepOutFrees.n = 0;
    defer StepOutFrees.releaseAll();
    const last = mlx.mlx_array_new_int(7);
    const out = try MtpHeadRef.mimoStepOut(Ok{}, StepOutFrees.record, last, 2, .logits);
    defer _ = mlx.mlx_array_free(out.logits);
    defer _ = mlx.mlx_array_free(out.hidden_next);
    try testing.expectEqual(@as(usize, 1), StepOutFrees.count(last.ctx));
    try testing.expect(out.rerank_x.ctx == null);

    const row = mlx.mlx_array_new_int(8);
    const mixed = try MtpHeadRef.mimoStepOut(Ok{}, StepOutFrees.record, row, 2, .mixed);
    defer _ = mlx.mlx_array_free(mixed.hidden_next);
    defer _ = mlx.mlx_array_free(mixed.rerank_x);
    try testing.expectEqual(@as(usize, 0), StepOutFrees.count(row.ctx));
    try testing.expect(mixed.rerank_x.ctx == row.ctx and mixed.logits.ctx == null);
}

test "argmax-only requests: greedy with nothing that reads past the argmax" {
    const greedy: SamplingParams = .{ .temperature = 0.0 };
    try std.testing.expect(argmaxOnlyRequest(greedy, 0));
    try std.testing.expect(argmaxOnlyRequest(.{ .temperature = 0.8, .top_k = 1 }, 0));
    try std.testing.expect(!argmaxOnlyRequest(.{ .temperature = 0.7 }, 0));
    try std.testing.expect(!argmaxOnlyRequest(.{ .temperature = 0.0, .repeat_penalty = 1.1 }, 0));
    try std.testing.expect(!argmaxOnlyRequest(.{ .temperature = 0.0, .presence_penalty = 0.5 }, 0));
    try std.testing.expect(!argmaxOnlyRequest(.{ .temperature = 0.0, .think_penalty = .{ .lambda = 1 } }, 0));
    try std.testing.expect(!argmaxOnlyRequest(greedy, 5));
    var cf: CallForce = undefined;
    try std.testing.expect(!argmaxOnlyRequest(.{ .temperature = 0.0, .call_force = &cf }, 0));
    var c: Constraint = undefined;
    try std.testing.expect(!argmaxOnlyRequest(.{ .temperature = 0.0, .constraint = &c }, 0));
}

test "a lazy penalty allocation failure fails the request instead of sampling unpenalized" {
    const s = mlx.gpuStream();
    const host = [_]f32{ 5, 4, 3, 2 };
    const logits = mlx.mlx_array_new_data(&host, &[_]c_int{ 1, 1, 4 }, 3, .float32);
    defer _ = mlx.mlx_array_free(logits);
    for (0..2) |fail_index| {
        var failing = testing.FailingAllocator.init(testing.allocator, .{ .fail_index = fail_index });
        try testing.expectError(error.OutOfMemory, penalizedSampleLazy(failing.allocator(), logits, .{ .temperature = 0, .presence_penalty = 2 }, &.{0}, s));
    }
}

pub fn installLogitBias(io: std.Io, xfm: *Transformer, tok: *const Tokenizer) !void {
    const setting = if (xfm.config.logit_bias_file_override) |*p| p.slice() else null;
    const path = setting orelse return;
    const body = try std.Io.Dir.cwd().readFileAlloc(io, path, xfm.allocator, .limited(16 << 20));
    defer xfm.allocator.free(body);
    const logits_dim: usize = xfm.config.vocab_size;
    const loaded = logit_bias.parse(xfm.allocator, body, std.fs.path.extension(path), tok, @min(logits_dim, tok.definedVocabSize())) catch |err| {
        log.err("[logit-bias] {s}: {s}\n", .{ path, @errorName(err) });
        return err;
    };
    xfm.logit_bias = loaded.biases;
    log.info("[logit-bias] {s}: {d} entries, {d} expanded ids, {d} multi-token spellings skipped\n", .{ path, loaded.entries, loaded.biases.len, loaded.skipped });
}

test "logit bias CPU: rewards and penalties require the full head while zero stays off" {
    const t = std.testing;
    for ([_]f32{ -2, 2 }) |delta| {
        const entries = [_]logit_bias.Bias{.{ .id = 1, .delta = delta }};
        const sampling = SamplingParams{ .temperature = 0, .think_penalty = .{ .biases = &entries } };
        try t.expect(!argmaxOnlyRequest(sampling, 0));
        var from_file = sampling;
        from_file.think_penalty.biases = &.{};
        from_file.think_penalty.file_biases = &entries;
        try t.expect(!argmaxOnlyRequest(from_file, 0));
    }
    const zero = [_]logit_bias.Bias{.{ .id = 1, .delta = 0 }};
    try t.expect(argmaxOnlyRequest(.{ .temperature = 0, .think_penalty = .{ .biases = &zero } }, 0));
    try t.expect(argmaxOnlyRequest(.{ .temperature = 0 }, 0));
}


test "a GLM lookup chain needs a long verbatim match and the budget for all four rows" {
    const draft = [_]u32{ 7, 8, 9, 10 };
    const strong = mtp_lookup.Match{ .draft = &draft, .suffix = mtp_lookup.STRONG_SUFFIX };
    const chain = glmLookupProposal(strong, 6, 64).?;
    try testing.expectEqualSlices(u32, &.{ 6, 7, 8, 9 }, chain.tokens[0..chain.count]);
    try testing.expectEqualSlices(i32, &.{ -1, 0, 1, 2 }, chain.parents[0..chain.count]);
    try testing.expect(glmLookupProposal(null, 6, 64) == null);
    try testing.expect(glmLookupProposal(.{ .draft = &draft, .suffix = mtp_lookup.STRONG_SUFFIX - 1 }, 6, 64) == null);
    try testing.expect(glmLookupProposal(.{ .draft = draft[0..2], .suffix = mtp_lookup.STRONG_SUFFIX }, 6, 64) == null);
    try testing.expect(glmLookupProposal(strong, 6, 3) == null);
}

test "GLM serving DFlash2 arms only requests the native verifier can reproduce" {
    try testing.expect(glmDflashEligible(.{ .temperature = 0 }, 0));
    try testing.expect(glmDflashEligible(.{ .temperature = 1, .top_k = 1 }, 0));
    try testing.expect(glmDflashEligible(.{ .temperature = 1 }, 0));
    try testing.expect(!glmDflashEligible(.{ .temperature = 0 }, 1));
    try testing.expect(!glmDflashEligible(.{ .temperature = 0, .presence_penalty = 1 }, 0));
    const ban = [_]logit_bias.Bias{.{ .id = 7, .delta = -100 }};
    try testing.expect(glmDflashEligible(.{ .temperature = 0, .think_penalty = .{ .biases = &ban } }, 0));
    try testing.expect(glmDflashEligible(.{ .temperature = 1, .think_penalty = .{ .file_biases = &ban } }, 0));
    try testing.expect(glmDflashEligible(.{ .temperature = 0, .think_penalty = .{ .lambda = 1, .closer_id = 3, .phase = .inside } }, 0));
}

test "GLM snapshot backoff lands the prompt-end checkpoint on a pool boundary" {
    for ([_]usize{ 0, 256, 1024 }) |offset| {
        for ([_]usize{ 31, 32, 33, 34, 35, 100, 299, 4097 }) |prefix_len| {
            const back = glmSnapshotBackoff(ssmSnapshotBackoff(true, prefix_len, offset > 0), offset, prefix_len);
            try testing.expect(back >= SSM_SNAPSHOT_BACKOFF and back <= SSM_SNAPSHOT_BACKOFF + 3 or back == prefix_len);
            if (back < prefix_len) try testing.expectEqual(@as(usize, 0), (offset + prefix_len - back) % 4);
        }
    }
    // A restored tail inside the window still forwards as one span.
    try testing.expectEqual(@as(usize, 12), glmSnapshotBackoff(ssmSnapshotBackoff(true, 12, true), 256, 12));
    // A short cold prompt still ends its loop on a pool boundary.
    try testing.expectEqual(@as(usize, 1), glmSnapshotBackoff(ssmSnapshotBackoff(true, 13, false), 0, 13));
}

test "GLM short prompts and odd chunk widths still checkpoint only on pool boundaries" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    var tok = Tokenizer.initEmptyForTests(a, .byte_level_bpe);
    defer tok.deinit();
    var weights = model_mod.Weights.init(a);
    defer weights.deinit();
    const cfg = try glmGeneratorFixture(&weights);
    const greedy = SamplingParams{ .temperature = 0.0 };
    var prompt: [150]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast((i * 5 + 1) % 4);
    glm_checkpoint_stride = 64;
    defer glm_checkpoint_stride = glm5_forward.prefill_chunk;
    for ([_]struct { len: usize, chunk: usize }{ .{ .len = 14, .chunk = 64 }, .{ .len = 150, .chunk = 37 } }) |case| {
        var xfm = try Transformer.init(testing.io, a, cfg, &weights);
        defer xfm.deinit();
        var gen = try Generator.initWithOptions(testing.io, a, &xfm, &tok, prompt[0..case.len], 2, greedy, &.{}, .{ .skip_lazy_preforward = true, .pinned_prefill_chunk = case.chunk, .adaptive_chunk_width = true, .ssm_checkpoint_stride = 1, .ssm_checkpoint_max = 16 });
        defer gen.deinit(a);
        try testing.expect(gen.ssm_checkpoints.items.len > 0);
        for (gen.ssm_checkpoints.items) |cp| try testing.expectEqual(@as(usize, 0), cp.pos % 4);
    }
}

test "an empty prompt is a typed error at the generator, never an index" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    var tok = Tokenizer.initEmptyForTests(a, .byte_level_bpe);
    defer tok.deinit();
    var weights = model_mod.Weights.init(a);
    defer weights.deinit();
    const cfg = try glmGeneratorFixture(&weights);
    var xfm = try Transformer.init(testing.io, a, cfg, &weights);
    defer xfm.deinit();
    try expectError(error.EmptyPrompt, Generator.initWithOptions(testing.io, a, &xfm, &tok, &.{}, 4, .{ .temperature = 0.0 }, &.{}, .{}));
}

fn glmGeneratorFixture(weights: *model_mod.Weights) !model_mod.ModelConfig {
    var cfg = try glm5_forward.nonzeroDecodeFixture(weights);
    cfg.max_position_embeddings = 4096;
    return cfg;
}

test "GLM prefill checkpoints sit on chunk boundaries, and a restore there prefills its suffix exactly like cold" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    var tok = Tokenizer.initEmptyForTests(a, .byte_level_bpe);
    defer tok.deinit();
    var first: [300]u32 = undefined;
    for (&first, 0..) |*t, i| t.* = @intCast((i * 7 + i / 5) % 4);
    var next: [330]u32 = undefined;
    @memcpy(next[0..290], first[0..290]);
    for (next[290..], 0..) |*t, i| t.* = @intCast((i * 3 + 1) % 4);
    const greedy = SamplingParams{ .temperature = 0.0 };
    const opts: Generator.InitOptions = .{ .skip_lazy_preforward = true, .pinned_prefill_chunk = 64, .adaptive_chunk_width = true, .ssm_checkpoint_stride = 64, .ssm_checkpoint_max = 16 };
    glm_checkpoint_stride = 64;
    defer glm_checkpoint_stride = glm5_forward.prefill_chunk;

    var weights = model_mod.Weights.init(a);
    defer weights.deinit();
    const cfg = try glmGeneratorFixture(&weights);
    var turn = try Transformer.init(testing.io, a, cfg, &weights);
    defer turn.deinit();
    var gen = try Generator.initWithOptions(testing.io, a, &turn, &tok, &first, 4, greedy, &.{}, opts);
    defer gen.deinit(a);
    var at: [5]usize = undefined;
    for (gen.ssm_checkpoints.items, 0..) |cp, i| at[i] = cp.pos;
    // Stride boundaries, then the backoff snapshot pulled back to a pool boundary (299 - 31).
    try testing.expectEqualSlices(usize, &.{ 64, 128, 192, 256, 268 }, at[0..gen.ssm_checkpoints.items.len]);
    var rows = try glm5_prefix.MlaRows.capture(a, &turn.glm5_request.?, 268, turn.s);
    defer rows.deinit();

    var cold_xfm = try Transformer.init(testing.io, a, cfg, &weights);
    defer cold_xfm.deinit();
    var cold = try Generator.initWithOptions(testing.io, a, &cold_xfm, &tok, &next, 4, greedy, &.{}, opts);
    defer cold.deinit(a);

    var warm_xfm = try Transformer.init(testing.io, a, cfg, &weights);
    defer warm_xfm.deinit();
    try glm5_prefix.restore(&warm_xfm.glm5_request.?, &rows, &gen.ssm_checkpoints.items[3]);
    warm_xfm.cache.step = 256;
    var warm_opts = opts;
    warm_opts.ssm_checkpoint_pos_offset = 256;
    var warm = try Generator.initWithOptions(testing.io, a, &warm_xfm, &tok, next[256..], 4, greedy, &.{}, warm_opts);
    defer warm.deinit(a);
    try glm5_prefix.expectSameLogicalState(&cold_xfm.glm5_request.?, &warm_xfm.glm5_request.?);
    try testing.expectEqual(cold.next_token_id, warm.next_token_id);
    try testing.expectEqual(cold.ssm_checkpoints.items[cold.ssm_checkpoints.items.len - 1].pos, warm.ssm_checkpoints.items[warm.ssm_checkpoints.items.len - 1].pos);
    const sched = glmCaptureSchedule(64, 16, 256, next.len - 256);
    try testing.expectEqual(sched.peak, warm.ssm_cp_peak);
    try testing.expectEqual(sched.newest, warm.ssm_checkpoints.items[warm.ssm_checkpoints.items.len - 1].pos);
}

test "the GLM capture schedule is what the prefill loop captures: peak, newest and count, cold and warm, at every cap" {
    const widths = [_]usize{ 2048, 512, 64 };
    for ([_]usize{ 64, 2048 }) |grid| for ([_]usize{ 0, 4, 60, 2044 }) |offset| {
        for ([_]usize{ 1, 2, 15, 16, 31, 33, 34, 64, 65, 200, 2048, 2049, 5000, 18432 }) |tail| for (0..9) |cap_usize| {
            const cap: u32 = @intCast(cap_usize);
            const got = glmCaptureSchedule(grid, cap, offset, tail);
            for (widths) |w| {
                const want = glmCaptureOracle(grid, cap, offset, tail, w);
                testing.expectEqual(want, got) catch |e| {
                    std.debug.print("grid {d} offset {d} tail {d} cap {d} width {d}: want {any} got {any}\n", .{ grid, offset, tail, cap, w, want, got });
                    return e;
                };
            }
        };
    };
    // 18,432 cold tokens: eight grid points, then the prompt-end checkpoint beside them before the thin.
    try testing.expectEqual(@as(u32, 9), glmCaptureSchedule(2048, 8, 0, 18432).peak);
    // A 16-token cold prompt captures one checkpoint, at 12.
    try testing.expectEqual(@as(usize, 12), glmCaptureSchedule(2048, 8, 0, 16).newest);
    // A grid point at a chunk's end is not absorbed by the tail merge that follows it.
    try testing.expectEqual(@as(usize, 2058), nextChunkEnd(0, 2058, 2048, true, 2048, 0, true));
    try testing.expectEqual(@as(usize, 2048), glmChunkEnd(0, 2058, 2048, 2048, 0, true));
}

test "a GLM prefill without checkpoints chunks like any other prefill" {
    for ([_]usize{ 0, 4096 }) |pos| for ([_]usize{ 2, 2058, 9000 }) |len| {
        const end = pos + len;
        try testing.expectEqual(nextChunkEnd(pos, end, 2048, false, 0, 0, true), glmChunkEnd(pos, end, 2048, 0, 0, true));
    };
}

test "the GLM prefill holds as many checkpoints as its schedule says, and ends on the newest" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    var tok = Tokenizer.initEmptyForTests(a, .byte_level_bpe);
    defer tok.deinit();
    var weights = model_mod.Weights.init(a);
    defer weights.deinit();
    const cfg = try glmGeneratorFixture(&weights);
    const greedy = SamplingParams{ .temperature = 0.0 };
    var prompt: [300]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast((i * 5 + 1) % 4);
    glm_checkpoint_stride = 64;
    defer glm_checkpoint_stride = glm5_forward.prefill_chunk;
    for ([_]usize{ 14, 40, 70, 150, 300 }) |len| for ([_]u32{ 2, 16 }) |cap| {
        var xfm = try Transformer.init(testing.io, a, cfg, &weights);
        defer xfm.deinit();
        var gen = try Generator.initWithOptions(testing.io, a, &xfm, &tok, prompt[0..len], 2, greedy, &.{}, .{ .skip_lazy_preforward = true, .pinned_prefill_chunk = 64, .adaptive_chunk_width = true, .ssm_checkpoint_stride = 1, .ssm_checkpoint_max = cap });
        defer gen.deinit(a);
        const sched = glmCaptureSchedule(64, cap, 0, len);
        try testing.expectEqual(sched.peak, gen.ssm_cp_peak);
        try testing.expectEqual(@min(sched.events, cap), @as(u32, @intCast(gen.ssm_checkpoints.items.len)));
        try testing.expectEqual(sched.newest, gen.ssm_checkpoints.items[gen.ssm_checkpoints.items.len - 1].pos);
    };
}

const GlmReleasing = struct {
    peak: usize = 0,

    fn call(raw: *anyopaque, _: usize, cps: []SSMCheckpoint) void {
        const self: *GlmReleasing = @ptrCast(@alignCast(raw));
        var held: usize = 0;
        var newest: ?usize = null;
        for (cps, 0..) |cp, i| {
            if (cp.isReleased()) continue;
            held += 1;
            newest = i;
        }
        self.peak = @max(self.peak, held);
        for (cps, 0..) |*cp, i| if (i != newest) cp.release();
    }
};

test "a GLM prefill whose hook releases landed checkpoints holds at most two states and lists every restore point" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    var tok = Tokenizer.initEmptyForTests(a, .byte_level_bpe);
    defer tok.deinit();
    var weights = model_mod.Weights.init(a);
    defer weights.deinit();
    const cfg = try glmGeneratorFixture(&weights);
    const greedy = SamplingParams{ .temperature = 0.0 };
    var prompt: [300]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast((i * 5 + 1) % 4);
    glm_checkpoint_stride = 64;
    defer glm_checkpoint_stride = glm5_forward.prefill_chunk;
    for ([_]usize{ 14, 70, 150, 300 }) |len| for ([_]u32{ 2, 16 }) |cap| {
        var kept_xfm = try Transformer.init(testing.io, a, cfg, &weights);
        defer kept_xfm.deinit();
        const base: Generator.InitOptions = .{ .skip_lazy_preforward = true, .pinned_prefill_chunk = 64, .adaptive_chunk_width = true, .ssm_checkpoint_stride = 1, .ssm_checkpoint_max = cap };
        var kept = try Generator.initWithOptions(testing.io, a, &kept_xfm, &tok, prompt[0..len], 2, greedy, &.{}, base);
        defer kept.deinit(a);

        var xfm = try Transformer.init(testing.io, a, cfg, &weights);
        defer xfm.deinit();
        var releasing = GlmReleasing{};
        var opts = base;
        opts.write_through_hook = .{ .ctx = &releasing, .call = GlmReleasing.call };
        var gen = try Generator.initWithOptions(testing.io, a, &xfm, &tok, prompt[0..len], 2, greedy, &.{}, opts);
        defer gen.deinit(a);

        const sched = glmCaptureSchedule(64, cap, 0, len);
        try testing.expect(releasing.peak <= @min(sched.peak, 2));
        if (len >= 300) try testing.expectEqual(@as(usize, 2), releasing.peak);
        try testing.expectEqual(kept.ssm_checkpoints.items.len, gen.ssm_checkpoints.items.len);
        var held: usize = 0;
        for (kept.ssm_checkpoints.items, gen.ssm_checkpoints.items) |want, got| {
            try testing.expectEqual(want.pos, got.pos);
            held += @intFromBool(!got.isReleased());
        }
        try testing.expect(held <= 2);
        try testing.expect(!gen.ssm_checkpoints.items[gen.ssm_checkpoints.items.len - 1].isReleased());
    };
}

const GlmStepDown = struct {
    fn call(_: *anyopaque, pos: usize, cur: u32, _: u32, _: *AdaptiveWidthState) u32 {
        return if (pos >= 128) 32 else cur;
    }
    fn confirm(_: *anyopaque, _: usize, _: u32) bool {
        return false;
    }
};

test "GLM checkpoints stay on the grid when a request steps its chunk down, and a restore there with the same widths matches cold" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    var tok = Tokenizer.initEmptyForTests(a, .byte_level_bpe);
    defer tok.deinit();
    var weights = model_mod.Weights.init(a);
    defer weights.deinit();
    const cfg = try glmGeneratorFixture(&weights);
    glm_checkpoint_stride = 64;
    defer glm_checkpoint_stride = glm5_forward.prefill_chunk;
    var first: [300]u32 = undefined;
    for (&first, 0..) |*t, i| t.* = @intCast((i * 7 + i / 5) % 4);
    var next: [330]u32 = undefined;
    @memcpy(next[0..290], first[0..290]);
    for (next[290..], 0..) |*t, i| t.* = @intCast((i * 3 + 1) % 4);
    const greedy = SamplingParams{ .temperature = 0.0 };
    var dummy: u8 = 0;
    const hook: Generator.ChunkWidthHook = .{ .ctx = &dummy, .call = GlmStepDown.call, .confirm = GlmStepDown.confirm };
    const opts: Generator.InitOptions = .{ .skip_lazy_preforward = true, .pinned_prefill_chunk = 64, .adaptive_chunk_width = true, .chunk_width_hook = hook, .ssm_checkpoint_stride = 64, .ssm_checkpoint_max = 16 };

    var turn = try Transformer.init(testing.io, a, cfg, &weights);
    defer turn.deinit();
    var gen = try Generator.initWithOptions(testing.io, a, &turn, &tok, &first, 2, greedy, &.{}, opts);
    defer gen.deinit(a);
    var at: [5]usize = undefined;
    for (gen.ssm_checkpoints.items, 0..) |cp, i| at[i] = cp.pos;
    // 64-wide chunks to 128, 32-wide after: the grid points are still chunk ends.
    try testing.expectEqualSlices(usize, &.{ 64, 128, 192, 256, 268 }, at[0..gen.ssm_checkpoints.items.len]);
    var rows = try glm5_prefix.MlaRows.capture(a, &turn.glm5_request.?, 268, turn.s);
    defer rows.deinit();

    var cold_xfm = try Transformer.init(testing.io, a, cfg, &weights);
    defer cold_xfm.deinit();
    var cold = try Generator.initWithOptions(testing.io, a, &cold_xfm, &tok, &next, 2, greedy, &.{}, opts);
    defer cold.deinit(a);

    var warm_xfm = try Transformer.init(testing.io, a, cfg, &weights);
    defer warm_xfm.deinit();
    try glm5_prefix.restore(&warm_xfm.glm5_request.?, &rows, &gen.ssm_checkpoints.items[2]);
    warm_xfm.cache.step = 192;
    var warm_opts = opts;
    warm_opts.ssm_checkpoint_pos_offset = 192;
    // The width in force at 192 in the cold run.
    warm_opts.pinned_prefill_chunk = 32;
    var warm = try Generator.initWithOptions(testing.io, a, &warm_xfm, &tok, next[192..], 2, greedy, &.{}, warm_opts);
    defer warm.deinit(a);
    try glm5_prefix.expectSameLogicalState(&cold_xfm.glm5_request.?, &warm_xfm.glm5_request.?);
    try testing.expectEqual(cold.next_token_id, warm.next_token_id);
}

test "a Qwen round below 8192 KV prices widths by the table's realized tokens and keeps the confidence-gated chunk B" {
    // The acceptance EMAs promise deep rounds; the table measured that a wider round realizes fewer tokens per ms.
    // The base follows the table at every KV and only a confident round extends past it.
    var xfm: Transformer = undefined;
    xfm.config.model_type = "qwen4_exp";
    xfm.round_cost = .{ .layout = .long, .first_use_logged = true };
    for (0..round_cost.MIN_SAMPLES) |_| {
        _ = xfm.round_cost.observe(2, 1000, 29.0, 2.6, true, false);
        _ = xfm.round_cost.observe(3, 1000, 31.0, 2.9, true, false);
        _ = xfm.round_cost.observe(4, 1000, 36.0, 3.1, true, false);
        _ = xfm.round_cost.observe(5, 1000, 42.0, 3.5, true, false);
    }
    var prompt: [1000]u32 = @splat(0);
    var gen = Generator{
        .xfm = &xfm,
        .ctx = undefined,
        .tok = undefined,
        .next_token_id = 0,
        .step = 0,
        .max_tokens = 1,
        .sampling = .{ .temperature = 0 },
        .prompt_tokens = 0,
        .completion_tokens = 0,
        .finish_reason = "length",
        .done = false,
        .eos_token_ids = &.{},
        .generated_ids = .empty,
        .timeout_ns = 0,
        .timer = io_util.Stopwatch.init(testing.io),
        .last_hidden = .{ .ctx = null },
        .has_last_hidden = false,
        .prompt_ids_owned = &prompt,
        .mtp_depth = 6,
        .mtp_depth_free = 6,
        .mtp_ev_rounds = 64,
        .mtp_ev_accept = .{ 0.92, 0.9, 0.88, 0.86, 0.84, 0.8, 0.8, 0.8 },
        .mtp_ev_m_lo_prev = 4,
        .spec_cost_solo = false,
    };
    const plan = gen.mtpRoundPlan();
    try testing.expectEqual(@as(u32, 3), plan.m_lo);
    try testing.expect(plan.m_hi > plan.m_lo);
    try testing.expect(plan.tau_ln < 0.0);
}

fn mtpBoundsPlanFixture(xfm: *Transformer, prompt: []u32, depth: u32) Generator {
    xfm.config.model_type = "qwen4_exp";
    xfm.round_cost = .{ .layout = .long, .first_use_logged = true };
    for (0..round_cost.MIN_SAMPLES) |_| {
        _ = xfm.round_cost.observe(2, 1000, 29.0, 2.6, true, false);
        _ = xfm.round_cost.observe(3, 1000, 31.0, 2.9, true, false);
        _ = xfm.round_cost.observe(4, 1000, 36.0, 3.1, true, false);
        _ = xfm.round_cost.observe(5, 1000, 42.0, 3.5, true, false);
    }
    return Generator{
        .xfm = xfm,
        .ctx = undefined,
        .tok = undefined,
        .next_token_id = 0,
        .step = 0,
        .max_tokens = 1,
        .sampling = .{ .temperature = 0 },
        .prompt_tokens = 0,
        .completion_tokens = 0,
        .finish_reason = "length",
        .done = false,
        .eos_token_ids = &.{},
        .generated_ids = .empty,
        .timeout_ns = 0,
        .timer = io_util.Stopwatch.init(testing.io),
        .last_hidden = .{ .ctx = null },
        .has_last_hidden = false,
        .prompt_ids_owned = prompt,
        .mtp_depth = depth,
        .mtp_depth_free = depth,
        .mtp_ev_rounds = 64,
        .mtp_ev_accept = .{ 0.92, 0.9, 0.88, 0.86, 0.84, 0.8, 0.8, 0.8 },
        .mtp_ev_m_lo_prev = 4,
        .spec_cost_solo = false,
    };
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

test "MTP depth bounds: the planner reads min..max from its table-priced plan" {
    const before = depth_bounds.active;
    defer depth_bounds.active = before;
    var xfm: Transformer = undefined;
    var prompt: [1000]u32 = @splat(0);

    depth_bounds.active = .{};
    var gen = mtpBoundsPlanFixture(&xfm, &prompt, 6);
    try testing.expectEqual(@as(u32, 3), gen.mtpRoundPlan().m_lo);

    depth_bounds.active = .{ .min = 5 };
    gen = mtpBoundsPlanFixture(&xfm, &prompt, 6);
    var plan = gen.mtpRoundPlan();
    try testing.expect(plan.m_lo >= 5 and plan.m_hi >= plan.m_lo and plan.m_hi <= 6);

    depth_bounds.active = .{ .max = 2 };
    gen = mtpBoundsPlanFixture(&xfm, &prompt, 2);
    plan = gen.mtpRoundPlan();
    try testing.expect(plan.m_lo <= 2 and plan.m_hi <= 2);
}

test "MTP depth bounds: min == max drafts that depth every round" {
    const before = depth_bounds.active;
    defer depth_bounds.active = before;
    var xfm: Transformer = undefined;
    var prompt: [1000]u32 = @splat(0);
    for ([_]u32{ 2, 4, 5 }) |depth| {
        depth_bounds.active = .{ .min = depth, .max = depth };
        var gen = mtpBoundsPlanFixture(&xfm, &prompt, depth);
        for (0..4) |_| {
            const plan = gen.mtpRoundPlan();
            try testing.expectEqual(depth, plan.m_lo);
            try testing.expectEqual(depth, plan.m_hi);
        }
    }
}

test "MTP depth bounds: the windowed controller never leaves min..max and never disables above a floor of 1" {
    const before = depth_bounds.active;
    defer depth_bounds.active = before;
    depth_bounds.active = .{ .min = 3, .max = 5 };
    var hot = mtpEvTestGenerator();
    hot.mtp_depth = 5;
    hot.mtp_depth_current = 3;
    for (0..64) |_| {
        hot.updateMtpDepth(5, 5);
        try testing.expect(hot.mtp_depth_current >= 3 and hot.mtp_depth_current <= 5);
    }
    try testing.expectEqual(@as(u32, 5), hot.mtp_depth_current);

    var cold = mtpEvTestGenerator();
    cold.mtp_depth = 5;
    cold.mtp_depth_current = 5;
    for (0..64) |_| {
        cold.updateMtpDepth(5, 0);
        try testing.expect(cold.mtp_depth_current >= 3 and cold.mtp_depth_current <= 5);
    }
    try testing.expectEqual(@as(u32, 3), cold.mtp_depth_current);
    try testing.expect(!cold.spec_disabled_runtime);
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
