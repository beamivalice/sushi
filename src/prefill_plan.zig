const std = @import("std");
const transformer_mod = @import("transformer.zig");
const model_mod = @import("model.zig");
const SSMCheckpoint = transformer_mod.SSMCheckpoint;
const testing = std.testing;
const generate = @import("generate.zig");

/// One layer's materialized-score budget for unfused-SDPA prefill (see
/// boundedPrefillChunk). 4 GiB keeps the full 8K chunk for every context up
/// to ~16K on 16-head models (further on fewer heads) and degrades gradually
/// to the 512 floor as heads × context grows toward 262K.
pub const PREFILL_SCORES_BUDGET_BYTES: u64 = 4 << 30;

/// Lower bound for the auto-capped prefill chunk; also its rounding grain
/// (repeating chunk sizes let the MLX allocator cache reuse score buffers).
pub const PREFILL_CHUNK_FLOOR: usize = 512;

/// MLX's fused SDPA kernels cover head_dim <= 128. A head_dim=256 trunk
/// falls back to the
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
/// (MoE gather buffers, per-chunk KV concat): +3% speed is not worth
/// +22 GB peak (a 64 GB Mac dies), so the cap stays keyed on raw head_dim.
///
/// `sliding_band_arch` (config.has_sliding_window) picks the policy family:
/// archs WITHOUT sliding-band layers (GDN + full attention)
/// additionally cap the auto chunk at 2048 — composed-causal prefill
/// measured strictly faster and ~9 GB lighter there (see the inline
/// comment). Sliding-band archs keep the formula-only policy for their fused band layers.
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
    // is the measured MoE lesson carried over as a ceiling only.
    // 2048 is the default (a 64k prompt prefills within noise of 4096); an
    // explicit `--prefill-chunk` may raise it to that ceiling.
    if (head_dim == 192 and transformer_mod.prefillHeadDimFused(head_dim)) {
        return @min(base_chunk, @as(usize, if (generate.prefill_chunk_explicit) 4096 else 2048));
    }
    // Non-sliding hd-256 archs under FUSED causal (the default since the
    // budgeted-dispatch flip): no score tensor exists, so the scores-budget
    // formula below is moot — and its old shrink (1024 at 64K on 24 heads)
    // starved the dequant+GEMM qmm route, which needs M >= 2048 to engage
    // (the 64K rung was the ladder's weakest for exactly this reason).
    // MoE keeps the 4096 cap: expert-gather transients scale with the chunk
    // (+3% speed for +22 GB peak is a bad trade).
    // DENSE hybrids have no gather transients and a full-size chunk halves
    // the per-chunk dequant sweeps: chunk 8192 measured +1.4% over 4096 at
    // the 8K rung, flat at 32K.
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
    // skipping for composed attention. Sliding-band archs keep big
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
    const base: usize = if (generate.prefill_chunk_explicit or pinned_chunk == 0)
        generate.prefill_chunk_override
    else
        @min(generate.prefill_chunk_override, pinned_chunk);
    return boundedPrefillChunk(base, head_dim, n_heads, total_ctx, sliding_band_arch, is_moe, long_ctx_gated);
}

/// The width a request forwards at, and the width its admission bills: `effectivePrefillChunk`,
/// never wider than the width handed in (0 = none). An explicit width outranks only the load-time pin.
pub fn requestPrefillChunk(head_dim: u32, n_heads: u32, total_ctx: usize, sliding_band_arch: bool, is_moe: bool, long_ctx_gated: bool, pinned_prefill_chunk: usize) usize {
    const width = effectivePrefillChunk(head_dim, n_heads, total_ctx, sliding_band_arch, is_moe, long_ctx_gated, pinned_prefill_chunk);
    return if (pinned_prefill_chunk == 0) width else @min(width, pinned_prefill_chunk);
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

/// Drops checkpoints until at most `max` hold a state (`max` 0 = no cap). A released one is a restore point
/// already on SSD and costs no memory, so it is never the one dropped nor counted.
pub fn thinHeldCheckpoints(allocator: std.mem.Allocator, list: *std.ArrayList(SSMCheckpoint), max: u32, policy: transformer_mod.ThinPolicy) void {
    if (max == 0) return;
    while (true) {
        var held: usize = 0;
        var first: usize = list.items.len;
        for (list.items, 0..) |*cp, i| {
            if (cp.isReleased()) continue;
            held += 1;
            first = @min(first, i);
        }
        if (held <= max) return;
        var dropped = list.orderedRemove(first + checkpointDropIndex(list.items[first..], policy));
        dropped.deinit(allocator);
    }
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
/// template suffix in the fleet with margin (ChatML+think ≈ 7 tokens, a
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

/// Effective SSM-checkpoint stride for a model, given the base (configured)
/// stride: checkpointing never sub-divides the prefill chunk, on ANY arch.
///
/// The old policy kept the fine base stride on dense (non-MoE) hybrids on the
/// theory that their prefill is compute-bound so extra chunks are ~free. That
/// was true before `prefillDqGemm`: since the dq route only engages at
/// M >= 2048, a fine stride pushes EVERY projection of EVERY chunk onto the
/// slow small-M qmm path, and the per-chunk fixed costs (graph build, eval
/// barrier, MTP history capture) multiply on top. Measured on a 27B
/// dense GDN trunk (M4 Max): stride 256 chunked an 8K prompt into 33
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

/// Walk a chunked prefill as `runPrefill` does, taking the width from `widths` at each boundary (the last entry repeats).
pub fn walkChunkEnds(
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

test "the held-state cap thins only states in memory: restore points already on SSD are never dropped" {
    const al = testing.allocator;
    var list = std.ArrayList(SSMCheckpoint).empty;
    defer list.deinit(al);
    // 130 points written through (their states released), then the newest 20 still held.
    for (0..130) |i| try list.append(al, .{ .pos = 8192 * (i + 1), .layers = &.{}, .released_bytes = 1 });
    for (0..20) |i| try list.append(al, .{ .pos = 8192 * (131 + i), .layers = &.{} });
    thinHeldCheckpoints(al, &list, 16, .min_span_recency);
    try testing.expectEqual(@as(usize, 146), list.items.len);
    for (list.items[0..130]) |cp| try testing.expect(cp.isReleased());
    try testing.expectEqual(@as(usize, 8192 * 150), list.items[list.items.len - 1].pos);
    // No cap, nothing thins.
    thinHeldCheckpoints(al, &list, 0, .min_span_recency);
    try testing.expectEqual(@as(usize, 146), list.items.len);
}

test "every ladder rung the adaptive width can take divides the checkpoint stride" {
    const t = testing;
    const stride = effectiveSsmCheckpointStride(256, generate.prefill_chunk_override);
    try t.expect(stride > 0);
    for ([_]usize{ 8192, 4096, 2048, 1024, PREFILL_CHUNK_FLOOR }) |rung| {
        try t.expect(rung <= stride);
        try t.expectEqual(@as(usize, 0), stride % rung);
    }
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
    // fleet-protection property: every head_dim 256 checkpoint must keep
    // full prefill throughput on typical prompts.
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
    // 16-head geometry: budget/(16*ctx*2) …
    // ctx 32768 -> exactly 4096.
    try testing.expectEqual(@as(usize, 4096), boundedPrefillChunk(8192, 256, 16, 32768, true, false, false));
    // ctx 100000 -> raw 1342, rounded down to the 512 grain -> 1024.
    try testing.expectEqual(@as(usize, 1024), boundedPrefillChunk(8192, 256, 16, 100_000, true, false, false));
    // ctx 262144 (the PR-#69 255K case) -> 512.
    try testing.expectEqual(@as(usize, 512), boundedPrefillChunk(8192, 256, 16, 262_144, true, false, false));
    // 24-head geometry at 262144: raw 341 -> floor 512.
    try testing.expectEqual(@as(usize, 512), boundedPrefillChunk(8192, 256, 24, 262_144, true, false, false));
    // 8-head geometry at 131072: exactly 2048.
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
    const saved_explicit = generate.prefill_chunk_explicit;
    defer generate.prefill_chunk_explicit = saved_explicit;
    generate.prefill_chunk_explicit = false;
    try testing.expectEqual(@as(usize, 2048), boundedPrefillChunk(8192, 192, 32, 8192, false, true, false));
    try testing.expectEqual(@as(usize, 2048), boundedPrefillChunk(8192, 192, 32, 262_144, false, true, false));
    try testing.expectEqual(@as(usize, 2048), boundedPrefillChunk(8192, 192, 32, 1_048_576, false, true, false));
    // An explicit `--prefill-chunk` raises it as far as the MoE ceiling.
    generate.prefill_chunk_explicit = true;
    try testing.expectEqual(@as(usize, 4096), boundedPrefillChunk(8192, 192, 32, 262_144, false, true, false));
    try testing.expectEqual(@as(usize, 3072), boundedPrefillChunk(3072, 192, 32, 262_144, false, true, false));
    generate.prefill_chunk_explicit = false;
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
    // non-sliding hd-256 class — and its old shrink to 1024 at 64K starved the
    // dequant+GEMM qmm route (engages at M >= 2048): the 64K rung was the
    // ladder's weakest for exactly this reason.
    std.debug.assert(transformer_mod.fused256_override == null);
    // DENSE hybrids: no expert-gather transients, and a
    // full-size chunk halves per-chunk dequant sweeps — chunk 8192 measured
    // +1.4% over 4096 at the 8K rung, flat at 32K.
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 256, 24, 8192, false, false, false));
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 256, 24, 65536, false, false, false));
    try testing.expectEqual(@as(usize, 8192), boundedPrefillChunk(8192, 256, 24, 140_000, false, false, false));
    // MoE keeps the 4096 cap: expert-gather transients scale with the chunk
    // (+3% speed for +22 GB peak is a bad trade).
    try testing.expectEqual(@as(usize, 4096), boundedPrefillChunk(8192, 256, 24, 8192, false, true, false));
    try testing.expectEqual(@as(usize, 4096), boundedPrefillChunk(8192, 256, 24, 140_000, false, true, false));
    // Never raises a caller-lowered base.
    try testing.expectEqual(@as(usize, 1024), boundedPrefillChunk(1024, 256, 24, 8192, false, false, false));
    // Sliding-band archs (fused band kernel wants big chunks) keep
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
    // fixed costs. Measured on a 27B dense trunk (M4 Max): stride
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
    for ([_][]const u8{ "glm5_next", "deepseek_v41", "not_a_served_arch" }) |mt| {
        var cfg = model_mod.ModelConfig{ .model_type = mt };
        try t.expect(!clearsPoolAtPrefillEnd(&cfg));
    }
}

test "countSpliceRows counts image and video placeholders, video only when declared" {
    const ids = [_]i32{ 7, 99, 99, 8, 9, 77 };
    try testing.expectEqual(@as(usize, 2), countSpliceRows(&ids, 99, 0));
    try testing.expectEqual(@as(usize, 3), countSpliceRows(&ids, 99, 77));
    try testing.expectEqual(@as(usize, 0), countSpliceRows(ids[0..1], 99, 77));
}
