const std = @import("std");
const mlx = @import("../mlx.zig");
const transformer_mod = @import("../transformer.zig");
const kv_cache = @import("../kv_cache.zig");
const qwen4_forward = @import("qwen4_forward.zig");
const tokenizer_mod = @import("../tokenizer.zig");
const model_mod = @import("../model.zig");
const io_util = @import("../io_util.zig");
const mtp_mod = @import("../mtp.zig");
const round_cost = @import("../round_cost.zig");
const group_planner = @import("../mtp_group_planner.zig");
const depth_bounds = @import("../mtp_depth_bounds.zig");
const Transformer = transformer_mod.Transformer;
const Tokenizer = tokenizer_mod.Tokenizer;
const ForwardCtx = transformer_mod.ForwardCtx;
const kv_disk_cache_mod = @import("../kv_disk_cache.zig");
const KVCache = transformer_mod.KVCache;
const testing = std.testing;
const expectError = @import("../test_expect.zig").expectError;
const sampler = @import("../sampler.zig");
const SamplingParams = sampler.SamplingParams;
const proposalTestRead = sampler.proposalTestRead;
const sampleRows = sampler.sampleRows;
const mtp_plan = @import("../mtp_plan.zig");
const ThoughtPoint = mtp_plan.ThoughtPoint;
const prefill_plan = @import("../prefill_plan.zig");
const effectiveSsmCheckpointStride = prefill_plan.effectiveSsmCheckpointStride;
const reservedPrefillTokens = prefill_plan.reservedPrefillTokens;
const walkChunkEnds = prefill_plan.walkChunkEnds;
const generate = @import("../generate.zig");
const Generator = generate.Generator;
const mtp_round = @import("../mtp_round.zig");
const mtp_policy = @import("../mtp_policy.zig");
const MtpHeadRef = generate.MtpHeadRef;

test "a 40k Qwen prefill checkpoints every 8192 tokens at any chunk width the ladder takes" {
    for ([_]usize{ 8192, 4096, 2048, 512 }) |width| {
        const stride = effectiveSsmCheckpointStride(256, @max(width, generate.prefill_chunk_override));
        try testing.expectEqual(@as(usize, kv_disk_cache_mod.POINT_GRID), stride);
        var buf: [128]usize = undefined;
        const ends = walkChunkEnds(&buf, 39969, &.{width}, true, stride, 0);
        var grid: [8]usize = undefined;
        var n: usize = 0;
        for (ends) |end| {
            if (end % stride != 0) continue;
            grid[n] = end;
            n += 1;
        }
        try testing.expectEqualSlices(usize, &.{ 8192, 16384, 24576, 32768 }, grid[0..n]);
    }
}

test "MTP EV seed round-trips on the qwen4 head; a fresh or absent head reads null" {
    // The in-checkpoint head stores the seed on `Qwen4Mtp` — per loaded model.
    // Before this it was a
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

test "mtpEvPlanFor: realistic warmup EMAs stop the generic surface early and never open qwen4 to depth 8" {
    const p = mtp_policy.MTP_EV_PRIOR;
    const a = [_]f32{ 0.97, 0.89, p, p, p, p, p, p };

    // The M1-M4 surface stops at the first expensive ramp position.
    const generic = Generator.mtpEvPlanFor(&a, 8, mtp_policy.MTP_EV_DEFAULT_COSTS, 3);
    try testing.expectEqual(@as(u32, 3), generic.m_lo);
    try testing.expect(generic.m_hi < 8);

    // qwen4_exp's measured surface has no NAX takeover — a verify row is
    // bytes at every width — so the same base depth stands but the same
    // evidence must NOT extend to depth 8.
    const qwen4_plan = Generator.mtpEvPlanFor(&a, 8, mtp_policy.MTP_EV_G17_NAX_QWEN4_Q4_GS64_COSTS, 3);
    try testing.expectEqual(@as(u32, 3), qwen4_plan.m_lo);
    try testing.expect(qwen4_plan.m_hi < 8);

    const cold = [_]f32{ 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1 };
    const cold_plan = Generator.mtpEvPlanFor(&cold, 8, mtp_policy.MTP_EV_G17_NAX_QWEN4_Q4_GS64_COSTS, 8);
    try testing.expectEqual(@as(u32, 1), cold_plan.m_lo);
    try testing.expectEqual(@as(u32, 1), cold_plan.m_hi);
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

test "qwen4: the coarse rerank head is READY before any draft has run" {
    // The head is a full `requantizeRows` of the trunk lm_head plus a
    // synchronous eval of its ~240 MB. Built lazily it landed inside the
    // FIRST request's draft chain — on the inference thread, mid-round,
    // draining the stream, and paid as first-token latency by whoever
    // happened to be first after a load. `qwen4BuildDraftRerank` is that
    // build as a standalone LOAD-time step: no Generator, no chain, no
    // request. Afterwards `canRerankDrafts()` is a pure read.
    try mlx.requireGpu();
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

test "mtpAdaptiveVoteFor: four real switch events of the qwen4 A/B" {
    // Four switches the v1 controller made on a 62.7k prose prompt where MTP was the faster
    // arm (56.8 vs 51.6 tok/s): the modeled token count ran 12-31% under the cell's own `tok`.
    const G = Generator;
    const margin = mtp_policy.MTP_ADAPTIVE_MARGIN;
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
            mtp_policy.MtpAdaptiveVote.mtp,
            G.mtpAdaptiveVoteFor(e.table_ms_tok, e.table_ms_tok, e.serial_ms_tok, margin),
        );
        // Must not vote serial on any row; an unfilled window is `.undecided`.
        const win: ?f32 = if (e.window_full) e.window_ms_tok else null;
        const got = G.mtpAdaptiveVoteFor(e.table_ms_tok, win, e.serial_ms_tok, margin);
        try testing.expect(got != .serial);
        try testing.expectEqual(
            if (e.window_full) mtp_policy.MtpAdaptiveVote.mtp else mtp_policy.MtpAdaptiveVote.undecided,
            got,
        );
    }
    // Rule B ungated reproduces the b1-long2 mistake.
    const b1l2 = events[1];
    try testing.expectEqual(
        mtp_policy.MtpAdaptiveVote.serial,
        G.mtpAdaptiveVoteFor(30.0, b1l2.window_ms_tok, b1l2.serial_ms_tok, margin),
    );
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
    for ([_][]const u8{ "glm5_next", "deepseek_v41", "not_a_served_arch" }) |mt| {
        var cfg = model_mod.ModelConfig{ .model_type = mt, .max_position_embeddings = 262_144 };
        try t.expectEqual(@as(u64, 0), reservedPrefillTokens(&cfg, seq, max_tokens, chunk));
    }

    var cache = try KVC.init(t.allocator, 4);
    defer cache.deinit();
    cache.reserve(0);
    try t.expectEqual(@as(usize, 0), cache.reserve_tokens);
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

    fn begin(g: *Generator, a: std.mem.Allocator, depth: u32, confidence: bool) !mtp_round.MtpRoundOpen {
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
    try mlx.requireGpu();
    const predraft = mtp_round.mtp_predraft_enabled;
    const bounds_before = depth_bounds.active;
    const planner = group_planner.enabled_override;
    defer {
        mtp_round.mtp_predraft_enabled = predraft;
        depth_bounds.active = bounds_before;
        group_planner.enabled_override = planner;
    }
    // Observe the completed-round stash before any successor consumes it.
    mtp_round.mtp_predraft_enabled = false;
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
    var tok = Tokenizer.initEmptyForTests(a);
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
                var ss: [4]?mtp_round.MtpRoundState = @splat(null);
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
                var opens: [4]mtp_round.MtpRoundOpen = undefined;
                var chains: [4]mtp_round.MtpPreDraft = undefined;
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
                var batch_states: [4]?mtp_round.MtpRoundState = @splat(null);
                defer for (&batch_states) |*state| {
                    if (state.*) |*p| p.deinit(a);
                };
                var batch_ptrs: [4]*mtp_round.MtpRoundState = undefined;
                for (0..n) |i| {
                    opens[i].chain = chains[i];
                    owns[i] = false;
                    batch_states[i] = try gens[i].mtpRoundContinue(a, opens[i]);
                    batch_ptrs[i] = &batch_states[i].?;
                }
                const grouped_verify = @import("../mtp_group_planner.zig").enabled();
                if (grouped_verify) {
                    try Generator.mtpGroupVerify(gens[0..n], batch_ptrs[0..n]);
                    try Generator.mtpGroupAccept(gens[0..n], batch_ptrs[0..n]);
                } else {
                    for (gens[0..n], batch_ptrs[0..n]) |g, state| try g.mtpRoundVerify(state);
                }
                for (0..n) |i| {
                    const bs = batch_ptrs[i];
                    errdefer std.debug.print("[mtp compare] N={d} depth={d} round={d} row={d}\n", .{ n, depth, round, i });
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
                }
            }
        }
    }
    if (transformer_mod.verifySharedHardware()) {
        try testing.expect(transformer_mod.mtp_coarse_pair_dispatches > coarse_before);
        {
            try testing.expect(qwen4_forward.mtp_verify_hc_prepared_calls > hc_prepared_before);
        }
        inline for (0..3) |i| {
            {
                try testing.expect(transformer_mod.mtp_verify_kernel_calls[i] > verify_kernel_before[i]);
            }
        }
    }
}

test "greedy tail on real qwen4 chains: depth 0 hands the verify its density, every later depth a one-hot at its draft (QWEN4_TEST_MODEL)" {
    const model_dir = std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest;
    try mlx.requireGpu();
    const predraft = mtp_round.mtp_predraft_enabled;
    const bounds_before = depth_bounds.active;
    const planner = group_planner.enabled_override;
    defer {
        mtp_round.mtp_predraft_enabled = predraft;
        depth_bounds.active = bounds_before;
        group_planner.enabled_override = planner;
    }
    mtp_round.mtp_predraft_enabled = false;
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
    var tok = Tokenizer.initEmptyForTests(a);
    defer tok.deinit();
    var prompt: [96]u32 = undefined;
    for (&prompt, 0..) |*id, i| id.* = @intCast(1 + (i * 7919) % 40000);

    const Check = struct {
        fn chain(c: *const mtp_round.MtpPreDraft, s: mlx.mlx_stream) !void {
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
        var opens: [2]mtp_round.MtpRoundOpen = undefined;
        var chains: [2]mtp_round.MtpPreDraft = undefined;
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
        var states: [2]?mtp_round.MtpRoundState = @splat(null);
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
    try mlx.requireGpu();
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
    var tok = Tokenizer.initEmptyForTests(a);
    defer tok.deinit();
    var prompt: [96]u32 = undefined;
    for (&prompt, 0..) |*id, i| id.* = @intCast(1 + (i * 7919) % 40000);
    var slots: [2]?*MtpChainTestSlot = @splat(null);
    defer for (slots) |slot| {
        if (slot) |p| p.deinit(a);
    };
    var states: [2]?mtp_round.MtpRoundState = @splat(null);
    defer for (&states) |*state| {
        if (state.*) |*p| p.deinit(a);
    };
    var gens: [2]*Generator = undefined;
    var ptrs: [2]*mtp_round.MtpRoundState = undefined;
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
    const syncs = kv_cache.deferred_ple_group_syncs;
    try Generator.mtpGroupVerify(&gens, &ptrs);
    try testing.expectEqual(@as(u64, 1), kv_cache.deferred_ple_group_syncs - syncs);
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
    try mlx.requireGpu();
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
    var tok = Tokenizer.initEmptyForTests(a);
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

test "cold grouped MTP rounds detach and resume exactly like solo heads" {
    const path = std.c.getenv("MTP_TEST_MODEL") orelse return error.SkipZigTest;
    try mlx.requireGpu();
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
    var tok = Tokenizer.initEmptyForTests(a);
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
        var opens: [2]mtp_round.MtpRoundOpen = undefined;
        var chains: [2]mtp_round.MtpPreDraft = undefined;
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
        var states: [2]?mtp_round.MtpRoundState = @splat(null);
        defer for (&states) |*state| {
            if (state.*) |*value| value.deinit(a);
        };
        var pointers: [2]*mtp_round.MtpRoundState = undefined;
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

test "qwen4 thought point: MTP rounds cut at the closer emit the tokens the uncut rounds do, and the state behind the closer is the newest checkpoint (MTP_TEST_MODEL)" {
    const path = std.c.getenv("MTP_TEST_MODEL") orelse return error.SkipZigTest;
    try mlx.requireGpu();
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
    var tok = Tokenizer.initEmptyForTests(a);
    defer tok.deinit();
    var prompt: [96]u32 = undefined;
    for (&prompt, 0..) |*id, i| id.* = @intCast(1 + (i * 7919) % 40000);
    const want: usize = 48;

    const whole = try MtpChainTestSlot.init(a, io, &xfm, &tok, &prompt, 4);
    defer whole.deinit(a);
    while (whole.gen.?.generated_ids.items.len < want) {
        const round = (try whole.gen.?.nextMtp(a)) orelse return error.MtpRoundDeclined;
        a.free(round.tokens);
    }
    const reference = whole.gen.?.generated_ids.items;
    // A token the stream emits mid-way stands in for the closer; its first occurrence is where the point sits.
    const closer = reference[20];
    const at = std.mem.indexOfScalar(u32, reference, closer).?;

    const cut = try MtpChainTestSlot.init(a, io, &xfm, &tok, &prompt, 4);
    defer cut.deinit(a);
    const gen = &cut.gen.?;
    gen.thought_point = .{ .closer = closer, .grid = 1, .min_tokens = 0, .base = prompt.len };
    while (gen.generated_ids.items.len < want) {
        const round = (try gen.nextMtp(a)) orelse return error.MtpRoundDeclined;
        a.free(round.tokens);
        // The round that took the point ended on the closer, and the state stands right behind it.
        if (gen.thought_point.?.state == .taken and gen.ssm_checkpoints.items.len == 1 and gen.generated_ids.items.len == at + 1) {
            try testing.expectEqual(closer, gen.generated_ids.items[at]);
            try testing.expectEqual(prompt.len + at + 1, gen.ssm_checkpoints.items[0].pos);
            try testing.expectEqual(prompt.len + at + 1, cut.off);
        }
    }
    try testing.expectEqual(ThoughtPoint.State.taken, gen.thought_point.?.state);
    try testing.expectEqual(@as(usize, 1), gen.ssm_checkpoints.items.len);
    try testing.expectEqual(prompt.len + at + 1, gen.ssm_checkpoints.items[0].pos);
    try testing.expectEqualSlices(u32, reference[0..want], gen.generated_ids.items[0..want]);
}
