const trunk_mod = @import("../trunk.zig");
const std = @import("std");
const dsv41_model = @import("dsv41_model.zig");
const dsv41_state = @import("dsv41_state.zig");
const dsv41_stream = @import("dsv41_stream.zig");
const kernels = @import("dsv41_kernels.zig");
const dsv41_borrow = @import("dsv41_borrow.zig");
const dsv41_pipelines = @import("dsv41_pipelines.zig");
const dsv41_config = @import("dsv41_config.zig");
const expert_stream_mod = @import("../expert_stream.zig");
const PrefetchCredit = @import("../prefetch_plan.zig").PrefetchCredit;
const sushi_exl3 = @import("sushi_exl3");
const expert_exl3_kernels = sushi_exl3.kernels;
const expert_quant_mod = @import("../expert_quant.zig");
const mlx = @import("../mlx.zig");
const model_mod = @import("../model.zig");
const log = @import("../log.zig");
const manifest_mod = @import("../pipeline_manifest.zig");
const load_clock = @import("../load_clock.zig");
const sampler = @import("../sampler.zig");
const seen_penalty = @import("../seen_penalty.zig");
const think_penalty = @import("../think_penalty.zig");
const logit_bias = @import("../logit_bias.zig");
const random = @import("../tensor/random.zig");
const spec_types = @import("../spec_types.zig");
const io_util_mod = @import("../io_util.zig");
const sched_policy = @import("../sched_policy.zig");
const ModelConfig = model_mod.ModelConfig;
const Weights = model_mod.Weights;
const diagEnvValueOn = io_util_mod.diagEnvValueOn;
const diagEnvOn = io_util_mod.diagEnvOn;
const ForwardCtx = trunk_mod.ForwardCtx;
const Trunk = trunk_mod.Trunk;
const initModuleShell = trunk_mod.initModuleShell;

/// Layers ahead of a layer whose picks its router input predicts.
const prefetch_lead_rows: u8 = 3;

pub fn initDsv41(allocator: std.mem.Allocator, config: ModelConfig, weights: *Weights, s: mlx.mlx_stream) !Trunk {
    const stq = config.dsv41.?.expert_dec != null;
    if (!config.expert_streaming or config.expert_layout != (if (stq) expert_quant_mod.Layout.exl3_k4 else .mxfp4_w123)) return error.ExpertStreamingRequired;
    // Decided before any kernel is made: the engine records each kernel as it is created.
    const own_decode = dsv41_borrow.ownDecode(&config.dsv41.?);
    if (own_decode) mlx.engine.enabled = true;
    if (own_decode) {
        const dsv41 = &config.dsv41.?;
        if (dsv41.expert_dec) |dec| expert_exl3_kernels.prepareGemmNax(dec, if (config.expert_tile_order == .pair_major) .tile_major else .pack);
        if (dsv41.trunk_dec) |dec| expert_exl3_kernels.prepareGemmNax(dec, .pack);
    }
    mlx.engine.seal(false);
    // A Cooler load's weights and expert slabs are the engine's own buffers, bound, sliced and cast by the engine: an op
    // on MLX's stream would take one as a second buffer over its pages.
    const own_load = dsv41_borrow.ownDecode(&config.dsv41.?);
    const engine = try Trunk.newExpertEngine(allocator, &config, if (own_load) mlx.engine.stream() else s);
    errdefer {
        engine.deinit();
        allocator.destroy(engine);
    }
    var plan = beginPipelines(allocator, &config, engine, own_decode);
    defer plan.deinit();
    // The load drops each tensor from the map once its arrays are bound.
    mlx.engine.own_weights = own_load;
    defer mlx.engine.own_weights = false;
    if (own_load) mlx.engine.active = true;
    const mdl = dsv41_model.init(allocator, config.dsv41.?, weights, config.expert_source_dir orelse return error.MissingExpertSourceDir, if (own_load) mlx.engine.stream() else s, .{ .experts = false, .dspark = config.dsparkOn(), .drop = engine.geometry.drop }) catch |e| {
        if (own_load) mlx.engine.active = false;
        return e;
    };
    errdefer mdl.deinit();
    if (own_load) {
        mlx.engine.active = false;
        mdl.s = s;
    }
    dsv41_model.logMemory(mdl);
    mdl.engine_decode = own_decode;
    const side: dsv41_borrow.DecodeSide = if (own_decode) .engine else dsv41_borrow.decodeSideOf(&config.dsv41.?);
    if (own_decode or side != .engine) log.info("{s}\n", .{dsv41_borrow.decodeLine(side)});
    // Prefill follows decode: a chunk of any width runs where a decode step does.
    mdl.engine_prefill = own_decode;
    if (own_decode) log.info("{s}\n", .{dsv41_borrow.prefill_line});
    dsv41_model.setFused(mdl, diagEnvOn("SUSHI_DSV41_COMPOSED"));
    if (diagEnvOn("SUSHI_DSV41_NO_DECODE_ENGRAM_AHEAD")) mdl.decode_engram_ahead = false;
    dsv41_model.setDsparkExact(mdl, if (std.c.getenv("SUSHI_DSV41_DSPARK_EXACT")) |v| diagEnvValueOn(v) else true);
    dsv41_model.setAttnNax(mdl, kernels.attn_nax_bar);
    dsv41_model.setIndexNax(mdl);
    dsv41_model.setReadDiag(diagEnvOn("SUSHI_DSV41_READ_DIAG"));
    if (mdl.ced_split) |split|
        log.info("[dsv41] CED: layers {d}..{d} run the last {d} rows of a prefill chunk\n", .{ split, mdl.cfg.layers - 1, mdl.cfg.window });
    log.info("[dsv41] prefill pipeline: {d} layers in flight\n", .{mdl.pipe_depth});
    const verify_profile = diagEnvOn("SUSHI_DSV41_VERIFY_PROFILE");
    if (verify_profile or diagEnvOn("SUSHI_DSV41_PREFILL_PROFILE")) {
        mdl.profile = try allocator.create(dsv41_model.Profile);
        mdl.profile.?.* = .init(verify_profile);
    }
    var shell = try initModuleShell(allocator, config, s);
    // Only the trunk half: `shell.arch_state` is filled last, so the model and engine have the errdefers above.
    errdefer shell.deinitTrunk();
    if (stq) {
        // CED runs the decoder over a chunk's last rows, so a prefill can take the decoder's pools for its working set.
        if (mdl.ced_split) |split| engine.borrow_from = @intCast(split);
        const experts = try allocator.create(dsv41_stream.StqStream);
        errdefer allocator.destroy(experts);
        experts.* = .{ .engine = engine, .cfg = &mdl.cfg };
        mdl.experts = .{ .hook = experts.hook() };
        shell.arch_state.dsv41.stq_experts = experts;
        try startExpertPrefetch(engine, mdl, config.ssdPrefetchPct());
    } else {
        const experts = try allocator.create(dsv41_stream.Stream);
        errdefer allocator.destroy(experts);
        experts.* = .{ .engine = engine, .cfg = &mdl.cfg, .act = mdl.act };
        mdl.experts = .{ .hook = experts.hook() };
        shell.arch_state.dsv41.experts = experts;
        if (prefetchIgnoredLine(&config)) |line| log.info("{s}\n", .{line});
    }
    shell.arch_state.dsv41.model = mdl;
    shell.expert_stream = engine;
    // A failed warm costs the first requests their compiles, not the load.
    var clock = expert_stream_mod.Clock.init();
    const had_error = mlx.errorPending();
    const warm_t0 = load_clock.now();
    const made_before = mlx.engine.pipelinesMade();
    const scope = plan.settle();
    if (warmAll(&shell, mdl, shell.arch_state.dsv41.stq_experts, scope)) {
        log.info("[dsv41] kernels warmed for 1..{d} rows{s}{s} in {d} ms\n", .{ expert_exl3_kernels.DECODE_ROWS_MAX, if (mdl.dspark != null) " and the DSpark rounds" else "", if (scope == .mlx) " (the engine's from the manifest)" else "", clock.lap() / std.time.ns_per_ms });
        if (scope == .all and own_decode) plan.store();
    } else |e| {
        mlx.dropLatchedErrorUnless(had_error);
        // A forward the bill does not cover would be a Metal OOM in a request.
        if (e == error.PoolOverBudget) return e;
        log.warn("[dsv41] kernel warm failed ({s}); requests compile their own\n", .{@errorName(e)});
    }
    const made = mlx.engine.pipelinesMade();
    const count_before = @import("dsv41_load.zig").takePipelinesBefore() orelse made_before.count;
    load_clock.warmDone(warm_t0, made.ns - made_before.ns + plan.made_ns, made.count - count_before + plan.made_count);
    mlx.engine.seal(true);
    // The warm's transients leave the pool, so the load's memory reading is the model's.
    _ = mlx.mlx_clear_cache();
    mlx.engine.trim();
    return shell;
}

/// PURE: what a load's warm runs after the model's own forwards. The sampled verdict and the wide chunks are the
/// engine's, so a manifest stands in for them; the spilled route's prefill chain is MLX's, which no manifest covers.
pub fn warmAfter(scope: dsv41_model.WarmScope, engine_prefill: bool) struct { sampled_verdict: bool, split: bool, wide: bool } {
    // The spilled route's chain on MLX is a prefill chunk's on MLX: a load that prefills on the engine never runs it.
    return .{ .sampled_verdict = scope == .all, .split = !engine_prefill, .wide = scope == .all and engine_prefill };
}

/// The warm of a load: with `.mlx` only the forwards on MLX, the engine's pipelines having come from a manifest. Every
/// forward of the engine runs under the budget a request's forward has (`dsv41_borrow.enterWarm`), and one the bill does
/// not cover fails the load.
pub fn warmAll(self: *Trunk, mdl: *dsv41_model.Model, stq: ?*dsv41_stream.StqStream, scope: dsv41_model.WarmScope) !void {
    return warmAllAt(self, mdl, stq, scope, dsv41_config.prefill_width);
}

/// `warmAll` with the widest wide chunk it runs.
pub fn warmAllAt(self: *Trunk, mdl: *dsv41_model.Model, stq: ?*dsv41_stream.StqStream, scope: dsv41_model.WarmScope, width: usize) !void {
    dsv41_borrow.warm_budget = true;
    defer dsv41_borrow.endWarmMlx();
    warmForwards(self, mdl, stq, scope, width) catch |e| {
        if (e == error.PoolOverBudget or mlx.takeErrorIf("PoolOverBudget")) {
            const r = mlx.engine.refusedBudget();
            log.err("[dsv41] a warm forward holds more than the bill it will have in a request: a buffer of {d} KB with {d} MB held for the pool, {d} MB of it asked by tensors in use, under a limit of {d} MB\n", .{ r.wanted >> 10, r.held >> 20, r.asked >> 20, r.limit >> 20 });
            return error.PoolOverBudget;
        }
        return e;
    };
}

fn warmForwards(self: *Trunk, mdl: *dsv41_model.Model, stq: ?*dsv41_stream.StqStream, scope: dsv41_model.WarmScope, width: usize) !void {
    try dsv41_model.warmKernelsFor(mdl, scope);
    const after = warmAfter(scope, mdl.engine_prefill);
    if (after.sampled_verdict) {
        dsv41_model.warmBudget(mdl, dsv41_config.verify_rows);
        try warmSampledVerdict(mdl);
        try warmSerialSampler(mdl);
    }
    if (after.split) if (stq) |st| try st.warmSplit(mdl, dsv41_config.prefill_width, false);
    if (after.wide) {
        var clock = expert_stream_mod.Clock.init();
        try warmWide(self, mdl, stq, width);
        log.info("[dsv41] wide kernels warmed in {d} ms\n", .{clock.lap() / std.time.ns_per_ms});
    }
}

/// The manifest to start the load from, its pipelines already being made. A load whose decode is on MLX makes none.
pub fn beginPipelines(allocator: std.mem.Allocator, config: *const ModelConfig, engine: *const expert_stream_mod.Engine, own_decode: bool) dsv41_pipelines.Plan {
    const io = std.Io.Threaded.global_single_threaded.io();
    const none: manifest_mod.Key = .{ .scope = @splat('0'), .build = @splat('0') };
    if (!own_decode) return dsv41_pipelines.begin(allocator, io, null, "decode runs on MLX", none, 0);
    const t0 = load_clock.now();
    const scope = dsv41_pipelines.scopeKey(allocator, io, config, engine) catch |e| return dsv41_pipelines.begin(allocator, io, null, @errorName(e), none, 0);
    const build = dsv41_pipelines.buildKey(allocator, io) catch |e| return dsv41_pipelines.begin(allocator, io, null, @errorName(e), none, 0);
    const key_ms = (load_clock.now() - t0) / std.time.ns_per_ms;
    var reason: []const u8 = "";
    const root = dsv41_pipelines.openRoot(io, &reason);
    return dsv41_pipelines.begin(allocator, io, root, reason, .{ .scope = scope, .build = build }, key_ms);
}

/// The wide forwards a prompt reaches, run once on the engine at the load so their kernels compile here: a full-width
/// state-only chunk and a head chunk (CED), head chunks whose rows fill the GEMM tiles whole, in part and not at all, the
/// held-back tail, and one chunk per pick-sort plan (`sort.planOf`: its block size follows the picks, a row's `top_k` of
/// them). Chunks wider than `width` are left out. Leaves no state.
pub fn warmWide(self: *Trunk, m: *dsv41_model.Model, stq: ?*dsv41_stream.StqStream, width: usize) !void {
    // The split chain is a wide chunk's work but no forward: it takes the budget such a chunk has, not the one the
    // decode-width forward before it left.
    dsv41_borrow.enterWarmMlx(width, width, &m.cfg);
    if (stq) |st| try st.warmSplit(m, width, true);
    const tail = @import("../prefill_plan.zig").SSM_SNAPSHOT_BACKOFF + 1;
    const k: usize = m.layers[0].top_k;
    var plans: [64][2]usize = undefined;
    var n_plans: usize = 0;
    const heads = [_]usize{ width, width * 17 / 32, width * 1100 / 2048 };
    plans[0] = .{ width, width };
    n_plans = 1;
    for (heads[1..]) |w| {
        plans[n_plans] = .{ w, tail };
        n_plans += 1;
    }
    // Rows that fill the narrow GEMM tiles whole.
    for ([_]usize{ 64, 128 }) |rows| {
        plans[n_plans] = .{ rows, 0 };
        n_plans += 1;
    }
    // A compressing layer's projection takes the chunk's compressed rows (`rows / ratio`), and a kernel of under 17 rows has
    // that count in its name: one chunk per count.
    var seen: [256]bool = @splat(false);
    for (m.cfg.ratios[0..m.cfg.layers]) |ratio| {
        if (ratio < 2 or ratio > 16 or seen[ratio]) continue;
        seen[ratio] = true;
        for (1..17) |c| {
            const rows = @max(c * ratio, dsv41_borrow.verify_rows + 1);
            if (rows >= (c + 1) * ratio or rows > width or n_plans == plans.len) continue;
            plans[n_plans] = .{ rows, 0 };
            n_plans += 1;
        }
    }
    // The widest chunk of each single-block sort size, and one past them all.
    for ([_]usize{ 128, 256, 512, 1024, 2048 }) |picks| {
        plans[n_plans] = .{ picks / k, 0 };
        n_plans += 1;
    }
    plans[n_plans] = .{ width, 0 };
    n_plans += 1;
    const ids = try m.gpa.alloc(i32, width);
    defer m.gpa.free(ids);
    for (ids, 0..) |*t, i| t.* = @intCast((i * 7 + 3) % m.cfg.vocab);
    for (plans[0..n_plans]) |plan| {
        const chunks = if (plan[1] == 0) plan[0..1] else plan[0..2];
        if (!planFits(chunks, width)) continue;
        var plan_clock = expert_stream_mod.Clock.init();
        try self.resetCache();
        const reset_ms = plan_clock.lap() / std.time.ns_per_ms;
        defer log.debug("[dsv41] wide warm {any}: reset {d} ms, forwards {d} ms\n", .{ chunks, reset_ms, plan_clock.lap() / std.time.ns_per_ms });
        var ctx = self.defaultCtx();
        for (chunks, 0..) |w, i| {
            ctx.ahead_min_rows = if (i + 1 < chunks.len) chunks[i + 1] else 0;
            const arr = mlx.mlx_array_new_data(ids.ptr, &[_]c_int{ 1, @intCast(w) }, 2, .int32);
            defer _ = mlx.mlx_array_free(arr);
            const out = try forwardDsv41With(self, &ctx, arr, m);
            defer _ = mlx.mlx_array_free(out);
            if (out.ctx != null) try mlx.check(mlx.mlx_array_eval(out));
        }
    }
    try self.resetCache();
}

fn planFits(plan: []const usize, width: usize) bool {
    for (plan) |w| if (w > width or w <= dsv41_borrow.verify_rows) return false;
    return true;
}

/// The draws of a sampled round: `rows` verify rows of `logits` (`[1, >= rows, vocab]`) filtered by `sampling` and
/// drawn one per row, row j for the generated position `sampling.draw + j`, gathered as one `[rows]` array not yet
/// evaluated. On the engine's stream the gathering vector is an engine vector, which the engine makes only while it
/// is active, and a row's key is `random.drawKey(sampling.draw_root, position)`, the request's seed or its OS-given
/// root, and MLX is not asked for a key or bits. On MLX's stream the sampler keys by `sampling.seed`.
pub fn sampledDraws(sampling: sampler.SamplingParams, logits: mlx.mlx_array, rows: usize, s: mlx.mlx_stream) !mlx.mlx_array {
    const own = mlx.engine.isStream(s);
    if (own) mlx.engine.active = true;
    defer if (own) {
        mlx.engine.active = false;
    };
    var draws: [dsv41_config.ring_slack]mlx.mlx_array = undefined;
    if (rows > draws.len) return error.SpecDecodeUnsupported;
    const probs = try sampler.probsAllPositions(logits, sampling, s);
    defer _ = mlx.mlx_array_free(probs);
    const rows2d = try spec_types.verifyRows2d(probs, @intCast(rows - 1), s);
    defer _ = mlx.mlx_array_free(rows2d);
    const vocab = mlx.getShape(rows2d)[1];
    var built: usize = 0;
    defer for (draws[0..built]) |d| {
        _ = mlx.mlx_array_free(d);
    };
    const vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    for (0..rows) |j| {
        var row = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(row);
        const at: c_int = @intCast(j);
        try mlx.check(mlx.mlx_slice(&row, rows2d, &.{ at, 0 }, 2, &.{ at + 1, vocab }, 2, &.{ 1, 1 }, 2, s));
        var keyed = sampling;
        keyed.draw +%= j;
        draws[built] = if (own) try engineDraw(row, random.drawKey(sampling.draw_root orelse return error.EngineDrawNeedsRoot, keyed.draw), s) else try sampler.sampleFromProbsLazy(row, keyed, s);
        built += 1;
        _ = mlx.mlx_vector_array_append_value(vec, draws[j]);
    }
    var flat = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_concatenate_axis(&flat, vec, 0, s));
    return flat;
}

/// `sampler.sampleFromProbsLazy` on the engine under a key of its own.
fn engineDraw(probs: mlx.mlx_array, key: [2]u32, s: mlx.mlx_stream) !mlx.mlx_array {
    var drawn = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(drawn);
    try sampler.drawFromProbs(&drawn, probs, key, s);
    return drawn;
}

const warm_root: u64 = 1;

/// The sampled verdict's chain on the engine, over every verify width and the four filter settings with a suppress
/// mask: each stock pipeline it launches compiles here, not inside a request. A seeded and an unseeded request run the
/// same chain (they differ in the root only), so the warm runs it unseeded. A request's own top-k only changes the
/// shapes the sorts run at, and so which block-sort variants.
pub fn warmSampledVerdict(mdl: *dsv41_model.Model) !void {
    if (!mdl.engine_decode or mdl.dspark == null) return;
    const s = mlx.engine.stream();
    const vocab: c_int = @intCast(mdl.cfg.vocab);
    var mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask);
    try mlx.check(mlx.mlx_zeros(&mask, &[_]c_int{vocab}, 1, .bool_, s));
    for (1..dsv41_config.ring_slack + 1) |rows| {
        var logits = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(logits);
        try mlx.check(mlx.mlx_zeros(&logits, &[_]c_int{ 1, @intCast(rows), vocab }, 3, .float32, s));
        for ([_][2]f32{ .{ 0, 1 }, .{ 40, 1 }, .{ 0, 0.95 }, .{ 40, 0.95 } }) |filter| {
            const sampling: sampler.SamplingParams = .{ .temperature = 0.8, .top_k = @intFromFloat(filter[0]), .top_p = filter[1], .suppress_mask = mask, .draw_root = warm_root };
            const draws = try sampledDraws(sampling, logits, rows, s);
            defer _ = mlx.mlx_array_free(draws);
            try mlx.check(mlx.mlx_array_eval(draws));
        }
    }
}

/// The serial sampler's chains on the engine over a forward's `[1, 1, vocab]` logits, and the penalty shifts of a verify
/// block at every width: each pipeline a request's first token, a serial step or a round with penalties launches
/// compiles here. The chains are the shared sampler's own functions, so what a request runs is what is warmed.
pub fn warmSerialSampler(mdl: *dsv41_model.Model) !void {
    if (!mdl.engine_decode) return;
    const s = mlx.engine.stream();
    const host = mlx.engine.enter(s);
    defer host.end();
    const a = std.heap.page_allocator;
    const vocab: c_int = @intCast(mdl.cfg.vocab);
    var mask = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(mask);
    try mlx.check(mlx.mlx_zeros(&mask, &[_]c_int{vocab}, 1, .bool_, s));
    var logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(logits);
    try mlx.check(mlx.mlx_zeros(&logits, &[_]c_int{ 1, 1, vocab }, 3, .float32, s));
    const seen = [_]u32{ 1, 2, 2, 7 };
    const penalties = [_]sampler.SamplingParams{
        .{ .temperature = 0.8, .repeat_penalty = 1.1 },
        .{ .temperature = 0.8, .presence_penalty = 0.5, .frequency_penalty = 0.5 },
        .{ .temperature = 0.8, .repeat_penalty = 1.1, .presence_penalty = 0.5, .frequency_penalty = 0.5 },
    };
    for ([_]bool{ false, true }) |masked| {
        for ([_][2]f32{ .{ 0, 1 }, .{ 40, 1 }, .{ 0, 0.95 }, .{ 40, 0.95 } }) |filter| {
            var sampling: sampler.SamplingParams = .{ .temperature = 0.8, .top_k = @intFromFloat(filter[0]), .top_p = filter[1], .draw_root = warm_root, .suppress_mask = if (masked) mask else null };
            const drawn = sampler.sampleTokenLazy(logits, sampling, s);
            defer _ = mlx.mlx_array_free(drawn);
            try mlx.check(mlx.mlx_array_eval(drawn));
            sampling.temperature = 0;
            const best = sampler.sampleTokenLazy(logits, sampling, s);
            defer _ = mlx.mlx_array_free(best);
            try mlx.check(mlx.mlx_array_eval(best));
        }
        var sampling: sampler.SamplingParams = .{ .temperature = 0.8, .top_k = 40, .top_p = 0.95, .draw_root = warm_root, .suppress_mask = if (masked) mask else null };
        for (penalties) |p| {
            sampling.repeat_penalty = p.repeat_penalty;
            sampling.presence_penalty = p.presence_penalty;
            sampling.frequency_penalty = p.frequency_penalty;
            const drawn = try sampler.penalizedSampleLazy(a, logits, sampling, &seen, s);
            defer _ = mlx.mlx_array_free(drawn);
            try mlx.check(mlx.mlx_array_eval(drawn));
            const tick = try sampler.sampleToken(a, logits, sampling, &seen, 3, null, s);
            a.free(tick.logprob_result.?.top_logprobs);
        }
    }
    const grammar = try a.alloc(bool, mdl.cfg.vocab);
    defer a.free(grammar);
    @memset(grammar, true);
    grammar[0] = false;
    var masked_logits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(masked_logits);
    try sampler.applyGrammarMask(a, &masked_logits, logits, grammar, s);
    try mlx.check(mlx.mlx_array_eval(masked_logits));
    for (1..dsv41_config.ring_slack + 1) |rows| try warmShiftRows(mdl, rows, mask);
}

/// A verify block of `rows` rows under the penalties, the thinking penalty and its logit bias, and a row mask.
fn warmShiftRows(mdl: *dsv41_model.Model, rows: usize, marker_mask: mlx.mlx_array) !void {
    const s = mlx.engine.stream();
    const a = std.heap.page_allocator;
    const vocab: c_int = @intCast(mdl.cfg.vocab);
    var block = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(block);
    try mlx.check(mlx.mlx_zeros(&block, &[_]c_int{ 1, @intCast(rows), vocab }, 3, .float32, s));
    var ids_host: [dsv41_config.ring_slack]i32 = undefined;
    for (ids_host[0..rows], 0..) |*id, i| id.* = @intCast(i + 1);
    const ids = mlx.mlx_array_new_data(&ids_host, &[_]c_int{ 1, @intCast(rows) }, 2, .int32);
    defer _ = mlx.mlx_array_free(ids);
    const committed = try seen_penalty.buildCounts(a, &.{ 1, 2, 2 }, mdl.cfg.vocab);
    defer _ = mlx.mlx_array_free(committed);
    const bias = [_]logit_bias.Bias{.{ .id = 3, .delta = 1 }};
    const pens = [_]seen_penalty.Penalties{ .{ .repeat = 1.1 }, .{ .presence = 0.5, .frequency = 0.5 }, .{ .repeat = 1.1, .presence = 0.5, .frequency = 0.5 } };
    for (pens) |pen| {
        var out = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(out);
        try seen_penalty.shiftRows(&out, block, ids, committed, pen, s);
        try mlx.check(mlx.mlx_array_eval(out));
    }
    for ([_]think_penalty.ThinkPenalty.Phase{ .before, .inside }) |phase| for ([_][]const logit_bias.Bias{ &.{}, &bias }) |biases| {
        var p: think_penalty.ThinkPenalty = .{ .lambda = 1, .opener_id = 1, .closer_id = 2, .phase = phase, .biases = biases };
        try p.prepare(a, mdl.cfg.vocab, marker_mask, s);
        defer p.deinitPrepared();
        var out = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(out);
        try think_penalty.shiftScoped(&out, block, marker_mask, p, ids, s);
        try mlx.check(mlx.mlx_array_eval(out));
    };
    const rows_mask = try a.alloc(bool, rows * mdl.cfg.vocab);
    defer a.free(rows_mask);
    @memset(rows_mask, true);
    rows_mask[0] = false;
    var masked = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(masked);
    try sampler.applyRowMasks(&masked, block, rows_mask, rows, mdl.cfg.vocab, s);
    try mlx.check(mlx.mlx_array_eval(masked));
}

/// A pack without the STQ stream has no expert prefetcher: `--ssd-prefetch` or the file's `ssd_prefetch` does nothing there.
fn prefetchIgnoredLine(config: *const ModelConfig) ?[]const u8 {
    if (config.ssd_prefetch != null) return "[dsv41] --ssd-prefetch is ignored: this pack has no STQ experts, so there is no expert prefetcher";
    if (config.ssd_prefetch_override != null) return "[dsv41] model-settings.json ssd_prefetch is ignored: this pack has no STQ experts, so there is no expert prefetcher";
    return null;
}

/// Starts the expert prefetch at `pct` percent of the read rate the serving pool measures now, or none at 0 or on a
/// resident load. A failure to start leaves the load without a prefetch.
fn startExpertPrefetch(engine: *expert_stream_mod.Engine, mdl: *dsv41_model.Model, pct: u8) !void {
    if (pct == 0) {
        log.info("[dsv41] expert prefetch off (--ssd-prefetch 0)\n", .{});
        return;
    }
    if (engine.plan.resident) {
        log.info("[dsv41] expert prefetch off (every routed expert is resident)\n", .{});
        return;
    }
    const probe = engine.probeReadRate() catch |e| {
        log.warn("[dsv41] expert prefetch off: the read-rate probe failed ({s})\n", .{@errorName(e)});
        return;
    };
    if (probe.max_bps <= 0) {
        log.warn("[dsv41] expert prefetch off: the read-rate probe read nothing\n", .{});
        return;
    }
    const targets = engine.prefetchTargets(prefetch_lead_rows);
    const credit = PrefetchCredit.init(pct, probe.max_bps, targets);
    engine.startPrefetch(credit) catch |e| {
        log.warn("[dsv41] expert prefetch off: {s}\n", .{@errorName(e)});
        return;
    };
    mdl.prefetch_lead_rows = prefetch_lead_rows;
    log.info("[dsv41] expert prefetch on: {d}% of {d:.1} GB/s (probe k2/k4/k8 {d:.1}/{d:.1}/{d:.1} GB/s)\n", .{ pct, probe.max_bps / 1e9, probe.bps[0] / 1e9, probe.bps[1] / 1e9, probe.bps[2] / 1e9 });
}

/// Whether the chunk's CED decoder tail is read by nothing (`dsv41_model.extendState`): no restore point at its end,
/// and the next chunk of the loop replays a full window, whose rows overwrite every ring slot a query reads. A next
/// chunk of `window` rows or fewer is a full pass that reads the rings this chunk's tail would have filled.
pub fn tailUnread(window: usize, rows: usize, ahead_min_rows: usize, holds_point: bool) bool {
    return window > 0 and !holds_point and rows > window and ahead_min_rows > window;
}

/// One request at a time: the model holds its state, which `cache.step` mirrors. Logits are the last row.
pub fn forwardDsv41With(self: *Trunk, ctx: *ForwardCtx, ids: mlx.mlx_array, m: *dsv41_model.Model) !mlx.mlx_array {
    if (ctx.cache.step == 0) dsv41_model.resetState(m);
    const st = try dsv41_model.ensureState(m);
    if (st.n != ctx.cache.step) return error.Dsv41CachePositionMismatch;
    m.s = self.s;
    const rows = mlx.getShape(ids);
    if (rows.len != 2 or rows[0] != 1) return error.InvalidDsv41Input;
    const n: usize = @intCast(rows[1]);
    // Where decode runs on the engine every forward does, at a decode width or a prefill's, with image rows or without.
    // On MLX a one-row greedy decode reads its token while layer 0 runs (`extendLazy`).
    const decode_width = m.engine_decode and (n <= expert_exl3_kernels.DECODE_ROWS_MAX or m.engine_prefill);
    const lazy = !(decode_width and ctx.vision_embeddings == null) and n == 1 and ctx.lazy_logits and ctx.vision_embeddings == null and ctx.ahead_ids.len == 0 and st.verify == null and dsv41_model.lazyStartOk(m);
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_astype(&flat, ids, .uint32, if (decode_width) mlx.engine.stream() else self.s));
    // Queued before any host work, so the token is ready as soon as the head is.
    if (lazy) try dsv41_model.submit(flat) else try mlx.check(mlx.mlx_array_eval(flat));
    var one: [1]u32 = undefined;
    const host: []const u32 = if (lazy) &one else (mlx.mlx_array_data_uint32(flat) orelse return error.InvalidDsv41Input)[0..n];
    if (self.expert_stream) |engine| {
        try dsv41_borrow.beforeForward(engine, &self.config, &m.cfg, n, st.n + n);
        engine.beginForward();
    }
    // The model splices the image rows itself: the span's embedding never leaves it.
    const vision: ?dsv41_model.VisionRows = if (ctx.vision_embeddings) |ve| .{ .rows = ve, .offset = ctx.vision_splice_offset } else null;
    const ahead = try m.gpa.alloc(u32, ctx.ahead_ids.len);
    defer m.gpa.free(ahead);
    for (ctx.ahead_ids, ahead) |t, *o| o.* = @intCast(t);
    const own = decode_width;
    if (m.ced_split != null and tailUnread(m.cfg.window, n, ctx.ahead_min_rows, ctx.holds_point) and st.verify == null) {
        if (own) _ = dsv41_model.ownBegin(m);
        const state_only = dsv41_model.extendState(m, host, dsv41_model.SPAN, vision, ahead);
        if (own) dsv41_model.ownEnd(m, self.s);
        try state_only;
        if (self.expert_stream) |engine| engine.finishForward(n);
        ctx.cache.step = st.n;
        return mlx.mlx_array_new();
    }
    m.defer_head = m.lazy_head and ctx.lazy_logits and n == 1;
    defer m.defer_head = false;
    if (own) keepMlxQueue(m);
    if (own) _ = dsv41_model.ownBegin(m);
    if (own) mlx.engine.phase(1);
    defer mlx.engine.phase(0);
    const forwarded = if (lazy) dsv41_model.extendLazy(m, flat, &one) else dsv41_model.extendAhead(m, host, .last, dsv41_model.SPAN, vision, ahead);
    if (own) dsv41_model.ownEnd(m, self.s);
    const last = try forwarded;
    defer _ = mlx.mlx_array_free(last);
    if (self.expert_stream) |engine| engine.finishForward(n);
    ctx.cache.step = st.n;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, last, &[_]c_int{ 1, 1, @intCast(m.cfg.vocab) }, 3, if (own) mlx.engine.stream() else self.s));
    return out;
}

/// MLX's queue runs nothing while decode is on the engine's, and the driver drops the resident set of a queue left
/// unused: the next MLX submission (a request's end, a prefill) then waits before any work runs. One MLX op per
/// warm tick keeps it, as the parked inference thread does between requests.
pub fn keepMlxQueue(m: *dsv41_model.Model) void {
    const now = mlx.engine.nowNs();
    if (now -| m.mlx_queue_used_ns < sched_policy.GPU_WARM_TICK_NS) return;
    m.mlx_queue_used_ns = now;
    mlx.gpuTouch();
}

/// DeepSeek-V4.1's DSpark round on the request's state (`ctx.cache.step` mirrors it): the drafter's block from `t1`.
pub fn dsv41DraftImpl(self: *Trunk, t1: u32) !dsv41_model.Draft {
    const m = dsv41_state.modelOf(self) orelse return error.Dsv41NoDspark;
    m.s = self.s;
    if (m.engine_decode) _ = dsv41_model.ownBegin(m);
    defer if (m.engine_decode) dsv41_model.ownEnd(m, self.s);
    mlx.engine.phase(2);
    defer mlx.engine.phase(0);
    return dsv41_model.draftHeads(m, t1, .conf);
}

/// Advances the request by `ids` (`[t1, drafts...]`) as a chunk `dsv41Finish` trims; the logits of every row,
/// f32 `[1, L, vocab]`.
pub fn dsv41VerifyImpl(self: *Trunk, ctx: *ForwardCtx, ids: []const u32) !mlx.mlx_array {
    const m = dsv41_state.modelOf(self) orelse return error.Dsv41NoDspark;
    m.s = self.s;
    const st = try dsv41_model.ensureState(m);
    if (st.n != ctx.cache.step) return error.Dsv41CachePositionMismatch;
    if (self.expert_stream) |engine| {
        try dsv41_borrow.beforeForward(engine, &self.config, &m.cfg, ids.len, st.n + ids.len);
        engine.beginForward();
    }
    if (m.engine_decode) keepMlxQueue(m);
    if (m.engine_decode) _ = dsv41_model.ownBegin(m);
    mlx.engine.phase(3);
    defer mlx.engine.phase(0);
    const verified = dsv41_model.verify(m, ids);
    if (m.engine_decode) dsv41_model.ownEnd(m, self.s);
    const all = try verified;
    defer _ = mlx.mlx_array_free(all);
    if (self.expert_stream) |engine| engine.finishForward(ids.len);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    // On the stream that made the logits: MLX's would wait for the head here, ahead of the verdict built on them.
    try mlx.check(mlx.mlx_reshape(&out, all, &[_]c_int{ 1, @intCast(ids.len), @intCast(m.cfg.vocab) }, 3, verdictStream(self, self.s)));
    return out;
}

/// The stream a round's verdict is taken on: the engine's while decode runs there, so the masked argmax rides the
/// verify's own passes and its read is the round's one wait.
pub fn verdictStream(self: *Trunk, s: mlx.mlx_stream) mlx.mlx_stream {
    const m = dsv41_state.modelOf(self) orelse return s;
    return if (m.engine_decode) mlx.engine.stream() else s;
}

/// Keeps the first `accepted + 1` rows of the open verify chunk.
pub fn dsv41FinishImpl(self: *Trunk, ctx: *ForwardCtx, accepted: usize) !void {
    const m = dsv41_state.modelOf(self) orelse return error.Dsv41NoDspark;
    m.s = self.s;
    {
        if (m.engine_decode) _ = dsv41_model.ownBegin(m);
        defer if (m.engine_decode) dsv41_model.ownEnd(m, self.s);
        mlx.engine.phase(4);
        defer mlx.engine.phase(0);
        try dsv41_model.finishVerify(m, accepted);
    }
    ctx.cache.step = (try dsv41_model.ensureState(m)).n;
}

test "a non-STQ DeepSeek pack names an ssd_prefetch it ignores, from the flag or the file" {
    var config = ModelConfig{};
    try std.testing.expect(prefetchIgnoredLine(&config) == null);
    config.ssd_prefetch_override = 30;
    try std.testing.expect(std.mem.indexOf(u8, prefetchIgnoredLine(&config).?, "model-settings.json ssd_prefetch is ignored") != null);
    config.ssd_prefetch = 50;
    try std.testing.expect(std.mem.indexOf(u8, prefetchIgnoredLine(&config).?, "--ssd-prefetch is ignored") != null);
}

test "dsv41 warm: a manifest stands in for the engine's sampled verdict and wide chunks; MLX's spilled-route prefill chain is warmed only where prefill is MLX's" {
    const full = warmAfter(.all, true);
    try std.testing.expect(full.sampled_verdict and full.wide and !full.split);
    const from_manifest = warmAfter(.mlx, true);
    try std.testing.expect(!from_manifest.sampled_verdict and !from_manifest.wide and !from_manifest.split);
    // Prefill on MLX (the opt-out) leaves the engine no wide chunk to warm, and MLX its chain, manifest or not.
    const prefill_on_mlx = warmAfter(.all, false);
    try std.testing.expect(prefill_on_mlx.sampled_verdict and prefill_on_mlx.split);
    try std.testing.expect(!prefill_on_mlx.wide);
    try std.testing.expect(warmAfter(.mlx, false).split);
}

test "tailUnread: a chunk is state only when no point is taken and the next chunk replays a full window" {
    const t = std.testing;
    try t.expect(tailUnread(128, 2048, 2048, false));
    try t.expect(!tailUnread(128, 2048, 2048, true));
    try t.expect(!tailUnread(128, 2048, 0, false));
    try t.expect(!tailUnread(128, 2048, 128, false));
    try t.expect(tailUnread(128, 2048, 129, false));
    try t.expect(!tailUnread(128, 128, 2048, false));
    try t.expect(!tailUnread(0, 2048, 2048, false));
}
