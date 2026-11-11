const trunk_mod = @import("../trunk.zig");
const std = @import("std");
const dsv41_model = @import("dsv41_model.zig");
const dsv41_state = @import("dsv41_state.zig");
const dsv41_stream = @import("dsv41_stream.zig");
const kernels = @import("dsv41_kernels.zig");
const dsv41_borrow = @import("dsv41_borrow.zig");
const dsv41_config = @import("dsv41_config.zig");
const expert_stream_mod = @import("../expert_stream.zig");
const PrefetchCredit = @import("../prefetch_plan.zig").PrefetchCredit;
const sushi_exl3 = @import("sushi_exl3");
const expert_exl3_kernels = sushi_exl3.kernels;
const expert_quant_mod = @import("../expert_quant.zig");
const mlx = @import("../mlx.zig");
const model_mod = @import("../model.zig");
const log = @import("../log.zig");
const load_clock = @import("../load_clock.zig");
const sampler = @import("../sampler.zig");
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
    const engine = try Trunk.newExpertEngine(allocator, &config, s);
    errdefer {
        engine.deinit();
        allocator.destroy(engine);
    }
    // The load drops each tensor from the map once its arrays are bound.
    const mdl = try dsv41_model.init(allocator, config.dsv41.?, weights, config.expert_source_dir orelse return error.MissingExpertSourceDir, s, .{ .experts = false, .dspark = config.dsparkOn(), .drop = engine.geometry.drop });
    errdefer mdl.deinit();
    dsv41_model.logMemory(mdl);
    mdl.engine_decode = own_decode;
    if (own_decode) {
        log.info("[dsv41] decode forwards (one row, DSpark draft and verify rows): on the engine of our own\n", .{});
    } else switch (dsv41_borrow.decodeSideOf(&config.dsv41.?)) {
        .engine => {},
        .mlx_not_stq => log.info("[dsv41] decode forwards: on MLX (the engine of our own has no quantized_matmul for an FP8 trunk or MXFP4 experts)\n", .{}),
        .mlx_no_matrix_units => log.info("[dsv41] decode forwards: on MLX (the engine of our own is proven on a GPU with matrix units only)\n", .{}),
    }
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
    if (warmAll(mdl, shell.arch_state.dsv41.stq_experts)) {
        log.info("[dsv41] kernels warmed for 1..{d} rows{s} in {d} ms\n", .{ expert_exl3_kernels.DECODE_ROWS_MAX, if (mdl.dspark != null) " and the DSpark rounds" else "", clock.lap() / std.time.ns_per_ms });
    } else |e| {
        mlx.dropLatchedErrorUnless(had_error);
        log.warn("[dsv41] kernel warm failed ({s}); requests compile their own\n", .{@errorName(e)});
    }
    const made = mlx.engine.pipelinesMade();
    load_clock.warmDone(warm_t0, made.ns - made_before.ns, made.count - made_before.count);
    mlx.engine.seal(true);
    // The warm's transients leave the pool, so the load's memory reading is the model's.
    _ = mlx.mlx_clear_cache();
    mlx.engine.trim();
    return shell;
}

pub fn warmAll(mdl: *dsv41_model.Model, stq: ?*dsv41_stream.StqStream) !void {
    try dsv41_model.warmKernels(mdl);
    try warmSampledVerdict(mdl);
    if (stq) |st| try st.warmSplit(mdl, dsv41_config.prefill_width);
}

/// The draws of a sampled round: `rows` verify rows of `logits` (`[1, >= rows, vocab]`) filtered by `sampling` and
/// drawn one per row, row j under the key of draw `sampling.draw + j`, gathered as one `[rows]` array not yet
/// evaluated. On the engine's stream the gathering vector is an engine vector, which the engine makes only while it
/// is active.
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
        draws[built] = try sampler.sampleFromProbsLazy(row, keyed, s);
        built += 1;
        _ = mlx.mlx_vector_array_append_value(vec, draws[j]);
    }
    var flat = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_concatenate_axis(&flat, vec, 0, s));
    return flat;
}

/// The sampled verdict's chain on the engine, over every verify width and the four filter settings with a suppress
/// mask, seeded: each stock pipeline it launches compiles here, not inside a request. The unseeded path reaches no
/// pipeline this does not (its bits come from MLX's next key); it is left out so the load does not advance MLX's key
/// sequence. A request's own top-k only changes the shapes the sorts run at, and so which block-sort variants.
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
            const sampling: sampler.SamplingParams = .{ .temperature = 0.8, .top_k = @intFromFloat(filter[0]), .top_p = filter[1], .seed = 1, .suppress_mask = mask };
            const draws = try sampledDraws(sampling, logits, rows, s);
            defer _ = mlx.mlx_array_free(draws);
            try mlx.check(mlx.mlx_array_eval(draws));
        }
    }
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
    // A forward of up to a verify block's rows is a decode-width forward, on the engine; one that splices image rows
    // stays on MLX. A one-row greedy decode reads its token while layer 0 runs (`extendLazy`).
    const decode_width = m.engine_decode and n <= expert_exl3_kernels.DECODE_ROWS_MAX and st.verify == null;
    const lazy = !(decode_width and ctx.vision_embeddings == null) and n == 1 and ctx.lazy_logits and ctx.vision_embeddings == null and ctx.ahead_ids.len == 0 and st.verify == null and dsv41_model.lazyStartOk(m);
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_astype(&flat, ids, .uint32, self.s));
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
    const own = decode_width and (vision == null or dsv41_model.imageRows(m, host) == 0);
    if (m.ced_split != null and tailUnread(m.cfg.window, n, ctx.ahead_min_rows, ctx.holds_point) and st.verify == null) {
        try dsv41_model.extendState(m, host, dsv41_model.SPAN, vision, ahead);
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
    try mlx.check(mlx.mlx_reshape(&out, last, &[_]c_int{ 1, 1, @intCast(m.cfg.vocab) }, 3, self.s));
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
