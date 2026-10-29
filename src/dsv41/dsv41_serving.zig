const trf = @import("../transformer.zig");
const std = @import("std");
const dsv41_model = @import("dsv41_model.zig");
const dsv41_stream = @import("dsv41_stream.zig");
const dsv41_borrow = @import("dsv41_borrow.zig");
const expert_stream_mod = @import("../expert_stream.zig");
const sushi_exl3 = @import("sushi_exl3");
const expert_exl3_kernels = sushi_exl3.kernels;
const expert_quant_mod = @import("../expert_quant.zig");
const mlx = @import("../mlx.zig");
const model_mod = @import("../model.zig");
const log = @import("../log.zig");
const io_util_mod = @import("../io_util.zig");
const ModelConfig = model_mod.ModelConfig;
const Weights = model_mod.Weights;
const diagEnvValueOn = io_util_mod.diagEnvValueOn;
const diagEnvOn = io_util_mod.diagEnvOn;
const ForwardCtx = trf.ForwardCtx;
const Transformer = trf.Transformer;
const initModuleShell = trf.initModuleShell;

pub fn initDsv41(allocator: std.mem.Allocator, config: ModelConfig, weights: *const Weights, s: mlx.mlx_stream) !Transformer {
    const stq = config.dsv41.?.expert_dec != null;
    if (!config.expert_streaming or config.expert_layout != (if (stq) expert_quant_mod.Layout.exl3_k4 else .mxfp4_w123)) return error.ExpertStreamingRequired;
    const engine = try Transformer.newExpertEngine(allocator, &config, s);
    errdefer {
        engine.deinit();
        allocator.destroy(engine);
    }
    // The load drops each tensor from the map once its arrays are bound.
    const mdl = try dsv41_model.init(allocator, config.dsv41.?, @constCast(weights), config.expert_source_dir orelse return error.MissingExpertSourceDir, s, .{ .experts = false, .dspark = config.dsparkOn(), .drop = engine.geometry.drop });
    errdefer mdl.deinit();
    dsv41_model.logMemory(mdl);
    dsv41_model.setFused(mdl, diagEnvOn("SUSHI_DSV41_COMPOSED"));
    if (diagEnvOn("SUSHI_DSV41_NO_DECODE_ENGRAM_AHEAD")) mdl.decode_engram_ahead = false;
    dsv41_model.setDsparkExact(mdl, if (std.c.getenv("SUSHI_DSV41_DSPARK_EXACT")) |v| diagEnvValueOn(v) else true);
    dsv41_model.setReadDiag(diagEnvOn("SUSHI_DSV41_READ_DIAG"));
    if (mdl.ced_split) |split|
        log.info("[dsv41] CED: layers {d}..{d} run the last {d} rows of a prefill chunk\n", .{ split, mdl.cfg.layers - 1, mdl.cfg.window });
    const verify_profile = diagEnvOn("SUSHI_DSV41_VERIFY_PROFILE");
    if (verify_profile or diagEnvOn("SUSHI_DSV41_PREFILL_PROFILE")) {
        mdl.profile = try allocator.create(dsv41_model.Profile);
        mdl.profile.?.* = .init(verify_profile);
    }
    var shell = try initModuleShell(allocator, config, s);
    if (stq) {
        // CED runs the decoder over a chunk's last rows, so a prefill can take the decoder's pools for its working set.
        if (mdl.ced_split) |split| engine.borrow_from = @intCast(split);
        const experts = try allocator.create(dsv41_stream.StqStream);
        experts.* = .{ .engine = engine, .cfg = &mdl.cfg };
        mdl.experts = .{ .hook = experts.hook() };
        shell.dsv41_stq_experts = experts;
        if (engine.startPrefetch()) {
            mdl.prefetch_lead_rows = 3;
        } else |e| log.warn("[dsv41] expert prefetch off: {s}\n", .{@errorName(e)});
    } else {
        const experts = try allocator.create(dsv41_stream.Stream);
        experts.* = .{ .engine = engine, .cfg = &mdl.cfg, .act = mdl.act };
        mdl.experts = .{ .hook = experts.hook() };
        shell.dsv41_experts = experts;
    }
    shell.dsv41 = mdl;
    shell.expert_stream = engine;
    // A failed warm costs the first requests their compiles, not the load.
    var clock = expert_stream_mod.Clock.init();
    const had_error = mlx.errorPending();
    if (dsv41_model.warmKernels(mdl)) {
        log.info("[dsv41] kernels warmed for 1..{d} rows{s} in {d} ms\n", .{ expert_exl3_kernels.DECODE_ROWS_MAX, if (mdl.dspark != null) " and the DSpark rounds" else "", clock.lap() / std.time.ns_per_ms });
    } else |e| {
        mlx.dropLatchedErrorUnless(had_error);
        log.warn("[dsv41] kernel warm failed ({s}); requests compile their own\n", .{@errorName(e)});
    }
    // The warm's transients leave the pool, so the load's memory reading is the model's.
    _ = mlx.mlx_clear_cache();
    return shell;
}

/// One request at a time: the model holds its state, which `cache.step` mirrors. Logits are the last row.
pub fn forwardDsv41With(self: *Transformer, ctx: *ForwardCtx, ids: mlx.mlx_array, m: *dsv41_model.Model) !mlx.mlx_array {
    if (ctx.cache.step == 0) dsv41_model.resetState(m);
    const st = try dsv41_model.ensureState(m);
    if (st.n != ctx.cache.step) return error.Dsv41CachePositionMismatch;
    m.s = self.s;
    const rows = mlx.getShape(ids);
    if (rows.len != 2 or rows[0] != 1) return error.InvalidDsv41Input;
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_astype(&flat, ids, .uint32, self.s));
    try mlx.check(mlx.mlx_array_eval(flat));
    const host = mlx.mlx_array_data_uint32(flat) orelse return error.InvalidDsv41Input;
    const n: usize = @intCast(rows[1]);
    if (self.expert_stream) |engine| {
        try dsv41_borrow.beforeForward(engine, &self.config, &m.cfg, n, st.n + n);
        engine.beginForward();
    }
    // The model splices the image rows itself: the span's embedding never leaves it.
    const vision: ?dsv41_model.VisionRows = if (ctx.vision_embeddings) |ve| .{ .rows = ve, .offset = ctx.vision_splice_offset } else null;
    const ahead = try m.gpa.alloc(u32, ctx.ahead_ids.len);
    defer m.gpa.free(ahead);
    for (ctx.ahead_ids, ahead) |t, *o| o.* = @intCast(t);
    const last = try dsv41_model.extendAhead(m, host[0..n], .last, dsv41_model.SPAN, vision, ahead);
    defer _ = mlx.mlx_array_free(last);
    if (self.expert_stream) |engine| engine.finishForward(n);
    ctx.cache.step = st.n;
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, last, &[_]c_int{ 1, 1, @intCast(m.cfg.vocab) }, 3, self.s));
    return out;
}

/// DeepSeek-V4.1's DSpark round on the request's state (`ctx.cache.step` mirrors it): the drafter's block from `t1`.
pub fn dsv41DraftImpl(self: *Transformer, t1: u32) !dsv41_model.Draft {
    const m = self.dsv41 orelse return error.Dsv41NoDspark;
    m.s = self.s;
    return dsv41_model.draftHeads(m, t1, .conf);
}

/// Advances the request by `ids` (`[t1, drafts...]`) as a chunk `dsv41Finish` trims; the logits of every row,
/// f32 `[1, L, vocab]`.
pub fn dsv41VerifyImpl(self: *Transformer, ctx: *ForwardCtx, ids: []const u32) !mlx.mlx_array {
    const m = self.dsv41 orelse return error.Dsv41NoDspark;
    m.s = self.s;
    const st = try dsv41_model.ensureState(m);
    if (st.n != ctx.cache.step) return error.Dsv41CachePositionMismatch;
    if (self.expert_stream) |engine| {
        try dsv41_borrow.beforeForward(engine, &self.config, &m.cfg, ids.len, st.n + ids.len);
        engine.beginForward();
    }
    const all = try dsv41_model.verify(m, ids);
    defer _ = mlx.mlx_array_free(all);
    if (self.expert_stream) |engine| engine.finishForward(ids.len);
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_reshape(&out, all, &[_]c_int{ 1, @intCast(ids.len), @intCast(m.cfg.vocab) }, 3, self.s));
    return out;
}

/// Keeps the first `accepted + 1` rows of the open verify chunk.
pub fn dsv41FinishImpl(self: *Transformer, ctx: *ForwardCtx, accepted: usize) !void {
    const m = self.dsv41 orelse return error.Dsv41NoDspark;
    m.s = self.s;
    try dsv41_model.finishVerify(m, accepted);
    ctx.cache.step = (try dsv41_model.ensureState(m)).n;
}
