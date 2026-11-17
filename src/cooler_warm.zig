//! The served warm of a load on Sushi Cooler: requests through the Generator as a served one runs them, at every prompt
//! width a kernel takes as a template value, past them and at the widest chunk, greedy and sampled, with the load's drafter and without. A
//! request after the seal then makes no pipeline it could have made here. Names no arch: what a request reaches is the
//! arch's, the sweep is the engine's.
const std = @import("std");
const generate = @import("generate.zig");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const sampler = @import("sampler.zig");
const Tokenizer = @import("tokenizer.zig").Tokenizer;
const Trunk = @import("trunk.zig").Trunk;
const prefill_plan = @import("prefill_plan.zig");
const bills = @import("bills.zig");
const arch_flags = @import("arch/flags.zig");
const arch_state = @import("arch/arch_state.zig");

/// Every chunk width to twice a verify block's rows: kernels take a narrow width as a template value, and a prompt
/// runs one row fewer than its length before its last token.
const narrow_max = @import("sampler_draw.zig").max_rows * 2 + 1;
/// Widths past every templated one: one off the GEMM tiles, one on them.
const wide = [_]usize{ 100, 128 };

/// Each filter a sampled request may set takes its own kernels: none, top-k, top-p, both.
const sampled = [_]sampler.SamplingParams{
    .{ .temperature = 0.7, .seed = 1 },
    .{ .temperature = 0.7, .top_k = 40, .seed = 1 },
    .{ .temperature = 0.7, .top_p = 0.9, .seed = 1 },
    .{ .temperature = 0.7, .top_p = 0.9, .top_k = 40, .seed = 1 },
};

/// `drafter`: the load's drafter, if it drafts; a DSpark one drafts in its own rounds, as the scheduler arms it.
/// Served warms run, in a test process.
pub var runs_for_test: u32 = 0;

pub fn warmServed(io: std.Io, a: std.mem.Allocator, xfm: *Trunk, drafter: ?generate.MtpHeadRef) !void {
    if (@import("builtin").is_test) runs_for_test += 1;
    arch_state.servedWarmBegin(&xfm.arch_state);
    defer arch_state.servedWarmEnd(&xfm.arch_state);
    var lap = @import("io_util.zig").Stopwatch.init(io);
    const made_before = mlx.engine.pipelinesMade().count;
    // The widest forward a request makes, a short prompt's whole chunk, and past a sparse-attention indexer's budget
    // (it selects blocks only there), off the tiles: paths a module takes from a row count on run only at such widths.
    const c = &xfm.config;
    // A request's chunk is sized by the live headroom: the warm's widest is what fits beside the load now.
    const top = prefill_plan.requestPrefillChunk(c.prefillScoreHeadDim(), c.num_attention_heads, 1, c.has_sliding_window, c.isMoe(), c.longCtxGated(), 0);
    const kv = bills.configuredKvQuantFor(c);
    const chunk = bills.widestChunkAdmitted(c, if (kv.scheme == .off) 16 else kv.bits, top, bills.prefillHeadroomNow(c, 0));
    if (chunk == 0) return error.WarmChunkDoesNotFit;
    if (chunk < top) log.info("[warm] the widest chunk is {d} rows, the most the headroom holds beside the load (of {d})\n", .{ chunk, top });
    const past_budget: usize = if (c.indexer_budget > 0) c.indexer_budget + 300 else 0;
    const longest = @max(past_budget, chunk + 1);
    const ids = try a.alloc(u32, @max(wide[wide.len - 1], longest));
    defer a.free(ids);
    // A repeated span: a model continues it, so its drafts are accepted and the round shapes acceptance unlocks (prompt
    // lookup, a two-chunk round) run in the long request.
    for (ids, 0..) |*t, i| t.* = @intCast((1000 + 7 * (i % 61)) % c.vocab_size);
    const dspark = if (drafter) |d| d.kind() == .dspark else false;
    const spec: Spec = .{ .drafter = if (dspark) null else drafter, .dspark = dspark, .chunk = chunk };
    for (1..narrow_max + 1) |n| try request(io, a, xfm, ids[0..n], spec, .{ .temperature = 0 }, 2, 0);
    for (wide) |n| try request(io, a, xfm, ids[0..n], spec, .{ .temperature = 0 }, 2, 0);
    // Split where a served prompt is, at its prefix-cache restore point: its tail and the drafter's history take the
    // pieces' shapes.
    {
        // The long request's rounds take every plan in turn: the planner opens a shape from the costs it measures, which
        // a warm cannot count on reaching.
        const policy = @import("mtp_policy.zig");
        var plans: [64]policy.MtpRoundPlan = undefined;
        if (spec.drafter != null) policy.warm_plans = policy.everyPlan(&plans, @import("generate.zig").Generator.mtpVerifyDraftsMax(c));
        defer policy.warm_plans = &.{};
        try request(io, a, xfm, ids[0..longest], spec, .{ .temperature = 0 }, 96, @import("server_state.zig").ssm_checkpoint_stride);
    }
    try narrowAfter(io, a, xfm, ids[0..longest], chunk);
    for (sampled) |sp| {
        try request(io, a, xfm, ids[0..9], spec, sp, 8, 0);
        try request(io, a, xfm, ids[0..9], .{}, sp, 4, 0);
    }
    try request(io, a, xfm, ids[0..9], .{}, .{ .temperature = 0 }, 4, 0);
    try mlx.engine.makeCopiesAhead();
    try xfm.resetCache();
    const drafts = if (dspark) "DSpark" else if (spec.drafter != null) "the MTP head" else "no drafter";
    log.info("[warm] served requests to {d} tokens with {s}: {d} pipelines in {d} ms\n", .{ longest, drafts, mlx.engine.pipelinesMade().count - made_before, lap.read() / std.time.ns_per_ms });
}

/// Every narrow chunk width once more, appended after `ids`: a prompt restored from the prefix cache prefills its tail
/// there, and past a sparse indexer's budget the attention's block selection takes the width as a template value too.
pub fn narrowAfter(io: std.Io, a: std.mem.Allocator, xfm: *Trunk, ids: []const u32, chunk: usize) !void {
    var tok = Tokenizer.initEmpty(a);
    defer tok.deinit();
    try xfm.resetCache();
    var gen = try generate.Generator.initWithOptions(io, a, xfm, &tok, ids, 1, .{ .temperature = 0 }, &.{}, .{ .skip_lazy_preforward = true, .pinned_prefill_chunk = chunk });
    defer gen.deinit(a);
    _ = try gen.next(a);
    var row: [narrow_max]i32 = undefined;
    for (&row, 0..) |*t, i| t.* = @intCast(ids[i % ids.len]);
    for (1..narrow_max + 1) |n| {
        const shape = [2]c_int{ 1, @intCast(n) };
        const in = try generate.idsArray(xfm, &row, &shape);
        defer _ = mlx.mlx_array_free(in);
        const logits = try xfm.forwardWith(&gen.ctx, in);
        defer _ = mlx.mlx_array_free(logits);
        try mlx.check(mlx.mlx_array_eval(logits));
    }
}

const Spec = struct {
    drafter: ?generate.MtpHeadRef = null,
    dspark: bool = false,
    /// The prefill chunk the warm's widest fits (`widestChunkAdmitted`).
    chunk: usize = 0,
};

/// One request of `n` tokens, its rounds taken as the scheduler takes them; `stride` its prompt's checkpoint stride.
fn request(io: std.Io, a: std.mem.Allocator, xfm: *Trunk, ids: []const u32, spec: Spec, sampling: sampler.SamplingParams, n: u32, stride: u32) !void {
    var tok = Tokenizer.initEmpty(a);
    defer tok.deinit();
    try xfm.resetCache();
    var gen = try generate.Generator.initWithOptions(io, a, xfm, &tok, ids, n, sampling, &.{}, .{
        .mtp = spec.drafter,
        .mtp_enabled = spec.drafter != null,
        .model_has_mtp = spec.drafter != null,
        .dspark_enabled = spec.dspark,
        .skip_lazy_preforward = spec.drafter == null and !spec.dspark,
        .ssm_checkpoint_stride = stride,
        .pinned_prefill_chunk = spec.chunk,
        // The load's acceptance rule and tail, as the scheduler hands every request: each is its own graph.
        .mtp_acceptance = arch_flags.mtpAcceptanceOf(&xfm.config).value,
        .mtp_greedy_tail = arch_flags.greedyTailOf(&xfm.config).value,
    });
    defer gen.deinit(a);
    if (gen.dspark) {
        while (try gen.nextDspark(a)) |r| a.free(r.tokens);
    } else if (gen.mtp != null) {
        while (try gen.nextMtp(a)) |r| a.free(r.tokens);
    } else {
        while (try gen.next(a)) |_| {}
    }
}
