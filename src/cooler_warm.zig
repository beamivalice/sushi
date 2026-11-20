//! The served warm of a load on Sushi Cooler: requests through the Generator as a served one runs them, at every prompt
//! width a kernel takes as a template value, past them and at the widest chunk, greedy and sampled, with the load's
//! drafter and without, across the indexer's budget, as joined rows and with an image. A request after the seal then
//! makes no pipeline it could have made here. Names no arch: what a request reaches is the arch's, the sweep is the
//! engine's.
const std = @import("std");
const generate = @import("generate.zig");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const sampler = @import("sampler.zig");
const Tokenizer = @import("tokenizer.zig").Tokenizer;
const trunk_mod = @import("trunk.zig");
const Trunk = trunk_mod.Trunk;
const bills = @import("bills.zig");
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

/// Each penalty a request may set takes its own kernels on the tokens said so far: a repeat penalty, presence and
/// frequency, greedy and sampled.
const penalized = [_]sampler.SamplingParams{
    .{ .temperature = 0, .repeat_penalty = 1.3 },
    .{ .temperature = 0, .presence_penalty = 1.0, .frequency_penalty = 0.7 },
    .{ .temperature = 0.7, .top_p = 0.9, .top_k = 40, .seed = 1, .repeat_penalty = 1.3, .presence_penalty = 1.0, .frequency_penalty = 0.7 },
};

/// `drafter`: the load's drafter, if it drafts; a DSpark one drafts in its own rounds, as the scheduler arms it.
/// Served warms run, in a test process.
pub var runs_for_test: u32 = 0;
/// Test seam: the bill the warm's widest request is held to, in place of its own.
pub var bill_for_test: ?u64 = null;

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
    const top = bills.widestServedRung(c);
    const kv = bills.configuredKvQuantFor(c);
    const headroom = bills.prefillHeadroomNow(c, 0);
    const chunk = bills.warmChunkAdmitted(c, if (kv.scheme == .off) 16 else kv.bits, top, headroom);
    if (chunk == 0) {
        const bits: u64 = if (kv.scheme == .off) 16 else kv.bits;
        const floor = bills.PREFILL_CHUNK_LADDER[bills.PREFILL_CHUNK_LADDER.len - 1];
        const ceil = bills.ceilingNow();
        var under_buf: [160]u8 = undefined;
        const under = if (ceil.declared) |d|
            std.fmt.bufPrint(&under_buf, "iogpu.wired_limit_mb={d} less the {d} MB {s} keeps unplanned", .{ d >> 20, ceil.unplanned >> 20, ceil.unplannedName() }) catch "the declared limit"
        else
            "the GPU working set";
        log.warn("[warm] no prefill chunk fits beside the load: the narrowest ({d} rows) needs {d} MB, the headroom is {d} MB (under {s}); a lower --ssd-budget-gb leaves room\n", .{
            floor, bills.warmChunkPeakBytes(c, bits, floor) >> 20, headroom >> 20, under,
        });
        return error.WarmChunkDoesNotFit;
    }
    log.info("[warm] the widest chunk is {d} rows of {d}: headroom {d} MB\n", .{ chunk, top, headroom >> 20 });
    const longest: usize = @intCast(bills.warmLongestPrompt(c, chunk));
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
        // The widest chunk's request, measured against its bill: a pack whose forwards ask more than the bill says, or
        // that the pool refuses, is refused here by name, not deep in a kernel on a served request. What the pool keeps
        // free up to its cap is not a bill's. The bill lends no decoder pool: the warm's chunk fits the headroom
        // (`warmChunkAdmitted` lends none either), so it releases none and takes its whole cap from the pool.
        const bits: u64 = if (kv.scheme == .off) 16 else kv.bits;
        const billed = bills.warmRequestBill(c, bits, chunk, spec.drafter != null);
        const bill = if (@import("builtin").is_test) bill_for_test orelse billed else billed;
        mlx.engine.close();
        const in_use0 = mlx.engine.poolTotal() - mlx.engine.poolFree();
        const held0 = mlx.engine.poolTotal();
        const refusals0 = mlx.engine.poolRefusals();
        mlx.engine.resetPoolPeak(.process);
        const ran = request(io, a, xfm, ids[0..longest], spec, .{ .temperature = 0 }, bills.WARM_REPLY_TOKENS, @import("server_state.zig").ssm_checkpoint_stride);
        mlx.engine.close();
        const asked = mlx.engine.poolPeak(.process) -| in_use0;
        const held = mlx.engine.poolHeldPeak(.process) -| held0;
        const refused = mlx.engine.poolRefusals() != refusals0;
        log.debug("[warm] a {d}-token request at {d}-row chunks: {d} MB in use at its peak, {d} MB held, bill {d} MB\n", .{ longest, chunk, asked >> 20, held >> 20, bill >> 20 });
        if (refused or asked > bill) {
            log.err("[warm] a {d}-token request at {d}-row chunks {s}: {d} MB in use at its peak ({d} MB held) against its {d} MB bill; the bill misses a term of this pack\n", .{ longest, chunk, if (refused) "was refused by the pool" else "asked more than its bill", asked >> 20, held >> 20, bill >> 20 });
            return error.WarmOverBill;
        }
        try ran;
    }
    try narrowAfter(io, a, xfm, ids[0..longest], chunk);
    try archWarmPackedGather(xfm);
    for (sampled ++ penalized) |sp| {
        try request(io, a, xfm, ids[0..9], spec, sp, 8, 0);
        try request(io, a, xfm, ids[0..9], .{}, sp, 4, 0);
    }
    try request(io, a, xfm, ids[0..9], .{}, .{ .temperature = 0 }, 4, 0);
    var added = @import("io_util.zig").Stopwatch.init(io);
    // A context that crosses the indexer's budget: attention stays dense to there, and over 1024 keys and more it takes
    // two passes, which no other warm request reaches. Prompt lookup is off for it: a model that continues the repeated
    // span drafts by lookup alone, and the head's own chains past the budget then never run.
    if (c.indexer_budget > narrow_max) {
        const lookup_was = generate.mtp_lookup_disabled;
        generate.mtp_lookup_disabled = true;
        defer generate.mtp_lookup_disabled = lookup_was;
        try request(io, a, xfm, ids[0 .. c.indexer_budget - 8], spec, .{ .temperature = 0 }, 48, 0);
    }
    if (spec.drafter) |d| for ([_]sampler.SamplingParams{ .{ .temperature = 0 }, sampled[3] }) |sp| {
        for (2..4) |rows| {
            const prompts = [_][]const u32{ ids[0..9], ids[0..40], ids[0..23] };
            try joinedRounds(io, a, xfm, d, prompts[0..rows], sp, 12);
        }
    };
    try joinedRows(io, a, xfm, ids);
    if (c.image_token_id != 0 and c.image_token_id < c.vocab_size and c.mrope_section[0] != 0) try imageRequest(io, a, xfm, spec.drafter, .{ .temperature = 0 }, 40, 16, 16);
    const added_ms = added.read() / std.time.ns_per_ms;
    if (c.image_token_id != 0) try spliceAhead(xfm);
    try mlx.engine.makeCopiesAhead();
    try xfm.resetCache();
    const drafts = if (dspark) "DSpark" else if (spec.drafter != null) "the MTP head" else "no drafter";
    log.info("[warm] served requests to {d} tokens with {s}: {d} pipelines in {d} ms ({d} ms across the budget, joined and with an image)\n", .{ longest, drafts, mlx.engine.pipelinesMade().count - made_before, lap.read() / std.time.ns_per_ms, added_ms });
}

/// An image request's prefill splices the tower's rows in at its placeholders, a path no text request takes: one small
/// splice before the seal makes its kernels, which depend on the dtypes and not on the rows.
pub fn spliceAhead(xfm: *Trunk) !void {
    const c = &xfm.config;
    const s = xfm.s;
    const scope = mlx.engine.enter(s);
    defer scope.end();
    var tids = [_]i32{ 1, 2, 0, 0, 0, 0, 3, 4 };
    for (tids[2..6]) |*t| t.* = @intCast(c.image_token_id);
    const tid = try mlx.newData(&tids, &[_]c_int{ 1, tids.len }, .int32);
    defer _ = mlx.mlx_array_free(tid);
    const hidden: c_int = @intCast(c.hidden_size);
    var rows = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(rows);
    try mlx.check(mlx.mlx_zeros(&rows, &[_]c_int{ 1, 4, hidden }, 3, .bfloat16, s));
    // Made last: the splice takes it on every path, so no fallible call may come between.
    var h = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_zeros(&h, &[_]c_int{ 1, tids.len, hidden }, 3, .bfloat16, s));
    const spliced = try Trunk.spliceVisionRows(h, tid, rows, c.image_token_id, c.video_token_id, 0, s);
    defer _ = mlx.mlx_array_free(spliced);
    try mlx.check(mlx.mlx_array_eval(spliced));
}

/// An image request of `len` tokens, `rows` of them the tower's: its prefill and every verify row after it splice the
/// rows in and rotate by M-RoPE positions, paths no text request takes. The kernels follow the dtypes and the row
/// counts, not the positions, so a table of text positions serves.
pub fn imageRequest(io: std.Io, a: std.mem.Allocator, xfm: *Trunk, drafter: ?generate.MtpHeadRef, sampling: sampler.SamplingParams, len: usize, rows: usize, n: u32) !void {
    const c = &xfm.config;
    const ids = try a.alloc(u32, len);
    defer a.free(ids);
    for (ids, 0..) |*t, i| t.* = if (i >= 1 and i <= rows) @intCast(c.image_token_id) else @intCast((1000 + 7 * (i % 61)) % c.vocab_size);
    const pos = try a.alloc(i32, 3 * len);
    defer a.free(pos);
    for (pos, 0..) |*p, i| p.* = @intCast(i % len);
    var ve = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(ve);
    {
        const scope = mlx.engine.enter(xfm.s);
        defer scope.end();
        try mlx.check(mlx.mlx_zeros(&ve, &[_]c_int{ 1, @intCast(rows), @intCast(c.hidden_size) }, 3, .bfloat16, xfm.s));
        try mlx.check(mlx.mlx_array_eval(ve));
    }
    try xfm.resetCache();
    var ctx = xfm.defaultCtx();
    ctx.vision_embeddings = ve;
    arch_state.setMrope(&ctx.arch_ctx, pos, len, 0);
    var tok = Tokenizer.initEmpty(a);
    defer tok.deinit();
    var gen = try generate.Generator.initWithOptions(io, a, xfm, &tok, ids, n, sampling, &.{}, .{
        .ctx = ctx,
        .mtp = drafter,
        .mtp_enabled = drafter != null,
        .model_has_mtp = drafter != null,
        .skip_lazy_preforward = drafter == null,
    });
    defer gen.deinit(a);
    try drain(a, &gen);
}

/// Concurrent requests whose drafter rounds join, as the scheduler's planner tick runs them
/// (`mtp_tick.runBatchedMtpHeadTick`): one draft chain over every row's head, one grouped verify.
pub fn joinedRounds(io: std.Io, a: std.mem.Allocator, xfm: *Trunk, drafter: generate.MtpHeadRef, prompts: []const []const u32, sampling: sampler.SamplingParams, n: u32) !void {
    const max = 8;
    if (prompts.len > max) return error.WarmJoinedRowsTooMany;
    var tok = Tokenizer.initEmpty(a);
    defer tok.deinit();
    var slots: [max]*Trunk.SpecWarmSlot = undefined;
    var gens: [max]generate.Generator = undefined;
    var ptrs: [max]*generate.Generator = undefined;
    var made: usize = 0;
    var started: usize = 0;
    defer for (slots[0..made]) |sl| {
        arch_state.releaseGroup(xfm, &sl.ctx) catch {};
        sl.deinit(xfm.allocator);
    };
    defer for (gens[0..started]) |*g| g.deinit(a);
    for (prompts, 0..) |p, i| {
        slots[i] = try Trunk.SpecWarmSlot.init(xfm, xfm.cache.config, p.len + n);
        made += 1;
        gens[i] = try generate.Generator.initWithOptions(io, a, xfm, &tok, p, n, sampling, &.{}, .{ .ctx = slots[i].ctx, .mtp = drafter, .mtp_enabled = true, .model_has_mtp = true });
        ptrs[i] = &gens[i];
        started += 1;
    }
    while (true) {
        var live: usize = 0;
        for (ptrs[0..started]) |g| live += @intFromBool(!g.done);
        if (live == 0) return;
        try joinedRound(a, ptrs[0..started]);
    }
}

/// Concurrent requests decoded as rows of one forward at every row count the load joins, with and without the hidden
/// rows a drafter's plain tick keeps, each tick's rows sampled greedy and sampled (`scheduler.runBatchedDecodeTick`).
/// An arch whose rows need a request's own state from the scheduler warms them itself.
pub fn joinedRows(io: std.Io, a: std.mem.Allocator, xfm: *Trunk, ids: []const u32) !void {
    const arch_sched = @import("arch/sched.zig");
    const most = @min(arch_sched.coolerRows(&xfm.config).rows, trunk_mod.MAX_BATCH_ROWS);
    if (most < 2 or !arch_sched.hasRowsForward(xfm)) return;
    var tok = Tokenizer.initEmpty(a);
    defer tok.deinit();
    var slots: [trunk_mod.MAX_BATCH_ROWS]*Trunk.SpecWarmSlot = undefined;
    var gens: [trunk_mod.MAX_BATCH_ROWS]generate.Generator = undefined;
    var made: usize = 0;
    var started: usize = 0;
    defer for (slots[0..made]) |sl| {
        arch_state.releaseGroup(xfm, &sl.ctx) catch {};
        sl.deinit(xfm.allocator);
    };
    defer for (gens[0..started]) |*g| g.deinit(a);
    for (0..most) |i| {
        slots[i] = try Trunk.SpecWarmSlot.init(xfm, xfm.cache.config, 9 + i + 64);
        made += 1;
        gens[i] = try generate.Generator.initWithOptions(io, a, xfm, &tok, ids[0 .. 9 + i], 64, .{ .temperature = 0 }, &.{}, .{ .ctx = slots[i].ctx, .skip_lazy_preforward = true });
        started += 1;
    }
    var toks: [trunk_mod.MAX_BATCH_ROWS]u32 = undefined;
    var ctxs: [trunk_mod.MAX_BATCH_ROWS]*trunk_mod.ForwardCtx = undefined;
    var offsets: [trunk_mod.MAX_BATCH_ROWS]u32 = undefined;
    var drawn: [trunk_mod.MAX_BATCH_ROWS]i32 = undefined;
    var params: [trunk_mod.MAX_BATCH_ROWS]sampler.SamplingParams = undefined;
    for (2..most + 1) |n| for ([_]bool{ false, true }) |capture| {
        for (gens[0..n], 0..) |*g, i| {
            toks[i] = g.next_token_id;
            ctxs[i] = &g.ctx;
            offsets[i] = @intCast(g.ctx.moe_seq_offset.*);
        }
        const arm = arch_sched.batchArm(xfm, ctxs[0..n]) orelse return error.WarmRowsNotReady;
        var hidden: ?[]mlx.mlx_array = null;
        defer if (hidden) |rows| {
            for (rows) |r| _ = mlx.mlx_array_free(r);
            xfm.allocator.free(rows);
        };
        const out = try arch_sched.rowsForward(xfm, toks[0..n], ctxs[0..n], offsets[0..n], arm, if (capture) &hidden else null);
        defer {
            for (out) |o| _ = mlx.mlx_array_free(o);
            xfm.allocator.free(out);
        }
        if (arm.advances_offset) for (ctxs[0..n]) |ctx| {
            ctx.moe_seq_offset.* += 1;
        };
        // Seeded rows draw one by one; unseeded rows under one sampler draw as one (`sampleRowsLazy`).
        for ([_]sampler.SamplingParams{ .{ .temperature = 0 }, sampled[3], .{ .temperature = 0.7, .top_p = 0.9, .top_k = 40 } }) |sp| {
            for (gens[0..n], 0..) |*g, i| {
                params[i] = sp;
                params[i].suppress_mask = g.sampling.suppress_mask;
                params[i].draw_root = i + 1;
            }
            try sampler.sampleRows(drawn[0..n], out, params[0..n], xfm.s);
        }
    };
}

fn joinedRound(a: std.mem.Allocator, gens: []const *generate.Generator) !void {
    const mtp_round = @import("mtp_round.zig");
    var opens: [8]mtp_round.MtpRoundOpen = undefined;
    var chains: [8]mtp_round.MtpPreDraft = undefined;
    var open_gens: [8]*generate.Generator = undefined;
    var states: [8]mtp_round.MtpRoundState = undefined;
    var n: usize = 0;
    // A chain is this round's until `mtpRoundContinue` takes it into its state.
    var handed: usize = 0;
    var stated: usize = 0;
    defer for (chains[handed..n]) |*ch| ch.deinit(a);
    defer for (states[0..stated]) |*st| st.deinit(a);
    defer for (gens) |g| {
        g.mtp_batch_head = false;
    };
    for (gens) |g| {
        if (g.done) continue;
        g.mtp_group_cap = 0;
        g.mtp_batch_head = true;
        switch (try g.mtpRoundBegin(a)) {
            .done => |r| if (r) |got| a.free(got.tokens),
            .verify => |begun| {
                var st = begun;
                defer st.deinit(a);
                try g.mtpRoundVerify(&st);
                if (try g.mtpRoundFinish(a, &st)) |got| a.free(got.tokens);
            },
            .open => |o| {
                opens[n] = o;
                chains[n] = o.chain;
                open_gens[n] = g;
                n += 1;
            },
        }
    }
    if (n == 0) return;
    var depth: u32 = 1;
    for (chains[0..n]) |ch| depth = @max(depth, ch.m);
    try generate.Generator.mtpChainBuildBatched(open_gens[0..n], chains[0..n], 0, depth);
    try generate.Generator.mtpChainDispatchBatched(chains[0..n]);
    var ptrs: [8]*mtp_round.MtpRoundState = undefined;
    for (0..n) |i| {
        opens[i].chain = chains[i];
        handed = i + 1;
        states[i] = try open_gens[i].mtpRoundContinue(a, opens[i]);
        ptrs[i] = &states[i];
        stated = i + 1;
    }
    try generate.Generator.mtpGroupVerify(open_gens[0..n], ptrs[0..n]);
    try generate.Generator.mtpGroupAccept(open_gens[0..n], ptrs[0..n]);
    for (open_gens[0..n], ptrs[0..n]) |g, st| if (try g.mtpRoundFinish(a, st)) |got| a.free(got.tokens);
}

fn drain(a: std.mem.Allocator, gen: *generate.Generator) !void {
    if (gen.dspark) {
        while (try gen.nextDspark(a)) |r| a.free(r.tokens);
    } else if (gen.mtp != null) {
        while (try gen.nextMtp(a)) |r| a.free(r.tokens);
    } else {
        while (try gen.next(a)) |_| {}
    }
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

/// What a prompt restored from the prefix cache past a sparse indexer's budget reads that no cold chunk does (the
/// cached keys in their stored form), made over the keys `narrowAfter` left in the cache, by the arch that has it.
fn archWarmPackedGather(xfm: *Trunk) !void {
    const registry = @import("arch/registry.zig");
    switch (registry.handleOf(xfm)) {
        inline else => |_, t| if (comptime @hasDecl(registry.Mod(t), "warmPackedGather")) try registry.Mod(t).warmPackedGather(xfm),
    }
}

const Spec = struct {
    drafter: ?generate.MtpHeadRef = null,
    dspark: bool = false,
    /// The prefill chunk the warm's widest fits (`warmChunkAdmitted`).
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
    });
    defer gen.deinit(a);
    try drain(a, &gen);
}
