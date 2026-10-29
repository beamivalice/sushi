const std = @import("std");
const mlx = @import("../mlx.zig");
const transformer_mod = @import("../transformer.zig");
const tokenizer_mod = @import("../tokenizer.zig");
const model_mod = @import("../model.zig");
const mtp_lookup = @import("../mtp_lookup.zig");
const mtp_mod = @import("../mtp.zig");
const mimo_mtp = @import("mimo_mtp.zig");
const think_penalty = @import("../think_penalty.zig");
const logit_bias = @import("../logit_bias.zig");
const Transformer = transformer_mod.Transformer;
const Tokenizer = tokenizer_mod.Tokenizer;
const ThinkPenalty = think_penalty.ThinkPenalty;
const testing = std.testing;
const sampler = @import("../sampler.zig");
const SamplingParams = sampler.SamplingParams;
const generate = @import("../generate.zig");
const mtp_policy = @import("../mtp_policy.zig");
const Generator = generate.Generator;
const MtpHeadRef = generate.MtpHeadRef;

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

test "mtpEvPlanFor: MiMo's surface drafts one on prose and three on code, where generic costs over-draft prose" {
    // Conditional per-index acceptance measured with the three heads (forced depth 3).
    const prose = [_]f32{ 0.59, 0.53, 0.42, 0, 0, 0, 0, 0 };
    const code = [_]f32{ 1.0, 0.91, 0.89, 0, 0, 0, 0, 0 };
    const mimo = Generator.mtpEvCostsForProfile(.mimo_exl3);
    try testing.expectEqual(@as(u32, 1), Generator.mtpEvPlanFor(&prose, 3, mimo, 3).m_lo);
    try testing.expectEqual(@as(u32, 3), Generator.mtpEvPlanFor(&code, 3, mimo, 3).m_lo);
    try testing.expect(Generator.mtpEvPlanFor(&prose, 3, mtp_policy.MTP_EV_DEFAULT_COSTS, 3).m_lo > 1);
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
    try mlx.requireGpu();
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
    try mlx.requireGpu();
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
    var tok = Tokenizer.initEmptyForTests(a);
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
    var free = Generator.mtpLookupCostsFor(mtp_policy.MtpCostSource.init(mtp_policy.MTP_EV_DEFAULT_COSTS, 1000, null), 3);
    for (&free.lookup) |*c| c.* = 0.01;
    try testing.expectEqual(Generator.mtpVerifyDraftsMax(true), mtp_lookup.gate(got, 1000, 8, 0, true, free));
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
        pub fn lmHeadLogits(_: @This(), _: mlx.mlx_array) !mlx.mlx_array {
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
        pub fn lmHeadLogits(_: @This(), _: mlx.mlx_array) !mlx.mlx_array {
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
