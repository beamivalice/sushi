const std = @import("std");
const mlx = @import("../mlx.zig");
const transformer_mod = @import("../transformer.zig");
const tokenizer_mod = @import("../tokenizer.zig");
const model_mod = @import("../model.zig");
const token_mask = @import("../token_mask.zig");
const rp_mod = @import("../reasoning_protocol.zig");
const depth_bounds = @import("../mtp_depth_bounds.zig");
const Transformer = transformer_mod.Transformer;
const Tokenizer = tokenizer_mod.Tokenizer;
const SSMCheckpoint = transformer_mod.SSMCheckpoint;
const glm5_forward = @import("../glm5/glm5_forward.zig");
const dsv41_prefix = @import("dsv41_prefix.zig");
const KVCache = transformer_mod.KVCache;
const testing = std.testing;
const prefill_plan = @import("../prefill_plan.zig");
const CACHE_CLEAR_INTERVAL = prefill_plan.CACHE_CLEAR_INTERVAL;
const sampler = @import("../sampler.zig");
const SamplingParams = sampler.SamplingParams;
const mtp_plan = @import("../mtp_plan.zig");
const SpecDisableReason = mtp_plan.SpecDisableReason;
const ThinkBound = mtp_plan.ThinkBound;
const generate = @import("../generate.zig");
const dsv41_dspark = @import("dsv41_dspark.zig");
const DsparkDiag = dsv41_dspark.DsparkDiag;
const Generator = generate.Generator;
const SchemaConstraint = generate.SchemaConstraint;
const dsparkEligible = dsv41_dspark.dsparkEligible;
const dsparkRowsForRoom = dsv41_dspark.dsparkRowsForRoom;
const dsv41_max_drafts = dsv41_dspark.dsv41_max_drafts;
const dsv41_max_verify_rows = dsv41_dspark.dsv41_max_verify_rows;
const parseDsparkDiag = dsv41_dspark.parseDsparkDiag;

test "SUSHI_DSV41_DSPARK_ROWS never forces a verify wider than dsv41Verify accepts" {
    const limit = dsv41_max_verify_rows;
    try testing.expectEqual(DsparkDiag{ .rows = limit }, parseDsparkDiag("17"));
    try testing.expectEqual(DsparkDiag{ .rows = limit }, parseDsparkDiag("255"));
    try testing.expectEqual(DsparkDiag{ .rows = 7 }, parseDsparkDiag("7"));
    try testing.expectEqual(DsparkDiag{ .rows = 1 }, parseDsparkDiag("0"));
    try testing.expectEqual(DsparkDiag.sweep, parseDsparkDiag("sweep"));
    try testing.expectEqual(DsparkDiag.off, parseDsparkDiag("wide"));
}

test "dsv41 generator: DSpark rounds through the Generator keep the budget, the stats and the first token (DSV41_TINY)" {
    try mlx.requireGpu();
    const dsv41_model = @import("dsv41_model.zig");
    const dir = try dsv41_model.fixtureDir();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var fx = try dsv41_model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    var sc = @import("dsv41_scope.zig").Scope.init(a, mlx.gpuStream());
    defer sc.deinit();
    const ids = try dsv41_model.tokens(&sc, &fx);
    defer a.free(ids);
    var config = try model_mod.parseConfig(io, a, dir);
    defer config.deinit(a);
    config.expert_streaming = true;
    config.stream_mtp_head = true;
    config.expert_cache_bytes = 3 * config.num_hidden_layers * try @import("dsv41_stream.zig").expertBytes(&config.dsv41.?);
    config.expert_source_dir = try a.dupe(u8, dir);
    var weights = try model_mod.loadWeightsForConfig(io, a, dir, &config, false);
    defer weights.deinit();
    var xfm = try Transformer.init(io, a, config, &weights);
    defer xfm.deinit();
    try testing.expect(xfm.dsv41.?.dspark != null and xfm.moduleStateSpecRollback());
    var tok = Tokenizer.initEmptyForTests(a);
    defer tok.deinit();
    const greedy = SamplingParams{ .temperature = 0.0 };
    const prompt = ids[0..30];
    const want: u32 = 30;

    var serial: std.ArrayList(u32) = .empty;
    defer serial.deinit(a);
    {
        var gen = try Generator.initWithOptions(io, a, &xfm, &tok, prompt, want, greedy, &.{}, .{ .skip_lazy_preforward = true });
        defer gen.deinit(a);
        while (try gen.next(a)) |t| try serial.append(a, t);
    }
    try testing.expectEqual(@as(usize, want), serial.items.len);
    try xfm.resetCache();

    var spec: std.ArrayList(u32) = .empty;
    defer spec.deinit(a);
    var gen = try Generator.initWithOptions(io, a, &xfm, &tok, prompt, want, greedy, &.{}, .{ .dspark_enabled = true });
    defer gen.deinit(a);
    try testing.expect(gen.dspark);
    var accepted_sum: u64 = 0;
    while (try gen.nextDspark(a)) |r| {
        defer a.free(r.tokens);
        try testing.expectEqual(@as(usize, r.accepted_tokens) + 1, r.tokens.len);
        accepted_sum += r.accepted_tokens;
        try spec.appendSlice(a, r.tokens);
    }
    // The bf16 span rows and the one-row decode round differently, so a near-tie may break either way: byte
    // identity with serial is pinned on the exact stream in dsv41_model.zig and on a real pack by the live check.
    try testing.expectEqual(serial.items.len, spec.items.len);
    try testing.expectEqual(serial.items[0], spec.items[0]);
    try testing.expect(gen.dspark_stats.rounds > 0);
    try testing.expectEqual(accepted_sum, gen.dspark_stats.accepted);
    try testing.expect(gen.dspark_stats.accepted <= gen.dspark_stats.drafted);
    try testing.expectEqualStrings("length", gen.finish_reason);

    // A forced verify width keeps the budget and the first token; one row is the serial decode step itself.
    for ([_]u8{ 1, 2, 3, 4, 6 }) |width| {
        try xfm.resetCache();
        var w = try Generator.initWithOptions(io, a, &xfm, &tok, prompt, want, greedy, &.{}, .{ .dspark_enabled = true });
        defer w.deinit(a);
        w.dspark_diag = .{ .rows = width };
        var got: std.ArrayList(u32) = .empty;
        defer got.deinit(a);
        while (try w.nextDspark(a)) |r| {
            defer a.free(r.tokens);
            try testing.expect(r.tokens.len <= width);
            try got.appendSlice(a, r.tokens);
        }
        try testing.expectEqual(serial.items.len, got.items.len);
        try testing.expectEqual(serial.items[0], got.items[0]);
        try testing.expect(w.dspark_stats.drafted <= w.dspark_stats.rounds * (width - 1));
        if (width == 1) try testing.expectEqualSlices(u32, serial.items, got.items);
    }

    // `--mtp-min-depth` / `--mtp-max-depth` set the width a DSpark request may verify: depth + 1 rows.
    {
        const saved = depth_bounds.active;
        defer depth_bounds.active = saved;
        depth_bounds.active = .{ .min = 2, .max = 2 };
        try xfm.resetCache();
        var pin = try Generator.initWithOptions(io, a, &xfm, &tok, prompt, want, greedy, &.{}, .{ .dspark_enabled = true });
        defer pin.deinit(a);
        try testing.expectEqual(@as(u32, 3), pin.dspark_policy.range.min_rows);
        try testing.expectEqual(@as(u32, 3), pin.dspark_policy.range.max_rows);
        while (try pin.nextDspark(a)) |r| a.free(r.tokens);
        try testing.expect(pin.dspark_stats.rounds > 0);
        // Only the closing round, cut to the tokens the budget still allows, may verify fewer rows.
        try testing.expect(pin.dspark_stats.plain <= 1);
        for (pin.dspark_stats.rows_hist, 0..) |n, rows| if (rows > 3) try testing.expectEqual(@as(u32, 0), n);
        depth_bounds.active = .{ .max = 1 };
        try xfm.resetCache();
        var cap = try Generator.initWithOptions(io, a, &xfm, &tok, prompt, want, greedy, &.{}, .{ .dspark_enabled = true });
        defer cap.deinit(a);
        while (try cap.nextDspark(a)) |r| a.free(r.tokens);
        try testing.expect(cap.dspark_stats.drafted <= cap.dspark_stats.rounds);
    }

    // A repeat penalty shifts the verify rows by the ids before them, as a serial tick does: no id repeats.
    const penalized_params = SamplingParams{ .temperature = 0.0, .repeat_penalty = 10.0 };
    var distinct: [2]std.ArrayList(u32) = .{ .empty, .empty };
    defer for (&distinct) |*r| r.deinit(a);
    {
        try xfm.resetCache();
        var g = try Generator.initWithOptions(io, a, &xfm, &tok, prompt, want, penalized_params, &.{}, .{ .skip_lazy_preforward = true });
        defer g.deinit(a);
        while (try g.next(a)) |t| try distinct[0].append(a, t);
        try xfm.resetCache();
        var d = try Generator.initWithOptions(io, a, &xfm, &tok, prompt, want, penalized_params, &.{}, .{ .dspark_enabled = true });
        defer d.deinit(a);
        try testing.expect(d.dspark);
        while (try d.nextDspark(a)) |r| {
            defer a.free(r.tokens);
            try distinct[1].appendSlice(a, r.tokens);
        }
        try testing.expect(d.dspark_stats.rounds > 0);
    }
    for (distinct) |list| {
        try testing.expectEqual(@as(usize, want), list.items.len);
        for (list.items, 0..) |t, i| try testing.expect(std.mem.indexOfScalar(u32, list.items[0..i], t) == null);
    }
    var logprob_req = try Generator.initWithOptions(io, a, &xfm, &tok, prompt, 4, .{ .temperature = 0.8 }, &.{}, .{ .dspark_enabled = true, .logprobs_n = 2 });
    defer logprob_req.deinit(a);
    try testing.expect(!logprob_req.dspark);
    try testing.expectError(error.SpecDecodeUnsupported, logprob_req.nextDspark(a));

    // A sampled request drafts too: the verify rows take the exact rule for a point-mass draft, and a seeded one
    // verifies at its fixed width, so it repeats.
    const sampled_params = SamplingParams{ .temperature = 0.8, .seed = 7 };
    var runs: [2]std.ArrayList(u32) = .{ .empty, .empty };
    defer for (&runs) |*r| r.deinit(a);
    for (&runs) |*out| {
        try xfm.resetCache();
        var sampled = try Generator.initWithOptions(io, a, &xfm, &tok, prompt, want, sampled_params, &.{}, .{ .dspark_enabled = true });
        defer sampled.deinit(a);
        try testing.expect(sampled.dspark);
        sampled.dspark_diag = .{ .rows = 3 };
        while (try sampled.nextDspark(a)) |r| {
            defer a.free(r.tokens);
            try out.appendSlice(a, r.tokens);
        }
        try testing.expect(sampled.dspark_stats.rounds > 0);
        try testing.expectEqualStrings("length", sampled.finish_reason);
    }
    try testing.expectEqual(@as(usize, want), runs[0].items.len);
    try testing.expectEqualSlices(u32, runs[0].items, runs[1].items);
    {
        try xfm.resetCache();
        var adaptive = try Generator.initWithOptions(io, a, &xfm, &tok, prompt, want, sampled_params, &.{}, .{ .dspark_enabled = true });
        defer adaptive.deinit(a);
        var n: usize = 0;
        while (try adaptive.nextDspark(a)) |r| {
            defer a.free(r.tokens);
            n += r.tokens.len;
        }
        try testing.expectEqual(@as(usize, want), n);
        try testing.expectEqualStrings("length", adaptive.finish_reason);
    }

    // One verified row draws its pending token from the filtered row: seeded, repeatable, not a placeholder.
    var single: [2]std.ArrayList(u32) = .{ .empty, .empty };
    defer for (&single) |*r| r.deinit(a);
    for (&single) |*out| {
        try xfm.resetCache();
        var one = try Generator.initWithOptions(io, a, &xfm, &tok, prompt, want, sampled_params, &.{}, .{ .dspark_enabled = true });
        defer one.deinit(a);
        one.dspark_diag = .{ .rows = 1 };
        while (try one.nextDspark(a)) |r| {
            defer a.free(r.tokens);
            try out.appendSlice(a, r.tokens);
        }
    }
    try testing.expectEqual(@as(usize, want), single[0].items.len);
    try testing.expectEqualSlices(u32, single[0].items, single[1].items);
    try testing.expect(std.mem.count(u32, single[0].items, &.{single[0].items[1]}) < want - 1);
}

test "dsv41 generator: with exact DSpark rounds greedy text is the same at every round-width mix (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    const dsv41_model = @import("dsv41_model.zig");
    const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var fx = try dsv41_model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    var sc = @import("dsv41_scope.zig").Scope.init(a, mlx.gpuStream());
    defer sc.deinit();
    const ids = try dsv41_model.tokens(&sc, &fx);
    defer a.free(ids);
    var config = try model_mod.parseConfig(io, a, dir);
    defer config.deinit(a);
    config.expert_streaming = true;
    config.stream_mtp_head = true;
    config.expert_cache_bytes = 64 << 20;
    config.expert_source_dir = try a.dupe(u8, dir);
    var weights = try model_mod.loadWeightsForConfig(io, a, dir, &config, false);
    defer weights.deinit();
    var xfm = try Transformer.init(io, a, config, &weights);
    defer xfm.deinit();
    try testing.expect(xfm.dsv41.?.dspark != null);
    var tok = Tokenizer.initEmptyForTests(a);
    defer tok.deinit();
    const greedy = SamplingParams{ .temperature = 0.0 };
    const prompt = ids[0..30];
    const want: u32 = 60;

    var first: std.ArrayList(u32) = .empty;
    defer first.deinit(a);

    // One width, a cycle through every width (`dspark_sweep_rows`), and the adaptive policy.
    const modes = [_]?DsparkDiag{ .{ .rows = 1 }, .{ .rows = 3 }, .{ .rows = 6 }, .sweep, null };
    for (modes, 0..) |mode, i| {
        try xfm.resetCache();
        var gen = try Generator.initWithOptions(io, a, &xfm, &tok, prompt, want, greedy, &.{}, .{ .dspark_enabled = true });
        defer gen.deinit(a);
        try testing.expect(gen.dspark);
        if (mode) |d| gen.dspark_diag = d;
        var got: std.ArrayList(u32) = .empty;
        defer got.deinit(a);
        while (try gen.nextDspark(a)) |r| {
            defer a.free(r.tokens);
            try got.appendSlice(a, r.tokens);
        }
        try testing.expectEqual(@as(usize, want), got.items.len);
        if (i == 0) try first.appendSlice(a, got.items) else try testing.expectEqualSlices(u32, first.items, got.items);
    }

    // `SUSHI_DSV41_DSPARK_EXACT=0` still decodes a whole request; its text may differ with the round mix.
    dsv41_model.setDsparkExact(xfm.dsv41.?, false);
    try xfm.resetCache();
    var fast = try Generator.initWithOptions(io, a, &xfm, &tok, prompt, want, greedy, &.{}, .{ .dspark_enabled = true });
    defer fast.deinit(a);
    var n: usize = 0;
    while (try fast.nextDspark(a)) |r| {
        defer a.free(r.tokens);
        for (r.tokens) |t| try testing.expect(t < xfm.dsv41.?.cfg.vocab);
        n += r.tokens.len;
    }
    try testing.expectEqual(@as(usize, want), n);
}

test "dsv41 generator: a seeded sampled request verifies at one width whatever the cost model says (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    const dsv41_model = @import("dsv41_model.zig");
    const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var fx = try dsv41_model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    var sc = @import("dsv41_scope.zig").Scope.init(a, mlx.gpuStream());
    defer sc.deinit();
    const ids = try dsv41_model.tokens(&sc, &fx);
    defer a.free(ids);
    var config = try model_mod.parseConfig(io, a, dir);
    defer config.deinit(a);
    config.expert_streaming = true;
    config.stream_mtp_head = true;
    config.expert_cache_bytes = 64 << 20;
    config.expert_source_dir = try a.dupe(u8, dir);
    var weights = try model_mod.loadWeightsForConfig(io, a, dir, &config, false);
    defer weights.deinit();
    var xfm = try Transformer.init(io, a, config, &weights);
    defer xfm.deinit();
    var tok = Tokenizer.initEmptyForTests(a);
    defer tok.deinit();
    const prompt = ids[0..30];
    const want: u32 = 60;
    dsv41_model.setDsparkExact(xfm.dsv41.?, false);

    const Perturb = enum { none, dear, cheap, owed };
    var base_hist: [dsv41_max_drafts + 2]u32 = undefined;
    var base_tokens: std.ArrayList(u32) = .empty;
    defer base_tokens.deinit(a);
    for ([_]Perturb{ .none, .dear, .cheap, .owed }) |perturb| {
        for ([_]SamplingParams{ .{ .temperature = 0.8, .seed = 7 }, .{ .temperature = 0.8, .seed = 7, .top_k = 20 } }, 0..) |params, which| {
            try xfm.resetCache();
            var gen = try Generator.initWithOptions(io, a, &xfm, &tok, prompt, want, params, &.{}, .{ .dspark_enabled = true });
            defer gen.deinit(a);
            try testing.expect(gen.dspark);
            switch (perturb) {
                .none => {},
                .dear => gen.dspark_policy.cost = .{ .plain_ms = 900, .step_ms = 40, .draft_ms = 40, .prior = 40 },
                .cheap => gen.dspark_policy.cost = .{ .plain_ms = 5, .step_ms = 3, .draft_ms = 1, .prior = 3 },
                .owed => {
                    gen.dspark_policy.owed = 6;
                    gen.dspark_policy.period = 8;
                },
            }
            var got: std.ArrayList(u32) = .empty;
            defer got.deinit(a);
            while (try gen.nextDspark(a)) |r| {
                defer a.free(r.tokens);
                try got.appendSlice(a, r.tokens);
            }
            try testing.expectEqual(@as(usize, want), got.items.len);
            if (which == 1) continue;
            if (perturb == .none) {
                base_hist = gen.dspark_stats.rows_hist;
                try base_tokens.appendSlice(a, got.items);
                continue;
            }
            try testing.expectEqualSlices(u32, &base_hist, &gen.dspark_stats.rows_hist);
            try testing.expectEqualSlices(u32, base_tokens.items, got.items);
        }
    }

    // An unseeded request keeps the adaptive policy: the cost model still moves its width.
    var widths: [2][dsv41_max_drafts + 2]u32 = undefined;
    for ([_]Perturb{ .dear, .cheap }, 0..) |perturb, i| {
        try xfm.resetCache();
        var gen = try Generator.initWithOptions(io, a, &xfm, &tok, prompt, want, .{ .temperature = 0.8 }, &.{}, .{ .dspark_enabled = true });
        defer gen.deinit(a);
        gen.dspark_policy.cost = if (perturb == .dear) .{ .plain_ms = 900, .step_ms = 40, .draft_ms = 40, .prior = 40 } else .{ .plain_ms = 5, .step_ms = 3, .draft_ms = 1, .prior = 3 };
        while (try gen.nextDspark(a)) |r| a.free(r.tokens);
        widths[i] = gen.dspark_stats.rows_hist;
    }
    try testing.expect(!std.mem.eql(u32, &widths[0], &widths[1]));
}

test "dsv41 generator: a DSpark round ends right behind the id it is told to, at any verify width, with the state standing there (DSV41_TINY)" {
    try mlx.requireGpu();
    const dsv41_model = @import("dsv41_model.zig");
    const dir = try dsv41_model.fixtureDir();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var fx = try dsv41_model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    var sc = @import("dsv41_scope.zig").Scope.init(a, mlx.gpuStream());
    defer sc.deinit();
    const ids = try dsv41_model.tokens(&sc, &fx);
    defer a.free(ids);
    var config = try model_mod.parseConfig(io, a, dir);
    defer config.deinit(a);
    config.expert_streaming = true;
    config.stream_mtp_head = true;
    config.expert_cache_bytes = 3 * config.num_hidden_layers * try @import("dsv41_stream.zig").expertBytes(&config.dsv41.?);
    config.expert_source_dir = try a.dupe(u8, dir);
    var weights = try model_mod.loadWeightsForConfig(io, a, dir, &config, false);
    defer weights.deinit();
    var xfm = try Transformer.init(io, a, config, &weights);
    defer xfm.deinit();
    var tok = Tokenizer.initEmptyForTests(a);
    defer tok.deinit();
    const greedy = SamplingParams{ .temperature = 0.0 };
    const prompt = ids[0..30];
    const want: u32 = 30;

    for ([_]u8{ 2, 4, 6 }) |width| {
        var plain: std.ArrayList(u32) = .empty;
        defer plain.deinit(a);
        try xfm.resetCache();
        {
            var gen = try Generator.initWithOptions(io, a, &xfm, &tok, prompt, want, greedy, &.{}, .{ .dspark_enabled = true });
            defer gen.deinit(a);
            gen.dspark_diag = .{ .rows = width };
            while (try gen.nextDspark(a)) |r| {
                defer a.free(r.tokens);
                try plain.appendSlice(a, r.tokens);
            }
        }
        const end_id = plain.items[9];
        try xfm.resetCache();
        var gen = try Generator.initWithOptions(io, a, &xfm, &tok, prompt, want, greedy, &.{}, .{ .dspark_enabled = true });
        defer gen.deinit(a);
        gen.dspark_diag = .{ .rows = width };
        gen.dspark_round_ends_at = end_id;
        var total: usize = 0;
        var ended_rounds: usize = 0;
        while (true) {
            const armed = gen.dspark_round_ends_at != null;
            const r = (try gen.nextDspark(a)) orelse break;
            defer a.free(r.tokens);
            total += r.tokens.len;
            // No round runs past the id, and the state stands on the committed tokens, so a restore point can be taken there.
            if (armed) if (std.mem.indexOfScalar(u32, r.tokens, end_id)) |at| {
                try testing.expectEqual(r.tokens.len - 1, at);
                var cp = try dsv41_prefix.capture(a, xfm.dsv41.?, prompt.len + total);
                cp.deinit(a);
                ended_rounds += 1;
                gen.dspark_round_ends_at = null;
            };
        }
        try testing.expectEqual(@as(usize, 1), ended_rounds);
        try testing.expectEqual(@as(usize, want), total);
    }
}

test "dsv41 generator: the restore point behind a thought's closer, written mid-decode, restores from an SSD file to the live state's continuation bit for bit (DSV41_TINY)" {
    try mlx.requireGpu();
    const dsv41_model = @import("dsv41_model.zig");
    const prefix_cache = @import("../prefix_cache.zig");
    const kv_disk_cache = @import("../kv_disk_cache.zig");
    const dir = try dsv41_model.fixtureDir();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var fx = try dsv41_model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    var sc = @import("dsv41_scope.zig").Scope.init(a, mlx.gpuStream());
    defer sc.deinit();
    const ids = try dsv41_model.tokens(&sc, &fx);
    defer a.free(ids);
    var config = try model_mod.parseConfig(io, a, dir);
    defer config.deinit(a);
    config.expert_streaming = true;
    config.stream_mtp_head = true;
    config.expert_cache_bytes = 3 * config.num_hidden_layers * try @import("dsv41_stream.zig").expertBytes(&config.dsv41.?);
    config.expert_source_dir = try a.dupe(u8, dir);
    var weights = try model_mod.loadWeightsForConfig(io, a, dir, &config, false);
    defer weights.deinit();
    var xfm = try Transformer.init(io, a, config, &weights);
    defer xfm.deinit();
    var tok = Tokenizer.initEmptyForTests(a);
    defer tok.deinit();
    const s = xfm.s;
    var prompt: [600]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = ids[(i * 5 + 3) % ids.len];
    const want: u32 = 40;
    // The suffix is what the next turn adds after the thought.
    var suffix: [50]u32 = undefined;
    for (&suffix, 0..) |*t, i| t.* = (ids[(i * 3 + 1) % ids.len] + 1) % 90;
    const greedy = SamplingParams{ .temperature = 0.0 };

    // The id a round is told to end behind: a token the plain run emits early.
    var plain: std.ArrayList(u32) = .empty;
    defer plain.deinit(a);
    {
        var gen = try Generator.initWithOptions(io, a, &xfm, &tok, &prompt, want, greedy, &.{}, .{ .dspark_enabled = true });
        defer gen.deinit(a);
        gen.dspark_diag = .{ .rows = 4 };
        while (try gen.nextDspark(a)) |r| {
            defer a.free(r.tokens);
            try plain.appendSlice(a, r.tokens);
        }
    }
    const end_id = plain.items[11];

    // The live state's continuation: the request stops where the round ended and the suffix runs on its state.
    var point: usize = 0;
    var next_buf: [600 + 40 + 50]u32 = undefined;
    var want_logits: ?mlx.mlx_array = null;
    defer if (want_logits) |w| {
        _ = mlx.mlx_array_free(w);
    };
    {
        try xfm.resetCache();
        var cache = try KVCache.initWithConfig(a, config.num_hidden_layers, transformer_mod.KVQuantConfig.engine_default);
        defer cache.deinit();
        if (config.swaRingTokens() > 0) cache.setSwaRing(config.sliding_window);
        var off: usize = 0;
        var ctx = xfm.defaultCtx();
        ctx.cache = &cache;
        ctx.moe_seq_offset = &off;
        var gen = try Generator.initWithOptions(io, a, &xfm, &tok, &prompt, want, greedy, &.{}, .{ .ctx = ctx, .dspark_enabled = true });
        defer gen.deinit(a);
        gen.dspark_diag = .{ .rows = 4 };
        gen.dspark_round_ends_at = end_id;
        while (try gen.nextDspark(a)) |r| {
            defer a.free(r.tokens);
            if (r.tokens[r.tokens.len - 1] != end_id) continue;
            point = prompt.len + gen.generated_ids.items.len;
            @memcpy(next_buf[0..prompt.len], &prompt);
            @memcpy(next_buf[prompt.len..point], gen.generated_ids.items);
            @memcpy(next_buf[point .. point + suffix.len], &suffix);
            want_logits = try dsv41_model.extend(xfm.dsv41.?, &suffix, .all);
            break;
        }
    }
    try testing.expect(point > prompt.len);
    const next = next_buf[0 .. point + suffix.len];

    // The same request again, writing the point where the round ends and decoding on to the end.
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var path_buf: [512]u8 = undefined;
    const base = path_buf[0..try tmp.dir.realPath(testing.io, &path_buf)];
    {
        var hc = prefix_cache.HotPrefixCache.initWithMem(a, 0, 0);
        defer hc.deinit();
        hc.ssd_first = true;
        hc.hybrid = true;
        hc.cp_thin = .min_span_recency;
        hc.disk = try kv_disk_cache.DiskTier.init(a, testing.io, base, "dsv41-thought", 0, 128);
        hc.disk.?.ssd_first = true;
        hc.disk.?.cp_thin = .min_span_recency;
        hc.disk.?.armTestSpace(256 << 30, 1 << 40);
        hc.disk.?.enableBackgroundWriter();

        try xfm.resetCache();
        var cache = try KVCache.initWithConfig(a, config.num_hidden_layers, transformer_mod.KVQuantConfig.engine_default);
        defer cache.deinit();
        if (config.swaRingTokens() > 0) cache.setSwaRing(config.sliding_window);
        var off: usize = 0;
        var ctx = xfm.defaultCtx();
        ctx.cache = &cache;
        ctx.moe_seq_offset = &off;
        var gen = try Generator.initWithOptions(io, a, &xfm, &tok, &prompt, want, greedy, &.{}, .{ .ctx = ctx, .dspark_enabled = true });
        defer gen.deinit(a);
        gen.dspark_diag = .{ .rows = 4 };
        gen.dspark_round_ends_at = end_id;
        var written = false;
        while (true) {
            const armed = gen.dspark_round_ends_at != null;
            const r = (try gen.nextDspark(a)) orelse break;
            defer a.free(r.tokens);
            if (!armed or r.tokens[r.tokens.len - 1] != end_id) continue;
            gen.dspark_round_ends_at = null;
            try testing.expectEqual(point, prompt.len + gen.generated_ids.items.len);
            try hc.writeDecodePointDsv41(&cache, xfm.dsv41.?, next[0..point], false, .{}, s);
            written = true;
        }
        try testing.expect(written);
        // The end-of-request commit: the generator's own points (none here) and the whole token record.
        try dsv41_prefix.markRows(&cache);
        var all = std.ArrayList(u32).empty;
        defer all.deinit(a);
        try all.appendSlice(a, &prompt);
        try all.appendSlice(a, gen.generated_ids.items);
        const st = try hc.commitWithRing(&cache, all.items, false, 0, 0, null, null, null, null, prompt.len, .{});
        try testing.expect(st == .disk_only);
        hc.flushPendingDisk(s);
        hc.disk.?.drainWriter();
        try testing.expectEqual(@as(u64, 0), hc.disk.?.writeErrors());
        try testing.expectEqualSlices(u32, &.{@intCast(point)}, hc.disk.?.entries.items[0].ssm_positions);
    }

    // A tier that never saw the write restores the point and runs the suffix to the same bits.
    var hc = prefix_cache.HotPrefixCache.initWithMem(a, 0, 0);
    defer hc.deinit();
    hc.ssd_first = true;
    hc.hybrid = true;
    hc.cp_thin = .min_span_recency;
    hc.disk = try kv_disk_cache.DiskTier.init(a, testing.io, base, "dsv41-thought", 0, 128);
    hc.disk.?.ssd_first = true;
    hc.disk.?.cp_thin = .min_span_recency;
    hc.disk.?.armTestSpace(256 << 30, 1 << 40);
    hc.disk.?.enableBackgroundWriter();
    try xfm.resetCache();
    var target = try KVCache.initWithConfig(a, config.num_hidden_layers, transformer_mod.KVQuantConfig.engine_default);
    defer target.deinit();
    var moe: usize = 0;
    const res = try hc.lookupAndRestoreDsv41(&target, &moe, xfm.dsv41.?, s, next, false, 0, null, &.{}, 9, false);
    try testing.expectEqual(point, res.matched);
    try testing.expect(hc.last_restored_disk_id != null);
    const got = try dsv41_model.extend(xfm.dsv41.?, &suffix, .all);
    defer _ = mlx.mlx_array_free(got);
    try glm5_forward.expectArrayBits(want_logits.?, got);
}

test "dsv41 generator: DSpark requests leave the MLX active memory flat (DSV41_TINY)" {
    try mlx.requireGpu();
    const dsv41_model = @import("dsv41_model.zig");
    const dir = try dsv41_model.fixtureDir();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var fx = try dsv41_model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    var sc = @import("dsv41_scope.zig").Scope.init(a, mlx.gpuStream());
    defer sc.deinit();
    const ids = try dsv41_model.tokens(&sc, &fx);
    defer a.free(ids);
    var config = try model_mod.parseConfig(io, a, dir);
    defer config.deinit(a);
    config.expert_streaming = true;
    config.stream_mtp_head = true;
    config.expert_cache_bytes = 3 * config.num_hidden_layers * try @import("dsv41_stream.zig").expertBytes(&config.dsv41.?);
    config.expert_source_dir = try a.dupe(u8, dir);
    var weights = try model_mod.loadWeightsForConfig(io, a, dir, &config, false);
    defer weights.deinit();
    var xfm = try Transformer.init(io, a, config, &weights);
    defer xfm.deinit();
    var tok = Tokenizer.initEmptyForTests(a);
    defer tok.deinit();

    var baseline: usize = 0;
    var active: usize = 0;
    for (0..12) |i| {
        try xfm.resetCache();
        var gen = try Generator.initWithOptions(io, a, &xfm, &tok, ids[0..30], 30, .{ .temperature = 0.0 }, &.{}, .{ .dspark_enabled = true });
        while (try gen.nextDspark(a)) |r| a.free(r.tokens);
        gen.deinit(a);
        if (i == 3) try mlx.check(mlx.mlx_get_active_memory(&baseline));
    }
    try mlx.check(mlx.mlx_get_active_memory(&active));
    try testing.expect(active <= baseline + 64 * 1024);
}

test "dsv41 generator: serial decode serves sampling, penalties and logprobs, one request after another (DSV41_TINY)" {
    try mlx.requireGpu();
    const dsv41_model = @import("dsv41_model.zig");
    const dir = try dsv41_model.fixtureDir();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var fx = try dsv41_model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    var sc = @import("dsv41_scope.zig").Scope.init(a, mlx.gpuStream());
    defer sc.deinit();
    const ids = try dsv41_model.tokens(&sc, &fx);
    defer a.free(ids);
    var config = try model_mod.parseConfig(io, a, dir);
    defer config.deinit(a);
    config.expert_streaming = true;
    config.stream_mtp_head = true;
    config.expert_cache_bytes = 3 * config.num_hidden_layers * try @import("dsv41_stream.zig").expertBytes(&config.dsv41.?);
    config.expert_source_dir = try a.dupe(u8, dir);
    var weights = try model_mod.loadWeightsForConfig(io, a, dir, &config, false);
    defer weights.deinit();
    var xfm = try Transformer.init(io, a, config, &weights);
    defer xfm.deinit();
    var tok = Tokenizer.initEmptyForTests(a);
    defer tok.deinit();
    const prompt = ids[0..30];
    const cases = [_]struct { sampling: SamplingParams, logprobs: u32, want: u32 = 12 }{
        .{ .sampling = .{ .temperature = 0.0 }, .logprobs = 0 },
        .{ .sampling = .{ .temperature = 0.8, .top_p = 0.9, .top_k = 20, .seed = 5 }, .logprobs = 0 },
        .{ .sampling = .{ .temperature = 0.8, .repeat_penalty = 1.3, .presence_penalty = 0.5, .seed = 5 }, .logprobs = 0 },
        .{ .sampling = .{ .temperature = 0.0 }, .logprobs = 3 },
        .{ .sampling = .{ .temperature = 0.0 }, .logprobs = 0, .want = CACHE_CLEAR_INTERVAL + 8 },
        .{ .sampling = .{ .temperature = 0.0 }, .logprobs = 3, .want = CACHE_CLEAR_INTERVAL + 8 },
    };
    for (cases) |c| {
        var runs: [2]std.ArrayList(u32) = .{ .empty, .empty };
        defer for (&runs) |*r| r.deinit(a);
        for (&runs) |*out| {
            try xfm.resetCache();
            var gen = try Generator.initWithOptions(io, a, &xfm, &tok, prompt, c.want, c.sampling, &.{}, .{ .skip_lazy_preforward = true, .logprobs_n = c.logprobs });
            defer gen.deinit(a);
            gen.logprobs_n = c.logprobs;
            while (try gen.next(a)) |t| {
                try out.append(a, t);
                if (c.logprobs > 0) {
                    const lp = gen.last_logprob orelse return error.MissingLogprobs;
                    try testing.expectEqual(@as(usize, c.logprobs), lp.top_logprobs.len);
                    try testing.expectEqual(t, lp.top_logprobs[0].token_id);
                    try testing.expectEqual(lp.top_logprobs[0].logprob, lp.token_logprob);
                }
            }
            if (c.want > CACHE_CLEAR_INTERVAL) try testing.expect(gen.last_cache_clear_step > 0);
        }
        try testing.expectEqual(@as(usize, c.want), runs[0].items.len);
        try testing.expectEqualSlices(u32, runs[0].items, runs[1].items);
    }
}

const LazyHeadRun = struct {
    tokens: std.ArrayList(u32) = .empty,
    logits: std.ArrayList(f32) = .empty,

    fn deinit(self: *LazyHeadRun, a: std.mem.Allocator) void {
        self.tokens.deinit(a);
        self.logits.deinit(a);
    }
};

fn forwardIds(xfm: *Transformer, ctx: *transformer_mod.ForwardCtx, ids: []const u32) !mlx.mlx_array {
    const a = testing.allocator;
    const ids32 = try a.alloc(i32, ids.len);
    defer a.free(ids32);
    for (ids, ids32) |t, *o| o.* = @intCast(t);
    const shape = [_]c_int{ 1, @intCast(ids.len) };
    const arr = mlx.mlx_array_new_data(ids32.ptr, &shape, 2, .int32);
    defer _ = mlx.mlx_array_free(arr);
    return xfm.forwardWith(ctx, arr);
}

fn argmaxHost(xfm: *Transformer, logits: mlx.mlx_array) !u32 {
    var am = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(am);
    try mlx.check(mlx.mlx_argmax_axis(&am, logits, -1, false, xfm.s));
    try mlx.check(mlx.mlx_array_eval(am));
    var v: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&v, am));
    return @intCast(v);
}

/// Greedy decode by hand: each step's token is evaluated from the logits before anything reads their bytes, as the
/// Generator does; with `lazy` the forward may hand the logits back unevaluated.
fn lazyHeadSteps(a: std.mem.Allocator, xfm: *Transformer, prompt: []const u32, steps: usize, lazy: bool) !LazyHeadRun {
    var run: LazyHeadRun = .{};
    errdefer run.deinit(a);
    try xfm.resetCache();
    var ctx = xfm.defaultCtx();
    var lg = try forwardIds(xfm, &ctx, prompt);
    var tok = try argmaxHost(xfm, lg);
    _ = mlx.mlx_array_free(lg);
    try run.tokens.append(a, tok);
    const vocab = xfm.dsv41.?.cfg.vocab;
    for (0..steps) |_| {
        ctx.lazy_logits = lazy;
        lg = try forwardIds(xfm, &ctx, &.{tok});
        defer _ = mlx.mlx_array_free(lg);
        tok = try argmaxHost(xfm, lg);
        try mlx.check(mlx.mlx_array_eval(lg));
        const data = mlx.mlx_array_data_float32(lg) orelse return error.NoLogits;
        try run.logits.appendSlice(a, data[0..vocab]);
        try run.tokens.append(a, tok);
    }
    return run;
}

/// `lazyHeadSteps` with each step's token handed to the forward as the argmax array itself, read back after the forward
/// returns; `queue` lets the forward queue layer 0 ahead of it.
fn lazyIdSteps(a: std.mem.Allocator, xfm: *Transformer, prompt: []const u32, steps: usize, queue: bool) !LazyHeadRun {
    var run: LazyHeadRun = .{};
    errdefer run.deinit(a);
    const m = xfm.dsv41.?;
    m.lazy_start = queue;
    defer m.lazy_start = true;
    try xfm.resetCache();
    var ctx = xfm.defaultCtx();
    var lg = try forwardIds(xfm, &ctx, prompt);
    defer _ = mlx.mlx_array_free(lg);
    const vocab = m.cfg.vocab;
    for (0..steps + 1) |step| {
        var am = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(am);
        try mlx.check(mlx.mlx_argmax_axis(&am, lg, -1, false, xfm.s));
        if (step < steps) {
            var one = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(one);
            try mlx.check(mlx.mlx_reshape(&one, am, &[_]c_int{ 1, 1 }, 2, xfm.s));
            ctx.lazy_logits = true;
            const next = try xfm.forwardWith(&ctx, one);
            _ = mlx.mlx_array_free(lg);
            lg = next;
        }
        try mlx.check(mlx.mlx_array_eval(am));
        var v: i32 = 0;
        try mlx.check(mlx.mlx_array_item_int32(&v, am));
        try run.tokens.append(a, @intCast(v));
        if (step < steps) {
            try mlx.check(mlx.mlx_array_eval(lg));
            try run.logits.appendSlice(a, (mlx.mlx_array_data_float32(lg) orelse return error.NoLogits)[0..vocab]);
        }
    }
    return run;
}

/// A lazy pass whose token the model refuses (an image id with no image rows) is dropped from the state it was queued on.
fn droppedStartLeavesNothing(a: std.mem.Allocator, xfm: *Transformer, prompt: []const u32, token: u32) !void {
    const m = xfm.dsv41.?;
    const vocab = m.cfg.vocab;
    var logits: [2]std.ArrayList(f32) = @splat(.empty);
    defer for (&logits) |*l| l.deinit(a);
    for (&logits, [_]bool{ false, true }) |*out, dropped| {
        try xfm.resetCache();
        var ctx = xfm.defaultCtx();
        const first = try forwardIds(xfm, &ctx, prompt);
        _ = mlx.mlx_array_free(first);
        if (dropped) {
            const image_id = m.cfg.image_token_id;
            m.cfg.image_token_id = token;
            defer m.cfg.image_token_id = image_id;
            const base = [_]i32{ @intCast(token), 0 };
            const held = mlx.mlx_array_new_data(&base, &[_]c_int{ 1, 1 }, 2, .int32);
            defer _ = mlx.mlx_array_free(held);
            const none_ = mlx.mlx_array_new_data(&base[1], &[_]c_int{ 1, 1 }, 2, .int32);
            defer _ = mlx.mlx_array_free(none_);
            var id = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(id);
            try mlx.check(mlx.mlx_add(&id, held, none_, xfm.s));
            ctx.lazy_logits = true;
            const before = m.lazy_starts;
            try testing.expectError(error.Dsv41VisionRowsMissing, xfm.forwardWith(&ctx, id));
            try testing.expectEqual(before, m.lazy_starts);
            try testing.expectEqual(prompt.len, ctx.cache.step);
        }
        ctx.lazy_logits = false;
        const lg = try forwardIds(xfm, &ctx, &.{token});
        defer _ = mlx.mlx_array_free(lg);
        try mlx.check(mlx.mlx_array_eval(lg));
        try out.appendSlice(a, (mlx.mlx_array_data_float32(lg) orelse return error.NoLogits)[0..vocab]);
    }
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(logits[0].items), std.mem.sliceAsBytes(logits[1].items));
}

fn lazyHeadProof(dir: []const u8, stq: bool, dspark: bool) !void {
    try mlx.requireGpu();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const dsv41_model = @import("dsv41_model.zig");
    var fx = try dsv41_model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    var sc = @import("dsv41_scope.zig").Scope.init(a, mlx.gpuStream());
    defer sc.deinit();
    const ids = try dsv41_model.tokens(&sc, &fx);
    defer a.free(ids);
    var config = try model_mod.parseConfig(io, a, dir);
    defer config.deinit(a);
    config.expert_streaming = true;
    config.stream_mtp_head = dspark;
    config.expert_cache_bytes = if (stq) 64 << 20 else 3 * config.num_hidden_layers * try @import("dsv41_stream.zig").expertBytes(&config.dsv41.?);
    config.expert_source_dir = try a.dupe(u8, dir);
    var weights = try model_mod.loadWeightsForConfig(io, a, dir, &config, false);
    defer weights.deinit();
    var xfm = try Transformer.init(io, a, config, &weights);
    defer xfm.deinit();
    const m = xfm.dsv41.?;
    try testing.expectEqual(dspark, m.dspark != null);
    var tok = Tokenizer.initEmptyForTests(a);
    defer tok.deinit();
    const prompt = ids[0..30];
    const steps: usize = 24;

    // The warm-up's DSpark verifies defer their heads too.
    const warmed = m.heads_deferred;
    var want = try lazyHeadSteps(a, &xfm, prompt, steps, false);
    defer want.deinit(a);
    try testing.expectEqual(warmed, m.heads_deferred);
    var got = try lazyHeadSteps(a, &xfm, prompt, steps, true);
    defer got.deinit(a);
    try testing.expectEqual(warmed + steps, m.heads_deferred);
    try testing.expectEqualSlices(u32, want.tokens.items, got.tokens.items);
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(want.logits.items), std.mem.sliceAsBytes(got.logits.items));

    // The Generator's pipelined greedy decode takes the same path and says the same tokens: the waited head, the
    // deferred head, and the deferred head with layer 0 queued ahead of the token.
    var served: [3]std.ArrayList(u32) = @splat(.empty);
    defer for (&served) |*r| r.deinit(a);
    var used: [3]u64 = undefined;
    var started: [3]u64 = undefined;
    for (&served, &used, &started, [_][2]bool{ .{ false, false }, .{ true, false }, .{ true, true } }) |*out, *n, *started_n, arm| {
        m.lazy_head, m.lazy_start = arm;
        const before = m.heads_deferred;
        const before_starts = m.lazy_starts;
        try xfm.resetCache();
        var gen = try Generator.initWithOptions(io, a, &xfm, &tok, prompt, @intCast(steps + 1), .{ .temperature = 0.0 }, &.{}, .{});
        defer gen.deinit(a);
        while (try gen.next(a)) |t| try out.append(a, t);
        n.* = m.heads_deferred - before;
        started_n.* = m.lazy_starts - before_starts;
    }
    m.lazy_head = true;
    m.lazy_start = true;
    try testing.expectEqual(@as(u64, 0), used[0]);
    try testing.expect(used[1] >= steps - 2);
    try testing.expectEqual([_]u64{ 0, 0 }, started[0..2].*);
    try testing.expect(started[2] >= steps - 2);
    try testing.expectEqual(@as(usize, steps + 1), served[0].items.len);
    try testing.expectEqualSlices(u32, served[0].items, served[1].items);
    try testing.expectEqualSlices(u32, served[0].items, served[2].items);

    // A token still on the GPU, as the Generator's argmax is: the logits and tokens match the token read first.
    var held_back = try lazyIdSteps(a, &xfm, prompt, steps, false);
    defer held_back.deinit(a);
    const queued_before = m.lazy_starts;
    var queued = try lazyIdSteps(a, &xfm, prompt, steps, true);
    defer queued.deinit(a);
    try testing.expectEqual(@as(u64, steps), m.lazy_starts - queued_before);
    try testing.expectEqualSlices(u32, want.tokens.items, queued.tokens.items);
    try testing.expectEqualSlices(u32, held_back.tokens.items, queued.tokens.items);
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(held_back.logits.items), std.mem.sliceAsBytes(queued.logits.items));
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(want.logits.items), std.mem.sliceAsBytes(queued.logits.items));

    // A token the pass refuses leaves nothing of its layer 0 behind: the same token decodes to the same bytes after.
    try droppedStartLeavesNothing(a, &xfm, prompt, want.tokens.items[0]);

    // A request that reads the logits themselves keeps the evaluated head.
    const kept = m.heads_deferred;
    try xfm.resetCache();
    var lp = try Generator.initWithOptions(io, a, &xfm, &tok, prompt, 8, .{ .temperature = 0.0 }, &.{}, .{ .logprobs_n = 3 });
    defer lp.deinit(a);
    lp.logprobs_n = 3;
    while (try lp.next(a)) |_| {}
    try testing.expectEqual(kept, m.heads_deferred);
}

test "dsv41 generator: a one-row decode that hands its logits back unevaluated and queues layer 0 ahead of its token gives the evaluated head's tokens and logits bit for bit (DSV41_TINY)" {
    const dir = try @import("dsv41_model.zig").fixtureDir();
    try lazyHeadProof(dir, false, false);
    try lazyHeadProof(dir, false, true);
}

test "dsv41 generator: a one-row decode that hands its logits back unevaluated and queues layer 0 ahead of its token gives the evaluated head's tokens and logits bit for bit (DSV41_STQ_TINY)" {
    const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    try lazyHeadProof(dir, true, false);
    try lazyHeadProof(dir, true, true);
}

test "dsv41 generator: a thinking bound fires between DSpark rounds, commits its closer and decodes the rest serially (DSV41_TINY)" {
    try mlx.requireGpu();
    const dsv41_model = @import("dsv41_model.zig");
    const dir = try dsv41_model.fixtureDir();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var fx = try dsv41_model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    var sc = @import("dsv41_scope.zig").Scope.init(a, mlx.gpuStream());
    defer sc.deinit();
    const ids = try dsv41_model.tokens(&sc, &fx);
    defer a.free(ids);
    var config = try model_mod.parseConfig(io, a, dir);
    defer config.deinit(a);
    config.expert_streaming = true;
    config.stream_mtp_head = true;
    config.expert_cache_bytes = 3 * config.num_hidden_layers * try @import("dsv41_stream.zig").expertBytes(&config.dsv41.?);
    config.expert_source_dir = try a.dupe(u8, dir);
    var weights = try model_mod.loadWeightsForConfig(io, a, dir, &config, false);
    defer weights.deinit();
    var xfm = try Transformer.init(io, a, config, &weights);
    defer xfm.deinit();
    var tok = Tokenizer.initEmptyForTests(a);
    defer tok.deinit();
    const closer: u32 = 4;
    const forced = [_]u32{ 3, closer, 10 };
    var bound = ThinkBound{ .budget = 6, .opener_id = null, .closer_id = closer, .forced = &forced, .in_think = true };
    const want: u32 = 30;
    var gen = try Generator.initWithOptions(io, a, &xfm, &tok, ids[0..30], want, .{ .temperature = 0.0, .think_bound = &bound }, &.{}, .{ .dspark_enabled = true });
    defer gen.deinit(a);
    try testing.expect(gen.dspark);
    var out: std.ArrayList(u32) = .empty;
    defer out.deinit(a);
    // The scheduler's order: the bound is checked before every tick; a fired bound decodes serially.
    while (true) {
        bound.observe(gen.generated_ids.items);
        if (bound.due()) {
            bound.fired = true;
            const r = try gen.commitForcedTokens(a, &forced);
            defer a.free(r.emitted);
            try out.appendSlice(a, r.emitted);
            if (r.stopped) break;
            continue;
        }
        if (bound.fired) {
            const t = (try gen.next(a)) orelse break;
            try out.append(a, t);
        } else {
            const r = (try gen.nextDspark(a)) orelse break;
            defer a.free(r.tokens);
            try out.appendSlice(a, r.tokens);
        }
    }
    try testing.expect(bound.fired);
    // The last forced id stays pending and is emitted by the serial tick that follows.
    const at = std.mem.indexOf(u32, out.items, forced[0 .. forced.len - 1]) orelse return error.ForcedTokensMissing;
    try testing.expectEqual(forced[forced.len - 1], out.items[at + forced.len - 1]);
    try testing.expect(at >= bound.budget);
    try testing.expectEqual(@as(usize, want), out.items.len);
}

test "a DSpark round never verifies past the position serial decode closes the thought at" {
    // Serial decode fires when the thought holds `budget` tokens; a round emits t1 plus its accepted drafts.
    for ([_]u32{ 1, 2, 3, 9, 48 }) |budget| {
        for ([_]usize{ 1, 2, 3, 6 }) |width| {
            for ([_]u32{ 0, 1, 2, 5 }) |accept_every| {
                var tb = ThinkBound{ .budget = budget, .opener_id = null, .closer_id = 99, .forced = &.{99}, .in_think = true };
                var emitted: u32 = 0;
                while (!tb.due()) {
                    const rows = dsparkRowsForRoom(width, tb.room());
                    try testing.expect(rows >= 1);
                    emitted += 1 + @min(@as(u32, @intCast(rows - 1)), accept_every);
                    tb.count = emitted;
                    try testing.expect(emitted <= budget);
                }
                try testing.expectEqual(budget, emitted);
            }
        }
    }
    try testing.expectEqual(@as(usize, 5), dsparkRowsForRoom(5, null));
    try testing.expectEqual(@as(usize, 1), dsparkRowsForRoom(5, 0));
    try testing.expectEqual(@as(usize, 2), dsparkRowsForRoom(5, 2));
}

/// `tail`: DSpark rounds up to the bound, then the old behavior (the forced commit, serial ticks after it).
const ThinkBoundMode = union(enum) { serial, tail, spark: DsparkDiag };

const ThinkBoundOutcome = struct {
    tokens: std.ArrayList(u32) = .empty,
    /// Tokens published when the bound fired.
    fired_at: ?usize = null,
    rounds_after: u64 = 0,
    drafted_after: u64 = 0,
    /// Rounds by verify width before the bound fired.
    rows_before: [dsv41_max_drafts + 2]u32 = @splat(0),
    resumed: bool = false,

    fn deinit(self: *ThinkBoundOutcome, a: std.mem.Allocator) void {
        self.tokens.deinit(a);
    }
};

/// The scheduler's order for one request: the bound is checked before every tick, `commitThoughtClose` commits the
/// forced ids and (DSpark) drains them back to a round entry.
fn thinkBoundRun(
    a: std.mem.Allocator,
    io: std.Io,
    xfm: *Transformer,
    tok: *Tokenizer,
    prompt: []const u32,
    want: u32,
    budget: u32,
    forced: []const u32,
    mode: ThinkBoundMode,
    params: SamplingParams,
) !ThinkBoundOutcome {
    var bound = ThinkBound{ .budget = budget, .opener_id = null, .closer_id = forced[forced.len - 2], .forced = forced, .in_think = true };
    try xfm.resetCache();
    var sampling = params;
    sampling.think_bound = &bound;
    var gen = try Generator.initWithOptions(io, a, xfm, tok, prompt, want, sampling, &.{}, switch (mode) {
        .serial => .{ .skip_lazy_preforward = true },
        .tail, .spark => .{ .dspark_enabled = true },
    });
    defer gen.deinit(a);
    if (mode != .serial) {
        try testing.expect(gen.dspark);
        gen.dspark_diag = if (mode == .spark) mode.spark else .{ .rows = 1 };
    }
    var out = ThinkBoundOutcome{};
    errdefer out.deinit(a);
    var at_fire = gen.dspark_stats;
    while (true) {
        bound.observe(gen.generated_ids.items);
        if (bound.due()) {
            bound.fired = true;
            out.fired_at = gen.generated_ids.items.len;
            at_fire = gen.dspark_stats;
            out.rows_before = at_fire.rows_hist;
            const r = try gen.commitForcedTokens(a, forced);
            defer a.free(r.emitted);
            try out.tokens.appendSlice(a, r.emitted);
            if (r.stopped) break;
            if (mode == .spark) switch (try gen.dsparkResumeAfterForced(a)) {
                .resumed => |closer| {
                    try out.tokens.append(a, closer);
                    out.resumed = true;
                    try testing.expect(!gen.spec_disabled_runtime);
                    try testing.expectEqual(SpecDisableReason.none, gen.spec_disable_reason);
                },
                .stopped => break,
                .refused => {},
            };
            continue;
        }
        if (mode != .serial and (!bound.fired or out.resumed)) {
            const r = (try gen.nextDspark(a)) orelse break;
            defer a.free(r.tokens);
            try out.tokens.appendSlice(a, r.tokens);
        } else {
            const t = (try gen.next(a)) orelse break;
            try out.tokens.append(a, t);
        }
    }
    out.rounds_after = gen.dspark_stats.rounds - at_fire.rounds;
    out.drafted_after = gen.dspark_stats.drafted - at_fire.drafted;
    return out;
}

fn thinkBoundSparkProof(env: [*:0]const u8, stq: bool) !void {
    try mlx.requireGpu();
    const dsv41_model = @import("dsv41_model.zig");
    const dir = std.mem.span(std.c.getenv(env) orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var fx = try dsv41_model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    var sc = @import("dsv41_scope.zig").Scope.init(a, mlx.gpuStream());
    defer sc.deinit();
    const ids = try dsv41_model.tokens(&sc, &fx);
    defer a.free(ids);
    var config = try model_mod.parseConfig(io, a, dir);
    defer config.deinit(a);
    config.expert_streaming = true;
    config.stream_mtp_head = true;
    config.expert_cache_bytes = if (stq) 64 << 20 else 3 * config.num_hidden_layers * try @import("dsv41_stream.zig").expertBytes(&config.dsv41.?);
    config.expert_source_dir = try a.dupe(u8, dir);
    var weights = try model_mod.loadWeightsForConfig(io, a, dir, &config, false);
    defer weights.deinit();
    var xfm = try Transformer.init(io, a, config, &weights);
    defer xfm.deinit();
    var tok = Tokenizer.initEmptyForTests(a);
    defer tok.deinit();
    const prompt = ids[0..30];
    const want: u32 = 30;
    const forced = [_]u32{ 3, 4, 10 };

    for ([_]u32{ 2, 4, 7 }) |budget| {
        // One-row rounds that do not sum by width are the serial step. The forced commit is one forward whose width
        // depends on the exit state (a spec exit re-feeds t1), so serial decode agrees through the forced ids only;
        // the answer is compared with the same commit followed by serial ticks.
        dsv41_model.setDsparkExact(xfm.dsv41.?, false);
        var serial = try thinkBoundRun(a, io, &xfm, &tok, prompt, want, budget, &forced, .serial, .{ .temperature = 0.0 });
        defer serial.deinit(a);
        var tail = try thinkBoundRun(a, io, &xfm, &tok, prompt, want, budget, &forced, .tail, .{ .temperature = 0.0 });
        defer tail.deinit(a);
        var one = try thinkBoundRun(a, io, &xfm, &tok, prompt, want, budget, &forced, .{ .spark = .{ .rows = 1 } }, .{ .temperature = 0.0 });
        defer one.deinit(a);
        try testing.expectEqual(@as(?usize, budget), serial.fired_at);
        try testing.expectEqual(serial.fired_at, one.fired_at);
        try testing.expect(one.resumed and one.rounds_after > 0);
        try testing.expectEqualSlices(u32, serial.tokens.items[0 .. budget + forced.len + 1], one.tokens.items[0 .. budget + forced.len + 1]);
        try testing.expectEqualSlices(u32, tail.tokens.items, one.tokens.items);

        // Exact rounds (the STQ trunk) give a row the same bytes at every width: any mix, cut by the budget, equals the
        // one-row stream.
        if (!stq) continue;
        dsv41_model.setDsparkExact(xfm.dsv41.?, true);
        var base = try thinkBoundRun(a, io, &xfm, &tok, prompt, want, budget, &forced, .{ .spark = .{ .rows = 1 } }, .{ .temperature = 0.0 });
        defer base.deinit(a);
        try testing.expectEqual(@as(?usize, budget), base.fired_at);
        for ([_]DsparkDiag{ .{ .rows = 3 }, .{ .rows = 6 }, .sweep, .off }) |diag| {
            var got = try thinkBoundRun(a, io, &xfm, &tok, prompt, want, budget, &forced, .{ .spark = diag }, .{ .temperature = 0.0 });
            defer got.deinit(a);
            try testing.expectEqual(base.fired_at, got.fired_at);
            for (got.rows_before, 0..) |n, rows| if (rows > budget) try testing.expectEqual(@as(u32, 0), n);
            try testing.expect(got.resumed and got.rounds_after > 0);
            try testing.expectEqualSlices(u32, base.tokens.items, got.tokens.items);
            if (diag == .rows and diag.rows > 1) try testing.expect(got.drafted_after > 0);
        }

        // A seeded sampled request draws per generated position, so the round mix (the adaptive width follows
        // measured ms) changes neither the thought nor the answer behind the forced close.
        const sampled = SamplingParams{ .temperature = 0.8, .seed = 7 };
        var seeded = try thinkBoundRun(a, io, &xfm, &tok, prompt, want, budget, &forced, .{ .spark = .{ .rows = 1 } }, sampled);
        defer seeded.deinit(a);
        for ([_]DsparkDiag{ .{ .rows = 3 }, .{ .rows = 6 }, .sweep, .off }) |diag| {
            var got = try thinkBoundRun(a, io, &xfm, &tok, prompt, want, budget, &forced, .{ .spark = diag }, sampled);
            defer got.deinit(a);
            try testing.expect(got.resumed and got.rounds_after > 0);
            try testing.expectEqualSlices(u32, seeded.tokens.items, got.tokens.items);
        }
    }
}

test "dsv41 generator: a thinking bound that fires mid-round cuts it at the serial token, and DSpark decodes the answer (DSV41_TINY)" {
    try thinkBoundSparkProof("DSV41_TINY", false);
}

test "dsv41 generator: a thinking bound that fires mid-round cuts it at the serial token, and DSpark decodes the answer (DSV41_STQ_TINY)" {
    try thinkBoundSparkProof("DSV41_STQ_TINY", true);
}

test "dspark eligibility: sampled, penalized and thinking-bound requests draft, logprobs do not" {
    try testing.expect(dsparkEligible(.{ .temperature = 0.0 }, 0));
    try testing.expect(dsparkEligible(.{ .temperature = 1.0, .top_p = 0.9, .top_k = 40 }, 0));
    try testing.expect(dsparkEligible(.{ .temperature = 1.0, .repeat_penalty = 1.1 }, 0));
    try testing.expect(dsparkEligible(.{ .temperature = 0.0, .presence_penalty = 0.5 }, 0));
    try testing.expect(!dsparkEligible(.{ .temperature = 1.0 }, 3));
    var bound = ThinkBound{ .budget = 10, .opener_id = null, .closer_id = 1, .forced = &.{}, .in_think = true };
    try testing.expect(dsparkEligible(.{ .temperature = 1.0, .think_bound = &bound }, 0));
}

test "dsv41 generator: prefill checkpoints sit on chunk boundaries, and a restore at one prefills its suffix exactly like cold (DSV41_TINY)" {
    try mlx.requireGpu();
    const dsv41_model = @import("dsv41_model.zig");
    const dir = try dsv41_model.fixtureDir();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var fx = try dsv41_model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    var sc = @import("dsv41_scope.zig").Scope.init(a, mlx.gpuStream());
    defer sc.deinit();
    const ids = try dsv41_model.tokens(&sc, &fx);
    defer a.free(ids);
    var config = try model_mod.parseConfig(io, a, dir);
    defer config.deinit(a);
    config.expert_streaming = true;
    config.stream_mtp_head = true;
    config.expert_cache_bytes = 3 * config.num_hidden_layers * try @import("dsv41_stream.zig").expertBytes(&config.dsv41.?);
    config.expert_source_dir = try a.dupe(u8, dir);
    var weights = try model_mod.loadWeightsForConfig(io, a, dir, &config, false);
    defer weights.deinit();
    var tok = Tokenizer.initEmptyForTests(a);
    defer tok.deinit();
    const greedy = SamplingParams{ .temperature = 0.0 };

    var first: [100]u32 = undefined;
    for (&first, 0..) |*t, i| t.* = ids[i % ids.len];
    var next: [100]u32 = undefined;
    @memcpy(next[0..80], first[0..80]);
    for (next[80..], 80..) |*t, i| t.* = ids[(i * 3 + 1) % ids.len];
    const opts: Generator.InitOptions = .{ .skip_lazy_preforward = true, .ssm_checkpoint_stride = 16, .ssm_checkpoint_max = 4 };

    var xfm = try Transformer.init(io, a, config, &weights);
    defer xfm.deinit();
    var gen = try Generator.initWithOptions(io, a, &xfm, &tok, &first, 4, greedy, &.{}, opts);
    defer gen.deinit(a);
    // The tiny prompt is one chunk, so the prompt-end snapshot (99 - 30) is the only restore point.
    try testing.expectEqual(@as(usize, 1), gen.ssm_checkpoints.items.len);
    try testing.expectEqual(@as(usize, 69), gen.ssm_checkpoints.items[0].pos);

    try xfm.resetCache();
    var want: std.ArrayList(u32) = .empty;
    defer want.deinit(a);
    {
        var cold = try Generator.initWithOptions(io, a, &xfm, &tok, &next, 6, greedy, &.{}, opts);
        defer cold.deinit(a);
        try want.append(a, cold.next_token_id);
        while (try cold.next(a)) |t| try want.append(a, t);
    }

    // The first request's own checkpoint at 69 serves the second, which shares 80 tokens with it.
    try xfm.resetCache();
    try dsv41_prefix.restore(xfm.dsv41.?, &gen.ssm_checkpoints.items[0], &next);
    xfm.cache.step = 69;
    var warm_opts = opts;
    warm_opts.ssm_checkpoint_pos_offset = 69;
    var warm = try Generator.initWithOptions(io, a, &xfm, &tok, next[69..], 6, greedy, &.{}, warm_opts);
    defer warm.deinit(a);
    var got: std.ArrayList(u32) = .empty;
    defer got.deinit(a);
    try got.append(a, warm.next_token_id);
    while (try warm.next(a)) |t| try got.append(a, t);
    try testing.expectEqualSlices(u32, want.items, got.items);
}

test "dsv41 generator: under a write-through every restore point the tier takes is gone before the next forward, and the tokens do not move (DSV41_TINY)" {
    try mlx.requireGpu();
    const dsv41_model = @import("dsv41_model.zig");
    const floor = @import("../kv_disk_cache.zig").MIN_PERSIST_TOKENS;
    const dir = try dsv41_model.fixtureDir();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var fx = try dsv41_model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    var sc = @import("dsv41_scope.zig").Scope.init(a, mlx.gpuStream());
    defer sc.deinit();
    const ids = try dsv41_model.tokens(&sc, &fx);
    defer a.free(ids);
    var config = try model_mod.parseConfig(io, a, dir);
    defer config.deinit(a);
    config.expert_streaming = true;
    config.stream_mtp_head = true;
    config.expert_cache_bytes = 3 * config.num_hidden_layers * try @import("dsv41_stream.zig").expertBytes(&config.dsv41.?);
    config.expert_source_dir = try a.dupe(u8, dir);
    var tok = Tokenizer.initEmptyForTests(a);
    defer tok.deinit();
    const greedy = SamplingParams{ .temperature = 0.0 };
    var prompt: [1400]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = ids[(i * 5 + 3) % ids.len];
    const saved_chunk = generate.prefill_chunk_override;
    defer generate.prefill_chunk_override = saved_chunk;
    generate.prefill_chunk_override = 256;
    const base: Generator.InitOptions = .{ .skip_lazy_preforward = true, .pinned_prefill_chunk = 256, .adaptive_chunk_width = false, .ssm_checkpoint_stride = 256, .ssm_checkpoint_max = 4 };

    var want: std.ArrayList(u32) = .empty;
    defer want.deinit(a);
    {
        // The load takes each tensor out of the map, so each model reads its own.
        var weights = try model_mod.loadWeightsForConfig(io, a, dir, &config, false);
        defer weights.deinit();
        var xfm = try Transformer.init(io, a, config, &weights);
        defer xfm.deinit();
        var kept = try Generator.initWithOptions(io, a, &xfm, &tok, &prompt, 6, greedy, &.{}, base);
        defer kept.deinit(a);
        try want.append(a, kept.next_token_id);
        while (try kept.next(a)) |t| try want.append(a, t);
    }

    // The tier's part: past its floor it takes every point it is shown and releases it, the newest too.
    const Tier = struct {
        seen: [256]usize = undefined,
        n: usize = 0,
        fn call(ctx: *anyopaque, pos: usize, cps: []SSMCheckpoint) void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            self.seen[self.n] = pos;
            self.n += 1;
            if (pos < floor) return;
            for (cps) |*cp| cp.release();
        }
    };
    var tier = Tier{};
    var opts = base;
    opts.write_through_hook = .{ .ctx = &tier, .call = Tier.call };
    var weights = try model_mod.loadWeightsForConfig(io, a, dir, &config, false);
    defer weights.deinit();
    var xfm = try Transformer.init(io, a, config, &weights);
    defer xfm.deinit();
    var gen = try Generator.initWithOptions(io, a, &xfm, &tok, &prompt, 6, greedy, &.{}, opts);
    defer gen.deinit(a);
    // Points at 256, 512 and 768 and the prompt-end snapshot at 1369, which the tier sees before the final span runs.
    try testing.expectEqual(@as(usize, 1369), tier.seen[tier.n - 1]);
    try testing.expectEqual(@as(usize, 4), gen.ssm_checkpoints.items.len);
    for (gen.ssm_checkpoints.items) |*cp| try testing.expect(cp.isReleased());
    var got: std.ArrayList(u32) = .empty;
    defer got.deinit(a);
    try got.append(a, gen.next_token_id);
    while (try gen.next(a)) |t| try got.append(a, t);
    try testing.expectEqualSlices(u32, want.items, got.items);
}

/// A request over the tiny pack whose prefill leaves the decoder tails nothing reads (or runs them all), kept with its
/// model so its restore points can be continued.
const TailArm = struct {
    weights: model_mod.Weights,
    xfm: Transformer,
    gen: Generator,
    tokens: std.ArrayList(u32) = .empty,

    fn start(a: std.mem.Allocator, config: model_mod.ModelConfig, dir: []const u8, tok: *Tokenizer, prompt: []const u32, offset: usize, skip: bool) !*TailArm {
        const io = std.Io.Threaded.global_single_threaded.io();
        const self = try a.create(TailArm);
        errdefer a.destroy(self);
        self.tokens = .empty;
        var cfg = config;
        self.weights = try model_mod.loadWeightsForConfig(io, a, dir, &cfg, false);
        errdefer self.weights.deinit();
        self.xfm = try Transformer.init(io, a, config, &self.weights);
        errdefer self.xfm.deinit();
        const m = self.xfm.dsv41.?;
        // Wider than a decode-width pass, as the release's window is: a narrower replay reads the ring below its floor.
        m.cfg.window = 20;
        m.s = self.xfm.s;
        if (offset > 0) {
            _ = mlx.mlx_array_free(try @import("dsv41_model.zig").extend(m, prompt[0..offset], .last));
            self.xfm.cache.step = offset;
        }
        const opts: Generator.InitOptions = .{
            .skip_lazy_preforward = true,
            .pinned_prefill_chunk = 256,
            .adaptive_chunk_width = false,
            .ssm_checkpoint_stride = 512,
            .ssm_checkpoint_max = 8,
            .ssm_checkpoint_pos_offset = offset,
            .skip_unread_tails = skip,
        };
        self.gen = try Generator.initWithOptions(io, a, &self.xfm, tok, prompt[offset..], 6, .{ .temperature = 0.0 }, &.{}, opts);
        errdefer self.gen.deinit(a);
        try self.tokens.append(a, self.gen.next_token_id);
        while (try self.gen.next(a)) |t| try self.tokens.append(a, t);
        return self;
    }

    fn destroy(self: *TailArm, a: std.mem.Allocator) void {
        self.tokens.deinit(a);
        self.gen.deinit(a);
        self.xfm.deinit();
        self.weights.deinit();
        a.destroy(self);
    }

    /// The logits of what a request does after restore point `i`: a pass of fewer rows than the window, then decode.
    fn continueAt(self: *TailArm, a: std.mem.Allocator, sc: *@import("dsv41_scope.zig").Scope, i: usize, prompt: []const u32) ![]f32 {
        const dsv41_model = @import("dsv41_model.zig");
        const m = self.xfm.dsv41.?;
        const cp = &self.gen.ssm_checkpoints.items[i];
        try dsv41_prefix.restore(m, cp, prompt);
        var rows: std.ArrayList(f32) = .empty;
        errdefer rows.deinit(a);
        var at = cp.pos;
        for ([_]usize{ 5, 1, 1, 1 }) |n| {
            const lg = try dsv41_model.extend(m, prompt[at..][0..n], .all);
            defer _ = mlx.mlx_array_free(lg);
            const h = try dsv41_model.hostRows(sc, lg);
            defer a.free(h);
            try rows.appendSlice(a, h);
            at += n;
        }
        return rows.toOwnedSlice(a);
    }
};

test "dsv41 generator: a prefill that skips the decoder tails nothing reads gives the tokens and restore points of the full passes, and a short continuation after each (DSV41_TINY)" {
    try mlx.requireGpu();
    const dsv41_model = @import("dsv41_model.zig");
    const dir = try dsv41_model.fixtureDir();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var fx = try dsv41_model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    var sc = @import("dsv41_scope.zig").Scope.init(a, mlx.gpuStream());
    defer sc.deinit();
    const ids = try dsv41_model.tokens(&sc, &fx);
    defer a.free(ids);
    var config = try model_mod.parseConfig(io, a, dir);
    defer config.deinit(a);
    config.expert_streaming = true;
    config.stream_mtp_head = true;
    config.expert_cache_bytes = 3 * config.num_hidden_layers * try @import("dsv41_stream.zig").expertBytes(&config.dsv41.?);
    config.expert_source_dir = try a.dupe(u8, dir);
    var tok = Tokenizer.initEmptyForTests(a);
    defer tok.deinit();
    var prompt: [1400]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = ids[(i * 5 + 3) % ids.len];
    const saved_chunk = generate.prefill_chunk_override;
    defer generate.prefill_chunk_override = saved_chunk;
    generate.prefill_chunk_override = 256;

    // Offset 0: chunks end at 256, 512, 768 and 1369 (a merged tail); points at 512 and the prompt end. The chunks
    // ending at 256 and 768 leave state only. Offset 246: the chunk ending at 502 is followed by one of ten rows,
    // which a skipped tail would leave stale; only the chunk ending at 768 is skipped.
    for ([_]struct { offset: usize, skipped: u64 }{ .{ .offset = 0, .skipped = 2 }, .{ .offset = 246, .skipped = 1 } }) |c| {
        const full = try TailArm.start(a, config, dir, &tok, &prompt, c.offset, false);
        defer full.destroy(a);
        const lean = try TailArm.start(a, config, dir, &tok, &prompt, c.offset, true);
        defer lean.destroy(a);
        try testing.expectEqual(@as(u64, 0), full.xfm.dsv41.?.state_only_used);
        try testing.expectEqual(c.skipped, lean.xfm.dsv41.?.state_only_used);
        try testing.expectEqualSlices(u32, full.tokens.items, lean.tokens.items);
        try testing.expectEqual(@as(usize, 2), full.gen.ssm_checkpoints.items.len);
        try testing.expectEqual(full.gen.ssm_checkpoints.items.len, lean.gen.ssm_checkpoints.items.len);
        for (full.gen.ssm_checkpoints.items, lean.gen.ssm_checkpoints.items, 0..) |f, l, i| {
            try testing.expectEqual(f.pos, l.pos);
            const want = try full.continueAt(a, &sc, i, &prompt);
            defer a.free(want);
            const got = try lean.continueAt(a, &sc, i, &prompt);
            defer a.free(got);
            try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(want), std.mem.sliceAsBytes(got));
        }
    }
}

test "dsv41 generator: a chunked prefill tells the model the tokens after each chunk, and the next pass takes their Engram rows (DSV41_TINY)" {
    try mlx.requireGpu();
    const dsv41_model = @import("dsv41_model.zig");
    const dir = try dsv41_model.fixtureDir();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var fx = try dsv41_model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    var sc = @import("dsv41_scope.zig").Scope.init(a, mlx.gpuStream());
    defer sc.deinit();
    const ids = try dsv41_model.tokens(&sc, &fx);
    defer a.free(ids);
    var config = try model_mod.parseConfig(io, a, dir);
    defer config.deinit(a);
    config.expert_streaming = true;
    config.stream_mtp_head = true;
    config.expert_cache_bytes = 3 * config.num_hidden_layers * try @import("dsv41_stream.zig").expertBytes(&config.dsv41.?);
    config.expert_source_dir = try a.dupe(u8, dir);
    var tok = Tokenizer.initEmptyForTests(a);
    defer tok.deinit();
    const greedy = SamplingParams{ .temperature = 0.0 };
    var prompt: [1400]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = ids[(i * 5 + 3) % ids.len];
    const saved_chunk = generate.prefill_chunk_override;
    defer generate.prefill_chunk_override = saved_chunk;
    generate.prefill_chunk_override = 256;

    // The hook runs once a chunk and once more at the prompt end, before the final span.
    const Count = struct {
        n: usize = 0,
        fn call(ctx: *anyopaque, _: usize, _: []SSMCheckpoint) void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            self.n += 1;
        }
    };
    var count = Count{};
    const opts: Generator.InitOptions = .{ .skip_lazy_preforward = true, .pinned_prefill_chunk = 256, .adaptive_chunk_width = false, .ssm_checkpoint_stride = 256, .ssm_checkpoint_max = 4, .write_through_hook = .{ .ctx = &count, .call = Count.call } };
    var weights = try model_mod.loadWeightsForConfig(io, a, dir, &config, false);
    defer weights.deinit();
    var xfm = try Transformer.init(io, a, config, &weights);
    defer xfm.deinit();
    var gen = try Generator.initWithOptions(io, a, &xfm, &tok, &prompt, 6, greedy, &.{}, opts);
    defer gen.deinit(a);
    try testing.expect(count.n > 4);
    // Every forward but the first, the final span included, started with the rows read during the one before it.
    try testing.expectEqual(@as(u64, count.n - 1), xfm.dsv41.?.engram_ahead_used);
    try testing.expect(xfm.dsv41.?.engram_ahead == null);
}

test "dsv41 generator: after a DSpark request the slot cache marks its rows and snapshots like the scheduler's commit (DSV41_TINY)" {
    try mlx.requireGpu();
    const dsv41_model = @import("dsv41_model.zig");
    const dir = try dsv41_model.fixtureDir();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var fx = try dsv41_model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    var sc = @import("dsv41_scope.zig").Scope.init(a, mlx.gpuStream());
    defer sc.deinit();
    const ids = try dsv41_model.tokens(&sc, &fx);
    defer a.free(ids);
    var config = try model_mod.parseConfig(io, a, dir);
    defer config.deinit(a);
    config.expert_streaming = true;
    config.stream_mtp_head = true;
    config.expert_cache_bytes = 3 * config.num_hidden_layers * try @import("dsv41_stream.zig").expertBytes(&config.dsv41.?);
    config.expert_source_dir = try a.dupe(u8, dir);
    var weights = try model_mod.loadWeightsForConfig(io, a, dir, &config, false);
    defer weights.deinit();
    var xfm = try Transformer.init(io, a, config, &weights);
    defer xfm.deinit();
    var tok = Tokenizer.initEmptyForTests(a);
    defer tok.deinit();
    // The scheduler's slot owns a cache with one entry per layer; the Transformer's own is a shell.
    // `initSlot` hands every slot the engine's default KV scheme, whatever the pack reports.
    var slot_cache = try KVCache.initWithConfig(a, config.num_hidden_layers, transformer_mod.KVQuantConfig.engine_default);
    defer slot_cache.deinit();
    if (config.swaRingTokens() > 0) slot_cache.setSwaRing(config.sliding_window);
    var off: usize = 0;
    var ctx = xfm.defaultCtx();
    ctx.cache = &slot_cache;
    ctx.moe_seq_offset = &off;
    var gen = try Generator.initWithOptions(io, a, &xfm, &tok, ids[0..30], 12, .{ .temperature = 0.0 }, &.{}, .{ .ctx = ctx, .dspark_enabled = true, .ssm_checkpoint_stride = 8, .ssm_checkpoint_max = 4 });
    defer gen.deinit(a);
    while (try gen.nextDspark(a)) |r| a.free(r.tokens);
    try testing.expect(slot_cache.step > 0);
    // the scheduler's commit: the generator's checkpoints and the whole token record, SSD-first with no RAM tier
    const cps = gen.takeSsmCheckpoints();
    var total: std.ArrayList(u32) = .empty;
    defer total.deinit(a);
    try total.appendSlice(a, ids[0..30]);
    try total.appendSlice(a, gen.generated_ids.items);
    try dsv41_prefix.markRows(&slot_cache);
    var snap = try slot_cache.snapshotRetained(xfm.s);
    defer snap.deinit();
    try testing.expectEqual(slot_cache.step, snap.step);
    try testing.expect(snap.entries[0].initialized);
    const prefix_cache = @import("../prefix_cache.zig");
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var buf: [512]u8 = undefined;
    const base = buf[0..try tmp.dir.realPath(testing.io, &buf)];
    var hc = prefix_cache.HotPrefixCache.initWithMem(a, 0, 0);
    defer hc.deinit();
    hc.ssd_first = true;
    hc.disk = try @import("../kv_disk_cache.zig").DiskTier.init(a, testing.io, base, "dsv41-live", 0, 128);
    hc.disk.?.ssd_first = true;
    hc.disk.?.armTestSpace(256 << 30, 1 << 40);
    hc.disk.?.enableBackgroundWriter();
    const st = try hc.commitWithRing(&slot_cache, total.items, false, 0, 0, null, if (cps.len > 0) cps else null, null, null, 30, .{});
    try testing.expect(st == .disk_only);
    try testing.expect(hc.pending_disk != null);
}

const TinyDsparkNet = struct {
    fx: model_mod.Weights,
    ids: []u32,
    config: model_mod.ModelConfig,
    weights: model_mod.Weights,

    fn init(a: std.mem.Allocator, io: std.Io) !TinyDsparkNet {
        const dsv41_model = @import("dsv41_model.zig");
        const dir = try dsv41_model.fixtureDir();
        var fx = try dsv41_model.loadFixture(dir, "fixture_noqat.safetensors");
        errdefer fx.deinit();
        var sc = @import("dsv41_scope.zig").Scope.init(a, mlx.gpuStream());
        defer sc.deinit();
        const ids = try dsv41_model.tokens(&sc, &fx);
        errdefer a.free(ids);
        var config = try model_mod.parseConfig(io, a, dir);
        config.expert_streaming = true;
        config.stream_mtp_head = true;
        config.expert_cache_bytes = 3 * config.num_hidden_layers * try @import("dsv41_stream.zig").expertBytes(&config.dsv41.?);
        config.expert_source_dir = try a.dupe(u8, dir);
        const weights = try model_mod.loadWeightsForConfig(io, a, dir, &config, false);
        return .{ .fx = fx, .ids = ids, .config = config, .weights = weights };
    }

    fn deinit(self: *TinyDsparkNet, a: std.mem.Allocator) void {
        self.weights.deinit();
        self.config.deinit(a);
        a.free(self.ids);
        self.fx.deinit();
    }
};

test "dsv41 DSpark: --max-mtp-ctx ends drafting once the context passes it (DSV41_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var st = try TinyDsparkNet.init(a, io);
    defer st.deinit(a);
    var xfm = try Transformer.init(io, a, st.config, &st.weights);
    defer xfm.deinit();
    var tok = Tokenizer.initEmptyForTests(a);
    defer tok.deinit();
    const prompt = st.ids[0..30];
    const saved = generate.max_mtp_ctx;
    defer generate.max_mtp_ctx = saved;
    // The prompt is admitted (30 <= 34); the ceiling is crossed a few tokens into the reply.
    generate.max_mtp_ctx = 34;
    var g = try Generator.initWithOptions(io, a, &xfm, &tok, prompt, 30, .{ .temperature = 0.0 }, &.{}, .{ .dspark_enabled = true });
    defer g.deinit(a);
    g.dspark_diag = .{ .rows = 3 };
    var drafted_past: u64 = 0;
    while (true) {
        const ctx_before = prompt.len + g.generated_ids.items.len;
        const drafted_before = g.dspark_stats.drafted;
        const r = (try g.nextDspark(a)) orelse break;
        a.free(r.tokens);
        if (ctx_before > generate.max_mtp_ctx) drafted_past += g.dspark_stats.drafted - drafted_before;
    }
    if (drafted_past > 0) {
        std.debug.print("DSpark drafted {d} rows past --max-mtp-ctx {d}\n", .{ drafted_past, generate.max_mtp_ctx });
        return error.DsparkIgnoresMaxMtpCtx;
    }
    try testing.expectEqual(SpecDisableReason.max_ctx, g.spec_disable_reason);
}

test "dsv41 DSpark: a drafted image-token id is cut from the verify chunk, since only a prompt may carry one (DSV41_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var st = try TinyDsparkNet.init(a, io);
    defer st.deinit(a);
    var xfm = try Transformer.init(io, a, st.config, &st.weights);
    defer xfm.deinit();
    var tok = Tokenizer.initEmptyForTests(a);
    defer tok.deinit();
    var g = try Generator.initWithOptions(io, a, &xfm, &tok, st.ids[0..30], 8, .{ .temperature = 0.0 }, &.{}, .{ .dspark_enabled = true });
    defer g.deinit(a);
    g.dspark_diag = .{ .rows = 4 };

    // The draft heads score the whole vocabulary, so the image id is one of their possible picks; this pack's
    // image id is made an id the draft already holds past row 1.
    const dsv = xfm.dsv41.?;
    var d = try xfm.dsv41Draft(g.next_token_id);
    defer d.deinit();
    const k = for (2..d.ids.len) |i| {
        if (d.ids[i] != d.ids[0] and d.ids[i] != d.ids[1]) break i;
    } else return error.NoDistinctDraft;
    dsv.cfg.image_token_id = d.ids[k];
    const round = g.nextDspark(a) catch |err| {
        std.debug.print("the verify chunk carrying the drafted image id failed: {s}\n", .{@errorName(err)});
        return err;
    };
    const r = round orelse return error.NoRound;
    defer a.free(r.tokens);
    try testing.expect(std.mem.indexOfScalar(u32, r.tokens, d.ids[k]) == null);
    try testing.expect(g.dspark_stats.drafted < k);
}

const grammar_test_schema =
    \\{"type":"object","properties":{"a":{"type":"boolean"},"b":{"type":"integer"}},"required":["a","b"],"additionalProperties":false}
;

/// JSON bytes cycled over the tiny vocabulary (id 0 pad), `closer` spelling the reasoning close.
fn grammarTestBytes(a: std.mem.Allocator, vocab: usize, closer: u32, eos: u32) !token_mask.TokenBytes {
    var arena = std.heap.ArenaAllocator.init(a);
    errdefer arena.deinit();
    const bytes = try arena.allocator().alloc(?[]const u8, vocab);
    const chars = "{}\":,abtruefals0123456789 -";
    for (bytes, 0..) |*b, i| b.* = if (i < 2) null else chars[(i - 2) % chars.len ..][0..1];
    bytes[closer] = rp_mod.BARE_THINK_CLOSER;
    bytes[eos] = null;
    return token_mask.TokenBytes.init(arena, bytes, eos);
}

const GrammarMode = union(enum) { serial, spark: ?DsparkDiag };

const GrammarOutcome = struct {
    tokens: std.ArrayList(u32) = .empty,
    /// Where the answer starts in the token stream, null when the reasoning never closed.
    payload_from: ?usize = null,
    finish: []const u8 = "",
    rounds: u64 = 0,
    drafted: u64 = 0,
    serial: u64 = 0,
    /// The answer's bytes, owned.
    payload: []u8 = &.{},

    fn deinit(self: *GrammarOutcome, a: std.mem.Allocator) void {
        self.tokens.deinit(a);
        a.free(self.payload);
    }
};

fn grammarRun(
    a: std.mem.Allocator,
    io: std.Io,
    xfm: *Transformer,
    tok: *Tokenizer,
    prompt: []const u32,
    want: u32,
    tb: *const token_mask.TokenBytes,
    closer: ?u32,
    eos: u32,
    params: SamplingParams,
    mode: GrammarMode,
) !GrammarOutcome {
    var parsed = try std.json.parseFromSlice(std.json.Value, a, grammar_test_schema, .{});
    defer parsed.deinit();
    var sc: SchemaConstraint = undefined;
    try sc.initFromValue(a, parsed.value, tb);
    defer sc.deinit();
    var proto = rp_mod.Protocol{ .kind = .bare_think };
    var candidates = [_]u32{closer orelse 0};
    if (closer != null) {
        try testing.expect(proto.setCloser(rp_mod.BARE_THINK_CLOSER));
        proto.closer_span_candidates = &candidates;
        try testing.expect(proto.setForced(&.{closer.?}));
        sc.deferWithProtocol(&proto);
    }
    sc.constraint.row_masks = true;
    var sampling = params;
    sampling.constraint = &sc.constraint;
    try xfm.resetCache();
    var gen = try Generator.initWithOptions(io, a, xfm, tok, prompt, want, sampling, &.{eos}, switch (mode) {
        .serial => .{ .skip_lazy_preforward = true },
        .spark => .{ .dspark_enabled = true },
    });
    defer gen.deinit(a);
    var out = GrammarOutcome{};
    errdefer out.deinit(a);
    switch (mode) {
        .serial => while (try gen.next(a)) |t| try out.tokens.append(a, t),
        .spark => |diag| {
            try testing.expect(gen.dspark);
            if (diag) |d| gen.dspark_diag = d;
            blocks: while (try gen.nextDspark(a)) |r| {
                defer a.free(r.tokens);
                for (r.tokens) |t| {
                    if (t == eos) break :blocks;
                    try out.tokens.append(a, t);
                }
            }
        },
    }
    out.finish = gen.finish_reason;
    out.rounds = gen.dspark_stats.rounds;
    out.drafted = gen.dspark_stats.drafted;
    out.serial = gen.dspark_stats.serial;
    var payload: std.ArrayList(u8) = .empty;
    errdefer payload.deinit(a);
    var lead: usize = 0;
    if (closer == null) {
        out.payload_from = 0;
    } else if (sc.constraint.pending_span) |span| {
        out.payload_from = span.token_index;
        lead = span.byte_offset;
    }
    if (out.payload_from) |from| for (out.tokens.items[@min(from, out.tokens.items.len)..]) |t| try payload.appendSlice(a, tb.bytes[t] orelse "");
    out.payload = try a.dupe(u8, payload.items[@min(lead, payload.items.len)..]);
    payload.deinit(a);
    return out;
}

/// A finished answer is JSON the schema accepts: both fields, nothing else.
fn expectAnswer(a: std.mem.Allocator, out: GrammarOutcome) !void {
    if (!std.mem.eql(u8, out.finish, "stop") or out.payload_from == null) return;
    const parsed = try std.json.parseFromSlice(std.json.Value, a, out.payload, .{});
    defer parsed.deinit();
    const obj = parsed.value.object;
    try testing.expect(obj.get("a").? == .bool and obj.get("b").? == .integer and obj.count() == 2);
}

test "dsv41 generator: DSpark rounds under a grammar emit serial decode's tokens, behind a reasoning boundary and from the first token (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    const dsv41_model = @import("dsv41_model.zig");
    const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var fx = try dsv41_model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    var sc = @import("dsv41_scope.zig").Scope.init(a, mlx.gpuStream());
    defer sc.deinit();
    const ids = try dsv41_model.tokens(&sc, &fx);
    defer a.free(ids);
    var config = try model_mod.parseConfig(io, a, dir);
    defer config.deinit(a);
    config.expert_streaming = true;
    config.stream_mtp_head = true;
    config.expert_cache_bytes = 64 << 20;
    config.expert_source_dir = try a.dupe(u8, dir);
    var weights = try model_mod.loadWeightsForConfig(io, a, dir, &config, false);
    defer weights.deinit();
    var xfm = try Transformer.init(io, a, config, &weights);
    defer xfm.deinit();
    try testing.expect(xfm.dsv41.?.dspark != null);
    var tok = Tokenizer.initEmptyForTests(a);
    defer tok.deinit();
    const greedy = SamplingParams{ .temperature = 0.0 };
    const prompt = ids[0..30];
    const want: u32 = 48;
    const vocab = xfm.dsv41.?.cfg.vocab;

    // The reasoning ends on a token the model picks a few steps in, so the boundary lands inside the request.
    var free: std.ArrayList(u32) = .empty;
    defer free.deinit(a);
    {
        var gen = try Generator.initWithOptions(io, a, &xfm, &tok, prompt, 12, greedy, &.{}, .{ .skip_lazy_preforward = true });
        defer gen.deinit(a);
        while (try gen.next(a)) |t| try free.append(a, t);
    }
    const closer: u32 = for (free.items[6..]) |t| {
        if (t >= 2) break t;
    } else 2;
    var tb = try grammarTestBytes(a, vocab, closer, 1);
    defer tb.deinit();

    // A one-row round is the serial step when the rows do not sum by width, so every token must match.
    dsv41_model.setDsparkExact(xfm.dsv41.?, false);
    for ([_]?u32{ closer, null }) |c| {
        var serial = try grammarRun(a, io, &xfm, &tok, prompt, want, &tb, c, 1, greedy, .serial);
        defer serial.deinit(a);
        var plain = try grammarRun(a, io, &xfm, &tok, prompt, want, &tb, c, 1, greedy, .{ .spark = .{ .rows = 1 } });
        defer plain.deinit(a);
        try testing.expect(serial.tokens.items.len > 8);
        try testing.expectEqualSlices(u32, serial.tokens.items, plain.tokens.items);
        try testing.expectEqualStrings(serial.finish, plain.finish);
        try testing.expect(plain.rounds > 0);
        try testing.expectEqual(serial.payload_from, plain.payload_from);
        try testing.expectEqualStrings(serial.payload, plain.payload);
        try expectAnswer(a, plain);
    }

    // Exact rounds give a row the same bytes at every width: any mix equals the one-row stream.
    dsv41_model.setDsparkExact(xfm.dsv41.?, true);
    const modes = [_]?DsparkDiag{ .{ .rows = 3 }, .{ .rows = 6 }, .sweep, null };
    for ([_]?u32{ closer, null }) |c| {
        var base = try grammarRun(a, io, &xfm, &tok, prompt, want, &tb, c, 1, greedy, .{ .spark = .{ .rows = 1 } });
        defer base.deinit(a);
        for (modes) |mode| {
            var got = try grammarRun(a, io, &xfm, &tok, prompt, want, &tb, c, 1, greedy, .{ .spark = mode });
            defer got.deinit(a);
            try testing.expectEqualSlices(u32, base.tokens.items, got.tokens.items);
            try testing.expectEqualStrings(base.payload, got.payload);
            try testing.expect(got.drafted > 0);
            try expectAnswer(a, got);
        }
    }

    // An EOS inside the reasoning forces the boundary instead of stopping, whichever round samples it.
    const eos_in_reasoning = free.items[3];
    try testing.expect(eos_in_reasoning >= 2 and eos_in_reasoning != closer);
    var forced_tb = try grammarTestBytes(a, vocab, closer, eos_in_reasoning);
    defer forced_tb.deinit();
    var base = try grammarRun(a, io, &xfm, &tok, prompt, want, &forced_tb, closer, eos_in_reasoning, greedy, .{ .spark = .{ .rows = 1 } });
    defer base.deinit(a);
    try testing.expect(base.payload_from != null);
    for (modes) |mode| {
        var got = try grammarRun(a, io, &xfm, &tok, prompt, want, &forced_tb, closer, eos_in_reasoning, greedy, .{ .spark = mode });
        defer got.deinit(a);
        try testing.expectEqualSlices(u32, base.tokens.items, got.tokens.items);
        try expectAnswer(a, got);
    }
    dsv41_model.setDsparkExact(xfm.dsv41.?, false);
    var serial = try grammarRun(a, io, &xfm, &tok, prompt, want, &forced_tb, closer, eos_in_reasoning, greedy, .serial);
    defer serial.deinit(a);
    var plain = try grammarRun(a, io, &xfm, &tok, prompt, want, &forced_tb, closer, eos_in_reasoning, greedy, .{ .spark = .{ .rows = 1 } });
    defer plain.deinit(a);
    try testing.expectEqualSlices(u32, serial.tokens.items, plain.tokens.items);
    dsv41_model.setDsparkExact(xfm.dsv41.?, true);

    // A sampled request keeps to the grammar and repeats at a forced width.
    const sampled = SamplingParams{ .temperature = 0.8, .seed = 7 };
    for ([_]?u32{ closer, null }) |c| {
        var first = try grammarRun(a, io, &xfm, &tok, prompt, want, &tb, c, 1, sampled, .{ .spark = .{ .rows = 3 } });
        defer first.deinit(a);
        var again = try grammarRun(a, io, &xfm, &tok, prompt, want, &tb, c, 1, sampled, .{ .spark = .{ .rows = 3 } });
        defer again.deinit(a);
        try testing.expectEqualSlices(u32, first.tokens.items, again.tokens.items);
        try expectAnswer(a, first);
    }
}

const WidthProbe = struct {
    const dsv41_model = @import("dsv41_model.zig");
    const A = mlx.mlx_array;

    /// One forward of `ids` as the server runs it: a verify round, or a plain chunk when `plain`.
    fn forward(xfm: *Transformer, ids: []const u32, plain: bool) !A {
        const m = xfm.dsv41.?;
        const engine = xfm.expert_stream.?;
        const st = try dsv41_model.ensureState(m);
        try @import("dsv41_borrow.zig").beforeForward(engine, &xfm.config, &m.cfg, ids.len, st.n + ids.len);
        engine.beginForward();
        const out = if (plain) try dsv41_model.extendAhead(m, ids, .last, dsv41_model.SPAN, null, &.{}) else try dsv41_model.verify(m, ids);
        engine.finishForward(ids.len);
        try mlx.check(mlx.mlx_array_eval(out));
        return out;
    }

    fn last(t: *const dsv41_model.Trace, name: []const u8) ?A {
        var i = t.items.items.len;
        while (i > 0) {
            i -= 1;
            if (std.mem.eql(u8, t.items.items[i].name, name)) return t.items.items[i].arr;
        }
        return null;
    }

    /// Row `row` of `a` (axis 0) as host f32.
    fn rowF32(a: std.mem.Allocator, x: A, row: usize) ![]f32 {
        const s = mlx.gpuStream();
        const sh = mlx.getShape(x);
        var start: [8]c_int = @splat(0);
        var stop: [8]c_int = undefined;
        const one: [8]c_int = @splat(1);
        @memcpy(stop[0..sh.len], sh);
        start[0] = @intCast(row);
        stop[0] = @intCast(row + 1);
        var sl = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(sl);
        try mlx.check(mlx.mlx_slice(&sl, x, &start, sh.len, &stop, sh.len, &one, sh.len, s));
        var f = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(f);
        try mlx.check(mlx.mlx_astype(&f, sl, .float32, s));
        try mlx.check(mlx.mlx_array_eval(f));
        const p = mlx.mlx_array_data_float32(f) orelse return error.HostRead;
        return a.dupe(f32, p[0..mlx.mlx_array_size(f)]);
    }

    /// Counts the elements of row `row` of `wide` that differ from row 0 of `narrow`; reports each name that differs.
    fn diffRow(a: std.mem.Allocator, wide: *const dsv41_model.Trace, narrow: *const dsv41_model.Trace, row: usize, verbose: bool) !usize {
        var differing: usize = 0;
        var seen: std.StringHashMapUnmanaged(void) = .empty;
        defer seen.deinit(a);
        for (wide.items.items) |it| {
            if (seen.contains(it.name) or std.mem.startsWith(u8, it.name, "ffn_")) continue;
            try seen.put(a, it.name, {});
            const w = last(wide, it.name).?;
            const n = last(narrow, it.name) orelse continue;
            const wr = try rowF32(a, w, row);
            defer a.free(wr);
            const nr = try rowF32(a, n, 0);
            defer a.free(nr);
            if (std.mem.startsWith(u8, it.name, "a_topk")) {
                // A wider block carries more picks, -1 past the row's own; packed picks index rows gathered per call,
                // so only which entries are -1 compares across widths.
                var same = wr.len >= nr.len;
                if (same) for (wr, 0..) |v, j| {
                    same = same and (v == -1) == (j >= nr.len or nr[j] == -1);
                };
                if (!same) {
                    differing += 1;
                    if (verbose) std.debug.print("  row {d} {s}: the picks differ (wide {d}, narrow {d})\n", .{ row, it.name, wr.len, nr.len });
                }
                continue;
            }
            if (wr.len != nr.len) return error.ShapeMismatch;
            var cnt: usize = 0;
            var maxd: f32 = 0;
            var maxv: f32 = 0;
            for (wr, nr) |x, y| {
                if (@as(u32, @bitCast(x)) != @as(u32, @bitCast(y))) cnt += 1;
                maxd = @max(maxd, @abs(x - y));
                maxv = @max(maxv, @abs(y));
            }
            if (cnt > 0) {
                differing += 1;
                if (verbose) std.debug.print("  row {d} {s}: {d}/{d} differ, max |d| {e:.3} (max |x| {e:.3})\n", .{ row, it.name, cnt, wr.len, maxd, maxv });
            }
        }
        return differing;
    }

    fn at(a: std.mem.Allocator, xfm: *Transformer, prompt: []const u32, cont: []const u32, W: usize, verbose: bool) !usize {
        const m = xfm.dsv41.?;
        // One-row rounds, each traced.
        dsv41_model.resetState(m);
        _ = mlx.mlx_array_free(try forward(xfm, prompt, true));
        var narrow: [16]dsv41_model.Trace = undefined;
        for (&narrow) |*t| t.* = .{ .gpa = a };
        defer for (&narrow) |*t| t.deinit();
        for (0..W + 1) |i| {
            dsv41_model.width_probe = &narrow[i];
            defer dsv41_model.width_probe = null;
            _ = mlx.mlx_array_free(try forward(xfm, cont[i .. i + 1], false));
            try dsv41_model.finishVerify(m, 0);
        }

        // The same tokens as one W-row verify that keeps every row, then a one-row round after it.
        dsv41_model.resetState(m);
        _ = mlx.mlx_array_free(try forward(xfm, prompt, true));
        var wide: dsv41_model.Trace = .{ .gpa = a };
        defer wide.deinit();
        var after: dsv41_model.Trace = .{ .gpa = a };
        defer after.deinit();
        {
            dsv41_model.width_probe = &wide;
            defer dsv41_model.width_probe = null;
            _ = mlx.mlx_array_free(try forward(xfm, cont[0..W], false));
            try dsv41_model.finishVerify(m, W - 1);
        }
        {
            dsv41_model.width_probe = &after;
            defer dsv41_model.width_probe = null;
            _ = mlx.mlx_array_free(try forward(xfm, cont[W .. W + 1], false));
            try dsv41_model.finishVerify(m, 0);
        }
        var bad: usize = 0;
        for (0..W) |r| bad += try diffRow(a, &wide, &narrow[r], r, verbose);
        if (verbose) std.debug.print("  -- the one-row round after the wide verify\n", .{});
        bad += try diffRow(a, &after, &narrow[W], 0, verbose);

        // A W-row verify that keeps two rows (its drafts after the first are wrong), then one-row rounds over the rest.
        dsv41_model.resetState(m);
        _ = mlx.mlx_array_free(try forward(xfm, prompt, true));
        {
            var ids: [16]u32 = undefined;
            @memcpy(ids[0..W], cont[0..W]);
            for (2..W) |k| ids[k] = (cont[k] + 7) % 1000 + 1000;
            var t: dsv41_model.Trace = .{ .gpa = a };
            defer t.deinit();
            dsv41_model.width_probe = &t;
            _ = mlx.mlx_array_free(try forward(xfm, ids[0..W], false));
            dsv41_model.width_probe = null;
            try dsv41_model.finishVerify(m, 1);
            if (verbose) std.debug.print("  -- rejected verify: kept rows\n", .{});
            for (0..2) |r| bad += try diffRow(a, &t, &narrow[r], r, verbose);
        }
        for (2..W + 1) |i| {
            var t: dsv41_model.Trace = .{ .gpa = a };
            defer t.deinit();
            dsv41_model.width_probe = &t;
            _ = mlx.mlx_array_free(try forward(xfm, cont[i .. i + 1], false));
            dsv41_model.width_probe = null;
            try dsv41_model.finishVerify(m, 0);
            if (verbose) std.debug.print("  -- one-row round {d} after the rollback\n", .{i});
            bad += try diffRow(a, &t, &narrow[i], 0, verbose);
        }

        // One-row rounds with a draft pass before each, as a drafted round runs it.
        dsv41_model.resetState(m);
        _ = mlx.mlx_array_free(try forward(xfm, prompt, true));
        for (0..W + 1) |i| {
            var d = try dsv41_model.draftHeads(m, cont[i], .conf);
            d.deinit();
            var t: dsv41_model.Trace = .{ .gpa = a };
            defer t.deinit();
            dsv41_model.width_probe = &t;
            _ = mlx.mlx_array_free(try forward(xfm, cont[i .. i + 1], false));
            dsv41_model.width_probe = null;
            try dsv41_model.finishVerify(m, 0);
            if (verbose) std.debug.print("  -- one-row round {d} behind a draft\n", .{i});
            bad += try diffRow(a, &t, &narrow[i], 0, verbose);
        }
        return bad;
    }
};

test "dsv41 DSpark: on the real pack each row of a wide verify is the bits of its own one-row round (SUSHI_DSV41_REAL_PACK)" {
    try mlx.requireGpu();
    const dir = std.mem.span(std.c.getenv("SUSHI_DSV41_REAL_PACK") orelse return error.SkipZigTest);
    const W: usize = if (std.c.getenv("SUSHI_DSV41_PROBE_WIDTH")) |v| try std.fmt.parseInt(usize, std.mem.span(v), 10) else 4;
    const verbose = std.c.getenv("SUSHI_DSV41_PROBE_VERBOSE") != null;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var config = try model_mod.parseConfig(io, a, dir);
    defer config.deinit(a);
    config.stream_mtp_head = true;
    config.has_vision = false;
    const budget: u64 = 40 << 30;
    const model_load = @import("../model_load.zig");
    const plan = try model_load.planExpertStreamingWith(io, a, &config, dir, budget, false, 0);
    model_load.applyStreamingPlan(&config, plan, budget);
    config.expert_source_dir = try a.dupe(u8, dir);
    var weights = try model_mod.loadWeightsForConfig(io, a, dir, &config, false);
    defer weights.deinit();
    var xfm = try Transformer.init(io, a, config, &weights);
    defer xfm.deinit();
    const m = xfm.dsv41.?;
    try testing.expect(m.dspark != null);
    var tok = try tokenizer_mod.loadTokenizer(io, a, dir);
    defer tok.deinit();
    var text_buf: []u8 = &.{};
    defer a.free(text_buf);
    const text: []const u8 = if (std.c.getenv("SUSHI_DSV41_PROBE_TEXT")) |path| blk: {
        text_buf = try std.Io.Dir.cwd().readFileAlloc(io, std.mem.span(path), a, .limited(1 << 20));
        break :blk text_buf;
    } else "Write a long, detailed story about a lighthouse keeper and the storms of one winter. The keeper climbed the iron stairs each night, counting the steps as the wind rose over the black water and the lamp turned.";
    const all = try tok.encode(a, text);
    defer a.free(all);
    var at_it = std.mem.tokenizeScalar(u8, if (std.c.getenv("SUSHI_DSV41_PROBE_AT")) |v| std.mem.span(v) else "17", ',');
    var bad: usize = 0;
    while (at_it.next()) |at_s| {
        const P = try std.fmt.parseInt(usize, at_s, 10);
        if (P + W + 1 > all.len) return error.ProbeTextTooShort;
        if (verbose) std.debug.print("== context {d}, width {d}\n", .{ P, W });
        bad += try WidthProbe.at(a, &xfm, all[0..P], all[P..], W, verbose);
    }
    try testing.expectEqual(@as(usize, 0), bad);
}
