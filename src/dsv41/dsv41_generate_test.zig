const std = @import("std");
const dsv41_state = @import("dsv41_state.zig");
const settings_resolve = @import("../settings_resolve.zig");
const mlx = @import("../mlx.zig");
const trunk_mod = @import("../trunk.zig");
const tokenizer_mod = @import("../tokenizer.zig");
const model_mod = @import("../model.zig");
const token_mask = @import("../token_mask.zig");
const rp_mod = @import("../reasoning_protocol.zig");
const depth_bounds = @import("../mtp_depth_bounds.zig");
const Trunk = trunk_mod.Trunk;
const Tokenizer = tokenizer_mod.Tokenizer;
const SSMCheckpoint = trunk_mod.SSMCheckpoint;
const mlx_ops = @import("../mlx_ops.zig");
const dsv41_prefix = @import("dsv41_prefix.zig");
const dsv41_serving = @import("dsv41_serving.zig");
const dsv41_borrow = @import("dsv41_borrow.zig");
const dsv41_config = @import("dsv41_config.zig");
const KVCache = trunk_mod.KVCache;
const test_support = @import("../test_support.zig");
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

test "SUSHI_DSV41_DSPARK_ROWS never forces a verify wider than dsv41VerifyImpl accepts" {
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
    var sc = @import("dsv41_scope.zig").Scope.plain(a, mlx.gpuStream());
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
    var xfm = try Trunk.init(io, a, config, &weights);
    defer xfm.deinit();
    try testing.expect(dsv41_state.modelOf(&xfm).?.dspark != null and xfm.moduleStateSpecRollback());
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

test "dsv41 generator: copy rounds keep greedy text the serial decode's, whole or partly wrong copies alike (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    const dsv41_model = @import("dsv41_model.zig");
    const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var fx = try dsv41_model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    var sc = @import("dsv41_scope.zig").Scope.plain(a, mlx.gpuStream());
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
    var xfm = try Trunk.init(io, a, config, &weights);
    defer xfm.deinit();
    var tok = Tokenizer.initEmptyForTests(a);
    defer tok.deinit();
    const greedy = SamplingParams{ .temperature = 0.0 };
    const prompt = ids[0..30];
    const want: u32 = 120;

    // One row a round is the serial decode, which exact DSpark rounds reproduce at every width.
    var serial: std.ArrayList(u32) = .empty;
    defer serial.deinit(a);
    {
        try xfm.resetCache();
        var gen = try Generator.initWithOptions(io, a, &xfm, &tok, prompt, want, greedy, &.{}, .{ .dspark_enabled = true });
        defer gen.deinit(a);
        gen.dspark_diag = .{ .rows = 1 };
        while (try gen.nextDspark(a)) |r| {
            defer a.free(r.tokens);
            try serial.appendSlice(a, r.tokens);
        }
    }
    try testing.expectEqual(@as(usize, want), serial.items.len);

    // Copy rounds keep to `dsv41_exact_verify_rows`, the widest verify this fixture matches serial decode at.
    // The lookup context is the prompt, the output the model is about to write, and the prompt again: the generated
    // tail then matches the output's start, and the copy is the model's own continuation. `wrong_every` spoils drafts.
    for ([_]usize{ 0, 11 }) |wrong_every| {
        const ctx = try a.alloc(u32, prompt.len * 2 + serial.items.len);
        defer a.free(ctx);
        @memcpy(ctx[0..prompt.len], prompt);
        @memcpy(ctx[prompt.len..][0..serial.items.len], serial.items);
        if (wrong_every > 0) {
            var i: usize = wrong_every;
            while (i < serial.items.len) : (i += wrong_every) ctx[prompt.len + i] +%= 1;
        }
        @memcpy(ctx[prompt.len + serial.items.len ..], prompt);
        for ([_]bool{ false, true }) |copy| {
            try xfm.resetCache();
            var gen = try Generator.initWithOptions(io, a, &xfm, &tok, prompt, want, greedy, &.{}, .{ .dspark_enabled = true, .lookup_prompt = ctx });
            defer gen.deinit(a);
            gen.dspark_copy = copy;
            var got: std.ArrayList(u32) = .empty;
            defer got.deinit(a);
            while (try gen.nextDspark(a)) |r| {
                defer a.free(r.tokens);
                try got.appendSlice(a, r.tokens);
            }
            try testing.expectEqualSlices(u32, serial.items, got.items);
            const st = gen.dspark_stats;
            try testing.expectEqual(copy, st.copy_rounds > 0);
            if (copy) {
                try testing.expect(st.copy_accepted <= st.copy_drafted);
                try testing.expect(wrong_every > 0 or st.copy_accepted == st.copy_drafted);
                try testing.expect(wrong_every == 0 or st.copy_accepted < st.copy_drafted);
            }
        }
    }
}

test "dsv41 generator: with exact DSpark rounds greedy text is the same at every round-width mix (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    const dsv41_model = @import("dsv41_model.zig");
    const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var fx = try dsv41_model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    var sc = @import("dsv41_scope.zig").Scope.plain(a, mlx.gpuStream());
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
    var xfm = try Trunk.init(io, a, config, &weights);
    defer xfm.deinit();
    try testing.expect(dsv41_state.modelOf(&xfm).?.dspark != null);
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
    dsv41_model.setDsparkExact(dsv41_state.modelOf(&xfm).?, false);
    try xfm.resetCache();
    var fast = try Generator.initWithOptions(io, a, &xfm, &tok, prompt, want, greedy, &.{}, .{ .dspark_enabled = true });
    defer fast.deinit(a);
    var n: usize = 0;
    while (try fast.nextDspark(a)) |r| {
        defer a.free(r.tokens);
        for (r.tokens) |t| try testing.expect(t < dsv41_state.modelOf(&xfm).?.cfg.vocab);
        n += r.tokens.len;
    }
    try testing.expectEqual(@as(usize, want), n);
}

test "dsv41 engine: DSpark rounds on the engine of our own say MLX's tokens at every round-width mix (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    const dsv41_model = @import("dsv41_model.zig");
    const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was);
    var fx = try dsv41_model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    var sc = @import("dsv41_scope.zig").Scope.plain(a, mlx.gpuStream());
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
    var xfm = try Trunk.init(io, a, config, &weights);
    defer xfm.deinit();
    const m = dsv41_state.modelOf(&xfm).?;
    try testing.expect(m.dspark != null);
    var tok = Tokenizer.initEmptyForTests(a);
    defer tok.deinit();
    const prompt = ids[0..30];
    const want: u32 = 60;
    var first: std.ArrayList(u32) = .empty;
    defer first.deinit(a);
    // MLX alone, then the engine's drafts and verifies under rounds of one row, of three rows, a cycle through every
    // width, and the adaptive policy. The prompt is MLX's either way, so each request crosses once.
    const Arm = struct { engine: bool, mode: ?DsparkDiag };
    const arms = [_]Arm{ .{ .engine = false, .mode = .{ .rows = 1 } }, .{ .engine = true, .mode = .{ .rows = 1 } }, .{ .engine = true, .mode = .{ .rows = 3 } }, .{ .engine = true, .mode = .sweep }, .{ .engine = true, .mode = null } };
    for (arms, 0..) |arm, i| {
        m.engine_decode = arm.engine;
        defer m.engine_decode = false;
        const before = mlx.engine.stats();
        try xfm.resetCache();
        var gen = try Generator.initWithOptions(io, a, &xfm, &tok, prompt, want, .{ .temperature = 0.0 }, &.{}, .{ .dspark_enabled = true });
        defer gen.deinit(a);
        if (arm.mode) |d| gen.dspark_diag = d;
        var got: std.ArrayList(u32) = .empty;
        defer got.deinit(a);
        while (try gen.nextDspark(a)) |r| {
            defer a.free(r.tokens);
            try got.appendSlice(a, r.tokens);
        }
        try testing.expectEqual(@as(usize, want), got.items.len);
        if (i == 0) try first.appendSlice(a, got.items) else try testing.expectEqualSlices(u32, first.items, got.items);
        if (i == 1) try testing.expect(mlx.engine.stats().dispatches > before.dispatches and mlx.engine.stats().lends > before.lends);
    }
}

/// The first word of the bits MLX would draw from its next key.
fn mlxNextWord() !u32 {
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    var bits = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(bits);
    try mlx.check(mlx.mlx_random_bits(&bits, &[_]c_int{1}, 1, 4, .{}, s));
    try mlx.check(mlx.mlx_array_eval(bits));
    return (mlx.mlx_array_data_uint32(bits) orelse return error.MlxArrayDataNull)[0];
}

/// One sampled DSpark request over `prompt`, at `mode`; on the engine no round lends to MLX, and the request leaves
/// MLX's key sequence where it was.
fn sampledRun(a: std.mem.Allocator, io: std.Io, xfm: *Trunk, tok: *Tokenizer, prompt: []const u32, want: u32, sampled: SamplingParams, mode: ?DsparkDiag, mask: ?mlx.mlx_array, on_engine: bool) !std.ArrayList(u32) {
    const m = dsv41_state.modelOf(xfm).?;
    m.engine_decode = on_engine;
    defer m.engine_decode = false;
    try xfm.resetCache();
    var gen = try Generator.initWithOptions(io, a, xfm, tok, prompt, want, sampled, &.{}, .{ .dspark_enabled = true });
    defer gen.deinit(a);
    gen.sampling.suppress_mask = mask;
    if (mode) |d| gen.dspark_diag = d;
    var got: std.ArrayList(u32) = .empty;
    errdefer got.deinit(a);
    _ = mlx.mlx_random_seed(99);
    while (true) {
        const before = mlx.engine.stats();
        const r = (try gen.nextDspark(a)) orelse break;
        defer a.free(r.tokens);
        if (on_engine) try testing.expectEqual(before.lends, mlx.engine.stats().lends);
        try got.appendSlice(a, r.tokens);
    }
    try testing.expectEqual(@as(usize, want), got.items.len);
    if (on_engine) {
        const after = try mlxNextWord();
        _ = mlx.mlx_random_seed(99);
        try testing.expectEqual(try mlxNextWord(), after);
    }
    return got;
}

test "dsv41 engine: a sampled DSpark request on the engine of our own repeats at every round-width mix, a seed or a root of its own gives other tokens, and no round touches MLX (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    const dsv41_model = @import("dsv41_model.zig");
    const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was);
    var fx = try dsv41_model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    var sc = @import("dsv41_scope.zig").Scope.plain(a, mlx.gpuStream());
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
    var xfm = try Trunk.init(io, a, config, &weights);
    defer xfm.deinit();
    const m = dsv41_state.modelOf(&xfm).?;
    try testing.expect(m.dspark != null);
    var tok = Tokenizer.initEmptyForTests(a);
    defer tok.deinit();
    const prompt = ids[0..30];
    const want: u32 = 60;
    const modes = [_]?DsparkDiag{ .{ .rows = 1 }, .{ .rows = 3 }, .sweep, null };
    // Every seventh id suppressed: the mask is the model's own array, which the engine imports.
    const vocab: usize = m.cfg.vocab;
    const flags = try a.alloc(bool, vocab);
    defer a.free(flags);
    for (flags, 0..) |*f, id| f.* = id % 7 == 0;
    const mask = mlx.mlx_array_new_data(flags.ptr, &[_]c_int{@intCast(vocab)}, 1, .bool_);
    defer _ = mlx.mlx_array_free(mask);

    // A seeded draw is keyed by its position, so every round mix says the same tokens, on each route by its own keys.
    var engine_ref: [2]std.ArrayList(u32) = .{ .empty, .empty };
    var mlx_ref: [2]std.ArrayList(u32) = .{ .empty, .empty };
    defer for (&engine_ref) |*r| r.deinit(a);
    defer for (&mlx_ref) |*r| r.deinit(a);
    const sampled = SamplingParams{ .temperature = 0.8, .top_k = 40, .top_p = 0.95, .seed = 7 };
    for ([_]bool{ false, true }) |masked| for (modes) |mode| for ([_]bool{ true, false }) |on_engine| {
        var got = try sampledRun(a, io, &xfm, &tok, prompt, want, sampled, mode, if (masked) mask else null, on_engine);
        defer got.deinit(a);
        const ref = &(if (on_engine) engine_ref else mlx_ref)[@intFromBool(masked)];
        if (ref.items.len == 0) try ref.appendSlice(a, got.items) else try testing.expectEqualSlices(u32, ref.items, got.items);
    };
    // A request that stops sooner is a prefix: how a stream ends moves no draw.
    var short = try sampledRun(a, io, &xfm, &tok, prompt, 37, sampled, .sweep, null, true);
    defer short.deinit(a);
    try testing.expectEqualSlices(u32, engine_ref[0].items[0..37], short.items);

    // Another seed, and two requests with none, draw other tokens (the first token is the serial sampler's, on MLX).
    var other = try sampledRun(a, io, &xfm, &tok, prompt, want, .{ .temperature = 0.8, .top_k = 40, .top_p = 0.95, .seed = 8 }, .{ .rows = 3 }, null, true);
    defer other.deinit(a);
    try testing.expect(!std.mem.eql(u32, engine_ref[0].items[1..], other.items[1..]));
    var unseeded: [2]std.ArrayList(u32) = .{ .empty, .empty };
    defer for (&unseeded) |*r| r.deinit(a);
    for (&unseeded) |*r| r.* = try sampledRun(a, io, &xfm, &tok, prompt, want, .{ .temperature = 0.8, .top_k = 40, .top_p = 0.95 }, .{ .rows = 3 }, null, true);
    try testing.expect(!std.mem.eql(u32, unseeded[0].items[1..], unseeded[1].items[1..]));
    try testing.expect(!std.mem.eql(u32, engine_ref[0].items[1..], unseeded[0].items[1..]));

    // The verdict itself puts no op on MLX and lends nothing, seeded or not, with a mask or without.
    m.engine_decode = true;
    defer m.engine_decode = false;
    const rows = 4;
    var host: [rows * 64]f32 = undefined;
    for (&host, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i * 2654435761) % 1000)) / 250.0;
    const logits = mlx.mlx_array_new_data(&host, &[_]c_int{ 1, rows, 64 }, 3, .float32);
    defer _ = mlx.mlx_array_free(logits);
    const drafts = [_]u32{ 3, 9, 20 };
    for ([_]?u64{ 7, null }) |seed| {
        try xfm.resetCache();
        var gen = try Generator.initWithOptions(io, a, &xfm, &tok, prompt, want, .{ .temperature = 0.8, .top_k = 40, .top_p = 0.95, .seed = seed }, &.{}, .{ .dspark_enabled = true });
        defer gen.deinit(a);
        const before = mlx.engine.stats();
        const ops = mlx.engine.mlx_ops;
        const verdict = try dsv41_dspark.dsparkSampledVerdict(&gen, logits, &drafts);
        try testing.expect(verdict.accepted <= drafts.len);
        try testing.expectEqual(ops, mlx.engine.mlx_ops);
        try testing.expectEqual(before.lends, mlx.engine.stats().lends);
    }
}

test "dsv41 generator: a seeded sampled request verifies at one width whatever the cost model says (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    const dsv41_model = @import("dsv41_model.zig");
    const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var fx = try dsv41_model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    var sc = @import("dsv41_scope.zig").Scope.plain(a, mlx.gpuStream());
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
    var xfm = try Trunk.init(io, a, config, &weights);
    defer xfm.deinit();
    var tok = Tokenizer.initEmptyForTests(a);
    defer tok.deinit();
    const prompt = ids[0..30];
    const want: u32 = 60;
    dsv41_model.setDsparkExact(dsv41_state.modelOf(&xfm).?, false);

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
    var sc = @import("dsv41_scope.zig").Scope.plain(a, mlx.gpuStream());
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
    var xfm = try Trunk.init(io, a, config, &weights);
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
                var cp = try dsv41_prefix.capture(a, dsv41_state.modelOf(&xfm).?, prompt.len + total);
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
    var sc = @import("dsv41_scope.zig").Scope.plain(a, mlx.gpuStream());
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
    var xfm = try Trunk.init(io, a, config, &weights);
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
        var cache = try KVCache.initWithConfig(a, config.num_hidden_layers, trunk_mod.KVQuantConfig.engine_default);
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
            want_logits = try dsv41_model.extend(dsv41_state.modelOf(&xfm).?, &suffix, .all);
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
        var cache = try KVCache.initWithConfig(a, config.num_hidden_layers, trunk_mod.KVQuantConfig.engine_default);
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
            try dsv41_prefix.writeDecodePointDsv41(&hc, &cache, dsv41_state.modelOf(&xfm).?, next[0..point], false, .{});
            written = true;
        }
        try testing.expect(written);
        // The end-of-request commit: the generator's own points (none here) and the whole token record.
        try dsv41_prefix.markRows(&cache, mlx.gpuStream());
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
    var target = try KVCache.initWithConfig(a, config.num_hidden_layers, trunk_mod.KVQuantConfig.engine_default);
    defer target.deinit();
    var moe: usize = 0;
    const res = try dsv41_prefix.lookupAndRestoreDsv41(&hc, &target, &moe, dsv41_state.modelOf(&xfm).?, s, next, false, 0, null, &.{}, 9, false);
    try testing.expectEqual(point, res.matched);
    try testing.expect(hc.last_restored_disk_id != null);
    const got = try dsv41_model.extend(dsv41_state.modelOf(&xfm).?, &suffix, .all);
    defer _ = mlx.mlx_array_free(got);
    try mlx_ops.expectArrayBits(want_logits.?, got);
}

test "dsv41 generator: DSpark requests leave the MLX active memory flat (DSV41_TINY)" {
    try mlx.requireGpu();
    const dsv41_model = @import("dsv41_model.zig");
    const dir = try dsv41_model.fixtureDir();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var fx = try dsv41_model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    var sc = @import("dsv41_scope.zig").Scope.plain(a, mlx.gpuStream());
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
    var xfm = try Trunk.init(io, a, config, &weights);
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
    var sc = @import("dsv41_scope.zig").Scope.plain(a, mlx.gpuStream());
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
    var xfm = try Trunk.init(io, a, config, &weights);
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

fn forwardIds(xfm: *Trunk, ctx: *trunk_mod.ForwardCtx, ids: []const u32) !mlx.mlx_array {
    const a = testing.allocator;
    const ids32 = try a.alloc(i32, ids.len);
    defer a.free(ids32);
    for (ids, ids32) |t, *o| o.* = @intCast(t);
    const shape = [_]c_int{ 1, @intCast(ids.len) };
    // On the side the forward's output is on, as the generator makes its token arrays.
    const host = mlx.engine.enter(xfm.outStream());
    const arr = mlx.mlx_array_new_data(ids32.ptr, &shape, 2, .int32);
    host.end();
    defer _ = mlx.mlx_array_free(arr);
    return xfm.forwardWith(ctx, arr);
}

fn argmaxHost(xfm: *Trunk, logits: mlx.mlx_array) !u32 {
    var am = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(am);
    try mlx.check(mlx.mlx_argmax_axis(&am, logits, -1, false, xfm.outStream()));
    try mlx.check(mlx.mlx_array_eval(am));
    var v: i32 = 0;
    try mlx.check(mlx.mlx_array_item_int32(&v, am));
    return @intCast(v);
}

/// Greedy decode by hand: each step's token is evaluated from the logits before anything reads their bytes, as the
/// Generator does; with `lazy` the forward may hand the logits back unevaluated.
fn lazyHeadSteps(a: std.mem.Allocator, xfm: *Trunk, prompt: []const u32, steps: usize, lazy: bool) !LazyHeadRun {
    var run: LazyHeadRun = .{};
    errdefer run.deinit(a);
    try xfm.resetCache();
    var ctx = xfm.defaultCtx();
    var lg = try forwardIds(xfm, &ctx, prompt);
    var tok = try argmaxHost(xfm, lg);
    _ = mlx.mlx_array_free(lg);
    try run.tokens.append(a, tok);
    const vocab = dsv41_state.modelOf(xfm).?.cfg.vocab;
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
fn lazyIdSteps(a: std.mem.Allocator, xfm: *Trunk, prompt: []const u32, steps: usize, queue: bool) !LazyHeadRun {
    var run: LazyHeadRun = .{};
    errdefer run.deinit(a);
    const m = dsv41_state.modelOf(xfm).?;
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
fn droppedStartLeavesNothing(a: std.mem.Allocator, xfm: *Trunk, prompt: []const u32, token: u32) !void {
    const m = dsv41_state.modelOf(xfm).?;
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
    var sc = @import("dsv41_scope.zig").Scope.plain(a, mlx.gpuStream());
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
    var xfm = try Trunk.init(io, a, config, &weights);
    defer xfm.deinit();
    const m = dsv41_state.modelOf(&xfm).?;
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

/// Greedy one-row decode on the engine of our own against the same steps on MLX.
fn engineDecodeProof(dir: []const u8, stq: bool) !void {
    try mlx.requireGpu();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const dsv41_model = @import("dsv41_model.zig");
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was);
    var fx = try dsv41_model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    var sc = @import("dsv41_scope.zig").Scope.plain(a, mlx.gpuStream());
    defer sc.deinit();
    const ids = try dsv41_model.tokens(&sc, &fx);
    defer a.free(ids);
    var config = try model_mod.parseConfig(io, a, dir);
    defer config.deinit(a);
    config.expert_streaming = true;
    config.expert_cache_bytes = if (stq) 64 << 20 else 3 * config.num_hidden_layers * try @import("dsv41_stream.zig").expertBytes(&config.dsv41.?);
    config.expert_source_dir = try a.dupe(u8, dir);
    var weights = try model_mod.loadWeightsForConfig(io, a, dir, &config, false);
    defer weights.deinit();
    var xfm = try Trunk.init(io, a, config, &weights);
    defer xfm.deinit();
    const m = dsv41_state.modelOf(&xfm).?;
    const prompt = ids[0..30];
    const steps: usize = 24;
    var want = try lazyHeadSteps(a, &xfm, prompt, steps, false);
    defer want.deinit(a);
    m.engine_decode = true;
    defer m.engine_decode = false;
    const before = mlx.engine.stats();
    var got = try lazyHeadSteps(a, &xfm, prompt, steps, false);
    defer got.deinit(a);
    try testing.expect(mlx.engine.stats().dispatches > before.dispatches);
    try testing.expectEqualSlices(u32, want.tokens.items, got.tokens.items);
    try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(want.logits.items), std.mem.sliceAsBytes(got.logits.items));
}

/// Every byte `rounds` DSpark rounds hand the generator: each draft's ids and confidences, then the logits of its
/// verify, whose width walks from one row to the widest verify and around.
fn dsparkRoundBytes(a: std.mem.Allocator, xfm: *Trunk, prompt: []const u32, rounds: usize) !std.ArrayList(u8) {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    const vocab = dsv41_state.modelOf(xfm).?.cfg.vocab;
    try xfm.resetCache();
    var ctx = xfm.defaultCtx();
    const first = try forwardIds(xfm, &ctx, prompt);
    var tok = try argmaxHost(xfm, first);
    _ = mlx.mlx_array_free(first);
    for (0..rounds) |r| {
        var d = try dsv41_serving.dsv41DraftImpl(xfm, tok);
        defer d.deinit();
        try out.appendSlice(a, std.mem.sliceAsBytes(d.ids));
        try out.appendSlice(a, std.mem.sliceAsBytes(d.conf));
        const rows = 1 + r % @min(d.ids.len, dsv41_max_verify_rows);
        const logits = try dsv41_serving.dsv41VerifyImpl(xfm, &ctx, d.ids[0..rows]);
        defer _ = mlx.mlx_array_free(logits);
        try mlx.check(mlx.mlx_array_eval(logits));
        const data = (mlx.mlx_array_data_float32(logits) orelse return error.NoLogits)[0 .. rows * vocab];
        try out.appendSlice(a, std.mem.sliceAsBytes(data));
        var accepted: usize = 0;
        while (accepted + 1 < rows and std.mem.indexOfMax(f32, data[accepted * vocab ..][0..vocab]) == d.ids[1 + accepted]) accepted += 1;
        tok = @intCast(std.mem.indexOfMax(f32, data[accepted * vocab ..][0..vocab]));
        try dsv41_serving.dsv41FinishImpl(xfm, &ctx, accepted);
    }
    return out;
}

test "dsv41 engine: a DSpark round on the engine gives MLX's draft, confidences and verify logits bit for bit at every width (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    const dsv41_model = @import("dsv41_model.zig");
    const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was);
    var fx = try dsv41_model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    var sc = @import("dsv41_scope.zig").Scope.plain(a, mlx.gpuStream());
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
    var xfm = try Trunk.init(io, a, config, &weights);
    defer xfm.deinit();
    const m = dsv41_state.modelOf(&xfm).?;
    try testing.expect(m.dspark != null);
    const rounds = 2 * @as(usize, dsv41_max_verify_rows);
    var want = try dsparkRoundBytes(a, &xfm, ids[0..30], rounds);
    defer want.deinit(a);
    m.engine_decode = true;
    defer m.engine_decode = false;
    const before = mlx.engine.stats();
    var got = try dsparkRoundBytes(a, &xfm, ids[0..30], rounds);
    defer got.deinit(a);
    try testing.expect(mlx.engine.stats().dispatches > before.dispatches);
    try testing.expectEqualSlices(u8, want.items, got.items);
}

test "dsv41 engine: a round that finds MLX's queue unused for a warm tick runs one MLX op on it first, and none sooner (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    const dsv41_model = @import("dsv41_model.zig");
    const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was);
    var fx = try dsv41_model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    var sc = @import("dsv41_scope.zig").Scope.plain(a, mlx.gpuStream());
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
    var xfm = try Trunk.init(io, a, config, &weights);
    defer xfm.deinit();
    const m = dsv41_state.modelOf(&xfm).?;

    const tick = @import("../sched_policy.zig").GPU_WARM_TICK_NS;
    var before: usize = 0;
    var after: usize = 0;
    const t0 = mlx.engine.nowNs();
    m.mlx_queue_used_ns = t0 - tick;
    try mlx.check(mlx.mlx_metal_command_buffer_count(&before));
    dsv41_serving.keepMlxQueue(m);
    try mlx.check(mlx.mlx_metal_command_buffer_count(&after));
    try testing.expectEqual(before + 1, after);
    try testing.expect(m.mlx_queue_used_ns >= t0);
    const kept = m.mlx_queue_used_ns;
    dsv41_serving.keepMlxQueue(m);
    try mlx.check(mlx.mlx_metal_command_buffer_count(&before));
    try testing.expectEqual(after, before);
    try testing.expectEqual(kept, m.mlx_queue_used_ns);

    // A round on the engine asks; a round on MLX uses the queue itself.
    m.mlx_queue_used_ns = 0;
    var on_mlx = try dsparkRoundBytes(a, &xfm, ids[0..30], 1);
    defer on_mlx.deinit(a);
    try testing.expectEqual(@as(u64, 0), m.mlx_queue_used_ns);
    m.engine_decode = true;
    defer m.engine_decode = false;
    var on_engine = try dsparkRoundBytes(a, &xfm, ids[0..30], 1);
    defer on_engine.deinit(a);
    try testing.expect(m.mlx_queue_used_ns >= t0);
}

test "dsv41 engine: the load clock times every phase of a load and the phases add up to its total (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    const load_clock = @import("../load_clock.zig");
    const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was);
    const own_was = dsv41_borrow.own_override;
    dsv41_borrow.own_override = true;
    defer dsv41_borrow.own_override = own_was;
    var config = try model_mod.parseConfig(io, a, dir);
    defer config.deinit(a);
    config.expert_streaming = true;
    config.stream_mtp_head = true;
    config.expert_cache_bytes = 64 << 20;
    config.expert_source_dir = try a.dupe(u8, dir);
    const made_before = mlx.engine.pipelinesMade();
    load_clock.begin();
    var weights = try model_mod.loadWeightsForConfig(io, a, dir, &config, false);
    defer weights.deinit();
    const build_t0 = load_clock.now();
    var xfm = try Trunk.init(io, a, config, &weights);
    defer xfm.deinit();
    load_clock.buildDone(build_t0);
    const wired_t0 = load_clock.now();
    _ = mlx.applyWiredPolicy();
    load_clock.wiredDone(wired_t0);
    const cache_t0 = load_clock.now();
    try xfm.expert_stream.?.warmCache();
    load_clock.cacheDone(cache_t0);
    load_clock.end();
    const p = load_clock.load;
    try testing.expect(p.headers_ns > 0 and p.read_ns > 0 and p.read_bytes > 0);
    try testing.expect(p.build_ns > 0 and p.warm_ns > 0 and p.wired_ns > 0 and p.cache_ns > 0);
    try testing.expectEqual(mlx.engine.pipelinesMade().count - made_before.count, p.compiled);
    try testing.expect(p.otherNs() * 20 <= p.total_ns);
}

/// The tiny STQ fixture loaded as the engine tests load it: engine decode on, the DSpark stages and a 64 MB cache.
const EngineFixture = struct {
    config: model_mod.ModelConfig,
    weights: model_mod.Weights,
    xfm: Trunk,
    ids: []u32,
    was_enabled: bool,
    was_own: ?bool,

    fn open(self: *EngineFixture, dir: []const u8) !void {
        return self.openAsking(dir, null);
    }

    /// The pack loaded and run on the MLX route: where a test takes MLX's bytes from. A route is a load's for its
    /// life, so the reference is a load of its own, closed before the engine's opens.
    fn openMlx(self: *EngineFixture, dir: []const u8) !void {
        return self.openAsking(dir, .flag);
    }

    /// With `ask` the model decodes on the side `decodeSideOf` gives that ask, the pack and the GPU; without, the engine is forced on.
    fn openAsking(self: *EngineFixture, dir: []const u8, ask: ?dsv41_config.MlxAsk) !void {
        return self.openSized(dir, ask, 64 << 20);
    }

    /// `openAsking` with an expert cache of `cache_bytes`: a small one makes every forward read experts.
    fn openSized(self: *EngineFixture, dir: []const u8, ask: ?dsv41_config.MlxAsk, cache_bytes: u64) !void {
        const a = testing.allocator;
        const io = std.Io.Threaded.global_single_threaded.io();
        const dsv41_model = @import("dsv41_model.zig");
        self.was_enabled = mlx.engine.enabled;
        self.was_own = dsv41_borrow.own_override;
        mlx.engine.enabled = true;
        dsv41_borrow.own_override = true;
        errdefer self.restore();
        var fx = try dsv41_model.loadFixture(dir, "fixture_noqat.safetensors");
        defer fx.deinit();
        var sc = @import("dsv41_scope.zig").Scope.plain(a, mlx.gpuStream());
        defer sc.deinit();
        self.ids = try dsv41_model.tokens(&sc, &fx);
        errdefer a.free(self.ids);
        self.config = try model_mod.parseConfig(io, a, dir);
        errdefer self.config.deinit(a);
        self.config.expert_streaming = true;
        self.config.stream_mtp_head = true;
        self.config.expert_cache_bytes = cache_bytes;
        self.config.expert_source_dir = try a.dupe(u8, dir);
        if (ask) |x| {
            self.config.dsv41.?.mlx_ask = x;
            dsv41_borrow.own_override = dsv41_borrow.decodeSideOf(&self.config.dsv41.?) == .engine;
        }
        self.weights = try model_mod.loadWeightsForConfig(io, a, dir, &self.config, false);
        errdefer self.weights.deinit();
        self.xfm = try Trunk.init(io, a, self.config, &self.weights);
    }

    fn restore(self: *EngineFixture) void {
        test_support.restoreEngine(self.was_enabled);
        dsv41_borrow.own_override = self.was_own;
    }

    fn deinit(self: *EngineFixture) void {
        self.xfm.deinit();
        self.weights.deinit();
        self.config.deinit(testing.allocator);
        testing.allocator.free(self.ids);
        self.restore();
    }

    fn model(self: *EngineFixture) *@import("dsv41_model.zig").Model {
        return dsv41_state.modelOf(&self.xfm).?;
    }

    /// `n` token ids, wrapping over the fixture's own.
    fn prompt(self: *const EngineFixture, buf: []u32) []const u32 {
        for (buf, 0..) |*t, i| t.* = self.ids[(i * 5 + 3) % self.ids.len];
        return buf;
    }
};

/// A prompt prefilled in one wide forward, then `k` more rows as the last prefill forward (a follow-up turn's tail), then
/// `steps` greedy one-row steps: the bytes of the tail's logits and of every step's, and the tokens.
fn shortPrefillBytes(a: std.mem.Allocator, xfm: *Trunk, prompt: []const u32, lead: usize, k: usize, steps: usize) !std.ArrayList(u8) {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    const vocab = dsv41_state.modelOf(xfm).?.cfg.vocab;
    try xfm.resetCache();
    var ctx = xfm.defaultCtx();
    const first = try forwardIds(xfm, &ctx, prompt[0..lead]);
    _ = mlx.mlx_array_free(first);
    var lg = try forwardIds(xfm, &ctx, prompt[lead..][0..k]);
    errdefer _ = mlx.mlx_array_free(lg);
    for (0..steps + 1) |_| {
        try mlx.check(mlx.mlx_array_eval(lg));
        const data = (mlx.mlx_array_data_float32(lg) orelse return error.NoLogits)[0..vocab];
        try out.appendSlice(a, std.mem.sliceAsBytes(data));
        const tok: u32 = @intCast(std.mem.indexOfMax(f32, data));
        try out.appendSlice(a, std.mem.asBytes(&tok));
        _ = mlx.mlx_array_free(lg);
        lg = .{ .ctx = null };
        lg = try forwardIds(xfm, &ctx, &.{tok});
    }
    _ = mlx.mlx_array_free(lg);
    return out;
}

test "dsv41 engine: the same greedy tokens come from a load with Sushi Cooler on and with the setting or the flag off, and an off load runs no engine dispatch (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    if (!mlx.engine.matrixUnits()) return error.SkipZigTest;
    const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    const a = testing.allocator;
    var bytes: [3]std.ArrayList(u8) = @splat(.empty);
    defer for (&bytes) |*b| b.deinit(a);
    for ([_]dsv41_config.MlxAsk{ .none, .setting, .flag }, 0..) |ask, i| {
        var fx: EngineFixture = undefined;
        try fx.openAsking(dir, ask);
        defer fx.deinit();
        try testing.expectEqual(ask == .none, fx.model().engine_decode);
        var buf: [48]u32 = undefined;
        const prompt = fx.prompt(&buf);
        const before = mlx.engine.stats().dispatches;
        bytes[i] = try shortPrefillBytes(a, &fx.xfm, prompt, 30, 4, 6);
        const ran = mlx.engine.stats().dispatches - before;
        try testing.expectEqual(ask == .none, ran > 0);
    }
    try testing.expectEqualSlices(u8, bytes[0].items, bytes[1].items);
    try testing.expectEqualSlices(u8, bytes[0].items, bytes[2].items);
}

/// A prompt of 40 rows whose rows 5 to 14 are image tokens that take `rows`, fed in chunks ending at `cuts`, then
/// `steps` greedy one-row steps: the bytes of every forward's logits. With `quiet`, MLX's keep-warm touch is held off.
fn imagePromptBytes(a: std.mem.Allocator, fx: *EngineFixture, rows: mlx.mlx_array, cuts: []const usize, steps: usize, quiet: bool) !std.ArrayList(u8) {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    const m = fx.model();
    var buf: [40]u32 = undefined;
    var ids = fx.prompt(&buf)[0..40].*;
    for (ids[5..15]) |*id| id.* = m.cfg.image_token_id;
    try fx.xfm.resetCache();
    var ctx = fx.xfm.defaultCtx();
    ctx.vision_embeddings = rows;
    var at: usize = 0;
    var lg: mlx.mlx_array = .{ .ctx = null };
    errdefer _ = mlx.mlx_array_free(lg);
    for (cuts) |end| {
        _ = mlx.mlx_array_free(lg);
        lg = .{ .ctx = null };
        if (quiet) m.mlx_queue_used_ns = mlx.engine.nowNs();
        lg = try forwardIds(&fx.xfm, &ctx, ids[at..end]);
        ctx.vision_splice_offset += std.mem.count(u32, ids[at..end], &.{m.cfg.image_token_id});
        at = end;
    }
    for (0..steps + 1) |_| {
        try mlx.check(mlx.mlx_array_eval(lg));
        const data = (mlx.mlx_array_data_float32(lg) orelse return error.NoLogits)[0..m.cfg.vocab];
        try out.appendSlice(a, std.mem.sliceAsBytes(data));
        const tok: u32 = @intCast(std.mem.indexOfMax(f32, data));
        _ = mlx.mlx_array_free(lg);
        lg = .{ .ctx = null };
        if (quiet) m.mlx_queue_used_ns = mlx.engine.nowNs();
        lg = try forwardIds(&fx.xfm, &ctx, &.{tok});
    }
    _ = mlx.mlx_array_free(lg);
    return out;
}

/// The fixture has no tower: its last id stands for the image token, every layer gets an image routing bias far from
/// the text one (a row that took the wrong bias picks other experts), and ten bf16 rows stand for a tower's.
const ImageStandIn = struct {
    fx: *EngineFixture,
    image_was: u32,
    rows: mlx.mlx_array,

    fn give(a: std.mem.Allocator, fx: *EngineFixture) !ImageStandIn {
        const m = fx.model();
        var self: ImageStandIn = .{ .fx = fx, .image_was = m.cfg.image_token_id, .rows = mlx.mlx_array_new() };
        m.cfg.image_token_id = m.cfg.vocab - 1;
        errdefer self.drop();
        // The image biases and the tower's rows are the load's own: of the side the model forwards on.
        const host = mlx.engine.enter(fx.xfm.outStream());
        defer host.end();
        for (m.layers[0..m.cfg.layers]) |*ly| {
            try mlx.check(mlx.mlx_array_eval(ly.gate_bias));
            const bias = (mlx.mlx_array_data_float32(ly.gate_bias) orelse return error.NoBias)[0..mlx.mlx_array_size(ly.gate_bias)];
            const shifted = try a.alloc(f32, bias.len);
            defer a.free(shifted);
            for (bias, shifted, 0..) |b, *o, e| o.* = b + (if (e % 2 == 0) @as(f32, 6) else -6);
            ly.gate_bias_vl = mlx.mlx_array_new_data(shifted.ptr, &[_]c_int{ 1, @intCast(bias.len) }, 2, .float32);
        }
        const span = try a.alloc(f32, 10 * m.cfg.dim);
        defer a.free(span);
        var state: u32 = 0x1a6e;
        for (span) |*v| {
            state = state *% 1664525 +% 1013904223;
            v.* = @as(f32, @floatFromInt(state >> 8)) / 16777216.0 - 0.5;
        }
        const wide = mlx.mlx_array_new_data(span.ptr, &[_]c_int{ 10, @intCast(m.cfg.dim) }, 2, .float32);
        defer _ = mlx.mlx_array_free(wide);
        try mlx.check(mlx.mlx_astype(&self.rows, wide, .bfloat16, fx.xfm.outStream()));
        try mlx.check(mlx.mlx_array_eval(self.rows));
        return self;
    }

    fn drop(self: *ImageStandIn) void {
        const m = self.fx.model();
        _ = mlx.mlx_array_free(self.rows);
        for (m.layers[0..m.cfg.layers]) |*ly| {
            if (ly.gate_bias_vl.ctx != null) _ = mlx.mlx_array_free(ly.gate_bias_vl);
            ly.gate_bias_vl = .{ .ctx = null };
        }
        m.cfg.image_token_id = self.image_was;
    }
};

test "dsv41 engine: a forward that splices image rows runs on the engine at a prefill width and at a decode width, with MLX's logits and no op on MLX (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    if (!mlx.engine.matrixUnits()) return error.SkipZigTest;
    const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    const a = testing.allocator;
    // One chunk wider than a verify block with the whole span; and cuts through the span at decode widths.
    const all_cuts = [_][]const usize{ &.{40}, &.{ 8, 16, 40 }, &.{ 4, 12, 13, 20, 40 } };
    var want: [all_cuts.len]std.ArrayList(u8) = @splat(.empty);
    defer for (&want) |*w| w.deinit(a);
    {
        var ref: EngineFixture = undefined;
        try ref.openMlx(dir);
        defer ref.deinit();
        var image = try ImageStandIn.give(a, &ref);
        defer image.drop();
        for (all_cuts, &want) |cuts, *w| w.* = try imagePromptBytes(a, &ref, image.rows, cuts, 3, false);
    }
    var fx: EngineFixture = undefined;
    try fx.open(dir);
    defer fx.deinit();
    const m = fx.model();
    var image = try ImageStandIn.give(a, &fx);
    defer image.drop();
    const rows = image.rows;
    // The load warms these pipelines for a pack that has the image bias; the fixture got its bias after the load.
    try @import("dsv41_model.zig").warmOwnImages(m);
    for (all_cuts, &want) |cuts, *w| {
        const ops = mlx.engine.mlx_ops;
        const made = mlx.engine.pipelinesMade().count;
        var got = try imagePromptBytes(a, &fx, rows, cuts, 3, true);
        defer got.deinit(a);
        try testing.expectEqual(ops, mlx.engine.mlx_ops);
        try testing.expectEqual(made, mlx.engine.pipelinesMade().count);
        try testing.expectEqualSlices(u8, w.items, got.items);
    }
}

test "dsv41 engine: a forward through the Trunk while a verify is open runs on the engine, with the MLX route's logits and no op on MLX (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    if (!mlx.engine.matrixUnits()) return error.SkipZigTest;
    const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const dsv41_model = @import("dsv41_model.zig");
    var bytes: [2][]u8 = .{ &.{}, &.{} };
    defer for (bytes) |b| a.free(b);
    for ([_]bool{ false, true }, 0..) |own, i| {
        var fx: EngineFixture = undefined;
        try if (own) fx.open(dir) else fx.openMlx(dir);
        defer fx.deinit();
        const m = fx.model();
        var buf: [48]u32 = undefined;
        const prompt = fx.prompt(&buf);
        try fx.xfm.resetCache();
        var ctx = fx.xfm.defaultCtx();
        _ = mlx.mlx_array_free(try forwardIds(&fx.xfm, &ctx, prompt[0..30]));
        const rows = try dsv41_serving.dsv41VerifyImpl(&fx.xfm, &ctx, prompt[30..33]);
        try mlx.check(mlx.mlx_array_eval(rows));
        _ = mlx.mlx_array_free(rows);
        ctx.cache.step = (try dsv41_model.ensureState(m)).n;
        m.mlx_queue_used_ns = mlx.engine.nowNs();
        const ops = mlx.engine.mlx_ops;
        const lg = try forwardIds(&fx.xfm, &ctx, prompt[33..34]);
        defer _ = mlx.mlx_array_free(lg);
        try mlx.check(mlx.mlx_array_eval(lg));
        if (own) try testing.expectEqual(ops, mlx.engine.mlx_ops);
        const data = (mlx.mlx_array_data_float32(lg) orelse return error.NoLogits)[0..m.cfg.vocab];
        bytes[i] = try a.dupe(u8, std.mem.sliceAsBytes(data));
    }
    try testing.expectEqualSlices(u8, bytes[0], bytes[1]);
}

/// What one phase of a request asked of MLX: every call into mlx-c, and the ops among them.
const Asked = struct { calls: u64, ops: u64 };

const Census = struct {
    calls: u64,
    ops: u64,
    names: std.ArrayList(mlx.NameCount) = .empty,
    stats: @TypeOf(mlx.engine.stats()) = undefined,

    /// Starts a phase; MLX's keep-warm touch (one per 500 ms of MLX idle) stays out of a phase this short.
    fn begin(model: ?*@import("dsv41_model.zig").Model) !Census {
        if (model) |m| m.mlx_queue_used_ns = mlx.engine.nowNs();
        var self: Census = .{ .calls = mlx.engine.mlxCalls(), .ops = mlx.engine.mlx_ops, .stats = mlx.engine.stats() };
        try mlx.callsByName(&self.names, testing.allocator);
        return self;
    }

    /// Ends it. `SUSHI_DSV41_CENSUS_TRACE` prints the phase's calls by entry point.
    fn end(self: *Census, comptime phase: []const u8) !Asked {
        defer {
            self.names.deinit(testing.allocator);
            self.names = .empty;
        }
        const asked: Asked = .{ .calls = mlx.engine.mlxCalls() - self.calls, .ops = mlx.engine.mlx_ops - self.ops };
        if (std.c.getenv("SUSHI_DSV41_CENSUS_TRACE") == null) return asked;
        var now: std.ArrayList(mlx.NameCount) = .empty;
        defer now.deinit(testing.allocator);
        try mlx.callsByName(&now, testing.allocator);
        std.debug.print("[census] {s}: {d} calls, {d} ops\n", .{ phase, asked.calls, asked.ops });
        const now_stats = mlx.engine.stats();
        std.debug.print("[census]   engine: {d} dispatches, {d} submits, imports {d}, lends {d}, appends in place {d}, copied {d}\n", .{ now_stats.dispatches - self.stats.dispatches, now_stats.submits - self.stats.submits, now_stats.imports - self.stats.imports, now_stats.lends - self.stats.lends, now_stats.updates_in_place - self.stats.updates_in_place, now_stats.updates_copied - self.stats.updates_copied });
        for (now.items) |n| {
            var was: u64 = 0;
            for (self.names.items) |b| if (std.mem.eql(u8, b.name, n.name)) {
                was = b.count;
            };
            if (n.count != was) std.debug.print("[census]   {s} {d}\n", .{ n.name, n.count - was });
        }
        return asked;
    }
};

/// One load, the phases of a request, the unload: what each asked of MLX. With `own_weights` the engine wraps the
/// pack's weights itself.
fn censusOf(dir: []const u8) ![9]Asked {
    const a = testing.allocator;
    var asked: [9]Asked = undefined;
    var fx: EngineFixture = undefined;
    {
        // What a process asks once (the device's description, a kernel probe) is its first load's: the load counted
        // is never that one, so a row reads the same alone and after other tests.
        var first: EngineFixture = undefined;
        try first.open(dir);
        first.deinit();
    }
    var c = try Census.begin(null);
    errdefer c.names.deinit(a);
    try fx.open(dir);
    var open = true;
    defer if (open) fx.deinit();
    asked[8] = try c.end("the load");
    const m = fx.model();
    var tok = Tokenizer.initEmptyForTests(a);
    defer tok.deinit();
    var buf: [48]u32 = undefined;
    const prompt = fx.prompt(&buf)[0..40];
    var image = try ImageStandIn.give(a, &fx);
    var image_given = true;
    defer if (image_given) image.drop();
    try @import("dsv41_model.zig").warmOwnImages(m);

    const Request = struct {
        fn run(f: *EngineFixture, t: *Tokenizer, ids: []const u32, sampling: SamplingParams, opts: Generator.InitOptions) !void {
            const al = testing.allocator;
            try f.xfm.resetCache();
            var gen = try Generator.initWithOptions(std.Io.Threaded.global_single_threaded.io(), al, &f.xfm, t, ids, 12, sampling, &.{}, opts);
            defer gen.deinit(al);
            if (opts.dspark_enabled) {
                gen.dspark_diag = .sweep;
                while (try gen.nextDspark(al)) |r| al.free(r.tokens);
            } else while (try gen.next(al)) |_| {}
        }
    };
    // Once through everything first: the expert cache, the pools and the pipelines are a load's, not a request's.
    const greedy: SamplingParams = .{ .temperature = 0.0 };
    const seeded: SamplingParams = .{ .temperature = 0.8, .top_p = 0.95, .seed = 5 };
    try Request.run(&fx, &tok, prompt, greedy, .{ .dspark_enabled = true });
    try Request.run(&fx, &tok, prompt, seeded, .{ .dspark_enabled = true });
    try Request.run(&fx, &tok, prompt, greedy, .{ .skip_lazy_preforward = true, .logprobs_n = 3 });

    c = try Census.begin(m);
    try Request.run(&fx, &tok, prompt, greedy, .{ .dspark_enabled = true });
    asked[0] = try c.end("greedy rounds: a 40-row prefill and 12 tokens");

    c = try Census.begin(m);
    try Request.run(&fx, &tok, prompt, seeded, .{ .dspark_enabled = true });
    asked[1] = try c.end("seeded rounds");

    c = try Census.begin(m);
    try Request.run(&fx, &tok, prompt, greedy, .{ .skip_lazy_preforward = true, .logprobs_n = 3 });
    asked[2] = try c.end("serial steps with logprobs");

    try fx.xfm.resetCache();
    var ctx = fx.xfm.defaultCtx();
    _ = mlx.mlx_array_free(try forwardIds(&fx.xfm, &ctx, prompt));
    c = try Census.begin(m);
    var cp = try dsv41_prefix.capture(a, m, prompt.len);
    defer cp.deinit(a);
    asked[3] = try c.end("a restore point captured");

    try fx.xfm.resetCache();
    c = try Census.begin(m);
    try dsv41_prefix.restore(m, &cp, prompt);
    asked[4] = try c.end("the point restored");

    ctx = fx.xfm.defaultCtx();
    ctx.cache.step = prompt.len;
    c = try Census.begin(m);
    _ = mlx.mlx_array_free(try forwardIds(&fx.xfm, &ctx, &.{prompt[3]}));
    asked[5] = try c.end("a step on the restored state");

    c = try Census.begin(m);
    var bytes = try imagePromptBytes(a, &fx, image.rows, &.{ 8, 16, 40 }, 3, true);
    bytes.deinit(a);
    asked[6] = try c.end("an image prompt in three chunks and 3 steps");

    image.drop();
    image_given = false;
    c = try Census.begin(m);
    open = false;
    fx.deinit();
    asked[7] = try c.end("the unload");

    return asked;
}

test "dsv41 engine: what a Cooler request still asks of MLX, phase by phase, pinned so that it only goes down (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    if (!mlx.engine.matrixUnits()) return error.SkipZigTest;
    const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    const asked = try censusOf(dir);
    // The target of every row is { 0, 0 }, and every phase of a request is there: no call at all. A load is left the
    // host arrays its warm makes outside a forward's scope and a few conversions.
    // Ops are exact. Calls are a ceiling a phase stays under and within a twentieth of, so a real cut moves it.
    const ceilings = [_]Asked{
        .{ .calls = 0, .ops = 0 }, // greedy rounds: a 40-row prefill and 12 tokens
        .{ .calls = 0, .ops = 0 }, // seeded rounds
        .{ .calls = 0, .ops = 0 }, // serial steps with logprobs
        .{ .calls = 0, .ops = 0 }, // a restore point captured
        .{ .calls = 0, .ops = 0 }, // the point restored
        .{ .calls = 0, .ops = 0 }, // a step on the restored state
        .{ .calls = 0, .ops = 0 }, // an image prompt
        .{ .calls = 7, .ops = 0 }, // the unload
        .{ .calls = 684, .ops = 6 }, // the load
    };
    for (ceilings, asked) |ceiling, got| {
        try testing.expectEqual(ceiling.ops, got.ops);
        try testing.expect(got.calls <= ceiling.calls);
        try testing.expect(got.calls + ceiling.calls / 20 >= ceiling.calls);
    }
}

test "dsv41 engine: a Cooler load writes a restore point to the SSD tier and restores it, byte for byte, with nothing asked of MLX (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    if (!mlx.engine.matrixUnits()) return error.SkipZigTest;
    const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = testing.io;
    const pc = @import("../prefix_cache.zig");
    const kv_disk_cache = @import("../kv_disk_cache.zig");
    var fx: EngineFixture = undefined;
    try fx.open(dir);
    defer fx.deinit();
    const m = fx.model();
    var tmp = testing.tmpDir(.{ .iterate = true });
    defer tmp.cleanup();
    var path_buf: [512]u8 = undefined;
    const base = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    var buf: [600]u32 = undefined;
    const prompt = fx.prompt(&buf);
    const point = 512;

    // The tier as the scheduler arms it, on a grid the point is on.
    const Tier = struct {
        fn open(f: *EngineFixture, at: []const u8) !pc.HotPrefixCache {
            var hc = pc.HotPrefixCache.initWithMem(testing.allocator, 0, 0);
            errdefer hc.deinit();
            hc.ssd_first = true;
            hc.hybrid = true;
            hc.disk = try kv_disk_cache.DiskTier.init(testing.allocator, testing.io, at, "dsv41-census", 0, 128);
            hc.disk.?.ssd_first = true;
            hc.disk.?.armTestSpace(256 << 30, 1 << 40);
            hc.disk.?.enableBackgroundWriter();
            const c = &f.config.dsv41.?;
            const divs = try dsv41_prefix.rowDivisors(testing.allocator, c, true);
            defer testing.allocator.free(divs);
            hc.disk.?.ssm_max_per_entry = dsv41_prefix.disk_checkpoint_cap;
            try hc.disk.?.setGrownSpec(256, divs, c.stateBytesPerToken());
            hc.arrays_on = dsv41_prefix.arraysOn(&f.config);
            hc.disk.?.arrays_on = hc.arrays_on;
            return hc;
        }
    };
    var asked: [2]Asked = undefined;
    // The state at the point as an owned copy, for the bytes the restore must give back.
    var want: trunk_mod.SSMCheckpoint = undefined;
    var wanted = false;
    defer if (wanted) want.deinit(a);
    {
        var hc = try Tier.open(&fx, base);
        defer hc.deinit();
        try fx.xfm.resetCache();
        var ctx = fx.xfm.defaultCtx();
        _ = mlx.mlx_array_free(try forwardIds(&fx.xfm, &ctx, prompt[0..point]));
        ctx.cache.step = point;
        want = try dsv41_prefix.capture(a, m, point);
        wanted = true;
        var c = try Census.begin(m);
        errdefer c.names.deinit(a);
        const cps = try a.alloc(trunk_mod.SSMCheckpoint, 1);
        cps[0] = (try dsv41_prefix.captureCheckpoint(a, &fx.xfm, point, true)).?;
        dsv41_prefix.writeThroughDsv41(&hc, m, ctx.cache, prompt[0..point], false, .{}, cps, "prefill");
        try testing.expect(cps[0].isReleased());
        try dsv41_prefix.markRows(ctx.cache, @import("dsv41_model.zig").stateStream(m));
        _ = try hc.commitWithState(ctx.cache, prompt[0..point], false, 0, cps, null, null);
        hc.flushPendingDisk(fx.xfm.s);
        hc.disk.?.drainWriter();
        asked[0] = try c.end("a 512-row restore point written to the SSD tier");
        try testing.expectEqual(@as(u64, 0), hc.disk.?.writeErrors());
        try testing.expectEqual(@as(usize, 1), hc.disk.?.entryCount());
    }
    {
        var hc = try Tier.open(&fx, base);
        defer hc.deinit();
        try fx.xfm.resetCache();
        const ctx = fx.xfm.defaultCtx();
        var moe: usize = 0;
        var c = try Census.begin(m);
        errdefer c.names.deinit(a);
        const res = try dsv41_prefix.lookupAndRestoreDsv41(&hc, ctx.cache, &moe, m, fx.xfm.s, prompt, false, 0, null, &.{}, null, false);
        asked[1] = try c.end("the point restored from the SSD tier");
        try testing.expectEqual(@as(usize, point), res.matched);
        var got = try dsv41_prefix.capture(a, m, point);
        defer got.deinit(a);
        try testing.expectEqual(want.layers.len, got.layers.len);
        for (want.layers, got.layers) |w, g| {
            for ([_][2]mlx.mlx_array{ .{ w.conv_state, g.conv_state }, .{ w.ssm_state, g.ssm_state } }) |pair| {
                try testing.expect(mlx.engine.isOurs(pair[1]));
                try testing.expectEqual(mlx.mlx_array_dtype(pair[0]), mlx.mlx_array_dtype(pair[1]));
                try testing.expectEqualSlices(c_int, mlx.getShape(pair[0]), mlx.getShape(pair[1]));
                const bytes = mlx.mlx_array_size(pair[0]) * mlx.mlx_array_itemsize(pair[0]);
                if (bytes == 0) continue;
                try testing.expectEqualSlices(u8, mlx.mlx_array_data_uint8(pair[0]).?[0..bytes], mlx.mlx_array_data_uint8(pair[1]).?[0..bytes]);
            }
        }
    }
    for (asked) |got| try testing.expectEqual(Asked{ .calls = 0, .ops = 0 }, got);
}

test "dsv41 engine: a Cooler load answers the fused hyper-connection kernel's probe as the MLX route does: the probe runs on the engine's stream (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    if (!mlx.engine.matrixUnits()) return error.SkipZigTest;
    const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    const kernels = @import("dsv41_kernels.zig");
    var probed: [2]bool = undefined;
    for ([_]dsv41_config.MlxAsk{ .flag, .none }, &probed) |ask, *on| {
        var fx: EngineFixture = undefined;
        try fx.openAsking(dir, ask);
        defer fx.deinit();
        on.* = kernels.hcExpandProbed();
        mlx.engine.close();
    }
    // Whatever this GPU's answer is, it is one answer: a probe the engine could not run would read as "differs".
    try testing.expectEqual(probed[0], probed[1]);
}

test "dsv41 engine: a restore point's copies take their own room in the pool, whatever cap the last forward left (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    if (!mlx.engine.matrixUnits()) return error.SkipZigTest;
    const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    const a = testing.allocator;
    var fx: EngineFixture = undefined;
    try fx.open(dir);
    defer fx.deinit();
    const m = fx.model();
    var buf: [48]u32 = undefined;
    const prompt = fx.prompt(&buf)[0..40];
    try fx.xfm.resetCache();
    var ctx = fx.xfm.defaultCtx();
    _ = mlx.mlx_array_free(try forwardIds(&fx.xfm, &ctx, prompt));
    // A later test sees no limit.
    defer mlx.engine.budget(1 << 62) catch {};
    // A cap with nothing to spare: what the tensors in use hold, their buffers' rounding included.
    const held = mlx.engine.stats();
    try mlx.engine.budget(held.pool_bytes - held.pool_free - held.pool_asked);
    const made = mlx.engine.pipelinesMade().count;
    var cp = try dsv41_prefix.capture(a, m, prompt.len);
    defer cp.deinit(a);
    // The load warmed the copies' pipelines.
    try testing.expectEqual(made, mlx.engine.pipelinesMade().count);
    // Every row the point holds is a tensor of the engine's (an absent slot is a placeholder).
    var held_rows: usize = 0;
    for (cp.layers) |l| for ([_]mlx.mlx_array{ l.conv_state, l.ssm_state }) |x| {
        if (!@import("dsv41_model.zig").present(x)) continue;
        held_rows += 1;
        try testing.expect(mlx.engine.isOurs(x));
    };
    try testing.expect(held_rows > 0);
    var said: [256]u8 = undefined;
    try testing.expectEqualStrings("", mlx.takeError(&said) orelse "");
}

test "dsv41 engine: a prefill forward of 2 to 16 rows gives MLX's logits and the steps after it MLX's bytes (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    const a = testing.allocator;
    var want: [17]std.ArrayList(u8) = @splat(.empty);
    defer for (&want) |*w| w.deinit(a);
    var buf: [48]u32 = undefined;
    {
        var ref: EngineFixture = undefined;
        try ref.openMlx(dir);
        defer ref.deinit();
        const prompt = ref.prompt(&buf);
        for (2..17) |k| want[k] = try shortPrefillBytes(a, &ref.xfm, prompt, 30, k, 3);
    }
    var fx: EngineFixture = undefined;
    try fx.open(dir);
    defer fx.deinit();
    const prompt = fx.prompt(&buf);
    for (2..17) |k| {
        var got = try shortPrefillBytes(a, &fx.xfm, prompt, 30, k, 3);
        defer got.deinit(a);
        try testing.expectEqualSlices(u8, want[k].items, got.items);
    }
}

test "dsv41 engine: a restored point followed by a prefill forward of 2, 7 or 16 rows decodes MLX's tokens (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    const dsv41_model = @import("dsv41_model.zig");
    const a = testing.allocator;
    const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    const widths = [_]usize{ 2, 7, 16 };
    var bytes: [2][widths.len]std.ArrayList(u8) = @splat(@splat(.empty));
    defer for (&bytes) |*route| for (route) |*b| b.deinit(a);
    // Each route captures its own point: a point's arrays are the route's.
    for (&bytes, [_]bool{ false, true }) |*route, engine| {
        var fx: EngineFixture = undefined;
        try if (engine) fx.open(dir) else fx.openMlx(dir);
        defer fx.deinit();
        const m = fx.model();
        var buf: [48]u32 = undefined;
        const prompt = fx.prompt(&buf);
        const lead: usize = 30;
        const vocab = m.cfg.vocab;
        try fx.xfm.resetCache();
        const point = try a.alloc(dsv41_model.Slot, dsv41_model.slotCount(m));
        defer a.free(point);
        @memset(point, @splat(.{ .ctx = null }));
        {
            var ctx = fx.xfm.defaultCtx();
            _ = mlx.mlx_array_free(try forwardIds(&fx.xfm, &ctx, prompt[0..lead]));
            try dsv41_model.captureState(m, point);
        }
        defer for (point) |*sl| dsv41_model.freeSlot(sl);
        for (widths, route) |k, *b| {
            try fx.xfm.resetCache();
            try dsv41_model.restoreState(m, point, lead, prompt);
            var ctx = fx.xfm.defaultCtx();
            ctx.cache.step = lead;
            var lg = try forwardIds(&fx.xfm, &ctx, prompt[lead..][0..k]);
            errdefer _ = mlx.mlx_array_free(lg);
            for (0..4) |_| {
                try mlx.check(mlx.mlx_array_eval(lg));
                const data = (mlx.mlx_array_data_float32(lg) orelse return error.NoLogits)[0..vocab];
                try b.appendSlice(a, std.mem.sliceAsBytes(data));
                const tok: u32 = @intCast(std.mem.indexOfMax(f32, data));
                _ = mlx.mlx_array_free(lg);
                lg = .{ .ctx = null };
                lg = try forwardIds(&fx.xfm, &ctx, &.{tok});
            }
            _ = mlx.mlx_array_free(lg);
        }
    }
    for (bytes[0], bytes[1]) |want, got| try testing.expectEqualSlices(u8, want.items, got.items);
}

test "dsv41 engine: a prefill forward of 2 to 16 rows costs MLX's queue what a one-row step does (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    var fx: EngineFixture = undefined;
    try fx.open(dir);
    defer fx.deinit();
    const m = fx.model();
    var buf: [48]u32 = undefined;
    const prompt = fx.prompt(&buf);
    var counts: [17]usize = undefined;
    for ([_]usize{ 1, 2, 9, 16 }) |k| {
        try fx.xfm.resetCache();
        var ctx = fx.xfm.defaultCtx();
        _ = mlx.mlx_array_free(try forwardIds(&fx.xfm, &ctx, prompt[0..30]));
        // A queue just used needs no keep-alive touch, so the count is the forward's own.
        m.mlx_queue_used_ns = mlx.engine.nowNs();
        var before: usize = 0;
        try mlx.check(mlx.mlx_metal_command_buffer_count(&before));
        const lg = try forwardIds(&fx.xfm, &ctx, prompt[30..][0..k]);
        defer _ = mlx.mlx_array_free(lg);
        try mlx.check(mlx.mlx_array_eval(lg));
        var after: usize = 0;
        try mlx.check(mlx.mlx_metal_command_buffer_count(&after));
        counts[k] = after - before;
    }
    for ([_]usize{ 2, 9, 16 }) |k| try testing.expectEqual(counts[1], counts[k]);
}

test "dsv41 engine: a load with decode on the engine keeps the NAX GEMM arm (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    if (!mlx.engine.matrixUnits()) return error.SkipZigTest;
    const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    var fx: EngineFixture = undefined;
    try fx.open(dir);
    defer fx.deinit();
    try testing.expect(!@import("sushi_exl3").kernels.gemm_nax_failed);
}

test "dsv41 engine: after the load a forward of every width from 1 to 16 and a DSpark round compile nothing, on either side (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    var fx: EngineFixture = undefined;
    try fx.open(dir);
    defer fx.deinit();
    const m = fx.model();
    try testing.expect(m.engine_decode);
    var buf: [48]u32 = undefined;
    const prompt = fx.prompt(&buf);
    const late = mlx.engine.latePipelines();
    const keys = mlx.kernel_keys.trackForTest();
    for (1..17) |k| {
        try fx.xfm.resetCache();
        var ctx = fx.xfm.defaultCtx();
        _ = mlx.mlx_array_free(try forwardIds(&fx.xfm, &ctx, prompt[0..30]));
        const lg = try forwardIds(&fx.xfm, &ctx, prompt[30..][0..k]);
        defer _ = mlx.mlx_array_free(lg);
        try mlx.check(mlx.mlx_array_eval(lg));
        const tok = try argmaxHost(&fx.xfm, lg);
        var d = try dsv41_serving.dsv41DraftImpl(&fx.xfm, tok);
        defer d.deinit();
        const rows = 1 + k % dsv41_max_verify_rows;
        const logits = try dsv41_serving.dsv41VerifyImpl(&fx.xfm, &ctx, d.ids[0..@min(rows, d.ids.len)]);
        defer _ = mlx.mlx_array_free(logits);
        try mlx.check(mlx.mlx_array_eval(logits));
        try dsv41_serving.dsv41FinishImpl(&fx.xfm, &ctx, 0);
    }
    try testing.expectEqual(late, mlx.engine.latePipelines());
    try testing.expectEqual(keys, mlx.kernel_keys.count());
}

test "dsv41 engine: after the load's warm no step or round compiles a kernel, whatever its width (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    const dsv41_model = @import("dsv41_model.zig");
    const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was);
    var fx = try dsv41_model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    var sc = @import("dsv41_scope.zig").Scope.plain(a, mlx.gpuStream());
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
    var xfm = try Trunk.init(io, a, config, &weights);
    defer xfm.deinit();
    const m = dsv41_state.modelOf(&xfm).?;
    try testing.expect(m.dspark != null);
    m.engine_decode = true;
    defer m.engine_decode = false;
    try dsv41_model.warmKernels(m);
    mlx.engine.seal(true);
    defer mlx.engine.seal(false);
    const late = mlx.engine.latePipelines();

    try xfm.resetCache();
    var ctx = xfm.defaultCtx();
    const first = try forwardIds(&xfm, &ctx, ids[0..30]);
    var tok = try argmaxHost(&xfm, first);
    _ = mlx.mlx_array_free(first);
    var round: [dsv41_max_verify_rows]u32 = undefined;
    // Twice over the widths: the second pass meets each one with a longer state behind it.
    for (0..2 * @as(usize, dsv41_max_verify_rows)) |r| {
        const one = try forwardIds(&xfm, &ctx, &.{tok});
        tok = try argmaxHost(&xfm, one);
        _ = mlx.mlx_array_free(one);
        var d = try dsv41_serving.dsv41DraftImpl(&xfm, tok);
        defer d.deinit();
        const rows = 1 + r % dsv41_max_verify_rows;
        // Past the block a copy round's rows are whatever the context held: the block's last draft stands in.
        for (round[0..rows], 0..) |*id, i| id.* = d.ids[@min(i, d.ids.len - 1)];
        const logits = try dsv41_serving.dsv41VerifyImpl(&xfm, &ctx, round[0..rows]);
        defer _ = mlx.mlx_array_free(logits);
        try mlx.check(mlx.mlx_array_eval(logits));
        try dsv41_serving.dsv41FinishImpl(&xfm, &ctx, r % rows);
        tok = round[r % rows];
    }
    try testing.expectEqual(late, mlx.engine.latePipelines());
}

test "dsv41 engine: after the load's warm a draft compiles no kernel at any fill of the stages' windows (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    const dsv41_model = @import("dsv41_model.zig");
    const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was);
    var fx = try dsv41_model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    var sc = @import("dsv41_scope.zig").Scope.plain(a, mlx.gpuStream());
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
    var xfm = try Trunk.init(io, a, config, &weights);
    defer xfm.deinit();
    const m = dsv41_state.modelOf(&xfm).?;
    try testing.expect(m.dspark != null);
    m.engine_decode = true;
    defer m.engine_decode = false;
    // The fixture's window of 8 rows is under every size at which MLX changes kernels: a wider one crosses them.
    m.cfg.window = 80;
    try dsv41_model.warmKernels(m);
    mlx.engine.seal(true);
    defer mlx.engine.seal(false);
    const late = mlx.engine.latePipelines();

    try xfm.resetCache();
    var ctx = xfm.defaultCtx();
    // A stage's attention is MLX's matmul and sum over the rows its window holds, and MLX picks their kernels by
    // that count: a prompt shorter than the window drafts at fills a long one never shows.
    for (0..m.cfg.window + 2) |i| {
        const one = try forwardIds(&xfm, &ctx, &.{ids[i % ids.len]});
        _ = mlx.mlx_array_free(one);
        var d = try dsv41_serving.dsv41DraftImpl(&xfm, ids[i % ids.len]);
        d.deinit();
    }
    try testing.expectEqual(late, mlx.engine.latePipelines());
}

test "dsv41 engine: the load's warm makes the split top-k a one-row step takes over 131072 keys (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    const dsv41_model = @import("dsv41_model.zig");
    const kernels = @import("dsv41_kernels.zig");
    const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was);
    var config = try model_mod.parseConfig(io, a, dir);
    defer config.deinit(a);
    config.expert_streaming = true;
    config.stream_mtp_head = true;
    config.expert_cache_bytes = 64 << 20;
    config.expert_source_dir = try a.dupe(u8, dir);
    var weights = try model_mod.loadWeightsForConfig(io, a, dir, &config, false);
    defer weights.deinit();
    var xfm = try Trunk.init(io, a, config, &weights);
    defer xfm.deinit();
    const m = dsv41_state.modelOf(&xfm).?;
    m.engine_decode = true;
    defer m.engine_decode = false;
    try dsv41_model.warmKernels(m);
    mlx.engine.seal(true);
    defer mlx.engine.seal(false);
    const late = mlx.engine.latePipelines();

    // The load cannot build that context: the kernels are reached here as the warm has to reach them.
    const stream = dsv41_model.ownBegin(m);
    defer dsv41_model.ownEnd(m, stream);
    var sc = m.scope();
    defer sc.deinit();
    const k: usize = m.cfg.index_topk;
    const picks = try kernels.topkIndicesSplit(&sc, try sc.zeros(&.{ 1, @intCast(kernels.topk_splits * k) }, .float32), k, kernels.topk_splits);
    try mlx.check(mlx.mlx_array_eval(picks));
    try testing.expectEqual(late, mlx.engine.latePipelines());
}

/// What a prefill leaves and what decode does after it: the state's digest and the last row's logits after the prefill,
/// the digest after the decode, and the decode's tokens.
const WideRun = struct {
    prefill_dispatches: u64 = 0,
    prefill_digest: [32]u8 = undefined,
    final_digest: [32]u8 = undefined,
    logits: std.ArrayList(f32) = .empty,
    tokens: std.ArrayList(u32) = .empty,

    fn deinit(self: *WideRun, a: std.mem.Allocator) void {
        self.logits.deinit(a);
        self.tokens.deinit(a);
    }

    fn expectSame(want: *const WideRun, got: *const WideRun) !void {
        try testing.expectEqualSlices(u8, &want.prefill_digest, &got.prefill_digest);
        try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(want.logits.items), std.mem.sliceAsBytes(got.logits.items));
        try testing.expectEqualSlices(u32, want.tokens.items, got.tokens.items);
        try testing.expectEqualSlices(u8, &want.final_digest, &got.final_digest);
    }
};

fn stateHash(m: *@import("dsv41_model.zig").Model) ![32]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    try dsv41_prefix.stateDigest(m, &h, m.s);
    return h.finalResult();
}

/// A prompt in `chunks`, the way the Generator hands it over (each chunk says how many rows the next has), then `steps`
/// greedy one-row decodes.
fn wideRun(a: std.mem.Allocator, xfm: *Trunk, prompt: []const u32, chunks: []const usize, steps: usize) !WideRun {
    var run: WideRun = .{};
    errdefer run.deinit(a);
    const m = dsv41_state.modelOf(xfm).?;
    try xfm.resetCache();
    var ctx = xfm.defaultCtx();
    var at: usize = 0;
    var tok: u32 = 0;
    const dispatches = mlx.engine.stats().dispatches;
    for (chunks, 0..) |n, i| {
        ctx.ahead_min_rows = if (i + 1 < chunks.len) chunks[i + 1] else 0;
        const lg = try forwardIds(xfm, &ctx, prompt[at..][0..n]);
        defer _ = mlx.mlx_array_free(lg);
        at += n;
        if (i + 1 < chunks.len) continue;
        tok = try argmaxHost(xfm, lg);
        try mlx.check(mlx.mlx_array_eval(lg));
        try run.logits.appendSlice(a, (mlx.mlx_array_data_float32(lg) orelse return error.NoLogits)[0..m.cfg.vocab]);
    }
    run.prefill_dispatches = mlx.engine.stats().dispatches - dispatches;
    run.prefill_digest = try stateHash(m);
    ctx.ahead_min_rows = 0;
    for (0..steps) |_| {
        try run.tokens.append(a, tok);
        const lg = try forwardIds(xfm, &ctx, &.{tok});
        defer _ = mlx.mlx_array_free(lg);
        tok = try argmaxHost(xfm, lg);
    }
    run.final_digest = try stateHash(m);
    return run;
}

/// Each of `plans` (a prompt cut into chunks) run on MLX and then on the engine of our own, with the wide chunks sent
/// there too.
/// `cache_bytes` below the experts of the pack makes the chunks stream (the union and the staged fill); `stall_ns` pauses
/// the staged fill after each rate group, on the engine's run only. With `refused`, the engine's run stages the fill on the
/// engine, which must stop with an error that names it.
fn wideChunkProof(dir: []const u8, plans: []const []const usize, cache_bytes: u64, stall_ns: u64, refused: ?[]const u8) !void {
    try mlx.requireGpu();
    // A refused forward unwinds without freeing what it held: it runs on an allocator that does not report that, and the
    // slabs it still holds time out their release at the end.
    const a = if (refused != null) std.heap.smp_allocator else testing.allocator;
    const timeouts = @import("../expert_stream.zig").slab_release_timeouts.load(.monotonic);
    defer if (refused != null) @import("../expert_stream.zig").slab_release_timeouts.store(timeouts, .monotonic);
    const io = std.Io.Threaded.global_single_threaded.io();
    const dsv41_model = @import("dsv41_model.zig");
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was);
    var fx = try dsv41_model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    var sc = @import("dsv41_scope.zig").Scope.plain(a, mlx.gpuStream());
    defer sc.deinit();
    const ids = try dsv41_model.tokens(&sc, &fx);
    defer testing.allocator.free(ids);
    var config = try model_mod.parseConfig(io, a, dir);
    defer config.deinit(a);
    config.expert_streaming = true;
    config.expert_cache_bytes = cache_bytes;
    config.expert_source_dir = try a.dupe(u8, dir);
    var weights = try model_mod.loadWeightsForConfig(io, a, dir, &config, false);
    defer weights.deinit();
    var xfm = try Trunk.init(io, a, config, &weights);
    defer xfm.deinit();
    const m = dsv41_state.modelOf(&xfm).?;
    const hook = dsv41_state.stqExpertsOf(&xfm).?;
    var prompt: [1100]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = ids[(i * 5 + 3) % ids.len];
    for (plans) |chunks| {
        var want = try wideRun(a, &xfm, &prompt, chunks, 12);
        defer want.deinit(a);
        m.engine_decode = true;
        m.engine_prefill = true;
        hook.stall_ns = stall_ns;
        hook.stage_on_engine = refused != null;
        const filled = hook.engine.fill_bytes_total;
        defer {
            m.engine_decode = false;
            m.engine_prefill = false;
            hook.stall_ns = 0;
            hook.stage_on_engine = false;
        }
        if (refused) |name| {
            if (wideRun(a, &xfm, &prompt, chunks, 12)) |ran| {
                var r = ran;
                r.deinit(a);
                return error.ExpectedRefusal;
            } else |_| {}
            var buf: [512]u8 = undefined;
            const said = mlx.takeError(&buf) orelse return error.NoRefusalMessage;
            mlx.engine.active = false;
            try testing.expect(std.mem.indexOf(u8, said, name) != null);
            // A refused launch is its own failure: the kernel arm it used stays on for the process.
            try testing.expect(!@import("sushi_exl3").kernels.gemm_nax_failed);
            return;
        }
        var got = wideRun(a, &xfm, &prompt, chunks, 12) catch |e| {
            var buf: [512]u8 = undefined;
            if (mlx.takeError(&buf)) |said| std.debug.print("{s}\n", .{said});
            return e;
        };
        defer got.deinit(a);
        try testing.expectEqual(@as(u64, 0), want.prefill_dispatches);
        try testing.expect(got.prefill_dispatches > 0);
        try testing.expectEqual(cache_bytes < (64 << 20), hook.engine.fill_bytes_total > filled);
        try want.expectSame(&got);
    }
}

test "dsv41 engine: a prefill chunk of 17 rows on the engine of our own gives MLX's state, logits and decode (DSV41_STQ_TINY)" {
    try wideChunkProof(std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest), &.{ &.{17}, &.{24}, &.{33}, &.{64}, &.{128}, &.{256}, &.{513}, &.{ 40, 17 }, &.{ 64, 64, 33 }, &.{ 300, 300, 17 } }, 64 << 20, 0, null);
}

test "dsv41 engine: a wide chunk whose experts stream through the union gives MLX's state, logits and decode, though the fill is slow (DSV41_STQ_TINY)" {
    // Three experts a layer: the cache is below the union, so every chunk fills.
    const cache = 3 * 8 * (256 * 128 * 3 * 64 / 128);
    try wideChunkProof(std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest), &.{ &.{ 64, 64, 33 }, &.{ 300, 17 } }, cache, 20 * std.time.ns_per_ms, null);
}

test "dsv41 engine: a pass that reads a slab its fill is still writing is refused by name, not run (DSV41_STQ_TINY)" {
    const cache = 3 * 8 * (256 * 128 * 3 * 64 / 128);
    try wideChunkProof(std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest), &.{&.{ 64, 64, 33 }}, cache, 20 * std.time.ns_per_ms, "ReadsFillingSlab");
}

test "dsv41 engine: after the load's wide warm, prompts cut at other widths compile no kernel, through the union (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const dsv41_model = @import("dsv41_model.zig");
    const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was);
    var fx = try dsv41_model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    var sc = @import("dsv41_scope.zig").Scope.plain(a, mlx.gpuStream());
    defer sc.deinit();
    const ids = try dsv41_model.tokens(&sc, &fx);
    defer a.free(ids);
    var config = try model_mod.parseConfig(io, a, dir);
    defer config.deinit(a);
    config.expert_streaming = true;
    config.expert_cache_bytes = 3 * 8 * (256 * 128 * 3 * 64 / 128);
    config.expert_source_dir = try a.dupe(u8, dir);
    var weights = try model_mod.loadWeightsForConfig(io, a, dir, &config, false);
    defer weights.deinit();
    var xfm = try Trunk.init(io, a, config, &weights);
    defer xfm.deinit();
    const m = dsv41_state.modelOf(&xfm).?;
    m.engine_decode = true;
    m.engine_prefill = true;
    dsv41_borrow.own_override = true;
    defer {
        m.engine_decode = false;
        m.engine_prefill = false;
        dsv41_borrow.own_override = null;
    }
    try dsv41_serving.warmAllAt(&xfm, m, dsv41_state.stqExpertsOf(&xfm), .all, 256);
    mlx.engine.seal(true);
    defer mlx.engine.seal(false);
    const late = mlx.engine.latePipelines();
    var prompt: [1100]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = ids[(i * 5 + 3) % ids.len];
    for ([_][]const usize{ &.{ 64, 64, 33 }, &.{25}, &.{ 90, 31 }, &.{ 200, 31 }, &.{128}, &.{ 256, 256 } }) |chunks| {
        var run = try wideRun(a, &xfm, &prompt, chunks, 3);
        run.deinit(a);
        errdefer std.debug.print("after the chunks {any}\n", .{chunks});
        try testing.expectEqual(late, mlx.engine.latePipelines());
    }
}

test "dsv41 engine: an image warm and an image prompt whose routes read experts take the engine's order on a Cooler load, with MLX's logits (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    if (!mlx.engine.matrixUnits()) return error.SkipZigTest;
    const a = testing.allocator;
    const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    // A cache this small misses at every forward, so each route fills slabs before its kernels read them.
    const small = 3 * 8 * (256 * 128 * 3 * 64 / 128);
    var want: std.ArrayList(u8) = .empty;
    defer want.deinit(a);
    {
        var ref: EngineFixture = undefined;
        try ref.openSized(dir, .flag, small);
        defer ref.deinit();
        var image = try ImageStandIn.give(a, &ref);
        defer image.drop();
        want = try imagePromptBytes(a, &ref, image.rows, &.{40}, 3, false);
    }
    var fx: EngineFixture = undefined;
    try fx.openSized(dir, null, small);
    defer fx.deinit();
    var image = try ImageStandIn.give(a, &fx);
    defer image.drop();
    // The load's image warm calls the model outside a request's scope, on the engine's stream all the same: its wide
    // routes take the engine's order, a group's kernels after that group has landed, never MLX's.
    try @import("dsv41_model.zig").warmOwnImages(fx.model());
    try testing.expectEqual(@as(u64, 0), dsv41_state.stqExpertsOf(&fx.xfm).?.mlx_order_routes);
    var got = try imagePromptBytes(a, &fx, image.rows, &.{40}, 3, true);
    defer got.deinit(a);
    try testing.expectEqualSlices(u8, want.items, got.items);
    var said: [256]u8 = undefined;
    try testing.expectEqualStrings("", mlx.takeError(&said) orelse "");
}

test "dsv41 engine: the load's warm runs every engine forward, decode-width and wide, under a serve forward's budget, and one that does not fit fails the load by name (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    const a = std.heap.smp_allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was);
    var config = try model_mod.parseConfig(io, a, dir);
    config.expert_streaming = true;
    config.expert_cache_bytes = 3 * 8 * (256 * 128 * 3 * 64 / 128);
    config.expert_source_dir = try a.dupe(u8, dir);
    var weights = try model_mod.loadWeightsForConfig(io, a, dir, &config, false);
    var xfm = try Trunk.init(io, a, config, &weights);
    const m = dsv41_state.modelOf(&xfm).?;
    const stq = dsv41_state.stqExpertsOf(&xfm);
    m.engine_decode = true;
    dsv41_borrow.own_override = true;
    defer {
        m.engine_decode = false;
        m.engine_prefill = false;
        dsv41_borrow.own_override = null;
        dsv41_borrow.warm_budget = false;
        dsv41_borrow.warm_budget_override = null;
        mlx.engine.active = false;
        // The budget is the engine's for the process: the tests after this one run with none.
        mlx.engine.budget(std.math.maxInt(u64)) catch {};
    }
    // The budget serve gives each forward covers the tiny model's forwards.
    try dsv41_serving.warmAllAt(&xfm, m, stq, .all, 256);
    // The wide warm installs a wide chunk's budget before its first work on the engine: the one a decode-width forward
    // left before it (a few KB here) covers no chunk.
    m.engine_prefill = true;
    dsv41_borrow.warm_budget = true;
    try mlx.engine.budget(16 << 10);
    try dsv41_serving.warmWide(&xfm, m, stq, 256);
    // No budget fits: the decode-width forwards fail the load, wide chunks or not.
    dsv41_borrow.warm_budget_override = 0;
    try testing.expectError(error.PoolOverBudget, dsv41_serving.warmAllAt(&xfm, m, stq, .all, 256));
    // And the wide chunks do when the decode-width forwards come from a manifest.
    m.engine_prefill = true;
    dsv41_borrow.warm_budget = true;
    try testing.expectError(error.MlxError, dsv41_serving.warmWide(&xfm, m, stq, 256));
    try testing.expect(mlx.takeErrorIf("PoolOverBudget"));
}

test "dsv41 engine: a prompt prefilled in chunks on the engine of our own, with restore points and state-only chunks, gives MLX's tokens and restore points (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    const dsv41_model = @import("dsv41_model.zig");
    const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was);
    var fx = try dsv41_model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    var sc = @import("dsv41_scope.zig").Scope.plain(a, mlx.gpuStream());
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
    var xfm = try Trunk.init(io, a, config, &weights);
    defer xfm.deinit();
    const m = dsv41_state.modelOf(&xfm).?;
    // Wider than a decode-width pass, as the release's window is: a narrower replay reads the ring below its floor.
    m.cfg.window = 20;
    var tok = Tokenizer.initEmptyForTests(a);
    defer tok.deinit();
    var prompt: [1400]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = ids[(i * 5 + 3) % ids.len];
    const saved_chunk = settings_resolve.prefill_chunk.value;
    defer settings_resolve.prefill_chunk.value = saved_chunk;
    settings_resolve.prefill_chunk.value = 256;
    const opts: Generator.InitOptions = .{
        .skip_lazy_preforward = true,
        .pinned_prefill_chunk = 256,
        .adaptive_chunk_width = false,
        .ssm_checkpoint_stride = 512,
        .ssm_checkpoint_max = 8,
        .skip_unread_tails = true,
    };
    var runs: [2]struct { tokens: std.ArrayList(u32) = .empty, points: std.ArrayList([32]u8) = .empty, positions: std.ArrayList(usize) = .empty, state_only: u64 = 0 } = .{ .{}, .{} };
    defer for (&runs) |*r| {
        r.tokens.deinit(a);
        r.points.deinit(a);
        r.positions.deinit(a);
    };
    for (&runs, [_]bool{ false, true }) |*r, on_engine| {
        m.engine_decode = on_engine;
        m.engine_prefill = on_engine;
        defer {
            m.engine_decode = false;
            m.engine_prefill = false;
        }
        try xfm.resetCache();
        const only_before = m.state_only_used;
        var gen = try Generator.initWithOptions(io, a, &xfm, &tok, &prompt, 12, .{ .temperature = 0.0 }, &.{}, opts);
        defer gen.deinit(a);
        try r.tokens.append(a, gen.next_token_id);
        while (try gen.next(a)) |t| try r.tokens.append(a, t);
        r.state_only = m.state_only_used - only_before;
        for (gen.ssm_checkpoints.items) |*cp| {
            try dsv41_prefix.restore(m, cp, &prompt);
            try r.points.append(a, try stateHash(m));
            try r.positions.append(a, cp.pos);
        }
    }
    try testing.expect(runs[0].state_only > 0);
    try testing.expectEqual(runs[0].state_only, runs[1].state_only);
    try testing.expect(runs[0].points.items.len >= 2);
    try testing.expectEqualSlices(usize, runs[0].positions.items, runs[1].positions.items);
    try testing.expectEqualSlices(u32, runs[0].tokens.items, runs[1].tokens.items);
    for (runs[0].points.items, runs[1].points.items) |want, got| try testing.expectEqualSlices(u8, &want, &got);
}

test "dsv41 engine: after the load's warm a sampled DSpark request, seeded or not, compiles no kernel, at any filter setting or with a suppress mask (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    const dsv41_model = @import("dsv41_model.zig");
    const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was);
    var fx = try dsv41_model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    var sc = @import("dsv41_scope.zig").Scope.plain(a, mlx.gpuStream());
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
    var xfm = try Trunk.init(io, a, config, &weights);
    defer xfm.deinit();
    const m = dsv41_state.modelOf(&xfm).?;
    try testing.expect(m.dspark != null);
    var tok = Tokenizer.initEmptyForTests(a);
    defer tok.deinit();
    m.engine_decode = true;
    dsv41_borrow.own_override = true;
    defer {
        m.engine_decode = false;
        dsv41_borrow.own_override = null;
        mlx.engine.budget(std.math.maxInt(u64)) catch {};
    }
    try dsv41_serving.warmAll(&xfm, m, null, .all);
    mlx.engine.seal(true);
    defer mlx.engine.seal(false);
    const late = mlx.engine.latePipelines();
    const flags = try a.alloc(bool, m.cfg.vocab);
    defer a.free(flags);
    for (flags, 0..) |*f, id| f.* = id % 7 == 0;
    const mask = mlx.mlx_array_new_data(flags.ptr, &[_]c_int{@intCast(m.cfg.vocab)}, 1, .bool_);
    defer _ = mlx.mlx_array_free(mask);
    for ([_]?u64{ 7, null }) |seed| for ([_]bool{ false, true }) |masked| for ([_][2]f32{ .{ 0, 1 }, .{ 40, 1 }, .{ 0, 0.95 }, .{ 40, 0.95 } }) |filter| {
        const sampled = SamplingParams{ .temperature = 0.8, .top_k = @intFromFloat(filter[0]), .top_p = filter[1], .seed = seed };
        try xfm.resetCache();
        var gen = try Generator.initWithOptions(io, a, &xfm, &tok, ids[0..30], 30, sampled, &.{}, .{ .dspark_enabled = true });
        defer gen.deinit(a);
        if (masked) gen.sampling.suppress_mask = mask;
        gen.dspark_diag = .sweep;
        while (try gen.nextDspark(a)) |r| a.free(r.tokens);
        try testing.expectEqual(late, mlx.engine.latePipelines());
    };
    // The serial tick, the first token and a round with penalties launch no pipeline the warm did not make.
    const serial = [_]struct { sampling: SamplingParams, logprobs: u32 = 0, masked: bool = false }{
        .{ .sampling = .{ .temperature = 0.0 } },
        .{ .sampling = .{ .temperature = 0.0 }, .masked = true },
        .{ .sampling = .{ .temperature = 0.8, .top_k = 40, .top_p = 0.95, .seed = 7 }, .masked = true },
        .{ .sampling = .{ .temperature = 0.8, .top_k = 40, .seed = 7 } },
        .{ .sampling = .{ .temperature = 0.8, .top_p = 0.9 } },
        .{ .sampling = .{ .temperature = 0.8, .repeat_penalty = 1.2, .seed = 7 } },
        .{ .sampling = .{ .temperature = 0.0, .presence_penalty = 0.5, .frequency_penalty = 0.3 } },
        .{ .sampling = .{ .temperature = 0.8, .top_k = 40, .top_p = 0.95, .seed = 7, .repeat_penalty = 1.2, .presence_penalty = 0.5 }, .logprobs = 3 },
        .{ .sampling = .{ .temperature = 0.0 }, .logprobs = 3 },
    };
    for (serial) |c| for ([_]bool{ true, false }) |skip_lazy| {
        try xfm.resetCache();
        var gen = try Generator.initWithOptions(io, a, &xfm, &tok, ids[0..30], 12, c.sampling, &.{}, .{ .skip_lazy_preforward = skip_lazy, .logprobs_n = c.logprobs });
        defer gen.deinit(a);
        gen.logprobs_n = c.logprobs;
        if (c.masked) gen.sampling.suppress_mask = mask;
        while (try gen.next(a)) |_| {}
        try testing.expectEqual(late, mlx.engine.latePipelines());
    };
    for ([_]SamplingParams{
        .{ .temperature = 0.8, .repeat_penalty = 1.2, .seed = 7 },
        .{ .temperature = 0.0, .presence_penalty = 0.5, .frequency_penalty = 0.3 },
    }) |c| {
        try xfm.resetCache();
        var gen = try Generator.initWithOptions(io, a, &xfm, &tok, ids[0..30], 30, c, &.{}, .{ .dspark_enabled = true });
        defer gen.deinit(a);
        gen.dspark_diag = .sweep;
        while (try gen.nextDspark(a)) |r| a.free(r.tokens);
        try testing.expectEqual(late, mlx.engine.latePipelines());
    }
}

test "dsv41 engine: a one-row decode on the engine of our own gives MLX's tokens and logits bit for bit (DSV41_STQ_TINY)" {
    try engineDecodeProof(std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest), true);
}

test "dsv41 generator: a thinking bound fires between DSpark rounds, commits its closer and decodes the rest serially (DSV41_TINY)" {
    try mlx.requireGpu();
    const dsv41_model = @import("dsv41_model.zig");
    const dir = try dsv41_model.fixtureDir();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var fx = try dsv41_model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    var sc = @import("dsv41_scope.zig").Scope.plain(a, mlx.gpuStream());
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
    var xfm = try Trunk.init(io, a, config, &weights);
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
    xfm: *Trunk,
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
    var sc = @import("dsv41_scope.zig").Scope.plain(a, mlx.gpuStream());
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
    var xfm = try Trunk.init(io, a, config, &weights);
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
        dsv41_model.setDsparkExact(dsv41_state.modelOf(&xfm).?, false);
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
        dsv41_model.setDsparkExact(dsv41_state.modelOf(&xfm).?, true);
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
    var sc = @import("dsv41_scope.zig").Scope.plain(a, mlx.gpuStream());
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

    var xfm = try Trunk.init(io, a, config, &weights);
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
    try dsv41_prefix.restore(dsv41_state.modelOf(&xfm).?, &gen.ssm_checkpoints.items[0], &next);
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
    var sc = @import("dsv41_scope.zig").Scope.plain(a, mlx.gpuStream());
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
    const saved_chunk = settings_resolve.prefill_chunk.value;
    defer settings_resolve.prefill_chunk.value = saved_chunk;
    settings_resolve.prefill_chunk.value = 256;
    const base: Generator.InitOptions = .{ .skip_lazy_preforward = true, .pinned_prefill_chunk = 256, .adaptive_chunk_width = false, .ssm_checkpoint_stride = 256, .ssm_checkpoint_max = 4 };

    var want: std.ArrayList(u32) = .empty;
    defer want.deinit(a);
    {
        // The load takes each tensor out of the map, so each model reads its own.
        var weights = try model_mod.loadWeightsForConfig(io, a, dir, &config, false);
        defer weights.deinit();
        var xfm = try Trunk.init(io, a, config, &weights);
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
    var xfm = try Trunk.init(io, a, config, &weights);
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
    xfm: Trunk,
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
        self.xfm = try Trunk.init(io, a, config, &self.weights);
        errdefer self.xfm.deinit();
        const m = dsv41_state.modelOf(&self.xfm).?;
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
        const m = dsv41_state.modelOf(&self.xfm).?;
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
    var sc = @import("dsv41_scope.zig").Scope.plain(a, mlx.gpuStream());
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
    const saved_chunk = settings_resolve.prefill_chunk.value;
    defer settings_resolve.prefill_chunk.value = saved_chunk;
    settings_resolve.prefill_chunk.value = 256;

    // Offset 0: chunks end at 256, 512, 768 and 1369 (a merged tail); points at 512 and the prompt end. The chunks
    // ending at 256 and 768 leave state only. Offset 246: the chunk ending at 502 is followed by one of ten rows,
    // which a skipped tail would leave stale; only the chunk ending at 768 is skipped.
    for ([_]struct { offset: usize, skipped: u64 }{ .{ .offset = 0, .skipped = 2 }, .{ .offset = 246, .skipped = 1 } }) |c| {
        const full = try TailArm.start(a, config, dir, &tok, &prompt, c.offset, false);
        defer full.destroy(a);
        const lean = try TailArm.start(a, config, dir, &tok, &prompt, c.offset, true);
        defer lean.destroy(a);
        try testing.expectEqual(@as(u64, 0), dsv41_state.modelOf(&full.xfm).?.state_only_used);
        try testing.expectEqual(c.skipped, dsv41_state.modelOf(&lean.xfm).?.state_only_used);
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
    var sc = @import("dsv41_scope.zig").Scope.plain(a, mlx.gpuStream());
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
    const saved_chunk = settings_resolve.prefill_chunk.value;
    defer settings_resolve.prefill_chunk.value = saved_chunk;
    settings_resolve.prefill_chunk.value = 256;

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
    var xfm = try Trunk.init(io, a, config, &weights);
    defer xfm.deinit();
    var gen = try Generator.initWithOptions(io, a, &xfm, &tok, &prompt, 6, greedy, &.{}, opts);
    defer gen.deinit(a);
    try testing.expect(count.n > 4);
    // Every forward but the first, the final span included, started with the rows read during the one before it.
    try testing.expectEqual(@as(u64, count.n - 1), dsv41_state.modelOf(&xfm).?.engram_ahead_used);
    try testing.expect(dsv41_state.modelOf(&xfm).?.engram_ahead == null);
}

test "dsv41 generator: after a DSpark request the slot cache marks its rows and snapshots like the scheduler's commit (DSV41_TINY)" {
    try mlx.requireGpu();
    const dsv41_model = @import("dsv41_model.zig");
    const dir = try dsv41_model.fixtureDir();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var fx = try dsv41_model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    var sc = @import("dsv41_scope.zig").Scope.plain(a, mlx.gpuStream());
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
    var xfm = try Trunk.init(io, a, config, &weights);
    defer xfm.deinit();
    var tok = Tokenizer.initEmptyForTests(a);
    defer tok.deinit();
    // The scheduler's slot owns a cache with one entry per layer; the Trunk's own is a shell.
    // `initSlot` hands every slot the engine's default KV scheme, whatever the pack reports.
    var slot_cache = try KVCache.initWithConfig(a, config.num_hidden_layers, trunk_mod.KVQuantConfig.engine_default);
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
    try dsv41_prefix.markRows(&slot_cache, mlx.gpuStream());
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
        var sc = @import("dsv41_scope.zig").Scope.plain(a, mlx.gpuStream());
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
    var xfm = try Trunk.init(io, a, st.config, &st.weights);
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
    var xfm = try Trunk.init(io, a, st.config, &st.weights);
    defer xfm.deinit();
    var tok = Tokenizer.initEmptyForTests(a);
    defer tok.deinit();
    var g = try Generator.initWithOptions(io, a, &xfm, &tok, st.ids[0..30], 8, .{ .temperature = 0.0 }, &.{}, .{ .dspark_enabled = true });
    defer g.deinit(a);
    g.dspark_diag = .{ .rows = 4 };

    // The draft heads score the whole vocabulary, so the image id is one of their possible picks; this pack's
    // image id is made an id the draft already holds past row 1.
    const dsv = dsv41_state.modelOf(&xfm).?;
    var d = try dsv41_serving.dsv41DraftImpl(&xfm, g.next_token_id);
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
    xfm: *Trunk,
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
    var sc = @import("dsv41_scope.zig").Scope.plain(a, mlx.gpuStream());
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
    var xfm = try Trunk.init(io, a, config, &weights);
    defer xfm.deinit();
    try testing.expect(dsv41_state.modelOf(&xfm).?.dspark != null);
    var tok = Tokenizer.initEmptyForTests(a);
    defer tok.deinit();
    const greedy = SamplingParams{ .temperature = 0.0 };
    const prompt = ids[0..30];
    const want: u32 = 48;
    const vocab = dsv41_state.modelOf(&xfm).?.cfg.vocab;

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
    dsv41_model.setDsparkExact(dsv41_state.modelOf(&xfm).?, false);
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
    dsv41_model.setDsparkExact(dsv41_state.modelOf(&xfm).?, true);
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
    dsv41_model.setDsparkExact(dsv41_state.modelOf(&xfm).?, false);
    var serial = try grammarRun(a, io, &xfm, &tok, prompt, want, &forced_tb, closer, eos_in_reasoning, greedy, .serial);
    defer serial.deinit(a);
    var plain = try grammarRun(a, io, &xfm, &tok, prompt, want, &forced_tb, closer, eos_in_reasoning, greedy, .{ .spark = .{ .rows = 1 } });
    defer plain.deinit(a);
    try testing.expectEqualSlices(u32, serial.tokens.items, plain.tokens.items);
    dsv41_model.setDsparkExact(dsv41_state.modelOf(&xfm).?, true);

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
    fn forward(xfm: *Trunk, ids: []const u32, plain: bool) !A {
        const m = dsv41_state.modelOf(xfm).?;
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

    fn at(a: std.mem.Allocator, xfm: *Trunk, prompt: []const u32, cont: []const u32, W: usize, verbose: bool) !usize {
        const m = dsv41_state.modelOf(xfm).?;
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
    var xfm = try Trunk.init(io, a, config, &weights);
    defer xfm.deinit();
    const m = dsv41_state.modelOf(&xfm).?;
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

test "dsv41 engine: on the real pack one-row steps and DSpark rounds on the engine give MLX's bytes (SUSHI_DSV41_REAL_PACK)" {
    try mlx.requireGpu();
    const dir = std.mem.span(std.c.getenv("SUSHI_DSV41_REAL_PACK") orelse return error.SkipZigTest);
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const was = mlx.engine.enabled;
    mlx.engine.enabled = true;
    defer test_support.restoreEngine(was);
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
    var xfm = try Trunk.init(io, a, config, &weights);
    defer xfm.deinit();
    const m = dsv41_state.modelOf(&xfm).?;
    try testing.expect(m.dspark != null);
    var tok = try tokenizer_mod.loadTokenizer(io, a, dir);
    defer tok.deinit();
    const prompt = try tok.encode(a, "Write a long, detailed story about a lighthouse keeper and the storms of one winter. The keeper climbed the iron stairs each night, counting the steps as the wind rose over the black water.");
    defer a.free(prompt);
    const rounds = 3 * @as(usize, dsv41_max_verify_rows);
    var want_rounds = try dsparkRoundBytes(a, &xfm, prompt, rounds);
    defer want_rounds.deinit(a);
    var want_steps = try lazyHeadSteps(a, &xfm, prompt, 24, false);
    defer want_steps.deinit(a);
    m.engine_decode = true;
    defer m.engine_decode = false;
    const before = mlx.engine.stats();
    var got_rounds = try dsparkRoundBytes(a, &xfm, prompt, rounds);
    defer got_rounds.deinit(a);
    var got_steps = try lazyHeadSteps(a, &xfm, prompt, 24, false);
    defer got_steps.deinit(a);
    try testing.expect(mlx.engine.stats().dispatches > before.dispatches);
    try testing.expectEqualSlices(u32, want_steps.tokens.items, got_steps.tokens.items);
    try testing.expect(std.mem.eql(u8, std.mem.sliceAsBytes(want_steps.logits.items), std.mem.sliceAsBytes(got_steps.logits.items)));
    try testing.expect(std.mem.eql(u8, want_rounds.items, got_rounds.items));
}

/// A cache of 1.5 MB is under the tiny pack's experts, so the load streams; 64 MB holds them all.
fn prefetchAt(dir: []const u8, budget_gb: u64, pct: ?u8, cache_bytes: u64) !bool {
    try mlx.requireGpu();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var config = try model_mod.parseConfig(io, a, dir);
    defer config.deinit(a);
    config.expert_streaming = true;
    config.expert_cache_bytes = cache_bytes;
    config.expert_ssd_budget_bytes = budget_gb << 30;
    config.ssd_prefetch = pct;
    config.expert_source_dir = try a.dupe(u8, dir);
    var weights = try model_mod.loadWeightsForConfig(io, a, dir, &config, false);
    defer weights.deinit();
    var xfm = try Trunk.init(io, a, config, &weights);
    defer xfm.deinit();
    const on = xfm.expert_stream.?.prefetcher != null;
    try testing.expectEqual(on, dsv41_state.modelOf(&xfm).?.prefetch_lead_rows != 0);
    return on;
}

test "dsv41 prefetch: --ssd-prefetch 0 and the default create no prefetch machinery, any other value does on a streamed load at any budget (DSV41_STQ_TINY)" {
    const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    const streamed: u64 = 3 << 19;
    for ([_]u64{ 1, 107, 108 }) |budget| {
        try testing.expect(!try prefetchAt(dir, budget, 0, streamed));
        try testing.expect(try prefetchAt(dir, budget, 65, streamed));
        try testing.expect(!try prefetchAt(dir, budget, null, streamed));
    }
    try testing.expect(try prefetchAt(dir, 1, 1, streamed));
    try testing.expect(!try prefetchAt(dir, 108, 65, 64 << 20));
}

test "every generator start path copies the launch round options (DSV41_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var st = try TinyDsparkNet.init(a, io);
    defer st.deinit(a);
    var xfm = try Trunk.init(io, a, st.config, &st.weights);
    defer xfm.deinit();
    var tok = Tokenizer.initEmptyForTests(a);
    defer tok.deinit();
    const saved_lookup = generate.mtp_lookup_disabled;
    const saved_ctx = generate.max_mtp_ctx;
    defer {
        generate.mtp_lookup_disabled = saved_lookup;
        generate.max_mtp_ctx = saved_ctx;
    }
    generate.mtp_lookup_disabled = true;
    generate.max_mtp_ctx = 77;
    const starts = [_]Generator.InitOptions{ .{}, .{ .skip_lazy_preforward = true }, .{ .dspark_enabled = true } };
    for (starts) |opts| {
        var g = try Generator.initWithOptions(io, a, &xfm, &tok, st.ids[0..30], 8, .{ .temperature = 0.0 }, &.{}, opts);
        defer g.deinit(a);
        try testing.expect(g.opts.lookup_disabled);
        try testing.expectEqual(@as(u32, 77), g.opts.max_mtp_ctx);
    }
}

test "dsv41 generator: a DFlash round is refused by name, not asserted (DSV41_TINY)" {
    try mlx.requireGpu();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var st = try TinyDsparkNet.init(a, io);
    defer st.deinit(a);
    var xfm = try Trunk.init(io, a, st.config, &st.weights);
    defer xfm.deinit();
    var tok = Tokenizer.initEmptyForTests(a);
    defer tok.deinit();
    var g = try Generator.initWithOptions(io, a, &xfm, &tok, st.ids[0..30], 8, .{ .temperature = 0.0 }, &.{}, .{});
    defer g.deinit(a);
    try testing.expectError(error.DrafterUnsupported, g.nextDflash(a));
}

test "dsv41 engine: a decode forward, a draft and a verify round of the same width build no kernel instance key once they have run (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    const kn = @import("../tensor/kernel.zig");
    var fx: EngineFixture = undefined;
    try fx.open(dir);
    defer fx.deinit();
    const xfm = &fx.xfm;
    try xfm.resetCache();
    var ctx = xfm.defaultCtx();
    var lg = try forwardIds(xfm, &ctx, fx.ids[0..30]);
    var tok = try argmaxHost(xfm, lg);
    _ = mlx.mlx_array_free(lg);
    var steady: [3]u64 = undefined;
    // The first of each kind is the one that fills the kernels' memos; the key builds of the second are counted.
    for (0..2) |round| {
        const before = kn.key_builds;
        lg = try forwardIds(xfm, &ctx, &.{tok});
        tok = try argmaxHost(xfm, lg);
        _ = mlx.mlx_array_free(lg);
        steady[0] = kn.key_builds - before;
        const draft_before = kn.key_builds;
        var d = try dsv41_serving.dsv41DraftImpl(xfm, tok);
        defer d.deinit();
        steady[1] = kn.key_builds - draft_before;
        const verify_before = kn.key_builds;
        const logits = try dsv41_serving.dsv41VerifyImpl(xfm, &ctx, d.ids[0..4]);
        defer _ = mlx.mlx_array_free(logits);
        try mlx.check(mlx.mlx_array_eval(logits));
        steady[2] = kn.key_builds - verify_before;
        try dsv41_serving.dsv41FinishImpl(xfm, &ctx, 1);
        tok = d.ids[1];
        _ = round;
    }
    try testing.expectEqual([3]u64{ 0, 0, 0 }, steady);
}

const CallKind = enum { config, query, read, eval, make, free, op };
const call_kinds = @intFromEnum(CallKind.op) + 1;

fn callKind(name: []const u8) CallKind {
    const has = std.mem.startsWith;
    if (has(u8, name, "mlx_fast_metal_kernel")) return .config;
    for ([_][]const u8{ "mlx_array_data_", "mlx_array_item_" }) |p| if (has(u8, name, p)) return .read;
    for ([_][]const u8{ "mlx_array_ndim", "mlx_array_shape", "mlx_array_size", "mlx_array_dtype", "mlx_array_strides", "mlx_array_itemsize", "_mlx_array_" }) |p| if (has(u8, name, p)) return .query;
    if (std.mem.endsWith(u8, name, "_free")) return .free;
    for ([_][]const u8{ "mlx_array_eval", "mlx_eval", "mlx_async_eval", "mlx_synchronize" }) |p| if (std.mem.eql(u8, name, p)) return .eval;
    if (std.mem.indexOf(u8, name, "_new") != null) return .make;
    return .op;
}

/// The calls into mlx-c made since `before`, by kind.
fn callsSince(before: []const mlx.NameCount) ![call_kinds]u64 {
    var now: std.ArrayList(mlx.NameCount) = .empty;
    defer now.deinit(testing.allocator);
    try mlx.callsByName(&now, testing.allocator);
    var split: [call_kinds]u64 = @splat(0);
    for (now.items) |n| {
        var was: u64 = 0;
        for (before) |b| if (std.mem.eql(u8, b.name, n.name)) {
            was = b.count;
        };
        if (!std.mem.startsWith(u8, n.name, "lend ")) split[@intFromEnum(callKind(n.name))] += n.count - was;
    }
    return split;
}

/// The ops that took a weight the engine wraps on MLX's side since `before`, as `op xN` joined by spaces.
fn lentSince(before: []const mlx.NameCount, buf: []u8) ![]const u8 {
    var now: std.ArrayList(mlx.NameCount) = .empty;
    defer now.deinit(testing.allocator);
    try mlx.callsByName(&now, testing.allocator);
    var w: std.Io.Writer = .fixed(buf);
    for (now.items) |n| {
        if (!std.mem.startsWith(u8, n.name, "lend ")) continue;
        var was: u64 = 0;
        for (before) |b| if (std.mem.eql(u8, b.name, n.name)) {
            was = b.count;
        };
        if (n.count > was) try w.print("{s} x{d} ", .{ n.name[5..], n.count - was });
    }
    return w.buffered();
}

const StepCalls = struct { total: u64, split: [call_kinds]u64 };

/// The calls into MLX the `steps`-th one-row step makes after a prompt of 30 tokens, greedy: the step is the token
/// array, the forward and the read of the argmax.
fn stepCalls(fx: *EngineFixture, prompt: []const u32, steps: usize) !StepCalls {
    try fx.xfm.resetCache();
    var ctx = fx.xfm.defaultCtx();
    _ = mlx.mlx_array_free(try forwardIds(&fx.xfm, &ctx, prompt[0..30]));
    var tok = prompt[30];
    var last: StepCalls = undefined;
    for (0..steps) |_| {
        var before: std.ArrayList(mlx.NameCount) = .empty;
        defer before.deinit(testing.allocator);
        try mlx.callsByName(&before, testing.allocator);
        const total = mlx.engine.mlxCalls();
        const lg = try forwardIds(&fx.xfm, &ctx, &.{tok});
        tok = try argmaxHost(&fx.xfm, lg);
        _ = mlx.mlx_array_free(lg);
        last = .{ .total = mlx.engine.mlxCalls() - total, .split = try callsSince(before.items) };
    }
    return last;
}

test "dsv41 engine: a one-row decode step in steady state makes no call into MLX (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    if (!mlx.engine.matrixUnits()) return error.SkipZigTest;
    const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    var fx: EngineFixture = undefined;
    try fx.open(dir);
    defer fx.deinit();
    try testing.expect(fx.model().engine_decode);
    var buf: [48]u32 = undefined;
    const prompt = fx.prompt(&buf);
    // The first run fills the expert cache and the pools; the next two walk the same tokens and must agree. The prompt's
    // prefill is the engine's, as a request's is.
    _ = try stepCalls(&fx, prompt, 5);
    const steady = try stepCalls(&fx, prompt, 5);
    const again = try stepCalls(&fx, prompt, 5);
    try testing.expectEqual(steady, again);
    // Zero, stage by stage: kernel configs, array queries, host reads, evals, makers, frees, other ops.
    try testing.expectEqual([_]u64{ 0, 0, 0, 0, 0, 0, 0 }, steady.split);
    try testing.expectEqual(@as(u64, 0), steady.total);
}

/// The ops a request puts on MLX in `want` serial ticks, MLX's keep-warm touch held off.
fn opsOver(m: *@import("dsv41_model.zig").Model, gen: *Generator, a: std.mem.Allocator, want: usize) !u64 {
    const before = mlx.engine.mlx_ops;
    for (0..want) |_| {
        m.mlx_queue_used_ns = mlx.engine.nowNs();
        _ = try gen.next(a);
    }
    return mlx.engine.mlx_ops - before;
}

test "dsv41 engine: a request's first token and every serial step put no op on MLX, greedy, seeded, penalized, with logprobs or a mask (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    if (!mlx.engine.matrixUnits()) return error.SkipZigTest;
    const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    var fx: EngineFixture = undefined;
    try fx.open(dir);
    defer fx.deinit();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var tok = Tokenizer.initEmptyForTests(a);
    defer tok.deinit();
    var buf: [48]u32 = undefined;
    const prompt = fx.prompt(&buf)[0..30];
    const m = fx.model();
    const flags = try a.alloc(bool, m.cfg.vocab);
    defer a.free(flags);
    for (flags, 0..) |*f, id| f.* = id % 7 == 0;
    const mask = mlx.mlx_array_new_data(flags.ptr, &[_]c_int{@intCast(m.cfg.vocab)}, 1, .bool_);
    defer _ = mlx.mlx_array_free(mask);
    const filtered: SamplingParams = .{ .temperature = 0.8, .top_p = 0.9, .top_k = 20, .seed = 5 };
    const cases = [_]struct { sampling: SamplingParams, logprobs: u32 = 0, masked: bool = false, opts: Generator.InitOptions = .{} }{
        .{ .sampling = .{ .temperature = 0.0 } },
        .{ .sampling = .{ .temperature = 0.0 }, .opts = .{ .skip_lazy_preforward = true } },
        .{ .sampling = .{ .temperature = 0.0 }, .masked = true },
        .{ .sampling = filtered },
        .{ .sampling = filtered, .opts = .{ .skip_lazy_preforward = true }, .masked = true },
        .{ .sampling = .{ .temperature = 0.8 } },
        .{ .sampling = .{ .temperature = 0.8, .repeat_penalty = 1.3, .presence_penalty = 0.5, .seed = 5 }, .opts = .{ .skip_lazy_preforward = true } },
        .{ .sampling = .{ .temperature = 0.0, .frequency_penalty = 0.4 }, .opts = .{ .skip_lazy_preforward = true } },
        .{ .sampling = .{ .temperature = 0.0 }, .logprobs = 3, .opts = .{ .skip_lazy_preforward = true } },
        .{ .sampling = filtered, .logprobs = 3, .opts = .{ .skip_lazy_preforward = true } },
    };
    for (cases) |c| {
        var opts = c.opts;
        opts.logprobs_n = c.logprobs;
        // The first token: the prefill is MLX's, what it is sampled and read with is not.
        {
            try fx.xfm.resetCache();
            var ctx = fx.xfm.defaultCtx();
            const lg = try forwardIds(&fx.xfm, &ctx, prompt);
            defer _ = mlx.mlx_array_free(lg);
            try mlx.check(mlx.mlx_array_eval(lg));
            var sampling = c.sampling;
            sampling.draw_root = 5;
            if (c.masked) sampling.suppress_mask = mask;
            const before = mlx.engine.mlx_ops;
            const lazy = try sampler.firstTokenSample(&fx.xfm, lg, sampling, fx.xfm.outStream());
            defer _ = mlx.mlx_array_free(lazy);
            try mlx.check(mlx.mlx_array_eval(lazy));
            var first: i32 = 0;
            try mlx.check(mlx.mlx_array_item_int32(&first, lazy));
            if (c.logprobs > 0) {
                const lp = (try sampler.firstTokenLogprobs(a, lg, @intCast(first), c.logprobs, fx.xfm.outStream())).?;
                a.free(lp.top_logprobs);
            }
            try testing.expectEqual(before, mlx.engine.mlx_ops);
        }
        try fx.xfm.resetCache();
        var gen = try Generator.initWithOptions(io, a, &fx.xfm, &tok, prompt, 12, c.sampling, &.{}, opts);
        defer gen.deinit(a);
        gen.logprobs_n = c.logprobs;
        if (c.masked) gen.sampling.suppress_mask = mask;
        _ = try opsOver(m, &gen, a, 2);
        try testing.expectEqual(@as(u64, 0), try opsOver(m, &gen, a, 8));
    }
}

test "dsv41 engine: a DSpark round with penalties, a thinking penalty, a logit bias or a row mask puts no op on MLX (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    if (!mlx.engine.matrixUnits()) return error.SkipZigTest;
    const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    var fx: EngineFixture = undefined;
    try fx.open(dir);
    defer fx.deinit();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var tok = Tokenizer.initEmptyForTests(a);
    defer tok.deinit();
    var buf: [48]u32 = undefined;
    const prompt = fx.prompt(&buf)[0..30];
    const m = fx.model();
    const think_ids = [_]u32{ 11, 12, 13 };
    fx.xfm.think_marker_mask = try @import("../think_penalty.zig").buildMask(a, &think_ids, m.cfg.vocab);
    defer {
        _ = mlx.mlx_array_free(fx.xfm.think_marker_mask.?);
        fx.xfm.think_marker_mask = null;
    }
    const bias = [_]@import("../logit_bias.zig").Bias{.{ .id = 14, .delta = 1.5, .scope = .reasoning }};
    const think: SamplingParams = .{ .temperature = 0.8, .seed = 3, .think_penalty = .{ .lambda = 2, .opener_id = 5, .closer_id = 6, .phase = .inside } };
    var biased = think;
    biased.think_penalty.biases = &bias;
    const cases = [_]SamplingParams{
        .{ .temperature = 0.8, .repeat_penalty = 1.3, .seed = 5 },
        .{ .temperature = 0.0, .frequency_penalty = 0.4, .presence_penalty = 0.2 },
        think,
        biased,
    };
    for (cases) |c| {
        try fx.xfm.resetCache();
        var gen = try Generator.initWithOptions(io, a, &fx.xfm, &tok, prompt, 30, c, &.{}, .{ .dspark_enabled = true });
        defer gen.deinit(a);
        gen.dspark_diag = .sweep;
        if (try gen.nextDspark(a)) |r| a.free(r.tokens);
        const before = mlx.engine.mlx_ops;
        var rounds: usize = 0;
        while (rounds < 5) : (rounds += 1) {
            m.mlx_queue_used_ns = mlx.engine.nowNs();
            const r = (try gen.nextDspark(a)) orelse break;
            a.free(r.tokens);
        }
        try testing.expect(rounds > 0);
        try testing.expectEqual(before, mlx.engine.mlx_ops);
    }
}

/// One request on the engine through `mode`, its ids.
const Mode = enum { serial, serial_lazy, serial_logprobs, rows1, rows3, sweep, adaptive };

fn engineIds(fx: *EngineFixture, tok: *Tokenizer, prompt: []const u32, want: u32, sampling: SamplingParams, mode: Mode, out: *std.ArrayList(u32)) !void {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    try fx.xfm.resetCache();
    const opts: Generator.InitOptions = switch (mode) {
        .serial => .{ .skip_lazy_preforward = true },
        .serial_lazy => .{},
        .serial_logprobs => .{ .skip_lazy_preforward = true, .logprobs_n = 3 },
        else => .{ .dspark_enabled = true },
    };
    var gen = try Generator.initWithOptions(io, a, &fx.xfm, tok, prompt, want, sampling, &.{}, opts);
    defer gen.deinit(a);
    switch (mode) {
        .serial, .serial_lazy => while (try gen.next(a)) |t| try out.append(a, t),
        .serial_logprobs => {
            gen.logprobs_n = 3;
            while (try gen.next(a)) |t| try out.append(a, t);
        },
        else => {
            gen.dspark_diag = switch (mode) {
                .rows1 => .{ .rows = 1 },
                .rows3 => .{ .rows = 3 },
                .sweep => .sweep,
                else => .off,
            };
            while (try gen.nextDspark(a)) |r| {
                defer a.free(r.tokens);
                try out.appendSlice(a, r.tokens);
            }
        },
    }
}

test "dsv41 engine: a seeded request gives the same tokens, the first one included, at every serial mix and at every DSpark round-width mix, streamed or not, and twice (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    if (!mlx.engine.matrixUnits()) return error.SkipZigTest;
    const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    var fx: EngineFixture = undefined;
    try fx.open(dir);
    defer fx.deinit();
    const a = testing.allocator;
    var tok = Tokenizer.initEmptyForTests(a);
    defer tok.deinit();
    var buf: [48]u32 = undefined;
    const prompt = fx.prompt(&buf)[0..30];
    const want: u32 = 40;
    const seeded: SamplingParams = .{ .temperature = 0.8, .top_k = 40, .top_p = 0.95, .seed = 7 };
    const penalized: SamplingParams = .{ .temperature = 0.8, .repeat_penalty = 1.3, .presence_penalty = 0.5, .seed = 7 };
    // A one-row decode and a one-row verify differ in the last bits of their logits, so a serial request and a round
    // request are two texts of one seed; each is the same at every mix of its own kind. The first token is one function's.
    const groups = [_][]const Mode{ &.{ .serial, .serial_lazy, .serial_logprobs }, &.{ .rows1, .rows3, .sweep, .adaptive } };
    for ([_]SamplingParams{ seeded, penalized }) |sampling| {
        var first: ?u32 = null;
        for (groups) |modes| {
            var ref: std.ArrayList(u32) = .empty;
            defer ref.deinit(a);
            for (modes) |mode| for (0..2) |_| {
                var got: std.ArrayList(u32) = .empty;
                defer got.deinit(a);
                try engineIds(&fx, &tok, prompt, want, sampling, mode, &got);
                try testing.expectEqual(@as(usize, want), got.items.len);
                if (ref.items.len == 0) try ref.appendSlice(a, got.items) else try testing.expectEqualSlices(u32, ref.items, got.items);
            };
            if (first) |f| try testing.expectEqual(f, ref.items[0]) else first = ref.items[0];
            // A request that stops sooner is a prefix, first token and all.
            var short: std.ArrayList(u32) = .empty;
            defer short.deinit(a);
            try engineIds(&fx, &tok, prompt, 17, sampling, modes[0], &short);
            try testing.expectEqualSlices(u32, ref.items[0..17], short.items);
            // Another seed, and requests with none, draw other tokens.
            var other: std.ArrayList(u32) = .empty;
            defer other.deinit(a);
            var changed = sampling;
            changed.seed = 8;
            try engineIds(&fx, &tok, prompt, want, changed, modes[0], &other);
            try testing.expect(!std.mem.eql(u32, ref.items, other.items));
            var unseeded: [2]std.ArrayList(u32) = .{ .empty, .empty };
            defer for (&unseeded) |*r| r.deinit(a);
            changed.seed = null;
            for (&unseeded) |*r| try engineIds(&fx, &tok, prompt, want, changed, modes[0], r);
            try testing.expect(!std.mem.eql(u32, unseeded[0].items, unseeded[1].items));
        }
    }
}

test "dsv41 engine: the serial sampler and a DSpark row draw one token for one position, at every filter, with a mask and under a penalty (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    if (!mlx.engine.matrixUnits()) return error.SkipZigTest;
    const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    var fx: EngineFixture = undefined;
    try fx.open(dir);
    defer fx.deinit();
    const vocab = fx.model().cfg.vocab;
    const flags = try testing.allocator.alloc(bool, vocab);
    defer testing.allocator.free(flags);
    for (flags, 0..) |*f, id| f.* = id % 7 == 0;
    const mask = mlx.mlx_array_new_data(flags.ptr, &[_]c_int{@intCast(vocab)}, 1, .bool_);
    defer _ = mlx.mlx_array_free(mask);
    const host = try testing.allocator.alloc(f32, vocab);
    defer testing.allocator.free(host);
    const s = mlx.engine.stream();
    var distinct = std.AutoHashMap(i32, void).init(testing.allocator);
    defer distinct.deinit();
    for ([_][2]f32{ .{ 0, 1 }, .{ 40, 1 }, .{ 0, 0.95 }, .{ 40, 0.95 } }) |filter| for ([_]bool{ false, true }) |masked| for (0..24) |position| {
        for (host, 0..) |*v, i| v.* = @as(f32, @floatFromInt((i * 2654435761 + position * 40503) % 1000)) / 250.0;
        const logits = mlx.mlx_array_new_data(host.ptr, &[_]c_int{ 1, 1, @intCast(vocab) }, 3, .float32);
        defer _ = mlx.mlx_array_free(logits);
        const sampling = SamplingParams{ .temperature = 0.8, .top_k = @intFromFloat(filter[0]), .top_p = filter[1], .draw = position, .draw_root = 11, .suppress_mask = if (masked) mask else null };
        const serial = sampler.sampleTokenLazy(logits, sampling, s);
        defer _ = mlx.mlx_array_free(serial);
        const row = try dsv41_serving.sampledDraws(sampling, logits, 1, s);
        defer _ = mlx.mlx_array_free(row);
        try mlx.check(mlx.mlx_array_eval(serial));
        try mlx.check(mlx.mlx_array_eval(row));
        var a: i32 = 0;
        var b: i32 = 0;
        try mlx.check(mlx.mlx_array_item_int32(&a, serial));
        try mlx.check(mlx.mlx_array_item_int32(&b, row));
        try testing.expectEqual(a, b);
        try distinct.put(a, {});
    };
    // The positions do not all draw one token.
    try testing.expect(distinct.count() > 3);
}

/// The ids of a request on the MLX route of the tiny STQ load (`engine_decode` off): the serial tick, or DSpark rounds of three rows.
fn mlxRouteIds(fx: *EngineFixture, sampling: SamplingParams, rounds: bool, out: *std.ArrayList(u32)) !void {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var tok = Tokenizer.initEmptyForTests(a);
    defer tok.deinit();
    var buf: [48]u32 = undefined;
    const prompt = fx.prompt(&buf)[0..30];
    try fx.xfm.resetCache();
    var gen = try Generator.initWithOptions(io, a, &fx.xfm, &tok, prompt, 12, sampling, &.{}, if (rounds) .{ .dspark_enabled = true } else .{ .skip_lazy_preforward = true });
    defer gen.deinit(a);
    if (rounds) gen.dspark_diag = .{ .rows = 3 };
    if (rounds) {
        while (try gen.nextDspark(a)) |r| {
            defer a.free(r.tokens);
            try out.appendSlice(a, r.tokens);
        }
    } else {
        while (try gen.next(a)) |t| try out.append(a, t);
    }
}

test "dsv41 engine: the MLX route's seeded ids, serial and in DSpark rounds, are what they were before the engine took the sampler (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    if (!mlx.engine.matrixUnits()) return error.SkipZigTest;
    const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    var fx: EngineFixture = undefined;
    try fx.openMlx(dir);
    defer fx.deinit();
    const seeded = SamplingParams{ .temperature = 0.8, .top_k = 40, .top_p = 0.95, .seed = 7 };
    const penalized = SamplingParams{ .temperature = 0.8, .repeat_penalty = 1.3, .presence_penalty = 0.5, .seed = 7 };
    const Case = struct { sampling: SamplingParams, rounds: bool, ids: [12]u32 };
    const cases = [_]Case{
        .{ .sampling = seeded, .rounds = false, .ids = .{ 116, 250, 45, 163, 22, 139, 130, 76, 119, 34, 40, 71 } },
        .{ .sampling = penalized, .rounds = false, .ids = .{ 116, 250, 45, 163, 22, 139, 130, 76, 16, 207, 237, 146 } },
        .{ .sampling = seeded, .rounds = true, .ids = .{ 116, 250, 45, 163, 22, 139, 130, 76, 188, 93, 74, 185 } },
        .{ .sampling = .{ .temperature = 0.0 }, .rounds = false, .ids = .{ 116, 250, 208, 47, 8, 159, 222, 227, 76, 78, 3, 57 } },
    };
    for (cases) |c| {
        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(testing.allocator);
        try mlxRouteIds(&fx, c.sampling, c.rounds, &ids);
        try testing.expectEqualSlices(u32, &c.ids, ids.items);
    }
}

/// The calls into MLX's `name` since `before`.
fn namedSince(before: []const mlx.NameCount, name: []const u8) !u64 {
    var now: std.ArrayList(mlx.NameCount) = .empty;
    defer now.deinit(testing.allocator);
    try mlx.callsByName(&now, testing.allocator);
    var count: u64 = 0;
    for (now.items) |n| {
        if (std.mem.eql(u8, n.name, name)) count += n.count;
    }
    for (before) |b| {
        if (std.mem.eql(u8, b.name, name)) count -= b.count;
    }
    return count;
}

test "dsv41 engine: a request's rings are the engine's from its first forward: MLX makes none, and no step asks MLX who holds one (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    if (!mlx.engine.matrixUnits()) return error.SkipZigTest;
    const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    var fx: EngineFixture = undefined;
    try fx.open(dir);
    defer fx.deinit();
    var buf: [48]u32 = undefined;
    const prompt = fx.prompt(&buf);
    _ = try stepCalls(&fx, prompt, 2);
    var before: std.ArrayList(mlx.NameCount) = .empty;
    defer before.deinit(testing.allocator);
    try mlx.callsByName(&before, testing.allocator);
    // A new request: its state, its prefill and five steps.
    _ = try stepCalls(&fx, prompt, 5);
    try testing.expectEqual(@as(u64, 0), try namedSince(before.items, "mlx_zeros"));
    try testing.expectEqual(@as(u64, 0), try namedSince(before.items, "_mlx_array_is_donatable"));
}

/// The first `n` greedy tokens after a prompt of 30.
fn greedyRun(fx: *EngineFixture, prompt: []const u32, comptime n: usize) ![n]u32 {
    try fx.xfm.resetCache();
    var ctx = fx.xfm.defaultCtx();
    var lg = try forwardIds(&fx.xfm, &ctx, prompt[0..30]);
    var out: [n]u32 = undefined;
    for (&out) |*tok| {
        tok.* = try argmaxHost(&fx.xfm, lg);
        _ = mlx.mlx_array_free(lg);
        lg = try forwardIds(&fx.xfm, &ctx, &.{tok.*});
    }
    _ = mlx.mlx_array_free(lg);
    return out;
}

test "dsv41 engine: a Cooler load wraps its weights itself: the MLX route's greedy tokens, steps that import no weight and lend none, and an unload that unmaps them (DSV41_STQ_TINY)" {
    try mlx.requireGpu();
    if (!mlx.engine.matrixUnits()) return error.SkipZigTest;
    const dir = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    var buf: [48]u32 = undefined;
    var want: [8]u32 = undefined;
    {
        // The MLX route, whose weights MLX wraps.
        var fx: EngineFixture = undefined;
        try fx.openAsking(dir, .flag);
        defer fx.deinit();
        want = try greedyRun(&fx, fx.prompt(&buf), 8);
    }
    mlx.engine.close();
    const regions = model_mod.ownedRegionCount();
    const mapped = mlx.engine.stats().mapped_bytes;
    var names: std.ArrayList(mlx.NameCount) = .empty;
    defer names.deinit(testing.allocator);
    try mlx.callsByName(&names, testing.allocator);
    const weight_lends = mlx.engine.stats().mapped_lends;
    {
        var fx: EngineFixture = undefined;
        try fx.open(dir);
        defer fx.deinit();
        try testing.expect(mlx.engine.stats().mapped_bytes > mapped);
        const prompt = fx.prompt(&buf);
        try testing.expectEqual(want, try greedyRun(&fx, prompt, 8));
        _ = try stepCalls(&fx, prompt, 5);
        const imports = mlx.engine.stats().imports;
        _ = try stepCalls(&fx, prompt, 5);
        // From the load on: no op on MLX's side took a weight the engine wraps.
        var lent: [1024]u8 = undefined;
        try testing.expectEqualStrings("", try lentSince(names.items, &lent));
        try testing.expectEqual(weight_lends, mlx.engine.stats().mapped_lends);
        // A prefill and five steps import nothing: the token arrays are the engine's own.
        try testing.expectEqual(@as(u64, 0), mlx.engine.stats().imports - imports);
    }
    mlx.engine.close();
    try testing.expectEqual(regions, model_mod.ownedRegionCount());
    try testing.expectEqual(mapped, mlx.engine.stats().mapped_bytes);
    // Nothing the load, the steps or the unload refused was swallowed.
    var said: [512]u8 = undefined;
    try testing.expectEqualStrings("", mlx.takeError(&said) orelse "");
}
