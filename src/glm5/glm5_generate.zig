const std = @import("std");
const mlx = @import("../mlx.zig");
const transformer_mod = @import("../transformer.zig");
const tokenizer_mod = @import("../tokenizer.zig");
const model_mod = @import("../model.zig");
const mtp_lookup = @import("../mtp_lookup.zig");
const round_cost = @import("../round_cost.zig");
const dflash_policy = @import("../dflash_policy.zig");
const logit_bias = @import("../logit_bias.zig");
const Transformer = transformer_mod.Transformer;
const Tokenizer = tokenizer_mod.Tokenizer;
const SSMCheckpoint = transformer_mod.SSMCheckpoint;
const glm5_forward = @import("glm5_forward.zig");
const glm5_prefix = @import("glm5_prefix.zig");
const kv_disk_cache_mod = @import("../kv_disk_cache.zig");
const KVCache = transformer_mod.KVCache;
const testing = std.testing;
const expectError = @import("../test_expect.zig").expectError;
const mtp_plan = @import("../mtp_plan.zig");
const AdaptiveWidthState = mtp_plan.AdaptiveWidthState;
const ThoughtPoint = mtp_plan.ThoughtPoint;
const draftsRefused = mtp_plan.draftsRefused;
const mtpRefusal = mtp_plan.mtpRefusal;
const mtpUnsupported = mtp_plan.mtpUnsupported;
const specDecodeUnsupported = mtp_plan.specDecodeUnsupported;
const prefill_plan = @import("../prefill_plan.zig");
const SSM_SNAPSHOT_BACKOFF = prefill_plan.SSM_SNAPSHOT_BACKOFF;
const nextChunkEnd = prefill_plan.nextChunkEnd;
const ssmSnapshotBackoff = prefill_plan.ssmSnapshotBackoff;
const sampler = @import("../sampler.zig");
const SamplingParams = sampler.SamplingParams;
const generate = @import("../generate.zig");
const CallForce = generate.CallForce;
const Generator = generate.Generator;
const glmDflashEligible = generate.glmDflashEligible;
const glmLookupProposal = generate.glmLookupProposal;

/// `nextChunkEnd` for a GLM prefill, whose checkpoint grid must be hit exactly: the tail merge never
/// absorbs a grid point.
pub fn glmChunkEnd(pos: usize, prefix_len: usize, default_chunk: usize, grid: usize, offset: usize, adaptive_width: bool) usize {
    // Grid 0 = no checkpoints for this request (prefix cache off, or admission shed them all).
    if (grid == 0) return nextChunkEnd(pos, prefix_len, default_chunk, false, 0, offset, adaptive_width);
    const next_grid = ((pos + offset) / grid + 1) * grid - offset;
    return @min(nextChunkEnd(pos, prefix_len, default_chunk, true, grid, offset, adaptive_width), next_grid);
}

/// GLM restores only on an IndexPool boundary, so its prompt-end snapshot backs off up to 3
/// more tokens to land on an absolute multiple of 4.
pub fn glmSnapshotBackoff(backoff: usize, offset: usize, prefix_len: usize) usize {
    if (backoff >= prefix_len) return backoff;
    return @min(backoff + (offset + prefix_len - backoff) % glm5_prefix.pool_size, prefix_len);
}

/// What a GLM prefill captures: the grid points inside `(offset, loop end]`, then one checkpoint at the
/// loop's end. Admission's bill and the prefill loop both read it.
pub const GlmCaptureSchedule = struct {
    /// Tail tokens held back from the chunk loop for the final span.
    backoff: usize,
    /// Checkpoints taken over the request.
    events: u32,
    /// Most held at once: one past the cap before each thin.
    peak: u32,
    /// Absolute position of the newest checkpoint, 0 = none.
    newest: usize,
};

/// The GLM prefill's capture schedule for a tail of `tail_len` tokens starting at absolute `offset`;
/// `cap` 0 thins nothing.
pub fn glmCaptureSchedule(grid: usize, cap: u32, offset: usize, tail_len: usize) GlmCaptureSchedule {
    var out: GlmCaptureSchedule = .{ .backoff = 0, .events = 0, .peak = 0, .newest = 0 };
    if (tail_len <= 1) return out;
    const prefix_len = tail_len - 1;
    out.backoff = glmSnapshotBackoff(ssmSnapshotBackoff(true, prefix_len, offset > 0), offset, prefix_len);
    const loop_end = prefix_len - out.backoff;
    if (loop_end == 0) return out;
    const end = offset + loop_end;
    var events: usize = end / grid - offset / grid;
    out.newest = end / grid * grid;
    // A grid point at the loop's end is already captured by the chunk loop.
    if (end % grid != 0) {
        events += 1;
        out.newest = end;
    }
    out.events = @intCast(@min(events, std.math.maxInt(u32)));
    out.peak = if (cap > 0) @min(out.events, cap + 1) else out.events;
    return out;
}

/// GLM restores only on an IndexPool boundary; every other arch at any checkpoint position.
pub fn restorableCheckpoint(glm: ?*glm5_forward.Request, pos: usize) bool {
    return glm == null or pos % glm5_prefix.pool_size == 0;
}

// DFlash has different economics from sequential-token drafters: one
// assistant call proposes the whole block and the M5 NAX verify lane makes
// wide blocks cheap. Per-draft percentage therefore penalizes block 16
// even when it is decisively profitable. The stable cross-width signal is
// accepted drafts per verify round. M5 sweeps put code/tool traffic at
// 4.4-15.0 accepted/round and the regressing prose/vision class at
// 1.0-1.5; two is the measured break-even boundary. M4 block-5 model-card
// workloads accepting 62-86% remain above it as well.
// Warmup 8, not 3 (2026-09-03): three rounds tripped on ONE dry opening
// (llmprobe's 16k rung: 0/3, drafter off for the whole request at 25
// tok/s; the same prompt restored from cache decoded at 73).
pub const DFLASH_GATE_WARMUP: u64 = 8;
/// A DFlash round that costs more per emitted token than this multiple of a serial step is dropped.
pub const DFLASH_GATE_MARGIN: f32 = 0.05;

/// Accepted drafts per round at which a round of `rows` verifier rows costs one serial step per
/// token, used while the request's context bucket holds no serial sample. The policy's own cost ladder.
pub fn dflashBreakEvenAccepted(rows: f32) f32 {
    return dflash_policy.breakEvenDrafts(rows);
}

/// Does the request's DFlash yield lose to serial decoding? With a measured serial price (the
/// round-cost table's cell) the test is ms per emitted token; without one it is the ladder's
/// break-even yield. Never decided before the warmup rounds.
pub fn dflashGateShouldDisable(attempted: u64, accepted: u64, wall_ms: f64, rows: f32, serial_ms_tok: ?f32) bool {
    if (attempted < DFLASH_GATE_WARMUP) return false;
    if (serial_ms_tok) |serial| {
        if (std.math.isFinite(serial) and serial > 0) {
            const emitted: f64 = @floatFromInt(attempted + accepted);
            return wall_ms / emitted > @as(f64, serial) * (1.0 + DFLASH_GATE_MARGIN);
        }
    }
    const avg = @as(f32, @floatFromInt(accepted)) / @as(f32, @floatFromInt(attempted));
    return avg < dflashBreakEvenAccepted(rows);
}

/// The rounds the gate judges: those after the warmup. The warmup's own wall carries one-time costs (graph builds,
/// first touches of a long context) that would decide a request the steady rounds win.
pub const DflashGateWindow = struct {
    open: bool = false,
    attempted: u64 = 0,
    accepted: u64 = 0,
    wall_ms: f64 = 0,

    pub fn shouldDisable(self: *DflashGateWindow, attempted: u64, accepted: u64, wall_ms: f64, rows: f32, serial_ms_tok: ?f32) bool {
        if (attempted < DFLASH_GATE_WARMUP) return false;
        if (!self.open) {
            self.* = .{ .open = true, .attempted = attempted, .accepted = accepted, .wall_ms = wall_ms };
            return false;
        }
        return dflashGateShouldDisable(attempted - self.attempted, accepted - self.accepted, wall_ms - self.wall_ms, rows, serial_ms_tok);
    }
};

/// The serial price a DFlash request is held to: the plain-tick cell at width 0, which this
/// path's own fallback ticks feed, else the serial row MTP's adaptive serial feeds.
pub fn dflashSerialPrice(t: *const round_cost.Table, bucket: usize) ?f32 {
    return t.msPerTok(0, bucket) orelse t.serialMsPerTok(bucket);
}

test "a repeat or presence penalty keeps DFlash and the drafter off and samples on the synchronous path" {
    try testing.expect(specDecodeUnsupported(.{ .presence_penalty = 0.5 }, 0));
    try testing.expect(specDecodeUnsupported(.{ .repeat_penalty = 1.1 }, 0));
    try testing.expect(draftsRefused(.{ .presence_penalty = 0.5 }));
    try testing.expect(!draftsRefused(.{}));
    var g: Generator = undefined;
    g.logprobs_n = 0;
    g.sampling = .{ .presence_penalty = 0.5 };
    try testing.expect(g.samplesSync());
    g.sampling = .{ .repeat_penalty = 1.2 };
    try testing.expect(g.samplesSync());
    g.sampling = .{};
    try testing.expect(!g.samplesSync());
    g.logprobs_n = 2;
    try testing.expect(g.samplesSync());
}

test "a penalty keeps DFlash and the drafter off but not MTP" {
    try testing.expect(!mtpUnsupported(.{ .presence_penalty = 1.5 }, 0));
    try testing.expect(!mtpUnsupported(.{ .repeat_penalty = 1.1, .presence_penalty = 0.5 }, 0));
    try testing.expect(mtpUnsupported(.{ .presence_penalty = 1.5 }, 2));
    var cf = CallForce{ .forced = &.{1}, .call_at = 0, .closer_id = null, .phase = .answer, .last = 0 };
    try testing.expect(mtpUnsupported(.{ .call_force = &cf, .presence_penalty = 1.5 }, 0));
    try testing.expect(specDecodeUnsupported(.{ .presence_penalty = 1.5 }, 0));
    try testing.expect(mtpRefusal(.{ .presence_penalty = 1.5 }, 0) == null);
    try testing.expectEqualStrings("logprobs requested", mtpRefusal(.{}, 1).?);
    try testing.expectEqualStrings("forced tool call", mtpRefusal(.{ .call_force = &cf }, 0).?);
}

test "Generator.dflashGateShouldDisable: a serial price decides by ms per token, the ladder break-even decides without one" {
    const G = Generator;
    const w = G.DFLASH_GATE_WARMUP;
    const round_ms: f64 = 30.6; // a 4-row round, Sushi-2.6bpw M5 Max
    // Never decided before the warmup rounds.
    try testing.expect(!G.dflashGateShouldDisable(w - 1, 0, round_ms * @as(f64, @floatFromInt(w - 1)), 4.0, 13.3));
    // Against a 13.3 ms serial token: one accepted draft per round costs 15.3 ms per token and loses,
    // one and a half costs 12.2 and wins.
    try testing.expect(G.dflashGateShouldDisable(w, w, round_ms * @as(f64, @floatFromInt(w)), 4.0, 13.3));
    try testing.expect(!G.dflashGateShouldDisable(w, w + w / 2, round_ms * @as(f64, @floatFromInt(w)), 4.0, 13.3));
    // A slower serial token (a larger pack) keeps rounds the 13.3 ms one would drop.
    try testing.expect(!G.dflashGateShouldDisable(w, w, round_ms * @as(f64, @floatFromInt(w)), 4.0, 20.0));
    // No serial sample: the ladder's break-even yield, 0.69 accepted drafts at 4 rows.
    try testing.expectApproxEqAbs(@as(f32, 0.69), G.dflashBreakEvenAccepted(4.0), 0.01);
    try testing.expect(G.dflashGateShouldDisable(w, w / 2, 0, 4.0, null));
    try testing.expect(!G.dflashGateShouldDisable(w, w * 9 / 10, 0, 4.0, null));
    // A round never offers fewer accepted drafts than its break-even asks for, up to eight rows.
    for ([_]f32{ 3, 4, 5, 6, 8 }) |rows| try testing.expect(G.dflashBreakEvenAccepted(rows) < rows - 1);
    // Rows the policy left out of a round are not a cost: 1.3 accepted per round at five rows on average
    // clears the bar, and the same yield at the eight-row block does not.
    const rounds = 3 * w;
    try testing.expect(!G.dflashGateShouldDisable(rounds, rounds * 13 / 10, 0, 5.0, null));
    try testing.expect(G.dflashGateShouldDisable(rounds, rounds * 13 / 10, 0, 8.0, null));
    // Normandy, 15k context: 1.18 accepted per round over 3.3 rows at 15.26 ms per token, against 17.74 ms serial,
    // is a request DFlash wins, whether or not the bucket held a serial price when the gate looked.
    const emitted: f64 = 1370 + 1617;
    try testing.expect(!G.dflashGateShouldDisable(1370, 1617, 15.26 * emitted, 3.33, 17.74));
    try testing.expect(!G.dflashGateShouldDisable(1370, 1617, 15.26 * emitted, 3.33, null));
}

test "Generator.DflashGateWindow: one-time costs in the warmup rounds do not decide the request, a steady loser is still dropped" {
    const G = Generator;
    const w: u64 = G.DFLASH_GATE_WARMUP;
    const serial: f32 = 14.5;
    // Warmup rounds of 80 ms (graph builds), then 30 ms rounds that emit 3.4 tokens: 8.8 ms per token.
    var win = G.DflashGateWindow{};
    var wall: f64 = 80.0 * @as(f64, @floatFromInt(w));
    try testing.expect(!win.shouldDisable(w, 2 * w, wall, 5.0, serial));
    var n: u64 = w;
    while (n < 4 * w) : (n += 1) {
        wall += 30.0;
        try testing.expect(!win.shouldDisable(n + 1, 2 * w + (n + 1 - w) * 5 / 2, wall, 5.0, serial));
    }
    // The cumulative price over the same rounds is above the serial token, so the unwindowed test dropped it.
    try testing.expect(G.dflashGateShouldDisable(w, 2 * w, 80.0 * @as(f64, @floatFromInt(w)), 5.0, serial));
    // Rounds of 60 ms emitting 2 tokens (30 ms per token) lose once a warmup's worth of them has run.
    var loser = G.DflashGateWindow{};
    var lw: f64 = 30.0 * @as(f64, @floatFromInt(w));
    try testing.expect(!loser.shouldDisable(w, w, lw, 5.0, serial));
    var dropped_at: u64 = 0;
    n = w;
    while (n < 4 * w) : (n += 1) {
        lw += 60.0;
        if (loser.shouldDisable(n + 1, w + (n + 1 - w), lw, 5.0, serial)) {
            dropped_at = n + 1;
            break;
        }
    }
    try testing.expectEqual(2 * w, dropped_at);
}

test "Generator.stopDflash: the drafter's window is released and no later tick can reach it" {
    var g: Generator = undefined;
    g.spec_disabled_runtime = false;
    g.dflash_ctx = .{ .cache = try KVCache.init(testing.allocator, 5), .base_pos = 128 };
    g.stopDflash();
    try testing.expect(g.spec_disabled_runtime);
    try testing.expect(g.dflash_ctx == null);
    g.stopDflash();
    try testing.expect(g.dflash_ctx == null);
}

test "Generator.dflashSerialPrice prices a context bucket by its own cell, never a shorter bucket's" {
    var t = round_cost.Table{ .layout = .long };
    const short_kv: u32 = 24000;
    const long_kv: u32 = 90000;
    for (0..round_cost.MIN_SAMPLES) |_| _ = t.observe(0, short_kv, 17.74, 1, true, false);
    try testing.expectApproxEqAbs(@as(f32, 17.74), Generator.dflashSerialPrice(&t, t.bucketOf(short_kv)).?, 1e-3);
    try testing.expect(Generator.dflashSerialPrice(&t, t.bucketOf(long_kv)) == null);
    for (0..round_cost.MIN_SAMPLES) |_| _ = t.observe(0, long_kv, 30.2, 1, true, false);
    try testing.expectApproxEqAbs(@as(f32, 30.2), Generator.dflashSerialPrice(&t, t.bucketOf(long_kv)).?, 1e-3);
    try testing.expectApproxEqAbs(@as(f32, 17.74), Generator.dflashSerialPrice(&t, t.bucketOf(short_kv)).?, 1e-3);
}

test "Generator.dflashSerialPrice reads the width-0 cell the DFlash fallback feeds, then the serial row" {
    var t = round_cost.Table{ .layout = .long };
    try testing.expect(Generator.dflashSerialPrice(&t, 0) == null);
    for (0..round_cost.MIN_SAMPLES) |_| _ = t.observe(0, 1000, 13.4, 1, true, false);
    try testing.expectApproxEqAbs(@as(f32, 13.4), Generator.dflashSerialPrice(&t, 0).?, 1e-3);
    var row = round_cost.Table{ .layout = .long };
    for (0..round_cost.MIN_SAMPLES) |_| _ = row.observeSerial(1000, 15.0, true, false);
    try testing.expectApproxEqAbs(@as(f32, 15.0), Generator.dflashSerialPrice(&row, 0).?, 1e-3);
}

/// The checkpoints a GLM prefill takes, replayed from the loop's own chunk ends (test oracle for `glmCaptureSchedule`).
pub fn glmCaptureOracle(chunk_grid: usize, grid: usize, cap: u32, offset: usize, tail_len: usize, width: usize) GlmCaptureSchedule {
    var r: GlmCaptureSchedule = .{ .backoff = 0, .events = 0, .peak = 0, .newest = 0 };
    if (tail_len <= 1) return r;
    const prefix_len = tail_len - 1;
    r.backoff = glmSnapshotBackoff(ssmSnapshotBackoff(true, prefix_len, offset > 0), offset, prefix_len);
    const loop_end = prefix_len - r.backoff;
    var held: u32 = 0;
    var pos: usize = 0;
    while (pos < loop_end) {
        pos = glmChunkEnd(pos, loop_end, width, chunk_grid, offset, true);
        if ((pos + offset) % grid != 0) continue;
        r.events += 1;
        r.newest = pos + offset;
        held += 1;
        r.peak = @max(r.peak, held);
        if (cap > 0 and held > cap) held -= 1;
    }
    if (loop_end > 0 and r.newest != loop_end + offset) {
        r.events += 1;
        r.newest = loop_end + offset;
        held += 1;
        r.peak = @max(r.peak, held);
    }
    return r;
}

test "--no-mtp-lookup: a GLM DFlash2 lookup chain is declined too" {
    const off = generate.mtp_lookup_disabled;
    defer generate.mtp_lookup_disabled = off;
    generate.mtp_lookup_disabled = true;
    var g: Generator = undefined;
    try testing.expect((try g.glmLookupChain(testing.allocator)) == null);
}

test "a GLM lookup chain needs a long verbatim match and the budget for all four rows" {
    const draft = [_]u32{ 7, 8, 9, 10 };
    const strong = mtp_lookup.Match{ .draft = &draft, .suffix = mtp_lookup.STRONG_SUFFIX };
    const chain = glmLookupProposal(strong, 6, 64).?;
    try testing.expectEqualSlices(u32, &.{ 6, 7, 8, 9 }, chain.tokens[0..chain.count]);
    try testing.expectEqualSlices(i32, &.{ -1, 0, 1, 2 }, chain.parents[0..chain.count]);
    try testing.expect(glmLookupProposal(null, 6, 64) == null);
    try testing.expect(glmLookupProposal(.{ .draft = &draft, .suffix = mtp_lookup.STRONG_SUFFIX - 1 }, 6, 64) == null);
    try testing.expect(glmLookupProposal(.{ .draft = draft[0..2], .suffix = mtp_lookup.STRONG_SUFFIX }, 6, 64) == null);
    try testing.expect(glmLookupProposal(strong, 6, 3) == null);
}

test "GLM serving DFlash2 arms only requests the native verifier can reproduce" {
    try testing.expect(glmDflashEligible(.{ .temperature = 0 }, 0));
    try testing.expect(glmDflashEligible(.{ .temperature = 1, .top_k = 1 }, 0));
    try testing.expect(glmDflashEligible(.{ .temperature = 1 }, 0));
    try testing.expect(!glmDflashEligible(.{ .temperature = 0 }, 1));
    try testing.expect(!glmDflashEligible(.{ .temperature = 0, .presence_penalty = 1 }, 0));
    const ban = [_]logit_bias.Bias{.{ .id = 7, .delta = -100 }};
    try testing.expect(glmDflashEligible(.{ .temperature = 0, .think_penalty = .{ .biases = &ban } }, 0));
    try testing.expect(glmDflashEligible(.{ .temperature = 1, .think_penalty = .{ .file_biases = &ban } }, 0));
    try testing.expect(glmDflashEligible(.{ .temperature = 0, .think_penalty = .{ .lambda = 1, .closer_id = 3, .phase = .inside } }, 0));
}

test "GLM snapshot backoff lands the prompt-end checkpoint on a pool boundary" {
    for ([_]usize{ 0, 256, 1024 }) |offset| {
        for ([_]usize{ 31, 32, 33, 34, 35, 100, 299, 4097 }) |prefix_len| {
            const back = glmSnapshotBackoff(ssmSnapshotBackoff(true, prefix_len, offset > 0), offset, prefix_len);
            try testing.expect(back >= SSM_SNAPSHOT_BACKOFF and back <= SSM_SNAPSHOT_BACKOFF + 3 or back == prefix_len);
            if (back < prefix_len) try testing.expectEqual(@as(usize, 0), (offset + prefix_len - back) % 4);
        }
    }
    // A restored tail inside the window still forwards as one span.
    try testing.expectEqual(@as(usize, 12), glmSnapshotBackoff(ssmSnapshotBackoff(true, 12, true), 256, 12));
    // A short cold prompt still ends its loop on a pool boundary.
    try testing.expectEqual(@as(usize, 1), glmSnapshotBackoff(ssmSnapshotBackoff(true, 13, false), 0, 13));
}

test "GLM short prompts and odd chunk widths still checkpoint only on pool boundaries" {
    try mlx.requireGpu();
    const a = testing.allocator;
    var tok = Tokenizer.initEmptyForTests(a);
    defer tok.deinit();
    var weights = model_mod.Weights.init(a);
    defer weights.deinit();
    const cfg = try glmGeneratorFixture(&weights);
    const greedy = SamplingParams{ .temperature = 0.0 };
    var prompt: [150]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast((i * 5 + 1) % 4);
    generate.glm_checkpoint_stride = 64;
    generate.glm_capture_stride = 64;
    defer generate.glm_checkpoint_stride = glm5_forward.prefill_chunk;
    defer generate.glm_capture_stride = kv_disk_cache_mod.POINT_GRID;
    for ([_]struct { len: usize, chunk: usize }{ .{ .len = 14, .chunk = 64 }, .{ .len = 150, .chunk = 37 } }) |case| {
        var xfm = try Transformer.init(testing.io, a, cfg, &weights);
        defer xfm.deinit();
        var gen = try Generator.initWithOptions(testing.io, a, &xfm, &tok, prompt[0..case.len], 2, greedy, &.{}, .{ .skip_lazy_preforward = true, .pinned_prefill_chunk = case.chunk, .adaptive_chunk_width = true, .ssm_checkpoint_stride = 1, .ssm_checkpoint_max = 16 });
        defer gen.deinit(a);
        try testing.expect(gen.ssm_checkpoints.items.len > 0);
        for (gen.ssm_checkpoints.items) |cp| try testing.expectEqual(@as(usize, 0), cp.pos % 4);
    }
}

test "an empty prompt is a typed error at the generator, never an index" {
    try mlx.requireGpu();
    const a = testing.allocator;
    var tok = Tokenizer.initEmptyForTests(a);
    defer tok.deinit();
    var weights = model_mod.Weights.init(a);
    defer weights.deinit();
    const cfg = try glmGeneratorFixture(&weights);
    var xfm = try Transformer.init(testing.io, a, cfg, &weights);
    defer xfm.deinit();
    try expectError(error.EmptyPrompt, Generator.initWithOptions(testing.io, a, &xfm, &tok, &.{}, 4, .{ .temperature = 0.0 }, &.{}, .{}));
}

fn glmGeneratorFixture(weights: *model_mod.Weights) !model_mod.ModelConfig {
    var cfg = try glm5_forward.nonzeroDecodeFixture(weights);
    cfg.max_position_embeddings = 4096;
    return cfg;
}

test "GLM prefill checkpoints sit on chunk boundaries, and a restore there prefills its suffix exactly like cold" {
    try mlx.requireGpu();
    const a = testing.allocator;
    var tok = Tokenizer.initEmptyForTests(a);
    defer tok.deinit();
    var first: [300]u32 = undefined;
    for (&first, 0..) |*t, i| t.* = @intCast((i * 7 + i / 5) % 4);
    var next: [330]u32 = undefined;
    @memcpy(next[0..290], first[0..290]);
    for (next[290..], 0..) |*t, i| t.* = @intCast((i * 3 + 1) % 4);
    const greedy = SamplingParams{ .temperature = 0.0 };
    const opts: Generator.InitOptions = .{ .skip_lazy_preforward = true, .pinned_prefill_chunk = 64, .adaptive_chunk_width = true, .ssm_checkpoint_stride = 64, .ssm_checkpoint_max = 16 };
    generate.glm_checkpoint_stride = 64;
    generate.glm_capture_stride = 64;
    defer generate.glm_checkpoint_stride = glm5_forward.prefill_chunk;
    defer generate.glm_capture_stride = kv_disk_cache_mod.POINT_GRID;

    var weights = model_mod.Weights.init(a);
    defer weights.deinit();
    const cfg = try glmGeneratorFixture(&weights);
    var turn = try Transformer.init(testing.io, a, cfg, &weights);
    defer turn.deinit();
    var gen = try Generator.initWithOptions(testing.io, a, &turn, &tok, &first, 4, greedy, &.{}, opts);
    defer gen.deinit(a);
    var at: [5]usize = undefined;
    for (gen.ssm_checkpoints.items, 0..) |cp, i| at[i] = cp.pos;
    // Stride boundaries, then the backoff snapshot pulled back to a pool boundary (299 - 31).
    try testing.expectEqualSlices(usize, &.{ 64, 128, 192, 256, 268 }, at[0..gen.ssm_checkpoints.items.len]);
    var rows = try glm5_prefix.MlaRows.capture(a, &turn.glm5_request.?, 268, turn.s);
    defer rows.deinit();

    var cold_xfm = try Transformer.init(testing.io, a, cfg, &weights);
    defer cold_xfm.deinit();
    var cold = try Generator.initWithOptions(testing.io, a, &cold_xfm, &tok, &next, 4, greedy, &.{}, opts);
    defer cold.deinit(a);

    var warm_xfm = try Transformer.init(testing.io, a, cfg, &weights);
    defer warm_xfm.deinit();
    try glm5_prefix.restore(&warm_xfm.glm5_request.?, &rows, &gen.ssm_checkpoints.items[3]);
    warm_xfm.cache.step = 256;
    var warm_opts = opts;
    warm_opts.ssm_checkpoint_pos_offset = 256;
    var warm = try Generator.initWithOptions(testing.io, a, &warm_xfm, &tok, next[256..], 4, greedy, &.{}, warm_opts);
    defer warm.deinit(a);
    try glm5_prefix.expectSameLogicalState(&cold_xfm.glm5_request.?, &warm_xfm.glm5_request.?);
    try testing.expectEqual(cold.next_token_id, warm.next_token_id);
    try testing.expectEqual(cold.ssm_checkpoints.items[cold.ssm_checkpoints.items.len - 1].pos, warm.ssm_checkpoints.items[warm.ssm_checkpoints.items.len - 1].pos);
    const sched = glmCaptureSchedule(64, 16, 256, next.len - 256);
    try testing.expectEqual(sched.peak, warm.ssm_cp_peak);
    try testing.expectEqual(sched.newest, warm.ssm_checkpoints.items[warm.ssm_checkpoints.items.len - 1].pos);
}

test "the GLM capture schedule is what the prefill loop captures: peak, newest and count, cold and warm, at every cap" {
    const widths = [_]usize{ 2048, 512, 64 };
    for ([_][2]usize{ .{ 64, 64 }, .{ 2048, 2048 }, .{ 2048, 8192 } }) |grids| for ([_]usize{ 0, 4, 60, 2044 }) |offset| {
        const chunk_grid = grids[0];
        const grid = grids[1];
        for ([_]usize{ 1, 2, 15, 16, 31, 33, 34, 64, 65, 200, 2048, 2049, 5000, 18432, 40000 }) |tail| for (0..9) |cap_usize| {
            const cap: u32 = @intCast(cap_usize);
            const got = glmCaptureSchedule(grid, cap, offset, tail);
            for (widths) |w| {
                const want = glmCaptureOracle(chunk_grid, grid, cap, offset, tail, w);
                testing.expectEqual(want, got) catch |e| {
                    std.debug.print("grid {d} offset {d} tail {d} cap {d} width {d}: want {any} got {any}\n", .{ grid, offset, tail, cap, w, want, got });
                    return e;
                };
            }
        };
    };
    // 18,432 cold tokens: eight grid points, then the prompt-end checkpoint beside them before the thin.
    try testing.expectEqual(@as(u32, 9), glmCaptureSchedule(2048, 8, 0, 18432).peak);
    // A 40k prompt keeps the 8192 grid and the prompt end only: 8192 x 4, then the end.
    const wide = glmCaptureSchedule(8192, 0, 0, 40000);
    try testing.expectEqual(@as(u32, 5), wide.events);
    // A 16-token cold prompt captures one checkpoint, at 12.
    try testing.expectEqual(@as(usize, 12), glmCaptureSchedule(2048, 8, 0, 16).newest);
    // A grid point at a chunk's end is not absorbed by the tail merge that follows it.
    try testing.expectEqual(@as(usize, 2058), nextChunkEnd(0, 2058, 2048, true, 2048, 0, true));
    try testing.expectEqual(@as(usize, 2048), glmChunkEnd(0, 2058, 2048, 2048, 0, true));
}

test "a GLM prefill without checkpoints chunks like any other prefill" {
    for ([_]usize{ 0, 4096 }) |pos| for ([_]usize{ 2, 2058, 9000 }) |len| {
        const end = pos + len;
        try testing.expectEqual(nextChunkEnd(pos, end, 2048, false, 0, 0, true), glmChunkEnd(pos, end, 2048, 0, 0, true));
    };
}

test "the GLM prefill holds as many checkpoints as its schedule says, and ends on the newest" {
    try mlx.requireGpu();
    const a = testing.allocator;
    var tok = Tokenizer.initEmptyForTests(a);
    defer tok.deinit();
    var weights = model_mod.Weights.init(a);
    defer weights.deinit();
    const cfg = try glmGeneratorFixture(&weights);
    const greedy = SamplingParams{ .temperature = 0.0 };
    var prompt: [300]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast((i * 5 + 1) % 4);
    generate.glm_checkpoint_stride = 64;
    generate.glm_capture_stride = 64;
    defer generate.glm_checkpoint_stride = glm5_forward.prefill_chunk;
    defer generate.glm_capture_stride = kv_disk_cache_mod.POINT_GRID;
    for ([_]usize{ 14, 70, 300 }) |len| for ([_]u32{ 2, 16 }) |cap| {
        var xfm = try Transformer.init(testing.io, a, cfg, &weights);
        defer xfm.deinit();
        var gen = try Generator.initWithOptions(testing.io, a, &xfm, &tok, prompt[0..len], 2, greedy, &.{}, .{ .skip_lazy_preforward = true, .pinned_prefill_chunk = 64, .adaptive_chunk_width = true, .ssm_checkpoint_stride = 1, .ssm_checkpoint_max = cap });
        defer gen.deinit(a);
        const sched = glmCaptureSchedule(64, cap, 0, len);
        try testing.expectEqual(sched.peak, gen.ssm_cp_peak);
        try testing.expectEqual(@min(sched.events, cap), @as(u32, @intCast(gen.ssm_checkpoints.items.len)));
        try testing.expectEqual(sched.newest, gen.ssm_checkpoints.items[gen.ssm_checkpoints.items.len - 1].pos);
    };
}

const GlmReleasing = struct {
    peak: usize = 0,

    fn call(raw: *anyopaque, _: usize, cps: []SSMCheckpoint) void {
        const self: *GlmReleasing = @ptrCast(@alignCast(raw));
        var held: usize = 0;
        var newest: ?usize = null;
        for (cps, 0..) |cp, i| {
            if (cp.isReleased()) continue;
            held += 1;
            newest = i;
        }
        self.peak = @max(self.peak, held);
        for (cps, 0..) |*cp, i| if (i != newest) cp.release();
    }
};

test "a GLM prefill whose hook releases landed checkpoints holds at most two states and lists every restore point" {
    try mlx.requireGpu();
    const a = testing.allocator;
    var tok = Tokenizer.initEmptyForTests(a);
    defer tok.deinit();
    var weights = model_mod.Weights.init(a);
    defer weights.deinit();
    const cfg = try glmGeneratorFixture(&weights);
    const greedy = SamplingParams{ .temperature = 0.0 };
    var prompt: [300]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast((i * 5 + 1) % 4);
    generate.glm_checkpoint_stride = 64;
    generate.glm_capture_stride = 64;
    defer generate.glm_checkpoint_stride = glm5_forward.prefill_chunk;
    defer generate.glm_capture_stride = kv_disk_cache_mod.POINT_GRID;
    for ([_]usize{ 14, 70, 300 }) |len| for ([_]u32{ 2, 16 }) |cap| {
        var kept_xfm = try Transformer.init(testing.io, a, cfg, &weights);
        defer kept_xfm.deinit();
        const base: Generator.InitOptions = .{ .skip_lazy_preforward = true, .pinned_prefill_chunk = 64, .adaptive_chunk_width = true, .ssm_checkpoint_stride = 1, .ssm_checkpoint_max = cap };
        var kept = try Generator.initWithOptions(testing.io, a, &kept_xfm, &tok, prompt[0..len], 2, greedy, &.{}, base);
        defer kept.deinit(a);

        var xfm = try Transformer.init(testing.io, a, cfg, &weights);
        defer xfm.deinit();
        var releasing = GlmReleasing{};
        var opts = base;
        opts.write_through_hook = .{ .ctx = &releasing, .call = GlmReleasing.call };
        var gen = try Generator.initWithOptions(testing.io, a, &xfm, &tok, prompt[0..len], 2, greedy, &.{}, opts);
        defer gen.deinit(a);

        const sched = glmCaptureSchedule(64, cap, 0, len);
        try testing.expect(releasing.peak <= @min(sched.peak, 2));
        if (len >= 300) try testing.expectEqual(@as(usize, 2), releasing.peak);
        // The cap bounds the states in memory, not the restore points listed: every capture stays.
        try testing.expectEqual(@as(usize, sched.events), gen.ssm_checkpoints.items.len);
        try testing.expectEqual(kept.ssm_checkpoints.items[kept.ssm_checkpoints.items.len - 1].pos, gen.ssm_checkpoints.items[gen.ssm_checkpoints.items.len - 1].pos);
        var held: usize = 0;
        for (gen.ssm_checkpoints.items) |got| held += @intFromBool(!got.isReleased());
        try testing.expect(held <= 2);
        try testing.expect(!gen.ssm_checkpoints.items[gen.ssm_checkpoints.items.len - 1].isReleased());
    };
}

const GlmStepDown = struct {
    fn call(_: *anyopaque, pos: usize, cur: u32, _: u32, _: *AdaptiveWidthState) u32 {
        return if (pos >= 128) 32 else cur;
    }
    fn confirm(_: *anyopaque, _: usize, _: u32) bool {
        return false;
    }
};

test "GLM checkpoints stay on the grid when a request steps its chunk down, and a restore there with the same widths matches cold" {
    try mlx.requireGpu();
    const a = testing.allocator;
    var tok = Tokenizer.initEmptyForTests(a);
    defer tok.deinit();
    var weights = model_mod.Weights.init(a);
    defer weights.deinit();
    const cfg = try glmGeneratorFixture(&weights);
    generate.glm_checkpoint_stride = 64;
    generate.glm_capture_stride = 64;
    defer generate.glm_checkpoint_stride = glm5_forward.prefill_chunk;
    defer generate.glm_capture_stride = kv_disk_cache_mod.POINT_GRID;
    var first: [300]u32 = undefined;
    for (&first, 0..) |*t, i| t.* = @intCast((i * 7 + i / 5) % 4);
    var next: [330]u32 = undefined;
    @memcpy(next[0..290], first[0..290]);
    for (next[290..], 0..) |*t, i| t.* = @intCast((i * 3 + 1) % 4);
    const greedy = SamplingParams{ .temperature = 0.0 };
    var dummy: u8 = 0;
    const hook: Generator.ChunkWidthHook = .{ .ctx = &dummy, .call = GlmStepDown.call, .confirm = GlmStepDown.confirm };
    const opts: Generator.InitOptions = .{ .skip_lazy_preforward = true, .pinned_prefill_chunk = 64, .adaptive_chunk_width = true, .chunk_width_hook = hook, .ssm_checkpoint_stride = 64, .ssm_checkpoint_max = 16 };

    var turn = try Transformer.init(testing.io, a, cfg, &weights);
    defer turn.deinit();
    var gen = try Generator.initWithOptions(testing.io, a, &turn, &tok, &first, 2, greedy, &.{}, opts);
    defer gen.deinit(a);
    var at: [5]usize = undefined;
    for (gen.ssm_checkpoints.items, 0..) |cp, i| at[i] = cp.pos;
    // 64-wide chunks to 128, 32-wide after: the grid points are still chunk ends.
    try testing.expectEqualSlices(usize, &.{ 64, 128, 192, 256, 268 }, at[0..gen.ssm_checkpoints.items.len]);
    var rows = try glm5_prefix.MlaRows.capture(a, &turn.glm5_request.?, 268, turn.s);
    defer rows.deinit();

    var cold_xfm = try Transformer.init(testing.io, a, cfg, &weights);
    defer cold_xfm.deinit();
    var cold = try Generator.initWithOptions(testing.io, a, &cold_xfm, &tok, &next, 2, greedy, &.{}, opts);
    defer cold.deinit(a);

    var warm_xfm = try Transformer.init(testing.io, a, cfg, &weights);
    defer warm_xfm.deinit();
    try glm5_prefix.restore(&warm_xfm.glm5_request.?, &rows, &gen.ssm_checkpoints.items[2]);
    warm_xfm.cache.step = 192;
    var warm_opts = opts;
    warm_opts.ssm_checkpoint_pos_offset = 192;
    // The width in force at 192 in the cold run.
    warm_opts.pinned_prefill_chunk = 32;
    var warm = try Generator.initWithOptions(testing.io, a, &warm_xfm, &tok, next[192..], 2, greedy, &.{}, warm_opts);
    defer warm.deinit(a);
    try glm5_prefix.expectSameLogicalState(&cold_xfm.glm5_request.?, &warm_xfm.glm5_request.?);
    try testing.expectEqual(cold.next_token_id, warm.next_token_id);
}

/// A speculative round as the generators commit one: `tokens` go through the model in one forward and join the generated ids.
fn commitTestRound(gen: *Generator, tokens: []const u32) !void {
    const ids = try testing.allocator.alloc(i32, tokens.len);
    defer testing.allocator.free(ids);
    for (tokens, ids) |t, *i| i.* = @intCast(t);
    const input = mlx.mlx_array_new_data(ids.ptr, &[_]c_int{ 1, @intCast(tokens.len) }, 2, .int32);
    defer _ = mlx.mlx_array_free(input);
    const logits = try gen.xfm.forwardWith(&gen.ctx, input);
    defer _ = mlx.mlx_array_free(logits);
    try mlx.check(mlx.mlx_array_eval(logits));
    try gen.generated_ids.appendSlice(testing.allocator, tokens);
    gen.advanceStep(@intCast(tokens.len));
}

test "GLM thought point: the round that carries the closer is cut to a pool boundary, its state is the newest checkpoint, and a restore from it is the live state" {
    try mlx.requireGpu();
    const a = testing.allocator;
    var tok = Tokenizer.initEmptyForTests(a);
    defer tok.deinit();
    var prompt: [300]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast((i * 7 + i / 5) % 3);
    const greedy = SamplingParams{ .temperature = 0.0 };
    const opts: Generator.InitOptions = .{ .skip_lazy_preforward = true, .pinned_prefill_chunk = 64, .adaptive_chunk_width = true, .ssm_checkpoint_stride = 64, .ssm_checkpoint_max = 16 };
    generate.glm_checkpoint_stride = 64;
    generate.glm_capture_stride = 64;
    defer generate.glm_checkpoint_stride = glm5_forward.prefill_chunk;
    defer generate.glm_capture_stride = kv_disk_cache_mod.POINT_GRID;

    var weights = model_mod.Weights.init(a);
    defer weights.deinit();
    const cfg = try glmGeneratorFixture(&weights);
    var turn = try Transformer.init(testing.io, a, cfg, &weights);
    defer turn.deinit();
    var gen = try Generator.initWithOptions(testing.io, a, &turn, &tok, &prompt, 4, greedy, &.{}, opts);
    defer gen.deinit(a);
    const closer: u32 = 3;
    const base = turn.glm5_request.?.offset;
    gen.thought_point = .{ .closer = closer, .grid = @intCast(glm5_prefix.pool_size), .min_tokens = 8, .base = base };
    const before = gen.ssm_checkpoints.items.len;

    // A round with no closer runs whole and takes nothing.
    const plain = [_]u32{ 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0 };
    try testing.expectEqual(@as(?usize, null), gen.thoughtRoundCut(&plain, base));
    try commitTestRound(&gen, &plain);
    gen.thoughtPointTake();
    try testing.expectEqual(before, gen.ssm_checkpoints.items.len);

    // The round that carries the closer keeps the tokens up to the pool boundary at or below it.
    const anchor = turn.glm5_request.?.offset;
    const closing = [_]u32{ 1, 2, 0, 1, 2, closer, 0, 1 };
    const want = (anchor + 5 + 1) / glm5_prefix.pool_size * glm5_prefix.pool_size - anchor;
    try testing.expect(want > 0 and want < closing.len);
    try testing.expectEqual(@as(?usize, want), gen.thoughtRoundCut(&closing, anchor));
    try commitTestRound(&gen, closing[0..want]);
    gen.thoughtPointTake();
    try testing.expectEqual(before + 1, gen.ssm_checkpoints.items.len);
    const point = &gen.ssm_checkpoints.items[before];
    try testing.expectEqual(anchor + want, point.pos);
    try testing.expectEqual(@as(usize, 0), point.pos % glm5_prefix.pool_size);
    try testing.expectEqual(ThoughtPoint.State.taken, gen.thought_point.?.state);

    // A restore from it is the live state, and the next token from both leaves them the same.
    var rows = try glm5_prefix.MlaRows.capture(a, &turn.glm5_request.?, point.pos, turn.s);
    defer rows.deinit();
    var warm_xfm = try Transformer.init(testing.io, a, cfg, &weights);
    defer warm_xfm.deinit();
    try glm5_prefix.restore(&warm_xfm.glm5_request.?, &rows, point);
    warm_xfm.cache.step = point.pos;
    try glm5_prefix.expectSameLogicalState(&turn.glm5_request.?, &warm_xfm.glm5_request.?);
    const next = [_]u32{closing[want]};
    try commitTestRound(&gen, &next);
    var warm_opts = opts;
    warm_opts.ssm_checkpoint_pos_offset = point.pos;
    var warm = try Generator.initWithOptions(testing.io, a, &warm_xfm, &tok, &next, 4, greedy, &.{}, warm_opts);
    defer warm.deinit(a);
    try glm5_prefix.expectSameLogicalState(&turn.glm5_request.?, &warm_xfm.glm5_request.?);

    // The commit may drop the point; the checkpoints are what they were.
    gen.dropThoughtPoint();
    try testing.expectEqual(before, gen.ssm_checkpoints.items.len);
}
