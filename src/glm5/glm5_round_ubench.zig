//! DIAGNOSTIC (SUSHI_GLM_ROUND_UBENCH=N): at load, N greedy DFlash2 rounds of one request per
//! `_CTX` (default 8192; comma list) prefilled from `_TEXT`, each drafted, verified and committed
//! as serving does. Prints the medians of draft, verify, replay and commit per round, accepted
//! drafts and rows per round.
const std = @import("std");
const mlx = @import("../mlx.zig");
const forward = @import("glm5_forward.zig");
const verifier = @import("glm5_dflash_model.zig");
const adapter = @import("glm5_dflash.zig");
const draft = @import("../dflash.zig");
const io_util = @import("../io_util.zig");
const log = @import("../log.zig");

const max_rounds = 512;
const warm = 3;
const budget = 1 << 20;

pub fn run(allocator: std.mem.Allocator, target: *forward.Model, assistant: *draft.DflashModel, latent_bits: u8, source: []const u32, ctx: usize, rounds_asked: usize) !void {
    if (ctx < 2048) return error.InvalidGlmRoundUbench;
    const io = std.Io.Threaded.global_single_threaded.io();
    const rounds = @min(rounds_asked, max_rounds);
    const s = target.s;
    var request = try forward.Request.initServing(allocator, target.layers.len);
    defer request.deinit();
    try request.setLatentBits(latent_bits);
    var context = try draft.DflashCtx.init(allocator, assistant, 0);
    defer context.deinit();
    const prompt = try allocator.alloc(u32, ctx);
    defer allocator.free(prompt);
    for (prompt, 0..) |*id, i| id.* = if (source.len > 0) source[i % source.len] else @intCast(1 + (i * 7919) % 150000);
    var pending: u32 = 0;
    var at: usize = 0;
    while (at < ctx) : (at += 2048) {
        const end = @min(ctx, at + 2048);
        const ids = mlx.mlx_array_new_data(prompt[at..end].ptr, &.{ 1, @intCast(end - at) }, 2, .uint32);
        defer _ = mlx.mlx_array_free(ids);
        pending = try adapter.prefill(assistant, &context, target, &request, ids);
    }
    _ = try @import("glm5_dflash_reserve.zig").reserve(&request, request.offset + (rounds + warm) * 4 + 64, std.math.maxInt(usize), s);
    const binding = verifier.bindDefaultSchedule();
    defer binding.restore();
    const taps = assistant.config.target_layer_ids;

    var draft_ns: [max_rounds]u64 = undefined;
    var verify_ns: [max_rounds]u64 = undefined;
    var replay_ns: [max_rounds]u64 = undefined;
    var commit_ns: [max_rounds]u64 = undefined;
    var accepted: usize = 0;
    var rows: usize = 0;
    for (0..rounds + warm) |round| {
        var sw = io_util.Stopwatch.init(io);
        var proposal = try adapter.proposeRound(assistant, &context, target, &request, pending, 2, budget, &.{}, 4);
        const d = sw.read();
        sw.reset();
        var verified = try verifier.verify(target, &request, proposal.tokens[0..proposal.count], proposal.parents[0..proposal.count], taps, .affine_rows_ffn);
        defer verified.deinit();
        const v = sw.read();
        const result = try adapter.finishRound(io, assistant, &context, target, &request, &proposal, &verified, budget, &.{});
        pending = result.pending orelse result.tokens[result.count - 1];
        if (round < warm) continue;
        const r = round - warm;
        draft_ns[r] = d;
        verify_ns[r] = v;
        replay_ns[r] = result.replay_ns;
        commit_ns[r] = result.commit_ns;
        accepted += result.accepted_drafts;
        rows += result.verified_rows;
    }
    const n: f64 = @floatFromInt(rounds);
    log.info("[glm-round-ubench] ctx={d} medians of {d} rounds: draft {d:.3} ms, verify {d:.3} ms, replay {d:.3} ms, commit {d:.3} ms; {d:.3} accepted drafts and {d:.3} rows per round\n", .{ ctx, rounds, median(draft_ns[0..rounds]), median(verify_ns[0..rounds]), median(replay_ns[0..rounds]), median(commit_ns[0..rounds]), @as(f64, @floatFromInt(accepted)) / n, @as(f64, @floatFromInt(rows)) / n });
}

fn median(values: []u64) f64 {
    if (values.len == 0) return 0;
    var copy: [max_rounds]u64 = undefined;
    @memcpy(copy[0..values.len], values);
    std.mem.sort(u64, copy[0..values.len], {}, std.sort.asc(u64));
    return @as(f64, @floatFromInt(copy[values.len / 2])) / 1e6;
}
