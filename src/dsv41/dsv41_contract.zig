//! What the shared Sushi Cooler contract (`src/cooler_contract_test.zig`) asks of a DeepSeek load: its tiny fixtures
//! and the reads only this arch can make.
const std = @import("std");
const mlx = @import("../mlx.zig");
const model_mod = @import("../model.zig");
const Trunk = @import("../trunk.zig").Trunk;
const dsv41_model = @import("dsv41_model.zig");
const dsv41_state = @import("dsv41_state.zig");
const Generator = @import("../generate.zig").Generator;
const Tokenizer = @import("../tokenizer.zig").Tokenizer;

pub const fixtures = [_][:0]const u8{ "DSV41_STQ_TINY", "DSV41_STQ_TINY_PM", "DSV41_TINY" };

/// A drafting request on a resident load rides the joined plain rows once three requests decode (`dsv41_sched.tickClass`):
/// a cell that needs one drafting request in a unit of its own asks it for logprobs.
pub const drafting_rides_rows = true;

/// A forward reads out its last row alone (`dsv41_model.Want.last`), however wide, under its own caps.
pub const reads_out_last_row_only = true;

/// Prompt lengths a request of the contract takes: one row, the decode widths' edge, a chunk past it.
pub const prompt_lengths = [_]usize{ 1, 2, 7, 16, 17, 40 };

/// What a test load sets on the parsed config: the experts stream, the DSpark stages stay resident. A tiny STQ fixture's
/// experts fit the cache, so its plan is resident and the load joins rows (the load refuses the claim over a streamed
/// plan); an MXFP4 store is never resident.
pub fn prepare(a: std.mem.Allocator, config: *model_mod.ModelConfig, dir: []const u8) !void {
    config.expert_streaming = true;
    config.stream_mtp_head = true;
    config.expert_cache_bytes = 64 << 20;
    config.expert_resident = if (config.dsv41) |c| c.expert_dec != null else false;
    config.expert_source_dir = try a.dupe(u8, dir);
}

/// While on, the load's expand probe serves the arm its parse did not guess (`dsv41_config.Config.wide_arms`), so a
/// config that kept the guess prices a wide forward the model does not run.
pub fn armsAgainstParse(on: bool) void {
    @import("dsv41_kernels.zig").hc_expand_probe_override = if (on) !@import("sushi_exl3").kernels.gemmNaxOn() else null;
}

/// The served tiny ends a chat or Responses reply within a few tokens, and holds 128: a long reply is a completion
/// with `ignore_eos`.
pub const server_long_replies = false;

/// The SSD budget a server test's load names (`ssd_budget_gb` in its model-settings.json): a Mac under 96 GB has no
/// RAM-tier default and refuses the load by name (`expert_streaming_required`), and the tiny's experts fit 1 GB whole.
pub const server_ssd_budget_gb: u32 = 1;

/// What a server test serves (`src/server_test.zig`), in `dir`: the DSpark tiny's weights, and a byte-level tokenizer
/// of the release's specials and the tiny's 96 ids, which the fixture lacks.
pub fn serverDir(dir: std.Io.Dir) !void {
    const tiny = std.mem.span(std.c.getenv("DSV41_TINY") orelse return error.SkipZigTest);
    try @import("../test_support.zig").linkInto(dir, tiny, &.{});
    const io = std.Io.Threaded.global_single_threaded.io();
    try dir.writeFile(io, .{ .sub_path = "tokenizer.json", .data = @embedFile("../fixtures/dsv41_tiny_tokenizer.json") });
    try dir.writeFile(io, .{ .sub_path = "tokenizer_config.json", .data = @embedFile("../fixtures/dsv41_tiny_tokenizer_config.json") });
}

/// The fixture's prompt ids (allocated with `a`), read before any load.
pub fn prompt(a: std.mem.Allocator, dir: []const u8) ![]u32 {
    var fx = try dsv41_model.loadFixture(dir, "fixture_noqat.safetensors");
    defer fx.deinit();
    var sc = @import("dsv41_scope.zig").Scope.plain(a, mlx.gpuStream());
    defer sc.deinit();
    return dsv41_model.tokens(&sc, &fx);
}

/// The weights of a Cooler load held outside an owned region: zero when every weight is mapped or built in one. The
/// rope tables and Engram's q.k products (MLX-made from a few KiB, the documented exception) are not counted.
pub fn weightsOutsideRegions(xfm: *Trunk) !usize {
    const m = dsv41_state.modelOf(xfm).?;
    var outside: usize = 0;
    for (m.owned.items) |arr| {
        if (arr.ctx == null) continue;
        try mlx.check(mlx.mlx_array_eval(arr));
        if (model_mod.ownsPointer(mlx.engine.storePointer(arr))) continue;
        var exempt = arr.ctx == m.rope_plain.freqs.ctx or arr.ctx == m.rope_yarn.freqs.ctx;
        for (m.engrams) |e| exempt = exempt or arr.ctx == e.qk.ctx;
        if (!exempt) outside += 1;
    }
    return outside;
}

/// Whether the load holds the drafting stages the fixtures ship.
pub fn headLoaded(xfm: *Trunk) bool {
    return dsv41_state.modelOf(xfm).?.dspark != null;
}

/// `n` greedy tokens over `ids` in DSpark rounds, every round width swept: the spec path a request takes.
pub fn specTokens(a: std.mem.Allocator, xfm: *Trunk, ids: []const u32, n: u32) !std.ArrayList(u32) {
    return specTokensWith(a, xfm, ids, n, .{ .temperature = 0.0 });
}

/// `specTokens` under `sampling`.
pub fn specTokensWith(a: std.mem.Allocator, xfm: *Trunk, ids: []const u32, n: u32, sampling: @import("../sampler.zig").SamplingParams) !std.ArrayList(u32) {
    var tok = Tokenizer.initEmpty(a);
    defer tok.deinit();
    try xfm.resetCache();
    var gen = try Generator.initWithOptions(std.Io.Threaded.global_single_threaded.io(), a, xfm, &tok, ids, n, sampling, &.{}, .{ .dspark_enabled = true });
    defer gen.deinit(a);
    gen.dspark_diag = .sweep;
    var got: std.ArrayList(u32) = .empty;
    errdefer got.deinit(a);
    while (try gen.nextDspark(a)) |r| {
        defer a.free(r.tokens);
        try got.appendSlice(a, r.tokens);
    }
    return got;
}

/// `n` greedy tokens for each of `prompts`: each request prefills on a decode state of its own, then all of them decode
/// as rows of one forward a step (`dsv41_serving.forwardRows`), as a load that joins rows serves them.
pub fn greedyRows(a: std.mem.Allocator, xfm: *Trunk, prompts: []const []const u32, n: u32) ![]std.ArrayList(u32) {
    const trunk_mod = @import("../trunk.zig");
    const m = dsv41_state.modelOf(xfm).?;
    const k = prompts.len;
    const reqs = try a.alloc(dsv41_model.Request, k);
    @memset(reqs, .{});
    defer {
        for (reqs) |*r| r.deinit();
        a.free(reqs);
    }
    const caches = try a.alloc(trunk_mod.KVCache, k);
    var made: usize = 0;
    defer {
        for (caches[0..made]) |*c| c.deinit();
        a.free(caches);
    }
    const offsets = try a.alloc(usize, k);
    defer a.free(offsets);
    @memset(offsets, 0);
    const out = try a.alloc(std.ArrayList(u32), k);
    for (out) |*o| o.* = .empty;
    errdefer {
        for (out) |*o| o.deinit(a);
        a.free(out);
    }
    const next = try a.alloc(u32, k);
    defer a.free(next);
    for (prompts, 0..) |p, i| {
        caches[i] = try trunk_mod.KVCache.initWithConfig(a, m.cfg.layers, @import("../kv_quant.zig").KVQuantConfig.dense);
        made += 1;
        var ctx = xfm.defaultCtx();
        ctx.cache = &caches[i];
        ctx.moe_seq_offset = &offsets[i];
        ctx.arch_ctx.dsv41.request = &reqs[i];
        // As a request prefills with no restore points: the prompt but its last token, then that token alone for the
        // logits (`Generator`'s final forward).
        if (p.len > 1) _ = mlx.mlx_array_free(try forwardIds(xfm, &ctx, p[0 .. p.len - 1]));
        const logits = try forwardIds(xfm, &ctx, p[p.len - 1 ..]);
        defer _ = mlx.mlx_array_free(logits);
        try argmaxRows(xfm, logits, next[i .. i + 1]);
        try out[i].append(a, next[i]);
    }
    const segs = try a.alloc(dsv41_model.Seg, k);
    defer a.free(segs);
    var step: u32 = 1;
    while (step < n) : (step += 1) {
        for (segs, reqs, next) |*sg, *r, *t| sg.* = .{ .request = r, .ids = @as(*const [1]u32, t) };
        const rows = try @import("dsv41_serving.zig").forwardRows(xfm, segs);
        defer _ = mlx.mlx_array_free(rows);
        try argmaxRows(xfm, rows, next);
        for (out, next) |*o, t| try o.append(a, t);
    }
    return out;
}

fn forwardIds(xfm: *Trunk, ctx: *@import("../trunk.zig").ForwardCtx, p: []const u32) !mlx.mlx_array {
    const ids = try std.heap.page_allocator.alloc(i32, p.len);
    defer std.heap.page_allocator.free(ids);
    for (p, ids) |t, *o| o.* = @intCast(t);
    const shape = [_]c_int{ 1, @intCast(p.len) };
    const host = mlx.engine.enter(xfm.outStream());
    const arr = mlx.newData(ids.ptr, &shape, .int32) catch |e| {
        host.end();
        return e;
    };
    host.end();
    defer _ = mlx.mlx_array_free(arr);
    return xfm.forwardWith(ctx, arr);
}

/// The greedy token of each of the last `into.len` rows of `logits` into `into`.
fn argmaxRows(xfm: *Trunk, logits: mlx.mlx_array, into: []u32) !void {
    var am = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(am);
    try mlx.check(mlx.mlx_argmax_axis(&am, logits, -1, false, xfm.outStream()));
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    const shape = [_]c_int{-1};
    try mlx.check(mlx.mlx_reshape(&flat, am, &shape, 1, xfm.outStream()));
    var as32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(as32);
    try mlx.check(mlx.mlx_astype(&as32, flat, .int32, xfm.outStream()));
    try mlx.check(mlx.mlx_array_eval(as32));
    // A prefill's logits may hold every position: its token is the last row's.
    const all = (mlx.mlx_array_data_int32(as32) orelse return error.InvalidDsv41Input)[0..mlx.mlx_array_size(as32)];
    for (all[all.len - into.len ..], into) |v, *o| o.* = @intCast(v);
}
