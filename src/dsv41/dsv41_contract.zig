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

/// Prompt lengths a request of the contract takes: one row, the decode widths' edge, a chunk past it.
pub const prompt_lengths = [_]usize{ 1, 2, 7, 16, 17, 40 };

/// What a test load sets on the parsed config: the experts stream, the DSpark stages stay resident.
pub fn prepare(a: std.mem.Allocator, config: *model_mod.ModelConfig, dir: []const u8) !void {
    config.expert_streaming = true;
    config.stream_mtp_head = true;
    config.expert_cache_bytes = 64 << 20;
    config.expert_source_dir = try a.dupe(u8, dir);
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
    var tok = Tokenizer.initEmpty(a);
    defer tok.deinit();
    try xfm.resetCache();
    var gen = try Generator.initWithOptions(std.Io.Threaded.global_single_threaded.io(), a, xfm, &tok, ids, n, .{ .temperature = 0.0 }, &.{}, .{ .dspark_enabled = true });
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
