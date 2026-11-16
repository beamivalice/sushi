//! What the shared Sushi Cooler contract (`src/cooler_contract_test.zig`) asks of a Flash-Next load: its tiny fixture
//! and the reads only this arch can make.
const std = @import("std");
const mlx = @import("../mlx.zig");
const model_mod = @import("../model.zig");
const Trunk = @import("../trunk.zig").Trunk;

pub const fixtures = [_][:0]const u8{"QWEN4_STQ_TINY"};

/// MTP's verdict is exact and its verify rows are a serial step's bytes, so a speculative request says its greedy tokens.
pub const spec_is_greedy = true;

/// Prompt lengths a request of the contract takes: one row, the verify widths' edge, a chunk past it.
pub const prompt_lengths = [_]usize{ 1, 2, 7, 8, 9, 40 };

/// A test load is resident, as the tiny pack is.
pub fn prepare(a: std.mem.Allocator, config: *model_mod.ModelConfig, dir: []const u8) !void {
    _ = a;
    _ = config;
    _ = dir;
}

/// The tiny pack ships no tokenizer: ids under the 128 its calibration drew from.
pub fn prompt(a: std.mem.Allocator, dir: []const u8) ![]u32 {
    _ = dir;
    const ids = try a.alloc(u32, 48);
    for (ids, 0..) |*t, i| t.* = @intCast((i * 37 + 11) % 128);
    return ids;
}

/// The weights of a Cooler load held outside an owned region: zero when every weight is mapped or built in one.
pub fn weightsOutsideRegions(xfm: *Trunk) !usize {
    const all = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(all);
    try @import("qwen4_cooler.zig").appendWeights(xfm, all);
    var outside: usize = 0;
    for (0..mlx.mlx_vector_array_size(all)) |i| {
        var arr: mlx.mlx_array = .{};
        defer _ = mlx.mlx_array_free(arr);
        try mlx.check(mlx.mlx_vector_array_get(&arr, all, i));
        try mlx.check(mlx.mlx_array_eval(arr));
        if (!model_mod.ownsPointer(mlx.engine.storePointer(arr))) outside += 1;
    }
    return outside;
}

/// `n` greedy tokens over `ids` in MTP draft and verify rounds: the spec path a request takes.
pub fn specTokens(a: std.mem.Allocator, xfm: *Trunk, ids: []const u32, n: u32) !std.ArrayList(u32) {
    return specTokensWith(a, xfm, ids, n, .{ .temperature = 0.0 });
}

/// `specTokens` under `sampling`.
pub fn specTokensWith(a: std.mem.Allocator, xfm: *Trunk, ids: []const u32, n: u32, sampling: @import("../sampler.zig").SamplingParams) !std.ArrayList(u32) {
    var tok = Tokenizer.initEmpty(a);
    defer tok.deinit();
    try xfm.resetCache();
    var gen = try Generator.initWithOptions(std.Io.Threaded.global_single_threaded.io(), a, xfm, &tok, ids, n, sampling, &.{}, .{
        .mtp = generate.MtpHeadRef.of("qwen4", xfm),
        .mtp_enabled = true,
        .model_has_mtp = true,
    });
    defer gen.deinit(a);
    var got: std.ArrayList(u32) = .empty;
    errdefer got.deinit(a);
    while (try gen.nextMtp(a)) |r| {
        defer a.free(r.tokens);
        try got.appendSlice(a, r.tokens);
    }
    return got;
}

const generate = @import("../generate.zig");
const Generator = generate.Generator;
const Tokenizer = @import("../tokenizer.zig").Tokenizer;

/// Whether the load holds the MTP head the fixture ships.
pub fn headLoaded(xfm: *Trunk) bool {
    return xfm.arch_state.qwen4.mtp != null;
}
