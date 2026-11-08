//! The shape, size and dtype queries (`mlx.query_count`, each an FFI call) of a Flash-Next decode tick on a pack: the
//! forward of one token after a prompt, counted on the pack's own layers.
const std = @import("std");
const mlx = @import("../mlx.zig");
const model_mod = @import("../model.zig");
const trunk_mod = @import("../trunk.zig");
const qwen4_forward = @import("qwen4_forward.zig");
const Qwen4TestSlot = @import("qwen4_test_support.zig").Qwen4TestSlot;
const KVQuantConfig = @import("../kv_quant.zig").KVQuantConfig;
const Trunk = trunk_mod.Trunk;
const testing = std.testing;

const Pinned = struct { pack: []const u8, ticks: [4]u64 };

/// The quantization-parameter cache hashes weight pointers, so a probe that misses its window costs queries that
/// vary by a few dozen from one process to the next.
const slack: u64 = 40;

const pinned = [_]Pinned{
    .{ .pack = "Qwen3.8-Flash-Next-Sushi-STQ-2.75bpw", .ticks = .{ 9083, 9083, 9373, 9083 } },
    .{ .pack = "Qwen3.8-Flash-Next-Sushi-4bpw", .ticks = .{ 9083, 9083, 9373, 9083 } },
};

fn tick(xfm: *Trunk, slot: *Qwen4TestSlot, token: i32) !u64 {
    const before = mlx.query_count;
    const logits = try slot.forward(xfm, &.{token});
    const used = mlx.query_count - before;
    defer _ = mlx.mlx_array_free(logits);
    try mlx.check(mlx.mlx_array_eval(logits));
    return used;
}

test "qwen4 queries: a decode tick through forwardQwen4With (QWEN4_TEST_MODEL)" {
    const model_dir = std.mem.span(std.c.getenv("QWEN4_TEST_MODEL") orelse return error.SkipZigTest);
    const name = std.fs.path.basename(std.mem.trimEnd(u8, model_dir, "/"));
    const want = for (pinned) |p| {
        if (std.mem.eql(u8, p.pack, name)) break p.ticks;
    } else return error.SkipZigTest;
    try mlx.requireGpu();
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var config = try model_mod.parseConfig(io, a, model_dir);
    defer if (config.ngram_table_path) |path| a.free(path);
    var weights = try model_mod.loadWeights(io, a, model_dir);
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Trunk.init(io, a, config, &weights);
    defer xfm.deinit();
    qwen4_forward.compileQwen4Hc(&xfm);
    qwen4_forward.compileGdnGate(&xfm);
    xfm.compileMoeRouting();
    const slot = try Qwen4TestSlot.init(a, config.num_hidden_layers);
    defer slot.deinit(a);
    try slot.cache.reinit(config.num_hidden_layers, KVQuantConfig.affine(8));
    const prompt = [_]i32{ 248045, 846, 198, 7734, 264, 2716, 3255, 883, 264, 11952, 6618, 310, 5984, 13, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271 };
    const prefill = try slot.forward(&xfm, &prompt);
    defer _ = mlx.mlx_array_free(prefill);
    try mlx.check(mlx.mlx_array_eval(prefill));
    const follow = [_]i32{ 4413, 220, 22, 18, 19, 11 };
    // The first ticks build what a launch keeps for the process.
    for (follow[0..2]) |t| _ = try tick(&xfm, slot, t);
    var got: [4]u64 = undefined;
    for (follow[2..], &got) |t, *n| n.* = try tick(&xfm, slot, t);
    for (want, got) |w, g| if (g + slack < w or g > w + slack) {
        std.debug.print("queries per tick {any}, pinned {any}\n", .{ got, want });
        return error.TestExpectedEqual;
    };
}
