//! The hyper-connection prefill kernels against Flash-Next's compiled closures.
const trunk = @import("../trunk.zig");
const KVCache = trunk.KVCache;
const Trunk = trunk.Trunk;
const test_support = @import("../test_support.zig");
const attn256MaxDiff = test_support.attn256MaxDiff;
const attn256RandBf16 = test_support.attn256RandBf16;
const mlx = @import("../mlx.zig");
const qwen4_state = @import("qwen4_state.zig");
const standinOnes = trunk.standinOnes;
const std = @import("std");
const testing = std.testing;

test "hc prefill: pending write, normalized streams, native inject reduction and BF16 mix across shapes" {
    try mlx.requireGpu();
    const hp = @import("../hc_prefill.zig");
    const s = mlx.gpuStream();
    var cache = try KVCache.init(testing.allocator, 0);
    defer cache.deinit();
    var xfm = std.mem.zeroInit(Trunk, .{ .s = s, .allocator = testing.allocator, .cache = cache });
    const q4 = qwen4_state.stateOf(&xfm);
    @import("qwen4_forward.zig").compileQwen4Hc(&xfm);
    defer q4.clearCompiled();
    try testing.expect(q4.compiled_hc_write != null and q4.compiled_hc_mix != null);
    var prng = std.Random.DefaultPrng.init(0x91831);
    const rnd = prng.random();
    const eps = mlx.mlx_array_new_float(1e-6);
    defer _ = mlx.mlx_array_free(eps);
    for ([_][4]c_int{ .{ 2, 17, 4, 2560 }, .{ 1, 65, 4, 2560 }, .{ 1, 513, 4, 2560 }, .{ 1, 17, 1, 128 }, .{ 1, 33, 2, 1536 }, .{ 1, 33, 3, 1536 }, .{ 1, 33, 5, 1536 }, .{ 1, 33, 6, 1536 }, .{ 1, 33, 7, 1536 }, .{ 1, 33, 8, 4096 }, .{ 1, 8703, 4, 2560 } }) |shape| {
        const b = shape[0];
        const seq = shape[1];
        xfm.config.hc_count = @intCast(shape[2]);
        xfm.config.hidden_size = @intCast(shape[3]);
        const hc: c_int = @intCast(xfm.config.hc_count);
        const hidden: c_int = @intCast(xfm.config.hidden_size);
        const width = hc * hidden;
        const w = try attn256RandBf16(rnd, &.{ hc, hidden }, s);
        defer _ = mlx.mlx_array_free(w);
        const iw = try attn256RandBf16(rnd, &.{ width, hc }, s);
        defer _ = mlx.mlx_array_free(iw);
        try testing.expect(hp.eligible(b, seq, xfm.config.hc_count, xfm.config.hidden_size, iw, .bfloat16));
        const ones = try standinOnes(&.{hidden}, s);
        defer _ = mlx.mlx_array_free(ones);
        const x = try attn256RandBf16(rnd, &.{ b, seq, hc, hidden }, s);
        defer _ = mlx.mlx_array_free(x);
        const wo = try attn256RandBf16(rnd, &.{ b, seq, 1, hidden }, s);
        defer _ = mlx.mlx_array_free(wo);
        const wi = try attn256RandBf16(rnd, &.{ b, seq, hc, 1 }, s);
        defer _ = mlx.mlx_array_free(wi);
        try testing.expect((try hp.norm(s, x, w, .{ .ctx = null }, eps, b, seq, null)) == null);
        try testing.expect((try hp.norm(s, x, w, iw, eps, b, 16, null)) == null);
        var f32_w = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(f32_w);
        try mlx.check(mlx.mlx_astype(&f32_w, w, .float32, s));
        try testing.expect((try hp.norm(s, x, f32_w, iw, eps, b, seq, null)) == null);
        try testing.expect((try hp.mix(s, x, x, b, seq + 1)) == null);
        const written = (try @import("qwen4_forward.zig").applyClosure(q4.compiled_hc_write, &.{ x, wo, wi })) orelse return error.MissingCompiledHc;
        defer _ = mlx.mlx_array_free(written);
        for ([_]bool{ false, true }) |pending| {
            const stream = if (pending) written else x;
            const n = (try hp.norm(s, x, w, iw, eps, b, seq, if (pending) .{ .out = wo, .inj = wi } else null)) orelse return error.HcFusedDeclined;
            defer n.deinit();
            var state = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(state);
            try mlx.check(mlx.mlx_reshape(&state, n.stream, &[_]c_int{ b, seq, hc, hidden }, 4, s));
            try testing.expectEqual(@as(f32, 0), try attn256MaxDiff(state, stream, s));
            var rms = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(rms);
            try mlx.check(mlx.mlx_fast_rms_norm(&rms, stream, ones, 1e-6, s));
            var normed = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(normed);
            try mlx.check(mlx.mlx_multiply(&normed, rms, w, s));
            try testing.expectEqual(@as(f32, 0), try attn256MaxDiff(n.normalized, normed, s));
            var flat = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(flat);
            try mlx.check(mlx.mlx_reshape(&flat, normed, &[_]c_int{ b, seq, width }, 3, s));
            var raw = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(raw);
            try mlx.check(mlx.mlx_matmul(&raw, flat, iw, s));
            try testing.expectEqual(@as(f32, 0), try attn256MaxDiff(n.raw_inject, raw, s));
            const expected = (try @import("qwen4_forward.zig").applyClosure(q4.compiled_hc_mix, &.{ x, normed })) orelse return error.MissingCompiledHc;
            defer _ = mlx.mlx_array_free(expected);
            const mixed = (try hp.mix(s, x, normed, b, seq)) orelse return error.HcFusedDeclined;
            defer _ = mlx.mlx_array_free(mixed);
            try testing.expectEqual(@as(f32, 0), try attn256MaxDiff(mixed, expected, s));
        }
    }
}
