const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const trf = @import("transformer.zig");
const testing = std.testing;
const test_support = @import("test_support.zig");
const attn256MaxDiff = test_support.attn256MaxDiff;
const attn256RandBf16 = test_support.attn256RandBf16;
const KVCache = trf.KVCache;
const Transformer = trf.Transformer;
const standinOnes = trf.standinOnes;

var kernels: [2]?mlx.mlx_fast_metal_kernel = .{ null, null };
var sigmoid_table: ?mlx.mlx_array = null;
var announced = false;

// Include the final remainder that the chunker may merge into an 8192-token chunk.
pub const max_seq: c_int = 8192 + @as(c_int, @intCast(@import("generate.zig").TAIL_MERGE_MAX)) - 1;

pub fn eligible(batch: c_int, seq: c_int, hc: u32, hidden: u32, inject: mlx.mlx_array, dtype: mlx.mlx_dtype) bool {
    return inject.ctx != null and dtype == .bfloat16 and mlx.mlx_array_dtype(inject) == .bfloat16 and geometry(batch, seq) and hc >= 1 and hc <= 8 and hidden >= 128 and hidden <= 4096 and hidden % 128 == 0;
}

fn geometry(batch: c_int, seq: c_int) bool {
    return batch >= 1 and batch <= 2 and seq > 16 and seq <= @divTrunc(max_seq, batch);
}

pub const Pending = struct { out: mlx.mlx_array, inj: mlx.mlx_array };
pub const Norm = struct {
    normalized: mlx.mlx_array,
    raw_inject: mlx.mlx_array,
    stream: mlx.mlx_array,

    pub fn deinit(self: Norm) void {
        _ = mlx.mlx_array_free(self.normalized);
        _ = mlx.mlx_array_free(self.raw_inject);
        _ = mlx.mlx_array_free(self.stream);
    }
};

fn getKernel(which: usize) !mlx.mlx_fast_metal_kernel {
    if (kernels[which]) |k| return k;
    const ni = [_][*:0]const u8{ "x", "w", "eps", "wo", "wi" };
    const no = [_][*:0]const u8{ "normed", "stream" };
    const mi = [_][*:0]const u8{ "up", "normed", "sigtab", "rows" };
    const mo = [_][*:0]const u8{"out"};
    const ins = if (which == 0) mlx.mlx_vector_string_new_data(&ni, ni.len) else mlx.mlx_vector_string_new_data(&mi, mi.len);
    defer _ = mlx.mlx_vector_string_free(ins);
    const outs = if (which == 0) mlx.mlx_vector_string_new_data(&no, no.len) else mlx.mlx_vector_string_new_data(&mo, mo.len);
    defer _ = mlx.mlx_vector_string_free(outs);
    const k = mlx.mlx_fast_metal_kernel_new(if (which == 0) "sushi_hc_prefill_norm" else "sushi_hc_prefill_mix", ins, outs, if (which == 0) @embedFile("kernels/hc_prefill_norm.metal") else @embedFile("kernels/hc_prefill_mix.metal"), "", true, false);
    if (k.ctx == null) return error.MetalKernelCompileFailed;
    kernels[which] = k;
    return k;
}

pub fn norm(s: mlx.mlx_stream, x: mlx.mlx_array, w: mlx.mlx_array, iw: mlx.mlx_array, eps: mlx.mlx_array, batch: c_int, seq: c_int, pending: ?Pending) !?Norm {
    if (!geometry(batch, seq) or iw.ctx == null) return null;
    const ws = mlx.getShape(w);
    if (ws.len != 2) return null;
    const hc = ws[0];
    const hidden = ws[1];
    if (hc < 1 or hc > 8 or hidden < 128 or hidden > 4096 or @mod(hidden, 128) != 0) return null;
    const width = hc * hidden;
    if (mlx.mlx_array_size(x) != @as(usize, @intCast(batch * seq * width)) or mlx.mlx_array_size(iw) != @as(usize, @intCast(width * hc))) return null;
    if (mlx.mlx_array_dtype(x) != .bfloat16 or mlx.mlx_array_dtype(w) != .bfloat16 or mlx.mlx_array_dtype(iw) != .bfloat16) return null;
    if (pending) |p| if (mlx.mlx_array_dtype(p.out) != .bfloat16 or mlx.mlx_array_dtype(p.inj) != .bfloat16) return null;
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const shape = [_]c_int{ batch, seq, hc, hidden };
    const xsh = if (pending != null) [_]c_int{ batch, seq, width } else [_]c_int{ 1, 1, 1 };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &shape, 4, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &xsh, 3, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, batch * seq * hc * @divTrunc(hidden, 4), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, @divTrunc(hidden, 4), 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "H", hidden));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "HC", hc));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "WR", @intFromBool(pending != null)));
    const arrays = [_]mlx.mlx_array{ x, w, eps, if (pending) |p| p.out else x, if (pending) |p| p.inj else x };
    const ins = mlx.mlx_vector_array_new_data(&arrays, arrays.len);
    defer _ = mlx.mlx_vector_array_free(ins);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, try getKernel(0), ins, cfg, s));
    var normalized = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(normalized);
    var stream = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(stream);
    try mlx.check(mlx.mlx_vector_array_get(&normalized, outs, 0));
    if (pending != null) try mlx.check(mlx.mlx_vector_array_get(&stream, outs, 1)) else try mlx.check(mlx.mlx_array_set(&stream, x));
    var raw = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(raw);
    var flat = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(flat);
    try mlx.check(mlx.mlx_reshape(&flat, normalized, &[_]c_int{ batch, seq, width }, 3, s));
    // Preserve the native inject reduction order for near-boundary BF16 gates.
    try mlx.check(mlx.mlx_matmul(&raw, flat, iw, s));
    // Keep the prefill engagement visible after a short server readiness probe.
    if (!announced and seq >= 512) {
        announced = true;
        log.info("[hc-prefill] engaged: S={d} B={d} pending={} native_inject=true\n", .{ seq, batch, pending != null });
    }
    return .{ .normalized = normalized, .raw_inject = raw, .stream = stream };
}

// Use MLX sigmoid for every BF16 bit pattern, including rounding-boundary inputs.
pub fn sigmoidTable(s: mlx.mlx_stream) !mlx.mlx_array {
    if (sigmoid_table) |table| return table;
    var bits: [65536]u16 = undefined;
    for (&bits, 0..) |*v, i| v.* = @intCast(i);
    const x = mlx.mlx_array_new_data(&bits, &[_]c_int{65536}, 1, .bfloat16);
    defer _ = mlx.mlx_array_free(x);
    var table = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(table);
    try mlx.check(mlx.mlx_sigmoid(&table, x, s));
    sigmoid_table = table;
    return table;
}

pub fn mix(s: mlx.mlx_stream, up: mlx.mlx_array, normalized: mlx.mlx_array, batch: c_int, seq: c_int) !?mlx.mlx_array {
    if (!geometry(batch, seq) or mlx.mlx_array_dtype(up) != .bfloat16 or mlx.mlx_array_dtype(normalized) != .bfloat16) return null;
    const shape4 = mlx.getShape(normalized);
    if (shape4.len != 4) return null;
    const hc = shape4[2];
    const hidden = shape4[3];
    if (hc < 1 or hc > 8 or hidden < 128 or hidden > 4096 or @mod(hidden, 128) != 0) return null;
    const count = @as(usize, @intCast(batch * seq * hc * hidden));
    if (mlx.mlx_array_size(up) != count or mlx.mlx_array_size(normalized) != count) return null;
    const cfg = mlx.mlx_fast_metal_kernel_config_new();
    defer _ = mlx.mlx_fast_metal_kernel_config_free(cfg);
    const shape = [_]c_int{ batch, seq, hidden };
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_output_arg(cfg, &shape, 3, .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_grid(cfg, batch * seq * hidden, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_set_thread_group(cfg, 256, 1, 1));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, "T", .bfloat16));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "HC", hc));
    try mlx.check(mlx.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, "H", hidden));
    const rows = mlx.mlx_array_new_int(batch * seq);
    defer _ = mlx.mlx_array_free(rows);
    const ins = mlx.mlx_vector_array_new_data(&.{ up, normalized, try sigmoidTable(s), rows }, 4);
    defer _ = mlx.mlx_vector_array_free(ins);
    var outs = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(outs);
    try mlx.check(mlx.mlx_fast_metal_kernel_apply(&outs, try getKernel(1), ins, cfg, s));
    var out = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(out);
    try mlx.check(mlx.mlx_vector_array_get(&out, outs, 0));
    return out;
}

test "hc prefill: pending write, normalized streams, native inject reduction and BF16 mix across shapes" {
    try mlx.requireGpu();
    const hp = @import("hc_prefill.zig");
    const s = mlx.gpuStream();
    var cache = try KVCache.init(testing.allocator, 0);
    defer cache.deinit();
    var xfm = std.mem.zeroInit(Transformer, .{ .s = s, .allocator = testing.allocator, .cache = cache });
    xfm.compileQwen4Hc();
    defer inline for (.{ xfm.compiled_hc_silu, xfm.compiled_hc_mix, xfm.compiled_hc_inj, xfm.compiled_hc_write }) |slot| {
        if (slot) |closure| _ = mlx.mlx_closure_free(closure);
    };
    try testing.expect(xfm.compiled_hc_write != null and xfm.compiled_hc_mix != null);
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
        const written = (try Transformer.applyClosure(xfm.compiled_hc_write, &.{ x, wo, wi })) orelse return error.MissingCompiledHc;
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
            const expected = (try Transformer.applyClosure(xfm.compiled_hc_mix, &.{ x, normed })) orelse return error.MissingCompiledHc;
            defer _ = mlx.mlx_array_free(expected);
            const mixed = (try hp.mix(s, x, normed, b, seq)) orelse return error.HcFusedDeclined;
            defer _ = mlx.mlx_array_free(mixed);
            try testing.expectEqual(@as(f32, 0), try attn256MaxDiff(mixed, expected, s));
        }
    }
}

test "hc prefill: unsupported configuration and coalesced chunk bounds decline" {
    const hp = @import("hc_prefill.zig");
    const inject = mlx.mlx_array_new_data(&[_]u16{0}, &.{1}, 1, .bfloat16);
    defer _ = mlx.mlx_array_free(inject);
    try testing.expect(hp.eligible(2, 4351, 4, 128, inject, .bfloat16));
    try testing.expect(!hp.eligible(2, 4352, 4, 128, inject, .bfloat16));
    for ([_][4]u32{ .{ 0, 17, 4, 2560 }, .{ 3, 17, 4, 2560 }, .{ 1, 16, 4, 2560 }, .{ 1, 8704, 4, 2560 }, .{ 1, 17, 0, 2560 }, .{ 1, 17, 9, 2560 }, .{ 1, 17, 4, 64 }, .{ 1, 17, 4, 4097 }, .{ 1, 17, 4, 129 } }) |shape| {
        try testing.expect(!hp.eligible(@intCast(shape[0]), @intCast(shape[1]), shape[2], shape[3], inject, .bfloat16));
    }
    try testing.expectEqual(@as(usize, @intCast(hp.max_seq)), @import("generate.zig").nextChunkEnd(0, 8703, 8192, false, 0, 0, true));
}
