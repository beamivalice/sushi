//! Trunk projections stored as STQ trellis banks, through the Flash-Next layers that read them: loading beside affine and
//! dense tensors, stacking, the head, and each layer against its dense twin (the bank decoded on the host).
const std = @import("std");
const mlx = @import("../mlx.zig");
const trf = @import("../transformer.zig");
const trunk_stq = @import("../trunk_stq.zig");
const mtp_mod = @import("../mtp.zig");
const model_mod = @import("../model.zig");
const sushi_exl3 = @import("sushi_exl3");
const qwen4_forward = @import("qwen4_forward.zig");
const test_support = @import("../test_support.zig");
const qwen4_test_support = @import("qwen4_test_support.zig");
const testing = std.testing;
const expert_exl3 = sushi_exl3.format;
const attn256RandBf16Scaled = test_support.attn256RandBf16Scaled;
const qwen4ReadF32 = test_support.qwen4ReadF32;
const FullAttnWeights = trf.FullAttnWeights;
const KVCache = trf.KVCache;
const ModelConfig = model_mod.ModelConfig;
const Qwen4TestSlot = qwen4_test_support.Qwen4TestSlot;
const Transformer = trf.Transformer;
const Weights = model_mod.Weights;

/// A trellis projection through the kernels against its host-decoded dense twin in bf16: the kernels' bar plus the
/// twin's own weight rounding (a GDN layer on it measures 3.6e-3 and 4.7e-3).
pub const STQ_PROJECTION_BAR: f64 = 1e-2;

/// Relative RMS error of `got` against `want`, both read as f32.
pub fn relRms(s: mlx.mlx_stream, got: mlx.mlx_array, want: mlx.mlx_array) !f64 {
    const g = try qwen4ReadF32(testing.allocator, got, s);
    defer testing.allocator.free(g);
    const w = try qwen4ReadF32(testing.allocator, want, s);
    defer testing.allocator.free(w);
    try testing.expectEqual(g.len, w.len);
    var ss: f64 = 0;
    var ref: f64 = 0;
    for (g, w) |a, b| {
        try testing.expect(std.math.isFinite(a));
        ss += (@as(f64, a) - b) * (@as(f64, a) - b);
        ref += @as(f64, b) * b;
    }
    return @sqrt(ss / @max(ref, 1e-30));
}

fn rand(rnd: std.Random, shape: []const c_int, scale: f32, s: mlx.mlx_stream) !mlx.mlx_array {
    return attn256RandBf16Scaled(rnd, shape, scale, s);
}

fn free(t: trunk_stq.Triple) void {
    inline for (.{ t.w, t.s, t.b }) |a| _ = mlx.mlx_array_free(a);
}

fn bareTransformer(s: mlx.mlx_stream, cache: KVCache) Transformer {
    return std.mem.zeroInit(Transformer, .{ .s = s, .allocator = testing.allocator, .cache = cache });
}

test "trunk STQ: qmatmul serves a trellis triple as its decoded dense twin and refuses one without trunk_quant" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var cache = try KVCache.init(testing.allocator, 0);
    defer cache.deinit();
    var xfm = bareTransformer(s, cache);
    const dec = expert_exl3.Decode.tabled(.spec1, .w15);
    xfm.config.trunk_quant = dec;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var prng = std.Random.DefaultPrng.init(0x7a11);
    const bank = try trunk_stq.TestBank.init(arena.allocator(), prng.random(), 256, 384, 64);
    const stq = try bank.triple(s);
    defer free(stq);
    const twin = try bank.twin(arena.allocator(), dec, s);
    defer free(twin);
    for ([_]c_int{ 1, 3, 40 }) |rows| {
        const x = try rand(prng.random(), &.{ 1, rows, 256 }, 4.0, s);
        defer _ = mlx.mlx_array_free(x);
        const got = try xfm.qmatmul(x, stq.w, stq.s, stq.b);
        defer _ = mlx.mlx_array_free(got);
        const want = try xfm.qmatmul(x, twin.w, twin.s, twin.b);
        defer _ = mlx.mlx_array_free(want);
        try testing.expectEqualSlices(c_int, &.{ 1, rows, 384 }, mlx.getShape(got));
        try testing.expect(try relRms(s, got, want) < STQ_PROJECTION_BAR);
    }
    xfm.config.trunk_quant = null;
    const x = try rand(prng.random(), &.{ 1, 1, 256 }, 4.0, s);
    defer _ = mlx.mlx_array_free(x);
    try testing.expectError(error.TrunkQuantMissing, xfm.qmatmul(x, stq.w, stq.s, stq.b));
}

test "trunk STQ: the joined decode and verify projections decline a trellis triple and leave it to the per-slot path" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var prng = std.Random.DefaultPrng.init(0x7a12);
    const bank = try trunk_stq.TestBank.init(arena.allocator(), prng.random(), 2560, 640, 48);
    const stq = try bank.triple(s);
    defer free(stq);
    const x = try rand(prng.random(), &.{ 1, 2, 2560 }, 4.0, s);
    defer _ = mlx.mlx_array_free(x);
    try testing.expectEqual(@as(?mlx.mlx_array, null), try trf.decodeRowsProjection(s, x, stq.w, stq.s, stq.b, &.{ 1, 1 }));
    try testing.expectEqual(@as(?mlx.mlx_array, null), try trf.verifyJoinedProjection(s, x, stq.w, stq.s, stq.b, &.{ 1, 1 }, .shared));
}

/// A named-tensor map entry of `shape` zeros in `dt`, standing in for a stored tensor the loader only binds.
fn putZeros(weights: *Weights, name: []const u8, shape: []const c_int, dt: mlx.mlx_dtype, s: mlx.mlx_stream) !void {
    var z = mlx.mlx_array_new();
    errdefer _ = mlx.mlx_array_free(z);
    try mlx.check(mlx.mlx_zeros(&z, shape.ptr, shape.len, dt, s));
    const key = try weights.allocator.dupe(u8, name);
    errdefer weights.allocator.free(key);
    try weights.map.put(key, z);
}

fn putBank(weights: *Weights, base: []const u8, bank: *const trunk_stq.TestBank) !void {
    const t = bank.tensors();
    inline for (.{ "trellis", "suh", "svh" }, 0..) |suffix, i| {
        const key = try std.fmt.allocPrint(weights.allocator, "{s}." ++ suffix, .{base});
        try weights.map.put(key, t[i]);
    }
}

/// A one-layer weight map with the norms every attention layer reads.
fn layerWeights(s: mlx.mlx_stream) !Weights {
    var weights = Weights.init(testing.allocator);
    errdefer weights.deinit();
    for ([_][]const u8{ "input_layernorm.weight", "post_attention_layernorm.weight" }) |n| {
        var key: [96]u8 = undefined;
        try putZeros(&weights, try std.fmt.bufPrint(&key, "model.layers.0.{s}", .{n}), &.{128}, .bfloat16, s);
    }
    for ([_][]const u8{ "q_norm", "k_norm" }) |n| {
        var key: [96]u8 = undefined;
        try putZeros(&weights, try std.fmt.bufPrint(&key, "model.layers.0.self_attn.{s}.weight", .{n}), &.{64}, .bfloat16, s);
    }
    return weights;
}

fn freeLoaded(loaded: anytype) void {
    testing.allocator.free(loaded.moe_layers);
    for (loaded.ssm_entries) |*e| {
        _ = mlx.mlx_array_free(e.conv_state);
        _ = mlx.mlx_array_free(e.ssm_state);
    }
    testing.allocator.free(loaded.ssm_entries);
    for (loaded.owned_bf16) |a| _ = mlx.mlx_array_free(a);
    testing.allocator.free(loaded.owned_bf16);
}

test "trunk STQ: a layer loads each projection as the pack stores it, trellis beside affine beside dense" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var prng = std.Random.DefaultPrng.init(0x7a13);
    const rnd = prng.random();
    const dec = expert_exl3.Decode.tabled(.spec1, .w15);
    var weights = try layerWeights(s);
    defer weights.deinit();
    // q and the gate are trellis banks, v and up stay affine, down is a plain bf16 weight; k and o are trellis too.
    const q = try trunk_stq.TestBank.init(arena.allocator(), rnd, 128, 256, 48);
    const k = try trunk_stq.TestBank.init(arena.allocator(), rnd, 128, 128, 32);
    const o = try trunk_stq.TestBank.init(arena.allocator(), rnd, 256, 128, 64);
    const gate = try trunk_stq.TestBank.init(arena.allocator(), rnd, 128, 256, 48);
    try putBank(&weights, "model.layers.0.self_attn.q_proj", &q);
    try putBank(&weights, "model.layers.0.self_attn.k_proj", &k);
    try putBank(&weights, "model.layers.0.self_attn.o_proj", &o);
    try putBank(&weights, "model.layers.0.mlp.gate_proj", &gate);
    for ([_][]const u8{ "self_attn.v_proj", "mlp.up_proj" }) |base| {
        inline for (.{ .{ "weight", .uint32, 16 }, .{ "scales", .bfloat16, 2 }, .{ "biases", .bfloat16, 2 } }) |part| {
            var key: [96]u8 = undefined;
            try putZeros(&weights, try std.fmt.bufPrint(&key, "model.layers.0.{s}.{s}", .{ base, part[0] }), &.{ 128, part[2] }, part[1], s);
        }
    }
    try putZeros(&weights, "model.layers.0.mlp.down_proj.weight", &.{ 128, 256 }, .bfloat16, s);

    var config = ModelConfig{ .model_type = "llama", .weight_prefix = "model", .num_hidden_layers = 1, .quant_bits = 4, .quant_group_size = 64, .trunk_quant = dec };
    var name_buf: [256]u8 = undefined;
    const loaded = try trf.initMoeLayers(testing.allocator, config, &weights, &name_buf, s);
    defer freeLoaded(loaded);
    const fa = loaded.moe_layers[0].attn.full;
    for ([_]mlx.mlx_array{ fa.q_w, fa.k_w, fa.o_w }) |w| try testing.expect(trunk_stq.isStq(w));
    try testing.expectEqualSlices(c_int, &.{ 1, 8, 16, 48 }, mlx.getShape(fa.q_w));
    try testing.expectEqualSlices(c_int, &.{ 1, 8, 8, 32 }, mlx.getShape(fa.k_w));
    try testing.expectEqualSlices(c_int, &.{ 1, 16, 8, 64 }, mlx.getShape(fa.o_w));
    try testing.expectEqual(mlx.mlx_dtype.uint32, mlx.mlx_array_dtype(fa.v_w));
    try testing.expectEqual(weights.get("model.layers.0.self_attn.v_proj.scales").?.ctx, fa.v_s.ctx);
    const dense = loaded.moe_layers[0].mlp.dense;
    try testing.expect(trunk_stq.isStq(dense.gate_w));
    try testing.expectEqualSlices(c_int, &.{ 1, 8, 16, 48 }, mlx.getShape(dense.gate_w));
    try testing.expectEqual(mlx.mlx_dtype.uint32, mlx.mlx_array_dtype(dense.up_w));
    try testing.expectEqualSlices(c_int, &.{ 256, 128 }, mlx.getShape(dense.down_w));
    try testing.expect(dense.down_s.ctx == null);

    // A trellis bank with no trunk_quant block is a named refusal.
    config.trunk_quant = null;
    try testing.expectError(error.TrunkQuantMissing, trf.initMoeLayers(testing.allocator, config, &weights, &name_buf, s));
}

test "trunk STQ: a trellis lm_head reads through the full head and keeps no coarse rerank copy" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var cache = try KVCache.init(testing.allocator, 0);
    defer cache.deinit();
    var xfm = bareTransformer(s, cache);
    const dec = expert_exl3.Decode.tabled(.spec1, .w15);
    xfm.config.trunk_quant = dec;
    xfm.config.hidden_size = 128;
    xfm.config.vocab_size = 1024;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var prng = std.Random.DefaultPrng.init(0x7a14);
    const rnd = prng.random();
    const bank = try trunk_stq.TestBank.init(arena.allocator(), rnd, 128, 1024, 96);
    const head = try bank.triple(s);
    defer free(head);
    const twin = try bank.twin(arena.allocator(), dec, s);
    defer free(twin);
    xfm.lm_head_w = head.w;
    xfm.lm_head_s = head.s;
    xfm.lm_head_b = head.b;
    try testing.expectEqual(@as(c_int, 1024), xfm.lmHeadRows());

    const x = try rand(rnd, &.{ 3, 1, 128 }, 4.0, s);
    defer _ = mlx.mlx_array_free(x);
    const want = try xfm.qmatmul(x, twin.w, twin.s, twin.b);
    defer _ = mlx.mlx_array_free(want);
    const logits = try xfm.lmHeadProject(x, false);
    defer _ = mlx.mlx_array_free(logits);
    try testing.expectEqualSlices(c_int, &.{ 3, 1, 1024 }, mlx.getShape(logits));
    try testing.expect(try relRms(s, logits, want) < STQ_PROJECTION_BAR);
    // The per-row head the MTP block reads, one row at a time through the same bank.
    const rows = try xfm.mtpProjectionRows(x, xfm.lm_head_w, xfm.lm_head_s, xfm.lm_head_b, true);
    defer _ = mlx.mlx_array_free(rows);
    try testing.expect(try relRms(s, rows, want) < STQ_PROJECTION_BAR);

    // No rows to gather: the coarse head declines, and the draft reads the head whole.
    try testing.expect(mtp_mod.buildRerankCoarse(s, &xfm, 3) == null);
    const x1 = try rand(rnd, &.{ 1, 1, 128 }, 4.0, s);
    defer _ = mlx.mlx_array_free(x1);
    const picked = try mtp_mod.fullReadoutArgmax(s, &xfm, x1, null);
    defer _ = mlx.mlx_array_free(picked);
    const truth = try xfm.qmatmul(x1, twin.w, twin.s, twin.b);
    defer _ = mlx.mlx_array_free(truth);
    const truth_host = try qwen4ReadF32(testing.allocator, truth, s);
    defer testing.allocator.free(truth_host);
    const id = try qwen4ReadF32(testing.allocator, picked, s);
    defer testing.allocator.free(id);
    var top: f32 = -std.math.inf(f32);
    var ss: f64 = 0;
    for (truth_host) |v| {
        top = @max(top, v);
        ss += @as(f64, v) * v;
    }
    const rms: f32 = @floatCast(@sqrt(ss / @as(f64, @floatFromInt(truth_host.len))));
    // The pick may differ from the twin's only where the twin's own top two sit inside the bf16 weight rounding.
    try testing.expect(truth_host[@intFromFloat(id[0])] >= top - 0.05 * rms);
}

test "trunk STQ: a layer stacks the projections that share an input and drops the stored banks" {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var prng = std.Random.DefaultPrng.init(0x7a15);
    const rnd = prng.random();
    const dec = expert_exl3.Decode.tabled(.spec1, .w15);
    var weights = try layerWeights(s);
    defer weights.deinit();
    const bases = [_][]const u8{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "mlp.gate_proj", "mlp.up_proj" };
    const outs = [_]usize{ 256, 128, 128, 256, 256 };
    var twins: [5][]f32 = undefined;
    for (bases, outs, &twins) |base, out, *twin| {
        const bank = try trunk_stq.TestBank.init(arena.allocator(), rnd, 128, out, 48);
        twin.* = try bank.dense(arena.allocator(), dec);
        var key: [96]u8 = undefined;
        try putBank(&weights, try std.fmt.bufPrint(&key, "model.layers.0.{s}", .{base}), &bank);
    }
    inline for (.{ .{ "weight", .uint32, 16 }, .{ "scales", .bfloat16, 2 }, .{ "biases", .bfloat16, 2 } }) |part| {
        var key: [96]u8 = undefined;
        try putZeros(&weights, try std.fmt.bufPrint(&key, "model.layers.0.self_attn.o_proj.{s}", .{part[0]}), &.{ 128, part[2] }, part[1], s);
    }
    try putZeros(&weights, "model.layers.0.mlp.down_proj.weight", &.{ 128, 256 }, .bfloat16, s);

    const config = ModelConfig{ .model_type = "llama", .weight_prefix = "model", .num_hidden_layers = 1, .quant_bits = 4, .quant_group_size = 64, .trunk_quant = dec };
    var name_buf: [256]u8 = undefined;
    const loaded = try trf.initMoeLayers(testing.allocator, config, &weights, &name_buf, s);
    defer freeLoaded(loaded);
    const fa = loaded.moe_layers[0].attn.full;
    const dense = loaded.moe_layers[0].mlp.dense;
    try testing.expectEqual(@as(usize, 3), fa.qkv_stack.?.count);
    try testing.expectEqual(@as(usize, 2), dense.gate_up_stack.?.count);
    const counts = @import("../layer_weights.zig").TrunkCounts.of(loaded.moe_layers);
    try testing.expectEqual(@as(usize, 5), counts.banks);
    try testing.expectEqual(@as(usize, 2), counts.stacks);
    for (bases) |base| {
        var key: [96]u8 = undefined;
        try testing.expectEqual(@as(?mlx.mlx_array, null), weights.get(try std.fmt.bufPrint(&key, "model.layers.0.{s}.trellis", .{base})));
    }
    // The parts are views of the stack: the layer still serves a chunk wider than a stacked launch.
    try testing.expect(trunk_stq.isStq(fa.v_w));
    try testing.expectEqualSlices(c_int, &.{ 1, 8, 8, 48 }, mlx.getShape(fa.v_w));

    var cache = try KVCache.init(testing.allocator, 0);
    defer cache.deinit();
    var xfm = bareTransformer(s, cache);
    xfm.config.trunk_quant = dec;
    const x = try rand(rnd, &.{ 1, 3, 128 }, 4.0, s);
    defer _ = mlx.mlx_array_free(x);
    var qkv = (try qwen4_forward.stackedAttnProjections(&xfm, x, &fa)).?;
    defer qkv.deinit();
    const pair = (try xfm.stackedPair(x, dense.gate_up_stack)).?;
    defer for (pair) |a| {
        _ = mlx.mlx_array_free(a);
    };
    const xs = try qwen4ReadF32(testing.allocator, x, s);
    defer testing.allocator.free(xs);
    for ([_]mlx.mlx_array{ qkv.q, qkv.k, qkv.v, pair[0], pair[1] }, outs, twins) |got, out, twin| {
        const want = try arena.allocator().alloc(f32, 3 * out);
        @memset(want, 0);
        for (0..3) |r| for (0..128) |k| for (0..out) |j| {
            want[r * out + j] += xs[r * 128 + k] * twin[k * out + j];
        };
        const w = mlx.mlx_array_new_data(want.ptr, &[_]c_int{ 1, 3, @intCast(out) }, 3, .float32);
        defer _ = mlx.mlx_array_free(w);
        try testing.expect(try relRms(s, got, w) < STQ_PROJECTION_BAR);
    }
    // Wider than a stacked launch: nothing is made, and the caller projects each alone.
    const wide = try rand(rnd, &.{ 1, 20, 128 }, 4.0, s);
    defer _ = mlx.mlx_array_free(wide);
    try testing.expect((try qwen4_forward.stackedAttnProjections(&xfm, wide, &fa)) == null);
}

test "trunk STQ: a full-attention layer on trellis projections tracks its dense twin at prefill, decode and verify widths" {
    try attnTwinCase(false);
    try attnTwinCase(true);
}

fn attnTwinCase(stacked: bool) !void {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var cache = try KVCache.init(testing.allocator, 0);
    defer cache.deinit();
    var xfm = bareTransformer(s, cache);
    const dec = expert_exl3.Decode.tabled(.spec1, .w15);
    const hidden: usize = 128;
    xfm.config.trunk_quant = dec;
    xfm.config.hidden_size = hidden;
    xfm.config.num_attention_heads = 2;
    xfm.config.num_key_value_heads = 1;
    xfm.config.head_dim = 128;
    xfm.config.query_pre_attn_scalar = 128;
    xfm.config.partial_rotary_factor = 0.25;
    xfm.config.attn_output_gate = true;
    xfm.config.rms_norm_eps = 1e-6;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var prng = std.Random.DefaultPrng.init(0x7a16);
    const rnd = prng.random();
    // q carries the output gate: 2 heads x 128 x 2.
    const dims = [4][2]usize{ .{ hidden, 512 }, .{ hidden, 128 }, .{ hidden, 128 }, .{ 256, hidden } };
    var stq: [4]trunk_stq.Triple = undefined;
    var twin: [4]trunk_stq.Triple = undefined;
    var held: std.ArrayList(mlx.mlx_array) = .empty;
    defer {
        for (held.items) |a| _ = mlx.mlx_array_free(a);
        held.deinit(testing.allocator);
    }
    for (dims, &stq, &twin) |d, *q, *w| {
        const bank = try trunk_stq.TestBank.init(arena.allocator(), rnd, d[0], d[1], 48);
        q.* = try bank.triple(s);
        try held.appendSlice(testing.allocator, &.{ q.w, q.s, q.b });
        w.* = try bank.twin(arena.allocator(), dec, s);
    }
    defer for (twin) |t| free(t);
    const stack = if (stacked) (try trunk_stq.fuse(&.{ &stq[0], &stq[1], &stq[2] }, &held, testing.allocator, s)).? else null;
    const q_norm = try rand(rnd, &.{128}, 2.0, s);
    defer _ = mlx.mlx_array_free(q_norm);
    const k_norm = try rand(rnd, &.{128}, 2.0, s);
    defer _ = mlx.mlx_array_free(k_norm);
    const none: mlx.mlx_array = .{ .ctx = null };
    const fa_stq = FullAttnWeights{
        .q_w = stq[0].w, .q_s = stq[0].s, .q_b = stq[0].b,
        .k_w = stq[1].w, .k_s = stq[1].s, .k_b = stq[1].b,
        .v_w = stq[2].w, .v_s = stq[2].s, .v_b = stq[2].b,
        .o_w = stq[3].w, .o_s = stq[3].s, .o_b = stq[3].b,
        .q_norm = q_norm, .k_norm = k_norm, .qkv_stack = stack,
    };
    const fa_twin = FullAttnWeights{
        .q_w = twin[0].w, .q_s = none, .q_b = none,
        .k_w = twin[1].w, .k_s = none, .k_b = none,
        .v_w = twin[2].w, .v_s = none, .v_b = none,
        .o_w = twin[3].w, .o_s = none, .o_b = none,
        .q_norm = q_norm, .k_norm = k_norm,
    };
    const slots = [2]*Qwen4TestSlot{ try Qwen4TestSlot.init(testing.allocator, 1), try Qwen4TestSlot.init(testing.allocator, 1) };
    defer for (slots) |sl| sl.deinit(testing.allocator);
    // A prefill chunk past the stacked launch, a verify block, then single decode ticks.
    var offset: c_int = 0;
    for ([_]c_int{ 12, 4, 1, 1 }) |seq| {
        const x = try rand(rnd, &.{ 1, seq, @intCast(hidden) }, 2.0, s);
        defer _ = mlx.mlx_array_free(x);
        var outs: [2]mlx.mlx_array = undefined;
        for ([_]*const FullAttnWeights{ &fa_stq, &fa_twin }, slots, &outs) |fa, sl, *o| {
            o.* = try xfm.qwen4AttnWith(&sl.ctx, x, fa, &sl.entries[0], 0, offset, 0, 1, seq, seq > 1);
        }
        defer for (outs) |o| {
            _ = mlx.mlx_array_free(o);
        };
        try testing.expect(try relRms(s, outs[0], outs[1]) < STQ_PROJECTION_BAR);
        offset += seq;
        for (slots) |sl| sl.cache.step = @intCast(offset);
    }
}

/// `a` widened to f32, which keeps every bf16 bit.
fn asF32(a: mlx.mlx_array, s: mlx.mlx_stream) !mlx.mlx_array {
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&out, a, .float32, s));
    return out;
}

fn withGdnProjections(base: *const trf.LinearAttnWeights, qkv: trunk_stq.Triple, z: trunk_stq.Triple, out: trunk_stq.Triple) trf.LinearAttnWeights {
    var la = base.*;
    la.qkv_w = qkv.w;
    la.qkv_s = qkv.s;
    la.qkv_b = qkv.b;
    la.z_w = z.w;
    la.z_s = z.s;
    la.z_b = z.b;
    la.out_w = out.w;
    la.out_s = out.s;
    la.out_b = out.b;
    return la;
}

test "trunk STQ: a GDN layer on trellis qkv, z and out projections tracks its dense twin at decode, verify and prefill widths" {
    try gdnTwinCase(false);
    try gdnTwinCase(true);
}

fn gdnTwinCase(stacked: bool) !void {
    try mlx.requireGpu();
    const s = mlx.gpuStream();
    var cache = try KVCache.init(testing.allocator, 0);
    defer cache.deinit();
    var xfm = bareTransformer(s, cache);
    defer inline for (.{ &xfm.gdn_q_scale, &xfm.gdn_k_scale, &xfm.gdn_eps }) |p| {
        if (p.*) |arr| _ = mlx.mlx_array_free(arr);
    };
    const dec = expert_exl3.Decode.tabled(.spec1, .w15);
    const hk: c_int = 16;
    const hv: c_int = 48;
    const hidden: c_int = 128;
    const c_dim: c_int = 2 * hk * 128 + hv * 128;
    xfm.config.trunk_quant = dec;
    xfm.config.linear_num_key_heads = @intCast(hk);
    xfm.config.linear_num_value_heads = @intCast(hv);
    xfm.config.linear_key_head_dim = 128;
    xfm.config.linear_value_head_dim = 128;
    xfm.config.linear_conv_kernel_dim = 4;
    xfm.config.rms_norm_eps = 1e-6;
    xfm.config.kda_sigmoid_out_gate = true;
    xfm.config.gdn_state_f32 = true;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    var prng = std.Random.DefaultPrng.init(0x6d5e2);
    const rnd = prng.random();
    const banks = [3]trunk_stq.TestBank{
        try trunk_stq.TestBank.init(arena.allocator(), rnd, @intCast(hidden), @intCast(c_dim), 48),
        try trunk_stq.TestBank.init(arena.allocator(), rnd, @intCast(hidden), @intCast(hv * 128), 48),
        try trunk_stq.TestBank.init(arena.allocator(), rnd, @intCast(hv * 128), @intCast(hidden), 48),
    };
    var stq: [3]trunk_stq.Triple = undefined;
    var twin: [3]trunk_stq.Triple = undefined;
    var held: std.ArrayList(mlx.mlx_array) = .empty;
    defer {
        for (held.items) |a| _ = mlx.mlx_array_free(a);
        held.deinit(testing.allocator);
    }
    for (banks, &stq, &twin) |b, *q, *w| {
        q.* = try b.triple(s);
        try held.appendSlice(testing.allocator, &.{ q.w, q.s, q.b });
        w.* = try b.twin(arena.allocator(), dec, s);
    }
    defer for (twin) |t| free(t);
    const stack = if (stacked) (try trunk_stq.fuse(&.{ &stq[0], &stq[1] }, &held, testing.allocator, s)).? else null;
    const none: mlx.mlx_array = .{ .ctx = null };
    const base = trf.LinearAttnWeights{
        .qkv_w = none, .qkv_s = none, .qkv_b = none,
        .z_w = none, .z_s = none, .z_b = none,
        .a_w = try rand(rnd, &.{ hidden, hv }, 2.0, s),
        .a_s = none,
        .a_b = none,
        .b_w = try rand(rnd, &.{ hidden, hv }, 2.0, s),
        .b_s = none,
        .b_b = none,
        .conv1d_w = try rand(rnd, &.{ c_dim, 4, 1 }, 1.0, s),
        .A_log = try rand(rnd, &.{hv}, 2.0, s),
        .dt_bias = try rand(rnd, &.{hv}, 2.0, s),
        .norm_w = try rand(rnd, &.{128}, 2.0, s),
        .out_w = none, .out_s = none, .out_b = none,
    };
    defer inline for (.{ base.a_w, base.b_w, base.conv1d_w, base.A_log, base.dt_bias, base.norm_w }) |arr| {
        _ = mlx.mlx_array_free(arr);
    };
    var la_stq = withGdnProjections(&base, stq[0], stq[1], stq[2]);
    la_stq.qkvz_stack = stack;
    const la_twin = withGdnProjections(&base, twin[0], twin[1], twin[2]);
    const conv0 = try rand(rnd, &.{ 1, 3, c_dim }, 4.0, s);
    defer _ = mlx.mlx_array_free(conv0);
    const state_bf = try rand(rnd, &.{ 1, hv, 128, 128 }, 1.0, s);
    defer _ = mlx.mlx_array_free(state_bf);
    const state0 = try asF32(state_bf, s);
    defer _ = mlx.mlx_array_free(state0);
    for ([_]c_int{ 1, 5, 20 }) |seq| {
        var entries: [2]trf.SSMCacheEntry = undefined;
        for (&entries) |*e| {
            e.* = .{ .conv_state = mlx.mlx_array_new(), .ssm_state = mlx.mlx_array_new(), .initialized = true };
            try mlx.check(mlx.mlx_array_set(&e.conv_state, conv0));
            try mlx.check(mlx.mlx_array_set(&e.ssm_state, state0));
        }
        defer for (&entries) |*e| {
            _ = mlx.mlx_array_free(e.conv_state);
            _ = mlx.mlx_array_free(e.ssm_state);
        };
        const x = try rand(rnd, &.{ 1, seq, hidden }, 2.0, s);
        defer _ = mlx.mlx_array_free(x);
        const got = try xfm.gatedDeltaNetProjected(x, &la_stq, &entries[0], 0, 1, seq, seq > 1, null, false);
        defer _ = mlx.mlx_array_free(got);
        const want = try xfm.gatedDeltaNetProjected(x, &la_twin, &entries[1], 0, 1, seq, seq > 1, null, false);
        defer _ = mlx.mlx_array_free(want);
        try testing.expect(try relRms(s, got, want) < STQ_PROJECTION_BAR);
    }
}
