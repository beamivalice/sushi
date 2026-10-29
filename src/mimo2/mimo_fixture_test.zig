const trf = @import("../transformer.zig");
const std = @import("std");
const expert_stream_mod = @import("../expert_stream.zig");
const mlx = @import("../mlx.zig");
const kv_quant = @import("../kv_quant.zig");
const mimo_forward = @import("mimo_forward.zig");
const MIMO_VERIFY_ROWS_MAX = mimo_forward.MIMO_VERIFY_ROWS_MAX;
const KVQuantConfig = kv_quant.KVQuantConfig;
const model_mod = @import("../model.zig");
const io_util_mod = @import("../io_util.zig");
const ModelConfig = model_mod.ModelConfig;
const Weights = model_mod.Weights;
const diagEnvOn = io_util_mod.diagEnvOn;
const testing = std.testing;
const expectError = @import("../test_expect.zig").expectError;
const QKV_MPP_DECODE_MIN_TK = mimo_forward.QKV_MPP_DECODE_MIN_TK;
const stackMimoFixtureExperts = mimo_forward.stackMimoFixtureExperts;
const test_support = @import("../test_support.zig");
const mimoExpectClose = test_support.mimoExpectClose;
const qwen4ReadF32 = test_support.qwen4ReadF32;
const snapshotsBitEqual = test_support.snapshotsBitEqual;
const CaptureLayers = trf.CaptureLayers;
const ForwardCtx = trf.ForwardCtx;
const KVCache = trf.KVCache;
const MimoExl3MoeHarness = trf.MimoExl3MoeHarness;
const Transformer = trf.Transformer;
const loadSwitchMlpBank = trf.loadSwitchMlpBank;

/// Prefill `prefix`, then either decode `rows` ids one tick at a time or verify them in
/// one `verify_rows` forward; the logits rows, `[rows, vocab]` f32.
fn mimoVerifyOrTicks(alloc: std.mem.Allocator, xfm: *Transformer, ids: []const i32, prefix: usize, rows: usize, verify: bool, kv: KVQuantConfig) ![]f32 {
    const s = xfm.s;
    try xfm.cache.reinit(xfm.config.num_hidden_layers, kv);
    xfm.cache.setSwaRing(xfm.config.sliding_window);
    xfm.moe_seq_offset = 0;
    var ctx = xfm.defaultCtx();
    const pre = mlx.mlx_array_new_data(ids.ptr, &[_]c_int{ 1, @intCast(prefix) }, 2, .int32);
    defer _ = mlx.mlx_array_free(pre);
    const pre_logits = try xfm.forwardWith(&ctx, pre);
    try mlx.check(mlx.mlx_array_eval(pre_logits));
    _ = mlx.mlx_array_free(pre_logits);
    const vocab: usize = @intCast(xfm.config.vocab_size);
    const out = try alloc.alloc(f32, rows * vocab);
    errdefer alloc.free(out);
    if (verify) {
        ctx.verify_rows = true;
        const block = mlx.mlx_array_new_data(ids[prefix..].ptr, &[_]c_int{ 1, @intCast(rows) }, 2, .int32);
        defer _ = mlx.mlx_array_free(block);
        const logits = try xfm.forwardWith(&ctx, block);
        defer _ = mlx.mlx_array_free(logits);
        const host = try qwen4ReadF32(alloc, logits, s);
        defer alloc.free(host);
        @memcpy(out, host);
    } else {
        for (0..rows) |r| {
            const one = mlx.mlx_array_new_data(ids[prefix + r ..].ptr, &[_]c_int{ 1, 1 }, 2, .int32);
            defer _ = mlx.mlx_array_free(one);
            const logits = try xfm.forwardWith(&ctx, one);
            defer _ = mlx.mlx_array_free(logits);
            const host = try qwen4ReadF32(alloc, logits, s);
            defer alloc.free(host);
            @memcpy(out[r * vocab ..][0..vocab], host);
        }
    }
    return out;
}

/// One decode slot of the batched-decode test: its own ringed cache and position.
const MimoBatchTestSlot = struct {
    cache: KVCache,
    offset: usize = 0,

    fn ctx(self: *MimoBatchTestSlot, argmax_only: bool) ForwardCtx {
        return .{ .cache = &self.cache, .moe_seq_offset = &self.offset, .ssm_entries = null, .capture_hidden = null, .vision_embeddings = null, .argmax_only = argmax_only };
    }
};

fn mimoBatchTestPrefill(xfm: *Transformer, slot: *MimoBatchTestSlot, ids: []const i32, kv: KVQuantConfig) !void {
    const cache = try KVCache.initWithConfig(testing.allocator, xfm.config.num_hidden_layers, kv);
    slot.* = .{ .cache = cache };
    errdefer slot.cache.deinit();
    slot.cache.setSwaRing(xfm.config.sliding_window);
    var c = slot.ctx(false);
    const pre = mlx.mlx_array_new_data(ids.ptr, &[_]c_int{ 1, @intCast(ids.len) }, 2, .int32);
    defer _ = mlx.mlx_array_free(pre);
    const logits = try xfm.forwardWith(&c, pre);
    defer _ = mlx.mlx_array_free(logits);
    try mlx.check(mlx.mlx_array_eval(logits));
}

test "mimo_v2 EXL3 real shard routed fork matches CPU dump oracle" {
    const fixture_path = std.c.getenv("MIMO_DIFFERENTIAL_FIXTURE") orelse return error.SkipZigTest;
    const pack_path = std.c.getenv("MIMO_DIFFERENTIAL_PACK") orelse return error.SkipZigTest;
    const layer_text = std.c.getenv("MIMO_DIFFERENTIAL_LAYER") orelse "1";
    const layer = try std.fmt.parseInt(u32, std.mem.span(layer_text), 10);
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var config = try model_mod.parseConfig(t.io, alloc, std.mem.span(pack_path));
    defer config.deinit(alloc);
    var fixture = try model_mod.loadWeightsSingleFile(alloc, std.mem.span(fixture_path));
    defer fixture.deinit();
    var shards: [3]Weights = undefined;
    var loaded: usize = 0;
    defer for (shards[0..loaded]) |*w| w.deinit();
    var weights = Weights.init(alloc);
    defer weights.map.deinit();
    for ([_][]const u8{ "gate", "up", "down" }, 0..) |proj, i| {
        const path = try std.fmt.allocPrint(alloc, "{s}/model-exl3-L{d:0>2}-{s}_proj.safetensors", .{ std.mem.span(pack_path), layer, proj });
        shards[i] = try model_mod.loadWeightsSingleFile(alloc, path);
        loaded += 1;
        var it = shards[i].map.iterator();
        while (it.next()) |entry| try weights.map.put(entry.key_ptr.*, entry.value_ptr.*);
    }
    var buf: [256]u8 = undefined;
    const bank = try loadSwitchMlpBank(&weights, &buf, "model", layer, true, &config);
    var h = try MimoExl3MoeHarness.init(alloc, s, .{ .n = 40 }, 1, 191, .mcg);
    defer h.deinit();
    h.xfm.config = config;
    h.mw.router_w = fixture.get("router") orelse return error.MissingWeight;
    h.mw.expert_bias = fixture.get("bias") orelse return error.MissingWeight;
    h.mw.switch_gate_w = bank.gate_w;
    h.mw.switch_gate_s = bank.gate_s;
    h.mw.switch_gate_b = bank.gate_b;
    h.mw.switch_up_w = bank.up_w;
    h.mw.switch_up_s = bank.up_s;
    h.mw.switch_up_b = bank.up_b;
    h.mw.switch_down_w = bank.down_w;
    h.mw.switch_down_s = bank.down_s;
    h.mw.switch_down_b = bank.down_b;
    const input = fixture.get("x") orelse return error.MissingWeight;
    const oracle = fixture.get("oracle_y") orelse return error.MissingWeight;
    const expected_ids = fixture.get("ids") orelse return error.MissingWeight;
    const expected_scores = fixture.get("scores") orelse return error.MissingWeight;
    for ([_]mlx.mlx_array{ oracle, expected_ids, expected_scores }) |a| try mlx.check(mlx.mlx_array_eval(a));
    const want = mlx.mlx_array_data_float32(oracle) orelse return error.F32Unreadable;
    const want_ids = mlx.mlx_array_data_uint32(expected_ids) orelse return error.ExpertIdsUnreadable;
    const want_scores = mlx.mlx_array_data_float32(expected_scores) orelse return error.F32Unreadable;
    var xb = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(xb);
    try mlx.check(mlx.mlx_astype(&xb, input, .bfloat16, s));
    for ([_]c_int{ 1, 14, 33 }) |rows| {
        var x = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x);
        try mlx.check(mlx.mlx_tile(&x, xb, &.{ 1, rows, 1 }, 3, s));
        const logits = try h.xfm.mimoRouterLogits(x, &h.mw);
        defer _ = mlx.mlx_array_free(logits);
        const routes = try h.xfm.computeMimoRouting(logits, h.mw.expert_bias.?);
        defer _ = mlx.mlx_array_free(routes.inds);
        defer _ = mlx.mlx_array_free(routes.norm_scores);
        var ids32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(ids32);
        try mlx.check(mlx.mlx_astype(&ids32, routes.inds, .uint32, s));
        try mlx.check(mlx.mlx_array_eval(ids32));
        try mlx.check(mlx.mlx_array_eval(routes.norm_scores));
        const ids = mlx.mlx_array_data_uint32(ids32) orelse return error.ExpertIdsUnreadable;
        const scores = mlx.mlx_array_data_float32(routes.norm_scores) orelse return error.F32Unreadable;
        for (0..@intCast(rows)) |r| {
            for (0..config.num_experts_per_tok) |k| {
                try t.expectEqual(want_ids[k], ids[r * config.num_experts_per_tok + k]);
                try t.expectApproxEqAbs(want_scores[k], scores[r * config.num_experts_per_tok + k], 1e-6);
            }
        }
        const y = try h.xfm.moeMLP(x, &h.mw);
        defer _ = mlx.mlx_array_free(y);
        try t.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(y));
        var y32 = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(y32);
        try mlx.check(mlx.mlx_astype(&y32, y, .float32, s));
        try mlx.check(mlx.mlx_array_eval(y32));
        const got = mlx.mlx_array_data_float32(y32) orelse return error.F32Unreadable;
        var ss: f64 = 0;
        var ref: f64 = 0;
        for (0..@intCast(rows)) |r| {
            for (0..config.hidden_size) |d| {
                const actual = got[r * config.hidden_size + d];
                try t.expect(std.math.isFinite(actual) and std.math.isFinite(want[d]));
                const delta: f64 = actual - want[d];
                ss += delta * delta;
                ref += @as(f64, want[d]) * want[d];
            }
        }
        const rel = @sqrt(ss / @max(ref, 1e-20));
        if (diagEnvOn("SUSHI_EXL3_LAYER_UBENCH"))
            std.debug.print("mimo real routed layer={d} rows={d} relative RMS={d:.8}\n", .{ layer, rows, rel });
        if (!(rel < 0.01)) {
            std.debug.print("mimo real routed layer={d} rows={d} relative RMS={d:.8}\n", .{ layer, rows, rel });
            return error.TestExpectedEqual;
        }
    }
}

test "mimo_v2 EXL3 sliced real scales and outliers match the f32 oracle" {
    const path = std.c.getenv("MIMO_SLICED_REAL_FIXTURE") orelse return error.SkipZigTest;
    const t = std.testing;
    const s = mlx.gpuStream();
    if (!mlx.streamIsGpu(s)) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    var weights = try model_mod.loadWeightsSingleFile(alloc, std.mem.span(path));
    defer weights.deinit();
    var h = try MimoExl3MoeHarness.init(alloc, s, .{ .n = 40 }, 1, 313, .{ .codebook = .mcg, .window = .w12 });
    defer h.deinit();
    h.mw.router_w = weights.get("router") orelse return error.MissingWeight;
    h.mw.expert_bias = weights.get("bias") orelse return error.MissingWeight;
    h.mw.switch_gate_w = weights.get("gt") orelse return error.MissingWeight;
    h.mw.switch_gate_s = weights.get("gsuh") orelse return error.MissingWeight;
    h.mw.switch_gate_b = weights.get("gsvh") orelse return error.MissingWeight;
    h.mw.switch_up_w = weights.get("ut") orelse return error.MissingWeight;
    h.mw.switch_up_s = weights.get("usuh") orelse return error.MissingWeight;
    h.mw.switch_up_b = weights.get("usvh") orelse return error.MissingWeight;
    h.mw.switch_down_w = weights.get("dt") orelse return error.MissingWeight;
    h.mw.switch_down_s = weights.get("dsuh") orelse return error.MissingWeight;
    h.mw.switch_down_b = weights.get("dsvh") orelse return error.MissingWeight;
    const input = weights.get("x") orelse return error.MissingWeight;
    const oracle = weights.get("oracle_y") orelse return error.MissingWeight;
    try mlx.check(mlx.mlx_array_eval(oracle));
    const want = mlx.mlx_array_data_float32(oracle) orelse return error.F32Unreadable;
    for ([_]c_int{ 1, 14, 33 }) |rows| {
        var view = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(view);
        try mlx.check(mlx.mlx_slice(&view, input, &.{ 0, 0, 0 }, 3, &.{ 1, rows, 128 }, 3, &.{ 1, 1, 1 }, 3, s));
        var x = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(x);
        try mlx.check(mlx.mlx_astype(&x, view, .bfloat16, s));
        const y = try h.xfm.moeMLP(x, &h.mw);
        defer _ = mlx.mlx_array_free(y);
        var yf = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(yf);
        try mlx.check(mlx.mlx_astype(&yf, y, .float32, s));
        try mlx.check(mlx.mlx_array_eval(yf));
        const got = mlx.mlx_array_data_float32(yf) orelse return error.F32Unreadable;
        if (std.c.getenv("MIMO_SLICED_OUTPUT")) |base| {
            const name = try std.fmt.allocPrint(alloc, "{s}.{d}.f32", .{ std.mem.span(base), rows });
            try std.Io.Dir.cwd().writeFile(t.io, .{ .sub_path = name, .data = std.mem.sliceAsBytes(got[0..@as(usize, @intCast(rows * 128))]) });
        }
        var maximum: f64 = 0;
        for (0..@intCast(rows)) |r| {
            var ss: f64 = 0;
            var ref: f64 = 0;
            for (0..128) |d| {
                const i = r * 128 + d;
                try t.expect(std.math.isFinite(got[i]) and std.math.isFinite(want[i]));
                const delta: f64 = got[i] - want[i];
                ss += delta * delta;
                ref += @as(f64, want[i]) * want[i];
            }
            maximum = @max(maximum, @sqrt(ss / @max(ref, 1e-30)));
        }
        if (diagEnvOn("SUSHI_EXL3_LAYER_UBENCH") or maximum >= 0.01)
            std.debug.print("mimo sliced real rows={d} max row relative RMS={d:.8}\n", .{ rows, maximum });
        try t.expect(maximum < 0.01);
    }
}

test "mimo v2 raw per-expert load failure releases partial construction" {
    const model_dir = std.c.getenv("MIMO_V2_MODEL") orelse return error.SkipZigTest;
    try mlx.requireGpu();
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();

    var config = try model_mod.parseConfig(io, allocator, std.mem.span(model_dir));
    defer config.deinit(allocator);
    var weights = try model_mod.loadWeights(io, allocator, std.mem.span(model_dir));
    defer weights.deinit();
    model_mod.resolveWeightPrefix(&config, &weights);

    // The fixture's dense layer 0 binds before the first MoE layer asks for
    // packed switch_mlp banks that are absent from the raw HF layout.
    try expectError(error.MissingWeight, Transformer.init(io, allocator, config, &weights));
}

test "mimo v2 fixture full and cached forwards track the HF reference (MIMO_V2_MODEL + MIMO_V2_FIXTURE)" {
    const model_dir = std.c.getenv("MIMO_V2_MODEL") orelse return error.SkipZigTest;
    const fixture_path = std.c.getenv("MIMO_V2_FIXTURE") orelse return error.SkipZigTest;
    try mlx.requireGpu();
    const allocator = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();

    var config = try model_mod.parseConfig(io, allocator, std.mem.span(model_dir));
    defer if (config.ngram_table_path) |p| allocator.free(p);
    try testing.expectEqualStrings("mimo_v2", config.model_type);
    var weights = try model_mod.loadWeights(io, allocator, std.mem.span(model_dir));
    defer weights.deinit();
    try stackMimoFixtureExperts(&weights, config, s);
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Transformer.init(io, allocator, config, &weights);
    defer xfm.deinit();

    var fx = try model_mod.loadWeightsSingleFile(allocator, std.mem.span(fixture_path));
    defer fx.deinit();
    const ids_arr = fx.get("input_ids") orelse return error.MissingFixtureTensor;
    try mlx.check(mlx.mlx_array_eval(ids_arr));
    const ids_src = mlx.mlx_array_data_int32(ids_arr) orelse return error.Unreadable;
    const token_count: usize = mlx.mlx_array_size(ids_arr);
    const vocab: usize = @intCast(config.vocab_size);
    const ref_full = try qwen4ReadF32(allocator, fx.get("logits_full") orelse return error.MissingFixtureTensor, s);
    defer allocator.free(ref_full);
    const ref_cache = try qwen4ReadF32(allocator, fx.get("cache_logits") orelse return error.MissingFixtureTensor, s);
    defer allocator.free(ref_cache);
    try testing.expectEqual(token_count * vocab, ref_full.len);
    try testing.expectEqual(token_count * vocab, ref_cache.len);

    const full_shape = [_]c_int{ 1, @intCast(token_count) };
    const full_ids = mlx.mlx_array_new_data(@ptrCast(ids_src), &full_shape, 2, .int32);
    defer _ = mlx.mlx_array_free(full_ids);
    const layer_count: usize = @intCast(config.num_hidden_layers);
    const capture_ids = try allocator.alloc(u32, layer_count);
    defer allocator.free(capture_ids);
    const capture_out = try allocator.alloc(mlx.mlx_array, layer_count);
    defer {
        for (capture_out) |arr| {
            if (arr.ctx != null) _ = mlx.mlx_array_free(arr);
        }
        allocator.free(capture_out);
    }
    for (capture_ids, capture_out, 0..) |*id, *out, i| {
        id.* = @intCast(i);
        out.* = mlx.mlx_array_new();
    }
    var captures: CaptureLayers = .{ .ids = capture_ids, .out = capture_out };
    var full_ctx = xfm.defaultCtx();
    full_ctx.capture_layers = &captures;
    const full_logits = try xfm.forwardWith(&full_ctx, full_ids);
    defer _ = mlx.mlx_array_free(full_logits);
    const ours_full = try qwen4ReadF32(allocator, full_logits, s);
    defer allocator.free(ours_full);

    const embed = try xfm.embedding(full_ids);
    defer _ = mlx.mlx_array_free(embed);
    try testing.expectEqual(mlx.mlx_dtype.float32, mlx.mlx_array_dtype(embed));
    const ref_embed = try qwen4ReadF32(allocator, fx.get("embed_out") orelse return error.MissingFixtureTensor, s);
    defer allocator.free(ref_embed);
    const ours_embed = try qwen4ReadF32(allocator, embed, s);
    defer allocator.free(ours_embed);
    try mimoExpectClose("embeddings", ours_embed, ref_embed, 0);

    {
        var component_xfm = try Transformer.init(io, allocator, config, &weights);
        defer component_xfm.deinit();
        var component_ctx = component_xfm.defaultCtx();
        const lw = &component_xfm.moe_layers.?[0];
        const normed = try component_xfm.rmsNorm(embed, lw.input_norm);
        defer _ = mlx.mlx_array_free(normed);
        const norm_host = try qwen4ReadF32(allocator, normed, s);
        defer allocator.free(norm_host);
        const value_proj = try component_xfm.qmatmul(normed, lw.attn.full.v_w, lw.attn.full.v_s, lw.attn.full.v_b);
        defer _ = mlx.mlx_array_free(value_proj);
        const value_host = try qwen4ReadF32(allocator, value_proj, s);
        defer allocator.free(value_host);
        const source_qkv = try qwen4ReadF32(allocator, weights.get("model.layers.0.self_attn.qkv_proj.weight").?, s);
        defer allocator.free(source_qkv);
        const vrows = config.layerKVHeads(0) * config.layerVHeadDim(0);
        const vstart = (config.layerNumHeads(0) + config.layerKVHeads(0)) * config.layerHeadDim(0);
        const truth_v = try allocator.alloc(f32, vrows);
        defer allocator.free(truth_v);
        for (truth_v, 0..) |*v, i| {
            var sum: f64 = 0;
            for (0..config.hidden_size) |j| sum += @as(f64, norm_host[j]) * source_qkv[(vstart + i) * config.hidden_size + j];
            v.* = @floatCast(sum);
        }
        try mimoExpectClose("layer 0 value projection", value_host[0..vrows], truth_v, 5e-4);
        var local_mask = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(local_mask);
        const attn = try component_xfm.mimoAttnWith(&component_ctx, normed, &lw.attn.full, 0, 0, 1, @intCast(token_count), true, &local_mask, .{ .ctx = null });
        defer _ = mlx.mlx_array_free(attn);
        const ours = try qwen4ReadF32(allocator, attn, s);
        defer allocator.free(ours);
        const ref = try qwen4ReadF32(allocator, fx.get("l0_attn_out").?, s);
        defer allocator.free(ref);
        try mimoExpectClose("layer 0 attention", ours, ref, 5e-4);
    }

    for (0..layer_count) |i| {
        var key_buf: [32]u8 = undefined;
        const key = std.fmt.bufPrint(&key_buf, "stream_{d}", .{i + 1}) catch unreachable;
        const ref = try qwen4ReadF32(allocator, fx.get(key) orelse return error.MissingFixtureTensor, s);
        defer allocator.free(ref);
        const ours = try qwen4ReadF32(allocator, capture_out[i], s);
        defer allocator.free(ours);
        try mimoExpectClose(key, ours, ref, 5e-4);
    }
    try mimoExpectClose("full logits", ours_full, ref_full, 5e-3);

    var cache_xfm = try Transformer.init(io, allocator, config, &weights);
    defer cache_xfm.deinit();
    const t_prefill: usize = token_count - 6;
    const pre_shape = [_]c_int{ 1, @intCast(t_prefill) };
    const pre_ids = mlx.mlx_array_new_data(@ptrCast(ids_src), &pre_shape, 2, .int32);
    defer _ = mlx.mlx_array_free(pre_ids);
    var cache_ctx = cache_xfm.defaultCtx();
    const pre_logits = try cache_xfm.forwardWith(&cache_ctx, pre_ids);
    defer _ = mlx.mlx_array_free(pre_logits);
    const pre_rows = try qwen4ReadF32(allocator, pre_logits, s);
    defer allocator.free(pre_rows);
    var ours_cache = try allocator.alloc(f32, ref_cache.len);
    defer allocator.free(ours_cache);
    @memcpy(ours_cache[0 .. t_prefill * vocab], pre_rows);

    for (t_prefill..token_count) |t| {
        const step_shape = [_]c_int{ 1, 1 };
        const step_id = mlx.mlx_array_new_data(@ptrCast(&ids_src[t]), &step_shape, 2, .int32);
        defer _ = mlx.mlx_array_free(step_id);
        var step_ctx = cache_xfm.defaultCtx();
        const step_logits = try cache_xfm.forwardWith(&step_ctx, step_id);
        defer _ = mlx.mlx_array_free(step_logits);
        const step_rows = try qwen4ReadF32(allocator, step_logits, s);
        defer allocator.free(step_rows);
        @memcpy(ours_cache[t * vocab ..][0..vocab], step_rows);
    }
    try mimoExpectClose("cached logits", ours_cache, ref_cache, 5e-3);
}

test "mimo v2 verify rows equal serial decode ticks bit for bit across the sliding window (MIMO_V2_MODEL)" {
    const model_dir = std.c.getenv("MIMO_V2_MODEL") orelse return error.SkipZigTest;
    try mlx.requireGpu();
    const a = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();
    var config = try model_mod.parseConfig(io, a, std.mem.span(model_dir));
    defer if (config.ngram_table_path) |p| a.free(p);
    // The served loader, so a real pack (MIMO_V2_MODEL=<pack>, a heavy GPU job) runs the same check.
    var weights = try model_mod.loadWeightsForConfig(io, a, std.mem.span(model_dir), &config, false);
    defer weights.deinit();
    try stackMimoFixtureExperts(&weights, config, s);
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Transformer.init(io, a, config, &weights);
    defer xfm.deinit();
    // Past the packed floor the global rows read the kv8 cache through matmul2d; a repeated
    // token makes the verify rows share experts, which takes the grouped expert GEMVs.
    const long_prefix: usize = @as(usize, @intCast(QKV_MPP_DECODE_MIN_TK)) + 3;
    var ids: [long_prefix + MIMO_VERIFY_ROWS_MAX]i32 = undefined;
    for ([_]bool{ false, true }) |repeat| {
        for ([_]KVQuantConfig{ KVQuantConfig.dense, KVQuantConfig.affine(8) }) |kv| {
            for ([_]usize{ 20, 126, 136, long_prefix }) |prefix| {
                for (&ids, 0..) |*v, i| v.* = @intCast(2 + (i * 37) % (config.vocab_size - 2));
                if (repeat) @memset(ids[prefix..], ids[prefix]);
                for ([_]usize{ 2, 3, 4, 5, 6, 7, 8 }) |rows| {
                    const ticks = try mimoVerifyOrTicks(a, &xfm, &ids, prefix, rows, false, kv);
                    defer a.free(ticks);
                    const verify = try mimoVerifyOrTicks(a, &xfm, &ids, prefix, rows, true, kv);
                    defer a.free(verify);
                    for (ticks, verify, 0..) |x, y, i| {
                        if (@as(u32, @bitCast(x)) != @as(u32, @bitCast(y))) {
                            std.debug.print("mimo verify row differs: kv={s} prefix={d} rows={d} repeat={} at {d}: {d} vs {d}\n", .{ @tagName(kv.scheme), prefix, rows, repeat, i, x, y });
                            return error.RowNotBitIdentical;
                        }
                    }
                }
            }
        }
    }
}

test "mimo batched decode rows equal each slot's own decode tick bit for bit (MIMO_V2_MODEL)" {
    const model_dir = std.c.getenv("MIMO_V2_MODEL") orelse return error.SkipZigTest;
    try mlx.requireGpu();
    const a = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();
    var config = try model_mod.parseConfig(io, a, std.mem.span(model_dir));
    defer if (config.ngram_table_path) |p| a.free(p);
    var weights = try model_mod.loadWeightsForConfig(io, a, std.mem.span(model_dir), &config, false);
    defer weights.deinit();
    try stackMimoFixtureExperts(&weights, config, s);
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Transformer.init(io, a, config, &weights);
    defer xfm.deinit();
    const window: usize = config.sliding_window;
    const ring_cap: usize = window + @as(usize, @intCast(ModelConfig.SWA_RING_SLACK));
    // Slots whose ticks cross the window edge, a ring compaction, nothing, and the packed global read.
    const prefixes = [_]usize{ window - 2, ring_cap - 2, 20, @as(usize, @intCast(QKV_MPP_DECODE_MIN_TK)) + 3 };
    const steps: usize = 4;
    var ids: [@as(usize, @intCast(QKV_MPP_DECODE_MIN_TK)) + 3 + steps]i32 = undefined;
    const vocab: usize = @intCast(config.vocab_size);
    for ([_]KVQuantConfig{ KVQuantConfig.dense, KVQuantConfig.affine(8) }) |kv| {
        for ([_]usize{ 2, 3, 4 }) |n| {
            var solo: [4]MimoBatchTestSlot = undefined;
            var batch: [4]MimoBatchTestSlot = undefined;
            var made: usize = 0;
            defer for (solo[0..made], batch[0..made]) |*x, *y| {
                x.cache.deinit();
                y.cache.deinit();
            };
            for (0..n) |i| {
                for (&ids, 0..) |*v, j| v.* = @intCast(2 + (j * 37 + i * 101) % (config.vocab_size - 2));
                try mimoBatchTestPrefill(&xfm, &solo[i], ids[0..prefixes[i]], kv);
                errdefer solo[i].cache.deinit();
                try mimoBatchTestPrefill(&xfm, &batch[i], ids[0..prefixes[i]], kv);
                made += 1;
            }
            for (0..steps) |step| {
                var tokens: [4]u32 = undefined;
                var offsets: [4]u32 = undefined;
                var batch_ctx: [4]ForwardCtx = undefined;
                var batch_ptrs: [4]*ForwardCtx = undefined;
                for (0..n) |i| {
                    tokens[i] = @intCast(2 + (step * 53 + i * 7) % (vocab - 2));
                    offsets[i] = @intCast(batch[i].offset);
                    // Odd rows read the full head, even rows the argmax shortlist.
                    batch_ctx[i] = batch[i].ctx(i % 2 == 0);
                    batch_ptrs[i] = &batch_ctx[i];
                }
                var hiddens: ?[]mlx.mlx_array = null;
                const rows = try xfm.forwardMimoBatchedDecode(tokens[0..n], batch_ptrs[0..n], offsets[0..n], &hiddens);
                defer {
                    for (rows) |r| _ = mlx.mlx_array_free(r);
                    a.free(rows);
                    for (hiddens.?) |h| _ = mlx.mlx_array_free(h);
                    a.free(hiddens.?);
                }
                for (0..n) |i| {
                    batch[i].offset += 1;
                    var c = solo[i].ctx(i % 2 == 0);
                    const tok: i32 = @intCast(tokens[i]);
                    const one = mlx.mlx_array_new_data(&tok, &[_]c_int{ 1, 1 }, 2, .int32);
                    defer _ = mlx.mlx_array_free(one);
                    // The solo tick's capture is what an MTP head resuming after plain ticks reads.
                    var want_hidden = mlx.mlx_array_new();
                    defer _ = mlx.mlx_array_free(want_hidden);
                    const want = try xfm.forwardWithCapture(&c, one, &want_hidden);
                    defer _ = mlx.mlx_array_free(want);
                    const hw = try qwen4ReadF32(a, want_hidden, s);
                    defer a.free(hw);
                    const hg = try qwen4ReadF32(a, hiddens.?[i], s);
                    defer a.free(hg);
                    try testing.expectEqual(hw.len, hg.len);
                    for (hw, hg) |x, y| try testing.expectEqual(@as(u32, @bitCast(x)), @as(u32, @bitCast(y)));
                    const want_host = try qwen4ReadF32(a, want, s);
                    defer a.free(want_host);
                    const got_host = try qwen4ReadF32(a, rows[i], s);
                    defer a.free(got_host);
                    try testing.expectEqual(want_host.len, got_host.len);
                    for (want_host, got_host, 0..) |x, y, k| {
                        if (@as(u32, @bitCast(x)) != @as(u32, @bitCast(y))) {
                            std.debug.print("mimo batched row differs: kv={s} n={d} step={d} slot={d} at {d}: {d} vs {d}\n", .{ @tagName(kv.scheme), n, step, i, k, x, y });
                            return error.RowNotBitIdentical;
                        }
                    }
                }
            }
            for (0..n) |i| {
                try testing.expectEqual(solo[i].offset, batch[i].offset);
                try testing.expectEqual(solo[i].cache.step, batch[i].cache.step);
                var x = try solo[i].cache.snapshotRetained(s);
                defer x.deinit();
                var y = try batch[i].cache.snapshotRetained(s);
                defer y.deinit();
                try testing.expect(try snapshotsBitEqual(&x, &y, s));
            }
        }
    }
}

test "mimo spec warm-up runs every verify row count on its own slot (MIMO_V2_MODEL)" {
    const model_dir = std.c.getenv("MIMO_V2_MODEL") orelse return error.SkipZigTest;
    try mlx.requireGpu();
    const a = testing.allocator;
    const s = mlx.gpuStream();
    const io = std.Io.Threaded.global_single_threaded.io();
    var config = try model_mod.parseConfig(io, a, std.mem.span(model_dir));
    defer if (config.ngram_table_path) |p| a.free(p);
    var weights = try model_mod.loadWeightsForConfig(io, a, std.mem.span(model_dir), &config, false);
    defer weights.deinit();
    try stackMimoFixtureExperts(&weights, config, s);
    model_mod.resolveWeightPrefix(&config, &weights);
    var xfm = try Transformer.init(io, a, config, &weights);
    defer xfm.deinit();
    var want: Transformer.MimoWarmRows = 0;
    var rows: usize = 2;
    while (rows <= MIMO_VERIFY_ROWS_MAX) : (rows += 1) want |= @as(Transformer.MimoWarmRows, 1) << @intCast(rows);
    for ([_]KVQuantConfig{ KVQuantConfig.dense, KVQuantConfig.affine(8) }) |kv| {
        try testing.expectEqual(want, try xfm.warmupMimoVerify(kv));
        for (xfm.cache.entries) |e| try testing.expect(!e.initialized);
    }
}

test "mimo v2 mixed-precision streamed and resident forwards agree" {
    const raw = std.c.getenv("MIMO_STREAM_FIXTURE") orelse return error.SkipZigTest;
    try mlx.requireGpu();
    const a = testing.allocator;
    const io = testing.io;
    const path = std.mem.span(raw);
    var config = try model_mod.parseConfig(io, a, path);
    defer config.deinit(a);
    var resident_weights = try model_mod.loadWeights(io, a, path);
    defer resident_weights.deinit();
    var streamed_weights = try model_mod.loadWeightsStreaming(io, a, path, .mxfp4_split, false, false);
    defer streamed_weights.deinit();
    const per_expert = try expert_stream_mod.mxfp4ExpertBytes(config.hidden_size, config.moe_intermediate_size);
    var stream_config = config;
    stream_config.expert_streaming = true;
    stream_config.expert_source_dir = try a.dupe(u8, path);
    defer a.free(stream_config.expert_source_dir.?);
    stream_config.expert_cache_bytes = 8 * config.expertLayerCount() * per_expert;
    for ([_]KVQuantConfig{ KVQuantConfig.dense, KVQuantConfig.affine(8) }) |kv| {
        var resident = try Transformer.init(io, a, config, &resident_weights);
        defer resident.deinit();
        var streamed = try Transformer.init(io, a, stream_config, &streamed_weights);
        defer streamed.deinit();
        try resident.cache.reinit(config.num_hidden_layers, kv);
        try streamed.cache.reinit(config.num_hidden_layers, kv);
        var tokens: [166]i32 = undefined;
        for (&tokens, 0..) |*token, i| token.* = @intCast(i % config.vocab_size);
        var offset: usize = 0;
        for ([_]usize{ 160, 1, 1, 1, 1, 1, 1 }) |width| {
            const shape = [_]c_int{ 1, @intCast(width) };
            const input = mlx.mlx_array_new_data(@ptrCast(tokens[offset..].ptr), &shape, 2, .int32);
            defer _ = mlx.mlx_array_free(input);
            var rctx = resident.defaultCtx();
            const rlogits = try resident.forwardWith(&rctx, input);
            defer _ = mlx.mlx_array_free(rlogits);
            const expected = try qwen4ReadF32(a, rlogits, resident.s);
            defer a.free(expected);
            var sctx = streamed.defaultCtx();
            const slogits = try streamed.forwardWith(&sctx, input);
            defer _ = mlx.mlx_array_free(slogits);
            const actual = try qwen4ReadF32(a, slogits, streamed.s);
            defer a.free(actual);
            try mimoExpectClose("MXFP4 streamed versus resident", actual, expected, 0);
            offset += width;
        }
    }
}
