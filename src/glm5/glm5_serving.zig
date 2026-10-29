const trf = @import("../transformer.zig");
const std = @import("std");
const glm5_mod = @import("glm5_forward.zig");
const expert_stream_mod = @import("../expert_stream.zig");
const sushi_exl3 = @import("sushi_exl3");
const expert_exl3 = sushi_exl3.format;
const mlx = @import("../mlx.zig");
const kv_quant = @import("../kv_quant.zig");
const KVQuantConfig = kv_quant.KVQuantConfig;
const model_mod = @import("../model.zig");
const ModelConfig = model_mod.ModelConfig;
const Weights = model_mod.Weights;
const testing = std.testing;
const CaptureLayers = trf.CaptureLayers;
const ForwardCtx = trf.ForwardCtx;
const KVCache = trf.KVCache;
const Transformer = trf.Transformer;
const initModuleShell = trf.initModuleShell;

pub fn initGlm5(allocator: std.mem.Allocator, config: ModelConfig, weights: *const Weights, s: mlx.mlx_stream) !Transformer {
    const engine = if (config.expert_streaming) try Transformer.newExpertEngine(allocator, &config, s) else null;
    errdefer if (engine) |e| {
        e.deinit();
        allocator.destroy(e);
    };
    const mdl = try allocator.create(glm5_mod.Model);
    errdefer allocator.destroy(mdl);
    mdl.* = try glm5_mod.Model.loadStreamed(allocator, config, weights, s, if (engine) |e| @import("glm5_stream.zig").Stream.serving(e, &config) else null);
    errdefer mdl.deinit();
    var request = try glm5_mod.Request.initServing(allocator, config.num_hidden_layers);
    errdefer request.deinit();
    var shell = try initModuleShell(allocator, config, s);
    shell.glm5 = mdl;
    shell.glm5_request = request;
    shell.expert_stream = engine;
    return shell;
}

pub fn forwardGlm5WithImpl(self: *Transformer, ctx: *ForwardCtx, ids: mlx.mlx_array, mdl: *glm5_mod.Model) !mlx.mlx_array {
    const request = ctx.glm5_request orelse return error.GlmRequestMissing;
    if (ctx.cache.step == 0) {
        request.reset();
        try request.setLatentBits(ctx.cache.config.glmLatentBits() orelse return error.GlmKvQuantUnsupported);
    }
    if (request.offset != ctx.cache.step) return error.GlmCachePositionMismatch;
    mdl.s = self.s;
    mdl.suppress_mask = self.suppress_mask;
    var capture: glm5_mod.Capture = undefined;
    if (ctx.capture_layers) |cl| {
        // Generic capture buffers start empty; native GLM requires live
        // destination handles before replacing them with captured arrays.
        for (cl.out) |*out| if (out.ctx == null) {
            out.* = mlx.mlx_array_new_float(0);
        };
        capture = .{ .ids = cl.ids, .out = cl.out };
        request.capture = &capture;
    }
    defer request.capture = null;
    var embedded: ?mlx.mlx_array = null;
    defer if (embedded) |h| {
        _ = mlx.mlx_array_free(h);
    };
    if (ctx.vision_embeddings) |media| {
        const h = try mdl.rawEmbedding(ids);
        embedded = self.spliceVisionEmbeddings(h, ids, media, self.config.image_token_id, self.config.video_token_id, ctx.vision_splice_offset) catch |err| {
            _ = mlx.mlx_array_free(h);
            return err;
        };
    }
    if (self.expert_stream) |engine| engine.beginForward();
    const logits = try mdl.forwardLastWithEmbedding(request, ids, true, embedded);
    if (self.expert_stream) |engine| engine.finishForward(@intCast(mlx.getShape(ids)[1]));
    ctx.cache.step = request.offset;
    return logits;
}

fn expectSameBits(expected: mlx.mlx_array, actual: mlx.mlx_array) ![]const u8 {
    try mlx.check(mlx.mlx_array_eval(expected));
    try mlx.check(mlx.mlx_array_eval(actual));
    try testing.expectEqual(mlx.mlx_array_dtype(expected), mlx.mlx_array_dtype(actual));
    try testing.expectEqualSlices(c_int, mlx.getShape(expected), mlx.getShape(actual));
    const bytes = mlx.mlx_array_size(expected) * mlx.mlx_array_itemsize(expected);
    const want = mlx.mlx_array_data_uint8(expected) orelse return error.UnreadableLogits;
    const got = mlx.mlx_array_data_uint8(actual) orelse return error.UnreadableLogits;
    try testing.expectEqualSlices(u8, want[0..bytes], got[0..bytes]);
    return want[0..bytes];
}

/// The fixture's routed bank as a one-shard EXL3 pack index the expert store opens.
fn writeGlmExl3Bank(dir: std.Io.Dir, path: []const u8, weights: *const Weights) !void {
    const a = testing.allocator;
    const map = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(map);
    var index: std.ArrayList(u8) = .empty;
    defer index.deinit(a);
    try index.appendSlice(a, "{\"weight_map\":{");
    for ([_][]const u8{ "gate_proj", "up_proj", "down_proj" }, 0..) |proj, pi| {
        for ([_][]const u8{ "trellis", "suh", "svh" }, 0..) |part, ci| {
            const key = try std.fmt.allocPrintSentinel(a, "{s}.{s}.{s}", .{ glm5_mod.ROUTED_BANK_PREFIX, proj, part }, 0);
            defer a.free(key);
            try mlx.check(mlx.mlx_map_string_to_array_insert(map, key.ptr, weights.get(key) orelse return error.MissingFixtureTensor));
            const entry = try std.fmt.allocPrint(a, "{s}\"{s}\":\"experts.safetensors\"", .{ if (pi + ci == 0) "" else ",", key });
            defer a.free(entry);
            try index.appendSlice(a, entry);
        }
    }
    try index.appendSlice(a, "}}");
    const meta = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta);
    const file = try std.fmt.allocPrintSentinel(a, "{s}/experts.safetensors", .{path}, 0);
    defer a.free(file);
    try mlx.check(mlx.mlx_save_safetensors(file.ptr, map, meta));
    try dir.writeFile(testing.io, .{ .sub_path = "model.safetensors.index.json", .data = index.items });
}

fn glmStreamedExl3MatchesResident(dec: expert_exl3.Decode) !void {
    try mlx.requireGpu();
    const a = testing.allocator;
    var weights = Weights.init(a);
    defer weights.deinit();
    var cfg = try glm5_mod.routedFixture(&weights, 4);
    cfg.expert_quant_codebook = dec.codebook;
    cfg.expert_quant_table = dec.table;
    var trunk = Weights.init(a);
    defer trunk.deinit();
    _ = try glm5_mod.routedFixture(&trunk, 4);
    try trunk.dropPrefix(glm5_mod.ROUTED_BANK_PREFIX);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = path_buf[0..try tmp.dir.realPath(testing.io, &path_buf)];
    try writeGlmExl3Bank(tmp.dir, path, &weights);
    var streamed_cfg = cfg;
    streamed_cfg.expert_streaming = true;
    streamed_cfg.expert_layout = .exl3_k4;
    streamed_cfg.expert_source_dir = path;
    streamed_cfg.expert_cache_bytes = 2 * try expert_stream_mod.expertBytesFor(a, path, cfg.expertGeometry(), .exl3_k4);
    var resident = try Transformer.init(testing.io, a, cfg, &weights);
    defer resident.deinit();
    var streamed = try Transformer.init(testing.io, a, streamed_cfg, &trunk);
    defer streamed.deinit();
    const engine = streamed.expert_stream orelse return error.MissingExpertStream;
    try engine.warmCache();
    var union_seen = false;
    for ([_]KVQuantConfig{ .dense, KVQuantConfig.affine(8), KVQuantConfig.affine(6) }) |kv| {
        for ([_]*Transformer{ &resident, &streamed }) |xfm| {
            try xfm.cache.reinit(cfg.num_hidden_layers, kv);
            try xfm.resetCache();
        }
        var first_bytes: ?[8]u8 = null;
        var logits_moved = false;
        for ([_][]const u32{ &.{ 0, 1, 2, 3, 1 }, &.{2}, &.{3}, &.{0}, &.{1}, &.{ 2, 0 }, &.{3} }) |chunk| {
            const ids = mlx.mlx_array_new_data(chunk.ptr, &[_]c_int{ 1, @intCast(chunk.len) }, 2, .uint32);
            defer _ = mlx.mlx_array_free(ids);
            const expected = try resident.forward(ids);
            defer _ = mlx.mlx_array_free(expected);
            const actual = try streamed.forward(ids);
            defer _ = mlx.mlx_array_free(actual);
            const want = try expectSameBits(expected, actual);
            if (first_bytes) |first| logits_moved = logits_moved or !std.mem.eql(u8, &first, want[0..8]) else first_bytes = want[0..8].*;
            union_seen = union_seen or engine.breakdown().union_members > 0;
        }
        try testing.expect(logits_moved);
        try testing.expectEqual(@as(u8, if (kv.isQuant()) kv.bits else 0), streamed.glm5_request.?.latent_bits);
    }
    try testing.expect(union_seen);
    try testing.expect(engine.fill_experts_total > 4);
}

test "GLM serving dispatch matches native forward across prefill decode and reset" {
    try mlx.requireGpu();
    const a = testing.allocator;
    var weights = Weights.init(a);
    defer weights.deinit();
    const cfg = try glm5_mod.completeFixture(&weights);
    var xfm = try Transformer.init(testing.io, a, cfg, &weights);
    defer xfm.deinit();
    try testing.expect(!xfm.ownsModuleDecodeState());
    var reference = try glm5_mod.Model.load(a, cfg, &weights, xfm.s);
    defer reference.deinit();
    var request = try glm5_mod.Request.init(a, cfg.num_hidden_layers);
    defer request.deinit();
    request.dense_prefill = true;
    request.prefill_async = true;
    for ([_]usize{ 3, 1, 2, 1 }) |n| {
        const ids = mlx.mlx_array_new_data(&[_]u32{ 0, 1, 2 }, &[_]c_int{ 1, @intCast(n) }, 2, .uint32);
        defer _ = mlx.mlx_array_free(ids);
        const actual = try xfm.forward(ids);
        defer _ = mlx.mlx_array_free(actual);
        const expected = try reference.forwardLast(&request, ids, true);
        defer _ = mlx.mlx_array_free(expected);
        var equal = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(equal);
        try mlx.check(mlx.mlx_array_equal(&equal, actual, expected, true, xfm.s));
            var same = false;
        try mlx.check(mlx.mlx_array_item_bool(&same, equal));
            try testing.expect(same);
        try testing.expectEqual(request.offset, xfm.cache.step);
    }
    try xfm.resetCache();
    try testing.expectEqual(@as(usize, 0), xfm.glm5_request.?.offset);
    request.reset();
    const ids = mlx.mlx_array_new_data(&[_]u32{0}, &[_]c_int{ 1, 1 }, 2, .uint32);
    defer _ = mlx.mlx_array_free(ids);
    const actual = try xfm.forward(ids);
    defer _ = mlx.mlx_array_free(actual);
    try testing.expectEqual(@as(usize, 1), xfm.glm5_request.?.offset);
    var ctx = xfm.defaultCtx();
    ctx.cache.step = 2;
    try testing.expectError(error.GlmCachePositionMismatch, xfm.forwardWith(&ctx, ids));
    try xfm.resetCache();
    var out = [_]mlx.mlx_array{mlx.mlx_array_new()};
    defer _ = mlx.mlx_array_free(out[0]);
    var captures = CaptureLayers{ .ids = &.{0}, .out = &out };
    ctx = xfm.defaultCtx();
    ctx.capture_layers = &captures;
    const captured = try xfm.forwardWith(&ctx, ids);
    defer _ = mlx.mlx_array_free(captured);
    try testing.expectEqualSlices(c_int, &.{ 1, 1, 128 }, mlx.getShape(out[0]));
}

test "GLM forward contexts carry their own request: interleaved streams equal each stream alone" {
    try mlx.requireGpu();
    const a = testing.allocator;
    var weights = Weights.init(a);
    defer weights.deinit();
    const cfg = try glm5_mod.completeFixture(&weights);
    var xfm = try Transformer.init(testing.io, a, cfg, &weights);
    defer xfm.deinit();
    try testing.expect(!xfm.ownsModuleDecodeState());
    try testing.expect(xfm.moduleSpecWiring());
    var reference = try glm5_mod.Model.load(a, cfg, &weights, xfm.s);
    defer reference.deinit();
    var caches: [2]KVCache = undefined;
    var offsets: [2]usize = .{ 0, 0 };
    var served: [2]glm5_mod.Request = undefined;
    var solo: [2]glm5_mod.Request = undefined;
    var ctxs: [2]ForwardCtx = undefined;
    for (0..2) |i| {
        caches[i] = try KVCache.init(a, cfg.num_hidden_layers);
        served[i] = try glm5_mod.Request.initServing(a, cfg.num_hidden_layers);
        solo[i] = try glm5_mod.Request.initServing(a, cfg.num_hidden_layers);
        ctxs[i] = .{ .cache = &caches[i], .moe_seq_offset = &offsets[i], .ssm_entries = null, .capture_hidden = null, .vision_embeddings = null, .glm5_request = &served[i] };
    }
    defer for (0..2) |i| {
        caches[i].deinit();
        served[i].deinit();
        solo[i].deinit();
    };
    const Step = struct { slot: usize, ids: []const u32 };
    const steps = [_]Step{ .{ .slot = 0, .ids = &.{ 0, 1, 2 } }, .{ .slot = 1, .ids = &.{ 2, 1 } }, .{ .slot = 0, .ids = &.{1} }, .{ .slot = 1, .ids = &.{0} }, .{ .slot = 1, .ids = &.{2} }, .{ .slot = 0, .ids = &.{2} } };
    for (steps) |step| {
        const ids = mlx.mlx_array_new_data(step.ids.ptr, &[_]c_int{ 1, @intCast(step.ids.len) }, 2, .uint32);
        defer _ = mlx.mlx_array_free(ids);
        const actual = try xfm.forwardWith(&ctxs[step.slot], ids);
        defer _ = mlx.mlx_array_free(actual);
        const expected = try reference.forwardLast(&solo[step.slot], ids, true);
        defer _ = mlx.mlx_array_free(expected);
        var equal = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(equal);
        try mlx.check(mlx.mlx_array_equal(&equal, actual, expected, true, xfm.s));
        var same = false;
        try mlx.check(mlx.mlx_array_item_bool(&same, equal));
        try testing.expect(same);
        try testing.expectEqual(solo[step.slot].offset, caches[step.slot].step);
    }
    // The shell's own request belongs to the default context only.
    try testing.expectEqual(@as(usize, 0), xfm.glm5_request.?.offset);
    var orphan = ctxs[0];
    orphan.glm5_request = null;
    const ids = mlx.mlx_array_new_data(&[_]u32{0}, &[_]c_int{ 1, 1 }, 2, .uint32);
    defer _ = mlx.mlx_array_free(ids);
    try testing.expectError(error.GlmRequestMissing, xfm.forwardWith(&orphan, ids));
}

test "a GLM prefill interleaved with another stream's decode builds the solo prefill's cache bits" {
    try mlx.requireGpu();
    const a = testing.allocator;
    var weights = Weights.init(a);
    defer weights.deinit();
    const cfg = try glm5_mod.completeFixture(&weights);
    var xfm = try Transformer.init(testing.io, a, cfg, &weights);
    defer xfm.deinit();
    var caches: [3]KVCache = undefined;
    var offsets: [3]usize = .{ 0, 0, 0 };
    var requests: [3]glm5_mod.Request = undefined;
    var ctxs: [3]ForwardCtx = undefined;
    for (0..3) |i| {
        caches[i] = try KVCache.init(a, cfg.num_hidden_layers);
        requests[i] = try glm5_mod.Request.initServing(a, cfg.num_hidden_layers);
        ctxs[i] = .{ .cache = &caches[i], .moe_seq_offset = &offsets[i], .ssm_entries = null, .capture_hidden = null, .vision_embeddings = null, .glm5_request = &requests[i] };
    }
    defer for (0..3) |i| {
        caches[i].deinit();
        requests[i].deinit();
    };
    // ctxs[0] prefills alone; ctxs[1] prefills the same chunks around ctxs[2]'s decode steps.
    const chunks = [_][]const u32{ &.{ 0, 1, 2 }, &.{ 3, 1 }, &.{2} };
    const other = [_][]const u32{ &.{ 1, 3 }, &.{0}, &.{2} };
    for (chunks, other) |chunk, step| {
        for ([_]usize{ 0, 2, 1 }, [_][]const u32{ chunk, step, chunk }) |slot, ids_slice| {
            const ids = mlx.mlx_array_new_data(ids_slice.ptr, &[_]c_int{ 1, @intCast(ids_slice.len) }, 2, .uint32);
            defer _ = mlx.mlx_array_free(ids);
            const out = try xfm.forwardWith(&ctxs[slot], ids);
            _ = mlx.mlx_array_free(out);
        }
    }
    for (requests[0].layers, requests[1].layers) |solo, mixed| {
        for ([_]mlx.mlx_array{ solo.recurrent.conv_state, solo.recurrent.ssm_state } ++ solo.attention.arrays(), [_]mlx.mlx_array{ mixed.recurrent.conv_state, mixed.recurrent.ssm_state } ++ mixed.attention.arrays()) |x, y| {
            if (x.ctx == null or y.ctx == null) {
                try testing.expect(x.ctx == null and y.ctx == null);
                continue;
            }
            var equal = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(equal);
            try mlx.check(mlx.mlx_array_equal(&equal, x, y, true, xfm.s));
            var same = false;
            try mlx.check(mlx.mlx_array_item_bool(&same, equal));
            try testing.expect(same);
        }
    }
}

test "a streamed GLM load keeps its one teacher stream exclusive" {
    var t: Transformer = undefined;
    t.qwen4 = null;
    t.dsv41 = null;
    t.expert_stream = null;
    var glm: glm5_mod.Model = undefined;
    glm.expert_stream = null;
    t.glm5 = &glm;
    try testing.expect(!t.ownsModuleDecodeState());
    try testing.expect(t.moduleSpecWiring());
    glm.expert_stream = @ptrFromInt(@alignOf(@import("glm5_stream.zig").Stream));
    try testing.expect(t.ownsModuleDecodeState());
}

test "GLM serving stores the latent its slot cache names and refuses kv4 by name" {
    try mlx.requireGpu();
    const a = testing.allocator;
    var weights = Weights.init(a);
    defer weights.deinit();
    const cfg = try glm5_mod.completeFixture(&weights);
    var xfm = try Transformer.init(testing.io, a, cfg, &weights);
    defer xfm.deinit();
    var reference = try glm5_mod.Model.load(a, cfg, &weights, xfm.s);
    defer reference.deinit();
    for ([_]u8{ 8, 6 }) |bits| {
        var request = try glm5_mod.Request.init(a, cfg.num_hidden_layers);
        defer request.deinit();
        request.dense_prefill = true;
        request.prefill_async = true;
        try request.setLatentBits(bits);
        try xfm.cache.reinit(cfg.num_hidden_layers, KVQuantConfig.affine(bits));
        try xfm.resetCache();
        for ([_]usize{ 3, 1 }) |n| {
            const ids = mlx.mlx_array_new_data(&[_]u32{ 0, 1, 2 }, &[_]c_int{ 1, @intCast(n) }, 2, .uint32);
            defer _ = mlx.mlx_array_free(ids);
            const actual = try xfm.forward(ids);
            defer _ = mlx.mlx_array_free(actual);
            const expected = try reference.forwardLast(&request, ids, true);
            defer _ = mlx.mlx_array_free(expected);
            var equal = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(equal);
            try mlx.check(mlx.mlx_array_equal(&equal, actual, expected, true, xfm.s));
            var same = false;
            try mlx.check(mlx.mlx_array_item_bool(&same, equal));
            try testing.expect(same);
        }
        const stored = &xfm.glm5_request.?;
        try testing.expectEqual(bits, stored.latent_bits);
        try testing.expectEqual(mlx.mlx_dtype.uint32, mlx.mlx_array_dtype(stored.layers[3].attention.latent));
    }
    const served = &xfm.glm5_request.?;
    const ids = mlx.mlx_array_new_data(&[_]u32{0}, &[_]c_int{ 1, 1 }, 2, .uint32);
    defer _ = mlx.mlx_array_free(ids);
    try xfm.cache.reinit(cfg.num_hidden_layers, KVQuantConfig.affine(4));
    try xfm.resetCache();
    try testing.expectError(error.GlmKvQuantUnsupported, xfm.forward(ids));
    try xfm.cache.reinit(cfg.num_hidden_layers, KVQuantConfig.dense);
    try xfm.resetCache();
    const plain = try xfm.forward(ids);
    defer _ = mlx.mlx_array_free(plain);
    try testing.expectEqual(@as(u8, 0), served.latent_bits);
    try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(served.layers[3].attention.latent));
}

test "GLM serving streams EXL3 experts through eviction and the union, bit-identical to resident at BF16 and kv8 latents" {
    try glmStreamedExl3MatchesResident(expert_exl3.Decode.mcg);
}

test "GLM serving streams std1 EXL3 experts bit-identical to resident" {
    try glmStreamedExl3MatchesResident(expert_exl3.Decode.tabled(.std1, .w15));
}

test "GLM serving streams BF16 source experts through the engine the native capture uses" {
    try mlx.requireGpu();
    const a = testing.allocator;
    var trunk = Weights.init(a);
    defer trunk.deinit();
    const cfg = try glm5_mod.routedFixture(&trunk, 4);
    try trunk.dropPrefix(glm5_mod.ROUTED_BANK_PREFIX);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try @import("glm_stream_fixture.zig").writeSized(a, tmp.dir, .none, 128, 128);
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = path_buf[0..try tmp.dir.realPath(testing.io, &path_buf)];
    var streamed_cfg = cfg;
    streamed_cfg.expert_streaming = true;
    streamed_cfg.expert_layout = .bf16_individual;
    streamed_cfg.expert_source_dir = path;
    streamed_cfg.expert_cache_bytes = 2 * try expert_stream_mod.expertBytesFor(a, path, cfg.expertGeometry(), .bf16_individual);
    var streamed = try Transformer.init(testing.io, a, streamed_cfg, &trunk);
    defer streamed.deinit();
    var engine = try expert_stream_mod.Engine.initWithOptions(a, path, cfg.expertGeometry(), streamed_cfg.expert_cache_bytes, streamed.s, .{ .layout = .bf16_individual });
    defer engine.deinit();
    var native = try glm5_mod.Model.loadStreamed(a, cfg, &trunk, streamed.s, .{ .engine = &engine, .max_tokens = 16, .max_chunk = 16 });
    defer native.deinit();
    var request = try glm5_mod.Request.init(a, cfg.num_hidden_layers);
    defer request.deinit();
    request.dense_prefill = true;
    for ([_][]const u32{ &.{ 0, 1, 2, 3, 1 }, &.{2}, &.{3}, &.{0} }) |chunk| {
        const ids = mlx.mlx_array_new_data(chunk.ptr, &[_]c_int{ 1, @intCast(chunk.len) }, 2, .uint32);
        defer _ = mlx.mlx_array_free(ids);
        const expected = try native.forwardLast(&request, ids, true);
        defer _ = mlx.mlx_array_free(expected);
        const actual = try streamed.forward(ids);
        defer _ = mlx.mlx_array_free(actual);
        _ = try expectSameBits(expected, actual);
    }
    try testing.expect(streamed.expert_stream.?.fill_bytes_total > 0);
}

test "GLM serving streams FP8 source experts through eviction and the union, the bits of an all-resident cache" {
    try mlx.requireGpu();
    const a = testing.allocator;
    var trunk = Weights.init(a);
    defer trunk.deinit();
    const cfg = try glm5_mod.routedFixture(&trunk, 4);
    try trunk.dropPrefix(glm5_mod.ROUTED_BANK_PREFIX);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try @import("glm_stream_fixture.zig").writeStorage(a, tmp.dir, .none, 128, 128, .fp8);
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path = path_buf[0..try tmp.dir.realPath(testing.io, &path_buf)];
    const per_expert = try expert_stream_mod.expertBytesFor(a, path, cfg.expertGeometry(), .fp8_individual);
    var streamed: [2]Transformer = undefined;
    for (&streamed, [_]u64{ 2, 4 }) |*xfm, slots| {
        var c = cfg;
        c.expert_streaming = true;
        c.expert_layout = .fp8_individual;
        c.expert_source_dir = path;
        c.expert_cache_bytes = slots * per_expert;
        xfm.* = try Transformer.init(testing.io, a, c, &trunk);
    }
    defer for (&streamed) |*xfm| xfm.deinit();
    var union_seen = false;
    var first_bytes: ?[8]u8 = null;
    var logits_moved = false;
    for ([_][]const u32{ &.{ 0, 1, 2, 3, 1 }, &.{2}, &.{3}, &.{0}, &.{1}, &.{ 2, 0 }, &.{3} }) |chunk| {
        const ids = mlx.mlx_array_new_data(chunk.ptr, &[_]c_int{ 1, @intCast(chunk.len) }, 2, .uint32);
        defer _ = mlx.mlx_array_free(ids);
        const expected = try streamed[1].forward(ids);
        defer _ = mlx.mlx_array_free(expected);
        const actual = try streamed[0].forward(ids);
        defer _ = mlx.mlx_array_free(actual);
        const want = try expectSameBits(expected, actual);
        if (first_bytes) |first| logits_moved = logits_moved or !std.mem.eql(u8, &first, want[0..8]) else first_bytes = want[0..8].*;
        union_seen = union_seen or streamed[0].expert_stream.?.breakdown().union_members > 0;
    }
    try testing.expect(logits_moved);
    try testing.expect(union_seen);
    try testing.expect(streamed[0].expert_stream.?.fill_experts_total > streamed[1].expert_stream.?.fill_experts_total);
}

test "GLM vision native serving splices media before HC expansion and preserves layer capture" {
    try mlx.requireGpu();
    const a = testing.allocator;
    var weights = Weights.init(a);
    defer weights.deinit();
    var cfg = try glm5_mod.completeFixture(&weights);
    cfg.image_token_id = 1;
    cfg.video_token_id = 2;
    var xfm = try Transformer.init(testing.io, a, cfg, &weights);
    defer xfm.deinit();
    var reference = try glm5_mod.Model.load(a, cfg, &weights, xfm.s);
    defer reference.deinit();
    var request = try glm5_mod.Request.init(a, cfg.num_hidden_layers);
    defer request.deinit();
    request.dense_prefill = true;
    request.prefill_async = true;
    var values: [2 * 128]f32 = undefined;
    for (&values, 0..) |*value, i| value.* = @as(f32, @floatFromInt(i % 13)) * 0.017;
    const media = mlx.mlx_array_new_data(&values, &[_]c_int{ 1, 2, 128 }, 3, .float32);
    defer _ = mlx.mlx_array_free(media);
    var capture_out = [_]mlx.mlx_array{mlx.mlx_array_new()};
    defer _ = mlx.mlx_array_free(capture_out[0]);
    const capture_ids = [_]u32{0};
    var capture = CaptureLayers{ .ids = &capture_ids, .out = &capture_out };
    for ([_]u32{ 1, 2 }, 0..) |token, offset| {
        const ids = mlx.mlx_array_new_data(&token, &[_]c_int{ 1, 1 }, 2, .uint32);
        defer _ = mlx.mlx_array_free(ids);
        var ctx = xfm.defaultCtx();
        ctx.vision_embeddings = media;
        ctx.vision_splice_offset = offset;
        ctx.capture_layers = &capture;
        const out = try xfm.forwardWith(&ctx, ids);
        defer _ = mlx.mlx_array_free(out);
        const embedded = try Transformer.spliceVisionRows(try reference.rawEmbedding(ids), ids, media, cfg.image_token_id, cfg.video_token_id, offset, xfm.s);
        defer _ = mlx.mlx_array_free(embedded);
        const want = try reference.forwardLastWithEmbedding(&request, ids, true, embedded);
        defer _ = mlx.mlx_array_free(want);
        var equal = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(equal);
        try mlx.check(mlx.mlx_array_equal(&equal, out, want, true, xfm.s));
        var same = false;
        try mlx.check(mlx.mlx_array_item_bool(&same, equal));
        try testing.expect(same);
        try testing.expectEqualSlices(c_int, &.{ 1, 1, 128 }, mlx.getShape(capture_out[0]));
        try testing.expectEqual(request.offset, ctx.cache.step);
    }
}
