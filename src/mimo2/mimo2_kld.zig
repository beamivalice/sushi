//! MiMo-V2.6-Flash on the offline tools: the names its source checkpoint gives the imatrix, and the tests of the
//! KLD, imatrix and hidden-capture paths over a tiny MiMo pack.
const std = @import("std");
const mlx = @import("../mlx.zig");
const imatrix = @import("../imatrix.zig");
const kld = @import("../kld.zig");
const transformer_mod = @import("../transformer.zig");
const testing = std.testing;

pub const imatrix_naming = imatrix.Naming{ .layer_prefix = "model.layers." };

const TinyMimo = struct {
    const hidden: usize = 128;
    const vocab: usize = 8;
    const experts: usize = 4;
    const layers: usize = 2;
    const qkv_rows: usize = 384; // (8 heads + 8 kv + 8 v) * 16
    const tiles: usize = hidden / 16;
    const trellis_n: usize = 40; // halfwords per 16x16 tile at k 2.5
};

const TinyTensor = struct {
    key: []const u8,
    dtype: []const u8,
    shape: []const u64,
    bytes: []const u8,
};

fn tinyBf16(a: std.mem.Allocator, count: usize, seed: u64) ![]u8 {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const out = try a.alloc(u8, count * 2);
    for (0..count) |i| {
        const v: f32 = r.float(f32) - 0.5;
        const bits: u32 = @bitCast(v);
        std.mem.writeInt(u16, out[i * 2 ..][0..2], @truncate(bits >> 16), .little);
    }
    return out;
}

/// e4m3 codes in [0.5, 0.94] with alternating signs — every byte a finite value,
/// so a pack that reads them raw still forwards instead of dying on a NaN.
fn tinyFp8(a: std.mem.Allocator, count: usize, seed: u64) ![]u8 {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const out = try a.alloc(u8, count);
    for (out) |*b| b.* = (if (r.boolean()) @as(u8, 0xB0) else @as(u8, 0x30)) | r.uintLessThan(u8, 8);
    return out;
}

/// f16 axis scales of magnitude [0.5, 1) with random signs.
fn tinyF16Axis(a: std.mem.Allocator, count: usize, seed: u64) ![]u8 {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const out = try a.alloc(u8, count * 2);
    for (0..count) |i| {
        const bits: u16 = (if (r.boolean()) @as(u16, 0x8000) else 0) | 0x3800 | r.uintLessThan(u16, 0x400);
        std.mem.writeInt(u16, out[i * 2 ..][0..2], bits, .little);
    }
    return out;
}

fn tinyU32(a: std.mem.Allocator, count: usize, seed: u64) ![]u8 {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const out = try a.alloc(u8, count * 4);
    for (0..count) |i| std.mem.writeInt(u32, out[i * 4 ..][0..4], r.int(u32), .little);
    return out;
}

fn tinyF32(a: std.mem.Allocator, values: []const f32) ![]u8 {
    const out = try a.alloc(u8, values.len * 4);
    for (values, 0..) |v, i| std.mem.writeInt(u32, out[i * 4 ..][0..4], @bitCast(v), .little);
    return out;
}

fn writeTinyShard(io: std.Io, a: std.mem.Allocator, dir: std.Io.Dir, tensors: []const TinyTensor) !void {
    var header: std.ArrayList(u8) = .empty;
    defer header.deinit(a);
    try header.append(a, '{');
    var payload: usize = 0;
    for (tensors, 0..) |t, i| {
        if (i != 0) try header.append(a, ',');
        try header.print(a, "\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ t.key, t.dtype });
        for (t.shape, 0..) |dim, j| {
            if (j != 0) try header.append(a, ',');
            try header.print(a, "{d}", .{dim});
        }
        try header.print(a, "],\"data_offsets\":[{d},{d}]}}", .{ payload, payload + t.bytes.len });
        payload += t.bytes.len;
    }
    try header.append(a, '}');

    const file_bytes = try a.alloc(u8, 8 + header.items.len + payload);
    defer a.free(file_bytes);
    std.mem.writeInt(u64, file_bytes[0..8], header.items.len, .little);
    @memcpy(file_bytes[8..][0..header.items.len], header.items);
    var at: usize = 8 + header.items.len;
    for (tensors) |t| {
        @memcpy(file_bytes[at..][0..t.bytes.len], t.bytes);
        at += t.bytes.len;
    }
    try dir.writeFile(io, .{ .sub_path = "model-00001.safetensors", .data = file_bytes });

    var index: std.ArrayList(u8) = .empty;
    defer index.deinit(a);
    try index.appendSlice(a, "{\"weight_map\":{");
    for (tensors, 0..) |t, i| {
        if (i != 0) try index.append(a, ',');
        try index.print(a, "\"{s}\":\"model-00001.safetensors\"", .{t.key});
    }
    try index.appendSlice(a, "}}");
    try dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index.items });
}

/// A two-layer mimo_v2 pack shaped like a converted MiMo: the FP8 source trunk
/// (layer 0 dense) beside resident EXL3 routed banks (layer 1). `expert_seed`
/// is the only thing that differs between two arms.
fn writeTinyMimoResidentPack(io: std.Io, a: std.mem.Allocator, dir: std.Io.Dir, expert_seed: u64) !void {
    return writeTinyMimoPackWith(io, a, dir, expert_seed, &.{});
}

/// `stored_affine` names `.weight` tensors the pack stores 8-bit g64 affine
/// (MLX's own packer over the same bf16), the way a converted pack carries them.
fn writeTinyMimoPackWith(io: std.Io, a: std.mem.Allocator, dir: std.Io.Dir, expert_seed: u64, stored_affine: []const []const u8) !void {
    const H = TinyMimo.hidden;
    try dir.writeFile(io, .{ .sub_path = "config.json", .data = try std.mem.concat(a, u8, &.{
        \\{"model_type":"mimo_v2","vocab_size":8,"hidden_size":128,
        \\ "num_hidden_layers":2,"intermediate_size":128,
        \\ "moe_intermediate_size":128,"n_routed_experts":4,
        \\ "num_experts_per_tok":2,"n_group":1,"topk_group":1,
        \\ "num_attention_heads":8,"num_key_value_heads":8,"head_dim":16,
        \\ "v_head_dim":16,"swa_num_attention_heads":8,
        \\ "swa_num_key_value_heads":8,"swa_head_dim":16,"swa_v_head_dim":16,
        \\ "hybrid_layer_pattern":[0,0],"moe_layer_freq":[0,1],
        \\ "attention_projection_layout":"fused_qkv",
        \\ "add_swa_attention_sink_bias":false,
        \\ "add_full_attention_sink_bias":false,
        \\ "expert_quant":{"format":"exl3","k":2.5,"codebook":"mcg"}
        ,
        "}",
    }) });
    try dir.writeFile(io, .{ .sub_path = "tokenizer_config.json", .data = "{}" });
    try dir.writeFile(io, .{ .sub_path = "tokenizer.json", .data =
        \\{"pre_tokenizer":{"type":"ByteLevel"},"model":{"type":"BPE",
        \\ "vocab":{"a":0,"b":1,"c":2,"d":3,"e":4,"f":5,"g":6,"h":7},"merges":[]}}
    });

    var tensors: std.ArrayList(TinyTensor) = .empty;
    defer tensors.deinit(a);
    const vec_shape = try a.dupe(u64, &[_]u64{H});
    try tensors.append(a, .{ .key = "model.embed_tokens.weight", .dtype = "BF16", .shape = try a.dupe(u64, &[_]u64{ TinyMimo.vocab, H }), .bytes = try tinyBf16(a, TinyMimo.vocab * H, 11) });
    try tensors.append(a, .{ .key = "lm_head.weight", .dtype = "BF16", .shape = try a.dupe(u64, &[_]u64{ TinyMimo.vocab, H }), .bytes = try tinyBf16(a, TinyMimo.vocab * H, 12) });
    try tensors.append(a, .{ .key = "model.norm.weight", .dtype = "BF16", .shape = vec_shape, .bytes = try tinyBf16(a, H, 13) });

    for (0..TinyMimo.layers) |li| {
        const p = try std.fmt.allocPrint(a, "model.layers.{d}", .{li});
        const seed: u64 = 100 + li;
        try tensors.append(a, .{ .key = try std.fmt.allocPrint(a, "{s}.input_layernorm.weight", .{p}), .dtype = "BF16", .shape = vec_shape, .bytes = try tinyBf16(a, H, seed) });
        try tensors.append(a, .{ .key = try std.fmt.allocPrint(a, "{s}.post_attention_layernorm.weight", .{p}), .dtype = "BF16", .shape = vec_shape, .bytes = try tinyBf16(a, H, seed + 1) });
        try tensors.append(a, .{ .key = try std.fmt.allocPrint(a, "{s}.self_attn.qkv_proj.weight", .{p}), .dtype = "F8_E4M3", .shape = try a.dupe(u64, &[_]u64{ TinyMimo.qkv_rows, H }), .bytes = try tinyFp8(a, TinyMimo.qkv_rows * H, seed + 2) });
        try tensors.append(a, .{ .key = try std.fmt.allocPrint(a, "{s}.self_attn.qkv_proj.weight_scale_inv", .{p}), .dtype = "F32", .shape = try a.dupe(u64, &[_]u64{ 4, 1 }), .bytes = try tinyF32(a, &[_]f32{ 1.0, 0.75, 1.25, 0.5 }) });
        try tensors.append(a, .{ .key = try std.fmt.allocPrint(a, "{s}.self_attn.o_proj.weight", .{p}), .dtype = "BF16", .shape = try a.dupe(u64, &[_]u64{ H, H }), .bytes = try tinyBf16(a, H * H, seed + 3) });

        if (li == 0) {
            for ([_][]const u8{ "gate", "up", "down" }, 0..) |proj, j| {
                try tensors.append(a, .{ .key = try std.fmt.allocPrint(a, "{s}.mlp.{s}_proj.weight", .{ p, proj }), .dtype = "F8_E4M3", .shape = try a.dupe(u64, &[_]u64{ H, H }), .bytes = try tinyFp8(a, H * H, seed + 10 + j) });
                try tensors.append(a, .{ .key = try std.fmt.allocPrint(a, "{s}.mlp.{s}_proj.weight_scale_inv", .{ p, proj }), .dtype = "F32", .shape = try a.dupe(u64, &[_]u64{ 1, 1 }), .bytes = try tinyF32(a, &[_]f32{1.0}) });
            }
            continue;
        }
        try tensors.append(a, .{ .key = try std.fmt.allocPrint(a, "{s}.mlp.gate.weight", .{p}), .dtype = "BF16", .shape = try a.dupe(u64, &[_]u64{ TinyMimo.experts, H }), .bytes = try tinyBf16(a, TinyMimo.experts * H, seed + 20) });
        try tensors.append(a, .{ .key = try std.fmt.allocPrint(a, "{s}.mlp.gate.e_score_correction_bias", .{p}), .dtype = "F32", .shape = try a.dupe(u64, &[_]u64{TinyMimo.experts}), .bytes = try tinyF32(a, &[_]f32{ 0.0, 0.1, -0.1, 0.05 }) });
        for ([_][]const u8{ "gate", "up", "down" }, 0..) |proj, j| {
            const E = TinyMimo.experts;
            const halfwords = E * TinyMimo.tiles * TinyMimo.tiles * TinyMimo.trellis_n;
            try tensors.append(a, .{ .key = try std.fmt.allocPrint(a, "{s}.mlp.switch_mlp.{s}_proj.trellis", .{ p, proj }), .dtype = "U16", .shape = try a.dupe(u64, &[_]u64{ E, TinyMimo.tiles, TinyMimo.tiles, TinyMimo.trellis_n }), .bytes = try tinyU32(a, halfwords / 2, expert_seed * 1000 + j) });
            try tensors.append(a, .{ .key = try std.fmt.allocPrint(a, "{s}.mlp.switch_mlp.{s}_proj.suh", .{ p, proj }), .dtype = "F16", .shape = try a.dupe(u64, &[_]u64{ E, H }), .bytes = try tinyF16Axis(a, E * H, expert_seed * 1000 + 10 + j) });
            try tensors.append(a, .{ .key = try std.fmt.allocPrint(a, "{s}.mlp.switch_mlp.{s}_proj.svh", .{ p, proj }), .dtype = "F16", .shape = try a.dupe(u64, &[_]u64{ E, H }), .bytes = try tinyF16Axis(a, E * H, expert_seed * 1000 + 20 + j) });
        }
    }
    var stored: std.ArrayList(TinyTensor) = .empty;
    defer stored.deinit(a);
    for (tensors.items) |t| {
        for (stored_affine) |name| {
            if (std.mem.eql(u8, name, t.key)) break;
        } else {
            try stored.append(a, t);
            continue;
        }
        try appendAffine8(a, &stored, t);
    }
    try writeTinyShard(io, a, dir, stored.items);
}

fn appendAffine8(a: std.mem.Allocator, out: *std.ArrayList(TinyTensor), t: TinyTensor) !void {
    const s = mlx.gpuStream();
    defer _ = mlx.mlx_stream_free(s);
    const dense = mlx.mlx_array_new_data(t.bytes.ptr, &[_]c_int{ @intCast(t.shape[0]), @intCast(t.shape[1]) }, 2, .bfloat16);
    defer _ = mlx.mlx_array_free(dense);
    var parts = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(parts);
    try mlx.check(mlx.mlx_quantize(&parts, dense, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(8), "affine", .{ .ctx = null }, s));
    const base = t.key[0 .. t.key.len - ".weight".len];
    for ([_][]const u8{ "weight", "scales", "biases" }, [_][]const u8{ "U32", "BF16", "BF16" }, 0..) |part, dtype, i| {
        var arr = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(arr);
        try mlx.check(mlx.mlx_vector_array_get(&arr, parts, i));
        try mlx.check(mlx.mlx_array_eval(arr));
        const shape = mlx.getShape(arr);
        const bytes = mlx.mlx_array_size(arr) * @as(usize, if (i == 0) 4 else 2);
        const src: [*]const u8 = if (i == 0)
            @ptrCast(mlx.mlx_array_data_uint32(arr) orelse return error.KldLogitsUnreadable)
        else
            @ptrCast(mlx.mlx_array_data_bfloat16(arr) orelse return error.KldLogitsUnreadable);
        try out.append(a, .{
            .key = try std.fmt.allocPrint(a, "{s}.{s}", .{ base, part }),
            .dtype = dtype,
            .shape = try a.dupe(u64, &[_]u64{ @intCast(shape[0]), @intCast(shape[1]) }),
            .bytes = try a.dupe(u8, src[0..bytes]),
        });
    }
}

test "kld: a resident mimo_v2 load takes the source trunk and its logits follow the routed experts" {
    try mlx.requireGpu();
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var metal: bool = false;
    mlx.check(mlx.mlx_metal_is_available(&metal)) catch return error.SkipZigTest;
    if (!metal) return error.SkipZigTest;

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const ids = [_]u32{ 1, 3, 5, 2, 7, 0 };
    var rows: [2][TinyMimo.vocab]f32 = undefined;
    for ([_][]const u8{ "pack_a", "pack_b" }, 0..) |sub, arm| {
        try tmp.dir.createDirPath(io, sub);
        var dir = try tmp.dir.openDir(io, sub, .{});
        defer dir.close(io);
        try writeTinyMimoResidentPack(io, arena, dir, @as(u64, arm) + 1);
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const path_len = try dir.realPath(io, &path_buf);

        const loaded = try kld.loadModel(io, allocator, .{ .model_dir = path_buf[0..path_len] });
        defer loaded.deinit();
        // The served QKV is the source's rank-local FP8 bytes, which only the
        // FP8 kernels read; no generic split of it is bound.
        const qkv = loaded.weights.get("model.layers.0.self_attn.qkv_proj.weight") orelse return error.MissingWeight;
        try testing.expectEqual(mlx.mlx_dtype.uint8, mlx.mlx_array_dtype(qkv));
        try testing.expect(loaded.weights.get("model.layers.0.self_attn.q_proj.weight") == null);

        var ctx = loaded.xfm.defaultCtx();
        const logits = try kld.forwardPrompt(allocator, loaded, &ctx, &ids);
        defer _ = mlx.mlx_array_free(logits);
        try kld.readLastRow(&loaded.xfm, logits, &rows[arm]);
        for (rows[arm]) |v| try testing.expect(std.math.isFinite(v));
    }
    try testing.expect(!std.mem.eql(f32, &rows[0], &rows[1]));
}

test "kld: a prompt longer than the chunk is forwarded chunk by chunk, a shorter one in one forward" {
    try mlx.requireGpu();
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var metal: bool = false;
    mlx.check(mlx.mlx_metal_is_available(&metal)) catch return error.SkipZigTest;
    if (!metal) return error.SkipZigTest;

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    try tmp.dir.createDirPath(io, "pack");
    var dir = try tmp.dir.openDir(io, "pack", .{});
    defer dir.close(io);
    try writeTinyMimoResidentPack(io, arena, dir, 1);
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try dir.realPath(io, &path_buf);
    const loaded = try kld.loadModel(io, allocator, .{ .model_dir = path_buf[0..path_len] });
    defer loaded.deinit();

    const chunk = kld.promptChunk(&loaded.config);
    const ids = try allocator.alloc(u32, 2 * chunk);
    defer allocator.free(ids);
    for (ids, 0..) |*t, i| t.* = @intCast((i * 5 + 1) % TinyMimo.vocab);
    for ([_]usize{ chunk, 2 * chunk }) |len| {
        try loaded.xfm.resetCache();
        var ctx = loaded.xfm.defaultCtx();
        const got = try kld.forwardPrompt(allocator, loaded, &ctx, ids[0..len]);
        defer _ = mlx.mlx_array_free(got);
        try loaded.xfm.resetCache();
        var ref_ctx = loaded.xfm.defaultCtx();
        const expected = try kld.forwardPromptChunks(allocator, loaded, &ref_ctx, ids[0..len], chunk);
        defer _ = mlx.mlx_array_free(expected);
        var same = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(same);
        try mlx.check(mlx.mlx_array_equal(&same, got, expected, true, loaded.xfm.s));
        var equal = false;
        try mlx.check(mlx.mlx_array_item_bool(&equal, same));
        try testing.expect(equal);
    }
}

test "kld: an imatrix capture records every mimo_v2 o_proj input and the lm_head input the forward read" {
    try mlx.requireGpu();
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var metal: bool = false;
    mlx.check(mlx.mlx_metal_is_available(&metal)) catch return error.SkipZigTest;
    if (!metal) return error.SkipZigTest;

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    try tmp.dir.createDirPath(io, "pack");
    var dir = try tmp.dir.openDir(io, "pack", .{});
    defer dir.close(io);
    try writeTinyMimoResidentPack(io, arena, dir, 1);
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const path_len = try dir.realPath(io, &path_buf);

    const loaded = try kld.loadModel(io, allocator, .{ .model_dir = path_buf[0..path_len] });
    defer loaded.deinit();
    const col = try imatrix.Collector.init(allocator, loaded.xfm.s, "/dev/null", TinyMimo.layers, TinyMimo.experts, imatrix_naming);
    defer col.deinit();
    loaded.xfm.imatrix = col;
    defer loaded.xfm.imatrix = null;

    const ids = [_]u32{ 1, 3, 5, 2, 7, 0 };
    var hidden = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(hidden);
    var ctx = loaded.xfm.defaultCtx();
    ctx.capture_hidden_all = &hidden;
    const logits = try kld.forwardPrompt(allocator, loaded, &ctx, &ids);
    defer _ = mlx.mlx_array_free(logits);
    try mlx.check(mlx.mlx_array_eval(logits));

    for (col.o_proj) |d| try testing.expectEqual(@as(u64, ids.len), d.rows);
    try testing.expectEqual(@as(u64, ids.len), col.lm_head.rows);

    // lm_head's statistic is the per-channel sum of squares of the final
    // normed hidden rows, the same rows the logits were projected from.
    var h32 = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(h32);
    try mlx.check(mlx.mlx_astype(&h32, hidden, .float32, loaded.xfm.s));
    try mlx.check(mlx.mlx_array_eval(h32));
    try mlx.check(mlx.mlx_array_eval(col.lm_head.acc));
    const h = mlx.mlx_array_data_float32(h32) orelse return error.KldLogitsUnreadable;
    const acc = mlx.mlx_array_data_float32(col.lm_head.acc) orelse return error.KldLogitsUnreadable;
    for (0..TinyMimo.hidden) |c| {
        var want: f32 = 0;
        for (0..ids.len) |r| want += h[r * TinyMimo.hidden + c] * h[r * TinyMimo.hidden + c];
        try testing.expectApproxEqRel(want, acc[c], 1e-5);
    }
}

fn bf16HostBytes(a: std.mem.Allocator, s: mlx.mlx_stream, arr: mlx.mlx_array) ![]u8 {
    try testing.expectEqual(mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(arr));
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_contiguous(&c, arr, false, s));
    try mlx.check(mlx.mlx_array_eval(c));
    const p = mlx.mlx_array_data_bfloat16(c) orelse return error.KldLogitsUnreadable;
    return a.dupe(u8, std.mem.sliceAsBytes(p[0..mlx.mlx_array_size(c)]));
}

fn bf16At(bytes: []const u8, i: usize) f32 {
    return @bitCast(@as(u32, std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little)) << 16);
}

test "kld: a hidden capture appends every block boundary of each prompt and leaves the teacher bit-identical" {
    try mlx.requireGpu();
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var metal: bool = false;
    mlx.check(mlx.mlx_metal_is_available(&metal)) catch return error.SkipZigTest;
    if (!metal) return error.SkipZigTest;

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    try tmp.dir.createDirPath(io, "pack");
    {
        var dir = try tmp.dir.openDir(io, "pack", .{});
        defer dir.close(io);
        try writeTinyMimoResidentPack(io, arena, dir, 1);
    }
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = try arena.dupe(u8, path_buf[0..try tmp.dir.realPath(io, &path_buf)]);
    try tmp.dir.writeFile(io, .{ .sub_path = "windows.jsonl", .data =
        \\{"id":"w0","prompt_ids":[1,3,5,2,7,0]}
        \\{"id":"w1","prompt_ids":[4,4,6,1,2]}
    });
    const windows = [_][]const u32{ &.{ 1, 3, 5, 2, 7, 0 }, &.{ 4, 4, 6, 1, 2 } };
    const total = windows[0].len + windows[1].len;

    const pack = try std.fmt.allocPrint(arena, "{s}/pack", .{root});
    const loaded = try kld.loadModel(io, allocator, .{ .model_dir = pack });
    defer loaded.deinit();
    var quiet: kld.Out = .{ .silent = true };
    const base: kld.Options = .{
        .model_dir = pack,
        .prompts = try std.fmt.allocPrint(arena, "{s}/windows.jsonl", .{root}),
        .tokens = 3,
        .no_template = true,
    };
    var plain = base;
    plain.out_dir = try std.fmt.allocPrint(arena, "{s}/plain", .{root});
    try kld.runCapture(io, allocator, loaded, plain, &quiet);
    // Off: nothing is written.
    if (tmp.dir.statFile(io, "hidden", .{})) |_| return error.TestUnexpectedResult else |_| {}

    var armed = base;
    armed.out_dir = try std.fmt.allocPrint(arena, "{s}/armed", .{root});
    armed.hidden_out = try std.fmt.allocPrint(arena, "{s}/hidden", .{root});
    try kld.runCapture(io, allocator, loaded, armed, &quiet);

    // The teacher is untouched: every logits row and greedy token, byte for byte.
    for ([_][]const u8{ "prompts/00_w0", "prompts/01_w1" }) |sub| {
        for ([_][]const u8{ "logits.f32", "generated_tokens.txt" }) |name| {
            const a = try tmp.dir.readFileAlloc(io, try std.fmt.allocPrint(arena, "plain/{s}/{s}", .{ sub, name }), arena, .limited(1 << 20));
            const b = try tmp.dir.readFileAlloc(io, try std.fmt.allocPrint(arena, "armed/{s}/{s}", .{ sub, name }), arena, .limited(1 << 20));
            try testing.expect(a.len > 0);
            try testing.expectEqualSlices(u8, a, b);
        }
    }

    const H = TinyMimo.hidden;
    const toks = try tmp.dir.readFileAlloc(io, "hidden/tokens.bin", arena, .limited(1 << 20));
    try testing.expectEqual(total * 4, toks.len);
    var files: [TinyMimo.layers + 1][]u8 = undefined;
    for (&files, 0..) |*f, b| {
        f.* = try tmp.dir.readFileAlloc(io, try std.fmt.allocPrint(arena, "hidden/boundary-{d:0>2}.bin", .{b}), arena, .limited(1 << 20));
        try testing.expectEqual(total * H * 2, f.len);
    }

    const embed = try tinyBf16(arena, TinyMimo.vocab * H, 11);
    const norm_w = try tinyBf16(arena, H, 13);
    const s = loaded.xfm.s;
    var row: usize = 0;
    for (windows) |w| {
        for (w, row..) |t, r| {
            try testing.expectEqual(t, std.mem.readInt(u32, toks[r * 4 ..][0..4], .little));
            // Boundary 0 is the token's own embedding row.
            try testing.expectEqualSlices(u8, embed[t * H * 2 ..][0 .. H * 2], files[0][r * H * 2 ..][0 .. H * 2]);
        }

        // Every boundary holds the forward's own residual at that depth.
        try loaded.xfm.resetCache();
        const layer_ids = [_]u32{ 0, 1 };
        var outs = [_]mlx.mlx_array{ mlx.mlx_array_new(), mlx.mlx_array_new() };
        defer for (outs) |o| {
            _ = mlx.mlx_array_free(o);
        };
        var input = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(input);
        var final = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(final);
        var cl: transformer_mod.CaptureLayers = .{ .ids = &layer_ids, .out = &outs, .input = &input };
        var ctx = loaded.xfm.defaultCtx();
        ctx.capture_layers = &cl;
        ctx.capture_hidden_all = &final;
        const logits = try kld.forwardPrompt(allocator, loaded, &ctx, w);
        defer _ = mlx.mlx_array_free(logits);
        try mlx.check(mlx.mlx_array_eval(logits));
        for ([_]mlx.mlx_array{ input, outs[0], outs[1] }, 0..) |arr, b| {
            const got = try bf16HostBytes(arena, s, arr);
            try testing.expectEqualSlices(u8, got, files[b][row * H * 2 ..][0 .. w.len * H * 2]);
        }

        // The last boundary is the residual the final norm reads.
        const fin = try bf16HostBytes(arena, s, final);
        const last = files[TinyMimo.layers][row * H * 2 ..][0 .. w.len * H * 2];
        for (0..w.len) |r| {
            var ms: f64 = 0;
            for (0..H) |c| ms += @as(f64, bf16At(last, r * H + c)) * bf16At(last, r * H + c);
            const inv = 1.0 / @sqrt(ms / @as(f64, H) + loaded.config.rms_norm_eps);
            for (0..H) |c| {
                const want: f64 = bf16At(last, r * H + c) * inv * bf16At(norm_w, c);
                try testing.expectApproxEqAbs(want, @as(f64, bf16At(fin, r * H + c)), 1e-2 + 1e-2 * @abs(want));
            }
        }
        row += w.len;
    }
}

test "kld: a pack storing o_proj, lm_head and embed_tokens affine serves them packed beside its bf16 twin" {
    try mlx.requireGpu();
    const allocator = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    var metal: bool = false;
    mlx.check(mlx.mlx_metal_is_available(&metal)) catch return error.SkipZigTest;
    if (!metal) return error.SkipZigTest;

    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const o_proj = [_][]const u8{ "model.layers.0.self_attn.o_proj.weight", "model.layers.1.self_attn.o_proj.weight" };
    const tables = [_][]const u8{ "lm_head.weight", "model.embed_tokens.weight" };
    const arms = [_][]const []const u8{ &.{}, &o_proj, &tables, tables[1..] };
    const ids = [_]u32{ 1, 3, 5, 2, 7, 0 };
    var rows: [arms.len][TinyMimo.vocab]f32 = undefined;
    for (arms, 0..) |stored, arm| {
        const sub = ([_][]const u8{ "source", "o_proj", "tables", "embed" })[arm];
        try tmp.dir.createDirPath(io, sub);
        var dir = try tmp.dir.openDir(io, sub, .{});
        defer dir.close(io);
        try writeTinyMimoPackWith(io, arena, dir, 1, stored);
        var path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const path_len = try dir.realPath(io, &path_buf);

        const loaded = try kld.loadModel(io, allocator, .{ .model_dir = path_buf[0..path_len] });
        defer loaded.deinit();
        for (o_proj ++ tables) |name| {
            const packed_here = for (stored) |s| {
                if (std.mem.eql(u8, s, name)) break true;
            } else false;
            const w = loaded.weights.get(name) orelse return error.MissingWeight;
            try testing.expectEqual(if (packed_here) mlx.mlx_dtype.uint32 else mlx.mlx_dtype.bfloat16, mlx.mlx_array_dtype(w));
        }

        var ctx = loaded.xfm.defaultCtx();
        const logits = try kld.forwardPrompt(allocator, loaded, &ctx, &ids);
        defer _ = mlx.mlx_array_free(logits);
        try kld.readLastRow(&loaded.xfm, logits, &rows[arm]);
        for (rows[arm]) |v| try testing.expect(std.math.isFinite(v));
    }
    // 8-bit weights move the logits only by their own rounding; a bf16 head
    // beside a packed embedding keeps its own (absent) scales.
    for (1..arms.len) |arm| {
        try testing.expect(!std.mem.eql(f32, &rows[0], &rows[arm]));
        var dot: f64 = 0;
        var na: f64 = 0;
        var nb: f64 = 0;
        for (rows[0], rows[arm]) |a, b| {
            dot += a * b;
            na += a * a;
            nb += b * b;
        }
        try testing.expect(dot / @sqrt(na * nb) > 0.98);
    }
}
