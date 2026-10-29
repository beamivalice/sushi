//! Small, deliberately shuffled source shards for the GLM BF16 and FP8 stream tests.
const std = @import("std");

pub const Fault = enum { none, missing, fp16, wrong_shape, truncated, dense_prefix, mixed, fp8_scale };
pub const Storage = enum { bf16, fp8 };

pub fn value(expert: usize, projection: usize) u16 {
    return @intCast(0x3f00 + expert * 16 + projection * 4);
}

/// A finite e4m3 code (never the 0x7f/0xff NaNs) that differs per expert, projection and element.
pub fn fp8Code(expert: usize, projection: usize, i: usize) u8 {
    var h: u32 = @intCast((expert * 131 + projection * 31 + i) & 0xffffffff);
    h = (h ^ (h >> 7)) *% 0x9e3779b1;
    const code: u8 = @truncate(h >> 13);
    return if (code & 0x7f == 0x7f) code ^ 1 else code;
}

/// A block-128 scale large enough that one routed expert moves a fixture model's logits.
pub fn fp8Scale(expert: usize, projection: usize, block: usize) f32 {
    return @as(f32, @floatFromInt(17 + expert * 5 + projection * 3 + block)) * 1.0e-4;
}

pub fn write(allocator: std.mem.Allocator, dir: std.Io.Dir, fault: Fault) !void {
    return writeSized(allocator, dir, fault, 32, 16);
}

pub fn writeSized(allocator: std.mem.Allocator, dir: std.Io.Dir, fault: Fault, hidden: usize, intermediate: usize) !void {
    return writeStorage(allocator, dir, fault, hidden, intermediate, .bf16);
}

pub fn writeStorage(allocator: std.mem.Allocator, dir: std.Io.Dir, fault: Fault, hidden: usize, intermediate: usize, storage: Storage) !void {
    const io = std.testing.io;
    var index: std.ArrayList(u8) = .empty;
    defer index.deinit(allocator);
    try index.appendSlice(allocator, "{\"weight_map\":{");
    var entries: usize = 0;
    for ([_][]const u8{ "gate", "up", "down" }, 0..) |projection, pi| {
        var header: std.ArrayList(u8) = .empty;
        defer header.deinit(allocator);
        try header.append(allocator, '{');
        const payload = try allocator.alloc(u8, 4 * (hidden * intermediate * 2 + hidden * intermediate / (128 * 128) * 4 + 4));
        defer allocator.free(payload);
        var offset: usize = 0;
        var count: usize = 0;
        for ([_]usize{ 3, 1, 0, 2 }) |expert| {
            if (fault == .missing and expert == 2 and pi == 1) continue;
            const layer: usize = if (fault == .dense_prefix and expert == 2 and pi == 1) 2 else 3;
            const rows: usize = if (pi == 2) hidden else intermediate;
            const cols: usize = if (fault == .wrong_shape and expert == 2 and pi == 1) hidden - 1 else if (pi == 2) intermediate else hidden;
            const parts: usize = if (storage == .fp8) 2 else 1;
            for (0..parts) |part| {
                if (part == 1 and fault == .fp8_scale and expert == 2 and pi == 1) continue;
                const key = try std.fmt.allocPrint(allocator, "model.language_model.layers.{d}.mlp.experts.{d}.{s}_proj.{s}", .{ layer, expert, projection, if (part == 0) "weight" else "weight_scale_inv" });
                defer allocator.free(key);
                const shape: [2]usize = if (part == 0) .{ rows, cols } else .{ rows / 128, cols / 128 };
                const elem: usize = if (part == 1) 4 else if (storage == .fp8) 1 else 2;
                const bytes = shape[0] * shape[1] * elem;
                const dtype: []const u8 = if (part == 1) "F32" else if (storage == .fp8) "F8_E4M3" else if (fault == .fp16 and expert == 2 and pi == 1) "F16" else "BF16";
                const entry = try std.fmt.allocPrint(allocator, "{s}\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[{d},{d}],\"data_offsets\":[{d},{d}]}}", .{ if (count == 0) "" else ",", key, dtype, shape[0], shape[1], offset, offset + bytes });
                defer allocator.free(entry);
                try header.appendSlice(allocator, entry);
                const mapping = try std.fmt.allocPrint(allocator, "{s}\"{s}\":\"{s}.safetensors\"", .{ if (entries == 0) "" else ",", key, projection });
                defer allocator.free(mapping);
                try index.appendSlice(allocator, mapping);
                for (0..shape[0] * shape[1]) |i| {
                    if (part == 1) {
                        std.mem.writeInt(u32, payload[offset + i * 4 ..][0..4], @bitCast(fp8Scale(expert, pi, i)), .little);
                    } else if (storage == .fp8) {
                        payload[offset + i] = fp8Code(expert, pi, i);
                    } else std.mem.writeInt(u16, payload[offset + i * 2 ..][0..2], value(expert, pi), .little);
                }
                offset += bytes;
                count += 1;
                entries += 1;
            }
        }
        try header.append(allocator, '}');
        var bytes: std.ArrayList(u8) = .empty;
        defer bytes.deinit(allocator);
        var len: [8]u8 = undefined;
        std.mem.writeInt(u64, &len, header.items.len, .little);
        try bytes.appendSlice(allocator, &len);
        try bytes.appendSlice(allocator, header.items);
        try bytes.appendSlice(allocator, payload[0..offset]);
        const filename = try std.fmt.allocPrint(allocator, "{s}.safetensors", .{projection});
        defer allocator.free(filename);
        const trim: usize = if (fault == .truncated and pi == 1) 1 else 0;
        try dir.writeFile(io, .{ .sub_path = filename, .data = bytes.items[0 .. bytes.items.len - trim] });
    }
    // Dense/shared weights are not opened by the expert store; nor is the MTP shard.
    try index.appendSlice(allocator, ",\"model.language_model.layers.0.mlp.gate_proj.weight\":\"dense-unopened.safetensors\",\"model.language_model.layers.3.mlp.shared_experts.gate_proj.weight\":\"shared-unopened.safetensors\",\"model.language_model.layers.4.mlp.experts.0.gate_proj.weight\":\"mtp-unopened.safetensors\"");
    if (fault == .mixed) try index.appendSlice(allocator, ",\"model.language_model.layers.3.mlp.switch_mlp.gate_proj.weight\":\"mixed.safetensors\"");
    if (fault == .fp8_scale and storage == .bf16) try index.appendSlice(allocator, ",\"model.language_model.layers.3.mlp.experts.0.gate_proj.weight_scale_inv\":\"fp8.safetensors\"");
    try index.appendSlice(allocator, "}}");
    try dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index.items });
}

/// A seeded GLM-5.3 BF16 source checkpoint small enough for a test: three dense KDA layers, an MLA MoE
/// layer and a KDA MoE layer, a random trunk, individual BF16 experts and a byte-level tokenizer.
pub const Tiny = struct {
    pub const vocab = 16;
    pub const hidden = 128;
    pub const layers = 5;
    pub const experts = 8;
};

const TinyTensor = struct { name: []const u8, dtype: []const u8, shape: []const usize, bytes: []u8 };

const TinyWriter = struct {
    a: std.mem.Allocator,
    rnd: std.Random,
    tensors: std.ArrayList(TinyTensor) = .empty,

    fn add(self: *TinyWriter, name: []const u8, shape: []const usize, f32_storage: bool, lo: f32, hi: f32) !void {
        var count: usize = 1;
        for (shape) |d| count *= d;
        const bytes = try self.a.alloc(u8, count * @as(usize, if (f32_storage) 4 else 2));
        for (0..count) |i| {
            const bits: u32 = @bitCast(lo + (hi - lo) * self.rnd.float(f32));
            if (f32_storage) std.mem.writeInt(u32, bytes[i * 4 ..][0..4], bits, .little) else std.mem.writeInt(u16, bytes[i * 2 ..][0..2], @truncate(bits >> 16), .little);
        }
        try self.tensors.append(self.a, .{ .name = try self.a.dupe(u8, name), .dtype = if (f32_storage) "F32" else "BF16", .shape = try self.a.dupe(usize, shape), .bytes = bytes });
    }

    fn linear(self: *TinyWriter, prefix: []const u8, leaf: []const u8, rows: usize, cols: usize) !void {
        const bound = 1.7 / @sqrt(@as(f32, @floatFromInt(cols)));
        try self.add(try std.fmt.allocPrint(self.a, "{s}.{s}", .{ prefix, leaf }), &.{ rows, cols }, false, -bound, bound);
    }

    fn shard(self: *TinyWriter, dir: std.Io.Dir, file: []const u8, tensors: []const TinyTensor, index: *std.ArrayList(u8)) !void {
        var header: std.ArrayList(u8) = .empty;
        try header.append(self.a, '{');
        var offset: usize = 0;
        for (tensors, 0..) |t, i| {
            try header.print(self.a, "{s}\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ if (i == 0) "" else ",", t.name, t.dtype });
            for (t.shape, 0..) |d, j| try header.print(self.a, "{s}{d}", .{ if (j == 0) "" else ",", d });
            try header.print(self.a, "],\"data_offsets\":[{d},{d}]}}", .{ offset, offset + t.bytes.len });
            try index.print(self.a, "{s}\"{s}\":\"{s}\"", .{ if (index.items.len == 0) "" else ",", t.name, file });
            offset += t.bytes.len;
        }
        try header.append(self.a, '}');
        while (header.items.len % 8 != 0) try header.append(self.a, ' ');
        var bytes: std.ArrayList(u8) = .empty;
        var len: [8]u8 = undefined;
        std.mem.writeInt(u64, &len, header.items.len, .little);
        try bytes.appendSlice(self.a, &len);
        try bytes.appendSlice(self.a, header.items);
        for (tensors) |t| try bytes.appendSlice(self.a, t.bytes);
        try dir.writeFile(std.testing.io, .{ .sub_path = file, .data = bytes.items });
    }
};

pub fn writeCheckpoint(allocator: std.mem.Allocator, dir: std.Io.Dir, seed: u64) !void {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    var prng = std.Random.DefaultPrng.init(seed);
    var w = TinyWriter{ .a = arena.allocator(), .rnd = prng.random() };
    const H = Tiny.hidden;
    const fmt = std.fmt.allocPrint;
    try w.add("model.language_model.embed_tokens.weight", &.{ Tiny.vocab, H }, false, -1, 1);
    try w.add("lm_head.weight", &.{ Tiny.vocab, H }, false, -0.5, 0.5);
    try w.add("model.language_model.norm.weight", &.{H}, false, 0.5, 1.5);
    for (0..Tiny.layers) |li| {
        const p = try fmt(w.a, "model.language_model.layers.{d}", .{li});
        for ([_][]const u8{ "hc_attn", "hc_ffn" }) |hc| {
            try w.add(try fmt(w.a, "{s}.{s}_fn", .{ p, hc }), &.{ 24, 4 * H }, true, -0.05, 0.05);
            try w.add(try fmt(w.a, "{s}.{s}_scale", .{ p, hc }), &.{3}, true, 0.5, 1.5);
            try w.add(try fmt(w.a, "{s}.{s}_base", .{ p, hc }), &.{24}, true, -0.5, 0.5);
        }
        for ([_][]const u8{ "input_layernorm", "post_attention_layernorm" }) |norm| try w.add(try fmt(w.a, "{s}.{s}.weight", .{ p, norm }), &.{H}, false, 0.5, 1.5);
        const ap = try fmt(w.a, "{s}.self_attn", .{p});
        if ((li + 1) % 4 != 0) {
            for ([_][]const u8{ "q_proj.weight", "k_proj.weight", "v_proj.weight", "f_a_proj.weight", "f_b_proj.weight", "g_a_proj.weight", "g_b_proj.weight", "o_proj.weight" }) |leaf| try w.linear(ap, leaf, H, H);
            try w.linear(ap, "b_proj.weight", 1, H);
            for ([_][]const u8{ "q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight" }) |leaf| try w.add(try fmt(w.a, "{s}.{s}", .{ ap, leaf }), &.{ H, 1, 4 }, false, -0.5, 0.5);
            try w.add(try fmt(w.a, "{s}.A_log", .{ap}), &.{1}, true, -0.5, 0.5);
            try w.add(try fmt(w.a, "{s}.dt_bias", .{ap}), &.{H}, true, -0.5, 0.5);
            try w.add(try fmt(w.a, "{s}.o_norm.weight", .{ap}), &.{H}, false, 0.5, 1.5);
        } else {
            // 256-wide query and value heads, as served, so dense prefill engages past 8 rows.
            for ([_][]const u8{ "q_a_proj.weight", "kv_a_proj_with_mqa.weight", "indexer.wq_b.weight", "indexer.wk.weight", "indexer.index_kpool_compress_gate" }) |leaf| try w.linear(ap, leaf, H, H);
            try w.linear(ap, "q_b_proj.weight", 256, H);
            try w.linear(ap, "kv_b_proj.weight", 512, H);
            try w.linear(ap, "o_proj.weight", H, 256);
            try w.linear(ap, "indexer.weights_proj.weight", 1, H);
            for ([_][]const u8{ "q_a_layernorm.weight", "kv_a_layernorm.weight", "indexer.k_norm.weight" }) |leaf| try w.add(try fmt(w.a, "{s}.{s}", .{ ap, leaf }), &.{H}, false, 0.5, 1.5);
            try w.add(try fmt(w.a, "{s}.indexer.k_norm.bias", .{ap}), &.{H}, false, -0.1, 0.1);
            try w.add(try fmt(w.a, "{s}.indexer.index_kpool_compress_ape", .{ap}), &.{ 4, H }, false, -0.1, 0.1);
        }
        const mp = try fmt(w.a, "{s}.mlp", .{p});
        const dense = if (li < 3) mp else try fmt(w.a, "{s}.shared_experts", .{mp});
        for ([_][]const u8{ "gate_proj.weight", "up_proj.weight", "down_proj.weight" }) |leaf| try w.linear(dense, leaf, H, H);
        if (li < 3) continue;
        try w.add(try fmt(w.a, "{s}.gate.weight", .{mp}), &.{ Tiny.experts, H }, true, -1, 1);
        try w.add(try fmt(w.a, "{s}.gate.e_score_correction_bias", .{mp}), &.{Tiny.experts}, true, -0.1, 0.1);
    }
    const trunk_count = w.tensors.items.len;
    for (3..Tiny.layers) |li| for (0..Tiny.experts) |e| {
        const p = try fmt(w.a, "model.language_model.layers.{d}.mlp.experts.{d}", .{ li, e });
        for ([_][]const u8{ "gate_proj.weight", "up_proj.weight", "down_proj.weight" }) |leaf| try w.linear(p, leaf, H, H);
    };
    var index: std.ArrayList(u8) = .empty;
    try w.shard(dir, "model-00001-of-00002.safetensors", w.tensors.items[0..trunk_count], &index);
    try w.shard(dir, "model-00002-of-00002.safetensors", w.tensors.items[trunk_count..], &index);
    const io = std.testing.io;
    try dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = try std.mem.concat(w.a, u8, &.{ "{\"weight_map\":{", index.items, "}}" }) });
    try dir.writeFile(io, .{ .sub_path = "config.json", .data =
        \\{"model_type":"glm5_next","tie_word_embeddings":false,"text_config":{
        \\ "model_type":"glm5_next_text","mhc":true,"attention_bias":false,"index_kpool_compress":true,
        \\ "tie_word_embeddings":false,"scoring_func":"sigmoid","topk_method":"noaux_tc","hidden_act":"silu",
        \\ "moe_router_dtype":"float32","hidden_size":128,"vocab_size":16,"intermediate_size":128,
        \\ "num_hidden_layers":5,"num_attention_heads":1,"num_experts_per_tok":2,"moe_intermediate_size":128,
        \\ "max_position_embeddings":4096,"rms_norm_eps":1e-05,"n_routed_experts":8,"first_k_dense_replace":3,
        \\ "n_shared_experts":1,"routed_scaling_factor":2.5,"norm_topk_prob":true,"n_group":1,"topk_group":1,
        \\ "q_lora_rank":128,"kv_lora_rank":128,"qk_nope_head_dim":256,"qk_rope_head_dim":0,"v_head_dim":256,
        \\ "mla_use_nope":true,"linear_attn_config":{"num_heads":1,"head_dim":128,"short_conv_kernel_size":4,
        \\ "gate_lower_bound":-5.0},"hc_mult":4,"hc_sinkhorn_iters":20,"hc_eps":1e-06,"swiglu_limit":10.0,
        \\ "index_n_heads":1,"index_head_dim":128,"index_topk":2048,"index_kpool":4,
        \\ "index_kpool_always_select_tail":true,"num_nextn_predict_layers":0,"eos_token_id":[15],
        \\ "mlp_layer_types":["dense","dense","dense","sparse","sparse"],
        \\ "layer_types":["linear_attention","linear_attention","linear_attention","deepseek_sparse_attention",
        \\ "linear_attention"]}}
    });
    try dir.writeFile(io, .{ .sub_path = "tokenizer_config.json", .data = "{}" });
    try dir.writeFile(io, .{ .sub_path = "tokenizer.json", .data =
        \\{"pre_tokenizer":{"type":"ByteLevel"},"model":{"type":"BPE","vocab":{"a":0,"b":1,"c":2,"d":3,
        \\ "e":4,"f":5,"g":6,"h":7,"i":8,"j":9,"k":10,"l":11,"m":12,"n":13,"o":14,"p":15},"merges":[]}}
    });
}
