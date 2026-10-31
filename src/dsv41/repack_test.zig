//! `sushi repack` against synthetic legacy packs generated here, and against a copy of the tiny STQ fixture.
const std = @import("std");
const testing = std.testing;
const exl3 = @import("sushi_exl3");
const repack = @import("repack.zig");
const model = @import("../model.zig");
const dsv41_source = @import("dsv41_source.zig");

const page: usize = 16384;
const hidden: u32 = 256;
const inter: u32 = 128;

fn io() std.Io {
    return std.Io.Threaded.global_single_threaded.io();
}

const config_text =
    \\{
    \\ "architectures": ["DeepseekV41ForCausalLM"],
    \\ "model_type": "deepseek_v41",
    \\ "dtype": "bfloat16",
    \\ "bos_token_id": 0,
    \\ "eos_token_id": 1,
    \\ "pad_token_id": 2,
    \\ "image_token_id": 129264,
    \\ "quantization_config": {"quant_method": "fp8", "activation_scheme": "dynamic", "weight_block_size": [32, 32], "scale_fmt": "ue8m0", "expert_dtype": "fp4"},
    \\ "text_config": {"model_type": "deepseek_v41_text", "vocab_size": 256, "hidden_size": 256, "moe_intermediate_size": 128, "num_hidden_layers": 8, "num_attention_heads": 2, "num_key_value_heads": 1, "hidden_act": "silu", "head_dim": 128, "qk_rope_head_dim": 16, "q_lora_rank": 128, "o_lora_rank": 128, "o_groups": 2, "swiglu_limit": 1.0, "rms_norm_eps": 1e-20, "max_position_embeddings": 128, "rope_theta": 10000.0, "rope_scaling": {"rope_type": "yarn", "factor": 4, "beta_fast": 32, "beta_slow": 1, "original_max_position_embeddings": 64}, "n_routed_experts": 5, "n_shared_experts": 1, "num_experts_per_tok": 3, "scoring_func": "sqrtsoftplus", "topk_method": "noaux_tc", "norm_topk_prob": true, "routed_scaling_factor": 1.5, "sliding_window": 8, "compress_ratios": [0, 0, 2, 2, 2, 1, 1, 1, 0, 0, 0], "compress_rope_theta": 40000.0, "kv_source_layer_ids": [2, 5], "index_source_layer_ids": [2, 5, 7], "index_n_heads": 8, "index_head_dim": 128, "index_topk": 4, "candidate_source_layer_id": 5, "candidate_topk_blocks": 3, "candidate_block_size": 2, "hc_mult": 4, "hc_sinkhorn_iters": 20, "hc_eps": 1e-06, "engram_layer_ids": [1, 2], "engram_max_ngram_size": 4, "engram_vocab_size": 50, "engram_n_heads": 2, "engram_head_dim": 64, "engram_pad_token_id": 2, "engram_compressed_vocab_size": 256, "num_nextn_predict_layers": 3, "dspark_block_size": 4, "dspark_noise_token_id": 255, "dspark_target_layer_ids": [5, 6, 7], "dspark_markov_rank": 32, "dspark_n_routed_experts": 4, "dspark_num_experts_per_tok": 2, "dtype": "bfloat16", "engram_num_embeddings": [384, 552]},
    \\ "expert_quant": {"format": "FORMAT", "k": 4.0, "codebook": "pair1", "window": 16, "codebook_table": "pair1-w16-v1", "codebook_digest": "c089b4eb"EXTRA},
    \\ "trunk_quant": {"format": "stq", "k": 6.0, "codebook": "spec1", "window": 15, "codebook_table": "spec1-w15", "codebook_digest": "37f72431"}
    \\}
    \\
;

const Fault = struct {
    model_type: []const u8 = "deepseek_v41",
    format: []const u8 = "stq",
    quant_extra: []const u8 = "",
    stamp_codebook: []const u8 = "pair1",
    stamp_k: []const u8 = "4",
    /// Shard 2 holds one trellis already renamed `trellis_pm` beside legacy ones.
    mixed: bool = false,
    /// Shard 2 is a symlink to a file beside the pack.
    link: bool = false,
    /// Shard 1 stamps a tile order.
    legacy_stamp: bool = false,
};

const shard_a = "model-00001-of-00002.safetensors";
const shard_b = "model-00002-of-00002.safetensors";

const Tensor = struct { key: []const u8, dtype: []const u8, shape: []const u64, bytes: []u8 };

var seed_state: u64 = 0x9e3779b97f4a7c15;

fn noise(a: std.mem.Allocator, len: usize) ![]u8 {
    const out = try a.alloc(u8, len);
    for (out) |*b| {
        seed_state ^= seed_state << 13;
        seed_state ^= seed_state >> 7;
        seed_state ^= seed_state << 17;
        b.* = @truncate(seed_state >> 24);
    }
    return out;
}

fn shapeBytes(shape: []const u64, elem: u64) usize {
    var n: u64 = elem;
    for (shape) |d| n *= d;
    return @intCast(n);
}

fn dupeShape(a: std.mem.Allocator, shape: []const u64) ![]u64 {
    return a.dupe(u64, shape);
}

fn push(a: std.mem.Allocator, list: *std.ArrayList(Tensor), key: []const u8, dtype: []const u8, shape: []const u64, elem: u64) !void {
    try list.append(a, .{ .key = try a.dupe(u8, key), .dtype = dtype, .shape = try dupeShape(a, shape), .bytes = try noise(a, shapeBytes(shape, elem)) });
}

const Group = struct { experts: u64, n_gate: u64, n_down: u64 };

/// One layer's routed tensors in the order the encoder lays a shard out: every trellis, then every suh, then every svh.
fn pushRouted(a: std.mem.Allocator, list: *std.ArrayList(Tensor), prefix: []const u8, groups: []const Group, mixed_first: bool) !void {
    const projs = [_][]const u8{ "down_proj", "gate_proj", "up_proj" };
    for (projs) |proj| for (groups, 0..) |g, gi| {
        const down = std.mem.eql(u8, proj, "down_proj");
        const in_t: u64 = if (down) inter / 16 else hidden / 16;
        const out_t: u64 = if (down) hidden / 16 else inter / 16;
        var key_buf: [160]u8 = undefined;
        const leaf = if (mixed_first and std.mem.eql(u8, proj, "down_proj") and gi == 0) "trellis_pm" else "trellis";
        const key = try std.fmt.bufPrint(&key_buf, "{s}{s}.g{d}.{s}", .{ prefix, proj, gi, leaf });
        if (std.mem.eql(u8, leaf, "trellis_pm")) {
            try push(a, list, key, "U16", &.{ g.experts, out_t / 2, in_t, 2, if (down) g.n_down else g.n_gate }, 2);
        } else {
            try push(a, list, key, "U16", &.{ g.experts, in_t, out_t, if (down) g.n_down else g.n_gate }, 2);
        }
    };
    for ([_][]const u8{ "suh", "svh" }) |leaf| for (projs) |proj| for (groups, 0..) |g, gi| {
        const down = std.mem.eql(u8, proj, "down_proj");
        const width: u64 = if (std.mem.eql(u8, leaf, "suh")) (if (down) inter else hidden) else (if (down) hidden else inter);
        var key_buf: [160]u8 = undefined;
        const key = try std.fmt.bufPrint(&key_buf, "{s}{s}.g{d}.{s}", .{ prefix, proj, gi, leaf });
        try push(a, list, key, "F16", &.{ g.experts, width }, 2);
    };
}

fn jsonString(w: *std.ArrayList(u8), a: std.mem.Allocator, s: []const u8) !void {
    try w.append(a, '"');
    try w.appendSlice(a, s);
    try w.append(a, '"');
}

fn writeShardFile(a: std.mem.Allocator, dir: []const u8, name: []const u8, meta: []const [2][]const u8, tensors: []const Tensor) !void {
    var header: std.ArrayList(u8) = .empty;
    try header.append(a, '{');
    if (meta.len > 0) {
        try header.appendSlice(a, "\"__metadata__\":{");
        for (meta, 0..) |kv, i| {
            if (i > 0) try header.append(a, ',');
            try jsonString(&header, a, kv[0]);
            try header.append(a, ':');
            try jsonString(&header, a, kv[1]);
        }
        try header.appendSlice(a, "},");
    }
    var at: usize = 0;
    for (tensors, 0..) |t, i| {
        if (i > 0) try header.append(a, ',');
        try jsonString(&header, a, t.key);
        try header.print(a, ":{{\"dtype\":\"{s}\",\"shape\":[", .{t.dtype});
        for (t.shape, 0..) |d, j| try header.print(a, "{s}{d}", .{ if (j > 0) "," else "", d });
        try header.print(a, "],\"data_offsets\":[{d},{d}]}}", .{ at, at + t.bytes.len });
        at += t.bytes.len;
    }
    try header.append(a, '}');
    while (header.items.len % 8 != 0) try header.append(a, ' ');
    var file: std.ArrayList(u8) = .empty;
    try file.appendNTimes(a, 0, 8);
    std.mem.writeInt(u64, file.items[0..8], header.items.len, .little);
    try file.appendSlice(a, header.items);
    for (tensors) |t| try file.appendSlice(a, t.bytes);
    try writeFile(dir, name, file.items);
}

fn writeFile(dir: []const u8, name: []const u8, data: []const u8) !void {
    var d = try std.Io.Dir.openDirAbsolute(io(), dir, .{});
    defer d.close(io());
    try d.writeFile(io(), .{ .sub_path = name, .data = data });
}

fn readFile(a: std.mem.Allocator, dir: []const u8, name: []const u8) ![]u8 {
    var d = try std.Io.Dir.openDirAbsolute(io(), dir, .{});
    defer d.close(io());
    return d.readFileAlloc(io(), name, a, .limited(256 << 20));
}

fn replaceAll(a: std.mem.Allocator, text: []const u8, from: []const u8, to: []const u8) ![]u8 {
    return std.mem.replaceOwned(u8, a, text, from, to);
}

/// A legacy (pack-order) pack of two shards: layer 0 and an MTP-less layer 1 plus a DSpark stage in the second.
fn writeLegacyPack(a: std.mem.Allocator, dir: []const u8, fault: Fault) !void {
    var cfg = try replaceAll(a, config_text, "FORMAT", fault.format);
    cfg = try replaceAll(a, cfg, "EXTRA", fault.quant_extra);
    cfg = try replaceAll(a, cfg, "\"deepseek_v41\"", try std.fmt.allocPrint(a, "\"{s}\"", .{fault.model_type}));
    try writeFile(dir, "config.json", cfg);

    const meta = [_][2][]const u8{
        .{ "format", "stq" },
        .{ "k", fault.stamp_k },
        .{ "codebook", fault.stamp_codebook },
        .{ "window", "16" },
    };
    const meta_stamped = meta ++ [_][2][]const u8{.{ "tile_order", "pair_major" }};

    var first: std.ArrayList(Tensor) = .empty;
    try push(a, &first, "model.embed_tokens.weight", "BF16", &.{ 4, 8 }, 2);
    for (0..4) |layer| {
        const prefix = try std.fmt.allocPrint(a, "model.layers.{d}.mlp.switch_mlp.", .{layer});
        try pushRouted(a, &first, prefix, &.{ .{ .experts = 3, .n_gate = 24, .n_down = 32 }, .{ .experts = 2, .n_gate = 32, .n_down = 32 } }, false);
    }
    try push(a, &first, "model.layers.0.input_layernorm.weight", "F32", &.{ 3, 5 }, 4);
    try writeShardFile(a, dir, shard_a, if (fault.legacy_stamp) &meta_stamped else &meta, first.items);

    var second: std.ArrayList(Tensor) = .empty;
    for (4..8) |layer| {
        const prefix = try std.fmt.allocPrint(a, "model.layers.{d}.mlp.switch_mlp.", .{layer});
        try pushRouted(a, &second, prefix, &.{.{ .experts = 5, .n_gate = 48, .n_down = 64 }}, fault.mixed and layer == 4);
    }
    try push(a, &second, "model.norm.weight", "BF16", &.{8}, 2);
    for (0..3) |stage| {
        const prefix = try std.fmt.allocPrint(a, "mtp.{d}.mlp.switch_mlp.", .{stage});
        try pushRouted(a, &second, prefix, &.{.{ .experts = 4, .n_gate = 32, .n_down = 32 }}, false);
    }
    if (fault.link) {
        try writeShardFile(a, dir, "real-shard.bin", &meta, second.items);
        var to_buf: [std.fs.max_path_bytes]u8 = undefined;
        var from_buf: [std.fs.max_path_bytes]u8 = undefined;
        try std.Io.Dir.symLinkAbsolute(io(), try std.fmt.bufPrint(&from_buf, "{s}/real-shard.bin", .{dir}), try std.fmt.bufPrint(&to_buf, "{s}/{s}", .{ dir, shard_b }), .{});
    } else try writeShardFile(a, dir, shard_b, &meta, second.items);

    var index: std.ArrayList(u8) = .empty;
    try index.appendSlice(a, "{\n \"metadata\": {},\n \"weight_map\": {\n");
    var n: usize = 0;
    for ([_]struct { []const u8, []const Tensor }{ .{ shard_a, first.items }, .{ shard_b, second.items } }) |entry| for (entry[1]) |t| {
        try index.print(a, "{s}  \"{s}\": \"{s}\"", .{ if (n > 0) ",\n" else "", t.key, entry[0] });
        n += 1;
    };
    try index.appendSlice(a, "\n }\n}\n");
    try writeFile(dir, "model.safetensors.index.json", index.items);
}

const Tmp = struct {
    tmp: testing.TmpDir,
    path: []u8,

    fn make() !Tmp {
        var tmp = testing.tmpDir(.{});
        errdefer tmp.cleanup();
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        const n = try tmp.dir.realPath(io(), &buf);
        return .{ .tmp = tmp, .path = try testing.allocator.dupe(u8, buf[0..n]) };
    }

    fn deinit(self: *Tmp) void {
        testing.allocator.free(self.path);
        self.tmp.cleanup();
    }
};

const Snapshot = struct {
    names: [][]u8,
    contents: [][]u8,

    fn take(a: std.mem.Allocator, dir: []const u8) !Snapshot {
        var d = try std.Io.Dir.openDirAbsolute(io(), dir, .{ .iterate = true });
        defer d.close(io());
        var names: std.ArrayList([]u8) = .empty;
        var it = d.iterate();
        while (try it.next(io())) |entry| try names.append(a, try a.dupe(u8, entry.name));
        std.mem.sort([]u8, names.items, {}, struct {
            fn lt(_: void, x: []u8, y: []u8) bool {
                return std.mem.lessThan(u8, x, y);
            }
        }.lt);
        const contents = try a.alloc([]u8, names.items.len);
        for (names.items, contents) |name, *c| c.* = readFile(a, dir, name) catch try a.dupe(u8, "<unreadable>");
        return .{ .names = names.items, .contents = contents };
    }

    fn expectEqual(self: Snapshot, other: Snapshot) !void {
        try testing.expectEqual(self.names.len, other.names.len);
        for (self.names, other.names, self.contents, other.contents) |n1, n2, c1, c2| {
            try testing.expectEqualStrings(n1, n2);
            try testing.expect(std.mem.eql(u8, c1, c2));
        }
    }
};

const Parsed = struct {
    value: std.json.Parsed(std.json.Value),
    data_start: usize,
    bytes: []const u8,

    fn open(a: std.mem.Allocator, bytes: []const u8) !Parsed {
        const len: usize = @intCast(std.mem.readInt(u64, bytes[0..8], .little));
        return .{ .value = try std.json.parseFromSlice(std.json.Value, a, bytes[8..][0..len], .{}), .data_start = 8 + len, .bytes = bytes };
    }

    fn tensor(self: Parsed, key: []const u8) ?[]const u8 {
        const t = (self.value.value.object.get(key) orelse return null).object;
        const offs = t.get("data_offsets").?.array.items;
        return self.bytes[self.data_start + @as(usize, @intCast(offs[0].integer)) ..][0..@intCast(offs[1].integer - offs[0].integer)];
    }

    fn offset(self: Parsed, key: []const u8) usize {
        return @intCast(self.value.value.object.get(key).?.object.get("data_offsets").?.array.items[0].integer);
    }
};

fn endsWith(s: []const u8, suffix: []const u8) bool {
    return std.mem.endsWith(u8, s, suffix);
}

/// `legacy` is the pack before the repack; `dir` after it. Checks every promise of the layout.
fn expectRepacked(a: std.mem.Allocator, legacy: Snapshot, dir: []const u8) !void {
    for ([_][]const u8{ shard_a, shard_b }) |shard| {
        const before_bytes = for (legacy.names, legacy.contents) |n, c| {
            if (std.mem.eql(u8, n, shard)) break c;
        } else return error.TestUnexpectedResult;
        const before = try Parsed.open(a, before_bytes);
        const after_bytes = try readFile(a, dir, shard);
        const after = try Parsed.open(a, after_bytes);
        try testing.expectEqual(@as(usize, 0), after.data_start % page);

        const meta = after.value.value.object.get("__metadata__").?.object;
        try testing.expectEqualStrings("pair_major", meta.get("tile_order").?.string);
        try testing.expectEqualStrings("__metadata__", after.value.value.object.keys()[0]);
        const before_meta = before.value.value.object.get("__metadata__").?.object;
        for (before_meta.keys(), before_meta.values()) |k, v| try testing.expectEqualStrings(v.string, meta.get(k).?.string);

        // The tensors tile the data section exactly, in offset order, with no gap.
        var end: usize = 0;
        var seen = std.StringHashMap(void).init(a);
        var in_order: std.ArrayList([]const u8) = .empty;
        const Ent = struct { key: []const u8, start: usize };
        var ents: std.ArrayList(Ent) = .empty;
        for (after.value.value.object.keys(), after.value.value.object.values()) |k, v| {
            if (std.mem.eql(u8, k, "__metadata__")) continue;
            try ents.append(a, .{ .key = k, .start = @intCast(v.object.get("data_offsets").?.array.items[0].integer) });
        }
        std.mem.sort(Ent, ents.items, {}, struct {
            fn lt(_: void, x: Ent, y: Ent) bool {
                return x.start < y.start;
            }
        }.lt);
        for (ents.items) |e| {
            const t = after.value.value.object.get(e.key).?.object;
            const offs = t.get("data_offsets").?.array.items;
            try testing.expectEqual(end, @as(usize, @intCast(offs[0].integer)));
            end = @intCast(offs[1].integer);
            try in_order.append(a, e.key);
            if (std.mem.startsWith(u8, e.key, "__pad__.")) {
                try testing.expect(std.mem.startsWith(u8, e.key, "__pad__." ++ "model-0000"));
                try testing.expectEqualStrings("U8", t.get("dtype").?.string);
                try testing.expectEqual(@as(usize, 1), t.get("shape").?.array.items.len);
                const bytes = after.tensor(e.key).?;
                try testing.expect(bytes.len > 0 and bytes.len < page);
                for (bytes) |b| try testing.expectEqual(@as(u8, 0), b);
            }
        }
        try testing.expectEqual(after_bytes.len, after.data_start + end);

        // Every legacy tensor is there: a trellis permuted per expert and on a page, the rest byte for byte.
        for (before.value.value.object.keys(), before.value.value.object.values()) |k, v| {
            if (std.mem.eql(u8, k, "__metadata__")) continue;
            const old = before.tensor(k).?;
            if (endsWith(k, ".trellis")) {
                const pm_key = try std.fmt.allocPrint(a, "{s}_pm", .{k});
                try seen.put(pm_key, {});
                const shape = v.object.get("shape").?.array.items;
                const experts: usize = @intCast(shape[0].integer);
                const it: usize = @intCast(shape[1].integer);
                const ot: usize = @intCast(shape[2].integer);
                const n: usize = @intCast(shape[3].integer);
                const new = after.tensor(pm_key) orelse return error.TestUnexpectedResult;
                try testing.expectEqual(@as(usize, 0), (after.data_start + after.offset(pm_key)) % page);
                try testing.expectEqual(old.len, new.len);
                const new_shape = after.value.value.object.get(pm_key).?.object.get("shape").?.array.items;
                try testing.expectEqual(@as(usize, 5), new_shape.len);
                try testing.expectEqual(@as(i64, @intCast(ot / 2)), new_shape[1].integer);
                const per = it * ot * n * 2;
                const want = try a.alloc(u8, per);
                for (0..experts) |e| {
                    exl3.group_layout.tileMajor(u8, want, old[e * per ..][0..per], it, ot, n * 2);
                    try testing.expect(std.mem.eql(u8, want, new[e * per ..][0..per]));
                }
            } else {
                try seen.put(k, {});
                try testing.expect(std.mem.eql(u8, old, after.tensor(k) orelse return error.TestUnexpectedResult));
                try testing.expectEqualStrings(v.object.get("dtype").?.string, after.value.value.object.get(k).?.object.get("dtype").?.string);
            }
        }
        for (in_order.items) |k| if (!std.mem.startsWith(u8, k, "__pad__.")) try testing.expect(seen.contains(k));
    }
    // A shard lists a routed projection as trellis_pm, suh, svh, group by group, gate then up then down.
    const first = try Parsed.open(a, try readFile(a, dir, shard_a));
    const want_order = [_][]const u8{
        "model.layers.0.mlp.switch_mlp.gate_proj.g0.trellis_pm", "model.layers.0.mlp.switch_mlp.gate_proj.g0.suh", "model.layers.0.mlp.switch_mlp.gate_proj.g0.svh",
        "model.layers.0.mlp.switch_mlp.gate_proj.g1.trellis_pm", "model.layers.0.mlp.switch_mlp.gate_proj.g1.suh", "model.layers.0.mlp.switch_mlp.gate_proj.g1.svh",
        "model.layers.0.mlp.switch_mlp.up_proj.g0.trellis_pm",   "model.layers.0.mlp.switch_mlp.up_proj.g0.suh",   "model.layers.0.mlp.switch_mlp.up_proj.g0.svh",
    };
    var last: usize = 0;
    for (want_order) |k| {
        const at = first.offset(k);
        try testing.expect(at >= last);
        last = at;
    }
    try testing.expect(first.offset("model.layers.0.mlp.switch_mlp.gate_proj.g0.suh") == first.offset("model.layers.0.mlp.switch_mlp.gate_proj.g0.trellis_pm") + first.tensor("model.layers.0.mlp.switch_mlp.gate_proj.g0.trellis_pm").?.len);

    // The index renames the trellis keys, lists the pads, and the config names the order.
    const index = try readFile(a, dir, "model.safetensors.index.json");
    try testing.expect(std.mem.indexOf(u8, index, ".trellis\"") == null);
    try testing.expect(std.mem.indexOf(u8, index, "mlp.switch_mlp.gate_proj.g0.trellis_pm\"") != null);
    try testing.expect(std.mem.indexOf(u8, index, "\"__pad__.model-00001-of-00002.0\"") != null);
    const cfg = try readFile(a, dir, "config.json");
    const parsed_cfg = try std.json.parseFromSlice(std.json.Value, a, cfg, .{});
    try testing.expectEqual(exl3.format.TileOrder.pair_major, (try exl3.parseExpertQuant(parsed_cfg.value.object)).tile_order);
    var left = try Snapshot.take(a, dir);
    for (left.names) |n| {
        try testing.expect(!endsWith(n, ".repack-tmp"));
        try testing.expect(!std.mem.eql(u8, n, ".sushi-repack.lock"));
    }
    left = left;
}

fn runOpts() repack.Options {
    return .{ .workers = 3 };
}

test "repack: a legacy pack becomes pair-major: permuted per expert on 16 KiB pages, pads zero, the rest verbatim, index and config renamed" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = try Tmp.make();
    defer tmp.deinit();
    try writeLegacyPack(a, tmp.path, .{});
    const legacy = try Snapshot.take(a, tmp.path);

    try testing.expectEqual(repack.Outcome.repacked, try repack.repackPack(testing.allocator, io(), tmp.path, runOpts()));
    try expectRepacked(a, legacy, tmp.path);

    // Only the shards, the index and the config changed.
    const after = try Snapshot.take(a, tmp.path);
    for (after.names, after.contents) |n, c| {
        const old = for (legacy.names, legacy.contents) |ln, lc| {
            if (std.mem.eql(u8, ln, n)) break lc;
        } else return error.TestUnexpectedResult;
        const changed = std.mem.eql(u8, n, shard_a) or std.mem.eql(u8, n, shard_b) or std.mem.eql(u8, n, "config.json") or std.mem.eql(u8, n, "model.safetensors.index.json");
        try testing.expectEqual(changed, !std.mem.eql(u8, old, c));
    }
    // The config differs by the one inserted member.
    const old_cfg = for (legacy.names, legacy.contents) |n, c| {
        if (std.mem.eql(u8, n, "config.json")) break c;
    } else unreachable;
    const new_cfg = try readFile(a, tmp.path, "config.json");
    const inserted = "\"tile_order\": \"pair_major\",";
    const at = std.mem.indexOf(u8, new_cfg, inserted).?;
    const without = try std.mem.concat(a, u8, &.{ new_cfg[0..at], new_cfg[at + inserted.len ..] });
    try testing.expectEqualStrings(old_cfg, without);
}

test "repack: a pair-major pack is reported and left alone, exit 0" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = try Tmp.make();
    defer tmp.deinit();
    try writeLegacyPack(a, tmp.path, .{});
    _ = try repack.repackPack(testing.allocator, io(), tmp.path, runOpts());
    const done = try Snapshot.take(a, tmp.path);
    try testing.expectEqual(repack.Outcome.already_pair_major, try repack.repackPack(testing.allocator, io(), tmp.path, runOpts()));
    try done.expectEqual(try Snapshot.take(a, tmp.path));
}

test "repack: the bytes do not depend on the worker count or the run" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var one = try Tmp.make();
    defer one.deinit();
    var two = try Tmp.make();
    defer two.deinit();
    seed_state = 12345;
    try writeLegacyPack(a, one.path, .{});
    seed_state = 12345;
    try writeLegacyPack(a, two.path, .{});
    _ = try repack.repackPack(testing.allocator, io(), one.path, .{ .workers = 1 });
    _ = try repack.repackPack(testing.allocator, io(), two.path, .{ .workers = 7 });
    try (try Snapshot.take(a, one.path)).expectEqual(try Snapshot.take(a, two.path));
}

test "repack: an interrupted run leaves a pack the loader refuses by name, and the rerun finishes the same bytes as an uninterrupted one" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var whole = try Tmp.make();
    defer whole.deinit();
    var cut = try Tmp.make();
    defer cut.deinit();
    seed_state = 777;
    try writeLegacyPack(a, whole.path, .{});
    seed_state = 777;
    try writeLegacyPack(a, cut.path, .{});
    const legacy = try Snapshot.take(a, cut.path);
    _ = try repack.repackPack(testing.allocator, io(), whole.path, runOpts());

    try testing.expectError(error.RepackInterrupted, repack.repackPack(testing.allocator, io(), cut.path, .{ .workers = 2, .fail_after_shards = 1 }));
    // Shard 1 is pair-major and the index and config still say the pack order.
    var config = try model.parseConfig(io(), testing.allocator, cut.path);
    defer config.deinit(testing.allocator);
    const refused = dsv41_source.validateTileOrder(io(), testing.allocator, cut.path, config.expert_tile_order, config.hidden_size, config.moe_intermediate_size);
    try testing.expect(refused == error.Exl3TileOrderMixed or refused == error.Exl3TileOrderMismatch);

    try testing.expectEqual(repack.Outcome.repacked, try repack.repackPack(testing.allocator, io(), cut.path, runOpts()));
    try expectRepacked(a, legacy, cut.path);
    try (try Snapshot.take(a, whole.path)).expectEqual(try Snapshot.take(a, cut.path));
}

test "repack: a stale tmp file and a dead process's lock do not stop a run" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = try Tmp.make();
    defer tmp.deinit();
    try writeLegacyPack(a, tmp.path, .{});
    try writeFile(tmp.path, shard_b ++ ".repack-tmp", "half a shard");
    try writeFile(tmp.path, ".sushi-repack.lock", "999999\n");
    try testing.expectEqual(repack.Outcome.repacked, try repack.repackPack(testing.allocator, io(), tmp.path, runOpts()));
}

const Refusal = struct { name: []const u8, fault: Fault, want: anyerror };

test "repack: every refusal is named and happens before a byte of the pack changes" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const cases = [_]Refusal{
        .{ .name = "other architecture", .fault = .{ .model_type = "qwen4_exp" }, .want = error.RepackArchitectureUnsupported },
        .{ .name = "not stq", .fault = .{ .format = "exl3" }, .want = error.Exl3TileOrderUnsupported },
        .{ .name = "unknown tile order", .fault = .{ .quant_extra = ", \"tile_order\": \"rowmajor\"" }, .want = error.Exl3TileOrderUnsupported },
        .{ .name = "stamp codebook", .fault = .{ .stamp_codebook = "mcg" }, .want = error.Exl3ShardStampMismatch },
        .{ .name = "stamp k wider", .fault = .{ .stamp_k = "8" }, .want = error.Exl3ShardStampMismatch },
        .{ .name = "legacy shard stamps an order", .fault = .{ .legacy_stamp = true }, .want = error.Exl3ShardStampMismatch },
        .{ .name = "shard mixes both names", .fault = .{ .mixed = true }, .want = error.Exl3TileOrderMixed },
        .{ .name = "shard is a link", .fault = .{ .link = true }, .want = error.RepackShardIsLink },
    };
    for (cases) |c| {
        var tmp = try Tmp.make();
        defer tmp.deinit();
        try writeLegacyPack(a, tmp.path, c.fault);
        const before = try Snapshot.take(a, tmp.path);
        testing.expectError(c.want, repack.repackPack(testing.allocator, io(), tmp.path, runOpts())) catch |e| {
            std.debug.print("case '{s}'\n", .{c.name});
            return e;
        };
        try before.expectEqual(try Snapshot.take(a, tmp.path));
    }
}

test "repack: a full disk, a live lock and an open shard are refused, and a missing pack is not downloaded" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = try Tmp.make();
    defer tmp.deinit();
    try writeLegacyPack(a, tmp.path, .{});
    const before = try Snapshot.take(a, tmp.path);

    try testing.expectError(error.RepackNoSpace, repack.repackPack(testing.allocator, io(), tmp.path, .{ .workers = 2, .free_bytes = 1 << 30 }));
    try before.expectEqual(try Snapshot.take(a, tmp.path));

    // The lock names this live process.
    const pid = try std.fmt.allocPrint(a, "{d}\n", .{std.c.getpid()});
    try writeFile(tmp.path, ".sushi-repack.lock", pid);
    try testing.expectError(error.RepackPackLocked, repack.repackPack(testing.allocator, io(), tmp.path, runOpts()));
    var d = try std.Io.Dir.openDirAbsolute(io(), tmp.path, .{});
    try d.deleteFile(io(), ".sushi-repack.lock");
    d.close(io());
    try before.expectEqual(try Snapshot.take(a, tmp.path));

    // A reader holds a shard open, as a running server does.
    const shard_path = try std.fmt.allocPrintSentinel(a, "{s}/{s}", .{ tmp.path, shard_b }, 0);
    const fd = std.c.open(shard_path.ptr, .{ .ACCMODE = .RDONLY });
    try testing.expect(fd >= 0);
    defer _ = std.c.close(fd);
    try testing.expectError(error.RepackPackInUse, repack.repackPack(testing.allocator, io(), tmp.path, .{ .workers = 2, .count_self = true }));
    try before.expectEqual(try Snapshot.take(a, tmp.path));

    var home = try Tmp.make();
    defer home.deinit();
    try testing.expectError(error.RepackNotDownloaded, repack.resolvePack(testing.allocator, io(), "deepseek-v4.1-flash", home.path));
    try testing.expectError(error.UnknownModel, repack.resolvePack(testing.allocator, io(), "no-such-pack", home.path));
    const abs = try repack.resolvePack(testing.allocator, io(), tmp.path, home.path);
    defer testing.allocator.free(abs);
    try testing.expectEqualStrings(tmp.path, abs);
}

/// sha256 of every file of the tiny STQ fixture after `sushi repack`, with the shard's pads: any change of the layout
/// or of a byte it writes shows here.
const tiny_hashes = [_]struct { []const u8, []const u8 }{
    .{ "config.json", "dc3165e23613fd28549c563eec0db6a7b32b283c574a7d8f6e6c6907dc83a31b" },
    .{ "engram_token_map.json", "ce694d5c7d7c923baab12ec1d6802c1c51fd2b080f75601205f07d9129708f35" },
    .{ "fixture.safetensors", "9db23659f056eef400bf8e0f65919d64facc08c26cd7872feeee031ba26fc27c" },
    .{ "fixture_noqat.safetensors", "4fd562da0f4386b657da40ccc8ea136665466c9a1ec693736e943bb454932b3d" },
    .{ "model-00001-of-00001.safetensors", "bb96e93d9dbd5c58bcc64fd02c285e3bcaad0f4068f13efa244820e7e94d8dc0" },
    .{ "model.safetensors.index.json", "10a376ecc004dde88de43a19ef95feebcc0146abbb2e4fc34d9b1e0e0ac1f147" },
    .{ "qat.safetensors", "276719ac2557bee970ac490b0655185862c26a513bc9120efda0a4bf31b231d4" },
};

fn hexSha(bytes: []const u8) [64]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}

test "repack: the tiny STQ fixture repacks to the committed bytes, and the fixture itself is not written (DSV41_STQ_TINY)" {
    const fixture = std.mem.span(std.c.getenv("DSV41_STQ_TINY") orelse return error.SkipZigTest);
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = try Tmp.make();
    defer tmp.deinit();
    var src = try std.Io.Dir.openDirAbsolute(io(), fixture, .{ .iterate = true });
    defer src.close(io());
    var it = src.iterate();
    while (try it.next(io())) |entry| {
        if (entry.kind != .file) continue;
        const bytes = try src.readFileAlloc(io(), entry.name, a, .limited(64 << 20));
        try writeFile(tmp.path, entry.name, bytes);
    }
    const fixture_before = try Snapshot.take(a, fixture);
    try testing.expectEqual(repack.Outcome.repacked, try repack.repackPack(testing.allocator, io(), tmp.path, runOpts()));
    try fixture_before.expectEqual(try Snapshot.take(a, fixture));

    const after = try Snapshot.take(a, tmp.path);
    var matched: usize = 0;
    for (after.names, after.contents) |name, bytes| {
        const hex = hexSha(bytes);
        for (tiny_hashes) |row| {
            if (!std.mem.eql(u8, row[0], name)) continue;
            matched += 1;
            testing.expectEqualStrings(row[1], &hex) catch |e| {
                std.debug.print("{s} sha256 {s}\n", .{ name, hex });
                return e;
            };
        }
    }
    try testing.expectEqual(after.names.len, matched);
}

