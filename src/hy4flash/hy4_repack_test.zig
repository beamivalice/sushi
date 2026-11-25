//! `sushi repack` on a generated Hy4-Flash STQ pack laid out as `sushi convert exl3` writes one: routed rate groups in
//! the layer shards, SPEC1 trunk and DFly2 shards whose dense trellis banks have no pair-major form. The routed experts
//! run at `routed_expert_hidden_size`, which is not the trunk's `hidden_size`.
const std = @import("std");
const testing = std.testing;
const exl3 = @import("sushi_exl3");
const repack = @import("../repack.zig");

const page: usize = 16384;
const hidden: u64 = 128;
const inter: u64 = 128;

fn io() std.Io {
    return std.Io.Threaded.global_single_threaded.io();
}

const config_text =
    \\{
    \\ "model_type": "hy_v4_flash",
    \\ "text_config": {"model_type": "hy_v4_flash", "hidden_size": 256, "moe_intermediate_size": 128, "routed_expert_hidden_size": 128, "n_routed_experts": 8},
    \\ "expert_quant": {"format": "stq", "k": 4, "codebook": "pair1", "window": 16, "codebook_table": "pair1-w16-v1", "codebook_digest": "c089b4eb", "encoder": "sushi-encoder-1"},
    \\ "trunk_quant": {"format": "stq", "k": 6, "codebook": "spec1", "window": 15, "codebook_table": "spec1-w15", "codebook_digest": "37f72431"}
    \\}
    \\
;

const Tensor = struct { key: []const u8, dtype: []const u8, shape: []const u64, bytes: []u8 };

var seed_state: u64 = 0x9e3779b97f4a7c15;

fn push(a: std.mem.Allocator, list: *std.ArrayList(Tensor), key: []const u8, dtype: []const u8, shape: []const u64, elem: u64) !void {
    var len: u64 = elem;
    for (shape) |d| len *= d;
    const bytes = try a.alloc(u8, @intCast(len));
    for (bytes) |*b| {
        seed_state ^= seed_state << 13;
        seed_state ^= seed_state >> 7;
        seed_state ^= seed_state << 17;
        b.* = @truncate(seed_state >> 24);
    }
    try list.append(a, .{ .key = try a.dupe(u8, key), .dtype = dtype, .shape = try a.dupe(u64, shape), .bytes = bytes });
}

/// One layer's routed banks in the encoder's order, every trellis, then every suh, then every svh. Each group holds
/// `experts` experts at its own gate and down widths (`n`, in u16 per 16x16 tile: 16 per bit of rate).
fn pushRouted(a: std.mem.Allocator, list: *std.ArrayList(Tensor), layer: usize, groups: []const struct { experts: u64, n_gate: u64, n_down: u64 }) !void {
    const projs = [_][]const u8{ "down_proj", "gate_proj", "up_proj" };
    var buf: [160]u8 = undefined;
    for (projs) |proj| for (groups, 0..) |g, gi| {
        const down = std.mem.eql(u8, proj, "down_proj");
        const key = try std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.switch_mlp.{s}.g{d}.trellis", .{ layer, proj, gi });
        try push(a, list, key, "U16", &.{ g.experts, (if (down) inter else hidden) / 16, (if (down) hidden else inter) / 16, if (down) g.n_down else g.n_gate }, 2);
    };
    for ([_][]const u8{ "suh", "svh" }) |leaf| for (projs) |proj| for (groups, 0..) |g, gi| {
        const down = std.mem.eql(u8, proj, "down_proj");
        const width = if (std.mem.eql(u8, leaf, "suh")) (if (down) inter else hidden) else (if (down) hidden else inter);
        const key = try std.fmt.bufPrint(&buf, "model.layers.{d}.mlp.switch_mlp.{s}.g{d}.{s}", .{ layer, proj, gi, leaf });
        try push(a, list, key, "F16", &.{ g.experts, width }, 2);
    };
}

fn writeFile(dir: []const u8, name: []const u8, data: []const u8) !void {
    var d = try std.Io.Dir.openDirAbsolute(io(), dir, .{});
    defer d.close(io());
    try d.writeFile(io(), .{ .sub_path = name, .data = data });
}

fn readFile(a: std.mem.Allocator, dir: []const u8, name: []const u8) ![]u8 {
    var d = try std.Io.Dir.openDirAbsolute(io(), dir, .{});
    defer d.close(io());
    return d.readFileAlloc(io(), name, a, .limited(64 << 20));
}

fn writeShardFile(a: std.mem.Allocator, dir: []const u8, name: []const u8, meta: []const [2][]const u8, tensors: []const Tensor) !void {
    var header: std.ArrayList(u8) = .empty;
    try header.appendSlice(a, "{\"__metadata__\":{");
    for (meta, 0..) |kv, i| try header.print(a, "{s}\"{s}\":\"{s}\"", .{ if (i > 0) "," else "", kv[0], kv[1] });
    try header.append(a, '}');
    var at: usize = 0;
    for (tensors) |t| {
        try header.print(a, ",\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ t.key, t.dtype });
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

const expert_stamp = [_][2][]const u8{ .{ "format", "stq" }, .{ "k", "4" }, .{ "codebook", "pair1" }, .{ "window", "16" }, .{ "encoder", "sushi-encoder-1" } };
const trunk_stamp = [_][2][]const u8{ .{ "format", "stq" }, .{ "k", "6" }, .{ "codebook", "spec1" }, .{ "window", "15" }, .{ "encoder", "sushi-encoder-1" } };

const layer_shards = [_][]const u8{ "model-layer-01.safetensors", "model-layer-02.safetensors" };
const plain_shards = [_][]const u8{ "model-globals.safetensors", "model-trunk-00.safetensors", "model-dfly.safetensors" };

/// Two layer shards (a one-group layer, a two-group layer), a trunk shard and the DFly2 shard with dense trellis
/// banks, the globals, and an index naming every tensor.
fn writePack(a: std.mem.Allocator, dir: []const u8) !void {
    try writeFile(dir, "config.json", config_text);
    var shards: [5]struct { name: []const u8, tensors: std.ArrayList(Tensor) } = undefined;
    for (&shards, [_][]const u8{ layer_shards[0], layer_shards[1], plain_shards[0], plain_shards[1], plain_shards[2] }) |*s, name| s.* = .{ .name = name, .tensors = .empty };
    try pushRouted(a, &shards[0].tensors, 1, &.{.{ .experts = 6, .n_gate = 64, .n_down = 64 }});
    try pushRouted(a, &shards[1].tensors, 2, &.{ .{ .experts = 4, .n_gate = 64, .n_down = 96 }, .{ .experts = 2, .n_gate = 96, .n_down = 64 } });
    try push(a, &shards[2].tensors, "model.embed_tokens.weight", "BF16", &.{ 16, 128 }, 2);
    try push(a, &shards[3].tensors, "model.layers.0.mlp.gate_proj.trellis", "U16", &.{ 8, 24, 96 }, 2);
    try push(a, &shards[3].tensors, "model.layers.0.mlp.gate_proj.suh", "F16", &.{128}, 2);
    try push(a, &shards[4].tensors, "model.dfly_layers.0.mlp.down_proj.trellis", "U16", &.{ 24, 8, 96 }, 2);
    try push(a, &shards[4].tensors, "model.dfly_layers.0.mlp.down_proj.svh", "F16", &.{128}, 2);
    var index: std.ArrayList(u8) = .empty;
    try index.appendSlice(a, "{\n \"metadata\": {},\n \"weight_map\": {\n");
    var n: usize = 0;
    for (shards) |s| {
        try writeShardFile(a, dir, s.name, if (s.tensors.items.len > 0 and std.mem.indexOf(u8, s.tensors.items[0].key, "switch_mlp") != null) &expert_stamp else &trunk_stamp, s.tensors.items);
        for (s.tensors.items) |t| {
            try index.print(a, "{s}  \"{s}\": \"{s}\"", .{ if (n > 0) ",\n" else "", t.key, s.name });
            n += 1;
        }
    }
    try index.appendSlice(a, "\n }\n}\n");
    try writeFile(dir, "model.safetensors.index.json", index.items);
}

const Parsed = struct {
    value: std.json.Parsed(std.json.Value),
    data_start: usize,
    bytes: []const u8,

    fn open(a: std.mem.Allocator, bytes: []const u8) !Parsed {
        const len: usize = @intCast(std.mem.readInt(u64, bytes[0..8], .little));
        return .{ .value = try std.json.parseFromSlice(std.json.Value, a, bytes[8..][0..len], .{}), .data_start = 8 + len, .bytes = bytes };
    }

    fn entry(self: Parsed, key: []const u8) ?std.json.ObjectMap {
        const v = self.value.value.object.get(key) orelse return null;
        return v.object;
    }

    fn offset(self: Parsed, key: []const u8) usize {
        return @intCast(self.entry(key).?.get("data_offsets").?.array.items[0].integer);
    }

    fn tensor(self: Parsed, key: []const u8) ?[]const u8 {
        const e = self.entry(key) orelse return null;
        const offs = e.get("data_offsets").?.array.items;
        return self.bytes[self.data_start + @as(usize, @intCast(offs[0].integer)) ..][0..@intCast(offs[1].integer - offs[0].integer)];
    }
};

fn untouched(a: std.mem.Allocator, before: []const u8, dir: []const u8, name: []const u8) !bool {
    return std.mem.eql(u8, before, try readFile(a, dir, name));
}

/// A routed shard after the repack against the one before it: every `trellis` is `trellis_pm`, each expert's tiles
/// permuted by `layout.tileMajor`, on a page; every other tensor equal byte for byte; the pads zero; nothing else there.
fn expectRepackedShard(a: std.mem.Allocator, before_bytes: []const u8, after_bytes: []const u8) !void {
    const before = try Parsed.open(a, before_bytes);
    const after = try Parsed.open(a, after_bytes);
    try testing.expectEqual(@as(usize, 0), after.data_start % page);
    try testing.expectEqualStrings("pair_major", after.entry("__metadata__").?.get("tile_order").?.string);
    var tensors: usize = 0;
    for (before.value.value.object.keys(), before.value.value.object.values()) |k, v| {
        if (std.mem.eql(u8, k, "__metadata__")) continue;
        tensors += 1;
        const old = before.tensor(k).?;
        if (!std.mem.endsWith(u8, k, ".trellis")) {
            try testing.expect(std.mem.eql(u8, old, after.tensor(k) orelse return error.TestUnexpectedResult));
            continue;
        }
        const pm_key = try std.fmt.allocPrint(a, "{s}_pm", .{k});
        const new = after.tensor(pm_key) orelse return error.TestUnexpectedResult;
        try testing.expect(after.value.value.object.get(k) == null);
        try testing.expectEqual(@as(usize, 0), (after.data_start + after.offset(pm_key)) % page);
        const shape = v.object.get("shape").?.array.items;
        const experts: usize = @intCast(shape[0].integer);
        const it: usize = @intCast(shape[1].integer);
        const ot: usize = @intCast(shape[2].integer);
        const n: usize = @intCast(shape[3].integer);
        const per = it * ot * n * 2;
        try testing.expectEqual(experts * per, new.len);
        const want = try a.alloc(u8, per);
        for (0..experts) |e| {
            exl3.group_layout.tileMajor(u8, want, old[e * per ..][0..per], it, ot, n * 2);
            try testing.expect(std.mem.eql(u8, want, new[e * per ..][0..per]));
        }
    }
    for (after.value.value.object.keys()) |k| {
        if (!std.mem.startsWith(u8, k, exl3.group_layout.pad_prefix)) continue;
        tensors += 1;
        for (after.tensor(k).?) |b| try testing.expectEqual(@as(u8, 0), b);
    }
    try testing.expectEqual(after.value.value.object.count() - 1, tensors);
}

test "hy4 repack: a pack-order STQ pack becomes pair-major: its routed trellis permuted, every other tensor and shard as it was" {
    if (@import("../arch/registry.zig").ofModelType("hy_v4_flash") == null) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const dir = path_buf[0..try tmp.dir.realPath(io(), &path_buf)];
    try writePack(a, dir);
    var before: [5][]u8 = undefined;
    for (&before, layer_shards ++ plain_shards) |*b, name| b.* = try readFile(a, dir, name);
    const old_config = try readFile(a, dir, "config.json");

    const outcome = repack.repackPack(testing.allocator, io(), dir, .{ .workers = 2 }) catch |err| {
        std.debug.print("repackPack: {s}\n", .{@errorName(err)});
        return err;
    };
    try testing.expectEqual(repack.Outcome.repacked, outcome);

    for (layer_shards, before[0..2]) |name, old| try expectRepackedShard(a, old, try readFile(a, dir, name));
    for (plain_shards, before[2..]) |name, old| try testing.expect(try untouched(a, old, dir, name));
    // The index renames the routed keys only: the DFly2 and trunk banks keep their dense `trellis` names.
    const index = try readFile(a, dir, "model.safetensors.index.json");
    try testing.expect(std.mem.indexOf(u8, index, "switch_mlp.gate_proj.g1.trellis_pm\"") != null);
    try testing.expect(std.mem.indexOf(u8, index, "switch_mlp.gate_proj.g0.trellis\"") == null);
    try testing.expect(std.mem.indexOf(u8, index, "model.dfly_layers.0.mlp.down_proj.trellis\"") != null);
    try testing.expect(std.mem.indexOf(u8, index, "model.layers.0.mlp.gate_proj.trellis\"") != null);
    // The config differs by the one inserted member.
    const new_config = try readFile(a, dir, "config.json");
    const inserted = "\"tile_order\": \"pair_major\",";
    const at = std.mem.indexOf(u8, new_config, inserted).?;
    try testing.expectEqualStrings(old_config, try std.mem.concat(a, u8, &.{ new_config[0..at], new_config[at + inserted.len ..] }));

    try testing.expectEqual(repack.Outcome.already_pair_major, try repack.repackPack(testing.allocator, io(), dir, .{ .workers = 2 }));
}
