//! deepseek_v41 weights from the release's own shards: the resident trunk is read by name, filtered
//! at the safetensors header so a tensor the engine does not hold is never read. Routed experts,
//! and the Engram tables are not part of it; the DSpark stages (`mtp.*`, experts included) and the vision tower
//! (with the trunk's image routing biases) load only when asked for.
const std = @import("std");
const mlx = @import("mlx.zig");
const model = @import("model.zig");
const expert_io = @import("expert_io.zig");
const log = @import("log.zig");

pub const Parts = struct {
    /// Routed experts as stacked-at-load banks (a checkpoint small enough to hold them); streamed ones are not read here.
    experts: bool = false,
    /// The DSpark drafter's stages with their own routed experts, always resident.
    dspark: bool = false,
    /// The vision tower, the aligner, the span delimiters and each backbone layer's image routing bias.
    vision: bool = false,
};

/// A tensor only an image request reads.
pub fn isVisionKey(key: []const u8) bool {
    if (std.mem.startsWith(u8, key, "mtp.")) return false;
    for ([_][]const u8{ "vision.", "aligner.", "image_" }) |p| if (std.mem.startsWith(u8, key, p)) return true;
    return std.mem.endsWith(u8, key, ".bias_vl");
}

/// Does the resident load keep `key`?
pub fn keeps(key: []const u8, parts: Parts) bool {
    // A DSpark draft is text: its stages' image routing biases are never read.
    if (std.mem.startsWith(u8, key, "mtp.")) return parts.dspark and !std.mem.endsWith(u8, key, ".bias_vl");
    if (isVisionKey(key)) return parts.vision;
    if (std.mem.indexOf(u8, key, ".engram.embed.") != null) return false;
    if (std.mem.indexOf(u8, key, ".ffn.experts.") != null) return parts.experts;
    // An STQ pack's routed rate groups, which stream.
    if (std.mem.startsWith(u8, key, "model.layers.") and std.mem.indexOf(u8, key, ".mlp.switch_mlp.") != null) return parts.experts;
    return true;
}

/// The checkpoint's weight shards in name order; the caller frees each name and the list.
fn shardNames(io: std.Io, allocator: std.mem.Allocator, dir_path: []const u8) !std.ArrayList([]const u8) {
    var dir = try std.Io.Dir.openDirAbsolute(io, dir_path, .{ .iterate = true });
    defer dir.close(io);
    var shards: std.ArrayList([]const u8) = .empty;
    errdefer freeShardNames(allocator, &shards);
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .file and entry.kind != .sym_link) continue;
        if (!std.mem.startsWith(u8, entry.name, "model") or !std.mem.endsWith(u8, entry.name, ".safetensors")) continue;
        try shards.append(allocator, try allocator.dupe(u8, entry.name));
    }
    std.mem.sort([]const u8, shards.items, {}, struct {
        fn lt(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.lt);
    if (shards.items.len == 0) return error.NoWeightFiles;
    return shards;
}

fn freeShardNames(allocator: std.mem.Allocator, shards: *std.ArrayList([]const u8)) void {
    for (shards.items) |f| allocator.free(f);
    shards.deinit(allocator);
}

/// A vision-tower tensor (not a routing bias): what a tower test loads without the trunk.
pub fn isTowerKey(key: []const u8) bool {
    return isVisionKey(key) and !std.mem.endsWith(u8, key, ".bias_vl");
}

/// Every tensor of the checkpoint at `dir_path` that `keeps`, by its release name.
pub fn loadResident(io: std.Io, allocator: std.mem.Allocator, dir_path: []const u8, parts: Parts) !model.Weights {
    return loadWhere(io, allocator, dir_path, parts, keeps);
}

/// Only the vision tower, aligner and span delimiters of the checkpoint at `dir_path`.
pub fn loadTower(io: std.Io, allocator: std.mem.Allocator, dir_path: []const u8) !model.Weights {
    return loadWhere(io, allocator, dir_path, {}, struct {
        fn f(key: []const u8, _: void) bool {
            return isTowerKey(key);
        }
    }.f);
}

fn loadWhere(io: std.Io, allocator: std.mem.Allocator, dir_path: []const u8, ctx: anytype, comptime keep: anytype) !model.Weights {
    var weights = model.Weights.init(allocator);
    errdefer weights.deinit();
    var reader = expert_io.ParallelReader.init(allocator, expert_io.ParallelReader.default_chunk, expert_io.ParallelReader.default_workers);
    defer reader.deinit();
    var shards = try shardNames(io, allocator, dir_path);
    defer freeShardNames(allocator, &shards);

    for (shards.items) |name| {
        const path = try std.fmt.allocPrintSentinel(allocator, "{s}/{s}", .{ dir_path, name }, 0);
        defer allocator.free(path);
        const fd = try expert_io.openHinted(path, .{ .nocache = true, .readahead_off = false });
        defer _ = std.c.close(fd);
        var header = try expert_io.SafetensorsHeader.read(allocator, fd);
        defer header.deinit();
        var tensors = header.parsed.value.object.iterator();
        while (tensors.next()) |entry| {
            const key = entry.key_ptr.*;
            if (std.mem.eql(u8, key, "__metadata__") or !keep(key, ctx)) continue;
            const value = try model.readSafetensor(&reader, fd, &header, entry.value_ptr.*);
            var value_owned = true;
            defer if (value_owned) {
                _ = mlx.mlx_array_free(value);
            };
            const owned = try allocator.dupe(u8, key);
            var key_owned = true;
            defer if (key_owned) allocator.free(owned);
            const gop = try weights.map.getOrPut(owned);
            if (gop.found_existing) return error.DuplicateTensor;
            gop.value_ptr.* = value;
            value_owned = false;
            key_owned = false;
        }
    }
    log.info("[dsv41] loaded {d} resident tensors from {d} shards\n", .{ weights.count(), shards.items.len });
    return weights;
}

/// Bytes the resident load holds once built: the stored tensors, the head widened to f32, and each
/// fp8 weight's 32x32 scale grid spread over its rows.
pub fn residentBytes(io: std.Io, allocator: std.mem.Allocator, dir_path: []const u8) !u64 {
    return sumBytes(io, allocator, dir_path, .trunk);
}

/// Bytes the vision tower adds once loaded: stored as built, bf16.
pub fn visionBytes(io: std.Io, allocator: std.mem.Allocator, dir_path: []const u8) !u64 {
    return sumBytes(io, allocator, dir_path, .vision);
}

/// Does the checkpoint ship the DSpark stages? Its weight index names `mtp.0.main_proj` (stored, or a trellis bank).
pub fn shipsDspark(io: std.Io, allocator: std.mem.Allocator, dir_path: []const u8) bool {
    var dir = std.Io.Dir.openDirAbsolute(io, dir_path, .{}) catch return false;
    defer dir.close(io);
    const raw = dir.readFileAlloc(io, "model.safetensors.index.json", allocator, .limited(64 << 20)) catch return false;
    defer allocator.free(raw);
    return std.mem.indexOf(u8, raw, "\"mtp.0.main_proj.weight\"") != null or std.mem.indexOf(u8, raw, "\"mtp.0.main_proj.trellis\"") != null;
}

/// Bytes the DSpark stages add once built: their tensors and experts, with the Markov head widened to f32.
pub fn dsparkBytes(io: std.Io, allocator: std.mem.Allocator, dir_path: []const u8) !u64 {
    return sumBytes(io, allocator, dir_path, .stages);
}

const Section = enum { trunk, stages, vision };

/// `keeps`' bytes as built, over the trunk, only the DSpark stages, or only the vision tower.
fn sumBytes(io: std.Io, allocator: std.mem.Allocator, dir_path: []const u8, section: Section) !u64 {
    var shards = try shardNames(io, allocator, dir_path);
    defer freeShardNames(allocator, &shards);
    var total: u64 = 0;
    for (shards.items) |name| {
        const path = try std.fmt.allocPrintSentinel(allocator, "{s}/{s}", .{ dir_path, name }, 0);
        defer allocator.free(path);
        const fd = try expert_io.openHinted(path, .{ .nocache = true, .readahead_off = false });
        defer _ = std.c.close(fd);
        var header = try expert_io.SafetensorsHeader.read(allocator, fd);
        defer header.deinit();
        var tensors = header.parsed.value.object.iterator();
        while (tensors.next()) |entry| {
            const key = entry.key_ptr.*;
            if (std.mem.eql(u8, key, "__metadata__") or !keeps(key, .{ .dspark = true, .vision = true })) continue;
            const mine: Section = if (isVisionKey(key)) .vision else if (std.mem.startsWith(u8, key, "mtp.")) .stages else .trunk;
            if (mine != section) continue;
            const offs = (entry.value_ptr.object.get("data_offsets") orelse return error.InvalidSafetensorsHeader).array.items;
            if (offs.len != 2 or offs[0] != .integer or offs[1] != .integer) return error.InvalidSafetensorsHeader;
            const bytes: u64 = @intCast(offs[1].integer - offs[0].integer);
            total +|= bytes *| widening(key);
        }
    }
    return total;
}

/// How many times the built tensor is as large as the stored one: an fp8 weight's block scale is repeated down its
/// 32 rows, and the bf16 tensors the forward holds in f32 (the heads, the routers, the confidence head) double; an
/// expert's per-row scales are stored as built.
fn widening(key: []const u8) u64 {
    if (std.mem.indexOf(u8, key, ".ffn.experts.") != null) return 1;
    if (std.mem.endsWith(u8, key, ".scale")) return 32;
    if (std.mem.eql(u8, key, "head.weight")) return 2;
    for ([_][]const u8{ ".markov_head.head.weight", ".ffn.gate.weight", ".confidence_head.proj.weight" }) |tail|
        if (std.mem.endsWith(u8, key, tail)) return 2;
    return 1;
}

/// Removes `name` from the map and frees its handle (an array views of it keep their buffer).
pub fn drop(w: *model.Weights, name: []const u8) void {
    if (w.map.fetchRemove(name)) |kv| {
        _ = mlx.mlx_array_free(kv.value);
        w.allocator.free(kv.key);
    }
}

const testing = std.testing;

test "dsv41 source: the resident filter keeps the trunk and drops experts, Engram rows, stages and the tower" {
    const keep = [_][]const u8{
        "embed.weight",                     "head.weight",                  "norm.weight",
        "layers.3.attn.wq_a.weight",        "layers.3.attn.wq_a.scale",     "layers.3.ffn.gate.weight",
        "layers.3.ffn.gate.bias",           "layers.3.ffn.shared_experts.w1.weight", "layers.1.hc_attn_fn",
        "layers.1.engram.wkv.weight",       "layers.1.engram.q_weight",     "layers.2.attn.compressor.wkv.weight",
    };
    for (keep) |k| try testing.expect(keeps(k, .{}));
    for ([_][]const u8{ "layers.3.attn.wq_a.trellis", "head.svh", "embed.scales", "mtp.0.mlp.switch_mlp.up_proj.g1.suh" }) |k|
        try testing.expect(keeps(k, .{ .dspark = true }));
    try testing.expect(!keeps("model.layers.3.mlp.switch_mlp.gate_proj.g2.trellis", .{ .dspark = true }));
    const drop_keys = [_][]const u8{
        "layers.3.ffn.experts.0.w1.weight", "layers.3.ffn.experts.383.w2.scale", "layers.1.engram.embed.weight",
        "layers.1.engram.embed.scale",      "mtp.0.attn.wq_a.weight",       "vision.blocks.0.attn.wo.weight",
        "aligner.w1.weight",                "image_start",                  "layers.3.ffn.gate.bias_vl",
    };
    for (drop_keys) |k| try testing.expect(!keeps(k, .{}));
    for ([_][]const u8{ "vision.blocks.0.attn.wo.weight", "aligner.w1.weight", "image_start", "image_newline", "layers.3.ffn.gate.bias_vl" }) |k| {
        try testing.expect(isVisionKey(k) and keeps(k, .{ .vision = true }));
        try testing.expect(!keeps(k, .{ .dspark = true, .experts = true }));
    }
    try testing.expect(!isVisionKey("layers.3.ffn.gate.bias") and !isVisionKey("embed.weight") and !isVisionKey("mtp.0.ffn.gate.bias_vl"));
    try testing.expect(keeps("layers.3.ffn.experts.0.w1.weight", .{ .experts = true }));
    try testing.expect(!keeps("layers.1.engram.embed.weight", .{ .experts = true }));
}

test "dsv41 source: DSpark loads its stages and their experts when asked and never the vision bias" {
    const stage = [_][]const u8{
        "mtp.0.attn.wq_a.weight",         "mtp.0.ffn.experts.127.w2.scale", "mtp.0.main_proj.weight",
        "mtp.2.markov_head.head.weight",  "mtp.2.confidence_head.proj.weight", "mtp.1.hc_attn_fn",
    };
    for (stage) |k| {
        try testing.expect(!keeps(k, .{}));
        try testing.expect(!keeps(k, .{ .experts = true }));
        try testing.expect(keeps(k, .{ .dspark = true }));
    }
    try testing.expect(!keeps("mtp.0.ffn.gate.bias_vl", .{ .dspark = true }));
    try testing.expect(!keeps("layers.1.engram.embed.weight", .{ .dspark = true }));
}

test "dsv41 source: a pack whose main projection is a trellis bank ships DSpark too" {
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = path_buf[0..try tmp.dir.realPath(io, &path_buf)];
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"mtp.0.main_proj.trellis\":\"model-a.safetensors\"}}" });
    try testing.expect(shipsDspark(io, testing.allocator, root));
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"mtp.0.attn.wq_a.trellis\":\"model-a.safetensors\"}}" });
    try testing.expect(!shipsDspark(io, testing.allocator, root));
}

/// A shard holding tensors of the given byte lengths (names -> bytes), as a safetensors file.
fn writeShard(dir: std.Io.Dir, name: []const u8, tensors: []const struct { []const u8, usize }) !void {
    var header: std.ArrayList(u8) = .empty;
    defer header.deinit(testing.allocator);
    var total: usize = 0;
    try header.append(testing.allocator, '{');
    for (tensors, 0..) |t, i| {
        if (i > 0) try header.append(testing.allocator, ',');
        const entry = try std.fmt.allocPrint(testing.allocator, "\"{s}\":{{\"dtype\":\"U8\",\"shape\":[{d}],\"data_offsets\":[{d},{d}]}}", .{ t[0], t[1], total, total + t[1] });
        defer testing.allocator.free(entry);
        try header.appendSlice(testing.allocator, entry);
        total += t[1];
    }
    try header.append(testing.allocator, '}');
    const file = try testing.allocator.alloc(u8, 8 + header.items.len + total);
    defer testing.allocator.free(file);
    std.mem.writeInt(u64, file[0..8], header.items.len, .little);
    @memcpy(file[8..][0..header.items.len], header.items);
    @memset(file[8 + header.items.len ..], 0);
    try dir.writeFile(testing.io, .{ .sub_path = name, .data = file });
}

test "dsv41 source: the tower's bytes are the vision section, the trunk's exclude them (the ssd budget's vision term)" {
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try writeShard(tmp.dir, "model-globals.safetensors", &.{
        .{ "embed.weight", 1000 },        .{ "layers.3.ffn.gate.bias", 40 },    .{ "layers.3.ffn.gate.bias_vl", 40 },
        .{ "vision.blocks.0.attn.wo.weight", 300 }, .{ "aligner.w1.weight", 500 }, .{ "image_start", 20 },
        .{ "mtp.0.attn.wq_a.weight", 7000 }, .{ "mtp.0.ffn.gate.bias_vl", 40 },
    });
    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const root = path_buf[0..try tmp.dir.realPath(testing.io, &path_buf)];
    try testing.expectEqual(@as(u64, 40 + 300 + 500 + 20), try visionBytes(testing.io, testing.allocator, root));
    try testing.expectEqual(@as(u64, 1040), try residentBytes(testing.io, testing.allocator, root));
    try testing.expectEqual(@as(u64, 7000), try dsparkBytes(testing.io, testing.allocator, root));
    var tower = try loadTower(testing.io, testing.allocator, root);
    defer tower.deinit();
    try testing.expectEqual(@as(usize, 3), tower.count());
    try testing.expect(tower.get("image_start") != null and tower.get("layers.3.ffn.gate.bias_vl") == null);
    var all = try loadResident(testing.io, testing.allocator, root, .{ .vision = true });
    defer all.deinit();
    try testing.expect(all.get("layers.3.ffn.gate.bias_vl") != null and all.get("mtp.0.ffn.gate.bias_vl") == null and all.get("mtp.0.attn.wq_a.weight") == null);
    var text = try loadResident(testing.io, testing.allocator, root, .{});
    defer text.deinit();
    try testing.expect(text.get("vision.blocks.0.attn.wo.weight") == null and text.get("layers.3.ffn.gate.bias") != null);
}
