//! Reads the safetensors shards of a checkpoint directory by name, keeping only the tensors a predicate selects. A tensor
//! over `model.import_min_bytes` is imported without a copy, so its bytes stay outside MLX's allocator.
const std = @import("std");
const mlx = @import("mlx.zig");
const model = @import("model.zig");
const expert_io = @import("expert_io.zig");
const log = @import("log.zig");

/// The checkpoint's weight shards in name order; the caller frees each name and the list.
pub fn shardNames(io: std.Io, allocator: std.mem.Allocator, dir_path: []const u8) !std.ArrayList([]const u8) {
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

pub fn freeShardNames(allocator: std.mem.Allocator, shards: *std.ArrayList([]const u8)) void {
    for (shards.items) |f| allocator.free(f);
    shards.deinit(allocator);
}

/// Every tensor of the shards at `dir_path` that `keep(key, ctx)` selects, logged under `tag`.
pub fn loadWhere(comptime tag: []const u8, io: std.Io, allocator: std.mem.Allocator, dir_path: []const u8, ctx: anytype, comptime keep: anytype) !model.Weights {
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
            const read = try model.readSafetensorArray(&reader, fd, &header, entry.value_ptr.*);
            const value = read.array;
            var value_owned = true;
            defer if (value_owned) {
                _ = mlx.mlx_array_free(value);
            };
            // A copied import would put the weight back in MLX's allocator, outside the ledger.
            if (read.bytes.len >= model.import_min_bytes and read.mapped == null) {
                log.err("[" ++ tag ++ "-mem] {s}: MLX refused the no-copy import of its {d} bytes\n", .{ key, read.bytes.len });
                return error.OwnedImportCopied;
            }
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
    log.info("[" ++ tag ++ "] loaded {d} resident tensors from {d} shards\n", .{ weights.count(), shards.items.len });
    return weights;
}

