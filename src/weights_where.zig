//! Reads the safetensors shards of a checkpoint directory by name, keeping only the tensors a predicate selects. A tensor
//! over `model.import_min_bytes` is imported without a copy, so its bytes stay outside MLX's allocator.
const std = @import("std");
const mlx = @import("mlx.zig");
const model = @import("model.zig");
const expert_io = @import("expert_io.zig");
const log = @import("log.zig");
const load_clock = @import("load_clock.zig");

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
        try shards.ensureUnusedCapacity(allocator, 1);
        shards.appendAssumeCapacity(try allocator.dupe(u8, entry.name));
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
/// Collects all wanted tensors across shards, sorts by `(shard, file_offset)`, and reads all payloads in parallel
/// directly into their final `MappedRegion`/host buffers via `expert_io.preadBatch` (no bounce copy, no per-tensor
/// thread spawn/join).
pub fn loadWhere(comptime tag: []const u8, io: std.Io, allocator: std.mem.Allocator, dir_path: []const u8, ctx: anytype, comptime keep: anytype) !model.Weights {
    var weights = model.Weights.init(allocator);
    errdefer weights.deinit();
    var shards = try shardNames(io, allocator, dir_path);
    defer freeShardNames(allocator, &shards);

    const Pending = struct {
        shard_idx: u32,
        key: []u8,
        meta: model.SafetensorMeta,
        region: ?*model.MappedRegion = null,
        host: ?[]u8 = null,
    };

    var open_fds: std.ArrayList(std.c.fd_t) = .empty;
    defer {
        for (open_fds.items) |fd| _ = std.c.close(fd);
        open_fds.deinit(allocator);
    }
    var pending: std.ArrayList(Pending) = .empty;
    defer {
        for (pending.items) |*p| {
            allocator.free(p.key);
            if (p.region) |r| model.destroyRegion(r);
            if (p.host) |h| allocator.free(h);
        }
        pending.deinit(allocator);
    }

    for (shards.items) |name| {
        const path = try std.fmt.allocPrintSentinel(allocator, "{s}/{s}", .{ dir_path, name }, 0);
        defer allocator.free(path);
        const fd = try expert_io.openHinted(path, .{ .nocache = true, .readahead_off = false });
        errdefer _ = std.c.close(fd);
        const header_t0 = load_clock.now();
        var header = try expert_io.SafetensorsHeader.read(allocator, fd);
        defer header.deinit();
        load_clock.headersDone(header_t0);
        var kept_in_shard = false;
        const shard_idx: u32 = @intCast(open_fds.items.len);
        var tensors = header.parsed.value.object.iterator();
        while (tensors.next()) |entry| {
            const key = entry.key_ptr.*;
            if (std.mem.eql(u8, key, "__metadata__") or !keep(key, ctx)) continue;
            const meta = try model.parseSafetensorMeta(&header, entry.value_ptr.*);
            const key_owned = try allocator.dupe(u8, key);
            errdefer allocator.free(key_owned);
            try pending.append(allocator, .{ .shard_idx = shard_idx, .key = key_owned, .meta = meta });
            kept_in_shard = true;
        }
        if (kept_in_shard) {
            try open_fds.append(allocator, fd);
        } else {
            _ = std.c.close(fd);
        }
    }

    // Sort by `(shard_idx, file_offset)` so every shard is read strictly sequentially.
    std.mem.sort(Pending, pending.items, {}, struct {
        fn lt(_: void, a: Pending, b: Pending) bool {
            if (a.shard_idx != b.shard_idx) return a.shard_idx < b.shard_idx;
            return a.meta.offset < b.meta.offset;
        }
    }.lt);

    const floor = model.importFloor(mlx.engine.own_weights);
    var reads: std.ArrayList(expert_io.BatchRead) = .empty;
    defer reads.deinit(allocator);
    try reads.ensureTotalCapacity(allocator, pending.items.len);
    var total_bytes: u64 = 0;

    for (pending.items) |*p| {
        if (p.meta.bytes == 0 or p.meta.bytes < floor) {
            const host = try allocator.alloc(u8, @max(p.meta.bytes, 1));
            p.host = host;
            if (p.meta.bytes > 0) {
                reads.appendAssumeCapacity(.{ .fd = open_fds.items[p.shard_idx], .offset = p.meta.offset, .dst = host[0..p.meta.bytes] });
            }
        } else {
            const region = try model.ownedRegion(p.meta.bytes);
            p.region = region;
            reads.appendAssumeCapacity(.{ .fd = open_fds.items[p.shard_idx], .offset = p.meta.offset, .dst = region.ptr[0..p.meta.bytes] });
        }
        total_bytes += p.meta.bytes;
    }

    const read_t0 = load_clock.now();
    try expert_io.preadBatch(allocator, reads.items, 8);
    load_clock.readDone(read_t0, total_bytes);

    for (pending.items) |*p| {
        const read = if (p.region) |region| blk: {
            p.region = null;
            break :blk try model.wrapFilledRegion(region, p.meta.bytes, p.meta.shape[0..p.meta.ndim], p.meta.dtype);
        } else blk: {
            const host = p.host.?;
            break :blk try model.wrapSmallHostBytes(host, p.meta.bytes, p.meta.shape[0..p.meta.ndim], p.meta.dtype);
        };
        const value = read.array;
        var value_owned = true;
        defer if (value_owned) {
            _ = mlx.mlx_array_free(value);
        };
        // A copied import would put the weight back in MLX's allocator, outside the ledger.
        if (read.bytes.len >= model.import_min_bytes and read.mapped == null) {
            log.err("[" ++ tag ++ "-mem] {s}: MLX refused the no-copy import of its {d} bytes\n", .{ p.key, read.bytes.len });
            return error.OwnedImportCopied;
        }
        const owned = try allocator.dupe(u8, p.key);
        var key_owned = true;
        defer if (key_owned) allocator.free(owned);
        const gop = try weights.map.getOrPut(owned);
        if (gop.found_existing) return error.DuplicateTensor;
        gop.value_ptr.* = value;
        value_owned = false;
        key_owned = false;
    }

    log.info("[" ++ tag ++ "] loaded {d} resident tensors from {d} shards\n", .{ weights.count(), shards.items.len });
    return weights;
}

