//! A conversation's live restore points as the SSD tier sees them: the points an arch's live record keeps of the state
//! it parks (each a position and the tier entry it was written to), which of them are only in RAM, and the writes that
//! take them there. What a point holds is the arch's; a point here is any struct with `pos: usize` and
//! `disk_id: ?u64`.
const std = @import("std");
const mlx = @import("mlx.zig");
const trunk = @import("trunk.zig");
const kv_quant = @import("kv_quant.zig");
const kv_disk_cache = @import("kv_disk_cache.zig");
const pc = @import("prefix_cache.zig");

const DiskTier = kv_disk_cache.DiskTier;

/// Whether the tier holds the point at `pos` where it was put (`disk_id`): its budget, a cache clear or a poisoned read
/// remove entries.
pub fn onDisk(disk_id: ?u64, pos: usize, disk: ?*const DiskTier) bool {
    const id = disk_id orelse return false;
    return (disk orelse return false).holdsPoint(id, pos);
}

/// Records the tier entry `id` on each point of `points` it now holds.
pub fn markOnDisk(points: anytype, disk: *const DiskTier, id: u64) void {
    for (points) |*p| {
        if (disk.holdsPoint(id, p.pos)) p.disk_id = id;
    }
}

/// Whether a point of `points` is only in RAM, not on `disk`.
pub fn anyUnflushed(points: anytype, disk: ?*const DiskTier) bool {
    for (points) |*p| if (!onDisk(p.disk_id, p.pos, disk)) return true;
    return false;
}

/// Releases the stored points of `cps` off the tier's row-segment grid: their rows past the last segment would be
/// written again by the next point, so they stay the live record's only. True when a point is left to write.
pub fn releaseOffGrid(disk: *const DiskTier, cps: []trunk.SSMCheckpoint) bool {
    const span: usize = if (disk.grown) |g| g.span else 0;
    var pending = false;
    for (cps) |*cp| {
        if (!cp.isReleased() and span != 0 and cp.pos % span != 0) cp.release();
        pending = pending or !cp.isReleased();
    }
    return pending;
}

/// What one write took to the tier: the bytes its points and segments staged, and the entry it stored.
pub const Written = struct { bytes: u64, id: ?u64 };

/// `cps` written over `tokens` as the cache's write-through writes a prefill's (`writeThroughHybrid`), the rows below
/// `base` the restore's or rollback's, which the tier links from the entry they came from (`DiskTier.rows_base`).
pub fn write(
    hc: *pc.HotPrefixCache,
    kv_entries: []const trunk.KVCacheEntry,
    config: kv_quant.KVQuantConfig,
    tokens: []const u32,
    has_tools: bool,
    media: pc.MediaStamp,
    media_chain: []const pc.MediaSpan,
    restored: ?kv_disk_cache.RestoredRows,
    cps: []trunk.SSMCheckpoint,
    base: usize,
    s: mlx.mlx_stream,
) Written {
    const d = &hc.disk.?;
    const before = d.point_bytes_written;
    d.rows_base = @intCast(@min(base, std.math.maxInt(u32)));
    defer d.rows_base = std.math.maxInt(u32);
    hc.commit_media_chain = media_chain;
    hc.writeThroughHybrid(kv_entries, config, tokens, has_tools, media, restored, cps, &.{}, std.math.maxInt(u64), s);
    return .{ .bytes = d.point_bytes_written - before, .id = d.last_store_id };
}

/// A live record's text and media, as a flush labels the points it writes.
pub const Record = struct {
    tokens: []const u32,
    has_tools: bool,
    vision_key: u64,
    media_start: ?usize,
    media_chain: []const pc.MediaSpan = &.{},
    restored: ?kv_disk_cache.RestoredRows = null,
    /// Rows below this position are a restore's or a rollback's (`write`).
    base: usize,
};

/// What a flush took to the tier.
pub const Flushed = struct { points: usize = 0, top: usize = 0, bytes: u64 = 0, id: ?u64 = null };

/// Writes the points of `points` that are not on the tier yet, at or past the tier's floor and at most `covered` (the
/// tokens the record and the state both hold), each captured as a stored point by `ctx.capture(allocator, point)`;
/// `ctx.prepare(cache)` readies the KV cache of `layers` layers at `quant` the record is written over (its rows to
/// the top point). The points of `points` the tier then holds are marked. A point that will not capture is skipped
/// with a line.
pub fn flush(ctx: anytype, hc: *pc.HotPrefixCache, points: anytype, rec: Record, covered: usize, layers: u32, quant: kv_quant.KVQuantConfig, s: mlx.mlx_stream, comptime tag: []const u8) Flushed {
    const log = @import("log.zig");
    const allocator = hc.allocator;
    var cps: std.ArrayList(trunk.SSMCheckpoint) = .empty;
    defer {
        for (cps.items) |*cp| cp.deinit(allocator);
        cps.deinit(allocator);
    }
    for (points) |*p| {
        if (onDisk(p.disk_id, p.pos, &hc.disk.?) or p.pos < kv_disk_cache.MIN_PERSIST_TOKENS or p.pos > covered) continue;
        var cp = ctx.capture(allocator, p) catch |err| {
            log.warn("  " ++ tag ++ " live point at {d} not flushed: {s}\n", .{ p.pos, @errorName(err) });
            continue;
        };
        cps.append(allocator, cp) catch {
            cp.deinit(allocator);
            continue;
        };
    }
    if (cps.items.len == 0) return .{};
    const top = cps.items[cps.items.len - 1].pos;
    var cache = trunk.KVCache.initWithConfig(allocator, layers, quant) catch return .{};
    defer cache.deinit();
    cache.step = top;
    ctx.prepare(&cache) catch |err| {
        log.warn("  " ++ tag ++ " live points not flushed, rows not marked: {s}\n", .{@errorName(err)});
        return .{};
    };
    const media = pc.truncatedMediaKey(rec.media_chain, rec.vision_key, rec.media_start, top);
    const w = write(hc, cache.entries, cache.config, rec.tokens[0..top], rec.has_tools, media, rec.media_chain, rec.restored, cps.items, rec.base, s);
    if (w.id) |id| markOnDisk(points, &hc.disk.?, id);
    return .{ .points = cps.items.len, .top = top, .bytes = w.bytes, .id = w.id };
}
