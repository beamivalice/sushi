//! DeepSeek-V4.1's prefill takes its working set from the expert cache. A forward that does not fit the free memory
//! first releases whole decoder-layer pools, latest layer first: under CED those layers run only a chunk's last
//! `window` rows, and a released layer's picks are read into the union workspace. The first decode forward after the
//! prefill reads the pools back. Admission bills a request's state and a decode forward, not its prefill working set.
const std = @import("std");
const mlx = @import("mlx.zig");
const log = @import("log.zig");
const status = @import("status.zig");
const stream = @import("expert_stream.zig");
const cfg_mod = @import("dsv41_config.zig");
const ModelConfig = @import("model.zig").ModelConfig;
const Config = cfg_mod.Config;

/// The live headroom (the GPU ceiling less MLX's active bytes), installed by the server. Unset (tests, `sushi kld`),
/// nothing is ever released.
pub var headroom_fn: ?*const fn (*const ModelConfig) u64 = null;

/// The widest verify block, `t1` and 16 drafts: up to it a forward is a decode forward, whose working set admission
/// reserves.
pub const verify_rows: u64 = 17;

/// The working set of a forward of `rows` that ends at position `seq` (a prefill chunk's, whose last chunk may absorb a
/// tail, or a decode or verify forward's) and the state its rows add.
pub fn forwardBytes(c: *const Config, rows: u64, seq: u64) u64 {
    const work = if (rows > verify_rows) c.transientBytes(rows -| cfg_mod.tail_merge_rows, seq) else c.decodeTransientBytes(rows, seq);
    return work +| rows *| c.stateBytesPerToken();
}

/// What a prefill-width forward must find free before it runs: `forwardBytes` with the request bill's quarter.
pub fn prefillNeed(c: *const Config, rows: u64, seq: u64) u64 {
    return forwardBytes(c, rows, seq) *| 5 / 4;
}

/// The pools a prefill may release when `cache_bytes` is split evenly over the layers: the load-time estimate the
/// engine's own count replaces once it is built.
pub fn plannedBytes(c: *const Config, stq: bool, cache_bytes: u64) u64 {
    if (!stq) return 0;
    const split = c.cedSplit() orelse return 0;
    return cache_bytes / c.layers * (c.layers - split);
}

/// PURE: how many `pools` (bytes each, in release order) must go before `headroom` covers `need`; null when even all
/// of them leave it short.
pub fn releaseCount(pools: []const u64, need: u64, headroom: u64) ?usize {
    var have = headroom;
    for (pools, 0..) |bytes, i| {
        if (have >= need) return i;
        have +|= bytes;
    }
    return if (have >= need) pools.len else null;
}

/// Before a forward of `rows` that ends at `seq`. A prefill-width forward releases decoder pools, latest first, until
/// its need fits the live headroom, and is refused when all of them do not cover it. A decode forward reads released
/// pools back while each fits beside its own working set.
pub fn beforeForward(engine: *stream.Engine, config: *const ModelConfig, c: *const Config, rows: u64, seq: u64) !void {
    const headroom = headroom_fn orelse return;
    if (engine.borrow_from >= engine.layers.len) return;
    if (rows <= verify_rows) return restore(engine, config, headroom, forwardBytes(c, rows, seq));
    const need = prefillNeed(c, rows, seq);
    if (headroom(config) >= need) return;
    _ = mlx.mlx_clear_cache();
    const before = headroom(config);
    const gpa = engine.allocator;
    const order = try gpa.alloc(u16, engine.layers.len - engine.borrow_from);
    defer gpa.free(order);
    const pools = try gpa.alloc(u64, order.len);
    defer gpa.free(pools);
    var n: usize = 0;
    var li = engine.layers.len;
    while (li > engine.borrow_from) {
        li -= 1;
        if (engine.isBorrowed(@intCast(li))) continue;
        order[n] = @intCast(li);
        pools[n] = engine.poolBytes(@intCast(li));
        n += 1;
    }
    // A resident pack pays for the union workspace out of its first release.
    const count = releaseCount(pools[0..n], need +| engine.unionPending(), before) orelse {
        log.warn("[dsv41-borrow] a {d}-row forward at {d} needs {d:.2} GB; the headroom and every decoder pool hold {d:.2} GB\n", .{ rows, seq, gb(need), gb((before +| sum(pools[0..n])) -| engine.unionPending()) });
        return error.PrefillDoesNotFit;
    };
    const wired = status.getWiredMemBytes();
    var released: u64 = 0;
    var have = before;
    var i: usize = 0;
    // The plan counts what the pools hold; the live headroom decides, so a short release takes the next pool too.
    while (i < n and (i < count or have < need)) : (i += 1) {
        released += try engine.borrowLayer(order[i]);
        have = headroom(config);
    }
    if (released > 0) {
        const waited = waitForUnwire(wired, released);
        log.info("[dsv41-borrow] released layers {d}..{d} ({d:.2} GB) for a {d}-row forward at {d}: need {d:.2} GB, headroom {d:.2} -> {d:.2} GB, unwired in {d} ms\n", .{
            order[i - 1], order[0], gb(released), rows, seq, gb(need), gb(before), gb(have), waited,
        });
    }
    if (have < need) return error.PrefillDoesNotFit;
}

fn restore(engine: *stream.Engine, config: *const ModelConfig, headroom: *const fn (*const ModelConfig) u64, work: u64) !void {
    var any = false;
    for (engine.borrow_from..engine.layers.len) |li| any = any or engine.isBorrowed(@intCast(li));
    if (!any) return;
    _ = mlx.mlx_clear_cache();
    const io = std.Io.Threaded.global_single_threaded.io();
    const t0 = std.Io.Timestamp.now(io, .awake);
    var bytes: u64 = 0;
    var count: usize = 0;
    var left: usize = 0;
    for (engine.borrow_from..engine.layers.len) |li| {
        if (!engine.isBorrowed(@intCast(li))) continue;
        if (left > 0 or headroom(config) < work +| engine.lentBytes(@intCast(li))) {
            left += 1;
            continue;
        }
        bytes += engine.lentBytes(@intCast(li));
        try engine.restoreLayer(@intCast(li));
        count += 1;
    }
    const ms: u64 = @intCast(@divTrunc(t0.untilNow(io, .awake).nanoseconds, std.time.ns_per_ms));
    if (count > 0) log.info("[dsv41-borrow] restored {d} layers ({d:.2} GB) in {d} ms before decode\n", .{ count, gb(bytes), ms });
    if (left > 0) log.info("[dsv41-borrow] {d} layers stay released: no room beside the decode forward\n", .{left});
}

/// The kernel unwires a freed Metal buffer asynchronously; the forward waits until most of what was released has left
/// the wired set, at most a second. Returns the ms waited.
fn waitForUnwire(wired_before: u64, freed: u64) u64 {
    const io = std.Io.Threaded.global_single_threaded.io();
    var waited: u64 = 0;
    while (wired_before -| status.getWiredMemBytes() < freed / 10 * 9 and waited < 1000) : (waited += 10) {
        std.Io.sleep(io, .fromMilliseconds(10), .real) catch break;
    }
    return waited;
}

fn sum(xs: []const u64) u64 {
    var total: u64 = 0;
    for (xs) |x| total +|= x;
    return total;
}

fn gb(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / 1e9;
}

const testing = std.testing;

test "dsv41 borrow: a forward releases pools in order until the headroom covers its need, and no more" {
    const pools = [_]u64{ 3, 3, 2, 4 };
    try testing.expectEqual(@as(?usize, 0), releaseCount(&pools, 10, 10));
    try testing.expectEqual(@as(?usize, 1), releaseCount(&pools, 10, 7));
    try testing.expectEqual(@as(?usize, 2), releaseCount(&pools, 10, 6));
    try testing.expectEqual(@as(?usize, 4), releaseCount(&pools, 12, 0));
    // Every pool short of the need: the forward is refused, never run over the edge.
    try testing.expectEqual(@as(?usize, null), releaseCount(&pools, 13, 0));
}

test "dsv41 borrow: a prefill chunk's need is its working set and its rows' state with the quarter, a decode forward's is far smaller" {
    const c = try cfg_mod.parseText(cfg_mod.release_config);
    const ctx: u64 = 1 << 20;
    try testing.expectEqual((c.transientBytes(2048 - cfg_mod.tail_merge_rows, ctx) + 2048 * c.stateBytesPerToken()) * 5 / 4, prefillNeed(&c, 2048, ctx));
    // A last chunk that absorbed its tail is priced at its own rows.
    try testing.expect(prefillNeed(&c, 2048 + 511, ctx) > prefillNeed(&c, 2048, ctx));
    try testing.expectEqual(c.decodeTransientBytes(6, ctx) + 6 * c.stateBytesPerToken(), forwardBytes(&c, 6, ctx));
    try testing.expect(forwardBytes(&c, verify_rows, ctx) * 3 < prefillNeed(&c, 512, ctx));
}

test "dsv41 borrow: the load-time estimate is the decoder layers' even share of the cache, STQ packs only" {
    const c = try cfg_mod.parseText(cfg_mod.release_config);
    try testing.expectEqual(@as(u64, 20 * 2_000_000_000), plannedBytes(&c, true, 80_000_000_000));
    try testing.expectEqual(@as(u64, 0), plannedBytes(&c, false, 80_000_000_000));
}
