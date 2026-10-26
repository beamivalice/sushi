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

/// What a prefill-width forward must find free before it runs: `forwardBytes` with the request bill's quarter, and the
/// buffer pool's growth above the decode cap (`poolGrowth`).
pub fn prefillNeed(c: *const Config, rows: u64, seq: u64, pool_growth: u64) u64 {
    return forwardBytes(c, rows, seq) *| 5 / 4 +| pool_growth;
}

/// PURE: the MLX buffer-pool cap of a forward of `rows`, and the pool its need bills: the decode cap up to a verify
/// block, then growing with the rows to the prefill cap at a default chunk (`prefill_width`), never above the RAM-wide
/// cap. A 2048-row chunk parks ~3.8 MB a row (twice the forward's own ~2 MB a row), so the cap spreads evenly over it.
pub fn poolCap(rows: u64, decode_cap: u64, prefill_cap: u64, ram_cap: u64) u64 {
    const decode = @min(ram_cap, decode_cap);
    if (rows <= verify_rows) return decode;
    const full = @min(ram_cap, @max(prefill_cap, decode_cap));
    return @max(decode, @min(full, full *| rows / cfg_mod.prefill_width));
}

/// The RAM-wide pool cap, installed by the server. Unset (tests, `sushi kld`), the pool is never moved.
pub var pool_ram_cap: u64 = 0;
/// The cap in force while it is above the decode cap; null at the decode cap the ledger set at load.
var pool_now: ?u64 = null;
/// The process-wide cap the load replaced with the decode cap; null while none is lowered.
var ram_limit: ?u64 = null;

pub const PoolOps = struct {
    set_limit: *const fn (bytes: u64) void,
    clear: *const fn () void,
};

fn rowsCap(rows: u64, config: *const ModelConfig) u64 {
    return poolCap(rows, config.expertPoolBytes(), config.prefillPoolBytes(), if (pool_ram_cap == 0) std.math.maxInt(u64) else pool_ram_cap);
}

/// What a forward of `rows` parks above the decode cap. The decode cap is a standing term of the live headroom
/// (`server.dsv41Headroom`), so a forward that stays at it bills nothing more.
pub fn poolGrowth(rows: u64, config: *const ModelConfig) u64 {
    return rowsCap(rows, config) -| config.expertPoolBytes();
}

fn moveCap(want: u64, config: *const ModelConfig, ops: PoolOps) bool {
    const decode = config.expertPoolBytes();
    const have = pool_now orelse decode;
    if (want == have) return false;
    pool_now = if (want > decode) want else null;
    ops.set_limit(want);
    // A lower cap trims lazily; the clear makes the parked pool what was billed.
    if (want < have) ops.clear();
    return true;
}

/// Moves the pool cap to the one `rows` bills, on the inference thread. True on a change.
pub fn enterForward(rows: u64, config: *const ModelConfig, ops: PoolOps) bool {
    if (pool_ram_cap == 0 or !config.isDsv41()) return false;
    return moveCap(rowsCap(rows, config), config, ops);
}

/// A request is over: whatever forward ran last, the cap drops back to the decode cap.
pub fn endRequest(config: *const ModelConfig, ops: PoolOps) bool {
    if (pool_ram_cap == 0 or !config.isDsv41()) return false;
    return moveCap(config.expertPoolBytes(), config, ops);
}

/// The load lowered the process-wide cap from `prev` to the ledger's decode cap.
pub fn poolCapped(prev: u64) void {
    if (ram_limit == null) ram_limit = prev;
}

/// A DeepSeek model is unloaded: the next model gets the process-wide cap back, and the raised cap is forgotten.
pub fn unloadModel(config: *const ModelConfig, ops: PoolOps) bool {
    if (!config.isDsv41()) return false;
    pool_now = null;
    const prev = ram_limit orelse return false;
    ram_limit = null;
    ops.set_limit(prev);
    return true;
}

/// The unload hook on the real MLX pool.
pub fn unloadModelMlx(config: *const ModelConfig) void {
    _ = unloadModel(config, mlx_ops);
}

/// The request-end hook on the real MLX pool.
pub fn endRequestMlx(config: *const ModelConfig) void {
    _ = endRequest(config, mlx_ops);
}

fn mlxSetLimit(bytes: u64) void {
    var prev: usize = 0;
    _ = mlx.mlx_set_cache_limit(&prev, @intCast(bytes));
}

fn mlxClear() void {
    _ = mlx.mlx_clear_cache();
}

const mlx_ops: PoolOps = .{ .set_limit = &mlxSetLimit, .clear = &mlxClear };

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
    _ = enterForward(rows, config, mlx_ops);
    const headroom = headroom_fn orelse return;
    if (engine.borrow_from >= engine.layers.len) return;
    if (rows <= verify_rows) return restore(engine, config, headroom, forwardBytes(c, rows, seq));
    const need = prefillNeed(c, rows, seq, poolGrowth(rows, config));
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
    try testing.expectEqual((c.transientBytes(2048 - cfg_mod.tail_merge_rows, ctx) + 2048 * c.stateBytesPerToken()) * 5 / 4, prefillNeed(&c, 2048, ctx, 0));
    // A last chunk that absorbed its tail is priced at its own rows.
    try testing.expect(prefillNeed(&c, 2048 + 511, ctx, 0) > prefillNeed(&c, 2048, ctx, 0));
    try testing.expectEqual(c.decodeTransientBytes(6, ctx) + 6 * c.stateBytesPerToken(), forwardBytes(&c, 6, ctx));
    try testing.expect(forwardBytes(&c, verify_rows, ctx) * 3 < prefillNeed(&c, 512, ctx, 0));
}

test "dsv41 borrow: the load-time estimate is the decoder layers' even share of the cache, STQ packs only" {
    const c = try cfg_mod.parseText(cfg_mod.release_config);
    try testing.expectEqual(@as(u64, 20 * 2_000_000_000), plannedBytes(&c, true, 80_000_000_000));
    try testing.expectEqual(@as(u64, 0), plannedBytes(&c, false, 80_000_000_000));
}

test "dsv41 borrow: a prefill chunk's need carries the pool it may park, and the decoder pools cover both at 1M" {
    const c = try cfg_mod.parseText(cfg_mod.release_config);
    const ctx: u64 = 1 << 20;
    const pool = @import("model.zig").dsv41_prefill_pool_bytes;
    try testing.expectEqual(prefillNeed(&c, 2048, ctx, 0) + pool, prefillNeed(&c, 2048, ctx, pool));
    // A 60 GB cache leaves the decoder layers this many bytes to lend.
    const lend = plannedBytes(&c, true, 60_000_000_000);
    try testing.expect(lend >= prefillNeed(&c, 2048 + 511, ctx, pool));
}

test "dsv41 borrow: the pool cap scales with the forward's rows, from the decode cap up to the prefill cap and never past the RAM-wide cap" {
    const decode: u64 = 1 << 30;
    const prefill: u64 = 8 << 30;
    const ram: u64 = 8 << 30;
    try testing.expectEqual(decode, poolCap(1, decode, prefill, ram));
    try testing.expectEqual(decode, poolCap(verify_rows, decode, prefill, ram));
    // A short tail parks the decode pool and no more; the bill and the MLX cap are the same number.
    try testing.expectEqual(decode, poolCap(31, decode, prefill, ram));
    try testing.expectEqual(@as(u64, 2 << 30), poolCap(512, decode, prefill, ram));
    try testing.expectEqual(@as(u64, 4 << 30), poolCap(1024, decode, prefill, ram));
    try testing.expectEqual(prefill, poolCap(cfg_mod.prefill_width, decode, prefill, ram));
    try testing.expectEqual(prefill, poolCap(2048 + 511, decode, prefill, ram));
    try testing.expectEqual(prefill, poolCap(1 << 20, decode, prefill, ram));
    var rows: u64 = 1;
    while (rows < 1 << 21) : (rows += 97) {
        try testing.expect(poolCap(rows, decode, prefill, ram) <= ram);
        try testing.expect(poolCap(rows, decode, prefill, 4 << 30) <= 4 << 30);
        try testing.expect(poolCap(rows, decode, prefill, ram) >= poolCap(rows -| 1, decode, prefill, ram));
    }
    try testing.expectEqual(@as(u64, 4 << 30), poolCap(2048, decode, prefill, 4 << 30));
    try testing.expectEqual(@as(u64, 512 << 20), poolCap(31, decode, prefill, 512 << 20));
    try testing.expectEqual(decode, poolCap(4096, decode, 1 << 20, ram));
}

test "dsv41 borrow: at the budget-115 geometry a 31-row tail fits the headroom with no release, a 2048-row chunk releases three pools" {
    const c = try cfg_mod.parseText(cfg_mod.release_config);
    const decode = @import("model.zig").dsv41_pool_bytes;
    const prefill = @import("model.zig").dsv41_prefill_pool_bytes;
    const ram: u64 = 8 << 30;
    // Live at `--ssd-budget-gb 115` (280 slots a layer): the decoder pools a prefill may release, and the headroom there.
    const pools = [_]u64{ 2_700_000_000, 2_700_000_000, 2_760_000_000 };
    const headroom: u64 = 8_950_000_000;
    const tail = prefillNeed(&c, 31, 14766, poolCap(31, decode, prefill, ram) - decode);
    try testing.expectEqual(@as(?usize, 0), releaseCount(&pools, tail, headroom));
    const chunk = prefillNeed(&c, 2048, 2048, poolCap(2048, decode, prefill, ram) - decode);
    try testing.expectEqual(@as(?usize, 3), releaseCount(&pools, chunk, headroom));
    // The same tail with the full pool billed is what forced the borrow.
    try testing.expectEqual(@as(?usize, 2), releaseCount(&pools, prefillNeed(&c, 31, 14766, prefill), headroom));
}

const PoolProbe = struct {
    var limits: [8]u64 = undefined;
    var n: usize = 0;
    var cleared: usize = 0;
    fn reset() void {
        n = 0;
        cleared = 0;
    }
    fn set(bytes: u64) void {
        limits[n] = bytes;
        n += 1;
    }
    fn clear() void {
        cleared += 1;
    }
    const ops: PoolOps = .{ .set_limit = &set, .clear = &clear };
};

fn withPoolState(ram_cap: u64, body: *const fn () anyerror!void) !void {
    const saved = .{ pool_ram_cap, pool_now, ram_limit };
    defer {
        pool_ram_cap = saved[0];
        pool_now = saved[1];
        ram_limit = saved[2];
    }
    pool_ram_cap = ram_cap;
    pool_now = null;
    ram_limit = null;
    PoolProbe.reset();
    try body();
}

test "dsv41 borrow: a forward moves the pool cap to its rows' cap, and lowering it clears the pool" {
    const config = ModelConfig{ .model_type = "deepseek_v41" };
    const Body = struct {
        fn run() anyerror!void {
            const decode = config.expertPoolBytes();
            // A decode forward finds the ledger's cap in force: nothing to move.
            try testing.expect(!enterForward(1, &config, PoolProbe.ops));
            try testing.expect(!enterForward(31, &config, PoolProbe.ops));
            try testing.expectEqual(@as(usize, 0), PoolProbe.n);
            try testing.expect(enterForward(2048, &config, PoolProbe.ops));
            try testing.expect(!enterForward(2048, &config, PoolProbe.ops));
            try testing.expectEqual(config.prefillPoolBytes(), PoolProbe.limits[0]);
            try testing.expectEqual(@as(usize, 0), PoolProbe.cleared);
            // A narrower forward after a wide one cannot leave the wide pool parked.
            try testing.expect(enterForward(1024, &config, PoolProbe.ops));
            try testing.expectEqual(@as(usize, 1), PoolProbe.cleared);
            try testing.expect(enterForward(1, &config, PoolProbe.ops));
            try testing.expectEqual(decode, PoolProbe.limits[PoolProbe.n - 1]);
            try testing.expectEqual(@as(usize, 2), PoolProbe.cleared);
        }
    };
    try withPoolState(8 << 30, &Body.run);
    // Unset (tests, `sushi kld`), the pool is never moved.
    PoolProbe.reset();
    try testing.expect(!enterForward(2048, &config, PoolProbe.ops));
    try testing.expectEqual(@as(usize, 0), PoolProbe.n);
}

test "dsv41 borrow: a request that ends without a decode forward still lowers the pool cap" {
    const config = ModelConfig{ .model_type = "deepseek_v41" };
    const Body = struct {
        fn run() anyerror!void {
            try testing.expect(enterForward(2048, &config, PoolProbe.ops));
            try testing.expect(endRequest(&config, PoolProbe.ops));
            try testing.expectEqual(config.expertPoolBytes(), PoolProbe.limits[PoolProbe.n - 1]);
            try testing.expectEqual(@as(usize, 1), PoolProbe.cleared);
            // Already at the decode cap: nothing to do.
            try testing.expect(!endRequest(&config, PoolProbe.ops));
            try testing.expectEqual(@as(usize, 2), PoolProbe.n);
            // Another architecture's request never touches the pool.
            try testing.expect(enterForward(2048, &config, PoolProbe.ops));
            const qwen = ModelConfig{ .model_type = "qwen4_exp" };
            try testing.expect(!endRequest(&qwen, PoolProbe.ops));
        }
    };
    try withPoolState(8 << 30, &Body.run);
}

test "dsv41 borrow: a forward bills the pool it parks above the decode cap, so a decode-width tail adds nothing" {
    const config = ModelConfig{ .model_type = "deepseek_v41" };
    const Body = struct {
        fn run() anyerror!void {
            const c = try cfg_mod.parseText(cfg_mod.release_config);
            const decode = config.expertPoolBytes();
            try testing.expectEqual(@as(u64, 0), poolGrowth(1, &config));
            try testing.expectEqual(@as(u64, 0), poolGrowth(31, &config));
            try testing.expectEqual(decode, poolGrowth(512, &config));
            try testing.expectEqual(config.prefillPoolBytes() - decode, poolGrowth(2048, &config));
            // The ledger holds the decode cap: the need of a 31-row tail is its working set and the quarter.
            try testing.expectEqual(forwardBytes(&c, 31, 14766) * 5 / 4, prefillNeed(&c, 31, 14766, poolGrowth(31, &config)));
            try testing.expectEqual(forwardBytes(&c, 2048, 2048) * 5 / 4 + config.prefillPoolBytes() - decode, prefillNeed(&c, 2048, 2048, poolGrowth(2048, &config)));
        }
    };
    try withPoolState(8 << 30, &Body.run);
}

test "dsv41 borrow: unloading DeepSeek gives the next model the process-wide pool cap and forgets the raised cap" {
    const config = ModelConfig{ .model_type = "deepseek_v41" };
    const Body = struct {
        fn run() anyerror!void {
            const ram: u64 = 8 << 30;
            poolCapped(ram);
            // A second load keeps the cap the first one replaced.
            poolCapped(config.expertPoolBytes());
            try testing.expect(enterForward(2048, &config, PoolProbe.ops));
            try testing.expect(pool_now != null);
            try testing.expect(unloadModel(&config, PoolProbe.ops));
            try testing.expectEqual(ram, PoolProbe.limits[PoolProbe.n - 1]);
            try testing.expectEqual(@as(?u64, null), pool_now);
            try testing.expectEqual(@as(?u64, null), ram_limit);
            // Nothing to give back twice, and another architecture's unload never touches the cap.
            const n = PoolProbe.n;
            try testing.expect(!unloadModel(&config, PoolProbe.ops));
            poolCapped(ram);
            const qwen = ModelConfig{ .model_type = "qwen4_exp" };
            try testing.expect(!unloadModel(&qwen, PoolProbe.ops));
            try testing.expectEqual(n, PoolProbe.n);
        }
    };
    try withPoolState(8 << 30, &Body.run);
}
